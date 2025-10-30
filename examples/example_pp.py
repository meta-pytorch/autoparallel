import logging
import os
from typing import Callable

import torch
from autoparallel._testing.models.dsv3 import (
    DeepSeekV3Model,
    DeepSeekV3ModelArgs,
    MoEArgs,
)
from autoparallel.api import AutoParallel
from autoparallel.graph_pp_runner import (
    GraphCallables,
    GraphMeta,
    GraphPipelineStage,
    run_backward_graph,
    run_forward_graph,
)

from torch.distributed.pipelining.schedules import (
    _PipelineSchedule,
    _PipelineScheduleRuntime,
    FORWARD,
    FULL_BACKWARD,
    get_schedule_class,
    PipelineScheduleMulti,
)
from torch.distributed.pipelining.stage import PipelineStage
from torch.distributed.tensor.placement_types import Shard
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.testing._internal.distributed.fake_pg import FakeStore

logger = logging.getLogger(__name__)


def build_pipeline_schedule(
    stages: list[PipelineStage],
    loss_fn: Callable,
    pipeline_parallel_schedule: str,
    microbatch_size: int,
    local_batch_size: int,
    pipeline_parallel_degree: int,
) -> _PipelineSchedule:
    """Builds a pipeline schedule for the given configuration and stages."""
    schedule_class = get_schedule_class(pipeline_parallel_schedule)

    looped_schedule = issubclass(schedule_class, PipelineScheduleMulti)
    # validate that the batch size is divisible by the microbatch_size otherwise we'll hang or error during training
    if local_batch_size % microbatch_size != 0:
        raise ValueError(
            f"Batch size {local_batch_size} must be divisible by {microbatch_size=}. "
        )
    n_microbatches = local_batch_size // microbatch_size
    # We expect that the number of local stages (`len(stages)`) is the same across all ranks
    num_total_stages = pipeline_parallel_degree * len(stages)
    if n_microbatches < num_total_stages:
        logger.warning(
            f"Number of microbatches ({n_microbatches}) is less than the total number "
            f"of stages ({num_total_stages}) which may result in a bubble in the pipeline."
        )

    schedule = schedule_class(
        stages if looped_schedule else stages[0],
        n_microbatches=n_microbatches,
        loss_fn=loss_fn,
    )
    logger.info(
        f"Using pipeline schedule {pipeline_parallel_schedule} "
        f"with {n_microbatches} microbatches and {num_total_stages} stages."
    )
    return schedule


def run_test():
    use_fake_pg = True
    if not use_fake_pg:
        # TODO(sankepurandare): Come back to this later
        torch.distributed.init_process_group()
        assert "WORLD_SIZE" in os.environ, "run with torchrun --nproc-per-node 4"
        world_size = int(os.getenv("WORLD_SIZE"))
        pp_degree = 2
        dp_degree = 2
        ep_degree = 2
        assert (
            world_size == pp_degree * dp_degree * ep_degree
        ), "world_size must be pp * dp * ep"
        world_mesh = torch.distributed.device_mesh.init_device_mesh(
            "cuda",
            (pp_degree, dp_degree, ep_degree),
            mesh_dim_names=(
                "pp",
                "dp",
                "ep",
            ),
        )
        rank = int(os.getenv("RANK"))
        local_rank = int(os.getenv("LOCAL_RANK"))
        device = torch.device(f"cuda:{local_rank}")
        pp_rank = world_mesh["pp"].get_local_rank()
    else:
        rank = int(os.getenv("RANK"))
        pp_degree = 4
        dp_degree = 4
        ep_degree = 64
        world_size = pp_degree * dp_degree * ep_degree

        fake_store = FakeStore()
        torch.distributed.init_process_group(
            "fake", store=fake_store, rank=rank, world_size=world_size
        )
        # mesh = torch.distributed.device_mesh.init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))
        world_mesh = torch.distributed.device_mesh.init_device_mesh(
            "cuda",
            (pp_degree, dp_degree, ep_degree),
            mesh_dim_names=(
                "pp",
                "dp",
                "ep",
            ),
        )
        device = torch.device("cuda")
        pp_rank = rank

    mesh = world_mesh["dp", "ep"]

    global_batch_size = 4 * mesh.shape[1] * mesh.shape[2]
    local_batch_size = global_batch_size // dp_degree
    n_microbatches = 4
    microbatch_size = local_batch_size // n_microbatches
    assert (
        microbatch_size >= 1
    ), f"invalid config {local_batch_size=}, {n_microbatches=}"
    spmd_batch_size = microbatch_size * dp_degree

    seq_len = 1024
    vocab_size = 102400
    dim = 2048

    def model_fn():
        config = DeepSeekV3ModelArgs(
            vocab_size=102400,
            max_seq_len=seq_len,
            dim=2048,
            inter_dim=10944,
            moe_inter_dim=1408,
            n_layers=1,  # 27,
            n_dense_layers=0,  # 1,
            n_heads=16,
            moe_args=MoEArgs(
                num_experts=64,
                num_shared_experts=2,
                top_k=6,
                score_func="softmax",
                route_norm=False,
                score_before_experts=False,
                mesh=mesh,
            ),
            q_lora_rank=0,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            mscale=0.70,
            use_flex_attn=False,
            attn_mask_type="causal",
        )

        m = DeepSeekV3Model(config).bfloat16()
        return m

    # def input_fn():
    #     bs = spmd_batch_size
    #     if pp_rank == 0:
    #         x = torch.randint(
    #             0,
    #             vocab_size,
    #             (bs, seq_len),
    #             device=device,
    #         )
    #     else:
    #         x = torch.randn(
    #             (bs, seq_len, dim),
    #             device=device,
    #             dtype=torch.bfloat16,
    #             requires_grad=True,
    #         )
    #     return x

    def input_fn():
        bs = spmd_batch_size
        return torch.randint(
            0,
            vocab_size,
            (bs, seq_len),
            device=device,
        )

    with torch.device("meta"):
        model = model_fn()

    # Based on pp rank split the model into logical stages and get the logical stages
    # for this rank
    logical_stage_mods: dict[int, torch.nn.Module] = {
        0: model,
    }
    # AutoParallel API call
    stage_mods: dict[int, torch.nn.Module] = {}
    stage_graphs: dict[int, GraphCallables] = {}
    stage_graph_metas: dict[int, GraphMeta] = {}
    for stage_idx, stage_mod in logical_stage_mods.items():
        with AutoParallel(stage_mod, input_fn, mesh, dynamic=True) as autop:
            autop.add_parameter_memory_constraint(low=None, high=None)

            # x_sharding = (Shard(0), Replicate())
            x_sharding = (Shard(0), Shard(0))

            autop.add_input_constraints([x_sharding])
            autop.add_output_constraints([x_sharding])

            sharding_placement = autop.optimize_placement()
            pp_mod = autop.apply_placement_pp(sharding_placement)

            pp_mod.to_empty(device="cuda")
            pp_mod.init_weights(buffer_device="cuda")

            stage_mods[stage_idx] = pp_mod
            stage_graphs[stage_idx] = GraphCallables(
                forward=pp_mod.fw_module, backward=pp_mod.bw_module
            )
            stage_graph_metas[stage_idx] = GraphMeta(
                num_mutate_inputs=pp_mod.num_mutate_inputs,
                num_user_outputs=pp_mod.num_user_outputs,
                num_symints_saved_for_bw=pp_mod.num_symints_saved_for_bw,
            )

    # run weight init on our sharded DTensor params
    torch.manual_seed(pp_rank)
    stages = []
    for pp_stage_idx, pp_stage_mod in stage_mods.items():
        stage = GraphPipelineStage(
            pp_stage_mod,
            stage_graphs[pp_stage_idx],
            stage_graph_metas[pp_stage_idx],
            stage_index=pp_stage_idx,
            num_stages=pp_degree,
            device=device,
            group=world_mesh.get_group("pp"),
        )
        stages.append(stage)
    schedule = build_pipeline_schedule(
        stages=stages,
        loss_fn=None,
        pipeline_parallel_schedule="1F1B",
        microbatch_size=microbatch_size,
        local_batch_size=spmd_batch_size,
        pipeline_parallel_degree=pp_degree,
    )
    assert isinstance(schedule, _PipelineScheduleRuntime)
    schedule.register_custom_function(FORWARD, run_forward_graph)
    schedule.register_custom_function(FULL_BACKWARD, run_backward_graph)
    x = input_fn()
    if pp_rank == 0:
        schedule.step(x)
    else:
        schedule.step()

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.cuda.synchronize()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    run_test()
