# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import functools
import logging
import os
from contextlib import nullcontext
from typing import Callable, Optional

import torch
import torch.distributed._tools.fake_collectives
import torch.nn as nn
from torch._logging import trace_structured
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed.pipelining.schedules import (
    FORWARD,
    FULL_BACKWARD,
    REDUCE_GRAD,
    RESHARD,
    UNSHARD,
    PipelineScheduleMulti,
    _PipelineSchedule,
    _PipelineScheduleRuntime,
    get_schedule_class,
)
from torch.distributed.pipelining.stage import PipelineStage
from torch.distributed.tensor.placement_types import Shard
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.testing._internal.distributed.fake_pg import FakeStore

from autoparallel._testing.models.dsv3 import (
    DeepSeekV3Model,
    DeepSeekV3ModelArgs,
    DeepSeekV3Stage0,
    DeepSeekV3StageI,
    DeepSeekV3StageN,
    MoEArgs,
)
from autoparallel.api import AutoParallelPP
from autoparallel.graph_pp_runner import (
    GraphCallables,
    GraphMeta,
    GraphPipelineStage,
    GraphPPRunner,
    stage_forward,
    stage_full_backward,
    stage_reduce_grad,
    stage_reshard,
    stage_unshard,
)
from autoparallel.utils import print_rank_by_rank

# Configure logging to show DEBUG messages
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def build_pipeline_schedule(
    stages: list[PipelineStage],
    loss_fn: Callable,
    pipeline_parallel_schedule: str,
    microbatch_size: int,
    local_batch_size: int,
    pipeline_parallel_degree: int,
    backward_requires_autograd: bool = False,
) -> _PipelineSchedule:
    """Builds a pipeline schedule for the given configuration and stages."""
    schedule_class = get_schedule_class(pipeline_parallel_schedule)

    looped_schedule = issubclass(schedule_class, PipelineScheduleMulti)
    assert looped_schedule, "Only looped schedules are supported"
    # validate that the batch size is divisible by the microbatch_size otherwise we'll hang or error during training
    if local_batch_size % microbatch_size != 0:
        raise ValueError(
            f"Batch size {local_batch_size} must be divisible by {microbatch_size=}. "
        )
    n_microbatches = local_batch_size // microbatch_size
    # We expect that the number of local stages (`len(stages)`) is the same across all pp ranks
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
        backward_requires_autograd=backward_requires_autograd,
    )
    logger.info(
        f"Using pipeline schedule {pipeline_parallel_schedule} "
        f"with {n_microbatches} microbatches and {num_total_stages} stages."
    )
    return schedule


def run_test(fake_evaluate: bool, debug_numerics: Optional[bool]):
    if not fake_evaluate:
        pp_degree = 2
        dp_mod_ep_degree = 2
        ep_degree = 2
    else:
        pp_degree = 4
        dp_mod_ep_degree = 4
        ep_degree = 64

    dp_degree = dp_mod_ep_degree * ep_degree
    world_size = pp_degree * dp_mod_ep_degree * ep_degree

    # Initialize process group based on evaluation mode
    if fake_evaluate:
        assert (
            "WORLD_SIZE" in os.environ
        ), "run with torchrun --standalone --nproc-per-node 4"
        assert (
            int(os.getenv("WORLD_SIZE")) == pp_degree
        ), "world_size must be 4, for fake evaluation"
        rank = int(os.getenv("RANK"))
        device = torch.device(f"cuda:{rank}")
        fake_store = FakeStore()
        torch.distributed.init_process_group(
            "fake",
            store=fake_store,
            rank=rank * dp_degree,  # global rank is pp_rank * spmd_size
            world_size=world_size,
        )
        pp_rank = rank
    else:
        assert (
            "WORLD_SIZE" in os.environ
        ), "run with torchrun --standalone --nproc-per-node 8"
        assert (
            int(os.getenv("WORLD_SIZE")) == world_size
        ), "Need at least 8 GPUs for real evaluation"
        local_rank = int(os.getenv("LOCAL_RANK"))
        device = torch.device(f"cuda:{local_rank}")
        torch.distributed.init_process_group(backend="nccl")

    # Initialize device mesh (common for both modes)
    world_mesh = torch.distributed.device_mesh.init_device_mesh(
        "cuda",
        (pp_degree, dp_mod_ep_degree, ep_degree),
        mesh_dim_names=(
            "pp",
            "dp_mod_ep",
            "ep",
        ),
    )

    # Set pp_rank based on evaluation mode
    if not fake_evaluate:
        pp_rank = world_mesh["pp"].get_local_rank()

    stages_per_rank = 2
    total_pp_stages = pp_degree * stages_per_rank

    # This is the spmd mesh to be used for tracing
    mesh = world_mesh[("dp_mod_ep", "ep")]

    global_batch_size = 32 * dp_degree
    # Batch size that will be supplied to the schedule and will be broken down into microbatches
    local_batch_size = global_batch_size // dp_degree
    n_microbatches = 16
    # Batch size with which the spmd graphs will actually be executed
    microbatch_size = local_batch_size // n_microbatches
    assert (
        microbatch_size >= 1
    ), f"invalid config {local_batch_size=}, {n_microbatches=}"
    # Batch size to be used for spmd tracing
    spmd_batch_size = microbatch_size * dp_degree

    seq_len = 1024

    if fake_evaluate:
        config = DeepSeekV3ModelArgs(
            vocab_size=102400,
            max_seq_len=seq_len,
            dim=2048,
            inter_dim=10944,
            moe_inter_dim=1408,
            n_layers=8,  # 27,
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
    else:
        config = DeepSeekV3ModelArgs(
            vocab_size=2048,
            max_seq_len=seq_len,
            dim=256,
            inter_dim=1024,
            moe_inter_dim=256,
            n_layers=4,
            n_dense_layers=0,  # 1,
            n_heads=16,
            moe_args=MoEArgs(
                num_experts=4,
                num_shared_experts=2,
                top_k=2,
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
        )

    with torch.device("meta"):
        model = DeepSeekV3Model(config).bfloat16()
        embed, layers, norm, output = list(model.children())
        items = list(layers.items())
        assert len(items) == config.n_layers
        n_layers_per_rank = len(items) // total_pp_stages
        layers = [
            nn.ModuleDict(items[i : i + n_layers_per_rank])
            for i in range(0, len(items), n_layers_per_rank)
        ]
        assert len(layers) == total_pp_stages
        for lst in layers:
            assert len(lst) * len(layers) == config.n_layers

    def tracing_input_fn():
        return torch.randint(
            0,
            config.vocab_size,
            (spmd_batch_size, seq_len),
            device=device,
        )

    def tracing_input_fn_after_first_stage():
        return torch.randn(
            (spmd_batch_size, seq_len, config.dim),
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )

    def runtime_input_fn():
        return torch.randint(
            0,
            config.vocab_size,
            (local_batch_size, seq_len),
            device=device,
        )

    def shape_inference_input_fn():
        return torch.randint(
            0,
            config.vocab_size,
            (microbatch_size, seq_len),
            device="meta",
        )

    def shape_inference_input_fn_after_first_stage():
        return torch.randn(
            (microbatch_size, seq_len, config.dim),
            device="meta",
            dtype=torch.bfloat16,
            requires_grad=True,
        )

    def shape_inference_output_fn_last_stage():
        return torch.randn(
            (microbatch_size, seq_len, config.vocab_size),
            device="meta",
            dtype=torch.bfloat16,
            requires_grad=True,
        )

    # Step 1. Construct the logical pipeline stages
    with torch.device("meta"):
        virtual_pp_stages = [DeepSeekV3Stage0(embed, layers[0], config)]
        for i in range(1, total_pp_stages - 1):
            virtual_pp_stages.append(DeepSeekV3StageI(layers[i], config))
        virtual_pp_stages.append(
            DeepSeekV3StageN(layers[total_pp_stages - 1], norm, output, config)
        )
    # Step 2. Assign each logical stage(s) to pp ranks for Interleaved1F1B schedule
    pp_rank_to_stage_indices: dict[int, list[int]] = {
        rank: [rank + i * pp_degree for i in range(stages_per_rank)]
        for rank in range(pp_degree)
    }
    assert len(pp_rank_to_stage_indices) == pp_degree
    for stages in pp_rank_to_stage_indices.values():
        assert len(stages) * pp_degree == len(virtual_pp_stages)
    stage_indices_current_pp_rank = pp_rank_to_stage_indices[pp_rank]
    stage_mods: dict[int, torch.nn.Module] = {}
    stage_graphs: dict[int, GraphCallables] = {}
    stage_graph_metas: dict[int, GraphMeta] = {}
    # Step 3. Apply AutoParallel to each logical stage assigned to this pp rank
    use_cache = fake_evaluate
    root_cache = "tmp"
    os.makedirs(root_cache, exist_ok=True)
    from autoparallel.api import AutoParallelPPModule

    for stage_idx in stage_indices_current_pp_rank:
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": f"begin_tracing_stage_{stage_idx}",
                "encoding": "string",
            },
            payload_fn=lambda: "placeholder text",
        )
        stage_mod = virtual_pp_stages[stage_idx]
        eval_mode = "fake" if fake_evaluate else "real"
        stage_file = os.path.join(root_cache, f"stage_{eval_mode}_{stage_idx}.pth")
        if os.path.exists(stage_file) and use_cache:
            cache = torch.load(stage_file, weights_only=False)
            graph_callables = cache["graph_callables"]
            graph_meta = cache["graph_meta"]
            cache["sharded_param_dict"] = {
                k: nn.Parameter(v.detach())
                for k, v in cache["sharded_param_dict"].items()
            }
            pp_mod = AutoParallelPPModule(
                cache["sharded_param_dict"], cache["sharded_buffer_dict"], stage_mod
            )
        else:
            if stage_idx == 0:
                input_fn = tracing_input_fn
            else:
                input_fn = tracing_input_fn_after_first_stage
            with AutoParallelPP(
                stage_mod, input_fn, mesh, dynamic=True, compile=False
            ) as autop:
                autop.add_parameter_memory_constraint(low=None, high=None)

                # x_sharding = (Shard(0), Replicate())
                x_sharding = (Shard(0), Shard(0))

                autop.add_input_constraints([x_sharding])
                autop.add_output_constraints([x_sharding])

                sharding_placement = autop.optimize_placement(verbose=False)
                cache = autop.apply_placement_pp(sharding_placement)
                graph_callables = cache["graph_callables"]
                graph_meta = cache["graph_meta"]
                pp_mod = AutoParallelPPModule(
                    cache["sharded_param_dict"],
                    cache["sharded_buffer_dict"],
                    autop.init_weights_model,
                )
                if use_cache:
                    torch.save(cache, stage_file)

        pp_mod.to_empty(device=device)
        pp_mod.init_weights(buffer_device=device)

        # Store each stage's information in stage_mods, stage_graphs, and stage_graph_metas
        stage_mods[stage_idx] = pp_mod
        stage_graphs[stage_idx] = GraphCallables(
            fw=graph_callables["fw"],
            full_bw=graph_callables["full_bw"],
            bw_dI=graph_callables["bw_dI"],
            bw_dW=graph_callables["bw_dW"],
            unshard=graph_callables["unshard"],
            reduce_grad=graph_callables["reduce_grad"],
        )
        stage_graph_metas[stage_idx] = GraphMeta(
            num_mutate_inputs=graph_meta["num_mutate_inputs"],
            num_user_outputs=graph_meta["num_user_outputs"],
            num_symints_saved_for_bw=graph_meta["num_symints_saved_for_bw"],
            num_params=graph_meta["num_params"],
            num_buffers=graph_meta["num_buffers"],
            num_input_grads=graph_meta["num_input_grads"],
        )
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": f"end_tracing_stage_{stage_idx}",
                "encoding": "string",
            },
            payload_fn=lambda: "placeholder text",
        )

    # Two stages per pp rank
    assert (
        len(stage_indices_current_pp_rank)
        == len(stage_mods)
        == len(stage_graphs)
        == len(stage_graph_metas)
    )

    # run weight init on our sharded DTensor params

    stages = []
    # Step 4. Construct pipeline stages for this pp_rank using the stage modules, graphs and metadata
    for pp_stage_idx, pp_stage_mod in stage_mods.items():
        stage = GraphPipelineStage(
            pp_stage_mod,
            stage_graphs[pp_stage_idx],
            stage_graph_metas[pp_stage_idx],
            stage_index=pp_stage_idx,
            num_stages=len(virtual_pp_stages),
            device=device,
            input_args=(
                shape_inference_input_fn()
                if pp_stage_idx == 0
                else shape_inference_input_fn_after_first_stage()
            ),
            output_args=(
                shape_inference_output_fn_last_stage()
                if pp_stage_idx == (len(virtual_pp_stages) - 1)
                else shape_inference_input_fn_after_first_stage()
            ),
            group=world_mesh.get_group("pp"),
        )
        stages.append(stage)
    # Step 5. Construct the pipeline runner using the pipeline stages for this pp_rank
    schedule = build_pipeline_schedule(
        stages=stages,
        loss_fn=None,
        pipeline_parallel_schedule="Interleaved1F1B",
        microbatch_size=microbatch_size,
        local_batch_size=local_batch_size,
        pipeline_parallel_degree=pp_degree,
        backward_requires_autograd=False,
    )
    assert isinstance(schedule, _PipelineScheduleRuntime)
    # Step 6. Override the pipeline runner's action implementations
    numerics_logs = []
    schedule.register_custom_function(
        FORWARD, functools.partial(stage_forward, numerics_logs=numerics_logs)
    )
    schedule.register_custom_function(FULL_BACKWARD, stage_full_backward)
    schedule.register_custom_function(REDUCE_GRAD, stage_reduce_grad)
    schedule.register_custom_function(RESHARD, stage_reshard)
    schedule.register_custom_function(UNSHARD, stage_unshard)

    # Step 7. Register the schedule with the graph runner

    graph_pp_runner = GraphPPRunner(schedule)

    # Step 8. Run the whole pipeline once using the graph runner
    with (
        FakeTensorMode(
            allow_non_fake_inputs=True,
            shape_env=ShapeEnv(),
        )
        if fake_evaluate
        else nullcontext()
    ):
        with torch.no_grad():
            if pp_rank == 0:
                x = runtime_input_fn()
                graph_pp_runner.step(x)
            else:
                graph_pp_runner.step()

    if debug_numerics:
        print_rank_by_rank("\n".join(numerics_logs))

    print("All good!")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.cuda.synchronize()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run DeepSeek V3 pipeline parallel example"
    )
    parser.add_argument(
        "--fake-evaluate",
        action="store_true",
        default=False,
        help="Use fake evaluation mode with FakeTensorMode (default: False)",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=None,
        help="Use a specific rng seed and deterministic algorithms for run-to-run invariance (default: None).",
    )
    args = parser.parse_args()

    if args.rng_seed is not None:
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(args.rng_seed)

    run_test(fake_evaluate=args.fake_evaluate, debug_numerics=args.rng_seed is not None)
