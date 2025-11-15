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
import torch.distributed as dist
import torch.distributed._tools.fake_collectives
import torch.nn as nn

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
from examples.example_ds3_pp import build_pipeline_schedule
from torch._C._distributed_c10d import FakeProcessGroup, FakeWork, PythonCallbackWork

from torch._logging import trace_structured
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed import DeviceMesh
from torch.distributed._local_tensor import (
    local_tensor_mode,
    LocalIntNode,
    LocalRunnerMode,
    LocalTensor,
    LocalTensorMode,
    maybe_disable_local_tensor_mode,
)
from torch.distributed._local_tensor._c10d import local_p2p_op
from torch.distributed.pipelining.schedules import (
    _PipelineSchedule,
    _PipelineScheduleRuntime,
    FORWARD,
    FULL_BACKWARD,
    get_schedule_class,
    PipelineScheduleMulti,
    REDUCE_GRAD,
    RESHARD,
    UNSHARD,
)
from torch.distributed.pipelining.stage import InputInfo, PipelineStage
from torch.distributed.tensor.placement_types import Shard
from torch.export._unlift import _assign_attr
from torch.export.unflatten import _AttrKind
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.testing._internal.distributed.fake_pg import FakeStore


# Configure logging to show DEBUG messages
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

_pg_groups: list[list[int]] = []


def enumerate_pp_groups(pp_mesh: DeviceMesh) -> list[list[int]]:
    pp_full_mesh = pp_mesh._layout.remap_to_tensor(pp_mesh._rank_map)
    pp_groups = []
    for i in range(pp_full_mesh.size(dim=0)):
        pp_group = pp_full_mesh[i].tolist()
        pp_groups.append(pp_group)
    return pp_groups


def combine_works(works: list[dist.Work], ctx: str | None = None) -> dist.Work:
    def _wait_all(timeout) -> bool:
        for w in works:
            print(f"{ctx} wait recv")
            w.wait()
        return True

    return PythonCallbackWork(_wait_all)


def get_pp_peer(self: int, peer: int) -> torch.SymInt:
    pp_ret = {}
    global _pp_groups
    for pp_group in _pp_groups:
        global_rank = pp_group[self]
        global_peer = pp_group[peer]
        pp_ret[global_rank] = global_peer
    return torch.SymInt(LocalIntNode(pp_ret))


def expand_p2p_ops(
    ops: list[dist.P2POp], pp_rank: int, ctx: str | None = None
) -> list[dist.P2POp]:
    # Ops where generated from a perspective of pp group where rank 0 is present.

    def multi_isend(tensor, dst=None, group=None, tag=0, group_src=None):
        assert group_src is not None, "Expected group rank"
        peer = get_pp_peer(pp_rank, group_src)
        print(f"PP peer {group_src} {ctx} multi_isend {peer=}")
        works = local_p2p_op(peer, tensor, dist.isend)
        return FakeWork()

    def multi_irecv(tensor, src=None, group=None, tag=0, group_src=None):
        assert group_src is not None, "Expected group rank"
        peer = get_pp_peer(pp_rank, group_src)
        print(f"PP peer {group_src} {ctx} multi_irecv {peer=}")
        works = local_p2p_op(peer, tensor, dist.irecv)
        return combine_works(works, f"PP peer {group_src} {ctx} multi_irecv {peer=}")

    send_ops = []
    recv_ops = []
    for p2p_op in ops:
        op = p2p_op.op
        if op is dist.isend:
            p2p_op.op = multi_isend
            send_ops.append(p2p_op)
        elif op is dist.irecv:
            p2p_op.op = multi_irecv
            recv_ops.append(p2p_op)
        else:
            raise AssertionError("Unxpected op {op}")

    # Execute send ops first and then recv because the latter are blocking
    return send_ops + recv_ops


class LocalGraphPipelineStage(GraphPipelineStage):
    def log_name(self) -> str:
        return (
            f"PP rank {self.group_rank} Stage {self.stage_index} of {self.num_stages}"
        )

    def _get_recv_ops(self, recv_infos: tuple[InputInfo, ...]) -> list[dist.P2POp]:
        ops = super()._get_recv_ops(recv_infos)
        ops = expand_p2p_ops(ops, self.group_rank, self.log_name() + " _get_recv_ops")
        return ops

    def get_fwd_send_ops(self, fwd_chunk_id: int) -> list[dist.P2POp]:
        ops = super().get_fwd_send_ops(fwd_chunk_id)
        ops = expand_p2p_ops(
            ops, self.group_rank, self.log_name() + " get_fwd_send_ops"
        )
        return ops

    def get_bwd_send_ops(self, bwd_chunk_id: int) -> list[dist.P2POp]:
        ops = super().get_bwd_send_ops(bwd_chunk_id)
        ops = expand_p2p_ops(
            ops, self.group_rank, self.log_name() + " get_bwd_send_ops"
        )
        return ops

    def _get_init_p2p_neighbors_ops(self) -> list[dist.P2POp]:
        ops = super()._get_init_p2p_neighbors_ops()
        ops = expand_p2p_ops(
            ops, self.group_rank, self.log_name() + " _get_init_p2p_neighbors_ops"
        )
        return ops


def run_test(run_local: bool, debug_numerics: Optional[bool]):
    pp_degree = 2
    dp_mod_ep_degree = 2
    ep_degree = 2

    dp_degree = dp_mod_ep_degree * ep_degree
    world_size = pp_degree * dp_mod_ep_degree * ep_degree

    # Initialize process group based on evaluation mode
    if run_local:
        assert (
            "WORLD_SIZE" in os.environ
        ), "run with torchrun --standalone --nproc-per-node 1"
        device = torch.device(f"cuda")
        default_pg = torch.distributed.init_process_group(
            "fake",
            rank=0,
            world_size=world_size,
        )
    else:
        assert (
            "WORLD_SIZE" in os.environ
        ), "run with torchrun --standalone --nproc-per-node 8"
        assert (
            int(os.getenv("WORLD_SIZE")) == world_size
        ), "Need at least 8 GPUs for real evaluation"
        local_rank = int(os.getenv("LOCAL_RANK"))
        device = torch.device(f"cuda:{local_rank}")
        default_pg = torch.distributed.init_process_group(backend="nccl")

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

    # Step 0. Construct the model and extract its layers to create stages from.
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

    stage_mods: dict[int, torch.nn.Module] = {}
    stage_graphs: dict[int, GraphCallables] = {}
    stage_graph_metas: dict[int, GraphMeta] = {}

    # Step 3. Apply AutoParallel to each logical stage
    from autoparallel.api import AutoParallelPPModule

    for stage_idx, stage_mod in enumerate(virtual_pp_stages):
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": f"begin_tracing_stage_{stage_idx}",
                "encoding": "string",
            },
            payload_fn=lambda: "placeholder text",
        )

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

    # At this point all stages have been compiles and parallelized.
    # NB: PP rank code
    if run_local:
        global _pp_groups
        _pp_groups = enumerate_pp_groups(world_mesh["pp"])
        # for pp_group_ranks in pp_groups:
        #     _pp_groups.append(default_pg.split_group(pp_group_ranks))

    def run_pp_rank(pp_rank: int):
        maybe_local_context = (
            LocalTensorMode(world_size) if run_local else nullcontext()
        )
        with maybe_local_context:
            # Step 4. Construct pipeline stages for this pp_rank using the stage modules, graphs and metadata
            stage_indices_current_pp_rank = pp_rank_to_stage_indices[pp_rank]
            stages = []
            for pp_stage_idx in stage_indices_current_pp_rank:
                pp_stage_mod = stage_mods[pp_stage_idx]

                # Convert module to local if running under local tensor mode
                maybe_make_module_local(pp_stage_mod)

                args = (
                    pp_stage_mod,
                    stage_graphs[pp_stage_idx],
                    stage_graph_metas[pp_stage_idx],
                )
                kwargs = {
                    "stage_index": pp_stage_idx,
                    "num_stages": len(virtual_pp_stages),
                    "device": device,
                    "input_args": (
                        shape_inference_input_fn()
                        if pp_stage_idx == 0
                        else shape_inference_input_fn_after_first_stage()
                    ),
                    "output_args": (
                        shape_inference_output_fn_last_stage()
                        if pp_stage_idx == (len(virtual_pp_stages) - 1)
                        else shape_inference_input_fn_after_first_stage()
                    ),
                    "group": world_mesh.get_group("pp"),
                }
                stage = (
                    LocalGraphPipelineStage(*args, **kwargs)
                    if run_local
                    else GraphPipelineStage(*args, **kwargs)
                )

                # NB: This is clearly a hack. The purpose of it is to override pp rank
                # that the stage obtained from the process group. Stage computes peers to
                # work with based on group rank.
                if run_local:
                    stage.group_rank = pp_rank

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
                FORWARD,
                functools.partial(stage_forward, numerics_logs=numerics_logs),
            )
            schedule.register_custom_function(FULL_BACKWARD, stage_full_backward)
            schedule.register_custom_function(REDUCE_GRAD, stage_reduce_grad)
            schedule.register_custom_function(RESHARD, stage_reshard)
            schedule.register_custom_function(UNSHARD, stage_unshard)

            # Step 7. Register the schedule with the graph runner
            graph_pp_runner = GraphPPRunner(schedule)

            # Step 8. Run the whole pipeline once using the graph runner
            with torch.no_grad():
                if pp_rank == 0:
                    x = runtime_input_fn()
                    graph_pp_runner.step(x)
                else:
                    graph_pp_runner.step()

            if debug_numerics:
                print_rank_by_rank("\n".join(numerics_logs))

    if run_local:
        with LocalRunnerMode(
            world_size,
            pp_degree,
            run_pp_rank,
        ):
            pass
    else:
        pp_rank = world_mesh["pp"].get_local_rank()
        run_pp_rank(pp_rank)

    print("All good!")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.cuda.synchronize()
        torch.distributed.destroy_process_group()


def local_tensor_mode_if_enabled(
    ltm: LocalTensorMode | None = None,
) -> LocalTensorMode | None:

    for _ in range(2):
        if ltm is not None and not ltm._disable:
            return ltm
        ltm = local_tensor_mode()

    return None


def maybe_make_tensor_local(
    tensor: torch.Tensor,
    ltm: LocalTensorMode | None = None,
) -> torch.Tensor:
    ltm = local_tensor_mode_if_enabled(ltm)
    if ltm is None:
        return tensor

    local_tensor = ltm.rank_map(lambda r: tensor.clone().detach())
    local_tensor.requires_grad = tensor.requires_grad
    return local_tensor


def maybe_make_module_local(
    module: nn.Module,
    ltm: LocalTensorMode | None = None,
) -> None:
    ltm = local_tensor_mode_if_enabled(ltm)
    if ltm is None:
        return

    for k, v in module.named_parameters():
        _assign_attr(
            nn.Parameter(
                data=maybe_make_tensor_local(v.data, ltm),
                requires_grad=v.requires_grad,
            ),
            module,
            k,
            attr_kind=_AttrKind.PARAMETER,
        )

    for k, v in module.named_buffers():
        _assign_attr(
            maybe_make_tensor_local(v, ltm), module, k, attr_kind=_AttrKind.BUFFER
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run DeepSeek V3 pipeline parallel example"
    )
    parser.add_argument(
        "--run-local",
        action="store_true",
        default=False,
        help="Use local tensor mode (default: False)",
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

    run_test(run_local=args.run_local, debug_numerics=args.rng_seed is not None)

# PYTHONPATH=. torchrun --standalone --nproc-per-node 8 examples/example_ds3_pp_local_tensor.py -- --rng-seed 1
# PYTHONPATH=. torchrun --standalone --nproc-per-node 1 examples/example_ds3_pp_local_tensor.py -- --rng-seed 1 --run-local
