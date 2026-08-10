# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import os
from functools import partial
from typing import Optional

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Partial, Replicate, Shard
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.testing._internal.distributed.fake_pg import FakeStore

from autoparallel._testing.models.dsv3 import (
    DeepSeekV3Model,
    MoE,
    _local_mapped_region_no_ep,
    _moe_local_map_kwargs,
    local_mapped_region,
    make_dsv3_config,
)
from autoparallel.api import AutoParallel
from autoparallel.collectives import (
    all_reduce,
    flex_local_map,
    get_flex_local_map_alternatives,
    local_map,
)
from autoparallel.shardings.placement_options import NumericsLogger

_DEFAULT_DTENSOR_RNG_SEED = 0
_LOCAL_MAP_CASES = (
    "plain-sharded",
    "plain-replicated-counts",
    "flex-default",
    "flex-replicated-counts",
    "flex-auto",
)


def _seed_dtensor_rng(rng_seed: Optional[int]) -> None:
    torch.manual_seed(_DEFAULT_DTENSOR_RNG_SEED if rng_seed is None else rng_seed)


def _moe_implementation_1(
    x,
    selected_experts_indices,
    top_scores,
    experts_w1,
    experts_w3,
    experts_w2,
    out,
    top_k,
    num_experts,
    score_before_experts,
    axis_name,
):
    out, tokens_per_expert = local_mapped_region(
        x,
        selected_experts_indices,
        top_scores,
        experts_w1,
        experts_w3,
        experts_w2,
        out,
        top_k,
        num_experts,
        score_before_experts,
        axis_name,
    )
    return out, all_reduce(tokens_per_expert, axis_name)


def _moe_implementation_2(
    x,
    selected_experts_indices,
    top_scores,
    experts_w1,
    experts_w3,
    experts_w2,
    out,
    *,
    top_k,
    num_experts,
    score_before_experts,
    axis_name,
):
    return local_mapped_region(
        x,
        selected_experts_indices,
        top_scores,
        experts_w1,
        experts_w3,
        experts_w2,
        out,
        top_k,
        num_experts,
        score_before_experts,
        axis_name,
    )


def _moe_implementation_no_ep(
    x,
    selected_experts_indices,
    top_scores,
    experts_w1,
    experts_w3,
    experts_w2,
    out,
    *,
    top_k,
    num_experts,
    score_before_experts,
    axis_name,
):
    return _local_mapped_region_no_ep(
        x,
        selected_experts_indices,
        top_scores,
        experts_w1,
        experts_w3,
        experts_w2,
        out,
        top_k,
        num_experts,
        score_before_experts,
        axis_name,
    )


def _moe_implementation_ep_output(*args, **kwargs):
    return (_moe_implementation_2(*args, **kwargs)[0],)


def _moe_implementation_no_ep_output(*args, **kwargs):
    return (_moe_implementation_no_ep(*args, **kwargs),)


def _unused_default_moe_implementation(*args):
    raise AssertionError("flex_local_map alternatives must provide their own fn")


def _replicated_counts_moe_local_map_kwargs(mesh):
    kwargs = _moe_local_map_kwargs(mesh)
    return {
        **kwargs,
        "out_placements": (
            kwargs["out_placements"][0],
            (Partial(reduce_op="sum"), Replicate()),
        ),
    }


def _no_ep_moe_local_map_kwargs(mesh):
    kwargs = _moe_local_map_kwargs(mesh)
    token_placements = (Shard(0), Replicate())
    weight_placements = (Replicate(), Replicate())
    return {
        **kwargs,
        "in_placements": (
            token_placements,
            token_placements,
            token_placements,
            weight_placements,
            weight_placements,
            weight_placements,
            token_placements,
            *kwargs["in_placements"][7:],
        ),
        "out_placements": (
            token_placements,
            (Partial(reduce_op="sum"), Replicate()),
        ),
    }


class _ForcedFlexMoELocalMap(torch.nn.Module):
    def __init__(self, moe, mesh, selected_index):
        super().__init__()
        sharded_kwargs = _moe_local_map_kwargs(mesh)
        replicated_counts_kwargs = _replicated_counts_moe_local_map_kwargs(mesh)
        static_kwargs = {
            "top_k": moe.router.top_k,
            "num_experts": moe.router.num_experts,
            "score_before_experts": moe.score_before_experts,
            "axis_name": moe.axis_name,
        }
        high_cost = 1_000_000.0
        self.mapped_region = flex_local_map(
            _unused_default_moe_implementation,
            alternatives=[
                {
                    "name": "sharded_ep_boundary",
                    "fn": partial(_moe_implementation_2, **static_kwargs),
                    "in_placements": sharded_kwargs["in_placements"][:7],
                    "out_placements": sharded_kwargs["out_placements"],
                    "cost_hint": 0.0 if selected_index == 0 else high_cost,
                },
                {
                    "name": "replicated_counts_boundary",
                    "fn": partial(_moe_implementation_1, **static_kwargs),
                    "in_placements": replicated_counts_kwargs["in_placements"][:7],
                    "out_placements": replicated_counts_kwargs["out_placements"],
                    "cost_hint": 0.0 if selected_index == 1 else high_cost,
                },
            ],
            device_mesh=mesh,
            redistribute_inputs=sharded_kwargs["redistribute_inputs"],
        )

    def forward(
        self,
        x,
        selected_experts_indices,
        top_scores,
        experts_w1,
        experts_w3,
        experts_w2,
        out,
        top_k,
        num_experts,
        score_before_experts,
        axis_name,
    ):
        return self.mapped_region(
            x,
            selected_experts_indices,
            top_scores,
            experts_w1,
            experts_w3,
            experts_w2,
            out,
        )


class _FlexMoELocalMap(torch.nn.Module):
    def __init__(self, moe, mesh):
        super().__init__()
        no_ep_kwargs = _no_ep_moe_local_map_kwargs(mesh)
        ep_kwargs = _moe_local_map_kwargs(mesh)
        static_kwargs = {
            "top_k": moe.router.top_k,
            "num_experts": moe.router.num_experts,
            "score_before_experts": moe.score_before_experts,
            "axis_name": moe.axis_name,
        }
        self.mapped_region = flex_local_map(
            _unused_default_moe_implementation,
            alternatives=[
                {
                    "name": "ep",
                    "fn": partial(_moe_implementation_ep_output, **static_kwargs),
                    "in_placements": ep_kwargs["in_placements"][:7],
                    "out_placements": (ep_kwargs["out_placements"][0],),
                },
                {
                    "name": "no_ep",
                    "fn": partial(_moe_implementation_no_ep_output, **static_kwargs),
                    "in_placements": no_ep_kwargs["in_placements"][:7],
                    "out_placements": (no_ep_kwargs["out_placements"][0],),
                },
            ],
            device_mesh=mesh,
            redistribute_inputs=ep_kwargs["redistribute_inputs"],
        )

    def forward(
        self,
        x,
        selected_experts_indices,
        top_scores,
        experts_w1,
        experts_w3,
        experts_w2,
        out,
        top_k,
        num_experts,
        score_before_experts,
        axis_name,
    ):
        mapped_output = self.mapped_region(
            x,
            selected_experts_indices,
            top_scores,
            experts_w1,
            experts_w3,
            experts_w2,
            out,
        )
        tokens_per_expert = torch.histc(
            selected_experts_indices.flatten(),
            bins=num_experts,
            min=0,
            max=num_experts,
        )
        return mapped_output[0], tokens_per_expert


def _enable_forced_flex_local_map(model, mesh, selected_index):
    for module in list(model.modules()):
        if isinstance(module, MoE):
            module.local_mapped_region = _ForcedFlexMoELocalMap(
                module, mesh, selected_index
            )


def _enable_flex_local_map(model, mesh):
    for module in list(model.modules()):
        if isinstance(module, MoE):
            module.local_mapped_region = _FlexMoELocalMap(module, mesh)


def _enable_replicated_counts_local_map(model, mesh):
    for module in list(model.modules()):
        if isinstance(module, MoE):
            module.local_mapped_region = local_map(
                _moe_implementation_1,
                **_replicated_counts_moe_local_map_kwargs(mesh),
            )


def _snapshot_tensor(value, global_tensor=False):
    with torch.no_grad():
        if isinstance(value, DTensor):
            value = value.full_tensor() if global_tensor else value.to_local()
        return value.detach().cpu()


def _save_validation_snapshot(numerics_logger, name, value):
    rank_dir = numerics_logger.dir / f"rank_{numerics_logger.rank}"
    rank_dir.mkdir(exist_ok=True)
    torch.save(value, rank_dir / name)


def run_test(
    fake_evaluate: bool,
    rng_seed: Optional[int],
    logs_dir: str,
    local_map_case: str,
):
    # Match TorchTitan's DeepSeek V3 debug model shape. This example is a
    # regression guard for placement/clustering issues that only appear at the
    # larger debug shape used by TorchTitan GraphTrainer.
    seq_len = 2048
    if fake_evaluate:
        world_size = 256

        fake_store = FakeStore()
        torch.distributed.init_process_group(
            "fake", store=fake_store, rank=0, world_size=world_size
        )
        local_rank = torch.distributed.get_rank()
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        _seed_dtensor_rng(rng_seed)
        mesh = torch.distributed.device_mesh.init_device_mesh(
            "cuda",
            (world_size // 64, 64),
            mesh_dim_names=("dp", "ep"),
        )

        config = make_dsv3_config(num_experts=64, max_seq_len=seq_len)
    else:
        dp_degree = 2
        ep_degree = 2
        world_size = dp_degree * ep_degree

        assert (
            "WORLD_SIZE" in os.environ
        ), f"run with torchrun --standalone --nproc-per-node {world_size}"
        assert (
            int(os.getenv("WORLD_SIZE")) == world_size
        ), f"Need at least {world_size} GPUs for real evaluation"
        local_rank = int(os.getenv("LOCAL_RANK"))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        _seed_dtensor_rng(rng_seed)
        torch.distributed.init_process_group(backend="nccl", device_id=device)
        mesh = torch.distributed.device_mesh.init_device_mesh(
            "cuda",
            (dp_degree, ep_degree),
            mesh_dim_names=("dp", "ep"),
        )

        config = make_dsv3_config(max_seq_len=seq_len)

    local_batch_size = 8
    global_batch_size = local_batch_size * mesh.shape[0] * mesh.shape[1]

    with torch.device("meta"):
        model = DeepSeekV3Model(
            config,
            mesh=mesh,
            compute_dtype=torch.bfloat16,
        )
    if local_map_case == "plain-replicated-counts":
        _enable_replicated_counts_local_map(model, mesh)
    elif local_map_case == "flex-default":
        _enable_forced_flex_local_map(model, mesh, selected_index=0)
    elif local_map_case == "flex-replicated-counts":
        _enable_forced_flex_local_map(model, mesh, selected_index=1)
    elif local_map_case == "flex-auto":
        _enable_flex_local_map(model, mesh)

    def input_fn():
        return torch.randint(
            0,
            config.vocab_size,
            (global_batch_size, seq_len),
            device=device,
        )

    numerics_logger = None
    if rng_seed is not None:
        numerics_logger = NumericsLogger(logs_dir)
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )
    with AutoParallel(
        model, input_fn, mesh, mp_policy=mp_policy, dynamic=True
    ) as autop:
        autop.add_parameter_memory_constraint(low=None, high=None)

        x_sharding = (Shard(0), Shard(0))

        autop.add_input_constraints([x_sharding])
        autop.add_output_constraints([x_sharding])

        selected_index = {
            "flex-default": 0,
            "flex-replicated-counts": 1,
        }.get(local_map_case)
        flex_nodes = []
        if selected_index is not None:
            flex_nodes = [
                node
                for node in autop.gm.graph.nodes
                if get_flex_local_map_alternatives(
                    node.meta.get("local_map_kwargs", {})
                )
                is not None
            ]
            assert flex_nodes
        sharding_placement = autop.optimize_placement(verbose=False)
        if selected_index is not None:
            for node in flex_nodes:
                spec = sharding_placement[node]
                assert spec.flex_local_map_alternative_index == selected_index
                assert spec.flex_local_map_alternative_name == (
                    "sharded_ep_boundary"
                    if selected_index == 0
                    else "replicated_counts_boundary"
                )
        parallel_mod = autop.apply_placement(sharding_placement)

    parallel_mod.to_empty(device=device)
    parallel_mod.init_weights(buffer_device=device, seed=rng_seed)
    if rng_seed is not None:
        numerics_logger.log_model_weights(parallel_mod)
        _save_validation_snapshot(
            numerics_logger,
            "model_state.pt",
            {
                **{
                    f"parameter:{name}": _snapshot_tensor(value)
                    for name, value in parallel_mod.named_parameters()
                },
                **{
                    f"buffer:{name}": _snapshot_tensor(value)
                    for name, value in parallel_mod.named_buffers()
                },
            },
        )
        _save_validation_snapshot(
            numerics_logger,
            "global_model_state.pt",
            {
                **{
                    f"parameter:{name}": _snapshot_tensor(value, global_tensor=True)
                    for name, value in parallel_mod.named_parameters()
                },
                **{
                    f"buffer:{name}": _snapshot_tensor(value, global_tensor=True)
                    for name, value in parallel_mod.named_buffers()
                },
            },
        )
        torch.manual_seed(rng_seed)

    n_microbatches = 16
    full_batch = torch.randint(
        0,
        config.vocab_size,
        (local_batch_size * n_microbatches, seq_len),
        device=device,
    )
    microbatches = torch.split(full_batch, local_batch_size, dim=0)
    assert len(microbatches) == n_microbatches
    if rng_seed:
        numerics_logger.log_diff(
            full_batch.to(torch.float32), prefix="full batch input"
        )
        _save_validation_snapshot(
            numerics_logger, "full_batch.pt", full_batch.detach().cpu()
        )

    with torch.autograd.set_multithreading_enabled(False):
        if fake_evaluate:
            shape_env = ShapeEnv()
            with FakeTensorMode(
                allow_non_fake_inputs=True,
                shape_env=shape_env,
            ):
                for x in microbatches:
                    out = parallel_mod(x)
                    out.backward(torch.ones_like(out))
        else:
            for i, x in enumerate(microbatches):
                assert x.shape[0] == local_batch_size
                out = parallel_mod(x)
                assert not torch.any(torch.isnan(out)), "Found NaNs in forward output"
                out.backward(torch.ones_like(out))
                if rng_seed is not None:
                    numerics_logger.log_diff(out, prefix=f"mb{i} fwd out")
                    _save_validation_snapshot(
                        numerics_logger,
                        f"mb{i}_output.pt",
                        _snapshot_tensor(out),
                    )

            if rng_seed is not None:
                gradients = {}
                for k, v in parallel_mod.named_parameters():
                    numerics_logger.log_diff(v.grad, prefix=f"grad {k}")
                    gradients[k] = _snapshot_tensor(v.grad)
                _save_validation_snapshot(numerics_logger, "gradients.pt", gradients)
                _save_validation_snapshot(
                    numerics_logger,
                    "global_gradients.pt",
                    {
                        name: _snapshot_tensor(value.grad, global_tensor=True)
                        for name, value in parallel_mod.named_parameters()
                    },
                )
                _save_validation_snapshot(
                    numerics_logger,
                    "final_buffers.pt",
                    {
                        name: _snapshot_tensor(value)
                        for name, value in parallel_mod.named_buffers()
                    },
                )
                _save_validation_snapshot(
                    numerics_logger,
                    "global_final_buffers.pt",
                    {
                        name: _snapshot_tensor(value, global_tensor=True)
                        for name, value in parallel_mod.named_buffers()
                    },
                )

    print("All good!")

    if torch.distributed.is_initialized():
        if torch.distributed.get_backend() == torch.distributed.Backend.NCCL:
            torch.distributed.barrier(device_ids=[local_rank])
        else:
            torch.distributed.barrier()
        torch.cuda.synchronize(device)
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run DeepSeek V3 local_map example")
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
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="out/",
        help="Directory to store logs (default: ./out/).",
    )
    parser.add_argument(
        "--flex-local-map",
        action="store_true",
        help="Compatibility alias for --local-map-case flex-auto.",
    )
    parser.add_argument(
        "--local-map-case",
        choices=_LOCAL_MAP_CASES,
        default=None,
        help="Select the MoE local_map boundary and flex alternative.",
    )
    args = parser.parse_args()

    if args.rng_seed is not None:
        torch.use_deterministic_algorithms(True)

    if args.flex_local_map and args.local_map_case is not None:
        parser.error("--flex-local-map and --local-map-case cannot be used together")
    local_map_case = (
        "flex-auto" if args.flex_local_map else (args.local_map_case or "plain-sharded")
    )

    run_test(
        fake_evaluate=args.fake_evaluate,
        rng_seed=args.rng_seed,
        logs_dir=args.logs_dir,
        local_map_case=local_map_case,
    )
