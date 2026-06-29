# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, cast

import torch
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor.placement_types import Placement, Replicate, Shard

from .cost_models.collective_runtime_estimation import (
    get_nccl_topo_config,
    reset_comms_cost_cache,
    set_nccl_topo_config,
)
from .cost_models.compute_estimation import reset_compute_cost_cache
from .cost_models.nccl_cost_model import (
    NCCLTopoConfig,
    derive_mesh_dim_topo,
    detect_nccl_topo_config,
)
from .optimize_sharding import ShardingOptimizer
from .shardings.placement_options import reset_placement_options_cache


def reset_mesh_search_caches() -> None:
    """Clear mesh-dependent strategy and cost caches."""

    reset_placement_options_cache()
    reset_comms_cost_cache()
    reset_compute_cost_cache()


def _set_cost_model_for_mesh(mesh, cost_model: Any) -> None:
    if isinstance(cost_model, NCCLTopoConfig):
        set_nccl_topo_config(cost_model)
    elif cost_model == "nccl":
        set_nccl_topo_config(detect_nccl_topo_config(mesh))
    else:
        set_nccl_topo_config(None)


def _placement_code(p: Placement) -> str:
    if isinstance(p, Shard):
        return f"S{p.dim}"
    if isinstance(p, Replicate):
        return "R"
    return type(p).__name__


def _split_dim_seed_dim_cost_model(
    cost_model: Any,
    mesh_shape: tuple[int, ...],
    dim_idx: int,
    *,
    fabric_aware: bool,
) -> Any:
    if not fabric_aware or not isinstance(cost_model, NCCLTopoConfig):
        return cost_model

    topo = derive_mesh_dim_topo(cost_model, mesh_shape, dim_idx)
    return replace(cost_model, mesh_dim_topo_override=topo)


def _split_dim_seed_cache_key(
    size: int,
    input_placement: Placement,
    cost_model: Any,
    mesh_shape: tuple[int, ...],
    dim_idx: int,
    *,
    fabric_aware: bool,
) -> tuple[Any, ...]:
    placement = _placement_code(input_placement)
    if isinstance(cost_model, NCCLTopoConfig):
        dim_cost_model = _split_dim_seed_dim_cost_model(
            cost_model, mesh_shape, dim_idx, fabric_aware=fabric_aware
        )
        topo = derive_mesh_dim_topo(dim_cost_model, (int(size),), 0)
        return (
            "nccl",
            int(size),
            placement,
            cost_model.arch.name,
            cost_model.num_nodes,
            cost_model.gpus_per_node,
            cost_model.bw_intra,
            cost_model.bw_inter,
            cost_model.num_channels,
            topo.n_nodes,
            topo.ppn,
            topo.bw_intra,
            topo.bw_inter,
            topo.n_channels,
            dim_cost_model.has_nvswitch,
            dim_cost_model.has_collnet,
            dim_cost_model.net_latency,
        )
    return (str(cost_model), int(size), placement)


def _first_output_placements(output_specs) -> tuple[Placement, ...] | None:
    if isinstance(output_specs, DTensorSpec):
        return output_specs.placements
    if isinstance(output_specs, (tuple, list)):
        for output_spec in output_specs:
            if isinstance(output_spec, DTensorSpec):
                return output_spec.placements
    return None


def build_split_dim_seed(
    gm: torch.fx.GraphModule,
    mesh_shape: tuple[int, ...],
    input_placements: tuple[Placement, ...],
    *,
    cost_model: Any = "nccl",
    force_grad_reduce_in_higher_precision: bool = False,
    repeated_subgraphs: bool = True,
    memory_high_fn: Callable[[int], float] | None = None,
    one_d_cache: dict[tuple[Any, ...], dict[str, Placement]] | None = None,
    device_type: str = "cuda",
    fabric_aware: bool = True,
) -> dict[str, tuple[Placement, ...]]:
    """Return a per-node placement seed for a target mesh shape.

    Args:
        gm: Joint graph to optimize.
        mesh_shape: Target mesh shape.
        input_placements: Required input placement for each target mesh dim.
        cost_model: Cost model identifier or NCCL topology config.
        force_grad_reduce_in_higher_precision: Whether gradient reductions use
            higher precision costs.
        repeated_subgraphs: Whether repeated graph regions share decisions.
        memory_high_fn: Function returning the parameter memory upper bound for
            a one-dimensional solve size.
        one_d_cache: Optional cache reused across calls.
        device_type: Device mesh type.
        fabric_aware: Whether one-dimensional solves use per-dim fabric topology.

    Returns:
        A mapping from FX node name to placement tuple.
    """

    ndim = len(mesh_shape)
    if len(input_placements) != ndim:
        raise ValueError(
            f"input_placements has {len(input_placements)} entries, expected {ndim}"
        )
    if memory_high_fn is None:
        memory_high_fn = lambda size: 1.0 / size  # noqa: E731

    cache = one_d_cache if one_d_cache is not None else {}
    seed_cost_model = cost_model
    if fabric_aware and cost_model == "nccl":
        with unset_fake_temporarily():
            full_mesh = init_device_mesh(
                device_type,
                mesh_shape,
                mesh_dim_names=tuple(f"d{i}" for i in range(ndim)),
            )
        seed_cost_model = detect_nccl_topo_config(full_mesh)

    per_dim: list[dict[str, Placement]] = []
    for dim_idx, size in enumerate(mesh_shape):
        input_placement = input_placements[dim_idx]
        key = _split_dim_seed_cache_key(
            int(size),
            input_placement,
            seed_cost_model,
            mesh_shape,
            dim_idx,
            fabric_aware=fabric_aware,
        )
        if key not in cache:
            with unset_fake_temporarily():
                mesh_1d = init_device_mesh(
                    device_type,
                    (int(size),),
                    mesh_dim_names=("d",),
                )
            prev = get_nccl_topo_config()
            try:
                dim_cost_model = _split_dim_seed_dim_cost_model(
                    seed_cost_model,
                    mesh_shape,
                    dim_idx,
                    fabric_aware=fabric_aware,
                )
                _set_cost_model_for_mesh(mesh_1d, dim_cost_model)
                reset_mesh_search_caches()
                opt = ShardingOptimizer(
                    gm,
                    mesh_1d,
                    force_grad_reduce_in_higher_precision,
                    repeated_subgraphs=repeated_subgraphs,
                )
                opt.add_sharded_input_constraint([(input_placement,)])
                opt.add_sharded_output_constraint([(input_placement,)])
                opt.add_parameter_memory_constraint(0.0, memory_high_fn(int(size)))
                solution = opt.get_solution()
            finally:
                set_nccl_topo_config(prev)

            node_placements: dict[str, Placement] = {}
            for node, strategy in solution.items():
                placements = _first_output_placements(strategy.output_specs)
                if placements is not None:
                    node_placements[node.name] = placements[0]
            cache[key] = node_placements
        per_dim.append(cache[key])

    seed: dict[str, tuple[Placement, ...]] = {}
    for node in gm.graph.nodes:
        if node.op == "output":
            continue
        seed[node.name] = tuple(
            per_dim[i].get(node.name, Replicate()) for i in range(ndim)
        )

    from torch._functorch._aot_autograd.fx_utils import (
        get_plain_input_and_grad_nodes,
        get_plain_output_and_tangent_nodes,
    )

    input_tuple = tuple(input_placements)
    for getter in (get_plain_input_and_grad_nodes, get_plain_output_and_tangent_nodes):
        for _desc, (node, companion) in cast(Any, getter(gm.graph)).items():
            seed[node.name] = input_tuple
            if companion is not None:
                seed[companion.name] = input_tuple

    return seed
