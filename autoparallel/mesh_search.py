# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from typing import Any, Callable, Iterator

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
from .shardings.propagation_rules import DTensorSpecSeed, OpSpecSeed, StrategySeed


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
    boundary_placements: tuple[Any, ...],
    cost_model: Any,
    mesh_shape: tuple[int, ...],
    dim_idx: int,
    *,
    fabric_aware: bool,
) -> tuple[Any, ...]:
    if isinstance(cost_model, NCCLTopoConfig):
        dim_cost_model = _split_dim_seed_dim_cost_model(
            cost_model, mesh_shape, dim_idx, fabric_aware=fabric_aware
        )
        topo = derive_mesh_dim_topo(dim_cost_model, (int(size),), 0)
        return (
            "nccl",
            int(size),
            boundary_placements,
            tuple(mesh_shape),
            dim_idx,
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
    return (
        str(cost_model),
        int(size),
        boundary_placements,
        tuple(mesh_shape),
        dim_idx,
    )


def _spec_to_seed(spec: DTensorSpec | None) -> DTensorSpecSeed | None:
    if spec is None:
        return None
    return DTensorSpecSeed(tuple(spec.placements), spec.tensor_meta)


def _op_spec_to_seed(strategy) -> OpSpecSeed:
    output_specs = strategy.output_specs
    seeded_outputs: DTensorSpecSeed | tuple[DTensorSpecSeed | None, ...] | None
    if isinstance(output_specs, DTensorSpec):
        seeded_outputs = _spec_to_seed(output_specs)
    elif isinstance(output_specs, (tuple, list)):
        seeded_outputs = tuple(_spec_to_seed(spec) for spec in output_specs)
    else:
        seeded_outputs = None

    input_specs = strategy.input_specs
    return OpSpecSeed(
        output_specs=seeded_outputs,
        input_specs=(
            None
            if input_specs is None
            else tuple(_spec_to_seed(spec) for spec in input_specs)
        ),
    )


def _first_seeded_output(seed: OpSpecSeed) -> DTensorSpecSeed | None:
    output_specs = seed.output_specs
    if isinstance(output_specs, DTensorSpecSeed):
        return output_specs
    if isinstance(output_specs, tuple):
        return next(
            (spec for spec in output_specs if isinstance(spec, DTensorSpecSeed)), None
        )
    return None


def _combine_spec_seeds(
    per_dim: list[DTensorSpecSeed | None],
) -> tuple[bool, DTensorSpecSeed | None]:
    if all(spec is None for spec in per_dim):
        return True, None
    if any(spec is None for spec in per_dim):
        return False, None

    specs = [spec for spec in per_dim if spec is not None]
    if any(len(spec.placements) != 1 for spec in specs):
        return False, None
    tensor_meta = specs[0].tensor_meta
    if any(spec.tensor_meta != tensor_meta for spec in specs[1:]):
        return False, None
    return True, DTensorSpecSeed(
        tuple(spec.placements[0] for spec in specs), tensor_meta
    )


def _combine_output_seeds(
    per_dim: list[DTensorSpecSeed | tuple[DTensorSpecSeed | None, ...] | None],
):
    if all(output is None for output in per_dim):
        return True, None
    if all(isinstance(output, DTensorSpecSeed) for output in per_dim):
        return _combine_spec_seeds(per_dim)  # type: ignore[arg-type]
    if not all(isinstance(output, tuple) for output in per_dim):
        return False, None

    outputs = [output for output in per_dim if isinstance(output, tuple)]
    if not outputs or any(len(output) != len(outputs[0]) for output in outputs[1:]):
        return False, None
    combined = []
    for index in range(len(outputs[0])):
        valid, spec = _combine_spec_seeds([output[index] for output in outputs])
        if not valid:
            return False, None
        combined.append(spec)
    return True, tuple(combined)


def _combine_op_spec_seeds(per_dim: list[OpSpecSeed]) -> OpSpecSeed | None:
    valid, output_specs = _combine_output_seeds(
        [strategy.output_specs for strategy in per_dim]
    )
    if not valid:
        return None

    input_specs = [strategy.input_specs for strategy in per_dim]
    if all(specs is None for specs in input_specs):
        combined_inputs = None
    elif any(specs is None for specs in input_specs):
        return None
    else:
        inputs = [specs for specs in input_specs if specs is not None]
        if any(len(specs) != len(inputs[0]) for specs in inputs[1:]):
            return None
        combined = []
        for index in range(len(inputs[0])):
            slot_valid, spec = _combine_spec_seeds([specs[index] for specs in inputs])
            if not slot_valid:
                return None
            combined.append(spec)
        combined_inputs = tuple(combined)

    return OpSpecSeed(output_specs=output_specs, input_specs=combined_inputs)


def _project_placements_to_dim(
    placements, dim_idx: int, ndim: int
) -> tuple[tuple[Placement, ...] | None, ...]:
    projected: list[tuple[Placement, ...] | None] = []
    for placement in placements:
        if placement is None:
            projected.append(None)
            continue
        if isinstance(placement, Placement):
            if ndim != 1:
                raise ValueError(
                    "A scalar local_map placement is only valid for a 1D mesh"
                )
            projected.append((placement,))
            continue
        if len(placement) != ndim:
            raise ValueError(
                f"local_map placement has {len(placement)} entries, expected {ndim}"
            )
        projected.append((placement[dim_idx],))
    return tuple(projected)


def _project_constraints_to_dim(constraints, dim_idx: int, ndim: int):
    if constraints is None:
        return None
    projected: list[tuple[Placement, ...] | None] = []
    for placements in constraints:
        if placements is None:
            projected.append(None)
            continue
        if len(placements) != ndim:
            raise ValueError(
                f"placement constraint has {len(placements)} entries, expected {ndim}"
            )
        projected.append((placements[dim_idx],))
    return projected


def _constraint_cache_key(input_constraints, output_constraints) -> tuple[Any, ...]:
    def placements_key(constraints):
        if constraints is None:
            return None
        return tuple(
            None
            if placements is None
            else tuple(_placement_code(placement) for placement in placements)
            for placements in constraints
        )

    return placements_key(input_constraints), placements_key(output_constraints)


@contextmanager
def _project_local_map_to_dim(
    gm: torch.fx.GraphModule, mesh_1d, dim_idx: int, ndim: int
) -> Iterator[None]:
    originals = []
    try:
        for node in gm.graph.nodes:
            local_map_kwargs = node.meta.get("local_map_kwargs")
            if local_map_kwargs is None:
                continue
            projected = dict(local_map_kwargs)
            projected["in_placements"] = _project_placements_to_dim(
                local_map_kwargs["in_placements"], dim_idx, ndim
            )
            projected["out_placements"] = _project_placements_to_dim(
                local_map_kwargs["out_placements"], dim_idx, ndim
            )
            projected["device_mesh"] = mesh_1d
            originals.append((node, local_map_kwargs))
            node.meta["local_map_kwargs"] = projected
        yield
    finally:
        for node, local_map_kwargs in originals:
            node.meta["local_map_kwargs"] = local_map_kwargs


def _build_split_dim_seed(
    gm: torch.fx.GraphModule,
    mesh_shape: tuple[int, ...],
    *,
    input_constraints=None,
    output_constraints=None,
    cost_model: Any = "nccl",
    force_grad_reduce_in_higher_precision: bool = False,
    repeated_subgraphs: bool = True,
    fast_build: bool = True,
    memory_high_fn: Callable[[int], float] | None = None,
    one_d_cache: dict[tuple[Any, ...], dict[str, OpSpecSeed]] | None = None,
    device_type: str = "cuda",
    fabric_aware: bool = True,
) -> dict[str, StrategySeed]:
    """Return per-node placement centers and complete strategy witnesses.

    Args:
        gm: Joint graph to optimize.
        mesh_shape: Target mesh shape.
        input_constraints: Optional user input placement constraints.
        output_constraints: Optional user output placement constraints.
        cost_model: Cost model identifier or NCCL topology config.
        force_grad_reduce_in_higher_precision: Whether gradient reductions use
            higher precision costs.
        repeated_subgraphs: Whether repeated graph regions share decisions.
        fast_build: Whether one-dimensional solves skip redundant enumeration costs.
        memory_high_fn: Function returning the parameter memory upper bound for
            a one-dimensional solve size.
        one_d_cache: Optional cache reused across calls.
        device_type: Device mesh type.
        fabric_aware: Whether one-dimensional solves use per-dim fabric topology.

    Returns:
        A mapping from FX node name to its placement center and optional witness.
    """

    ndim = len(mesh_shape)
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

    per_dim: list[dict[str, OpSpecSeed]] = []
    for dim_idx, size in enumerate(mesh_shape):
        dim_inputs = _project_constraints_to_dim(input_constraints, dim_idx, ndim)
        dim_outputs = _project_constraints_to_dim(output_constraints, dim_idx, ndim)
        key = _split_dim_seed_cache_key(
            int(size),
            _constraint_cache_key(dim_inputs, dim_outputs),
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
                with _project_local_map_to_dim(gm, mesh_1d, dim_idx, ndim):
                    opt = ShardingOptimizer(
                        gm,
                        mesh_1d,
                        force_grad_reduce_in_higher_precision,
                        repeated_subgraphs=repeated_subgraphs,
                        fast_build=fast_build,
                    )
                    if dim_inputs is not None:
                        opt.add_sharded_input_constraint(dim_inputs)
                    if dim_outputs is not None:
                        opt.add_sharded_output_constraint(dim_outputs)
                    opt.add_parameter_memory_constraint(0.0, memory_high_fn(int(size)))
                    solution = opt.get_solution()
            finally:
                set_nccl_topo_config(prev)

            node_strategies: dict[str, OpSpecSeed] = {}
            for node, strategy in solution.items():
                node_strategies[node.name] = _op_spec_to_seed(strategy)
            cache[key] = node_strategies
        per_dim.append(cache[key])

    seed: dict[str, StrategySeed] = {}
    for node in gm.graph.nodes:
        if node.op == "output":
            continue
        output_placements = tuple(
            (
                first_output.placements[0]
                if (strategy := per_dim[i].get(node.name)) is not None
                and (first_output := _first_seeded_output(strategy)) is not None
                and len(first_output.placements) == 1
                else Replicate()
            )
            for i in range(ndim)
        )
        per_dim_witnesses = [
            strategies[node.name] for strategies in per_dim if node.name in strategies
        ]
        witness = (
            _combine_op_spec_seeds(per_dim_witnesses)
            if len(per_dim_witnesses) == ndim
            else None
        )
        seed[node.name] = StrategySeed(output_placements, witness)

    return seed
