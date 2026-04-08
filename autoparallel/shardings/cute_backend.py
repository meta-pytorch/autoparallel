"""
CuTe sharding backend for AutoParallel optimizer.

Uses ShardedLayout + 5-primitive propagation engine + per-mesh-dim GPU stride
redistribution planner. Supports non-standard shardings (XorStride, cat
non-contiguous) that DTensor cannot represent.
"""

from __future__ import annotations

from typing import Any

from .backend import OpOption, ShardingBackend
from .cute import (
    ShardedLayout,
    enumerate_shardings,
    enumerate_strategies,
    get_propagation_rule,
    get_sharding_hints,
    plan_redistribute,
)


class CuTeBackend:
    """ShardingBackend implementation using CuTe ShardedLayouts."""

    def enumerate_options(
        self,
        mesh: Any,
        node: Any,
        input_options: list[list[OpOption]],
        user_args: tuple,
        user_kwargs: dict,
    ) -> list[OpOption]:
        """Generate all valid strategies for a graph node.

        Looks up the propagation rule in the op registry, runs it on each
        combination of input shardings, and wraps successful results in OpOption.
        """
        # Get the ATen op name from the FX node
        op_name = _get_op_name(node)
        if op_name is None:
            return []

        fn = get_propagation_rule(op_name)
        if fn is None:
            return []

        mesh_shape = _get_mesh_shape(mesh)

        # Extract ShardedLayout candidates per input
        input_candidates = []
        for opts in input_options:
            candidates = [opt.output_spec for opt in opts
                          if isinstance(opt.output_spec, ShardedLayout)]
            input_candidates.append(candidates)

        if not input_candidates or any(len(c) == 0 for c in input_candidates):
            return []

        # Extract non-tensor args from user_args/user_kwargs
        # The propagation function expects ATen-matching args:
        # tensor args replaced by ShardedLayouts, non-tensor args passed through
        tensor_arg_indices = _get_tensor_arg_indices(node, user_args)
        non_tensor_args = _extract_non_tensor_args(user_args, tensor_arg_indices)

        # Run enumerate_strategies with non-tensor args
        strategies = enumerate_strategies(
            op_name, input_candidates, *non_tensor_args, **user_kwargs
        )

        # Convert to OpOption with costs
        results = []
        for input_shardings, output_sharding in strategies:
            # Compute redistribute costs: for each input arg, cost to go from
            # each source option's output to this strategy's required input
            redist_costs = []
            total_comm = 0.0
            for arg_idx, inp_sharding in enumerate(input_shardings):
                arg_costs = []
                for src_opt in input_options[arg_idx]:
                    src_spec = src_opt.output_spec
                    if isinstance(src_spec, ShardedLayout) and src_spec != inp_sharding:
                        cost = self.redistribute_cost(src_spec, inp_sharding, mesh)
                    else:
                        cost = 0.0
                    arg_costs.append(cost)
                redist_costs.append(arg_costs)
                # Use min cost as the comm cost for this strategy
                if arg_costs:
                    total_comm += min(arg_costs)

            results.append(OpOption(
                output_spec=output_sharding,
                input_specs=tuple(input_shardings),
                compute_cost=0.0,  # TODO: compute cost estimation
                comm_cost=total_comm,
                redistribute_costs=redist_costs,
            ))

        return results

    def create_all_options(
        self,
        mesh: Any,
        node: Any,
    ) -> list[OpOption]:
        """Generate all possible shardings for a tensor."""
        tensor = node.meta["val"]
        tensor_shape = tuple(tensor.shape)
        mesh_shape = _get_mesh_shape(mesh)
        shardings = enumerate_shardings(tensor_shape, mesh_shape)

        # Add op-specific hints if available (merged at strategy enumeration time)
        return [
            OpOption(
                output_spec=sl,
                input_specs=(sl,),
                compute_cost=0.0,
                comm_cost=0.0,
                redistribute_costs=[],
            )
            for sl in shardings
        ]

    def redistribute_cost(
        self,
        src_spec: Any,
        tgt_spec: Any,
        mesh: Any,
    ) -> float:
        """Cost of redistributing from src to tgt sharding."""
        if not isinstance(src_spec, ShardedLayout) or not isinstance(tgt_spec, ShardedLayout):
            return float('inf')

        if src_spec == tgt_spec:
            return 0.0

        collectives = plan_redistribute(src_spec, tgt_spec)
        if not collectives:
            return 0.0

        # Map collective types to cost estimates
        total_cost = 0.0
        mesh_shape = _get_mesh_shape(mesh)
        tensor_elements = src_spec.num_elements
        # Rough cost model: proportional to data transferred
        # TODO: use autoparallel/cost_models/ for accurate estimates
        bytes_per_element = 4  # assume float32
        tensor_bytes = tensor_elements * bytes_per_element

        for coll_type, mesh_dim, info in collectives:
            mesh_dim_size = mesh_shape[mesh_dim] if mesh_dim < len(mesh_shape) else 1
            if coll_type == "all_gather":
                # Each device sends its shard to all others
                total_cost += tensor_bytes * (mesh_dim_size - 1) / mesh_dim_size
            elif coll_type == "all_reduce":
                # Ring all-reduce: 2 * (n-1)/n * data
                total_cost += 2 * tensor_bytes * (mesh_dim_size - 1) / mesh_dim_size
            elif coll_type == "reduce_scatter":
                total_cost += tensor_bytes * (mesh_dim_size - 1) / mesh_dim_size
            elif coll_type == "ppermute":
                # 1-to-1 transfer: each device sends tensor_bytes / mesh_dim_size
                total_cost += tensor_bytes / mesh_dim_size
            elif coll_type == "all_to_all":
                total_cost += tensor_bytes * (mesh_dim_size - 1) / mesh_dim_size

        return total_cost

    def apply_solution(
        self,
        gm: Any,
        solution: dict,
        mesh: Any,
    ) -> Any:
        """Apply the chosen sharding solution to the FX graph.

        DEFERRED — focus on strategy selection first.
        """
        raise NotImplementedError(
            "CuTe apply_solution not yet implemented. "
            "Use DTensorBackend for graph application."
        )


# =============================================================================
# Helpers
# =============================================================================


def _get_op_name(node) -> str | None:
    """Extract the ATen op name string from an FX node."""
    if node.op != "call_function":
        return None
    target = node.target
    if hasattr(target, "name"):
        # torch.ops.aten.mm.default -> "aten.mm.default"
        return target.name()
    if hasattr(target, "__name__"):
        return target.__name__
    return str(target)


def _get_mesh_shape(mesh) -> tuple[int, ...]:
    """Extract mesh shape from a DeviceMesh or tuple."""
    if isinstance(mesh, tuple):
        return mesh
    if hasattr(mesh, "shape"):
        return tuple(mesh.shape)
    if hasattr(mesh, "size"):
        # 1D mesh
        return (mesh.size(),)
    return (1,)


def _get_tensor_arg_indices(node, user_args) -> list[int]:
    """Find which positional args are tensors (will be replaced by ShardedLayouts)."""
    import torch
    indices = []
    for i, arg in enumerate(user_args):
        if isinstance(arg, torch.Tensor):
            indices.append(i)
    return indices


def _extract_non_tensor_args(user_args, tensor_indices) -> tuple:
    """Extract non-tensor args, preserving order."""
    return tuple(arg for i, arg in enumerate(user_args) if i not in tensor_indices)
