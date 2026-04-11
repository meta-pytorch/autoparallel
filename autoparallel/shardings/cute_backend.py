"""
CuTe sharding backend for AutoParallel optimizer.

Uses ShardedLayout + 5-primitive propagation engine + per-mesh-dim GPU stride
redistribution planner. Supports non-standard shardings (XorStride, cat
non-contiguous) that DTensor cannot represent.
"""

from __future__ import annotations

import operator
from typing import Any

import torch
import torch.fx.traceback as fx_traceback
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._pytree import tree_flatten

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
        params_spec: Any = None,
        buffers_spec: Any = None,
    ) -> Any:
        """Apply the chosen sharding solution to the FX graph.

        Uses CuTe's redistribute_tensor for collective insertion.
        Parameters and buffers are sharded via DTensor conversion.

        Returns:
            (parallel_gm, sharded_param_dict, sharded_buffer_dict)
        """
        from ..apply_sharding import (
            _copy_descriptors_and_rename_placeholders,
            _get_inductor_decomp_table,
            _shard_params_and_buffers,
        )
        from ..graph_passes.graph_utils import cleanup_graph

        # Convert solution to DTensorSpec for parameter sharding
        dtensor_solution = _convert_solution_to_dtensor(solution, mesh)

        # Create local args from placeholder nodes
        local_args = _create_local_args(gm, solution)

        # Interpret graph with collective insertion
        from .cute.redistribute_tensor import redistribute_tensor

        interp = _CuTeApplyShardingInterpreter(gm, solution, mesh, redistribute_tensor)
        with fx_traceback.preserve_node_meta():
            parallel_gm0 = make_fx(interp.run)(*local_args)
        cleanup_graph(parallel_gm0)

        # Decompose high-level collectives
        decomp_table = _get_inductor_decomp_table()
        interp2 = torch.fx.Interpreter(parallel_gm0)
        with fx_traceback.preserve_node_meta():
            parallel_gm = make_fx(interp2.run, decomposition_table=decomp_table)(
                *local_args
            )
        cleanup_graph(parallel_gm)

        # Copy descriptors and rename placeholders (if present)
        try:
            _copy_descriptors_and_rename_placeholders(gm, parallel_gm)
        except (KeyError, StopIteration):
            pass  # No descriptors in source graph (e.g., unit test)

        # Shard parameters and buffers via DTensor
        sharded_param_dict, sharded_buffer_dict = {}, {}
        if params_spec is not None or buffers_spec is not None:
            sharded_param_dict, sharded_buffer_dict = _shard_params_and_buffers(
                gm, dtensor_solution, params_spec or {}, buffers_spec or {}
            )

        return parallel_gm, sharded_param_dict, sharded_buffer_dict


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


# =============================================================================
# apply_solution helpers
# =============================================================================


class _CuTeApplyShardingInterpreter(torch.fx.Interpreter):
    """FX Interpreter that inserts CuTe collectives at sharding mismatches.

    For each graph node, compares the source output ShardedLayout with the
    target input ShardedLayout. On mismatch, calls redistribute_tensor to
    insert the appropriate funcol collective operations.
    """

    def __init__(self, module, sharding_placement, mesh, redistribute_fn):
        super().__init__(module, garbage_collect_values=True, graph=None)
        self.sharding_placement = sharding_placement
        self.mesh = mesh
        self.redistribute_fn = redistribute_fn

    def run_node(self, n):
        self._curr_node = n
        return super().run_node(n)

    def _get_input_nodes(self, node):
        from ..graph_passes.graph_utils import all_input_nodes
        return all_input_nodes(node)

    def call_function(self, target, args, kwargs):
        node = self._curr_node

        if node not in self.sharding_placement:
            return super().call_function(target, args, kwargs)

        if target == operator.getitem:
            return self._handle_getitem(target, args, kwargs)

        input_nodes = self._get_input_nodes(node)
        node_opt = self.sharding_placement[node]

        flat_args, treespec = tree_flatten(args)
        new_flat_args = list(flat_args)
        tensor_idx = 0

        for i, arg in enumerate(flat_args):
            if isinstance(arg, torch.Tensor) and tensor_idx < len(input_nodes):
                inp_node = input_nodes[tensor_idx]

                if inp_node in self.sharding_placement:
                    src_spec = self.sharding_placement[inp_node].output_spec
                    tgt_spec = node_opt.input_specs[tensor_idx]

                    if (isinstance(src_spec, ShardedLayout)
                            and isinstance(tgt_spec, ShardedLayout)
                            and src_spec != tgt_spec):
                        new_flat_args[i] = self.redistribute_fn(
                            arg, src_spec, tgt_spec, self.mesh
                        )

                tensor_idx += 1

        new_args = list(treespec.unflatten(new_flat_args))

        # Adjust shape for factory ops (zeros, ones, etc.)
        from .propagation_rules import TENSOR_FACTORY_OPS
        if target in TENSOR_FACTORY_OPS and target != torch.ops.aten.scalar_tensor.default:
            out_spec = node_opt.output_spec
            if isinstance(out_spec, ShardedLayout):
                local_sizes = out_spec.local_sizes
                val = list(new_args[0])
                for dim in range(len(val)):
                    mesh_dims = out_spec.mesh_dim_map.get(dim, ())
                    if mesh_dims:
                        val[dim] = local_sizes[dim]
                new_args[0] = tuple(val)

        return super().call_function(target, tuple(new_args), kwargs)

    def _handle_getitem(self, target, args, kwargs):
        node = self._curr_node
        idx = args[1]
        arg = args[0][idx]

        if isinstance(arg, torch.Tensor):
            input_nodes = self._get_input_nodes(node)
            if input_nodes and input_nodes[0] in self.sharding_placement:
                producer_opt = self.sharding_placement[input_nodes[0]]
                # Producer output may be a tuple of specs
                src_spec = producer_opt.output_spec
                if isinstance(src_spec, (tuple, list)):
                    src_spec = src_spec[idx]

                if node in self.sharding_placement:
                    tgt_spec = self.sharding_placement[node].input_specs[0]
                    if (isinstance(src_spec, ShardedLayout)
                            and isinstance(tgt_spec, ShardedLayout)
                            and src_spec != tgt_spec):
                        new_args_0 = list(args[0])
                        new_args_0[idx] = self.redistribute_fn(
                            arg, src_spec, tgt_spec, self.mesh
                        )
                        return super().call_function(target, (tuple(new_args_0), idx), kwargs)

        return super().call_function(target, args, kwargs)


def _create_local_args(gm, solution):
    """Create local tensor args for placeholder nodes based on ShardedLayout specs."""
    local_args = []
    for node in gm.graph.find_nodes(op="placeholder"):
        tensor = node.meta["val"]
        if node in solution:
            spec = solution[node].output_spec
            if isinstance(spec, ShardedLayout):
                local_shape = spec.local_sizes
                local_t = torch.randn(local_shape, dtype=tensor.dtype, device="meta")
                local_args.append(local_t)
                continue
        # Replicate or not in solution — use original shape
        local_args.append(torch.randn(tensor.shape, dtype=tensor.dtype, device="meta"))
    return local_args


def _sharded_layout_to_dtensor_spec(sl, mesh):
    """Convert ShardedLayout to DTensorSpec for parameter sharding.

    For each mesh dim: if it shards tensor dim d → Shard(d), else Replicate.
    If Partial → _Partial(reduce_op).
    Works for standard shardings. Non-standard (XorStride, cat) not supported.
    """
    from torch.distributed.tensor._dtensor_spec import DTensorSpec
    from torch.distributed.tensor.placement_types import Partial, Replicate, Shard

    # Build reverse map: mesh_dim -> (tensor_dim, ...)
    mesh_ndim = mesh.ndim if hasattr(mesh, 'ndim') else len(mesh.shape)
    placements = []
    for md in range(mesh_ndim):
        placed = False
        for td, mesh_dims in sl.mesh_dim_map.items():
            if md in mesh_dims:
                placements.append(Shard(td))
                placed = True
                break
        if not placed:
            if md in sl.partial:
                placements.append(Partial(sl.partial[md]))
            else:
                placements.append(Replicate())

    return DTensorSpec(mesh, tuple(placements))


def _convert_solution_to_dtensor(solution, mesh):
    """Convert a ShardedLayout solution to DTensorSpec solution for param sharding."""
    from torch.distributed.tensor._dtensor_spec import DTensorSpec
    from torch.distributed.tensor.placement_types import Replicate

    dtensor_solution = {}
    for node, opt in solution.items():
        out_spec = opt.output_spec
        if isinstance(out_spec, ShardedLayout):
            dt_out = _sharded_layout_to_dtensor_spec(out_spec, mesh)
        else:
            dt_out = out_spec

        dt_inputs = []
        for inp_spec in opt.input_specs:
            if isinstance(inp_spec, ShardedLayout):
                dt_inputs.append(_sharded_layout_to_dtensor_spec(inp_spec, mesh))
            else:
                dt_inputs.append(inp_spec)

        # Create a minimal OpOption-like object with DTensorSpec
        # that _shard_params_and_buffers can use
        dtensor_solution[node] = OpOption(
            output_spec=dt_out,
            input_specs=tuple(dt_inputs),
            redistribute_costs=opt.redistribute_costs,
        )

    return dtensor_solution
