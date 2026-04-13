"""
CuTe sharding backend for AutoParallel optimizer.

Uses ShardedLayout + 5-primitive propagation engine + per-mesh-dim GPU stride
redistribution planner. Supports non-standard shardings (XorStride, cat
non-contiguous) that DTensor cannot represent.
"""

from __future__ import annotations

import logging
import operator
from typing import Any

import torch
import torch.fx.traceback as fx_traceback
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._pytree import tree_flatten

from .backend import OpOption, OpOptionList, ShardingBackend

logger = logging.getLogger(__name__)
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

        # Handle getitem: inherit specs from producer's tuple output
        if node.target == operator.getitem:
            return self._handle_getitem(node, input_options)

        if op_name is None:
            return self._replicate_fallback(node, input_options)

        fn = get_propagation_rule(op_name)
        if fn is None:
            return self._replicate_fallback(node, input_options, op_name)

        mesh_shape = _get_mesh_shape(mesh)

        # Extract ShardedLayout candidates per input (skip non-OpOptionList entries)
        input_candidates = []
        input_options_list = []  # parallel list of OpOptionList for cost computation
        for opts in input_options:
            if not isinstance(opts, OpOptionList):
                continue
            candidates = [opt.output_spec for opt in opts
                          if isinstance(opt.output_spec, ShardedLayout)]
            input_candidates.append(candidates)
            input_options_list.append(opts)

        if not input_candidates or (not any(len(c) > 0 for c in input_candidates)):
            logger.debug("Node %s (%s): no non-empty input_candidates (n=%d, lengths=%s), falling back",
                         node.name, op_name, len(input_candidates),
                         [len(c) for c in input_candidates])
            return self._replicate_fallback(node, input_options, op_name)

        # Extract non-tensor args from user_args/user_kwargs
        # The propagation function expects ATen-matching args:
        # tensor args replaced by ShardedLayouts, non-tensor args passed through
        tensor_arg_indices = _get_tensor_arg_indices(node, user_args)
        non_tensor_args = _extract_non_tensor_args(user_args, tensor_arg_indices)

        # Run enumerate_strategies with non-tensor args.
        # For multi-input ops like SDPA, the Cartesian product of all input
        # candidates can be huge (21^8 ≈ 38B for SDPA backward on a 2D mesh).
        # Use single-input enumeration for ops where all inputs must match.
        _SDPA_OPS = {
            "aten._scaled_dot_product_flash_attention.default",
            "aten._scaled_dot_product_efficient_attention.default",
            "aten._scaled_dot_product_cudnn_attention.default",
            "aten._scaled_dot_product_flash_attention_backward.default",
            "aten._scaled_dot_product_efficient_attention_backward.default",
            "aten._scaled_dot_product_cudnn_attention_backward.default",
        }
        if op_name in _SDPA_OPS and input_candidates:
            # SDPA: enumerate only with compatible input shapes.
            # All 4D inputs must have the same sharding; non-4D inputs get replicate.
            # Use first 4D input's candidates as the enumeration source.
            strategies = []
            first_4d_idx = None
            for ci, cands in enumerate(input_candidates):
                if cands and len(cands[0].global_shape) >= 4:
                    first_4d_idx = ci
                    break

            if first_4d_idx is not None:
                for sl in input_candidates[first_4d_idx]:
                    combo = []
                    for cands in input_candidates:
                        if not cands:
                            # Empty candidates (non-tensor input with None specs)
                            # Use a dummy replicate spec with shape (1,)
                            combo.append(ShardedLayout.replicate((1,)))
                        elif len(cands[0].global_shape) == len(sl.global_shape):
                            combo.append(sl)
                        else:
                            # Different ndim — use replicate for this input
                            combo.append(ShardedLayout.replicate(cands[0].global_shape))
                    combo = tuple(combo)
                    try:
                        output = fn(*list(combo) + list(non_tensor_args), **user_kwargs)
                    except Exception as e:
                        logger.debug("SDPA %s: candidate failed: %s", op_name, e)
                        continue
                    if output is not None:
                        strategies.append((combo, output))
            if not strategies:
                logger.debug("SDPA %s: all candidates failed. first_4d_idx=%s, n_inputs=%d, "
                             "candidate_shapes=%s",
                             op_name, first_4d_idx, len(input_candidates),
                             [len(c[0].global_shape) if c else 'empty' for c in input_candidates])
        else:
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
                if arg_idx >= len(input_options_list):
                    break
                arg_costs = []
                for src_opt in input_options_list[arg_idx]:
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

        # For boundary ops (alias, clone, contiguous), enumerate all possible
        # output shardings independently — like placeholders. This decouples
        # backward gradient placements from the backward compute chain,
        # letting the optimizer choose the best placement with redistribution
        # cost on the edge.
        _BOUNDARY_OPS = {
            "aten.alias.default",
            "aten.clone.default",
            "aten.contiguous.default",
        }
        if op_name in _BOUNDARY_OPS and node.meta.get("val") is not None:
            tensor = node.meta["val"]
            if hasattr(tensor, 'shape'):
                all_shardings = enumerate_shardings(tuple(tensor.shape), mesh_shape)
                existing = {s.output_spec.placements for s in results
                            if isinstance(s.output_spec, ShardedLayout)}
                for sl in all_shardings:
                    if sl.placements in existing:
                        continue
                    existing.add(sl.placements)
                    redist_costs = []
                    for opts in input_options_list:
                        arg_costs = []
                        for src_opt in opts:
                            src_spec = src_opt.output_spec
                            if isinstance(src_spec, ShardedLayout) and src_spec != sl:
                                cost = self.redistribute_cost(src_spec, sl, mesh)
                            else:
                                cost = 0.0
                            arg_costs.append(cost)
                        redist_costs.append(arg_costs)
                    results.append(OpOption(
                        output_spec=sl,
                        input_specs=(sl,),
                        compute_cost=0.0,
                        comm_cost=min(c for costs in redist_costs for c in costs) if redist_costs else 0.0,
                        redistribute_costs=redist_costs,
                    ))

        return results if results else self._replicate_fallback(node, input_options, op_name)

    def _handle_getitem(self, node, input_options):
        """Handle operator.getitem: inherit specs from producer's tuple output."""
        # node.args = (producer_node, index)
        # input_options has the producer's OpOptionList at position 0
        idx = node.args[1]
        producer_opts = None
        for opts in input_options:
            if isinstance(opts, OpOptionList) and len(opts) > 0:
                producer_opts = opts
                break

        if producer_opts is None:
            logger.debug("getitem %s: no non-empty OpOptionList in input_options (types: %s, lengths: %s)",
                         node.name,
                         [type(x).__name__ for x in input_options],
                         [len(x) if isinstance(x, (list, OpOptionList)) else 'N/A' for x in input_options])
            return self._replicate_fallback(node, input_options)

        results = []
        for i, opt in enumerate(producer_opts):
            out_spec = opt.output_spec
            # If producer output is a tuple of specs, extract the indexed one
            if isinstance(out_spec, (tuple, list)):
                out_spec = out_spec[idx] if idx < len(out_spec) else None
            results.append(OpOption(
                output_spec=out_spec,
                input_specs=(opt.output_spec,),
                compute_cost=0.0,
                comm_cost=0.0,
                redistribute_costs=[[0.0] * len(producer_opts)],
            ))
        return results if results else self._replicate_fallback(node, input_options)

    def _replicate_fallback(self, node, input_options, op_name=None):
        """Fallback: all inputs and output replicate. Always valid."""
        if op_name:
            logger.warning("No CuTe propagation rule for %s — falling back to replicate", op_name)
        else:
            logger.warning("No CuTe propagation rule for node %s — falling back to replicate", node.name)
        tensor = node.meta.get("val")
        if tensor is None:
            return []

        # Handle tuple outputs (e.g., SDPA returns (output, logsumexp, ...))
        if isinstance(tensor, (tuple, list)):
            out_specs = []
            for t in tensor:
                if t is not None and hasattr(t, 'shape'):
                    out_specs.append(ShardedLayout.replicate(tuple(t.shape)))
                else:
                    out_specs.append(None)
            out_spec = tuple(out_specs)
        elif hasattr(tensor, 'shape'):
            out_spec = ShardedLayout.replicate(tuple(tensor.shape))
        else:
            return []

        # Build replicate input specs — only process OpOptionList entries
        rep_inputs = []
        redist_costs = []
        for opts in input_options:
            if isinstance(opts, OpOptionList) and opts and hasattr(opts[0], 'output_spec'):
                first = opts[0]
                first_spec = first.output_spec
                if hasattr(first_spec, 'global_shape'):
                    rep_inputs.append(ShardedLayout.replicate(first_spec.global_shape))
                elif isinstance(first_spec, (tuple, list)):
                    # Producer has tuple output — pass through as-is for getitem
                    rep_inputs.append(first_spec)
                else:
                    rep_inputs.append(out_spec)
                redist_costs.append([0.0] * len(opts))

        return [OpOption(
            output_spec=out_spec,
            input_specs=tuple(rep_inputs),
            compute_cost=0.0,
            comm_cost=0.0,
            redistribute_costs=redist_costs,
        )]

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
        # redistribute_costs: one list per input arg, one cost per source option.
        # For placeholders, there's one "input" (self) and N options.
        # Cost from option i to option j: 0 if same, redistribute_cost otherwise.
        n = len(shardings)
        return [
            OpOption(
                output_spec=sl,
                input_specs=(sl,),
                compute_cost=0.0,
                comm_cost=0.0,
                redistribute_costs=[[
                    0.0 if sl == shardings[j]
                    else self.redistribute_cost(shardings[j], sl, mesh)
                    for j in range(n)
                ]],
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
            if mesh_dim is None:
                # Global collective across all mesh dims
                mesh_dim_size = 1
                for s in mesh_shape:
                    mesh_dim_size *= s
            elif isinstance(mesh_dim, int) and mesh_dim < len(mesh_shape):
                mesh_dim_size = mesh_shape[mesh_dim]
            else:
                mesh_dim_size = 1
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

        # Create local args from placeholder nodes via DTensor sharding
        local_args = _create_local_args(gm, solution, dtensor_solution, mesh)

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
    """Extract the ATen op name string from an FX node.

    Returns the dotted format "aten.mm.default" that matches op_registry keys.
    """
    if node.op != "call_function":
        return None
    target = node.target
    # OpOverload: torch.ops.aten.mm.default -> "aten.mm.default"
    if isinstance(target, torch._ops.OpOverload):
        ns = target.namespace
        op_name = target._schema.name.split("::")[-1]
        overload = target._overloadname
        return f"{ns}.{op_name}.{overload}"
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

        # Adjust shape for view/reshape ops: convert global shape to local shape
        _VIEW_TARGETS = {
            torch.ops.aten.view.default,
            torch.ops.aten.reshape.default,
            torch.ops.aten._unsafe_view.default,
            torch.ops.aten.view_copy.default,
        }
        if target in _VIEW_TARGETS:
            out_spec = node_opt.output_spec
            if isinstance(out_spec, ShardedLayout):
                local_sizes = out_spec.local_sizes
                new_args[1] = list(local_sizes)
            elif isinstance(out_spec, (tuple, list)):
                # Tuple output (shouldn't happen for view, but be safe)
                pass

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


def _create_local_args(gm, solution, dtensor_solution, mesh):
    """Create local tensor args for placeholder nodes.

    Uses DTensor's shard_placeholder_inputs when dtensor_solution is available,
    falls back to raw tensor creation for empty solutions.
    """
    if dtensor_solution:
        from ..apply_sharding import shard_placeholder_inputs
        try:
            sharded = shard_placeholder_inputs(gm, dtensor_solution)
            return [arg.to_local() for arg in sharded]
        except (KeyError, AttributeError) as e:
            import logging
            logging.getLogger(__name__).warning("shard_placeholder_inputs failed: %s, falling back", e)

    # Fallback: create meta tensors with local shapes
    local_args = []
    for node in gm.graph.find_nodes(op="placeholder"):
        tensor = node.meta["val"]
        device = tensor.device if hasattr(tensor, 'device') else "meta"
        if node in solution:
            spec = solution[node].output_spec
            if isinstance(spec, ShardedLayout):
                local_t = torch.randn(spec.local_sizes, dtype=tensor.dtype, device=device)
                local_args.append(local_t)
                continue
        local_args.append(torch.randn(tensor.shape, dtype=tensor.dtype, device=device))
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

    spec = DTensorSpec(mesh, tuple(placements))
    # Set tensor_meta for compatibility with _shard_params_and_buffers
    if hasattr(sl, 'global_shape'):
        from torch.distributed.tensor._dtensor_spec import TensorMeta
        shape = torch.Size(sl.global_shape)
        # Row-major strides
        strides = []
        acc = 1
        for s in reversed(sl.global_shape):
            strides.append(acc)
            acc *= s
        strides.reverse()
        spec.tensor_meta = TensorMeta(shape=shape, stride=tuple(strides), dtype=torch.float32)
    return spec


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
