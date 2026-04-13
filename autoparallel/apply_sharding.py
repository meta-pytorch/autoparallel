# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import copy
import functools
import logging
import operator
import time

import torch
import torch.fx.traceback as fx_traceback
import torch.nn as nn
from torch._functorch._aot_autograd.fx_utils import (
    get_named_buffer_nodes,
    get_named_param_nodes,
)
from torch._inductor.decomposition import select_decomp_table
from torch._subclasses.fake_tensor import FakeTensor, unset_fake_temporarily
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._dtensor_spec import DTensorSpec, ShardOrderEntry
from torch.distributed.tensor._ops._view_ops import (
    Flatten,
    InputDim,
    Singleton,
    Split,
    view_groups,
)
from torch.distributed.tensor.placement_types import Partial, Replicate, Shard  # noqa
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._pytree import tree_flatten, tree_map_only

from .graph_passes.graph_utils import all_input_nodes, cleanup_graph
from .shardings.ordered_sharding import (
    compute_optimal_placement_order_for_parameters,
    ordered_redistribute_local_tensor,
)
from .shardings.propagation_rules import TENSOR_FACTORY_OPS

logger = logging.getLogger(__name__)

_VIEW_OPS = {
    torch.ops.aten._unsafe_view.default,
    torch.ops.aten.view.default,
    torch.ops.aten.expand.default,
}


def _args_have_symints(args):
    """Check if any args (including nested lists) contain SymInts."""
    flat, _ = tree_flatten(args)
    return any(isinstance(x, torch.SymInt) for x in flat)


def _concretize_shape(shape):
    """Concretize a shape tuple, replacing SymInts with their hint values."""
    return tuple(s.node.hint if isinstance(s, torch.SymInt) else s for s in shape)


def _compute_local_dim(dim_spec, local_input_shape, output_dim, output_spec):
    """Compute local size of one output dim from the view_groups dim mapping."""
    if isinstance(dim_spec, InputDim):
        return local_input_shape[dim_spec.input_dim]
    elif isinstance(dim_spec, Flatten):
        return functools.reduce(
            operator.mul,
            [
                _compute_local_dim(d, local_input_shape, output_dim, output_spec)
                for d in dim_spec.input_dims
            ],
        )
    elif isinstance(dim_spec, Split):
        inner_local = _compute_local_dim(
            dim_spec.input_dim, local_input_shape, output_dim, output_spec
        )
        is_sharded = any(
            p.is_shard() and p.dim == output_dim for p in output_spec.placements
        )
        if is_sharded and isinstance(inner_local, torch.SymInt):
            # Derive from symbolic inner size to preserve dynamism.
            # inner_local = local product of all pieces; divide out the other pieces.
            other_product = 1
            for i, s in enumerate(dim_spec.group_shape):
                if i != dim_spec.split_id:
                    other_product *= s
            return inner_local // other_product
        else:
            # Concrete: use group_shape, adjust for sharding.
            val = dim_spec.group_shape[dim_spec.split_id]
            for mesh_dim, placement in enumerate(output_spec.placements):
                if placement.is_shard() and placement.dim == output_dim:
                    val = val // output_spec.mesh.size(mesh_dim)
            return val
    elif isinstance(dim_spec, Singleton):
        return 1
    else:
        raise ValueError(f"Unknown dim spec: {dim_spec}")


def _compute_local_view_shape(
    global_input_shape, global_output_shape, local_input_shape, output_spec
):
    """Compute local view output shape using view_groups dim mapping.

    Uses concrete global shapes to determine the dim mapping (Flatten, Split,
    InputDim), then applies the mapping to the local input shape (which may
    have symbolic dims from make_fx) to produce the local output shape.
    """
    mapping = view_groups(global_input_shape, global_output_shape)
    return [
        _compute_local_dim(dim_spec, local_input_shape, out_dim, output_spec)
        for out_dim, dim_spec in enumerate(mapping)
    ]


def _compute_shard_order(shard_order, reverse: bool):
    result = []
    for tensor_dim, mesh_dims in shard_order:
        default_order = sorted(mesh_dims)
        if reverse:
            default_order = default_order[::-1]
        result.append(
            ShardOrderEntry(tensor_dim=tensor_dim, mesh_dims=tuple(default_order))
        )
    return tuple(result)


def _filter_specs_for_local_map(flat_args, curr_specs, tgt_specs):
    """Filter curr/tgt specs to tensor-only entries for local_map HOPs.

    Other ops filter out non-tensor/symint args from their specs already,
    but local_map keeps them, so we need to strip them here.
    """
    curr_specs_t = []
    tgt_specs_t = []
    for i, arg in enumerate(flat_args):
        if isinstance(arg, torch.Tensor):
            curr_specs_t.append(curr_specs[i])
            tgt_specs_t.append(tgt_specs[i])
        elif isinstance(arg, torch.SymInt):
            assert curr_specs[i] is None
            assert tgt_specs[i] is None
        else:
            raise ValueError("Unexpected local_map HOP argument")

    assert len(curr_specs_t) == len(tgt_specs_t)
    return curr_specs_t, tgt_specs_t


class ApplyShardingInterpreter(torch.fx.Interpreter):
    def __init__(
        self,
        module,
        sharding_placement,
        enable_ordered_sharding_optimization: bool = True,
    ):
        super().__init__(module, garbage_collect_values=True, graph=None)
        self.sharding_placement = sharding_placement
        param_placement_order = {}
        if enable_ordered_sharding_optimization:
            param_placement_order = compute_optimal_placement_order_for_parameters(
                module, sharding_placement
            )
        self.param_placement_order = param_placement_order

    def run_node(self, n):
        self._curr_node = n
        return super().run_node(n)

    def _get_input_nodes(self, node):
        # node.all_input_nodes deduplicates, but we need repeated nodes preserved.
        # Filter out shape-computation nodes (sym_size, operator.mul, etc.)
        # that produce scalars and have no sharding placement.
        return [n for n in all_input_nodes(node) if n in self.sharding_placement]

    def _set_origin_and_target_device_order(self, node, curr_spec, tgt_spec):
        # shard_order should be automatically assigned once `placements` is set
        assert curr_spec.shard_order is not None
        assert tgt_spec.shard_order is not None
        if node in self.param_placement_order:
            is_target_reversed_order, need_reorder = self.param_placement_order[node]
            curr_spec.shard_order = _compute_shard_order(
                curr_spec.shard_order,
                reverse=not (is_target_reversed_order and need_reorder),
            )
            tgt_spec.shard_order = _compute_shard_order(
                tgt_spec.shard_order,
                reverse=is_target_reversed_order,
            )

    def redistribute_tensor(self, arg, curr_spec, tgt_spec, node):
        tgt_placements = tuple(
            p if not p.is_partial() else Replicate() for p in tgt_spec.placements
        )
        x = arg
        if node in self.param_placement_order and self.param_placement_order[node][1]:
            assert curr_spec.placements != tgt_spec.placements
        self._set_origin_and_target_device_order(node, curr_spec, tgt_spec)
        if (
            curr_spec.placements != tgt_spec.placements
            or curr_spec.shard_order != tgt_spec.shard_order
        ):
            tgt_spec_c = DTensorSpec(
                tgt_spec.mesh,
                tgt_placements,
                tensor_meta=tgt_spec.tensor_meta,
                shard_order=tgt_spec.shard_order,
            )
            x = ordered_redistribute_local_tensor(
                arg,
                curr_spec,
                tgt_spec_c,
            )
        return x

    def _call_getitem(self, args):
        node = self._curr_node
        assert node.target == operator.getitem, f"Got {node.target}"
        idx = args[1]
        arg = args[0][idx]

        new_args_0 = list(args[0])
        if isinstance(arg, torch.Tensor):
            all_input_nodes = self._get_input_nodes(node)
            curr_spec = self.sharding_placement[all_input_nodes[0]].output_specs[idx]
            tgt_spec = self.sharding_placement[node].input_specs[0]
            new_args_0[idx] = self.redistribute_tensor(arg, curr_spec, tgt_spec, node)
        else:
            tgt_spec = None

        return [tuple(new_args_0), idx], tgt_spec

    def _redistribute_and_adjust_args(self, target, args):
        node = self._curr_node
        all_input_nodes = self._get_input_nodes(node)
        num_input_nodes = len(all_input_nodes)
        curr_specs = [
            self.sharding_placement[n].output_specs for n in all_input_nodes
        ]  # FIXME ?
        tgt_specs = [
            self.sharding_placement[node].input_specs[c] for c in range(num_input_nodes)
        ]

        flat_args, treespec = tree_flatten(args)
        flat_args_t = [x for x in flat_args if isinstance(x, torch.Tensor)]
        if len(flat_args_t) < len(flat_args) and "local_map" in node.name:
            curr_specs, tgt_specs = _filter_specs_for_local_map(
                flat_args, curr_specs, tgt_specs
            )

        assert len(flat_args_t) == len(curr_specs) == len(tgt_specs)
        last_tgt_spec = None
        new_flat_args_t = []
        for n, arg, curr_spec, tgt_spec in zip(
            all_input_nodes, flat_args_t, curr_specs, tgt_specs
        ):
            x = self.redistribute_tensor(arg, curr_spec, tgt_spec, node)
            new_flat_args_t.append(x)
            last_tgt_spec = tgt_spec

        new_flat_args = []
        counter = 0
        for x in flat_args:
            if isinstance(x, torch.Tensor):
                x = new_flat_args_t[counter]
                counter += 1
            new_flat_args.append(x)

        new_args = list(treespec.unflatten(new_flat_args))

        # apply sharding to constructor functions as well
        if target in TENSOR_FACTORY_OPS:
            # scalar_tensor has a scalar as first arg, not a shape
            if target != torch.ops.aten.scalar_tensor.default:
                val = list(new_args[0])
                spec = self.sharding_placement[node].output_specs
                for mesh_size, placement in zip(spec.mesh.shape, spec.placements):
                    if placement.is_shard():
                        # TODO: fix uneven cases ?
                        val[placement.dim] //= mesh_size
                new_args[0] = tuple(val)

        return new_args, last_tgt_spec

    def call_function(self, target, args, kwargs):
        if self._curr_node not in self.sharding_placement:
            # Shape-computation nodes (sym_size, operator.mul, etc.) produce
            # scalars, not tensors — just execute them directly.
            return super().call_function(target, tuple(args), kwargs)

        if self._curr_node.target == operator.getitem:
            new_args, tgt_spec = self._call_getitem(args)
        else:
            new_args, tgt_spec = self._redistribute_and_adjust_args(target, args)

        # View ops need special handling to convert global shape args to local.
        # For static shapes, DTensor machinery handles this conversion.
        # For dynamic shapes, we compute the local output shape using view_groups
        # dim mapping applied to the local input tensor's shape. This avoids
        # DTensor dispatch which can't handle symbolic expressions from make_fx.
        if target in _VIEW_OPS and tgt_spec is not None:
            if _args_have_symints(new_args):
                output_spec = self.sharding_placement[self._curr_node].output_specs
                # Get concrete global shapes from the joint graph's metadata.
                # The tensor input node is the first Node arg of the view node.
                input_node = [
                    n
                    for n in self._curr_node.all_input_nodes
                    if isinstance(n.meta.get("val"), torch.Tensor)
                ][0]
                global_input_shape = _concretize_shape(input_node.meta["val"].shape)
                global_output_shape = _concretize_shape(
                    self._curr_node.meta["val"].shape
                )
                local_input_shape = new_args[0].shape
                new_args[1] = _compute_local_view_shape(
                    global_input_shape,
                    global_output_shape,
                    local_input_shape,
                    output_spec,
                )
            else:
                new_args[0] = DTensor.from_local(
                    new_args[0], tgt_spec.mesh, tgt_spec.placements
                )
                # TODO: once `from_local` accept device order arg, we can remove the following
                new_args[0]._spec.shard_order = tgt_spec.shard_order

                # TODO: see if we can remove this contiguous properly
                new_args[0] = new_args[0].contiguous()

        out = super().call_function(target, tuple(new_args), kwargs)
        out = tree_map_only(DTensor, lambda x: x.to_local(), out)
        return out


def shard_node_given_placements(node, sharding_placement):
    tgt_spec = sharding_placement[node].input_specs[0]
    mesh = tgt_spec.mesh
    curr_placement = (Replicate(),) * mesh.ndim
    tensor = node.meta["val"]

    assert isinstance(
        tensor, FakeTensor
    ), f"only FakeTensor params supported for now, got {type(tensor)}"
    with unset_fake_temporarily():
        tensor = torch.empty(tensor.shape, dtype=tensor.dtype, device="meta")
        sharded_tensor = DTensor.from_local(tensor, mesh, curr_placement).redistribute(
            mesh, tgt_spec.placements
        )

    return sharded_tensor


def rename_placeholder_node(
    fx_g: torch.fx.GraphModule, node: torch.fx.Node, new_name: str
):
    assert node.op == "placeholder", f"only placeholder node supported, got {node.op}"
    with fx_g.graph.inserting_before(node):
        new_node = fx_g.graph.placeholder(new_name)
        new_node.meta.update(node.meta)
        node.replace_all_uses_with(new_node)
        fx_g.graph.erase_node(node)


def _get_inductor_decomp_table():
    decomp_table = copy.copy(select_decomp_table())
    # desugar our custom operator now that we've computed the sharding decision
    decomp_table[torch.ops.autoparallel.dtype_cast.default] = lambda x, dtype: x.to(
        dtype
    )
    return decomp_table


def _has_symbolic_shapes(gm):
    """Check if the graph has any placeholder nodes with symbolic (SymInt) dimensions."""
    for node in gm.graph.nodes:
        if node.op == "placeholder" and isinstance(node.meta.get("val"), torch.Tensor):
            for s in node.meta["val"].shape:
                if isinstance(s, torch.SymInt):
                    return True
    return False


def _make_local_args(gm, sharding_placement):
    """Create local FakeTensors for each placeholder by dividing sharded dims.

    Computes the local shape by dividing sharded dims by the mesh size
    (clean integer division, assumes even sharding). For dynamic shapes,
    SymInt dims produce symbolic local sizes (e.g., s0 // 32).
    """
    local_args = []
    for node in gm.graph.find_nodes(op="placeholder"):
        tensor = node.meta["val"]
        assert isinstance(
            tensor, FakeTensor
        ), f"expected FakeTensor placeholder, got {type(tensor)}"
        tgt_spec = sharding_placement[node].input_specs[0]
        mesh = tgt_spec.mesh

        local_shape = list(tensor.shape)
        for mesh_dim, placement in enumerate(tgt_spec.placements):
            if placement.is_shard():
                dim = placement.dim
                local_shape[dim] = local_shape[dim] // mesh.size(mesh_dim)

        with tensor.fake_mode:
            local = torch.empty(local_shape, dtype=tensor.dtype, device=tensor.device)
        local_args.append(local)
    return local_args


def _re_symbolize_graph(parallel_gm):
    """Re-trace a parallel graph to replace old SymInts with fresh symbols.

    The parallel graph produced by make_fx inside an existing FakeTensorMode
    carries SymInts from the joint graph's ShapeEnv. Inductor can't codegen
    these because they aren't derivable from the parallel graph's inputs.

    This re-traces the graph with make_fx(tracing_mode='symbolic') outside
    the FakeTensorMode, using real tensors on the original device. make_fx
    creates a fresh ShapeEnv with clean symbols tied to the new placeholders.
    """
    concrete_args = []
    for node in parallel_gm.graph.nodes:
        if node.op != "placeholder":
            continue
        val = node.meta["val"]
        if isinstance(val, torch.Tensor):
            shape = tuple(
                s.node.hint if isinstance(s, torch.SymInt) else s for s in val.shape
            )
            concrete_args.append(torch.empty(shape, dtype=val.dtype, device=val.device))
        else:
            concrete_args.append(val)

    with unset_fake_temporarily():
        return make_fx(
            parallel_gm, tracing_mode="symbolic", _allow_non_fake_inputs=True
        )(*concrete_args)


def _lower_to_parallel_graph(gm, sharding_placement, local_args, dynamic=False):
    """Two-pass lowering: interpret with sharding collectives, then decompose."""
    decomp_table = _get_inductor_decomp_table()

    interp = ApplyShardingInterpreter(gm, sharding_placement)

    tracing_mode = "symbolic" if dynamic else "real"

    with fx_traceback.preserve_node_meta():
        parallel_gm0 = make_fx(interp.run, tracing_mode=tracing_mode)(*local_args)
    cleanup_graph(parallel_gm0)

    interp2 = torch.fx.Interpreter(parallel_gm0)
    with fx_traceback.preserve_node_meta():
        parallel_gm = make_fx(
            interp2.run,
            decomposition_table=decomp_table,
            tracing_mode=tracing_mode,
        )(*local_args)
    cleanup_graph(parallel_gm)

    if dynamic:
        parallel_gm = _re_symbolize_graph(parallel_gm)
        cleanup_graph(parallel_gm)

    return parallel_gm


def _copy_descriptors_and_rename_placeholders(source_gm, target_gm):
    """Copy node descriptors from source graph and rename placeholders to match."""
    for n1, n2 in zip(
        (n for n in source_gm.graph.nodes if n.op in ("placeholder", "output")),
        (n for n in target_gm.graph.nodes if n.op in ("placeholder", "output")),
    ):
        n2.meta["desc"] = n1.meta["desc"]
        if n2.op == "placeholder":
            n2.target = n1.target
            # node renaming is needed for partitioner as it searches for tangent
            # nodes. See https://fburl.com/kc4jtc3t for one case where it's used
            rename_placeholder_node(target_gm, n2, n1.name)
    target_gm.recompile()


def _shard_params_and_buffers(gm, sharding_placement, params_spec, buffers_spec):
    """Shard parameters and buffers according to the sharding placement."""
    # NB: ok to NOT use the parallel_gm here because we will just reapply the
    # correct sharding placement via sharding_placement
    fqn_to_param = get_named_param_nodes(gm.graph)
    fqn_to_buffer = get_named_buffer_nodes(gm.graph)

    sharded_param_dict = {}
    for fqn in params_spec:
        n = fqn_to_param[fqn]
        with unset_fake_temporarily():
            sharded_param_dict[fqn] = nn.Parameter(
                shard_node_given_placements(n, sharding_placement)
            )
            tgt_spec = sharding_placement[n].input_specs[0]
            sharded_param_dict[fqn]._spec.shard_order = tgt_spec.shard_order

    sharded_buffer_dict = {}
    for fqn in buffers_spec:
        n = fqn_to_buffer[fqn]
        sharded_buffer_dict[fqn] = shard_node_given_placements(n, sharding_placement)

    return sharded_param_dict, sharded_buffer_dict


def apply_sharding_to_model(gm, sharding_placement, params_spec, buffers_spec):
    t0 = time.perf_counter()
    dynamic = _has_symbolic_shapes(gm)
    local_args = _make_local_args(gm, sharding_placement)
    t1 = time.perf_counter()

    parallel_gm = _lower_to_parallel_graph(gm, sharding_placement, local_args, dynamic)
    t2 = time.perf_counter()

    _copy_descriptors_and_rename_placeholders(gm, parallel_gm)
    t3 = time.perf_counter()

    sharded_param_dict, sharded_buffer_dict = _shard_params_and_buffers(
        gm, sharding_placement, params_spec, buffers_spec
    )
    t4 = time.perf_counter()

    logger.debug(
        "apply_sharding_to_model breakdown: "
        "shard_inputs=%.3fs, lower_graph=%.3fs, "
        "rename=%.3fs, shard_params=%.3fs",
        t1 - t0,
        t2 - t1,
        t3 - t2,
        t4 - t3,
    )

    return parallel_gm, sharded_param_dict, sharded_buffer_dict
