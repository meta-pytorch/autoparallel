# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import copy
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


def _concretize_shape(shape):
    """Concretize a shape tuple, replacing SymInts with their hint values."""
    from autoparallel.optimize_sharding import concretize_symint

    return tuple(concretize_symint(s) for s in shape)


def _localize_shape_arg(node, shape_arg, output_spec):
    """Convert a global shape arg to local by dividing sharded dims.

    Computes the local shape from the global shape in node.meta["val"],
    dividing sharded dims by the corresponding mesh sizes. SymInt values
    in shape_arg (computed from local tensors via sym_size/mul nodes)
    are preserved as-is since they are already local.
    """
    global_shape = _concretize_shape(node.meta["val"].shape)
    local_shape = list(global_shape)
    for mesh_size, placement in zip(output_spec.mesh.shape, output_spec.placements):
        if placement.is_shard():
            dim = placement.dim
            local_shape[dim] = (local_shape[dim] + mesh_size - 1) // mesh_size
    # Restore SymInt values from the interpreter (already local)
    for i, s in enumerate(shape_arg):
        if isinstance(s, torch.SymInt):
            local_shape[i] = s
    return local_shape


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


class ApplyShardingInterpreter(torch.fx.Interpreter):
    def __init__(
        self,
        module,
        sharding_placement,
        enable_ordered_sharding_optimization: bool = True,
        dynamic: bool = False,
    ):
        super().__init__(module, garbage_collect_values=True, graph=None)
        self.sharding_placement = sharding_placement
        self.dynamic = dynamic
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
        # Filter out nodes without sharding entries:
        # - get_attr: HOP submodule nodes (GraphModules)
        # - call_function producing non-tensors: shape-computation nodes
        #   (sym_size, operator.mul, etc.)
        result = []
        for x in all_input_nodes(node):
            if x in self.sharding_placement:
                result.append(x)
            elif x.op != "get_attr":
                # call_function nodes not in sharding_placement should only be
                # scalar shape-computation nodes, not tensor producers.
                val = x.meta.get("val")
                assert not isinstance(val, torch.Tensor), (
                    f"Tensor-producing node {x} (op={x.op}) unexpectedly "
                    f"missing from sharding_placement"
                )
        return result

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
            # curr_spec/tgt_spec can be None for HOP-internal activations:
            # tensors with data-dependent shapes (unbacked SymInts) that are
            # saved by the forward local_map HOP and only consumed by the
            # backward local_map HOP. These don't need redistribution because
            # the HOP manages their distribution internally.
            if curr_spec is not None and tgt_spec is not None:
                new_args_0[idx] = self.redistribute_tensor(
                    arg, curr_spec, tgt_spec, node
                )
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
        if len(flat_args_t) < len(curr_specs):
            # HOPs have mixed arg types (tensors, SymInts, etc.).
            # Filter specs to tensor-only entries matching flat_args_t.
            filtered_nodes = []
            filtered_curr = []
            filtered_tgt = []
            for n, cs, ts in zip(all_input_nodes, curr_specs, tgt_specs):
                if ts is not None:
                    filtered_nodes.append(n)
                    filtered_curr.append(cs)
                    filtered_tgt.append(ts)
            all_input_nodes = filtered_nodes
            curr_specs = filtered_curr
            tgt_specs = filtered_tgt

        assert len(flat_args_t) == len(curr_specs) == len(tgt_specs)
        last_tgt_spec = None
        new_flat_args_t = []
        for n, arg, curr_spec, tgt_spec in zip(
            all_input_nodes, flat_args_t, curr_specs, tgt_specs
        ):
            if curr_spec is None or tgt_spec is None:
                # HOP-internal activations with data-dependent shapes (unbacked
                # SymInts). See comment in _call_getitem for details.
                new_flat_args_t.append(arg)
            else:
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

        # Localize shape args for factory and view ops.
        # In static mode, shape args are concrete global values baked into the
        # graph. In dynamic mode, SymInt args are already local (computed from
        # local tensors via sym_size/mul), but concrete args are still global.
        # _localize_shape_arg handles both: divides global shape by mesh size,
        # then preserves any SymInt values from the interpreter.
        if target in TENSOR_FACTORY_OPS:
            if target != torch.ops.aten.scalar_tensor.default:
                spec = self.sharding_placement[node].output_specs
                new_args[0] = tuple(_localize_shape_arg(node, new_args[0], spec))
        elif self.dynamic and target in _VIEW_OPS:
            spec = self.sharding_placement[node].output_specs
            new_args[1] = _localize_shape_arg(node, new_args[1], spec)

        # In static mode, view ops use DTensor wrapping to convert global
        # shape args to local (DTensor handles the global→local conversion
        # internally). Not needed in dynamic mode since shape args are already
        # localized above.
        if not self.dynamic and target in _VIEW_OPS and tgt_spec is not None:
            new_args[0] = DTensor.from_local(
                new_args[0], tgt_spec.mesh, tgt_spec.placements
            )
            new_args[0]._spec.shard_order = tgt_spec.shard_order
            new_args[0] = new_args[0].contiguous()

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


def _has_rank_varying_size(dim_idx, global_shape, spec):
    """Check if different ranks can have different local sizes for a tensor dim.

    Returns True when a Shard placement on dim_idx doesn't evenly divide the
    (effective) dim size, meaning some ranks get ceil(size/mesh) elements and
    others get fewer. Only meaningful for concrete (non-SymInt) dims — SymInt
    dims are already marked DYNAMIC by the caller.
    """
    size = global_shape[dim_idx]
    for mesh_size, placement in zip(spec.mesh.shape, spec.placements):
        if placement.is_shard() and placement.dim == dim_idx:
            if size % mesh_size != 0:
                return True
            size = size // mesh_size
    return False


def _make_local_args(gm, sharding_placement):
    """Create local tensors for each placeholder via DTensor redistribute.

    Uses DTensor's redistribute to compute correct local shapes and strides.
    When the FakeTensorMode has a ShapeEnv (dynamic shapes), re-creates the
    local tensors with fresh SymInts for batch-dependent dims. The caller
    must swap the ShapeEnv to a fresh one before calling this function.
    """
    from torch.fx.experimental.symbolic_shapes import (
        DimDynamic,
        StatelessSymbolicContext,
    )

    local_args = []
    for node in gm.graph.find_nodes(op="placeholder"):
        tensor = node.meta["val"]
        tgt_spec = sharding_placement[node].input_specs[0]
        mesh = tgt_spec.mesh
        curr_placement = (Replicate(),) * mesh.ndim

        # Use DTensor to compute the correct local shape and strides.
        # Concretize any SymInts before DTensor to avoid ShapeEnv conflicts.
        concrete_tensor = tensor
        if isinstance(tensor, FakeTensor) and any(
            isinstance(s, torch.SymInt) for s in tensor.shape
        ):
            with tensor.fake_mode:
                concrete_tensor = torch.empty_strided(
                    _concretize_shape(tensor.shape),
                    _concretize_shape(tensor.stride()),
                    dtype=tensor.dtype,
                    device=tensor.device,
                )

        sharded = DTensor.from_local(
            concrete_tensor, mesh, curr_placement
        ).redistribute(mesh, tgt_spec.placements)
        local = sharded.to_local()

        # For dynamic shapes, re-create with fresh SymInts.
        # A dim is DYNAMIC if it's genuinely symbolic (a free SymInt variable
        # like the batch dim, not a guarded model constant like hidden_dim
        # whose expr collapsed to a number), or if uneven sharding causes
        # rank-varying local sizes.
        if isinstance(tensor, FakeTensor) and tensor.fake_mode.shape_env is not None:
            dynamic_sizes = [
                DimDynamic.DYNAMIC
                if (isinstance(s, torch.SymInt) and not s.node.expr.is_number)
                or _has_rank_varying_size(i, tensor.shape, tgt_spec)
                else DimDynamic.STATIC
                for i, s in enumerate(tensor.shape)
            ]
            # Use unset_fake_temporarily so torch.empty creates a real meta
            # tensor. Inside the active fake mode, torch.empty would produce
            # a FakeTensor that from_tensor returns from cache, ignoring
            # symbolic_context.
            with unset_fake_temporarily():
                real = torch.empty(
                    _concretize_shape(local.shape),
                    dtype=local.dtype,
                    device="meta",
                    requires_grad=tensor.requires_grad,
                )
            sym_ctx = StatelessSymbolicContext(dynamic_sizes=dynamic_sizes)
            local = tensor.fake_mode.from_tensor(real, symbolic_context=sym_ctx)
            with tensor.fake_mode:
                local = local.to(tensor.device)

        local_args.append(local)
    return local_args


def _lower_to_parallel_graph(gm, sharding_placement, local_args, dynamic=False):
    """Two-pass lowering: interpret with sharding collectives, then decompose."""
    decomp_table = _get_inductor_decomp_table()

    interp = ApplyShardingInterpreter(gm, sharding_placement, dynamic=dynamic)

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

    # For dynamic shapes, swap ShapeEnv to a fresh one so that _make_local_args
    # creates FakeTensors with fresh SymInts, and make_fx propagates them
    # through the parallel graph. This produces symbols derivable from the
    # parallel graph's own placeholders, which Inductor can codegen.
    # The new ShapeEnv is kept (not restored) because the parallel graph's
    # metadata references its symbols. update_joint_with_descriptors copies
    # this metadata into joint_with_descriptors, making the old ShapeEnv unused.
    if dynamic:
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        # Find the shared FakeTensorMode and swap its ShapeEnv.
        # All placeholder FakeTensors must share the same mode.
        fake_mode = None
        for node in gm.graph.find_nodes(op="placeholder"):
            val = node.meta.get("val")
            if isinstance(val, FakeTensor):
                if fake_mode is None:
                    fake_mode = val.fake_mode
                else:
                    assert (
                        val.fake_mode is fake_mode
                    ), "All placeholder FakeTensors must share the same FakeTensorMode"
        if fake_mode is not None:
            fake_mode.shape_env = ShapeEnv()
            fake_mode.static_shapes = False

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
