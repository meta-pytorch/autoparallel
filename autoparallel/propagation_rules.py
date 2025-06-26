# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import collections
import copy
import itertools
import math
import operator

import torch
from torch.distributed._tensor.placement_types import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import OpSchema, OpSpec, OpStrategy
from torch.distributed.tensor._ops._view_ops import (
    RuntimeSchemaInfo,
    dim_maps,
    dim_transpose,
    propagate_shape_and_sharding,
    register_op_strategy_map,
)
from torch.distributed.tensor._ops.utils import generate_redistribute_costs
from torch.distributed.tensor.placement_types import Replicate, Shard

# TODO: move this to PyTorch
dim_maps[torch.t] = lambda input: dim_transpose(input.ndim, -2, -1)

register_op_strategy_map(
    torch.ops.aten.t.default, torch.t, schema_info=RuntimeSchemaInfo(1)
)


_op_rules = {}


def register_rule(op):
    global _op_rules

    def wrapper(impl):
        _op_rules[op] = impl
        return impl

    return wrapper


_op_partial_rules = {}


def register_opschema_rule(op):
    global _op_partial_rules

    def wrapper(impl):
        _op_partial_rules[op] = impl
        return impl

    return wrapper


def _gen_tensor_meta(shape, dtype=None):
    if isinstance(shape, torch.Tensor):
        empty_tensor = shape
    else:
        if dtype is None:
            dtype = torch.float32
        empty_tensor = torch.empty(shape, dtype=dtype, device="meta")
    return TensorMeta(
        empty_tensor.shape,
        empty_tensor.stride(),
        empty_tensor.dtype,
    )


def _build_meta_tensor(tensor_meta):
    return torch.empty_strided(
        tensor_meta.shape, tensor_meta.stride, dtype=tensor_meta.dtype, device="meta"
    )


def _create_all_options_no_nested_sharding(mesh, shape, tensor_meta=None):
    if tensor_meta is None:
        tensor_meta = _gen_tensor_meta(shape)
    # TODO: take partial into account as well?
    possible_options = [-1] + list(range(mesh.ndim))
    all_options = list(
        itertools.product(*[possible_options for _ in range(len(shape))])
    )
    # print(list(all_options))
    strats = []
    for op in all_options:
        c = collections.Counter(op)
        # print("here", op,c)
        if any(count > 1 for obj, count in c.most_common() if obj != -1):
            # print("skipping ", op, c)
            continue
        spec = DTensorSpec.from_dim_map(mesh, op, [], tensor_meta)
        strats.append(OpSpec(spec, input_specs=[spec], redistribute_cost=[[0.0]]))
    return OpStrategy(strats)


def _create_all_options(mesh, shape, tensor_meta=None, tensor=None):
    # TODO: clean up shape / tensor_meta / tensor mess
    if tensor is not None:
        assert tensor_meta is None
        assert shape == tensor.shape
        tensor_meta = TensorMeta(tensor.shape, tensor.stride(), tensor.dtype)
    if tensor_meta is None:
        tensor_meta = _gen_tensor_meta(shape)
    # TODO: take partial into account as well?
    possible_options = [Replicate()] + [Shard(i) for i in range(len(shape))]
    all_options = list(itertools.product(*[possible_options for _ in range(mesh.ndim)]))
    strats = []
    for placement in all_options:
        spec = DTensorSpec(mesh, placement, tensor_meta=tensor_meta)
        strats.append(OpSpec(spec, input_specs=[spec], redistribute_cost=[[0.0]]))
    return OpStrategy(strats)


@register_rule(operator.getitem)
def getitem_rule(mesh, specs):
    op_spec = specs[0]
    index = specs[1]
    strats = []
    new_inp = OpStrategy(
        [
            OpSpec(strat.output_specs[index], input_specs=strat.output_specs)
            for strat in op_spec.strategies
        ]
    )
    for strat in op_spec.strategies:
        input_specs = strat.output_specs
        output_specs = input_specs[index]
        redistribute_costs = [generate_redistribute_costs(new_inp, output_specs)]
        # TODO: fix this to take input_specs as argument
        # this will require fixing apply_sharding as well, see other TODO
        # s = OpSpec(output_specs, input_specs=input_specs)
        s = OpSpec(output_specs, input_specs=(output_specs,))
        # s.redistribute_cost = [[0.0]] * len(input_specs)
        # s.redistribute_cost[index] = redistribute_costs
        s.redistribute_cost = redistribute_costs
        strats.append(s)
    return OpStrategy(strats)


@register_rule(torch.ops.aten.view.default)
def view_rule(mesh, specs):
    # reverting https://github.com/pytorch/pytorch/pull/149764
    # as it would raise errors on some cases
    op_spec = specs[0]
    shape = specs[1]
    strats = []
    dim_map = dim_maps[torch.Tensor.view]
    rules = dim_map(op_spec, shape)
    global_shape = op_spec.shape
    in_tensor = _build_meta_tensor(op_spec.strategies[0].output_specs.tensor_meta)
    out_tensor = torch.ops.aten.view.default(in_tensor, shape)
    out_tensor_meta = _gen_tensor_meta(out_tensor)
    for strat in op_spec.strategies:
        input_specs = strat.output_specs
        input_tgt_placements, output_placements = propagate_shape_and_sharding(
            input_specs.placements, global_shape, rules, mesh.shape, strict_view=False
        )

        input_tgt_spec = DTensorSpec(
            placements=tuple(input_tgt_placements),
            mesh=mesh,
            tensor_meta=input_specs.tensor_meta,
        )
        output_spec = DTensorSpec(
            mesh=mesh,
            placements=tuple(output_placements),
            tensor_meta=out_tensor_meta,
        )

        redistribute_costs = [generate_redistribute_costs(op_spec, input_tgt_spec)]
        s = OpSpec(
            output_spec,
            input_specs=(input_tgt_spec,),
            redistribute_cost=redistribute_costs,
        )
        strats.append(s)
    return OpStrategy(strats)


@register_rule(torch.ops.aten.alias.default)
def alias_rule(mesh, specs):
    op_spec = specs[0]
    strats = []
    tensor_meta = op_spec.strategies[0].output_specs.tensor_meta
    all_ops = _create_all_options(mesh, tensor_meta.shape, tensor_meta)
    # for strat in op_spec.strategies:
    for strat in all_ops.strategies:
        input_specs = strat.output_specs
        output_specs = input_specs
        redistribute_costs = [generate_redistribute_costs(op_spec, output_specs)]
        s = OpSpec(output_specs, input_specs=(input_specs,))
        s.redistribute_cost = redistribute_costs
        strats.append(s)
    return OpStrategy(strats)


@register_rule(torch.ops.aten.unbind.int)
def unbind_rule(mesh, specs):
    op_spec = specs[0]
    dim = specs[1]
    strats = []

    banned_idxs = set()
    for i, ss in enumerate(op_spec.strategies):
        for placement in ss.output_spec.placements:
            if placement.is_shard(dim) or placement.is_partial():
                banned_idxs.add(i)
    for strat in op_spec.strategies:
        input_specs = strat.output_spec
        tensor_meta = input_specs.tensor_meta
        inp_t = _build_meta_tensor(tensor_meta)
        out_ts = inp_t.unbind(dim)
        placements = input_specs.placements
        if any(p.is_shard(dim) or p.is_partial() for p in placements):
            continue
        output_specs = tuple(
            DTensorSpec(mesh, placements, tensor_meta=_gen_tensor_meta(out_t))
            for out_t in out_ts
        )

        redistribute_costs = generate_redistribute_costs(op_spec, output_specs[0])
        for banned in banned_idxs:
            redistribute_costs[banned] = math.inf

        s = OpSpec(output_specs, input_specs=(input_specs,))
        s.redistribute_cost = [redistribute_costs]
        strats.append(s)
    return OpStrategy(strats)


@register_rule(torch.ops.aten.split_with_sizes.default)
def split_with_sizes_rule(mesh, specs):
    op_spec = specs[0]
    sizes = specs[1]
    dim = 0
    if len(specs) > 2:
        # TODO: kwargs!!!!
        dim = specs[2]
    strats = []

    banned_idxs = set()
    for i, ss in enumerate(op_spec.strategies):
        for placement in ss.output_spec.placements:
            if placement.is_shard(dim) or placement.is_partial():
                banned_idxs.add(i)
    for strat in op_spec.strategies:
        input_specs = strat.output_spec
        tensor_meta = input_specs.tensor_meta
        inp_t = _build_meta_tensor(tensor_meta)
        out_ts = inp_t.split(sizes, dim)
        placements = input_specs.placements
        if any(p.is_shard(dim) or p.is_partial() for p in placements):
            continue
        output_specs = tuple(
            DTensorSpec(mesh, placements, tensor_meta=_gen_tensor_meta(out_t))
            for out_t in out_ts
        )

        redistribute_costs = generate_redistribute_costs(op_spec, output_specs[0])
        for banned in banned_idxs:
            redistribute_costs[banned] = math.inf

        s = OpSpec(output_specs, input_specs=(input_specs,))
        s.redistribute_cost = [redistribute_costs]
        strats.append(s)
    return OpStrategy(strats)


@register_rule(torch.ops.aten.cat.default)
def cat_rule(mesh, specs):
    op_spec = specs[0]
    dim = specs[1]
    strats = []

    num_tensors = len(op_spec)

    inp_ts = []
    for i in range(num_tensors):
        tm = op_spec[i].strategies[0].output_spec.tensor_meta
        inp_ts.append(_build_meta_tensor(tm))

    out_t = torch.cat(inp_ts, dim)

    banned_idxs = set()
    for i, ss in enumerate(op_spec[0].strategies):
        for placement in ss.output_spec.placements:
            if placement.is_shard(dim) or placement.is_partial():
                banned_idxs.add(i)

    for i in range(1, num_tensors):
        assert len(op_spec[i].strategies) == len(
            op_spec[0].strategies
        ), "Assume each cat input has same number of strategies"

    for strat_idx, strat in enumerate(op_spec[0].strategies):
        placements = strat.output_spec.placements
        if any(p.is_shard(dim) or p.is_partial() for p in placements):
            continue
        output_spec = DTensorSpec(mesh, placements, tensor_meta=_gen_tensor_meta(out_t))

        all_costs = []
        input_specs = []
        for i in range(num_tensors):
            input_specs.append(op_spec[i].strategies[strat_idx].output_spec)
            redistribute_costs = generate_redistribute_costs(op_spec[i], output_spec)
            for banned in banned_idxs:
                redistribute_costs[banned] = math.inf
            all_costs.append(redistribute_costs)

        s = OpSpec(output_spec, input_specs=input_specs)
        s.redistribute_cost = all_costs
        strats.append(s)
    return OpStrategy(strats)


@register_rule(torch.ops.prims.iota.default)
def iota_rule(mesh, specs):
    raise NotImplementedError("Needs hardening, only tested on a few cases")
    shape = [specs[0]]
    tensor_meta = _gen_tensor_meta(shape, dtype=torch.int64)
    placement = (Replicate(),) * mesh.ndim
    spec = DTensorSpec(mesh, placement, tensor_meta=tensor_meta)
    # return OpStrategy([OpSpec(spec, input_specs=[spec], redistribute_cost=[[0.0]])])
    return OpStrategy([OpSpec(spec, input_specs=[spec], redistribute_cost=[[0.0]])])


@register_rule(torch.ops.aten.randperm.default)
def randperm_rule(mesh, specs):
    raise NotImplementedError("Needs hardening, only tested on a few cases")
    shape = [specs[0]]
    tensor_meta = _gen_tensor_meta(shape, dtype=torch.int64)
    placement = (Replicate(),) * mesh.ndim
    spec = DTensorSpec(mesh, placement, tensor_meta=tensor_meta)
    return OpStrategy([OpSpec(spec, input_specs=[spec], redistribute_cost=[[0.0]])])


@register_rule(torch.ops.aten.full.default)
def full_rule(mesh, specs):
    raise NotImplementedError("Needs hardening, only tested on a few cases")
    shape = specs[0]
    # TODO: get the dtype
    tensor_meta = _gen_tensor_meta(shape)
    # TODO: I'm hard-coding this here, I'll probably need to do something else about this
    placement = (Shard(0),) + (Replicate(),) * (mesh.ndim - 1)
    # placement = (Replicate(),) * mesh.ndim
    input_placement = (Replicate(),) * mesh.ndim
    spec = DTensorSpec(mesh, placement, tensor_meta=tensor_meta)
    input_spec = DTensorSpec(mesh, input_placement, tensor_meta=tensor_meta)
    # return OpStrategy([OpSpec(spec, input_specs=[spec], redistribute_cost=[[0.0]])])
    return OpStrategy(
        [OpSpec(spec, input_specs=[input_spec], redistribute_cost=[[0.0]])]
    )


# ======================================
# the following ops require meta_tensor fix


@register_opschema_rule(torch.ops.aten.native_layer_norm.default)
def native_layer_norm_rule(mesh, op_schema):
    from torch.distributed.tensor._ops._math_ops import (
        Sequence,
        normalize_to_torch_size,
    )
    from torch.distributed.tensor._ops._pointwise_ops import pointwise_strategy

    # mesh = op_schema.get_mesh_from_args()
    # args must be: input, normalized_shape, weight, bias, eps
    # for None weight and bias, their corresponding objects will
    # be None as well. layer_norm_strategy returns one OpStrategy
    # for the triple return values (out, mean, rstd).
    assert len(op_schema.args_schema) == 5
    (
        input_strategy,
        normalized_shape,
        weight_strategy,
        bias_strategy,
        _,
    ) = op_schema.args_schema

    # the current layer norm implementation requires that all
    # input DTensor's sharding must be in form of OpStrategy
    assert isinstance(input_strategy, OpStrategy)
    assert isinstance(normalized_shape, (int, Sequence, torch.Size))
    normalized_size = normalize_to_torch_size(normalized_shape)

    input_ndim = input_strategy.ndim
    axis = input_ndim - len(normalized_size)

    output_strategy = pointwise_strategy(op_schema, linearity=False)

    # now let's remove the cases that are invalid, as they require
    # reduction on a sharded dimension
    kept = []
    for strategy in output_strategy.strategies:
        is_valid = True
        for plc in strategy.input_specs[0].placements:
            if plc.is_shard() and plc.dim >= axis:
                is_valid = False
                break
        if is_valid:
            output_spec = strategy.output_specs
            input_spec = strategy.input_specs[0]
            output_spec.tensor_meta = input_spec.tensor_meta
            assert output_spec.tensor_meta is not None
            mesh = strategy.mesh
            # the output spec is the same as input spec
            shape = input_spec.tensor_meta.shape[:axis] + (1,) * len(normalized_size)
            mean_std_tgt_spec = DTensorSpec(
                mesh=mesh,
                placements=output_spec.placements,
                tensor_meta=_gen_tensor_meta(shape),
            )
            output_target_spec = (
                output_spec,
                mean_std_tgt_spec,
                mean_std_tgt_spec,
            )
            if len(output_target_spec) == 1:
                output_target_spec = output_target_spec[0]
            strategy.output_specs = output_target_spec
            kept.append(strategy)

    return OpStrategy(kept)


@register_opschema_rule(torch.ops.aten.native_layer_norm_backward.default)
def native_layer_norm_backward_rule(mesh, op_schema):
    from torch.distributed.tensor._ops._math_ops import (
        Sequence,
        normalize_to_torch_size,
    )
    from torch.distributed.tensor._ops._pointwise_ops import pointwise_strategy

    assert len(op_schema.args_schema) == 8
    (
        grad_out_strategy,
        input_strategy,
        normalized_shape,
        mean_strategy,
        rstd_strategy,
        weight_strategy,
        bias_strategy,
        _,
    ) = op_schema.args_schema

    assert isinstance(input_strategy, OpStrategy)
    assert isinstance(normalized_shape, (int, Sequence, torch.Size))
    normalized_size = normalize_to_torch_size(normalized_shape)

    input_ndim = input_strategy.ndim
    axis = input_ndim - len(normalized_size)

    output_strategy = pointwise_strategy(op_schema, linearity=False)

    # now let's remove the cases that are invalid, as they require
    # reduction on a sharded dimension
    kept = []
    for strategy in output_strategy.strategies:
        is_valid = True
        input_spec = strategy.input_specs[1]
        for plc in input_spec.placements:
            if plc.is_shard() and plc.dim >= axis:
                is_valid = False
                break
        if is_valid:
            mesh = strategy.mesh
            grad_input_spec = DTensorSpec(
                mesh=mesh,
                placements=strategy.output_specs.placements,
                tensor_meta=strategy.output_specs.tensor_meta,
            )
            weight_spec = strategy.input_specs[4]
            bias_spec = strategy.input_specs[5]
            grad_input_spec.tensor_meta = input_spec.tensor_meta
            assert grad_input_spec.tensor_meta is not None
            weight_tgt_spec = DTensorSpec(
                mesh=mesh,
                placements=weight_spec.placements,
                tensor_meta=weight_spec.tensor_meta,
            )
            bias_tgt_spec = DTensorSpec(
                mesh=mesh,
                placements=bias_spec.placements,
                tensor_meta=bias_spec.tensor_meta,
            )
            output_target_spec = (
                grad_input_spec,
                weight_tgt_spec,
                bias_tgt_spec,
            )
            if len(output_target_spec) == 1:
                output_target_spec = output_target_spec[0]
            strategy.output_specs = output_target_spec
            kept.append(strategy)

    return OpStrategy(kept)


@register_opschema_rule(torch.ops.prims.convert_element_type.default)
def convert_element_type_rule(mesh, op_schema):
    from torch.distributed.tensor._ops._tensor_ops import default_strategy

    # TODO: API has changed in latest main
    out_strat = default_strategy(mesh, op_schema)
    return out_strat


@register_opschema_rule(torch.ops.aten.split.Tensor)
def split_rule(mesh, op_schema):
    strat = op_schema.args_schema
    op = torch.ops.aten.split.Tensor
    from torch.distributed.tensor._ops._tensor_ops import split_rule

    res = []
    oo = []
    for i, ss in enumerate(strat[0].strategies):
        ispec = ss.input_specs[0]
        assert ss.output_spec == ispec
        o = split_rule(OpSchema(op, (ispec, strat[1], strat[2]), {}))
        # res.append(o)
        oo.append(o)
        if o.output_spec is not None:
            s = OpSpec(o.output_spec, input_specs=(ispec,))
            s.redistribute_cost = [[math.inf] * len(ss.redistribute_cost[0])]
            # s.redistribute_cost = [[0.0] * len(ss.redistribute_cost[0])]
            s.redistribute_cost[0][i] = 0.0
            res.append(s)

    out_strat = OpStrategy(res)
    return out_strat


@register_opschema_rule(torch.ops.aten._unsafe_index.Tensor)
def _unsafe_index_rule(mesh, op_schema):
    raise NotImplementedError()


@register_opschema_rule(torch.ops.aten.index.Tensor)
def index_rule(mesh, op_schema):
    raise NotImplementedError("Needs hardening, only tested on a few cases")
    strat = op_schema.args_schema
    specs = strat  # TODO: clean this up
    res = []
    idxs_placements = [(Replicate(), Replicate()), (Shard(0), Replicate())]
    if strat[1].childs[0] is None:
        idxs_placements = idxs_placements[:1]
    else:
        idxs_placements = idxs_placements[1:]
    # TODO: this is a nasty hack and won't work for most of the cases
    for i, ss in enumerate(strat[0].strategies):
        for plt in idxs_placements:
            ispec = ss.input_specs[0]
            ospec = DTensorSpec(mesh=mesh, placements=ispec.placements)
            assert ss.output_spec == ispec
            idxs_strats = [
                DTensorSpec(mesh, placements=plt)
                for x in strat[1].childs
                if x is not None
            ]
            if len(idxs_strats) == 2:
                # TODO: VERY NASTY HACK
                idxs_strats[1] = DTensorSpec(
                    mesh, placements=(Replicate(), Replicate())
                )
            kspc = [x for x in strat[1].childs if x is not None]
            s = OpSpec(output_specs=ospec, input_specs=[ispec] + idxs_strats)

            redistribute_costs = [generate_redistribute_costs(specs[0], ospec),] + [
                generate_redistribute_costs(kk, idxs_strat)
                for kk, idxs_strat in zip(kspc, idxs_strats)
            ]
            s.redistribute_cost = redistribute_costs

            res.append(s)
    out_strat = OpStrategy(res)
    return out_strat


@register_opschema_rule(torch.ops.aten.index_put.default)
def index_put_rule(mesh, op_schema):
    raise NotImplementedError("Needs hardening, only tested on a few cases")
    strat = op_schema.args_schema
    specs = strat  # TODO: clean this up
    res = []
    idxs_placements = [(Replicate(), Replicate()), (Shard(0), Replicate())]
    if strat[1].childs[0] is None:
        idxs_placements = idxs_placements[:1]
    else:
        idxs_placements = idxs_placements[1:]
    # TODO: this is a nasty hack and won't work for most of the cases
    for i, ss in enumerate(strat[0].strategies):
        for plt in idxs_placements:
            ispec = ss.input_specs[0]
            ospec = DTensorSpec(mesh=mesh, placements=ispec.placements)
            assert ss.output_spec == ispec, f"{ss.output_spec}, {ispec}"
            idxs_strats = [
                DTensorSpec(mesh, placements=plt)
                for x in strat[1].childs
                if x is not None
            ]
            kspc = [x for x in strat[1].childs if x is not None]
            t_strats = [DTensorSpec(mesh, placements=ispec.placements)]
            s = OpSpec(output_specs=ospec, input_specs=[ispec] + idxs_strats + t_strats)

            redistribute_costs = (
                [generate_redistribute_costs(specs[0], ospec)]
                + [
                    generate_redistribute_costs(kk, idxs_strat)
                    for kk, idxs_strat in zip(kspc, idxs_strats)
                ]
                + [generate_redistribute_costs(specs[2], t_strats[0])]
            )
            s.redistribute_cost = redistribute_costs
            res.append(s)
    out_strat = OpStrategy(res)
    return out_strat


@register_opschema_rule(torch.ops.aten._scaled_dot_product_efficient_attention.default)
def sdpa_rule(mesh, op_schema):
    op = torch.ops.aten._scaled_dot_product_efficient_attention.default
    out_strat = torch.distributed.tensor.DTensor._op_dispatcher.sharding_propagator.op_strategy_funcs[
        op
    ](
        op_schema
    )
    # remove wrong context-parallel strategy
    # https://github.com/pytorch/pytorch/pull/131351#discussion_r1716164659
    new_strats = []
    for ss in out_strat.strategies:
        if (
            torch.distributed.tensor.placement_types.Shard(2)
            not in ss.input_specs[0].placements
        ):
            os = ss.output_specs
            new_os = []
            for s in os:
                # replace None with Replicate() in the output_spec
                # as this is done by default but somewhere further
                # down the line in DTensor
                if s is None:
                    s = DTensorSpec(mesh=mesh, placements=(Replicate(),) * mesh.ndim)
                new_os.append(s)
            ss.output_specs = tuple(new_os)
            new_strats.append(ss)
    out_strat.strategies = new_strats
    return out_strat


@register_opschema_rule(torch.ops.aten.reshape.default)
def reshape_rule(mesh, op_schema):
    op = torch.ops.aten.reshape.default
    out_strat = torch.distributed.tensor.DTensor._op_dispatcher.sharding_propagator.op_strategy_funcs[
        op
    ](
        op_schema
    )
    if mesh.ndim == 1:
        # remove duplicate strategy
        # TODO: hack, fixme
        if len(out_strat.strategies) > 2 and str(out_strat.strategies[2]) == str(
            out_strat.strategies[0]
        ):
            print("removing")
            out_strat.strategies.pop(2)
    return out_strat


@register_opschema_rule(torch.ops.aten.expand.default)
def expand_rule(mesh, op_schema_):
    op = torch.ops.aten.expand.default
    op_schema = copy.deepcopy(op_schema_)
    input_strat = op_schema.args_schema[0]
    orig_shape = input_strat.strategies[0].output_specs.tensor_meta.shape
    dest_shape = op_schema.args_schema[1]
    expand_dim = [
        i
        for i, (s1, s2) in enumerate(zip(orig_shape, dest_shape))
        if s1 == 1 and s2 != s1
    ]
    if len(expand_dim) != 1:
        assert len(expand_dim) == 0
        return torch.distributed.tensor.DTensor._op_dispatcher.sharding_propagator.op_strategy_funcs[
            op
        ](
            op_schema
        )
    assert len(expand_dim) == 1, f"{expand_dim}"
    expand_dim = expand_dim[0]
    to_remove = []
    for i, ss in enumerate(input_strat.strategies):
        for plc in ss.output_spec.placements:
            if plc.is_shard(expand_dim):
                # need to remove this and add back afterwards
                to_remove.append(i)
                break

    removed = []
    for i in reversed(to_remove):
        removed.append(input_strat.strategies.pop(i))
    out_strat = torch.distributed.tensor.DTensor._op_dispatcher.sharding_propagator.op_strategy_funcs[
        op
    ](
        op_schema
    )
    for i, ss in enumerate(out_strat.strategies):
        for remov in to_remove:
            ss.redistribute_cost[0].insert(remov, math.inf)
    return out_strat
