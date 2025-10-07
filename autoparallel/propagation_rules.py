# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
Custom sharding propagation rules for automatic parallelization.

This module extends and overrides PyTorch's DTensor operation rules to provide
custom sharding strategies specifically for autoparallel. All of these should
eventually be upstreamed to PyTorch proper.

Based on PyTorch DTensor implementation:
- Core DTensor ops: torch/distributed/tensor/_ops/
- Sharding propagation: torch/distributed/tensor/_sharding_prop.py
- Op strategies: torch/distributed/tensor/_op_schema.py
- Reference: https://pytorch.org/docs/stable/distributed.tensor.html
"""

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
from torch.distributed.tensor._ops.utils import (
    generate_redistribute_costs,
    is_tensor_shardable,
)
from torch.distributed.tensor.placement_types import Replicate, Shard

# need to import this to have the dtype_cast registered
from .cast_parametrization import dtype_cast  # noqa
from .dtensor_util import get_op_strategy

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


def register_opschema_rule(ops):
    global _op_partial_rules

    def wrapper(impl):
        if isinstance(ops, list):
            for op in ops:
                _op_partial_rules[op] = impl
        else:
            _op_partial_rules[ops] = impl
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


def remove_invalid_configs(out_strat, mesh):
    kept = []
    for strategy in out_strat.strategies:
        is_valid = True
        output_specs = strategy.output_specs
        if isinstance(output_specs, DTensorSpec):
            output_specs = [output_specs]
        if strategy.input_specs is not None:
            if output_specs is None:
                specs = list(strategy.input_specs)
            else:
                specs = list(strategy.input_specs) + list(output_specs)
        else:
            # special case for ops like full, empty, which have no inputs. See further comments by `TENSOR_FACTORY_OPS`
            specs = list(output_specs)

        for spec in specs:
            if spec is None:
                continue
            shape = list(spec.tensor_meta.shape)
            for mesh_shape, plc in zip(mesh.shape, spec.placements):
                if plc.is_shard():
                    dim = plc.dim
                    if shape[dim] % mesh_shape == 0:
                        shape[dim] //= mesh_shape
                    else:
                        is_valid = False
                        break
        if is_valid:
            kept.append(strategy)

    return OpStrategy(kept)


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
    out_strats = OpStrategy(strats)
    out_strats = remove_invalid_configs(out_strats, mesh)
    return out_strats


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
    out_strats = OpStrategy(strats)
    out_strats = remove_invalid_configs(out_strats, mesh)
    return out_strats


# For when dst_spec is None
def generate_dummy_redistribute_costs(src_strategy: OpStrategy) -> list[float]:
    return [0.0] * len(src_strategy.strategies)


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
        if output_specs is None:
            # if getitem doesn't return a tensor, there are no costs
            redistribute_costs = [generate_dummy_redistribute_costs(new_inp)]
        else:
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


# We do a few special things for factory ops
# - use the factory rule below
# - fake that they have input schemas so the solver doesn't freak out
# - convert their sizes to 'local tensor sizes' (divide by mesh dim) during ApplySharding
TENSOR_FACTORY_OPS = [
    torch.ops.aten.zeros.default,
    torch.ops.aten.ones.default,
    torch.ops.aten.full.default,
    torch.ops.aten.empty.memory_format,
    torch.ops.aten.rand.default,
    torch.ops.aten.randn.default,
]


@register_opschema_rule(TENSOR_FACTORY_OPS)
def factory_rule(mesh, op_schema: OpSchema) -> OpStrategy:
    """
    This is an auto-parallel specific util that won't be upstreamed becuase of a UX mismatch.

    In regular DTensor programs, a user has to either call `torch.full` to get a regular tensor, or
    `torch.distributed.tensor.full` (with placements specified) to get a DTensor.

    There is no point registering a strategy in DTensor for factories like 'full' since there is no way they
    could be used by DTensor's dispatching logic.  (Note: DTensor does provide strategies for similar ops like
    'new_full' and 'full_like', the difference being there is an input tensor to trigger dispatch off of and to
    use to direct the placement options.)

    This util applies to any factory function that takes 'size' as the first argument,
    and supports Replication and Shard placements all at zero cost.
    """
    assert isinstance(op_schema.args_schema[0], (torch.Size, list))
    shape = op_schema.args_schema[0]
    x = torch.empty(shape, device="meta")
    stride = x.stride()
    dtype = torch.get_default_dtype()
    if len(op_schema.args_schema) >= 3:
        assert isinstance(op_schema.args_schema[2], torch.dtype)
        dtype = op_schema.args_schema[2]
        assert isinstance(dtype, torch.dtype), dtype

    # TODO: ensure the solver knows that it is more expensive to Replicate factory functions than shard
    # for now, put replicate last since this might encourage sharding.  (Experimentally it seemed so, but definitely
    # this is not a stable gaurantee and we should fix this properly.)
    single_mesh_dim_strategies = [[Shard(i)] for i in range(len(shape))] + [
        [Replicate()]
    ]

    """
    Expand the single_mesh_dim_strategies to full mesh dim strategies.
    see docs for `expand_to_full_mesh_op_strategy` in _tensor_ops.py in pytorch
    """
    all_mesh_dim_strategies = [single_mesh_dim_strategies] * mesh.ndim

    strategy_combs = list(itertools.product(*all_mesh_dim_strategies))

    all_strategies = []
    for strategy_comb in strategy_combs:
        spec_list = [DTensorSpec(mesh, specs) for specs in zip(*strategy_comb)]
        output_specs = spec_list[0]
        output_specs.tensor_meta = TensorMeta(shape, stride, dtype)  # type: ignore[arg-type]

        if not is_tensor_shardable(shape, output_specs):
            continue

        redistribute_cost = [
            # TODO: there shouldn't actually be a row here, since there is no input to the op and the rows correspond
            # to the inputs. However, the optimization code is not set up to tolerate input-less ops, so hack around it
            # (see "/data/users/whc/autoparallel/autoparallel/optimize_sharding.py", line 226, in walk_over_options)
            [0.0]
            * len(strategy_combs)
        ]

        # NOTE: why do we have input_specs for constructor nodes, given that they have no inputs?
        # This is because the optimizer code expects to see input_specs for all nodes, and it
        # uses the input_specs to determine the sharding of the output.  So we have to give it
        # something, even though it is in principle not needed.
        strategy = OpSpec(
            output_specs=output_specs,
            input_specs=[output_specs],
            redistribute_cost=redistribute_cost,
        )
        all_strategies.append(strategy)
    return OpStrategy(all_strategies)


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


@register_opschema_rule(
    [
        torch.ops.prims.convert_element_type.default,
        torch.ops.autoparallel.dtype_cast.default,
    ]
)
def convert_element_type_rule(mesh, op_schema):
    from torch.distributed.tensor._ops._tensor_ops import (
        propagate_single_input_strategy,
    )

    out_strat = propagate_single_input_strategy(op_schema)
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


def sdpa_rule(op, mesh, op_schema):
    out_strat = get_op_strategy(op, op_schema)
    # remove wrong context-parallel strategy
    # https://github.com/pytorch/pytorch/pull/131351#discussion_r1716164659
    new_strats = []
    for ss in out_strat.strategies:
        if (
            torch.distributed.tensor.placement_types.Shard(2)
            not in ss.input_specs[0].placements
        ):
            new_strats.append(ss)
    out_strat.strategies = new_strats
    return out_strat


@register_opschema_rule(torch.ops.aten._scaled_dot_product_efficient_attention.default)
def _(mesh, op_schema):
    op = torch.ops.aten._scaled_dot_product_efficient_attention.default
    return sdpa_rule(op, mesh, op_schema)


@register_opschema_rule(torch.ops.aten._scaled_dot_product_flash_attention.default)
def _(mesh, op_schema):
    op = torch.ops.aten._scaled_dot_product_flash_attention.default
    return sdpa_rule(op, mesh, op_schema)


@register_opschema_rule(
    torch.ops.aten._scaled_dot_product_flash_attention_backward.default
)
def _(mesh, op_schema):
    op = torch.ops.aten._scaled_dot_product_flash_attention_backward.default
    return sdpa_rule(op, mesh, op_schema)


@register_opschema_rule(
    torch.ops.aten._scaled_dot_product_efficient_attention_backward.default
)
def _(mesh, op_schema):
    op = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default
    return sdpa_rule(op, mesh, op_schema)


@register_opschema_rule(torch.ops.aten.reshape.default)
def reshape_rule(mesh, op_schema):
    op = torch.ops.aten.reshape.default
    out_strat = get_op_strategy(op, op_schema)
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
    from torch._subclasses.fake_tensor import unset_fake_temporarily

    with unset_fake_temporarily():
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
        return get_op_strategy(op, op_schema)
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
    out_strat = get_op_strategy(op, op_schema)
    for i, ss in enumerate(out_strat.strategies):
        for remov in to_remove:
            ss.redistribute_cost[0].insert(remov, math.inf)
    return out_strat


@register_opschema_rule(torch.ops.aten.einsum.default)
def einsum_rule(mesh, op_schema):
    from torch.distributed.tensor._op_schema import TupleStrategy
    from torch.distributed.tensor._ops._matrix_ops import _mm_like_strategy

    mm_equation, mat_strategy = op_schema.args_schema
    assert isinstance(mm_equation, str)
    assert isinstance(mat_strategy, TupleStrategy)

    assert len(mat_strategy.children) == 2, "Only two args to einsum supported for now"

    self_strategy, mat2_strategy = mat_strategy.children

    # dispatch to mm_like_strategy
    new_op_schema = OpSchema(
        torch.ops.aten.einsum.default,
        args_schema=(self_strategy, mat2_strategy),
        kwargs_schema={},
    )
    return _mm_like_strategy(mm_equation, mesh, new_op_schema)


@register_opschema_rule(torch.ops.aten.scatter.src)
def scatter_strategy(mesh, op_schema: OpSchema):
    # taken from scatter_add strategy from PyTorch
    from torch.distributed.tensor._ops._tensor_ops import (
        PlacementList,
        expand_to_full_mesh_op_strategy,
        normalize_dim,
    )

    input_strategy = op_schema.args_schema[0]
    dim = op_schema.args_schema[1]
    index_strategy = op_schema.args_schema[2]

    assert isinstance(input_strategy, OpStrategy)
    assert isinstance(index_strategy, OpStrategy)
    assert isinstance(dim, int)
    dim = normalize_dim(dim, input_strategy.ndim)
    mesh = input_strategy.mesh
    input_shape = input_strategy.shape
    index_shape = index_strategy.shape

    single_mesh_dim_strategies = []

    # placement list stores placements of [output, input, index, src]
    # first we always have replicate all for inputs and output
    all_replicate: PlacementList = [Replicate()] * 4
    single_mesh_dim_strategies.append(all_replicate)

    if len(input_shape) == len(index_shape):
        for d in range(len(input_shape)):
            if d != dim and input_shape[d] == index_shape[d]:
                sharding: PlacementList = [Shard(d), Shard(d), Shard(d), Shard(d)]
                single_mesh_dim_strategies.append(sharding)

    return expand_to_full_mesh_op_strategy(
        mesh, op_schema, single_mesh_dim_strategies, input_index=1
    )
