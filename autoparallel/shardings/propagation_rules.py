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
import logging
import math
import operator

import torch
from torch.distributed._tensor.placement_types import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import OpSchema, OpSpec, OpStrategy
from torch.distributed.tensor._ops._view_ops import (
    dim_maps,
    propagate_shape_and_sharding,
)
from torch.distributed.tensor._ops.utils import (
    generate_redistribute_costs,
    is_tensor_shardable,
)
from torch.distributed.tensor.placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
)

# need to import this to have the dtype_cast registered
from ..cast_parametrization import dtype_cast  # noqa
from .dtensor_sharding_helpers import _try_single_dim_strategy, get_op_strategy

logger = logging.getLogger(__name__)


_op_rules = {}


def register_rule(ops):
    global _op_rules

    def wrapper(impl):
        if isinstance(ops, list):
            for op in ops:
                _op_rules[op] = impl
        else:
            _op_rules[ops] = impl
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
                    if shape[dim] >= mesh_shape:
                        shape[dim] = (shape[dim] + mesh_shape - 1) // mesh_shape
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
def getitem_rule(mesh, op_schema):
    op_spec = op_schema.args_schema[0]
    index = op_schema.args_schema[1]
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
def view_rule(mesh, op_schema):
    op_spec = op_schema.args_schema[0]
    shape = op_schema.args_schema[1]
    strats = []
    dim_map = dim_maps[torch.Tensor.view]
    rules = dim_map(op_spec, shape)
    global_shape = op_spec.shape
    in_tensor = _build_meta_tensor(op_spec.strategies[0].output_specs.tensor_meta)
    out_tensor = torch.ops.aten.view.default(in_tensor, shape)
    out_tensor_meta = _gen_tensor_meta(out_tensor)
    for strat in op_spec.strategies:
        input_specs = strat.output_specs
        try:
            input_tgt_placements, output_placements = propagate_shape_and_sharding(
                input_specs.placements,
                global_shape,
                rules,
                mesh.shape,
                strict_view=False,
            )
        except AssertionError:
            # PyTorch may raise when a sharded dim is not divisible by the
            # mesh size (e.g. unflatten nheads=48 on mesh dim size=32).
            # With strict_view=False this should demote to Replicate, but
            # upstream validation is overly strict.  Skip this strategy;
            # the replicated variant is already covered by another iteration.
            continue

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


@register_rule(torch.ops.aten.view.dtype)
def view_dtype_rule(mesh, op_schema):
    """view(dtype=...) reinterprets the last dim's bytes as a different dtype.

    Shape changes only on the last dim (e.g. [N, 8] uint8 -> [N, 1] int64).
    Sharding on any non-last dim is preserved; sharding on the last dim is
    banned since the byte layout changes.
    """
    op_spec = op_schema.args_schema[0]
    target_dtype = op_schema.args_schema[1]
    in_meta = op_spec.strategies[0].output_specs.tensor_meta
    in_tensor = _build_meta_tensor(in_meta)
    out_tensor = torch.ops.aten.view.dtype(in_tensor, target_dtype)
    out_meta = _gen_tensor_meta(out_tensor)
    last_dim = in_tensor.ndim - 1

    strats = []
    for strat in op_spec.strategies:
        input_specs = strat.output_specs
        if any(p.is_shard(last_dim) for p in input_specs.placements):
            continue
        output_spec = DTensorSpec(mesh, input_specs.placements, tensor_meta=out_meta)
        redistribute_costs = [generate_redistribute_costs(op_spec, input_specs)]
        s = OpSpec(output_spec, input_specs=(input_specs,))
        s.redistribute_cost = redistribute_costs
        strats.append(s)
    return OpStrategy(strats)


@register_rule(torch.ops.aten.select.int)
def select_int_rule(mesh, op_schema):
    """Wrapper around DTensor's select_int_strategy that fixes two issues:

    1. Negative dim/index: DTensor doesn't handle negative values.
       We normalize them before delegating.

    2. Shared output/input specs: DTensor sets ``output_specs = input_specs``
       (same object) when the selected dim is not sharded. Then
       ``propagate_tensor_meta`` overwrites the shared spec's tensor_meta,
       corrupting the input node's strategy. We copy the shared spec.
    """
    import dataclasses

    from torch.distributed.tensor._op_schema import OpSchema

    input_strategy, dim, index = op_schema.args_schema
    ndim = input_strategy.ndim
    if isinstance(dim, int) and dim < 0:
        dim = dim + ndim
    if isinstance(index, int) and index < 0:
        index = index + input_strategy.shape[dim]
    if dim != op_schema.args_schema[1] or index != op_schema.args_schema[2]:
        op_schema = OpSchema(
            op_schema.op,
            (input_strategy, dim, index),
            op_schema.kwargs_schema,
            op_schema.schema_info,
        )

    out_strat = get_op_strategy(torch.ops.aten.select.int, op_schema)
    for s in out_strat.strategies:
        if s.output_specs is not None and s.input_specs is not None:
            for ispec in s.input_specs:
                if ispec is not None and ispec is s.output_specs:
                    s.output_specs = dataclasses.replace(s.output_specs)
                    break
    return out_strat


@register_rule(torch.ops.aten.alias.default)
def alias_rule(mesh, op_schema):
    op_spec = op_schema.args_schema[0]
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


@register_rule(torch.ops.aten.split_with_sizes.default)
def split_with_sizes_rule(mesh, op_schema):
    op_spec = op_schema.args_schema[0]
    sizes = op_schema.args_schema[1]
    dim = 0
    if len(op_schema.args_schema) > 2:
        dim = op_schema.args_schema[2]
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
def iota_rule(mesh, op_schema):
    # Replicate-only: iota's output values are position-dependent
    # ([0, 1, ..., length-1]), so Shard would require adjusting `start`
    # per rank. Replicate is correct and cheap (small index tensors).
    shape = [op_schema.args_schema[0]]
    tensor_meta = _gen_tensor_meta(shape, dtype=torch.int64)
    placement = (Replicate(),) * mesh.ndim
    spec = DTensorSpec(mesh, placement, tensor_meta=tensor_meta)
    return OpStrategy([OpSpec(spec, input_specs=[spec], redistribute_cost=[[0.0]])])


@register_rule(torch.ops.aten.randperm.default)
def randperm_rule(mesh, op_schema):
    raise NotImplementedError("Needs hardening, only tested on a few cases")
    shape = [op_schema.args_schema[0]]
    tensor_meta = _gen_tensor_meta(shape, dtype=torch.int64)
    placement = (Replicate(),) * mesh.ndim
    spec = DTensorSpec(mesh, placement, tensor_meta=tensor_meta)
    return OpStrategy([OpSpec(spec, input_specs=[spec], redistribute_cost=[[0.0]])])


# Factory ops that take a shape as the first argument
_SHAPE_FACTORY_OPS = [
    torch.ops.aten.zeros.default,
    torch.ops.aten.ones.default,
    torch.ops.aten.full.default,
    torch.ops.aten.empty.memory_format,
    torch.ops.aten.rand.default,
    torch.ops.aten.randn.default,
]

# We do a few special things for factory ops
# - use the factory rule below
# - fake that they have input schemas so the solver doesn't freak out
# - convert their sizes to 'local tensor sizes' (divide by mesh dim) during ApplySharding
TENSOR_FACTORY_OPS = _SHAPE_FACTORY_OPS + [
    torch.ops.aten.scalar_tensor.default,  # Special case: creates 0-dim tensor
]


@register_rule(torch.ops.aten.scalar_tensor.default)
def scalar_tensor_rule(mesh, op_schema: OpSchema) -> OpStrategy:
    """
    Rule for aten.scalar_tensor which creates a scalar (0-dimensional) tensor.
    Unlike other factory ops, this doesn't take a shape parameter.

    Schema: scalar_tensor(Scalar s, *, ScalarType? dtype=None, ...) -> Tensor
    """
    # scalar_tensor creates a 0-dimensional tensor
    shape = ()
    stride = ()
    dtype = torch.get_default_dtype()

    # Check if dtype is specified in kwargs or args
    if len(op_schema.args_schema) >= 2 and op_schema.args_schema[1] is not None:
        dtype = op_schema.args_schema[1]  # type: ignore[assignment]

    tensor_meta = TensorMeta(shape, stride, dtype)  # type: ignore[arg-type]

    # For a scalar (0-dim) tensor, we can only replicate across all mesh dimensions
    placement = (Replicate(),) * mesh.ndim
    output_specs = DTensorSpec(mesh, placement, tensor_meta=tensor_meta)

    # Similar to factory_rule, we add a dummy input_specs for solver compatibility
    strategy = OpSpec(
        output_specs=output_specs,
        input_specs=[output_specs],
        redistribute_cost=[[0.0]],
    )

    return OpStrategy([strategy])


@register_rule(_SHAPE_FACTORY_OPS)
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


@register_rule(torch.ops.autoparallel.dtype_cast.default)
def dtype_cast_rule(mesh, op_schema):
    from torch.distributed.tensor._ops._tensor_ops import (
        propagate_single_input_strategy,
    )

    out_strat = propagate_single_input_strategy(op_schema)
    return out_strat


@register_rule(torch.ops.aten.split.Tensor)
def split_rule(mesh, op_schema):
    strat = op_schema.args_schema
    op = torch.ops.aten.split.Tensor
    from torch.distributed.tensor._ops._tensor_ops import split_strategy

    op_schema = OpSchema(op, (strat[0], strat[1], strat[2]), {})
    upstream_strat = split_strategy(op_schema)

    res = []
    n_input_strats = len(strat[0].strategies)
    for i, os in enumerate(upstream_strat.strategies):
        s = OpSpec(os.output_specs, input_specs=os.input_specs)
        s.redistribute_cost = [[math.inf] * n_input_strats]
        s.redistribute_cost[0][i] = 0.0
        res.append(s)

    out_strat = OpStrategy(res)
    return out_strat


@register_rule(torch.ops.aten._unsafe_index.Tensor)
def _unsafe_index_rule(mesh, op_schema):
    raise NotImplementedError()


@register_rule(torch.ops.aten.reshape.default)
def reshape_rule(mesh, op_schema):
    op = torch.ops.aten.reshape.default
    out_strat = get_op_strategy(op, op_schema)
    if mesh.ndim == 1:
        # remove duplicate strategy
        # TODO: hack, fixme
        if len(out_strat.strategies) > 2 and str(out_strat.strategies[2]) == str(
            out_strat.strategies[0]
        ):
            logger.debug("removing")
            out_strat.strategies.pop(2)
    return out_strat


@register_rule(torch.ops.aten.expand.default)
def expand_rule(mesh, op_schema_):
    op = torch.ops.aten.expand.default
    from torch._subclasses.fake_tensor import unset_fake_temporarily

    with unset_fake_temporarily():
        op_schema = copy.deepcopy(op_schema_)
    input_strat = op_schema.args_schema[0]
    orig_shape = input_strat.strategies[0].output_specs.tensor_meta.shape
    dest_shape = op_schema.args_schema[1]
    expand_dims = [
        i
        for i, (s1, s2) in enumerate(zip(orig_shape, dest_shape))
        if s1 == 1 and s2 != s1
    ]
    if len(expand_dims) == 0:
        return get_op_strategy(op, op_schema)
    to_remove = []
    for expand_dim in expand_dims:
        for i, ss in enumerate(input_strat.strategies):
            for plc in ss.output_spec.placements:
                if plc.is_shard(expand_dim) and i not in to_remove:
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


def _einsum_single_dim_strategy(op, args_schema, kwargs_schema):
    from torch.distributed.tensor._ops._matrix_ops import (
        gen_single_dim_einsum_strategies,
    )

    equation, input_metas = args_schema
    assert isinstance(equation, str)
    assert isinstance(input_metas, tuple)
    assert len(input_metas) == 2, "Only two args to einsum supported for now"

    # ignore strategies with Partial inputs.
    return [
        strategy
        for strategy in gen_single_dim_einsum_strategies(equation)
        if not any(isinstance(placement, Partial) for placement in strategy[1:])
    ]


def _register_einsum_single_dim_strategy():
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor._ops.single_dim_strategy import (
        register_single_dim_strategy,
    )

    op = torch.ops.aten.einsum.default
    propagator = DTensor._op_dispatcher.sharding_propagator
    if op not in propagator.op_single_dim_strategy_funcs:
        register_single_dim_strategy(op, allow_unbacked_sharding=True)(
            _einsum_single_dim_strategy
        )


_register_einsum_single_dim_strategy()


@register_rule(torch.ops.aten.einsum.default)
def einsum_rule(mesh, op_schema):
    from torch.distributed.tensor._op_schema import TupleStrategy

    mm_equation, mat_strategy = op_schema.args_schema
    assert isinstance(mm_equation, str)
    assert isinstance(mat_strategy, TupleStrategy)

    assert len(mat_strategy.children) == 2, "Only two args to einsum supported for now"

    out_strat = _try_single_dim_strategy(torch.ops.aten.einsum.default, op_schema)
    assert out_strat is not None
    return out_strat


@register_rule(torch.ops.aten.scatter.src)
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


@register_rule(torch.ops.aten.stack.default)
def stack_strategy(mesh, op_schema: OpSchema):
    from torch.distributed.tensor._ops._tensor_ops import (
        PlacementList,
        TupleStrategy,
        cast,
        expand_to_full_mesh_op_strategy,
        normalize_dim,
    )

    input_tuple_strategy = op_schema.args_schema[0]
    assert isinstance(input_tuple_strategy, TupleStrategy)

    num_input_tensor = len(input_tuple_strategy.children)
    first_input_strategy = input_tuple_strategy.children[0]
    assert isinstance(first_input_strategy, OpStrategy)
    common_input_ndim = first_input_strategy.ndim

    dim = cast(int, op_schema.args_schema[1]) if len(op_schema.args_schema) > 1 else 0
    # normalize the dim to be within the common input ndim
    dim = normalize_dim(dim, common_input_ndim)

    possible_input_strategies: PlacementList = [Replicate()] + [  # type: ignore[assignment]
        Shard(i) for i in range(common_input_ndim)
    ]
    possible_output_strategies: PlacementList = (
        [Replicate()]  # type: ignore[assignment]
        + [Shard(i) for i in range(dim)]
        + [Shard(i + 1) for i in range(dim, common_input_ndim)]
    )

    single_mesh_dim_strategies = []
    for input_strategy, output_strategy in zip(
        possible_input_strategies, possible_output_strategies
    ):
        strategy: PlacementList = [output_strategy] + [
            input_strategy
        ] * num_input_tensor
        single_mesh_dim_strategies.append(strategy)

    s = expand_to_full_mesh_op_strategy(
        mesh, op_schema, single_mesh_dim_strategies, input_index=1
    )
    return s


# ======================================
# Convolution ops


@register_rule(torch.ops.aten.convolution_backward.default)
def convolution_backward_strategy(mesh, op_schema: OpSchema):
    from torch.distributed.tensor._ops.utils import expand_to_full_mesh_op_strategy

    bias_shape_opt = op_schema.args_schema[3]
    has_bias = bias_shape_opt is not None

    # Outputs: grad_input, grad_weight, grad_bias (3)
    # Inputs (tensor): grad_output, input, weight (3)
    # Placement list: [grad_input, grad_weight, grad_bias, grad_output, input, weight]
    single_mesh_dim_strategies: list[list[Placement | None]] = []

    if has_bias:
        single_mesh_dim_strategies.append([Replicate()] * 6)
        single_mesh_dim_strategies.append(
            [Shard(0), Partial(), Partial(), Shard(0), Shard(0), Replicate()]
        )
    else:
        single_mesh_dim_strategies.append(
            [Replicate(), Replicate(), None, Replicate(), Replicate(), Replicate()]
        )
        single_mesh_dim_strategies.append(
            [Shard(0), Partial(), None, Shard(0), Shard(0), Replicate()]
        )

    return expand_to_full_mesh_op_strategy(
        mesh, op_schema, single_mesh_dim_strategies, input_index=3
    )


# ======================================
# Random ops


@register_rule(torch.ops.aten.uniform.default)
def uniform_strategy(mesh, op_schema: OpSchema):
    from torch.distributed.tensor._ops.utils import expand_to_full_mesh_op_strategy

    input_strategy = op_schema.args_schema[0]
    assert isinstance(input_strategy, OpStrategy)
    ndim = input_strategy.ndim

    # [output, input] — any shard of input maps to same shard of output
    single_mesh_dim_strategies: list[list[Placement | None]] = [
        [Replicate(), Replicate()]
    ]
    for d in range(ndim):
        single_mesh_dim_strategies.append([Shard(d), Shard(d)])

    return expand_to_full_mesh_op_strategy(
        mesh, op_schema, single_mesh_dim_strategies, input_index=1
    )
