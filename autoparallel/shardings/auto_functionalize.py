# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""Sharding support for ``auto_functionalized_v2`` nodes.

``aot_export_module`` functionalizes a custom op that declares ``mutates_args`` into a
call to the higher-order op ``auto_functionalized_v2`` (see
``torch/_higher_order_ops/auto_functionalize.py``). The node looks like::

    auto_functionalized_v2(op, <non-mutated arg>=..., _<mutated arg>_base_index=i,
                           ..., _all_bases=[base_0, base_1, ...])

    value = (op returns..., updated base_0, updated base_1, ...)

so, unlike every other node AutoParallel sees:

* ``node.args`` is just ``(op,)``; every tensor input is in ``node.kwargs``;
* a *mutated* argument is not an input at all: it is regenerated from ``_all_bases``, and the
  node's value carries the updated bases after the op's own returns.

The wrapped op's sharding rule is the one registered with DTensor (``register_sharding`` /
``register_op_strategy``) for the op's own schema, in which the mutated tensor is an ordinary
argument. This module rebuilds that call from the node, asks the DTensor registry for the
strategy, and maps it back: input specs are laid out in ``all_input_nodes`` order, the op's
returns keep the op's output specs, and each updated base takes the placement its mutated
argument was given.

Nothing here refers to a particular op's signature: it is derived from the op's schema and
the node's structure, so an op that later turns functional (no mutated arguments, no bases)
goes through the same code.
"""

import contextlib
import dataclasses
from typing import Any, Iterator, Optional

import torch
import torch.utils._pytree as pytree
from torch._higher_order_ops.auto_functionalize import (
    AliasViewInfo,
    NotView,
    get_mutable_args_from_schema,
    read_view_information_from_args,
)
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import (
    OpSchema,
    OpSpec,
    OpStrategy,
    RuntimeSchemaInfo,
    TupleStrategy,
)
from torch.distributed.tensor._ops.utils import generate_redistribute_costs
from torch.distributed.tensor.placement_types import Replicate

from .dtensor_sharding_helpers import get_op_strategy, with_implicit_strategies
from .propagation_rules import _op_rules, remove_invalid_configs

_BASES = "_all_bases"

# One input slot of an auto_functionalized_v2 node: a key identifying where the tensor
# sits in the node (("base", i) or (kwarg name,) or (kwarg name, i) for a tensor list).
SlotKey = tuple


def is_auto_functionalized_v2(node: torch.fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target is torch.ops.higher_order.auto_functionalized_v2
    )


@dataclasses.dataclass(frozen=True)
class AutoFunctionalizedCall:
    """The wrapped op of an ``auto_functionalized_v2`` node, decoded.

    ``op``: the wrapped custom op. ``bases``: the ``_all_bases`` nodes. ``base_of_arg``:
    mutated schema argument name -> index into ``bases``. ``replicate_only``: some mutated
    argument is a strict view of its base (slice / as_strided), or two mutated arguments
    share a base; the base then has a different shape than the argument the op rule is
    written for, so the node is only given the all-replicated strategy.
    """

    op: torch._ops.OpOverload
    bases: tuple[Any, ...]
    base_of_arg: dict[str, int]
    replicate_only: bool

    @property
    def schema(self) -> torch.FunctionSchema:
        return self.op._schema


def decode_auto_functionalized_v2(node: torch.fx.Node) -> AutoFunctionalizedCall:
    if not is_auto_functionalized_v2(node):
        raise ValueError(f"{node} is not an auto_functionalized_v2 node")
    op = node.args[0]
    if not isinstance(op, torch._ops.OpOverload):
        raise NotImplementedError(
            f"auto_functionalized_v2 of {op!r}: only OpOverload targets are supported"
        )
    if any(isinstance(x, torch.fx.Node) for x in pytree.tree_leaves(node.args)):
        raise NotImplementedError(
            f"{node}: tensors in auto_functionalized_v2 positional args"
        )
    schema = op._schema
    kwargs = dict(node.kwargs)
    bases = tuple(kwargs.pop(_BASES, ()))
    names, types = get_mutable_args_from_schema(schema)
    if torch._library.utils.is_out(op):
        # out= variant: write-only buffers, described by _<name>_size/_stride, no bases.
        raise NotImplementedError(
            f"auto_functionalized_v2 of out= op {op}: not supported"
        )
    for name, typ in zip(names, types):
        if torch._library.utils.is_tensorlist_like_type(typ):
            raise NotImplementedError(
                f"{op}: mutated tensor-list argument {name!r} is not supported"
            )
    view_info = read_view_information_from_args(names, types, kwargs, list(bases))
    base_of_arg: dict[str, int] = {}
    replicate_only = False
    for name in names:
        info = view_info[name]
        if info is None:  # Optional[Tensor] passed as None
            continue
        if not isinstance(info, (NotView, AliasViewInfo)):
            replicate_only = True
        base_of_arg[name] = info.base_index
    if len(set(base_of_arg.values())) != len(base_of_arg):
        replicate_only = True
    return AutoFunctionalizedCall(op, bases, base_of_arg, replicate_only)


def _is_strategy(x: Any) -> bool:
    return isinstance(x, OpStrategy)


def input_slots(kwarg_specs: dict[str, Any]) -> list[tuple[SlotKey, OpStrategy]]:
    """The tensor inputs of the node, in ``all_input_nodes`` order, with their producers'
    strategies. ``kwarg_specs`` is ``node.kwargs`` with every node that the solver tracks
    replaced by its ``OpStrategy``."""
    slots: list[tuple[SlotKey, OpStrategy]] = []
    for name, value in kwarg_specs.items():
        if isinstance(value, (list, tuple)):
            for i, x in enumerate(value):
                if _is_strategy(x):
                    key: SlotKey = ("base", i) if name == _BASES else (name, i)
                    slots.append((key, x))
        elif _is_strategy(value):
            slots.append(((name,), value))
    return slots


def _tensor_meta(t: Any) -> Optional[TensorMeta]:
    if not isinstance(t, torch.Tensor):
        return None
    return TensorMeta(t.shape, t.stride(), t.dtype)


def _slot_key_of_arg(call: AutoFunctionalizedCall, name: str) -> SlotKey:
    if name in call.base_of_arg:
        return ("base", call.base_of_arg[name])
    return (name,)


def _inner_op_schema(
    call: AutoFunctionalizedCall,
    kwarg_specs: dict[str, Any],
    user_kwargs: dict[str, Any],
) -> tuple[OpSchema, list[SlotKey]]:
    """The call ``op(*args, **kwargs)`` the node wraps, with strategies for tensors, and
    the slot of every tensor input in the order DTensor lists them in ``input_specs``.
    """
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    slots: list[SlotKey] = []
    needs_pytree = False
    for arg in call.schema.arguments:
        name = arg.name
        value: Any
        if name in call.base_of_arg:
            i = call.base_of_arg[name]
            value = kwarg_specs[_BASES][i]
            slots.append(("base", i))
        elif name in kwarg_specs:
            value = kwarg_specs[name]
            if isinstance(value, (list, tuple)) and any(_is_strategy(x) for x in value):
                value = TupleStrategy(tuple(value))
                needs_pytree = True
                slots.extend((name, i) for i in range(len(value.children)))
            elif _is_strategy(value):
                slots.append((name,))
            else:
                value = user_kwargs[name]
        elif arg.has_default_value():
            value = arg.default_value
        else:
            # A mutated Optional[Tensor] passed as None, or an arg AOT left out.
            value = None
        if arg.kwarg_only:
            kwargs[name] = value
        else:
            args.append(value)
    schema = OpSchema(
        call.op, tuple(args), kwargs, RuntimeSchemaInfo(needs_pytree=needs_pytree)
    )
    return schema, slots


def _as_spec_list(output_specs: Any) -> list[Optional[DTensorSpec]]:
    if output_specs is None:
        return []
    if isinstance(output_specs, DTensorSpec):
        return [output_specs]
    return list(output_specs)


@contextlib.contextmanager
def _not_inplace_for_strategy() -> Iterator[None]:
    """Make ``OpSchema.is_inplace_op`` false while a strategy is generated.

    DTensor calls an op "in place" by its name (a trailing ``_``) and then drops every
    strategy whose placement for the first argument differs from the placement that argument
    *currently has*: eager DTensor cannot redistribute an operand it is about to overwrite.
    Inside ``auto_functionalized_v2`` the mutation is functional (the mutated tensor is a
    base, copied or re-inplaced later), and the solver, not the runtime, chooses the
    placement of every input, so that restriction would only discard valid strategies.
    """
    original = OpSchema.is_inplace_op
    OpSchema.is_inplace_op = lambda self: False  # type: ignore[method-assign]
    try:
        yield
    finally:
        OpSchema.is_inplace_op = original  # type: ignore[method-assign]


def get_auto_functionalized_placement_option(
    mesh, node: torch.fx.Node, kwarg_specs: dict[str, Any], user_kwargs: dict[str, Any]
) -> OpStrategy:
    """OpStrategy of an ``auto_functionalized_v2`` node.

    Outputs: the wrapped op's returns, then one entry per updated base (the node's value is
    ``(*op returns, *updated bases)``), each updated base placed like the mutated argument it
    is regenerated into. The mutation is therefore placement-preserving by construction.
    """
    call = decode_auto_functionalized_v2(node)
    slots = input_slots(kwarg_specs)
    val = node.meta["val"]
    n_bases = len(call.bases)
    n_returns = len(val) - n_bases
    base_metas = [_tensor_meta(b.meta["val"]) for b in call.bases]
    ret_metas = [_tensor_meta(v) for v in val[:n_returns]]

    def make(
        in_specs: dict[SlotKey, DTensorSpec], ret_specs: list[Optional[DTensorSpec]]
    ) -> OpSpec:
        outs: list[Optional[DTensorSpec]] = []
        for j, meta in enumerate(ret_metas):
            spec = ret_specs[j] if j < len(ret_specs) else None
            if meta is None:
                outs.append(None)
                continue
            if spec is None:
                spec = DTensorSpec(mesh, (Replicate(),) * mesh.ndim)
            outs.append(DTensorSpec(mesh, spec.placements, tensor_meta=meta))
        for i, meta in enumerate(base_metas):
            src = in_specs[("base", i)]
            outs.append(DTensorSpec(mesh, src.placements, tensor_meta=meta))
        ordered_in = [in_specs[key] for key, _ in slots]
        return OpSpec(
            output_specs=tuple(outs),
            input_specs=tuple(ordered_in),
            redistribute_cost=[
                generate_redistribute_costs(strategy, spec)
                for (_, strategy), spec in zip(slots, ordered_in)
            ],
        )

    if call.replicate_only:
        replicated = (Replicate(),) * mesh.ndim
        in_specs = {
            key: DTensorSpec(
                mesh,
                replicated,
                tensor_meta=strategy.strategies[0].output_spec.tensor_meta,
            )
            for key, strategy in slots
        }
        return OpStrategy([make(in_specs, [])])

    op_schema, inner_slots = _inner_op_schema(call, kwarg_specs, user_kwargs)
    with _not_inplace_for_strategy():
        if call.op in _op_rules:
            inner = _op_rules[call.op](mesh, op_schema)
        else:
            with with_implicit_strategies():
                inner = get_op_strategy(call.op, op_schema)
    if isinstance(inner, TupleStrategy) or not isinstance(inner, OpStrategy):
        raise NotImplementedError(f"{call.op}: strategy of type {type(inner)}")

    slot_strategy = dict(slots)
    strategies = []
    for s in inner.strategies:
        if s.input_specs is None or len(s.input_specs) != len(inner_slots):
            raise NotImplementedError(
                f"{call.op}: strategy has {None if s.input_specs is None else len(s.input_specs)} "
                f"input specs for {len(inner_slots)} tensor inputs"
            )
        in_specs = {}
        for key, spec in zip(inner_slots, s.input_specs):
            if spec.tensor_meta is None:
                spec = DTensorSpec(
                    mesh,
                    spec.placements,
                    tensor_meta=slot_strategy[key]
                    .strategies[0]
                    .output_spec.tensor_meta,
                )
            in_specs[key] = spec
        strategies.append(make(in_specs, _as_spec_list(s.output_specs)))
    out = remove_invalid_configs(OpStrategy(strategies), mesh)
    if not out.strategies:
        raise RuntimeError(
            f"{node}: no valid sharding of {call.op} for mesh shape {tuple(mesh.shape)}"
        )
    return out
