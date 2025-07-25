# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.distributed._tensor.placement_types import TensorMeta
from torch.distributed.device_mesh import _get_device_handle
from torch.distributed.tensor._op_schema import OpSchema, OpStrategy, TupleStrategy
from torch.distributed.tensor._ops.utils import generate_redistribute_costs
from torch.utils._pytree import tree_flatten, tree_map_only

from .propagation_rules import (
    TENSOR_FACTORY_OPS,
    _op_partial_rules,
    _op_rules,
    remove_invalid_configs,
)


def _get_meta_tensors_for_op(op, user_args, user_kwargs):
    out_t = op(*user_args, **user_kwargs)

    if isinstance(out_t, torch.Tensor):
        out_tensor_meta = TensorMeta(out_t.shape, out_t.stride(), out_t.dtype)
    else:
        out_tensor_meta = tree_map_only(
            torch.Tensor, lambda x: TensorMeta(x.shape, x.stride(), x.dtype), out_t
        )

    input_tensor_metas = tree_flatten(user_args)[0]
    input_tensor_metas = tree_map_only(
        torch.Tensor,
        lambda x: TensorMeta(x.shape, x.stride(), x.dtype),
        input_tensor_metas,
    )
    input_tensor_metas = tuple(
        x for x in input_tensor_metas if isinstance(x, TensorMeta)
    )
    return out_tensor_meta, input_tensor_metas


def propagate_tensor_meta(op, user_args, user_kwargs, out_strat):
    new_tensor_meta, tensor_metas = _get_meta_tensors_for_op(op, user_args, user_kwargs)

    for strat in out_strat.strategies:
        if isinstance(new_tensor_meta, TensorMeta):
            strat.output_spec.tensor_meta = new_tensor_meta
        else:
            for ospec, tm in zip(strat.output_specs, new_tensor_meta):
                if ospec is not None:
                    if ospec.tensor_meta != tm:
                        # This is overcoming some limitations of the lack of
                        # tensor_meta for sdpa which returns None
                        # we should just fix this all across the board
                        if ospec.tensor_meta is None:
                            ospec.tensor_meta = tm
                        else:
                            assert tm is None
        if strat.input_specs is None:
            # TODO: this should be cleaned up

            supported_ops = {
                torch.ops.prims.convert_element_type.default,
                torch.ops.aten.clone.default,
                torch.ops.aten.slice.Tensor,
            }
            assert op in supported_ops, (
                f"{op} strategy doesn't have input_specs, only harcoded "
                "{supported_ops} for now"
            )
            strat.input_specs = (strat.output_specs,)
            assert strat.redistribute_cost is None
        # TODO: this invariant wrt factory ops is something I believe
        # I'll keep for the solver, so we need to have some consistency here
        # i.e., even though factory ops don't have inputs, we do put an
        # input spec for it which is equal to the output spec
        if op not in TENSOR_FACTORY_OPS:
            assert len(tensor_metas) == len(
                strat.input_specs
            ), f"{op}, {len(tensor_metas)}, {len(strat.input_specs)}"
        for tm, ispec in zip(tensor_metas, strat.input_specs):
            if ispec.tensor_meta is None:
                ispec.tensor_meta = tm


def fill_missing_redistribute_cost(op, specs, out_strat):
    for strat in out_strat.strategies:
        # TODO: check me
        if strat.redistribute_cost is None:
            # TODO: the torch.ops.aten.slice.Tensor is wrong here and in the input_spec!!!!!
            handled_ops = {
                torch.ops.aten.ones_like.default,
                torch.ops.aten.full_like.default,
                torch.ops.aten.empty_like.default,
                torch.ops.prims.convert_element_type.default,
                torch.ops.aten.slice.Tensor,
            }
            assert op in handled_ops, f"got {op}, supported ops here are {handled_ops}"
            # assert len(specs) == 1, f"Expected len(specs) == 1, got {len(specs)}"
            redistribute_costs = [
                generate_redistribute_costs(specs[0], strat.output_spec)
            ]
            strat.redistribute_cost = redistribute_costs


def _generate_dummy_strategy(mesh, op, user_args, user_kwargs, input_strategies):
    from torch.distributed.tensor._dtensor_spec import DTensorSpec
    from torch.distributed.tensor._op_schema import OpSpec
    from torch.distributed.tensor.placement_types import Replicate

    placements = (Replicate(),) * mesh.ndim

    out_tensor_meta, input_tensor_metas = _get_meta_tensors_for_op(
        op, user_args, user_kwargs
    )

    input_specs = [
        DTensorSpec(mesh=mesh, placements=placements, tensor_meta=tm)
        for tm in input_tensor_metas
    ]
    if isinstance(out_tensor_meta, TensorMeta):
        output_spec = DTensorSpec(
            mesh=mesh, placements=placements, tensor_meta=out_tensor_meta
        )
    else:
        output_spec = tuple(
            DTensorSpec(mesh=mesh, placements=placements, tensor_meta=tm)
            for tm in out_tensor_meta
        )

    out_strat = OpSpec(output_specs=output_spec, input_specs=input_specs)
    num_input_args = len(input_tensor_metas)
    input_strategies_flat = [
        x for x in tree_flatten(input_strategies)[0] if isinstance(x, OpStrategy)
    ]
    assert num_input_args == len(
        input_strategies_flat
    ), f"{op}, {num_input_args}, {len(input_strategies_flat)}"
    redistribute_cost = [
        generate_redistribute_costs(input_strategies_flat[i], input_specs[i])
        for i in range(num_input_args)
    ]
    out_strat.redistribute_cost = redistribute_cost

    assert len(out_strat.redistribute_cost) == num_input_args
    out_strat = OpStrategy([out_strat])
    return out_strat


def get_placement_options(mesh, op, specs, user_args, user_kwargs):
    # print(op)

    if op in _op_rules:
        out_strat = _op_rules[op](mesh, specs)
        out_strat = remove_invalid_configs(out_strat, mesh)
        return out_strat

    strat = []
    for spec in specs:
        if isinstance(spec, OpStrategy):
            strat.append(spec)
        elif (
            isinstance(spec, list)
            and len(spec) > 0
            and any(isinstance(x, OpStrategy) for x in spec)
        ):
            strat.append(TupleStrategy(spec))
        else:
            strat.append(spec)
    strat = tuple(strat)

    op_schema = OpSchema(op, strat, {})

    if op in _op_partial_rules:
        out_strat = _op_partial_rules[op](mesh, op_schema)
    elif (
        op
        in torch.distributed.tensor.DTensor._op_dispatcher.sharding_propagator.op_strategy_funcs
        and op
        not in {
            torch.ops.aten._softmax_backward_data.default,
        }
    ):
        out_strat = torch.distributed.tensor.DTensor._op_dispatcher.sharding_propagator.op_strategy_funcs[
            op
        ](
            op_schema
        )
    else:
        print(f"Ops that need to be implemented {op}")
        out_strat = _generate_dummy_strategy(mesh, op, user_args, user_kwargs, strat)

    propagate_tensor_meta(op, user_args, user_kwargs, out_strat)
    fill_missing_redistribute_cost(op, specs, out_strat)
    out_strat = remove_invalid_configs(out_strat, mesh)

    return out_strat


def _get_device_from_mesh(mesh):
    if mesh.device_type == "cpu":
        return torch.device("cpu")
    device_handle = _get_device_handle(mesh.device_type)
    return torch.device(mesh.device_type, device_handle.current_device())
