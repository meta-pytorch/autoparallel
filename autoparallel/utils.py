# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union

import torch
from torch.distributed._tensor.placement_types import Placement, TensorMeta
from torch.distributed.device_mesh import _get_device_handle
from torch.distributed.tensor import DeviceMesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._op_schema import (
    OpSchema,
    OpSpec,
    OpStrategy,
    TupleStrategy,
)
from torch.distributed.tensor._ops.utils import generate_redistribute_costs
from torch.distributed.tensor.experimental._func_map import (
    InputPlacements,
    OutputPlacements,
)
from torch.utils._pytree import tree_flatten, tree_map_only

from .propagation_rules import _op_partial_rules, _op_rules, remove_invalid_configs


def propagate_tensor_meta(op, user_args, user_kwargs, out_strat):
    out_t = op(*user_args, **user_kwargs)

    if isinstance(out_t, torch.Tensor):
        new_tensor_meta = TensorMeta(out_t.shape, out_t.stride(), out_t.dtype)
    else:
        new_tensor_meta = tree_map_only(
            torch.Tensor, lambda x: TensorMeta(x.shape, x.stride(), x.dtype), out_t
        )

    tensor_metas = tree_flatten(user_args)[0]
    tensor_metas = tree_map_only(
        torch.Tensor, lambda x: TensorMeta(x.shape, x.stride(), x.dtype), tensor_metas
    )
    tensor_metas = tuple(x for x in tensor_metas if isinstance(x, TensorMeta))

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
    else:
        out_strat = torch.distributed.tensor.DTensor._op_dispatcher.sharding_propagator.op_strategy_funcs[
            op
        ](
            op_schema
        )

    propagate_tensor_meta(op, user_args, user_kwargs, out_strat)
    fill_missing_redistribute_cost(op, specs, out_strat)
    out_strat = remove_invalid_configs(out_strat, mesh)

    return out_strat


def get_local_map_placement_option(
    mesh: DeviceMesh,
    specs: tuple[OpStrategy],
    user_args: tuple[torch.Tensor],
    output_val: Union[torch.Tensor, tuple[torch.Tensor]],
    in_placements: InputPlacements,
    out_placements: OutputPlacements,
):
    in_specs = []
    for example, placement in zip(user_args, in_placements):
        if placement is None:
            # not a dtensor
            assert False, "Not sure how to create DTensorSpec for this input"

        assert isinstance(placement, (list, tuple)), "Not implemented"
        in_specs.append(
            DTensorSpec(
                mesh=mesh,
                placements=placement,
                tensor_meta=TensorMeta(
                    shape=example.shape,
                    stride=example.stride(),
                    dtype=example.dtype,
                ),
            )
        )

    out_specs = []
    assert isinstance(output_val, (torch.Tensor, list, tuple))
    outs = output_val if isinstance(output_val, (list, tuple)) else [output_val]
    for example, placement in zip(outs, out_placements):
        if placement is None:
            # not a dtensor
            assert False, "Not sure how to create DTensorSpec for this output"
        elif isinstance(placement, Placement):
            placement = [placement]

        assert isinstance(placement, (list, tuple)), "Not implemented"
        out_specs.append(
            DTensorSpec(
                mesh=mesh,
                placements=placement,
                tensor_meta=TensorMeta(
                    shape=example.shape,
                    stride=example.stride(),
                    dtype=example.dtype,
                ),
            )
        )

    if len(out_specs) == 1:
        out_specs = out_specs[0]

    redistribute_costs = []
    for user_strategy, input_spec in zip(specs, in_specs):
        costs = generate_redistribute_costs(user_strategy, input_spec)
        redistribute_costs.append(costs)

    return OpStrategy(
        [
            OpSpec(
                output_specs=out_specs,
                input_specs=in_specs,
                redistribute_cost=redistribute_costs,
            )
        ]
    )


def _get_device_from_mesh(mesh):
    if mesh.device_type == "cpu":
        return torch.device("cpu")
    device_handle = _get_device_handle(mesh.device_type)
    return torch.device(mesh.device_type, device_handle.current_device())
