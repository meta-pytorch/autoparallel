# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import copy
import itertools
import warnings
from contextlib import contextmanager
from typing import Callable, Optional, TypeVar

import torch
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._op_schema import (
    OpSchema,
    OpSpec,
    OpStrategy,
    PlacementList,
    RuntimeSchemaInfo,
    StrategyType,
    TupleStrategy,
)
from torch.distributed.tensor._ops.utils import (
    expand_to_full_mesh_op_strategy,
    generate_redistribute_costs,
    is_tensor_shardable,
)
from torch.distributed.tensor.placement_types import Placement, Replicate, Shard

try:
    from torch.utils._cxx_pytree import register_pytree_node, tree_leaves
except ImportError:
    from torch.utils._pytree import tree_leaves, register_pytree_node  # type: ignore[no-redef]

from typing_extensions import ParamSpec

# TODO: remove once https://github.com/pytorch/pytorch/pull/158046 is merged
try:
    register_pytree_node(
        TupleStrategy,
        lambda node: (node.children, None),
        lambda children, _: TupleStrategy(tuple(children)),
    )
except ValueError:
    # already registered TupleStrategy, skip
    pass


aten = torch.ops.aten

_T = TypeVar("_T")
_P = ParamSpec("_P")


# -------------define universal op strategy-------------
def replicate_op_strategy(op_schema: OpSchema) -> OpStrategy:
    """
    Fallback strategy all use Replication()
    """
    inputs_strategy = op_schema.args_strategy
    # TODO(zpcore): handle kwarg_inputs_strategy
    # kwarg_inputs_strategy = op_schema.kwargs_schema
    output_type = [str(ret.type) for ret in op_schema.op._schema.returns]
    # TODO(zpcore): Confirm if view op can be handle properly or not. Prevent
    # handling view ops until confirmed.
    if op_schema.op.is_view:
        raise RuntimeError(
            "fallback strategy is unable to handle view ops until confirmed"
        )
    if "List[Tensor]" in output_type:
        raise RuntimeError(
            "fallback strategy is unable to handle ops with List[Tensor] output "
            "because size of the list may depend on the op's input value"
        )

    mesh = inputs_strategy[0].mesh

    dim_sharding: PlacementList = [Replicate()] * (len(inputs_strategy) + 1)
    single_dim_placement = [dim_sharding]
    return expand_to_full_mesh_op_strategy(mesh, op_schema, single_dim_placement)


def batch_shard_strategy(
    op_schema: OpSchema,
    input_shard_dim: list[Optional[int]],
    output_shard_dim: list[Optional[int]],
    enable_shard_batch_dim_over_multiple_axis: bool = False,
) -> OpStrategy:
    """
    Shard the input tensor over the specified dimensions. The strategy will map
    batch dim of input/output tensors to the same device mesh axis (or same
    multiple device axises). All input must either have one specified batch dim
    or no batch dim. If an input doesn't have batch dim, the strategy will
    assume the tensor will be broadcasted to batch dim and processed by the
    operator. For inputs specified with a batch dim, user need to make sure the
    batch dim size are all the same. Output should always have a batch dim.

    Args:
        op_schema (OpSchema): the op schema.

        input_shard_dim (list[Optional[int]]): the list of shard dimensions to
        consider for each input tensor argument. Use `None` if no batch dim of
        the input arg. If an arg is List[Tenor], we flatten it first and then
        match with input_shard_dim. Since the dim is not specific to the device
        mesh axis, it can be a combination of any device axises. Example 1:
        input tensor A[1024,64,8], B[1024,64,16], with input_shard_dim = [1,1],
        it can shard A's dim 0 over device axis X, and shard B's dim 0 over
        device axis X. X can be any of device axises. The output follow the same
        sharding as input. Example 2: input tensor A[64,8], B[64,16,1024],
        C[64,8], with input_shard_dim = [None,2,None], it will Replicate A,C
        over all device dim and only shard B's dim 2 over the device mesh.
        Assume the device mesh has 3 axis, then tensor B's placement can be
        (Shard(2), Shard(2), Replicate()), (Shard(2), Replicate(), Shard(2)),
        (Replicate(), Shard(2), Shard(2)).

        output_shard_dim (list[Optional[int]]): the list of shard dimensions to
        consider for each output tensor argument. Use `None` if no batch dim of
        the output arg. For example, if the output is a single tensor and is
        sharded on dim 0, pass in [0] then.

        enable_shard_batch_dim_over_multiple_axis (bool): if True, the strategy
        will try also map batch dim to multiple device axis. Default is False.

    Note: It is the user's responsibility to make sure the sharded tensor for
    processing is correct in shape.
    """
    output_type = [str(ret.type) for ret in op_schema.op._schema.returns]
    # TODO(zpcore): Confirm if view op can be handle properly or not. Prevent
    # handling view ops until confirmed.
    if op_schema.op.is_view:
        raise RuntimeError(
            "fallback strategy is unable to handle view ops until confirmed"
        )
    if "List[Tensor]" in output_type:
        raise RuntimeError(
            "fallback strategy is unable to handle ops with List[Tensor] output "
            "because size of the list may depend on the op's input value"
        )
    inputs_strategy = tree_leaves(op_schema.args_strategy)
    assert len(inputs_strategy) == len(input_shard_dim)
    output_strategy = OpStrategy([])
    mesh = inputs_strategy[0].mesh
    device_axis = list(range(mesh.ndim))
    use_how_many_axis = (
        [i + 1 for i in range(mesh.ndim)]
        if enable_shard_batch_dim_over_multiple_axis
        else [1]
    )
    # number of device axises to shard on for the batch dim
    for num_axis in use_how_many_axis:
        device_combinations = list(itertools.combinations(device_axis, num_axis))
        # e.g., if num_axis == 2, device_combinations = [(0,1), (0,2), (1,2),
        # ...]. Then One feasible strategy is to shard tensor dim on both axis
        # (0,1). We check all combinations in device_combinations below.
        for comb in device_combinations:
            input_specs_list: list[DTensorSpec] = []
            output_specs_list: list[DTensorSpec] = []
            is_shardable = True
            for op_stratgy, dim in zip(inputs_strategy, input_shard_dim):
                # create a new list of shard_dim_option
                new_placements: list[Placement] = [Replicate()] * mesh.ndim
                for axis in comb:
                    new_placements[axis] = Shard(dim) if dim else Replicate()
                tensor_meta = op_stratgy.strategies[0].output_spec.tensor_meta
                new_input_spec = DTensorSpec(
                    mesh,
                    tuple(new_placements),
                    tensor_meta=op_stratgy.strategies[0].output_spec.tensor_meta,
                )
                if not is_tensor_shardable(tensor_meta.shape, new_input_spec):
                    is_shardable = False
                    break
                input_specs_list.append(new_input_spec)
            if not is_shardable:
                continue
            for dim in output_shard_dim:
                new_placements = [Replicate()] * mesh.ndim
                for axis in comb:
                    new_placements[axis] = Shard(dim) if dim else Replicate()
                output_spec = DTensorSpec(
                    mesh,
                    tuple(new_placements),
                )
                output_specs_list.append(output_spec)

            output_specs = (
                output_specs_list[0]
                if len(output_specs_list) == 1
                else tuple(output_specs_list)
            )
            input_specs = input_specs_list
            redistribute_cost = [
                generate_redistribute_costs(strat, input_specs_list[i])
                for i, strat in enumerate(inputs_strategy)
            ]
            output_strategy.strategies.append(
                OpSpec(output_specs, input_specs, redistribute_cost)  # type: ignore
            )
    return output_strategy


class StrategyPool:
    def __init__(self) -> None:
        # collect the existing strategy from the DTensor upstream
        self.op_strategy_funcs: dict[
            torch._ops.OpOverload, Callable[[OpSchema], StrategyType]
        ] = copy.deepcopy(
            torch.distributed.tensor.DTensor._op_dispatcher.sharding_propagator.op_strategy_funcs
        )
        # we probably don't need to care about existing op_to_schema_info for AP
        self.op_to_schema_info = {
            k: copy.deepcopy(v)
            for k, v in torch.distributed.tensor.DTensor._op_dispatcher.sharding_propagator.op_to_schema_info.items()
            if v is not None
        }

        self.enable_implicit_replication: bool = False
        self.implicit_strategy_op_tracker: list[torch._ops.OpOverload] = []

    def get_op_strategy(
        self, op: torch._ops.OpOverload, op_schema: OpSchema
    ) -> StrategyType:
        if op not in self.op_strategy_funcs:
            if not self.enable_implicit_replication:
                raise NotImplementedError(
                    f"Operator {op} does not have a sharding strategy registered."
                )
            else:
                self.implicit_strategy_op_tracker.append(op)
                warnings.warn(
                    f"implicitly register sharding strategy op {op.name()} using {replicate_op_strategy.__name__}"
                )
                self.register_op_strategy(op)(replicate_op_strategy)
        return self.op_strategy_funcs[op](op_schema)

    def register_op_strategy(
        self,
        op: torch._ops.OpOverload,
        schema_info=RuntimeSchemaInfo(needs_pytree=True),
    ) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
        # pyre-fixme[2]: Parameter must be annotated.
        # always enable pytree as dispatching overhead is not a concern in AP.
        def wrapper(impl):
            if isinstance(op, list):
                overloads = op
            else:
                overloads = [op]

            for overload in overloads:
                self.op_strategy_funcs[overload] = impl
                self.op_to_schema_info[overload] = schema_info
            return impl

        return wrapper

    @contextmanager
    def replicate_for_unsupported_operators(self):
        """
        Context manager for setting and clearing implicit strategy.
        """
        try:
            if self.enable_implicit_replication:
                raise RuntimeError(
                    "Implicit strategy is already enabled. Cannot enable it again."
                )
            self.enable_implicit_replication = True
            yield
        finally:
            self.enable_implicit_replication = False
            op_to_remove = self.implicit_strategy_op_tracker
            for op_overload in op_to_remove:
                if op_overload in self.op_strategy_funcs:
                    del self.op_strategy_funcs[op_overload]
                if op_overload in self.op_to_schema_info:
                    del self.op_to_schema_info[op_overload]
            self.implicit_strategy_op_tracker.clear()

    # TODO: automatic generate redistribute cost for strategies. There exists a
    # `fill_missing_redistribute_cost` in autoparallel/utils.py, which is a hack
    # to generate redistribute cost given input specs, and only tested on
    # certain ops. We can potentially make an improvement.
    def fill_missing_redistribute_cost(
        self, op: torch._ops.OpOverload, op_schema: OpSchema
    ):
        """
        Fill missing redistribute cost for strategies.
        """
        ...
