# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
from contextlib import ExitStack, contextmanager
from typing import Optional

import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._op_schema import (
    OpSchema,
    OpSpec,
    OpStrategy,
    StrategyType,
)
from torch.distributed.tensor._ops.utils import (
    generate_redistribute_costs,
    is_tensor_shardable,
    register_op_strategy,
)
from torch.distributed.tensor.placement_types import Placement, Replicate, Shard

try:
    from torch.utils._cxx_pytree import tree_leaves
except ImportError:
    from torch.utils._pytree import tree_leaves  # type: ignore[no-redef]


logger = logging.getLogger(__name__)

aten = torch.ops.aten

# reference to existing sharding_propagator DTensor upstream
propagator = DTensor._op_dispatcher.sharding_propagator

enable_implicit_replication = False
_current_stack = None

replicate_op_strategy = torch.distributed.tensor._ops.utils.replicate_op_strategy


# TODO: remove and refer to
# https://github.com/pytorch/pytorch/blob/9c107606629de6383f55e3b48b42e594d23407b1/test/distributed/tensor/test_op_strategy.py#L446
# once the function is moved outside of the test folder in upstream
@contextmanager
def op_strategy_context(op_overload, strategy_func, schema_info=None):
    """
    Context manager for setting and clearing op strategies.
    Args:
        op_overload: The operator overload to set or clear the strategy for.
        strategy_func: The strategy function to set for the operator overload.
        schema_info: Optional schema information for the operator overload.
    Yields:
        None
    """
    propagator = DTensor._op_dispatcher.sharding_propagator
    _origin_op_strategy_funcs = None
    _origin_op_strategy_schema = None
    try:
        # register the op strategy
        if op_overload in propagator.op_strategy_funcs:
            _origin_op_strategy_funcs = propagator.op_strategy_funcs[op_overload]
            del propagator.op_strategy_funcs[op_overload]
        if op_overload in propagator.op_to_schema_info:
            _origin_op_strategy_schema = propagator.op_to_schema_info[op_overload]
            del propagator.op_to_schema_info[op_overload]
        register_op_strategy(op_overload, schema_info=schema_info)(strategy_func)
        yield
    finally:
        # clear this op strategy cache
        if _origin_op_strategy_funcs is None:
            if op_overload in propagator.op_strategy_funcs:
                del propagator.op_strategy_funcs[op_overload]
        else:
            propagator.op_strategy_funcs[op_overload] = _origin_op_strategy_funcs
        if _origin_op_strategy_schema is None:
            if op_overload in propagator.op_to_schema_info:
                del propagator.op_to_schema_info[op_overload]
        else:
            propagator.op_to_schema_info[op_overload] = _origin_op_strategy_schema
        propagator.propagate_op_sharding.cache.cache_clear()


# -------------define universal op strategy-------------
def batch_shard_strategy(
    op_schema: OpSchema,
    input_shard_dim: list[Optional[int]],
    output_shard_dim: list[Optional[int]],
    enable_shard_batch_dim_over_multiple_axis: bool = False,
) -> OpStrategy:
    """
    Shard the input tensor over the specified dimensions. The strategy will map
    batch dim of input/output tensors to the same device mesh axis (or same
    multiple device axes). All input must either have one specified batch dim or
    no batch dim. If an input doesn't have batch dim, the strategy will assume
    the tensor will be broadcasted to batch dim and processed by the operator.
    For inputs specified with a batch dim, user need to make sure the batch dim
    size are all the same. Output should always have a batch dim.

    Args:
        op_schema (OpSchema): the op schema.

        input_shard_dim (list[Optional[int]]): the list of shard dimensions to
        consider for each input tensor argument. Use `None` if no batch dim of
        the input arg. If an arg is List[Tenor], we flatten it first and then
        match with input_shard_dim. Since the dim is not specific to the device
        mesh axis, it can be a combination of any device axes. Example 1: input
        tensor A[1024,64,8], B[1024,64,16], with input_shard_dim = [1,1], it can
        shard A's dim 0 over device axis X, and shard B's dim 0 over device axis
        X. X can be any of device axes. The output follow the same sharding as
        input. Example 2: input tensor A[64,8], B[64,16,1024], C[64,8], with
        input_shard_dim = [None,2,None], it will Replicate A,C over all device
        dim and only shard B's dim 2 over the device mesh. Assume the device
        mesh has 3 axis, then tensor B's placement can be (Shard(2), Shard(2),
        Replicate()), (Shard(2), Replicate(), Shard(2)), (Replicate(), Shard(2),
        Shard(2)).

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
    # number of device axes to shard on for the batch dim
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


def get_op_strategy(op: torch._ops.OpOverload, op_schema: OpSchema) -> StrategyType:
    global enable_implicit_replication, _current_stack

    if op not in propagator.op_strategy_funcs:
        if not enable_implicit_replication:
            raise NotImplementedError(
                f"Operator {op} does not have a sharding strategy registered."
            )
        else:
            # Use the current stack if available
            if _current_stack is not None:
                _current_stack.enter_context(
                    op_strategy_context(op, replicate_op_strategy)
                )
            else:
                # No stack available, just register permanently
                register_op_strategy(op)(replicate_op_strategy)
            logger.warning(
                f"implicitly registering `{op}` with `{replicate_op_strategy.__name__}`"
            )
    return propagator.op_strategy_funcs[op](op_schema)


@contextmanager
def with_implicit_strategies():
    """Context manager to enable implicit replication and clean up strategies."""
    global enable_implicit_replication, _current_stack

    # Create a fresh ExitStack for this context
    with ExitStack() as local_stack:
        # Store the stack as a global variable
        old_stack = _current_stack
        _current_stack = local_stack

        # Enable implicit replication
        old_value = enable_implicit_replication
        enable_implicit_replication = True
        try:
            yield
        finally:
            # Restore the original values
            _current_stack = old_stack
            enable_implicit_replication = old_value


# TODO: automatic generate redistribute cost for strategies. There exists a
# `fill_missing_redistribute_cost` in autoparallel/utils.py, which is a hack
# to generate redistribute cost given input specs, and only tested on
# certain ops. We can potentially make an improvement.
def fill_missing_redistribute_cost(op: torch._ops.OpOverload, op_schema: OpSchema):
    """
    Fill missing redistribute cost for strategies.
    """
    ...
