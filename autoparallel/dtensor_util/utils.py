# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import logging
from contextlib import ExitStack, contextmanager

import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._op_schema import OpSchema, StrategyType
from torch.distributed.tensor._ops.utils import register_op_strategy

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
    try:
        # register the op strategy
        register_op_strategy(op_overload, schema_info=schema_info)(strategy_func)
        yield
    finally:
        # clear this op strategy cache
        if op_overload in propagator.op_strategy_funcs:
            del propagator.op_strategy_funcs[op_overload]
        if op_overload in propagator.op_to_schema_info:
            del propagator.op_to_schema_info[op_overload]
        propagator.propagate_op_sharding.cache.cache_clear()


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
