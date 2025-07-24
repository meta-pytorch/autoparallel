# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import logging
from contextlib import contextmanager
from typing import Callable, TypeVar

import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._op_schema import (
    OpSchema,
    OutputSharding,
    RuntimeSchemaInfo,
    StrategyType,
)
from typing_extensions import ParamSpec

logger = logging.getLogger(__name__)

aten = torch.ops.aten

_T = TypeVar("_T")
_P = ParamSpec("_P")


# -------------define universal op strategy-------------
replicate_op_strategy = torch.distributed.tensor._ops.utils.replicate_op_strategy


class StrategyPool:
    def __init__(self) -> None:
        # reference to existing strategy from the DTensor upstream
        self.op_strategy_funcs: dict[
            torch._ops.OpOverload, Callable[[OpSchema], StrategyType]
        ] = DTensor._op_dispatcher.sharding_propagator.op_strategy_funcs
        # reference to existing rules
        self.op_to_rules: dict[
            torch._ops.OpOverload, Callable[[OpSchema], OutputSharding]
        ] = DTensor._op_dispatcher.sharding_propagator.op_to_rules
        # we probably don't need to care about existing op_to_schema_info for AP
        self.op_to_schema_info = (
            DTensor._op_dispatcher.sharding_propagator.op_to_schema_info
        )

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
                logger.warning(
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
