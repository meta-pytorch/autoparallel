# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# functions to expose
from .utils import (
    get_op_strategy,
    op_strategy_context,
    replicate_op_strategy,
    with_implicit_strategies,
)

__all__ = [
    "replicate_op_strategy",
    "get_op_strategy",
    "with_implicit_strategies",
    "op_strategy_context",
]
