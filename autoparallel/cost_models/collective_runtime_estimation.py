# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
Communication cost estimation for DTensor redistribution.

NOTE: This module depends on PyTorch PR that adds `all_to_all_cost` and
`include_compute_cost` to `torch.distributed.tensor._collective_utils`.
See: pytorch/pytorch branch `cost-model-consolidation`
"""

import torch.distributed.tensor._dtensor_spec as dtensor_spec
from torch.distributed.tensor._collective_utils import (
    MeshTopoInfo,
    all_to_all_cost,
    allgather_cost,
    allreduce_cost,
)
from torch.distributed.tensor._collective_utils import (
    redistribute_cost as _pytorch_redistribute_cost,
)
from torch.distributed.tensor._collective_utils import (
    reduce_scatter_cost,
    spec_to_bytes,
)


def redistribute_cost(
    current_spec: "dtensor_spec.DTensorSpec",
    target_spec: "dtensor_spec.DTensorSpec",
    order: list[int] | None = None,
) -> float:
    """
    Estimate the cost of redistributing from current to target DTensorSpec.

    This is a thin wrapper around PyTorch's redistribute_cost that enables
    compute cost estimation by default (for accurate sharding strategy selection).

    Args:
        current_spec: The current DTensorSpec.
        target_spec: The target DTensorSpec.
        order: Deprecated. Previously used for custom iteration order.
            PyTorch now uses _gen_transform_infos for optimal ordering.

    Returns:
        The estimated cost of redistribution in microseconds.
    """
    # Use PyTorch's upstream redistribute_cost with compute costs enabled
    # This accounts for reshuffle overhead on non-dim-0 shards
    return _pytorch_redistribute_cost(
        current_spec,
        target_spec,
        include_compute_cost=True,
    )


def estimate_strategy_comms_cost(
    src_spec: "dtensor_spec.DTensorSpec",
    tgt_spec: "dtensor_spec.DTensorSpec",
) -> float:
    """
    Estimate communication cost for a sharding strategy transition.

    Args:
        src_spec: Source DTensorSpec (current sharding).
        tgt_spec: Target DTensorSpec (desired sharding).

    Returns:
        Estimated communication cost in microseconds.
    """
    return redistribute_cost(src_spec, tgt_spec)


# Re-export for convenience
__all__ = [
    "redistribute_cost",
    "estimate_strategy_comms_cost",
    "all_to_all_cost",
    "allgather_cost",
    "allreduce_cost",
    "reduce_scatter_cost",
    "MeshTopoInfo",
    "spec_to_bytes",
]
