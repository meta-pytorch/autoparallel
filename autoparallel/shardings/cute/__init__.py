"""
CuTe-based sharding placement and propagation.

ShardedLayout = hierarchical CuTe Layout (local, mesh) per dim.
Produced by logical_divide(Layout(tensor_shape), local_sizes).
"""

from .placement import ShardedLayout

# Keep TiledLayout as alias during transition
TiledLayout = ShardedLayout
from .propagation import (
    propagate_broadcast,
    propagate_einsum,
    propagate_gather,
    propagate_permute,
    propagate_pointwise,
    propagate_reduction,
    propagate_slice,
    propagate_transpose,
    propagate_unsqueeze,
    propagate_view,
)

__all__ = [
    "TiledLayout",
    "propagate_broadcast",
    "propagate_einsum",
    "propagate_gather",
    "propagate_permute",
    "propagate_pointwise",
    "propagate_reduction",
    "propagate_slice",
    "propagate_transpose",
    "propagate_unsqueeze",
    "propagate_view",
]
