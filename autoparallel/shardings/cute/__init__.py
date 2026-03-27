"""
CuTe-based sharding placement and propagation.

TiledLayout = tensor_layout + mesh_tiler.
shard_layout = logical_divide(tensor_layout, mesh_tiler) — derived, not stored.
"""

from .placement import TiledLayout
from .propagation import (
    propagate_einsum,
    propagate_gather,
    propagate_permute,
    propagate_pointwise,
    propagate_reduction,
    propagate_slice,
    propagate_transpose,
    propagate_view,
)

__all__ = [
    "TiledLayout",
    "propagate_view",
    "propagate_transpose",
    "propagate_permute",
    "propagate_slice",
    "propagate_gather",
    "propagate_einsum",
    "propagate_pointwise",
    "propagate_reduction",
]
