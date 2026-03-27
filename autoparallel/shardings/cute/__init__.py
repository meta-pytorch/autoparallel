"""
CuTe-based sharding placement and propagation.

This package provides:
- TiledLayout: Sharding as composition of tensor layout and shard layout
- Propagation rules for view, transpose, einsum, pointwise, and reduction ops
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
