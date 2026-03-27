"""
CuTe-based sharding placement and propagation.

This package provides:
- TiledLayout: Sharding as composition of tensor layout and shard layout
- Propagation rules for view, transpose, einsum, pointwise, and reduction ops
"""

from .placement import TiledLayout
from .propagation import (
    propagate_einsum,
    propagate_pointwise,
    propagate_reduction,
    propagate_transpose,
    propagate_view,
)

__all__ = [
    "TiledLayout",
    "propagate_view",
    "propagate_transpose",
    "propagate_einsum",
    "propagate_pointwise",
    "propagate_reduction",
]
