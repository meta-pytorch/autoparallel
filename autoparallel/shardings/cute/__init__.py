"""
CuTe-based sharding placement and propagation.

This package provides:
- CutePlacement: A unified placement type using CuTe layouts
- Propagation rules for view, einsum, pointwise, and reduction ops
"""

from .placement import CutePlacement
from .propagation import (
    propagate_einsum,
    propagate_pointwise,
    propagate_reduction,
    propagate_view,
)

__all__ = [
    "CutePlacement",
    "propagate_view",
    "propagate_einsum",
    "propagate_pointwise",
    "propagate_reduction",
]
