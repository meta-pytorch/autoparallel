"""
max_common_layout: Find the maximum common sub-layout between two layouts.

Ported from C++ CuTe (layout.hpp lines 1370-1426).

The maximum common layout is the largest contiguous sub-layout that both
layouts agree on — the "identity prefix" of composition(a, right_inverse(b)).
Used for auto-vectorization: determines how many elements can be processed
at once when copying/computing between two layouts.
"""

from torch.distributed._pycute import (
    Layout,
    coalesce,
    is_tuple,
    right_inverse,
)

# Use our ScaledBasis-aware composition
from . import composition


def max_common_vector(a, b):
    """
    Return the maximum number of contiguous elements that logically
    correspond in layouts a and b.

    For all 0 <= i < result: a(b.right_inverse(i)) == i

    Args:
        a, b: CuTe Layouts with the same codomain.

    Returns:
        int: Maximum common vector size (>= 1).
    """
    inv_b = right_inverse(b)
    if inv_b is None:
        return 1
    common = coalesce(composition(a, inv_b))
    if is_tuple(common.shape):
        first_stride = common.stride[0]
        first_shape = common.shape[0]
    else:
        first_stride = common.stride
        first_shape = common.shape
    if first_stride == 1:
        return first_shape
    return 1


def max_common_layout(a, b):
    """
    Return a layout pointing to the maximum number of contiguous elements
    that logically correspond in layouts a and b.

    For all 0 <= i < size(result): a(result(i)) == i and b(result(i)) == i

    Args:
        a, b: CuTe Layouts with the same codomain.

    Returns:
        Layout mapping indices to coordinates in b's domain.
    """
    inv_b = right_inverse(b)
    if inv_b is None:
        return Layout(1, 0)
    common = coalesce(composition(a, inv_b))
    if is_tuple(common.shape):
        first_stride = common.stride[0]
        first_shape = common.shape[0]
    else:
        first_stride = common.stride
        first_shape = common.shape
    if first_stride == 1:
        return composition(inv_b, Layout(first_shape, 1))
    return Layout(1, 0)
