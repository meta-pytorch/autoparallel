"""
Extended pycute with coordinate-producing strides (ScaledBasis).

Re-exports everything from torch.distributed._pycute and adds
ScaledBasis/ArithmeticTuple for coordinate layouts.
"""

from torch.distributed._pycute import *  # noqa: F401,F403
from torch.distributed._pycute import (
    Layout,
    coalesce,
    complement,
    composition,
    cosize,
    crd2idx,
    filter,
    flatten,
    has_none,
    idx2crd,
    inner_product,
    is_int,
    is_layout,
    is_tuple,
    left_inverse,
    logical_divide,
    logical_product,
    make_layout,
    product,
    right_inverse,
    shape_div,
    size,
    slice_,
    suffix_product,
)

from .scaled_basis import (
    ArithmeticTuple,
    E,
    ScaledBasis,
    is_arithmetic_tuple,
    is_coord_stride,
    is_scaled_basis,
    make_basis_like,
)
