"""
Extended pycute with coordinate-producing strides, codomain_divide,
and ScaledBasis-aware composition.

Re-exports everything from torch.distributed._pycute and adds:
- ScaledBasis/ArithmeticTuple for coordinate layouts
- codomain_divide for codomain decomposition (dual of logical_divide)
- composition override that handles ScaledBasis strides (mode selection)
"""

from torch.distributed._pycute import *  # noqa: F401,F403
from torch.distributed._pycute import (
    Layout,
    coalesce,
    complement,
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
# Import base composition for non-ScaledBasis cases
from torch.distributed._pycute import composition as _base_composition

from .codomain_divide import codomain_divide
from .scaled_basis import (
    ArithmeticTuple,
    E,
    ScaledBasis,
    is_arithmetic_tuple,
    is_coord_stride,
    is_scaled_basis,
    make_basis_like,
)


def _basis_get(basis, tup):
    """Select element from tuple at the position indicated by basis.index."""
    if isinstance(basis, ScaledBasis):
        if is_tuple(tup):
            return tup[basis.index]
        return tup  # scalar, no indexing needed
    return tup


def composition(layout_a, layout_b):
    """
    Layout composition with ScaledBasis support.

    When layout_b has ScaledBasis strides, each mode of B selects the
    corresponding mode from A via basis_get, then composes with the
    scalar basis value. This enables permutation/transpose:

        composition(Layout((4,8),(8,1)), Layout((8,4),(E(1),E(0))))
        = Layout((8,4),(1,8))  -- transpose!

    For non-ScaledBasis cases, delegates to torch's composition.
    """
    if layout_b is None:
        return layout_a
    elif is_int(layout_b):
        return composition(layout_a, Layout(layout_b))
    elif is_tuple(layout_b):
        # By-mode: compose each mode of A with corresponding element of B
        assert len(layout_a) >= len(layout_b)
        from itertools import chain

        return make_layout(
            chain(
                (composition(layout_a[i], layout_b[i]) for i in range(len(layout_b))),
                (layout_a[i] for i in range(len(layout_b), len(layout_a))),
            )
        )
    elif is_tuple(layout_b.shape):
        # B has tuple shape — compose each sub-layout of B with A
        return make_layout(composition(layout_a, b_i) for b_i in layout_b)

    # B is rank-1: check for ScaledBasis stride
    if isinstance(layout_b.stride, ScaledBasis):
        # Select the LHS mode indicated by the basis index
        selected_shape = _basis_get(layout_b.stride, layout_a.shape)
        selected_stride = _basis_get(layout_b.stride, layout_a.stride)
        # Recurse with selected mode and basis scalar value
        selected_layout = Layout(selected_shape, selected_stride)
        scalar_b = Layout(layout_b.shape, layout_b.stride.value)
        return composition(selected_layout, scalar_b)

    # Non-ScaledBasis: delegate to base composition
    return _base_composition(layout_a, layout_b)
