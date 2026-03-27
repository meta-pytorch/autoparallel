"""
codomain_divide: Decompose a layout by reshaping its codomain.

This is the dual of logical_divide:
  logical_divide(L, tiler)       -> reshape DOMAIN into (tile, rest)
  codomain_divide(L, group_shape) -> reshape CODOMAIN into per-piece coverage

Implemented as composition with a coordinate-producing split layout:
  composition(Layout(group_shape, (E(0), E(1), ...)), layout)
The result has E(k) strides tagging each mode with its piece index.
Per-piece coverage is read from the mode sizes grouped by E(k).
"""

from torch.distributed._pycute import composition, flatten, is_tuple

from .scaled_basis import ScaledBasis, make_basis_like


def _read_coverage(result, n_pieces):
    """
    Read per-piece coverage from a layout with E(k) coordinate strides.

    Each mode's ScaledBasis stride tags which piece it belongs to.
    Coverage = product of sizes per piece.
    """
    coverage = {k: 1 for k in range(n_pieces)}

    if is_tuple(result.shape):
        modes = zip(flatten(result.stride), flatten(result.shape))
    else:
        modes = ((result.stride, result.shape),)

    for stride, size in modes:
        if isinstance(stride, ScaledBasis):
            coverage[stride.index] *= size

    return coverage


def codomain_divide(layout, group_shape):
    """
    Determine per-piece coverage when reshaping a layout's codomain.

    Accepts either:
    - A layout with integer strides: composes with coordinate split first
    - A layout with E(k) coordinate strides: reads coverage directly

    Args:
        layout: A CuTe Layout (integer or coordinate strides).
        group_shape: Tuple of ints (G0, G1, ...) defining the codomain reshape.

    Returns:
        Dict mapping piece_idx -> coverage count.

    Complexity: O(1) — structural composition on shape/stride, no enumeration.
    """
    if _has_coord_strides(layout):
        # Already composed with coordinate strides — read directly
        return _read_coverage(layout, len(group_shape))

    # Integer strides — compose with coordinate layout first
    from torch.distributed._pycute import Layout as _Layout

    coord_split = _Layout(group_shape, make_basis_like(group_shape))
    result = composition(coord_split, layout)
    return _read_coverage(result, len(group_shape))


def _has_coord_strides(layout):
    """Check if a layout has ScaledBasis (coordinate) strides."""
    strides = flatten(layout.stride) if is_tuple(layout.stride) else (layout.stride,)
    return any(isinstance(s, ScaledBasis) for s in strides)
