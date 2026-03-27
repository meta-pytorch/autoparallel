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

# Import composition from torch — we need it here but avoid circular imports
# by importing at function level if needed
from torch.distributed._pycute import composition, flatten, is_tuple

from .scaled_basis import E, ScaledBasis, make_basis_like


def codomain_divide(layout, group_shape):
    """
    Determine per-piece coverage when reshaping a layout's codomain.

    Uses CuTe composition with coordinate-producing strides:
    composition(Layout(group_shape, (E(0), E(1), ...)), layout)
    decomposes the layout's modes through the split shape, tagging each
    resulting mode with an E(k) stride indicating which piece it belongs to.

    Args:
        layout: A CuTe Layout mapping indices -> flat offsets.
        group_shape: Tuple of ints (G0, G1, ...) defining the codomain reshape.
            Row-major: G0 is outermost (largest stride), G_{n-1} is innermost.

    Returns:
        Dict mapping piece_idx -> coverage count.
        coverage[k] = number of unique G_k coordinates the layout covers.
        If coverage[k] == group_shape[k], piece k is fully covered.
        If coverage[k] < group_shape[k], piece k is partially covered (sharded).

    Complexity: O(1) — structural composition on shape/stride, no enumeration.
    """
    # Avoid circular: Layout is used by composition, import here
    from torch.distributed._pycute import Layout as _Layout

    from .scaled_basis import make_basis_like

    coord_split = _Layout(group_shape, make_basis_like(group_shape))
    result = composition(coord_split, layout)

    # Extract modes and group by piece (E(k) stride)
    if is_tuple(result.shape):
        modes = list(zip(flatten(result.stride), flatten(result.shape)))
    else:
        modes = [(result.stride, result.shape)]

    n_pieces = len(group_shape)
    coverage = {k: 1 for k in range(n_pieces)}

    for stride, size in modes:
        if isinstance(stride, ScaledBasis):
            coverage[stride.index] *= size

    return coverage
