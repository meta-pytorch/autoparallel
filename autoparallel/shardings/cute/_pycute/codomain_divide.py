"""
codomain_divide: Decompose a layout by reshaping its codomain.

This is the dual of logical_divide:
  logical_divide(L, tiler)       -> reshape DOMAIN into (tile, rest)
  codomain_divide(L, group_shape) -> reshape CODOMAIN into per-piece coverage

Given a layout L mapping local_idx -> flat_offset, and a group_shape (G0, G1, ...)
that describes how to reshape the flat codomain, determine how many unique
coordinates each piece covers.

The flat codomain has row-major strides: S_k = product(group_shape[k+1:]).
Each mode of L (with stride d and size s) falls into the piece k where
S_k <= d < S_k * G_k. It contributes min(s, S_k * G_k / d) to piece k's
coverage, with overflow becoming a virtual mode at stride S_k * G_k that
is processed by outer pieces.

This operation answers: "If I reshape the flat offsets this layout produces
into (G0, G1, ...), how many unique values does each piece get?"
"""

from torch.distributed._pycute import flatten, is_tuple


def codomain_divide(layout, group_shape):
    """
    Determine per-piece coverage when reshaping a layout's codomain.

    Args:
        layout: A CuTe Layout mapping indices -> flat offsets.
        group_shape: Tuple of ints (G0, G1, ...) defining the codomain reshape.
            Row-major: G0 is outermost (largest stride), G_{n-1} is innermost.

    Returns:
        Dict mapping piece_idx -> coverage count.
        coverage[k] = number of unique G_k coordinates the layout covers.
        If coverage[k] == group_shape[k], piece k is fully covered.
        If coverage[k] < group_shape[k], piece k is partially covered (sharded).

    Complexity: O(modes × pieces) — no element enumeration.
    """
    n_pieces = len(group_shape)

    # Row-major strides for the codomain shape
    split_strides = [1] * n_pieces
    for k in range(n_pieces - 2, -1, -1):
        split_strides[k] = split_strides[k + 1] * group_shape[k + 1]

    # Extract layout modes as (stride, size) pairs
    if is_tuple(layout.shape):
        modes = list(zip(flatten(layout.stride), flatten(layout.shape)))
    else:
        modes = [(layout.stride, layout.shape)]

    coverage = {k: 1 for k in range(n_pieces)}

    for mode_stride, mode_size in modes:
        if mode_size <= 1:
            continue

        # Process this mode through pieces from innermost to outermost.
        # A mode at stride d with size s falls into piece k where S_k <= d < S_k*G_k.
        # It contributes steps_in_piece = min(s, S_k*G_k / d) to piece k.
        # If s > steps_in_piece, the overflow has effective stride S_k*G_k
        # and size s // steps_in_piece, processed by outer pieces.
        current_stride = mode_stride
        remaining = mode_size

        for k in range(n_pieces - 1, -1, -1):
            if remaining <= 1:
                break
            piece_stride = split_strides[k]
            piece_end = piece_stride * group_shape[k]

            if current_stride >= piece_stride and current_stride < piece_end:
                steps_in_piece = piece_end // current_stride
                contrib = min(remaining, steps_in_piece)
                coverage[k] *= contrib
                if remaining > steps_in_piece:
                    remaining = remaining // steps_in_piece
                    current_stride = piece_end  # overflow stride
                else:
                    remaining = 0

    return coverage
