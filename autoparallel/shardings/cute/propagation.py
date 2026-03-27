"""
CuTe-based sharding propagation rules.

All rules operate on TiledLayout = (tensor_layout, mesh_tiler).
shard_layout = logical_divide(tensor_layout, mesh_tiler) is derived.

Both tensor_layout and mesh_tiler transform under CuTe operations:
- View: tensor_layout changes, mesh_tiler invariant
- Transpose/Permute: reorder modes in both
- Slice: CuTe slice both with same coordinate

Rules are strictly redistribution-free.
"""

from ._pycute import Layout, is_tuple, product
from .placement import TiledLayout


# =============================================================================
# View / Reshape
# =============================================================================


def propagate_view(tiled, new_shape):
    """View: tensor_layout changes, mesh_tiler invariant."""
    if product(tiled.tensor_shape) != product(new_shape):
        return None
    return TiledLayout(Layout(new_shape), tiled.mesh_tiler)


# =============================================================================
# Transpose / Permute
# =============================================================================


def propagate_transpose(tiled, dim0, dim1):
    """Transpose: swap modes in both tensor_layout and mesh_tiler."""
    def _swap(layout, d0, d1):
        shape = list(layout.shape if is_tuple(layout.shape) else (layout.shape,))
        stride = list(layout.stride if is_tuple(layout.stride) else (layout.stride,))
        shape[d0], shape[d1] = shape[d1], shape[d0]
        stride[d0], stride[d1] = stride[d1], stride[d0]
        return Layout(tuple(shape), tuple(stride))

    return TiledLayout(_swap(tiled.tensor_layout, dim0, dim1),
                       _swap(tiled.mesh_tiler, dim0, dim1))


def propagate_permute(tiled, dims):
    """Permute: reorder modes in both tensor_layout and mesh_tiler."""
    def _permute(layout, dims):
        shape = layout.shape if is_tuple(layout.shape) else (layout.shape,)
        stride = layout.stride if is_tuple(layout.stride) else (layout.stride,)
        return Layout(tuple(shape[d] for d in dims), tuple(stride[d] for d in dims))

    return TiledLayout(_permute(tiled.tensor_layout, dims),
                       _permute(tiled.mesh_tiler, dims))


# =============================================================================
# Slice
# =============================================================================


def propagate_slice(tiled, dim, index):
    """Slice: CuTe slice both tensor_layout and mesh_tiler.

    If dim is sharded (tiler size < tensor size), returns None.
    """
    t_shape = tiled.tensor_layout.shape if is_tuple(tiled.tensor_layout.shape) else (tiled.tensor_layout.shape,)
    m_shape = tiled.mesh_tiler.shape if is_tuple(tiled.mesh_tiler.shape) else (tiled.mesh_tiler.shape,)

    if m_shape[dim] < t_shape[dim]:
        return None  # slicing a sharded dim requires redistribution

    ndim = len(t_shape)
    coord = tuple(index if k == dim else None for k in range(ndim))

    new_tensor = tiled.tensor_layout(*coord)
    new_tiler = tiled.mesh_tiler(*coord)

    return TiledLayout(new_tensor, new_tiler)


# =============================================================================
# Gather
# =============================================================================


def propagate_gather(tiled, dim, index_layout):
    """Gather with CuTe-expressible index pattern along dim.

    If dim is sharded, returns None.
    """
    t_shape = tiled.tensor_layout.shape if is_tuple(tiled.tensor_layout.shape) else (tiled.tensor_layout.shape,)
    m_shape = tiled.mesh_tiler.shape if is_tuple(tiled.mesh_tiler.shape) else (tiled.mesh_tiler.shape,)

    if m_shape[dim] < t_shape[dim]:
        return None

    # Replace dim in tensor_layout with gathered strides
    t_stride = tiled.tensor_layout.stride if is_tuple(tiled.tensor_layout.stride) else (tiled.tensor_layout.stride,)
    dim_stride = t_stride[dim]

    if is_tuple(index_layout.stride):
        new_dim_stride = tuple(s * dim_stride for s in index_layout.stride)
    else:
        new_dim_stride = index_layout.stride * dim_stride

    new_shape = t_shape[:dim] + (index_layout.shape,) + t_shape[dim + 1:]
    new_stride = t_stride[:dim] + (new_dim_stride,) + t_stride[dim + 1:]
    new_tensor = Layout(new_shape, new_stride)

    # Tiler: dim was replicate (full size), replace with gathered size
    m_stride = tiled.mesh_tiler.stride if is_tuple(tiled.mesh_tiler.stride) else (tiled.mesh_tiler.stride,)
    new_m_shape = m_shape[:dim] + (index_layout.shape,) + m_shape[dim + 1:]
    new_m_stride = m_stride[:dim] + (new_dim_stride,) + m_stride[dim + 1:]
    new_tiler = Layout(new_m_shape, new_m_stride)

    return TiledLayout(new_tensor, new_tiler)


# =============================================================================
# Einsum / Matmul
# =============================================================================


def _parse_einsum(equation):
    if "->" not in equation:
        raise ValueError(f"Einsum equation must contain '->': {equation}")
    inputs_str, output = equation.split("->")
    return inputs_str.split(","), output


def _classify_einsum_dims(inputs, output):
    assert len(inputs) == 2
    a_dims, b_dims = set(inputs[0]), set(inputs[1])
    out_dims = set(output)

    categories = {}
    for d in a_dims | b_dims | out_dims:
        in_a, in_b, in_out = d in a_dims, d in b_dims, d in out_dims
        if in_a and in_b and in_out:
            categories[d] = "batch"
        elif in_a and in_b and not in_out:
            categories[d] = "contract"
        elif in_a and not in_b and in_out:
            categories[d] = "m"
        elif not in_a and in_b and in_out:
            categories[d] = "n"
        else:
            categories[d] = "other"

    return categories


def _get_shard_info(tiled):
    """Get list of (dim, mesh_size) for sharded dims."""
    placements = tiled.get_placements()
    return [(entry[1], entry[2]) for entry in placements if entry[0] == "shard"]


def propagate_einsum(equation, tiled_a, tiled_b, output_shape):
    """Einsum: redistribution-free strategies."""
    inputs, output = _parse_einsum(equation)
    categories = _classify_einsum_dims(inputs, output)

    a_shards = _get_shard_info(tiled_a)
    b_shards = _get_shard_info(tiled_b)

    if not a_shards and not b_shards:
        return TiledLayout.replicate(output_shape)

    # Build output from each shard
    out = TiledLayout.replicate(output_shape)
    is_partial = False

    # Check each sharded dim
    for a_dim, a_mesh in a_shards:
        if a_dim >= len(inputs[0]):
            return None
        a_label = inputs[0][a_dim]
        cat = categories.get(a_label, "other")

        # Find matching shard in b
        b_match = None
        for b_dim, b_mesh in b_shards:
            if b_dim < len(inputs[1]) and inputs[1][b_dim] == a_label:
                b_match = (b_dim, b_mesh)
                break

        if cat == "m":
            if b_match:
                return None  # can't have both sharded on M dim
            out_dim = output.index(a_label)
            out = TiledLayout.shard(output_shape, out_dim, a_mesh)
        elif cat == "batch":
            if not b_match:
                return None  # batch must be sharded on both
            out_dim = output.index(a_label)
            out = TiledLayout.shard(output_shape, out_dim, a_mesh)
        elif cat == "contract":
            if not b_match:
                return None  # K shard needs both
            is_partial = True
        else:
            return None

    for b_dim, b_mesh in b_shards:
        if b_dim >= len(inputs[1]):
            return None
        b_label = inputs[1][b_dim]
        cat = categories.get(b_label, "other")
        # Skip if already handled as batch/contract
        if any(a_dim < len(inputs[0]) and inputs[0][a_dim] == b_label for a_dim, _ in a_shards):
            continue
        if cat == "n":
            out_dim = output.index(b_label)
            out = TiledLayout.shard(output_shape, out_dim, b_mesh)
        else:
            return None

    if is_partial:
        out._is_partial = True

    return out


# =============================================================================
# Pointwise
# =============================================================================


def propagate_pointwise(all_tileds, output_shape):
    """Pointwise: all inputs must have compatible mesh_tilers."""
    if not all_tileds:
        return None

    ref = all_tileds[0]
    for tiled in all_tileds[1:]:
        if tiled.mesh_tiler != ref.mesh_tiler:
            if tiled.is_replicate():
                continue
            if ref.is_replicate():
                ref = tiled
                continue
            return None

    return TiledLayout(Layout(output_shape), ref.mesh_tiler)


# =============================================================================
# Reduction
# =============================================================================


def propagate_reduction(tiled, reduce_dim, keepdim, output_shape):
    """Reduction: if sharded dim reduced -> Partial. Otherwise adjust."""
    placements = tiled.get_placements()

    for entry in placements:
        if entry[0] == "shard" and entry[1] == reduce_dim:
            result = TiledLayout.replicate(output_shape)
            result._is_partial = True
            return result

    for entry in placements:
        if entry[0] == "shard":
            new_dim = entry[1] if entry[1] < reduce_dim or keepdim else entry[1] - 1
            return TiledLayout.shard(output_shape, new_dim, entry[2])

    return TiledLayout.replicate(output_shape)
