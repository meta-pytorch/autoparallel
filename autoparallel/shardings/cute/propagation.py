"""
CuTe-based sharding propagation rules.

TiledLayout = tensor_layout + tuple of mesh_tilers (one per mesh dim).
shard_layout = sequential logical_divide.

Propagation:
- View: tensor_layout changes, mesh_tilers invariant
- Transpose/Permute: reorder modes in tensor_layout and all tilers
- Slice: CuTe slice tensor_layout and all tilers
"""

from ._pycute import E, Layout, composition, is_tuple, product
from .placement import TiledLayout


# =============================================================================
# View / Reshape
# =============================================================================


def propagate_view(tiled, new_shape):
    """View: tensor_layout changes, mesh_tilers invariant."""
    if product(tiled.tensor_shape) != product(new_shape):
        return None
    return TiledLayout(Layout(new_shape), tiled.mesh_tilers)


# =============================================================================
# Transpose / Permute
# =============================================================================


def _make_perm_layout(layout, dims):
    """Build a permutation layout for the given layout and dim ordering."""
    shape = layout.shape if is_tuple(layout.shape) else (layout.shape,)
    perm_shape = tuple(shape[d] for d in dims)
    perm_stride = tuple(E(d) for d in dims)
    return Layout(perm_shape, perm_stride)


def propagate_transpose(tiled, dim0, dim1):
    """Transpose: composition with permutation layout using ScaledBasis."""
    t_shape = tiled.tensor_layout.shape if is_tuple(tiled.tensor_layout.shape) else (tiled.tensor_layout.shape,)
    ndim = len(t_shape)
    dims = list(range(ndim))
    dims[dim0], dims[dim1] = dims[dim1], dims[dim0]

    new_tensor = composition(tiled.tensor_layout, _make_perm_layout(tiled.tensor_layout, dims))
    new_tilers = tuple(composition(t, _make_perm_layout(t, dims)) for t in tiled.mesh_tilers)
    return TiledLayout(new_tensor, new_tilers)


def propagate_permute(tiled, dims):
    """Permute: composition with permutation layout using ScaledBasis."""
    new_tensor = composition(tiled.tensor_layout, _make_perm_layout(tiled.tensor_layout, dims))
    new_tilers = tuple(composition(t, _make_perm_layout(t, dims)) for t in tiled.mesh_tilers)
    return TiledLayout(new_tensor, new_tilers)


# =============================================================================
# Slice
# =============================================================================


def propagate_slice(tiled, dim, index):
    """Slice: CuTe slice tensor_layout and all tilers with same coord."""
    # Check if any tiler shards this dim
    t_shape = tiled.tensor_layout.shape if is_tuple(tiled.tensor_layout.shape) else (tiled.tensor_layout.shape,)

    for tiler in tiled.mesh_tilers:
        m_shape = tiler.shape if is_tuple(tiler.shape) else (tiler.shape,)
        if len(m_shape) == len(t_shape) and m_shape[dim] < t_shape[dim]:
            return None  # slicing a sharded dim

    ndim = len(t_shape)
    coord = tuple(index if k == dim else None for k in range(ndim))

    new_tensor = tiled.tensor_layout(*coord)
    new_tilers = tuple(t(*coord) for t in tiled.mesh_tilers)
    return TiledLayout(new_tensor, new_tilers)


# =============================================================================
# Gather
# =============================================================================


def propagate_gather(tiled, dim, index_layout):
    """Gather with CuTe-expressible index pattern along dim."""
    t_shape = tiled.tensor_layout.shape if is_tuple(tiled.tensor_layout.shape) else (tiled.tensor_layout.shape,)

    for tiler in tiled.mesh_tilers:
        m_shape = tiler.shape if is_tuple(tiler.shape) else (tiler.shape,)
        if len(m_shape) == len(t_shape) and m_shape[dim] < t_shape[dim]:
            return None

    t_stride = tiled.tensor_layout.stride if is_tuple(tiled.tensor_layout.stride) else (tiled.tensor_layout.stride,)
    dim_stride = t_stride[dim]

    new_dim_stride = (
        tuple(s * dim_stride for s in index_layout.stride)
        if is_tuple(index_layout.stride) else index_layout.stride * dim_stride
    )
    new_shape = t_shape[:dim] + (index_layout.shape,) + t_shape[dim + 1:]
    new_stride = t_stride[:dim] + (new_dim_stride,) + t_stride[dim + 1:]
    new_tensor = Layout(new_shape, new_stride)

    # Update tilers: replace dim with gathered size (was replicate = full)
    new_tilers = []
    for tiler in tiled.mesh_tilers:
        m_shape = tiler.shape if is_tuple(tiler.shape) else (tiler.shape,)
        m_stride = tiler.stride if is_tuple(tiler.stride) else (tiler.stride,)
        nm_shape = m_shape[:dim] + (index_layout.shape,) + m_shape[dim + 1:]
        nm_stride = m_stride[:dim] + (new_dim_stride,) + m_stride[dim + 1:]
        new_tilers.append(Layout(nm_shape, nm_stride))

    return TiledLayout(new_tensor, tuple(new_tilers))


# =============================================================================
# Einsum
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


def propagate_einsum(equation, tiled_a, tiled_b, output_shape):
    """Einsum: redistribution-free strategies."""
    inputs, output = _parse_einsum(equation)
    categories = _classify_einsum_dims(inputs, output)

    a_placements = tiled_a.get_placements()
    b_placements = tiled_b.get_placements()

    # Build output shard specs
    shard_specs = []
    is_partial = False

    # For each mesh dim in A
    for entry in a_placements:
        if entry[0] != "shard":
            continue
        a_dim, a_mesh = entry[1], entry[2]
        if a_dim >= len(inputs[0]):
            return None
        a_label = inputs[0][a_dim]
        cat = categories.get(a_label, "other")

        b_match = any(
            e[0] == "shard" and e[1] < len(inputs[1]) and inputs[1][e[1]] == a_label
            for e in b_placements
        )

        if cat == "m":
            if b_match:
                return None
            out_dim = output.index(a_label)
            shard_specs.append((out_dim, a_mesh))
        elif cat == "batch":
            if not b_match:
                return None
            out_dim = output.index(a_label)
            shard_specs.append((out_dim, a_mesh))
        elif cat == "contract":
            if not b_match:
                return None
            is_partial = True
        else:
            return None

    # For each mesh dim in B not already handled
    for entry in b_placements:
        if entry[0] != "shard":
            continue
        b_dim, b_mesh = entry[1], entry[2]
        if b_dim >= len(inputs[1]):
            return None
        b_label = inputs[1][b_dim]
        if any(e[0] == "shard" and e[1] < len(inputs[0]) and inputs[0][e[1]] == b_label for e in a_placements):
            continue  # already handled
        cat = categories.get(b_label, "other")
        if cat == "n":
            out_dim = output.index(b_label)
            shard_specs.append((out_dim, b_mesh))
        else:
            return None

    if shard_specs:
        out = TiledLayout.shard_multi(output_shape, shard_specs)
    else:
        out = TiledLayout.replicate(output_shape)

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
        if tiled.mesh_tilers != ref.mesh_tilers:
            if tiled.is_replicate():
                continue
            if ref.is_replicate():
                ref = tiled
                continue
            return None

    return TiledLayout(Layout(output_shape), ref.mesh_tilers)


# =============================================================================
# Reduction
# =============================================================================


def propagate_reduction(tiled, reduce_dim, keepdim, output_shape):
    """Reduction: sharded dim reduced -> Partial. Otherwise adjust."""
    placements = tiled.get_placements()

    for entry in placements:
        if entry[0] == "shard" and entry[1] == reduce_dim:
            result = TiledLayout.replicate(output_shape)
            result._is_partial = True
            return result

    # Rebuild for output shape
    shard_specs = []
    for entry in placements:
        if entry[0] == "shard":
            new_dim = entry[1] if entry[1] < reduce_dim or keepdim else entry[1] - 1
            shard_specs.append((new_dim, entry[2]))

    if shard_specs:
        return TiledLayout.shard_multi(output_shape, shard_specs)
    return TiledLayout.replicate(output_shape)
