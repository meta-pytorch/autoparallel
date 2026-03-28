"""
CuTe-based sharding propagation rules.

TiledLayout = tensor_layout + tuple of mesh_tilers (one per mesh dim).
shard_layout = sequential logical_divide.

Propagation:
- View: tensor_layout changes, mesh_tilers invariant
- Transpose/Permute: reorder modes in tensor_layout and all tilers
- Slice: CuTe slice tensor_layout and all tilers
"""

from ._pycute import E, Layout, codomain_divide, composition, is_tuple, product
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
    """Einsum via tensor folding (CuTe paper section 1.3).

    Works directly with CuTe tiler layouts — no get_placements().
    Each tiler mode is classified by the einsum equation labels.

    For rank-matched tilers (tiler rank == equation rank): direct mode
    classification via labels. For rank-mismatched (after view):
    codomain_divide decomposes the tiler through the tensor shape,
    recovering per-dim local sizes for classification.
    """
    inputs, output_labels = _parse_einsum(equation)
    categories = _classify_einsum_dims(inputs, output_labels)

    out_tilers = []
    is_partial = False

    n_mesh = max(len(tiled_a.mesh_tilers), len(tiled_b.mesh_tilers))

    for idx in range(n_mesh):
        a_tiler = tiled_a.mesh_tilers[idx] if idx < len(tiled_a.mesh_tilers) else None
        b_tiler = tiled_b.mesh_tilers[idx] if idx < len(tiled_b.mesh_tilers) else None

        a_cat = _classify_tiler(a_tiler, tiled_a.tensor_layout, inputs[0], categories) if a_tiler else None
        b_cat = _classify_tiler(b_tiler, tiled_b.tensor_layout, inputs[1], categories) if b_tiler else None

        if a_cat is None and b_cat is None:
            continue

        if a_cat == "m":
            if b_cat is not None:
                return None
            out_tilers.append(_tiler_for_output(
                a_tiler, tiled_a.tensor_layout, inputs[0],
                categories, output_labels, output_shape))
        elif a_cat == "batch":
            if b_cat != "batch":
                return None
            out_tilers.append(_tiler_for_output(
                a_tiler, tiled_a.tensor_layout, inputs[0],
                categories, output_labels, output_shape))
        elif a_cat == "contract":
            if b_cat != "contract":
                return None
            is_partial = True
        elif a_cat is None and b_cat == "n":
            out_tilers.append(_tiler_for_output(
                b_tiler, tiled_b.tensor_layout, inputs[1],
                categories, output_labels, output_shape))
        else:
            return None

    out = TiledLayout(Layout(output_shape), tuple(out_tilers))
    if is_partial:
        out._is_partial = True
    return out


def _get_local_sizes(tiler, tensor_layout):
    """Get per-dim local sizes for a tiler relative to its tensor.

    For rank-matched tilers, returns tiler shape directly.
    For rank-mismatched (after view), uses codomain_divide.
    Returns a dict {dim_index: local_size}.
    """
    t_shape = tensor_layout.shape if is_tuple(tensor_layout.shape) else (tensor_layout.shape,)
    m_shape = tiler.shape if is_tuple(tiler.shape) else (tiler.shape,)

    if len(m_shape) == len(t_shape):
        return {i: (product(m_shape[i]) if is_tuple(m_shape[i]) else m_shape[i])
                for i in range(len(t_shape))}
    else:
        return codomain_divide(tiler, tuple(t_shape))


def _classify_tiler(tiler, tensor_layout, input_labels, categories):
    """Find the einsum category of the dim a tiler shards.

    Returns 'm', 'n', 'batch', 'contract', or None (replicate).
    """
    t_shape = tensor_layout.shape if is_tuple(tensor_layout.shape) else (tensor_layout.shape,)
    local_sizes = _get_local_sizes(tiler, tensor_layout)

    for i in range(min(len(t_shape), len(input_labels))):
        t_s = product(t_shape[i]) if is_tuple(t_shape[i]) else t_shape[i]
        l_s = local_sizes.get(i, t_s)
        if l_s < t_s:
            return categories.get(input_labels[i])
    return None


def _tiler_for_output(tiler, tensor_layout, input_labels,
                      categories, output_labels, output_shape):
    """Build output tiler from an input tiler using equation labels.

    Rank-matched (tiler rank == equation rank): per-dim local sizes are
    mapped to output positions via labels. Clean, no stride comparison.

    Rank-mismatched (after view): preserves tiler mode structure via stride
    scaling. Each mode is classified as M/batch vs K by comparing its stride
    against the input tensor's strides. M strides are scaled, K modes dropped.
    """
    t_shape = tiler.shape if is_tuple(tiler.shape) else (tiler.shape,)
    o_shape = output_shape if is_tuple(output_shape) else (output_shape,)

    # Output row-major strides
    o_strides = [1] * len(o_shape)
    for k in range(len(o_shape) - 2, -1, -1):
        o_strides[k] = o_strides[k + 1] * o_shape[k + 1]

    # --- Rank-matched: direct label-based transformation ---
    if len(t_shape) == len(input_labels):
        new_shape = []
        new_stride = []
        output_dims_added = set()

        for i, label in enumerate(input_labels):
            cat = categories.get(label, "other")
            if cat == "contract":
                continue  # skip K dims
            out_idx = list(output_labels).index(label)
            output_dims_added.add(out_idx)
            new_shape.append(t_shape[i])
            new_stride.append(o_strides[out_idx])

        # Add output dims not from input (N for A, M for B)
        for i in range(len(o_shape)):
            if i not in output_dims_added:
                new_shape.append(o_shape[i])
                new_stride.append(o_strides[i])

        if len(new_shape) == 1:
            return Layout(new_shape[0], new_stride[0])
        return Layout(tuple(new_shape), tuple(new_stride))

    # --- Rank-mismatched: stride-based mode classification ---
    # Preserves hierarchical mode structure from the original tensor shape.
    t_stride = tiler.stride if is_tuple(tiler.stride) else (tiler.stride,)
    i_shape = tensor_layout.shape if is_tuple(tensor_layout.shape) else (tensor_layout.shape,)

    # Input row-major strides
    i_strides = [1] * len(i_shape)
    for k in range(len(i_shape) - 2, -1, -1):
        i_strides[k] = i_strides[k + 1] * i_shape[k + 1]

    # Compute totals for stride scaling
    k_total = 1
    other_total = 1
    for i, label in enumerate(input_labels):
        cat = categories.get(label)
        if cat == "contract":
            k_total *= i_shape[i]
    # "other" = dims from the OTHER input that appear in output but not this one
    for i in range(len(o_shape)):
        if output_labels[i] not in input_labels:
            other_total *= o_shape[i]

    m_stride_in = max(i_strides)
    scale = other_total // k_total if k_total > 0 else 1

    new_shape = []
    new_stride = []
    for i in range(len(t_shape)):
        s = t_stride[i]
        if s == 0:
            new_shape.append(t_shape[i])
            new_stride.append(0)
        elif s < m_stride_in:
            continue  # K mode — skip
        else:
            new_shape.append(t_shape[i])
            new_stride.append(s * scale)

    # Add the replacement dim (N for A, M for B)
    new_shape.append(other_total)
    new_stride.append(1)

    if len(new_shape) == 1:
        return Layout(new_shape[0], new_stride[0])
    return Layout(tuple(new_shape), tuple(new_stride))


# =============================================================================
# Pointwise
# =============================================================================


def propagate_pointwise(all_tileds, output_shape):
    """Pointwise: all inputs must have compatible mesh_tilers.

    Pointwise is identity on coordinates — same shard pattern passes through.
    """
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


def _make_reduce_layout(layout, reduce_dim):
    """Build projection layout that removes reduce_dim using E(k) strides."""
    shape = layout.shape if is_tuple(layout.shape) else (layout.shape,)
    ndim = len(shape)
    out_shape = []
    out_stride = []
    for k in range(ndim):
        if k != reduce_dim:
            out_shape.append(shape[k])
            out_stride.append(E(k))
    if len(out_shape) == 1:
        return Layout(out_shape[0], out_stride[0])
    return Layout(tuple(out_shape), tuple(out_stride))


def propagate_reduction(tiled, reduce_dim, keepdim, output_shape):
    """Reduction = composition with projection that removes reduce_dim.

    If sharded dim reduced -> Partial (tiler becomes replicate after projection).
    """
    # Check if reduce_dim is sharded
    placements = tiled.get_placements()
    is_sharded_reduce = any(
        e[0] == "shard" and e[1] == reduce_dim for e in placements
    )

    if keepdim:
        # keepdim: output has same rank, reduced dim size 1
        # Use slice at index 0 on the reduced dim (size 1 = same as any index)
        new_tensor = Layout(output_shape)
        # Tilers: reduce_dim becomes size 1
        new_tilers = []
        for tiler in tiled.mesh_tilers:
            t_shape = tiler.shape if is_tuple(tiler.shape) else (tiler.shape,)
            new_shape = tuple(1 if k == reduce_dim else t_shape[k] for k in range(len(t_shape)))
            t_stride = tiler.stride if is_tuple(tiler.stride) else (tiler.stride,)
            new_tilers.append(Layout(new_shape, t_stride))
        result = TiledLayout(new_tensor, tuple(new_tilers))
    else:
        # Project out reduce_dim via composition with E(k) projection
        new_tensor = composition(tiled.tensor_layout, _make_reduce_layout(tiled.tensor_layout, reduce_dim))
        new_tilers = tuple(
            composition(t, _make_reduce_layout(t, reduce_dim))
            for t in tiled.mesh_tilers
        )
        result = TiledLayout(new_tensor, new_tilers)

    if is_sharded_reduce:
        result._is_partial = True

    return result
