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
    """Einsum via tensor folding (CuTe paper section 1.3).

    Classify dims as M̂ (row), N̂ (col), K̂ (reduction), L̂ (batch).
    For each tiler, determine which dim it shards, apply GEMM rules,
    and transform the tiler for the output element space.

    M-dim tilers from A pass to C with strides scaled by N_total / K_total.
    N-dim tilers from B pass to C with strides scaled by M_total / K_total.
    K-dim tilers -> Partial. L-dim tilers pass through if both inputs match.
    """
    inputs, output = _parse_einsum(equation)
    categories = _classify_einsum_dims(inputs, output)

    a_placements = tiled_a.get_placements()
    b_placements = tiled_b.get_placements()

    # Compute dim sizes for stride scaling
    a_shape = tiled_a.tensor_layout.shape if is_tuple(tiled_a.tensor_layout.shape) else (tiled_a.tensor_layout.shape,)
    b_shape = tiled_b.tensor_layout.shape if is_tuple(tiled_b.tensor_layout.shape) else (tiled_b.tensor_layout.shape,)

    k_total = 1
    n_total = 1
    for i, label in enumerate(inputs[0]):
        if categories.get(label) == "contract":
            k_total *= a_shape[i]
    for i, label in enumerate(inputs[1]):
        if categories.get(label) == "n":
            n_total *= b_shape[i]

    m_total = 1
    for i, label in enumerate(inputs[0]):
        if categories.get(label) == "m":
            m_total *= a_shape[i]

    out_tilers = []
    is_partial = False

    for idx, entry in enumerate(a_placements):
        if entry[0] != "shard":
            continue
        a_dim = entry[1]
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
            # M-dim: transform tiler from A's element space to C's
            # Scale M strides by (N_total / K_total), replace K modes with N
            tiler = tiled_a.mesh_tilers[idx]
            new_tiler = _transform_tiler_for_output(
                tiler, inputs[0], categories, a_shape,
                output, output_shape, "m", n_total, k_total
            )
            out_tilers.append(new_tiler)
        elif cat == "batch":
            if not b_match:
                return None
            tiler = tiled_a.mesh_tilers[idx]
            # Batch: same elements, but K replaced by N in output
            new_tiler = _transform_tiler_for_output(
                tiler, inputs[0], categories, a_shape,
                output, output_shape, "batch", n_total, k_total
            )
            out_tilers.append(new_tiler)
        elif cat == "contract":
            if not b_match:
                return None
            is_partial = True
        else:
            return None

    for idx, entry in enumerate(b_placements):
        if entry[0] != "shard":
            continue
        b_dim = entry[1]
        if b_dim >= len(inputs[1]):
            return None
        b_label = inputs[1][b_dim]
        if any(
            e[0] == "shard" and e[1] < len(inputs[0]) and inputs[0][e[1]] == b_label
            for e in a_placements
        ):
            continue
        cat = categories.get(b_label, "other")
        if cat == "n":
            tiler = tiled_b.mesh_tilers[idx]
            new_tiler = _transform_tiler_for_output(
                tiler, inputs[1], categories, b_shape,
                output, output_shape, "n", m_total, k_total
            )
            out_tilers.append(new_tiler)
        else:
            return None

    out = TiledLayout(Layout(output_shape), tuple(out_tilers))
    if is_partial:
        out._is_partial = True
    return out


def _transform_tiler_for_output(tiler, input_labels, categories, input_shape,
                                 output_labels, output_shape, shard_cat,
                                 other_total, k_total):
    """Transform a tiler from input element space to output element space.

    Detects which tiler modes are M/batch vs K by comparing strides
    against the input tensor's strides. M/batch strides are scaled by
    output_inner / input_inner. K modes are replaced by N/M (full coverage).
    """
    t_shape = tiler.shape if is_tuple(tiler.shape) else (tiler.shape,)
    t_stride = tiler.stride if is_tuple(tiler.stride) else (tiler.stride,)
    i_shape = input_shape if is_tuple(input_shape) else (input_shape,)
    o_shape = output_shape if is_tuple(output_shape) else (output_shape,)

    # Compute output row-major strides
    o_strides = [1] * len(o_shape)
    for k in range(len(o_shape) - 2, -1, -1):
        o_strides[k] = o_strides[k + 1] * o_shape[k + 1]

    # Compute input row-major strides
    i_strides = [1] * len(i_shape)
    for k in range(len(i_shape) - 2, -1, -1):
        i_strides[k] = i_strides[k + 1] * i_shape[k + 1]

    # If tiler rank matches input labels, do mode-by-mode transformation
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
            new_stride.append(o_strides[out_idx] * t_shape[i] // t_shape[i]
                             if t_shape[i] == i_shape[i]
                             else o_strides[out_idx])
            # Simpler: use output stride for this dim's position
            new_stride[-1] = o_strides[out_idx]

        # Add output dims not from input
        for i in range(len(o_shape)):
            if i not in output_dims_added:
                new_shape.append(o_shape[i])
                new_stride.append(o_strides[i])

        if len(new_shape) == 1:
            return Layout(new_shape[0], new_stride[0])
        return Layout(tuple(new_shape), tuple(new_stride))

    # Tiler rank differs from input rank (e.g., after view).
    # Detect M vs K modes by stride comparison.
    # K modes have strides < input's M-dim stride.
    # For 'mk,kn->mn': K is the innermost dim of input.

    # Find the boundary stride: strides >= input's outermost K stride are M modes.
    # Input has K dims with known strides. Find the max M stride.
    k_dim_strides = set()
    for i, label in enumerate(input_labels):
        if categories.get(label) == "contract":
            k_dim_strides.add(i_strides[i])

    # The M-dim stride in the input is the stride of the first non-K dim
    # For (32, 16) with 'mk': m stride = 16, k stride = 1.
    m_stride_in = max(i_strides)  # outermost stride = M stride
    k_stride_in = min(i_strides)  # innermost stride = K stride

    # Scale factor: how M strides change from input to output
    # Input M stride = product of K dims. Output M stride = product of N dims.
    scale = other_total // k_total if k_total > 0 else 1

    new_shape = []
    new_stride = []

    for i in range(len(t_shape)):
        s = t_stride[i]
        if s == 0:
            # stride 0 mode: pass through
            new_shape.append(t_shape[i])
            new_stride.append(0)
        elif s >= k_stride_in and s < m_stride_in:
            # This mode's stride is in the K-dim range — it IS a K mode
            # Replace with corresponding N/output dim
            continue  # skip K modes
        else:
            # M/batch mode — scale stride
            new_shape.append(t_shape[i])
            new_stride.append(s * scale)

    # Add the N/output dim that replaces K. Size = other_total (N for A, M for B).
    new_shape.append(other_total)
    new_stride.append(1)  # innermost stride

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
