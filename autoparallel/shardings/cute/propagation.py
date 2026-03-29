"""
CuTe-based sharding propagation rules.

ShardedLayout = hierarchical CuTe Layout (local, mesh) per dim.

Propagation:
- View: reshape hier_layout to match new shape (merge/split local parts, mesh invariant)
- Transpose/Permute: swap top-level modes in hier_layout
- Slice: partial evaluation on hier_layout
- Pointwise: broadcast hierarchical shapes
- Reduction: drop dim (flat=normal, hierarchical=Partial)
- Einsum: = broadcast + reduce (ranks always match thanks to view)
"""

from ._pycute import Layout, flatten, is_tuple, logical_divide, product, suffix_product
from .placement import ShardedLayout

from torch.distributed.tensor._ops._view_ops import (
    Flatten,
    InputDim,
    Split,
    view_groups,
)


# =============================================================================
# View / Reshape
# =============================================================================


def _get_local(mode_shape):
    """Get the local (scalar) size from a hierarchical mode shape."""
    if _is_level2(mode_shape):
        # Level-2: product of each sub-dim's local
        result = 1
        for sub in mode_shape:
            result *= _get_local(sub)
        return result
    if is_tuple(mode_shape):
        return mode_shape[0]
    return mode_shape


def _get_mesh(mode_shape):
    """Get the mesh modes from a hierarchical mode shape. () if flat."""
    if _is_level2(mode_shape):
        # Level-2: collect mesh from all sub-dims
        result = []
        for sub in mode_shape:
            result.extend(_get_mesh(sub))
        return tuple(result)
    if is_tuple(mode_shape):
        return mode_shape[1:]
    return ()


def _make_mode(local, mesh_modes):
    """Build a hierarchical mode: scalar if flat, tuple if sharded."""
    if not mesh_modes:
        return local
    return (local,) + tuple(mesh_modes)


def _is_level2(mode):
    """Level-2 = tuple where ALL elements are tuples (each is a sub-dim)."""
    if not is_tuple(mode):
        return False
    return all(is_tuple(m) for m in mode)


def _wrap_subdim(s):
    """Wrap a flat scalar or level-3 mode as a sub-dim for level-2 grouping."""
    if is_tuple(s):
        return s
    return (s,)


def _unwrap_subdim(s):
    """Unwrap a 1-tuple back to scalar: (4,) → 4."""
    if is_tuple(s) and len(s) == 1:
        return s[0]
    return s


def _get_subdims(mode, mode_stride):
    """Get list of (sub_shape, sub_stride) from a mode."""
    if _is_level2(mode):
        return list(zip(mode, mode_stride))
    else:
        return [(_wrap_subdim(mode), mode_stride)]


def _split_subdim(sub, sub_st, split_point):
    """Split a sub-dim at a global-size boundary.
    Returns ((left_shape, left_stride), (right_shape, right_stride)) or None."""
    gs = product(sub)
    if gs == split_point:
        return (sub, sub_st), None
    if split_point <= 0 or gs % split_point != 0:
        return None

    if len(sub) == 1:
        # Flat sub-dim (X,): split into two flat pieces
        local = sub[0]
        local_st = sub_st[0] if is_tuple(sub_st) else sub_st
        left = (split_point,)
        right = (local // split_point,)
        left_st = (local_st,)
        right_st = (local_st * split_point,)
        return (left, left_st), (right, right_st)
    else:
        # Sharded sub-dim (local, mesh...): split the local part
        local = sub[0]
        mesh = sub[1:]
        local_st = sub_st[0] if is_tuple(sub_st) else sub_st
        mesh_st = sub_st[1:] if is_tuple(sub_st) else ()
        mesh_product = product(mesh)

        if split_point >= mesh_product and split_point % mesh_product == 0:
            # Mesh fits in left piece
            left_local = split_point // mesh_product
            right_local = local // left_local
            left = (left_local,) + tuple(mesh)
            right = (right_local,)
            left_st = (local_st,) + tuple(mesh_st)
            right_st = (local_st * left_local * mesh_product,)
            return (left, left_st), (right, right_st)
        else:
            return None


def _collect_input_dims(op):
    """Recursively collect InputDim indices from a view_groups op."""
    if isinstance(op, InputDim):
        return [op.input_dim]
    elif isinstance(op, Flatten):
        dims = []
        for sub in op.input_dims:
            dims.extend(_collect_input_dims(sub))
        return dims
    elif isinstance(op, Split):
        return _collect_input_dims(op.input_dim)
    return []


def _view_hier(hier_layout, old_global, new_global):
    """Transform hier_layout using 3-level nesting.

    Uses view_groups for merge/split grouping:
    - InputDim: pass through
    - Flatten: nest source dims into level-2 tuple (merge = nest)
    - Split: peel level-2 if aligned, split sub-dims if not (split = peel)
    """
    h_shape = hier_layout.shape if is_tuple(hier_layout.shape) else (hier_layout.shape,)
    h_stride = hier_layout.stride if is_tuple(hier_layout.stride) else (hier_layout.stride,)
    old_g = old_global if is_tuple(old_global) else (old_global,)
    new_g = new_global if is_tuple(new_global) else (new_global,)

    ops = view_groups(old_g, new_g)

    # Extract per-source-dim sub-dims
    src_subdims = []
    for s, st in zip(h_shape, h_stride):
        src_subdims.append(_get_subdims(s, st))

    # Identify split groups
    split_groups = {}
    for out_i, op in enumerate(ops):
        if isinstance(op, Split):
            key = id(op.input_dim)
            if key not in split_groups:
                split_groups[key] = {
                    "input_op": op.input_dim,
                    "group_shape": op.group_shape,
                    "outputs": [],
                }
            split_groups[key]["outputs"].append((out_i, op.split_id))

    out_shape = [None] * len(new_g)
    out_stride = [None] * len(new_g)
    processed = set()

    for out_i, op in enumerate(ops):
        if out_i in processed:
            continue

        if isinstance(op, InputDim):
            out_shape[out_i] = h_shape[op.input_dim]
            out_stride[out_i] = h_stride[op.input_dim]

        elif isinstance(op, Flatten):
            # Merge: concatenate sub-dim lists from all source dims
            input_dims = _collect_input_dims(op)
            merged = []
            for d in input_dims:
                merged.extend(src_subdims[d])

            if len(merged) == 1:
                out_shape[out_i] = _unwrap_subdim(merged[0][0])
                out_stride[out_i] = merged[0][1]
            else:
                out_shape[out_i] = tuple(s for s, _ in merged)
                out_stride[out_i] = tuple(st for _, st in merged)

        elif isinstance(op, Split):
            group = split_groups[id(op.input_dim)]
            input_dims = _collect_input_dims(group["input_op"])
            group_shape = group["group_shape"]
            outputs = sorted(group["outputs"], key=lambda x: x[1])

            # Collect all sub-dims from source
            all_subs = []
            for d in input_dims:
                all_subs.extend(src_subdims[d])

            # Greedily assign sub-dims to each split piece,
            # splitting individual sub-dims at boundaries when needed
            sub_idx = 0
            pieces = []
            ok = True

            for target_size in group_shape:
                acc = 1
                piece = []

                while acc < target_size and sub_idx < len(all_subs):
                    sub_s, sub_st = all_subs[sub_idx]
                    gs = product(sub_s)

                    if acc * gs <= target_size:
                        acc *= gs
                        piece.append((sub_s, sub_st))
                        sub_idx += 1
                    else:
                        needed = target_size // acc
                        result = _split_subdim(sub_s, sub_st, needed)
                        if result is None:
                            ok = False
                            break
                        (left_s, left_st), right = result
                        piece.append((left_s, left_st))
                        acc *= product(left_s)
                        if right is not None:
                            all_subs[sub_idx] = right
                        else:
                            sub_idx += 1
                        break

                if not ok or acc != target_size:
                    return None
                pieces.append(piece)

            for (oi, sid), piece in zip(outputs, pieces):
                if len(piece) == 1:
                    out_shape[oi] = _unwrap_subdim(piece[0][0])
                    out_stride[oi] = piece[0][1]
                else:
                    out_shape[oi] = tuple(s for s, _ in piece)
                    out_stride[oi] = tuple(st for _, st in piece)
                processed.add(oi)

            for oi, _ in outputs:
                processed.add(oi)

    if any(s is None for s in out_shape):
        return None
    if len(out_shape) == 1:
        return Layout(out_shape[0], out_stride[0])
    return Layout(tuple(out_shape), tuple(out_stride))


def propagate_view(sharded, new_shape):
    """View: reshape hier_layout using 3-level nesting."""
    if product(sharded.global_shape) != product(new_shape):
        return None
    new_hier = _view_hier(sharded.hier_layout, sharded.global_shape, new_shape)
    if new_hier is None:
        return None
    return ShardedLayout(new_hier)


# =============================================================================
# Transpose / Permute
# =============================================================================


def propagate_transpose(sharded, dim0, dim1):
    """Transpose: swap top-level modes in hier_layout."""
    g_shape = sharded.global_shape if is_tuple(sharded.global_shape) else (sharded.global_shape,)
    ndim = len(g_shape)
    dims = list(range(ndim))
    dims[dim0], dims[dim1] = dims[dim1], dims[dim0]
    new_global = tuple(g_shape[d] for d in dims)

    h_shape = sharded.hier_layout.shape if is_tuple(sharded.hier_layout.shape) else (sharded.hier_layout.shape,)
    h_stride = sharded.hier_layout.stride if is_tuple(sharded.hier_layout.stride) else (sharded.hier_layout.stride,)
    new_h_shape = tuple(h_shape[d] for d in dims)
    new_h_stride = tuple(h_stride[d] for d in dims)
    new_hier = Layout(new_h_shape, new_h_stride)
    return ShardedLayout(new_hier)


def propagate_permute(sharded, dims):
    """Permute: reorder top-level modes in hier_layout."""
    g_shape = sharded.global_shape if is_tuple(sharded.global_shape) else (sharded.global_shape,)
    new_global = tuple(g_shape[d] for d in dims)

    h_shape = sharded.hier_layout.shape if is_tuple(sharded.hier_layout.shape) else (sharded.hier_layout.shape,)
    h_stride = sharded.hier_layout.stride if is_tuple(sharded.hier_layout.stride) else (sharded.hier_layout.stride,)
    new_h_shape = tuple(h_shape[d] for d in dims)
    new_h_stride = tuple(h_stride[d] for d in dims)
    new_hier = Layout(new_h_shape, new_h_stride)
    return ShardedLayout(new_hier)


# =============================================================================
# Slice
# =============================================================================


def propagate_slice(sharded, dim, index):
    """Slice: partial evaluation on hier_layout."""
    h_shape = sharded.hier_layout.shape if is_tuple(sharded.hier_layout.shape) else (sharded.hier_layout.shape,)

    # Check if dim is sharded (hierarchical shape)
    if dim < len(h_shape) and is_tuple(h_shape[dim]):
        local_size = h_shape[dim][0]
        g_shape = sharded.global_shape if is_tuple(sharded.global_shape) else (sharded.global_shape,)
        if local_size < g_shape[dim]:
            return None  # slicing a sharded dim

    ndim = len(h_shape)
    coord = tuple(index if k == dim else None for k in range(ndim))

    new_hier = sharded.hier_layout(*coord)
    g_shape = sharded.global_shape if is_tuple(sharded.global_shape) else (sharded.global_shape,)
    new_global = tuple(s for i, s in enumerate(g_shape) if i != dim)
    return ShardedLayout(new_hier)


# =============================================================================
# Gather
# =============================================================================


def propagate_gather(sharded, dim, index_layout):
    """Gather with CuTe-expressible index pattern along dim."""
    h_shape = sharded.hier_layout.shape if is_tuple(sharded.hier_layout.shape) else (sharded.hier_layout.shape,)

    # Check if dim is sharded
    if dim < len(h_shape) and is_tuple(h_shape[dim]):
        g_shape = sharded.global_shape if is_tuple(sharded.global_shape) else (sharded.global_shape,)
        local_size = h_shape[dim][0]
        if local_size < g_shape[dim]:
            return None

    h_stride = sharded.hier_layout.stride if is_tuple(sharded.hier_layout.stride) else (sharded.hier_layout.stride,)
    dim_stride = h_stride[dim]

    new_dim_stride = (
        tuple(s * dim_stride for s in index_layout.stride)
        if is_tuple(index_layout.stride) else index_layout.stride * dim_stride
    )
    new_shape = h_shape[:dim] + (index_layout.shape,) + h_shape[dim + 1:]
    new_stride = h_stride[:dim] + (new_dim_stride,) + h_stride[dim + 1:]
    new_hier = Layout(new_shape, new_stride)

    g_shape = sharded.global_shape if is_tuple(sharded.global_shape) else (sharded.global_shape,)
    g_dim_size = index_layout.shape if not is_tuple(index_layout.shape) else product(index_layout.shape)
    new_global = g_shape[:dim] + (g_dim_size,) + g_shape[dim + 1:]
    return ShardedLayout(new_hier)


# =============================================================================
# Pointwise — broadcast hierarchical shapes
# =============================================================================


def _hier_broadcast_shapes(a_shape, b_shape):
    """Broadcast two hierarchical shapes.

    Rules:
    - Both flat, same size: OK
    - One is size 1 (broadcast): other wins
    - One sharded, other size 1: sharded wins
    - One sharded, other flat same size: INCOMPATIBLE (different device assignment)
    - Both sharded: must match exactly
    """
    result = []
    for sa, sb in zip(a_shape, b_shape):
        if sa == sb:
            result.append(sa)
        elif sa == 1:
            result.append(sb)
        elif sb == 1:
            result.append(sa)
        elif is_tuple(sa) and is_tuple(sb):
            return None  # both sharded differently
        elif is_tuple(sa) and not is_tuple(sb):
            # sa sharded, sb flat — incompatible unless sb == 1 (handled above)
            return None
        elif not is_tuple(sa) and is_tuple(sb):
            return None
        else:
            return None  # different flat sizes
    return tuple(result)


def propagate_pointwise(all_shardeds, output_shape):
    """Pointwise: broadcast hierarchical shapes."""
    if not all_shardeds:
        return None

    ref = all_shardeds[0]
    ref_shape = ref.hier_layout.shape if is_tuple(ref.hier_layout.shape) else (ref.hier_layout.shape,)

    for sharded in all_shardeds[1:]:
        if sharded.is_replicate():
            continue
        if ref.is_replicate():
            ref = sharded
            ref_shape = ref.hier_layout.shape if is_tuple(ref.hier_layout.shape) else (ref.hier_layout.shape,)
            continue

        s_shape = sharded.hier_layout.shape if is_tuple(sharded.hier_layout.shape) else (sharded.hier_layout.shape,)
        if len(s_shape) != len(ref_shape):
            return None
            return None

        bc = _hier_broadcast_shapes(ref_shape, s_shape)
        if bc is None:
            return None
        # Use the sharded one as ref (non-replicate wins)
        if ref.is_replicate() and not sharded.is_replicate():
            ref = sharded
            ref_shape = s_shape

    return ShardedLayout(ref.hier_layout)


# =============================================================================
# Reduction
# =============================================================================


def _is_hier_dim_sharded(mode_shape):
    """Check if a hierarchical mode shape is sharded (has mesh modes with product > 1)."""
    if is_tuple(mode_shape) and len(mode_shape) > 1:
        return product(mode_shape[1:]) > 1
    return False


def propagate_reduction(sharded, reduce_dim, keepdim, output_shape):
    """Reduction: drop dim from hier_layout.

    If reduced dim is hierarchical (sharded) → Partial.
    """
    h_shape = sharded.hier_layout.shape if is_tuple(sharded.hier_layout.shape) else (sharded.hier_layout.shape,)
    is_sharded_reduce = _is_hier_dim_sharded(h_shape[reduce_dim] if reduce_dim < len(h_shape) else 1)

    # Remove or collapse the reduced dim
    ndim = len(h_shape)
    if keepdim:
        coord = tuple(0 if k == reduce_dim else None for k in range(ndim))
        new_hier = sharded.hier_layout(*coord)
        # Re-insert size-1 dim at reduce_dim position
        nh_shape = new_hier.shape if is_tuple(new_hier.shape) else (new_hier.shape,)
        nh_stride = new_hier.stride if is_tuple(new_hier.stride) else (new_hier.stride,)
        ins_shape = nh_shape[:reduce_dim] + (1,) + nh_shape[reduce_dim:]
        ins_stride = nh_stride[:reduce_dim] + (0,) + nh_stride[reduce_dim:]
        new_hier = Layout(ins_shape, ins_stride)
    else:
        coord = tuple(0 if k == reduce_dim else None for k in range(ndim))
        new_hier = sharded.hier_layout(*coord)

    result = ShardedLayout(new_hier)
    if is_sharded_reduce:
        result._is_partial = True
    return result


# =============================================================================
# Einsum = broadcast + reduce
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


def propagate_einsum(equation, sharded_a, sharded_b, output_shape):
    """Einsum = unsqueeze + broadcast + reduce.

    Works at the original dim level via hierarchical shapes.
    """
    inputs, output_labels = _parse_einsum(equation)
    categories = _classify_einsum_dims(inputs, output_labels)

    a_shape = sharded_a.hier_layout.shape
    a_shape = a_shape if is_tuple(a_shape) else (a_shape,)
    b_shape = sharded_b.hier_layout.shape
    b_shape = b_shape if is_tuple(b_shape) else (b_shape,)

    # Build the joint label space from the equation
    all_labels = list(dict.fromkeys(list(inputs[0]) + list(inputs[1])))

    # Expand A and B to joint space (unsqueeze missing dims as size 1)
    a_expanded = []
    for label in all_labels:
        if label in inputs[0]:
            idx = list(inputs[0]).index(label)
            a_expanded.append(a_shape[idx] if idx < len(a_shape) else 1)
        else:
            a_expanded.append(1)

    b_expanded = []
    for label in all_labels:
        if label in inputs[1]:
            idx = list(inputs[1]).index(label)
            b_expanded.append(b_shape[idx] if idx < len(b_shape) else 1)
        else:
            b_expanded.append(1)

    # Broadcast
    bc = _hier_broadcast_shapes(tuple(a_expanded), tuple(b_expanded))
    if bc is None:
        return None

    # Check compatibility for einsum rules
    for i, label in enumerate(all_labels):
        cat = categories.get(label)
        a_sharded = _is_hier_dim_sharded(a_expanded[i])
        b_sharded = _is_hier_dim_sharded(b_expanded[i])

        if cat == "m" and b_sharded:
            return None  # M sharded in B
        if cat == "n" and a_sharded:
            return None  # N sharded in A
        if cat == "batch":
            if a_sharded != b_sharded:
                return None  # batch must match
        if cat == "contract":
            if a_sharded != b_sharded:
                return None  # K must match on both

    # Reduce: drop contracted dims, detect Partial
    is_partial = False
    out_hier_shape = []
    for i, label in enumerate(all_labels):
        cat = categories.get(label)
        if cat == "contract":
            if _is_hier_dim_sharded(bc[i]):
                is_partial = True
            continue  # drop from output
        if label in output_labels:
            out_hier_shape.append(bc[i])

    # Build output hier_layout preserving 3-level structure from broadcast.
    # The broadcast gives us the correct SHAPE; we need OUTPUT strides.
    # Strategy: construct a replicate output layout, then use _view_hier
    # to reshape it to match the broadcast structure. The broadcast shape
    # encodes which dims are sharded, and _view_hier computes the strides.
    o_shape = output_shape if is_tuple(output_shape) else (output_shape,)

    # Collect output modes from broadcast (non-contracted, in output order)
    out_modes = []
    for i, label in enumerate(all_labels):
        if categories.get(label) == "contract":
            continue
        if label in output_labels:
            out_modes.append(bc[i])

    # Compute output local sizes
    out_local = tuple(_get_local(mode) for mode in out_modes)

    # Build output hier with output-space strides
    from .placement import _clean_hier
    out_hier = _clean_hier(logical_divide(Layout(o_shape), out_local))

    # If broadcast has level-2 structure (from merged dims), overlay it.
    # The simple logical_divide gives flat (local, mesh) per dim.
    # The broadcast may have level-2 nesting ((sub0, sub1), ...) from view merge.
    # For the output, we carry the input's level-2 structure.
    # This happens when the einsum equation has fewer labels than the input's
    # hier rank (rank-mismatch from view), but with 3-level nesting on view,
    # the input's hier is already reshaped to match, so bc has the right structure.

    # Overlay broadcast shape onto output hier strides
    oh_shape = out_hier.shape if is_tuple(out_hier.shape) else (out_hier.shape,)
    oh_stride = out_hier.stride if is_tuple(out_hier.stride) else (out_hier.stride,)

    final_shape = list(oh_shape)
    final_stride = list(oh_stride)
    for idx, (mode_bc, s, st) in enumerate(zip(out_modes, oh_shape, oh_stride)):
        if _is_level2(mode_bc):
            # Level-2: get strides from the contributing input
            label = output_labels[idx]
            if label in inputs[0]:
                src_idx = list(inputs[0]).index(label)
                a_st = sharded_a.hier_layout.stride
                a_st = a_st if is_tuple(a_st) else (a_st,)
                if src_idx < len(a_st):
                    final_shape[idx] = mode_bc
                    final_stride[idx] = a_st[src_idx]
            elif label in inputs[1]:
                src_idx = list(inputs[1]).index(label)
                b_st = sharded_b.hier_layout.stride
                b_st = b_st if is_tuple(b_st) else (b_st,)
                if src_idx < len(b_st):
                    final_shape[idx] = mode_bc
                    final_stride[idx] = b_st[src_idx]

    if len(final_shape) == 1:
        out_hier = Layout(final_shape[0], final_stride[0])
    else:
        out_hier = Layout(tuple(final_shape), tuple(final_stride))

    result = ShardedLayout(out_hier)
    if is_partial:
        result._is_partial = True
    return result
