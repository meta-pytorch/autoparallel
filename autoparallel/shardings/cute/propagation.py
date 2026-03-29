"""
CuTe-based sharding propagation rules.

ShardedLayout uses uniform 3-level hierarchical nesting:
- Level 1: tensor dims (tuple of level-2 modes)
- Level 2: sub-dims (tuple of level-3 sub-dims)
- Level 3: (local, mesh...) — tuple starting with scalar local

Propagation:
- View: merge = concatenate sub-dims, split = peel or split sub-dims
- Transpose/Permute: swap top-level modes
- Pointwise: broadcast hierarchical shapes
- Reduction: drop dim (flat=normal, sharded=Partial)
- Einsum: = broadcast + reduce
"""

from ._pycute import Layout, is_tuple, logical_divide, product, suffix_product
from .placement import ShardedLayout, _ensure_tuple, _local_size, _mode_has_mesh, _to_uniform

from torch.distributed.tensor._ops._view_ops import (
    Flatten,
    InputDim,
    Split,
    view_groups,
)


# =============================================================================
# View / Reshape
# =============================================================================


def _split_subdim(sub, sub_st, split_point):
    """Split a level-3 sub-dim (local, mesh...) at a global-size boundary.
    Returns ((left, left_st), (right, right_st)) or None.
    Sub-dim always has at least (local, mesh) with mesh=1 for replicate."""
    gs = product(sub)
    if gs == split_point:
        return (sub, sub_st), None
    if split_point <= 0 or gs % split_point != 0:
        return None

    local = sub[0]
    mesh = sub[1:]
    local_st = sub_st[0] if is_tuple(sub_st) else sub_st
    mesh_st = sub_st[1:] if is_tuple(sub_st) else ()
    mesh_product = product(mesh)

    if split_point >= mesh_product and split_point % mesh_product == 0:
        left_local = split_point // mesh_product
        right_local = local // left_local
        left = (left_local,) + tuple(mesh)
        right = (right_local, 1)
        left_st = (local_st,) + tuple(mesh_st)
        right_st = (local_st * left_local * mesh_product, 0)
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
    """Transform hier_layout using uniform 3-level nesting.

    Every level-1 mode is a tuple of sub-dims (level-2).
    Each sub-dim is a tuple (local, mesh...) (level-3).

    Merge = concatenate sub-dim lists.
    Split = greedily assign sub-dims, splitting at boundaries.
    """
    h_shape = hier_layout.shape
    h_stride = hier_layout.stride
    old_g = _ensure_tuple(old_global)
    new_g = _ensure_tuple(new_global)

    ops = view_groups(old_g, new_g)

    # With uniform 3-level, each mode IS a tuple of sub-dims
    src_subdims = []
    for s, st in zip(h_shape, h_stride):
        src_subdims.append(list(zip(s, st)))

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
            # Merge: concatenate sub-dim lists
            input_dims = _collect_input_dims(op)
            merged = []
            for d in input_dims:
                merged.extend(src_subdims[d])
            out_shape[out_i] = tuple(s for s, _ in merged)
            out_stride[out_i] = tuple(st for _, st in merged)

        elif isinstance(op, Split):
            group = split_groups[id(op.input_dim)]
            input_dims = _collect_input_dims(group["input_op"])
            group_shape = group["group_shape"]
            outputs = sorted(group["outputs"], key=lambda x: x[1])

            all_subs = []
            for d in input_dims:
                all_subs.extend(src_subdims[d])

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
                out_shape[oi] = tuple(s for s, _ in piece)
                out_stride[oi] = tuple(st for _, st in piece)
                processed.add(oi)

            for oi, _ in outputs:
                processed.add(oi)

    if any(s is None for s in out_shape):
        return None
    return Layout(tuple(out_shape), tuple(out_stride))


def propagate_view(sharded, new_shape):
    """View: reshape hier_layout using uniform 3-level nesting."""
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
    """Transpose: swap top-level modes."""
    h_shape = sharded.hier_layout.shape
    h_stride = sharded.hier_layout.stride
    ndim = len(h_shape)
    dims = list(range(ndim))
    dims[dim0], dims[dim1] = dims[dim1], dims[dim0]
    new_hier = Layout(tuple(h_shape[d] for d in dims), tuple(h_stride[d] for d in dims))
    return ShardedLayout(new_hier)


def propagate_permute(sharded, dims):
    """Permute: reorder top-level modes."""
    h_shape = sharded.hier_layout.shape
    h_stride = sharded.hier_layout.stride
    new_hier = Layout(tuple(h_shape[d] for d in dims), tuple(h_stride[d] for d in dims))
    return ShardedLayout(new_hier)


# =============================================================================
# Slice
# =============================================================================


def propagate_slice(sharded, dim, index):
    """Slice: partial evaluation on hier_layout."""
    h_shape = sharded.hier_layout.shape

    if _mode_has_mesh(h_shape[dim]):
        return None  # slicing a sharded dim

    ndim = len(h_shape)
    coord = tuple(index if k == dim else None for k in range(ndim))
    new_hier = sharded.hier_layout(*coord)
    return ShardedLayout(new_hier)


# =============================================================================
# Gather
# =============================================================================


def propagate_gather(sharded, dim, index_layout):
    """Gather with CuTe-expressible index pattern along dim."""
    h_shape = sharded.hier_layout.shape

    if _mode_has_mesh(h_shape[dim]):
        return None

    h_stride = sharded.hier_layout.stride
    dim_stride = h_stride[dim]

    # Build new dim from index_layout, wrapped in uniform 3-level
    idx_shape = index_layout.shape
    idx_stride = index_layout.stride
    if is_tuple(idx_shape):
        # Multi-mode index — wrap each as a sub-dim
        new_dim_shape = tuple((s,) for s in idx_shape)
        new_dim_stride = tuple((st * dim_stride,) if not is_tuple(st) else
                               tuple(x * dim_stride for x in st) for st in idx_stride)
    else:
        new_dim_shape = ((idx_shape,),)
        new_dim_stride = ((idx_stride * dim_stride,),) if not is_tuple(dim_stride) else \
                         ((idx_stride * dim_stride[0],),)

    new_shape = h_shape[:dim] + (new_dim_shape,) + h_shape[dim + 1:]
    new_stride = h_stride[:dim] + (new_dim_stride,) + h_stride[dim + 1:]
    new_hier = Layout(new_shape, new_stride)
    return ShardedLayout(new_hier)


# =============================================================================
# Pointwise — broadcast hierarchical shapes
# =============================================================================


def _hier_broadcast_shapes(a_shape, b_shape):
    """Broadcast two uniform 3-level shapes.

    Each mode is a tuple of sub-dims. Compatible if:
    - Same structure (equal)
    - One is size-1 broadcast
    - Incompatible: different shardings or different non-broadcast sizes
    """
    result = []
    for sa, sb in zip(a_shape, b_shape):
        if sa == sb:
            result.append(sa)
        elif product(sa) == 1:
            result.append(sb)
        elif product(sb) == 1:
            result.append(sa)
        elif _mode_has_mesh(sa) or _mode_has_mesh(sb):
            return None  # incompatible sharding
        elif product(sa) != product(sb):
            return None  # different sizes
        else:
            result.append(sa)  # same global size, both flat
    return tuple(result)


def propagate_pointwise(all_shardeds, output_shape):
    """Pointwise: broadcast hierarchical shapes."""
    if not all_shardeds:
        return None

    ref = all_shardeds[0]

    for sharded in all_shardeds[1:]:
        if sharded.is_replicate():
            continue
        if ref.is_replicate():
            ref = sharded
            continue

        bc = _hier_broadcast_shapes(ref.hier_layout.shape, sharded.hier_layout.shape)
        if bc is None:
            return None

    return ShardedLayout(ref.hier_layout)


# =============================================================================
# Reduction
# =============================================================================


def propagate_reduction(sharded, reduce_dim, keepdim, output_shape):
    """Reduction: drop dim from hier_layout.

    If reduced dim has mesh → Partial.
    """
    h_shape = sharded.hier_layout.shape
    is_sharded_reduce = _mode_has_mesh(h_shape[reduce_dim])

    ndim = len(h_shape)
    if keepdim:
        coord = tuple(0 if k == reduce_dim else None for k in range(ndim))
        new_hier = sharded.hier_layout(*coord)
        nh_shape = new_hier.shape if is_tuple(new_hier.shape) else (new_hier.shape,)
        nh_stride = new_hier.stride if is_tuple(new_hier.stride) else (new_hier.stride,)
        ins_shape = nh_shape[:reduce_dim] + (((1, 1),),) + nh_shape[reduce_dim:]
        ins_stride = nh_stride[:reduce_dim] + (((0, 0),),) + nh_stride[reduce_dim:]
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


# Uniform 3-level size-1 mode: ((local=1, mesh=1),)
_SIZE_1_MODE = ((1, 1),)


def propagate_einsum(equation, sharded_a, sharded_b, output_shape):
    """Einsum = unsqueeze + broadcast + reduce."""
    inputs, output_labels = _parse_einsum(equation)
    categories = _classify_einsum_dims(inputs, output_labels)

    a_shape = sharded_a.hier_layout.shape
    b_shape = sharded_b.hier_layout.shape

    # Build joint label space
    all_labels = list(dict.fromkeys(list(inputs[0]) + list(inputs[1])))

    # Expand A and B to joint space
    a_expanded = []
    for label in all_labels:
        if label in inputs[0]:
            idx = list(inputs[0]).index(label)
            a_expanded.append(a_shape[idx])
        else:
            a_expanded.append(_SIZE_1_MODE)

    b_expanded = []
    for label in all_labels:
        if label in inputs[1]:
            idx = list(inputs[1]).index(label)
            b_expanded.append(b_shape[idx])
        else:
            b_expanded.append(_SIZE_1_MODE)

    # Broadcast
    bc = _hier_broadcast_shapes(tuple(a_expanded), tuple(b_expanded))
    if bc is None:
        return None

    # Check einsum compatibility
    for i, label in enumerate(all_labels):
        cat = categories.get(label)
        a_sharded = _mode_has_mesh(a_expanded[i])
        b_sharded = _mode_has_mesh(b_expanded[i])

        if cat == "m" and b_sharded:
            return None
        if cat == "n" and a_sharded:
            return None
        if cat == "batch" and a_sharded != b_sharded:
            return None
        if cat == "contract" and a_sharded != b_sharded:
            return None

    # Reduce: drop contracted dims
    is_partial = False
    out_modes_bc = []
    for i, label in enumerate(all_labels):
        if categories.get(label) == "contract":
            if _mode_has_mesh(bc[i]):
                is_partial = True
            continue
        if label in output_labels:
            out_modes_bc.append(bc[i])

    # Build output hier with output-space strides
    o_shape = _ensure_tuple(output_shape)
    out_local = tuple(_local_size(mode) for mode in out_modes_bc)
    out_hier = _to_uniform(logical_divide(Layout(o_shape), out_local))

    # Overlay broadcast level-2 structure (for merged dims with mesh)
    oh_shape = out_hier.shape
    final_shape = list(oh_shape)
    final_stride = list(out_hier.stride)

    for idx, mode_bc in enumerate(out_modes_bc):
        if len(mode_bc) > 1 or _mode_has_mesh(mode_bc):
            label = output_labels[idx]
            if label in inputs[0]:
                src_idx = list(inputs[0]).index(label)
                final_shape[idx] = mode_bc
                final_stride[idx] = sharded_a.hier_layout.stride[src_idx]
            elif label in inputs[1]:
                src_idx = list(inputs[1]).index(label)
                final_shape[idx] = mode_bc
                final_stride[idx] = sharded_b.hier_layout.stride[src_idx]

    out_hier = Layout(tuple(final_shape), tuple(final_stride))
    result = ShardedLayout(out_hier)
    if is_partial:
        result._is_partial = True
    return result
