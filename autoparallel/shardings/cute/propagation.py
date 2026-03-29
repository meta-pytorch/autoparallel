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

from ._pycute import Layout, is_tuple, product
from .placement import ShardedLayout, _ensure_tuple, _mode_has_mesh

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
    """View: reshape hier_layout using uniform 3-level nesting.
    mesh_dim_map is transformed: merge combines entries, split distributes them.
    """
    if product(sharded.global_shape) != product(new_shape):
        return None
    new_hier = _view_hier(sharded.hier_layout, sharded.global_shape, new_shape)
    if new_hier is None:
        return None

    # Transform mesh_dim_map through the view
    old_g = _ensure_tuple(sharded.global_shape)
    new_g = _ensure_tuple(new_shape)
    ops = view_groups(old_g, new_g)

    new_map = {}
    for out_i, op in enumerate(ops):
        if isinstance(op, InputDim):
            src = op.input_dim
            if src in sharded.mesh_dim_map:
                new_map[out_i] = sharded.mesh_dim_map[src]
        elif isinstance(op, Flatten):
            # Merge: collect mesh dims from all source dims
            src_dims = _collect_input_dims(op)
            mesh_dims = []
            for d in src_dims:
                if d in sharded.mesh_dim_map:
                    entry = sharded.mesh_dim_map[d]
                    if isinstance(entry, tuple):
                        mesh_dims.extend(entry)
                    else:
                        mesh_dims.append(entry)
            if len(mesh_dims) == 1:
                new_map[out_i] = mesh_dims[0]
            elif len(mesh_dims) > 1:
                new_map[out_i] = tuple(mesh_dims)
        elif isinstance(op, Split):
            # Split: distribute mesh dims to output dims
            # The 3-level sub-dim structure determines which output dim gets which mesh
            # For now: mesh dims go to the first output dim that has mesh in the hier shape
            src_dims = _collect_input_dims(op)
            if op.split_id == 0:
                # Collect all mesh dims from source
                mesh_dims = []
                for d in src_dims:
                    if d in sharded.mesh_dim_map:
                        entry = sharded.mesh_dim_map[d]
                        if isinstance(entry, tuple):
                            mesh_dims.extend(entry)
                        else:
                            mesh_dims.append(entry)
                if mesh_dims:
                    # Distribute: check which output dims have mesh in the hier shape
                    split_key = id(op.input_dim)
                    split_outputs = [(oi2, o.split_id) for oi2, o in enumerate(ops)
                                     if isinstance(o, Split) and id(o.input_dim) == split_key]
                    split_outputs.sort(key=lambda x: x[1])
                    mesh_idx = 0
                    for oi2, _ in split_outputs:
                        if mesh_idx < len(mesh_dims) and _mode_has_mesh(new_hier.shape[oi2]):
                            new_map[oi2] = mesh_dims[mesh_idx]
                            mesh_idx += 1

    return ShardedLayout(new_hier, new_map)


# =============================================================================
# Transpose / Permute
# =============================================================================


def propagate_transpose(sharded, dim0, dim1):
    """Transpose: swap top-level modes and mesh_dim_map entries."""
    h_shape = sharded.hier_layout.shape
    h_stride = sharded.hier_layout.stride
    ndim = len(h_shape)
    dims = list(range(ndim))
    dims[dim0], dims[dim1] = dims[dim1], dims[dim0]
    new_hier = Layout(tuple(h_shape[d] for d in dims), tuple(h_stride[d] for d in dims))

    # Swap mesh_dim_map keys
    new_map = {}
    for src, mesh_dim in sharded.mesh_dim_map.items():
        if src == dim0:
            new_map[dim1] = mesh_dim
        elif src == dim1:
            new_map[dim0] = mesh_dim
        else:
            new_map[src] = mesh_dim

    return ShardedLayout(new_hier, new_map)


def propagate_permute(sharded, dims):
    """Permute: reorder top-level modes and mesh_dim_map entries."""
    h_shape = sharded.hier_layout.shape
    h_stride = sharded.hier_layout.stride
    new_hier = Layout(tuple(h_shape[d] for d in dims), tuple(h_stride[d] for d in dims))

    # Remap mesh_dim_map keys: old dim i → new position
    inv_perm = {d: i for i, d in enumerate(dims)}
    new_map = {inv_perm[src]: mesh_dim for src, mesh_dim in sharded.mesh_dim_map.items()}

    return ShardedLayout(new_hier, new_map)


# =============================================================================
# Slice
# =============================================================================


def propagate_slice(sharded, dim, index):
    """Slice: partial evaluation on hier_layout."""
    h_shape = sharded.hier_layout.shape

    if _mode_has_mesh(h_shape[dim]):
        return None

    ndim = len(h_shape)
    coord = tuple(index if k == dim else None for k in range(ndim))
    new_hier = sharded.hier_layout(*coord)

    # Remove sliced dim from map, shift higher dims down
    new_map = {}
    for src, mesh_dim in sharded.mesh_dim_map.items():
        if src == dim:
            continue
        new_key = src if src < dim else src - 1
        new_map[new_key] = mesh_dim

    return ShardedLayout(new_hier, new_map)


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
    return ShardedLayout(new_hier, dict(sharded.mesh_dim_map))


# =============================================================================
# Pointwise — broadcast hierarchical shapes
# =============================================================================


def _hier_broadcast_shapes(a_shape, b_shape):
    """Broadcast two uniform 3-level shapes.

    Each mode is a tuple of sub-dims. Compatible if:
    - Same structure (equal)
    - One is size-1 (broadcast dim)
    - One has no mesh (replicate on this dim): sharded one wins
    - Both have mesh: must match exactly
    """
    result = []
    for sa, sb in zip(a_shape, b_shape):
        if sa == sb:
            result.append(sa)
        elif product(sa) == 1:
            result.append(sb)
        elif product(sb) == 1:
            result.append(sa)
        elif _mode_has_mesh(sa) and _mode_has_mesh(sb):
            return None  # both sharded differently
        elif _mode_has_mesh(sa) and not _mode_has_mesh(sb):
            result.append(sa)  # sharded wins over replicate
        elif not _mode_has_mesh(sa) and _mode_has_mesh(sb):
            result.append(sb)  # sharded wins over replicate
        elif product(sa) != product(sb):
            return None  # different sizes, both flat
        else:
            result.append(sa)  # same global size, both flat
    return tuple(result)


def propagate_pointwise(all_shardeds, output_shape):
    """Pointwise: broadcast hierarchical shapes. Merge mesh_dim_maps."""
    if not all_shardeds:
        return None

    result = all_shardeds[0]
    result_map = dict(result.mesh_dim_map)

    for sharded in all_shardeds[1:]:
        r_shape = result.hier_layout.shape
        s_shape = sharded.hier_layout.shape

        if len(r_shape) != len(s_shape):
            return None

        bc = _hier_broadcast_shapes(r_shape, s_shape)
        if bc is None:
            return None

        # Check mesh dim compatibility:
        # 1. Same tensor dim must use same mesh dim
        for dim in set(result_map) & set(sharded.mesh_dim_map):
            if result_map[dim] != sharded.mesh_dim_map[dim]:
                return None

        # 2. Same mesh dim must not be on different tensor dims
        result_mesh_to_tensor = {v: k for k, v in result_map.items() if not isinstance(v, tuple)}
        for dim, mesh_dim in sharded.mesh_dim_map.items():
            if isinstance(mesh_dim, tuple):
                continue
            if mesh_dim in result_mesh_to_tensor and result_mesh_to_tensor[mesh_dim] != dim:
                return None  # same mesh dim on different tensor dims

        # Build broadcast layout
        bc_stride = []
        for i in range(len(bc)):
            if product(r_shape[i]) > 1 and (product(s_shape[i]) == 1 or _mode_has_mesh(r_shape[i])):
                bc_stride.append(result.hier_layout.stride[i])
            elif product(s_shape[i]) > 1:
                bc_stride.append(sharded.hier_layout.stride[i])
            else:
                bc_stride.append(result.hier_layout.stride[i])

        # Merge mesh_dim_maps
        merged_map = dict(result_map)
        for dim, mesh_dim in sharded.mesh_dim_map.items():
            if dim not in merged_map:
                merged_map[dim] = mesh_dim

        result = ShardedLayout(Layout(bc, tuple(bc_stride)), merged_map)
        result_map = merged_map

    return result


# =============================================================================
# Reduction
# =============================================================================


def propagate_reduction(sharded, reduce_dim, keepdim=False, output_shape=None):
    """Reduction: drop one or more dims from hier_layout.

    reduce_dim: int or list/tuple of ints.
    If any reduced dim has mesh → Partial.
    """
    h_shape = sharded.hier_layout.shape

    if isinstance(reduce_dim, int):
        reduce_dims = {reduce_dim}
    else:
        reduce_dims = set(reduce_dim)

    is_sharded_reduce = any(_mode_has_mesh(h_shape[d]) for d in reduce_dims)

    ndim = len(h_shape)
    coord = tuple(0 if k in reduce_dims else None for k in range(ndim))
    new_hier = sharded.hier_layout(*coord)

    if keepdim:
        nh_shape = _ensure_tuple(new_hier.shape)
        nh_stride = _ensure_tuple(new_hier.stride)
        result_shape = list(nh_shape)
        result_stride = list(nh_stride)
        for d in sorted(reduce_dims):
            result_shape.insert(d, ((1, 1),))
            result_stride.insert(d, ((0, 0),))
        new_hier = Layout(tuple(result_shape), tuple(result_stride))

    # Transform mesh_dim_map: remove reduced dims, shift others
    new_map = {}
    if keepdim:
        # Dims don't shift with keepdim
        for src, mesh_dim in sharded.mesh_dim_map.items():
            if src not in reduce_dims:
                new_map[src] = mesh_dim
    else:
        # Shift higher dims down
        for src, mesh_dim in sharded.mesh_dim_map.items():
            if src in reduce_dims:
                continue
            shift = sum(1 for rd in reduce_dims if rd < src)
            new_map[src - shift] = mesh_dim

    result = ShardedLayout(new_hier, new_map)
    if is_sharded_reduce:
        result._is_partial = True
    return result


# =============================================================================
# Einsum = expand + pointwise + reduce
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


def _expand_to_joint(sharded, input_labels, all_labels):
    """Expand a ShardedLayout to the joint label space by inserting size-1 modes."""
    h_shape = sharded.hier_layout.shape
    h_stride = sharded.hier_layout.stride

    new_shape = []
    new_stride = []
    new_map = {}
    for out_i, label in enumerate(all_labels):
        if label in input_labels:
            idx = list(input_labels).index(label)
            new_shape.append(h_shape[idx])
            new_stride.append(h_stride[idx])
            if idx in sharded.mesh_dim_map:
                new_map[out_i] = sharded.mesh_dim_map[idx]
        else:
            new_shape.append(_SIZE_1_MODE)
            new_stride.append(((0, 0),))

    return ShardedLayout(Layout(tuple(new_shape), tuple(new_stride)), new_map)


def propagate_einsum(equation, sharded_a, sharded_b, output_shape):
    """Einsum = expand + pointwise + reduce.

    1. Parse equation, classify dims, check einsum-specific compatibility
    2. Expand A and B to joint label space (unsqueeze)
    3. Pointwise broadcast
    4. Reduce contracted dims
    """
    inputs, output_labels = _parse_einsum(equation)
    categories = _classify_einsum_dims(inputs, output_labels)

    # Build joint label space
    all_labels = list(dict.fromkeys(list(inputs[0]) + list(inputs[1])))

    # Step 1: Check einsum-specific compatibility BEFORE expanding
    a_shape = sharded_a.hier_layout.shape
    b_shape = sharded_b.hier_layout.shape
    for label in all_labels:
        cat = categories.get(label)
        a_sharded = _mode_has_mesh(a_shape[list(inputs[0]).index(label)]) if label in inputs[0] else False
        b_sharded = _mode_has_mesh(b_shape[list(inputs[1]).index(label)]) if label in inputs[1] else False

        if cat == "m" and b_sharded:
            return None
        if cat == "n" and a_sharded:
            return None
        if cat == "batch" and a_sharded != b_sharded:
            return None
        if cat == "contract" and a_sharded != b_sharded:
            return None

    # Step 2: Expand to joint space
    expanded_a = _expand_to_joint(sharded_a, inputs[0], all_labels)
    expanded_b = _expand_to_joint(sharded_b, inputs[1], all_labels)

    # Step 3: Pointwise broadcast
    joint = propagate_pointwise([expanded_a, expanded_b], output_shape)
    if joint is None:
        return None

    # Step 4: Reduce contracted dims
    contract_dims = [i for i, label in enumerate(all_labels) if categories.get(label) == "contract"]
    if not contract_dims:
        return joint

    result = propagate_reduction(joint, contract_dims, keepdim=False, output_shape=output_shape)
    return result
