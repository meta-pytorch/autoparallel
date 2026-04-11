"""
Redistribute planning via per-mesh-dim GPU stride classification.

The GPU stride per mesh dim is the fundamental binary classifier:
  gs = 0: this mesh dim doesn't partition this tensor dim (Replicate / Partial)
  gs > 0: this mesh dim partitions this tensor dim (Shard)

Per-mesh-dim GPU strides are extracted from hier_layout (3-level),
NOT from _ld() (which fuses all mesh dims into one stride).

For each mesh dim independently, compare src_gs vs tgt_gs:
  src>0, tgt=0 -> all_gather
  src=0, tgt>0 -> local reinterpret
  src=0, tgt=0 -> no_op (or all_reduce if Partial)
  src>0, tgt>0, same -> no_op
  src>0, tgt>0, diff -> ppermute (1-to-1 device mapping) or all_to_all

Refinements:
  ppermute detection: check if each src device maps to exactly one tgt device
  max_common_vector(src_local, tgt_local) -> vectorization factor
  composition(right_inverse(tgt), src) -> full element-level send/recv schedule
"""

from ._pycute import Layout, is_tuple, logical_divide, product, composition, right_inverse, coalesce, max_common_vector

from .placement import _has_mesh, _local_size


# =============================================================================
# Redistribute planning — per-mesh-dim GPU stride classification
# =============================================================================


def _get_per_mesh_dim_gpu_stride(sharded, mesh_dim):
    """Get the GPU stride for a specific mesh dim from hier_layout.

    Walks hier_layout's 3-level structure to find the stride element
    corresponding to the given mesh dim. Returns (tensor_dim, gpu_stride)
    or None if this mesh dim doesn't shard any tensor dim.

    The sub-dim layout has order (local, mesh0, mesh1, ...) matching
    the mesh_dim_map order (construction order from shard_multi).
    """
    for tensor_dim, mesh_dims in sharded.mesh_dim_map.items():
        if mesh_dim in mesh_dims:
            mesh_idx_in_dim = list(mesh_dims).index(mesh_dim)

            dim_shape = sharded.hier_layout.shape[tensor_dim]
            dim_stride = sharded.hier_layout.stride[tensor_dim]

            # Collect all mesh elements across sub-dims
            all_mesh = []
            for sub_s, sub_st in zip(dim_shape, dim_stride):
                if _has_mesh(sub_s):
                    mesh_parts = sub_s[1:]
                    mesh_strides = sub_st[1:] if is_tuple(sub_st) else ()

                    def _flatten_mesh(s, st):
                        if is_tuple(s):
                            result = []
                            for si, sti in zip(s, st):
                                result.extend(_flatten_mesh(si, sti))
                            return result
                        return [(s, st)]

                    all_mesh.extend(_flatten_mesh(mesh_parts, mesh_strides))

            if mesh_idx_in_dim < len(all_mesh):
                return tensor_dim, all_mesh[mesh_idx_in_dim][1]

    return None


def _get_local_layout(ld):
    """Extract coalesced local sub-layout from a logical_divide result."""
    shapes = []
    strides = []
    for s, st in zip(ld.shape, ld.stride):
        if is_tuple(s):
            shapes.append(s[0])
            strides.append(st[0])
        else:
            shapes.append(s)
            strides.append(st)
    return coalesce(Layout(tuple(shapes), tuple(strides)))


def _reverse_mesh_map(mesh_dim_map):
    reverse = {}
    for tensor_dim, mesh_dims in mesh_dim_map.items():
        for md in mesh_dims:
            reverse.setdefault(md, set()).add(tensor_dim)
    return reverse


def _detect_permutation(source, target, mesh_dim, src_info, tgt_info):
    """Detect if the redistribution on a mesh dim is a permutation (ppermute).

    For each source device, checks if ALL its elements map to exactly one
    target device. If yes, returns the permutation as [(src, dst), ...].
    If no (data splits across multiple target devices), returns None.

    Uses the hier_layout to compute element-to-device mappings for a few
    representative elements per device, which is cheap for typical mesh sizes.
    """
    src_td, src_gs = src_info
    tgt_td, tgt_gs = tgt_info

    # Get mesh sizes for this mesh dim from the sub-dim structure
    src_mesh_size = None
    for tensor_dim, mesh_dims in source.mesh_dim_map.items():
        if mesh_dim in mesh_dims:
            mode = source.hier_layout.shape[tensor_dim]
            g = product(mode)
            l = _local_size(mode)
            src_mesh_size = g // l
            break

    tgt_mesh_size = None
    for tensor_dim, mesh_dims in target.mesh_dim_map.items():
        if mesh_dim in mesh_dims:
            mode = target.hier_layout.shape[tensor_dim]
            g = product(mode)
            l = _local_size(mode)
            tgt_mesh_size = g // l
            break

    if src_mesh_size is None or tgt_mesh_size is None:
        return None
    if src_mesh_size != tgt_mesh_size:
        return None  # different mesh sizes, can't be a permutation

    mesh_size = src_mesh_size

    # Compute element-to-device mapping for source and target.
    # For each element index e:
    #   src_device = e // src_gs % mesh_size  (for integer strides)
    #   tgt_device = e // tgt_gs % mesh_size
    # But for XorStride/ModStride, this formula doesn't apply directly.
    #
    # Use the _ld() layouts to compute device assignments.
    # _ld() gives (local, mesh) per dim. For an element at global position,
    # the mesh coordinate = device index.
    #
    # Simpler approach for typical cases: use the GPU strides directly.
    # For integer strides: element e is on device (e // local_size) % mesh_size
    # where local_size = global_size // mesh_size.

    src_local = source.local_sizes
    tgt_local = target.local_sizes
    global_shape = source.global_shape

    # For 1D tensors or the specific sharded dim, compute the device mapping.
    # We check a few elements from each source device to see if they all
    # map to the same target device.

    # Find the sharded tensor dims for this mesh dim
    src_sharded_dim = src_td
    tgt_sharded_dim = tgt_td

    src_dim_local = src_local[src_sharded_dim]
    tgt_dim_local = tgt_local[tgt_sharded_dim]
    src_dim_global = global_shape[src_sharded_dim]
    tgt_dim_global = global_shape[tgt_sharded_dim]

    # For simple integer strides: device = element_in_dim // local_size
    # Check if this gives a clean permutation
    if not isinstance(src_gs, int) or not isinstance(tgt_gs, int):
        # XorStride or ModStride — need element-level check
        # For now, fall back to checking all device pairs
        return _detect_permutation_element_level(
            source, target, mesh_dim, mesh_size
        )

    # Integer strides: compute device mapping analytically
    perm = {}
    for src_dev in range(mesh_size):
        # Elements on src_dev: indices where (idx // src_dim_local) % mesh_size == src_dev
        # Pick a representative element: src_dev * src_dim_local
        repr_elem = src_dev * src_dim_local
        # Which tgt device is this element on?
        tgt_dev = (repr_elem // tgt_dim_local) % mesh_size
        if src_dev in perm:
            return None  # shouldn't happen
        perm[src_dev] = tgt_dev

    # Verify it's a bijection
    if len(set(perm.values())) != mesh_size:
        return None  # not a permutation (some tgt device receives from multiple sources)

    return [(src, dst) for src, dst in sorted(perm.items())]


def _detect_permutation_element_level(source, target, mesh_dim, mesh_size):
    """Detect permutation by checking element-level device assignments.

    Used for non-integer strides (XorStride, ModStride) where analytical
    computation isn't straightforward. Checks representative elements
    from each source device.
    """
    src_ld = source._ld()
    tgt_ld = target._ld()

    # Element mapping: src (local, mesh) -> tgt coordinate
    mapping = composition(right_inverse(tgt_ld), src_ld)

    # For each source device, evaluate the mapping at a few local coordinates
    # and extract the target mesh coordinate.
    ndim = len(source.global_shape)
    src_local_sizes = source.local_sizes
    total_local = product(src_local_sizes)

    perm = {}
    for src_dev in range(mesh_size):
        tgt_devs = set()
        # Check a few representative local elements
        n_checks = min(total_local, 4)  # check up to 4 local elements
        for local_idx in range(n_checks):
            # Build the source coordinate: (local_coords..., mesh_coord)
            # For 1D: (local_idx, src_dev)
            # For nD: need to decompose local_idx into per-dim local coords
            # Simplified: use _ld() which is 2-level (local, mesh) per dim
            # Evaluate src_ld at (local_idx_per_dim, src_dev_per_dim)

            # For the specific mesh_dim, set mesh coord = src_dev
            # For other dims, use local_idx as the flat local index

            # This is getting complex for multi-dim. For now, use the
            # element index directly:
            # src device src_dev holds elements at positions where
            # the mesh coordinate in src_ld equals src_dev.

            # Pick element: for the sharded dim, position = src_dev * local_size + local_offset
            src_sharded_dim = None
            for td, mds in source.mesh_dim_map.items():
                if mesh_dim in mds:
                    src_sharded_dim = td
                    break

            local_size_on_dim = src_local_sizes[src_sharded_dim]
            elem_on_dim = src_dev * local_size_on_dim + (local_idx % local_size_on_dim)

            # For target: find which device this element belongs to
            tgt_sharded_dim = None
            for td, mds in target.mesh_dim_map.items():
                if mesh_dim in mds:
                    tgt_sharded_dim = td
                    break

            tgt_local_size = target.local_sizes[tgt_sharded_dim]
            tgt_dev = (elem_on_dim // tgt_local_size) % mesh_size
            tgt_devs.add(tgt_dev)

        if len(tgt_devs) != 1:
            return None  # this src device maps to multiple tgt devices
        perm[src_dev] = tgt_devs.pop()

    # Verify bijection
    if len(set(perm.values())) != mesh_size:
        return None

    return [(src, dst) for src, dst in sorted(perm.items())]


def _get_mesh_dim_size(sharded, mesh_dim):
    """Get the mesh size for a specific mesh dim from hier_layout."""
    for tensor_dim, mesh_dims in sharded.mesh_dim_map.items():
        if mesh_dim in mesh_dims:
            idx = list(mesh_dims).index(mesh_dim)
            dim_shape = sharded.hier_layout.shape[tensor_dim]
            # Collect mesh parts from sub-dims
            for sub_s in dim_shape:
                if _has_mesh(sub_s):
                    mesh_parts = sub_s[1:]
                    if is_tuple(mesh_parts):
                        if idx < len(mesh_parts):
                            return mesh_parts[idx]
                    elif idx == 0:
                        return mesh_parts
    return None


def _build_rank_to_chunk(sharded, tensor_dim, mesh_dims, mesh_shape):
    """Build a CuTe Layout mapping rank -> chunk index for coupled mesh dims.

    For a tensor dim with sub-dim (local, mesh0, mesh1, ...):
    - chunk_stride_i = sub_stride[mesh_pos_i] // (local_size * local_stride)
    - Shape is reversed (M_last, ..., M_first) so CuTe's col-major matches
      mesh's row-major rank indexing.

    For non-integer local strides (XorStride), uses mesh strides directly
    divided by the smallest mesh stride as the chunk unit.

    Args:
        sharded: ShardedLayout
        tensor_dim: which tensor dim has the coupled sharding
        mesh_dims: tuple of mesh dim indices that shard this tensor dim
        mesh_shape: tuple of mesh dim sizes in mesh dim order

    Returns:
        (Layout mapping flat rank -> chunk index, total_chunks) or (None, None)
    """
    dim_shape = sharded.hier_layout.shape[tensor_dim]
    dim_stride = sharded.hier_layout.stride[tensor_dim]

    # Find the sub-dim with mesh sharding
    for sub_s, sub_st in zip(dim_shape, dim_stride):
        if _has_mesh(sub_s):
            local_size = sub_s[0]
            local_stride = sub_st[0]

            mesh_parts_s = sub_s[1:]
            mesh_parts_st = sub_st[1:]

            # Compute divisor: for integer local strides, use local_size * local_stride.
            # For non-integer (XorStride), use the minimum mesh stride as unit.
            if isinstance(local_stride, int) and local_stride > 0:
                divisor = local_size * local_stride
            else:
                # XorStride local: mesh strides are the chunk strides directly
                # (each "chunk" is the set of elements for one rank at one pair index)
                divisor = 1

            chunk_strides = {}
            total_chunks = 1
            for i, md in enumerate(mesh_dims):
                st = mesh_parts_st[i]
                if isinstance(st, int):
                    chunk_strides[md] = st // divisor if divisor > 1 else st
                else:
                    chunk_strides[md] = st  # keep ModStride/XorStride as-is
                total_chunks *= mesh_parts_s[i]

            all_mds = sorted(chunk_strides.keys())
            rev_shape = tuple(mesh_shape[md] for md in reversed(all_mds))
            rev_strides = tuple(chunk_strides[md] for md in reversed(all_mds))

            if len(rev_shape) == 1:
                return Layout(rev_shape[0], rev_strides[0]), total_chunks
            return Layout(rev_shape, rev_strides), total_chunks

    return None, None


def _compute_coupled_ppermute(source, target, tensor_dim, src_mesh_dims, tgt_mesh_dims, mesh_shape):
    """Compute rank permutation for coupled mesh dims on the same tensor dim.

    Uses CuTe rank-to-chunk composition:
    perm = composition(right_inverse(tgt_r2c), src_r2c)

    When offsets differ, adjusts chunk indices by the offset difference
    (in chunk space, modulo total_chunks).

    Returns list of (src_rank, tgt_rank) pairs, or None if identity.
    """
    src_r2c, src_total = _build_rank_to_chunk(source, tensor_dim, src_mesh_dims, mesh_shape)
    tgt_r2c, tgt_total = _build_rank_to_chunk(target, tensor_dim, tgt_mesh_dims, mesh_shape)

    if src_r2c is None or tgt_r2c is None:
        return None

    total_ranks = product(mesh_shape)

    # Compute offset difference in chunk space
    src_offset = getattr(source, 'offset', {}).get(tensor_dim, 0)
    tgt_offset = getattr(target, 'offset', {}).get(tensor_dim, 0)

    # Get local_size and local_stride for chunk conversion
    dim_shape = source.hier_layout.shape[tensor_dim]
    dim_stride = source.hier_layout.stride[tensor_dim]
    divisor = 1
    total_chunks = src_total
    for sub_s, sub_st in zip(dim_shape, dim_stride):
        if _has_mesh(sub_s):
            local_stride = sub_st[0]
            if isinstance(local_stride, int) and local_stride > 0:
                divisor = sub_s[0] * local_stride
            break

    src_off_chunks = (src_offset // divisor) % total_chunks if divisor > 0 else 0
    tgt_off_chunks = (tgt_offset // divisor) % total_chunks if divisor > 0 else 0

    if src_off_chunks == tgt_off_chunks:
        # No offset difference — use CuTe composition
        perm_layout = composition(right_inverse(tgt_r2c), src_r2c)
        perm = []
        is_identity = True
        for r in range(total_ranks):
            tgt_r = perm_layout(r)
            perm.append((r, tgt_r))
            if r != tgt_r:
                is_identity = False
        return None if is_identity else perm

    # Offset differs — enumerate directly with modular arithmetic
    # For each rank, compute which chunk it holds in src and tgt
    src_chunk_to_rank = {}
    for r in range(total_ranks):
        chunk = (src_off_chunks + src_r2c(r)) % total_chunks
        src_chunk_to_rank[chunk] = r

    perm = []
    is_identity = True
    for tgt_rank in range(total_ranks):
        tgt_chunk = (tgt_off_chunks + tgt_r2c(tgt_rank)) % total_chunks
        src_rank = src_chunk_to_rank.get(tgt_chunk)
        if src_rank is None:
            return None  # not a permutation
        perm.append((src_rank, tgt_rank))
        if src_rank != tgt_rank:
            is_identity = False

    # Verify bijection
    if len(set(p[0] for p in perm)) != total_ranks:
        return None

    return None if is_identity else perm


def plan_redistribute(source, target):
    """Plan collectives using per-mesh-dim GPU stride classification.

    For each mesh dim independently:
    1. Extract the GPU stride from hier_layout for src and tgt
    2. Compare using the 2x2 matrix (gs=0 vs gs>0)
    3. Apply Partial refinement

    Returns list of (collective_type, mesh_dim, info) tuples.
    """
    if source.global_shape != target.global_shape:
        raise ValueError(
            f"Cannot redistribute: shapes differ {source.global_shape} vs {target.global_shape}"
        )

    src_reverse = _reverse_mesh_map(source.mesh_dim_map)
    tgt_reverse = _reverse_mesh_map(target.mesh_dim_map)
    all_mesh_dims = set(src_reverse) | set(tgt_reverse) | set(source.partial)

    collectives = []

    # Handle partials first
    for md, reduce_op in source.partial.items():
        tgt_dims = tgt_reverse.get(md, set())
        if tgt_dims:
            collectives.append(("reduce_scatter", md, {"reduce_op": reduce_op}))
        else:
            collectives.append(("all_reduce", md, {"reduce_op": reduce_op}))
        all_mesh_dims.discard(md)

    # Check if offsets differ on any tensor dim with mesh sharding
    src_offset = getattr(source, 'offset', {})
    tgt_offset = getattr(target, 'offset', {})
    offset_differs = {}
    for td in set(list(src_offset.keys()) + list(tgt_offset.keys())):
        so = src_offset.get(td, 0)
        to = tgt_offset.get(td, 0)
        if so != to and source.mesh_dim_map.get(td, ()):
            offset_differs[td] = (so, to)

    # Per mesh dim: compare GPU strides from hier_layout
    for md in sorted(all_mesh_dims):
        src_info = _get_per_mesh_dim_gpu_stride(source, md)
        tgt_info = _get_per_mesh_dim_gpu_stride(target, md)

        src_gs = src_info[1] if src_info else 0
        tgt_gs = tgt_info[1] if tgt_info else 0

        src_dims = src_reverse.get(md, set())
        tgt_dims = tgt_reverse.get(md, set())

        # Check if any tensor dim for this mesh dim has an offset difference
        has_offset_diff = any(
            td in offset_differs
            for td in src_dims | tgt_dims
        )

        if src_gs == 0 and tgt_gs == 0:
            continue  # no_op
        elif src_gs != 0 and tgt_gs == 0:
            collectives.append(("all_gather", md, {"tensor_dims": src_dims}))
        elif src_gs == 0 and tgt_gs != 0:
            continue  # local reinterpret
        elif src_gs == tgt_gs and not has_offset_diff:
            continue  # same stride and same offset, no communication
        else:
            # Both sharded, different strides.
            # Try to detect if it's a permutation (ppermute) first.
            perm = None
            if src_info and tgt_info:
                perm = _detect_permutation(source, target, md, src_info, tgt_info)

            if perm is not None:
                collectives.append(("ppermute", md, {
                    "perm": perm,
                    "source_dims": src_dims,
                    "target_dims": tgt_dims,
                }))
            else:
                collectives.append(("all_to_all", md, {
                    "source_dims": src_dims,
                    "target_dims": tgt_dims,
                }))

    # Post-pass: handle cases where per-mesh-dim analysis is insufficient.
    # This covers:
    # 1. Coupled mesh dims (S(0)S(0) with different strides/orderings)
    # 2. Offset differences (same strides but different element-to-rank mapping)
    # In both cases, compute the correct rank permutation via rank-to-chunk
    # with offset-aware modular arithmetic.
    for td, src_mds in source.mesh_dim_map.items():
        if not src_mds:
            continue
        tgt_mds = target.mesh_dim_map.get(td, ())
        if set(src_mds) != set(tgt_mds):
            continue

        is_coupled = len(src_mds) > 1
        has_offset = td in offset_differs

        if not is_coupled and not has_offset:
            continue

        # Check if any of these mesh dims had a collective emitted
        affected = [i for i, (ct, md, info) in enumerate(collectives)
                    if md in src_mds]
        if not affected and not has_offset:
            continue

        # Derive mesh shape from the sub-dim structure
        mesh_shape = {}
        for md in src_mds:
            ms = _get_mesh_dim_size(source, md)
            if ms is not None:
                mesh_shape[md] = ms

        all_mds = sorted(mesh_shape.keys())
        mesh_shape_tuple = tuple(mesh_shape[md] for md in all_mds)

        perm = _compute_coupled_ppermute(
            source, target, td, src_mds, tgt_mds, mesh_shape_tuple
        )

        # Remove the per-mesh-dim entries
        for i in reversed(affected):
            collectives.pop(i)

        # Add global ppermute if non-identity
        if perm is not None:
            collectives.append(("ppermute", None, {"perm": perm}))

    return collectives


def plan_redistribute_detailed(source, target):
    """Extended redistribution plan with vectorization and element mapping."""
    plan = plan_redistribute(source, target)

    src_ld = source._ld()
    tgt_ld = target._ld()

    src_local = _get_local_layout(src_ld)
    tgt_local = _get_local_layout(tgt_ld)
    vec_factor = max_common_vector(src_local, tgt_local)

    element_mapping = composition(right_inverse(tgt_ld), src_ld)

    return {
        "collectives": plan,
        "vectorization_factor": vec_factor,
        "element_mapping": element_mapping,
    }
