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


def _compute_ppermute(source, target, tensor_dim, src_mesh_dims, tgt_mesh_dims, mesh_shape):
    """Compute rank permutation for a tensor dim's mesh group.

    Builds rank-to-chunk CuTe layouts for src and tgt, then determines
    the rank permutation via composition or offset-aware enumeration.

    Handles all sharded→sharded cases uniformly:
    - Single or multiple mesh dims
    - Same or different strides/orderings
    - Same or different offsets (ring attention)
    - Integer or non-integer strides (XorStride/ModStride)

    Args:
        source, target: ShardedLayout
        tensor_dim: which tensor dim has the sharding
        src_mesh_dims, tgt_mesh_dims: mesh dim tuples for this tensor dim
        mesh_shape: tuple of mesh dim sizes in sorted mesh dim order

    Returns:
        list of (src_rank, tgt_rank) pairs if ppermute,
        None if identity (no communication needed),
        "all_to_all" string if not a valid permutation.
    """
    src_r2c, src_total = _build_rank_to_chunk(source, tensor_dim, src_mesh_dims, mesh_shape)
    tgt_r2c, tgt_total = _build_rank_to_chunk(target, tensor_dim, tgt_mesh_dims, mesh_shape)

    if src_r2c is None or tgt_r2c is None:
        return "all_to_all"

    total_ranks = product(mesh_shape)

    # Compute offset difference in chunk space
    src_offset = getattr(source, 'offset', {}).get(tensor_dim, 0)
    tgt_offset = getattr(target, 'offset', {}).get(tensor_dim, 0)

    # Get divisor for element-to-chunk conversion
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
        # No offset difference — try CuTe composition (fast path)
        try:
            perm_layout = composition(right_inverse(tgt_r2c), src_r2c)
            perm = []
            is_identity = True
            for r in range(total_ranks):
                tgt_r = perm_layout(r)
                perm.append((r, tgt_r))
                if r != tgt_r:
                    is_identity = False
            if is_identity:
                return None
            if len(set(p[1] for p in perm)) == total_ranks:
                return perm
            return "all_to_all"
        except Exception:
            return "all_to_all"

    # Offset differs — enumerate directly with modular arithmetic
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
            return "all_to_all"
        perm.append((src_rank, tgt_rank))
        if src_rank != tgt_rank:
            is_identity = False

    if is_identity:
        return None
    if len(set(p[0] for p in perm)) != total_ranks:
        return "all_to_all"
    return perm


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

    # Identify tensor dims that need joint (non-per-mesh-dim) analysis:
    # coupled mesh dims (S(0)S(0)) or offset differences.
    # Their mesh dims are skipped in the per-mesh-dim loop and handled after.
    joint_tensor_dims = {}  # td -> (src_mds, tgt_mds)
    skip_mesh_dims = set()
    for td, src_mds in source.mesh_dim_map.items():
        if not src_mds:
            continue
        tgt_mds = target.mesh_dim_map.get(td, ())
        if set(src_mds) != set(tgt_mds):
            continue
        is_coupled = len(src_mds) > 1
        has_offset = td in offset_differs
        if is_coupled or has_offset:
            joint_tensor_dims[td] = (src_mds, tgt_mds)
            skip_mesh_dims.update(src_mds)

    # Per mesh dim: compare GPU strides from hier_layout
    for md in sorted(all_mesh_dims):
        if md in skip_mesh_dims:
            continue  # handled by joint tensor dim analysis below

        src_info = _get_per_mesh_dim_gpu_stride(source, md)
        tgt_info = _get_per_mesh_dim_gpu_stride(target, md)

        src_gs = src_info[1] if src_info else 0
        tgt_gs = tgt_info[1] if tgt_info else 0

        src_dims = src_reverse.get(md, set())
        tgt_dims = tgt_reverse.get(md, set())

        if src_gs == 0 and tgt_gs == 0:
            continue  # no_op
        elif src_gs != 0 and tgt_gs == 0:
            collectives.append(("all_gather", md, {"tensor_dims": src_dims}))
        elif src_gs == 0 and tgt_gs != 0:
            continue  # local reinterpret
        elif src_gs == tgt_gs:
            continue  # same stride, no communication
        else:
            # Both sharded, different strides.
            # Use rank-to-chunk composition to classify as ppermute or all_to_all.
            src_td = src_info[0] if src_info else None
            tgt_td = tgt_info[0] if tgt_info else None

            if src_td is not None and tgt_td is not None:
                src_mds = source.mesh_dim_map.get(src_td, ())
                tgt_mds = target.mesh_dim_map.get(tgt_td, ())

                if src_td == tgt_td and len(src_mds) == 1 and len(tgt_mds) == 1:
                    src_ms = _get_mesh_dim_size(source, md)
                    tgt_ms = _get_mesh_dim_size(target, md)
                    if src_ms is not None and tgt_ms is not None and src_ms == tgt_ms:
                        mesh_shape = (src_ms,)
                        result = _compute_ppermute(
                            source, target, src_td, src_mds, tgt_mds, mesh_shape
                        )
                        if result is None:
                            continue  # identity
                        elif result != "all_to_all":
                            collectives.append(("ppermute", md, {
                                "perm": result,
                                "source_dims": src_dims,
                                "target_dims": tgt_dims,
                            }))
                            continue

            # Fall through to all_to_all
            collectives.append(("all_to_all", md, {
                "source_dims": src_dims,
                "target_dims": tgt_dims,
            }))

    # Joint tensor dim analysis: coupled mesh dims and/or offset differences.
    # These can't be decomposed into per-mesh-dim collectives.
    for td, (src_mds, tgt_mds) in joint_tensor_dims.items():
        mesh_shape = {}
        for md in src_mds:
            ms = _get_mesh_dim_size(source, md)
            if ms is not None:
                mesh_shape[md] = ms

        all_mds = sorted(mesh_shape.keys())
        mesh_shape_tuple = tuple(mesh_shape[md] for md in all_mds)

        result = _compute_ppermute(
            source, target, td, src_mds, tgt_mds, mesh_shape_tuple
        )

        if result is None:
            continue  # identity
        elif result == "all_to_all":
            collectives.append(("all_to_all", None, {
                "source_dims": {td},
                "target_dims": {td},
            }))
        else:
            collectives.append(("ppermute", None, {"perm": result}))

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
