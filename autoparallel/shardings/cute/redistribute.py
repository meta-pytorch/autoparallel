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
  src>0, tgt>0, diff -> all_to_all

Refinements:
  max_common_vector(src_local, tgt_local) -> vectorization factor
  composition(right_inverse(tgt), src) -> full element-level send/recv schedule
"""

from ._pycute import Layout, is_tuple, logical_divide, product, composition, right_inverse, coalesce, max_common_vector

from .placement import _has_mesh


# =============================================================================
# Redistribute planning — per-mesh-dim GPU stride classification
# =============================================================================


def _get_per_mesh_dim_gpu_stride(sharded, mesh_dim):
    """Get the GPU stride for a specific mesh dim from hier_layout.

    Walks hier_layout's 3-level structure to find the stride element
    corresponding to the given mesh dim. Returns (tensor_dim, gpu_stride)
    or None if this mesh dim doesn't shard any tensor dim.

    For S(0)S(0), nested logical_divide produces innermost-first ordering
    in the tuple (complement from last divide comes before mesh from first divide),
    but mesh_dim_map is outermost-first (first spec = mesh_dim 0 = outermost).
    So we reverse the flattened mesh elements to match mesh_dim_map order.
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

            # Reverse: nested logical_divide produces innermost-first,
            # but mesh_dim_map is outermost-first (construction order).
            all_mesh.reverse()

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

    # Per mesh dim: compare GPU strides from hier_layout
    for md in sorted(all_mesh_dims):
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
            collectives.append(("all_to_all", md, {
                "source_dims": src_dims,
                "target_dims": tgt_dims,
            }))

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
