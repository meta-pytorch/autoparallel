"""
CuTe-based sharding propagation rules.

All rules operate on TiledLayout = (tensor_layout, shard_layout).
Reshape/view/transpose change tensor_layout; shard_layout is invariant.

Rules are strictly redistribution-free: they return output placements
assuming the given inputs are already in place, or None if incompatible.
"""

from ._pycute import Layout, codomain_divide, is_tuple, product
from .placement import TiledLayout


# =============================================================================
# View / Reshape propagation
# =============================================================================


def propagate_view(tiled, new_shape):
    """
    Propagate TiledLayout through a view/reshape.

    Reshape changes the tensor_layout (how coords map to memory).
    The shard_layout (which device holds which elements) is invariant.
    """
    if product(tiled.tensor_shape) != product(new_shape):
        return None
    return TiledLayout(Layout(new_shape), tiled.shard_layout, tiled.mesh_ndim)


# =============================================================================
# Transpose / Permute propagation
# =============================================================================


def propagate_transpose(tiled, dim0, dim1):
    """
    Propagate TiledLayout through a transpose.

    Transpose reorders tensor_layout modes. shard_layout unchanged.
    """
    shape = list(
        tiled.tensor_layout.shape
        if is_tuple(tiled.tensor_layout.shape)
        else (tiled.tensor_layout.shape,)
    )
    stride = list(
        tiled.tensor_layout.stride
        if is_tuple(tiled.tensor_layout.stride)
        else (tiled.tensor_layout.stride,)
    )
    shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
    stride[dim0], stride[dim1] = stride[dim1], stride[dim0]
    new_tensor = Layout(tuple(shape), tuple(stride))
    return TiledLayout(new_tensor, tiled.shard_layout, tiled.mesh_ndim)


# =============================================================================
# Einsum / Matmul propagation
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


def _get_shard_dim(tiled, mesh_dim):
    """Get which tensor dim a specific mesh dim shards, or None for replicate."""
    placements = tiled.get_placements()
    if mesh_dim < len(placements):
        p_type, p_dim = placements[mesh_dim]
        if p_type == "shard":
            return p_dim
    return None


def propagate_einsum(equation, tiled_a, tiled_b, output_shape):
    """
    Propagate TiledLayout through an einsum (redistribution-free).

    Valid strategies:
      Both replicate on a mesh dim -> output replicate
      Both shard on same batch/contract dim -> batch passes through, contract -> Partial
      One shards M (or N), other replicate -> output shards M (or N)
    """
    inputs, output = _parse_einsum(equation)
    categories = _classify_einsum_dims(inputs, output)

    assert tiled_a.mesh_ndim == tiled_b.mesh_ndim
    mesh_ndim = tiled_a.mesh_ndim
    mesh_shape = tiled_a.mesh_shape

    # Build output tensor layout
    out_tensor_layout = Layout(output_shape)

    # For each mesh dim, determine output shard
    # Start from replicate and modify
    out_tiled = TiledLayout.replicate(output_shape, mesh_shape)
    is_partial = False

    for md in range(mesh_ndim):
        a_shard = _get_shard_dim(tiled_a, md)
        b_shard = _get_shard_dim(tiled_b, md)

        if a_shard is None and b_shard is None:
            continue  # both replicate

        if a_shard is not None and b_shard is not None:
            # Both sharded — must be on same label
            if a_shard >= len(inputs[0]) or b_shard >= len(inputs[1]):
                return None
            a_label = inputs[0][a_shard]
            b_label = inputs[1][b_shard]
            if a_label != b_label:
                return None

            cat = categories.get(a_label, "other")
            if cat == "batch":
                out_dim = output.index(a_label)
                out_tiled = TiledLayout.shard(
                    output_shape, mesh_shape, out_dim, md
                )
            elif cat == "contract":
                is_partial = True
            else:
                return None
            continue

        # Exactly one sharded
        if a_shard is not None:
            if a_shard >= len(inputs[0]):
                return None
            label = inputs[0][a_shard]
            cat = categories.get(label, "other")
            if cat == "m":
                out_dim = output.index(label)
                out_tiled = TiledLayout.shard(
                    output_shape, mesh_shape, out_dim, md
                )
            else:
                return None
        else:
            if b_shard >= len(inputs[1]):
                return None
            label = inputs[1][b_shard]
            cat = categories.get(label, "other")
            if cat == "n":
                out_dim = output.index(label)
                out_tiled = TiledLayout.shard(
                    output_shape, mesh_shape, out_dim, md
                )
            else:
                return None

    if is_partial:
        out_tiled._is_partial = True

    return out_tiled


# =============================================================================
# Pointwise propagation
# =============================================================================


def propagate_pointwise(all_tileds, output_shape):
    """
    Propagate TiledLayout through a pointwise op (redistribution-free).

    All inputs must have compatible shard_layouts.
    """
    if not all_tileds:
        return None

    ref = all_tileds[0]
    for tiled in all_tileds[1:]:
        if tiled.shard_layout != ref.shard_layout:
            # Check if one is replicate (broadcasting from size-1 dim)
            if tiled.is_replicate():
                continue
            if ref.is_replicate():
                ref = tiled
                continue
            return None

    # Output has same shard layout, new tensor shape
    return TiledLayout(Layout(output_shape), ref.shard_layout, ref.mesh_ndim)


# =============================================================================
# Reduction propagation
# =============================================================================


def propagate_reduction(tiled, reduce_dim, keepdim, output_shape):
    """
    Propagate TiledLayout through a reduction (redistribution-free).

    If the reduced dim is sharded -> Partial.
    Otherwise -> adjust tensor_layout, shard_layout unchanged.
    """
    # Check if reduce_dim is sharded
    placements = tiled.get_placements()
    for p_type, p_dim in placements:
        if p_type == "shard" and p_dim == reduce_dim:
            # Reducing a sharded dim -> Partial
            result = TiledLayout.replicate(output_shape, tiled.mesh_shape)
            result._is_partial = True
            return result

    # Reducing a non-sharded dim: shard_layout is still valid
    # (element indices don't change for the remaining elements,
    # but we need to account for the collapsed dim)
    # For now, find which dim is sharded and rebuild
    for p_type, p_dim in placements:
        if p_type == "shard":
            # Adjust dim index if needed
            new_dim = p_dim
            if not keepdim and p_dim > reduce_dim:
                new_dim = p_dim - 1
            # Find mesh_dim for this placement
            mesh_dim = placements.index((p_type, p_dim))
            return TiledLayout.shard(
                output_shape, tiled.mesh_shape, new_dim, mesh_dim
            )

    # All replicate
    return TiledLayout.replicate(output_shape, tiled.mesh_shape)
