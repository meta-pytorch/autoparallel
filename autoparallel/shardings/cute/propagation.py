"""
CuTe-based sharding propagation rules.

All rules operate on TiledLayout = (tensor_layout, shard_layout).
Reshape/view/transpose/permute change tensor_layout; shard_layout is invariant.
Slice and gather modify the element space, changing shard_layout via composition.

Rules are strictly redistribution-free: they return output placements
assuming the given inputs are already in place, or None if incompatible.
"""

from ._pycute import Layout, codomain_divide, composition, is_tuple, product
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
# Permute propagation
# =============================================================================


def propagate_permute(tiled, dims):
    """
    Propagate TiledLayout through a permute.

    Reorders tensor_layout modes according to dims. shard_layout unchanged.
    permute(dims) means output[i] = input[dims[i]].
    """
    shape = (
        tiled.tensor_layout.shape
        if is_tuple(tiled.tensor_layout.shape)
        else (tiled.tensor_layout.shape,)
    )
    stride = (
        tiled.tensor_layout.stride
        if is_tuple(tiled.tensor_layout.stride)
        else (tiled.tensor_layout.stride,)
    )
    new_shape = tuple(shape[d] for d in dims)
    new_stride = tuple(stride[d] for d in dims)
    new_tensor = Layout(new_shape, new_stride)
    return TiledLayout(new_tensor, tiled.shard_layout, tiled.mesh_ndim)


# =============================================================================
# Slice propagation
# =============================================================================


def propagate_slice(tiled, dim, index):
    """
    Propagate TiledLayout through a slice (select a single index along dim).

    Uses CuTe's partial evaluation on tensor_layout.

    If dim is sharded, returns None (requires redistribution).
    If dim is not sharded, rebuild shard for the output shape with
    adjusted dim indices.
    """
    # Check if the sliced dim is sharded
    placements = tiled.get_placements()
    for p_type, p_dim in placements:
        if p_type == "shard" and p_dim == dim:
            return None

    # Slice tensor_layout using CuTe slicing
    t_shape = (
        tiled.tensor_layout.shape
        if is_tuple(tiled.tensor_layout.shape)
        else (tiled.tensor_layout.shape,)
    )
    slice_coord = tuple(index if k == dim else None for k in range(len(t_shape)))
    new_tensor = tiled.tensor_layout(*slice_coord)
    output_shape = (
        new_tensor.shape if is_tuple(new_tensor.shape) else (new_tensor.shape,)
    )

    # Rebuild shard for output shape with adjusted dim indices
    for p_type, p_dim in placements:
        if p_type == "shard":
            new_dim = p_dim if p_dim < dim else p_dim - 1
            mesh_dim = placements.index((p_type, p_dim))
            return TiledLayout.shard(
                output_shape, tiled.mesh_shape, new_dim, mesh_dim
            )

    return TiledLayout.replicate(output_shape, tiled.mesh_shape)


# =============================================================================
# Gather propagation
# =============================================================================


def propagate_gather(tiled, dim, index_layout):
    """
    Propagate TiledLayout through a structured gather along dim.

    C[..., i, ...] = A[..., index_layout(i), ...] where index_layout
    is a CuTe Layout mapping output indices to input indices along dim.

    Only works for CuTe-expressible index patterns (contiguous, strided, tiled).
    Arbitrary index tensors cannot be represented as Layouts.

    If dim is sharded, returns None (would need redistribution).
    If dim is not sharded, the gather is local — compose the index mapping
    into the tensor layout.
    """
    # Check if the gathered dim is sharded
    placements = tiled.get_placements()
    for p_type, p_dim in placements:
        if p_type == "shard" and p_dim == dim:
            return None  # gathering on a sharded dim requires redistribution

    # Build output tensor shape: replace dim's size with index_layout's size
    t_shape = (
        tiled.tensor_layout.shape
        if is_tuple(tiled.tensor_layout.shape)
        else (tiled.tensor_layout.shape,)
    )
    t_stride = (
        tiled.tensor_layout.stride
        if is_tuple(tiled.tensor_layout.stride)
        else (tiled.tensor_layout.stride,)
    )

    gather_size = index_layout.size()
    output_shape = t_shape[:dim] + (gather_size,) + t_shape[dim + 1 :]

    # The tensor_layout's dim mode gets composed with index_layout:
    # original: tensor_coord[dim] -> memory offset contribution = coord * stride[dim]
    # after gather: output_idx -> index_layout(output_idx) -> memory offset
    # = index_layout(output_idx) * stride[dim]
    # This is composition of stride[dim] * identity with index_layout,
    # which is just Layout(index_layout.shape, index_layout.stride * stride[dim])

    dim_stride = t_stride[dim]
    if is_tuple(index_layout.stride):
        new_dim_stride = tuple(s * dim_stride for s in index_layout.stride)
    else:
        new_dim_stride = index_layout.stride * dim_stride

    new_dim_shape = index_layout.shape

    output_stride = t_stride[:dim] + (new_dim_stride,) + t_stride[dim + 1 :]
    output_shape_full = t_shape[:dim] + (new_dim_shape,) + t_shape[dim + 1 :]
    new_tensor = Layout(output_shape_full, output_stride)

    # shard_layout: element-index based. Rebuild for output shape.
    for p_type, p_dim in placements:
        if p_type == "shard":
            new_dim = p_dim  # gather doesn't change dim indices (replaces in-place)
            mesh_dim = placements.index((p_type, p_dim))
            return TiledLayout.shard(
                output_shape, tiled.mesh_shape, new_dim, mesh_dim
            )

    return TiledLayout.replicate(output_shape, tiled.mesh_shape)


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
