"""
CuTe-based sharding propagation rules.

All rules operate purely on CutePlacement (one per mesh dimension).
They determine how placements transform through tensor operations:
- view/reshape: flatten, split, permute
- einsum/matmul: batch, M, N, K dimension handling
- pointwise: element-wise ops with broadcasting
- reduction: sum, mean, etc. along a dimension
"""

import re

from torch.distributed._pycute import Layout, coalesce, flatten, is_tuple, product, suffix_product
from .placement import CutePlacement


# =============================================================================
# View / Reshape propagation
# =============================================================================


def _compute_view_dim_mapping(input_shape, output_shape):
    """
    Compute how input dimensions map to output dimensions in a view/reshape.

    Returns a list of (output_dim, mapping) tuples where mapping is one of:
        ("identity", input_dim)
        ("flatten", [input_dim, ...])    — multiple input dims collapse into one output dim
        ("split", input_dim, group_shape, split_id)  — one input dim splits into multiple output dims

    This is a simplified version of PyTorch's dim_maps that works directly
    on shapes without needing the op schema machinery.
    """
    # Use the view_groups algorithm: greedily match input/output dims
    # by finding groups where the product of input dims equals the product
    # of output dims.
    i, j = 0, 0
    mappings = []  # list of (input_dims, output_dims) groups

    while i < len(input_shape) and j < len(output_shape):
        in_group = [i]
        out_group = [j]
        in_prod = input_shape[i]
        out_prod = output_shape[j]
        i += 1
        j += 1

        while in_prod != out_prod:
            if in_prod < out_prod:
                in_group.append(i)
                in_prod *= input_shape[i]
                i += 1
            else:
                out_group.append(j)
                out_prod *= output_shape[j]
                j += 1

        mappings.append((in_group, out_group))

    # Handle trailing size-1 dims
    while i < len(input_shape):
        assert input_shape[i] == 1
        mappings.append(([i], []))
        i += 1
    while j < len(output_shape):
        assert output_shape[j] == 1
        mappings.append(([], [j]))
        j += 1

    # Convert to typed mappings per output dim
    result = {}  # output_dim -> mapping info
    for in_group, out_group in mappings:
        if len(in_group) == 1 and len(out_group) == 1:
            # Identity: one input dim -> one output dim
            result[out_group[0]] = ("identity", in_group[0])
        elif len(in_group) > 1 and len(out_group) == 1:
            # Flatten: multiple input dims -> one output dim
            result[out_group[0]] = ("flatten", in_group)
        elif len(in_group) == 1 and len(out_group) > 1:
            # Split: one input dim -> multiple output dims
            group_shape = tuple(output_shape[d] for d in out_group)
            for idx, out_dim in enumerate(out_group):
                result[out_dim] = ("split", in_group[0], group_shape, idx)
        elif len(in_group) == 0 and len(out_group) == 1:
            # New dim of size 1
            result[out_group[0]] = ("new",)
        elif len(in_group) == 1 and len(out_group) == 0:
            # Removed dim of size 1
            pass
        else:
            # General reshape: flatten then split
            # Treat as flatten of in_group, then split into out_group
            flat_dim = out_group[0]  # assign to first output dim
            group_shape = tuple(output_shape[d] for d in out_group)
            for idx, out_dim in enumerate(out_group):
                result[out_dim] = ("flatten_split", in_group, group_shape, idx)

    return result


def _build_inverse_mapping(dim_mapping, input_shape, output_shape):
    """
    Build inverse mapping: for each input dim, which output dim(s) does it appear in?
    Returns dict: input_dim -> list of (output_dim, mapping_type, ...) entries
    """
    inv = {}
    for out_dim, mapping in dim_mapping.items():
        if mapping[0] == "identity":
            in_dim = mapping[1]
            inv.setdefault(in_dim, []).append(("identity", out_dim))
        elif mapping[0] == "flatten":
            for in_dim in mapping[1]:
                inv.setdefault(in_dim, []).append(("flatten", out_dim, mapping[1]))
        elif mapping[0] == "split":
            in_dim = mapping[1]
            inv.setdefault(in_dim, []).append(
                ("split", out_dim, mapping[2], mapping[3])
            )
    return inv


def propagate_view(placements, input_shape, output_shape, mesh_sizes):
    """
    Propagate CutePlacements through a view/reshape operation.

    Args:
        placements: tuple of CutePlacement, one per mesh dimension
        input_shape: tuple of ints, global input tensor shape
        output_shape: tuple of ints, global output tensor shape
        mesh_sizes: tuple of ints, size of each mesh dimension

    Returns:
        (input_target_placements, output_placements): tuple of CutePlacement tuples.
        input_target_placements: what the input should be redistributed to (if needed).
        output_placements: the resulting placements after the view.
    """
    dim_mapping = _compute_view_dim_mapping(input_shape, output_shape)
    inv_mapping = _build_inverse_mapping(dim_mapping, input_shape, output_shape)

    input_targets = []
    output_places = []

    for mesh_dim, placement in enumerate(placements):
        mesh_size = mesh_sizes[mesh_dim]

        if placement.is_replicate():
            # Replicate passes through any view unchanged
            input_targets.append(placement)
            output_places.append(CutePlacement.replicate(mesh_size))
            continue

        shard_dim = placement.dim
        if shard_dim not in inv_mapping:
            # Sharded dim was removed (size-1 dim squeezed out)
            input_targets.append(CutePlacement.replicate(mesh_size))
            output_places.append(CutePlacement.replicate(mesh_size))
            continue

        entries = inv_mapping[shard_dim]
        # We only handle the case where the input dim maps to exactly one output group
        entry = entries[0]

        if entry[0] == "identity":
            # Direct 1:1 mapping: rewrite dim, keep layout
            out_dim = entry[1]
            input_targets.append(placement)
            output_places.append(
                CutePlacement(dim=out_dim, layout=placement.layout)
            )

        elif entry[0] == "flatten":
            out_dim = entry[1]
            in_dims = entry[2]  # list of input dims being flattened

            # Find position of our sharded dim within the flatten group
            pos = in_dims.index(shard_dim)
            dim_sizes = tuple(input_shape[d] for d in in_dims)
            shard_dim_size = dim_sizes[pos]

            if shard_dim_size % mesh_size != 0:
                # Can't shard: not evenly divisible
                input_targets.append(CutePlacement.replicate(mesh_size))
                output_places.append(CutePlacement.replicate(mesh_size))
                continue

            local_shard = shard_dim_size // mesh_size

            # In the flattened output dim, the elements are laid out in
            # colexicographic order of the input dims (rightmost varies fastest
            # for C-contiguous / row-major view semantics — but PyTorch views
            # use row-major, meaning leftmost varies slowest).
            #
            # For a flatten of dims with sizes (S0, S1, ..., Sn):
            #   flat_idx = s0 * (S1*S2*...*Sn) + s1 * (S2*...*Sn) + ... + sn
            #
            # If we shard dim at position `pos` (with size Sk) by D devices,
            # device d holds sk in [d*Sk/D, (d+1)*Sk/D).
            # The local elements are at flat indices:
            #   { s0*stride0 + ... + sk*stridek + ... + sn*striden }
            # where sk ranges over [0, Sk/D) and all other si range fully.
            #
            # This forms a hierarchical CuTe layout:
            # - "pre" dims (i < pos): sizes Si, strides = product of sizes after
            # - the shard dim: size Sk/D, stride = product of sizes after pos
            # - "post" dims (i > pos): sizes Si, strides = product of sizes after i
            #
            # We build this as: shape = (*pre_sizes, local_shard, *post_sizes)
            #                   stride = (*pre_strides, post_product, *post_strides)

            # Compute strides for each dim in the flatten group (row-major)
            # stride[i] = product of dim_sizes[i+1:]
            n = len(dim_sizes)
            strides = [1] * n
            for k in range(n - 2, -1, -1):
                strides[k] = strides[k + 1] * dim_sizes[k + 1]

            shape_parts = []
            stride_parts = []

            # Pre dims (before the sharded dim)
            for k in range(pos):
                if dim_sizes[k] > 1:
                    shape_parts.append(dim_sizes[k])
                    stride_parts.append(strides[k])

            # The sharded dim (local portion)
            shape_parts.append(local_shard)
            stride_parts.append(strides[pos])

            # Post dims (after the sharded dim)
            for k in range(pos + 1, n):
                if dim_sizes[k] > 1:
                    shape_parts.append(dim_sizes[k])
                    stride_parts.append(strides[k])

            if len(shape_parts) == 1:
                out_layout = Layout(shape_parts[0], stride_parts[0])
            else:
                out_layout = Layout(
                    tuple(shape_parts), tuple(stride_parts)
                )

            input_targets.append(placement)
            output_places.append(CutePlacement(dim=out_dim, layout=out_layout))

        elif entry[0] == "split":
            out_dim = entry[1]
            group_shape = entry[2]
            split_id = entry[3]

            shard_dim_size = input_shape[shard_dim]
            if shard_dim_size % mesh_size != 0:
                input_targets.append(CutePlacement.replicate(mesh_size))
                output_places.append(CutePlacement.replicate(mesh_size))
                continue

            if split_id == 0:
                # First piece of the split: shard propagates if divisible
                piece_size = group_shape[0]
                if piece_size % mesh_size == 0:
                    input_targets.append(placement)
                    output_places.append(
                        CutePlacement.shard(out_dim, piece_size, mesh_size)
                    )
                else:
                    # Can't shard this split piece
                    input_targets.append(CutePlacement.replicate(mesh_size))
                    output_places.append(CutePlacement.replicate(mesh_size))
            else:
                # Non-first split piece: can't shard with standard approach
                # Use CuTe composition to see if we can represent it
                input_targets.append(CutePlacement.replicate(mesh_size))
                output_places.append(CutePlacement.replicate(mesh_size))

        else:
            # Unknown mapping: fall back to replicate
            input_targets.append(CutePlacement.replicate(mesh_size))
            output_places.append(CutePlacement.replicate(mesh_size))

    return tuple(input_targets), tuple(output_places)


# =============================================================================
# Einsum / Matmul propagation
# =============================================================================


def _parse_einsum(equation):
    """
    Parse an einsum equation into input and output subscripts.

    Returns:
        (inputs, output) where inputs is a list of strings and output is a string.
        e.g. "bmk,bkn->bmn" -> (["bmk", "bkn"], "bmn")
    """
    if "->" in equation:
        inputs_str, output = equation.split("->")
    else:
        raise ValueError(f"Einsum equation must contain '->': {equation}")
    inputs = inputs_str.split(",")
    return inputs, output


def _classify_einsum_dims(inputs, output):
    """
    Classify each dimension label in an einsum into categories.

    Returns:
        dict mapping label -> one of:
            "batch"  — appears in all inputs and output
            "contract" — appears in all inputs but not output
            "m"  — appears in first input and output only
            "n"  — appears in second input and output only
    """
    assert len(inputs) == 2, "Only 2-operand einsum supported"
    a_dims, b_dims = set(inputs[0]), set(inputs[1])
    out_dims = set(output)

    categories = {}
    all_dims = a_dims | b_dims | out_dims

    for d in all_dims:
        in_a = d in a_dims
        in_b = d in b_dims
        in_out = d in out_dims

        if in_a and in_b and in_out:
            categories[d] = "batch"
        elif in_a and in_b and not in_out:
            categories[d] = "contract"
        elif in_a and not in_b and in_out:
            categories[d] = "m"
        elif not in_a and in_b and in_out:
            categories[d] = "n"
        else:
            # Other patterns (e.g., diagonal, trace) - treat as unsupported
            categories[d] = "other"

    return categories


def propagate_einsum(
    equation, placements_a, placements_b, shape_a, shape_b, mesh_sizes
):
    """
    Propagate CutePlacements through an einsum operation.

    Args:
        equation: einsum equation string, e.g. "bmk,bkn->bmn"
        placements_a: tuple of CutePlacement for first input
        placements_b: tuple of CutePlacement for second input
        shape_a: global shape of first input
        shape_b: global shape of second input
        mesh_sizes: tuple of mesh dimension sizes

    Returns:
        (target_a, target_b, output_placements): tuples of CutePlacement
    """
    inputs, output = _parse_einsum(equation)
    categories = _classify_einsum_dims(inputs, output)

    # Build dim index maps: label -> position in tensor
    a_label_to_dim = {label: i for i, label in enumerate(inputs[0])}
    b_label_to_dim = {label: i for i, label in enumerate(inputs[1])}
    out_label_to_dim = {label: i for i, label in enumerate(output)}

    target_a = list(placements_a)
    target_b = list(placements_b)
    output_placements = [CutePlacement.replicate(m) for m in mesh_sizes]

    for mesh_dim in range(len(mesh_sizes)):
        mesh_size = mesh_sizes[mesh_dim]
        pa = placements_a[mesh_dim]
        pb = placements_b[mesh_dim]

        if pa.is_replicate() and pb.is_replicate():
            # Both replicate: output replicate
            continue

        if not pa.is_replicate():
            # A is sharded on some dim
            a_dim = pa.dim
            if a_dim is not None and a_dim < len(inputs[0]):
                label = inputs[0][a_dim]
                cat = categories.get(label, "other")

                if cat == "batch":
                    # Batch dim: passes through to output
                    out_dim = out_label_to_dim[label]
                    target_b[mesh_dim] = CutePlacement(
                        dim=b_label_to_dim[label], layout=pa.layout
                    )
                    output_placements[mesh_dim] = CutePlacement(
                        dim=out_dim, layout=pa.layout
                    )
                elif cat == "m":
                    # M dim (in A and output, not in B): B should be replicate
                    out_dim = out_label_to_dim[label]
                    target_b[mesh_dim] = CutePlacement.replicate(mesh_size)
                    output_placements[mesh_dim] = CutePlacement(
                        dim=out_dim, layout=pa.layout
                    )
                elif cat == "contract":
                    # Contraction dim: both inputs sharded, output is partial
                    # We represent partial as replicate here (caller handles partial)
                    target_b[mesh_dim] = CutePlacement(
                        dim=b_label_to_dim[label], layout=pa.layout
                    )
                    output_placements[mesh_dim] = CutePlacement.replicate(
                        mesh_size
                    )
                    # Mark as partial via a special attribute
                    output_placements[mesh_dim]._is_partial = True
                else:
                    # Fall back to replicate
                    target_a[mesh_dim] = CutePlacement.replicate(mesh_size)

        elif not pb.is_replicate():
            # B is sharded, A is replicate
            b_dim = pb.dim
            if b_dim is not None and b_dim < len(inputs[1]):
                label = inputs[1][b_dim]
                cat = categories.get(label, "other")

                if cat == "batch":
                    out_dim = out_label_to_dim[label]
                    target_a[mesh_dim] = CutePlacement(
                        dim=a_label_to_dim[label], layout=pb.layout
                    )
                    output_placements[mesh_dim] = CutePlacement(
                        dim=out_dim, layout=pb.layout
                    )
                elif cat == "n":
                    out_dim = out_label_to_dim[label]
                    target_a[mesh_dim] = CutePlacement.replicate(mesh_size)
                    output_placements[mesh_dim] = CutePlacement(
                        dim=out_dim, layout=pb.layout
                    )
                elif cat == "contract":
                    target_a[mesh_dim] = CutePlacement(
                        dim=a_label_to_dim[label], layout=pb.layout
                    )
                    output_placements[mesh_dim] = CutePlacement.replicate(
                        mesh_size
                    )
                    output_placements[mesh_dim]._is_partial = True
                else:
                    target_b[mesh_dim] = CutePlacement.replicate(mesh_size)

    return tuple(target_a), tuple(target_b), tuple(output_placements)


# =============================================================================
# Pointwise propagation
# =============================================================================


def propagate_pointwise(all_placements, shapes, mesh_sizes):
    """
    Propagate CutePlacements through a pointwise (element-wise) operation.

    All inputs must agree on placement per mesh dim (after broadcasting).
    Broadcasting: an input with size 1 on a sharded dim is treated as replicate.

    Args:
        all_placements: list of tuples of CutePlacement, one tuple per input
        shapes: list of input shapes
        mesh_sizes: tuple of mesh dimension sizes

    Returns:
        output_placements: tuple of CutePlacement for the output
    """
    num_mesh_dims = len(mesh_sizes)
    output_placements = []

    for mesh_dim in range(num_mesh_dims):
        mesh_size = mesh_sizes[mesh_dim]
        # Collect non-replicate placements across inputs
        candidate = None
        all_agree = True

        for inp_idx, inp_placements in enumerate(all_placements):
            p = inp_placements[mesh_dim]

            if p.is_replicate():
                continue

            # Check if this input has size 1 on the sharded dim (broadcast)
            if p.dim is not None and p.dim < len(shapes[inp_idx]):
                if shapes[inp_idx][p.dim] == 1:
                    # Broadcasting from size 1: effectively replicate
                    continue

            if candidate is None:
                candidate = p
            elif candidate != p:
                all_agree = False
                break

        if candidate is not None and all_agree:
            output_placements.append(candidate)
        else:
            output_placements.append(CutePlacement.replicate(mesh_size))

    return tuple(output_placements)


# =============================================================================
# Reduction propagation
# =============================================================================


def propagate_reduction(placements, reduce_dim, keepdim, mesh_sizes):
    """
    Propagate CutePlacements through a reduction operation.

    Args:
        placements: tuple of CutePlacement, one per mesh dimension
        reduce_dim: the tensor dimension being reduced
        keepdim: whether the reduced dimension is kept (as size 1)
        mesh_sizes: tuple of mesh dimension sizes

    Returns:
        output_placements: tuple of CutePlacement for the output.
        Placements that shard on the reduce_dim will have _is_partial=True.
    """
    output_placements = []

    for mesh_dim, placement in enumerate(placements):
        mesh_size = mesh_sizes[mesh_dim]

        if placement.is_replicate():
            output_placements.append(CutePlacement.replicate(mesh_size))
            continue

        if placement.dim == reduce_dim:
            # Sharding on the reduced dimension: output is partial
            p = CutePlacement.replicate(mesh_size)
            p._is_partial = True
            output_placements.append(p)
        elif placement.dim is not None and placement.dim > reduce_dim and not keepdim:
            # Dims after the reduced dim shift down by 1
            output_placements.append(
                CutePlacement(dim=placement.dim - 1, layout=placement.layout)
            )
        else:
            # Dim before the reduced dim or keepdim=True: unchanged
            output_placements.append(placement)

    return tuple(output_placements)
