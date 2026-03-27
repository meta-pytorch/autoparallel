"""
CuTe-based sharding propagation rules.

All rules operate on CutePlacement with rank-2 layouts:
    (device_idx, local_idx...) -> flat_offset

Rules are strictly redistribution-free: they return output placements
assuming the given inputs are already in place, or None if incompatible.

View propagation follows S_C = S_A . R where R is the reshape bijection.
All view cases (identity, flatten, split) use composition(R, tiler).
"""

from ._pycute import (
    Layout,
    ScaledBasis,
    coalesce,
    composition,
    flatten,
    is_tuple,
    make_basis_like,
    make_layout,
    product,
)
from .placement import CutePlacement


# =============================================================================
# View / Reshape propagation
# =============================================================================


def _compute_view_dim_mapping(input_shape, output_shape):
    """
    Compute how input dimensions map to output dimensions in a view/reshape.

    Returns a dict: output_dim -> mapping, where mapping is one of:
        ("identity", input_dim)
        ("flatten", [input_dim, ...])
        ("split", input_dim, group_shape, split_id)
        ("new",)
    """
    i, j = 0, 0
    mappings = []

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

    while i < len(input_shape):
        assert input_shape[i] == 1
        mappings.append(([i], []))
        i += 1
    while j < len(output_shape):
        assert output_shape[j] == 1
        mappings.append(([], [j]))
        j += 1

    result = {}
    for in_group, out_group in mappings:
        if len(in_group) == 1 and len(out_group) == 1:
            result[out_group[0]] = ("identity", in_group[0])
        elif len(in_group) > 1 and len(out_group) == 1:
            result[out_group[0]] = ("flatten", in_group)
        elif len(in_group) == 1 and len(out_group) > 1:
            group_shape = tuple(output_shape[d] for d in out_group)
            for idx, out_dim in enumerate(out_group):
                result[out_dim] = ("split", in_group[0], group_shape, idx)
        elif len(in_group) == 0 and len(out_group) == 1:
            result[out_group[0]] = ("new",)
        elif len(in_group) == 1 and len(out_group) == 0:
            pass
        else:
            group_shape = tuple(output_shape[d] for d in out_group)
            for idx, out_dim in enumerate(out_group):
                result[out_dim] = ("flatten_split", in_group, group_shape, idx)

    return result


def _build_inverse_mapping(dim_mapping):
    """input_dim -> list of (mapping_type, out_dim, ...) entries."""
    inv = {}
    for out_dim, mapping in dim_mapping.items():
        if mapping[0] == "identity":
            inv.setdefault(mapping[1], []).append(("identity", out_dim))
        elif mapping[0] == "flatten":
            for in_dim in mapping[1]:
                inv.setdefault(in_dim, []).append(("flatten", out_dim, mapping[1]))
        elif mapping[0] == "split":
            inv.setdefault(mapping[1], []).append(
                ("split", out_dim, mapping[2], mapping[3])
            )
    return inv


def _has_coord_strides(layout):
    """Check if a layout has ScaledBasis (coordinate) strides."""
    strides = flatten(layout.stride) if is_tuple(layout.stride) else (layout.stride,)
    return any(isinstance(s, ScaledBasis) for s in strides)


def _build_reshape_composition(entries, placement, input_shape):
    """
    Build the reshape bijection R and tiler for S_C = composition(R, tiler).

    Returns (R, tiler, device_stride, out_dim) or None if incompatible.
    For split entries, out_dim is determined after inspecting the result.
    """
    shard_dim = placement.dim
    mesh_size = placement.mesh_dim_size
    entry = entries[0]

    if entry[0] == "identity":
        dim_size = input_shape[shard_dim]
        if dim_size % mesh_size != 0:
            return None
        local_shard = dim_size // mesh_size
        R = Layout(dim_size)
        tiler = Layout(local_shard)
        device_stride = local_shard
        return R, tiler, device_stride, entry[1]

    elif entry[0] == "flatten":
        in_dims = entry[2]
        pos = in_dims.index(shard_dim)
        dim_sizes = tuple(input_shape[d] for d in in_dims)
        shard_dim_size = dim_sizes[pos]

        if shard_dim_size % mesh_size != 0:
            return None
        local_shard = shard_dim_size // mesh_size

        R = Layout(dim_sizes)
        tiler = tuple(
            Layout(local_shard) if k == pos else Layout(dim_sizes[k])
            for k in range(len(dim_sizes))
        )
        R_stride_at_pos = R.stride[pos] if is_tuple(R.stride) else R.stride
        device_stride = R_stride_at_pos * local_shard
        return R, tiler, device_stride, entry[1]

    elif entries[0][0] == "split":
        group_shape = entries[0][2]
        # Use coordinate strides so composition tags per-piece coverage
        R = Layout(group_shape, make_basis_like(group_shape))
        local_layout = Layout(placement.local_shape, placement.local_stride)
        tiler = local_layout
        return R, tiler, None, None  # out_dim determined from result

    return None


def propagate_view(placements, input_shape, output_shape, mesh_sizes):
    """
    Propagate CutePlacements through a view/reshape operation.

    Each device performs a local view — no communication involved.
    All cases follow S_C = S_A . R via composition(R, tiler).

    Returns:
        tuple of CutePlacement, or None if incompatible.
    """
    dim_mapping = _compute_view_dim_mapping(input_shape, output_shape)
    inv_mapping = _build_inverse_mapping(dim_mapping)

    output_places = []

    for mesh_dim, placement in enumerate(placements):
        mesh_size = mesh_sizes[mesh_dim]

        if placement.is_replicate():
            output_places.append(CutePlacement.replicate(mesh_size))
            continue

        shard_dim = placement.dim
        if shard_dim not in inv_mapping:
            return None

        entries = inv_mapping[shard_dim]

        # Build R and tiler for this reshape
        args = _build_reshape_composition(entries, placement, input_shape)
        if args is None:
            return None
        R, tiler, device_stride, out_dim = args

        # S_C = composition(R, tiler), then coalesce
        result = coalesce(composition(R, tiler))

        if _has_coord_strides(result):
            # Split case: result has E(k) strides tagging each mode with
            # its piece index. Find the sharded piece (size < group_shape[k]).
            group_shape = entries[0][2]

            strides = flatten(result.stride) if is_tuple(result.stride) else (result.stride,)
            shapes = flatten(result.shape) if is_tuple(result.shape) else (result.shape,)

            shard_piece = None
            for stride, size in zip(strides, shapes):
                if isinstance(stride, ScaledBasis):
                    k = stride.index
                    if size < group_shape[k]:
                        if shard_piece is not None:
                            return None  # multiple sharded pieces
                        shard_piece = k

            if shard_piece is None:
                return None

            piece_size = group_shape[shard_piece]
            if piece_size % mesh_size != 0:
                return None

            target_entry = next((e for e in entries if e[3] == shard_piece), None)
            if target_entry is None:
                return None

            # Local layout for the sharded piece: coverage contiguous elements
            coverage = piece_size // mesh_size
            out_dim = target_entry[1]
            device_stride = coverage
            result = Layout(coverage)

        # Build output CutePlacement: (device_mode, local_layout)
        out_layout = make_layout(Layout(mesh_size, device_stride), result)
        output_places.append(CutePlacement(dim=out_dim, layout=out_layout))

    return tuple(output_places)


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


def propagate_einsum(
    equation, placements_a, placements_b, shape_a, shape_b, mesh_sizes
):
    """
    Propagate CutePlacements through an einsum (redistribution-free).

    Valid strategies per mesh dim:
      (R, R) -> R
      (S(m), R) -> S(m)
      (R, S(n)) -> S(n)
      (S(k), S(k)) -> Partial
      (S(b), S(b)) -> S(b)  [batch dims only]
    """
    inputs, output = _parse_einsum(equation)
    categories = _classify_einsum_dims(inputs, output)
    out_label_to_dim = {label: i for i, label in enumerate(output)}

    output_placements = []

    for mesh_dim in range(len(mesh_sizes)):
        mesh_size = mesh_sizes[mesh_dim]
        pa = placements_a[mesh_dim]
        pb = placements_b[mesh_dim]

        if pa.is_replicate() and pb.is_replicate():
            output_placements.append(CutePlacement.replicate(mesh_size))
            continue

        if not pa.is_replicate() and not pb.is_replicate():
            a_dim, b_dim = pa.dim, pb.dim
            if (
                a_dim is not None
                and b_dim is not None
                and a_dim < len(inputs[0])
                and b_dim < len(inputs[1])
            ):
                a_label = inputs[0][a_dim]
                b_label = inputs[1][b_dim]
                if a_label == b_label:
                    cat = categories.get(a_label, "other")
                    if cat == "contract":
                        p = CutePlacement.replicate(mesh_size)
                        p._is_partial = True
                        output_placements.append(p)
                        continue
                    elif cat == "batch":
                        out_dim = out_label_to_dim[a_label]
                        output_placements.append(
                            CutePlacement(dim=out_dim, layout=pa.layout)
                        )
                        continue
            return None

        if not pa.is_replicate():
            a_dim = pa.dim
            if a_dim is None or a_dim >= len(inputs[0]):
                return None
            label = inputs[0][a_dim]
            if categories.get(label) == "m":
                out_dim = out_label_to_dim[label]
                output_placements.append(
                    CutePlacement(dim=out_dim, layout=pa.layout)
                )
            else:
                return None
        else:
            b_dim = pb.dim
            if b_dim is None or b_dim >= len(inputs[1]):
                return None
            label = inputs[1][b_dim]
            if categories.get(label) == "n":
                out_dim = out_label_to_dim[label]
                output_placements.append(
                    CutePlacement(dim=out_dim, layout=pb.layout)
                )
            else:
                return None

    return tuple(output_placements)


# =============================================================================
# Pointwise propagation
# =============================================================================


def propagate_pointwise(all_placements, shapes, mesh_sizes):
    """
    Propagate CutePlacements through a pointwise op (redistribution-free).

    All sharded inputs must agree. Replicate inputs must have size 1 on
    the sharded dim (broadcasting) — otherwise they'd need slicing.
    """
    output_placements = []

    for mesh_dim in range(len(mesh_sizes)):
        mesh_size = mesh_sizes[mesh_dim]
        candidate = None

        for inp_idx, inp_placements in enumerate(all_placements):
            p = inp_placements[mesh_dim]
            if p.is_replicate():
                continue
            if p.dim is not None and p.dim < len(shapes[inp_idx]):
                if shapes[inp_idx][p.dim] == 1:
                    continue
            if candidate is None:
                candidate = p
            elif candidate != p:
                return None

        if candidate is not None:
            for inp_idx, inp_placements in enumerate(all_placements):
                p = inp_placements[mesh_dim]
                if not p.is_replicate():
                    continue
                if candidate.dim is not None and candidate.dim < len(shapes[inp_idx]):
                    if shapes[inp_idx][candidate.dim] > 1:
                        return None
            output_placements.append(candidate)
        else:
            output_placements.append(CutePlacement.replicate(mesh_size))

    return tuple(output_placements)


# =============================================================================
# Reduction propagation
# =============================================================================


def propagate_reduction(placements, reduce_dim, keepdim, mesh_sizes):
    """
    Propagate CutePlacements through a reduction (redistribution-free).

    Shard on reduce_dim -> Partial (needs all-reduce, inherent to algorithm).
    Shard on other dim -> dim adjustment if not keepdim.
    """
    output_placements = []

    for mesh_dim, placement in enumerate(placements):
        mesh_size = mesh_sizes[mesh_dim]

        if placement.is_replicate():
            output_placements.append(CutePlacement.replicate(mesh_size))
        elif placement.dim == reduce_dim:
            p = CutePlacement.replicate(mesh_size)
            p._is_partial = True
            output_placements.append(p)
        elif placement.dim is not None and placement.dim > reduce_dim and not keepdim:
            output_placements.append(
                CutePlacement(dim=placement.dim - 1, layout=placement.layout)
            )
        else:
            output_placements.append(placement)

    return tuple(output_placements)
