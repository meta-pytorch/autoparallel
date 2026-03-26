"""
CuTe-based sharding propagation rules.

All rules operate purely on CutePlacement (one per mesh dimension).
They determine how placements transform through tensor operations,
assuming the given input placements are already in place (no redistribution).

Each rule returns output placements, or None for incompatible inputs.
The caller is responsible for enumerating input placement strategies
and computing redistribution costs.

Supported operations:
- view/reshape: flatten, split, permute
- einsum/matmul: batch, M, N, K dimension handling
- pointwise: element-wise ops with broadcasting
- reduction: sum, mean, etc. along a dimension
"""

from torch.distributed._pycute import Layout, coalesce, flatten, is_tuple, product
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
            pass  # removed size-1 dim
        else:
            group_shape = tuple(output_shape[d] for d in out_group)
            for idx, out_dim in enumerate(out_group):
                result[out_dim] = ("flatten_split", in_group, group_shape, idx)

    return result


def _build_inverse_mapping(dim_mapping):
    """
    Build inverse mapping: input_dim -> list of (mapping_type, out_dim, ...) entries.
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

    Given the input placements, computes the output placements that result
    from each device performing a local view — NO communication involved.

    Args:
        placements: tuple of CutePlacement, one per mesh dimension
        input_shape: tuple of ints, global input tensor shape
        output_shape: tuple of ints, global output tensor shape
        mesh_sizes: tuple of ints, size of each mesh dimension

    Returns:
        output_placements: tuple of CutePlacement, or None if the view is
        incompatible with the given placements (would require redistribution).
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
            # Sharded dim was removed (size-1 dim squeezed out) — incompatible
            return None

        entries = inv_mapping[shard_dim]
        entry = entries[0]

        if entry[0] == "identity":
            out_dim = entry[1]
            output_places.append(
                CutePlacement(dim=out_dim, layout=placement.layout)
            )

        elif entry[0] == "flatten":
            out_dim = entry[1]
            in_dims = entry[2]

            pos = in_dims.index(shard_dim)
            dim_sizes = tuple(input_shape[d] for d in in_dims)
            shard_dim_size = dim_sizes[pos]

            if shard_dim_size % mesh_size != 0:
                return None  # incompatible

            local_shard = shard_dim_size // mesh_size

            # Each device does a local view. In the flattened output dim,
            # the device's elements form a pattern determined by:
            # - which input dims are in the flatten group
            # - which one is sharded and at what position
            #
            # Row-major strides: stride[k] = product(dim_sizes[k+1:])
            n = len(dim_sizes)
            strides = [1] * n
            for k in range(n - 2, -1, -1):
                strides[k] = strides[k + 1] * dim_sizes[k + 1]

            shape_parts = []
            stride_parts = []

            for k in range(pos):
                if dim_sizes[k] > 1:
                    shape_parts.append(dim_sizes[k])
                    stride_parts.append(strides[k])

            shape_parts.append(local_shard)
            stride_parts.append(strides[pos])

            for k in range(pos + 1, n):
                if dim_sizes[k] > 1:
                    shape_parts.append(dim_sizes[k])
                    stride_parts.append(strides[k])

            if len(shape_parts) == 1:
                out_layout = Layout(shape_parts[0], stride_parts[0])
            else:
                out_layout = Layout(tuple(shape_parts), tuple(stride_parts))

            output_places.append(CutePlacement(dim=out_dim, layout=out_layout))

        elif entry[0] == "split":
            out_dim = entry[1]
            group_shape = entry[2]
            split_id = entry[3]

            shard_dim_size = input_shape[shard_dim]
            if shard_dim_size % mesh_size != 0:
                return None

            if split_id == 0:
                piece_size = group_shape[0]
                if piece_size % mesh_size == 0:
                    output_places.append(
                        CutePlacement.shard(out_dim, piece_size, mesh_size)
                    )
                else:
                    return None  # can't shard this split piece
            else:
                return None  # non-first split piece incompatible

        else:
            return None  # unknown mapping

    return tuple(output_places)


# =============================================================================
# Einsum / Matmul propagation
# =============================================================================


def _parse_einsum(equation):
    """Parse einsum equation into (input_subscripts, output_subscript)."""
    if "->" not in equation:
        raise ValueError(f"Einsum equation must contain '->': {equation}")
    inputs_str, output = equation.split("->")
    return inputs_str.split(","), output


def _classify_einsum_dims(inputs, output):
    """
    Classify each dimension label into: batch, contract, m, n, or other.
    """
    assert len(inputs) == 2, "Only 2-operand einsum supported"
    a_dims, b_dims = set(inputs[0]), set(inputs[1])
    out_dims = set(output)

    categories = {}
    for d in a_dims | b_dims | out_dims:
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
            categories[d] = "other"

    return categories


def propagate_einsum(
    equation, placements_a, placements_b, shape_a, shape_b, mesh_sizes
):
    """
    Propagate CutePlacements through an einsum, assuming the given input
    placements are already in place (no redistribution).

    For each mesh dimension, determines the output placement based on
    how A and B are sharded. The only redistribution-free strategies are:

    For mk,kn->mn:
      (R, R) -> R
      (S(m), R) -> S(m)       — A sharded on M, B replicate
      (R, S(n)) -> S(n)       — A replicate, B sharded on N
      (S(k), S(k)) -> P       — both sharded on contraction dim

    For batch dims (appear in both inputs and output):
      (S(b), S(b)) -> S(b)    — both sharded on same batch dim

    Batch dims require BOTH inputs to be sharded identically because
    the local matmul needs matching batch sizes. (S(b), R) is NOT free —
    B has all batches locally but the matmul kernel sees mismatched shapes.

    Returns:
        output_placements tuple, or None if incompatible.
        Partial outputs have _is_partial=True attribute.
    """
    inputs, output = _parse_einsum(equation)
    categories = _classify_einsum_dims(inputs, output)

    a_label_to_dim = {label: i for i, label in enumerate(inputs[0])}
    b_label_to_dim = {label: i for i, label in enumerate(inputs[1])}
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
            # Both sharded — must be on the same dim label
            a_dim = pa.dim
            b_dim = pb.dim
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

        # Exactly one is sharded
        if not pa.is_replicate():
            a_dim = pa.dim
            if a_dim is None or a_dim >= len(inputs[0]):
                return None
            label = inputs[0][a_dim]
            cat = categories.get(label, "other")

            if cat == "m":
                out_dim = out_label_to_dim[label]
                output_placements.append(
                    CutePlacement(dim=out_dim, layout=pa.layout)
                )
            else:
                # batch, contract, n, other — all incompatible when only A is sharded
                return None
        else:
            b_dim = pb.dim
            if b_dim is None or b_dim >= len(inputs[1]):
                return None
            label = inputs[1][b_dim]
            cat = categories.get(label, "other")

            if cat == "n":
                out_dim = out_label_to_dim[label]
                output_placements.append(
                    CutePlacement(dim=out_dim, layout=pb.layout)
                )
            else:
                # batch, contract, m, other — all incompatible when only B is sharded
                return None

    return tuple(output_placements)


# =============================================================================
# Pointwise propagation
# =============================================================================


def propagate_pointwise(all_placements, shapes, mesh_sizes):
    """
    Propagate CutePlacements through a pointwise (element-wise) operation.

    Given the input placements, computes output placements assuming no
    redistribution. All inputs must have compatible placements per mesh dim:
    - All sharded inputs must agree on the same placement.
    - A replicate input is only compatible with a sharded input if the
      replicate tensor has size 1 (or is scalar) on the sharded dim,
      so that broadcasting handles it correctly with no conversion.
      Otherwise, the replicate tensor would need to be sliced to match
      the shard — that's a placement conversion, not redistribution-free.

    Returns:
        output_placements tuple, or None if incompatible.
    """
    num_mesh_dims = len(mesh_sizes)
    output_placements = []

    for mesh_dim in range(num_mesh_dims):
        mesh_size = mesh_sizes[mesh_dim]
        candidate = None

        for inp_idx, inp_placements in enumerate(all_placements):
            p = inp_placements[mesh_dim]

            if p.is_replicate():
                # Will check compatibility with candidate after the loop
                continue

            # Sharded input — check if it's broadcasting (size 1 on sharded dim)
            if p.dim is not None and p.dim < len(shapes[inp_idx]):
                if shapes[inp_idx][p.dim] == 1:
                    # Size-1 shard is effectively replicate
                    continue

            if candidate is None:
                candidate = p
            elif candidate != p:
                return None  # conflicting sharded placements

        if candidate is not None:
            # We have a sharded candidate. Verify all replicate inputs
            # are compatible: they must have size 1 (or be scalar) on
            # the sharded dim so broadcasting works without conversion.
            for inp_idx, inp_placements in enumerate(all_placements):
                p = inp_placements[mesh_dim]
                if not p.is_replicate():
                    continue
                # Replicate input: check if it has size > 1 on the sharded dim
                if candidate.dim is not None and candidate.dim < len(shapes[inp_idx]):
                    if shapes[inp_idx][candidate.dim] > 1:
                        # Replicate tensor has full size on the sharded dim —
                        # local op would see shape mismatch without slicing
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
    Propagate CutePlacements through a reduction operation.

    Given the input placements, computes output placements assuming no
    redistribution during the reduction itself.

    If the input is sharded on the reduce_dim, the output is partial
    (each device holds a partial result that needs an all-reduce — this
    is inherent to the algorithm, not a redistribution).

    Returns:
        output_placements tuple. Partial outputs have _is_partial=True.
    """
    output_placements = []

    for mesh_dim, placement in enumerate(placements):
        mesh_size = mesh_sizes[mesh_dim]

        if placement.is_replicate():
            output_placements.append(CutePlacement.replicate(mesh_size))
            continue

        if placement.dim == reduce_dim:
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
