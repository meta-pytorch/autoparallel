"""
CuTe-based sharding propagation via 5 primitives.

Every propagation rule is a dim recipe composed from:
  Carry  — output dim inherits sharding from input dim(s)
  Insert — new replicate dim
  Remove — dim disappears; if sharded -> Partial
  Merge  — multiple dims become one (view flatten, cat)
  Split  — one dim becomes multiple (view unflatten)

A generic engine applies the recipe on ShardedLayout.

CuTe is NOT used for propagation rules (no composition).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ._pycute import Layout, is_tuple, product

from .placement import (
    ShardedLayout,
    _ensure_tuple,
    _has_mesh,
    _local_size,
    _mode_has_mesh,
    _SIZE_1_MODE,
    _SIZE_1_STRIDE,
)

from torch.distributed.tensor._ops._view_ops import (
    Flatten,
    InputDim,
    Split as ViewSplit,
    view_groups,
)


# =============================================================================
# DimSpec: the 5 primitives
# =============================================================================


@dataclass
class Carry:
    """Output dim inherits sharding from input dim(s).
    Multiple sources -> check compatibility, pick sharded one.
    index_layout: if set, dim mode is built from this Layout (for gather)."""
    sources: list  # [(input_idx, dim_idx), ...]
    index_layout: object = None  # Layout for gather — replaces the dim mode


@dataclass
class Insert:
    """New replicate dim."""
    size: int = 1


@dataclass
class Merge:
    """Multiple dims become one.
    cross_tensor=False: view flatten (concatenate sub-dims from same input).
    cross_tensor=True: cat (extend with source-index mode)."""
    sources: list  # [(input_idx, dim_idx), ...]
    cross_tensor: bool = False


@dataclass
class Split:
    """One dim becomes multiple. This spec represents ONE part of the split."""
    source: tuple  # (input_idx, dim_idx)
    sizes: tuple   # full group_shape
    split_idx: int  # which part of the split this output dim is


@dataclass
class Remove:
    """Input dim is removed. If sharded and reduce_op is set -> Partial.
    If sharded and reduce_op is None -> reject (return None)."""
    source: tuple  # (input_idx, dim_idx)
    reduce_op: Optional[str] = None


# =============================================================================
# Primitive implementations
# =============================================================================


def _apply_carry(spec, inputs):
    """Apply Carry primitive. Returns ((mode_shape, mode_stride), mesh_dims) or None."""
    if spec.index_layout is not None:
        # Gather: build dim mode from index_layout, scaled by source dim stride
        inp_idx, dim_idx = spec.sources[0]
        inp = inputs[inp_idx]
        dim_stride = inp.hier_layout.stride[dim_idx]

        idx_shape = spec.index_layout.shape
        idx_stride = spec.index_layout.stride
        if is_tuple(idx_shape):
            new_shape = tuple((s,) for s in idx_shape)
            new_stride = tuple((st * dim_stride,) if not is_tuple(st) else
                               tuple(x * dim_stride for x in st) for st in idx_stride)
        else:
            first_st = dim_stride[0] if is_tuple(dim_stride) else dim_stride
            new_shape = ((idx_shape,),)
            new_stride = ((idx_stride * first_st,),)
        return (new_shape, new_stride), ()  # gather dim is always replicate

    if len(spec.sources) == 1:
        inp_idx, dim_idx = spec.sources[0]
        inp = inputs[inp_idx]
        return (inp.hier_layout.shape[dim_idx], inp.hier_layout.stride[dim_idx]), inp.mesh_dim_map[dim_idx]

    # Multiple sources: check compatibility, pick the sharded one
    modes = []
    for inp_idx, dim_idx in spec.sources:
        inp = inputs[inp_idx]
        mode_shape = inp.hier_layout.shape[dim_idx]
        mode_stride = inp.hier_layout.stride[dim_idx]
        mesh_dims = inp.mesh_dim_map[dim_idx]
        is_sharded = _mode_has_mesh(mode_shape)
        is_size_1 = product(mode_shape) == 1
        modes.append((mode_shape, mode_stride, mesh_dims, is_sharded, is_size_1))

    # Find the winner: sharded > non-size-1-replicate > size-1
    winner = None
    for i, (ms, mst, md, sharded, size1) in enumerate(modes):
        if size1:
            continue
        if winner is None:
            winner = i
        elif sharded and not modes[winner][3]:
            winner = i  # sharded wins over replicate
        elif sharded and modes[winner][3]:
            # Both sharded: must be compatible (same mesh_dims AND same structure)
            if md != modes[winner][2]:
                return None  # incompatible mesh dims
            if ms != modes[winner][0]:
                return None  # incompatible sharding structure (different mesh sizes)
    if winner is None:
        winner = 0  # all size-1

    # Verify all non-size-1 sharded modes are compatible
    w_ms, w_mst, w_md, w_sharded, _ = modes[winner]
    for i, (ms, mst, md, sharded, size1) in enumerate(modes):
        if i == winner or size1:
            continue
        if sharded and w_sharded:
            if md != w_md or ms != w_ms:
                return None

    return (w_ms, w_mst), w_md


def _apply_insert(spec):
    """Apply Insert primitive."""
    if spec.size == 1:
        return (_SIZE_1_MODE, _SIZE_1_STRIDE), ()
    return (((spec.size, 1),), ((1, 0),)), ()


def _apply_merge(spec, inputs):
    """Apply Merge primitive. Returns ((merged_shape, merged_stride), mesh_dims)."""
    if spec.cross_tensor:
        return _apply_merge_cross_tensor(spec, inputs)
    return _apply_merge_within_tensor(spec, inputs)


def _apply_merge_within_tensor(spec, inputs):
    """View flatten: concatenate sub-dim lists from source dims."""
    merged_shape = ()
    merged_stride = ()
    merged_mesh = ()
    for inp_idx, dim_idx in spec.sources:
        inp = inputs[inp_idx]
        merged_shape = merged_shape + inp.hier_layout.shape[dim_idx]
        merged_stride = merged_stride + inp.hier_layout.stride[dim_idx]
        merged_mesh = merged_mesh + inp.mesh_dim_map[dim_idx]
    return (merged_shape, merged_stride), merged_mesh


def _apply_merge_cross_tensor(spec, inputs):
    """Cat: extend dim with source-index mode.

    Each device holds its local chunks from all sources.
    Output has non-contiguous layout in global index space.
    """
    # All sources must have compatible sharding on this dim
    src_modes = []
    all_mesh = set()
    common_mesh = None
    for inp_idx, dim_idx in spec.sources:
        inp = inputs[inp_idx]
        mode_shape = inp.hier_layout.shape[dim_idx]
        mode_stride = inp.hier_layout.stride[dim_idx]
        mesh_dims = inp.mesh_dim_map[dim_idx]
        src_modes.append((mode_shape, mode_stride, mesh_dims))
        if mesh_dims:
            if common_mesh is None:
                common_mesh = mesh_dims
            elif mesh_dims != common_mesh:
                return None  # incompatible cat dim shardings

    if common_mesh is None:
        common_mesh = ()

    # Check if all sources have the same sharding structure on this dim
    all_same = all(m[0] == src_modes[0][0] for m in src_modes)
    if not all_same and common_mesh:
        return None  # can't cat with different sharding structures

    # For replicate cat dim: just build a bigger replicate dim
    if not common_mesh:
        total_size = sum(product(m[0]) for m in src_modes)
        return (((total_size, 1),), ((1, 0),)), ()

    # Sharded cat dim: each device holds local chunks from all sources.
    # Keep the original sub-dim structure, add a source-index sub-dim.
    base_shape = src_modes[0][0]  # all same
    base_stride = src_modes[0][1]
    num_sources = len(spec.sources)

    # The source offset = global size of one source tensor on this dim
    source_global_size = product(base_shape)

    # Compute element stride for this dim from the first input's layout
    first_inp = inputs[spec.sources[0][0]]
    dim_idx = spec.sources[0][1]
    elem_stride = _get_elem_stride(first_inp, dim_idx)

    # Output sub-dims: original sub-dims + a (num_sources, 1) source-index sub-dim
    # The source-index sub-dim has stride = source_global_size * elem_stride
    # and is local (each device holds all sources), so mesh=1
    source_sub = (num_sources, 1)
    source_sub_stride = (source_global_size * elem_stride, 0)

    out_shape = base_shape + (source_sub,)
    out_stride = base_stride + (source_sub_stride,)

    return (out_shape, out_stride), common_mesh


def _get_elem_stride(sharded, dim_idx):
    """Get the element stride for a dim (stride of the local part)."""
    mode_stride = sharded.hier_layout.stride[dim_idx]
    # First sub-dim's first element stride
    first_sub_st = mode_stride[0]
    if is_tuple(first_sub_st):
        return first_sub_st[0]
    return first_sub_st


def _split_subdim(sub, sub_st, split_point):
    """Split a level-3 sub-dim at a global-size boundary."""
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
    return None


def _apply_split(spec, inputs, all_split_specs):
    """Apply Split primitive for one part of a split group.

    Returns list of ((mode_shape, mode_stride), mesh_dims) for ALL parts
    of this split group (called once for split_idx=0, returns all).
    """
    inp_idx, dim_idx = spec.source
    inp = inputs[inp_idx]
    h_shape = inp.hier_layout.shape[dim_idx]
    h_stride = inp.hier_layout.stride[dim_idx]
    src_mesh = inp.mesh_dim_map[dim_idx]

    src_subdims = list(zip(h_shape, h_stride))
    sub_idx = 0
    pieces = []

    for target_size in spec.sizes:
        acc = 1
        piece = []
        while acc < target_size and sub_idx < len(src_subdims):
            sub_s, sub_st = src_subdims[sub_idx]
            gs = product(sub_s)
            if acc * gs <= target_size:
                acc *= gs
                piece.append((sub_s, sub_st))
                sub_idx += 1
            else:
                needed = target_size // acc
                result = _split_subdim(sub_s, sub_st, needed)
                if result is None:
                    return None
                (left_s, left_st), right = result
                piece.append((left_s, left_st))
                acc *= product(left_s)
                if right is not None:
                    src_subdims[sub_idx] = right
                else:
                    sub_idx += 1
                break
        if acc != target_size:
            return None
        pieces.append(piece)

    # Build output modes and distribute mesh_dims
    results = []
    mesh_dims_list = list(src_mesh)
    mesh_idx = 0
    for piece in pieces:
        out_shape = tuple(s for s, _ in piece)
        out_stride = tuple(st for _, st in piece)
        # Assign mesh dims to this piece based on which sub-dims have mesh
        piece_mesh = []
        for sub_s, sub_st in piece:
            if _has_mesh(sub_s):
                if mesh_idx < len(mesh_dims_list):
                    piece_mesh.append(mesh_dims_list[mesh_idx])
                    mesh_idx += 1
        results.append(((out_shape, out_stride), tuple(piece_mesh)))

    return results


# =============================================================================
# Generic propagation engine
# =============================================================================


def propagate(recipe, removed, inputs):
    """Apply a dim recipe to produce the output ShardedLayout.

    recipe: list of DimSpec (Carry, Insert, Merge, Split) for output dims
    removed: list of Remove specs for contracted/sliced dims
    inputs: list of ShardedLayout
    """
    # 1. Process removed dims
    partial = {}
    for r in removed:
        inp_idx, dim_idx = r.source
        inp = inputs[inp_idx]
        if _mode_has_mesh(inp.hier_layout.shape[dim_idx]):
            if r.reduce_op is not None:
                for md in inp.mesh_dim_map[dim_idx]:
                    partial[md] = r.reduce_op
            else:
                return None  # reject (e.g., slice on sharded dim)

    # Also check mesh_dim_map compatibility for multi-source Carry
    # Same mesh dim on different tensor dims -> incompatible
    all_carry_sources = []
    for spec in recipe:
        if isinstance(spec, Carry) and len(spec.sources) > 1:
            all_carry_sources.append(spec)

    if len(inputs) > 1 and all_carry_sources:
        # Check: same mesh dim must not be assigned to different output dims
        mesh_to_out = {}
        for out_dim, spec in enumerate(recipe):
            if isinstance(spec, Carry):
                for inp_idx, dim_idx in spec.sources:
                    for md in inputs[inp_idx].mesh_dim_map[dim_idx]:
                        if md in mesh_to_out and mesh_to_out[md] != out_dim:
                            return None
                        mesh_to_out[md] = out_dim

    # 2. Build output dims
    out_shapes = []
    out_strides = []
    out_map = {}

    # Pre-compute all Split groups (a Split group shares the same source)
    split_cache = {}

    for out_dim, spec in enumerate(recipe):
        if isinstance(spec, Carry):
            result = _apply_carry(spec, inputs)
            if result is None:
                return None
            (mode_shape, mode_stride), mesh_dims = result
            out_shapes.append(mode_shape)
            out_strides.append(mode_stride)
            out_map[out_dim] = mesh_dims

        elif isinstance(spec, Insert):
            (mode_shape, mode_stride), mesh_dims = _apply_insert(spec)
            out_shapes.append(mode_shape)
            out_strides.append(mode_stride)
            out_map[out_dim] = mesh_dims

        elif isinstance(spec, Merge):
            result = _apply_merge(spec, inputs)
            if result is None:
                return None
            (mode_shape, mode_stride), mesh_dims = result
            out_shapes.append(mode_shape)
            out_strides.append(mode_stride)
            out_map[out_dim] = mesh_dims

        elif isinstance(spec, Split):
            cache_key = (spec.source, spec.sizes)
            if cache_key not in split_cache:
                all_parts = _apply_split(spec, inputs, recipe)
                if all_parts is None:
                    return None
                split_cache[cache_key] = all_parts
            parts = split_cache[cache_key]
            (mode_shape, mode_stride), mesh_dims = parts[spec.split_idx]
            out_shapes.append(mode_shape)
            out_strides.append(mode_stride)
            out_map[out_dim] = mesh_dims

    out_hier = Layout(tuple(out_shapes), tuple(out_strides))
    result = ShardedLayout(out_hier, out_map)
    result.partial = partial

    # Inherit partials from inputs
    for inp in inputs:
        for md, op in inp.partial.items():
            if md in result.partial and result.partial[md] != op:
                return None
            result.partial[md] = op if md not in result.partial else result.partial[md]

    return result


# =============================================================================
# Recipe generators for each operator
# =============================================================================


def _collect_input_dims(op):
    """Recursively collect InputDim indices from a view_groups op."""
    if isinstance(op, InputDim):
        return [op.input_dim]
    elif isinstance(op, Flatten):
        dims = []
        for sub in op.input_dims:
            dims.extend(_collect_input_dims(sub))
        return dims
    elif isinstance(op, ViewSplit):
        return _collect_input_dims(op.input_dim)
    return []


def propagate_transpose(sharded, dim0, dim1):
    ndim = len(sharded.global_shape)
    perm = list(range(ndim))
    perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
    recipe = [Carry([(0, perm[i])]) for i in range(ndim)]
    return propagate(recipe, [], [sharded])


def propagate_permute(sharded, dims):
    recipe = [Carry([(0, dims[i])]) for i in range(len(dims))]
    return propagate(recipe, [], [sharded])


def propagate_unsqueeze(sharded, dim):
    ndim = len(sharded.global_shape)
    recipe = []
    src = 0
    for i in range(ndim + 1):
        if i == dim:
            recipe.append(Insert())
        else:
            recipe.append(Carry([(0, src)]))
            src += 1
    return propagate(recipe, [], [sharded])


def propagate_slice(sharded, dim, index):
    ndim = len(sharded.global_shape)
    recipe = [Carry([(0, i)]) for i in range(ndim) if i != dim]
    removed = [Remove((0, dim), reduce_op=None)]
    return propagate(recipe, removed, [sharded])


def propagate_gather(sharded, dim, index_layout):
    ndim = len(sharded.global_shape)
    # Gathered dim: Remove (reject if sharded) + Carry from index_layout
    recipe = []
    removed = []
    for i in range(ndim):
        if i == dim:
            # Replace this dim with index_layout structure
            removed.append(Remove((0, i), reduce_op=None))
            recipe.append(Carry([(0, i)], index_layout=index_layout))
        else:
            recipe.append(Carry([(0, i)]))
    return propagate(recipe, removed, [sharded])


def propagate_reduction(sharded, reduce_dim, keepdim=False, reduce_op="sum"):
    ndim = len(sharded.global_shape)
    if isinstance(reduce_dim, int):
        reduce_dims = {reduce_dim}
    else:
        reduce_dims = set(reduce_dim)

    recipe = []
    removed = []
    for i in range(ndim):
        if i in reduce_dims:
            removed.append(Remove((0, i), reduce_op=reduce_op))
            if keepdim:
                recipe.append(Insert())
        else:
            recipe.append(Carry([(0, i)]))

    return propagate(recipe, removed, [sharded])


def propagate_broadcast(a, b):
    a_shape = a.global_shape
    b_shape = b.global_shape
    ndim_a = len(a_shape)
    ndim_b = len(b_shape)
    ndim_out = max(ndim_a, ndim_b)

    recipe = []
    for i in range(ndim_out):
        a_idx = i - (ndim_out - ndim_a)
        b_idx = i - (ndim_out - ndim_b)
        has_a = a_idx >= 0
        has_b = b_idx >= 0

        if not has_a:
            recipe.append(Carry([(1, b_idx)]))
        elif not has_b:
            recipe.append(Carry([(0, a_idx)]))
        elif a_shape[a_idx] == 1 and b_shape[b_idx] == 1:
            recipe.append(Carry([(0, a_idx)]))
        elif a_shape[a_idx] == 1:
            recipe.append(Carry([(1, b_idx)]))
        elif b_shape[b_idx] == 1:
            recipe.append(Carry([(0, a_idx)]))
        else:
            recipe.append(Carry([(0, a_idx), (1, b_idx)]))

    return propagate(recipe, [], [a, b])


def propagate_pointwise(all_shardeds):
    if not all_shardeds:
        return None
    result = all_shardeds[0]
    for sharded in all_shardeds[1:]:
        result = propagate_broadcast(result, sharded)
        if result is None:
            return None
    return result


def propagate_view(sharded, new_shape):
    if product(sharded.global_shape) != product(new_shape):
        return None

    old_g = _ensure_tuple(sharded.global_shape)
    new_g = _ensure_tuple(new_shape)
    ops = view_groups(old_g, new_g)

    # Identify split groups
    split_groups = {}
    for out_i, op in enumerate(ops):
        if isinstance(op, ViewSplit):
            key = id(op.input_dim)
            if key not in split_groups:
                split_groups[key] = {
                    "input_op": op.input_dim,
                    "group_shape": op.group_shape,
                    "outputs": [],
                }
            split_groups[key]["outputs"].append((out_i, op.split_id))

    # For Flatten+Split cases (Split whose source is a Flatten of multiple dims),
    # we first merge those dims into a temporary ShardedLayout, then split.
    # Collect which input dims need pre-merging.
    merge_split_groups = {}  # key -> (merged input dims, group_shape, outputs)
    for key, group in split_groups.items():
        input_dims = _collect_input_dims(group["input_op"])
        if len(input_dims) > 1:
            merge_split_groups[key] = (input_dims, group["group_shape"], group["outputs"])

    # Pre-merge: create a temporary ShardedLayout with merged dims
    # We'll reference this temporary input for the Split specs
    temp_input = None
    temp_dim_idx = None
    if merge_split_groups:
        # There should be one merge-split group per view (simplification for common cases)
        # Build a temporary layout by merging the source dims
        for key, (input_dims, group_shape, outputs) in merge_split_groups.items():
            merge_recipe = [Merge([(0, d) for d in input_dims])]
            # Also carry all dims NOT in the merge group
            all_merged = set(input_dims)
            for i in range(len(old_g)):
                if i not in all_merged:
                    merge_recipe.append(Carry([(0, i)]))
            temp_input = propagate(merge_recipe, [], [sharded])
            if temp_input is None:
                return None
            temp_dim_idx = 0  # merged dim is always first in our recipe

    # Build the recipe
    recipe = [None] * len(new_g)
    processed = set()

    for out_i, op in enumerate(ops):
        if out_i in processed:
            continue

        if isinstance(op, InputDim):
            recipe[out_i] = Carry([(0, op.input_dim)])

        elif isinstance(op, Flatten):
            input_dims = _collect_input_dims(op)
            recipe[out_i] = Merge([(0, d) for d in input_dims])

        elif isinstance(op, ViewSplit):
            group = split_groups[id(op.input_dim)]
            input_dims = _collect_input_dims(group["input_op"])
            group_shape = group["group_shape"]
            outputs = sorted(group["outputs"], key=lambda x: x[1])

            if len(input_dims) == 1:
                # Simple split from a single source dim
                source_dim = input_dims[0]
                for oi, sid in outputs:
                    recipe[oi] = Split(
                        source=(0, source_dim),
                        sizes=tuple(group_shape),
                        split_idx=sid,
                    )
                    processed.add(oi)
            else:
                # Flatten+Split: use the pre-merged temporary input
                # The Split references the merged dim in temp_input
                for oi, sid in outputs:
                    recipe[oi] = Split(
                        source=(1, temp_dim_idx),  # input 1 = temp_input, dim 0 = merged
                        sizes=tuple(group_shape),
                        split_idx=sid,
                    )
                    processed.add(oi)

    if any(r is None for r in recipe):
        return None

    if merge_split_groups:
        # We have Split specs referencing temp_input (input idx 1)
        # Also remap any Carry specs: they should reference the original input (idx 0)
        # But some dims in the original are consumed by the merge — those
        # surviving dims are in temp_input at shifted positions.
        # Simpler approach: run the split specs against temp_input,
        # and carry specs against the original input.

        # Rebuild recipe: splits reference temp_input, carries reference original
        # We need to pass both as inputs to propagate
        # But propagate expects consistent input references per spec.
        # Instead, just split the temp_input directly.

        # Build a split-only recipe on temp_input
        split_recipe = []
        carry_from_original = []
        for out_i, spec in enumerate(recipe):
            if isinstance(spec, Split) and spec.source[0] == 1:
                # Remap to input 0 (temp_input will be the sole input)
                split_recipe.append(Split(
                    source=(0, spec.source[1]),
                    sizes=spec.sizes,
                    split_idx=spec.split_idx,
                ))
            elif isinstance(spec, Carry):
                # This dim was carried from original. Find it in temp_input.
                # In the merge recipe, non-merged dims were appended after the merged dim.
                orig_dim = spec.sources[0][1]
                all_merged = set()
                for _, (input_dims, _, _) in merge_split_groups.items():
                    all_merged.update(input_dims)
                # Position in temp_input: merged dim is 0, then non-merged in order
                non_merged = [i for i in range(len(old_g)) if i not in all_merged]
                temp_pos = 1 + non_merged.index(orig_dim)
                split_recipe.append(Carry([(0, temp_pos)]))
            elif isinstance(spec, Merge):
                split_recipe.append(spec)
            else:
                split_recipe.append(spec)

        return propagate(split_recipe, [], [temp_input])

    return propagate(recipe, [], [sharded])


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


def propagate_einsum(equation, sharded_a, sharded_b):
    inputs_labels, output_labels = _parse_einsum(equation)
    categories = _classify_einsum_dims(inputs_labels, output_labels)
    all_labels = list(dict.fromkeys(list(inputs_labels[0]) + list(inputs_labels[1])))

    # Pre-check einsum-specific constraints
    for label in all_labels:
        cat = categories.get(label)
        a_idx = inputs_labels[0].index(label) if label in inputs_labels[0] else None
        b_idx = inputs_labels[1].index(label) if label in inputs_labels[1] else None
        a_sharded = a_idx is not None and _mode_has_mesh(sharded_a.hier_layout.shape[a_idx])
        b_sharded = b_idx is not None and _mode_has_mesh(sharded_b.hier_layout.shape[b_idx])

        if cat == "m" and b_sharded:
            return None
        if cat == "n" and a_sharded:
            return None
        if cat == "batch" and a_sharded != b_sharded:
            return None
        if cat == "contract" and a_sharded != b_sharded:
            return None

    # Build recipe
    recipe = []
    for label in output_labels:
        sources = []
        if label in inputs_labels[0]:
            sources.append((0, inputs_labels[0].index(label)))
        if label in inputs_labels[1]:
            sources.append((1, inputs_labels[1].index(label)))
        recipe.append(Carry(sources))

    removed = []
    for label in all_labels:
        if categories.get(label) == "contract":
            if label in inputs_labels[0]:
                removed.append(Remove((0, inputs_labels[0].index(label)), reduce_op="sum"))

    return propagate(recipe, removed, [sharded_a, sharded_b])


def propagate_cat(shardeds, dim):
    """Cat: cross-tensor merge on dim, carry on other dims."""
    if not shardeds:
        return None
    ndim = len(shardeds[0].global_shape)

    recipe = []
    for d in range(ndim):
        if d == dim:
            recipe.append(Merge(
                [(i, d) for i in range(len(shardeds))],
                cross_tensor=True
            ))
        else:
            recipe.append(Carry([(i, d) for i in range(len(shardeds))]))

    return propagate(recipe, [], shardeds)


# =============================================================================
# Identity ops: clone, contiguous, detach, fill_, zero_, etc.
# All dims Carry from the single input.
# =============================================================================


def propagate_identity(sharded):
    """Identity: all dims carried through (clone, contiguous, detach, etc.)."""
    ndim = len(sharded.global_shape)
    recipe = [Carry([(0, i)]) for i in range(ndim)]
    return propagate(recipe, [], [sharded])


# =============================================================================
# Named matrix ops: mm, bmm, addmm, dot — wrappers over einsum
# =============================================================================


def propagate_mm(sharded_a, sharded_b):
    """mm: (M, K) @ (K, N) -> (M, N)."""
    return propagate_einsum("mk,kn->mn", sharded_a, sharded_b)


def propagate_bmm(sharded_a, sharded_b):
    """bmm: (B, M, K) @ (B, K, N) -> (B, M, N)."""
    return propagate_einsum("bmk,bkn->bmn", sharded_a, sharded_b)


def propagate_addmm(sharded_bias, sharded_a, sharded_b):
    """addmm: bias + A @ B. Propagate mm, then broadcast with bias."""
    mm_result = propagate_mm(sharded_a, sharded_b)
    if mm_result is None:
        return None
    return propagate_broadcast(sharded_bias, mm_result)


def propagate_baddbmm(sharded_self, sharded_a, sharded_b):
    """baddbmm: self + batch(A @ B)."""
    bmm_result = propagate_bmm(sharded_a, sharded_b)
    if bmm_result is None:
        return None
    return propagate_broadcast(sharded_self, bmm_result)


def propagate_dot(sharded_a, sharded_b):
    """dot: (K,) . (K,) -> scalar."""
    return propagate_einsum("k,k->", sharded_a, sharded_b)


def propagate_t(sharded):
    """t: 2D matrix transpose."""
    return propagate_transpose(sharded, 0, 1)


def propagate_movedim(sharded, source, destination):
    """movedim: move dimension from source to destination position."""
    ndim = len(sharded.global_shape)
    dims = list(range(ndim))
    dims.pop(source)
    dims.insert(destination, source)
    return propagate_permute(sharded, dims)


# =============================================================================
# View variants: squeeze, expand, flatten, unflatten, repeat
# =============================================================================


def propagate_squeeze(sharded, dim=None):
    """squeeze: remove size-1 dims. If dim specified, only that dim."""
    shape = sharded.global_shape
    if dim is not None:
        if shape[dim] != 1:
            return propagate_identity(sharded)
        new_shape = shape[:dim] + shape[dim + 1:]
    else:
        new_shape = tuple(s for s in shape if s != 1)
    if not new_shape:
        new_shape = (1,)  # scalar
    return propagate_view(sharded, new_shape)


def propagate_flatten(sharded, start_dim=0, end_dim=-1):
    """flatten: merge dims [start_dim, end_dim] into one."""
    shape = sharded.global_shape
    ndim = len(shape)
    if end_dim < 0:
        end_dim = ndim + end_dim
    new_shape = shape[:start_dim] + (product(shape[start_dim:end_dim + 1]),) + shape[end_dim + 1:]
    return propagate_view(sharded, new_shape)


def propagate_unflatten(sharded, dim, sizes):
    """unflatten: split dim into sizes."""
    shape = sharded.global_shape
    new_shape = shape[:dim] + tuple(sizes) + shape[dim + 1:]
    return propagate_view(sharded, new_shape)


def propagate_expand(sharded, new_shape):
    """expand: broadcast size-1 dims to new_shape. No data copy."""
    old_shape = sharded.global_shape
    ndim_out = len(new_shape)
    ndim_in = len(old_shape)

    # Align from the right (expand can add leading dims)
    recipe = []
    removed = []
    for i in range(ndim_out):
        in_idx = i - (ndim_out - ndim_in)
        if in_idx < 0:
            # Leading dim added by expand
            recipe.append(Insert(size=new_shape[i]))
        elif old_shape[in_idx] == 1 and new_shape[i] != 1:
            # Broadcasting size-1 to new_shape[i]: replace with expanded size
            removed.append(Remove((0, in_idx), reduce_op=None))
            recipe.append(Insert(size=new_shape[i]))
        else:
            recipe.append(Carry([(0, in_idx)]))

    return propagate(recipe, removed, [sharded])


def propagate_repeat(sharded, repeats):
    """repeat: tile tensor by repeats. Dims with repeat > 1 must be replicate."""
    shape = sharded.global_shape
    ndim = len(shape)
    ndim_out = len(repeats)

    # repeat can add leading dims if len(repeats) > ndim
    recipe = []
    removed = []
    for i in range(ndim_out):
        in_idx = i - (ndim_out - ndim)
        if in_idx < 0:
            # Leading dim added by repeat
            recipe.append(Insert(size=repeats[i]))
        elif repeats[i] == 1:
            recipe.append(Carry([(0, in_idx)]))
        else:
            # Repeated dim: must not be sharded (modular, can't express)
            removed.append(Remove((0, in_idx), reduce_op=None))
            recipe.append(Insert(size=shape[in_idx] * repeats[i]))

    return propagate(recipe, removed, [sharded])


# =============================================================================
# Stack, split, unbind — inverses of cat
# =============================================================================


def propagate_stack(shardeds, dim):
    """stack: insert new dim, then cat. All inputs must have same shape."""
    if not shardeds:
        return None
    # Unsqueeze each input at dim, then cat along dim
    unsqueezed = []
    for s in shardeds:
        u = propagate_unsqueeze(s, dim)
        if u is None:
            return None
        unsqueezed.append(u)
    return propagate_cat(unsqueezed, dim)


def propagate_split(sharded, split_sizes, dim):
    """split: inverse of cat. Returns list of ShardedLayouts.

    Each output has the same sharding on all dims, just different size on split dim.
    If dim is sharded, reject (can't split a sharded dim cleanly).
    """
    if _mode_has_mesh(sharded.hier_layout.shape[dim]):
        return None

    ndim = len(sharded.global_shape)
    results = []
    for size in split_sizes:
        # Carry all dims, but replace split dim's mode with the right size
        recipe = []
        for i in range(ndim):
            if i == dim:
                recipe.append(Insert(size=size))
            else:
                recipe.append(Carry([(0, i)]))
        removed = [Remove((0, dim), reduce_op=None)]
        out = propagate(recipe, removed, [sharded])
        if out is None:
            return None
        results.append(out)
    return results


def propagate_unbind(sharded, dim):
    """unbind: remove dim, returning one ShardedLayout per element.

    Equivalent to selecting each index along dim.
    If dim is sharded, reject.
    """
    if _mode_has_mesh(sharded.hier_layout.shape[dim]):
        return None

    size = sharded.global_shape[dim]
    results = []
    for i in range(size):
        out = propagate_slice(sharded, dim, i)
        if out is None:
            return None
        results.append(out)
    return results


# =============================================================================
# Replicate-on-affected-dim ops: flip, roll, sort, topk, argmax, argmin,
# cumsum, cumprod, softmax, layer_norm
#
# Pattern: dims touched by the op must be replicate. Other dims carry through.
# If a touched dim is sharded, reject.
# =============================================================================


def _propagate_replicate_affected(sharded, affected_dims):
    """Generic: carry all dims, reject if any affected dim is sharded."""
    ndim = len(sharded.global_shape)
    if isinstance(affected_dims, int):
        affected_dims = {affected_dims}
    else:
        affected_dims = set(affected_dims)

    for d in affected_dims:
        if _mode_has_mesh(sharded.hier_layout.shape[d]):
            return None

    recipe = [Carry([(0, i)]) for i in range(ndim)]
    return propagate(recipe, [], [sharded])


def propagate_flip(sharded, dims):
    """flip: reverse elements along dims. Affected dims must be replicate."""
    return _propagate_replicate_affected(sharded, dims)


def propagate_roll(sharded, shifts, dims):
    """roll: circular shift along dims. Affected dims must be replicate."""
    return _propagate_replicate_affected(sharded, dims)


def propagate_sort(sharded, dim):
    """sort: sort along dim. Affected dim must be replicate. Returns (values, indices)."""
    result = _propagate_replicate_affected(sharded, dim)
    if result is None:
        return None
    return result, result  # both outputs have same sharding


def propagate_topk(sharded, dim, k):
    """topk: top-k along dim. Affected dim must be replicate."""
    result = _propagate_replicate_affected(sharded, dim)
    if result is None:
        return None
    # Output dim changes size to k — but sharding on other dims is unchanged
    ndim = len(sharded.global_shape)
    recipe = []
    for i in range(ndim):
        if i == dim:
            recipe.append(Insert(size=k))
        else:
            recipe.append(Carry([(0, i)]))
    removed = [Remove((0, dim), reduce_op=None)]
    out = propagate(recipe, removed, [sharded])
    if out is None:
        return None
    return out, out  # values and indices


def propagate_argmax(sharded, dim=None, keepdim=False):
    """argmax: indices of max. Non-linear reduction — must replicate reduced dim."""
    if dim is None:
        # Global argmax — all dims reduced, must all be replicate
        for d in range(len(sharded.global_shape)):
            if _mode_has_mesh(sharded.hier_layout.shape[d]):
                return None
        return ShardedLayout.replicate(())
    return _propagate_replicate_affected(sharded, dim)


def propagate_argmin(sharded, dim=None, keepdim=False):
    """argmin: indices of min. Same as argmax."""
    return propagate_argmax(sharded, dim, keepdim)


def propagate_cumsum(sharded, dim):
    """cumsum: cumulative sum. Affected dim must be replicate."""
    return _propagate_replicate_affected(sharded, dim)


def propagate_cumprod(sharded, dim):
    """cumprod: cumulative product. Affected dim must be replicate."""
    return _propagate_replicate_affected(sharded, dim)


def propagate_softmax(sharded, dim):
    """softmax: normalize along dim. Affected dim must be replicate."""
    return _propagate_replicate_affected(sharded, dim)


def propagate_layer_norm(sharded, normalized_dims):
    """layer_norm: normalize along last N dims. Affected dims must be replicate."""
    ndim = len(sharded.global_shape)
    if isinstance(normalized_dims, int):
        affected = list(range(ndim - normalized_dims, ndim))
    else:
        affected = list(normalized_dims)
    return _propagate_replicate_affected(sharded, affected)


# =============================================================================
# Select (= single-index slice)
# =============================================================================


def propagate_select(sharded, dim, index):
    """select: remove one element along dim. Same as slice with single index."""
    return propagate_slice(sharded, dim, index)


# =============================================================================
# Index_select (= gather with contiguous index)
# =============================================================================


def propagate_index_select(sharded, dim, index_size):
    """index_select: select elements by indices along dim.
    Indices must be replicate. Same as gather with Layout(index_size, 1)."""
    return propagate_gather(sharded, dim, Layout(index_size, 1))


# =============================================================================
# Scatter (conservative: reject if sharded on scatter dim)
# =============================================================================


def propagate_scatter(sharded, dim, src_sharded):
    """scatter: write src into self at index positions.
    Conservative: reject if scatter dim is sharded on either input."""
    if _mode_has_mesh(sharded.hier_layout.shape[dim]):
        return None
    if _mode_has_mesh(src_sharded.hier_layout.shape[dim]):
        return None
    return propagate_broadcast(sharded, src_sharded)


# =============================================================================
# Embedding (specialized gather: rowwise, colwise, or batch)
# =============================================================================


def propagate_embedding(weight_sharded, indices_sharded, mode="rowwise"):
    """embedding: look up rows from weight by indices.

    modes:
    - 'rowwise': weight S(0) on vocab dim -> Partial (each rank has subset of rows)
    - 'colwise': weight S(1) on embed dim -> output sharded on embed dim
    - 'batch': weight replicate, indices sharded on batch dim -> output sharded on batch
    """
    if mode == "colwise":
        # Weight sharded on dim 1 (embedding dim), indices replicate
        # Output: same shape as indices + embed_dim, sharded on last dim
        ndim_idx = len(indices_sharded.global_shape)
        recipe = [Carry([(1, i)]) for i in range(ndim_idx)]
        # Add embedding dim from weight
        recipe.append(Carry([(0, 1)]))
        return propagate(recipe, [], [weight_sharded, indices_sharded])

    elif mode == "batch":
        # Weight replicate, indices sharded on some dim
        # Output follows indices sharding + replicate embed dim
        ndim_idx = len(indices_sharded.global_shape)
        embed_dim = weight_sharded.global_shape[1]
        recipe = [Carry([(1, i)]) for i in range(ndim_idx)]
        recipe.append(Insert(size=embed_dim))
        return propagate(recipe, [], [weight_sharded, indices_sharded])

    else:  # rowwise
        # Weight sharded on dim 0 (vocab), output has Partial
        # Each rank looks up its local rows; non-matching indices get zeros
        ndim_idx = len(indices_sharded.global_shape)
        recipe = [Carry([(1, i)]) for i in range(ndim_idx)]
        recipe.append(Carry([(0, 1)]))
        removed = [Remove((0, 0), reduce_op="sum")]
        return propagate(recipe, removed, [weight_sharded, indices_sharded])


# =============================================================================
# Convolution (carry batch/channel dims, replicate spatial dims if sharded)
# =============================================================================


def propagate_convolution(input_sharded, weight_sharded):
    """convolution: (N, C_in, *spatial) * (C_out, C_in, *kernel) -> (N, C_out, *spatial_out).

    Batch dim (0): carry from input.
    C_out (1): carry from weight dim 0.
    C_in: contracted (like einsum K dim).
    Spatial dims: carry from input (reject if sharded — would need halo exchange).
    """
    ndim = len(input_sharded.global_shape)

    # Check: spatial dims on input must not be sharded
    for d in range(2, ndim):
        if _mode_has_mesh(input_sharded.hier_layout.shape[d]):
            return None

    # Check: C_in on both must match
    a_cin_sharded = _mode_has_mesh(input_sharded.hier_layout.shape[1])
    b_cin_sharded = _mode_has_mesh(weight_sharded.hier_layout.shape[1])
    if a_cin_sharded != b_cin_sharded:
        return None

    recipe = []
    # Batch dim from input
    recipe.append(Carry([(0, 0)]))
    # C_out from weight dim 0
    recipe.append(Carry([(1, 0)]))
    # Spatial dims from input
    for d in range(2, ndim):
        recipe.append(Carry([(0, d)]))

    # C_in is contracted
    removed = []
    if a_cin_sharded and b_cin_sharded:
        removed.append(Remove((0, 1), reduce_op="sum"))

    return propagate(recipe, removed, [input_sharded, weight_sharded])


# =============================================================================
# Dropout / random ops (identity — same sharding, just reject Partial)
# =============================================================================


def propagate_dropout(sharded):
    """dropout: identity sharding. Reject if Partial (need synchronized randomness)."""
    if sharded.partial:
        return None
    return propagate_identity(sharded)

