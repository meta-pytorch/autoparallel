"""
Generic tensor redistribution using functional collectives.

Takes a local tensor shard + plan_redistribute output and executes the
collectives using torch.distributed._functional_collectives primitives.
All ops are graph-capturable (use torch.ops._c10d_functional internally).

Supports all collective types from plan_redistribute:
  all_gather, all_reduce, reduce_scatter, ppermute, all_to_all
"""

from __future__ import annotations

import torch
import torch.distributed._functional_collectives as funcol

from .placement import ShardedLayout
from .redistribute import plan_redistribute


def _find_shard_dim(layout, mesh_dim):
    """Find which tensor dim is sharded on the given mesh dim."""
    for tensor_dim, mesh_dims in layout.mesh_dim_map.items():
        if mesh_dim in mesh_dims:
            return tensor_dim
    return None


def _all_to_all_cross_dim(tensor, source_dim, target_dim, mesh_size, group):
    """Perform all_to_all to reshard from source_dim to target_dim.

    Local tensor is sharded on source_dim (has src_local elements) and
    full on target_dim (has tgt_full elements). After the all_to_all,
    the result is full on source_dim and sharded on target_dim.

    Strategy: use all_gather on source_dim, then reduce_scatter-style
    local slicing on target_dim. But since we want to avoid the 2x memory
    of all_gather, we implement as all_to_all_single with reshaping:

    1. Split target_dim into (mesh_size, tgt_local)
    2. Permute dims so (mesh_size, source_dim) are dims 0 and 1
    3. Merge dims 0,1 -> all_to_all_single with equal splits on dim 0
    4. Split dim 0 into (mesh_size, src_local), permute back, merge
    """
    ndim = tensor.ndim
    shape = list(tensor.shape)
    src_local = shape[source_dim]
    tgt_local = shape[target_dim] // mesh_size

    # Step 1: split target_dim -> (mesh_size, tgt_local)
    split_shape = shape[:target_dim] + [mesh_size, tgt_local] + shape[target_dim + 1:]
    tensor = tensor.reshape(split_shape)
    # ndim+1 dims now. mesh_size at target_dim, tgt_local at target_dim+1.

    # Adjust source_dim for the extra dim
    adj_src = source_dim if source_dim < target_dim else source_dim + 1
    mesh_pos = target_dim  # position of mesh_size factor

    # Step 2: permute so (mesh_pos, adj_src) become dims (0, 1)
    # Build the rest dims in order
    rest = [i for i in range(ndim + 1) if i != mesh_pos and i != adj_src]
    fwd_perm = [mesh_pos, adj_src] + rest
    tensor = tensor.permute(fwd_perm).contiguous()
    # Shape: (mesh_size, src_local, ...rest...)

    # Step 3: merge dims 0,1 and do all_to_all_single
    rest_shape = list(tensor.shape[2:])
    tensor = tensor.reshape([mesh_size * src_local] + rest_shape)
    tensor = funcol.all_to_all_single(tensor, None, None, group)

    # Step 4: split dim 0 back into (mesh_size, src_local)
    tensor = tensor.reshape([mesh_size, src_local] + rest_shape)

    # Now dim 0 = mesh_size chunks from different ranks (to be merged into source_dim)
    # dim 1 = src_local (per-chunk source elements)
    # rest = same as before

    # Merge dims 0,1 -> src_full = mesh_size * src_local
    src_full = mesh_size * src_local
    tensor = tensor.reshape([src_full] + rest_shape)

    # Now we need to get dims back to the original order.
    # Currently: dim 0 = src_full, then rest dims in the order of `rest`.
    # Target: original dim order with source_dim=src_full, target_dim=tgt_local.

    # rest was [i for i in range(ndim+1) if i != mesh_pos and i != adj_src]
    # In the original (ndim+1)-dim tensor, tgt_local was at target_dim+1.
    # After removing mesh_pos and adj_src, the rest dims map to:
    # - tgt_local (was at target_dim+1 in the split tensor)
    # - all other original dims

    # Build the inverse permutation for the final ndim-dim tensor.
    # The final tensor has ndim dims (same as input).
    # dim source_dim -> src_full (currently at position 0)
    # dim target_dim -> tgt_local (currently at some position in rest+1)
    # other dims -> in their original positions

    # Figure out where each original dim ended up.
    # In the split tensor (ndim+1 dims), the original dims mapped as:
    #   orig dim d -> d if d < target_dim, d+1 if d >= target_dim (except target_dim -> mesh_pos, target_dim+1 -> tgt_local)
    # Actually it's simpler: just build the target shape and use permute.

    # The current dim order is: [merged_src] + rest
    # rest came from fwd_perm[2:], which are the indices in the (ndim+1)-dim tensor
    # excluding mesh_pos and adj_src.

    # Map (ndim+1)-dim indices back to ndim-dim indices:
    # In the original ndim-dim tensor:
    #   - source_dim is the source dim
    #   - target_dim is the target dim
    # In the (ndim+1)-dim tensor after split:
    #   - mesh_pos = target_dim: mesh_size factor (will be merged into src_full)
    #   - adj_src: source_dim data (will be merged into src_full)
    #   - target_dim+1: tgt_local

    # After merge, our dim 0 corresponds to original source_dim.
    # The rest dims correspond to original dims, but we need to map them.

    # Map each rest index (in ndim+1 space) back to original ndim space:
    def to_orig(idx):
        """Map (ndim+1)-dim index to original ndim-dim index."""
        if idx == target_dim + 1:
            return target_dim  # tgt_local -> target_dim
        elif idx < target_dim:
            return idx  # unchanged
        else:  # idx > target_dim + 1 (since mesh_pos=target_dim and target_dim+1 are handled)
            return idx - 1  # shift back

    # Current order: [source_dim] + [to_orig(r) for r in rest]
    current_to_orig = [source_dim] + [to_orig(r) for r in rest]

    # We want the final order to be [0, 1, ..., ndim-1]
    # Build permutation: inv[current_to_orig[i]] = i
    inv_perm = [0] * ndim
    for i, orig_dim in enumerate(current_to_orig):
        inv_perm[orig_dim] = i

    tensor = tensor.permute(inv_perm).contiguous()
    return tensor


def redistribute_tensor(
    local_tensor: torch.Tensor,
    source_layout: ShardedLayout,
    target_layout: ShardedLayout,
    mesh,
) -> torch.Tensor:
    """Redistribute a local tensor shard from source to target layout.

    Uses plan_redistribute to classify collectives, then dispatches to
    torch.distributed._functional_collectives primitives. All ops are
    graph-capturable.

    Args:
        local_tensor: the local shard on this rank
        source_layout: current ShardedLayout
        target_layout: desired ShardedLayout
        mesh: DeviceMesh for collective communication

    Returns:
        New local tensor shard matching target_layout
    """
    collectives = plan_redistribute(source_layout, target_layout)
    if not collectives:
        return local_tensor

    tensor = local_tensor

    for coll_type, mesh_dim, info in collectives:
        if mesh_dim is None:
            group = mesh  # full mesh, all ranks
        else:
            group = (mesh, mesh_dim)

        if coll_type == "all_gather":
            gather_dim = _find_shard_dim(source_layout, mesh_dim)
            assert gather_dim is not None, (
                f"all_gather on mesh_dim {mesh_dim} but no tensor dim is sharded on it"
            )
            tensor = funcol.all_gather_tensor(tensor, gather_dim, group)

        elif coll_type == "all_reduce":
            reduce_op = info["reduce_op"]
            tensor = funcol.all_reduce(tensor, reduce_op, group)

        elif coll_type == "reduce_scatter":
            reduce_op = info["reduce_op"]
            scatter_dim = _find_shard_dim(target_layout, mesh_dim)
            assert scatter_dim is not None, (
                f"reduce_scatter on mesh_dim {mesh_dim} but no target tensor dim is sharded on it"
            )
            tensor = funcol.reduce_scatter_tensor(tensor, reduce_op, scatter_dim, group)

        elif coll_type == "ppermute":
            perm = info["perm"]
            # permute_tensor expects src_dst[src] = dst mapping list
            if mesh_dim is None:
                mesh_size = mesh.size()  # total ranks across all dims
            else:
                mesh_size = mesh.size(mesh_dim)
            src_dst = list(range(mesh_size))  # identity by default
            for src, dst in perm:
                src_dst[src] = dst
            tensor = funcol.permute_tensor(tensor, src_dst, group)

        elif coll_type == "all_to_all":
            source_dim = _find_shard_dim(source_layout, mesh_dim)
            target_dim = _find_shard_dim(target_layout, mesh_dim)
            assert source_dim is not None and target_dim is not None, (
                f"all_to_all on mesh_dim {mesh_dim} but source or target dim not found"
            )
            mesh_size = mesh.size(mesh_dim)

            if source_dim == target_dim:
                tensor = funcol.all_to_all_single(tensor, None, None, group)
            else:
                tensor = _all_to_all_cross_dim(
                    tensor, source_dim, target_dim, mesh_size, group
                )

        else:
            raise ValueError(f"Unknown collective type: {coll_type}")

    if hasattr(tensor, 'wait'):
        tensor = tensor.wait()
    return tensor
