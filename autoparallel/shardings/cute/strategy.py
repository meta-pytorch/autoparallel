"""
Strategy enumeration for sharding optimization.

Enumerates valid ShardedLayouts for tensors on a mesh, and valid
(input_shardings, output_sharding) strategies for ops. Supports
op-specific sharding hints for non-standard layouts (e.g., XorStride).

Search space: standard shardings (Replicate, Shard, S(0)S(0)) enumerated
per mesh dim, combined via Cartesian product. Non-standard layouts emerge
from propagation, not enumeration.
"""

from itertools import product as cartesian_product, permutations

from .placement import ShardedLayout
from .op_registry import get_propagation_rule


# =============================================================================
# Enumerate shardings for a single tensor
# =============================================================================


def enumerate_shardings(tensor_shape, mesh_shape):
    """Enumerate all valid ShardedLayouts for tensor_shape on a mesh.

    Args:
        tensor_shape: tuple of tensor dim sizes, e.g., (16, 32)
        mesh_shape: tuple of mesh dim sizes, e.g., (2, 4) for a 2D mesh

    Returns:
        list of ShardedLayout — all valid shardings including Replicate,
        single-dim Shard, S(0)S(1), and S(0)S(0) with all orderings.

    Strategy: enumerate per-mesh-dim placements, then Cartesian product.
    For S(0)S(0) (same tensor dim on multiple mesh dims), all application
    orderings are enumerated (LTR, RTL) since they produce different layouts.
    """
    ndim = len(tensor_shape)
    n_mesh = len(mesh_shape)

    # Per-mesh-dim placements: None = Replicate, int = Shard(dim=d)
    per_mesh_placements = []
    for m in range(n_mesh):
        mesh_size = mesh_shape[m]
        placements = [None]  # Replicate is always valid
        for d in range(ndim):
            if tensor_shape[d] % mesh_size == 0:
                placements.append(d)
        per_mesh_placements.append(placements)

    # Cartesian product across mesh dims
    results = []
    seen = set()

    def _add(sl):
        h = hash(sl)
        if h not in seen:
            seen.add(h)
            results.append(sl)

    for combo in cartesian_product(*per_mesh_placements):
        # combo is a tuple: one placement per mesh dim
        # e.g., (None, 1) = Replicate on mesh0, Shard(dim=1) on mesh1
        # e.g., (0, 0) = Shard(dim=0) on mesh0 AND mesh1 = S(0)S(0)

        # Collect shard specs: (tensor_dim, mesh_dim, mesh_size)
        shard_specs = []
        for mesh_dim, placement in enumerate(combo):
            if placement is not None:
                shard_specs.append((placement, mesh_dim, mesh_shape[mesh_dim]))

        if not shard_specs:
            # Fully replicate
            _add(ShardedLayout.replicate(tensor_shape))
            continue

        # Group by tensor dim to detect S(0)S(0)
        tensor_dim_to_mesh = {}
        for td, md, ms in shard_specs:
            tensor_dim_to_mesh.setdefault(td, []).append((md, ms))

        # Check if any tensor dim has multiple mesh dims (S(0)S(0))
        has_same_dim = any(len(v) > 1 for v in tensor_dim_to_mesh.values())

        if not has_same_dim:
            # Simple case: each tensor dim sharded on at most one mesh dim
            # Check divisibility
            valid = True
            for td, meshes in tensor_dim_to_mesh.items():
                total_mesh = 1
                for _, ms in meshes:
                    total_mesh *= ms
                if tensor_shape[td] % total_mesh != 0:
                    valid = False
                    break
            if not valid:
                continue

            if len(shard_specs) == 1:
                td, md, ms = shard_specs[0]
                sl = ShardedLayout.shard(tensor_shape, shard_dim=td,
                                         mesh_dim_size=ms, mesh_dim=md)
            else:
                specs = [(td, ms) for td, md, ms in shard_specs]
                sl = ShardedLayout.shard_multi(tensor_shape, specs)

            _add(sl)
        else:
            # S(0)S(0) case: enumerate all orderings of mesh dims on the same tensor dim
            # Build the list of shard specs for shard_multi, trying all permutations
            # of mesh dims that share a tensor dim

            # Separate same-dim groups from single-dim specs
            same_dim_groups = {td: meshes for td, meshes in tensor_dim_to_mesh.items()
                               if len(meshes) > 1}
            single_specs = [(td, md, ms) for td, md, ms in shard_specs
                            if len(tensor_dim_to_mesh[td]) == 1]

            # Check total divisibility for same-dim groups
            valid = True
            for td, meshes in same_dim_groups.items():
                total_mesh = 1
                for _, ms in meshes:
                    total_mesh *= ms
                if tensor_shape[td] % total_mesh != 0:
                    valid = False
                    break
            if not valid:
                continue

            # For each same-dim group, enumerate all orderings
            group_orderings = []
            for td, meshes in sorted(same_dim_groups.items()):
                group_orderings.append(
                    [(td, perm) for perm in permutations(meshes)]
                )

            # Cartesian product of orderings across groups
            for ordering_combo in cartesian_product(*group_orderings):
                specs = []
                # Add single-dim specs (order doesn't matter)
                for td, md, ms in single_specs:
                    specs.append((td, ms, md))
                # Add same-dim specs in the chosen ordering,
                # preserving the actual mesh dim index
                for td, perm in ordering_combo:
                    for md, ms in perm:
                        specs.append((td, ms, md))

                try:
                    sl = ShardedLayout.shard_multi(tensor_shape, specs)
                    _add(sl)
                except (AssertionError, ValueError):
                    continue  # invalid combination

    return results


# =============================================================================
# Enumerate strategies for an op
# =============================================================================


def enumerate_strategies(op_name, input_candidates, *op_args, **op_kwargs):
    """Enumerate valid (input_shardings, output_sharding) pairs for an op.

    Args:
        op_name: ATen op name, e.g., "aten.mm.default"
        input_candidates: list of lists — per-input candidate ShardedLayouts
        *op_args, **op_kwargs: non-tensor op arguments (dim, keepdim, etc.)

    Returns:
        list of (input_shardings_tuple, output_sharding) pairs where
        propagation succeeds.
    """
    fn = get_propagation_rule(op_name)
    if fn is None:
        return []

    results = []
    for combo in cartesian_product(*input_candidates):
        # Build the full argument list: replace tensor positions with ShardedLayouts
        args = list(combo) + list(op_args)
        try:
            output = fn(*args, **op_kwargs)
        except (TypeError, IndexError, AssertionError):
            continue
        if output is not None:
            results.append((combo, output))

    return results


# =============================================================================
# Op-specific sharding hints
# =============================================================================


_OP_SHARDING_HINTS = {}


def register_sharding_hint(op_name, hint_fn):
    """Register additional sharding candidates for an op.

    hint_fn(tensor_shapes, mesh_shape) -> list[list[ShardedLayout]]
    Returns additional per-input candidates to merge with standard enumeration.
    Each inner list corresponds to one input's additional candidates.
    """
    _OP_SHARDING_HINTS.setdefault(op_name, []).append(hint_fn)


def get_sharding_hints(op_name, tensor_shapes, mesh_shape):
    """Get additional candidates from registered hints.

    Returns list of lists — per-input additional candidates.
    If no hints registered, returns empty list.
    """
    all_hints = []
    for hint_fn in _OP_SHARDING_HINTS.get(op_name, []):
        hints = hint_fn(tensor_shapes, mesh_shape)
        if hints:
            all_hints.extend(hints)
    return all_hints
