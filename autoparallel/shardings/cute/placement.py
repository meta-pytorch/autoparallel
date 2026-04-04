"""
ShardedLayout: Sharding as a hierarchical CuTe Layout + mesh dim map.

Always uses uniform 3-level nesting:
- Level 1: tensor dims — always a tuple of level-2 modes
- Level 2: sub-dims (from merge/split/cat history) — always a tuple of level-3 sub-dims
- Level 3: (local, mesh...) — always a tuple with at least (local, 1) for replicate

mesh_dim_map: always has an entry for every tensor dim.
  - Replicate dim: empty tuple ()
  - Sharded dim: tuple of mesh dim indices, e.g. (0,) or (0, 1) for S(0),S(0)

CuTe is used for: representation (hierarchical layouts), construction (logical_divide).
"""

from ._pycute import Layout, is_tuple, logical_divide, product


def _ensure_tuple(x):
    return x if is_tuple(x) else (x,)


def _to_uniform(layout):
    """Convert a CuTe Layout to uniform 3-level nesting."""
    shape = _ensure_tuple(layout.shape)
    stride = _ensure_tuple(layout.stride)
    new_shape = []
    new_stride = []
    for s, st in zip(shape, stride):
        if is_tuple(s):
            new_shape.append((s,))
            new_stride.append((st,))
        else:
            new_shape.append(((s, 1),))
            new_stride.append(((st, 0),))
    return Layout(tuple(new_shape), tuple(new_stride))


def _empty_map(ndim):
    return {i: () for i in range(ndim)}


def _has_mesh(subdim):
    """Check if a level-3 sub-dim has mesh sharding."""
    return len(subdim) > 1 and product(subdim[1:]) > 1


def _local_size(mode):
    """Get the local (per-device) size of a level-2 mode (tuple of sub-dims)."""
    result = 1
    for sub in mode:
        result *= sub[0]
    return result


def _mode_has_mesh(mode):
    """Check if a level-2 mode has any mesh sharding."""
    return any(_has_mesh(sub) for sub in mode)


def _global_shape(hier_layout):
    shape = hier_layout.shape
    if not is_tuple(shape):
        return (shape,)
    return tuple(product(s) for s in shape)


_SIZE_1_MODE = ((1, 1),)
_SIZE_1_STRIDE = ((0, 0),)


class ShardedLayout:
    """
    Sharding = hierarchical CuTe Layout + mesh dim identity map + partial.

    hier_layout: 3-level nesting (tensor dims -> sub-dims -> (local, mesh...))
    mesh_dim_map: {tensor_dim: (mesh_dim_ids...)}
    partial: {mesh_dim: reduce_op}
    """

    def __init__(self, hier_layout, mesh_dim_map=None, partial=None):
        self.hier_layout = hier_layout
        ndim = len(_ensure_tuple(hier_layout.shape))
        if mesh_dim_map is None:
            self.mesh_dim_map = _empty_map(ndim)
        else:
            self.mesh_dim_map = {i: mesh_dim_map.get(i, ()) for i in range(ndim)}
        self.partial = partial or {}

    @staticmethod
    def replicate(tensor_shape):
        t_shape = _ensure_tuple(tensor_shape)
        return ShardedLayout(_to_uniform(Layout(t_shape)))

    @staticmethod
    def shard(tensor_shape, shard_dim, mesh_dim_size, mesh_dim=0):
        t_shape = _ensure_tuple(tensor_shape)
        assert t_shape[shard_dim] % mesh_dim_size == 0
        local = list(t_shape)
        local[shard_dim] //= mesh_dim_size
        hier = logical_divide(Layout(t_shape), tuple(local))
        return ShardedLayout(_to_uniform(hier), {shard_dim: (mesh_dim,)})

    @staticmethod
    def shard_multi(tensor_shape, shard_specs):
        t_shape = _ensure_tuple(tensor_shape)
        mesh_dim_map = _empty_map(len(t_shape))
        for mesh_dim_idx, (shard_dim, _) in enumerate(shard_specs):
            mesh_dim_map[shard_dim] = mesh_dim_map[shard_dim] + (mesh_dim_idx,)

        dim_counts = {}
        for dim, _ in shard_specs:
            dim_counts[dim] = dim_counts.get(dim, 0) + 1

        if not any(c > 1 for c in dim_counts.values()):
            local = list(t_shape)
            for dim, mesh_size in shard_specs:
                assert local[dim] % mesh_size == 0
                local[dim] //= mesh_size
            hier = logical_divide(Layout(t_shape), tuple(local))
            return ShardedLayout(_to_uniform(hier), mesh_dim_map)

        # S(0),S(0) case: sequential logical_divide
        hier = Layout(t_shape)
        current_local = list(t_shape)
        for dim, mesh_size in shard_specs:
            assert current_local[dim] % mesh_size == 0
            chunk = current_local[dim] // mesh_size
            divisor = tuple(chunk if i == dim else current_local[i]
                            for i in range(len(t_shape)))
            hier = logical_divide(hier, divisor)
            current_local[dim] = chunk
        return ShardedLayout(_to_uniform(hier), mesh_dim_map)

    @property
    def global_shape(self):
        return _global_shape(self.hier_layout)

    @property
    def local_sizes(self):
        return tuple(_local_size(mode) for mode in self.hier_layout.shape)

    def _ld(self):
        return logical_divide(Layout(self.global_shape), self.local_sizes)

    @property
    def tensor_shape(self):
        return self.global_shape

    @property
    def num_elements(self):
        return product(self.global_shape)

    def is_replicate(self):
        return all(len(v) == 0 for v in self.mesh_dim_map.values())

    def get_placements(self):
        if self.is_replicate():
            return [("replicate", None, None)]
        shape = self.hier_layout.shape
        if not is_tuple(shape):
            return [("replicate", None, None)]
        placements = []
        for i, mode in enumerate(shape):
            if self.mesh_dim_map[i]:
                g = product(mode)
                l = _local_size(mode)
                mesh_size = g // l
                placements.append(("shard", i, mesh_size, self.mesh_dim_map[i]))
        return placements or [("replicate", None, None)]

    def __eq__(self, other):
        if not isinstance(other, ShardedLayout):
            return NotImplemented
        return (self.hier_layout == other.hier_layout
                and self.mesh_dim_map == other.mesh_dim_map)

    def __hash__(self):
        return hash((repr(self.hier_layout), tuple(sorted(self.mesh_dim_map.items()))))

    def __repr__(self):
        return f"ShardedLayout(hier={self.hier_layout}, mesh_map={self.mesh_dim_map})"
