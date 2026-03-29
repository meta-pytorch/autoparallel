"""
ShardedLayout: Sharding as a single hierarchical CuTe Layout.

Always uses uniform 3-level nesting:
- Level 1: tensor dims — always a tuple of level-2 modes
- Level 2: sub-dims (from merge/split history) — always a tuple of level-3 sub-dims
- Level 3: (local, mesh...) — always a tuple starting with scalar local

Examples:
  Replicate (4, 8, 16):    (((4,),), ((8,),), ((16,),))
  S(0)=2 on (4, 8, 16):   (((2, 2),), ((8,),), ((16,),))
  After merge (32, 16):    (((2, 2), (8,)), ((16,),))
"""

from ._pycute import Layout, is_tuple, logical_divide, product


def _to_uniform(layout):
    """Convert a CuTe Layout to uniform 3-level nesting.

    Takes output of logical_divide and wraps every mode into
    tuple-of-tuples-of-tuples structure.
    """
    shape = layout.shape if is_tuple(layout.shape) else (layout.shape,)
    stride = layout.stride if is_tuple(layout.stride) else (layout.stride,)

    new_shape = []
    new_stride = []
    for s, st in zip(shape, stride):
        if is_tuple(s):
            # Could be (local, mesh...) from logical_divide
            # Check for trivial (X, 1) — collapse to (X,)
            if len(s) == 2 and s[1] == 1:
                sub_s = (s[0],)
                sub_st = (st[0] if is_tuple(st) else st,)
            else:
                sub_s = s
                sub_st = st
            # Wrap as single sub-dim in level-2
            new_shape.append((sub_s,))
            new_stride.append((sub_st,))
        else:
            # Flat scalar — wrap as ((X,),)
            new_shape.append(((s,),))
            new_stride.append(((st,),))

    return Layout(tuple(new_shape), tuple(new_stride))


def _has_mesh(subdim):
    """Check if a level-3 sub-dim has mesh sharding. Sub-dim is always a tuple."""
    return len(subdim) > 1 and product(subdim[1:]) > 1


def _local_size(mode):
    """Get the local (per-device) size of a level-2 mode (tuple of sub-dims)."""
    result = 1
    for sub in mode:
        result *= sub[0]  # first element of each sub-dim is local
    return result


def _mode_has_mesh(mode):
    """Check if a level-2 mode has any mesh sharding."""
    return any(_has_mesh(sub) for sub in mode)


def _global_shape(hier_layout):
    """Derive global tensor shape from uniform 3-level layout."""
    shape = hier_layout.shape
    if not is_tuple(shape):
        return (shape,)
    return tuple(product(s) for s in shape)


class ShardedLayout:
    """
    Sharding described by a single hierarchical CuTe Layout.

    Always uses uniform 3-level nesting:
    - Level 1: tuple of level-2 modes (one per tensor dim)
    - Level 2: tuple of level-3 sub-dims (from merge/split history)
    - Level 3: tuple (local, mesh...) — local is always scalar

    global_shape is derived as product per top-level mode.
    """

    def __init__(self, hier_layout):
        self.hier_layout = hier_layout
        self._is_partial = False

    @staticmethod
    def replicate(tensor_shape):
        """All devices hold all elements."""
        t_shape = tensor_shape if is_tuple(tensor_shape) else (tensor_shape,)
        return ShardedLayout(_to_uniform(Layout(t_shape)))

    @staticmethod
    def shard(tensor_shape, shard_dim, mesh_dim_size):
        """Shard tensor_shape[shard_dim] across mesh_dim_size devices."""
        t_shape = tensor_shape if is_tuple(tensor_shape) else (tensor_shape,)
        assert t_shape[shard_dim] % mesh_dim_size == 0

        local = list(t_shape)
        local[shard_dim] //= mesh_dim_size
        hier = logical_divide(Layout(t_shape), tuple(local))
        return ShardedLayout(_to_uniform(hier))

    @staticmethod
    def shard_multi(tensor_shape, shard_specs):
        """Shard with multiple mesh dims.

        shard_specs: list of (shard_dim, mesh_dim_size) per mesh dim.
        For S(0),S(0) (same dim repeated), applies sequential divisions.
        """
        t_shape = tensor_shape if is_tuple(tensor_shape) else (tensor_shape,)

        dim_counts = {}
        for dim, _ in shard_specs:
            dim_counts[dim] = dim_counts.get(dim, 0) + 1

        has_repeated = any(c > 1 for c in dim_counts.values())

        if not has_repeated:
            local = list(t_shape)
            for dim, mesh_size in shard_specs:
                assert local[dim] % mesh_size == 0
                local[dim] //= mesh_size
            hier = logical_divide(Layout(t_shape), tuple(local))
            return ShardedLayout(_to_uniform(hier))

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

        return ShardedLayout(_to_uniform(hier))

    @property
    def global_shape(self):
        """Derived global tensor shape."""
        return _global_shape(self.hier_layout)

    @property
    def tensor_shape(self):
        """Alias for global_shape."""
        return self.global_shape

    @property
    def num_elements(self):
        return product(self.global_shape)

    def is_replicate(self):
        """True if no dim is sharded."""
        shape = self.hier_layout.shape
        if not is_tuple(shape):
            return True
        return not any(_mode_has_mesh(mode) for mode in shape)

    def get_placements(self):
        """Extract per-mesh-dim placements from hierarchical shape."""
        if self.is_replicate():
            return [("replicate", None, None)]

        shape = self.hier_layout.shape
        if not is_tuple(shape):
            return [("replicate", None, None)]

        placements = []
        for i, mode in enumerate(shape):
            if _mode_has_mesh(mode):
                g = product(mode)
                l = _local_size(mode)
                mesh_size = g // l
                placements.append(("shard", i, mesh_size))
        if not placements:
            return [("replicate", None, None)]
        return placements

    def __eq__(self, other):
        if not isinstance(other, ShardedLayout):
            return NotImplemented
        return self.hier_layout == other.hier_layout

    def __hash__(self):
        return hash(self.hier_layout)

    def __repr__(self):
        return f"ShardedLayout(hier={self.hier_layout})"

    def __str__(self):
        placements = self.get_placements()
        parts = []
        for entry in placements:
            if entry[0] == "replicate":
                parts.append("R")
            else:
                parts.append(f"S({entry[1]})")
        return f"[{', '.join(parts)}] on {self.global_shape}"


# Alias for backward compatibility during transition
TiledLayout = ShardedLayout
