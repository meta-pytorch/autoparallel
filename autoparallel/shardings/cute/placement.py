"""
ShardedLayout: Sharding as a single hierarchical CuTe Layout.

The hierarchical layout is produced by logical_divide(Layout(tensor_shape), local_sizes).
Each sharded dim has shape (local, mesh_size), replicated dims are flat.
Convention: (local, mesh) — local is tile (stride 1, contiguous blocks).

The hier_layout is VIEW-INVARIANT (element-space strides).
View only changes global_shape metadata.
"""

from ._pycute import Layout, codomain_divide, flatten, is_tuple, logical_divide, product


def _clean_hier(layout):
    """Collapse trivial (X, 1):(s, 0) modes to flat X:s.

    logical_divide always produces (tile, rest) pairs. For replicated dims
    (divided by full size), rest=1. This cleanup makes replicate dims flat,
    so is_tuple(mode) ↔ sharded.
    """
    shape = layout.shape if is_tuple(layout.shape) else (layout.shape,)
    stride = layout.stride if is_tuple(layout.stride) else (layout.stride,)
    new_shape = []
    new_stride = []
    for s, st in zip(shape, stride):
        if is_tuple(s) and len(s) == 2 and s[1] == 1:
            new_shape.append(s[0])
            new_stride.append(st[0] if is_tuple(st) else st)
        else:
            new_shape.append(s)
            new_stride.append(st)
    if len(new_shape) == 1:
        return Layout(new_shape[0], new_stride[0])
    return Layout(tuple(new_shape), tuple(new_stride))


def _is_level2(mode):
    """Level-2 = tuple where ALL elements are tuples."""
    if not is_tuple(mode):
        return False
    return all(is_tuple(m) for m in mode)


def _has_mesh(mode):
    """Check if a mode has any mesh sharding."""
    if not is_tuple(mode):
        return False
    if _is_level2(mode):
        return any(_has_mesh(sub) for sub in mode)
    return product(mode[1:]) > 1


def _local_size(mode):
    """Get the local (per-device) size of a mode."""
    if not is_tuple(mode):
        return mode
    if _is_level2(mode):
        result = 1
        for sub in mode:
            result *= _local_size(sub)
        return result
    return mode[0]


class ShardedLayout:
    """
    Sharding described by a single hierarchical CuTe Layout.

    Attributes:
        hier_layout: Layout with hierarchical shape (local, mesh) per dim.
            Produced by logical_divide(Layout(tensor_shape), local_sizes).
            Uses element-space strides — invariant under view.
        global_shape: Current tensor shape (changes on view/transpose/slice).
    """

    def __init__(self, hier_layout, global_shape):
        self.hier_layout = hier_layout
        self.global_shape = global_shape
        self._is_partial = False

    @staticmethod
    def replicate(tensor_shape):
        """All devices hold all elements."""
        return ShardedLayout(Layout(tensor_shape), tensor_shape)

    @staticmethod
    def shard(tensor_shape, shard_dim, mesh_dim_size):
        """Shard tensor_shape[shard_dim] across mesh_dim_size devices."""
        t_shape = tensor_shape if is_tuple(tensor_shape) else (tensor_shape,)
        dim_size = t_shape[shard_dim]
        assert dim_size % mesh_dim_size == 0

        local = list(t_shape)
        local[shard_dim] //= mesh_dim_size
        hier = _clean_hier(logical_divide(Layout(t_shape), tuple(local)))
        return ShardedLayout(hier, t_shape)

    @staticmethod
    def shard_multi(tensor_shape, shard_specs):
        """Shard with multiple mesh dims.

        shard_specs: list of (shard_dim, mesh_dim_size) per mesh dim.
        For S(0),S(0) (same dim repeated), applies sequential divisions.
        """
        t_shape = tensor_shape if is_tuple(tensor_shape) else (tensor_shape,)

        # Check if any dim is sharded by multiple mesh dims
        dim_counts = {}
        for dim, _ in shard_specs:
            dim_counts[dim] = dim_counts.get(dim, 0) + 1

        has_repeated = any(c > 1 for c in dim_counts.values())

        if not has_repeated:
            # Simple case: each dim sharded at most once
            local = list(t_shape)
            for dim, mesh_size in shard_specs:
                assert local[dim] % mesh_size == 0
                local[dim] //= mesh_size
            hier = _clean_hier(logical_divide(Layout(t_shape), tuple(local)))
            return ShardedLayout(hier, t_shape)

        # S(0),S(0) case: sequential logical_divide
        hier = Layout(t_shape)
        current_local = list(t_shape)
        for dim, mesh_size in shard_specs:
            assert current_local[dim] % mesh_size == 0
            current_local[dim] //= mesh_size
            divisor = tuple(current_local[i] if i == dim else
                            (current_local[i] if dim_counts.get(i, 0) > 1 else t_shape[i])
                            for i in range(len(t_shape)))
            # Actually, sequential divide is simpler: divide the current hier layout
            local_for_this = list(t_shape)
            local_for_this[dim] = current_local[dim]
            # For repeated dims, we need to divide the rest of the previous division
            pass

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

        return ShardedLayout(_clean_hier(hier), t_shape)

    @property
    def tensor_shape(self):
        """Current tensor shape (for compatibility with existing API)."""
        return self.global_shape

    @property
    def num_elements(self):
        return product(self.global_shape)

    def is_replicate(self):
        """True if no dim is sharded (all mesh parts are 1)."""
        shape = self.hier_layout.shape
        if not is_tuple(shape):
            return True
        return not any(_has_mesh(s) for s in shape)

    def _hier_mode_shape(self, dim):
        """Get the hierarchical shape of a specific dim."""
        shape = self.hier_layout.shape
        if not is_tuple(shape):
            return shape if dim == 0 else None
        if dim < len(shape):
            return shape[dim]
        return None

    def get_placements(self):
        """Extract per-mesh-dim placements from hierarchical shape.

        Each dim with shape (local, mesh) → Shard.
        Flat dims → Replicate.
        """
        if self.is_replicate():
            return [("replicate", None, None)]

        shape = self.hier_layout.shape
        if not is_tuple(shape):
            shape = (shape,)

        # If hier rank matches global rank, direct read
        if len(shape) == len(self.global_shape if is_tuple(self.global_shape) else (self.global_shape,)):
            placements = []
            for i, s in enumerate(shape):
                if _has_mesh(s):
                    # Compute total mesh product for this dim
                    g = product(s)
                    l = _local_size(s)
                    mesh_size = g // l
                    placements.append(("shard", i, mesh_size))
            if not placements:
                return [("replicate", None, None)]
            return placements

        # Rank mismatch (after view): use codomain_divide
        g_shape = self.global_shape if is_tuple(self.global_shape) else (self.global_shape,)
        coverage = codomain_divide(self.hier_layout, g_shape)
        placements = []
        for k in range(len(g_shape)):
            g_s = g_shape[k]
            c_s = coverage.get(k, g_s)
            if c_s < g_s:
                mesh_size = g_s // c_s
                placements.append(("shard", k, mesh_size))
        if not placements:
            return [("replicate", None, None)]
        return placements

    def __eq__(self, other):
        if not isinstance(other, ShardedLayout):
            return NotImplemented
        return (
            self.hier_layout == other.hier_layout
            and self.global_shape == other.global_shape
        )

    def __hash__(self):
        return hash((self.hier_layout, self.global_shape))

    def __repr__(self):
        return f"ShardedLayout(hier={self.hier_layout}, shape={self.global_shape})"

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
