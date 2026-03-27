"""
TiledLayout: Sharding as tensor_layout + mesh_tilers.

Each mesh_tiler describes one mesh dimension's contribution to sharding.
shard_layout = sequential logical_divide(tensor_layout, tiler_0, tiler_1, ...).

Both tensor_layout and mesh_tilers transform under CuTe operations:
- View: tensor_layout changes, mesh_tilers invariant
- Transpose/Permute: reorder modes in all
- Slice: CuTe slice all with same coordinate
"""

from ._pycute import Layout, coalesce, codomain_divide, flatten, is_tuple, logical_divide, product


class TiledLayout:
    """
    Sharding described by tensor_layout + tuple of mesh_tilers.

    Attributes:
        tensor_layout: Layout mapping tensor_coords -> element_index.
        mesh_tilers: Tuple of Layouts, one per mesh dim. Each describes
            that mesh dim's local element pattern using tensor strides.
            Applied sequentially via logical_divide.
    """

    def __init__(self, tensor_layout, mesh_tilers):
        self.tensor_layout = tensor_layout
        if isinstance(mesh_tilers, Layout):
            mesh_tilers = (mesh_tilers,)
        self.mesh_tilers = tuple(mesh_tilers)

    @property
    def tensor_shape(self):
        return self.tensor_layout.shape

    @property
    def num_elements(self):
        return product(self.tensor_shape)

    @property
    def mesh_ndim(self):
        return len(self.mesh_tilers)

    @property
    def local_size(self):
        """Number of elements per device."""
        size = self.num_elements
        for tiler in self.mesh_tilers:
            # Each tiler divides: rest = size / tiler.size()
            # local after all = product of tiler sizes / num_elements...
            # Actually: final local = last tiler's size
            pass
        # Simpler: local = product of all tiler sizes / product of intermediate rests
        # Or just: local = smallest tiler's size for the final level
        # Actually: local = num_elements / product of mesh dim sizes
        local = self.num_elements
        for tiler in self.mesh_tilers:
            local = tiler.size()
        return local

    @property
    def shard_layout(self):
        """Derived: sequential logical_divide with all mesh_tilers."""
        result = self.tensor_layout
        for tiler in self.mesh_tilers:
            result = logical_divide(result, tiler)
        return result

    @staticmethod
    def replicate(tensor_shape):
        """All devices hold all elements. No mesh tilers."""
        return TiledLayout(Layout(tensor_shape), ())

    @staticmethod
    def shard(tensor_shape, shard_dim, mesh_dim_size):
        """Shard tensor_shape[shard_dim] across mesh_dim_size devices."""
        t_shape = tensor_shape if is_tuple(tensor_shape) else (tensor_shape,)

        dim_size = t_shape[shard_dim]
        assert dim_size % mesh_dim_size == 0
        chunk = dim_size // mesh_dim_size

        # Tensor strides (row-major)
        t_strides = [1] * len(t_shape)
        for k in range(len(t_shape) - 2, -1, -1):
            t_strides[k] = t_strides[k + 1] * t_shape[k + 1]

        # Tiler: same as tensor but shard_dim has local size
        tiler_shape = tuple(chunk if k == shard_dim else t_shape[k] for k in range(len(t_shape)))
        mesh_tiler = Layout(tiler_shape, tuple(t_strides))

        return TiledLayout(Layout(tensor_shape), (mesh_tiler,))

    @staticmethod
    def shard_multi(tensor_shape, shard_specs):
        """Shard with multiple mesh dims.

        shard_specs: list of (shard_dim, mesh_dim_size) per mesh dim.
        """
        t_shape = tensor_shape if is_tuple(tensor_shape) else (tensor_shape,)

        # Tensor strides (row-major)
        t_strides = [1] * len(t_shape)
        for k in range(len(t_shape) - 2, -1, -1):
            t_strides[k] = t_strides[k + 1] * t_shape[k + 1]

        # Build tilers sequentially — each operates on the previous tiler's local
        mesh_tilers = []
        current_local = list(t_shape)  # track local sizes per dim

        for shard_dim, mesh_dim_size in shard_specs:
            assert current_local[shard_dim] % mesh_dim_size == 0
            chunk = current_local[shard_dim] // mesh_dim_size

            tiler_shape = tuple(chunk if k == shard_dim else current_local[k] for k in range(len(t_shape)))
            mesh_tilers.append(Layout(tiler_shape, tuple(t_strides)))

            current_local[shard_dim] = chunk

        return TiledLayout(Layout(tensor_shape), tuple(mesh_tilers))

    def is_replicate(self):
        """True if no mesh tilers (all devices hold everything)."""
        return len(self.mesh_tilers) == 0

    def get_placements(self):
        """Extract per-mesh-dim placements.

        Each tiler is compared against the tensor shape (or previous tiler's
        local shape) to find which dim it shards.
        """
        if self.is_replicate():
            return [("replicate", None, None)]

        t_shape = self.tensor_shape if is_tuple(self.tensor_shape) else (self.tensor_shape,)

        placements = []
        current_shape = list(t_shape)

        for tiler in self.mesh_tilers:
            m_shape = tiler.shape if is_tuple(tiler.shape) else (tiler.shape,)

            # If tiler rank matches current shape rank, direct comparison
            if len(m_shape) == len(current_shape):
                found = False
                for k in range(len(current_shape)):
                    t_s = current_shape[k] if not is_tuple(current_shape[k]) else product(current_shape[k])
                    m_s = m_shape[k] if not is_tuple(m_shape[k]) else product(m_shape[k])
                    if m_s < t_s:
                        mesh_size = t_s // m_s
                        placements.append(("shard", k, mesh_size))
                        current_shape[k] = m_s
                        found = True
                        break
                if not found:
                    placements.append(("replicate", None, None))
            else:
                # Rank mismatch (after view): use codomain_divide
                coverage = codomain_divide(tiler, tuple(current_shape))
                found = False
                for k in range(len(current_shape)):
                    if coverage[k] < current_shape[k]:
                        mesh_size = current_shape[k] // coverage[k]
                        placements.append(("shard", k, mesh_size))
                        current_shape[k] = coverage[k]
                        found = True
                        break
                if not found:
                    placements.append(("replicate", None, None))

        return placements

    def __eq__(self, other):
        if not isinstance(other, TiledLayout):
            return NotImplemented
        return (
            self.tensor_layout == other.tensor_layout
            and self.mesh_tilers == other.mesh_tilers
        )

    def __hash__(self):
        return hash((self.tensor_layout, self.mesh_tilers))

    def __repr__(self):
        tilers_str = ", ".join(str(t) for t in self.mesh_tilers)
        return f"TiledLayout(tensor={self.tensor_layout}, tilers=[{tilers_str}])"

    def __str__(self):
        placements = self.get_placements()
        parts = []
        for entry in placements:
            if entry[0] == "replicate":
                parts.append("R")
            else:
                parts.append(f"S({entry[1]})")
        return f"[{', '.join(parts)}] on {self.tensor_shape}"
