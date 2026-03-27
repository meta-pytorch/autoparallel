"""
TiledLayout: Sharding as tensor_layout + mesh_tiler.

The mesh_tiler is a CuTe Layout that describes which elements one device holds,
using the tensor's strides. shard_layout = logical_divide(tensor_layout, mesh_tiler)
is derived, never stored.

Both tensor_layout and mesh_tiler transform under the same CuTe operations:
- View: tensor_layout changes, mesh_tiler invariant (uses tensor strides)
- Transpose/Permute: reorder modes in both
- Slice: CuTe slice both with same coordinate
"""

from ._pycute import Layout, coalesce, codomain_divide, flatten, is_tuple, logical_divide, product


class TiledLayout:
    """
    Sharding described by tensor_layout + mesh_tiler.

    Attributes:
        tensor_layout: Layout mapping tensor_coords -> element_index.
        mesh_tiler: Layout with same rank as tensor, describing one device's
            local element pattern using tensor strides.
            Shape = local sizes per dim (dim_size for replicate, dim_size/D for shard).
            Stride = tensor's strides.
    """

    def __init__(self, tensor_layout, mesh_tiler):
        self.tensor_layout = tensor_layout
        self.mesh_tiler = mesh_tiler

    @property
    def tensor_shape(self):
        return self.tensor_layout.shape

    @property
    def num_elements(self):
        return product(self.tensor_shape)

    @property
    def local_size(self):
        return self.mesh_tiler.size()

    @property
    def shard_layout(self):
        """Derived: logical_divide(tensor_layout, mesh_tiler)."""
        return logical_divide(self.tensor_layout, self.mesh_tiler)

    @staticmethod
    def replicate(tensor_shape):
        """All devices hold all elements. Tiler = full tensor."""
        tensor_layout = Layout(tensor_shape)
        mesh_tiler = Layout(tensor_shape)  # local = full tensor
        return TiledLayout(tensor_layout, mesh_tiler)

    @staticmethod
    def shard(tensor_shape, shard_dim, mesh_dim_size):
        """Shard tensor_shape[shard_dim] across mesh_dim_size devices."""
        tensor_layout = Layout(tensor_shape)
        t_shape = tensor_shape if is_tuple(tensor_shape) else (tensor_shape,)

        dim_size = t_shape[shard_dim]
        assert dim_size % mesh_dim_size == 0
        chunk = dim_size // mesh_dim_size

        # Compute tensor strides (row-major)
        t_strides = [1] * len(t_shape)
        for k in range(len(t_shape) - 2, -1, -1):
            t_strides[k] = t_strides[k + 1] * t_shape[k + 1]

        # Tiler: same strides as tensor, but shard_dim has local size
        tiler_shape = tuple(chunk if k == shard_dim else t_shape[k] for k in range(len(t_shape)))
        mesh_tiler = Layout(tiler_shape, tuple(t_strides))

        return TiledLayout(tensor_layout, mesh_tiler)

    def is_replicate(self):
        """True if tiler covers all elements (local_size == num_elements)."""
        return self.local_size == self.num_elements

    def get_placements(self):
        """
        Extract which dims are sharded by decomposing tiler through tensor shape.

        Uses codomain_divide when tiler rank differs from tensor rank (after view).
        """
        if self.is_replicate():
            return [("replicate", None, None)]

        t_shape = self.tensor_shape if is_tuple(self.tensor_shape) else (self.tensor_shape,)
        m_shape = self.mesh_tiler.shape if is_tuple(self.mesh_tiler.shape) else (self.mesh_tiler.shape,)

        # If ranks match, direct comparison
        if len(t_shape) == len(m_shape):
            placements = []
            for k in range(len(t_shape)):
                t_s = t_shape[k] if not is_tuple(t_shape[k]) else product(t_shape[k])
                m_s = m_shape[k] if not is_tuple(m_shape[k]) else product(m_shape[k])
                if m_s < t_s:
                    placements.append(("shard", k, t_s // m_s))
            return placements if placements else [("replicate", None, None)]

        # Ranks differ (after view): use codomain_divide
        coverage = codomain_divide(self.mesh_tiler, t_shape)
        placements = []
        for k in range(len(t_shape)):
            if coverage[k] < t_shape[k]:
                placements.append(("shard", k, t_shape[k] // coverage[k]))
        return placements if placements else [("replicate", None, None)]

    def __eq__(self, other):
        if not isinstance(other, TiledLayout):
            return NotImplemented
        return (
            self.tensor_layout == other.tensor_layout
            and self.mesh_tiler == other.mesh_tiler
        )

    def __hash__(self):
        return hash((self.tensor_layout, self.mesh_tiler))

    def __repr__(self):
        return (
            f"TiledLayout(\n"
            f"  tensor={self.tensor_layout},\n"
            f"  tiler={self.mesh_tiler}\n"
            f")"
        )

    def __str__(self):
        placements = self.get_placements()
        parts = []
        for entry in placements:
            if entry[0] == "replicate":
                parts.append("R")
            else:
                parts.append(f"S({entry[1]})")
        return f"[{', '.join(parts)}] on {self.tensor_shape}"
