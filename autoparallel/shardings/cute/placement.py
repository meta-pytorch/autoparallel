"""
TiledLayout: Sharding as composition of tensor layout and shard layout.

This mirrors CuTe's tensor definition T = accessor ∘ layout:
- tensor_layout: tensor_coords -> flat_memory_offset (how tensor is stored)
- shard_layout: (mesh_coords, local_coords) -> element_index (how devices partition)

The full mapping: composition(tensor_layout, shard_layout) gives
(mesh, local) -> memory_offset.

Separation of concerns:
- Reshape/view/transpose change tensor_layout, leave shard_layout unchanged
- Shard/redistribute change shard_layout, leave tensor_layout unchanged
"""

from ._pycute import Layout, coalesce, codomain_divide, flatten, is_tuple, make_layout, product


class TiledLayout:
    """
    Sharding described by two CuTe Layouts.

    Attributes:
        tensor_layout: Layout mapping tensor_coords -> flat_memory_offset.
            Shape = global tensor shape, strides = memory strides.
        shard_layout: Layout mapping (mesh_coords..., local_coords...) -> element_index.
            First modes = mesh shape, remaining = per-device element pattern.
            element_index is a flat index in [0, num_elements).
        mesh_ndim: Number of leading modes that are mesh dimensions.
    """

    def __init__(self, tensor_layout, shard_layout, mesh_ndim):
        self.tensor_layout = tensor_layout
        self.shard_layout = shard_layout
        self.mesh_ndim = mesh_ndim

    @property
    def tensor_shape(self):
        """Global tensor shape."""
        return self.tensor_layout.shape

    @property
    def num_elements(self):
        """Total number of elements in the tensor."""
        return product(self.tensor_shape)

    @property
    def mesh_shape(self):
        """Shape of the mesh (first mesh_ndim modes of shard_layout)."""
        if is_tuple(self.shard_layout.shape):
            return self.shard_layout.shape[: self.mesh_ndim]
        return (self.shard_layout.shape,)

    @property
    def local_shape(self):
        """Shape of local modes (after mesh modes in shard_layout)."""
        if is_tuple(self.shard_layout.shape):
            return self.shard_layout.shape[self.mesh_ndim :]
        return ()

    @property
    def local_size(self):
        """Number of elements per device."""
        ls = self.local_shape
        return product(ls) if ls else 1

    @staticmethod
    def replicate(tensor_shape, mesh_shape):
        """All devices hold all elements."""
        n = product(tensor_shape) if is_tuple(tensor_shape) else tensor_shape
        tensor_layout = Layout(tensor_shape)
        # Mesh modes with stride 0, local mode covers all elements
        mesh_ndim = len(mesh_shape) if is_tuple(mesh_shape) else 1
        shape = mesh_shape + (n,) if is_tuple(mesh_shape) else (mesh_shape, n)
        stride = (0,) * mesh_ndim + (1,)
        shard_layout = Layout(shape, stride)
        return TiledLayout(tensor_layout, shard_layout, mesh_ndim)

    @staticmethod
    def shard(tensor_shape, mesh_shape, shard_dim, mesh_dim=0):
        """Shard tensor_shape[shard_dim] across mesh_shape[mesh_dim]."""
        tensor_layout = Layout(tensor_shape)
        n = product(tensor_shape) if is_tuple(tensor_shape) else tensor_shape
        mesh_ndim = len(mesh_shape) if is_tuple(mesh_shape) else 1
        mesh_shape_t = mesh_shape if is_tuple(mesh_shape) else (mesh_shape,)

        # Compute tensor strides (row-major)
        t_shape = tensor_shape if is_tuple(tensor_shape) else (tensor_shape,)
        t_strides = [1] * len(t_shape)
        for k in range(len(t_shape) - 2, -1, -1):
            t_strides[k] = t_strides[k + 1] * t_shape[k + 1]

        D = mesh_shape_t[mesh_dim]
        dim_size = t_shape[shard_dim]
        assert dim_size % D == 0
        chunk = dim_size // D

        # Build shard layout:
        # Mesh modes: stride 0 for all except mesh_dim, which gets chunk * t_strides[shard_dim]
        mesh_strides = [0] * mesh_ndim
        mesh_strides[mesh_dim] = chunk * t_strides[shard_dim]

        # Local modes: same as tensor dims, but shard_dim has size chunk instead of dim_size
        local_shape = []
        local_strides = []
        for k in range(len(t_shape)):
            sz = chunk if k == shard_dim else t_shape[k]
            local_shape.append(sz)
            local_strides.append(t_strides[k])

        shape = tuple(mesh_shape_t) + tuple(local_shape)
        stride = tuple(mesh_strides) + tuple(local_strides)
        shard_layout = Layout(shape, stride)

        return TiledLayout(tensor_layout, shard_layout, mesh_ndim)

    def is_replicate(self):
        """True if all mesh modes have stride 0."""
        if is_tuple(self.shard_layout.stride):
            mesh_strides = self.shard_layout.stride[: self.mesh_ndim]
            if is_tuple(mesh_strides):
                return all(s == 0 for s in mesh_strides)
            return mesh_strides == 0
        return self.shard_layout.stride == 0

    def get_placements(self):
        """
        Extract per-mesh-dim DTensor-style placements.

        Returns list of (placement_type, dim_or_none) tuples:
            ("replicate", None) or ("shard", dim_index)
        """
        if self.is_replicate():
            mesh_shape_t = self.mesh_shape if is_tuple(self.mesh_shape) else (self.mesh_shape,)
            return [("replicate", None)] * len(mesh_shape_t)

        # Use codomain_divide on local layout vs tensor shape to find
        # which tensor dims are sharded
        t_shape = self.tensor_shape if is_tuple(self.tensor_shape) else (self.tensor_shape,)
        local_layout = Layout(self.local_shape, self.local_stride)
        coverage = codomain_divide(local_layout, t_shape)

        # For each mesh dim, determine which tensor dim it shards
        mesh_shape_t = self.mesh_shape if is_tuple(self.mesh_shape) else (self.mesh_shape,)
        shard_strides = self.shard_layout.stride if is_tuple(self.shard_layout.stride) else (self.shard_layout.stride,)
        mesh_strides = shard_strides[: self.mesh_ndim]

        # Compute tensor row-major strides for comparison
        t_strides = [1] * len(t_shape)
        for k in range(len(t_shape) - 2, -1, -1):
            t_strides[k] = t_strides[k + 1] * t_shape[k + 1]

        placements = []
        for md in range(len(mesh_shape_t)):
            ms = mesh_strides[md] if is_tuple(mesh_strides) else mesh_strides
            if ms == 0:
                placements.append(("replicate", None))
            else:
                # Find which tensor dim this mesh dim shards by matching stride
                shard_dim = None
                for k in range(len(t_shape)):
                    # mesh stride should be chunk * tensor_stride[k]
                    # where chunk = dim_size / mesh_dim_size
                    if t_strides[k] > 0 and ms % t_strides[k] == 0:
                        chunk = ms // t_strides[k]
                        if chunk * mesh_shape_t[md] == t_shape[k]:
                            shard_dim = k
                            break
                placements.append(("shard", shard_dim))

        return placements

    @property
    def local_stride(self):
        if is_tuple(self.shard_layout.stride):
            return self.shard_layout.stride[self.mesh_ndim :]
        return ()

    def __eq__(self, other):
        if not isinstance(other, TiledLayout):
            return NotImplemented
        return (
            self.tensor_layout == other.tensor_layout
            and self.shard_layout == other.shard_layout
            and self.mesh_ndim == other.mesh_ndim
        )

    def __repr__(self):
        return (
            f"TiledLayout(\n"
            f"  tensor={self.tensor_layout},\n"
            f"  shard={self.shard_layout},\n"
            f"  mesh_ndim={self.mesh_ndim}\n"
            f")"
        )

    def __str__(self):
        placements = self.get_placements()
        parts = []
        for p_type, p_dim in placements:
            if p_type == "replicate":
                parts.append("R")
            else:
                parts.append(f"S({p_dim})")
        return f"[{', '.join(parts)}] on {self.tensor_shape}"
