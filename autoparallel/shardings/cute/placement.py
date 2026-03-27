"""
CutePlacement: A unified sharding placement using CuTe layouts.

Convention:
    The layout is a rank-2 CuTe Layout mapping (device_idx, local_idx...) -> flat_offset
    within the tensor dimension. Mode 0 is the device mode (size = mesh_dim_size),
    remaining modes describe the local element pattern per device.

    This is analogous to CuTe's thread-value (TV) layouts, where device = thread
    and local elements = values.

Examples:
    Replicate on mesh dim of size 4:
        Layout((4, 16), (0, 1))
        All 4 devices hold the same 16 contiguous elements (stride-0 device mode).

    Shard(dim=1) for tensor dim size 16, mesh dim size 4:
        Layout((4, 4), (4, 1))
        Device d starts at offset d*4, holding 4 contiguous elements.

    After flatten (B=4, S=8) -> (B*S=32), Shard(1) on S by D=2 devices:
        Layout((2, 4, 4), (4, 8, 1))
        Device d holds 16 elements in a strided pattern:
        4 batches (stride 8) x 4 seq elements (stride 1), offset by d*4.
"""

from ._pycute import Layout, coalesce, flatten, is_tuple, make_layout, product


class CutePlacement:
    """
    Unified sharding placement using a rank-2 CuTe Layout.

    Attributes:
        dim: Tensor dimension being sharded, or None for replicate.
        layout: CuTe Layout mapping (device_idx, local_idx...) -> flat_offset.
            Mode 0 = device mode (size = mesh_dim_size).
            Remaining modes = local element pattern.
    """

    def __init__(self, dim, layout):
        self.dim = dim
        self.layout = layout

    @staticmethod
    def replicate(mesh_dim_size, tensor_dim_size=1):
        """Create a replicate placement (stride-0 device mode)."""
        return CutePlacement(
            dim=None,
            layout=Layout((mesh_dim_size, tensor_dim_size), (0, 1)),
        )

    @staticmethod
    def shard(dim, tensor_dim_size, mesh_dim_size):
        """Create a standard shard placement (contiguous chunks)."""
        assert tensor_dim_size % mesh_dim_size == 0, (
            f"Tensor dim size {tensor_dim_size} must be divisible by "
            f"mesh dim size {mesh_dim_size}"
        )
        chunk_size = tensor_dim_size // mesh_dim_size
        return CutePlacement(
            dim=dim,
            layout=Layout((mesh_dim_size, chunk_size), (chunk_size, 1)),
        )

    @staticmethod
    def from_placement(placement, tensor_dim_size, mesh_dim_size):
        """Convert a DTensor Placement to CutePlacement."""
        if hasattr(placement, "dim") and hasattr(placement, "is_shard"):
            if placement.is_shard():
                return CutePlacement.shard(
                    placement.dim, tensor_dim_size, mesh_dim_size
                )
        return CutePlacement.replicate(mesh_dim_size, tensor_dim_size)

    @property
    def mesh_dim_size(self):
        """Number of devices along this mesh dimension."""
        if is_tuple(self.layout.shape):
            return self.layout.shape[0]
        return self.layout.shape

    @property
    def local_shape(self):
        """Shape of the local modes (excluding device mode)."""
        if is_tuple(self.layout.shape):
            return self.layout.shape[1:]
        return ()

    @property
    def local_stride(self):
        """Stride of the local modes (excluding device mode)."""
        if is_tuple(self.layout.stride):
            return self.layout.stride[1:]
        return ()

    @property
    def local_size(self):
        """Number of elements each device holds."""
        return product(self.local_shape) if self.local_shape else 1

    @property
    def device_stride(self):
        """Stride of the device mode."""
        if is_tuple(self.layout.stride):
            return self.layout.stride[0]
        return self.layout.stride

    def is_replicate(self):
        """True if all devices hold the same data (stride-0 device mode)."""
        if self.dim is None:
            return True
        return self.device_stride == 0

    def is_shard(self):
        """True if this is equivalent to a simple Shard(dim).

        A simple shard has a contiguous local layout (coalesces to N:1)
        and device stride = chunk size.
        """
        if self.dim is None:
            return False
        if self.device_stride == 0:
            return False
        # Check that local modes are contiguous (coalesce to N:1)
        local = Layout(self.local_shape, self.local_stride)
        c = coalesce(local)
        if is_tuple(c.shape) or is_tuple(c.stride):
            return False
        return c.stride == 1

    def to_placement(self):
        """Convert back to a DTensor Placement if possible."""
        from torch.distributed.tensor.placement_types import Replicate, Shard

        if self.is_replicate():
            return Replicate()
        if self.is_shard():
            return Shard(self.dim)
        return None

    def __eq__(self, other):
        if not isinstance(other, CutePlacement):
            return NotImplemented
        return self.dim == other.dim and self.layout == other.layout

    def __hash__(self):
        return hash((self.dim, self.layout))

    def __repr__(self):
        if self.is_replicate():
            return f"CutePlacement.replicate({self.mesh_dim_size})"
        if self.is_shard():
            return f"CutePlacement.shard(dim={self.dim}, chunk={self.local_size})"
        return f"CutePlacement(dim={self.dim}, layout={self.layout})"

    def __str__(self):
        if self.is_replicate():
            return "R"
        if self.is_shard():
            return f"S({self.dim})"
        return f"CS({self.dim}, {self.layout})"
