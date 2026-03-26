"""
CutePlacement: A unified sharding placement using CuTe layouts.

This module provides a generalized placement type that represents both
Shard and Replicate (and more complex patterns like _StridedShard) using
CuTe layouts. The key advantage is that CuTe layouts can represent
hierarchical, strided sharding patterns that arise from view/reshape ops
but cannot be expressed with flat Shard(dim).

Examples:
    Replicate on mesh dim of size 4:
        CutePlacement(dim=None, layout=Layout(4, 0))

    Shard(dim=1) for tensor dim size 16, mesh dim size 4:
        CutePlacement(dim=1, layout=Layout(4, 4))
        (device d starts at offset d*4, holding 4 elements)

    After flatten (B=2, S=8) -> (B*S=16), Shard(1) on S by 4 devices becomes:
        CutePlacement(dim=0, layout=Layout((2, 2), (8, 1)))
        (device d holds a strided pattern in the flat dim)
"""

from ._pycute import Layout, coalesce, flatten, is_tuple, product


class CutePlacement:
    """
    Unified sharding placement using a CuTe Layout.

    Attributes:
        dim: Tensor dimension being sharded, or None for replicate.
        layout: Rank-1 CuTe Layout with size = mesh_dim_size.
            Maps device_idx -> starting offset within the tensor dimension.
            Stride 0 means replicate (all devices hold the same data).
    """

    def __init__(self, dim, layout):
        self.dim = dim
        self.layout = layout

    @staticmethod
    def replicate(mesh_dim_size):
        """Create a replicate placement (stride-0 layout)."""
        return CutePlacement(dim=None, layout=Layout(mesh_dim_size, 0))

    @staticmethod
    def shard(dim, tensor_dim_size, mesh_dim_size):
        """Create a standard shard placement (contiguous chunks)."""
        assert tensor_dim_size % mesh_dim_size == 0, (
            f"Tensor dim size {tensor_dim_size} must be divisible by "
            f"mesh dim size {mesh_dim_size}"
        )
        chunk_size = tensor_dim_size // mesh_dim_size
        return CutePlacement(dim=dim, layout=Layout(mesh_dim_size, chunk_size))

    @staticmethod
    def from_placement(placement, tensor_dim_size, mesh_dim_size):
        """
        Convert a DTensor Placement to CutePlacement.

        Args:
            placement: A Shard(dim) or Replicate() instance.
            tensor_dim_size: Size of the tensor dimension being sharded.
            mesh_dim_size: Size of the mesh dimension.
        """
        # Check by attribute/method to avoid importing torch placement types
        if hasattr(placement, "dim") and hasattr(placement, "is_shard"):
            if placement.is_shard():
                return CutePlacement.shard(
                    placement.dim, tensor_dim_size, mesh_dim_size
                )
        # Replicate or unknown -> replicate
        return CutePlacement.replicate(mesh_dim_size)

    def is_replicate(self):
        """True if all devices hold the same data (stride-0)."""
        if self.dim is None:
            return True
        c = coalesce(self.layout)
        all_strides = flatten(c.stride)
        return all(s == 0 for s in all_strides)

    def is_shard(self):
        """True if this is equivalent to a simple Shard(dim).

        A simple shard means each device holds a contiguous chunk.
        The layout maps device_idx -> starting offset, with stride = chunk_size.
        This is Layout(D, chunk_size) where chunk_size > 0.

        We also recognize layouts that coalesce to this form, e.g.,
        (2, 8):(8, 1) from a flatten of (B, S) sharded on B means
        each device holds B/D * S contiguous elements — this is still
        a simple shard with chunk_size = B/D * S.
        """
        if self.dim is None:
            return False
        c = coalesce(self.layout)
        # Simple case: rank-1 layout with positive stride
        if not is_tuple(c.shape) and not is_tuple(c.stride):
            return c.stride > 0
        # Check if the layout describes contiguous elements per device.
        # A layout is "contiguous per device" if its right_inverse has size
        # equal to the layout's cosize, i.e., it covers all elements in [0, cosize).
        # Simpler check: the layout, when treated as a function, maps
        # range(size) to a contiguous range. We check this by verifying
        # that the coalesced layout of the "local" part is just N:1.
        # For (2,8):(8,1) -> coalesced is (2,8):(8,1) which is NOT N:1.
        # But (8,2):(1,8) -> coalesced is 16:1 which IS N:1.
        # The issue is ordering — we need to check if the image is contiguous
        # regardless of order.
        # Practical check: is the layout a permutation of N:1?
        # i.e., does it produce all values 0..N-1 for inputs 0..N-1?
        sz = c.size()
        if sz <= 64:  # Only check for small layouts
            vals = sorted(c(i) for i in range(sz))
            if vals == list(range(sz)):
                return True
        return False

    def to_placement(self):
        """
        Convert back to a DTensor Placement if possible.

        Returns:
            - Shard(dim) if this is a simple contiguous shard
            - Replicate() if this is replicate
            - None if this is a complex CuTe pattern that can't be
              represented as a standard placement
        """
        # Import lazily to avoid hard dependency on torch
        from torch.distributed.tensor.placement_types import Replicate, Shard

        if self.is_replicate():
            return Replicate()
        if self.is_shard():
            return Shard(self.dim)
        return None

    @property
    def mesh_dim_size(self):
        """Number of devices along this mesh dimension."""
        return product(self.layout.shape)

    def local_size(self, tensor_dim_size):
        """Size of the local shard for each device."""
        if self.is_replicate():
            return tensor_dim_size
        return tensor_dim_size // self.mesh_dim_size

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
            c = coalesce(self.layout)
            return f"CutePlacement.shard(dim={self.dim}, chunk={c.stride})"
        return f"CutePlacement(dim={self.dim}, layout={self.layout})"

    def __str__(self):
        if self.is_replicate():
            return "R"
        if self.is_shard():
            c = coalesce(self.layout)
            return f"S({self.dim})"
        return f"CS({self.dim}, {self.layout})"
