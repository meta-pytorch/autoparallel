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

import torch
from ._pycute import Layout, is_tuple, logical_divide, product


def _ensure_tuple(x):
    return x if is_tuple(x) else (x,)


def _flatten_subdim(s, st):
    """Flatten a possibly-nested sub-dim into a flat tuple of scalars.

    E.g., (1, (4, 2)) with strides (16, (16, 64)) -> (1, 4, 2) with strides (16, 16, 64).
    Scalar s -> (s, 1) with strides (st, 0).
    Already-flat tuple (local, mesh) -> unchanged.
    """
    if not is_tuple(s):
        return (s, 1), (st, 0)

    flat_s = []
    flat_st = []
    for si, sti in zip(s, st if is_tuple(st) else (st,)):
        if is_tuple(si):
            # Recurse into nested tuple
            inner_s, inner_st = _flatten_subdim(si, sti)
            flat_s.extend(inner_s)
            flat_st.extend(inner_st)
        else:
            flat_s.append(si)
            flat_st.append(sti)
    return tuple(flat_s), tuple(flat_st)


def _to_uniform(layout):
    """Convert a CuTe Layout to uniform 3-level nesting.

    Guarantees:
    - Level 1: each tensor dim is a tuple of level-2 sub-dims
    - Level 2: each sub-dim is a flat tuple (local, mesh...) of scalars
    - No nested tuples at level 3, no scalar sub-dims at level 2

    Detection:
    - If all elements of a mode are tuples -> already level 2 (from view merge),
      normalize each sub-dim individually (flatten nested, wrap scalars)
    - If any element is a scalar -> level 1 (from logical_divide / S(0)S(0)),
      flatten the whole mode and wrap as a single sub-dim
    """
    shape = _ensure_tuple(layout.shape)
    stride = _ensure_tuple(layout.stride)
    new_shape = []
    new_stride = []
    for s, st in zip(shape, stride):
        if not is_tuple(s):
            # Scalar mode: wrap as ((s, 1),)
            new_shape.append(((s, 1),))
            new_stride.append(((st, 0),))
        elif all(is_tuple(elem) for elem in s):
            # Already level 2: normalize each sub-dim individually
            norm_subs = []
            norm_strides = []
            for sub_s, sub_st in zip(s, st):
                flat_s, flat_st = _flatten_subdim(sub_s, sub_st)
                norm_subs.append(flat_s)
                norm_strides.append(flat_st)
            new_shape.append(tuple(norm_subs))
            new_stride.append(tuple(norm_strides))
        else:
            # Level 1 (may have nested tuples from S(0)S(0) or scalars):
            # flatten and wrap as single sub-dim
            flat_s, flat_st = _flatten_subdim(s, st)
            new_shape.append((flat_s,))
            new_stride.append((flat_st,))
    return Layout(tuple(new_shape), tuple(new_stride))


def _empty_map(ndim):
    return {i: () for i in range(ndim)}


def _has_mesh(subdim):
    """Check if a level-3 sub-dim has mesh sharding.

    Sub-dim is a flat tuple (local, mesh...) where all elements are scalars.
    S(0)S(0) has len > 2: (local, mesh0, mesh1).
    """
    return len(subdim) > 1 and product(subdim[1:]) > 1


def _local_size(mode):
    """Get the local (per-device) size of a level-2 mode (tuple of sub-dims).

    Each sub-dim is a flat tuple (local, mesh...). sub[0] is always the scalar local size.
    """
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


class _TensorMeta:
    """Minimal TensorMeta-compatible object for ShardedLayout."""
    __slots__ = ['shape', 'stride', 'dtype']
    def __init__(self, shape):
        self.shape = torch.Size(shape)
        strides = []
        acc = 1
        for s in reversed(shape):
            strides.append(acc)
            acc *= s
        strides.reverse()
        self.stride = tuple(strides)
        self.dtype = torch.float32


class ShardedLayout:
    """
    Sharding = hierarchical CuTe Layout + mesh dim identity map + partial.

    hier_layout: 3-level nesting (tensor dims -> sub-dims -> (local, mesh...))
    mesh_dim_map: {tensor_dim: (mesh_dim_ids...)}
    partial: {mesh_dim: reduce_op}
    """

    def __init__(self, hier_layout, mesh_dim_map=None, partial=None, offset=None):
        self.hier_layout = hier_layout
        ndim = len(_ensure_tuple(hier_layout.shape))
        if mesh_dim_map is None:
            self.mesh_dim_map = _empty_map(ndim)
        else:
            self.mesh_dim_map = {i: mesh_dim_map.get(i, ()) for i in range(ndim)}
        self.partial = partial or {}
        self.offset = offset or {}  # {tensor_dim: int} — per-dim element offset

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
        """Create a ShardedLayout with multiple mesh dims sharding tensor dims.

        Args:
            tensor_shape: tuple of tensor dim sizes
            shard_specs: list of (shard_dim, mesh_size) or (shard_dim, mesh_size, mesh_dim_idx).
                The 2-tuple form assigns mesh_dim_idx sequentially (0, 1, 2, ...).
                The 3-tuple form allows explicit mesh dim assignment, needed when
                the nesting order differs from the mesh dim order (e.g., RTL).
        """
        t_shape = _ensure_tuple(tensor_shape)

        # Normalize to 3-tuples: (shard_dim, mesh_size, mesh_dim_idx)
        normalized = []
        for i, spec in enumerate(shard_specs):
            if len(spec) == 3:
                normalized.append(spec)
            else:
                normalized.append((spec[0], spec[1], i))

        mesh_dim_map = _empty_map(len(t_shape))
        for shard_dim, _, mesh_dim_idx in normalized:
            mesh_dim_map[shard_dim] = mesh_dim_map[shard_dim] + (mesh_dim_idx,)

        dim_counts = {}
        for dim, mesh_size, _ in normalized:
            dim_counts[dim] = dim_counts.get(dim, 0) + 1

        if not any(c > 1 for c in dim_counts.values()):
            local = list(t_shape)
            for dim, mesh_size, _ in normalized:
                assert local[dim] % mesh_size == 0
                local[dim] //= mesh_size
            hier = logical_divide(Layout(t_shape), tuple(local))
            return ShardedLayout(_to_uniform(hier), mesh_dim_map)

        # S(0),S(0) case: direct construction of 3-level hierarchy.
        #
        # Sequential logical_divide can collapse mesh sub-dims due to
        # coalescing inside composition (when strides align). Instead,
        # construct the (local, mesh0, mesh1, ...) sub-dim directly.
        #
        # For shard_specs [(dim, m0), (dim, m1), ...] on a dim of size G:
        #   local = G / product(all m_i)
        #   local_stride = base_stride (from the dim's position in the tensor)
        #   m_k_stride = base_stride * local * product(m_{k+1}, ..., m_last)
        #
        # This gives contiguous-block partitioning: each rank gets a contiguous
        # chunk whose size equals local.

        # Group shard specs by tensor dim, preserving mesh dim indices
        dim_specs = {}
        for dim, mesh_size, mesh_dim_idx in normalized:
            dim_specs.setdefault(dim, []).append((mesh_size, mesh_dim_idx))

        # Build the per-dim sub-dim shapes and strides
        sub_shapes = []
        sub_strides = []
        # Per-dim mesh_dim ordering for mesh_dim_map
        dim_mesh_order = {}

        # Compute base strides for each tensor dim (row-major)
        base_strides = []
        acc = 1
        for d in range(len(t_shape) - 1, -1, -1):
            base_strides.append(acc)
            acc *= t_shape[d]
        base_strides.reverse()

        for d in range(len(t_shape)):
            specs_for_dim = dim_specs.get(d, [])
            if not specs_for_dim:
                # Replicate dim
                sub_shapes.append((t_shape[d], 1))
                sub_strides.append((base_strides[d], 0))
            else:
                mesh_sizes = [ms for ms, _ in specs_for_dim]
                mesh_dim_indices = [mdi for _, mdi in specs_for_dim]
                dim_mesh_order[d] = tuple(mesh_dim_indices)

                total_mesh = 1
                for m in mesh_sizes:
                    total_mesh *= m
                assert t_shape[d] % total_mesh == 0
                local = t_shape[d] // total_mesh
                s = base_strides[d]

                shape_parts = [local]
                stride_parts = [s]
                # Compute mesh strides: m_k_stride = s * local * product(m_{k+1}...)
                for k, mk in enumerate(mesh_sizes):
                    suffix_product = 1
                    for j in range(k + 1, len(mesh_sizes)):
                        suffix_product *= mesh_sizes[j]
                    shape_parts.append(mk)
                    stride_parts.append(s * local * suffix_product)

                sub_shapes.append(tuple(shape_parts))
                sub_strides.append(tuple(stride_parts))

        hier = Layout(
            tuple((sub_s,) for sub_s in sub_shapes),
            tuple((sub_st,) for sub_st in sub_strides),
        )
        return ShardedLayout(hier, mesh_dim_map)

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

    @property
    def placements(self):
        """DTensorSpec-compatible placements for optimizer constraint matching."""
        from torch.distributed.tensor.placement_types import Replicate, Shard, Partial
        # Build one placement per mesh dim
        # Invert mesh_dim_map: mesh_dim -> (tensor_dim, ...)
        max_mesh_dim = -1
        mesh_to_tensor = {}
        for td, mds in self.mesh_dim_map.items():
            for md in mds:
                mesh_to_tensor[md] = td
                if md > max_mesh_dim:
                    max_mesh_dim = md
        result = []
        for md in range(max_mesh_dim + 1):
            if md in self.partial:
                result.append(Partial(self.partial[md]))
            elif md in mesh_to_tensor:
                result.append(Shard(mesh_to_tensor[md]))
            else:
                result.append(Replicate())
        return tuple(result) if result else (Replicate(),)

    @property
    def tensor_meta(self):
        """DTensorSpec-compatible tensor_meta for cost model compatibility."""
        return _TensorMeta(self.global_shape)

    def with_offset(self, offsets):
        """Return a new ShardedLayout with the given offsets added.

        Args:
            offsets: dict {tensor_dim: int_offset} in element space.
        """
        new_offset = dict(self.offset)
        for dim, off in offsets.items():
            new_offset[dim] = new_offset.get(dim, 0) + off
        return ShardedLayout(self.hier_layout, self.mesh_dim_map, self.partial, new_offset)

    def __eq__(self, other):
        if not isinstance(other, ShardedLayout):
            return NotImplemented
        return (self.hier_layout == other.hier_layout
                and self.mesh_dim_map == other.mesh_dim_map
                and self.offset == other.offset)

    def __hash__(self):
        return hash((repr(self.hier_layout),
                      tuple(sorted(self.mesh_dim_map.items())),
                      tuple(sorted(self.offset.items()))))

    def __repr__(self):
        parts = f"ShardedLayout(hier={self.hier_layout}, mesh_map={self.mesh_dim_map}"
        if self.offset:
            parts += f", offset={self.offset}"
        return parts + ")"
