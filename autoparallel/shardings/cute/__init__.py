"""
CuTe-based sharding placement, propagation, and redistribution.

- placement: ShardedLayout = hierarchical CuTe Layout (local, mesh) per dim
- propagation: 5-primitive recipe engine (Carry, Insert, Remove, Merge, Split)
- redistribute: per-mesh-dim GPU stride classification for collective planning
"""

from .placement import ShardedLayout

from .propagation import (
    Carry,
    Insert,
    Merge,
    Remove,
    Split,
    propagate,
    propagate_addmm,
    propagate_argmax,
    propagate_baddbmm,
    propagate_bmm,
    propagate_broadcast,
    propagate_cat,
    propagate_convolution,
    propagate_dot,
    propagate_dropout,
    propagate_einsum,
    propagate_embedding,
    propagate_expand,
    propagate_flatten,
    propagate_gather,
    propagate_identity,
    propagate_index_select,
    propagate_layer_norm,
    propagate_mm,
    propagate_movedim,
    propagate_permute,
    propagate_pointwise,
    propagate_reduction,
    propagate_repeat,
    propagate_replicate_affected,
    propagate_scatter,
    propagate_slice,
    propagate_sort,
    propagate_split,
    propagate_squeeze,
    propagate_stack,
    propagate_t,
    propagate_topk,
    propagate_transpose,
    propagate_unbind,
    propagate_unflatten,
    propagate_unsqueeze,
    propagate_view,
)

from .redistribute import (
    plan_redistribute,
    plan_redistribute_detailed,
)

from .op_registry import (
    OP_REGISTRY,
    get_propagation_rule,
)
