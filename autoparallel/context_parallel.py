# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
)

from .collectives import local_map

_DP_REPLICATE_NAMES = {"dp_replicate", "ddp"}
_DP_SHARD_NAMES = {"dp", "dp_shard", "fsdp", "data", "data_parallel"}
_CP_NAMES = {"cp", "context", "context_parallel"}
_TP_NAMES = {"tp", "tensor", "tensor_parallel"}


@dataclass(frozen=True)
class ContextParallelPlacements:
    """Placement groups for context-parallel attention."""

    q: tuple[Placement, ...]
    kv: tuple[Placement, ...]
    out: tuple[Placement, ...]
    q_grad: tuple[Placement, ...]
    kv_grad: tuple[Placement, ...]

    @property
    def in_placements(self) -> tuple[tuple[Placement, ...], ...]:
        return (self.q, self.kv, self.kv)

    @property
    def out_placements(self) -> tuple[tuple[Placement, ...], ...]:
        return (self.out,)

    @property
    def in_grad_placements(self) -> tuple[tuple[Placement, ...], ...]:
        return (self.q_grad, self.kv_grad, self.kv_grad)


def _mesh_dim_names(mesh: DeviceMesh) -> tuple[str, ...]:
    names = getattr(mesh, "mesh_dim_names", None)
    if names is not None and all(name is not None for name in names):
        return tuple(names)

    # 2-D meshes are ambiguous: [dp_shard, cp] and [dp_shard, tp] both
    # matter for CP. Require names there so K/V are not silently misplaced.
    if mesh.ndim == 3:
        return ("dp_shard", "cp", "tp")
    if mesh.ndim == 4:
        return ("dp_replicate", "dp_shard", "cp", "tp")
    raise ValueError(
        "context_parallel_attention_placements requires mesh_dim_names for "
        "1-D/2-D meshes. Use names such as ('dp_shard', 'cp') or "
        "('dp_shard', 'tp')."
    )


def _mesh_axis_role(name: str) -> str:
    normalized = name.lower()
    if normalized in _DP_REPLICATE_NAMES:
        return "dp_replicate"
    if normalized in _DP_SHARD_NAMES:
        return "dp_shard"
    if normalized in _CP_NAMES:
        return "cp"
    if normalized in _TP_NAMES:
        return "tp"
    raise ValueError(
        f"Unsupported mesh axis {name!r} for context parallel attention. "
        "Expected axes like dp_shard/dp, cp, tp, and optionally dp_replicate."
    )


def context_parallel_attention_placements(
    mesh: DeviceMesh,
    *,
    batch_dim: int = 0,
    seq_dim: int = 1,
    head_dim: int = 2,
) -> ContextParallelPlacements:
    """Return attention placements for the given mesh.

    Args:
        mesh: Device mesh with named DP, CP, and/or TP dimensions.
        batch_dim: Batch dimension in Q/K/V tensors.
        seq_dim: Sequence dimension in Q/K/V tensors.
        head_dim: Head dimension in Q/K/V tensors.
    """

    q: list[Placement] = []
    kv: list[Placement] = []
    q_grad: list[Placement] = []
    kv_grad: list[Placement] = []

    for name in _mesh_dim_names(mesh):
        role = _mesh_axis_role(name)
        if role in ("dp_replicate", "dp_shard"):
            q_p = kv_p = q_grad_p = kv_grad_p = Shard(batch_dim)
        elif role == "cp":
            q_p = q_grad_p = Shard(seq_dim)
            kv_p = Replicate()
            kv_grad_p = Partial()
        elif role == "tp":
            q_p = kv_p = q_grad_p = kv_grad_p = Shard(head_dim)

        q.append(q_p)
        kv.append(kv_p)
        q_grad.append(q_grad_p)
        kv_grad.append(kv_grad_p)

    q_t = tuple(q)
    return ContextParallelPlacements(
        q=q_t,
        kv=tuple(kv),
        out=q_t,
        q_grad=tuple(q_grad),
        kv_grad=tuple(kv_grad),
    )


def context_parallel_local_map(
    fn: Callable | None = None,
    *,
    mesh: DeviceMesh,
    batch_dim: int = 0,
    seq_dim: int = 1,
    head_dim: int = 2,
    redistribute_inputs: bool = True,
):
    """Wrap an attention callable with context-parallel placements.

    Args:
        fn: Callable whose first three arguments are Q, K, and V tensors.
        mesh: Device mesh with named DP, CP, and/or TP dimensions.
        batch_dim: Batch dimension in Q/K/V tensors.
        seq_dim: Sequence dimension in Q/K/V tensors.
        head_dim: Head dimension in Q/K/V tensors.
        redistribute_inputs: Whether to redistribute inputs to the requested
            placements.
    """

    placements = context_parallel_attention_placements(
        mesh, batch_dim=batch_dim, seq_dim=seq_dim, head_dim=head_dim
    )

    def wrap(inner_fn: Callable):
        return local_map(
            inner_fn,
            out_placements=placements.out_placements,
            in_placements=placements.in_placements,
            redistribute_inputs=redistribute_inputs,
            in_grad_placements=placements.in_grad_placements,
            device_mesh=mesh,
        )

    if fn is None:
        return wrap
    return wrap(fn)


def make_context_parallel_sdpa(
    mesh: DeviceMesh,
    *,
    batch_dim: int = 0,
    seq_dim: int = 2,
    head_dim: int = 1,
    is_causal: bool = True,
    dropout_p: float = 0.0,
    scale: float | None = None,
    enable_gqa: bool = False,
):
    """Build a context-parallel SDPA callable.

    Args:
        mesh: Device mesh with named DP, CP, and/or TP dimensions.
        batch_dim: Batch dimension in Q/K/V tensors.
        seq_dim: Sequence dimension in Q/K/V tensors.
        head_dim: Head dimension in Q/K/V tensors.
        is_causal: Whether SDPA applies a causal mask.
        dropout_p: Dropout probability for SDPA.
        scale: Optional SDPA scale value.
        enable_gqa: Whether SDPA uses grouped query attention.
    """

    @context_parallel_local_map(
        mesh=mesh,
        batch_dim=batch_dim,
        seq_dim=seq_dim,
        head_dim=head_dim,
    )
    def _context_parallel_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        kwargs = {
            "is_causal": is_causal,
            "dropout_p": dropout_p,
            "enable_gqa": enable_gqa,
        }
        if scale is not None:
            kwargs["scale"] = scale
        return F.scaled_dot_product_attention(q, k, v, **kwargs)

    return _context_parallel_sdpa
