# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import Placement, Shard

from .collectives import all_gather, axis_size, local_map, reduce_scatter

_DP_REPLICATE_NAMES = {"dp_replicate", "ddp"}
_DP_SHARD_NAMES = {"dp", "dp_shard", "fsdp", "data", "data_parallel"}
_CP_NAMES = {"cp", "context", "context_parallel"}
_TP_NAMES = {"tp", "tensor", "tensor_parallel"}


@dataclass(frozen=True)
class ContextParallelPlacements:
    """Placement groups for context-parallel attention."""

    qkv: tuple[Placement, ...]
    out: tuple[Placement, ...]

    @property
    def in_placements(self) -> tuple[tuple[Placement, ...], ...]:
        return (self.qkv, self.qkv, self.qkv)

    @property
    def out_placements(self) -> tuple[tuple[Placement, ...], ...]:
        return (self.out,)


def _mesh_dim_names(mesh: DeviceMesh) -> tuple[str, ...]:
    names = getattr(mesh, "mesh_dim_names", None)
    if names is not None and all(name is not None for name in names):
        return tuple(names)

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


def _cp_axis_name(mesh: DeviceMesh) -> str | None:
    cp_names = [
        name for name in _mesh_dim_names(mesh) if _mesh_axis_role(name) == "cp"
    ]
    if not cp_names:
        return None
    if len(cp_names) > 1:
        raise ValueError("Only one context-parallel mesh axis is supported.")
    return cp_names[0]


def context_parallel_attention_placements(
    mesh: DeviceMesh,
    *,
    batch_dim: int = 0,
    seq_dim: int = 1,
    head_dim: int = 2,
) -> ContextParallelPlacements:
    """Return Q/K/V-sharded attention placements for the given mesh.

    Args:
        mesh: Device mesh with named DP, CP, and/or TP dimensions.
        batch_dim: Batch dimension in Q/K/V tensors.
        seq_dim: Sequence dimension in Q/K/V tensors.
        head_dim: Head dimension in Q/K/V tensors.
    """

    qkv: list[Placement] = []
    for name in _mesh_dim_names(mesh):
        role = _mesh_axis_role(name)
        if role in ("dp_replicate", "dp_shard"):
            qkv.append(Shard(batch_dim))
        elif role == "cp":
            qkv.append(Shard(seq_dim))
        elif role == "tp":
            qkv.append(Shard(head_dim))

    qkv_t = tuple(qkv)
    return ContextParallelPlacements(qkv=qkv_t, out=qkv_t)


def context_parallel_local_map(
    fn: Callable | None = None,
    *,
    mesh: DeviceMesh,
    batch_dim: int = 0,
    seq_dim: int = 1,
    head_dim: int = 2,
    redistribute_inputs: bool = True,
):
    """Wrap an attention callable for context-parallel execution.

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
    cp_axis = _cp_axis_name(mesh)

    def wrap(inner_fn: Callable):
        def _with_kv_all_gather(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
            if cp_axis is not None:
                k = all_gather(k, seq_dim, cp_axis)
                v = all_gather(v, seq_dim, cp_axis)
            return inner_fn(q, k, v)

        return local_map(
            _with_kv_all_gather,
            out_placements=placements.out_placements,
            in_placements=placements.in_placements,
            redistribute_inputs=redistribute_inputs,
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
    """Build an SDPA callable for context-parallel attention.

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

    cp_axis = _cp_axis_name(mesh)
    if cp_axis is not None and dropout_p != 0.0:
        raise ValueError("Context-parallel SDPA does not support dropout.")

    @context_parallel_local_map(
        mesh=mesh,
        batch_dim=batch_dim,
        seq_dim=seq_dim,
        head_dim=head_dim,
    )
    def _context_parallel_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        kwargs = {
            "dropout_p": dropout_p,
            "enable_gqa": enable_gqa,
        }
        if scale is not None:
            kwargs["scale"] = scale

        if is_causal and cp_axis is not None and axis_size(cp_axis) > 1:
            cp_size = axis_size(cp_axis)
            q = all_gather(q, seq_dim, cp_axis)
            kwargs["is_causal"] = True
            out = F.scaled_dot_product_attention(q, k, v, **kwargs)
            return reduce_scatter(out / cp_size, seq_dim, cp_axis)

        kwargs["is_causal"] = is_causal
        return F.scaled_dot_product_attention(q, k, v, **kwargs)

    return _context_parallel_sdpa
