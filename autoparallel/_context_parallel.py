# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Callable, Literal

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.distributed.tensor.placement_types import Placement, Replicate, Shard
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from .collectives import local_map

_DP_REPLICATE_NAMES = {"dp_replicate", "ddp"}
_DP_SHARD_NAMES = {
    "dp",
    "dp_shard",
    "dp_shard_mod_ep",
    "dp_shard_in_ep",
    "fsdp",
    "data",
    "data_parallel",
}
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


@dataclass(frozen=True)
class _ContextParallelBlockMask:
    kv_num_blocks: torch.Tensor
    kv_indices: torch.Tensor
    full_kv_num_blocks: torch.Tensor | None
    full_kv_indices: torch.Tensor | None
    q_offsets: torch.Tensor
    block_size: int | tuple[int, int]
    mask_mod: Callable | None
    local_seq_len: int
    kv_seq_len: int

    @classmethod
    def from_block_mask(
        cls,
        block_mask: BlockMask,
        *,
        cp_size: int,
        device: torch.device,
    ) -> "_ContextParallelBlockMask":
        q_seq_len, kv_seq_len = block_mask.seq_lengths
        q_block_size = block_mask.BLOCK_SIZE[0]
        if q_seq_len % cp_size != 0:
            raise ValueError(
                "Context-parallel FlexAttention requires the BlockMask query "
                "length to be divisible by the CP degree."
            )

        local_seq_len = q_seq_len // cp_size
        if local_seq_len % q_block_size != 0:
            raise ValueError(
                "Context-parallel FlexAttention requires each CP shard length "
                "to be divisible by the BlockMask query block size."
            )

        q_offsets = torch.arange(cp_size, device=device, dtype=torch.int64)
        q_offsets = q_offsets * local_seq_len
        return cls(
            kv_num_blocks=block_mask.kv_num_blocks,
            kv_indices=block_mask.kv_indices,
            full_kv_num_blocks=block_mask.full_kv_num_blocks,
            full_kv_indices=block_mask.full_kv_indices,
            q_offsets=q_offsets,
            block_size=block_mask.BLOCK_SIZE,
            mask_mod=block_mask.mask_mod,
            local_seq_len=local_seq_len,
            kv_seq_len=kv_seq_len,
        )

    def args(self) -> tuple[object, ...]:
        return (
            self.kv_num_blocks,
            self.kv_indices,
            self.full_kv_num_blocks,
            self.full_kv_indices,
            self.q_offsets,
        )

    def _tensor_placements(
        self, tensor: torch.Tensor, mesh: DeviceMesh
    ) -> tuple[Placement, ...]:
        placements: list[Placement] = []
        for mesh_dim, name in enumerate(_mesh_dim_names(mesh)):
            role = _mesh_axis_role(name)
            mesh_size = mesh.size(mesh_dim)
            if role in ("dp_replicate", "dp_shard"):
                dim = 0
            elif role == "tp":
                dim = 1
            elif role == "cp":
                dim = 2
            else:
                placements.append(Replicate())
                continue

            if tensor.size(dim) == 1:
                placements.append(Replicate())
            elif tensor.size(dim) % mesh_size == 0:
                placements.append(Shard(dim))
            else:
                raise ValueError(
                    "Context-parallel FlexAttention requires explicit BlockMask "
                    "batch, head, and query-block dimensions to be divisible by "
                    "their matching mesh dimensions."
                )
        return tuple(placements)

    def placements(self, mesh: DeviceMesh, cp_axis: str | None) -> tuple[object, ...]:
        offset_placements = [Replicate() for _ in range(mesh.ndim)]
        if cp_axis is not None:
            cp_dim = _mesh_dim_names(mesh).index(cp_axis)
            offset_placements[cp_dim] = Shard(0)
        return (
            self._tensor_placements(self.kv_num_blocks, mesh),
            self._tensor_placements(self.kv_indices, mesh),
            (
                self._tensor_placements(self.full_kv_num_blocks, mesh)
                if self.full_kv_num_blocks is not None
                else None
            ),
            (
                self._tensor_placements(self.full_kv_indices, mesh)
                if self.full_kv_indices is not None
                else None
            ),
            tuple(offset_placements),
        )

    def rebuild(
        self,
        kv_num_blocks: torch.Tensor,
        kv_indices: torch.Tensor,
        full_kv_num_blocks: torch.Tensor | None,
        full_kv_indices: torch.Tensor | None,
        q_offsets: torch.Tensor,
    ) -> BlockMask:
        q_offset = q_offsets.reshape(())
        mask_mod = self.mask_mod
        shifted_mask_mod: Callable | None
        if mask_mod is not None:

            def shifted_mask_mod(b, h, q_idx, kv_idx):
                return mask_mod(b, h, q_idx + q_offset, kv_idx)

        else:
            shifted_mask_mod = None

        return BlockMask.from_kv_blocks(
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            BLOCK_SIZE=self.block_size,
            mask_mod=shifted_mask_mod,
            seq_lengths=(self.local_seq_len, self.kv_seq_len),
        )


def _mesh_dim_names(mesh: DeviceMesh) -> tuple[str, ...]:
    names = getattr(mesh, "mesh_dim_names", None)
    if names is not None and all(name is not None for name in names):
        return tuple(names)

    if mesh.ndim == 3:
        return ("dp_shard", "cp", "tp")
    if mesh.ndim == 4:
        return ("dp_replicate", "dp_shard", "cp", "tp")
    if mesh.ndim == 5:
        return ("dp_replicate", "dp_shard_mod_ep", "dp_shard_in_ep", "cp", "tp")
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
    cp_names = [name for name in _mesh_dim_names(mesh) if _mesh_axis_role(name) == "cp"]
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


def make_context_parallel(
    mesh: DeviceMesh,
    *,
    kind: Literal["sdpa", "flex_attention"] = "sdpa",
    batch_dim: int = 0,
    seq_dim: int = 2,
    head_dim: int = 1,
    is_causal: bool = True,
    dropout_p: float = 0.0,
    scale: float | None = None,
    enable_gqa: bool = False,
    score_mod: Callable | None = None,
    block_mask: BlockMask | None = None,
    kernel_options: dict | None = None,
):
    """Build a callable for context-parallel attention.

    Args:
        mesh: Device mesh with named DP, CP, and/or TP dimensions.
        kind: Attention implementation to wrap.
        batch_dim: Batch dimension in Q/K/V tensors.
        seq_dim: Sequence dimension in Q/K/V tensors.
        head_dim: Head dimension in Q/K/V tensors.
        is_causal: Whether SDPA applies a causal mask.
        dropout_p: Dropout probability for SDPA.
        scale: Optional attention scale value.
        enable_gqa: Whether attention uses grouped query attention.
        score_mod: Optional FlexAttention score modifier.
        block_mask: Optional FlexAttention block mask.
        kernel_options: Optional FlexAttention kernel options.
    """

    if kind == "sdpa":
        return _make_context_parallel_sdpa(
            mesh,
            batch_dim=batch_dim,
            seq_dim=seq_dim,
            head_dim=head_dim,
            is_causal=is_causal,
            dropout_p=dropout_p,
            scale=scale,
            enable_gqa=enable_gqa,
        )
    if kind == "flex_attention":
        return _make_context_parallel_flex_attention(
            mesh,
            batch_dim=batch_dim,
            seq_dim=seq_dim,
            head_dim=head_dim,
            scale=scale,
            enable_gqa=enable_gqa,
            score_mod=score_mod,
            block_mask=block_mask,
            kernel_options=kernel_options,
        )
    raise ValueError(f"Unsupported context-parallel attention kind: {kind!r}")


def _make_context_parallel_sdpa(
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
    cp_axis = _cp_axis_name(mesh)
    if cp_axis is not None and dropout_p != 0.0:
        raise ValueError("Context-parallel SDPA does not support dropout.")

    placements = context_parallel_attention_placements(
        mesh, batch_dim=batch_dim, seq_dim=seq_dim, head_dim=head_dim
    )

    cp_allgather = None
    cp_group_name = None
    cp_size = None
    rank_placements = None
    if cp_axis is not None:
        from torch.distributed.tensor.experimental._context_parallel._attention import (
            flex_cp_allgather,
        )

        cp_mesh = mesh[cp_axis]
        cp_allgather = flex_cp_allgather
        cp_group_name = dist._get_process_group_name(cp_mesh.get_group())
        cp_dim = _mesh_dim_names(mesh).index(cp_axis)
        cp_size = mesh.size(cp_dim)
        rank_placements = tuple(
            Shard(0) if mesh_dim == cp_dim else Replicate()
            for mesh_dim in range(mesh.ndim)
        )

    def cp_sdpa(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *rank_args: torch.Tensor,
    ):
        if cp_allgather is not None:
            k, v = cp_allgather(k.contiguous(), v.contiguous(), seq_dim, cp_group_name)

        attn_mask = None
        sdpa_is_causal = is_causal
        if rank_args:
            rank_index = rank_args[0].reshape(())
            local_q_len = q.size(seq_dim)
            q_positions = (
                torch.arange(local_q_len, device=q.device) + rank_index * local_q_len
            )
            kv_positions = torch.arange(k.size(seq_dim), device=q.device)
            attn_mask = q_positions[:, None] >= kv_positions[None, :]
            sdpa_is_causal = False

        kwargs = {
            "attn_mask": attn_mask,
            "dropout_p": dropout_p,
            "enable_gqa": enable_gqa,
            "is_causal": sdpa_is_causal,
        }
        if scale is not None:
            kwargs["scale"] = scale

        return F.scaled_dot_product_attention(q, k, v, **kwargs)

    mapped = local_map(
        cp_sdpa,
        out_placements=placements.out_placements,
        in_placements=placements.in_placements
        + ((rank_placements,) if rank_placements is not None and is_causal else ()),
        redistribute_inputs=True,
        device_mesh=mesh,
    )

    if cp_size is None or not is_causal:
        return mapped
    assert rank_placements is not None

    def call(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        rank_indices = torch.arange(cp_size, device=q.device)
        if isinstance(q, DTensor):
            rank_indices = distribute_tensor(rank_indices, mesh, rank_placements)
        return mapped(q, k, v, rank_indices)

    return call


def _make_context_parallel_flex_attention(
    mesh: DeviceMesh,
    *,
    batch_dim: int = 0,
    seq_dim: int = 2,
    head_dim: int = 1,
    scale: float | None = None,
    enable_gqa: bool = False,
    score_mod: Callable | None = None,
    block_mask: BlockMask | None = None,
    kernel_options: dict | None = None,
):
    cp_axis = _cp_axis_name(mesh)
    if cp_axis is not None and score_mod is not None:
        raise NotImplementedError(
            "FlexAttention score_mod is not supported with context parallel."
        )

    placements = context_parallel_attention_placements(
        mesh, batch_dim=batch_dim, seq_dim=seq_dim, head_dim=head_dim
    )

    cp_allgather = None
    cp_group_name = None
    if cp_axis is not None:
        from torch.distributed.tensor.experimental._context_parallel._attention import (
            flex_cp_allgather,
        )

        cp_mesh = mesh[cp_axis]
        cp_allgather = flex_cp_allgather
        cp_group_name = dist._get_process_group_name(cp_mesh.get_group())

    block_mask_args: tuple[object, ...] = ()
    block_mask_placements: tuple[object, ...] = ()
    cp_block_mask: _ContextParallelBlockMask | None = None
    if block_mask is not None:
        cp_size = mesh.size(_mesh_dim_names(mesh).index(cp_axis)) if cp_axis else 1
        cp_block_mask = _ContextParallelBlockMask.from_block_mask(
            block_mask,
            cp_size=cp_size,
            device=block_mask.kv_indices.device,
        )
        block_mask_args = cp_block_mask.args()
        block_mask_placements = cp_block_mask.placements(mesh, cp_axis)

    def cp_flex(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *mask_args: object,
    ):
        if cp_allgather is not None:
            k, v = cp_allgather(k.contiguous(), v.contiguous(), seq_dim, cp_group_name)

        local_block_mask = None
        if cp_block_mask is not None:
            local_block_mask = cp_block_mask.rebuild(*mask_args)

        return flex_attention(
            q,
            k,
            v,
            score_mod=score_mod,
            block_mask=local_block_mask,
            scale=scale,
            enable_gqa=enable_gqa,
            kernel_options=kernel_options,
        )

    mapped = local_map(
        cp_flex,
        out_placements=placements.out_placements,
        in_placements=placements.in_placements + tuple(block_mask_placements),
        redistribute_inputs=True,
        device_mesh=mesh,
    )

    if not block_mask_args:
        return mapped

    def call(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # The block-mask tensors are captured as plain tensors; when Q/K/V are
        # DTensors, distribute them to matching placements so local_map can
        # redistribute every input consistently.
        has_dtensor_input = any(isinstance(arg, DTensor) for arg in (q, k, v))
        mask_args = []
        for arg, placement in zip(block_mask_args, block_mask_placements):
            if (
                has_dtensor_input
                and isinstance(arg, torch.Tensor)
                and placement is not None
                and not isinstance(arg, DTensor)
            ):
                arg = distribute_tensor(arg, mesh, placement)
            mask_args.append(arg)
        return mapped(q, k, v, *mask_args)

    return call


def make_context_parallel_sdpa(mesh: DeviceMesh, **kwargs):
    """Build an SDPA callable for context-parallel attention.

    Args:
        mesh: Device mesh with named DP, CP, and/or TP dimensions.
        **kwargs: Arguments forwarded to ``make_context_parallel``.
    """

    return make_context_parallel(mesh, kind="sdpa", **kwargs)
