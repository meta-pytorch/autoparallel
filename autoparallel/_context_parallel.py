# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, Callable, Literal

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.distributed.tensor.placement_types import Placement, Replicate, Shard
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from torch.utils import _pytree

from ._local_map_regions import _DeferredLocalMapBody
from .collectives import local_map


if BlockMask not in _pytree.SUPPORTED_NODES:
    _pytree.register_pytree_node(
        BlockMask,
        BlockMask._flatten,
        BlockMask._unflatten,
        flatten_with_keys_fn=BlockMask._flatten_with_keys,
    )


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
    leaves: tuple[torch.Tensor, ...]
    tree_spec: Any
    leaf_placements: tuple[tuple[Placement, ...], ...]
    tensor_attrs: tuple[str, ...]
    num_regular_leaves: int
    q_offsets: torch.Tensor
    block_size: int | tuple[int, int]
    local_seq_len: int
    kv_seq_len: int

    @classmethod
    def from_block_mask(
        cls,
        block_mask: BlockMask,
        *,
        mesh: DeviceMesh,
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
        leaves, tree_spec = _pytree.tree_flatten(block_mask)
        if not all(isinstance(leaf, torch.Tensor) for leaf in leaves):
            raise TypeError(
                "Context-parallel FlexAttention requires BlockMask pytree leaves "
                "to be tensors."
            )

        optional_attrs = {
            attr
            for attr in BlockMask._TENSOR_ATTRS
            if getattr(block_mask, attr) is None
        }
        tensor_attrs = [
            attr for attr in BlockMask._TENSOR_ATTRS if attr not in optional_attrs
        ]
        replicated = tuple(Replicate() for _ in range(mesh.ndim))
        leaf_placements = []
        for index, leaf in enumerate(leaves):
            if index < len(tensor_attrs) and tensor_attrs[index] in {
                "kv_num_blocks",
                "kv_indices",
                "full_kv_num_blocks",
                "full_kv_indices",
            }:
                leaf_placements.append(cls._placements_for_tensor(leaf, mesh))
            else:
                leaf_placements.append(replicated)
        return cls(
            leaves=tuple(leaves),
            tree_spec=tree_spec,
            leaf_placements=tuple(leaf_placements),
            tensor_attrs=tuple(tensor_attrs),
            num_regular_leaves=len(tensor_attrs),
            q_offsets=q_offsets,
            block_size=block_mask.BLOCK_SIZE,
            local_seq_len=local_seq_len,
            kv_seq_len=kv_seq_len,
        )

    def args(self) -> tuple[object, ...]:
        return (*self.leaves, self.q_offsets)

    @staticmethod
    def _placements_for_tensor(
        tensor: torch.Tensor, mesh: DeviceMesh
    ) -> tuple[Placement, ...]:
        base_placements = context_parallel_attention_placements(
            mesh, batch_dim=0, seq_dim=2, head_dim=1
        ).qkv
        placements: list[Placement] = []
        for mesh_dim, base_placement in enumerate(base_placements):
            mesh_size = mesh.size(mesh_dim)
            dim = base_placement.dim

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
        return (*self.leaf_placements, tuple(offset_placements))


def _rebuild_context_parallel_block_mask(
    args: tuple[object, ...],
    *,
    tree_spec: Any,
    num_regular_leaves: int,
    block_size: int | tuple[int, int],
    local_seq_len: int,
    kv_seq_len: int,
) -> BlockMask:
    *leaves, q_offsets = args
    leaves = [
        leaf if index < num_regular_leaves else leaf.clone()
        for index, leaf in enumerate(leaves)
    ]
    block_mask = _pytree.tree_unflatten(list(leaves), tree_spec)
    if not isinstance(block_mask, BlockMask):
        raise TypeError("Expected the BlockMask pytree to rebuild a BlockMask.")
    assert isinstance(q_offsets, torch.Tensor)
    q_offset = q_offsets.reshape(())
    mask_mod = block_mask.mask_mod
    shifted_mask_mod: Callable | None
    if mask_mod is not None:

        def shifted_mask_mod(b, h, q_idx, kv_idx):
            return mask_mod(b, h, q_idx + q_offset, kv_idx)

    else:
        shifted_mask_mod = None

    return BlockMask.from_kv_blocks(
        block_mask.kv_num_blocks,
        block_mask.kv_indices,
        block_mask.full_kv_num_blocks,
        block_mask.full_kv_indices,
        BLOCK_SIZE=block_size,
        mask_mod=shifted_mask_mod,
        seq_lengths=(local_seq_len, kv_seq_len),
    )


def _mesh_dim_names(mesh: DeviceMesh) -> tuple[str, ...]:
    names = getattr(mesh, "mesh_dim_names", None)
    if names is None or any(name is None for name in names):
        raise ValueError(
            "context_parallel_attention_placements requires named mesh axes."
        )
    return tuple(names)


def _cp_axis_name(mesh: DeviceMesh) -> str | None:
    names = _mesh_dim_names(mesh)
    context_parallel_attention_placements(mesh)
    cp_names = [name for name in names if name == "cp"]
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
    for axis_name in _mesh_dim_names(mesh):
        match axis_name:
            case "dp" | "dp_replicate" | "dp_shard":
                placement = Shard(batch_dim)
            case "dp_shard_mod_ep" | "dp_shard_in_ep":
                placement = Shard(batch_dim)
            case "cp":
                placement = Shard(seq_dim)
            case "tp":
                placement = Shard(head_dim)
            case _:
                raise ValueError(
                    f"Unsupported mesh axis {axis_name!r} for context parallel "
                    "attention."
                )
        qkv.append(placement)

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


def make_context_parallel_body(
    mesh: DeviceMesh,
    *,
    kind: Literal["sdpa"] = "sdpa",
    seq_dim: int = 2,
    is_causal: bool = True,
    dropout_p: float = 0.0,
    scale: float | None = None,
    enable_gqa: bool = False,
):
    """Build an unwrapped local-tensor context-parallel attention body.

    The returned callable can be passed to ``local_map``. During eager execution
    it runs the same DTensor context-parallel dispatcher path as TorchTitan.
    """
    if kind != "sdpa":
        raise ValueError(f"Unsupported context-parallel attention kind: {kind!r}")

    cp_axis = _cp_axis_name(mesh)
    if cp_axis is not None and dropout_p != 0.0:
        raise ValueError("Context-parallel SDPA does not support dropout.")

    kwargs = {
        "dropout_p": dropout_p,
        "enable_gqa": enable_gqa,
        "is_causal": is_causal,
    }
    if scale is not None:
        kwargs["scale"] = scale

    def local_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        return F.scaled_dot_product_attention(q, k, v, **kwargs)

    if cp_axis is None:
        return local_sdpa

    from torch.distributed.tensor.experimental._attention import (
        _enable_context_parallel_dispatcher,
    )

    _enable_context_parallel_dispatcher()
    cp_mesh = mesh[cp_axis]
    cp_placements = (Shard(seq_dim),)

    def runtime_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        q = DTensor.from_local(q, cp_mesh, cp_placements, run_check=False)
        k = DTensor.from_local(k, cp_mesh, cp_placements, run_check=False)
        v = DTensor.from_local(v, cp_mesh, cp_placements, run_check=False)
        out = F.scaled_dot_product_attention(q, k, v, **kwargs)
        return out.to_local()

    return _DeferredLocalMapBody(
        surrogate_fn=local_sdpa,
        runtime_fn=runtime_sdpa,
    )


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
    placements = context_parallel_attention_placements(
        mesh, batch_dim=batch_dim, seq_dim=seq_dim, head_dim=head_dim
    )
    body = make_context_parallel_body(
        mesh,
        kind="sdpa",
        seq_dim=seq_dim,
        is_causal=is_causal,
        dropout_p=dropout_p,
        scale=scale,
        enable_gqa=enable_gqa,
    )
    return local_map(
        body,
        out_placements=placements.out_placements,
        in_placements=placements.in_placements,
        redistribute_inputs=True,
        device_mesh=mesh,
    )


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

    def build_mapped(
        active_block_mask: BlockMask | None,
    ) -> tuple[
        Callable,
        tuple[object, ...],
        tuple[object, ...],
        _ContextParallelBlockMask | None,
    ]:
        block_mask_args: tuple[object, ...] = ()
        block_mask_placements: tuple[object, ...] = ()
        cp_block_mask: _ContextParallelBlockMask | None = None
        if active_block_mask is not None:
            cp_size = mesh.size(_mesh_dim_names(mesh).index(cp_axis)) if cp_axis else 1
            cp_block_mask = _ContextParallelBlockMask.from_block_mask(
                active_block_mask,
                mesh=mesh,
                cp_size=cp_size,
                device=active_block_mask.kv_indices.device,
            )
            block_mask_args = cp_block_mask.args()
            block_mask_placements = cp_block_mask.placements(mesh, cp_axis)
            tree_spec = cp_block_mask.tree_spec
            num_regular_leaves = cp_block_mask.num_regular_leaves
            block_size = cp_block_mask.block_size
            local_seq_len = cp_block_mask.local_seq_len
            kv_seq_len = cp_block_mask.kv_seq_len
        else:
            tree_spec = None
            num_regular_leaves = 0
            block_size = 0
            local_seq_len = 0
            kv_seq_len = 0

        def run_flex(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            mask_args: tuple[object, ...],
        ):
            if cp_allgather is not None:
                k, v = cp_allgather(
                    k.contiguous(), v.contiguous(), seq_dim, cp_group_name
                )

            local_block_mask = None
            if tree_spec is not None:
                local_block_mask = _rebuild_context_parallel_block_mask(
                    mask_args,
                    tree_spec=tree_spec,
                    num_regular_leaves=num_regular_leaves,
                    block_size=block_size,
                    local_seq_len=local_seq_len,
                    kv_seq_len=kv_seq_len,
                )

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

        def cp_flex(q, k, v, *mask_args):
            return run_flex(q, k, v, mask_args)

        def surrogate_flex(q, k, v, *mask_args):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
            for arg in mask_args:
                if isinstance(arg, torch.Tensor):
                    out = out + arg.reshape(-1)[0].to(out.dtype) * 0
            return out

        mapped = local_map(
            _DeferredLocalMapBody(
                surrogate_fn=surrogate_flex,
                runtime_fn=cp_flex,
            ),
            out_placements=placements.out_placements,
            in_placements=placements.in_placements + tuple(block_mask_placements),
            redistribute_inputs=True,
            device_mesh=mesh,
        )
        return mapped, block_mask_args, block_mask_placements, cp_block_mask

    def invoke(
        mapped: Callable,
        block_mask_args: tuple[object, ...],
        block_mask_placements: tuple[object, ...],
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        if not block_mask_args:
            return mapped(q, k, v)

        # When Q/K/V are DTensors, distribute the BlockMask tensor leaves to
        # matching placements so local_map can redistribute every input.
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

    (
        factory_mapped,
        factory_args,
        factory_placements,
        factory_block_mask,
    ) = build_mapped(block_mask)
    has_factory_block_mask = factory_block_mask is not None
    factory_tensor_attrs = (
        factory_block_mask.tensor_attrs if factory_block_mask is not None else ()
    )
    factory_num_regular_leaves = (
        factory_block_mask.num_regular_leaves if factory_block_mask is not None else 0
    )
    factory_closure_count = (
        len(factory_block_mask.leaves) - factory_num_regular_leaves
        if factory_block_mask is not None
        else 0
    )

    def call(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        block_mask: BlockMask | None = None,
        mask_mod_buffers: tuple[torch.Tensor, ...] | None = None,
    ):
        if block_mask is None:
            return invoke(
                factory_mapped,
                factory_args,
                factory_placements,
                q,
                k,
                v,
            )

        if has_factory_block_mask:
            regular_leaves = tuple(
                getattr(block_mask, attr) for attr in factory_tensor_attrs
            )
            if mask_mod_buffers is None:
                runtime_leaves, _ = _pytree.tree_flatten(block_mask)
                mask_mod_buffers = tuple(runtime_leaves[factory_num_regular_leaves:])
            if len(mask_mod_buffers) != factory_closure_count:
                raise ValueError(
                    "Runtime mask_mod_buffers must match the factory-time "
                    "BlockMask closure tensor count."
                )
            return invoke(
                factory_mapped,
                (*regular_leaves, *mask_mod_buffers, factory_args[-1]),
                factory_placements,
                q,
                k,
                v,
            )

        if mask_mod_buffers is not None:
            raise ValueError(
                "mask_mod_buffers requires a factory-time BlockMask template."
            )
        runtime_mapped, runtime_args, runtime_placements, _ = build_mapped(block_mask)
        return invoke(
            runtime_mapped,
            runtime_args,
            runtime_placements,
            q,
            k,
            v,
        )

    return call


def make_context_parallel_sdpa(mesh: DeviceMesh, **kwargs):
    """Build an SDPA callable for context-parallel attention.

    Args:
        mesh: Device mesh with named DP, CP, and/or TP dimensions.
        **kwargs: Arguments forwarded to ``make_context_parallel``.
    """

    return make_context_parallel(mesh, kind="sdpa", **kwargs)
