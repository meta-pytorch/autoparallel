# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import Placement, Shard


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
    if names is None or any(name is None for name in names):
        raise ValueError(
            "context_parallel_attention_placements requires named mesh axes."
        )
    return tuple(names)


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
