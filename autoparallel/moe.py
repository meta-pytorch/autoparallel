# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""Public mesh and placement helpers for aligned MoE ``local_map`` regions."""

from dataclasses import dataclass

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DeviceMesh
from torch.distributed.tensor.placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
)

__all__ = [
    "MoEMeshRoles",
    "build_moe_local_map_placements",
    "build_moe_mesh",
]


@dataclass(frozen=True)
class MoEMeshRoles:
    """Which ``DeviceMesh`` axes are expert-parallel inside an MoE ``local_map``.

    ``ep_axis_names`` names the axes whose flatten forms the all-to-all group.
    ``ep_group_name`` is the mesh-dimension name used by region collectives.
    """

    ep_axis_names: tuple[str, ...]
    ep_group_name: str


def build_moe_local_map_placements(
    mesh_dim_names: tuple[str, ...], roles: MoEMeshRoles
) -> tuple[tuple[Placement, ...], tuple[Placement, ...], tuple[Placement, ...]]:
    """Return token, expert-weight, and token-count placements, in that order."""
    token: list[Placement] = []
    weight: list[Placement] = []
    count: list[Placement] = []
    for name in mesh_dim_names:
        token.append(Shard(0))
        count.append(Partial(reduce_op="sum"))
        weight.append(Shard(0) if name in roles.ep_axis_names else Replicate())
    return tuple(token), tuple(weight), tuple(count)


def build_moe_mesh(
    *,
    dp_replicate: int,
    dp_shard: int,
    cp: int,
    tp: int,
    ep: int,
    device_type: str = "cuda",
) -> tuple[DeviceMesh, MoEMeshRoles]:
    """Build one AutoParallel mesh and role map for an aligned MoE config.

    ``ep`` is formed from axes already contained in ``dp_shard * cp * tp``; it
    is not an additional world-size factor. The resulting world size is
    ``dp_replicate * dp_shard * cp * tp``, and the external FSDP degree is
    ``dp_shard * cp * tp / ep``.

    Only aligned configs are supported: ``ep >= 2``, ``ep`` must be a multiple
    of ``cp * tp``, and ``ep / (cp * tp)`` must divide ``dp_shard``. Size-one
    axes are omitted. Callers must also ensure ``num_experts`` is divisible by
    ``ep``; this builder cannot see the model configuration.

    Returns:
        The shared ``DeviceMesh`` and the ``MoEMeshRoles`` describing its EP
        axes and collective group.
    """
    if ep < 2:
        raise ValueError(
            f"build_moe_mesh requires ep >= 2 (expert parallelism); got ep={ep}. "
            "Use the dense path for models without expert parallelism."
        )
    if ep % (cp * tp) != 0:
        raise ValueError(
            f"Non-aligned MoE config: ep({ep}) must be a multiple of cp*tp "
            f"({cp}*{tp}); splitting cp/tp across ep is not supported."
        )
    dp_shard_in_ep = ep // (cp * tp)
    if dp_shard % dp_shard_in_ep != 0:
        raise ValueError(
            f"Non-aligned MoE config: ep/(cp*tp)={dp_shard_in_ep} must divide "
            f"dp_shard({dp_shard})."
        )
    dp_shard_mod_ep = dp_shard // dp_shard_in_ep  # == efsdp == dp_shard*cp*tp/ep

    # dp outermost, ep-constituents contiguous innermost so the EP group is
    # contiguous (matches TorchTitan's world unflatten).
    ordered = [
        ("dp_replicate", dp_replicate),
        ("dp_shard_mod_ep", dp_shard_mod_ep),
        ("dp_shard_in_ep", dp_shard_in_ep),
        ("cp", cp),
        ("tp", tp),
    ]
    atoms = [(n, d) for n, d in ordered if d > 1]
    if not atoms:
        raise ValueError("Degenerate MoE config: all parallelism degrees are 1.")
    names = tuple(n for n, _ in atoms)
    mesh = init_device_mesh(
        device_type, tuple(d for _, d in atoms), mesh_dim_names=names
    )

    ep_atoms = tuple(n for n in ("dp_shard_in_ep", "cp", "tp") if n in names)
    assert ep_atoms, "aligned config with ep>1 always keeps >=1 ep atom"
    if len(ep_atoms) == 1:
        ep_group_name = ep_atoms[0]
    else:
        ep_group_name = "ep"
        mesh[ep_atoms]._flatten(ep_group_name)

    roles = MoEMeshRoles(ep_axis_names=ep_atoms, ep_group_name=ep_group_name)
    return mesh, roles
