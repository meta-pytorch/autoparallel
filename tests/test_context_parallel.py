# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest
import torch
from conftest import apply_cuda_patches
from torch.distributed.tensor.placement_types import Shard

from autoparallel import context_parallel_attention_placements


def _fake_mesh(mesh_dim_names=None, ndim=None):
    if ndim is None:
        ndim = len(mesh_dim_names)
    return SimpleNamespace(mesh_dim_names=mesh_dim_names, ndim=ndim)


@pytest.mark.parametrize(
    "mesh_dim_names,expected_placements",
    (
        (("dp", "cp", "tp"), (Shard(0), Shard(2), Shard(1))),
        (
            ("dp_replicate", "dp_shard", "cp", "tp"),
            (Shard(0), Shard(0), Shard(2), Shard(1)),
        ),
        (
            ("dp_shard_mod_ep", "dp_shard_in_ep", "cp", "tp"),
            (Shard(0), Shard(0), Shard(2), Shard(1)),
        ),
        (
            (
                "dp_replicate",
                "dp_shard_mod_ep",
                "dp_shard_in_ep",
                "cp",
                "tp",
            ),
            (Shard(0), Shard(0), Shard(0), Shard(2), Shard(1)),
        ),
        (("cp", "tp"), (Shard(2), Shard(1))),
        (("tp", "cp"), (Shard(1), Shard(2))),
        (("dp_shard", "tp"), (Shard(0), Shard(1))),
    ),
)
def test_context_parallel_attention_placement_names(
    mesh_dim_names, expected_placements
):
    placements = context_parallel_attention_placements(
        _fake_mesh(mesh_dim_names),
        batch_dim=0,
        seq_dim=2,
        head_dim=1,
    )

    assert placements.qkv == expected_placements
    assert placements.in_placements == (expected_placements,) * 3
    assert placements.out_placements == (expected_placements,)


@pytest.mark.parametrize(
    "axis_name",
    (
        "bad",
        "context",
        "context_parallel",
        "tensor",
        "tensor_parallel",
        "ddp",
        "data",
        "data_parallel",
        "fsdp",
    ),
)
def test_context_parallel_attention_placements_reject_noncanonical_axis(axis_name):
    with pytest.raises(ValueError, match="Unsupported mesh axis"):
        context_parallel_attention_placements(_fake_mesh(("dp", axis_name)))


@pytest.mark.parametrize("ndim", (1, 2, 3, 4, 5))
def test_context_parallel_attention_placements_reject_unnamed_meshes(ndim):
    with pytest.raises(ValueError, match="requires named mesh axes"):
        context_parallel_attention_placements(_fake_mesh(None, ndim=ndim))


@apply_cuda_patches
@pytest.mark.parametrize(
    "mesh_shape,mesh_dim_names,qkv_placements",
    (
        (
            (2, 2, 2),
            ("dp_shard", "cp", "tp"),
            (Shard(0), Shard(2), Shard(1)),
        ),
        (
            (2, 2),
            ("dp_shard", "cp"),
            (Shard(0), Shard(2)),
        ),
        (
            (2, 2),
            ("cp", "tp"),
            (Shard(2), Shard(1)),
        ),
        (
            (2, 2),
            ("tp", "cp"),
            (Shard(1), Shard(2)),
        ),
        (
            (2, 2, 2, 2),
            ("dp_replicate", "dp_shard", "cp", "tp"),
            (Shard(0), Shard(0), Shard(2), Shard(1)),
        ),
        (
            (2, 2, 2, 2),
            ("dp_shard_mod_ep", "dp_shard_in_ep", "cp", "tp"),
            (Shard(0), Shard(0), Shard(2), Shard(1)),
        ),
        (
            (2, 2, 2, 2, 2),
            (
                "dp_replicate",
                "dp_shard_mod_ep",
                "dp_shard_in_ep",
                "cp",
                "tp",
            ),
            (Shard(0), Shard(0), Shard(0), Shard(2), Shard(1)),
        ),
    ),
)
def test_context_parallel_attention_placements_are_qkv_sharded(
    mesh_shape,
    mesh_dim_names,
    qkv_placements,
):
    mesh = torch.distributed.device_mesh.init_device_mesh(
        "cuda", mesh_shape, mesh_dim_names=mesh_dim_names
    )

    placements = context_parallel_attention_placements(
        mesh,
        batch_dim=0,
        seq_dim=2,
        head_dim=1,
    )

    assert placements.in_placements == (qkv_placements,) * 3
    assert placements.out_placements == (qkv_placements,)
