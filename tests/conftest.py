# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import pytest
import torch
from torch.testing._internal.distributed.fake_pg import FakeStore

# Patches for running tests on machines with fewer GPUs than the fake world
# size. Without these, deferred CUDA calls (e.g., _check_capability) try to
# access non-existent devices.
_CUDA_PATCHES = [
    patch("torch.cuda.device_count", lambda: 8),
    patch("torch.cuda.get_device_name", lambda *args, **kwargs: "H100"),
    patch("torch.cuda.get_device_capability", lambda *args, **kwargs: (9, 0)),
    patch(
        "torch.cuda.get_device_properties",
        lambda *args, **kwargs: type(
            "Props",
            (),
            {
                "major": 9,
                "minor": 0,
                "name": "H100",
                "total_memory": 80 * 1024**3,
                "multi_processor_count": 132,
            },
        )(),
    ),
]


def apply_cuda_patches(func):
    """Decorator that applies all CUDA device patches for fake multi-GPU testing."""
    for p in reversed(_CUDA_PATCHES):
        func = p(func)
    return func


@pytest.fixture(scope="module", autouse=True)
def init_pg():
    world_size = 256
    fake_store = FakeStore()
    if torch.distributed.is_initialized():
        return
    torch.distributed.init_process_group(
        "fake", store=fake_store, rank=0, world_size=world_size
    )


@pytest.fixture(scope="module")
def device_mesh_1d():
    world_size = torch.distributed.get_world_size()
    mesh = torch.distributed.device_mesh.init_device_mesh(
        "cuda", (world_size,), mesh_dim_names=("dp",)
    )
    return mesh


@pytest.fixture(scope="module")
def device_mesh_2d():
    world_size = torch.distributed.get_world_size()
    mesh = torch.distributed.device_mesh.init_device_mesh(
        "cuda",
        (world_size // 8, 8),
        mesh_dim_names=("dp", "tp"),
    )
    return mesh


@pytest.fixture(scope="module")
def device_mesh_3d():
    world_size = torch.distributed.get_world_size()
    mesh = torch.distributed.device_mesh.init_device_mesh(
        "cuda",
        (world_size // 32, 8, 4),
        mesh_dim_names=("dp", "tp", "cp"),
    )
    return mesh
