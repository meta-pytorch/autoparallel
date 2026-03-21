# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch.testing._internal.distributed.fake_pg import FakeStore


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
