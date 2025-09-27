# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch import nn
from torch.distributed.tensor.placement_types import Shard
from torch.testing._internal.distributed.fake_pg import FakeStore

from autoparallel.api import AutoParallel


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


def test_from_meta_model(device_mesh_1d):
    class Model(nn.Module):
        def __init__(self, dim1):
            super().__init__()
            self.linear = nn.Linear(dim1, dim1)
            self.register_buffer("buf", torch.rand(1))

        def forward(self, x, y):
            return y + 2, self.linear(x) * self.buf, x + y + self.buf

    dim = 128
    with torch.device("meta"):
        model = Model(dim)

    def input_fn():
        b = 32
        inputs = (
            torch.rand(b, dim, device="cuda"),
            torch.rand(b, 1, device="cuda"),
        )
        return inputs

    auto_p = AutoParallel(
        model,
        input_fn,
        device_mesh_1d,
    )
    assert isinstance(
        auto_p.model.get_parameter("linear.weight"), torch._subclasses.FakeTensor
    )
    assert isinstance(auto_p.model.get_buffer("buf"), torch._subclasses.FakeTensor)


def test_init(device_mesh_1d):
    dim = 128

    class Model(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear = nn.Linear(dim, dim)
            self.register_buffer("buf", torch.empty(dim))

        def forward(self, x):
            return self.linear(x) + self.buf

        def init_weights(self):
            self.linear.weight = torch.nn.Parameter(torch.ones(dim, dim) * 9.0)
            with torch.no_grad():
                self.linear.bias.fill_(98.6)
            self.buf = torch.arange(dim)

    def input_fn():
        b = 512
        inputs = (torch.rand(b, dim, device="cuda"),)
        return inputs

    with torch.device("meta"):
        model = Model(dim)
    with AutoParallel(
        model,
        input_fn,
        device_mesh_1d,
    ) as autop:
        x_sharding = (Shard(0),)
        autop.add_input_constraints([x_sharding])
        sharding_placement = autop.optimize_placement()

        # AutoParallel produces a module with meta-DTensor parameters that need to be initialized
        parallel_mod = autop.apply_placement(sharding_placement)
    parallel_mod.to_empty(device="cuda")
    parallel_mod.init_weights()
    assert torch.equal(
        parallel_mod.get_parameter("linear.weight").full_tensor(),
        torch.full((dim, dim), 9.0, device="cuda"),
    )
    assert torch.equal(
        parallel_mod.get_parameter("linear.bias").full_tensor(),
        torch.full((dim,), 98.6, device="cuda"),
    )
    assert torch.equal(
        parallel_mod.get_buffer("buf").full_tensor(), torch.arange(dim, device="cuda")
    )
