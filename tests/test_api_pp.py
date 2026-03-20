# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch import nn
from torch.distributed.tensor import DTensor
from torch.testing._internal.distributed.fake_pg import FakeStore

from autoparallel.api_pp import _make_pp_module


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


def _make_sharded_dicts(model, device_mesh):
    """Create DTensor param/buffer dicts from a model (replicated placement)."""
    param_dict = {}
    for name, param in model.named_parameters():
        local = torch.empty_like(param, device="cuda")
        dt = DTensor.from_local(local, device_mesh=device_mesh)
        param_dict[name] = nn.Parameter(dt, requires_grad=param.requires_grad)
    buffer_dict = {}
    for name, buf in model.named_buffers():
        local = torch.empty_like(buf, device="cuda")
        dt = DTensor.from_local(local, device_mesh=device_mesh)
        buffer_dict[name] = dt
    return param_dict, buffer_dict


def test_pp_init_weights_basic(device_mesh_1d):
    """Basic init_weights with in-place fills on the PP module."""
    dim = 128

    class Model(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear = nn.Linear(dim, dim)
            self.register_buffer("buf", torch.empty(dim))

        def forward(self, x):
            return self.linear(x) + self.buf

        def init_weights(self):
            with torch.no_grad():
                self.linear.weight.fill_(3.0)
                self.linear.bias.fill_(7.0)
                self.buf.fill_(5.0)

    with torch.device("meta"):
        model = Model(dim)

    param_dict, buffer_dict = _make_sharded_dicts(model, device_mesh_1d)
    pp_mod = _make_pp_module(param_dict, buffer_dict, model)
    pp_mod.init_weights()

    assert torch.equal(
        pp_mod.get_parameter("linear.weight").full_tensor(),
        torch.full((dim, dim), 3.0, device="cuda"),
    )
    assert torch.equal(
        pp_mod.get_parameter("linear.bias").full_tensor(),
        torch.full((dim,), 7.0, device="cuda"),
    )
    assert torch.equal(
        pp_mod.get_buffer("buf").full_tensor(),
        torch.full((dim,), 5.0, device="cuda"),
    )


def test_pp_init_weights_setattr(device_mesh_1d):
    """init_weights that assigns new Parameters and buffers via setattr."""
    dim = 128

    class Model(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear = nn.Linear(dim, dim)
            self.register_buffer("buf", torch.empty(dim))

        def forward(self, x):
            return self.linear(x) + self.buf

        def init_weights(self):
            self.linear.weight = nn.Parameter(torch.ones(dim, dim) * 9.0)
            self.buf = torch.arange(dim, dtype=torch.float32)

    with torch.device("meta"):
        model = Model(dim)

    param_dict, buffer_dict = _make_sharded_dicts(model, device_mesh_1d)
    pp_mod = _make_pp_module(param_dict, buffer_dict, model)
    pp_mod.init_weights()

    assert torch.equal(
        pp_mod.get_parameter("linear.weight").full_tensor(),
        torch.full((dim, dim), 9.0, device="cuda"),
    )
    assert torch.equal(
        pp_mod.get_buffer("buf").full_tensor(),
        torch.arange(dim, dtype=torch.float32, device="cuda"),
    )


def test_pp_init_weights_submodule(device_mesh_1d):
    """init_weights that delegates to submodule init_weights."""
    dim = 128

    class MLP(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.fc1 = nn.Linear(dim, dim)
            self.fc2 = nn.Linear(dim, dim)

        def forward(self, x):
            return self.fc2(self.fc1(x))

        def init_weights(self):
            with torch.no_grad():
                self.fc1.weight.fill_(1.0)
                self.fc1.bias.fill_(0.0)
                self.fc2.weight.fill_(2.0)
                self.fc2.bias.fill_(0.5)

    class Model(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.mlp = MLP(dim)

        def forward(self, x):
            return self.mlp(x)

        def init_weights(self):
            self.mlp.init_weights()

    with torch.device("meta"):
        model = Model(dim)

    param_dict, buffer_dict = _make_sharded_dicts(model, device_mesh_1d)
    pp_mod = _make_pp_module(param_dict, buffer_dict, model)
    pp_mod.init_weights()

    assert torch.equal(
        pp_mod.get_parameter("mlp.fc1.weight").full_tensor(),
        torch.ones(dim, dim, device="cuda"),
    )
    assert torch.equal(
        pp_mod.get_parameter("mlp.fc2.weight").full_tensor(),
        torch.full((dim, dim), 2.0, device="cuda"),
    )
    assert torch.equal(
        pp_mod.get_parameter("mlp.fc2.bias").full_tensor(),
        torch.full((dim,), 0.5, device="cuda"),
    )


def test_pp_init_weights_load_state_dict(device_mesh_1d):
    """init_weights that uses load_state_dict."""
    dim = 128

    class Model(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear = nn.Linear(dim, dim)

        def forward(self, x):
            return self.linear(x)

        def init_weights(self):
            state = {
                "linear.weight": torch.ones(dim, dim) * 4.0,
                "linear.bias": torch.full((dim,), 2.0),
            }
            self.load_state_dict(state)

    with torch.device("meta"):
        model = Model(dim)

    param_dict, buffer_dict = _make_sharded_dicts(model, device_mesh_1d)
    pp_mod = _make_pp_module(param_dict, buffer_dict, model)
    pp_mod.init_weights()

    assert torch.equal(
        pp_mod.get_parameter("linear.weight").full_tensor(),
        torch.full((dim, dim), 4.0, device="cuda"),
    )
    assert torch.equal(
        pp_mod.get_parameter("linear.bias").full_tensor(),
        torch.full((dim,), 2.0, device="cuda"),
    )


def test_pp_init_weights_user_helper_method(device_mesh_1d):
    """init_weights that calls a user-defined helper method on self."""
    dim = 128

    class Model(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear = nn.Linear(dim, dim)

        def forward(self, x):
            return self.linear(x)

        def _init_linear(self, linear):
            with torch.no_grad():
                linear.weight.fill_(6.0)
                linear.bias.fill_(1.0)

        def init_weights(self):
            self._init_linear(self.linear)

    with torch.device("meta"):
        model = Model(dim)

    param_dict, buffer_dict = _make_sharded_dicts(model, device_mesh_1d)
    pp_mod = _make_pp_module(param_dict, buffer_dict, model)

    assert isinstance(pp_mod, Model)
    pp_mod.init_weights()

    assert torch.equal(
        pp_mod.get_parameter("linear.weight").full_tensor(),
        torch.full((dim, dim), 6.0, device="cuda"),
    )
    assert torch.equal(
        pp_mod.get_parameter("linear.bias").full_tensor(),
        torch.full((dim,), 1.0, device="cuda"),
    )


def test_pp_init_weights_named_parameters(device_mesh_1d):
    """init_weights that iterates self.named_parameters()."""
    dim = 128

    class Model(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear1 = nn.Linear(dim, dim)
            self.linear2 = nn.Linear(dim, dim)

        def forward(self, x):
            return self.linear2(self.linear1(x))

        def init_weights(self):
            for name, param in self.named_parameters():
                with torch.no_grad():
                    if "weight" in name:
                        param.fill_(1.0)
                    else:
                        param.fill_(0.0)

    with torch.device("meta"):
        model = Model(dim)

    param_dict, buffer_dict = _make_sharded_dicts(model, device_mesh_1d)
    pp_mod = _make_pp_module(param_dict, buffer_dict, model)
    pp_mod.init_weights()

    assert torch.equal(
        pp_mod.get_parameter("linear1.weight").full_tensor(),
        torch.ones(dim, dim, device="cuda"),
    )
    assert torch.equal(
        pp_mod.get_parameter("linear2.bias").full_tensor(),
        torch.zeros(dim, device="cuda"),
    )
