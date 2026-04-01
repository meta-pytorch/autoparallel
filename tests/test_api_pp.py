# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.distributed.tensor import DTensor

from autoparallel.api_pp import make_pp_module


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
    pp_mod = make_pp_module(param_dict, buffer_dict, model)
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
    pp_mod = make_pp_module(param_dict, buffer_dict, model)
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
    pp_mod = make_pp_module(param_dict, buffer_dict, model)
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
    pp_mod = make_pp_module(param_dict, buffer_dict, model)
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
    pp_mod = make_pp_module(param_dict, buffer_dict, model)

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
    pp_mod = make_pp_module(param_dict, buffer_dict, model)
    pp_mod.init_weights()

    assert torch.equal(
        pp_mod.get_parameter("linear1.weight").full_tensor(),
        torch.ones(dim, dim, device="cuda"),
    )
    assert torch.equal(
        pp_mod.get_parameter("linear2.bias").full_tensor(),
        torch.zeros(dim, device="cuda"),
    )


def test_pp_init_weights_optional_submodule(device_mesh_1d):
    """init_weights that checks for an optional submodule (self.rope is not None).

    Mirrors the torchtitan Decoder pattern where rope may or may not be present.
    When rope is None, the parallel model must still have the attribute so the
    None check doesn't raise AttributeError.
    """
    dim = 128

    class Model(nn.Module):
        def __init__(self, dim, use_rope=False):
            super().__init__()
            self.linear = nn.Linear(dim, dim)
            self.rope = nn.Linear(dim, dim) if use_rope else None

        def forward(self, x):
            return self.linear(x)

        def init_weights(self):
            with torch.no_grad():
                self.linear.weight.fill_(1.0)
                self.linear.bias.fill_(0.0)
            if self.rope is not None:
                with torch.no_grad():
                    self.rope.weight.fill_(2.0)

    with torch.device("meta"):
        model = Model(dim, use_rope=False)

    param_dict, buffer_dict = _make_sharded_dicts(model, device_mesh_1d)
    pp_mod = make_pp_module(param_dict, buffer_dict, model)
    pp_mod.init_weights()

    assert pp_mod.rope is None
    assert torch.equal(
        pp_mod.get_parameter("linear.weight").full_tensor(),
        torch.ones(dim, dim, device="cuda"),
    )
