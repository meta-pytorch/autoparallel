# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn

from autoparallel.module_construction import make_parallel_module


def _make_param_and_buffer_dicts(model):
    """Create plain-tensor param/buffer dicts from a meta model."""
    param_dict = {}
    for name, param in model.named_parameters():
        param_dict[name] = nn.Parameter(
            torch.randn(param.shape), requires_grad=param.requires_grad
        )
    buffer_dict = {}
    for name, buf in model.named_buffers():
        buffer_dict[name] = torch.randn(buf.shape)
    return param_dict, buffer_dict


# --- basic construction ---


def test_params_and_buffers_registered():
    """Parameters and buffers appear on the parallel module."""
    dim = 16

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(dim, dim)
            self.register_buffer("buf", torch.empty(dim))

        def forward(self, x):
            return self.linear(x) + self.buf

    with torch.device("meta"):
        model = Model()

    param_dict, buffer_dict = _make_param_and_buffer_dicts(model)
    mod = make_parallel_module(model, param_dict, buffer_dict)

    param_names = {n for n, _ in mod.named_parameters()}
    assert "linear.weight" in param_names
    assert "linear.bias" in param_names

    buffer_names = {n for n, _ in mod.named_buffers()}
    assert "buf" in buffer_names


def test_isinstance_user_class():
    """Parallel module is an instance of the user's model class."""
    dim = 16

    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(dim, dim)

        def forward(self, x):
            return self.linear(x)

    with torch.device("meta"):
        model = MyModel()

    param_dict, buffer_dict = _make_param_and_buffer_dicts(model)
    mod = make_parallel_module(model, param_dict, buffer_dict)

    assert isinstance(mod, MyModel)


def test_user_attrs_copied():
    """User-defined instance attributes are available on the parallel module."""
    dim = 16

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(dim, dim)
            self.dim = dim
            self.name = "test"

        def forward(self, x):
            return self.linear(x)

    with torch.device("meta"):
        model = Model()

    param_dict, buffer_dict = _make_param_and_buffer_dicts(model)
    mod = make_parallel_module(model, param_dict, buffer_dict)

    assert mod.dim == dim
    assert mod.name == "test"


def test_user_methods_inherited():
    """User-defined methods are accessible via class inheritance."""
    dim = 16

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(dim, dim)

        def forward(self, x):
            return self.linear(x)

        def get_num_params(self):
            return sum(p.numel() for p in self.parameters())

    with torch.device("meta"):
        model = Model()

    param_dict, buffer_dict = _make_param_and_buffer_dicts(model)
    mod = make_parallel_module(model, param_dict, buffer_dict)

    assert hasattr(mod, "get_num_params")
    assert mod.get_num_params() == dim * dim + dim


def test_custom_forward_fn():
    """A custom forward_fn is used as the forward method."""
    dim = 16
    called = []

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(dim, dim)

        def forward(self, x):
            raise AssertionError("should not be called")

    def my_forward(self, x):
        called.append(True)
        return x

    with torch.device("meta"):
        model = Model()

    param_dict, buffer_dict = _make_param_and_buffer_dicts(model)
    mod = make_parallel_module(model, param_dict, buffer_dict, forward_fn=my_forward)

    mod(torch.randn(2, dim))
    assert len(called) == 1


# --- ModuleDict preservation ---


def test_moduledict_preserved():
    """nn.ModuleDict structure is preserved from the ref model."""
    dim = 16

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleDict(
                {"a": nn.Linear(dim, dim), "b": nn.Linear(dim, dim)}
            )

        def forward(self, x):
            return self.layers["a"](x) + self.layers["b"](x)

    with torch.device("meta"):
        model = Model()

    param_dict, buffer_dict = _make_param_and_buffer_dicts(model)
    mod = make_parallel_module(model, param_dict, buffer_dict)

    assert isinstance(mod.layers, nn.ModuleDict)
    assert "a" in mod.layers
    assert "b" in mod.layers


# --- alias re-registration ---


def test_param_alias_reregistered():
    """Aliased parameters (weight tying) are re-established."""
    dim = 16

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(dim, dim, bias=False)
            self.lm_head = nn.Linear(dim, dim, bias=False)
            self.lm_head.weight = self.embed.weight

        def forward(self, x):
            return self.embed(x)

    with torch.device("meta"):
        model = Model()

    # Only the canonical FQN ends up in the dict (tracer deduplicates)
    param_dict = {"embed.weight": nn.Parameter(torch.randn(dim, dim))}
    buffer_dict = {}

    mod = make_parallel_module(model, param_dict, buffer_dict)

    assert hasattr(mod, "lm_head")
    assert mod.get_parameter("lm_head.weight") is mod.get_parameter("embed.weight")


def test_buffer_alias_reregistered():
    """Aliased buffers (e.g. rope.cache / freqs_cis) are re-established."""
    dim = 16

    class RoPE(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("cache", torch.zeros(dim), persistent=False)

        def forward(self, x):
            return x + self.cache

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(dim, dim)
            self.rope = RoPE()
            self.register_buffer("freqs_cis", self.rope.cache, persistent=False)

        def forward(self, x):
            return self.linear(x) + self.freqs_cis

    with torch.device("meta"):
        model = Model()

    param_dict, _ = _make_param_and_buffer_dicts(model)
    # Only the canonical buffer survives deduplication; figure out which one.
    seen: dict[int, str] = {}
    for fqn, buf in model.named_buffers():
        seen[id(buf)] = fqn
    canonical_fqn: str | None = None
    for fqn, buf in model.named_buffers(remove_duplicate=False):
        if fqn not in seen.values():
            canonical_fqn = seen[id(buf)]
            break
    assert canonical_fqn is not None
    buffer_dict = {canonical_fqn: torch.randn(dim)}

    mod = make_parallel_module(model, param_dict, buffer_dict)

    # The alias is whichever FQN wasn't canonical
    alias_fqn = next(
        fqn
        for fqn, buf in model.named_buffers(remove_duplicate=False)
        if fqn not in seen.values()
    )
    assert mod.get_buffer(alias_fqn) is mod.get_buffer(canonical_fqn)
    assert "freqs_cis" not in mod.state_dict()
    assert "rope.cache" not in mod.state_dict()


def test_module_alias_reestablished():
    """Aliased submodules (e.g. model_ema = teacher) are re-established."""
    dim = 16

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.student = nn.Linear(dim, dim)
            self.teacher = nn.Linear(dim, dim)
            self.model_ema = self.teacher

        def forward(self, x):
            return self.student(x) + self.teacher(x)

    with torch.device("meta"):
        model = Model()

    assert model.model_ema is model.teacher

    param_dict, buffer_dict = _make_param_and_buffer_dicts(model)
    mod = make_parallel_module(model, param_dict, buffer_dict)

    assert mod.model_ema is mod.teacher


# --- orphan submodule copying ---


def test_orphan_submodule_copied():
    """Submodules with no params/buffers are copied from ref_model."""
    dim = 16

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(dim, dim)
            self.rope = None

        def forward(self, x):
            return self.linear(x)

    with torch.device("meta"):
        model = Model()

    param_dict, buffer_dict = _make_param_and_buffer_dicts(model)
    mod = make_parallel_module(model, param_dict, buffer_dict)

    assert mod.rope is None


# --- inheritance ---


def test_inherited_methods_available():
    """Methods from base classes in the MRO are accessible."""
    dim = 16

    class BaseModel(nn.Module):
        def get_num_params(self):
            return sum(p.numel() for p in self.parameters())

    class Model(BaseModel):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(dim, dim)

        def forward(self, x):
            return self.linear(x)

    with torch.device("meta"):
        model = Model()

    param_dict, buffer_dict = _make_param_and_buffer_dicts(model)
    mod = make_parallel_module(model, param_dict, buffer_dict)

    assert isinstance(mod, Model)
    assert isinstance(mod, BaseModel)
    assert mod.get_num_params() == dim * dim + dim


def test_classmethod_and_property():
    """Classmethods and properties from the user class are accessible."""
    dim = 16

    class Model(nn.Module):
        _registry = []

        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(dim, dim)
            self._hidden_dim = dim * 4

        def forward(self, x):
            return self.linear(x)

        @classmethod
        def from_config(cls, d):
            return cls()

        @property
        def hidden_dim(self):
            return self._hidden_dim

    with torch.device("meta"):
        model = Model()

    param_dict, buffer_dict = _make_param_and_buffer_dicts(model)
    mod = make_parallel_module(model, param_dict, buffer_dict)

    assert mod.hidden_dim == dim * 4
    assert hasattr(type(mod), "from_config")
    assert type(mod)._registry is Model._registry
