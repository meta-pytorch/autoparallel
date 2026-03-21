# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode

from autoparallel.tracing import move_to_fake


def _make_fake_mode():
    return FakeTensorMode()


def test_move_to_fake_param_alias():
    """Aliased parameters (e.g. tied embeddings) remain the same object."""

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(16, 16, bias=False)
            self.lm_head = nn.Linear(16, 16, bias=False)
            self.lm_head.weight = self.embed.weight

        def forward(self, x):
            return self.lm_head(self.embed(x))

    with torch.device("meta"):
        model = Model()

    assert model.embed.weight is model.lm_head.weight

    mode = _make_fake_mode()
    move_to_fake(model, mode, torch.device("cpu"))

    assert model.embed.weight is model.lm_head.weight
    assert isinstance(model.embed.weight, nn.Parameter)


def test_move_to_fake_buffer_alias():
    """Aliased buffers (e.g. rope cache registered under two names) remain the same object."""

    class RoPE(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("cache", torch.zeros(16), persistent=False)

        def forward(self, x):
            return x + self.cache

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.rope = RoPE()
            self.register_buffer("freqs_cis", self.rope.cache, persistent=False)

        def forward(self, x):
            return self.rope(x) + self.freqs_cis

    with torch.device("meta"):
        model = Model()

    assert model.freqs_cis is model.rope.cache

    mode = _make_fake_mode()
    move_to_fake(model, mode, torch.device("cpu"))

    assert model.freqs_cis is model.rope.cache


def test_move_to_fake_module_alias():
    """Aliased submodules (e.g. model_ema = teacher) remain the same object,
    and their parameters remain aliased too."""

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.student = nn.Linear(16, 16)
            self.teacher = nn.Linear(16, 16)
            self.model_ema = self.teacher

        def forward(self, x):
            return self.student(x) + self.teacher(x)

    with torch.device("meta"):
        model = Model()

    assert model.model_ema is model.teacher

    mode = _make_fake_mode()
    move_to_fake(model, mode, torch.device("cpu"))

    assert model.model_ema is model.teacher
    assert model.model_ema.weight is model.teacher.weight
    assert model.model_ema.bias is model.teacher.bias


def test_move_to_fake_no_false_aliasing():
    """Distinct parameters must remain distinct after move_to_fake."""

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(16, 16, bias=False)
            self.b = nn.Linear(16, 16, bias=False)

        def forward(self, x):
            return self.a(x) + self.b(x)

    with torch.device("meta"):
        model = Model()

    assert model.a.weight is not model.b.weight

    mode = _make_fake_mode()
    move_to_fake(model, mode, torch.device("cpu"))

    assert model.a.weight is not model.b.weight
