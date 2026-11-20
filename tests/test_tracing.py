# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx

from autoparallel.tracing import _get_decomp_table, move_to_fake

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


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


def _trace_addmm(beta, alpha):
    def f(bias, x, w):
        return torch.addmm(bias, x, w, beta=beta, alpha=alpha)

    return make_fx(f, decomposition_table=_get_decomp_table())


def test_addmm_decomp_default_scalars_graph():
    args = (torch.randn(8), torch.randn(4, 16), torch.randn(16, 8))
    gm = _trace_addmm(1, 1)(*args)
    targets = [n.target for n in gm.graph.nodes if n.op == "call_function"]
    assert targets == [torch.ops.aten.mm.default, torch.ops.aten.add.Tensor]
    torch.testing.assert_close(gm(*args), torch.addmm(*args))


def test_addmm_decomp_scalars():
    args = (torch.randn(8), torch.randn(4, 16), torch.randn(16, 8))
    for beta, alpha in [(0.5, 2.0), (1, 3.0), (-2.0, 1)]:
        gm = _trace_addmm(beta, alpha)(*args)
        torch.testing.assert_close(
            gm(*args), torch.addmm(*args, beta=beta, alpha=alpha)
        )

    nan_bias = torch.full((8,), float("nan"))
    gm = _trace_addmm(0, 2.0)(nan_bias, *args[1:])
    out = gm(nan_bias, *args[1:])
    torch.testing.assert_close(out, torch.addmm(nan_bias, *args[1:], beta=0, alpha=2.0))
    assert not out.isnan().any()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=requires_cuda)])
def test_addmm_decomp_fp16_scales_in_opmath(device):
    args = [
        torch.tensor(t, dtype=torch.float16, device=device)
        for t in ([[-40000.0]], [[40000.0]], [[1.0]])
    ]
    out = _trace_addmm(1, 2)(*args)(*args)
    assert out.dtype == torch.float16
    torch.testing.assert_close(out, torch.addmm(*args, alpha=2))


def test_addmm_decomp_integral_truncates_scalars():
    # aten.addmm has no int8 CUDA kernel, so this is CPU only
    args = [torch.tensor(t, dtype=torch.int8) for t in ([7], [[2, 3]], [[4], [5]])]
    out = _trace_addmm(0.5, 2.0)(*args)(*args)
    ref = torch.addmm(*args, beta=0.5, alpha=2.0)
    assert out.dtype == ref.dtype == torch.int8
    torch.testing.assert_close(out, ref)


def _trace_uniform(x, tracing_mode="real", **kwargs):
    def f(t):
        return torch.ops.aten.uniform(t, -1.0, 2.0, **kwargs)

    return make_fx(
        f, decomposition_table=_get_decomp_table(), tracing_mode=tracing_mode
    )(x)


def _call_targets(gm):
    return {n.target for n in gm.graph.nodes if n.op == "call_function"}


@pytest.mark.parametrize("tracing_mode", ["real", "symbolic"])
@pytest.mark.parametrize(
    "x",
    [
        torch.empty(()),
        torch.empty(1),
        torch.empty(4, 8),
        torch.empty(8, 4).t(),
        torch.empty(2, 3, 4, 5, memory_format=torch.channels_last),
        torch.empty(4, 1, 8).transpose(0, 1),
    ],
    ids=["0d", "one_element", "contiguous", "transposed", "channels_last", "unit_dim"],
)
def test_uniform_decomp_keeps_dense_layout_without_as_strided(x, tracing_mode):
    gm = _trace_uniform(x, tracing_mode)
    targets = _call_targets(gm)
    assert torch.ops.aten.as_strided.default not in targets
    assert torch.ops.aten.rand.default in targets

    out = gm(x)
    assert out.shape == x.shape
    # Strides of size-1 dims don't constrain later views
    assert [s for s, n in zip(out.stride(), x.shape) if n > 1] == [
        s for s, n in zip(x.stride(), x.shape) if n > 1
    ]
    assert ((out >= -1.0) & (out < 2.0)).all()


def test_uniform_decomp_non_dense_falls_back_to_inductor():
    x = torch.empty(4, 8)[:, :4]
    gm = _trace_uniform(x)
    assert torch.ops.aten.as_strided.default in _call_targets(gm)
    assert gm(x).stride() == x.stride()


def test_uniform_decomp_generator():
    x = torch.empty(4, 8)
    generator = torch.Generator()
    gm = _trace_uniform(x, generator=generator)
    assert torch.ops.aten.rand.generator in _call_targets(gm)
    generator.manual_seed(0)
    out = gm(x)
    generator.manual_seed(0)
    torch.testing.assert_close(gm(x), out)
