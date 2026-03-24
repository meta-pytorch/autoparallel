# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.fx.experimental.proxy_tensor import make_fx

from autoparallel.graph_passes.graph_utils import _replace_view_mm_view_with_einsum


def _count_ops(gm, target):
    return len(gm.graph.find_nodes(op="call_function", target=target))


def test_forward_pattern_3d():
    """view(x, [-1,K]) -> mm(_, w) -> view(_, [B,S,N]) is replaced by einsum."""
    B, S, K, N = 2, 8, 16, 32

    def f(x, w):
        flat = torch.ops.aten.view.default(x, [B * S, K])
        out = torch.ops.aten.mm.default(flat, w)
        return torch.ops.aten.view.default(out, [B, S, N])

    x = torch.randn(B, S, K)
    w = torch.randn(K, N)
    gm = make_fx(f, tracing_mode="fake")(x, w)

    _replace_view_mm_view_with_einsum(gm)

    assert _count_ops(gm, torch.ops.aten.einsum.default) == 1
    assert _count_ops(gm, torch.ops.aten.mm.default) == 0
    assert _count_ops(gm, torch.ops.aten.view.default) == 0


def test_forward_pattern_4d():
    """Same pattern but with 4D input [B,S,T,K] -> 3 batch dims in equation."""
    B, S, T, K, N = 2, 4, 3, 16, 32

    def f(x, w):
        flat = torch.ops.aten.view.default(x, [B * S * T, K])
        out = torch.ops.aten.mm.default(flat, w)
        return torch.ops.aten.view.default(out, [B, S, T, N])

    x = torch.randn(B, S, T, K)
    w = torch.randn(K, N)
    gm = make_fx(f, tracing_mode="fake")(x, w)

    _replace_view_mm_view_with_einsum(gm)

    assert _count_ops(gm, torch.ops.aten.einsum.default) == 1
    assert _count_ops(gm, torch.ops.aten.mm.default) == 0
    assert _count_ops(gm, torch.ops.aten.view.default) == 0


def test_backward_pattern():
    """view -> permute -> mm -> permute is replaced by einsum."""
    B, S, K, N = 2, 8, 16, 32

    def f(grad_out, x):
        # grad_out: [B, S, N], x: [B, S, K]
        # flatten both, permute grad_out^T, mm, permute result
        flat_grad = torch.ops.aten.view.default(grad_out, [B * S, N])
        perm_grad = torch.ops.aten.permute.default(flat_grad, [1, 0])
        flat_x = torch.ops.aten.view.default(x, [B * S, K])
        out = torch.ops.aten.mm.default(perm_grad, flat_x)
        return torch.ops.aten.permute.default(out, [1, 0])

    grad_out = torch.randn(B, S, N)
    x = torch.randn(B, S, K)
    gm = make_fx(f, tracing_mode="fake")(grad_out, x)

    _replace_view_mm_view_with_einsum(gm)

    assert _count_ops(gm, torch.ops.aten.einsum.default) == 1
    assert _count_ops(gm, torch.ops.aten.mm.default) == 0
    assert _count_ops(gm, torch.ops.aten.view.default) == 0


def test_no_match_plain_mm():
    """Plain mm without surrounding views should be left untouched."""

    def f(a, b):
        return torch.ops.aten.mm.default(a, b)

    a = torch.randn(4, 8)
    b = torch.randn(8, 16)
    gm = make_fx(f, tracing_mode="fake")(a, b)

    _replace_view_mm_view_with_einsum(gm)

    assert _count_ops(gm, torch.ops.aten.einsum.default) == 0
    assert _count_ops(gm, torch.ops.aten.mm.default) == 1


def test_no_match_shape_mismatch():
    """view -> mm -> view where leading dims of input/output don't match."""
    K, N = 16, 32

    def f(x, w):
        # input [6, K] viewed as [6, K], output viewed as [2, 3, N]
        # leading dims: (6,) vs (2, 3) — different, so no match
        flat = torch.ops.aten.view.default(x, [6, K])
        out = torch.ops.aten.mm.default(flat, w)
        return torch.ops.aten.view.default(out, [2, 3, N])

    x = torch.randn(6, K)
    w = torch.randn(K, N)
    gm = make_fx(f, tracing_mode="fake")(x, w)

    _replace_view_mm_view_with_einsum(gm)

    assert _count_ops(gm, torch.ops.aten.einsum.default) == 0
    assert _count_ops(gm, torch.ops.aten.mm.default) == 1
