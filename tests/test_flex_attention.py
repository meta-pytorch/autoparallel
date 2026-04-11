# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.nn.attention.flex_attention import flex_attention

from autoparallel.api import auto_parallel
from autoparallel.shardings.placement_options import reset_placement_options_cache


class FlexAttnModel(torch.nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wq = torch.nn.Linear(dim, dim, bias=False)
        self.wk = torch.nn.Linear(dim, dim, bias=False)
        self.wv = torch.nn.Linear(dim, dim, bias=False)
        self.wo = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        bsz, seqlen, _ = x.shape
        q = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        out = flex_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(out)


@pytest.fixture(autouse=True)
def _clear_caches():
    reset_placement_options_cache()
    yield
    reset_placement_options_cache()


def test_flex_attention_1d_mesh(device_mesh_1d):
    dim = 128
    n_heads = 8
    local_bs = 64

    with torch.device("meta"):
        model = FlexAttnModel(dim, n_heads)

    x = DTensor.from_local(
        torch.randn(local_bs, 16, dim, device="cuda"),
        device_mesh_1d,
        [Shard(0)],
    )

    parallel_model = auto_parallel(
        model,
        device_mesh_1d,
        sample_inputs=(x,),
        out_shardings=(Shard(0),),
        compile=False,
    )
    assert parallel_model is not None


def test_flex_attention_2d_mesh(device_mesh_2d):
    dim = 128
    n_heads = 8
    local_bs = 64

    with torch.device("meta"):
        model = FlexAttnModel(dim, n_heads)

    x = DTensor.from_local(
        torch.randn(local_bs, 16, dim, device="cuda"),
        device_mesh_2d,
        [Shard(0), Replicate()],
    )

    parallel_model = auto_parallel(
        model,
        device_mesh_2d,
        sample_inputs=(x,),
        out_shardings=(Shard(0), Replicate()),
        compile=False,
    )
    assert parallel_model is not None
