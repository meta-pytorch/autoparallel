# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from test_correctness import run_correctness_test
from torch.distributed.tensor.placement_types import Shard

from autoparallel.context_parallel import make_context_parallel_sdpa


class AttentionKernel(nn.Module):
    def __init__(self, mesh):
        super().__init__()
        self.cp_sdpa = (
            make_context_parallel_sdpa(mesh, is_causal=False)
            if mesh is not None
            else None
        )

    def forward(self, q, k, v):
        if self.cp_sdpa is None:
            return F.scaled_dot_product_attention(q, k, v, is_causal=False)
        return self.cp_sdpa(q, k, v)


@pytest.mark.parametrize(
    "mesh_shape,mesh_dim_names,qkv_placements",
    (
        (
            (2, 2, 2),
            ("dp_shard", "cp", "tp"),
            (Shard(0), Shard(2), Shard(1)),
        ),
        (
            (2, 2),
            ("dp_shard", "cp"),
            (Shard(0), Shard(2)),
        ),
        (
            (2, 2),
            ("dp_shard", "tp"),
            (Shard(0), Shard(1)),
        ),
        (
            (2, 2, 2, 2),
            ("dp_replicate", "dp_shard", "cp", "tp"),
            (Shard(0), Shard(0), Shard(2), Shard(1)),
        ),
    ),
)
def test_context_parallel_attention_correctness(
    mesh_shape,
    mesh_dim_names,
    qkv_placements,
):
    batch_size = 8
    nheads = 4
    seq_len = 8
    head_dim = 4

    mesh = torch.distributed.device_mesh.init_device_mesh(
        "cuda", mesh_shape, mesh_dim_names=mesh_dim_names
    )

    def model_fn():
        return AttentionKernel(mesh)

    def reference_model_fn():
        return AttentionKernel(None)

    def input_fn():
        return (
            torch.randn(
                batch_size,
                nheads,
                seq_len,
                head_dim,
                device="cuda",
                requires_grad=True,
            ),
            torch.randn(
                batch_size,
                nheads,
                seq_len,
                head_dim,
                device="cuda",
                requires_grad=True,
            ),
            torch.randn(
                batch_size,
                nheads,
                seq_len,
                head_dim,
                device="cuda",
                requires_grad=True,
            ),
        )

    run_correctness_test(
        model_fn,
        input_fn,
        mesh_shape,
        mesh_dim_names=mesh_dim_names,
        input_placements=(qkv_placements, qkv_placements, qkv_placements),
        output_placements=qkv_placements,
        reference_model_fn=reference_model_fn,
        parameter_memory_constraint=False,
        atol=1e-4,
        rtol=1e-4,
    )
