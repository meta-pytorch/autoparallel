# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from conftest import apply_cuda_patches
from test_correctness import _run_correctness_test
from torch.distributed.tensor.placement_types import Shard

import autoparallel
from autoparallel import (
    context_parallel_attention_placements,
    make_context_parallel_sdpa,
)


def _fake_mesh(mesh_dim_names=None, ndim=None):
    if ndim is None:
        ndim = len(mesh_dim_names)
    return SimpleNamespace(mesh_dim_names=mesh_dim_names, ndim=ndim)


def test_context_parallel_api_is_exported_from_autoparallel():
    assert hasattr(autoparallel, "ContextParallelPlacements")
    assert callable(autoparallel.context_parallel_attention_placements)
    assert callable(autoparallel.context_parallel_local_map)
    assert callable(autoparallel.make_context_parallel_sdpa)


@apply_cuda_patches
def test_llama_build_attention_accepts_context_parallel_mesh():
    from autoparallel._testing.models.llama3 import build_attention

    mesh = torch.distributed.device_mesh.init_device_mesh(
        "cuda", (2, 2), mesh_dim_names=("cp", "tp")
    )

    attention = build_attention(
        use_flex_attn=False,
        attn_mask_type="causal",
        context_parallel_mesh=mesh,
    )

    assert callable(attention)


@pytest.mark.parametrize(
    "kwargs,match",
    (
        ({"use_flex_attn": True}, "FlexAttention"),
        ({"fixed_block_size": 128}, "fixed_block_size"),
        ({"attn_mask_type": "document"}, "causal"),
    ),
)
def test_llama_build_attention_rejects_unsupported_context_parallel_modes(
    kwargs, match
):
    from autoparallel._testing.models.llama3 import build_attention

    args = {
        "use_flex_attn": False,
        "attn_mask_type": "causal",
        "context_parallel_mesh": _fake_mesh(("cp", "tp")),
    }
    args.update(kwargs)

    with pytest.raises(ValueError, match=match):
        build_attention(**args)


@pytest.mark.parametrize(
    "mesh_dim_names,expected_placements",
    (
        (("dp", "context", "tensor"), (Shard(0), Shard(2), Shard(1))),
        (
            ("fsdp", "context_parallel", "tensor_parallel"),
            (Shard(0), Shard(2), Shard(1)),
        ),
        (
            ("ddp", "data_parallel", "cp", "tp"),
            (Shard(0), Shard(0), Shard(2), Shard(1)),
        ),
        (("cp", "tp"), (Shard(2), Shard(1))),
        (("tp", "cp"), (Shard(1), Shard(2))),
        (("dp_shard", "tp"), (Shard(0), Shard(1))),
    ),
)
def test_context_parallel_attention_placement_names(
    mesh_dim_names, expected_placements
):
    placements = context_parallel_attention_placements(
        _fake_mesh(mesh_dim_names),
        batch_dim=0,
        seq_dim=2,
        head_dim=1,
    )

    assert placements.qkv == expected_placements
    assert placements.in_placements == (expected_placements,) * 3
    assert placements.out_placements == (expected_placements,)


@pytest.mark.parametrize(
    "mesh,expected_placements",
    (
        (_fake_mesh(None, ndim=3), (Shard(0), Shard(2), Shard(1))),
        (_fake_mesh(None, ndim=4), (Shard(0), Shard(0), Shard(2), Shard(1))),
    ),
)
def test_context_parallel_attention_placement_defaults(mesh, expected_placements):
    placements = context_parallel_attention_placements(
        mesh,
        batch_dim=0,
        seq_dim=2,
        head_dim=1,
    )

    assert placements.qkv == expected_placements


def test_context_parallel_attention_placements_reject_invalid_axis():
    with pytest.raises(ValueError, match="Unsupported mesh axis"):
        context_parallel_attention_placements(_fake_mesh(("dp", "bad")))


@pytest.mark.parametrize("ndim", (1, 2))
def test_context_parallel_attention_placements_reject_unnamed_ambiguous_meshes(ndim):
    with pytest.raises(ValueError, match="requires mesh_dim_names"):
        context_parallel_attention_placements(_fake_mesh(None, ndim=ndim))


def test_make_context_parallel_sdpa_rejects_context_parallel_dropout():
    with pytest.raises(ValueError, match="dropout"):
        make_context_parallel_sdpa(_fake_mesh(("cp", "tp")), dropout_p=0.1)


@apply_cuda_patches
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
            ("cp", "tp"),
            (Shard(2), Shard(1)),
        ),
        (
            (2, 2),
            ("tp", "cp"),
            (Shard(1), Shard(2)),
        ),
        (
            (2, 2, 2, 2),
            ("dp_replicate", "dp_shard", "cp", "tp"),
            (Shard(0), Shard(0), Shard(2), Shard(1)),
        ),
    ),
)
def test_context_parallel_attention_placements_are_qkv_sharded(
    mesh_shape,
    mesh_dim_names,
    qkv_placements,
):
    mesh = torch.distributed.device_mesh.init_device_mesh(
        "cuda", mesh_shape, mesh_dim_names=mesh_dim_names
    )

    placements = context_parallel_attention_placements(
        mesh,
        batch_dim=0,
        seq_dim=2,
        head_dim=1,
    )

    assert placements.in_placements == (qkv_placements,) * 3
    assert placements.out_placements == (qkv_placements,)


class AttentionKernel(nn.Module):
    def __init__(self, mesh, *, is_causal=False):
        super().__init__()
        self.cp_sdpa = (
            make_context_parallel_sdpa(mesh, is_causal=is_causal)
            if mesh is not None
            else None
        )
        self.is_causal = is_causal

    def forward(self, q, k, v):
        if self.cp_sdpa is None:
            return F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal)
        return self.cp_sdpa(q, k, v)


@pytest.mark.parametrize(
    "mesh_shape,mesh_dim_names,qkv_placements,is_causal",
    (
        (
            (2, 2),
            ("dp_shard", "cp"),
            (Shard(0), Shard(2)),
            False,
        ),
        (
            (2, 2),
            ("dp_shard", "cp"),
            (Shard(0), Shard(2)),
            True,
        ),
        (
            (2, 2),
            ("cp", "tp"),
            (Shard(2), Shard(1)),
            False,
        ),
        (
            (2, 2),
            ("cp", "tp"),
            (Shard(2), Shard(1)),
            True,
        ),
        (
            (2, 2, 2),
            ("dp_shard", "cp", "tp"),
            (Shard(0), Shard(2), Shard(1)),
            True,
        ),
        (
            (2, 2),
            ("dp_shard", "tp"),
            (Shard(0), Shard(1)),
            False,
        ),
    ),
)
def test_context_parallel_attention_correctness(
    mesh_shape,
    mesh_dim_names,
    qkv_placements,
    is_causal,
):
    batch_size = 8
    nheads = 4
    seq_len = 8
    head_dim = 4

    mesh = torch.distributed.device_mesh.init_device_mesh(
        "cuda", mesh_shape, mesh_dim_names=mesh_dim_names
    )

    def model_fn():
        return AttentionKernel(mesh, is_causal=is_causal)

    def reference_model_fn():
        return AttentionKernel(None, is_causal=is_causal)

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

    _run_correctness_test(
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
