# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import sys
from pathlib import Path

import pytest
import torch
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Replicate, Shard

from autoparallel._testing.models.qwen3 import (
    Qwen3ModelArgs,
    Transformer,
    apply_rotary_emb_cos_sin,
    qwen3_debug_args,
    qwen3_args_from_torchtitan_config,
    qwen3_moe_debug_args,
)
from autoparallel.api import AutoParallel, auto_parallel


def _tiny_args(**overrides) -> Qwen3ModelArgs:
    args = Qwen3ModelArgs(
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        hidden_dim=128,
        vocab_size=128,
        max_seq_len=16,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    args.__post_init__()
    return args


def _tiny_moe_args(**overrides) -> Qwen3ModelArgs:
    args = Qwen3ModelArgs(
        dim=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        head_dim=8,
        hidden_dim=64,
        vocab_size=64,
        max_seq_len=4,
        moe_enabled=True,
        moe_hidden_dim=16,
        num_experts=64,
        top_k=8,
        route_norm=True,
        score_before_experts=False,
        moe_axis_name="tp",
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    args.__post_init__()
    return args


def test_qwen3_forward_shape():
    args = _tiny_args()
    model = Transformer(args)
    model.init_weights(seed=0)

    tokens = torch.randint(0, args.vocab_size, (2, args.max_seq_len))
    logits = model(tokens)

    assert logits.shape == (2, args.max_seq_len, args.vocab_size)


def test_qwen3_qk_norm_changes_logits():
    args = _tiny_args(n_layers=1)
    model = Transformer(args)
    model.init_weights(seed=0)

    tokens = torch.randint(0, args.vocab_size, (2, args.max_seq_len))
    logits = model(tokens)

    with torch.no_grad():
        model.layers["0"].attention.q_norm.weight.zero_()
    logits_without_q = model(tokens)

    assert not torch.allclose(logits, logits_without_q)


def test_qwen3_weight_tying_survives_init_weights():
    args = _tiny_args(enable_weight_tying=True)
    model = Transformer(args)

    assert model.tok_embeddings.weight is model.lm_head.weight
    model.init_weights(seed=0)
    assert model.tok_embeddings.weight is model.lm_head.weight


def test_qwen3_debug_args_matches_torchtitan_dense_shape():
    args = qwen3_debug_args(max_seq_len=32)

    assert args.dim == 256
    assert args.n_layers == 8
    assert args.n_heads == 16
    assert args.n_kv_heads == 8
    assert args.head_dim == 128
    assert args.hidden_dim == 3072
    assert args.vocab_size == 2048
    assert args.rope_theta == 1000000.0
    assert args.enable_weight_tying


def test_qwen3_moe_debug_args_matches_torchtitan_shape():
    args = qwen3_moe_debug_args(max_seq_len=32)

    assert args.dim == 256
    assert args.n_layers == 8
    assert args.n_heads == 16
    assert args.n_kv_heads == 8
    assert args.head_dim == 128
    assert args.moe_enabled
    assert args.moe_hidden_dim == 768
    assert args.num_experts == 64
    assert args.top_k == 8
    assert args.route_norm
    assert not args.score_before_experts


@pytest.mark.parametrize(
    ("flavor", "expected"),
    [
        (
            "8B",
            {
                "dim": 4096,
                "n_layers": 36,
                "n_heads": 32,
                "n_kv_heads": 8,
                "head_dim": 128,
                "hidden_dim": 12288,
                "vocab_size": 151936,
                "moe_enabled": False,
                "num_experts": 0,
                "top_k": 1,
                "max_seq_len": 4096,
            },
        ),
        (
            "30B-A3B",
            {
                "dim": 2048,
                "n_layers": 48,
                "n_heads": 32,
                "n_kv_heads": 4,
                "head_dim": 128,
                "hidden_dim": 0,
                "vocab_size": 151936,
                "moe_enabled": True,
                "moe_hidden_dim": 768,
                "num_experts": 128,
                "top_k": 8,
                "route_norm": True,
                "score_before_experts": False,
                "max_seq_len": 262144,
            },
        ),
    ],
)
def test_qwen3_args_from_torchtitan_config(flavor, expected):
    torchtitan_root = Path(__file__).resolve().parents[2] / "torchtitan"
    if not torchtitan_root.exists():
        pytest.skip("torchtitan sibling checkout not found")
    sys.path.insert(0, str(torchtitan_root))

    try:
        from torchtitan.models.qwen3 import qwen3_configs  # type: ignore[import-not-found]
    except Exception as exc:
        pytest.skip(f"torchtitan Qwen3 config unavailable: {exc}")

    args = qwen3_args_from_torchtitan_config(
        qwen3_configs[flavor](attn_backend="sdpa")
    )

    for attr, value in expected.items():
        assert getattr(args, attr) == value
    assert args.rope_theta == 1000000.0
    assert args.norm_eps == 1e-6


def test_qwen3_cos_sin_rope_matches_torchtitan_helper():
    torchtitan_root = Path(__file__).resolve().parents[2] / "torchtitan"
    if not torchtitan_root.exists():
        pytest.skip("torchtitan sibling checkout not found")
    sys.path.insert(0, str(torchtitan_root))

    try:
        from torchtitan.models.common.rope import (  # type: ignore[import-not-found]
            RoPE,
            apply_rotary_emb_cos_sin as tt_apply_rotary_emb_cos_sin,
        )
    except Exception as exc:
        pytest.skip(f"torchtitan Qwen3 RoPE helper unavailable: {exc}")

    args = _tiny_args()
    rope = RoPE(
        RoPE.Config(
            dim=args.head_dim,
            max_seq_len=args.max_seq_len,
            theta=args.rope_theta,
            backend="cos_sin",
        )
    )
    xq = torch.randn(2, args.max_seq_len, args.n_heads, args.head_dim)
    xk = torch.randn(2, args.max_seq_len, args.n_kv_heads, args.head_dim)

    actual = apply_rotary_emb_cos_sin(xq, xk, rope.cache)
    expected = tt_apply_rotary_emb_cos_sin(xq, xk, rope.cache)

    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_qwen3_autoparallel_pipeline_smoke(device_mesh_2d):
    args = _tiny_args(n_layers=2, max_seq_len=8)
    batch_size = 2 * device_mesh_2d.shape[0]

    with torch.device("meta"):
        model = Transformer(args)

    def input_fn():
        return torch.randint(
            0,
            args.vocab_size,
            (batch_size, args.max_seq_len),
            device="cuda",
        )

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )

    with AutoParallel(
        model,
        input_fn,
        device_mesh_2d,
        mp_policy,
        repeated_subgraphs=True,
    ) as autop:
        autop.add_input_constraints([(Shard(0), Replicate())])
        autop.add_output_constraints([(Shard(0), Shard(2))])
        sharding_placement = autop.optimize_placement(verbose=False)
        parallel_mod = autop.apply_placement(sharding_placement)

    assert isinstance(parallel_mod, Transformer)


def test_qwen3_moe_auto_parallel_smoke(device_mesh_2d):
    args = _tiny_moe_args()
    local_batch_size = 1

    with torch.device("meta"):
        model = Transformer(args, mesh=device_mesh_2d, moe_axis_name="tp")

    expected_param_shapes = {
        name: tuple(param.shape) for name, param in model.named_parameters()
    }
    expected_nparams = sum(param.numel() for param in model.parameters())

    tokens = DTensor.from_local(
        torch.randint(
            0,
            args.vocab_size,
            (local_batch_size, args.max_seq_len),
            device="cuda",
        ),
        device_mesh_2d,
        [Shard(0), Shard(0)],
    )

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )
    parallel_mod = auto_parallel(
        model,
        device_mesh_2d,
        sample_inputs=(tokens,),
        out_shardings=(Shard(0), Shard(2)),
        mp_policy=mp_policy,
        dynamic=True,
    )

    assert isinstance(parallel_mod, Transformer)
    assert sum(param.numel() for param in parallel_mod.parameters()) == expected_nparams
    assert {
        name: tuple(param.shape) for name, param in parallel_mod.named_parameters()
    } == expected_param_shapes
    assert parallel_mod.layers["0"].moe.experts.w1.shape == (
        args.num_experts,
        args.moe_hidden_dim,
        args.dim,
    )

    parallel_mod.to_empty(device="cuda")
    parallel_mod.init_weights(buffer_device=torch.device("cuda"), seed=0)

    local_tokens = torch.randint(
        0,
        args.vocab_size,
        (local_batch_size, args.max_seq_len),
        device="cuda",
    )
    out = parallel_mod(local_tokens)
    assert out.shape == (
        local_batch_size * device_mesh_2d.shape[1],
        args.max_seq_len,
        args.vocab_size // device_mesh_2d.shape[1],
    )
    out.backward(torch.randn_like(out))
