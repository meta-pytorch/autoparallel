# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from conftest import apply_cuda_patches
from test_correctness import _get_parallel_graph_and_placements, _run_correctness_test
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.experimental._context_parallel import (
    _context_parallel_shard,
)
from torch.distributed.tensor.placement_types import Shard
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

import autoparallel
from autoparallel import (
    context_parallel_attention_placements,
    make_context_parallel,
    make_context_parallel_body,
    make_context_parallel_sdpa,
)


def _fake_mesh(mesh_dim_names=None, ndim=None):
    if ndim is None:
        ndim = len(mesh_dim_names)
    return SimpleNamespace(mesh_dim_names=mesh_dim_names, ndim=ndim)


def test_context_parallel_api_is_exported_from_autoparallel():
    assert hasattr(autoparallel, "ContextParallelPlacements")
    assert callable(autoparallel.context_parallel_attention_placements)
    assert callable(autoparallel.make_context_parallel)
    assert callable(autoparallel.make_context_parallel_body)
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


def test_make_context_parallel_rejects_unknown_kind():
    with pytest.raises(ValueError, match="Unsupported"):
        make_context_parallel(_fake_mesh(("cp", "tp")), kind="unknown")


def test_make_context_parallel_flex_rejects_context_parallel_score_mod():
    def score_mod(score, batch, head, q_idx, kv_idx):
        return score

    with pytest.raises(NotImplementedError, match="score_mod"):
        make_context_parallel(
            _fake_mesh(("cp", "tp")),
            kind="flex_attention",
            score_mod=score_mod,
        )


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


def _causal_block_mask(seq_len, *, batch_size=None, nheads=None, device="cuda"):
    def causal(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    return create_block_mask(
        causal,
        B=batch_size,
        H=nheads,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device,
        BLOCK_SIZE=1,
    )


@pytest.mark.parametrize(
    "mesh_shape,mesh_dim_names,qkv_placements,is_causal",
    (
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


def _local_shard_for_placements(tensor, placements, mesh_shape, coordinate):
    local = tensor
    for mesh_dim, placement in enumerate(placements):
        local = local.chunk(mesh_shape[mesh_dim], dim=placement.dim)[
            coordinate[mesh_dim]
        ].contiguous()
    return local


def _close_metrics(actual, expected, *, rank, name, atol, rtol):
    diff = (actual - expected).abs()
    tolerance = atol + rtol * expected.abs()
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    violations = ((diff > tolerance) | ~finite).sum().item()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    max_tolerance_ratio = (diff / tolerance).max().item()
    print(
        f"NUMERICS rank={rank} tensor={name} max_abs={max_abs:.9e} "
        f"mean_abs={mean_abs:.9e} "
        f"max_tolerance_ratio={max_tolerance_ratio:.9e} "
        f"violations={violations} elements={actual.numel()}",
        flush=True,
    )
    return violations


def _context_parallel_sdpa_worker(rank, case, init_file):
    world_size = case["world_size"]
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    try:
        import torch.distributed.config as dist_config

        from autoparallel._testing.models.llama3 import TransformerModelArgs

        mesh_shape = case["mesh_shape"]
        mesh = torch.distributed.device_mesh.init_device_mesh(
            "cuda", mesh_shape, mesh_dim_names=case["mesh_dim_names"]
        )
        coordinate = mesh.get_coordinate()
        assert coordinate is not None
        placements = tuple(Shard(dim) for dim in case["placement_dims"])

        class CPAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.sdpa = make_context_parallel_sdpa(
                    mesh,
                    is_causal=case["is_causal"],
                    scale=case["scale"],
                    enable_gqa=case["enable_gqa"],
                )

            def forward(self, q, k, v):
                return self.sdpa(q, k, v)

        model_args = TransformerModelArgs(n_kv_heads=8)
        batch_size = 4
        seq_len = model_args.max_seq_len
        nheads = model_args.n_heads
        head_dim = model_args.dim // nheads

        def model_fn():
            return CPAttention()

        # LLaMA3 repeat_kv expands K/V to n_heads before the SDPA call.
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

        cp_mesh = mesh["cp"] if "cp" in case["mesh_dim_names"] else mesh
        _context_parallel_shard(cp_mesh, input_fn(), (2, 2, 2), load_balancer=None)

        with dist_config.patch(compile_on_one_rank=True):
            parallel_gm = _get_parallel_graph_and_placements(
                model_fn,
                input_fn,
                mesh,
                input_placements=(placements,) * 3,
                output_placements=placements,
                parameter_memory_constraint=False,
            )[0]

        torch.manual_seed(0)
        q, k, v = input_fn()
        ref = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=case["is_causal"],
            scale=case["scale"],
            enable_gqa=case["enable_gqa"],
        )
        grad_out = torch.randn_like(ref)
        ref.backward(grad_out)

        local_inputs = tuple(
            _local_shard_for_placements(
                tensor.detach(), placements, mesh_shape, coordinate
            ).clone()
            for tensor in (q, k, v)
        )
        local_grad_out = _local_shard_for_placements(
            grad_out, placements, mesh_shape, coordinate
        )
        local_outputs, local_grads = parallel_gm(*local_inputs, local_grad_out)
        failures = []

        expected_output = _local_shard_for_placements(
            ref.detach(), placements, mesh_shape, coordinate
        )
        failures.append(
            _close_metrics(
                local_outputs[0],
                expected_output,
                rank=rank,
                name="autoparallel.output",
                atol=1e-4,
                rtol=1e-4,
            )
        )
        for name, actual, full_input in zip(("q", "k", "v"), local_grads, (q, k, v)):
            expected = _local_shard_for_placements(
                full_input.grad, placements, mesh_shape, coordinate
            )
            failures.append(
                _close_metrics(
                    actual,
                    expected,
                    rank=rank,
                    name=f"autoparallel.{name}_grad",
                    atol=1e-4,
                    rtol=1e-4,
                )
            )

        eager_local_inputs = tuple(
            _local_shard_for_placements(
                tensor.detach(), placements, mesh_shape, coordinate
            )
            .clone()
            .requires_grad_()
            for tensor in (q, k, v)
        )
        dist.barrier()
        eager_sdpa = make_context_parallel_body(
            mesh,
            is_causal=case["is_causal"],
            scale=case["scale"],
            enable_gqa=case["enable_gqa"],
        )
        eager_output = eager_sdpa(*eager_local_inputs)
        eager_output.backward(local_grad_out)
        dist.barrier()
        failures.append(
            _close_metrics(
                eager_output,
                expected_output,
                rank=rank,
                name="eager.output",
                atol=1e-4,
                rtol=1e-4,
            )
        )
        for name, local_input, full_input in zip(
            ("q", "k", "v"), eager_local_inputs, (q, k, v)
        ):
            expected = _local_shard_for_placements(
                full_input.grad, placements, mesh_shape, coordinate
            )
            failures.append(
                _close_metrics(
                    local_input.grad,
                    expected,
                    rank=rank,
                    name=f"eager.{name}_grad",
                    atol=1e-4,
                    rtol=1e-4,
                )
            )
        assert not any(failures), f"numerical mismatches: {failures}"
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "case",
    (
        {
            "world_size": 2,
            "mesh_shape": (2,),
            "mesh_dim_names": ("cp",),
            "placement_dims": (2,),
            "is_causal": True,
            "scale": None,
            "enable_gqa": False,
        },
        {
            "world_size": 2,
            "mesh_shape": (2,),
            "mesh_dim_names": ("cp",),
            "placement_dims": (2,),
            "is_causal": False,
            "scale": None,
            "enable_gqa": False,
        },
        {
            "world_size": 4,
            "mesh_shape": (2, 2),
            "mesh_dim_names": ("cp", "tp"),
            "placement_dims": (2, 1),
            "is_causal": True,
            "scale": None,
            "enable_gqa": False,
        },
    ),
    ids=("cp2_causal", "cp2_noncausal", "cp2_tp2_causal"),
)
def test_context_parallel_sdpa_real_distributed_correctness(case):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if torch.cuda.device_count() < case["world_size"]:
        pytest.skip(f"requires {case['world_size']} CUDA devices")

    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = os.path.join(tmpdir, "init")
        mp.spawn(
            _context_parallel_sdpa_worker,
            args=(case, init_file),
            nprocs=case["world_size"],
            join=True,
        )


def _context_parallel_flex_worker(rank, case, init_file):
    world_size = case["world_size"]
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    try:
        mesh_shape = case["mesh_shape"]
        placements = tuple(Shard(dim) for dim in case["placement_dims"])
        mesh = torch.distributed.device_mesh.init_device_mesh(
            "cuda", mesh_shape, mesh_dim_names=case["mesh_dim_names"]
        )
        coordinate = mesh.get_coordinate()

        batch_size = 2
        seq_len = 8
        head_dim = 4
        nheads = case["nheads"]
        nkv_heads = case["nkv_heads"]
        block_mask = (
            _causal_block_mask(
                seq_len,
                batch_size=batch_size if case["block_mask_batch"] else None,
                nheads=nheads if case["block_mask_heads"] else None,
            )
            if case["block_mask"]
            else None
        )

        torch.manual_seed(0)
        q = torch.randn(
            batch_size,
            nheads,
            seq_len,
            head_dim,
            device="cuda",
            requires_grad=True,
        )
        k = torch.randn(
            batch_size,
            nkv_heads,
            seq_len,
            head_dim,
            device="cuda",
            requires_grad=True,
        )
        v = torch.randn(
            batch_size,
            nkv_heads,
            seq_len,
            head_dim,
            device="cuda",
            requires_grad=True,
        )

        q_local = (
            _local_shard_for_placements(q.detach(), placements, mesh_shape, coordinate)
            .clone()
            .requires_grad_()
        )
        k_local = (
            _local_shard_for_placements(k.detach(), placements, mesh_shape, coordinate)
            .clone()
            .requires_grad_()
        )
        v_local = (
            _local_shard_for_placements(v.detach(), placements, mesh_shape, coordinate)
            .clone()
            .requires_grad_()
        )
        q_dtensor = DTensor.from_local(q_local, mesh, placements, run_check=False)
        k_dtensor = DTensor.from_local(k_local, mesh, placements, run_check=False)
        v_dtensor = DTensor.from_local(v_local, mesh, placements, run_check=False)

        cp_flex = make_context_parallel(
            mesh,
            kind="flex_attention",
            block_mask=block_mask,
            scale=case["scale"],
            enable_gqa=case["enable_gqa"],
        )
        with mesh:
            out = cp_flex(q_dtensor, k_dtensor, v_dtensor)

        ref = flex_attention(
            q,
            k,
            v,
            block_mask=block_mask,
            scale=case["scale"],
            enable_gqa=case["enable_gqa"],
        )
        torch.testing.assert_close(out.full_tensor(), ref, atol=1e-4, rtol=1e-4)

        out.to_local().float().sum().backward()
        ref.sum().backward()
        for local_tensor, ref_grad in (
            (q_local, q.grad),
            (k_local, k.grad),
            (v_local, v.grad),
        ):
            grad = DTensor.from_local(
                local_tensor.grad, mesh, placements, run_check=False
            ).full_tensor()
            torch.testing.assert_close(grad, ref_grad, atol=1e-4, rtol=1e-4)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "case",
    (
        {
            "world_size": 2,
            "mesh_shape": (2,),
            "mesh_dim_names": ("cp",),
            "placement_dims": (2,),
            "block_mask": False,
            "block_mask_batch": False,
            "block_mask_heads": False,
            "scale": None,
            "enable_gqa": False,
            "nheads": 4,
            "nkv_heads": 4,
        },
        {
            "world_size": 4,
            "mesh_shape": (2, 2),
            "mesh_dim_names": ("cp", "tp"),
            "placement_dims": (2, 1),
            "block_mask": True,
            "block_mask_batch": False,
            "block_mask_heads": True,
            "scale": 0.5,
            "enable_gqa": True,
            "nheads": 4,
            "nkv_heads": 2,
        },
    ),
    ids=("cp2", "cp2_tp2_block_mask_gqa"),
)
def test_context_parallel_flex_attention_real_distributed_correctness(case):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if torch.cuda.device_count() < case["world_size"]:
        pytest.skip(f"requires {case['world_size']} CUDA devices")

    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = os.path.join(tmpdir, "init")
        mp.spawn(
            _context_parallel_flex_worker,
            args=(case, init_file),
            nprocs=case["world_size"],
            join=True,
        )
