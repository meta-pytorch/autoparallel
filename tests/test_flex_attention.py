# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from autoparallel.api import auto_parallel
from autoparallel.shardings.placement_options import reset_placement_options_cache

DIM = 128
N_HEADS = 8
HEAD_DIM = DIM // N_HEADS
LOCAL_BS = 64
SEQLEN = 16


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


class FlexAttnScoreModModel(torch.nn.Module):
    """Model with a custom score_mod (ALiBi-style position bias)."""

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

        def alibi(score, batch, head, q_idx, kv_idx):
            return score - (q_idx - kv_idx).abs().float()

        out = flex_attention(q, k, v, score_mod=alibi)
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(out)


class FlexAttnBlockMaskModel(torch.nn.Module):
    """Model with a causal block_mask."""

    def __init__(self, dim, n_heads, seqlen, batch_size=None):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wq = torch.nn.Linear(dim, dim, bias=False)
        self.wk = torch.nn.Linear(dim, dim, bias=False)
        self.wv = torch.nn.Linear(dim, dim, bias=False)
        self.wo = torch.nn.Linear(dim, dim, bias=False)

        def causal(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        self.block_mask = create_block_mask(
            causal,
            B=batch_size,
            H=n_heads if batch_size is not None else None,
            Q_LEN=seqlen,
            KV_LEN=seqlen,
            device="cuda",
        )

    def forward(self, x):
        bsz, seqlen, _ = x.shape
        q = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        out = flex_attention(q, k, v, block_mask=self.block_mask)
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(out)


class FlexAttnScaleModel(torch.nn.Module):
    """Model with explicit scale parameter."""

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
        out = flex_attention(q, k, v, scale=0.5)
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(out)


class FlexAttnGQAModel(torch.nn.Module):
    """Model with Grouped Query Attention (fewer KV heads than Q heads)."""

    def __init__(self, dim, n_heads, n_kv_heads):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.wq = torch.nn.Linear(dim, dim, bias=False)
        self.wk = torch.nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = torch.nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        bsz, seqlen, _ = x.shape
        q = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)
        out = flex_attention(q, k, v, enable_gqa=True)
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(out)


class FlexAttnOtherBuffersModel(torch.nn.Module):
    """Model where score_mod captures external tensors (other_buffers).

    Uses a buffer with shape [B, H] which coincidentally matches the batch and
    head dimensions of Q. Without explicit other_buffers handling, this would
    be misclassified as an attention tensor and incorrectly sharded.
    """

    def __init__(self, dim, n_heads, batch_size):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wq = torch.nn.Linear(dim, dim, bias=False)
        self.wk = torch.nn.Linear(dim, dim, bias=False)
        self.wv = torch.nn.Linear(dim, dim, bias=False)
        self.wo = torch.nn.Linear(dim, dim, bias=False)
        # Shape [B, H] — deliberately matches attention tensor dims
        self.per_batch_head_bias = torch.nn.Parameter(
            torch.randn(batch_size, n_heads), requires_grad=False
        )

    def forward(self, x):
        bsz, seqlen, _ = x.shape
        q = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        bias = self.per_batch_head_bias

        def score_mod(score, batch, head, q_idx, kv_idx):
            return score + bias[batch, head]

        out = flex_attention(q, k, v, score_mod=score_mod)
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(out)


class FlexAttnScoreModBlockMaskModel(torch.nn.Module):
    """Model with both score_mod and block_mask together."""

    def __init__(self, dim, n_heads, seqlen):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wq = torch.nn.Linear(dim, dim, bias=False)
        self.wk = torch.nn.Linear(dim, dim, bias=False)
        self.wv = torch.nn.Linear(dim, dim, bias=False)
        self.wo = torch.nn.Linear(dim, dim, bias=False)

        def causal(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        self.block_mask = create_block_mask(
            causal, B=None, H=None, Q_LEN=seqlen, KV_LEN=seqlen, device="cuda"
        )

    def forward(self, x):
        bsz, seqlen, _ = x.shape
        q = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)

        def alibi(score, batch, head, q_idx, kv_idx):
            return score - (q_idx - kv_idx).abs().float()

        out = flex_attention(
            q, k, v, score_mod=alibi, block_mask=self.block_mask, scale=0.25
        )
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(out)


@pytest.fixture(autouse=True)
def _clear_caches():
    reset_placement_options_cache()
    yield
    reset_placement_options_cache()


def _make_input(mesh, dim):
    placements = [Shard(0)] + [Replicate()] * (mesh.ndim - 1)
    return DTensor.from_local(
        torch.randn(LOCAL_BS, SEQLEN, dim, device="cuda"),
        mesh,
        placements,
    )


def _out_shardings(mesh):
    return tuple([Shard(0)] + [Replicate()] * (mesh.ndim - 1))


def _run_auto_parallel(model, mesh, dim=DIM):
    x = _make_input(mesh, dim)
    parallel_model = auto_parallel(
        model,
        mesh,
        sample_inputs=(x,),
        out_shardings=_out_shardings(mesh),
        compile=False,
    )
    assert parallel_model is not None
    return parallel_model


def test_flex_attention_1d_mesh(device_mesh_1d):
    with torch.device("meta"):
        model = FlexAttnModel(DIM, N_HEADS)
    _run_auto_parallel(model, device_mesh_1d)


def test_flex_attention_2d_mesh(device_mesh_2d):
    with torch.device("meta"):
        model = FlexAttnModel(DIM, N_HEADS)
    _run_auto_parallel(model, device_mesh_2d)


def test_flex_attention_score_mod(device_mesh_1d):
    with torch.device("meta"):
        model = FlexAttnScoreModModel(DIM, N_HEADS)
    _run_auto_parallel(model, device_mesh_1d)


def test_flex_attention_score_mod_2d(device_mesh_2d):
    with torch.device("meta"):
        model = FlexAttnScoreModModel(DIM, N_HEADS)
    _run_auto_parallel(model, device_mesh_2d)


def test_flex_attention_block_mask(device_mesh_1d):
    model = FlexAttnBlockMaskModel(DIM, N_HEADS, SEQLEN)
    model = model.to("meta")
    _run_auto_parallel(model, device_mesh_1d)


def test_flex_attention_block_mask_2d(device_mesh_2d):
    model = FlexAttnBlockMaskModel(DIM, N_HEADS, SEQLEN)
    model = model.to("meta")
    _run_auto_parallel(model, device_mesh_2d)


def test_flex_attention_block_mask_explicit_bh(device_mesh_1d):
    """block_mask with B=global_batch and H=n_heads (non-broadcast masks)."""
    global_bs = LOCAL_BS * device_mesh_1d.size()
    model = FlexAttnBlockMaskModel(DIM, N_HEADS, SEQLEN, batch_size=global_bs)
    model = model.to("meta")
    _run_auto_parallel(model, device_mesh_1d)


def test_flex_attention_block_mask_sharding_matches_shape(device_mesh_1d):
    """Verify block_mask tensors are shardable only on dims that match Q.

    - B=global_batch, H=n_heads: block_mask can be sharded on both batch and heads.
    - B=None, H=n_heads: block_mask has shape[0]=1, so batch dim must stay
      Replicate, but head dim (shape[1]=H) can still be Shard(1).
    - B=None, H=None: block_mask shape is [1,1,...], must be fully Replicate.
    """
    global_bs = LOCAL_BS * device_mesh_1d.size()
    mesh = device_mesh_1d

    from autoparallel.api import AutoParallel
    from autoparallel.input_validation import (
        _extract_input_info,
        _flatten_out_shardings,
        _make_input_fn,
    )

    def _get_flex_attn_strat(model):
        x = _make_input(mesh, DIM)
        shapes, dtypes, input_placements, treespec = _extract_input_info((x,), mesh)
        input_fn = _make_input_fn(shapes, dtypes, treespec)
        output_placements = _flatten_out_shardings(_out_shardings(mesh))

        with AutoParallel(model, input_fn, mesh, compile=False) as autop:
            autop.add_input_constraints(input_placements)
            autop.add_output_constraints(output_placements)

            for node in autop.gm.graph.nodes:
                if node.op != "call_function":
                    continue
                if not isinstance(node.target, torch._ops.HigherOrderOperator):
                    continue
                if "backward" in node.target.name():
                    continue
                return autop.sharding_optimizer.strats[node]
        raise AssertionError("flex_attention node not found")

    def _block_mask_specs(strat):
        """Collect input specs for block_mask tensors (not Q/K/V, not GraphModule)."""
        specs = []
        for s in strat.strategies:
            for ispec in s.input_specs:
                if ispec is None:
                    continue
                # Block_mask tensors have a trailing dim of size 1
                if ispec.tensor_meta.shape[-1] == 1:
                    specs.append(ispec)
        return specs

    # B=global_batch, H=n_heads: block_mask can be Shard(0)
    model = FlexAttnBlockMaskModel(DIM, N_HEADS, SEQLEN, batch_size=global_bs)
    model = model.to("meta")
    bm_specs = _block_mask_specs(_get_flex_attn_strat(model))
    assert any(
        any(p == Shard(0) for p in spec.placements) for spec in bm_specs
    ), "block_mask with explicit B should allow Shard(0)"

    # B=None, H=None: block_mask shape [1,1,...] must be fully Replicate
    model = FlexAttnBlockMaskModel(DIM, N_HEADS, SEQLEN)
    model = model.to("meta")
    bm_specs = _block_mask_specs(_get_flex_attn_strat(model))
    for spec in bm_specs:
        assert all(
            p == Replicate() for p in spec.placements
        ), f"block_mask with B=None, H=None should be Replicate, got {spec.placements}"


def test_flex_attention_scale(device_mesh_1d):
    with torch.device("meta"):
        model = FlexAttnScaleModel(DIM, N_HEADS)
    _run_auto_parallel(model, device_mesh_1d)


def test_flex_attention_gqa(device_mesh_1d):
    n_kv_heads = 2
    with torch.device("meta"):
        model = FlexAttnGQAModel(DIM, N_HEADS, n_kv_heads)
    _run_auto_parallel(model, device_mesh_1d)


def test_flex_attention_gqa_2d(device_mesh_2d):
    n_kv_heads = 2
    with torch.device("meta"):
        model = FlexAttnGQAModel(DIM, N_HEADS, n_kv_heads)
    _run_auto_parallel(model, device_mesh_2d)


def test_flex_attention_score_mod_block_mask(device_mesh_1d):
    model = FlexAttnScoreModBlockMaskModel(DIM, N_HEADS, SEQLEN)
    model = model.to("meta")
    _run_auto_parallel(model, device_mesh_1d)


def test_flex_attention_score_mod_block_mask_2d(device_mesh_2d):
    model = FlexAttnScoreModBlockMaskModel(DIM, N_HEADS, SEQLEN)
    model = model.to("meta")
    _run_auto_parallel(model, device_mesh_2d)


def test_flex_attention_other_buffers(device_mesh_1d):
    """score_mod capturing a [B, H] tensor as other_buffer."""
    global_bs = LOCAL_BS * device_mesh_1d.size()
    model = FlexAttnOtherBuffersModel(DIM, N_HEADS, global_bs)
    model = model.to("meta")
    _run_auto_parallel(model, device_mesh_1d)


def test_flex_attention_other_buffers_replicated(device_mesh_1d):
    """Verify other_buffers with shape [B, H] are Replicate, not co-sharded.

    The per_batch_head_bias has shape [B, H] which matches the attention tensor
    dimensions. Without explicit other_buffers handling, tensor_placement would
    assign it the same Shard(0)/Shard(1) placement as Q — but score_mod indexes
    it arbitrarily, so it must be Replicate.
    """
    global_bs = LOCAL_BS * device_mesh_1d.size()
    model = FlexAttnOtherBuffersModel(DIM, N_HEADS, global_bs)
    model = model.to("meta")
    mesh = device_mesh_1d

    from autoparallel.api import AutoParallel
    from autoparallel.input_validation import (
        _extract_input_info,
        _flatten_out_shardings,
        _make_input_fn,
    )

    x = _make_input(mesh, DIM)
    shapes, dtypes, input_placements, treespec = _extract_input_info((x,), mesh)
    input_fn = _make_input_fn(shapes, dtypes, treespec)
    output_placements = _flatten_out_shardings(_out_shardings(mesh))

    with AutoParallel(model, input_fn, mesh, compile=False) as autop:
        autop.add_input_constraints(input_placements)
        autop.add_output_constraints(output_placements)

        strats = autop.sharding_optimizer.strats
        for node in autop.gm.graph.nodes:
            if node.op != "call_function":
                continue
            if not isinstance(node.target, torch._ops.HigherOrderOperator):
                continue
            if "backward" in node.target.name():
                continue

            strat = strats[node]
            q_shape = strat.strategies[0].input_specs[0].tensor_meta.shape
            B, H = q_shape[0], q_shape[1]

            # Find the [B, H] other_buffer and verify it's always Replicate.
            for si, s in enumerate(strat.strategies):
                for i, ispec in enumerate(s.input_specs):
                    if ispec is None:
                        continue
                    if ispec.tensor_meta.shape == torch.Size([B, H]):
                        assert all(p == Replicate() for p in ispec.placements), (
                            f"Strategy {si}, other_buffer Input {i} "
                            f"(shape [{B}, {H}]) should be Replicate "
                            f"but got {ispec.placements}"
                        )
