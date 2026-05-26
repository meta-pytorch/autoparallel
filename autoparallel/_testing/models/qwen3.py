# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass
from typing import Callable, ClassVar, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DeviceMesh
from torch.distributed.tensor.placement_types import Partial, Replicate, Shard
from torch.fx import traceback as fx_traceback
from torch.nn.attention import sdpa_kernel, SDPBackend

from autoparallel._testing.models.dsv3 import (
    _permute,
    _run_experts_for_loop,
    _run_experts_grouped_mm,
    _token_combine,
)
from autoparallel.collectives import all_to_all, axis_size, local_map


def has_cuda_capability(major: int, minor: int) -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (
        major,
        minor,
    )


class ScaledDotProductAttention(torch.nn.Module):
    backends: ClassVar[list[SDPBackend]] = []

    def __init__(self, attn_mask_type: str) -> None:
        super().__init__()
        if attn_mask_type != "causal":
            raise ValueError("Qwen3 with SDPA currently only supports causal mask.")

        ScaledDotProductAttention._init_backend()

    @classmethod
    def _init_backend(cls) -> None:
        if cls.backends:
            return

        cls.backends = [
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ]
        if has_cuda_capability(10, 0):
            cls.backends.insert(0, SDPBackend.CUDNN_ATTENTION)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float | None = None,
    ) -> torch.Tensor:
        assert self.backends, "SDPA backends should not be empty."
        with sdpa_kernel(self.backends, set_priority=True):
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                scale=scale,
            )


def build_attention(attn_mask_type: str):
    if attn_mask_type != "causal":
        raise ValueError("Qwen3 with SDPA currently only supports causal mask.")
    return ScaledDotProductAttention(attn_mask_type)


@dataclass
class Qwen3ModelArgs:
    dim: int = 4096
    n_layers: int = 36
    n_heads: int = 32
    n_kv_heads: Optional[int] = 8
    head_dim: int = 128
    hidden_dim: int = 12288
    vocab_size: int = 151936
    norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    max_seq_len: int = 4096
    depth_init: bool = True
    attn_mask_type: str = "causal"
    eos_id: int = 0
    enable_weight_tying: bool = False
    moe_enabled: bool = False
    moe_hidden_dim: int = 768
    num_experts: int = 64
    top_k: int = 8
    route_norm: bool = True
    route_scale: float = 1.0
    score_before_experts: bool = False
    use_grouped_mm: bool = True
    load_balance_coeff: Optional[float] = 1e-3
    moe_axis_name: str = "ep"

    def __post_init__(self) -> None:
        n_kv_heads = self.n_heads if self.n_kv_heads is None else self.n_kv_heads
        if self.n_heads % n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by "
                f"n_kv_heads ({n_kv_heads})."
            )
        if self.moe_enabled and self.top_k > self.num_experts:
            raise ValueError(
                f"top_k ({self.top_k}) must be <= num_experts ({self.num_experts})."
            )

    def update_from_config(self, job_config, tokenizer) -> None:
        self.vocab_size = tokenizer.n_words
        self.max_seq_len = job_config.training.seq_len
        self.eos_id = tokenizer.eos_id

    def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
        nparams = sum(p.numel() for p in model.parameters())
        nparams_embedding = sum(
            sum(p.numel() for p in m.parameters())
            for m in model.children()
            if isinstance(m, nn.Embedding)
        )

        l, h, q, t = (
            self.n_layers,
            self.n_heads,
            self.head_dim,
            seq_len,
        )
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        return nparams, num_flops_per_token


def qwen3_args_from_torchtitan_config(config) -> Qwen3ModelArgs:
    """Build AutoParallel Qwen3 args from TorchTitan's Qwen3Model.Config."""
    if not config.layers:
        raise ValueError("Qwen3 config must contain at least one layer.")

    first_layer = config.layers[0]
    attention = first_layer.attention
    moe = first_layer.moe

    if getattr(attention, "fuse_qkv", False):
        raise ValueError("AutoParallel Qwen3 does not support fused QKV yet.")

    moe_enabled = moe is not None
    if moe_enabled:
        hidden_dim = 0
        moe_hidden_dim = moe.experts.hidden_dim
        num_experts = moe.num_experts
        top_k = moe.router.top_k
        route_norm = moe.router.route_norm
        route_scale = moe.router.route_scale
        score_before_experts = moe.experts.token_dispatcher.score_before_experts
        load_balance_coeff = moe.load_balance_coeff
    else:
        hidden_dim = first_layer.feed_forward.w1.out_features
        moe_hidden_dim = 0
        num_experts = 0
        top_k = 1
        route_norm = True
        route_scale = 1.0
        score_before_experts = False
        load_balance_coeff = None

    return Qwen3ModelArgs(
        dim=config.dim,
        n_layers=len(config.layers),
        n_heads=attention.n_heads,
        n_kv_heads=attention.n_kv_heads,
        head_dim=attention.head_dim,
        hidden_dim=hidden_dim,
        vocab_size=config.vocab_size,
        norm_eps=config.norm.eps,
        rope_theta=config.rope.theta,
        max_seq_len=config.rope.max_seq_len,
        attn_mask_type=attention.mask_type,
        enable_weight_tying=config.enable_weight_tying,
        moe_enabled=moe_enabled,
        moe_hidden_dim=moe_hidden_dim,
        num_experts=num_experts,
        top_k=top_k,
        route_norm=route_norm,
        route_scale=route_scale,
        score_before_experts=score_before_experts,
        load_balance_coeff=load_balance_coeff,
    )


def qwen3_debug_args(**overrides) -> Qwen3ModelArgs:
    args = Qwen3ModelArgs(
        dim=256,
        n_layers=8,
        n_heads=16,
        n_kv_heads=8,
        head_dim=128,
        hidden_dim=3072,
        vocab_size=2048,
        max_seq_len=4096,
        enable_weight_tying=True,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    args.__post_init__()
    return args


def qwen3_0_6b_args(**overrides) -> Qwen3ModelArgs:
    args = Qwen3ModelArgs(
        dim=1024,
        n_layers=28,
        n_heads=16,
        n_kv_heads=8,
        head_dim=128,
        hidden_dim=3072,
        vocab_size=151936,
        enable_weight_tying=True,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    args.__post_init__()
    return args


def qwen3_1_7b_args(**overrides) -> Qwen3ModelArgs:
    args = Qwen3ModelArgs(
        dim=2048,
        n_layers=28,
        n_heads=16,
        n_kv_heads=8,
        head_dim=128,
        hidden_dim=6144,
        vocab_size=151936,
        enable_weight_tying=True,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    args.__post_init__()
    return args


def qwen3_4b_args(**overrides) -> Qwen3ModelArgs:
    args = Qwen3ModelArgs(
        dim=2560,
        n_layers=36,
        n_heads=32,
        n_kv_heads=8,
        head_dim=128,
        hidden_dim=9728,
        vocab_size=151936,
        enable_weight_tying=True,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    args.__post_init__()
    return args


def qwen3_8b_args(**overrides) -> Qwen3ModelArgs:
    args = Qwen3ModelArgs()
    for key, value in overrides.items():
        setattr(args, key, value)
    args.__post_init__()
    return args


def qwen3_moe_debug_args(**overrides) -> Qwen3ModelArgs:
    args = Qwen3ModelArgs(
        dim=256,
        n_layers=8,
        n_heads=16,
        n_kv_heads=8,
        head_dim=128,
        hidden_dim=3072,
        vocab_size=2048,
        max_seq_len=4096,
        moe_enabled=True,
        moe_hidden_dim=768,
        num_experts=64,
        top_k=8,
        route_norm=True,
        score_before_experts=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    args.__post_init__()
    return args


def qwen3_30b_a3b_args(**overrides) -> Qwen3ModelArgs:
    args = Qwen3ModelArgs(
        dim=2048,
        n_layers=48,
        n_heads=32,
        n_kv_heads=4,
        head_dim=128,
        hidden_dim=6144,
        vocab_size=151936,
        max_seq_len=262144,
        moe_enabled=True,
        moe_hidden_dim=768,
        num_experts=128,
        top_k=8,
        route_norm=True,
        score_before_experts=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    args.__post_init__()
    return args


def qwen3_235b_a22b_args(**overrides) -> Qwen3ModelArgs:
    args = Qwen3ModelArgs(
        dim=4096,
        n_layers=94,
        n_heads=64,
        n_kv_heads=4,
        head_dim=128,
        hidden_dim=12288,
        vocab_size=151936,
        max_seq_len=4096,
        rope_theta=5000000.0,
        moe_enabled=True,
        moe_hidden_dim=1536,
        num_experts=128,
        top_k=8,
        route_norm=True,
        score_before_experts=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    args.__post_init__()
    return args


def precompute_freqs_cos_sin(
    dim: int,
    max_seq_len: int,
    theta: float = 1000000.0,
) -> torch.Tensor:
    freq = theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
    inv_freq = 1.0 / freq
    t = torch.arange(max_seq_len, dtype=inv_freq.dtype, device=inv_freq.device)
    freqs = torch.outer(t, inv_freq).float()
    freqs = torch.cat([freqs, freqs], dim=-1)
    cos = freqs.cos()
    sin = freqs.sin()
    return torch.cat([cos, sin], dim=-1)


def reshape_for_broadcast_cos_sin(
    rope_cache: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    bsz, seqlen, _, head_dim = x.shape
    rope_cache = rope_cache[0:seqlen]
    assert rope_cache.shape == (seqlen, head_dim * 2)
    return rope_cache.view(1, seqlen, 1, head_dim * 2).expand(bsz, -1, -1, -1)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb_cos_sin(
    xq: torch.Tensor,
    xk: torch.Tensor,
    rope_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    head_dim = xq.shape[-1]
    rope_cache = reshape_for_broadcast_cos_sin(rope_cache, xq)
    cos = rope_cache[..., :head_dim].to(device=xq.device)
    sin = rope_cache[..., head_dim:].to(device=xq.device)
    xq_f = xq.float()
    xk_f = xk.float()
    xq_out = (xq_f * cos) + (_rotate_half(xq_f) * sin)
    xk_out = (xk_f * cos) + (_rotate_half(xk_f) * sin)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        torch.unsqueeze(x, dim=3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


def _to_activation_device(tensor: torch.Tensor, activation: torch.Tensor) -> torch.Tensor:
    if tensor.device != activation.device and tensor.device.type == "meta":
        return tensor.to(activation.device)
    return tensor


def _rms_norm(x: torch.Tensor, norm: nn.RMSNorm) -> torch.Tensor:
    weight = (
        _to_activation_device(norm.weight, x)
        if norm.weight is not None
        else None
    )
    if weight is not None and weight.dtype != x.dtype:
        weight = weight.to(dtype=x.dtype)
    return F.rms_norm(x, norm.normalized_shape, weight, norm.eps).to(dtype=x.dtype)


def _linear(x: torch.Tensor, linear: nn.Linear) -> torch.Tensor:
    weight = _to_activation_device(linear.weight, x)
    bias = (
        _to_activation_device(linear.bias, x)
        if linear.bias is not None
        else None
    )
    if weight.dtype != x.dtype:
        weight = weight.to(dtype=x.dtype)
    if bias is not None and bias.dtype != x.dtype:
        bias = bias.to(dtype=x.dtype)
    return F.linear(x, weight, bias)


class Attention(nn.Module):
    def __init__(self, model_args: Qwen3ModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.n_kv_heads = (
            model_args.n_heads
            if model_args.n_kv_heads is None
            else model_args.n_kv_heads
        )
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = model_args.head_dim
        self.scale = self.head_dim**-0.5

        self.wq = nn.Linear(
            model_args.dim, model_args.n_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(
            model_args.n_heads * self.head_dim, model_args.dim, bias=False
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=model_args.norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=model_args.norm_eps)
        self.sdpa = build_attention(model_args.attn_mask_type)

    def init_weights(self, init_std: float):
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)
        self.q_norm.reset_parameters()
        self.k_norm.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos_sin: torch.Tensor,
    ):
        bs, seqlen, _ = x.shape
        xq, xk, xv = _linear(x, self.wq), _linear(x, self.wk), _linear(x, self.wv)

        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        xq = _rms_norm(xq, self.q_norm)
        xk = _rms_norm(xk, self.k_norm)
        freqs_cos_sin = _to_activation_device(freqs_cos_sin, xq)
        xq, xk = apply_rotary_emb_cos_sin(xq, xk, freqs_cos_sin)

        keys = repeat_kv(xk, self.n_rep)
        values = repeat_kv(xv, self.n_rep)

        xq = xq.transpose(1, 2)
        xk = keys.transpose(1, 2)
        xv = values.transpose(1, 2)

        output = self.sdpa(xq, xk, xv, scale=self.scale)

        output = output.transpose(1, 2).contiguous()
        output = output.view(bs, seqlen, -1)
        return _linear(output, self.wo)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return _linear(F.silu(_linear(x, self.w1)) * _linear(x, self.w3), self.w2)

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


class GroupedExperts(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        use_grouped_mm: bool,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.w1 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.w3 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.use_grouped_mm = use_grouped_mm

    def forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_grouped_mm:
            return _run_experts_grouped_mm(
                self.w1, self.w2, self.w3, x, num_tokens_per_expert
            )
        return _run_experts_for_loop(
            self.w1, self.w2, self.w3, x, num_tokens_per_expert
        )

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.w3, mean=0.0, std=init_std)


def _qwen3_token_dispatch(routed_input, num_tokens_per_expert, axis_name):
    ep_size = axis_size(axis_name)
    num_tokens_per_expert_group = all_to_all(
        num_tokens_per_expert,
        None,
        None,
        axis_name,
    )

    with torch.no_grad():
        input_splits = (
            num_tokens_per_expert.view(ep_size, -1)
            .sum(dim=1)
            .to(torch.device("cpu"), non_blocking=True)
        )
        output_splits = (
            num_tokens_per_expert_group.view(ep_size, -1)
            .sum(dim=1)
            .to(torch.device("cpu"), non_blocking=False)
        )
        input_splits = input_splits.tolist()
        output_splits = output_splits.tolist()

    with fx_traceback.annotate({"comm_region": "token_dispatch"}):
        routed_input = all_to_all(
            routed_input,
            output_splits,
            input_splits,
            axis_name,
        )

    num_local_experts = num_tokens_per_expert_group.shape[0] // ep_size
    return (
        *_permute(
            routed_input,
            num_tokens_per_expert_group,
            ep_size,
            num_local_experts,
        ),
        input_splits,
        output_splits,
    )


def qwen3_moe_local_mapped_region(
    x: torch.Tensor,
    selected_experts_indices: torch.Tensor,
    top_scores: torch.Tensor,
    experts_w1: torch.Tensor,
    experts_w3: torch.Tensor,
    experts_w2: torch.Tensor,
    out: torch.Tensor,
    top_k: int,
    num_experts: int,
    score_before_experts: bool,
    axis_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    dim = x.shape[-1]
    ep_size = axis_size(axis_name)
    if num_experts % ep_size != 0:
        raise ValueError(
            f"num_experts ({num_experts}) must be divisible by "
            f"axis_size({axis_name!r}) ({ep_size})."
        )

    num_tokens_per_expert = torch.histc(
        selected_experts_indices.flatten(),
        bins=num_experts,
        min=0,
        max=num_experts,
    ).view(-1)

    token_indices_experts_sorted = torch.argsort(
        selected_experts_indices.view(-1), stable=True
    )
    top_scores_experts_sorted = top_scores.view(-1)[token_indices_experts_sorted]
    token_indices_experts_sorted = token_indices_experts_sorted // top_k

    routed_input = x[token_indices_experts_sorted]
    if score_before_experts:
        routed_input = (
            routed_input.to(torch.float32) * top_scores_experts_sorted.reshape(-1, 1)
        ).to(x.dtype)

    shape = routed_input.shape
    (
        input_shape,
        routed_input,
        permuted_indices,
        num_tokens_per_expert_group,
        input_splits,
        output_splits,
    ) = _qwen3_token_dispatch(routed_input, num_tokens_per_expert, axis_name)

    routed_output = _run_experts_grouped_mm(
        experts_w1,
        experts_w2,
        experts_w3,
        routed_input,
        num_tokens_per_expert_group,
    )
    routed_output = _token_combine(
        routed_output,
        input_shape,
        permuted_indices,
        input_splits,
        output_splits,
        axis_name,
    )

    torch._check(routed_output.shape[0] == shape[0])
    if not score_before_experts:
        routed_output = (
            routed_output.to(torch.float32) * top_scores_experts_sorted.reshape(-1, 1)
        ).to(routed_output.dtype)

    out = out.scatter_add(
        dim=0,
        index=token_indices_experts_sorted.reshape(-1, 1).expand(-1, dim),
        src=routed_output,
    )
    return out, num_tokens_per_expert

class MoE(nn.Module):
    def __init__(
        self,
        model_args: Qwen3ModelArgs,
        mesh: DeviceMesh | None = None,
        axis_name: str | None = None,
    ):
        super().__init__()
        self.mesh = mesh
        self.axis_name = axis_name or model_args.moe_axis_name
        self.num_experts = model_args.num_experts
        self.top_k = model_args.top_k
        self.route_norm = model_args.route_norm
        self.route_scale = model_args.route_scale
        self.score_before_experts = model_args.score_before_experts
        self.load_balance_coeff = model_args.load_balance_coeff

        self.router = nn.Linear(model_args.dim, model_args.num_experts, bias=False)
        self.experts = GroupedExperts(
            dim=model_args.dim,
            hidden_dim=model_args.moe_hidden_dim,
            num_experts=model_args.num_experts,
            use_grouped_mm=model_args.use_grouped_mm,
        )
        self.register_buffer(
            "expert_bias",
            torch.zeros(model_args.num_experts, dtype=torch.float32),
            persistent=self.load_balance_coeff is not None,
        )
        self.register_buffer(
            "tokens_per_expert",
            torch.zeros(model_args.num_experts, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, slen, dim = x.shape
        x = x.view(-1, dim)
        experts_w1, experts_w2, experts_w3 = self.experts.parameters()
        experts_w1 = _to_activation_device(experts_w1, x)
        experts_w2 = _to_activation_device(experts_w2, x)
        experts_w3 = _to_activation_device(experts_w3, x)

        scores = F.linear(
            x.to(torch.float32),
            _to_activation_device(self.router.weight, x).to(torch.float32),
            None,
        )
        scores = F.softmax(scores, dim=-1)
        expert_bias = _to_activation_device(self.expert_bias, scores)
        scores_for_choice = (
            scores + expert_bias
            if self.load_balance_coeff is not None
            else scores
        )
        _, selected_experts_indices = torch.topk(
            scores_for_choice,
            k=self.top_k,
            dim=-1,
            sorted=False,
        )

        top_scores = scores.gather(dim=-1, index=selected_experts_indices)
        if self.route_norm:
            denominator = top_scores.sum(dim=-1, keepdim=True) + 1e-20
            top_scores = top_scores / denominator
        top_scores = top_scores * self.route_scale

        # Qwen3 MoE has no shared expert path, but keeping the initial output
        # differentiably tied to x matches the DSv3 local_map autograd shape.
        out = x * 0
        out, num_tokens_per_expert = local_map(
            qwen3_moe_local_mapped_region,
            out_placements=(
                (Shard(0), Shard(0)),
                (Partial(reduce_op="sum"), Partial(reduce_op="sum")),
            ),
            in_placements=(
                (Shard(0), Shard(0)),
                (Shard(0), Shard(0)),
                (Shard(0), Shard(0)),
                (Replicate(), Shard(0)),
                (Replicate(), Shard(0)),
                (Replicate(), Shard(0)),
                (Shard(0), Shard(0)),
                None,
                None,
                None,
                None,
            ),
            redistribute_inputs=True,
            in_grad_placements=None,
            device_mesh=self.mesh,
        )(
            x,
            selected_experts_indices,
            top_scores,
            experts_w1,
            experts_w3,
            experts_w2,
            out,
            self.top_k,
            self.num_experts,
            self.score_before_experts,
            self.axis_name,
        )
        # This counter is only used for runtime load-balance diagnostics. During
        # AutoParallel graph capture the module buffers are fake/meta tensors
        # while the traced local_map output can be CUDA-fake, and recording this
        # mutation is not needed for the solved training graph.
        if not torch.compiler.is_compiling():
            with torch.no_grad():
                self.tokens_per_expert.add_(num_tokens_per_expert)  # type: ignore[operator]
        return out.reshape(bs, slen, dim)

    def init_weights(
        self,
        init_std: float,
        buffer_device: torch.device,
    ):
        nn.init.trunc_normal_(self.router.weight, mean=0.0, std=init_std)
        self.experts.init_weights(init_std)
        with torch.device(buffer_device):
            self.tokens_per_expert.zero_()  # type: ignore[operator]
            self.expert_bias.zero_()  # type: ignore[operator]


class TransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        model_args: Qwen3ModelArgs,
        mesh: DeviceMesh | None = None,
        moe_axis_name: str | None = None,
    ):
        super().__init__()
        self.attention = Attention(model_args)
        self.moe_enabled = model_args.moe_enabled
        if self.moe_enabled:
            self.moe = MoE(model_args, mesh=mesh, axis_name=moe_axis_name)
        else:
            self.feed_forward = FeedForward(
                dim=model_args.dim,
                hidden_dim=model_args.hidden_dim,
            )
        self.attention_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)

        if model_args.depth_init:
            self.weight_init_std = 0.02 / math.sqrt(2 * (layer_id + 1))
        else:
            self.weight_init_std = 0.02 / math.sqrt(2 * model_args.n_layers)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos_sin: torch.Tensor,
    ):
        h = x + self.attention(_rms_norm(x, self.attention_norm), freqs_cos_sin)
        if self.moe_enabled:
            out = h + self.moe(_rms_norm(h, self.ffn_norm))
        else:
            out = h + self.feed_forward(_rms_norm(h, self.ffn_norm))
        return out

    def init_weights(self, buffer_device: torch.device):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        if self.moe_enabled:
            self.moe.init_weights(self.weight_init_std, buffer_device)
        else:
            self.feed_forward.init_weights(self.weight_init_std)


class Transformer(nn.Module):
    def __init__(
        self,
        model_args: Qwen3ModelArgs,
        mesh: DeviceMesh | None = None,
        moe_axis_name: str | None = None,
    ):
        super().__init__()
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers
        self.eos_id = model_args.eos_id
        self.enable_weight_tying = model_args.enable_weight_tying
        self.mesh = mesh
        self.moe_axis_name = moe_axis_name or model_args.moe_axis_name

        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)
        self.register_buffer(
            "freqs_cos_sin",
            self._precompute_freqs_cos_sin(),
            persistent=True,
        )

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(
                layer_id,
                model_args,
                mesh=mesh,
                moe_axis_name=self.moe_axis_name,
            )
        self.norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.lm_head = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)

        if self.enable_weight_tying:
            self.tok_embeddings.weight = self.lm_head.weight

    def init_weights(
        self,
        buffer_device: Optional[torch.device] = None,
        seed: int | None = None,
    ):
        if seed is not None:
            torch.manual_seed(seed)

        if self.enable_weight_tying:
            self.tok_embeddings.weight = self.lm_head.weight

        buffer_device = buffer_device or self.freqs_cos_sin.device  # type: ignore[assignment]
        with torch.device(buffer_device):  # type: ignore[arg-type]
            self.freqs_cos_sin = self._precompute_freqs_cos_sin()

        if not self.enable_weight_tying and self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight)
        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights(buffer_device)  # type: ignore[operator]
        if self.norm is not None:
            self.norm.reset_parameters()

        final_out_std = self.model_args.dim**-0.5
        cutoff_factor = 3
        if self.lm_head is not None:
            nn.init.trunc_normal_(
                self.lm_head.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )

        if self.enable_weight_tying:
            self.tok_embeddings.weight = self.lm_head.weight

    def _precompute_freqs_cos_sin(self) -> torch.Tensor:
        return precompute_freqs_cos_sin(
            self.model_args.head_dim,
            self.model_args.max_seq_len,
            self.model_args.rope_theta,
        )

    def _token_embedding(self, tokens: torch.Tensor) -> torch.Tensor:
        weight = self.tok_embeddings.weight
        if weight.device != tokens.device and weight.device.type == "meta":
            weight = weight.to(tokens.device)
        return F.embedding(tokens, weight)

    def forward(self, tokens: torch.Tensor, input_batch: Optional[torch.Tensor] = None):
        h = self._token_embedding(tokens) if self.tok_embeddings is not None else tokens

        for layer in self.layers.values():
            h = layer(h, self.freqs_cos_sin)

        h = _rms_norm(h, self.norm) if self.norm is not None else h
        output = _linear(h, self.lm_head) if self.lm_head is not None else h
        return output


_MODULE_FQN = "module_fqn"


def _annotate_once(fn: Callable, meta: dict):
    if getattr(fn, "_graph_trainer_annotated", False):
        return fn
    wrapped = fx_traceback.annotate_fn(meta)(fn)
    setattr(wrapped, "_graph_trainer_annotated", True)
    return wrapped


def _annotate_module_fqns(model: nn.Module) -> None:
    for fqn, submodule in model.named_modules():
        if fqn:
            submodule.forward = _annotate_once(
                submodule.forward,
                {_MODULE_FQN: fqn},
            )


def annotate_qwen3_for_graph_trainer(model: Transformer) -> None:
    """Attach graph_trainer-compatible FX annotations to AP's Qwen3 model."""
    global qwen3_moe_local_mapped_region

    qwen3_moe_local_mapped_region = _annotate_once(
        qwen3_moe_local_mapped_region,
        {"EP": "compute"},
    )
    MoE.forward = _annotate_once(  # type: ignore[method-assign]
        MoE.forward,
        {"EP": "compute"},
    )
    _annotate_module_fqns(model)
