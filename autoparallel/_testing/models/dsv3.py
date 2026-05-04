# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
from typing import Callable, ClassVar, Literal, Optional, Tuple

import torch
import torch.fx.traceback as fx_traceback
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn
from torch.distributed.tensor import DeviceMesh, DTensor
from torch.distributed.tensor.placement_types import Partial, Replicate, Shard
from torch.nn.attention import SDPBackend, sdpa_kernel

from autoparallel.collectives import all_to_all, axis_size, local_map

_MODULE_FQN = "module_fqn"


def _to_compute_dtype(
    x: torch.Tensor,
    compute_dtype: torch.dtype | None,
) -> torch.Tensor:
    if compute_dtype is None or not torch.is_floating_point(x):
        return x
    return x.to(compute_dtype)


def _linear_compute(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    compute_dtype: torch.dtype | None,
) -> torch.Tensor:
    if compute_dtype is None:
        return F.linear(x, weight, bias)
    bias = None if bias is None else bias.to(compute_dtype)
    return F.linear(x.to(compute_dtype), weight.to(compute_dtype), bias)


def _rms_norm_compute(
    x: torch.Tensor,
    norm: nn.RMSNorm,
    compute_dtype: torch.dtype | None,
) -> torch.Tensor:
    if compute_dtype is None:
        return norm(x)
    weight = None if norm.weight is None else norm.weight.to(compute_dtype)
    return F.rms_norm(
        x.to(compute_dtype),
        norm.normalized_shape,
        weight,
        norm.eps,
    )


@triton.jit
def _fill_indices_kernel(
    tokens_per_expert_group_ptr,
    start_index_values_ptr,
    write_offsets_ptr,
    output_ptr,
    experts_per_rank: tl.constexpr,
    num_ranks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)

    for expert_id in range(pid, experts_per_rank, num_programs):
        write_offset = tl.load(write_offsets_ptr + expert_id)

        for r in range(num_ranks):
            i = r * experts_per_rank + expert_id
            start_index = tl.load(start_index_values_ptr + i)
            length = tl.load(tokens_per_expert_group_ptr + i)
            offsets = tl.arange(0, BLOCK_SIZE)

            for chunk_start in range(0, length, BLOCK_SIZE):
                chunk_offsets = chunk_start + offsets
                mask = chunk_offsets < length
                values = start_index + chunk_offsets
                dest_indices = write_offset + chunk_offsets
                tl.store(output_ptr + dest_indices, values, mask=mask)

            write_offset += length


@torch.library.custom_op("autoparallel::fill_indices_functional", mutates_args=())
def fill_indices_functional(
    tokens_per_expert_group: torch.Tensor,
    start_index_values: torch.Tensor,
    write_offsets: torch.Tensor,
    experts_per_rank: int,
    num_ranks: int,
    max_len: int,
    block_size: int = 128,
    max_blocks: int = 1024,
) -> torch.Tensor:
    permuted_indices = torch.full(
        (max_len,), -1, dtype=torch.int32, device=tokens_per_expert_group.device
    )

    num_blocks = min(experts_per_rank, max_blocks)
    _fill_indices_kernel[(num_blocks,)](
        tokens_per_expert_group,
        start_index_values,
        write_offsets,
        permuted_indices,
        experts_per_rank,
        num_ranks,
        BLOCK_SIZE=block_size,
    )
    return permuted_indices


@fill_indices_functional.register_fake
def _(
    tokens_per_expert_group: torch.Tensor,
    start_index_values: torch.Tensor,
    write_offsets: torch.Tensor,
    experts_per_rank: int,
    num_ranks: int,
    max_len: int,
    block_size: int = 128,
    max_blocks: int = 1024,
) -> torch.Tensor:
    return torch.full(
        (max_len,), -1, dtype=torch.int32, device=tokens_per_expert_group.device
    )


def generate_permute_indices(
    tokens_per_expert_group: torch.Tensor,
    experts_per_rank: int,
    num_ranks: int,
    max_len: int,
    alignment: int,
):
    start_index_values = (
        torch.cumsum(tokens_per_expert_group, 0) - tokens_per_expert_group
    )
    total_tokens_per_expert = tokens_per_expert_group.view(num_ranks, -1).sum(0)
    total_tokens_per_expert = torch.clamp_min(total_tokens_per_expert, alignment)
    m_sizes = ((total_tokens_per_expert + alignment - 1) // alignment * alignment).to(
        torch.int32
    )
    m_offsets = torch.cumsum(m_sizes, 0)
    write_offsets = m_offsets - m_sizes
    permuted_indices = fill_indices_functional(
        tokens_per_expert_group,
        start_index_values,
        write_offsets,
        experts_per_rank,
        num_ranks,
        max_len,
    )
    return permuted_indices, m_sizes, m_offsets.to(torch.int32)


TOKEN_GROUP_ALIGN_SIZE_M = 8


def _round_up(x: int, y: int) -> int:
    x_ceil_div_y = (x + y - 1) // y
    return x_ceil_div_y * y


def functional_feed_forward(
    w1,
    w2,
    w3,
    x,
    compute_dtype: torch.dtype | None = None,
):
    h1 = _linear_compute(x, w1, None, compute_dtype)
    h3 = _linear_compute(x, w3, None, compute_dtype)
    return _linear_compute(F.silu(h1) * h3, w2, None, compute_dtype)


# can be used as dense FFN layer or shared experts in MoE layers
class FeedForward(nn.Module):
    """
    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension of the feedforward layer.

    Attributes:
        w1 (Linear): Linear transformation for the first layer.
        w2 (Linear): Linear transformation for the second layer.
        w3 (Linear): Linear transformation for the third layer.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.compute_dtype = compute_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return functional_feed_forward(
            self.w1.weight,
            self.w2.weight,
            self.w3.weight,
            x,
            self.compute_dtype,
        )

    def init_weights(self, init_std: float = 0.02):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


def _run_experts_for_loop(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x_: torch.Tensor,
    num_tokens_per_expert_: torch.Tensor,
) -> torch.Tensor:
    if isinstance(w1, DTensor):
        assert isinstance(w2, DTensor)
        assert isinstance(w3, DTensor)
        w1 = w1.to_local()
        w2 = w2.to_local()
        w3 = w3.to_local()

    # NOTE: this would incur a synchronization between device and host
    num_tokens_per_expert: list[int] = num_tokens_per_expert_.tolist()

    # a tuple of tensors indexed by experts
    # each with shape (tokens_per_expert(varying), dim)
    x: tuple[torch.Tensor, ...] = torch.split(
        x_,
        split_size_or_sections=num_tokens_per_expert,
        dim=0,
    )
    out_experts_splits = []
    for expert_idx, x_expert in enumerate(x):
        compute_dtype = x_expert.dtype if torch.is_floating_point(x_expert) else None
        x_expert = _to_compute_dtype(x_expert, compute_dtype)
        h = F.silu(
            torch.matmul(
                x_expert,
                _to_compute_dtype(w1[expert_idx], compute_dtype).transpose(-2, -1),
            )
        )
        h = h * torch.matmul(
            x_expert,
            _to_compute_dtype(w3[expert_idx], compute_dtype).transpose(-2, -1),
        )
        h = torch.matmul(
            h,
            _to_compute_dtype(w2[expert_idx], compute_dtype).transpose(-2, -1),
        )
        # h shape (tokens_per_expert(varying), dim)
        out_experts_splits.append(h)

    return torch.cat(out_experts_splits, dim=0)


def _run_experts_grouped_mm(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    if isinstance(w1, DTensor):
        assert isinstance(w2, DTensor)
        assert isinstance(w3, DTensor)
        w1 = w1.to_local()
        w2 = w2.to_local()
        w3 = w3.to_local()

    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
    # grouped mm between a 2D tensor and a 3D tensor
    assert x.dim() == 2

    h = F.silu(
        torch._grouped_mm(x.bfloat16(), w1.bfloat16().transpose(-2, -1), offs=offsets)
    )
    h = h * torch._grouped_mm(
        x.bfloat16(), w3.bfloat16().transpose(-2, -1), offs=offsets
    )
    out = torch._grouped_mm(h, w2.bfloat16().transpose(-2, -1), offs=offsets).type_as(x)

    return out


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
        else:
            return _run_experts_for_loop(
                self.w1, self.w2, self.w3, x, num_tokens_per_expert
            )

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.w3, mean=0.0, std=init_std)


@torch.library.custom_op("autoparallel::batched_histc", mutates_args=())
def batched_histc(
    x: torch.Tensor, bins: int = 100, min: int = 0, max: int = 0
) -> torch.Tensor:
    assert x.ndim == 2
    out = []
    for t in x:
        out.append(torch.histc(t, bins, min, max))
    return torch.stack(out, 0)


@batched_histc.register_fake
def batched_histc_fake(
    x: torch.Tensor, bins: int = 100, min: int = 0, max: int = 0
) -> torch.Tensor:
    assert max - min == bins
    out = torch.empty((x.shape[0], bins), dtype=torch.int64, device=x.device)
    return out


class TokenChoiceTopKRouter(nn.Module):
    """This class implements token-choice routing. In token-choice top-K routing, each token is
        routed to top K experts based on the router scores.

    Args:
        dim (int): Dimension of input tokens.
        num_experts (int): Number of experts in each moe layer.
        top_k (int): Number of experts each token will be routed to in token-choice routing.
        score_func (Literal["softmax", "sigmoid"]): Whether to use sigmoid or softmax for router scores.
        route_norm (bool): Whether to normalize the routing scores when using sigmoid.
        route_scale (float): Scaling factor applied to the routing scores.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int,
        score_func: Literal["softmax", "sigmoid"],
        route_norm: bool,
        route_scale: float,
    ):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.num_experts = num_experts
        self.top_k = top_k
        self.score_func = score_func
        self.route_norm = route_norm
        self.route_scale = route_scale

    def forward(
        self,
        x: torch.Tensor,
        gate_weight: torch.nn.Parameter,
        expert_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs*slen, dim)``.
            expert_bias (torch.Tensor | None, optional): Optional bias tensor for experts with shape ``(num_experts,)``.
                Used for load balancing. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - top_scores (torch.Tensor):
                    Routing scores for selected experts with shape ``(bs*slen, top_k)``.
                - selected_experts_indices (torch.Tensor):
                    Expert indices selected for each token with shape ``(bs*slen, top_k)``.
                - num_tokens_per_expert (torch.Tensor):
                    Number of tokens assigned to each expert with shape ``(num_experts,)``.
        """
        # scores shape (bs*slen, num_experts)
        # scores = self.gate(x)
        scores = _linear_compute(
            x,
            gate_weight,
            None,
            x.dtype if torch.is_floating_point(x) else None,
        )

        # By default, sigmoid or softmax is performed in float32 to avoid loss explosion
        if self.score_func == "sigmoid":
            scores = torch.sigmoid(scores.to(torch.float32))
        elif self.score_func == "softmax":
            scores = F.softmax(scores.to(torch.float32), dim=-1)
        else:
            raise NotImplementedError(f"Unknown score function {self.score_func}")

        # top scores shape (bs*slen, top_k)
        # NOTE: The expert_bias is only used for routing. The gating value
        #       top_scores is still derived from the original scores.
        if expert_bias is not None:
            _, selected_experts_indices = torch.topk(
                scores + expert_bias, k=self.top_k, dim=-1
            )
            top_scores = scores.gather(dim=-1, index=selected_experts_indices)
        else:
            top_scores, selected_experts_indices = torch.topk(
                scores, k=self.top_k, dim=-1
            )

        if self.score_func == "sigmoid" and self.route_norm:
            denominator = top_scores.sum(dim=-1, keepdim=True) + 1e-20
            top_scores = top_scores / denominator
        top_scores = top_scores * self.route_scale

        return top_scores, selected_experts_indices

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.gate.weight, mean=0.0, std=init_std)


# NOTE: the reason we make this a stateless module is to support
#       expert_tensor_parallel_degree=1 with consistent TP/EP APIs.
class TokenReorderer(nn.Module):
    """
    This module reorders token indices to match the order of experts, enabling
    efficient parallel processing of tokens by experts.

    Args:
        num_experts (int): Number of experts in the MoE layer.
        top_k (int): Number of experts each token will be routed to.
    """

    def __init__(self, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(
        self,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Reorders token indices to match the order of experts for MoE routing.

        Args:
            top_scores (torch.Tensor): Routing scores for selected experts,
                shape (batch_size*seq_len, top_k)
            selected_experts_indices (torch.Tensor): Expert indices selected for each token,
                shape (batch_size*seq_len, top_k)

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - top_scores_experts_sorted: Scores reordered to match expert ordering
                - token_indices_experts_sorted: Token indices reordered to match expert ordering
                - num_tokens_per_expert: Number of tokens assigned to each expert
        """
        # group tokens together by expert indices from 0 to num_experts and pass that to experts forward
        # num_tokens_per_expert = torch.histc(
        #     selected_experts_indices.view(-1),
        #     bins=self.num_experts,
        #     min=0,
        #     max=self.num_experts,
        # )

        # Reorder the token indices to match the order of the experts
        # token_indices_experts_sorted shape (bs*slen*top_k,)
        token_indices_experts_sorted = torch.argsort(
            selected_experts_indices.flatten(1), dim=-1, stable=True
        )

        # top_scores_experts_sorted = top_scores.view(-1)[token_indices_experts_sorted]
        top_scores_experts_sorted = top_scores.view_as(
            token_indices_experts_sorted
        ).gather(1, token_indices_experts_sorted)
        token_indices_experts_sorted = token_indices_experts_sorted // self.top_k

        return (
            top_scores_experts_sorted,
            token_indices_experts_sorted,
            # num_tokens_per_expert,
        )


def _permute(routed_input, num_tokens_per_expert_group, ep_size, num_local_experts):
    """Reorder tokens from rank-major to expert-major layout.

    This is the local_map-friendly version of TorchTitan's dispatcher permute:
    it uses a fixed-size custom op rather than repeat_interleave with a dynamic
    output shape, which local_map cannot currently capture.
    """
    x_padded_per_expert = (
        routed_input.shape[0] + num_local_experts * TOKEN_GROUP_ALIGN_SIZE_M
    )
    padded_max_len = _round_up(x_padded_per_expert, TOKEN_GROUP_ALIGN_SIZE_M)
    with torch.no_grad():
        (permuted_indices, num_tokens_per_expert, _,) = generate_permute_indices(
            num_tokens_per_expert_group,
            num_local_experts,
            ep_size,
            padded_max_len,
            TOKEN_GROUP_ALIGN_SIZE_M,
        )

    routed_input = torch.vstack(
        (routed_input, routed_input.new_zeros((routed_input.shape[-1])))
    )
    return (
        routed_input.shape,
        routed_input[permuted_indices, :],
        permuted_indices,
        num_tokens_per_expert,
    )


def _unpermute(routed_output, input_shape, permuted_indices):
    """Reverse expert-major reordering."""
    out_unpermuted = routed_output.new_empty(input_shape)
    out_unpermuted[permuted_indices, :] = routed_output
    return out_unpermuted[:-1]


def _token_dispatch(routed_input, num_tokens_per_expert, axis_name):

    ep_size = axis_size(axis_name)

    with torch.no_grad():
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
        # NOTE: this would incur a device-to-host sync
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

    # Reorder from rank-major to expert-major via _permute.
    #
    # num_tokens_per_expert_group layout after all-to-all:
    #   (e0,r0), (e1,r0), ..., (e0,r1), (e1,r1), ...  (rank-major)
    # _permute reshuffles to:
    #   (e0,r0), (e0,r1), ..., (e1,r0), (e1,r1), ...  (expert-major)
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


def _token_combine(
    routed_output, input_shape, permuted_indices, input_splits, output_splits, axis_name
):
    routed_output = _unpermute(routed_output, input_shape, permuted_indices)
    with fx_traceback.annotate({"comm_region": "token_combine"}):
        routed_output = all_to_all(
            routed_output,
            input_splits,
            output_splits,
            axis_name,
        )
        return routed_output


# @torch.library.custom_op("autoparallel::local_mapped_region", mutates_args=())
def local_mapped_region(
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
    # assert False, f"{x.shape}, {selected_experts_indices.shape}, {top_scores.shape}, {out.shape}"

    dim = x.shape[-1]

    # num_tokens_per_expert = torch.ops.autoparallel.batched_histc(
    num_tokens_per_expert = torch.histc(
        selected_experts_indices.flatten(),
        bins=num_experts,
        min=0,
        max=num_experts,
    )

    # total_tokens_per_expert = all_reduce(num_tokens_per_expert, axis_name)
    total_tokens_per_expert = num_tokens_per_expert

    token_indices_experts_sorted = torch.argsort(
        selected_experts_indices.view(-1), stable=True
    )

    top_scores_experts_sorted = top_scores.view(-1)[token_indices_experts_sorted]
    token_indices_experts_sorted = token_indices_experts_sorted // top_k

    # shape (bs*slen*top_k, dim)
    routed_input = x[token_indices_experts_sorted]
    if score_before_experts:
        routed_input = (
            routed_input.to(torch.float32) * top_scores_experts_sorted.reshape(-1, 1)
        ).to(x.dtype)

    shape = routed_input.shape
    dim = shape[-1]
    num_tokens_per_expert = num_tokens_per_expert.view(-1)
    (
        input_shape,
        routed_input,
        permuted_indices,
        num_tokens_per_expert_group,
        input_splits,
        output_splits,
    ) = _token_dispatch(routed_input, num_tokens_per_expert, axis_name)

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
    return out, total_tokens_per_expert


# @local_mapped_region.register_fake
def _(
    routed_input: torch.Tensor,
    selected_expert_indices: torch.Tensor,
    top_scores: torch.Tensor,
    out: torch.Tensor,
    experts_w1: torch.Tensor,
    experts_w2: torch.Tensor,
    experts_w3: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_experts = 64
    return torch.empty_like(routed_input), torch.empty(
        (1, num_experts), dtype=routed_input.dtype, device=routed_input.device
    )


# @torch.library.custom_op("autoparallel::local_mapped_region_grad", mutates_args=())
# def local_mapped_region_grad(
#     routed_input: torch.Tensor,
#     selected_experts_indices: torch.Tensor,
#     top_scores: torch.Tensor,
#     out: torch.Tensor,
#     experts_w1: torch.Tensor,
#     experts_w2: torch.Tensor,
#     experts_w3: torch.Tensor,
# ) -> tuple[
#     torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
# ]:
#     grad_i = torch.empty_like(routed_input)
#     grad_o = torch.empty_like(out)
#     grad_s = torch.empty_like(top_scores)
#     g1 = torch.empty_like(experts_w1)
#     g2 = torch.empty_like(experts_w2)
#     g3 = torch.empty_like(experts_w3)
#     return grad_i, grad_s, grad_o, g1, g2, g3


# @local_mapped_region_grad.register_fake
# def _(
#     routed_input: torch.Tensor,
#     selected_experts_indices: torch.Tensor,
#     top_scores: torch.Tensor,
#     out: torch.Tensor,
#     experts_w1: torch.Tensor,
#     experts_w2: torch.Tensor,
#     experts_w3: torch.Tensor,
# ) -> tuple[
#     torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
# ]:
#     grad_i = torch.empty_like(routed_input)
#     grad_o = torch.empty_like(out)
#     grad_s = torch.empty_like(top_scores)
#     g1 = torch.empty_like(experts_w1)
#     g2 = torch.empty_like(experts_w2)
#     g3 = torch.empty_like(experts_w3)
#     return grad_i, grad_s, grad_o, g1, g2, g3


# def setup_context_local_mapped_region(ctx, inputs, output):
#     # routed_input, num_tokens_per_expert, experts_w1, experts_w2, experts_w3 = inputs
#     ctx.save_for_backward(*inputs)


# def backward_local_mapped_region(ctx, grad, grad2):
#     (
#         routed_input,
#         selected_experts_indices,
#         top_scores,
#         out,
#         experts_w1,
#         experts_w2,
#         experts_w3,
#     ) = ctx.saved_tensors
#     grad_i, grad_s, grad_o, g1, g2, g3 = local_mapped_region_grad(
#         routed_input,
#         selected_experts_indices,
#         top_scores,
#         out,
#         experts_w1,
#         experts_w2,
#         experts_w3,
#     )
#     return grad_i, None, grad_s, grad_o, g1, g2, g3


# torch.library.register_autograd(
#     "autoparallel::local_mapped_region",
#     backward_local_mapped_region,
#     setup_context=setup_context_local_mapped_region,
# )


def _moe_forward(
    x: torch.Tensor,
    router_gate_weight: torch.Tensor,
    expert_bias: Optional[torch.Tensor],
    experts_w1: torch.Tensor,
    experts_w3: torch.Tensor,
    experts_w2: torch.Tensor,
    shared_w1: torch.Tensor,
    shared_w3: torch.Tensor,
    shared_w2: torch.Tensor,
    router: TokenChoiceTopKRouter,
    reorderer: TokenReorderer,
    mesh: Optional[DeviceMesh],
    axis_name: str,
    score_before_experts: bool,
    compute_dtype: torch.dtype | None = None,
):
    # x: 64, 2048, 256
    bs, slen, dim = x.shape
    x = x.view(-1, dim)

    # top_scores and selected_experts_indices shape (bs*slen*top_k,)
    # num_tokens_per_expert shape (num_experts,)
    (
        top_scores,
        selected_experts_indices,
    ) = router(x, router_gate_weight, expert_bias)

    # tokens_per_expert will be used to update the expert bias for load balancing.
    # and also to count the expert usage
    # TODO: Activation Checkpointing has the side effect of double counting tokens_per_expert --
    #       first in the forward pass, and then in the backward pass. However, this has no
    #       effect on the expert bias update thanks to the torch.sign() operator.
    # moved out to remove mutation
    # with torch.no_grad():
    #     tokens_per_expert.add_(num_tokens_per_expert)

    # top_scores and token_indices_experts_sorted shape (bs*slen*top_k,)
    # num_tokens_per_expert shape (num_experts,)
    # NOTE: the reason we need to compute num_tokens_per_expert again is:
    #       1st computation in router is to update self.tokens_per_expert
    #       which would be the same across all TP ranks.
    #       2nd computation in reorderer is for the actual routing and experts computation
    #       which would be sharded over TP ranks if expert_tensor_parallel_degree==1.
    #       If tensor_paralllel_degree == expert_tensor_parallel_degree, they agree.
    # (
    #     top_scores_experts_sorted,
    #     token_indices_experts_sorted,
    #     # _, #num_tokens_per_expert,
    # ) = reorderer(top_scores, selected_experts_indices)

    # shape (bs*slen*top_k, dim)
    # routed_output = experts(routed_input, num_tokens_per_expert)

    out = functional_feed_forward(shared_w1, shared_w2, shared_w3, x, compute_dtype)

    ######################################################
    # This is in the local_map region
    ######################################################

    # expert_placements = ((Replicate(), Shard(0)),) * 3
    # in_placements = (
    #     (Shard(0), Shard(0)),
    #     (Shard(0), Shard(0)),
    #     (Shard(0), Shard(0)),
    #     (Shard(0), Shard(0)),
    # )
    # Dynamo reorders captured variables (lifted freevars) before explicit
    # arguments, so x must come first in the input order and placements.
    reordered_placements = (
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
    )

    out, num_tokens_per_expert = local_map(
        local_mapped_region,
        out_placements=(
            (Shard(0), Shard(0)),
            (Partial(reduce_op="sum"), Partial(reduce_op="sum")),
        ),
        in_placements=reordered_placements,
        redistribute_inputs=True,
        in_grad_placements=None,
        device_mesh=mesh,
    )(
        x,
        selected_experts_indices,
        top_scores,
        experts_w1,
        experts_w3,
        experts_w2,
        out,
        router.top_k,
        router.num_experts,
        score_before_experts,
        axis_name,
    )
    # assert False, f"there: {out.shape}, {num_tokens_per_expert.shape}"

    ######################################################
    # end of the local_map region
    ######################################################

    # shared expert
    # if shared_experts is not None:
    #     out = shared_experts(x)
    # else:
    #     out = torch.zeros_like(x)

    # assert False, f"{out.shape}, {token_indices_experts_sorted.shape}, {routed_output.shape}"
    out = out.reshape(bs, slen, dim)
    return out, num_tokens_per_expert


class MoE(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        top_k: int,
        shared_experts_hidden_dim: int,
        score_func: Literal["softmax", "sigmoid"] = "sigmoid",
        route_norm: bool = False,
        route_scale: float = 1.0,
        score_before_experts: bool = True,
        use_grouped_mm: bool = True,
        load_balance_coeff: float | None = 1e-3,
        mesh: DeviceMesh | None = None,
        compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.mesh = mesh
        self.axis_name = "ep"
        self.compute_dtype = compute_dtype
        self.experts = GroupedExperts(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            use_grouped_mm=use_grouped_mm,
        )
        self.router = TokenChoiceTopKRouter(
            dim=dim,
            num_experts=num_experts,
            top_k=top_k,
            score_func=score_func,
            route_norm=route_norm,
            route_scale=route_scale,
        )
        self.reorderer = TokenReorderer(num_experts=num_experts, top_k=top_k)
        self.shared_experts = FeedForward(dim=dim, hidden_dim=shared_experts_hidden_dim)
        self.score_before_experts = score_before_experts

        self.load_balance_coeff = load_balance_coeff
        if self.load_balance_coeff is not None:
            assert self.load_balance_coeff > 0.0
            self.register_buffer(
                "expert_bias",
                torch.zeros(num_experts, dtype=torch.float32),
                persistent=True,
            )
        else:
            self.expert_bias = None
        # tokens_per_expert will be used to track expert usage and to update the expert bias for load balancing
        self.register_buffer(
            "tokens_per_expert",
            torch.zeros(num_experts, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs, slen, dim)``.

        Returns:
            out (torch.Tensor): Output tensor with shape ``(bs, slen, dim)``.
        """
        experts_w1, experts_w2, experts_w3 = self.experts.parameters()
        shared_w1, shared_w2, shared_w3 = self.shared_experts.parameters()
        out, num_tokens_per_expert = _moe_forward(
            x,
            self.router.gate.weight,
            self.expert_bias,
            experts_w1,
            experts_w3,
            experts_w2,
            shared_w1,
            shared_w3,
            shared_w2,
            self.router,
            self.reorderer,
            self.mesh,
            self.axis_name,
            self.score_before_experts,
            self.compute_dtype,
        )

        # HOPs don't support buffer mutations, keep this outside
        with torch.no_grad():
            self.tokens_per_expert.add_(num_tokens_per_expert)  # type: ignore[operator]
        return out

    def init_weights(
        self,
        init_std: float,
        buffer_device: torch.device,
    ):
        self.experts.init_weights(init_std)
        self.router.init_weights(init_std)
        if self.shared_experts is not None:
            self.shared_experts.init_weights(init_std)

        with torch.device(buffer_device):
            self.tokens_per_expert.zero_()  # type: ignore[operator]
            if self.load_balance_coeff is not None:
                assert isinstance(self.expert_bias, torch.Tensor)
                self.expert_bias.zero_()  # type: ignore[operator]


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
            raise ValueError(
                "TorchTitan with SDPA currently only supports causal mask."
            )

        ScaledDotProductAttention._init_backend()

    @classmethod
    def _init_backend(cls) -> None:
        if cls.backends:
            return

        # Add CuDNN on B200 w/ highest priority
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
        assert self.backends, "SDPA Backends should not be empty."
        with sdpa_kernel(self.backends, set_priority=True):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)


def build_attention(
    use_flex_attn: bool, attn_mask_type: str, fixed_block_size: int | None = None
):
    if fixed_block_size is not None:
        raise ValueError(
            "TorchTitan with SDPA currently does not support fixed_block_size."
        )
    if attn_mask_type != "causal":
        raise ValueError("TorchTitan with SDPA currently only supports causal mask.")
    return ScaledDotProductAttention(attn_mask_type)


@dataclass
class RoPEConfig:
    dim: int = 64
    max_seq_len: int = 4096 * 4
    theta: float = 10000.0
    rope_factor: float = 40.0
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    original_seq_len: int = 4096


@dataclass
class NormConfig:
    eps: float = 1e-5


@dataclass
class LinearConfig:
    in_features: int = 0
    out_features: int = 0


@dataclass
class SDPAConfig:
    pass


@dataclass
class AttentionConfig:
    n_heads: int = 16
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    mscale: float = 1.0
    mask_type: str = "causal"
    inner_attention: SDPAConfig = field(default_factory=SDPAConfig)


@dataclass
class TokenDispatcherConfig:
    score_before_experts: bool = True


@dataclass
class ExpertsConfig:
    hidden_dim: int = 1408
    use_grouped_mm: bool = True
    token_dispatcher: TokenDispatcherConfig = field(
        default_factory=TokenDispatcherConfig
    )


@dataclass
class RouterConfig:
    top_k: int = 1
    score_func: str = "sigmoid"
    route_norm: bool = False
    route_scale: float = 1.0


@dataclass
class FeedForwardConfig:
    w1: LinearConfig = field(default_factory=LinearConfig)


@dataclass
class MoEConfig:
    num_experts: int = 8
    experts: ExpertsConfig = field(default_factory=ExpertsConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    load_balance_coeff: float | None = 1e-3
    shared_experts: FeedForwardConfig = field(default_factory=FeedForwardConfig)


@dataclass
class LayerConfig:
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    attention_norm: NormConfig = field(default_factory=NormConfig)
    ffn_norm: NormConfig = field(default_factory=NormConfig)
    feed_forward: FeedForwardConfig | None = None
    moe: MoEConfig | None = None


@dataclass
class DeepSeekV3Config:
    """Hierarchical config for DeepSeekV3Model.

    Attribute paths are compatible with torchtitan's DeepSeekV3Model.Config,
    so either config type can be passed to DeepSeekV3Model.
    """

    dim: int = 2048
    vocab_size: int = 102400
    rope: RoPEConfig = field(default_factory=RoPEConfig)
    norm: NormConfig = field(default_factory=NormConfig)
    layers: list = field(default_factory=list)


def make_dsv3_config(
    dim: int = 256,
    vocab_size: int = 2048,
    n_layers: int = 6,
    n_dense_layers: int = 1,
    n_heads: int = 16,
    q_lora_rank: int = 0,
    kv_lora_rank: int = 512,
    qk_nope_head_dim: int = 128,
    qk_rope_head_dim: int = 64,
    v_head_dim: int = 128,
    mscale: float = 0.70,
    dense_hidden_dim: int = 1024,
    moe_hidden_dim: int = 256,
    num_experts: int = 8,
    num_shared_experts: int = 2,
    top_k: int = 3,
    score_func: str = "softmax",
    route_norm: bool = False,
    score_before_experts: bool = False,
    max_seq_len: int = 4096 * 4,
    rope_theta: float = 10000.0,
    rope_factor: float = 40.0,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    original_seq_len: int = 4096,
    load_balance_coeff: float | None = 1e-3,
) -> DeepSeekV3Config:
    layers = []
    for layer_id in range(n_layers):
        attn = AttentionConfig(
            n_heads=n_heads,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            mscale=mscale,
        )
        if layer_id < n_dense_layers:
            ff = FeedForwardConfig(w1=LinearConfig(out_features=dense_hidden_dim))
            moe = None
        else:
            ff = None
            moe = MoEConfig(
                num_experts=num_experts,
                experts=ExpertsConfig(
                    hidden_dim=moe_hidden_dim,
                    token_dispatcher=TokenDispatcherConfig(
                        score_before_experts=score_before_experts,
                    ),
                ),
                router=RouterConfig(
                    top_k=top_k,
                    score_func=score_func,
                    route_norm=route_norm,
                ),
                load_balance_coeff=load_balance_coeff,
                shared_experts=FeedForwardConfig(
                    w1=LinearConfig(
                        out_features=moe_hidden_dim * num_shared_experts,
                    ),
                ),
            )
        layers.append(
            LayerConfig(
                attention=attn,
                feed_forward=ff,
                moe=moe,
            )
        )

    return DeepSeekV3Config(
        dim=dim,
        vocab_size=vocab_size,
        rope=RoPEConfig(
            dim=qk_rope_head_dim,
            max_seq_len=max_seq_len,
            theta=rope_theta,
            rope_factor=rope_factor,
            beta_fast=beta_fast,
            beta_slow=beta_slow,
            original_seq_len=original_seq_len,
        ),
        layers=layers,
    )


def precompute_freqs_cis(config) -> torch.Tensor:
    rope = config.rope
    dim = rope.dim
    seqlen = rope.max_seq_len
    beta_fast = rope.beta_fast
    beta_slow = rope.beta_slow
    base = rope.theta
    factor = rope.rope_factor

    def find_correction_dim(
        num_rotations: float, dim: int, base: float, max_seq_len: int
    ) -> float:
        """
        Computes the correction dimension for a given number of rotations in the rotary positional embedding.

        Args:
            num_rotations (float): Number of rotations to compute the correction for.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            float: The correction dimension based on the input parameters.
        """
        return (
            dim
            * math.log(max_seq_len / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

    def find_correction_range(
        low_rot: float, high_rot: float, dim: int, base: float, max_seq_len: int
    ) -> Tuple[int, int]:
        """
        Computes the range of correction dimensions for rotary positional embeddings.

        Args:
            low_rot (float): Lower bound for the number of rotations.
            high_rot (float): Upper bound for the number of rotations.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            Tuple[int, int]: The range of correction dimensions (low, high), clamped to valid indices.
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min: float, max: float, dim: int) -> torch.Tensor:
        """
        Computes a linear ramp function used to smooth values between a minimum and maximum range.

        Args:
            min (float): Minimum value for the ramp function.
            max (float): Maximum value for the ramp function.
            dim (int): Dimensionality of the ramp tensor.

        Returns:
            torch.Tensor: A tensor of shape (dim,) with values linearly interpolated between 0 and 1,
                clamped to the range [0, 1].
        """
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    # Basic RoPE frequency calculation
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    # YaRN scaling for extended context. YaRN is used to extend the context length after pre-training.
    if seqlen > rope.original_seq_len:
        low, high = find_correction_range(
            beta_fast, beta_slow, dim, base, rope.original_seq_len
        )
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    # Create position indices
    t = torch.arange(seqlen)

    # Outer product: [positions] × [frequencies]
    freqs = torch.outer(t, freqs)

    # Convert to complex exponentials: e^(i*freq*pos)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)


class Attention(nn.Module):
    """
    Multi-head attention (MLA) module.
    """

    def __init__(
        self,
        attn_config,
        model_config,
        compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.dim = model_config.dim
        self.n_heads = attn_config.n_heads
        self.q_lora_rank = attn_config.q_lora_rank
        self.kv_lora_rank = attn_config.kv_lora_rank
        self.qk_nope_head_dim = attn_config.qk_nope_head_dim
        self.qk_rope_head_dim = attn_config.qk_rope_head_dim
        self.qk_head_dim = attn_config.qk_nope_head_dim + attn_config.qk_rope_head_dim
        self.v_head_dim = attn_config.v_head_dim
        self.compute_dtype = compute_dtype

        norm_eps = model_config.norm.eps

        if self.q_lora_rank == 0:
            self.wq = nn.Linear(self.dim, self.n_heads * self.qk_head_dim, bias=False)
        else:
            self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False)
            self.q_norm = nn.RMSNorm(self.q_lora_rank, eps=norm_eps)
            self.wq_b = nn.Linear(
                self.q_lora_rank, self.n_heads * self.qk_head_dim, bias=False
            )
        self.wkv_a = nn.Linear(
            self.dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=False
        )
        self.kv_norm = nn.RMSNorm(self.kv_lora_rank, eps=norm_eps)
        self.wkv_b = nn.Linear(
            self.kv_lora_rank,
            self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim, bias=False)
        self.softmax_scale = self.qk_head_dim**-0.5

        rope_cfg = model_config.rope
        if rope_cfg.max_seq_len > rope_cfg.original_seq_len:
            mscale = 0.1 * attn_config.mscale * math.log(rope_cfg.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        use_flex_attn = "FlexAttention" in type(attn_config.inner_attention).__name__
        self.sdpa = build_attention(use_flex_attn, attn_config.mask_type)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        """
        Forward pass for the Multi-Head Latent Attention (MLA) Layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        bsz, seqlen, _ = x.size()

        # Query projection
        if self.q_lora_rank == 0:
            q = _linear_compute(
                x,
                self.wq.weight,
                self.wq.bias,
                self.compute_dtype,
            )  # (bsz, seqlen, n_heads * qk_head_dim)
        else:
            q = _linear_compute(
                x,
                self.wq_a.weight,
                self.wq_a.bias,
                self.compute_dtype,
            )
            q = _linear_compute(
                _rms_norm_compute(q, self.q_norm, self.compute_dtype),
                self.wq_b.weight,
                self.wq_b.bias,
                self.compute_dtype,
            )
        # Use -1 instead of `n_heads` (or `n_kv_heads`) to infer the actual
        # local heads from sizes of q and kv as TP may have sharded them after
        # the above linear ops.
        q = q.view(bsz, seqlen, -1, self.qk_head_dim)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        q = torch.cat([q_nope, q_pe], dim=-1)  # (bsz, seqlen, n_heads, qk_head_dim)

        # Key-value projection
        kv = _linear_compute(
            x,
            self.wkv_a.weight,
            self.wkv_a.bias,
            self.compute_dtype,
        )  # (bsz, seqlen, kv_lora_rank + qk_rope_head_dim)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        k_pe = apply_rotary_emb(
            k_pe.unsqueeze(2), freqs_cis
        )  # (bsz, seqlen, 1, qk_rope_head_dim)

        kv = _linear_compute(
            _rms_norm_compute(kv, self.kv_norm, self.compute_dtype),
            self.wkv_b.weight,
            self.wkv_b.bias,
            self.compute_dtype,
        )  # (bsz, seqlen, n_heads * (qk_nope_head_dim + v_head_dim))
        kv = kv.view(bsz, seqlen, -1, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = torch.cat(
            [k_nope, k_pe.expand(-1, -1, self.n_heads, -1)], dim=-1
        )  # (bsz, seqlen, n_heads, qk_head_dim)

        q = q.transpose(1, 2)  # (bsz, n_heads, seqlen, qk_head_dim)
        k = k.transpose(1, 2)  # (bsz, n_heads, seqlen, qk_head_dim)
        v = v.transpose(1, 2)  # (bsz, n_heads, seqlen, v_head_dim)
        q = _to_compute_dtype(q, self.compute_dtype)
        k = _to_compute_dtype(k, self.compute_dtype)
        v = _to_compute_dtype(v, self.compute_dtype)

        output = self.sdpa(q, k, v, scale=self.softmax_scale)

        # Reshape and project output
        output = output.transpose(
            1, 2
        ).contiguous()  # (bsz, seqlen, n_heads, v_head_dim)
        output = output.view(bsz, seqlen, -1)  # (bsz, seqlen, n_heads * v_head_dim)
        return _linear_compute(
            output,
            self.wo.weight,
            self.wo.bias,
            self.compute_dtype,
        )  # (bsz, seqlen, dim)

    def init_weights(self, init_std: float):
        linear_list = [
            self.wkv_a,
            self.wkv_b,
        ]
        if self.q_lora_rank > 0:
            linear_list.extend([self.wq_a, self.wq_b])
        else:
            linear_list.append(self.wq)

        for linear in linear_list:
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

        self.kv_norm.reset_parameters()
        if self.q_lora_rank > 0:
            self.q_norm.reset_parameters()


class TransformerBlock(nn.Module):
    """
    Transformer block with attention and feed-forward layers.
    """

    def __init__(
        self,
        layer_id: int,
        layer_config,
        model_config,
        mesh: DeviceMesh | None = None,
        compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        dim = model_config.dim
        self.compute_dtype = compute_dtype
        self.attention = Attention(
            layer_config.attention,
            model_config,
            compute_dtype=compute_dtype,
        )
        self.attention_norm = nn.RMSNorm(dim, eps=layer_config.attention_norm.eps)
        self.ffn_norm = nn.RMSNorm(dim, eps=layer_config.ffn_norm.eps)

        self.moe_enabled = layer_config.moe is not None
        if self.moe_enabled:
            moe_cfg = layer_config.moe
            self.moe = MoE(
                dim=dim,
                hidden_dim=moe_cfg.experts.hidden_dim,
                num_experts=moe_cfg.num_experts,
                top_k=moe_cfg.router.top_k,
                shared_experts_hidden_dim=moe_cfg.shared_experts.w1.out_features,
                score_func=moe_cfg.router.score_func,
                route_norm=moe_cfg.router.route_norm,
                route_scale=moe_cfg.router.route_scale,
                score_before_experts=moe_cfg.experts.token_dispatcher.score_before_experts,
                use_grouped_mm=moe_cfg.experts.use_grouped_mm,
                load_balance_coeff=moe_cfg.load_balance_coeff,
                mesh=mesh,
                compute_dtype=compute_dtype,
            )
        else:
            ff_cfg = layer_config.feed_forward
            self.feed_forward = FeedForward(
                dim,
                ff_cfg.w1.out_features,
                compute_dtype=compute_dtype,
            )

        self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        self.layer_id = layer_id

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        x = x + self.attention(
            _rms_norm_compute(x, self.attention_norm, self.compute_dtype),
            freqs_cis,
        )
        if self.moe_enabled:
            x = x + self.moe(_rms_norm_compute(x, self.ffn_norm, self.compute_dtype))
        else:
            x = x + self.feed_forward(
                _rms_norm_compute(x, self.ffn_norm, self.compute_dtype)
            )
        return x

    def init_weights(self, buffer_device: torch.device):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        if self.moe_enabled:
            self.moe.init_weights(self.weight_init_std, buffer_device)
        else:
            self.feed_forward.init_weights(self.weight_init_std)


class DeepSeekV3Model(nn.Module):
    """
    DeepSeek-V3 Transformer model with attention and feed-forward layers.
    """

    def __init__(
        self,
        config,
        mesh: DeviceMesh | None = None,
        compute_dtype: torch.dtype | None = None,
    ):
        # Explicitly call nn.Module.__init__ to avoid MRO issues when this class
        # is used with multiple inheritance (e.g., with ModelProtocol in torchtitan)
        nn.Module.__init__(self)
        self.compute_dtype = compute_dtype
        self.max_seq_len = config.rope.max_seq_len
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.register_buffer(
            "freqs_cis", precompute_freqs_cis(config), persistent=False
        )

        self.layers = torch.nn.ModuleDict()
        for layer_id, layer_config in enumerate(config.layers):
            self.layers[str(layer_id)] = TransformerBlock(
                layer_id,
                layer_config,
                config,
                mesh,
                compute_dtype=compute_dtype,
            )

        self.norm = nn.RMSNorm(config.dim, eps=config.norm.eps)
        self.lm_head = nn.Linear(
            config.dim,
            config.vocab_size,
            dtype=torch.get_default_dtype(),
            bias=False,
        )
        self.model_args = config

    def init_weights(
        self, buffer_device: torch.device | None = None, seed: int | None = None
    ) -> None:
        _init_weights_tok_embeddings(self, seed)
        _init_weights_layers(self, buffer_device, seed)
        _init_weights_norm_and_output(self)

    def forward(
        self,
        tokens: torch.Tensor,
        input_batch: torch.Tensor | None = None,
    ):
        """
        Forward pass for the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices if pipeline parallelism is not enabled.
                If pipeline parallelism is enabled, this will be the input token indices
                for the ranks on the first pipeline stage. This will be the activation of the
                previous pipeline stage if the current rank is not on the first stage.
            input_batch (torch.Tensor): The input batch read from the dataloader.
                This will always be the input batch regardless of the pipeline stage.
                This field is required for non-first PP stages to perform document
                masking attention (to analyze the boundary of the document).

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, vocab_size).
        """

        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
        h = _to_compute_dtype(h, self.compute_dtype)

        for layer in self.layers.values():
            h = layer(h, self.freqs_cis)
        h = (
            _rms_norm_compute(h, self.norm, self.compute_dtype)
            if self.norm is not None
            else h
        )
        output = (
            _linear_compute(
                h,
                self.lm_head.weight,
                self.lm_head.bias,
                self.compute_dtype,
            )
            if self.lm_head is not None
            else h
        )
        return output


def _init_weights_tok_embeddings(self: DeepSeekV3Model, seed: int | None = None):
    if seed is not None:
        torch.manual_seed(seed)
    if self.tok_embeddings is not None:
        nn.init.normal_(self.tok_embeddings.weight)


def _init_weights_layers(
    self: DeepSeekV3Model,
    buffer_device: torch.device | None,
    seed: int | None = None,
):
    if buffer_device is None:
        buffer_device = self.freqs_cis.device  # type: ignore[assignment]
    with torch.device(buffer_device):  # type: ignore[arg-type]
        self.freqs_cis = precompute_freqs_cis(self.model_args)
    for i, layer in enumerate(self.layers.values()):
        if seed is not None:
            torch.manual_seed(seed)
        if layer is not None:
            assert isinstance(layer, TransformerBlock)
            layer.init_weights(buffer_device)  # type: ignore[arg-type]


def _init_weights_norm_and_output(self: DeepSeekV3Model):
    if self.norm is not None:
        self.norm.reset_parameters()
    if self.lm_head is not None:
        final_out_std = self.model_args.dim**-0.5
        cutoff_factor = 3
        nn.init.trunc_normal_(
            self.lm_head.weight,
            mean=0.0,
            std=final_out_std,
            a=-cutoff_factor * final_out_std,
            b=cutoff_factor * final_out_std,
        )


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


def annotate_deepseekv3_for_graph_trainer(model: DeepSeekV3Model) -> None:
    """Attach graph_trainer-compatible FX annotations to AP's DSv3 model."""
    global local_mapped_region

    local_mapped_region = _annotate_once(
        local_mapped_region,
        {"EP": "compute"},
    )
    MoE.forward = _annotate_once(  # type: ignore[method-assign]
        MoE.forward,
        {"EP": "compute"},
    )
    _annotate_module_fqns(model)
