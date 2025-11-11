# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Callable, Literal

import torch
import torch.nn.functional as F

from autoparallel._testing.models.moe_utils import (
    generate_permute_indices,
    TOKEN_GROUP_ALIGN_SIZE_M,
)
from torch import nn
from torch.distributed.tensor.placement_types import Replicate, Shard


def expert_parallel(func: Callable) -> Callable:
    """
    This is a wrapper applied to the GroupedExperts computation, serving
    the following three purposes:
    1. Convert parameters from DTensors to plain Tensors, to work with
    dynamic-shape inputs which cannot be easily expressed as DTensors.
    2. In Expert Parallel, apply the generate_permute_indices kernel to
    permute the inputs to be ordered by local experts (see the _token_dispatch
    function in ExpertParallel) and permute the outputs back.
    3. In order to use torch._grouped_mm, we need to make sure the number of
    tokens each expert gets is a multiple of ALIGN_SIZE_M. The generate_permute_indices
    kernel also helps achieve this via padding, without incurring synchronization
    between device and host. Note that this will create side effects when wrapping
    the for-loop implementation of GroupedExperts, as it does not need padding.

    Among the above:
    1 and 2 are needed only when expert_parallel_degree > 1.
    3 is needed even for single-device computation.
    2 can be moved to ExpertParallel _token_dispatch if not coupled with 3.
    """

    def wrapper(
        w1: torch.Tensor,
        w2: torch.Tensor,
        w3: torch.Tensor,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:

        experts_per_ep_rank = w1.shape[0]
        num_ep_ranks = num_tokens_per_expert.shape[0] // experts_per_ep_rank

        with torch.no_grad():
            (
                permuted_indices,
                num_tokens_per_expert,
                _,  # offsets,
            ) = generate_permute_indices(
                num_tokens_per_expert,
                experts_per_ep_rank,
                num_ep_ranks,
                x.shape[0] + experts_per_ep_rank * TOKEN_GROUP_ALIGN_SIZE_M,
                TOKEN_GROUP_ALIGN_SIZE_M,
            )

        x = torch.vstack((x, x.new_zeros((x.shape[-1]))))
        input_shape = x.shape
        x = x[permuted_indices, :]

        out = func(w1, w2, w3, x, num_tokens_per_expert)

        out_unpermuted = out.new_empty(input_shape)
        out_unpermuted[permuted_indices, :] = out
        out = out_unpermuted[:-1]

        return out

    return wrapper


@dataclass
class MoEArgs:
    num_experts: int = 8
    num_shared_experts: int = 1

    # router
    score_func: Literal["softmax", "sigmoid"] = "sigmoid"
    route_norm: bool = False
    route_scale: float = 1.0
    score_before_experts: bool = True

    # token-choice
    top_k: int = 1
    use_grouped_mm: bool = True  # grouped mm or for-loop for the experts computation
    load_balance_coeff: float | None = 1e-3


def _run_shared_experts(
    shared_w1: torch.Tensor,
    shared_w2: torch.Tensor,
    shared_w3: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:

    h = F.silu(x @ shared_w1.transpose(-2, -1))
    h = h * x @ shared_w3.transpose(-2, -1)
    out = h @ shared_w2.transpose(-2, -1)
    return out


@expert_parallel
def _run_experts_grouped_mm(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
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


def _topk_token_choice_router(
    x: torch.Tensor,
    gate: torch.Tensor,
    top_k: int,
    num_experts: int,
    score_func: Literal["softmax", "sigmoid"],
    route_norm: bool,
    route_scale: float,
    expert_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # scores shape (bs*slen, num_experts)
    scores = x @ gate.transpose(-2, -1)

    # By default, sigmoid or softmax is performed in float32 to avoid loss explosion
    if score_func == "sigmoid":
        scores = torch.sigmoid(scores.to(torch.float32))
    elif score_func == "softmax":
        scores = F.softmax(scores.to(torch.float32), dim=1)
    else:
        raise NotImplementedError(f"Unknown score function {score_func}")

    # top scores shape (bs*slen, top_k)
    # NOTE: The expert_bias is only used for routing. The gating value
    #       top_scores is still derived from the original scores.
    if expert_bias is not None:
        _, selected_experts_indices = torch.topk(scores + expert_bias, k=top_k, dim=1)
        top_scores = scores.gather(dim=1, index=selected_experts_indices)
    else:
        top_scores, selected_experts_indices = torch.topk(scores, k=top_k, dim=1)

    if score_func == "sigmoid" and route_norm:
        denominator = top_scores.sum(dim=-1, keepdim=True) + 1e-20
        top_scores = top_scores / denominator
    top_scores = top_scores * route_scale

    # group tokens together by expert indices from 0 to num_experts and pass that to experts forward
    num_tokens_per_expert = torch.histc(
        selected_experts_indices.view(-1),
        bins=num_experts,
        min=0,
        max=num_experts,
    )

    return top_scores, selected_experts_indices, num_tokens_per_expert


def _reorder_tokens_by_experts(
    top_scores: torch.Tensor,
    selected_experts_indices: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    top_k: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Reorders token indices to match the order of experts for MoE routing.
    # NOTE: the reason we need to compute num_tokens_per_expert again is:
    #       1st computation in router is to update self.tokens_per_expert
    #       which would be the same across all TP ranks.
    #       2nd computation in reorderer is for the actual routing and experts computation
    #       which would be sharded over TP ranks if expert_tensor_parallel_degree==1.
    #       If tensor_paralllel_degree == expert_tensor_parallel_degree, they agree.
    # num_tokens_per_expert = torch.histc(
    #     selected_experts_indices.view(-1),
    #     bins=num_experts,
    #     min=0,
    #     max=num_experts,
    # )

    # Reorder the token indices to match the order of the experts
    # token_indices_experts_sorted shape (bs*slen*top_k,)
    token_indices_experts_sorted = torch.argsort(
        selected_experts_indices.view(-1), stable=True
    )

    top_scores_experts_sorted = top_scores.view(-1)[token_indices_experts_sorted]
    token_indices_experts_sorted = token_indices_experts_sorted // top_k

    return (
        top_scores_experts_sorted,
        token_indices_experts_sorted,
    )


def _moe_forward(
    x: torch.Tensor,
    gate: torch.Tensor,
    expert_bias: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    shared_w1: torch.Tensor,
    shared_w2: torch.Tensor,
    shared_w3: torch.Tensor,
    top_k: int,
    num_experts: int,
    score_func: Literal["softmax", "sigmoid"],
    route_norm: bool,
    route_scale: float,
    score_before_experts: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    bs, slen, dim = x.shape
    x = x.view(-1, dim)
    # top_scores and selected_experts_indices shape (bs*slen*top_k,)
    # num_tokens_per_expert shape (num_experts,)
    (
        top_scores,
        selected_experts_indices,
        num_tokens_per_expert,
    ) = _topk_token_choice_router(
        x,
        gate,
        top_k,
        num_experts,
        score_func,
        route_norm,
        route_scale,
        expert_bias,
    )
    # top_scores_experts_sorted and token_indices_experts_sorted shape (bs*slen*top_k,)
    (
        top_scores_experts_sorted,
        token_indices_experts_sorted,
    ) = _reorder_tokens_by_experts(
        top_scores,
        selected_experts_indices,
        num_tokens_per_expert,
        top_k,
        num_experts,
    )

    # shape (bs*slen*top_k, dim)
    token_indices_experts_sorted = token_indices_experts_sorted.reshape(-1, 1).expand(
        -1, dim
    )

    # shape (bs*slen*top_k, dim)
    routed_input = torch.gather(x, dim=0, index=token_indices_experts_sorted)

    if score_before_experts:
        routed_input = (
            routed_input.to(torch.float32) * top_scores_experts_sorted.reshape(-1, 1)
        ).to(x.dtype)

    # shape (bs*slen*top_k, dim)
    routed_output = _run_experts_grouped_mm(
        routed_input, w1, w2, w3, num_tokens_per_expert
    )

    if not score_before_experts:
        routed_output = (
            routed_output.to(torch.float32) * top_scores_experts_sorted.reshape(-1, 1)
        ).to(x.dtype)

    # shared expert
    # shape (bs*slen*top_k, dim)
    out = _run_shared_experts(shared_w1, shared_w2, shared_w3, x)
    out = out.scatter_add(dim=0, index=token_indices_experts_sorted, src=routed_output)
    out = out.reshape(bs, slen, dim)
    return out, num_tokens_per_expert


class MoE(nn.Module):
    def __init__(self, moe_args: MoEArgs, dim: int, hidden_dim: int):
        super().__init__()

        # Routed Experts
        self.num_experts = moe_args.num_experts
        self.w1 = nn.Parameter(torch.empty(self.num_experts, hidden_dim, dim))
        self.w2 = nn.Parameter(torch.empty(self.num_experts, dim, hidden_dim))
        self.w3 = nn.Parameter(torch.empty(self.num_experts, hidden_dim, dim))

        # Router
        self.top_k = moe_args.top_k
        self.score_func = moe_args.score_func
        self.route_norm = moe_args.route_norm
        self.route_scale = moe_args.route_scale
        self.gate = nn.Parameter(torch.empty(self.num_experts, dim))

        # Shared Experts
        self.use_shared_experts = moe_args.num_shared_experts > 0
        if self.use_shared_experts:
            self.shared_w1 = nn.Parameter(
                torch.empty(hidden_dim * moe_args.num_shared_experts, dim)
            )
            self.shared_w2 = nn.Parameter(
                torch.empty(dim, hidden_dim * moe_args.num_shared_experts)
            )
            self.shared_w3 = nn.Parameter(
                torch.empty(hidden_dim * moe_args.num_shared_experts, dim)
            )

        self.score_before_experts = moe_args.score_before_experts

        # define fields for auxiliary-loss-free load balancing (https://arxiv.org/abs/2408.15664)
        # NOTE: tokens_per_expert is accumulated in the model forward pass.
        #       expert_bias is updated outside the model in an optimizer step pre hook
        #       to work with gradient accumulation.
        self.load_balance_coeff = moe_args.load_balance_coeff
        if self.load_balance_coeff is not None:
            assert self.load_balance_coeff > 0.0
            self.register_buffer(
                "expert_bias",
                torch.zeros(self.num_experts, dtype=torch.float32),
                persistent=True,
            )
        else:
            self.expert_bias = None
        # tokens_per_expert will be used to track expert usage and to update the expert bias for load balancing
        self.register_buffer(
            "tokens_per_expert",
            torch.zeros(self.num_experts, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs, slen, dim)``.

        Returns:
            out (torch.Tensor): Output tensor with shape ``(bs, slen, dim)``.
        """
        # tokens_per_expert will be used to update the expert bias for load balancing.
        # and also to count the expert usage
        # TODO: Activation Checkpointing has the side effect of double counting tokens_per_expert --
        #       first in the forward pass, and then in the backward pass. However, this has no
        #       effect on the expert bias update thanks to the torch.sign() operator.
        # moved out to remove mutation
        assert self.expert_bias is not None, "Load balance coeff must be set"
        assert self.use_shared_experts, "Shared experts must be enabled"
        out, num_tokens_per_expert = _moe_forward(
            x,
            self.gate,
            self.expert_bias,
            self.w1,
            self.w2,
            self.w3,
            self.shared_w1,
            self.shared_w2,
            self.shared_w3,
            self.top_k,
            self.num_experts,
            self.score_func,
            self.route_norm,
            self.route_scale,
            self.score_before_experts,
        )

        # HOPs don't support buffer mutations, keep this outside
        with torch.no_grad():
            self.tokens_per_expert.add_(num_tokens_per_expert)
        return out

    def init_weights(
        self,
        init_std: float,
        buffer_device: torch.device,
    ):
        # Initialize Routed Expert Weights
        nn.init.trunc_normal_(self.w1, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.w3, mean=0.0, std=init_std)

        # Initialize Router Weight
        nn.init.trunc_normal_(self.gate, mean=0.0, std=init_std)

        # Initialize Shared Expert Weights
        if self.use_shared_experts:
            nn.init.trunc_normal_(self.shared_w1, mean=0.0, std=0.02)
            nn.init.trunc_normal_(self.shared_w2, mean=0.0, std=init_std)
            nn.init.trunc_normal_(self.shared_w3, mean=0.0, std=init_std)

        # Initialize Buffers
        with torch.device(buffer_device):
            self.tokens_per_expert = torch.zeros(
                self.experts.num_experts, dtype=torch.float32
            )
            if self.load_balance_coeff is not None:
                self.expert_bias = torch.zeros(
                    self.experts.num_experts, dtype=torch.float32
                )
