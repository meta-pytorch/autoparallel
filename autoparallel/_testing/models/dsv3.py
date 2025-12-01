# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
from typing import Callable, ClassVar, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn

# from torchtitan.distributed.expert_parallel import expert_parallel
from torch.distributed.tensor import DeviceMesh, DTensor
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.nn.attention import SDPBackend, sdpa_kernel

from autoparallel.collectives import all_to_all, axis_size, local_map


# parallelized kernel
@triton.jit
def _fill_indices_kernel(
    tokens_per_expert_group_ptr,
    start_index_values_ptr,
    write_offsets_ptr,
    output_ptr,
    experts_per_rank: tl.constexpr,
    num_ranks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # Number of threads per block
):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)

    # map programs (blocks) to the experts and loop (grid stride) if needed
    for expert_id in range(pid, experts_per_rank, num_programs):
        # read this experts write offset
        write_offset = tl.load(write_offsets_ptr + expert_id)

        for r in range(num_ranks):
            # index into tokens_per_expert_group array
            i = r * experts_per_rank + expert_id

            # load start index and number of tokens for this expert-rank pair
            start_index = tl.load(start_index_values_ptr + i)
            length = tl.load(tokens_per_expert_group_ptr + i)

            # each thread in block processes tokens in parallel
            offsets = tl.arange(0, BLOCK_SIZE)

            # tokens are processed in chunks of BLOCK_SIZE
            for chunk_start in range(0, length, BLOCK_SIZE):
                chunk_offsets = chunk_start + offsets

                # mask valid indices
                mask = chunk_offsets < length

                values = start_index + chunk_offsets

                # destination
                dest_indices = write_offset + chunk_offsets

                # store
                tl.store(output_ptr + dest_indices, values, mask=mask)

            # update write offset for next rank
            write_offset += length


# ==============
# wrapper
# ==============


# workaround until local_map functionalization is fixed: https://github.com/pytorch/pytorch/issues/167568
@torch.library.custom_op("autoparallel::fill_indices_functional", mutates_args=())
def fill_indices_functional(
    tokens_per_expert_group: torch.Tensor,
    start_index_values: torch.Tensor,
    write_offsets: torch.Tensor,
    experts_per_rank: int,
    num_ranks: int,
    max_len: int,
    block_size: int = 128,
    max_blocks: int = 1024,  # cap on total number of blocks to launch
) -> torch.Tensor:
    # preallocate output
    permuted_indices = torch.full(
        (max_len,), -1, dtype=torch.int32, device=tokens_per_expert_group.device
    )

    # write offsets is per local expert...
    num_blocks = min(experts_per_rank, max_blocks)
    # grid = one block per expert unless capped and then we loop...
    grid = (num_blocks,)

    # launch kernel
    _fill_indices_kernel[grid](
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
    max_blocks: int = 1024,  # cap on total number of blocks to launch
) -> torch.Tensor:
    return torch.full(
        (max_len,), -1, dtype=torch.int32, device=tokens_per_expert_group.device
    )


# reference
def fill_indices_cpu(
    tokens_per_expert_group: torch.Tensor,
    start_index_values: torch.Tensor,
    write_offsets: torch.Tensor,
    experts_per_rank: int,
    num_ranks: int,
    max_len: int,
):
    # We need to preallocate the output - we ignore device and force it on cpu
    # device = tokens_per_expert_group.device
    permuted_indices = torch.full(
        (max_len,),
        -1,
        dtype=torch.int32,
    )  # device=device)
    # Fill the permuted indices
    # For each local expert
    for e in range(experts_per_rank):
        write_start = write_offsets[e].item()
        assert isinstance(write_start, int)
        # For each remote rank
        for r in range(num_ranks):
            i: int = r * experts_per_rank + e
            start_index = start_index_values[i].item()
            length = tokens_per_expert_group[i].item()
            assert isinstance(start_index, int)
            assert isinstance(length, int)
            # Fill in the indices
            if length > 0:
                end_idx: int = min(write_start + length, max_len)
                permuted_indices[write_start:end_idx] = torch.arange(
                    start_index,
                    start_index + (end_idx - write_start),
                    dtype=torch.int32,
                )
            write_start += length
    return permuted_indices


def generate_permute_indices(
    tokens_per_expert_group: torch.Tensor,
    experts_per_rank: int,
    num_ranks: int,
    max_len: int,
    alignment: int,
    use_cpu: bool = False,
):
    """
    Prepare permutation indices and the number of tokens for each expert.

    Args:
        tokens_per_expert_group: number of tokens for each expert from all ranks.
        experts_per_rank: number of experts per rank.
        num_ranks: number of ranks.
        max_len: maximum length of the output index vector.
        alignment: alignment for each returned element in `m_sizes` and padding min for zero token experts.
        use_cpu: whether to use CPU implementation.


    Returns:
        permuted_indices: Tensor of indices that map original token order to the expert-grouped order.
        m_sizes: aligned number of tokens for each expert (padded to alignment boundary).
        m_offsets: Cumulative sum of m_sizes. The exclusive ending position for each expert's tokens.

    Explanatory details:
        `tokens_per_expert_group` is of shape (num_ranks * experts_per_rank,), for example:
        From: |       rank 0      |       rank 1      |
        To:   | E0 | E1 | E2 | E3 | E0 | E1 | E2 | E3 |
              |  4 |  2 |  1 |  3 |  1 |  2 |  3 |  4 |
    """

    # prefix sum to get start index of each expert (parallel scan kernel in future?)
    start_index_values = (
        torch.cumsum(tokens_per_expert_group, 0) - tokens_per_expert_group
    )

    # total tokens for each expert (sum over ranks)
    total_tokens_per_expert = tokens_per_expert_group.view(num_ranks, -1).sum(0)

    # pad out empty experts to alignment requirement
    total_tokens_per_expert = torch.clamp_min(total_tokens_per_expert, alignment)

    # align the chunk sizes (cdiv)
    m_sizes = ((total_tokens_per_expert + alignment - 1) // alignment * alignment).to(
        torch.int32
    )

    # additional prefix sum to get write offset of each expert in permuted_indices
    # write offsets is per local expert, not global
    m_offsets = torch.cumsum(m_sizes, 0)
    write_offsets = m_offsets - m_sizes

    # Select the implementation to use
    if use_cpu:
        permuted_indices = fill_indices_cpu(
            tokens_per_expert_group,
            start_index_values,
            write_offsets,
            experts_per_rank,
            num_ranks,
            max_len,
        )
    else:
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
    """Round up x to the nearest multiple of y."""
    x_ceil_div_y = (x + y - 1) // y
    return x_ceil_div_y * y


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
        global TOKEN_GROUP_ALIGN_SIZE_M
        if isinstance(w1, DTensor):
            assert isinstance(w2, DTensor)
            assert isinstance(w3, DTensor)
            w1 = w1.to_local()
            w2 = w2.to_local()
            w3 = w3.to_local()

        experts_per_ep_rank = w1.shape[0]
        num_ep_ranks = num_tokens_per_expert.shape[0] // experts_per_ep_rank
        # assert (
        #     num_ep_ranks == 64
        # ), f"{num_ep_ranks}, {experts_per_ep_rank}, num_tokens_per_expert.shape: {num_tokens_per_expert.shape}, x={x.ndim}, w={w1.shape}"

        # Make sure max_len of permuted token indicies is divisible by TOKEN_GROUP_ALIGN_SIZE_M,
        # by padding it to the nearest multiple of TOKEN_GROUP_ALIGN_SIZE_M.
        x_padded_per_expert = (
            x.shape[0] + experts_per_ep_rank * TOKEN_GROUP_ALIGN_SIZE_M
        )
        padded_max_len = _round_up(x_padded_per_expert, TOKEN_GROUP_ALIGN_SIZE_M)
        with torch.no_grad():
            (
                permuted_indices,
                num_tokens_per_expert,
                _,  # offsets,
            ) = generate_permute_indices(
                num_tokens_per_expert,
                experts_per_ep_rank,
                num_ep_ranks,
                padded_max_len,
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


def functional_feed_forward(w1, w2, w3, x):
    return F.linear(F.silu(F.linear(x, w1)) * F.linear(x, w3), w2)


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
    ):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float = 0.02):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


# TODO: keeping this for-loop implementation for comparison
#       and readability, may remove later
@expert_parallel
def _run_experts_for_loop(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x_: torch.Tensor,
    num_tokens_per_expert_: torch.Tensor,
) -> torch.Tensor:
    # NOTE: this would incur a synchronization between device and host
    num_tokens_per_expert: list[int] = num_tokens_per_expert_.tolist()

    # side-effect code due to the usage of generate_permute_indices
    num_padding: int = x_.shape[0] - sum(num_tokens_per_expert)

    # a tuple of tensors indexed by experts
    # each with shape (tokens_per_expert(varying), dim)
    x: tuple[torch.Tensor, ...] = torch.split(
        x_[: sum(num_tokens_per_expert)],
        split_size_or_sections=num_tokens_per_expert,
        dim=0,
    )
    out_experts_splits = []
    for expert_idx, x_expert in enumerate(x):
        h = F.silu(torch.matmul(x_expert, w1[expert_idx].transpose(-2, -1)))
        h = h * torch.matmul(x_expert, w3[expert_idx].transpose(-2, -1))
        h = torch.matmul(h, w2[expert_idx].transpose(-2, -1))
        # h shape (tokens_per_expert(varying), dim)
        out_experts_splits.append(h)
    out = torch.cat(out_experts_splits, dim=0)

    # side-effect code due to the usage of generate_permute_indices
    out = torch.vstack((out, out.new_zeros((num_padding, out.shape[-1]))))

    return out


@expert_parallel
def _run_experts_grouped_mm(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
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
        scores = torch.nn.functional.linear(x, gate_weight)

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


def _token_dispatch(routed_input, num_tokens_per_expert, axis_name):
    # annotate module input placements/sharding with input_layouts
    # ep_size = device_mesh.shape[0]
    ep_size = axis_size(axis_name)

    # generate the input splits and output splits for all-to-all
    with torch.no_grad():
        num_tokens_per_expert_group = all_to_all(
            num_tokens_per_expert,
            None,
            None,
            axis_name,
        )
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

    # perform all-to-all
    routed_input = all_to_all(
        routed_input,
        output_splits,
        input_splits,
        axis_name,
    )

    # NOTE: After this all-to-all, the routed input is put on proper EP rank.
    # However, the num_tokens_per_expert_group is not of the final target format
    # [#tokens for local expert 0, #tokens for local expert 1, ...]
    # Rather, it is of the format
    # [#tokens for local expert 0 from EP rank 0, #tokens for local expert 1 from EP rank 0, ...,
    #  #tokens for local expert 0 from EP rank 1, #tokens for local expert 1 from EP rank 1, ...]
    # We need to perform another shuffle to get the correct format -- this is done via the function
    # generate_permute_indices in moe.py, which also does padding to make sure the number of tokens
    # each expert gets locally is a multiple of ALIGN_SIZE_M.

    return routed_input, num_tokens_per_expert_group, input_splits, output_splits


def _token_combine(routed_output, input_splits, output_splits, axis_name):
    routed_output = all_to_all(
        routed_output,
        input_splits,
        output_splits,
        axis_name,
    )
    return routed_output


# @torch.library.custom_op("autoparallel::local_mapped_region", mutates_args=())
def local_mapped_region(
    selected_experts_indices: torch.Tensor,
    top_scores: torch.Tensor,
    x: torch.Tensor,
    experts_w1: torch.Tensor,
    experts_w3: torch.Tensor,
    experts_w2: torch.Tensor,
    out: torch.Tensor,
    top_k: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    axis_name = "ep"
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
        selected_experts_indices.flatten(1), dim=-1, stable=True
    )

    # top_scores_experts_sorted = top_scores.view(-1)[token_indices_experts_sorted]
    top_scores_experts_sorted = top_scores.view_as(token_indices_experts_sorted).gather(
        1, token_indices_experts_sorted
    )
    token_indices_experts_sorted = token_indices_experts_sorted // top_k

    # shape (bs*slen*top_k, dim)
    token_indices_experts_sorted = token_indices_experts_sorted[..., None].expand(
        -1, -1, dim
    )

    # shape (bs*slen*top_k, dim)
    routed_input = torch.gather(
        x.view(token_indices_experts_sorted.shape[0], -1, dim),
        dim=1,
        index=token_indices_experts_sorted,
    )
    routed_input = (
        routed_input.to(torch.float32) * top_scores_experts_sorted[..., None]
    ).to(x.dtype)

    shape = routed_input.shape
    dim = shape[-1]
    routed_input = routed_input.view(-1, dim)
    num_tokens_per_expert = num_tokens_per_expert.view(-1)
    (
        routed_input,
        num_tokens_per_expert_group,
        input_splits,
        output_splits,
    ) = _token_dispatch(routed_input, num_tokens_per_expert, axis_name)

    routed_output = _run_experts_grouped_mm(
        # experts_w1, experts_w2, experts_w3, routed_input, num_tokens_per_expert
        experts_w1,
        experts_w2,
        experts_w3,
        routed_input,
        num_tokens_per_expert_group,
    )

    routed_output = _token_combine(
        routed_output, input_splits, output_splits, axis_name
    )

    torch._check(routed_output.shape[0] == shape[0] * shape[1])

    routed_output = routed_output.view(shape)

    out = out.scatter_add(dim=1, index=token_indices_experts_sorted, src=routed_output)
    return out, total_tokens_per_expert[None, :]


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
    router: TokenChoiceTopKRouter,  # None
    reorderer: TokenReorderer,  # None
    mesh: Optional[DeviceMesh],  # None
):
    # x: 64, 2048, 256
    bs, slen, dim = x.shape

    # local_batch_size = 4
    # num_gpus_participating = 32 * 2
    # num_experts_per_groups = local_batch_size * num_gpus_participating
    # x = x.unflatten(0, (-1, num_experts_per_groups))
    # x = x.view(-1, dim)

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

    out = functional_feed_forward(shared_w1, shared_w2, shared_w3, x)

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
    )

    # assert False, f"{x.shape}, {selected_experts_indices.shape}, {top_scores.shape}, {out.shape}"
    # [selected_experts_indices, top_scores_1, rms_norm_2, v_2, v_4, v_3, out]
    out, num_tokens_per_expert = local_map(
        local_mapped_region,
        out_placements=((Shard(0), Shard(0)), (Shard(0), Shard(0))),
        in_placements=reordered_placements,
        redistribute_inputs=True,
        in_grad_placements=None,
        device_mesh=mesh,
    )(
        selected_experts_indices,
        top_scores,
        x,
        experts_w1,
        experts_w3,
        experts_w2,
        out,
        router.top_k,
        router.num_experts,
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
    return out, num_tokens_per_expert.sum(0)


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

    _debug_force_load_balance: bool = False
    # if True, we force each experts get same amount of token via round-robin
    mesh: Optional[DeviceMesh] = None


class MoE(nn.Module):
    def __init__(self, moe_args: MoEArgs, dim: int, hidden_dim: int):
        super().__init__()

        num_experts = moe_args.num_experts
        self.mesh = moe_args.mesh
        self.experts = GroupedExperts(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            use_grouped_mm=moe_args.use_grouped_mm,
        )
        self.router = TokenChoiceTopKRouter(
            dim=dim,
            num_experts=num_experts,
            top_k=moe_args.top_k,
            score_func=moe_args.score_func,
            route_norm=moe_args.route_norm,
            route_scale=moe_args.route_scale,
        )
        self.reorderer = TokenReorderer(num_experts=num_experts, top_k=moe_args.top_k)
        assert moe_args.num_shared_experts > 0
        self.shared_experts = FeedForward(
            dim=dim, hidden_dim=hidden_dim * moe_args.num_shared_experts
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
            self.router,  # None
            self.reorderer,  # None
            self.mesh,  # None
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


# Reference: https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
@dataclass
class DeepSeekV3ModelArgs:
    """
    Data class for defining model arguments and hyperparameters.

    Attributes:
        max_batch_size (int): Maximum batch size.
        max_seq_len (int): Maximum sequence length.
        vocab_size (int): Vocabulary size.
        dim (int): Model dimension.
        inter_dim (int): Intermediate dimension for MLP layers.
        moe_inter_dim (int): Intermediate dimension for MoE layers.
        n_layers (int): Number of transformer layers.
        n_dense_layers (int): Number of dense layers in the model.
        n_heads (int): Number of attention heads.
        norm_eps (float): Epsilon value used for RMSNorm.
        moe_args (MoEArgs): MoE configuration.
        n_expert_groups (int): Number of expert groups.
        n_limited_groups (int): Number of limited groups for MoE routing.
        q_lora_rank (int): LoRA rank for query projections.
        kv_lora_rank (int): LoRA rank for key-value projections.
        qk_nope_head_dim (int): Dimension for query-key projections without positional embeddings.
        qk_rope_head_dim (int): Dimension for query-key projections with rotary embeddings.
        v_head_dim (int): Dimension for value projections.
        use_flex_attn (bool): Whether to use FlexAttention.
        attn_mask_type (str): Type of attention mask.
        original_seq_len (int): Original sequence length.
        rope_theta (float): Base for rotary positional encoding.
        rope_factor (float): Scaling factor for extended sequence lengths.
        beta_fast (int): Fast beta correction factor.
        beta_slow (int): Slow beta correction factor.
    """

    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4
    vocab_size: int = 102400
    dim: int = 2048
    inter_dim: int = 10944
    moe_inter_dim: int = 1408
    n_layers: int = 27
    n_dense_layers: int = 1
    n_heads: int = 16
    norm_eps: float = 1e-5  # eps used for RMSNorm

    # MoE
    moe_args: MoEArgs = field(default_factory=MoEArgs)
    # TODO: node-limited routing is not supported yet
    n_expert_groups: int = 1
    n_limited_groups: int = 1

    # Multi-Head Latent Attention (MLA)
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    use_flex_attn: bool = False
    attn_mask_type: str = "causal"

    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0


# Adapted from https://github.com/DeepSeek-ai/DeepSeek-V3/blob/main/inference/model.py#L294
def precompute_freqs_cis(args: DeepSeekV3ModelArgs) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args (DeepSeekV3ModelArgs): Model arguments containing positional embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

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
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(
            beta_fast, beta_slow, dim, base, args.original_seq_len
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

    def __init__(self, model_args: DeepSeekV3ModelArgs):
        super().__init__()
        self.dim = model_args.dim
        self.n_heads = model_args.n_heads
        self.q_lora_rank = model_args.q_lora_rank
        self.kv_lora_rank = model_args.kv_lora_rank
        self.qk_nope_head_dim = model_args.qk_nope_head_dim
        self.qk_rope_head_dim = model_args.qk_rope_head_dim
        self.qk_head_dim = model_args.qk_nope_head_dim + model_args.qk_rope_head_dim
        self.v_head_dim = model_args.v_head_dim

        if self.q_lora_rank == 0:
            self.wq = nn.Linear(self.dim, self.n_heads * self.qk_head_dim, bias=False)
        else:
            self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False)
            self.q_norm = nn.RMSNorm(self.q_lora_rank, eps=model_args.norm_eps)
            self.wq_b = nn.Linear(
                self.q_lora_rank, self.n_heads * self.qk_head_dim, bias=False
            )
        self.wkv_a = nn.Linear(
            self.dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=False
        )
        self.kv_norm = nn.RMSNorm(self.kv_lora_rank, eps=model_args.norm_eps)
        self.wkv_b = nn.Linear(
            self.kv_lora_rank,
            self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim, bias=False)
        self.softmax_scale = self.qk_head_dim**-0.5

        if model_args.max_seq_len > model_args.original_seq_len:
            mscale = 0.1 * model_args.mscale * math.log(model_args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        self.sdpa = build_attention(model_args.use_flex_attn, model_args.attn_mask_type)

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
            q = self.wq(x)  # (bsz, seqlen, n_heads * qk_head_dim)
        else:
            q = self.wq_a(x)
            q = self.wq_b(self.q_norm(q))
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
        kv = self.wkv_a(x)  # (bsz, seqlen, kv_lora_rank + qk_rope_head_dim)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        k_pe = apply_rotary_emb(
            k_pe.unsqueeze(2), freqs_cis
        )  # (bsz, seqlen, 1, qk_rope_head_dim)

        kv = self.wkv_b(
            self.kv_norm(kv)
        )  # (bsz, seqlen, n_heads * (qk_nope_head_dim + v_head_dim))
        kv = kv.view(bsz, seqlen, -1, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = torch.cat(
            [k_nope, k_pe.expand(-1, -1, self.n_heads, -1)], dim=-1
        )  # (bsz, seqlen, n_heads, qk_head_dim)

        q = q.transpose(1, 2)  # (bsz, n_heads, seqlen, qk_head_dim)
        k = k.transpose(1, 2)  # (bsz, n_heads, seqlen, qk_head_dim)
        v = v.transpose(1, 2)  # (bsz, n_heads, seqlen, v_head_dim)

        output = self.sdpa(q, k, v, scale=self.softmax_scale)

        # Reshape and project output
        output = output.transpose(
            1, 2
        ).contiguous()  # (bsz, seqlen, n_heads, v_head_dim)
        output = output.view(bsz, seqlen, -1)  # (bsz, seqlen, n_heads * v_head_dim)
        return self.wo(output)  # (bsz, seqlen, dim)

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

    def __init__(self, layer_id: int, model_args: DeepSeekV3ModelArgs):

        super().__init__()
        self.attention = Attention(model_args)
        self.attention_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)

        self.moe_enabled = layer_id >= model_args.n_dense_layers
        if self.moe_enabled:
            self.moe = MoE(
                model_args.moe_args,
                dim=model_args.dim,
                hidden_dim=model_args.moe_inter_dim,
            )
        else:
            self.feed_forward = FeedForward(model_args.dim, model_args.inter_dim)

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
        x = x + self.attention(self.attention_norm(x), freqs_cis)
        if self.moe_enabled:
            x = x + self.moe(self.ffn_norm(x))
        else:
            x = x + self.feed_forward(self.ffn_norm(x))
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

    def __init__(self, model_args: DeepSeekV3ModelArgs):
        super().__init__()
        self.max_seq_len = model_args.max_seq_len
        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)
        self.register_buffer(
            "freqs_cis", precompute_freqs_cis(model_args), persistent=False
        )

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, model_args)

        self.norm = nn.RMSNorm(model_args.dim)
        self.output = nn.Linear(
            model_args.dim,
            model_args.vocab_size,
            dtype=torch.get_default_dtype(),
            bias=False,
        )
        self.model_args = model_args

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

        for layer in self.layers.values():
            h = layer(h, self.freqs_cis)
        h = self.norm(h) if self.norm is not None else h
        output = self.output(h) if self.output is not None else h
        return output


def dsv3_loss_fn(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(
        pred.flatten(0, 1).float(), labels.flatten(0, 1)
    )


########################
# Pipeline stuff start #
########################


class DeepSeekV3StageI(nn.Module):
    def __init__(self, layers, model_args):
        super().__init__()
        self.layers = layers
        self.register_buffer(
            "freqs_cis", precompute_freqs_cis(model_args), persistent=False
        )
        self.model_args = model_args

    def forward(self, h):
        # intermediate stages only have layers
        for layer in self.layers.values():
            h = layer(h, self.freqs_cis)
        return h

    def init_weights(
        self, buffer_device: torch.device | None = None, seed: int | None = None
    ) -> None:
        _init_weights_layers(self, buffer_device, seed)


class DeepSeekV3Stage0(DeepSeekV3StageI):
    def __init__(self, embed, layers, model_args):
        super().__init__(layers, model_args)
        self.tok_embeddings = embed

    def forward(self, tokens):
        # torch.Size([1024, 1024])
        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
        # torch.Size([1024, 1024, 2048])
        return super().forward(h)

    def init_weights(
        self, buffer_device: torch.device | None = None, seed: int | None = None
    ) -> None:
        _init_weights_tok_embeddings(self, seed)
        super().init_weights(buffer_device, seed)


class DeepSeekV3StageN(DeepSeekV3StageI):
    def __init__(self, layers, norm, output, model_args):
        super().__init__(layers, model_args)
        self.norm = norm
        self.output = output
        self.model_args = model_args

    def forward(self, h):
        h = super().forward(h)
        h = self.norm(h) if self.norm is not None else h
        output = self.output(h) if self.output is not None else h
        return output

    def init_weights(
        self, buffer_device: torch.device | None = None, seed: int | None = None
    ) -> None:
        super().init_weights(buffer_device, seed)
        _init_weights_norm_and_output(self)


######################
# Pipeline stuff end #
######################


def _init_weights_tok_embeddings(
    self: Union[DeepSeekV3Model, DeepSeekV3Stage0], seed: int | None = None
):
    if seed is not None:
        torch.manual_seed(seed)
    if self.tok_embeddings is not None:
        nn.init.normal_(self.tok_embeddings.weight)


def _init_weights_layers(
    self: Union[DeepSeekV3Model, DeepSeekV3StageI],
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


def _init_weights_norm_and_output(self: Union[DeepSeekV3Model, DeepSeekV3StageN]):
    if self.norm is not None:
        self.norm.reset_parameters()
    if self.output is not None:
        final_out_std = self.model_args.dim**-0.5
        cutoff_factor = 3
        nn.init.trunc_normal_(
            self.output.weight,
            mean=0.0,
            std=final_out_std,
            a=-cutoff_factor * final_out_std,
            b=cutoff_factor * final_out_std,
        )
