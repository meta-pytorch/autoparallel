# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore

from autoparallel.api import AutoParallel


# Reference: https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
@dataclass
class DeepSeekV3ModelArgs:
    """
    Data class for defining model arguments and hyperparameters.

    Attributes:
        max_batch_size (int): Maximum batch size.
        max_seq_len (int): Maximum sequence length.
        dtype (Literal["bf16", "fp8"]): Data type for computations.
        vocab_size (int): Vocabulary size.
        dim (int): Model dimension.
        inter_dim (int): Intermediate dimension for MLP layers.
        moe_inter_dim (int): Intermediate dimension for MoE layers.
        n_layers (int): Number of transformer layers.
        n_dense_layers (int): Number of dense layers in the model.
        n_heads (int): Number of attention heads.
        n_routed_experts (int): Number of routed experts for MoE layers.
        n_shared_experts (int): Number of shared experts for MoE layers.
        n_activated_experts (int): Number of activated experts in MoE layers.
        n_expert_groups (int): Number of expert groups.
        n_limited_groups (int): Number of limited groups for MoE routing.
        score_func (Literal["softmax", "sigmoid"]): Scoring function for MoE routing.
        route_scale (float): Scaling factor for routing scores.
        use_grouped_mm (bool): Whether to use grouped matrix multiplication for MoE layers.
        load_balance_coeff (float | None): Auxiliary-Loss-Free Load balancing coefficient for MoE layers.
        q_lora_rank (int): LoRA rank for query projections.
        kv_lora_rank (int): LoRA rank for key-value projections.
        qk_nope_head_dim (int): Dimension for query-key projections without positional embeddings.
        qk_rope_head_dim (int): Dimension for query-key projections with rotary embeddings.
        v_head_dim (int): Dimension for value projections.
        original_seq_len (int): Original sequence length.
        rope_theta (float): Base for rotary positional encoding.
        rope_factor (float): Scaling factor for extended sequence lengths.
        beta_fast (int): Fast beta correction factor.
        beta_slow (int): Slow beta correction factor.
    """

    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4
    dtype: Literal["bf16", "fp8"] = "bf16"
    vocab_size: int = 102400
    dim: int = 2048
    inter_dim: int = 10944
    moe_inter_dim: int = 1408
    n_layers: int = 27
    n_dense_layers: int = 1
    n_heads: int = 16
    norm_eps: float = 1e-5  # eps used for RMSNorm
    # MoE
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.0
    use_grouped_mm: bool = True
    load_balance_coeff: float = 1e-3
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

    def update_from_config(self, job_config, tokenizer) -> None:
        """
        Update the model_config config from the given job config.
        """
        self.vocab_size = tokenizer.vocab_size
        self.max_seq_len = job_config.training.seq_len

    def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
        """
        Adopted from llama4 implementation.
        """
        nparams_embedding = 0
        nparams_moe_router = 0
        nparams_shared_expert = 0
        nparams_experts = 0
        nparams_dense = 0

        for name, p in model.named_parameters():
            if "embedding" in name:
                nparams_embedding += p.numel()
                nparams_dense += p.numel()
            elif "moe.shared_expert" in name:
                nparams_shared_expert += p.numel()
            elif "moe.router" in name:
                nparams_moe_router += p.numel()
            elif "moe.experts" in name:
                nparams_experts += p.numel()
            else:
                nparams_dense += p.numel()

        nparams_sparse = nparams_moe_router + nparams_shared_expert + nparams_experts
        nparams = nparams_dense + nparams_sparse
        nparams_sparse_active = (
            nparams_moe_router
            + nparams_shared_expert
            + nparams_experts * self.n_activated_experts // self.n_routed_experts
        )

        # logger.info(
        #    f"Total parameter count: dense {nparams_dense:,}, "
        #    f"sparse {nparams_sparse:,}, active {nparams_dense + nparams_sparse_active:,}"
        # )

        l, h, q, t = (
            self.n_layers,
            self.n_heads,
            self.dim // self.n_heads,
            seq_len,
        )
        # Reasoning behind the factor of 12 for the self-attention part of the formula:
        # 1. each self-attention has 2 matmul in the forward and 4 in the backward (6)
        # 2. the flash attention does 1 more matmul recomputation in the backward
        #    but recomputation should not be counted in calculating MFU           (+0)
        # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
        # 4. we follow the convention and do not account for sparsity in causal attention
        num_flops_per_token = (
            6 * (nparams_dense - nparams_embedding + nparams_sparse_active)
            + 12 * l * h * q * t
        )

        return nparams, num_flops_per_token


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


@torch.library.custom_op("autoparallel::fill_indices_wrapper", mutates_args=())
def fill_indices_wrapper(
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


@fill_indices_wrapper.register_fake
def _(
    tokens_per_expert_group: torch.Tensor,
    start_index_values: torch.Tensor,
    write_offsets: torch.Tensor,
    experts_per_rank: int,
    num_ranks: int,
    max_len: int,
    block_size: int = 128,
    max_blocks: int = 1024,  # cap on total number of blocks to launch
):
    # preallocate output
    permuted_indices = torch.empty(
        (max_len,), dtype=torch.int32, device=tokens_per_expert_group.device
    )
    return permuted_indices


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
        # For each remote rank
        for r in range(num_ranks):
            i = r * experts_per_rank + e
            start_index = start_index_values[i].item()
            length = tokens_per_expert_group[i].item()
            # Fill in the indices
            if length > 0:
                end_idx = min(write_start + length, max_len)
                permuted_indices[write_start:end_idx] = torch.arange(
                    start_index,
                    start_index + (end_idx - write_start),
                    dtype=torch.int32,
                    # device=device,
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
        permuted_indices = fill_indices_wrapper(
            tokens_per_expert_group,
            start_index_values,
            write_offsets,
            experts_per_rank,
            num_ranks,
            max_len,
        )

    return permuted_indices, m_sizes, m_offsets.to(torch.int32)


def expert_parallel(func):
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
        num_tokens_per_expert: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if isinstance(w1, DTensor):
            w1 = w1.to_local()
            w2 = w2.to_local()
            w3 = w3.to_local()

        if num_tokens_per_expert is not None:
            experts_per_ep_rank = w1.shape[0]
            num_ep_ranks = num_tokens_per_expert.shape[0] // experts_per_ep_rank

            ALIGN_SIZE_M = 16
            with torch.no_grad():
                (
                    permuted_indices,
                    num_tokens_per_expert,
                    _,  # offsets,
                ) = generate_permute_indices(
                    num_tokens_per_expert,
                    experts_per_ep_rank,
                    num_ep_ranks,
                    x.shape[0] + experts_per_ep_rank * ALIGN_SIZE_M,
                    ALIGN_SIZE_M,
                )

            x = torch.vstack((x, x.new_zeros((x.shape[-1]))))
            input_shape = x.shape
            x = x[permuted_indices, :]

        out = func(w1, w2, w3, x, num_tokens_per_expert)

        if num_tokens_per_expert is not None:
            out_unpermuted = out.new_empty(input_shape)
            out_unpermuted[permuted_indices, :] = out
            out = out_unpermuted[:-1]

        return out

    return wrapper


class FeedForward(nn.Module):
    """
    FeedForward module

    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension of the feedforward layer.
        multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
        ffn_dim_multiplier (float | None): Custom multiplier for hidden dimension. Defaults to None.

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
        self.w1 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.w3 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.use_grouped_mm = use_grouped_mm

    def forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_grouped_mm:
            return GroupedExperts._run_experts_grouped_mm(
                self.w1, self.w2, self.w3, x, num_tokens_per_expert
            )
        else:
            return GroupedExperts._run_experts_for_loop(
                self.w1, self.w2, self.w3, x, num_tokens_per_expert
            )

    # TODO: keeping this for-loop implementation for comparison
    #       and readability, may remove later
    @expert_parallel
    @staticmethod
    def _run_experts_for_loop(
        w1: torch.Tensor,
        w2: torch.Tensor,
        w3: torch.Tensor,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if num_tokens_per_expert is not None:
            # NOTE: this would incur a synchronization between device and host
            num_tokens_per_expert = num_tokens_per_expert.tolist()

            # side-effect code due to the usage of generate_permute_indices
            num_padding = x.shape[0] - sum(num_tokens_per_expert)

            # a tuple of tensors indexed by experts
            # each with shape (tokens_per_expert(varying), dim)
            x = torch.split(
                x[: sum(num_tokens_per_expert)],
                split_size_or_sections=num_tokens_per_expert,
                dim=0,
            )
            out_experts_splits = []
            for expert_idx, x_expert in enumerate(x):
                h = F.silu(torch.matmul(x_expert, w1[expert_idx]))
                h = h * torch.matmul(x_expert, w3[expert_idx])
                h = torch.matmul(h, w2[expert_idx])
                # h shape (tokens_per_expert(varying), dim)
                out_experts_splits.append(h)
            out = torch.cat(out_experts_splits, dim=0)

            # side-effect code due to the usage of generate_permute_indices
            out = torch.vstack((out, out.new_zeros((num_padding, out.shape[-1]))))
        else:
            # x shape (num_experts, tokens_per_expert, dim)
            h = F.silu(torch.bmm(x, w1))
            h = h * torch.bmm(x, w3)
            # out shape (num_experts, tokens_per_expert, dim)
            out = torch.bmm(h, w2)

        return out

    @expert_parallel
    @staticmethod
    def _run_experts_grouped_mm(
        w1: torch.Tensor,
        w2: torch.Tensor,
        w3: torch.Tensor,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if num_tokens_per_expert is not None:
            offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
            # grouped mm between a 2D tensor and a 3D tensor
            assert x.dim() == 2
        else:
            offsets = None
            # fall back to regular bmm between 3D tensors
            assert x.dim() == 3

        assert (
            x.dtype == w1.dtype == w2.dtype == w3.dtype == torch.bfloat16
        ), "torch._grouped_mm only supports bf16 dtypes"

        h = F.silu(torch._grouped_mm(x, w1, offs=offsets))
        h = h * torch._grouped_mm(x, w3, offs=offsets)
        out = torch._grouped_mm(h, w2, offs=offsets)

        return out

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.w3, mean=0.0, std=init_std)


class TokenChoiceTopKRouter(nn.Module):
    """This class implements token-choice routing. In token-choice top-K routing, each token is
        routed to top K experts based on the router scores.

    Args:
        gate (nn.Module): Gate module to calculate the scores, typically nn.Linear(dim, num_experts).
        num_experts (int): Number of experts in each moe layer.
        top_k (int): Number of experts each token will be routed to in token-choice routing.
        use_sigmoid (bool): Whether to use sigmoid or softmax for router scores. Default is False.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int,
        use_sigmoid: bool = False,
        route_sclaing_factor: float = 1.0,
    ):
        super().__init__()

        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_sigmoid = use_sigmoid
        self.route_sclaing_factor = route_sclaing_factor
        self.gate = nn.Linear(self.dim, self.num_experts, bias=False)

    def forward(
        self, x: torch.Tensor, expert_bias: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        TODO: We haven't implement the group-based routing (node limit routing),
        and currently EP is not supporting node limit routing yet.

        Args:
            x (torch.Tensor): Input tensor with shape ``(bs*slen, dim)``.

        Returns:
            routed_input (torch.Tensor):
                Tokens grouped together by experts indices with shape ``(bs*slen*top_k,)``.
            token_indices (torch.Tensor):
                Token indices for routed_input with shape ``(bs*slen*top_k,)``.
            num_tokens_per_expert (torch.Tensor):
                Number of tokens assigned to each expert with shape ``(num_experts,)``.
        """
        # scores shape (bs*slen, num_experts)
        scores = self.gate(x)

        # By default, sigmoid or softmax is performed in float32 to avoid loss explosion
        if self.use_sigmoid:
            scores = torch.sigmoid(scores.to(torch.float32))
        else:
            scores = F.softmax(scores.to(torch.float32), dim=1)

        # top scores shape (bs*slen, top_k)
        # NOTE: The expert_bias is only used for routing. The gating value
        #       top_scores is still derived from the original scores.
        if expert_bias is not None:
            _, selected_experts_indices = torch.topk(
                scores + expert_bias, k=self.top_k, dim=1
            )
            top_scores = scores.gather(dim=1, index=selected_experts_indices)
        else:
            top_scores, selected_experts_indices = torch.topk(
                scores, k=self.top_k, dim=1
            )

        # group tokens together by expert indices from 0 to num_experts and pass that to experts forward
        num_tokens_per_expert = torch.histc(
            selected_experts_indices.view(-1),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )

        # Reorder the token indices to match the order of the experts
        # token_indices_experts_sorted shape (bs*slen*top_k,)
        token_indices_experts_sorted = torch.argsort(
            selected_experts_indices.view(-1), stable=True
        )

        # reorder the scores to match the order of the token indices
        top_scores = top_scores.view(-1)[token_indices_experts_sorted]
        token_indices_experts_sorted = token_indices_experts_sorted // self.top_k

        top_scores = (
            top_scores * self.route_sclaing_factor
        )  # must multiply the scaling factor
        return top_scores, token_indices_experts_sorted, num_tokens_per_expert

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.gate.weight, mean=0.0, std=init_std)


class MoE(nn.Module):
    def __init__(self, model_args: DeepSeekV3ModelArgs):

        super().__init__()
        dim = model_args.dim

        num_experts = model_args.n_routed_experts
        hidden_dim = model_args.moe_inter_dim
        top_k = model_args.n_activated_experts
        route_scaling_factor = model_args.route_scale

        self.experts = GroupedExperts(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            use_grouped_mm=model_args.use_grouped_mm,
        )
        self.router = TokenChoiceTopKRouter(
            dim=dim,
            num_experts=num_experts,
            top_k=top_k,
            use_sigmoid=model_args.score_func == "sigmoid",
            route_sclaing_factor=route_scaling_factor,
        )
        self.shared_expert = (
            # Reference: https://huggingface.co/deepseek-ai/DeepSeek-V3-Base/blob/main/modeling_deepseek.py#L517
            GroupedExperts(
                dim=dim,
                hidden_dim=hidden_dim * model_args.n_shared_experts,
                num_experts=1,  # Here needs to be 1 to make it equivalent to the MLP
                use_grouped_mm=model_args.use_grouped_mm,
            )
            if model_args.n_shared_experts > 0
            else None
        )

        # auxiliary-loss-free load balancing
        self.load_balance_coeff = model_args.load_balance_coeff
        if self.load_balance_coeff is not None:
            assert self.load_balance_coeff > 0.0
            self.register_buffer(
                "expert_bias",
                torch.zeros(num_experts, dtype=torch.float32),
                persistent=True,
            )
            self.register_buffer(
                "tokens_per_expert",
                torch.zeros(num_experts, dtype=torch.float32),
                persistent=True,
            )
        else:
            self.expert_bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs, slen, dim)``.

        Returns:
            out (torch.Tensor): Output tensor with shape ``(bs, slen, dim)``.
        """
        bs, slen, dim = x.shape

        # top_scores and selected_indices shape (bs*slen*top_k,)
        # num_tokens_per_expert shape (num_experts,)
        (
            top_scores,
            token_indices,
            num_tokens_per_expert,
        ) = self.router(x.reshape(bs * slen, dim), self.expert_bias)

        # tokens_per_expert will be used to update the expert bias for load balancing.
        # Prevent extra local tokens accumulation on evaluation or activation recomputation.
        if self.load_balance_coeff is not None and torch.is_grad_enabled():
            with torch.no_grad():
                self.tokens_per_expert.add_(num_tokens_per_expert)
        # shape (bs*slen*top_k, dim)
        token_indices = token_indices.reshape(-1, 1).expand(-1, dim)

        # shape (bs*slen*top_k, dim)
        routed_input = torch.gather(
            x.view(-1, dim),
            dim=0,
            index=token_indices,
        )

        # shape (bs*slen*top_k, dim)
        routed_output = self.experts(routed_input, num_tokens_per_expert)

        routed_output = (routed_output.to(torch.float32) * top_scores.unsqueeze(-1)).to(
            x.dtype
        )

        # shared expert
        if self.shared_expert is not None:
            out = self.shared_expert(x.reshape(1, bs * slen, dim)).reshape(
                bs * slen, dim
            )
        else:
            out = torch.zeros_like(x.reshape(bs * slen, dim))

        # Accumulate multiple expert results becase each token can be routed to multiple experts
        out = out.scatter_add(dim=0, index=token_indices, src=routed_output)
        out = out.reshape(bs, slen, dim)
        return out

    def init_weights(
        self,
        init_std: float,
        buffer_device: torch.device,
    ):
        self.experts.init_weights(init_std)
        self.router.init_weights(init_std)
        if self.shared_expert is not None:
            self.shared_expert.init_weights(init_std)

        if self.load_balance_coeff is not None:
            with torch.device(buffer_device):
                self.expert_bias = torch.zeros(
                    self.experts.num_experts, dtype=torch.float32
                )
                self.tokens_per_expert = torch.zeros(
                    self.experts.num_experts, dtype=torch.float32
                )


world_size = 256

fake_store = FakeStore()
torch.distributed.init_process_group(
    "fake", store=fake_store, rank=0, world_size=world_size
)
# mesh = torch.distributed.device_mesh.init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))
mesh = torch.distributed.device_mesh.init_device_mesh(
    "cuda",
    (world_size // 8, 8),
    mesh_dim_names=(
        "dp",
        "tp",
    ),
)

bs = 8 * mesh.shape[0]
seqlen = 1024
dim = 4096


def input_fn():
    return torch.randn(
        bs, seqlen, dim, dtype=torch.bfloat16, device="cuda", requires_grad=True
    )


args = DeepSeekV3ModelArgs(dim=dim, n_layers=1)

# parallelize the model
with torch.device("meta"):
    model = MoE(args).bfloat16()

autop = AutoParallel(model, input_fn, mesh)
autop.add_parameter_memory_constraint(low=None, high=None)

x_sharding = (Shard(0), Replicate())

autop.add_input_constraints([x_sharding])
autop.add_output_constraints([x_sharding])

sharding_placement = autop.optimize_placement()
parallel_mod = autop.apply_placement(sharding_placement)

# run weight init on our sharded DTensor params
parallel_mod.to_empty(device="cuda")
parallel_mod.init_weights(init_std=0.02, buffer_device="cuda")  # maybe not correct value

# # now let's run it
x = (
    torch.randn(
        # 0,
        # args.vocab_size,
        (bs // mesh.shape[0], seqlen, dim),
        device=torch.device("cuda"),
        dtype=torch.bfloat16
    ),
)
out = parallel_mod(*x)
out.backward(torch.randn_like(out))
print("All good!")
