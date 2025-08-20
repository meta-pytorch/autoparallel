from typing import cast

import torch
from autoparallel.propagation_rules import register_opschema_rule
from dynamic_shard import DynamicShard
from moe_utils import (
    batched_generate_permute_indices,
    batched_permute_and_pad,
    batched_unpermute_and_unpad,
    TOKEN_GROUP_ALIGN_SIZE_M,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._op_schema import OpSchema, OpStrategy
from torch.distributed.tensor.placement_types import Partial, Replicate, Shard
from torch.utils.flop_counter import register_flop_formula


@torch.library.custom_op("autoparallel::batched_grouped_mm", mutates_args=())
def batched_grouped_mm(
    mat1: torch.Tensor, mat2: torch.Tensor, offs: torch.Tensor
) -> torch.Tensor:
    assert offs.ndim == 2  # [ob, num_experts]
    assert mat1.ndim == 3  # [ob, ib, dim]
    assert mat2.ndim == 3, f"{mat2.shape}"  # [num_experts, dim, hidden_dim]
    ob1, num_experts1 = offs.shape
    ob2, _, dim = mat1.shape
    num_experts2, dim, _ = mat2.shape
    assert ob1 == ob2, f"{mat1.shape} vs {offs.shape}"
    assert num_experts1 == num_experts2, f"{mat2.shape} vs {offs.shape}"
    assert dim == dim, f"{mat1.shape} vs {mat2.shape}"
    res = []
    for a, off in zip(mat1, offs):
        res.append(torch._grouped_mm(a, mat2, off))
    return torch.stack(res, 0)


def setup_context(ctx, inputs, output):
    mat1, mat2, offs = inputs
    ctx.save_for_backward(mat1, mat2, offs)


def backward(ctx, grad):
    mat1, mat2, offs = ctx.saved_tensors
    grad1 = batched_grouped_mm(grad, mat2.transpose(-2, -1), offs)
    grad2 = batched_grouped_mm(mat1.transpose(-2, -1), grad, offs)
    return grad1, grad2, None


torch.library.register_autograd(
    "autoparallel::batched_grouped_mm", backward, setup_context=setup_context
)


@batched_grouped_mm.register_fake
def batched_grouped_mm_meta(
    mat1: torch.Tensor, mat2: torch.Tensor, offs: torch.Tensor
) -> torch.Tensor:
    assert offs.ndim == 2  # [ob, num_experts]
    assert mat1.ndim == 3  # [ob, ib, dim]
    assert mat2.ndim == 3, f"{mat2.shape}"  # [num_experts, dim, hidden_dim]
    ob1, num_experts1 = offs.shape
    ob2, _, dim = mat1.shape
    num_experts2, dim, _ = mat2.shape
    assert ob1 == ob2, f"{mat1.shape} vs {offs.shape}"
    assert num_experts1 == num_experts2, f"{mat2.shape} vs {offs.shape}"
    assert dim == dim, f"{mat1.shape} vs {mat2.shape}"

    out = torch.empty(
        mat1.shape[0],
        mat1.shape[1],
        mat2.shape[2],
        dtype=mat1.dtype,
        device=mat1.device,
    )
    return out


@register_flop_formula(torch.ops.autoparallel.batched_grouped_mm)
def batched_grouped_mm_flop_count(
    mat1_shape,
    mat2_shape,
    offsets_shape=None,
    bias_shape=None,
    out_shape=None,
    **kwargs,
) -> int:
    """
    Count floating-point operations for batched grouped matrix multiplication.

    This operation performs matrix multiplication between mat1 and mat2, where tokens
    are grouped by experts (common in MoE models). The offsets tensor defines how
    tokens are distributed across expert groups.

    Args:
        mat1_shape: Shape of first input matrix
        mat2_shape: Shape of second input matrix
        offsets_shape: Shape of offsets tensor defining token grouping

    Returns:
        Total number of floating-point operations
    """

    # Parse mat1 dimensions based on whether it's already batched or flat
    if len(mat1_shape) == 2:
        # Case: Flat tokens that need to be grouped by experts
        # mat1_shape: (total_tokens, input_dim)
        assert offsets_shape is not None, "Offsets required for flat token tensor"
        total_tokens, input_dim = mat1_shape
        num_expert_groups = offsets_shape[0]  # Number of expert groups
        # Assume roughly balanced distribution of tokens across experts
        avg_tokens_per_group = total_tokens // num_expert_groups

    elif len(mat1_shape) == 3:
        # Case: Pre-batched tokens already grouped by experts
        # mat1_shape: (batch_size, total_tokens, input_dim)
        assert offsets_shape is not None, "Offsets required to determine grouping"
        batch_size, total_tokens, input_dim = mat1_shape
        offset_batch_size, num_expert_groups = offsets_shape

        # Validate consistency
        assert (
            batch_size == offset_batch_size
        ), f"Batch size mismatch: {batch_size} vs {offset_batch_size}"

        # Average tokens per expert group (assuming balanced distribution)
        avg_tokens_per_group = total_tokens // num_expert_groups

    else:
        raise ValueError(f"Unsupported mat1 shape: {mat1_shape}")

    # Parse mat2 dimensions
    if len(mat2_shape) == 2:
        # Case: Flat weight matrix that will be grouped
        # mat2_shape: (input_dim, total_output_dim)
        input_dim_2, total_output_dim = mat2_shape
        assert offsets_shape is not None, "Offsets required for flat weight matrix"
        # Assume output dimension is distributed across expert groups
        avg_output_dim_per_group = total_output_dim // num_expert_groups

    elif len(mat2_shape) == 3:
        # Case: Pre-grouped expert weight matrices
        # mat2_shape: (num_experts, input_dim, output_dim_per_expert)
        num_experts, input_dim_2, output_dim_per_expert = mat2_shape

        # Validate dimensions match
        assert (
            num_experts == num_expert_groups
        ), f"Expert count mismatch: {num_experts} vs {num_expert_groups}"
        avg_output_dim_per_group = output_dim_per_expert

    else:
        raise ValueError(f"Unsupported mat2 shape: {mat2_shape}")

    # Validate input dimensions match
    assert (
        input_dim == input_dim_2
    ), f"Input dimension mismatch: {input_dim} vs {input_dim_2}"

    # Calculate FLOPs for matrix multiplication: C = A @ B
    # For each output element C[i,j], we compute: sum_k(A[i,k] * B[k,j])
    # This requires k multiplications and (k-1) additions ≈ 2k operations
    total_flops = (
        num_expert_groups  # Number of expert groups to process
        * avg_tokens_per_group  # Average tokens per expert group
        * avg_output_dim_per_group  # Output features per expert
        * 2
        * input_dim  # 2 ops per inner product (multiply + add)
    )

    return total_flops


@register_opschema_rule(torch.ops.autoparallel.batched_grouped_mm.default)
def _(mesh: DeviceMesh, op_schema: OpSchema):
    from torch.distributed.tensor._op_schema import PlacementList
    from torch.distributed.tensor._ops.utils import expand_to_full_mesh_op_strategy

    mat1_strategy = cast(OpStrategy, op_schema.args_schema[0])
    mat2_strategy = cast(OpStrategy, op_schema.args_schema[1])
    offs_strategy = cast(OpStrategy, op_schema.args_schema[2])

    assert len(mat1_strategy.shape) == 3
    assert len(mat2_strategy.shape) == 3
    assert len(offs_strategy.shape) == 2
    assert mat1_strategy.shape[0] == offs_strategy.shape[0]
    assert mat2_strategy.shape[0] == offs_strategy.shape[1]
    assert mat1_strategy.shape[2] == mat2_strategy.shape[1]

    single_mesh_dim_strategies = []

    # placement list stores placements of [output, mat1, mat2, offs]
    # first we always have replicate all for inputs and output
    all_replicate: PlacementList = [Replicate()] * 4
    single_mesh_dim_strategies.append(all_replicate)
    # mat1 sharded on outer batch, mat2 is replicated, offs is sharded on outer batch
    #  -> output is sharded on outer batch
    single_mesh_dim_strategies.append([Shard(0), Shard(0), Replicate(), Shard(0)])
    # mat1 is replicated, mat2 is sharded on column dim, offs is replicated
    #  -> output is sharded on column dim
    single_mesh_dim_strategies.append([Shard(2), Replicate(), Shard(2), Replicate()])
    # mat1 is sharded on column dim, mat2 is sharded on row dim, offs is replicated
    #  -> output is partial
    single_mesh_dim_strategies.append([Partial(), Shard(2), Shard(1), Replicate()])
    # mat1 is dynamically sharded on row dim, mat2 is sharded on experts dim,
    # offs is sharded on experts dim -> output is dynamically sharded on row dim
    single_mesh_dim_strategies.append(
        [DynamicShard(1), DynamicShard(1), Shard(0), Shard(1)]
    )

    return expand_to_full_mesh_op_strategy(
        mesh, op_schema, single_mesh_dim_strategies, input_index=1
    )


@torch.library.custom_op("autoparallel::batched_histc", mutates_args=())
def batched_histc(
    x: torch.Tensor, bins: int = 100, min_val: int = 0, max_val: int = 1
) -> torch.Tensor:
    assert x.ndim == 2
    out = []
    for t in x:
        out.append(torch.histc(t, bins, min_val, max_val))
    return torch.stack(out, 0)


@batched_histc.register_fake
def batched_hist_meta(
    x: torch.Tensor, bins: int = 100, min_val: int = 0, max_val: int = 1
) -> torch.Tensor:
    out = torch.empty((x.shape[0], bins), dtype=x.dtype, device=x.device)
    return out


@torch.library.custom_op("autoparallel::token_dispatch", mutates_args=())
def token_dispatch_op(
    x: torch.Tensor,
    top_scores: torch.Tensor,
    selected_experts_indices: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    num_experts: int,
    top_k: int,
    score_before_experts: bool = False,
    mesh: DeviceMesh | None = None,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Size]
]:  # (padded_permuted_routed_input, top_scores_sorted, token_indices_sorted, batch_permuted_indices, routed_input_shape_before_permute)
    """
    Token dispatch operator with custom backward.

    Forward: Dispatch tokens to experts based on their expert assignments.
    Backward: Use token combine logic to scatter gradients back to original positions.

    Args:
        x: Input tensor of shape (ob, ib * slen, dim)
        top_scores: Tensor of shape (ob, ib * slen, top_k)
        selected_experts_indices: Tensor of shape (ob, ib * slen, top_k)
        num_tokens_per_expert: Tensor of shape (ob, num_experts)
        num_experts: Number of experts
        top_k: Number of experts to route to
        score_before_experts: Whether to apply scores before expert processing

    Returns:
        padded_permuted_routed_input:
        (ob, ib * slen * top_k + (num_experts * TOKEN_GROUP_ALIGN_SIZE_M), dim) - padded and permuted routed input
        top_scores_sorted: (ob, ib * slen * top_k) - sorted scores
        token_indices_sorted: (ob, ib * slen * top_k) - sorted token indices
        batch_permuted_indices: (ob, ib * slen * top_k + (num_experts * TOKEN_GROUP_ALIGN_SIZE_M)) - permuted indices
        routed_input_shape_before_permute: list of shape before permute
    """
    assert x.ndim == 3
    ob, ib_slen, dim = x.shape

    # Flatten expert indices and scores
    expert_indices_flat = selected_experts_indices.reshape(
        ob, ib_slen * top_k
    )  # (ob, ib_slen * top_k)
    scores_flat = top_scores.reshape(ob, ib_slen * top_k)  # (ob, ib_slen * top_k)

    # Sort within each batch
    sort_indices = torch.argsort(
        expert_indices_flat, dim=1, stable=True
    )  # (ob, ib_slen * top_k)

    # Gather sorted scores
    batch_indices = torch.arange(ob, device=x.device).unsqueeze(1)  # (ob, 1)
    scores_sorted = scores_flat[batch_indices, sort_indices]  # (ob, ib_slen * top_k)

    # Convert to token indices
    token_indices = sort_indices // top_k  # (ob, ib_slen * top_k)

    # Expand for gather
    token_indices_expanded = token_indices.unsqueeze(-1).expand(
        -1, -1, dim
    )  # (ob, ib_slen * top_k, dim)

    # Gather tokens
    routed_input = torch.gather(
        x, dim=1, index=token_indices_expanded
    )  # (ob, ib_slen * top_k, dim)

    if score_before_experts:
        routed_input = (
            routed_input.to(torch.float32) * scores_sorted.unsqueeze(-1)
        ).to(
            x.dtype
        )  # (ob, ib_slen * top_k, dim)

    exps_per_rank = num_tokens_per_expert.shape[1]
    num_ep_ranks = num_experts // exps_per_rank

    # Pad and permute routed input
    permuted_indices, m_sizes, m_offsets = batched_generate_permute_indices(
        batched_tokens_per_expert_group=num_tokens_per_expert,
        experts_per_rank=exps_per_rank,
        num_ranks=num_ep_ranks,
        batched_max_len=[
            routed_input[b].shape[0] + exps_per_rank * TOKEN_GROUP_ALIGN_SIZE_M
            for b in range(ob)
        ],
        alignment=TOKEN_GROUP_ALIGN_SIZE_M,
        use_cpu=False,
    )
    (
        padded_permuted_routed_input,
        routed_input_shapes_before_permute,
    ) = batched_permute_and_pad(
        batched_routed_input=routed_input,
        batched_permute_indices=permuted_indices,
    )
    return (
        padded_permuted_routed_input,
        scores_sorted,
        token_indices,
        permuted_indices,
        routed_input_shapes_before_permute,
    )


@torch.library.custom_op("autoparallel::token_combine", mutates_args=())
def token_combine_op(
    base_output: torch.Tensor,
    padded_permuted_routed_output: torch.Tensor,
    top_scores_sorted: torch.Tensor,
    token_indices_sorted: torch.Tensor,
    batch_permuted_indices: torch.Tensor,
    routed_input_shapes_before_permute: list[torch.Size],
    num_tokens_per_expert: torch.Tensor,
    num_experts: int,
    top_k: int,
    score_before_experts: bool = False,
    mesh: DeviceMesh | None = None,
) -> torch.Tensor:
    """
    Token combine operator with custom backward.

    Forward: Combine expert outputs back to original token positions.
    Backward: Use token dispatch logic to gather gradients for expert processing.

    Args:
        base_output: Base output tensor (ob, ib * slen, dim) -
            e.g., from shared experts or zeros if no shared experts
        paddded_permuted_routed_output: Padded and permuted routed output
         (ob, ib * slender * top_k + (num_experts * TOKEN_GROUP_ALIGN_SIZE_M), dim)
        top_scores_sorted: Sorted scores (ob, ib * slen * top_k)
        token_indices_sorted: Sorted token indices (ob, ib * slen * top_k)
        batch_permuted_indices: Permute indices (ob, ib * slen * top_k + (num_experts * TOKEN_GROUP_ALIGN_SIZE_M))
        routed_input_shape_before_permute: routed input shapes before permute
        num_tokens_per_expert: Tensor of shape (ob, num_experts)
        num_experts: Number of experts
        top_k: Number of experts to route to
        score_before_experts: Whether scores were applied before experts

    Returns:
        combined_output: (ob, ib * slen, dim) - final combined output
    """
    ob, ib_slen, dim = base_output.shape

    # Unpad and permute routed output
    routed_output = batched_unpermute_and_unpad(
        batched_padded_permuted_routed_output=padded_permuted_routed_output,
        batched_permute_indices=batch_permuted_indices,
        batched_routed_input_shapes_before_permute=routed_input_shapes_before_permute,
    )
    # Apply scores if not applied before experts
    if not score_before_experts:
        routed_output = (
            routed_output.to(torch.float32) * top_scores_sorted.unsqueeze(-1)
        ).to(routed_output.dtype)

    # Expand token indices for scatter_add
    token_indices_expanded = token_indices_sorted.unsqueeze(-1).expand(-1, -1, dim)

    # Scatter and add expert outputs to base output
    combined_output = base_output.scatter_add(
        dim=1, index=token_indices_expanded, src=routed_output
    )

    return combined_output


def setup_token_dispatch_context(ctx, inputs, output):
    """Setup context for token_dispatch backward pass."""
    (
        x,
        top_scores,
        selected_experts_indices,
        num_tokens_per_expert,
        num_experts,
        top_k,
        score_before_experts,
    ) = inputs

    (
        padded_permuted_routed_input,
        top_scores_sorted,
        token_indices_sorted,
        batch_permuted_indices,
        routed_input_shape_before_permute,
    ) = output

    # Save tensors needed for backward
    # For exact score gradients, we need to save the unscaled routed_input
    if score_before_experts:
        # Save unscaled routed_input (before score multiplication)
        unscaled_routed_input = padded_permuted_routed_input / (
            top_scores_sorted.unsqueeze(-1) + 1e-8
        )
        ctx.save_for_backward(
            top_scores_sorted,
            token_indices_sorted,
            num_tokens_per_expert,
            unscaled_routed_input,
            batch_permuted_indices,
        )
        ctx.routed_input_shape_before_permute = routed_input_shape_before_permute
    else:
        ctx.save_for_backward(
            top_scores_sorted,
            token_indices_sorted,
            num_tokens_per_expert,
            batch_permuted_indices,
        )
        ctx.routed_input_shape_before_permute = routed_input_shape_before_permute

    ctx.num_experts = num_experts
    ctx.top_k = top_k
    ctx.score_before_experts = score_before_experts
    ctx.input_shape = x.shape


def token_dispatch_backward(
    ctx,
    grad_padded_permuted_routed_input,
    grad_top_scores_sorted,
    grad_token_indices_sorted,
    grad_batch_permuted_indices,
    grad_routed_input_shape_before_permute,
):
    """
    Backward pass for token_dispatch.

    Scatter gradients back to original token positions.
    """
    if ctx.score_before_experts:
        (
            top_scores_sorted,
            token_indices_sorted,
            num_tokens_per_expert,
            unscaled_routed_input,
            batch_permuted_indices,
        ) = ctx.saved_tensors
    else:
        (
            top_scores_sorted,
            token_indices_sorted,
            num_tokens_per_expert,
            batch_permuted_indices,
        ) = ctx.saved_tensors
        unscaled_routed_input = None

    ob, ib_slen, dim = ctx.input_shape

    if grad_padded_permuted_routed_input is None:
        return None, None, None, None, None, None, None

    # Step 1: Handle score gradients if scores were applied before experts
    grad_routed_input_for_scatter = grad_padded_permuted_routed_input
    grad_top_scores = None

    if ctx.score_before_experts:
        assert unscaled_routed_input is not None

        # Unpad and unpermute both gradients and unscaled input for score gradient computation
        grad_routed_input_unpermuted = batched_unpermute_and_unpad(
            batched_padded_permuted_routed_output=grad_padded_permuted_routed_input,
            batched_permute_indices=batch_permuted_indices,
            batched_routed_input_shapes_before_permute=ctx.routed_input_shape_before_permute,
        )

        unscaled_routed_input_unpermuted = batched_unpermute_and_unpad(
            batched_padded_permuted_routed_output=unscaled_routed_input,
            batched_permute_indices=batch_permuted_indices,
            batched_routed_input_shapes_before_permute=ctx.routed_input_shape_before_permute,
        )

        # Chain rule: ∂L/∂scores = grad_routed_input * unscaled_routed_input
        grad_top_scores_flat = (
            grad_routed_input_unpermuted * unscaled_routed_input_unpermuted
        ).sum(dim=-1)
        grad_top_scores = grad_top_scores_flat.reshape(ob, ib_slen, ctx.top_k)

        # Remove score scaling from gradients for scatter operation
        grad_routed_input_for_scatter = grad_padded_permuted_routed_input / (
            top_scores_sorted.unsqueeze(-1) + 1e-8
        )

    # Step 2: Unpad and unpermute gradients
    grad_routed_input = batched_unpermute_and_unpad(
        batched_padded_permuted_routed_output=grad_routed_input_for_scatter,
        batched_permute_indices=batch_permuted_indices,
        batched_routed_input_shapes_before_permute=ctx.routed_input_shape_before_permute,
    )

    # Step 3: Scatter gradients back to original token positions (inverse of gather)
    grad_x = torch.zeros(
        (ob, ib_slen, dim),
        dtype=grad_routed_input.dtype,
        device=grad_routed_input.device,
    )
    token_indices_expanded = token_indices_sorted.unsqueeze(-1).expand(-1, -1, dim)
    grad_x = grad_x.scatter_add(
        dim=1, index=token_indices_expanded, src=grad_routed_input
    )

    return grad_x, grad_top_scores, None, None, None, None, None


def setup_token_combine_context(ctx, inputs, output):
    """Setup context for token_combine backward pass."""
    (
        base_output,
        padded_permuted_routed_output,
        top_scores_sorted,
        token_indices_sorted,
        batch_permuted_indices,
        routed_input_shapes_before_permute,
        num_tokens_per_expert,
        num_experts,
        top_k,
        score_before_experts,
    ) = inputs

    # Save tensors needed for backward
    if not score_before_experts:
        # Save routed output before score multiplication for gradient computation
        unpadded_unscaled_routed_output = batched_unpermute_and_unpad(
            batched_padded_permuted_routed_output=padded_permuted_routed_output,
            batched_permute_indices=batch_permuted_indices,
            batched_routed_input_shapes_before_permute=routed_input_shapes_before_permute,
        )
        ctx.save_for_backward(
            top_scores_sorted,
            token_indices_sorted,
            num_tokens_per_expert,
            unpadded_unscaled_routed_output,
            batch_permuted_indices,
        )
    else:
        ctx.save_for_backward(
            top_scores_sorted,
            token_indices_sorted,
            num_tokens_per_expert,
            batch_permuted_indices,
        )

    ctx.num_experts = num_experts
    ctx.top_k = top_k
    ctx.score_before_experts = score_before_experts
    ctx.base_shape = base_output.shape
    ctx.routed_shape = padded_permuted_routed_output.shape
    ctx.routed_input_shapes_before_permute = routed_input_shapes_before_permute


def token_combine_backward(ctx, grad_combined_output):
    """
    Backward pass for token_combine.

    token_combine forward: unpad_unpermute → scatter_add
    token_combine backward: gather → pad_permute
    """
    if not ctx.score_before_experts:
        (
            top_scores_sorted,
            token_indices_sorted,
            num_tokens_per_expert,
            unscaled_routed_output,
            batch_permuted_indices,
        ) = ctx.saved_tensors
    else:
        (
            top_scores_sorted,
            token_indices_sorted,
            num_tokens_per_expert,
            batch_permuted_indices,
        ) = ctx.saved_tensors
        unscaled_routed_output = None

    ob, ib_slen, dim = ctx.base_shape

    if grad_combined_output is None:
        return None, None, None, None, None, None, None, None, None, None

    # Step 1: Base output gets the full gradient (residual connection)
    grad_base_output = grad_combined_output.clone()

    # Step 2: Gather gradients from original positions (inverse of scatter_add)
    token_indices_expanded = token_indices_sorted.unsqueeze(-1).expand(-1, -1, dim)
    grad_routed_output = torch.gather(
        grad_combined_output, dim=1, index=token_indices_expanded
    )

    # Step 3: Handle score gradients if scores were applied in forward pass
    grad_top_scores_sorted = None
    if not ctx.score_before_experts and unscaled_routed_output is not None:
        # Chain rule: In forward we had: final_output = unscaled_routed_output * scores
        # So: ∂L/∂scores = grad_final_output * unscaled_routed_output
        grad_top_scores_sorted = (grad_routed_output * unscaled_routed_output).sum(
            dim=-1
        )

    # Step 4: Apply permutation and padding (inverse of unpad_unpermute)
    (
        grad_padded_permuted_routed_output,
        _,  # Don't need the shapes
    ) = batched_permute_and_pad(
        batched_routed_input=grad_routed_output,
        batched_permute_indices=batch_permuted_indices,
    )

    return (
        grad_base_output,  # grad_base_output
        grad_padded_permuted_routed_output,  # grad_padded_permuted_routed_output
        grad_top_scores_sorted,  # grad_top_scores_sorted
        None,  # grad_token_indices_sorted
        None,  # grad_batch_permuted_indices
        None,  # grad_routed_input_shapes_before_permute
        None,  # grad_num_tokens_per_expert
        None,  # grad_num_experts
        None,  # grad_top_k
        None,  # grad_score_before_experts
    )


# Register backward functions
torch.library.register_autograd(
    "autoparallel::token_dispatch",
    token_dispatch_backward,
    setup_context=setup_token_dispatch_context,
)

torch.library.register_autograd(
    "autoparallel::token_combine",
    token_combine_backward,
    setup_context=setup_token_combine_context,
)


# Register fake implementations for meta tensors
@token_dispatch_op.register_fake
def token_dispatch_meta(
    x: torch.Tensor,
    top_scores: torch.Tensor,
    selected_experts_indices: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    num_experts: int,
    top_k: int,
    score_before_experts: bool = False,
    mesh: DeviceMesh | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Size]]:
    ob, ib_slen, dim = x.shape

    # Get maximum possible length based on max tokens per expert across all batches
    # Use max to handle different padding amounts per batch
    max_len = ib_slen * top_k + (num_experts * TOKEN_GROUP_ALIGN_SIZE_M)

    # Padded and permuted routed input with maximum possible size
    padded_permuted_routed_input = torch.empty(
        (ob, max_len, dim),
        dtype=x.dtype,
        device=x.device,
    )
    scores_sorted = torch.empty(
        (ob, ib_slen * top_k), dtype=top_scores.dtype, device=top_scores.device
    )
    token_indices = torch.empty(
        (ob, ib_slen * top_k),
        dtype=selected_experts_indices.dtype,
        device=selected_experts_indices.device,
    )
    batch_permuted_indices = torch.empty(
        (ob, max_len),
        dtype=torch.long,
        device=x.device,
    )
    # Shapes before permute: each batch has (ib_slen * top_k + 1, dim) where +1 is the zero row
    routed_input_shape_before_permute = [
        torch.Size([ib_slen * top_k + 1, dim]) for _ in range(ob)
    ]

    return (
        padded_permuted_routed_input,
        scores_sorted,
        token_indices,
        batch_permuted_indices,
        routed_input_shape_before_permute,
    )


@token_combine_op.register_fake
def token_combine_meta(
    base_output: torch.Tensor,
    paddded_permuted_routed_output: torch.Tensor,
    top_scores_sorted: torch.Tensor,
    token_indices_sorted: torch.Tensor,
    batch_permuted_indices: torch.Tensor,
    routed_input_shape_before_permute: list[torch.Size],
    num_tokens_per_expert: torch.Tensor,
    num_experts: int,
    top_k: int,
    score_before_experts: bool = False,
    mesh: DeviceMesh | None = None,
) -> torch.Tensor:
    return torch.empty_like(base_output)
