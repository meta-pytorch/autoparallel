# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch

# Importing triggers registration of torch_attn::_varlen_attn{,_backward}.
import torch.nn.attention.varlen as _varlen  # noqa: F401
from torch.utils.flop_counter import (
    register_flop_formula,
    sdpa_backward_flop_count,
    sdpa_flop_count,
)


def permutation(x: torch.Tensor, axis: int = 0, independent: bool = False):
    """Randomly permute elements of a tensor along an axis.

    Similar to jax.random.permutation, this function returns a tensor with
    elements shuffled along the specified axis.

    Args:
        x: Input tensor to permute.
        axis: The axis along which to permute. Defaults to 0.
        independent: If False (default), applies the same random permutation
            to all slices along the axis (like shuffling rows of a matrix
            together). If True, generates independent random permutations
            for each slice, meaning each position along other dimensions
            gets its own random ordering.

    Returns:
        A tensor with the same shape as x, with elements permuted along
        the specified axis.

    Examples:
        >>> x = torch.arange(12).reshape(3, 4)
        >>> # Shuffle rows (axis=0), same permutation for all columns
        >>> permutation(x, axis=0, independent=False)
        >>> # Shuffle rows independently for each column
        >>> permutation(x, axis=0, independent=True)
    """
    if independent is False:
        idxs = torch.randperm(x.shape[axis], device=x.device)
        return x.index_select(axis, idxs)

    # generate random permutation matrix which is independent per axis
    idxs = torch.rand_like(x, dtype=torch.float32).argsort(axis)
    return x.gather(axis, idxs)


# ---------------------------------------------------------------------------
# doc_packed_attn: document-packed variable-length attention with a shape that
# AutoParallel can shard.
#
# Q and KV may have different sequence capacities and document boundaries.
# Their [B, S, H, D] tensors and [B, MAX_DOCS+1] boundaries share a batch axis,
# so Shard(0) preserves paired documents across DP ranks.
#
# Internally we reshape q/k/v to THD layout, build a flat cu_seq_q with
# per-batch-element offsets of ``b*S``, and dispatch to
# ``torch_attn::_varlen_attn``. ``n_docs`` is uniform across batch elements;
# batch elements with fewer real documents pad their row with repeated ``S``
# so the kernel sees zero-length trailing docs and skips them.
#
# The custom op returns ``(out, lse, rng_state)`` (mirroring the upstream
# ``torch_attn::_varlen_attn`` shape) so ``setup_context`` can save the
# residuals for backward without re-running the forward kernel. End users who
# want just the output should call :func:`doc_packed_attn`, the thin Python
# helper below.
# ---------------------------------------------------------------------------


def _build_flat_cu_seq(cu_seq: torch.Tensor, n_docs: int, S: int) -> torch.Tensor:
    """Slice boundaries to ``n_docs+1`` entries per row, then offset and flatten.

    For ``B == 1`` returns a view (no copy, no sync).
    For ``B > 1`` returns ``[0, d0_1, ..., S, S+d1_1, ..., 2S, ...]`` of length
    ``B*n_docs + 1`` — sync-free arithmetic on a tiny tensor.
    """
    B = cu_seq.shape[0]
    sliced = cu_seq[:, : n_docs + 1]  # [B, n_docs+1]
    if B == 1:
        return sliced[0]
    # Offset each row by b*S so doc boundaries are absolute positions in the
    # flattened T=B*S dimension.
    offsets = torch.arange(B, dtype=cu_seq.dtype, device=cu_seq.device) * S
    shifted = sliced + offsets.unsqueeze(1)  # [B, n_docs+1]
    # Each row's leading entry equals b*S (== prior row's trailing entry), so
    # drop it from rows 1..B-1 to avoid duplicating boundary positions.
    return torch.cat([shifted[0], shifted[1:, 1:].flatten()])


def _validate_doc_packed_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seq_q: torch.Tensor,
    n_docs: int,
    cu_seq_k: torch.Tensor | None,
    enable_gqa: bool,
) -> torch.Tensor:
    if any(t.ndim != 4 for t in (query, key, value)):
        raise ValueError("query, key and value must have shape [B, S, H, D]")
    if key.shape != value.shape:
        raise ValueError("key and value must have the same shape")
    if query.shape[0] != key.shape[0] or query.shape[3] != key.shape[3]:
        raise ValueError(
            "query and key/value must have matching batch size and head dimension"
        )
    if cu_seq_k is None:
        if query.shape[1] != key.shape[1]:
            raise ValueError(
                "cu_seq_k is required when query and key/value sequence capacities differ"
            )
        cu_seq_k = cu_seq_q
    if n_docs < 1:
        raise ValueError("n_docs must be positive")
    for name, cu in (("cu_seq_q", cu_seq_q), ("cu_seq_k", cu_seq_k)):
        if cu.ndim != 2 or cu.shape[0] != query.shape[0] or cu.shape[1] < n_docs + 1:
            raise ValueError(
                f"{name} must have shape [B, MAX_DOCS+1] with MAX_DOCS >= n_docs"
            )
        if cu.dtype != torch.int32:
            raise ValueError(f"{name} must have dtype int32")
    if any(t.device != query.device for t in (key, value, cu_seq_q, cu_seq_k)):
        raise ValueError(
            "query, key, value and document boundaries must share a device"
        )
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError("query, key and value must have the same dtype")
    num_heads_q, num_heads_k = query.shape[2], key.shape[2]
    if num_heads_q < 1 or num_heads_k < 1:
        raise ValueError("query and key/value must have at least one head")
    if not enable_gqa and num_heads_q != num_heads_k:
        raise ValueError(
            f"Expect query and key/value to have the same number of heads "
            f"but got Hq={num_heads_q} and Hkv={num_heads_k}. "
            f"Try setting enable_gqa=True for GQA."
        )
    if enable_gqa and num_heads_q % num_heads_k != 0:
        raise ValueError(
            f"Expect number of query heads to be a multiple of kv heads for GQA "
            f"but got Hq={num_heads_q} and Hkv={num_heads_k}."
        )
    return cu_seq_k


@torch.library.custom_op("autoparallel::doc_packed_attn_op", mutates_args=())
def _doc_packed_attn_op(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seq_q: torch.Tensor,
    n_docs: int,
    scale: float | None = None,
    window_size: list[int] | None = None,
    enable_gqa: bool = False,
    cu_seq_k: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Custom op: returns ``(out, lse, rng_state)``.

    ``out`` has the same shape as ``query``. ``lse`` and ``rng_state`` come
    straight from the underlying flash-attention kernel and are passed back
    into ``torch_attn::_varlen_attn_backward`` during backward.
    """
    cu_seq_k = _validate_doc_packed_attn(
        query, key, value, cu_seq_q, n_docs, cu_seq_k, enable_gqa
    )
    B, S_q = query.shape[:2]
    S_k = key.shape[1]
    cu_q = _build_flat_cu_seq(cu_seq_q, n_docs, S_q)
    cu_k = cu_q if cu_seq_k is cu_seq_q else _build_flat_cu_seq(cu_seq_k, n_docs, S_k)

    ws = list(window_size) if window_size is not None else [-1, -1]
    is_causal = ws == [-1, 0]
    out_thd, lse, rng_state = torch.ops.torch_attn._varlen_attn(
        query.reshape(B * S_q, *query.shape[2:]),
        key.reshape(B * S_k, *key.shape[2:]),
        value.reshape(B * S_k, *value.shape[2:]),
        cu_q,
        cu_k,
        S_q,
        S_k,
        is_causal,
        scale,
        ws,
        enable_gqa,
        None,  # seqused_k — inference-only per upstream
        None,  # block_table — inference-only per upstream
        None,  # num_splits — perf-tuning knob, not exposed
    )
    return out_thd.view_as(query), lse, rng_state


@_doc_packed_attn_op.register_fake
def _doc_packed_attn_op_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seq_q: torch.Tensor,
    n_docs: int,
    scale: float | None = None,
    window_size: list[int] | None = None,
    enable_gqa: bool = False,
    cu_seq_k: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate_doc_packed_attn(query, key, value, cu_seq_q, n_docs, cu_seq_k, enable_gqa)
    B, S, Hq = query.shape[:3]
    out = torch.empty_like(query)
    # Matches torch_attn::_varlen_attn fake shapes (after THD flatten):
    #   lse: (num_heads, total_q) float32
    #   rng_state: (2,) uint64 placeholder
    lse = torch.empty((Hq, B * S), dtype=torch.float32, device=query.device)
    rng_state = torch.empty((2,), dtype=torch.uint64, device=query.device)
    return out, lse, rng_state


@register_flop_formula(torch.ops.autoparallel.doc_packed_attn_op)
def _doc_packed_attn_forward_flop(query, key, value, *args, **kwargs) -> int:
    """Count a per-batch upper bound because document lengths are data-dependent."""
    B, S, H_q, D_q = query
    B_k, S_k, H_kv, D_k = key
    B_v, S_v, H_v, D_v = value
    return sdpa_flop_count(
        (B, H_q, S, D_q),
        (B_k, H_kv, S_k, D_k),
        (B_v, H_v, S_v, D_v),
    )


def _doc_packed_attn_setup_context(ctx: Any, inputs: tuple, output: Any) -> None:
    (
        query,
        key,
        value,
        cu_seq_q,
        n_docs,
        scale,
        window_size,
        enable_gqa,
        cu_seq_k,
    ) = inputs
    out, lse, rng_state = output
    ctx.mark_non_differentiable(lse, rng_state)
    ctx.save_for_backward(
        query,
        key,
        value,
        cu_seq_q,
        out,
        lse,
        rng_state,
        cu_seq_k,
    )
    ctx.n_docs = n_docs
    ctx.scale = scale
    ctx.window_size = window_size


@torch.library.custom_op("autoparallel::doc_packed_attn_backward_op", mutates_args=())
def _doc_packed_attn_backward_op(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cu_seq_q: torch.Tensor,
    rng_state: torch.Tensor,
    n_docs: int,
    scale: float | None = None,
    window_size: list[int] | None = None,
    cu_seq_k: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward custom op for ``doc_packed_attn``.

    Takes the same ``[B, S, H, D]`` / ``[B, MAX_DOCS+1]`` shapes as the forward.
    The THD reshape and the flat ``cu_seq_q`` construction happen inside the
    op body and are invisible to AutoParallel — so the AP sharding strategy
    can reason about the natural ``[B, S, H, D]`` layout symmetrically with
    the forward, and the divisibility check uses B directly.
    """
    cu_seq_k = _validate_doc_packed_attn(
        query, key, value, cu_seq_q, n_docs, cu_seq_k, enable_gqa=True
    )
    B, S_q = query.shape[:2]
    S_k = key.shape[1]
    ws = list(window_size) if window_size is not None else [-1, -1]
    is_causal = ws == [-1, 0]
    cu_q = _build_flat_cu_seq(cu_seq_q, n_docs, S_q)
    cu_k = cu_q if cu_seq_k is cu_seq_q else _build_flat_cu_seq(cu_seq_k, n_docs, S_k)

    dq_thd, dk_thd, dv_thd = torch.ops.torch_attn._varlen_attn_backward(
        grad_out.reshape(B * S_q, *grad_out.shape[2:]),
        query.reshape(B * S_q, *query.shape[2:]),
        key.reshape(B * S_k, *key.shape[2:]),
        value.reshape(B * S_k, *value.shape[2:]),
        out.reshape(B * S_q, *out.shape[2:]),
        lse,
        cu_q,
        cu_k,
        S_q,
        S_k,
        is_causal,
        rng_state,
        scale,
        ws,
    )
    return (
        dq_thd.view_as(query),
        dk_thd.view_as(key),
        dv_thd.view_as(value),
    )


@_doc_packed_attn_backward_op.register_fake
def _doc_packed_attn_backward_op_fake(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cu_seq_q: torch.Tensor,
    rng_state: torch.Tensor,
    n_docs: int,
    scale: float | None = None,
    window_size: list[int] | None = None,
    cu_seq_k: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate_doc_packed_attn(
        query, key, value, cu_seq_q, n_docs, cu_seq_k, enable_gqa=True
    )
    return torch.empty_like(query), torch.empty_like(key), torch.empty_like(value)


@register_flop_formula(torch.ops.autoparallel.doc_packed_attn_backward_op)
def _doc_packed_attn_backward_flop(grad_out, query, key, value, *args, **kwargs) -> int:
    """Count a per-batch upper bound because document lengths are data-dependent."""
    B, S, H_q, D_q = query
    B_k, S_k, H_kv, D_k = key
    B_v, S_v, H_v, D_v = value
    B_g, S_g, H_g, D_g = grad_out
    return sdpa_backward_flop_count(
        (B_g, H_g, S_g, D_g),
        (B, H_q, S, D_q),
        (B_k, H_kv, S_k, D_k),
        (B_v, H_v, S_v, D_v),
    )


def _doc_packed_attn_backward(
    ctx: Any,
    grad_out: torch.Tensor,
    grad_lse: torch.Tensor,
    grad_rng: torch.Tensor,
) -> tuple[torch.Tensor | None, ...]:
    """Backward callback for doc_packed_attn.

    Routes to the ``doc_packed_attn_backward_op`` custom op so that the
    THD reshape stays hidden from AutoParallel.
    """
    query, key, value, cu_seq_q, out, lse, rng_state, cu_seq_k = ctx.saved_tensors
    dq, dk, dv = torch.ops.autoparallel.doc_packed_attn_backward_op(
        grad_out,
        query,
        key,
        value,
        out,
        lse,
        cu_seq_q,
        rng_state,
        ctx.n_docs,
        ctx.scale,
        ctx.window_size,
        cu_seq_k,
    )
    # Match the forward signature: query, key, value, cu_seq_q, n_docs,
    # scale, window_size, enable_gqa, cu_seq_k. Only q/k/v get gradients.
    return (dq, dk, dv, None, None, None, None, None, None)


torch.library.register_autograd(
    "autoparallel::doc_packed_attn_op",
    _doc_packed_attn_backward,
    setup_context=_doc_packed_attn_setup_context,
)


def doc_packed_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seq_q: torch.Tensor,
    n_docs: int,
    *,
    cu_seq_k: torch.Tensor | None = None,
    scale: float | None = None,
    window_size: tuple[int, int] = (-1, -1),
    enable_gqa: bool = False,
) -> torch.Tensor:
    """Document-packed variable-length attention.

    Mirrors :func:`torch.nn.attention.varlen.varlen_attn`'s API, but takes
    ``query``/``key``/``value`` in ``[B, S, H, D]`` layout and ``cu_seq_q`` as
    padded ``[B, MAX_DOCS+1]`` boundary tensors so the leading dim can be sharded
    as ``Shard(0)`` by AutoParallel. Document slot ``i`` in Q attends only to
    slot ``i`` in KV, which must refer to the same example. Q and KV may have
    different sequence capacities and document lengths.

    Args:
        query: ``[B, S_q, H_q, D]`` query tensor.
        key:   ``[B, S_k, H_kv, D]`` key tensor.
        value: ``[B, S_k, H_kv, D]`` value tensor.
        cu_seq_q: ``[B, MAX_DOCS+1]`` int32 padded cumulative sequence positions.
            Each row is nondecreasing, starts at zero and ends at ``S_q``. Real
            document boundaries occupy entries ``1..n_real_docs[b]``; entries
            ``n_real_docs[b]..n_docs`` should be padded with ``S_q`` so the
            trailing slots appear as zero-length documents (which the kernel
            skips). ``n_docs`` is uniform across batch elements.
        n_docs: Document count per batch element (SymInt under ``dynamic=True``).
            Batches with fewer real documents pad as described above.
        cu_seq_k: Optional KV boundaries, with the same ordered document slots
            as ``cu_seq_q``. Each row starts at zero and ends at ``S_k``;
            boundaries must be nondecreasing. Trailing unused slots repeat
            ``S_k``. Defaults to ``cu_seq_q``, requiring ``S_q == S_k``.
            Tensor padding must be assigned separate document slots so it
            cannot participate in a real document's attention.
        scale: Softmax scale. ``None`` uses ``1 / sqrt(D)``.
        window_size: ``(left, right)`` sliding-window sizes per token.
            ``(-1, -1)`` is full attention (default), ``(-1, 0)`` is causal,
            ``(W, 0)`` is causal sliding window of size ``W``.
            For unequal document lengths, windows use Flash Attention's
            bottom-right alignment: query position ``i`` aligns with key
            position ``i + length_k - length_q``.
        enable_gqa: Allow ``H_kv != H_q`` via grouped-query attention.

    Returns:
        Output tensor of the same shape as ``query``.
    """
    out, _lse, _rng = torch.ops.autoparallel.doc_packed_attn_op(
        query,
        key,
        value,
        cu_seq_q,
        n_docs,
        scale,
        list(window_size),
        enable_gqa,
        cu_seq_k,
    )
    return out
