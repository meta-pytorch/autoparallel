# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch

# Importing triggers registration of torch_attn::_varlen_attn{,_backward}.
import torch.nn.attention.varlen as _varlen  # noqa: F401


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
# The user-facing op takes q/k/v in [B, S, H, D] layout and a cu_seq_q tensor
# in [B, MAX_DOCS+1] layout. Sharding ``Shard(0)`` on the leading dim of all
# four inputs gives clean DP across the batch.
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


def _build_flat_cu_seq_q(cu_seq_q: torch.Tensor, n_docs: int, S: int) -> torch.Tensor:
    """Slice ``cu_seq_q`` to ``n_docs+1`` entries per row, then offset and flatten.

    For ``B == 1`` returns a view (no copy, no sync).
    For ``B > 1`` returns ``[0, d0_1, ..., S, S+d1_1, ..., 2S, ...]`` of length
    ``B*n_docs + 1`` — sync-free arithmetic on a tiny tensor.
    """
    B = cu_seq_q.shape[0]
    sliced = cu_seq_q[:, : n_docs + 1]  # [B, n_docs+1]
    if B == 1:
        return sliced[0]
    # Offset each row by b*S so doc boundaries are absolute positions in the
    # flattened T=B*S dimension.
    offsets = torch.arange(B, dtype=cu_seq_q.dtype, device=cu_seq_q.device) * S
    shifted = sliced + offsets.unsqueeze(1)  # [B, n_docs+1]
    # Each row's leading entry equals b*S (== prior row's trailing entry), so
    # drop it from rows 1..B-1 to avoid duplicating boundary positions.
    return torch.cat([shifted[0], shifted[1:, 1:].flatten()])


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Custom op: returns ``(out, lse, rng_state)``.

    ``out`` has the same shape as ``query``. ``lse`` and ``rng_state`` come
    straight from the underlying flash-attention kernel and are passed back
    into ``torch_attn::_varlen_attn_backward`` during backward.
    """
    B, S = query.shape[:2]
    cu = _build_flat_cu_seq_q(cu_seq_q, n_docs, S)

    ws = list(window_size) if window_size is not None else [-1, -1]
    is_causal = ws == [-1, 0]
    out_thd, lse, rng_state = torch.ops.torch_attn._varlen_attn(
        query.reshape(B * S, *query.shape[2:]),
        key.reshape(B * S, *key.shape[2:]),
        value.reshape(B * S, *value.shape[2:]),
        cu,
        cu,
        S,
        S,
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, S, Hq = query.shape[:3]
    out = torch.empty_like(query)
    # Matches torch_attn::_varlen_attn fake shapes (after THD flatten):
    #   lse: (num_heads, total_q) float32
    #   rng_state: (2,) int64 placeholder
    lse = torch.empty((Hq, B * S), dtype=torch.float32, device=query.device)
    rng_state = torch.empty((2,), dtype=torch.int64, device=query.device)
    return out, lse, rng_state


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
    ) = inputs
    out, lse, rng_state = output
    ctx.save_for_backward(query, key, value, cu_seq_q, out, lse, rng_state)
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward custom op for ``doc_packed_attn``.

    Takes the same ``[B, S, H, D]`` / ``[B, MAX_DOCS+1]`` shapes as the forward.
    The THD reshape and the flat ``cu_seq_q`` construction happen inside the
    op body and are invisible to AutoParallel — so the AP sharding strategy
    can reason about the natural ``[B, S, H, D]`` layout symmetrically with
    the forward, and the divisibility check uses B directly.
    """
    B, S = query.shape[:2]
    ws = list(window_size) if window_size is not None else [-1, -1]
    is_causal = ws == [-1, 0]
    cu = _build_flat_cu_seq_q(cu_seq_q, n_docs, S)

    dq_thd, dk_thd, dv_thd = torch.ops.torch_attn._varlen_attn_backward(
        grad_out.reshape(B * S, *grad_out.shape[2:]),
        query.reshape(B * S, *query.shape[2:]),
        key.reshape(B * S, *key.shape[2:]),
        value.reshape(B * S, *value.shape[2:]),
        out.reshape(B * S, *out.shape[2:]),
        lse,
        cu,
        cu,
        S,
        S,
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.empty_like(query), torch.empty_like(key), torch.empty_like(value)


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
    query, key, value, cu_seq_q, out, lse, rng_state = ctx.saved_tensors
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
    )
    # Match the forward signature: query, key, value, cu_seq_q, n_docs,
    # scale, window_size, enable_gqa. Only q/k/v get gradients.
    return (dq, dk, dv, None, None, None, None, None)


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
    scale: float | None = None,
    window_size: tuple[int, int] = (-1, -1),
    enable_gqa: bool = False,
) -> torch.Tensor:
    """Document-packed variable-length attention.

    Mirrors :func:`torch.nn.attention.varlen.varlen_attn`'s API, but takes
    ``query``/``key``/``value`` in ``[B, S, H, D]`` layout and ``cu_seq_q`` as
    a padded ``[B, MAX_DOCS+1]`` tensor so the leading dim can be sharded as
    ``Shard(0)`` by AutoParallel. ``cu_seq_k`` is assumed to equal ``cu_seq_q``
    (self-attention) and ``max_q == max_k == S``.

    Args:
        query: ``[B, S, H_q, D]`` query tensor.
        key:   ``[B, S, H_kv, D]`` key tensor.
        value: ``[B, S, H_kv, D]`` value tensor.
        cu_seq_q: ``[B, MAX_DOCS+1]`` int32 padded cumulative sequence positions.
            ``cu_seq_q[b, 0] == 0`` and ``cu_seq_q[b, n_docs] == S``. Real
            document boundaries occupy entries ``1..n_real_docs[b]``; entries
            ``n_real_docs[b]..n_docs`` should be padded with ``S`` so the
            trailing slots appear as zero-length documents (which the kernel
            skips). ``n_docs`` is uniform across batch elements.
        n_docs: Document count per batch element (SymInt under ``dynamic=True``).
            Batches with fewer real documents pad as described above.
        scale: Softmax scale. ``None`` uses ``1 / sqrt(D)``.
        window_size: ``(left, right)`` sliding-window sizes per token.
            ``(-1, -1)`` is full attention (default), ``(-1, 0)`` is causal,
            ``(W, 0)`` is causal sliding window of size ``W``.
        enable_gqa: Allow ``H_kv != H_q`` via grouped-query attention.

    Returns:
        Output tensor of the same shape as ``query``.
    """
    num_heads_q = query.size(2)
    num_heads_k = key.size(2)
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

    out, _lse, _rng = torch.ops.autoparallel.doc_packed_attn_op(
        query,
        key,
        value,
        cu_seq_q,
        n_docs,
        scale,
        list(window_size),
        enable_gqa,
    )
    return out
