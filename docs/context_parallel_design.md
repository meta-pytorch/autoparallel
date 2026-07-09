# Context Parallel Attention Design

This document describes how AutoParallel represents context-parallel attention
inside the graph. The user-facing API is documented in
[Context Parallel Attention](context_parallel.md).

## Goal

The CP attention helper is model support for transformer attention, not a
standalone parallel runtime. The intended contract is:

- AutoParallel sees Q, K, V, and output with the same placement tuple.
- The CP axis shards the sequence dimension for all four tensors.
- The graph outside the attention region does not need special K/V replicate
  constraints or attention-specific gradient constraints.
- Attention-specific communication stays inside the wrapped attention region.

## Placement Contract

For tensors shaped `(batch, heads, sequence, head_dim)`, the default mapping is:

| Mesh role | Placement |
| --- | --- |
| DP shard | `Shard(batch_dim)` |
| CP | `Shard(seq_dim)` |
| TP | `Shard(head_dim)` |
| Extra DP axis | `Shard(batch_dim)` |

The same tuple is used for Q, K, V, and output. This differs from the earlier
TorchTitan-style boundary where Q was sequence-sharded and K/V were replicated
on the CP axis.

## Attention Region

The helper uses `local_map` as the boundary between AutoParallel placement
optimization and the attention kernel. The wrapped function receives local
tensor shards and can use `autoparallel.collectives` directly.

For non-causal SDPA:

- K and V are gathered across the CP axis before SDPA runs.
- Q remains sequence-local.
- The output remains sequence-local.

For causal SDPA:

- Q, K, and V are gathered across the CP axis before SDPA runs.
- Each rank computes the same full-sequence causal attention.
- The full output is reduce-scattered back to the CP sequence shard.

The causal path uses duplicate full attention computation to preserve the
simple Q/K/V-sharded boundary without requiring rank-dependent masks in the
traced local_map body.

## Gradient Behavior

The collective wrappers in `autoparallel.collectives` define autograd formulas:

- `all_gather` backward is reduce-scatter.
- `reduce_scatter` backward is all-gather.

That keeps the backward boundary consistent with the forward placement
contract. No `in_grad_placements` are required on the local_map call.

## Public Surface

The user-facing APIs are exported from `autoparallel`:

- `make_context_parallel_sdpa`
- `context_parallel_attention_placements`
- `context_parallel_local_map`
- `ContextParallelPlacements`

The implementation lives in a private module so model code can depend on the
stable package-level API instead of an implementation file name.

## Model Integration

The in-repo Llama model can opt in through
`TransformerModelArgs.context_parallel_mesh`. When this field is set,
`build_attention` constructs SDPA through the package-level
`make_context_parallel_sdpa` API. When it is unset, the existing non-CP SDPA
path is unchanged.

## Test Coverage

The tests cover:

- Mesh axis aliases, defaults, and validation errors.
- Q/K/V/output placement contracts.
- Public package-level imports.
- In-repo Llama attention construction through the public API.
- End-to-end LocalTensor correctness for causal and non-causal SDPA.
