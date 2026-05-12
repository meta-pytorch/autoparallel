# Adaptive Sharding: Sequence-Parallel vs Column-Parallel

With a more faithful communication cost model (the NCCL cost model), the
solver's choice between sequence-parallel and column-parallel sharding for
LLaMA3 models is explained by a combination of the forward memory-traffic
crossover (`M` vs `N`) and backward-pass considerations such as gradient
reduction costs and residual connection compatibility. This document walks
through the trade-off and how it produces the solver's strategies for
LLaMA3-8B and LLaMA3-70B at the same training configuration.

## Background

For an `nn.Linear` with 3D input `[B, S, D_in]` and weight `[D_in, D_out]`
on a 2D mesh `(DP, TP)`, there are two main sharding strategies:

**Column-parallel** (standard FSDP+TP): the activation is replicated across
TP ranks and the weight is sharded on the output dimension. Each GPU reads
the full-sequence activation but only its slice of the weight.

**Sequence-parallel**: the activation is sharded across TP ranks on the
sequence dimension and the weight is fully replicated. Each GPU reads only
its slice of the activation but the entire weight.

Both strategies compute the same FLOPs — the difference is purely in how
much data each GPU reads from memory.

## The Crossover

For a weight `[K, N]` with TP degree `T` and `M = B × S / DP` tokens per
DP rank, the total bytes read per GPU are:

| Strategy | Activation | Weight | Total |
|---|---|---|---|
| Column-parallel | `M × K` | `K × N / T` | `M×K + K×N/T` |
| Sequence-parallel | `M × K / T` | `K × N` | `M×K/T + K×N` |

Setting the two equal and simplifying:

```
M × K × (1 − 1/T) = K × N × (1 − 1/T)
M = N
```

- When **M > N** (more tokens than output features): sequence-parallel
  reads less total data.
- When **M < N** (fewer tokens than output features): column-parallel reads
  less total data.

The crossover point is `M = N`: when the number of tokens per DP rank
equals the output dimension of the linear layer. This rule describes the
dominant local memory-traffic trade-off for individual projections.
However, the full-graph solver also considers backward-pass costs
(gradient reduction, weight gradient dtype), residual connection layout
compatibility, and redistribution costs between adjacent ops. These
additional factors can shift the decision away from what the forward-only
`M` vs `N` analysis predicts.

## Impact on LLaMA3 Models

### LLaMA3-8B

Model dimensions:
- Attention: wq/wo `[4096, 4096]`, wk/wv `[4096, 1024]` (GQA, 8 KV heads)
- MLP: w1/w3 `[4096, 14336]`, w2 `[14336, 4096]`

Training config (from the official recipe): `batch_size=2, seqlen=8192`,
giving `M = 2 × 8192 = 16,384` tokens per DP rank.

| Layer | N | M vs N | Strategy |
|---|---|---|---|
| wq | 4,096 | 16,384 > 4,096 | Column-parallel |
| wk, wv | 1,024 | 16,384 > 1,024 | Column-parallel |
| wo | 4,096 | 16,384 > 4,096 | Sequence-parallel |
| w1, w3 | 14,336 | 16,384 ≈ 14,336 | Column-parallel |
| w2 | 4,096 | 16,384 > 4,096 | Row-parallel (TP) |

Despite `M > N` for the attention projections (which would favor
sequence-parallel by the forward-only analysis), the solver chooses
column-parallel for wq/wk/wv. This is because column-parallel produces
weight gradients with `P(sum)S(0)` placement — only one reduce-scatter
dimension — which is cheaper than the `P(sum)P(sum)` gradient from
sequence-parallel that requires a full 2D reduce-scatter.

The wo projection stays sequence-parallel because its output feeds
directly into the residual add, which expects `S(0)S(1)` placement.
Column-parallel wo would produce `S(0)P(sum)`, requiring a costly
all-reduce before the add. Sequence-parallel wo outputs `S(0)S(1)`,
matching the residual for free.

### LLaMA3-70B

Model dimensions:
- Attention: wq/wo `[8192, 8192]`, wk/wv `[8192, 1024]` (GQA, 8 KV heads)
- MLP: w1/w3 `[8192, 28672]`, w2 `[28672, 8192]`

Same training config: `batch_size=2, seqlen=8192`, `M = 16,384`.

| Layer | N | M vs N | Strategy |
|---|---|---|---|
| wq, wo | 8,192 | 16,384 > 8,192 | Column-parallel |
| wk, wv | 1,024 | 16,384 > 1,024 | Column-parallel |
| w1, w3 | 28,672 | 16,384 < 28,672 | Column-parallel |
| w2 | 8,192 | 16,384 > 8,192 | Row-parallel (TP) |

The 70B uses column-parallel TP uniformly across all projections —
matching the standard Megatron-style parallelism recipe. With `M/N = 2.0`
for attention (closer to the crossover than the 8B's `M/N = 4.0`), the
backward-pass cost of `P(sum)P(sum)` gradient reduction outweighs the
forward memory-traffic advantage of sequence-parallel. For MLP,
column-parallel is clearly favored (`M/N = 0.57`).

Unlike the 8B, wo also uses column-parallel here. The larger weight
size (8192×8192 vs 4096×4096) makes the sequence-parallel weight
all-gather more expensive, tilting the balance toward column-parallel
despite the residual add cost.

### Summary

| Model | wq/wk/wv | wo | MLP (w1/w3) | w2 |
|---|---|---|---|---|
| LLaMA3-8B | Col-par | Seq-par | Col-par | Row-par |
| LLaMA3-70B | Col-par | Col-par | Col-par | Row-par |

The 70B converges to standard Megatron-style TP everywhere. The 8B uses
a hybrid where wo remains sequence-parallel for residual add
compatibility. In both cases, the MLP uses column-parallel TP paired with
row-parallel w2.

## What This Means in Practice

The solver's decisions align with the conventional Megatron-style TP
recipe for LLaMA3-70B. For the smaller 8B model, the solver discovers a
minor variation: wo uses sequence-parallel to avoid an all-reduce before
the residual connection, while the rest of the attention uses
column-parallel.

The `M` vs `N` crossover remains a useful first-order heuristic for
understanding strategy selection: layers where the weight is large
relative to the activation (MLP) clearly favor column-parallel, while
layers with small weights relative to the activation could go either
way. But the full-graph solver accounts for additional costs — gradient
reduction in the backward pass, redistribution between adjacent ops, and
residual connection compatibility — that shift some decisions away from
the forward-only prediction.

The NCCL cost model surfaces these trade-offs because it prices
communication faithfully enough for the solver to distinguish between
strategies that differ by small bandwidth margins.

## SDPA (Attention Computation)

Regardless of which strategy is used for the linear projections, SDPA
always runs head-parallel: each GPU handles a subset of attention heads.
When sequence-parallel projections feed into head-parallel SDPA, all-to-all
transitions convert between sequence-sharded and head-sharded layouts.
These transitions are intra-node on NVSwitch, where all-to-all is much
less punitive than in inter-node settings and can be competitive with
all-gather for the sizes relevant here.

## Requirements

This adaptive behavior requires two features:

1. **NCCL cost model** (`cost_model="nccl"`, the default): provides
   physically grounded communication costs that distinguish intra-node
   from inter-node bandwidth.

2. **Einsum fusion** (`_APPLY_VIEW_MM_VIEW_PATTERN=True`, the default):
   preserves the sequence dimension in the graph representation. Without
   it, the `view → mm → view` decomposition folds sequence into batch,
   making sequence-parallel strategies invisible to the solver.
