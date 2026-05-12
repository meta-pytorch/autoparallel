# Adaptive Sharding: Sequence-Parallel vs Column-Parallel

With a more faithful communication cost model (the NCCL cost model), the
solver's choice between sequence-parallel and column-parallel sharding for
LLaMA3 models is clearly explained by a single ratio: the number of tokens
per DP rank vs the output dimension of each linear layer. This document
walks through the trade-off and how it produces different strategies for
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
equals the output dimension of the linear layer. This rule is most
directly relevant for forward projections like wq, wk, wv, wo, w1, and
w3. The down-projection w2 operates in a row-parallel configuration
where the trade-off is slightly different, but the same directional
intuition applies.

This crossover describes the dominant local memory-traffic trade-off for
individual projections; full-graph solver decisions can also reflect
redistribution and backward-pass interactions.

## Impact on LLaMA3 Models

### LLaMA3-8B

Model dimensions:
- Attention: wq/wo `[4096, 4096]`, wk/wv `[4096, 1024]` (GQA, 8 KV heads)
- MLP: w1/w3 `[4096, 14336]`, w2 `[14336, 4096]`

Training config (from the official recipe): `batch_size=2, seqlen=8192`,
giving `M = 2 × 8192 = 16,384` tokens per DP rank.

| Layer | N | M vs N | Strategy |
|---|---|---|---|
| wq, wo | 4,096 | 16,384 > 4,096 | Sequence-parallel |
| wk, wv | 1,024 | 16,384 > 1,024 | Sequence-parallel |
| w1, w3 | 14,336 | 16,384 ≈ 14,336 | Column-parallel |
| w2 | 4,096 | 16,384 > 4,096 | Row-parallel (TP) |

The attention projections are firmly in sequence-parallel territory
(`M/N ≥ 4`). The MLP gate/up projections (w1/w3) are near the crossover
(`M/N = 1.14`), and the solver chooses column-parallel TP. The
down-projection w2 uses row-parallel TP, pairing naturally with the
column-parallel w1/w3.

### LLaMA3-70B

Model dimensions:
- Attention: wq/wo `[8192, 8192]`, wk/wv `[8192, 1024]` (GQA, 8 KV heads)
- MLP: w1/w3 `[8192, 28672]`, w2 `[28672, 8192]`

Same training config: `batch_size=2, seqlen=8192`, `M = 16,384`.

| Layer | N | M vs N | Strategy |
|---|---|---|---|
| wq, wo | 8,192 | 16,384 > 8,192 | Sequence-parallel |
| wk, wv | 1,024 | 16,384 > 1,024 | Sequence-parallel |
| w1, w3 | 28,672 | 16,384 < 28,672 | Column-parallel |
| w2 | 8,192 | 16,384 > 8,192 | Row-parallel (TP) |

The 70B follows the same hybrid pattern as the 8B: sequence-parallel for
attention, column-parallel TP for MLP. The MLP is more firmly in
column-parallel territory (`M/N = 0.57`) compared to the 8B's marginal
case.

### Summary

| Model | Attention | MLP (w1/w3) | Notes |
|---|---|---|---|
| LLaMA3-8B (seqlen=8K) | Seq-par | Col-par (marginal) | M/N = 1.14 for MLP |
| LLaMA3-70B (seqlen=8K) | Seq-par | Col-par | M/N = 0.57 for MLP |

Both models use the same hybrid strategy: sequence-parallel for attention
(where `M > N`) and column-parallel TP for MLP (where `M ≈ N` or `M < N`).
The key insight is that attention projections have relatively small output
dimensions (4096 or 8192), making it cheaper to all-gather the weight and
keep activations sequence-sharded. MLP projections have large output
dimensions (14336 or 28672), making it cheaper to shard the weight on the
output dimension and replicate the activation across TP ranks.

## What This Means in Practice

The conventional parallelism recipe applies column-parallel TP uniformly
across all linear layers. The solver instead discovers a hybrid strategy
that uses sequence-parallel for attention and column-parallel TP for MLP.
This follows naturally from the `M` vs `N` crossover: attention weights
are small relative to the number of tokens, while MLP weights are large.

The hybrid strategy avoids the weight all-gather cost for MLP (where
weights are 3.5-7x larger than attention weights) while keeping
activations sequence-sharded for attention (where the per-GPU activation
is larger than the weight). The w2 down-projection uses row-parallel TP,
which pairs with the column-parallel w1/w3 to form the standard
Megatron-style MLP parallelism.

The NCCL cost model surfaces this trade-off because it prices
communication faithfully enough for the solver to distinguish between
strategies that differ by small bandwidth margins. The previous default
cost model inflated all-to-all costs by 5x, which masked these
differences and pushed the solver toward column-parallel uniformly. With
more accurate costs, the solver's decisions become interpretable through
the simple `M` vs `N` lens.

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
