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
| w1, w3 | 14,336 | 16,384 > 14,336 | Sequence-parallel (marginal) |
| w2 | 4,096 | 16,384 > 4,096 | Sequence-parallel |

At this training config, the NCCL cost model favors sequence-parallel
across the major linear projections. The MLP gate/up projections (w1/w3)
are close to the crossover — `M` exceeds `N` by only 14%. With a shorter
sequence (e.g. `seqlen=4096`, `M = 8,192`), the MLP flips to
column-parallel while attention stays sequence-parallel.

### LLaMA3-70B

Model dimensions:
- Attention: wq/wo `[8192, 8192]`, wk/wv `[8192, 1024]` (GQA, 8 KV heads)
- MLP: w1/w3 `[8192, 28672]`, w2 `[28672, 8192]`

Same training config: `batch_size=2, seqlen=8192`, `M = 16,384`.

| Layer | N | M vs N | Strategy |
|---|---|---|---|
| wq, wo | 8,192 | 16,384 > 8,192 | Sequence-parallel |
| wk, wv | 1,024 | 16,384 > 1,024 | Sequence-parallel |
| w1, w3 | 28,672 | 16,384 < 28,672 | **Column-parallel** |
| w2 | 8,192 | 16,384 > 8,192 | Sequence-parallel |

The solver discovers a **hybrid strategy**: sequence-parallel for attention
projections and column-parallel for the MLP gate/up projections. This
happens because the 70B MLP has a much larger output dimension (28,672 vs
14,336 for 8B), pushing those layers below the crossover.

### Summary

| Model | Attention | MLP (w1/w3) | Notes |
|---|---|---|---|
| LLaMA3-8B (seqlen=8K) | Seq-par | Seq-par (marginal) | Near crossover in MLP |
| LLaMA3-8B (seqlen=4K) | Seq-par | Col-par | Mixed regime |
| LLaMA3-70B (seqlen=8K) | Seq-par | Col-par | Hybrid attention/MLP |

## What This Means in Practice

The conventional parallelism recipe applies column-parallel TP uniformly
across all linear layers. This was designed for large models (70B+) where
TP is essential for fitting weights in memory, and where the MLP's large
output dimension makes column-parallel bandwidth-efficient.

For smaller models like LLaMA3-8B at long sequence lengths, the solver
finds that sequence-parallel can be more efficient — the activation
all-gather that column-parallel requires becomes the bottleneck when there
are many tokens per GPU. This is consistent with the possibility that the
standard 8-way TP configuration used for LLaMA3-8B training was inherited
from larger-model recipes rather than optimized specifically for the 8B
scale.

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
