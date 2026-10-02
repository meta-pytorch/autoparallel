# Full-shard planning and overlap uncertainty

Use this workflow only when parameter storage is near fully sharded, prefetch
assumptions materially affect the objective, or the selected compute layout is
topology-inverted. Ordinary plans do not need these extra solves.

## What the solver does and does not establish

A `1 / world_size` parameter bound forces every sufficiently shardable
parameter toward its minimum local fraction. It constrains persistent parameter
placement; it does not model the full training-memory peak, optimizer state,
activation checkpointing, or the lifetime of materialized FSDP buffers.

The current memory constraint averages each parameter tensor's local sharding
ratio with equal weight. It is not byte-weighted. This distinction is usually
irrelevant at the exact fully-sharded endpoint, but it matters for relaxed
bounds and heterogeneous tensor sizes. Always compute the selected local
parameter bytes after solving.

`apply_prefetch_discount(scale)` multiplies eligible communication costs in
place. Repeated calls compound and cannot reconstruct the undiscounted costs.
Use a fresh capture or reload the same undiscounted saved optimizer for each
endpoint.

The discount is candidate-independent even though realizable overlap depends on
the placement, bucket sizes, available compute, collective resources, and live
buffer memory. Do not tune the scalar to make the solver emit a preferred plan;
that is circular evidence.

## Build a bounded placement envelope

Capture once and retain enough optimizer state to reproduce independent solves.
Use at most these candidates unless the user asks for a sweep:

1. **Serial endpoint:** no prefetch discount. This is a pessimistic overlap
   proxy, not a runtime upper bound with guaranteed estimator accuracy.
2. **Optimistic endpoint:** `apply_prefetch_discount(scale=0.0)` on a fresh
   optimizer. This treats eligible parameter and terminal communication as
   free; it is not an executable prediction and excludes transient-buffer
   pressure.
3. **Compute-placement baseline, when diagnostic value justifies one more
   solve:** remove or omit the parameter-memory constraint. This shows the
   preferred compute layout but is not a viable training plan until persistent
   state is assessed.

Do not run intermediate discount values by default. Deduplicate candidates by
their persistent parameter placements, heavy forward/backward operation
placements, and dominant redistribution families.

If the serial and optimistic endpoints choose substantially the same layout,
call the placement overlap-robust within the modeled endpoints. If they differ,
do not select an intermediate discount as truth. Return the distinct placement
families as a provisional candidate set and evaluate them after lowering.

## Inspect storage, compute, and topology separately

For each candidate, report:

- parameter placements by count and by global parameter bytes;
- achieved local parameter bytes and byte-weighted fraction;
- heavy matmul, attention, normalization, embedding, and output placements;
- which physical mesh axis carries latency-sensitive TP or EP collectives;
- the largest parameter-derived, terminal-gradient, and activation
  redistributions;
- global batch, effective DP degree, and local batch per DP group.

Mesh dimension names are intentions, not constraints on the ILP. Flag a
topology inversion when large-model compute uses a slow inter-node axis while a
fast intra-node axis primarily shards batch or tokens. Investigate it; do not
automatically force the conventional orientation.

Distinguish these two fully-sharded storage regimes:

```text
pure FSDP: persistent S(dp)S(tp) -> compute R(dp)R(tp)
FSDP + TP: persistent S(dp)S(tp) -> compute R(dp)S(tp)
```

They may have identical persistent parameter memory. FSDP + TP retains a
smaller materialized weight and trades parameter communication for TP
activation communication. A zero prefetch discount erases much of that
advantage and therefore tends to favor pure FSDP when batch parallelism is
available.

## Report memory without claiming a pre-AC peak

Separate quantities that are known before graph partitioning from those that
are not:

- Report persistent parameter bytes exactly from the selected placements.
- Estimate gradients, master weights, optimizer moments, and teacher state only
  from the stated training dtypes and ownership assumptions. Label estimates.
- Do not claim an activation or total peak before forward/backward partitioning,
  AC selection, and scheduling.
- After lowering, report forward and backward peaks separately, including
  materialized parameter and communication buffers represented by the
  estimator.

## Evaluate only distinct lowered candidates

Apply and lower the distinct envelope candidates with the same topology,
mixed-precision policy, AC configuration, and autobucketing settings. Use the
actual reordering pass to collect forward and backward critical path, exposed
communication, and peak memory. A discounted ILP objective does not replace
this evaluation. Record collective counts before and after bucketing when they
are available. Two plans with similar heavy-compute DP/TP roles can lower very
differently when their persistent placements produce different materialization
groups or prevent compiler-realizable buckets. Also report candidate-specific
initialization or lowering warnings that globally materialize a sharded value.

When lowering is unavailable, keep the recommendation provisional and state
which candidate-dependent overlap or memory signal is missing. Do not report
the optimistic endpoint as the selected deployment.

## Confidence card

Include this card for a full-shard plan:

```text
SPMD semantics: audited | unresolved
topology model: explicit NCCL | detected NCCL | PyTorch fallback
memory scope: persistent parameters | estimated persistent state | post-AC peak
overlap sensitivity: stable endpoints | placement-changing
evidence: ILP-feasible | lowered | compiled | numerically checked | measured
recommendation: final | provisional candidate set
```

A final recommendation requires resolved SPMD semantics, an appropriate
topology model, and evidence beyond a placement-changing discounted ILP. Target
measurement is still needed when modeled candidates are close.
