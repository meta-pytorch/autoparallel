# Interpreting AutoParallel Results

## ILP objective

Use `sharding_optimizer.get_json()["summary"]` for the selected plan's base
total, compute, communication, and transition costs. `search_objective` is the
actual objective after any prefetch scale; with undiscounted communication it
matches the base total. Compute and communication are runtime estimates in
microseconds. The transition cost is a `1.0` tie-breaker that favors fewer
redistribution steps when equal-cost redistributions could be fused; do not
present it as a measured kernel-launch cost.

With undiscounted communication, the sum is a modeled serial upper-bound proxy
for forward-plus-backward runtime: it ignores communication/compute overlap,
while its component models remain approximate. If
`apply_prefetch_discount(scale < 1)` was used, call `search_objective` a
discounted latency proxy and report the scale. The base cost breakdown remains
undiscounted. A zero scale is an optimistic endpoint, not evidence that all
eligible communication is hidden.

The discount applies one scale across candidate placements. Realized exposure
can change with collective sizes, bucket composition, available compute,
communicator contention, and live-buffer memory. A scale calibrated for one
placement does not establish another placement's cost.

Meta or fake-tensor execution can produce this objective without allocating
the real model. Use it to check capture/lowering legality, explore meshes and
memory, rank candidates, and decide which target runs are most informative. It
is not runtime evidence. A calibrated meta-to-target mapping may support a
relative claim only when its regime-matched held-out error envelope preserves
the decision. A broad global fit with large architecture outliers remains
ranking evidence (`META_RUNTIME_ESTIMATE_RANKING_ONLY`) even when its median
error is modest.

## Reordered graph metrics

`estimate_graph_metrics()` returns:

- `total_time`: estimated critical-path duration;
- `compute_time`: sum of compute-node durations;
- `communication_time`: sum of collective durations;
- `exposed_comm_time`: estimated compute-stream stalls at synchronization;
- `peak_memory`: estimated peak live tensor bytes.

Collect these independently for the reordered forward and backward graphs. Add
their `total_time` values for an estimated forward-plus-backward duration, but
report peak memory separately because phase peaks are not additive. The
parameter update is absent unless separately captured.

These values remain modeled. Use `measured` only for timed execution on the
stated hardware and topology.

Report findings are triage signals. `HIGH_COMMUNICATION_FRACTION` compares the
selected plan's communication cost with its undiscounted total and includes the
largest priced edges. `REPLICATED_PARAMETER_STORAGE` compares byte-weighted
local and global persistent parameter storage. `CLUSTERING_GROUPS_SKIPPED`
means the solve remained valid but used less ILP compression.
`EXTRAPOLATED_CRITICAL_TERM` identifies topology cost coverage that needs a
benchmark or calibration case. None of these findings proves a numerical
correctness defect or a measured performance regression by itself.

## Compare plans

Compare, when available:

1. the ILP serial or discounted proxy;
2. the reordered forward-plus-backward critical path;
3. measured iteration time.

Hold workload, mesh, dtype, activation checkpointing, estimator, and scheduling
configuration fixed. ILP-versus-reordered disagreement indicates overlap,
bucketing, or scheduling effects. Reordered-versus-measured disagreement points
to compiler, kernel, topology, or cost-model error. Prefer measurement for close
candidates.

A mesh change alters both legal strategies and collective costs. Treat 1D and
2D meshes as separate optimization runs and record why an inferred mesh was
chosen.

A batch or sequence-shape change is also a different execution configuration.
Do not use a larger synthetic batch to accept a plan for an underfilled user
workload. A bounded sensitivity sweep can show where the optimizer changes
placement families, but the original workload keeps its own routing decision.

For a full-shard sensitivity comparison, solve and save the serial endpoint,
then replace the objective scale and solve the zero-discount endpoint. If their
material parameter and heavy-compute layouts agree, report endpoint stability.
If they differ, retain the distinct layouts as provisional candidates and
compare post-reordering metrics; do not select an intermediate scale merely
because it produces a familiar strategy.

## Interpret selected replication

The exported node placement is the selected strategy, not the operator's full
strategy space. A replicated selection therefore does not establish a missing
propagation rule. Use the report's `summary.strategy_space` counts or inspect a
trusted live/full optimizer to separate replicate-only option sets from
replicated selections that had sharded alternatives. Correlate true
replicate-only nodes with implicit-fallback warnings and their shapes before
proposing a propagation rule.

If most replicated selections had sharded alternatives, investigate the cost
objective, boundary and memory constraints, and workload representativeness.
In particular, a fixed per-operation launch floor can give replicated and
batch-sharded small kernels identical compute estimates. That makes duplicated
global work look competitive when it avoids gradient communication. Treat this
as cost-model sensitivity until post-lowering estimates or target measurements
support the ranking.

## Compare with DDP and FSDP counterfactuals

A replicated or fully-sharded AutoParallel placement is not automatically a
faithful PyTorch DDP or FSDP baseline. Use a versioned implementation-aware
template and follow `baseline-counterfactuals.md`. Hold the execution contract
fixed and record which reducer, bucketing, prefetch, resharding, and memory
assumptions the template represents.

For measured regressions, compare collective launch counts and bucket
composition as well as aggregate communication time. A sharded plan can move
less persistent state yet lose because it emits several times more reductions
or decomposes matmuls into smaller kernels. Use all-rank profiles and keep this
distinct from optimizer self time.

Do not infer that every excess reduction is bucketable. Use post-lowering
diagnostics to separate terminal gradient reductions from collectives on the
activation critical path. A later collective that is a descendant of an
earlier collective cannot be horizontally merged without changing the
computation. If these dependency-bound collectives dominate, the issue is the
selected tensor-parallel placement or its search cost, not merely the
reordering pass. Compare a bounded replicated or lower-TP candidate and use
post-lowering metrics to rerank when they recover the measured ordering.

On a multi-dimensional mesh, inspect collective counts per process group, not
only in aggregate. A group discovered from FSDP-like all-gathers may not cover
all-reduce-only DP or TP groups in the same graph. A large pre/post discrepancy
for omitted groups is bucketing-coverage evidence, but promoting broader group
discovery requires target execution: changing the interleaving of collectives
on overlapping communicators can affect liveness as well as performance.
Even safe, successful bucketing can be unprofitable when the saved launches are
small relative to packing, synchronization, or local-kernel overhead. Require
repeated end-to-end improvement on a held-out architecture; do not infer
profitability from collective-count reduction alone.

Aggregate per-rank profiles can also hide communicator-ordering stalls. Compare
collective kernel time on every rank and map stable high/low cohorts to mesh
coordinates and process-group membership. If broad bucketing reduces max-rank
time far more than the ideal collective model predicts, record
`CROSS_GROUP_RANK_SKEW_UNMODELED`; the missing term is peer wait or launch-order
coordination, not necessarily bandwidth or a universal per-launch constant.

A solver timeout is not plan or lowering evidence. Distinguish graph/ILP
construction time from CBC import/presolve and timed search, record the
configured solver limit plus end-to-end elapsed time, and reject a timed-out
incumbent even if CBC found a feasible objective. Use a smaller architectural
canary only to isolate a mechanism; do not present it as evidence that the
original workload parallelizes successfully.

For relative verdicts, use the envelopes, classifier, and Pareto guard in
`baseline-counterfactuals.md` and the acceptance rules in
`decision-principles.md`. Applying one reordering pass to every candidate does
not make the comparison measured or automatically fair: the same estimator can
have strategy-dependent bias.

Treat memory as a separate hard constraint with an explicit scope. Persistent
parameter storage alone cannot establish DDP/FSDP feasibility, and a baseline
OOM establishes `baseline_infeasible`, not throughput parity.

## Evidence ladder

- **ILP-feasible:** the enumerated decision and graph-flow constraints admit the
  selected strategy.
- **Lowered:** AutoParallel produced a parallel graph or module.
- **Compiled:** forward and backward compiled successfully.
- **Numerically checked:** outputs and gradients matched a stated reference and
  tolerances.
- **Executed on target:** real collectives ran on the stated topology.
- **Measured:** runtime or memory came from target execution.

AutoParallel constructs legal plans within its supported strategy rules.
Numerical and target checks can still expose implementation, compiler,
custom-op, or opaque-region defects.

Parallelization selection is separate from the evidence ladder. Record
`accepted` only when AutoParallel meets the explicit acceptance criteria. Use
`below_bar` when evidence demonstrates a miss, `failed` for an attempted path
that errors, `ineligible` when required preconditions are absent, and
`inconclusive` when the necessary evidence was not collected. Selecting DDP or
FSDP must retain the linked AutoParallel outcome and its stable reason codes.
Relative competitiveness is another independent dimension: record whether it
was modeled, measured, or mixed, and do not upgrade target execution evidence
because an offline baseline estimate exists.

Treat placement application as a separate legality gate. Strategy propagation
cannot establish that custom operations, dynamic rank-dependent shapes, model
initialization, or later compiler transformations preserve the planned
contract. Report such failures at their actual evidence level, and use a
replicated counterfactual to isolate the cause when useful; do not silently
substitute that counterfactual for the selected plan.

## Boundaries

- A valid plan is meaningful only when the model expresses the intended global
  SPMD computation.
- A replicate-only region may be legal and slow; highlight material replication.
- `local_map` exposes a boundary but does not price or optimize internal
  communication.
- Dynamic-shape costs use traced shape hints unless the estimator models the
  range explicitly.
- Parameter-memory bounds and the reported achieved fraction cover persistent
  parameter bytes, not total training memory.
- Persistent parameter memory excludes optimizer state, activation peaks, and
  transient materialization buffers unless those were evaluated separately.
- Full optimizer state is trusted, version-coupled pickle data. Placement JSON
  is smaller but still requires a matching graph and mesh.
