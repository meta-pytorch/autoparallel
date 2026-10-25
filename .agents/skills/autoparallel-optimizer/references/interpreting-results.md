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
memory, rank candidates, and decide which target runs are most informative.
Record those affirmative uses as `valid_meta_uses: legality/feasibility
checking, candidate ranking, scale/memory exploration, benchmark
prioritization`; do not leave one implicit merely because the response names
what meta execution cannot prove.
When the calibration envelope cannot preserve the modeled advantage, record
`verdict: inconclusive` explicitly before retaining the production fallback.
It is not runtime evidence. A calibrated meta-to-target mapping may support a
relative claim only when its regime-matched held-out error envelope preserves
the decision. A broad global fit with large architecture outliers remains
`META_RUNTIME_ESTIMATE_RANKING_ONLY` even when its median error is modest.

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

For each strategy, keep the optimistic legal-overlap estimate, the expected
reordered estimate, and the pessimistic serialized-communication estimate.
Expand that envelope by held-out calibration error. The same estimator can
still have strategy-dependent bias, so applying one reordering pass to every
candidate does not make the comparison measured or automatically fair.

Use `robustly_competitive` only when AutoParallel's pessimistic calibrated
bound remains competitive with the strongest baseline's optimistic calibrated
bound. Use `robustly_worse` only when AutoParallel's optimistic bound loses to
a baseline's pessimistic bound. Overlapping intervals are
`schedule_sensitive`; preserve an existing baseline or use the normal fallback
rather than declaring a win. If a template or calibration is missing, use
`comparison_unavailable`.

Latency tolerance absorbs uncertainty; it is not a reason to select a plan
that loses on every acceptance dimension. When one baseline's pessimistic
latency is no worse than AutoParallel's optimistic latency and it also uses no
more complete-step peak memory and is no worse on recorded startup or other
objectives, use `pareto_dominated` and route to that baseline. Require robust
envelopes for the compared dimensions rather than point-estimate dominance.

Treat memory as a separate hard constraint with an explicit scope. Persistent
parameter storage alone cannot establish DDP/FSDP feasibility. A baseline OOM
can establish `baseline_infeasible`, but it does not establish throughput
parity. Never label a counterfactual as measured unless that implementation ran
on the stated target.

A measured forward/loss/backward peak also does not establish optimizer-step
feasibility when the optimizer state was absent. Report the measured peak and
the projected training peak separately. Add only persistent tensors absent from
the measured boundary: if parameters and gradients are already live, an AdamW
projection normally adds its two local moment tensors at their actual dtypes
and placements. Include an explicit operational reserve. If the projection
misses the limit, the candidate fails the memory bar even when its measured
forward/backward latency wins.
Keep all supplied evidence labels explicit even when memory decides routing:
label the partial-step latency and forward/backward peak `measured`, the added
optimizer-state bytes `analytical`, and the complete-step peak `projected`.

Use calibration only for the regime it covers. Record in-sample and held-out
error, and retain the larger applicable uncertainty for architecture families,
model sizes, or launch-bound workloads with sparse coverage. A low aggregate
median does not override a large held-out outlier; use
`comparison_unavailable` when no conservative comparable error bound exists.

Match the latency boundary to the decision. Forward/loss/backward measurements
can diagnose the sharding and communication schedule, but parameter placement
also changes optimizer update work and DTensor overhead. Do not transfer a
partial-boundary ranking to end-to-end training unless optimizer cost was
measured or conservatively modeled for every strategy. Record
`OPTIMIZER_PERFORMANCE_UNMODELED` or `STEP_BOUNDARY_MISMATCH` rather than
folding the unknown cost into communication-model calibration.
When gradient accumulation is enabled, match its frequency semantics too. A
cycle with `G` microsteps generally has `G - 1` reduction-suppressed
microsteps, one synchronized microstep, and one optimizer update. Compare the
whole cycle or its correctly amortized latency; a single synchronized sample
plus update can reverse the production ranking.
Treat an unset/default foreach or fused option as a configured policy, not an
observed implementation. Profile or otherwise record the realized optimizer
kernel family for each strategy before attributing a gap to sharding; tensor
subclasses and placements may change the framework's automatic choice.
For a modeled update, a useful placement-dependent feature is local parameter
or optimizer-state bytes, combined with a fixed dispatch term. Regard the fit
as valid only for its measured kernel family, dtype policy, framework version,
hardware, and workload regime. A small in-sample residual is not routing
evidence: freeze the fit, validate it prospectively on held-out architectures,
and retain the maximum relevant prediction error in the comparison envelope.
When using a profiler, fit the optimizer's descendant update kernels rather
than the enclosing `Optimizer.step` device span. The parent span may include
backward communication that was still pending when the update began; retain
that tail in the full-step communication/scheduling analysis.

A composed prediction has two uncertainty sources: the base
forward/loss/backward envelope and the optimizer increment. Freeze both, not
just the increment model. Prefer contemporaneous independent launches for the
base. If a historical base is reused, expand it by a validated temporal-drift
bound for the same template and regime. When the optimizer kernels predict
well but a refreshed partial control explains a full-step miss, record
`COMPOSED_FULL_STEP_BASE_DRIFT`; do not inflate the optimizer-model error or
claim that the increment model failed.

Iteration samples from one process do not measure launch-to-launch variance.
They also do not cover separate-process compilation, allocator initialization,
or system-state variance.
Before accepting AutoParallel from measured relative performance, run every
strategy in at least three independent launches and compare the combined launch
envelope. A close single-launch result is schedule-sensitive or inconclusive.
Use `MEASURED_LAUNCH_VARIANCE_UNMODELED`; do not call a tight p10-p90 interval
within one launch robust evidence.

Time-to-ready and steady-state latency answer different questions. The first
synchronized warmup can include lazy graph compilation, specialization, and
allocator initialization. Keep it out of CUDA-event step samples, but include
it when estimating a finite job's time-to-solution. Report the step-count
break-even when startup and steady-state rankings disagree.

Distinguish an optimizer-objective miss from a propagation failure. If the
selected plan has broad sharding coverage but retains substantially more local
parameter bytes than full-shard, meets the forward/backward bar, and loses
after the optimizer update is included, record
`OPTIMIZER_COST_MISSING_FROM_OBJECTIVE`. The current sharding objective does
not price eager optimizer update work. Use the full-step fallback result for
routing and retain the supplied numeric local/global parameter fractions plus
every supplied partial- and full-boundary latency and memory result so the
optimizer cost can later be added to search. If only the ordering is supplied,
preserve it and note that exact values are unavailable rather than inventing
them. State the partial-boundary and full-step orderings separately, including
when the measured full-step fallback is both faster and smaller. Keep a compact
stable-finding record with the reason code, parameter-storage comparison,
`partial_boundary:` observed ordering, and `full_boundary:` latency plus memory
ordering. A statement about what the objective favored is not a record of the
observed partial-boundary winner.
When authorized, validate this diagnosis with one narrow parameter-storage
counterfactual near the full-shard fraction. An improvement in both full-step
latency and memory despite extra communication isolates the missing optimizer
term; it does not make a hard storage constraint the new default.

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
