# Interpreting AutoParallel Results

## ILP objective

Use `sharding_optimizer.get_json()["summary"]` for the selected plan's total,
compute, communication, and transition costs. Compute and communication are
runtime estimates in microseconds. The transition cost is a `1.0` tie-breaker
that favors fewer redistribution steps when equal-cost redistributions could be
fused; do not present it as a measured kernel-launch cost.

With undiscounted communication, the sum is a modeled serial upper-bound proxy
for forward-plus-backward runtime: it ignores communication/compute overlap,
while its component models remain approximate. If
`apply_prefetch_discount(scale < 1)` was used, call the objective a discounted
latency proxy and report the scale. A zero scale is an optimistic endpoint, not
evidence that all eligible communication is hidden.

The discount applies one scale across candidate placements. Realized exposure
can change with collective sizes, bucket composition, available compute,
communicator contention, and live-buffer memory. A scale calibrated for one
placement does not establish another placement's cost.

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

For a full-shard sensitivity comparison, use independently loaded serial and
zero-discount optimizers. If their material parameter and heavy-compute layouts
agree, report endpoint stability. If they differ, retain the distinct layouts
as provisional candidates and compare post-reordering metrics; do not select an
intermediate scale merely because it produces a familiar strategy.

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

Treat placement application as a separate legality gate. Strategy propagation
may admit a shard whose local shape is ceil-padded when a dimension is not
divisible by the mesh. A later fixed `view` can reject that padded shape.
Likewise, tied parameters or aliased buffers can plan successfully but be
reconstructed with an incompatible local shape or missing state. Report these
as lowering failures, and use a replicated counterfactual to isolate the cause
when useful; do not silently substitute that counterfactual for the selected
plan.

## Boundaries

- A valid plan is meaningful only when the model expresses the intended global
  SPMD computation.
- A replicate-only region may be legal and slow; highlight material replication.
- `local_map` exposes a boundary but does not price or optimize internal
  communication.
- Dynamic-shape costs use traced shape hints unless the estimator models the
  range explicitly.
- The parameter-memory constraint currently averages per-tensor local ratios,
  not local parameter bytes. Report the achieved byte-weighted fraction for a
  relaxed constraint.
- Persistent parameter memory excludes optimizer state, activation peaks, and
  transient materialization buffers unless those were evaluated separately.
- Full optimizer state is trusted, version-coupled pickle data. Placement JSON
  is smaller but still requires a matching graph and mesh.
