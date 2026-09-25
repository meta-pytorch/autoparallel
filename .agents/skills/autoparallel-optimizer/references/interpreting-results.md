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
latency proxy and report the scale.

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

## Boundaries

- A valid plan is meaningful only when the model expresses the intended global
  SPMD computation.
- A replicate-only region may be legal and slow; highlight material replication.
- `local_map` exposes a boundary but does not price or optimize internal
  communication.
- Dynamic-shape costs use traced shape hints unless the estimator models the
  range explicitly.
- Full optimizer state is trusted, version-coupled pickle data. Placement JSON
  is smaller but still requires a matching graph and mesh.
