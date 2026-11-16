# Interpreting AutoParallel Results

## ILP objective

Use `sharding_optimizer.get_json()["summary"]` for the selected plan's base
`total`, `compute`, `comm`, and `transition` costs. `search_objective` is the
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

For prefetch-discount sensitivity, compare endpoints as described in
`full-shard-planning.md` rather than selecting an intermediate scale.

## Interpret selected replication

An exported placement is the selected strategy, not the operator's option set,
so a replicated selection does not prove a missing propagation rule. Use
`explain_placement(target_placement)` on a live optimizer to see where a
sharded alternative stops being available and how its cost compares where it
is available. Operators that fall through to the replicate fallback, after
registered, single-dim, and decomposition-derived strategies fail, log
`implicitly registering ... with replicate_op_strategy` and are replicate-only.

When sharded alternatives existed, investigate the objective, boundary and
memory constraints, and workload size instead. Compute estimates have a 7 µs
kernel-launch floor, which `explain_placement` marks `[floor]`: on small
per-rank work, replicated and batch-sharded kernels can cost the same, so
duplicated global work looks competitive when it avoids gradient
communication. Treat that ranking as cost-model sensitivity until reordered
estimates or measurements support it.

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
cannot establish that custom operations, dynamic rank-dependent shapes, model
initialization, or later compiler transformations preserve the planned
contract. Report such failures at their actual evidence level, and use a
replicated counterfactual to isolate the cause when useful; do not silently
substitute that counterfactual for the selected plan.

Uneven shards follow `torch.chunk`: every rank gets the ceiling-sized extent
except the last ones, which are shorter or empty (10 over 4 ranks is
3, 3, 3, 1). A fixed view or reshape that assumes `local * world_size` equals
the global size fails on such a plan. Validate uneven-shard and tied-parameter
plans on both long and short ranks, and compare output shapes, values, and
gradients with the unsharded reference.

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
