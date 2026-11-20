# Interpreting AutoParallel Results

## ILP objective

Use `sharding_optimizer.get_json()["summary"]` for the selected plan's base
`total`, `compute`, `comm`, and `transition` costs. Compute and communication
are runtime estimates in microseconds. The transition cost is a `1.0`
tie-breaker that favors fewer redistribution steps when equal-cost
redistributions could be fused; do not present it as a measured cost.

With undiscounted communication, the total is a modeled serial upper-bound
proxy for forward-plus-backward runtime: it ignores communication/compute
overlap, and its component models are approximate. `search_objective` is the
objective actually solved; with a prefetch discount it is a discounted proxy
while the base breakdown stays undiscounted. Report the scale, and see
`full-shard-planning.md` for why one scale does not transfer between
placements.

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

## Compare plans

Compare, when available, the ILP proxy, the reordered forward-plus-backward
critical path, and measured iteration time. ILP-versus-reordered disagreement
indicates overlap, bucketing, or scheduling effects; reordered-versus-measured
disagreement points to compiler, kernel, topology, or cost-model error. Prefer
measurement for close candidates.

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

- `local_map` exposes a boundary but does not price or optimize internal
  communication.
- Dynamic-shape costs use traced shape hints unless the estimator models the
  range explicitly.
- Parameter-memory bounds and the achieved fraction cover persistent parameter
  bytes only, not optimizer state, activation peaks, or transient
  materialization buffers.
- Placement JSON is smaller than full optimizer state but still requires a
  matching graph and mesh.
