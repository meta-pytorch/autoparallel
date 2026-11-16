---
name: autoparallel-optimizer
description: Optimize, inspect, compare, and validate PyTorch model sharding with AutoParallel, including ILP placement search and post-reordering forward/backward cost and memory estimates. Use for AutoParallel planning or performance investigations in an AutoParallel checkout; do not use for generic PyTorch tuning unrelated to AutoParallel.
---

# AutoParallel Optimizer

Use AutoParallel's existing APIs as the planner and its compiler scheduling
passes as the evaluator. Do not add a public plan or scoring API to automate
this workflow.

## Choose the requested depth

- **Explain:** inspect an existing optimizer or solution, without capturing,
  compiling, or benchmarking unless the question requires it.
- **Plan:** capture, constrain, solve, inspect, and export a placement.
- **Evaluate:** lower a plan, reorder its forward and backward graphs, and
  collect modeled time and memory.
- **Validate or measure:** run correctness checks or hardware benchmarks.

Stop at the shallowest depth that answers the user.

## Preconditions

- Capture the joint graph on a CUDA machine. A CPU-only machine may load a
  GPU-captured `ShardingOptimizer.save()` artifact to inspect or re-solve it;
  `explain_placement()` currently needs a live optimizer.
- AutoParallel requires global SPMD semantics: it traces global logical inputs,
  and the lowered module runs on local per-rank inputs. A successful trace does
  not prove the model expresses the intended global computation.
- Construct the model under `torch.device("meta")`; `.to("meta")` on a
  materialized model can untie shared parameters.
- Reuse the user's `DeviceMesh`. Otherwise establish the intended allocation,
  since visible devices do not prove it, and propose a topology-aligned 1D or
  2D mesh as a hypothesis. More than two dimensions makes the ILP impractical.
- A final multi-node plan needs a detected or explicit NCCL topology; the
  PyTorch fallback cost model is provisional.
- Preserve dtype, mixed precision, initialization, and unrelated worktree
  changes. Do not rewrite the model or force a placement without authorization.

Read [workflow.md](references/workflow.md) before planning or evaluation and
[interpreting-results.md](references/interpreting-results.md) before comparing
costs. Read [global-spmd-modeling.md](references/global-spmd-modeling.md) when
the forward may assume process-local execution,
[mesh-selection.md](references/mesh-selection.md) when no mesh was supplied, and
[full-shard-planning.md](references/full-shard-planning.md) for a near
`1 / world_size` memory bound, a prefetch discount, or TP on a slow mesh axis.

## Method

1. Start from the closest working example or entry point and audit global SPMD
   semantics. For a mismatch, emit an `SPMD_MODELING_ASSUMPTION` warning with
   the source location and a proposed global formulation, and ask before a
   semantics-changing rewrite.
2. Record the allocation source, topology, mesh shape and names, global batch,
   effective DP degree, local batch, dtypes, and compiler settings.
3. Capture once, add only intended constraints, and call
   `optimize_placement()`. A `None` boundary leaf means the default `Shard(0)`;
   use `UNCONSTRAINED` or omit the call to let the solver choose.
   `add_parameter_memory_constraint()` with no arguments is a byte-weighted
   `1 / world_size` upper bound, not a neutral constraint. Pass
   `solver_time_limit_seconds` to bound large solves.
4. Save placements and `get_json()` before experimenting. Explore with
   `add_node_constraint()`, `resolve()`, `diff_solutions()`, and
   `remove_constraints()` on the stable graph; retrace only when graph-producing
   inputs or configuration change.
5. Evaluate with `scripts/collect_reordered_metrics.py`. Keep one cost model
   from planning through compilation, compile with
   `autoparallel_backend(overlap_scheduling=False)`, and run forward and
   backward inside the collector context.
6. Smoke-test execution. Compare outputs and gradients with an unsharded
   reference when correctness matters. Time hardware only when requested or
   needed to separate close candidates.

## Evidence rules

- Keep the ILP objective, reordered estimates, and measured runtime separate;
  the evidence ladder is in `interpreting-results.md`.
- Compare candidates only under the same workload, mesh, estimator, activation
  checkpointing, and reordering configuration.
- Report forward and backward peak memory separately. The optimizer update is
  absent unless captured and evaluated separately.
- A prefetch discount is a sensitivity assumption; a zero discount is an
  optimistic endpoint, not a prediction.
- Treat `local_map` regions as opaque unless they have an explicit cost or
  benchmark.

## Failure handling

- On capture failure, reduce to the smallest failing submodule and keep the
  original traceback. Do not hide it behind `local_map`.
- On infeasibility, read the violated-constraint log, remove temporary node
  constraints, then relax output or memory constraints one at a time and report
  which change restored feasibility.
- On unexpected replication, check the memory constraint, the discount, and
  whether sharded alternatives existed (`interpreting-results.md`).
- A successful solve is not lowering evidence. Report failures from custom
  operations, uneven shards, tied parameters, initialization, or compilation at
  the stage where they occurred.
- On metric failure, check that both phases compiled and ran inside the
  collector and that planning and evaluation shared one topology.
- Consult `docs/troubleshooting.md` for established procedures.

## Output

For **Explain**, return the observation, supporting node data, and limitations.
For **Plan**, add the mesh, constraints, ILP summary, placements, SPMD audit,
and artifact paths. For **Evaluate**, add separate forward/backward metrics and
the cost-model status. For **Validate or measure**, state exactly what compiled,
executed, was numerically compared, or was timed; say `estimated` otherwise.
For a provisional full-shard result, return the competing candidates and the
confidence card from `full-shard-planning.md`.
