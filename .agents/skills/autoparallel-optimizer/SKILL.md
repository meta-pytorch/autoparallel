---
name: autoparallel-optimizer
description: Optimize, inspect, compare, and validate PyTorch model sharding with AutoParallel, including ILP placement search and post-reordering forward/backward cost and memory estimates. Use for AutoParallel planning or performance investigations in an AutoParallel checkout; do not use for generic PyTorch tuning unrelated to AutoParallel.
---

# AutoParallel Optimizer

Use AutoParallel's existing APIs as the planner and its compiler scheduling
passes as the evaluator. Do not add a public plan or scoring API to automate
this workflow.

## Choose the requested depth

- **Explain:** inspect an existing optimizer or solution; do not capture,
  compile, or benchmark unless the question requires it.
- **Plan:** capture, constrain, solve, inspect, and export a placement.
- **Evaluate:** lower a plan and collect reordered forward/backward estimates.
- **Validate or measure:** run correctness checks or hardware benchmarks.

Stop at the shallowest depth that answers the user.

## Preconditions

- AutoParallel traces global logical inputs and runs the lowered module on
  local per-rank inputs. A successful trace does not prove the model expresses
  the intended global computation; audit it first.
- Capture on a CUDA machine, with the model constructed under
  `torch.device("meta")`. A CPU-only machine can load a saved optimizer to
  inspect or re-solve it.
- Reuse the user's `DeviceMesh`. Otherwise establish the intended allocation
  (visible devices do not prove it) and propose a topology-aligned mesh of at
  most two dimensions as a hypothesis.
- A final multi-node plan needs a detected or explicit NCCL topology.
- Do not rewrite the model, force a placement, or touch unrelated worktree
  changes without authorization.

Read [workflow.md](references/workflow.md) before planning, exploring, or
evaluating, and [interpreting-results.md](references/interpreting-results.md)
before comparing costs. Read
[global-spmd-modeling.md](references/global-spmd-modeling.md) when the forward
may assume process-local execution,
[mesh-selection.md](references/mesh-selection.md) when no mesh was supplied, and
[full-shard-planning.md](references/full-shard-planning.md) for a near
`1 / world_size` memory bound, a prefetch discount, or TP on a slow mesh axis.

## Method

1. Start from the closest working example and audit global SPMD semantics. For
   a mismatch, emit an `SPMD_MODELING_ASSUMPTION` warning and ask before a
   semantics-changing rewrite.
2. Record the allocation, topology, mesh, global batch, effective DP degree,
   local batch, dtypes, and compiler settings.
3. Capture once, add only intended constraints, and call `optimize_placement()`.
   A `None` boundary leaf means `Shard(0)`, not unconstrained, and
   `add_parameter_memory_constraint()` without arguments is a `1 / world_size`
   bound, not a neutral one.
4. Save placements and `get_json()`, then explore counterfactuals on the stable
   graph with node constraints, `resolve()`, and `diff_solutions()`.
5. Evaluate with `scripts/collect_reordered_metrics.py`: keep its cost model
   from planning through compilation, compile with
   `autoparallel_backend(overlap_scheduling=False)`, and run forward and
   backward inside its context.
6. Smoke-test execution, compare with an unsharded reference when correctness
   matters, and time hardware only when requested or needed.

## Evidence rules

- Keep the ILP objective, reordered estimates, and measurements separate, and
  report the highest evidence level reached (`interpreting-results.md`).
- Compare candidates only under the same workload, mesh, estimator, activation
  checkpointing, and reordering configuration.
- Treat a prefetch discount as a sensitivity assumption and `local_map` regions
  as unpriced.

## Failure handling

- Capture failure: reduce to the smallest failing submodule and keep the
  traceback; do not hide it behind `local_map`.
- Infeasibility: read the violated-constraint log, remove temporary node
  constraints, then relax output or memory constraints one at a time and report
  which change restored feasibility.
- Unexpected replication: check the memory constraint, the discount, and
  whether sharded alternatives existed (`interpreting-results.md`).
- A successful solve is not lowering evidence: report failures from custom ops,
  uneven shards, tied parameters, initialization, or compilation at the stage
  where they occurred.
- See `docs/troubleshooting.md` for established procedures.

## Output

Return what the depth established: the observation and supporting node data for
Explain; mesh, constraints, ILP summary, placements, SPMD audit, and artifact
paths for Plan; separate forward/backward metrics and cost-model status for
Evaluate; and exactly what compiled, ran, was compared, or was timed for
Validate. Say `estimated` for anything not measured on target hardware. For a
provisional full-shard result, return the competing candidates with the
confidence card from `full-shard-planning.md`.
