---
name: autoparallel-optimizer
description: Optimize, inspect, compare, and validate PyTorch model sharding with AutoParallel, including ILP placement search and post-reordering forward/backward cost and memory estimates. Use for AutoParallel planning or performance investigations in an AutoParallel checkout; do not use for generic PyTorch tuning unrelated to AutoParallel.
---

# AutoParallel Optimizer

Use AutoParallel's existing APIs as the planner and compiler scheduling passes as
the refined evaluator. Do not add a public plan or scoring API to automate this
workflow.

## Choose the requested depth

- **Explain:** inspect an existing optimizer or solution. Do not capture,
  compile, or benchmark unless the question requires it.
- **Plan:** capture, constrain, solve, inspect, and export a placement.
- **Evaluate:** lower a plan, reorder its partitioned forward and backward
  graphs, and collect modeled time and memory.
- **Validate or measure:** execute correctness checks or hardware benchmarks.

Stop at the shallowest mode that answers the user. A question about one node
does not require a full evaluation report.

## Preconditions

- Capture the joint training graph in a CUDA-enabled PyTorch environment. A
  CPU-only machine may load a GPU-captured `ShardingOptimizer.save()` artifact
  and inspect or re-solve it, but should not recapture the graph.
- AutoParallel requires global SPMD semantics: tracing sees the global logical
  inputs and sharding creates per-rank execution. Audit local-batch shuffles,
  sampling, batch reductions, manual collectives, and rank-dependent branches.
  A successful trace does not prove these semantics.
- Reuse the user's `DeviceMesh`. If absent, infer a topology-aligned 1D or 2D
  mesh and label it as a hypothesis. Do not use more than two dimensions;
  AutoParallel's ILP solve time currently grows impractically.
- Trace with global logical input shapes and execute the lowered module with
  local per-rank inputs. Preserve the model's dtype, mixed-precision, launch,
  and initialization conventions.
- Preserve unrelated worktree changes. Do not rewrite the model or force a
  placement unless the user authorized optimization changes.

Read [references/workflow.md](references/workflow.md) before planning or
evaluation and [references/interpreting-results.md](references/interpreting-results.md)
before comparing costs. Read
[references/global-spmd-modeling.md](references/global-spmd-modeling.md) when
the forward may assume process-local execution. Read
[references/mesh-selection.md](references/mesh-selection.md) when no mesh was
provided.

## Method

1. Start from the closest working example or training entry point. Audit global
   SPMD semantics before treating a plan as meaningful.
2. If an adaptation is needed, emit an `SPMD_MODELING_ASSUMPTION` warning with
   the source location, mismatch, and proposed global formulation. Ask before a
   non-trivial or semantics-changing rewrite.
3. Select the supplied mesh or evaluate a bounded 1D/2D candidate set. Record
   GPU count, physical topology, mesh shape and names, workload shapes, dtypes,
   and compiler settings.
4. Capture once, add only intended boundary and memory constraints, and call
   `optimize_placement()`. This is the plan operation. Note that
   `add_parameter_memory_constraint()` with no arguments sets the upper bound to
   `1 / world_size`; it is not a neutral constraint.
5. Save placements, `get_json()`, and optionally the full optimizer before
   experimentation. Explore the stable joint graph by adding node constraints,
   calling `resolve()` and `diff_solutions()`, then removing the temporary
   constraints. Retrace only when graph-producing inputs or configuration
   change.
6. For evaluation, preserve one communication cost model from planning through
   compile and graph estimation. Use
   `scripts/collect_reordered_metrics.py`; record when NCCL detection returns
   `None` and the PyTorch fallback is used.
7. Apply the plan, compile with
   `autoparallel_backend(overlap_scheduling=False)`, and execute forward and
   backward inside the metrics-collector context. The collector's ATen pass must
   be the only overlap scheduler. Require independent metrics for both phases;
   retain every record when specialization recompiles a graph.
8. Run an execution smoke test. Compare outputs and gradients with an unsharded
   reference when correctness is at issue. Time hardware only when requested or
   needed to distinguish close candidates.

## Evidence rules

- Keep ILP objective, reordered estimates, and measured runtime separate. The
  interpretation and claim ladder are in `references/interpreting-results.md`.
- Compare candidates only with the same workload, mesh, estimator, activation
  checkpointing, and reordering configuration. Show both rankings if ILP and
  reordered estimates disagree.
- Report forward and backward peak memory separately. Do not count the optimizer
  update unless it was separately captured and evaluated.
- Treat `local_map` regions as opaque unless they have an explicit cost or
  benchmark.

## Failure handling

- On capture failure, reduce to the smallest failing submodule and preserve the
  original traceback. Do not automatically hide it behind `local_map`.
- On infeasibility, remove temporary node constraints first, then relax output
  or memory constraints one at a time and report what restored feasibility.
- On unexpected replication, inspect the parameter-memory constraint and the
  communication discount configuration.
- On metric failure, verify that compilation produced partitioned graphs, that
  forward and backward both ran, and that planning and evaluation shared the
  same topology configuration.
- Consult `docs/troubleshooting.md` for established repository procedures.

## Output

For **Explain**, return the placement or cost observation, supporting node data,
and relevant limitations. For **Plan**, add mesh selection, constraints, ILP
summary, placements, SPMD audit, and artifact paths. For **Evaluate**, add
separate forward/backward `GraphMetrics` and the cost-model status. For
**Validate or measure**, state exactly what compiled, executed, was numerically
compared, or was timed. Say `estimated` unless a target-hardware run was timed.
