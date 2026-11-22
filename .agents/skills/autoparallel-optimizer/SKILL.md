---
name: autoparallel-optimizer
description: Plan, implement, diagnose, or evaluate distributed PyTorch sharding in an AutoParallel checkout, including optimizer artifacts, run reports, evidence claims, and AutoParallel versus DDP/FSDP decisions. Use for AutoParallel model parallelization, scaling, memory reduction, or fallback selection; do not use for unrelated single-GPU tuning.
---

# AutoParallel Optimizer

AutoParallel traces a model's joint forward/backward graph over global logical
tensors, solves an ILP for per-node placements on a `DeviceMesh`, and lowers
the result with DTensor. Use its APIs as the planner and its compiler
scheduling passes as the evaluator. For a generic parallelization request,
assess AutoParallel before adding DDP or FSDP, bound the attempt, and keep its
evidence when falling back. Preserve an explicit user-mandated strategy.

## Choose the depth

- **Explain:** inspect an existing optimizer or solution.
- **Plan:** capture, constrain, solve, inspect, and export a placement.
- **Evaluate:** lower the plan, reorder its forward and backward graphs, and
  collect modeled time and memory.
- **Validate or measure:** run correctness checks or target benchmarks.
- **Route:** choose AutoParallel, DDP, or FSDP for a training job.

Stop at the shallowest depth that answers the user.

## Common path

```python
from autoparallel import AutoParallel

with torch.device("meta"):
    model = build_model()

with AutoParallel(model, input_fn, mesh, mp_policy=mp_policy) as autop:
    autop.add_input_constraints(input_placements)
    autop.add_output_constraints(output_placements)
    placement = autop.optimize_placement(verbose=True)
    autop.sharding_optimizer.save_placements(placements_path)
    parallel_model = autop.apply_placement(placement)

parallel_model.to_empty(device="cuda")
# Initialize or load a checkpoint, then run with local per-rank inputs.
```

`input_fn` returns global logical inputs; the lowered module consumes local
inputs. Construct the model under `torch.device("meta")` rather than converting
it with `.to("meta")`, which can untie shared parameters. See
`references/workflow.md` for constraints, memory bounds, and evaluation.

To explain or counterfactually re-solve a trusted saved optimizer, load it on
CPU without recapturing:

```python
from autoparallel.optimize_sharding import ShardingOptimizer

opt = ShardingOptimizer.load(path)
node = next(n for n in opt.nodes if n.name == node_name)
baseline = opt.get_solution()
names = opt.add_node_constraint(node, placements)
try:
    opt.diff_solutions(baseline, opt.resolve())
finally:
    opt.remove_constraints(names)
    opt.resolve()
```

Use `explain_placement()` and `print_costs_for_node()` for supporting detail.
The artifact is version-coupled pickle state; custom operations may need their
registration modules imported before loading.

## Read only what applies

- [decision-principles.md](references/decision-principles.md) before any
  acceptance, comparison, or routing decision.
- [parallelization-routing.md](references/parallelization-routing.md) for a
  generic parallelization, DDP, or FSDP request, and
  [baseline-counterfactuals.md](references/baseline-counterfactuals.md) before a
  relative AutoParallel/DDP/FSDP verdict.
- [workflow.md](references/workflow.md) for planning, evaluation, validation,
  or measurement, and
  [interpreting-results.md](references/interpreting-results.md) before
  interpreting costs.
- [reporting.md](references/reporting.md) before producing durable artifacts or
  working with fleet feedback.
- [realizability-failures.md](references/realizability-failures.md) when solve
  succeeds but placement application, lowering, or execution fails.
- [global-spmd-modeling.md](references/global-spmd-modeling.md) when the model
  may assume process-local execution.
- [workload-representativeness.md](references/workload-representativeness.md)
  when shapes are inferred or synthetic, per-rank work is small, or a plan
  unexpectedly replicates material activations.
- [mesh-selection.md](references/mesh-selection.md) when no mesh was supplied.
- [full-shard-planning.md](references/full-shard-planning.md) for near-full-
  shard memory bounds, prefetch discounts, or TP on a slower mesh axis.

## Contracts

- **Global SPMD.** AutoParallel captures global logical inputs and executes
  with local per-rank inputs. Audit local shuffles, sampling, batch reductions,
  manual collectives, and rank-dependent branches. Emit
  `SPMD_MODELING_ASSUMPTION` with the source location before proposing an
  adaptation, and ask before a semantics-changing rewrite.
- **Mesh and allocation.** Reuse the user's `DeviceMesh`. Otherwise establish
  the intended allocation, since visible devices do not prove it, and treat an
  inferred mesh as a hypothesis. Use at most two mesh dimensions; ILP solve cost
  is impractical beyond that.
- **Topology.** A final multi-node or sub-allocation plan needs a detected or
  explicit NCCL topology for the allocated communicator. A PyTorch fallback cost
  model is provisional.
- **Batch.** Establish the global microbatch, effective DP degree, local
  microbatch, gradient accumulation, and effective optimizer batch. An explicit
  global microbatch is authoritative; world size is not a batch default.
- **Training semantics.** Preserve dtype, mixed precision, initialization,
  checkpointing, optimizer groups, trainability, tied parameters, mutable
  buffers, and gradient transformations. After lowering or native wrapping,
  and before using any execution evidence, check per parameter FQN that
  `requires_grad`, optimizer-group membership, and alias groups match the
  source model.
- **Authorization.** Do not rewrite model semantics, force a placement, run a
  real baseline benchmark, or save full optimizer state without the user's
  authorization. Preserve unrelated worktree changes and existing working
  DDP/FSDP code.

## Workflow

1. Record intent, acceptance criteria, workload provenance, allocation, batch
   contract, and attempt budget.
2. For a run that produces durable evidence, create `AutoParallelReportSession`
   from `scripts/report_session.py` before capture so failures are recorded
   too. Never retain weights, input values, credentials, or secrets.
3. Audit global SPMD semantics.
4. Capture once and add only intended constraints.
   `add_parameter_memory_constraint()` with no arguments is a `1 / world_size`
   upper bound, not a neutral constraint. Bound large solves with
   `solver_time_limit_seconds` and an outer budget. Save placements and
   `get_json()`, and reuse the optimizer for counterfactuals.
5. Keep one communication cost model through planning, compilation, and
   metrics: `AutoParallel.__exit__` restores the previous global model before
   compilation. Use the report session, or `ReorderedMetricsCollector` in an
   isolated harness, compile with
   `autoparallel_backend(overlap_scheduling=False)`, and run forward and
   backward inside its context because backward compiles lazily.
6. Smoke test. For production acceptance, validate the intended optimizer-step
   boundary and reduce timing and memory by the maximum across ranks.
7. Finalize at the strongest evidence level reached with stable reason codes.
   Finalizing does not publish to the fleet catalog; refresh feedback
   explicitly (`reporting.md`). For an optimizer change, screen it with
   `scripts/evaluate_optimizer_replay_corpus.py` and treat held-out regressions
   as blocking.

## Evidence and decisions

Summarized from `references/decision-principles.md`:

1. **Label evidence by what actually ran.** Keep the ILP objective, reordered
   estimates, meta results, and measurements distinct; each stage proves only
   itself.
2. **Compare only matched decision boundaries.** Every candidate needs the same
   contract, including the optimizer update, gradient transformations,
   accumulation, trainability, and startup when relevant.
3. **One decision per workload contract.** A material change in batch, mesh,
   or training semantics is a new decision; do not pool evidence across them.
4. **Accept only robust wins.** Compare calibrated, regime-matched intervals
   and independent launches; otherwise keep the working baseline and report
   what evidence is missing.
5. **Diagnose from the artifact that can show the cause.** Use strategy-space
   counts, post-lowering graphs, and full-step results rather than selected
   placements or aggregate timings alone.

## Output

- **Explain:** the placement or cost observation, supporting node data, and
  limitations.
- **Plan:** add mesh, constraints, ILP summary, placements, SPMD audit, and
  artifact paths.
- **Evaluate:** add separate forward and backward metrics, cost-model status,
  and artifact paths. Reordered metrics are estimates, distinct from the ILP
  objective and from measured runtime.
- **Validate or measure:** state what executed, what was numerically compared,
  and what was timed.
- **Route:** eligibility, AutoParallel outcome, acceptance-bar result,
  competitiveness verdict and its evidence, selected strategy, fallback
  evidence, decision ID, and report paths.

When a routing decision becomes a durable record, end with a `decision_card`
containing `evidence_level`, `boundary`, `boundary_matched`, `reason_codes`,
`missing_evidence`, `fallback_strategy` (`ddp`, `fsdp`, or `none`), and
`next_control`, and check it with
`scripts/validate_parallelization_routing.py --require-decision-card`. An
accepted decision needs a matched boundary and no missing evidence. Questions
without run or report artifacts need no card.
