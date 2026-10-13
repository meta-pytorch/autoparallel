---
name: autoparallel-optimizer
description: Plan, implement, diagnose, or evaluate distributed PyTorch sharding in an AutoParallel checkout, including optimizer artifacts, run reports, evidence claims, and AutoParallel versus DDP/FSDP decisions. Use for AutoParallel model parallelization, scaling, memory reduction, or fallback selection; do not use for unrelated single-GPU tuning.
---

# AutoParallel Optimizer

Use AutoParallel's existing APIs as the planner and its compiler scheduling
passes as the refined evaluator. For a generic parallelization request, assess
AutoParallel before directly adding DDP or FSDP. Bound the attempt, evaluate it
against an explicit bar, and retain any fallback evidence. Preserve an explicit
user-mandated strategy.

## Choose the requested depth

- **Route:** assess eligibility, make a bounded attempt when eligible, and
  select AutoParallel, DDP, or FSDP.
- **Explain:** inspect an existing optimizer or solution without capturing,
  compiling, or benchmarking unless the question requires it.
- **Plan:** capture, constrain, solve, inspect, and export a placement.
- **Evaluate:** lower a plan, reorder partitioned forward and backward graphs,
  and collect modeled time and memory.
- **Validate or measure:** execute correctness checks or target benchmarks.

Stop at the shallowest mode that answers the user.

## Exact shallow workflows

For a trusted saved optimizer, do not recapture or initialize distributed
state. Load and inspect it on CPU with the concrete APIs:

```python
from autoparallel.optimize_sharding import ShardingOptimizer

opt = ShardingOptimizer.load("artifacts/model.ap")
node = next(node for node in opt.nodes if node.name == "mm_3")
original = opt.get_solution()
names = opt.add_node_constraint(node, target_placements)
counterfactual = opt.resolve()
opt.diff_solutions(original, counterfactual)
opt.remove_constraints(names)
opt.resolve()
```

Use a `try/finally` around the constraint loop in real code. Loading uses
trusted, version-coupled pickle state; non-ATen custom operations may require
their registration modules to be imported first. This workflow is explain-only
evidence: it does not lower, compile, execute, or modify the artifact.

When a finalized run is absent from fleet feedback, preserve the immutable run
bundle and name the exact one-shot command
`python scripts/refresh_report_feedback.py <report-root>`, or use that command
with a cluster-supervised `--watch` process. Finalization never performs this
refresh. The refresh owns a non-blocking lock, builds `catalog-v1.json` and both
improvement-queue products in memory, publishes them atomically, and writes
`feedback-status-v1.json` last. Verify a successful, non-stale status, then the
run's presence in the rebuilt catalog and queue inputs; report
`skipped_locked`, degraded, or failed refreshes rather than editing the bundle.

## Read only what applies

For every generic parallelization, DDP, or FSDP request, read
[parallelization-routing.md](references/parallelization-routing.md). Before a
relative AutoParallel/DDP/FSDP decision, also read
[baseline-counterfactuals.md](references/baseline-counterfactuals.md).

For planning, evaluation, validation, or measurement, read
[workflow.md](references/workflow.md). Before interpreting costs, read
[interpreting-results.md](references/interpreting-results.md). Before producing
durable artifacts, read [reporting.md](references/reporting.md).

For any diagnosis where solve succeeds but placement application, lowering, or
execution fails, or where aliases or repeated-subgraph clustering are involved,
read [realizability-failures.md](references/realizability-failures.md), including
in Explain mode.

Read the following references only when their condition applies:

- [global-spmd-modeling.md](references/global-spmd-modeling.md) when the model
  may assume process-local execution;
- [workload-representativeness.md](references/workload-representativeness.md)
  when shapes are inferred or synthetic, per-rank work is underfilled, or a
  plan unexpectedly replicates material activations;
- [mesh-selection.md](references/mesh-selection.md) when no mesh was supplied;
- [full-shard-planning.md](references/full-shard-planning.md) for near-full-
  shard memory bounds, material full sharding, prefetch discounts, or TP on a
  slower mesh axis.

For a training route involving checkpointing, optimizer groups, frozen or tied
parameters, gradient clipping or accumulation, optimizer-state memory, or
time-to-ready, use the matching training-contract guidance in `workflow.md`
even when no plan will be built.

## Non-negotiable contracts

- AutoParallel uses global SPMD semantics: capture global logical inputs and
  execute lowered code with local per-rank inputs. Audit local shuffles,
  sampling, reductions, manual collectives, and rank-dependent branches.
- Reuse the user's `DeviceMesh`. Otherwise establish the intended allocation
  before proposing a topology-aligned 1D or 2D mesh. Visible devices do not
  prove world size, and meshes above two dimensions are currently impractical
  for the ILP.
- Require detected or explicit NCCL topology for a final multi-node plan.
  Treat a PyTorch fallback cost model as provisional.
- Establish global microbatch, effective DP degree, local microbatch, gradient
  accumulation, and effective optimizer batch. World size is not a batch-size
  default. A synthetic workload is smoke evidence, not a production decision.
- Preserve dtype, mixed precision, launch, initialization, checkpointing,
  optimizer, trainability, aliasing, and gradient-transformation semantics.
- Before using training execution or timing evidence, assert after lowering or
  native wrapping that every parameter FQN preserves `requires_grad`, optimizer
  group membership, and tied-parameter aliases. A mismatch is a semantic
  failure, not performance evidence. Use a qualified fallback and require a
  fresh AutoParallel evaluation under the exact original contract after the
  invariant is fixed. In the response, call this a post-lowering or
  post-wrapping per-FQN invariant check, not merely a requirement to preserve
  the mask.
- Do not rewrite model semantics, force a placement, run a real baseline
  benchmark, or save full optimizer state without the user's authorization.
- Preserve unrelated worktree changes and existing working DDP/FSDP code.

## Core workflow

1. Record intent, acceptance criteria, workload provenance, allocation, batch
   contract, and attempt budget. For routing, follow
   `parallelization-routing.md` and keep one decision ID per materially distinct
   workload contract.
2. For an applying run that produces durable evidence, create
   `AutoParallelReportSession` before capture so failures and successes reach
   the configured report root. Record sanitized launch and software provenance;
   never retain weights, input values, credentials, or secrets.
3. Audit global SPMD semantics. Emit `SPMD_MODELING_ASSUMPTION` with source
   context before proposing an adaptation, and ask before any non-trivial or
   semantics-changing rewrite.
4. Capture once, mark substages, add only intended constraints, and call
   `optimize_placement()`. A parameter-memory constraint with no arguments is a
   `1 / world_size` bound, not a neutral constraint. Save placements and
   `get_json()`; reuse the stable optimizer for bounded counterfactuals.
5. Preserve one communication cost model through planning, lowering, compile,
   and graph estimation. Use `scripts/report_session.py` for durable evaluation
   or its lower-level `ReorderedMetricsCollector` for an isolated harness. Keep
   that collector as the only overlap scheduler; do not reimplement its pass.
6. Apply and compile the plan with
   `autoparallel_backend(overlap_scheduling=False)`. Require independent
   forward and backward metrics and retain specialization recompiles. In an
   isolated harness, call `collector.require_forward_backward()` and
   `collector.write_json(metrics_path)`; printed JSON is not a durable artifact.
   Accept an explicit output path and leave no backup or scratch file in the
   workspace. In the handoff, distinguish these estimates from the ILP
   objective and state that the pass adds no control dependencies, so later
   Inductor scheduling or fusion may move the modeled order. After editing an
   isolated harness, run
   `scripts/validate_evaluation_harness.py PATH_TO_HARNESS`; fix every reported
   invariant and include every returned `handoff_requirements` statement in the
   final response.
7. Execute a smoke test. For production training acceptance, validate the
   intended optimizer-step memory and performance boundary, not only
   forward/loss/backward. Reduce timing and memory by the maximum across ranks.
8. Finalize at the strongest evidence level reached. Record stable outcome,
   competitiveness, fallback, and decision reason codes. Refresh fleet
   feedback explicitly at campaign handoff; finalizing a bundle does not make
   it catalog-visible. For an optimizer change, evaluate replayable queue
   representatives with `scripts/evaluate_optimizer_replay_corpus.py`, retain
   the deterministic training/held-out split, and reject held-out regressions
   before selecting target validation cases.

## Evidence and failure guardrails

- Keep ILP objective, reordered estimate, meta/fake-tensor result, and measured
  target runtime distinct. Meta runtime is ranking evidence unless a matched
  held-out error envelope makes the decision robust. For a meta result, report
  `valid_meta_uses: legality/feasibility checking, candidate ranking,
  scale/memory exploration, benchmark prioritization`; these uses do not prove
  numerical correctness, compilation, or target runtime. Unless a matched
  conservative envelope preserves the decision, also record `verdict:
  inconclusive` rather than implying that retaining the fallback is a measured
  AutoParallel loss.
- Compare candidates only under the same workload, mesh, estimator,
  checkpointing, optimizer and gradient semantics, and measured step boundary.
- Include or conservatively project optimizer state before accepting training
  memory. Include measured or calibrated optimizer-update cost before making a
  production throughput claim.
- Keep time-to-ready separate from steady-state latency and amortize it when
  the intended run length is known. A finite-job total computed from measured
  startup and step components is still projected: label that in prose and in
  the decision-card boundary rather than implying the total ran end to end.
  Do not infer execution provenance from supplied timing numbers. Only use
  `executed_on_target` when those components explicitly came from target
  execution, and do not use `measured` unless the finite-job boundary ran end
  to end. When every candidate passes the acceptance bar but startup is not
  amortized, use `STARTUP_NOT_AMORTIZED`, not `PERFORMANCE_BAR_MISSED`.
  Require three independent launches per strategy for measured relative
  acceptance. A close single-launch comparison remains inconclusive even with
  tight within-launch samples: those samples do not cover process launch,
  compilation, allocator, or system-state variance. Call the result
  schedule-sensitive or inconclusive, record
  `MEASURED_LAUNCH_VARIANCE_UNMODELED`, and request matched independent-launch
  distributions for both strategies.
- Do not accept a Pareto-dominated plan merely because it falls inside a
  latency tolerance. Preserve an existing baseline when comparison evidence is
  unavailable, unmatched, or schedule-sensitive.
- Bound large CBC searches with `solver_time_limit_seconds` and an outer
  end-to-end budget. A timed-out feasible incumbent is not an optimal plan.
- A successful solve is not lowering or execution evidence. Preserve capture,
  lowering, compiler, metric, and semantic failures rather than hiding them
  behind `local_map` or a native fallback.
- For a strategy-generation coverage failure, reuse an existing operator rule
  only when its schema and in-contract semantics are equivalent; otherwise
  start with an explicitly allowlisted replicate-only strategy. Reject invalid
  input options candidate-locally rather than aborting the whole node, account
  for redistribution, and remap output placements after removed dimensions.
  On both 1D and 2D meshes, lower every emitted strategy before executing
  output-structure, shape, value, and gradient checks against the native
  reference. Strategy generation alone is only planning coverage.
- Diagnose unexpected replication, communication, and bucketing from the
  post-lowering graph and applicable reference procedure; collective count or
  a single-rank timing alone is not proof of the cause.
- When comparing optimizer revisions, rescore the recorded placement and the
  new placement under one evaluator. Raw objectives from different revisions
  are not directly comparable when strategy or cost code changed. Treat replay
  results as optimizer-screening evidence; they do not establish lowering,
  numerical correctness, or target runtime.

Use these exact guards for recurring close decisions:

- An explicit global microbatch is authoritative. Check divisibility, capture
  that global logical shape, derive `local = global / effective_dp_degree`, and
  verify the executed local shape. Write the divisibility assertion explicitly,
  such as `global_batch % effective_dp_degree == 0`; consistent arithmetic does
  not leave the check implicit. A different batch is a separate sensitivity
  configuration, never a substitute.
- A world-size or mesh change needs a new decision ID: it changes local batch,
  collective group sizes and scale, and potentially the optimized placement.
  Exact retries and baselines on one mesh share that mesh's ID. For a GPU
  sub-allocation that canonical topology detection cannot represent, require
  an explicit NCCL topology for the allocated communicator; do not use a
  fallback cost model or unused visible GPUs as final routing evidence. State
  all three changed quantities—local batch, collective scale, and potential
  selected placement—and say plainly that timing evidence cannot be pooled.
- Label a mesh inferred from allocation and topology as an optimization
  hypothesis and provisional starting point. Explicitly exclude dimensions
  above two because current ILP solve cost is impractical, not as a generic
  preference.
- An underfilled synthetic run proves smoke coverage only. Separate
  replicate-only or implicit-fallback option sets from replicated selections
  that had sharded alternatives; include the explicit conclusion **world size
  does not define global batch**, and request the production batch before
  routing. Do not leave that conclusion implied by calling the input synthetic.
- A forward/loss/backward comparison cannot establish full-step throughput
  when optimizer work is absent. Parameter placement changes local update
  bytes, DTensor dispatch, and the realized fused/foreach kernel family.
  Preserve the working fallback and record `OPTIMIZER_PERFORMANCE_UNMODELED`
  or `STEP_BOUNDARY_MISMATCH` until every strategy has the same measured update
  or a versioned, prospectively held-out, regime-matched calibrated envelope.
  Before applying a frozen envelope, explicitly report all three calibration
  regime groups: **platform** (hardware, framework, dtype policy, realized
  kernel family), **optimizer contract** (groups, update frequency,
  accumulation, state placement, dispatch), and **scale** (calibrated local
  parameter-byte and tensor-count ranges). Do not leave supplied matches
  implicit. Widen or reject the envelope when any group does not match. A
  modeled-only comparison with no current candidate execution must not use
  `executed_on_target` or `measured` as its decision-card evidence level.
- When broad sharded-option coverage and a forward/backward win coexist with
  substantially higher local parameter bytes and a measured full-step loss,
  diagnose an objective-boundary miss rather than missing propagation. Select
  the measured full-step winner, record `OPTIMIZER_COST_MISSING_FROM_OBJECTIVE`,
  and retain the supplied numeric local/global parameter fractions plus every
  supplied partial- and full-step latency and memory result. When only result
  ordering is supplied, preserve that qualitative ordering and say that exact
  values were not supplied; never invent them. State both the partial-boundary
  ordering and the full-step ordering explicitly, including when the full-step
  winner is both faster and smaller. Do not infer which individual parameters
  stayed replicated without node data. Write the stable finding with four
  fields: reason code, parameter-storage comparison, partial-boundary ordering,
  and full-step latency plus memory ordering. Use explicit `partial_boundary:`
  and `full_boundary:` fields; do not substitute a description of what the
  objective favored for the observed partial-boundary winner.
- Gradient clipping is part of the workload contract. Preserve the global-norm
  algorithm, threshold, position between backward and the update, clipping and
  update frequency, local work, and placement-dependent reduction. Report
  `local_norm_work` and `placement_dependent_global_reduction` as separate
  boundary fields rather than letting "global norm" imply the local work. Give
  the clipped contract a new decision ID relative to an unclipped report, then
  use that same clipped ID for its exact retries and linked baselines. In the
  recommendation, state explicitly that both strategies use the same clipping
  frequency and optimizer-update frequency, normally once per completed
  accumulation cycle.
- When projecting missing optimizer state onto an otherwise measured partial
  step, retain separate labels: supplied latency and forward/backward peak are
  **measured**, optimizer-state additions are **analytical**, and the resulting
  complete-step peak is **projected**. Do not omit the measured latency label
  merely because routing is decided by memory.
- For a full-shard overlap envelope, explicitly name the undiscounted serial
  endpoint as pessimistic and zero-discount as optimistic and non-executable;
  save both independently, deduplicate placement families, and lower only
  distinct candidates with topology, AC, and autobucketing held fixed. Report
  the achieved byte-weighted parameter-storage fraction and distinguish its
  persistent storage placement from heavy-operation compute placement; do not
  collapse either field into total memory or a generic placement summary.
  Diagnose inter-node TP as possible topology inversion, and include effective
  DP degree and local batch before returning a provisional candidate set.
- For an uneven-shard realizability retry, close with all four checks named
  explicitly: placement application on both long and short rank-local extents,
  output-shape parity, output-value parity, and gradient parity against the
  unsharded reference. Generic "numerical checks" do not imply shape parity.
- For a trainability-mask failure, say explicitly that the per-FQN
  `requires_grad` and optimizer-group invariant is checked **after lowering or
  wrapping and before execution evidence is accepted**. A generic request to
  preserve or verify the mask is not the same gate.
- Treat strategy clustering as optional ILP compression. An unclustered retry
  can establish ILP feasibility but does not establish lowering, numerical
  correctness, or performance; state both facts explicitly. Record the labeled
  finding `clustering role: optional ILP compression` before interpreting the
  retry.
- For a composed full-step prediction, freeze and propagate uncertainty from
  both the partial-step base and optimizer increment. The first discriminating
  control is a contemporaneous independent-launch measurement of the same
  partial-step base, followed by recomposition; alternatively use a validated
  temporal-drift bound. A full-step benchmark can decide routing but does not
  isolate the source. Preserve the current fallback and record
  `COMPOSED_FULL_STEP_BASE_DRIFT` when the refreshed base explains the miss;
  name that conditional classification in the recommendation.
- With alternating DDP `no_sync()` and synchronized microsteps, use
  `static_graph=False` unless that exact template executed successfully.
  Qualify each FSDP wrapping template separately and retain its actual latency,
  complete-step memory, and startup costs.

Before finalizing structured routing fields, validate their consistency with
`scripts/validate_parallelization_routing.py --require-decision-card`. Evidence
supplied to that validator must come from the run; the validator does not
manufacture missing facts.

## Output

For **Explain**, return the placement or cost observation, supporting node
data, and limitations. For **Plan**, add mesh, constraints, ILP summary,
placements, SPMD audit, and artifact paths. For **Evaluate**, add separate
forward/backward metrics, cost-model status, and artifact paths. Label reordered
metrics as estimates distinct from the ILP objective; note that the pass adds
no overlap control dependencies, so later Inductor scheduling or fusion can
move execution order. For **Validate or measure**, state exactly what executed,
was numerically compared, or was timed; say `estimated` unless target hardware
was measured.

For **Route**, return eligibility and disposition, AutoParallel outcome,
acceptance-bar result, competitiveness verdict and evidence, benchmark action,
selected strategy, fallback evidence, decision ID, and report paths. For any
durable run, return its report path and highest evidence level. When reviewing
fleet feedback, include catalog freshness and relevant stable candidate IDs.
When routing rejects evidence for a training-semantic mismatch, explicitly name
the post-lowering or post-wrapping per-FQN and optimizer-group invariant that
must pass before the candidate is re-evaluated.

For Route, and for Explain when asked to accept or choose from real run evidence
that should become a durable routing record, always end with a compact
`decision_card` containing only `evidence_level`, the
measured or modeled `boundary`, `boundary_matched`, stable `reason_codes`,
explicit `missing_evidence`, `fallback_strategy`, and one `next_control`. Use an
evidence level from `none`, `captured`, `ilp_feasible`, `lowered`, `compiled`,
`numerically_checked`, `executed_on_target`, or `measured`; use fallback
`ddp`, `fsdp`, or `none`. Set `boundary_matched=false` when the production
decision boundary is missing even if a partial boundary was measured. An
accepted decision needs a matched boundary and no missing evidence. An
inconclusive decision needs non-empty missing evidence and a next control. Keep
weighted quality, hard safety/semantic gates, and evidence level separate.
A read-only hypothetical or arithmetic comparison with no run/report artifacts
should answer directly and omit the card; do not manufacture execution
provenance merely to populate structured fields.
