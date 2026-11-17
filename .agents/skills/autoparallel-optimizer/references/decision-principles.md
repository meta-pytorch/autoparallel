# Evidence and decision principles

These principles govern every claim, comparison, and routing decision. Apply
them to whatever boundary, optimizer, or measurement the task involves rather
than searching for a matching example.

## 1. Label evidence by what actually ran

The evidence ladder is `captured` → `ilp_feasible` → `lowered` → `compiled` →
`numerically_checked` → `executed_on_target` → `measured`; see
`interpreting-results.md` for definitions. Each level establishes only itself.

- The ILP objective, a reordered-graph estimate, a meta/fake-tensor result, and
  a target measurement answer different questions. Keep them separate.
- Every number carries its provenance: measured, analytical (derived from
  shapes and dtypes), modeled (a cost model or estimator), or projected (a
  combination). A value composed from measured and non-measured parts is
  projected. Supplied numbers do not reveal their provenance; use
  `executed_on_target` or `measured` only for components that ran on the target.
- A stage does not prove the next one. A feasible solve is not lowering.
  Strategy generation is not lowering or numerical correctness. A retry with
  an optional ILP compression disabled, such as repeated-subgraph clustering,
  shows feasibility, not equivalence or performance. Offline replay screens
  optimizer changes but says nothing about lowering, numerics, or runtime. A
  timed-out CBC incumbent is not an optimal plan.
- Meta/fake-tensor results support legality checks, candidate ranking,
  scale and memory exploration, and choosing which target runs to spend. They
  become runtime evidence only through a regime-matched, held-out calibration
  (principle 4).
- Report the strongest level reached and keep each failure at the stage where
  it happened. Do not hide it behind `local_map`, a replicated counterfactual,
  or a native fallback.

## 2. Compare only matched decision boundaries

A comparison is valid only when every candidate runs the same contract and the
measured or modeled boundary covers the production decision.

The contract includes workload and batch, mesh and topology, dtype and mixed
precision, activation checkpointing as realized rather than as configured,
estimator and scheduling settings, and everything between one backward and the
next forward: optimizer update and state, gradient transformations such as
clipping, gradient accumulation and reduction frequency, trainability mask and
optimizer parameter groups, tied parameters, mutable buffers, checkpoint/resume
when in scope, and startup when the job length is finite.

- A component missing for any candidate makes the result a partial-boundary
  result. Report it, keep end-to-end competitiveness inconclusive, and do not
  transfer the partial ranking: placement changes local optimizer bytes,
  DTensor dispatch, the realized kernel family, and the cost of any
  placement-dependent reduction.
- Count components at their real frequency. With gradient accumulation `G`, a
  cycle has `G - 1` reduction-suppressed microsteps, one synchronized
  microstep, and one update. Compare whole cycles, or cycles amortized per
  microstep.
- Accept memory against the complete training step. When optimizer state was
  absent from a measurement, add the missing persistent tensors analytically
  (gradients and optimizer state only for local trainable parameters; frozen
  parameters still occupy storage) and include operational headroom.
- Time-to-ready and steady-state latency are different quantities. For a
  finite job, compare startup plus steps times step latency and report the
  break-even step count; with an unknown job length the steady-state decision is
  conditional.
- A semantic mismatch, such as changed trainability, untied aliases, or broken
  buffer mutation, invalidates the candidate's timing. It is a failure, not a
  slow result.

Reason codes: `OPTIMIZER_PERFORMANCE_UNMODELED`, `STEP_BOUNDARY_MISMATCH`,
`OPTIMIZER_STATE_MEMORY_INFEASIBLE`, `TRAINING_MEMORY_HEADROOM_UNAVAILABLE`,
and `STARTUP_NOT_AMORTIZED` when every candidate meets the bar but startup is
not recovered within the job.

## 3. One decision per workload contract

A routing decision belongs to one contract. Give a materially different
contract a new `decision_id`; retries and linked baselines of the same contract
share it.

Material changes include per-microstep global batch, gradient accumulation,
reduction or update frequency, clipping algorithm or threshold, sequence
length, trainability mask or optimizer groups, activation-checkpoint policy,
mesh shape or world size, and execution mode. A world-size change, for example,
changes local batch, collective group sizes, and possibly the selected
placement, so timings from different meshes cannot be pooled regardless of
which is newer.

- An explicit global microbatch is authoritative. Assert
  `global_batch % effective_dp_degree == 0`, capture with the global shape, and
  execute with `global_batch // effective_dp_degree` per DP rank.
- World size does not define the batch. A batch chosen for convenience is smoke
  evidence and cannot rank strategies; obtain the production configuration
  before routing.
- A batch-sensitivity experiment (`workload-representativeness.md`) is a
  separate configuration with its own aligned baseline and never replaces the
  user's workload.

## 4. Accept only robust wins; otherwise keep the working baseline

Compare intervals, not point estimates.

- **Modeled:** use optimistic, expected, and pessimistic envelopes widened by
  held-out calibration error (`baseline-counterfactuals.md`). A calibration
  applies only inside its regime: platform (hardware, framework version, dtype
  policy, realized kernel family), optimizer contract (groups, update
  frequency, accumulation, state placement, dispatch), and scale (local
  parameter bytes and tensor counts). Check all three and widen or reject the
  envelope on a mismatch or material unknown. Use the maximum relevant held-out
  error, not the median or winner accuracy.
- **Measured:** samples inside one process miss launch-to-launch variance from
  compilation, allocator, and system state. Use at least three independent
  launches per strategy before accepting a measured win; a close
  single-launch result is inconclusive (`MEASURED_LAUNCH_VARIANCE_UNMODELED`).
- **Composed:** a modeled increment added to a measured base carries both
  uncertainties. Reusing an older base requires a refreshed base or a validated
  temporal-drift bound (`COMPOSED_FULL_STEP_BASE_DRIFT` when drift explains a
  miss).
- Overlapping envelopes are `schedule_sensitive`; a missing template or
  calibration is `comparison_unavailable`. Neither is an AutoParallel win or a
  proven loss: record the AutoParallel outcome as `inconclusive`, keep the
  existing or normal fallback, and identify the missing evidence and the next
  discriminating measurement.
- Pareto dominance overrides tolerance. A baseline that is robustly no worse on
  latency, complete-step memory, and every other recorded criterion wins even
  when AutoParallel is inside the latency tolerance.
- Do not run a real baseline benchmark to resolve uncertainty without the
  user's authorization.

## 5. Diagnose from the artifact that can show the cause

- **Selected versus available strategies.** An exported placement is the
  winner, not the option set. Before claiming missing propagation rules,
  separate replicate-only option sets and implicit-fallback warnings from
  replicated selections that had sharded alternatives
  (`summary.strategy_space`). Broad sharded coverage with replicated selections
  points to the objective, constraints, or workload.
- **Objective boundary.** The sharding objective does not price the eager
  optimizer update. A plan that wins on forward/backward, keeps much more local
  parameter storage than full sharding, and loses once the update is included
  indicates `OPTIMIZER_COST_MISSING_FROM_OBJECTIVE`. Route on the full step,
  retain the parameter-storage fractions and both boundary results for
  objective work, and test the hypothesis with at most one parameter-memory
  counterfactual near the full-shard fraction.
- **Communication and bucketing.** Use the post-lowering graph and
  `interpreting-results.md`; a collective count or a single-rank timing alone
  does not identify the cause.
- **Realizability.** When solve succeeds but placement application, lowering,
  or execution fails, follow `realizability-failures.md`.
- **Optimizer revisions.** Rescore the recorded and new placements under one
  evaluator; raw objectives from different revisions are not comparable.
