# AutoParallel-first routing

Use this policy when the user asks to parallelize a PyTorch model, scale
training across GPUs, reduce per-rank training memory, or add or compare DDP or
FSDP. The goal is to consider AutoParallel before committing to a manual
wrapper, without overriding an explicit user choice or spending an unbounded
amount of time on an unsupported model.

## Decision flow

1. Record the parallelization intent, target allocation, representative
   workload boundary and provenance, global and local microbatch, gradient
   accumulation, effective batch, and user acceptance criteria. World size
   does not define global batch. When the global microbatch is specified, use
   that exact value for AutoParallel capture and evaluation and derive the local
   batch from effective DP degree. If the user supplied no bar, use
   `default_acceptance_criteria()` and `default_attempt_budget()` from
   `scripts/parallelization_policy.py`.
2. Assess AutoParallel eligibility. An unavailable checkout, missing model or
   input boundary, no CUDA capture environment or saved optimizer, or a clearly
   incompatible global-SPMD computation can make the attempt ineligible. Local
   inconvenience alone is not evidence that DDP or FSDP is better.
3. If eligible, make one unmodified AutoParallel attempt. Permit at most one
   targeted semantics-preserving retry by default. Stop after ten minutes
   without stage progress unless the user supplied a different budget. Never
   make a semantics-changing rewrite without authorization. A rewrite that
   preserves outputs but materially changes work, such as evaluating every MoE
   expert for every token, is diagnostic lowering evidence only and cannot
   establish strategy quality.
4. Accept AutoParallel only when it meets the recorded bar. Otherwise record
   `below_bar`, `failed`, `ineligible`, or `inconclusive` with stable reason
   codes and select a fallback.
5. When AutoParallel passes the absolute bar, follow
   [baseline-counterfactuals.md](baseline-counterfactuals.md). Compare it with
   versioned DDP/FSDP counterfactuals under calibrated optimistic, expected,
   and pessimistic scheduling assumptions. The reordered estimate is the
   expected point, not ground truth. Do not run a real baseline benchmark
   without explicit user authorization.
6. Accept AutoParallel as relatively competitive only when its calibrated
   envelope is robustly competitive, every applicable baseline is infeasible,
   or the user explicitly accepts a documented tradeoff. Treat overlapping,
   uncalibrated, or unavailable comparisons as inconclusive and preserve an
   existing working baseline.
7. Prefer DDP when the replicated parameters, gradients, and optimizer state
   fit per rank and the requested computation is ordinary data parallelism.
   When graph capture is the incompatibility, a versioned native eager DDP or
   FSDP wrapper is a valid fallback. Record its execution backend explicitly;
   do not treat eager fallback timing as a matched compiled competitiveness
   comparison.
   Prefer FSDP when replicated training state does not fit or parameter
   sharding is required. Apply the same execution, memory, correctness, and
   performance bar to the fallback.
8. Record the final selected strategy and outcome. A fallback is evidence, not
   a silent disappearance of the AutoParallel failure.

## Keep routing fields orthogonal

Use each routing field for one question:

- **Disposition** records whether AutoParallel was attempted, only assessed, or
  skipped. An explicit DDP/FSDP requirement normally makes AutoParallel
  `assess_only`, with an `ineligible` AutoParallel outcome for that decision.
- **Outcome** records the AutoParallel candidate's absolute acceptance state.
  Use `failed` when an attempted path errors or violates a semantic invariant,
  `below_bar` when sufficient evidence proves an acceptance criterion was
  missed, and `inconclusive` when required evidence is absent, mismatched, or
  still pending. Do not turn uncertainty into `below_bar` merely because the
  conservative final selection is DDP or FSDP.
- **Competitiveness verdict** records the relative performance comparison, not
  the whole acceptance decision. AutoParallel can be `robustly_competitive`
  on steady-state latency yet remain `below_bar` because complete-step memory
  or finite-job time-to-solution loses. Use `pareto_dominated` only for robust
  no-worse evidence across every recorded acceptance dimension; otherwise keep
  the latency-envelope verdict.
- **Benchmark action** is `run_authorized` only when the user authorized the
  needed benchmark, `skip` when a benchmark was explicitly considered but is
  not authorized, and `none` when no benchmark action is part of the decision.
- **Selected strategy** may be `none` while an authorized close comparison is
  still running. When evidence is unavailable and no benchmark is pending,
  select the normal conservative fallback without changing an inconclusive
  AutoParallel outcome into a proven loss.

Before finalizing these fields, run
`scripts/validate_parallelization_routing.py` on the classification. Its
optional evidence document accepts facts such as `requested_strategy`,
`autoparallel_attempted`, `attempt_error_observed`,
`semantic_violation_observed`, `required_evidence_missing`,
`workload_representative`, `benchmark_needed`, `benchmark_authorized`,
`benchmark_pending`, and `qualified_fallback`. Only provide facts established
by the run. The validator rejects contradictory fields; it does not infer
missing evidence or replace the routing decision.

When only a synthetic or inferred workload is available, an AutoParallel run
may establish smoke evidence but not a production quality decision. For a
controlled benchmark or authorized tuning task, use the bounded sensitivity
procedure in `workload-representativeness.md`. Do not change a user's fixed
microbatch to rescue AutoParallel; route the intended workload independently.

When the request is only a calibration, counterfactual, or implementation
experiment and no production strategy will be selected, record
`intent.kind="diagnostic_evaluation"`. Keep real routing attempts as
`distributed_training`; an experiment linked to an unresolved routing request
does not become diagnostic merely because it is inconclusive.

If the user explicitly requires DDP or FSDP, preserve that choice. Assess
AutoParallel briefly when it remains within scope, report the alternative, and
do not substitute it without permission.

## Default acceptance bar

The default production bar is:

- the measured or modeled workload represents the intended per-step training
  contract rather than a convenience input derived from world size;
- an explicit global microbatch matches the captured global input and executed
  local input after accounting for effective DP degree;
- the intended global-SPMD formulation was audited;
- the selected plan lowered and executed forward and backward on the target;
- observed or conservatively estimated per-rank memory for the intended
  training step, including optimizer state and operational headroom, fits the
  target limit;
- no unresolved error-severity finding remains;
- a user-supplied throughput or latency target is met, when one exists; and
- numerical validation passes when the user requested it or correctness is in
  doubt.

For a production training-performance decision, the relative boundary must
also include the intended optimizer update or a calibrated optimizer-cost
envelope for every candidate. Parameter placement changes optimizer work and
DTensor dispatch overhead, so a forward/loss/backward-only comparison cannot
be assumed to preserve the full-step ranking. If that cost is missing, use
`OPTIMIZER_PERFORMANCE_UNMODELED`; keep an existing working baseline or select
the normal fallback rather than accepting AutoParallel on the partial boundary.
When measured full-step evidence shows that a plan with more local parameter
storage loses only after the optimizer update, classify it as
`OPTIMIZER_COST_MISSING_FROM_OBJECTIVE`, not as missing sharding propagation.

The default relative-quality bar additionally requires
`robustly_competitive`, `baseline_infeasible`, or `user_accepted_tradeoff`.
`pareto_dominated`, `schedule_sensitive`, `robustly_worse`, and
`comparison_unavailable` do not establish that AutoParallel is as good as
DDP/FSDP. Apply the Pareto guard after the latency envelope: a baseline that is
robustly no worse on latency, complete-step memory, and all other recorded
acceptance dimensions wins even if AutoParallel is inside the latency
tolerance. A comparison may remain modeled; the AutoParallel candidate itself
must still reach the required target execution evidence.

An ILP-feasible or lowered-only plan is useful evidence but is not an accepted
production parallelization. If target hardware is unavailable, report the
decision as inconclusive rather than accepting modeled evidence as execution.
A DDP baseline that cannot run because it exceeds memory does not invalidate a
successful AutoParallel result.

Do not equate a forward/loss/backward measurement with optimizer-step memory
feasibility. When the optimizer update is outside the timed boundary, record
the observed peak and a separate projection for persistent optimizer state. If
the observed boundary already materializes parameters and gradients, add the
local optimizer tensors that are still absent; for AdamW this is normally its
two moment tensors, using their actual dtypes and sharding. Apply the same
projection to DDP and FSDP. Reject or leave inconclusive a plan whose projection
exceeds the per-rank limit or leaves less than the required cluster/user
headroom, using `OPTIMIZER_STATE_MEMORY_INFEASIBLE` or
`TRAINING_MEMORY_HEADROOM_UNAVAILABLE`.

## Durable decision records

Give all reports for one routing decision the same `decision_id`. Configure an
AutoParallel report session before capture:

Keep materially different workload contracts under different decision IDs.
Changing per-microstep global batch, gradient-accumulation count, reduction or
optimizer-update frequency, gradient-clipping algorithm or threshold, sequence
length, trainability mask or optimizer parameter groups, mesh shape or world
size, or execution mode is a new decision even when the model name is
unchanged. A
world-size change can alter local batch, collective group sizes and scale, and
the selected placement, so do not pool its timing or routing evidence with
another mesh. Retries and linked baselines for that exact contract continue to
share one ID. For a sub-allocation that canonical topology detection cannot
represent, require an explicit NCCL topology for the allocated communicator;
fallback topology costs and unused visible GPUs are not final routing evidence.
In the routing response, enumerate local batch, collective scale, and potential
selected-placement changes, then state explicitly that timing evidence cannot
be pooled across the meshes.

```python
from parallelization_policy import (
    classify_modeled_competitiveness,
    default_acceptance_criteria,
    default_attempt_budget,
)

parallelization = {
    "decision_id": decision_id,
    "intent": {
        "kind": "distributed_training",
        "requested_strategy": "automatic",
        "attempt_budget": default_attempt_budget(),
    },
    "acceptance_criteria": default_acceptance_criteria(
        memory_limit_bytes=memory_limit_bytes,
        performance_target=performance_target,
    ),
}

with AutoParallelReportSession(
    mesh,
    request=request,
    parallelization=parallelization,
    mode="validate",
) as session:
    # Capture, solve, lower, and execute as usual.
    assessment = classify_modeled_competitiveness(
        autoparallel_latency,
        baseline_estimates,
    )
    session.record_competitiveness_assessment(
        assessment["verdict"],
        evidence="modeled",
        autoparallel_estimate=autoparallel_estimate,
        baselines=baseline_estimates,
        reason_codes=assessment["reason_codes"],
        details=assessment["details"],
    )
    acceptable = parallelization["acceptance_criteria"]["relative_quality"][
        "acceptable_verdicts"
    ]
    if assessment["verdict"] in acceptable:
        session.record_autoparallel_outcome("accepted")
        session.record_parallelization_decision("autoparallel", "accepted")
    else:
        session.record_autoparallel_outcome(
            "below_bar",
            reason_codes=assessment["reason_codes"],
        )
```

The latency input to the classifier is the structured latency envelope inside
`autoparallel_estimate`; each baseline contains its own latency envelope plus
strategy, versioned template, feasibility, configuration hash, and memory
scope. Record `below_bar` and select the normal fallback when the relative
verdict is not acceptable.

An uncaught exception in `AutoParallelReportSession` automatically records a
stable substage-specific failure outcome when recognized, or
`AUTOPARALLEL_EXCEPTION` otherwise. Run DDP or FSDP in a linked report
with the same decision ID, copy the AutoParallel outcome with
`related_run_id=<autoparallel run id>`, then call `record_fallback_outcome()`
and `record_parallelization_decision()`. Use stable reason codes such as
`CAPTURE_UNSUPPORTED`, `LOWERING_FAILED`, `MEMORY_BAR_MISSED`,
`CAPTURE_DYNAMIC_OUTPUT_SHAPE_UNSUPPORTED`,
`PERFORMANCE_BAR_MISSED`, `MODELED_ROBUSTLY_WORSE`,
`MODELED_SCHEDULE_SENSITIVE`, `BASELINE_COMPARISON_UNAVAILABLE`, or
`WORKLOAD_NOT_REPRESENTATIVE`, `OPTIMIZER_STATE_MEMORY_INFEASIBLE`,
`TRAINING_MEMORY_HEADROOM_UNAVAILABLE`, `OPTIMIZER_PERFORMANCE_UNMODELED`,
`STEP_BOUNDARY_MISMATCH`, `STARTUP_NOT_AMORTIZED`, `USER_REQUIRED_FSDP`; put
variable diagnostic text in `details`. Use `STARTUP_NOT_AMORTIZED` when all
candidates meet the acceptance bar but a finite run ends before a slower
startup is recovered; this is not `PERFORMANCE_BAR_MISSED`.

The catalog exposes privacy-reduced outcome, selected-strategy, and fallback
counts. Selecting DDP or FSDP also emits `AUTOPARALLEL_FALLBACK`, allowing the
improvement queue to identify recurring reasons without changing defaults
automatically.

## Behavioral evaluation

For a fast routing canary, `agent-routing-evals.json` covers generic
parallelization, explicit FSDP, failure, memory, modeled win/loss, schedule
sensitivity, benchmark authorization, and unrelated tuning. Give only prompts
and context to the agent, collect one structured response per case, then score
them with:

```bash
python scripts/evaluate_parallelization_routing.py responses.json
```

Use `--template` to print the response shape. This evaluation checks routing
behavior; it does not replace model capture, execution, or performance tests.

For release-quality behavioral evaluation, use the isolated suite in
`tests/agent_skills/autoparallel_optimizer`. The runner copies only declared
fixtures, omits answer keys from agent workspaces, repeats each case, runs the
configured no-skill baseline, grades against the hidden rubric in a separate
workspace, and preserves responses, traces, diffs, and scores:

```bash
python scripts/run_behavioral_evaluation.py /path/to/new-results \
  --pairwise-grade --jobs 4
```

Blind comparison grades every response pair once in each A/B order. It records
both judgments but counts a directional consensus only when both orders prefer
the same variant; an order disagreement becomes a tie. Use the raw judgment
position diagnostics to detect grader position effects.

Use `--plan` to inspect the invocation set without running it and `--case` with
`--repeats 1` for a bounded runner smoke test. The complete default protocol is
substantial; do not launch it without authorization for its model cost.
After an interrupted campaign, rerun the identical command with `--resume`.
Resume requires matching source, skill, case, fixture, rubric, model, command,
timeout, selection, evaluator, and grading provenance; it reuses only
successful complete runs and valid reciprocal blind judgments, and reruns
incomplete or failed artifacts. A
campaign-wide non-blocking lock rejects concurrent writers to the same output.
Compare completed campaigns with
`python scripts/compare_behavioral_evaluations.py LEFT RIGHT`. Treat score
changes as directly comparable only when the reported model, reasoning,
repeats, case, fixture, rubric, baseline, and grading protocol fields match;
the evaluator hash must match, while the skill hash is the treatment and may
intentionally differ.
