# Modeled DDP and FSDP counterfactuals

Use this procedure when AutoParallel must be compared with an applicable DDP
or FSDP baseline without running an additional user workload. It estimates
relative competitiveness; it does not establish measured parity with PyTorch's
runtime implementations.

## Preserve the user's execution contract

Build every candidate from one captured joint graph and hold these inputs
fixed: model and source revision, training boundary, global batch, input
specialization, dtype and mixed precision, optimizer-state assumptions,
activation checkpointing, mesh and physical topology, compilation settings,
and the included step boundary. A comparison is unavailable when those values
cannot be aligned.

For an end-to-end training decision, the boundary includes optimizer update
cost. A forward/backward graph model may still support a partial comparison,
but it is not a production throughput verdict unless a versioned optimizer
template adds calibrated update cost for every strategy. Record optimizer
implementation, fused/foreach mode, update frequency, state dtype, and state
placement; otherwise use `comparison_unavailable` for full-step performance.

Prefer the user's existing DDP or FSDP configuration as the baseline template.
Otherwise use a versioned standard template and record that it was inferred.
Do not call an arbitrary replicated or fully-sharded AutoParallel placement
"DDP" or "FSDP" without checking the following contracts.

### DDP-like template

- shard the logical batch over the data-parallel axis;
- keep parameters replicated over that axis;
- represent terminal gradient reduction and the intended reduction dtype;
- record bucket size, bucket order, and whether gradient communication may
  overlap backward compute; and
- record mutable-buffer broadcast and update semantics. When AutoParallel uses
  global logical BatchNorm statistics, ordinary rank-local DDP BatchNorm is not
  equivalent; use SyncBatchNorm or exclude that baseline from comparison; and
- for gradient accumulation, record `no_sync()` frequency and the DDP
  `static_graph` setting. Use `static_graph=False` when alternating suppressed
  and synchronized reductions unless the exact compiled template has been
  validated; and
- include replicated parameters, gradients, and optimizer state in the memory
  feasibility calculation.

### FSDP-like template

- record the sharding mode, parameter groups or wrapping policy, and DP mesh;
- represent parameter all-gather, gradient reduce-scatter, and the requested
  reshard-after-forward behavior;
- record forward and backward prefetch, communication bucketing, mixed
  precision, and rate-limiting assumptions; and
- include persistent state, materialized parameters, communication buffers,
  activations, and optimizer-state assumptions in the stated memory scope.

When the graph cannot express an implementation feature, widen uncertainty or
mark the comparison unavailable. Do not silently credit AutoParallel with an
optimization that the baseline template omits.

## Estimate a scheduling envelope

For AutoParallel and every feasible baseline, record one latency envelope:

- **optimistic:** the lowest latency under legal overlap assumptions;
- **expected:** the implementation-aware or reordered critical-path estimate;
- **pessimistic:** the estimate with eligible communication serialized; and
- **calibration relative error:** the held-out error allowance for this cost
  and scheduling model on a comparable cluster and collective family.

The values must use one unit and satisfy
`optimistic <= expected <= pessimistic`. The reordering pass supplies the
expected point; it is not the acceptance oracle. An uncalibrated model should
use a conservative error allowance or produce `comparison_unavailable`.

Use this report shape for each latency envelope:

```json
{
  "unit": "us",
  "optimistic": 800.0,
  "expected": 950.0,
  "pessimistic": 1200.0,
  "calibration_relative_error": 0.15,
  "schedule_model": "reordered-graph-v1"
}
```

An AutoParallel estimate contains `latency` and a structured `memory` scope.
Each baseline additionally contains `strategy`, `template_id`,
`template_version`, `configuration_hash`, `feasibility`, `latency`, and
`memory`. Use `latency: null` when a baseline is infeasible or cannot be
estimated, and retain the reason in the assessment details.

Use `classify_modeled_competitiveness()` from
`scripts/parallelization_policy.py` for the default latency verdict. It expands
the scheduling endpoints by calibration error and compares AutoParallel with
the strongest feasible baseline. Pass the resolved cluster profile's
`modeled_latency_tolerance_fraction`; the packaged default is `0.05`.

- `robustly_competitive`: AutoParallel's pessimistic calibrated bound remains
  within the configured tolerance of the baseline's optimistic bound;
- `robustly_worse`: AutoParallel's optimistic calibrated bound is worse than a
  baseline's pessimistic bound beyond tolerance;
- `schedule_sensitive`: the calibrated intervals overlap;
- `baseline_infeasible`: every applicable baseline misses a hard feasibility
  constraint; or
- `comparison_unavailable`: a comparable template, estimate, or calibration is
  missing.

After the latency verdict, call `find_pareto_dominating_baseline()` on the
complete strategy records. Use `pareto_dominated` when one feasible baseline's
pessimistic latency is no worse than AutoParallel's optimistic latency, its
complete-step peak memory is no larger, and it is no worse on every other
recorded acceptance dimension such as time-to-ready, with at least one strict
improvement. This guard takes precedence over the latency tolerance.

Memory remains a hard constraint rather than being folded into a latency
score. A baseline is infeasible only when the complete stated memory scope
misses the target, not merely because persistent parameter storage is too
large. If some applicable baseline remains feasible, compare against it.

For training, the complete scope includes optimizer state and an explicit
operational reserve even when latency is estimated over forward/loss/backward
without the update. Record observed and projected peaks separately and use the
same optimizer-state dtype and sharding assumptions for AutoParallel, DDP, and
FSDP. A projection is analytical evidence, not a measured peak.

When adding a calibrated optimizer increment to a measured partial-step base,
carry the base launch envelope and temporal-drift error separately from the
increment-model error. A historical one-launch base is not a zero-error
constant. Refresh it or classify the composed comparison as schedule-sensitive
when its applicable drift bound can change the routing result.

Runtime compatibility qualification is template-specific. If a native FSDP
wrapping boundary warns that a returned view may lose its pre-backward
all-gather hook, treat that exact template as performance-only until the
consumer semantics or a warning-free boundary is qualified. Do not select it
from timing alone. A root-only or output-cloning template may remove the warning
but must be measured with its real memory, startup, and copy costs. Do not
disqualify every FSDP template for the model family because one block policy
warned; keep DDP as the correctness-qualified comparator when warning-free FSDP
alternatives do not meet the performance bar.

Shared parameters constrain nested FSDP ownership. Independently wrapping two
subtrees that reference the same embedding can create overlapping mesh
ownership; record that as a template incompatibility and try only a bounded,
semantics-preserving alternative.

## Route conservatively

AutoParallel may pass the relative-quality gate when it is
`robustly_competitive`, when all applicable baselines are infeasible, or when
the user explicitly accepts a documented tradeoff. A `pareto_dominated`,
`schedule_sensitive`, `robustly_worse`, or `comparison_unavailable` result does
not establish that AutoParallel is as good as the baseline.

Preserve an existing working DDP or FSDP integration when modeled evidence is
close or unavailable. For a new integration, prefer the normal DDP/FSDP
fallback when AutoParallel misses the relative bar. Do not run a real baseline
benchmark unless the user explicitly authorized that resource cost. An
explicit request to compare performance can provide that authorization; a
generic request to parallelize a model does not.

For `user_accepted_tradeoff`, record `details.user_authorized=true` plus the
tradeoff the user accepted. This is an explicit override, not modeled evidence.

Label every result as modeled, measured, or mixed. Never describe a modeled
counterfactual as measured. A DDP or FSDP OOM establishes a feasibility result,
not throughput parity.

## Record enough evidence to improve the model

Record the baseline strategy, template ID and version, configuration hash,
feasibility, latency envelope, memory scope, cost-model and topology versions,
verdict, stable reason codes, and any user-authorized tradeoff. Give a real
baseline run the same decision ID when one is later measured.

The fleet catalog should aggregate verdict and coverage counts without copying
model inputs or raw graph data. Use controlled benchmark suites and occasional
authorized measurements to calibrate scheduling error. A recurring modeled
loss, schedule-sensitive result, or unavailable comparison should create an
improvement candidate; it must not automatically change cluster defaults.
