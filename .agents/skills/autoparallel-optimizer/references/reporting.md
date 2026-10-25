# AutoParallel run reports

Create a run report for planning, evaluation, validation, and measurement work.
Start it before capture so failures are retained. The report is an evidence
bundle, not a claim that every stage succeeded.

## Configuration

The v1 writer uses filesystem report roots. It resolves configuration in this
order:

1. explicit Python arguments;
2. `AUTOPARALLEL_CLUSTER_PROFILE`, `AUTOPARALLEL_CLUSTER_PROFILES`, and
   `AUTOPARALLEL_REPORT_ROOT`;
3. an environment match or default in
   [cluster-profiles.json](cluster-profiles.json).

The packaged profiles contain only site-neutral defaults. Keep real cluster
names, scheduler metadata, usernames, and filesystem roots in a
cluster-managed file outside the checkout. Start from
[cluster-profiles.example.json](cluster-profiles.example.json), install one
private file per cluster, and point `AUTOPARALLEL_CLUSTER_PROFILES` at it. The
external file is merged over the packaged profiles and may declare
`match_env`, `report_root`, `topology_profile`, `capture_level`,
`retention_days`, feedback refresh and triage defaults, and policies.

Prefer explicit selection from the cluster's job environment:

```bash
export AUTOPARALLEL_CLUSTER_PROFILES=/etc/autoparallel/profiles.json
export AUTOPARALLEL_CLUSTER_PROFILE=site-default
```

Use a stable, non-sensitive profile ID or calibration-domain name because it
is retained in run reports. When automatic matching is useful, keep the real
scheduler values in the external file's `match_env`; never commit them or infer
a cluster from visible GPUs. `AUTOPARALLEL_REPORT_ROOT` can override only the
destination. Do not keep the private file inside the repository, even if it is
gitignored, and do not put credentials in a profile.

Reports identify the packaged source as `builtin` and an external source by a
content digest. They do not retain the external configuration path,
`report_root`, or `match_env` values.

If profile selection is ambiguous, set `AUTOPARALLEL_CLUSTER_PROFILE`
explicitly. The packaged `local` profile writes under
`artifacts/autoparallel-reports` and is the fallback when no external site
profile is configured.

## Parallelization decisions

For a generic request to parallelize a model, follow
[parallelization-routing.md](parallelization-routing.md) and attach a
`parallelization` record to the report session. It contains a stable decision
ID, the original intent, and the acceptance criteria. Record the AutoParallel
outcome before selecting a strategy. If DDP or FSDP is used, record its outcome
and the final decision with the same decision ID in a linked report.

Use `intent.kind="diagnostic_evaluation"` for a bounded calibration,
counterfactual, or implementation experiment that is not expected to select a
production strategy. The catalog retains those reports and counts them
separately, but excludes them from routing-decision rates and improvement
candidates. Do not use this intent to hide an unresolved user routing request;
linked experiments for such a request keep its original routing intent and
decision ID.

Use `record_autoparallel_outcome()`, `record_fallback_outcome()`, and
`record_parallelization_decision()` with stable reason codes. Before accepting
AutoParallel under the default policy, use
`record_competitiveness_assessment()` to store the modeled or measured relative
verdict, AutoParallel estimate, versioned DDP/FSDP baseline templates,
scheduling envelopes, memory scopes, and calibration uncertainty. Variable
error messages and measurements belong in `details`. An uncaught session
exception automatically records `failed`; known capture, lowering, and compile
substages use stable specific reason codes, while unclassified failures use
`AUTOPARALLEL_EXCEPTION`. A successful session without an explicit acceptance decision records
`inconclusive` with `ACCEPTANCE_NOT_EVALUATED`. Neither condition silently
accepts a plan.

Record `pareto_dominated` with `BASELINE_PARETO_DOMINATES` when a feasible
baseline is robustly no worse on latency, complete-step peak memory, and every
other recorded acceptance dimension. Keep the per-dimension bounds in
`details`; do not reduce this decision to a weighted score.

The report writer rejects an accepted AutoParallel decision when the recorded
acceptance criteria require relative quality and no acceptable competitiveness
verdict exists. `robustly_competitive`, `baseline_infeasible`, and an explicit
`user_accepted_tradeoff` satisfy the default relative bar. Real DDP/FSDP
benchmarking remains explicit-authorization-only; modeled evidence must not be
labeled measured.

## Writing a report

Import the session helper from the skill's `scripts` directory as described in
[workflow.md](workflow.md). Do not pass a profile name as a Python argument
during ordinary cluster runs. Let the cluster environment select it with
`AUTOPARALLEL_CLUSTER_PROFILE` or the external file's `match_env`; the report
records `selection_source=environment` or `environment_match`, respectively.

```python
from pathlib import Path

from report_session import AutoParallelReportSession

request = {
    "workload": {
        "mode": "training",
        "global_batch": 64,
        "local_batch": 8,
        "gradient_accumulation_steps": 4,
        "effective_global_batch": 256,
        "batch_scope": "forward-backward-microstep",
        "configuration_source": "training-entrypoint",
    },
    "mesh": {"shape": [8], "names": ["dp"]},
    "constraints": {},
    "spmd_assumptions": [],
}
with AutoParallelReportSession(
    mesh,
    request=request,
    campaign={
        "id": "torchbench-high-comm-v1",
        "suite": "torchbench",
        "case_id": model_name,
        "expected_case_count": 6,
    },
    compiler_config={"enable_ac": False, "overlap_scheduling": False},
    source_root=Path.cwd(),
    external_source_roots={"torchbench": torchbench_root},
    entry_point="training.entrypoint:main",
    launch_config={
        "launcher": "torchrun",
        "nodes": 1,
        "processes_per_node": 8,
        "world_size": 8,
    },
    mode="measure",
) as session:
    session.mark_substage("capture")
    with AutoParallel(
        model,
        input_fn,
        mesh,
        mp_policy=mp_policy,
        cost_model=session.cost_model,
    ) as autop:
        session.mark_substage("solve")
        placement = autop.optimize_placement()
        session.record_plan(autop.sharding_optimizer)
        session.mark_substage("lower")
        parallel_model = autop.apply_placement(placement)
        session.mark_lowered()

    # Compile and run forward/backward while the session is active.
    session.mark_substage("compile")
    compiled = torch.compile(
        parallel_model,
        backend=autoparallel_backend(
            enable_ac=False,
            overlap_scheduling=False,
        ),
    )
    def step():
        parallel_model.zero_grad(set_to_none=True)
        output = compiled(*local_inputs)
        model_specific_loss(output).backward()

    session.mark_substage("measure")
    session.measure_step("target-step", step)
```

The session detects topology by default, records whether it was detected or
explicit, keeps that cost model active through compilation, saves placements
through scratch space outside the bundle, and records forward/backward metrics
before validation. The collector runs on every distributed rank, while rank 0
alone owns the shared report bundle; `RANK` is used before the process group is
initialized. Use `mode="plan"` when no lowering or compilation will run. Pass
an explicit topology only for non-standard hardware.

The session records a reproducibility block containing the supplied model entry
point and sanitized launch configuration. It also records the source revision,
whether the checkout is dirty, and a state fingerprint covering tracked changes
and untracked paths. `external_source_roots` records the same state for model or
benchmark repositories; a missing external revision makes reproduction
incomplete. Do not pass raw command lines, environment dumps, dataset paths, or
secrets as launch metadata.

Use one stable campaign ID for a suite and one stable case ID per workload.
`expected_case_count` lets the catalog distinguish missing cases from failures.
Retries reuse the case ID; campaign summaries use the latest attempt while
retaining the total attempt count. Campaign metadata is operational and does
not split the execution-configuration hash.

Record enough workload provenance to decide whether the run is representative:
global and local microbatch, effective DP degree, gradient accumulation,
effective optimizer batch, material input dimensions, and whether the values
came from the user, training entry point, maintained benchmark, or a synthetic
smoke input. Do not encode global batch as world size by convention. Batch
sensitivity points are distinct execution configurations and should be linked
by campaign rather than written as retries of one case.

The request's explicit global microbatch must match the global logical input
used for capture. Record the derived local batch and effective DP degree, and
verify that the executed local input uses that local batch. Emit
`BATCH_CONTRACT_MISMATCH` with error severity rather than publishing a quality
comparison when these values disagree. If the supplied global batch is an
effective optimizer batch, separately record the per-microstep global batch and
gradient-accumulation steps.

The context manager finalizes successful attempts and records an uncaught
exception with its original traceback before re-raising it. The lower-level
`AutoParallelRunReport` and `ReorderedMetricsCollector` remain available for
specialized workflows that cannot use the session.

Use `record_validation()` only for work that actually ran. Its evidence level
must identify the strongest supported claim: `compiled`, `numerically_checked`,
`executed_on_target`, or `measured`. Keep seeds, input hashes, tolerances,
hardware, and timing method in the validation result.

For target performance feedback, use `mode="measure"` and `measure_step()`.
Warmup runs occur before CUDA peak-memory reset. Each measured forward/backward
iteration waits for full-device completion; a new graph specialization during
that window invalidates the measurement. CUDA-event samples and peak
allocated/reserved bytes are reduced by maximum across ranks, then stored as
`measured` evidence. The result derives a measurement-configuration hash from
the report configuration, measured boundary, timing method, and iteration
counts, so comparable estimate errors can be aggregated without a DDP or other
unsharded baseline. Exclude the optimizer update unless it has a corresponding
graph estimate. When a production measurement intentionally includes it, pass
`optimizer_update=True` so the report declares the wider boundary. Use one
stable input specialization throughout warmup and measurement.
The timing result also records max-rank wall time for the first warmup and the
complete warmup. Treat this as time-to-ready evidence that includes lazy
compilation when applicable, not as steady-state step latency or a compiler-only
measurement.

Add machine-readable findings with `add_finding()`. Prefer stable codes such as
`SPMD_MODELING_ASSUMPTION`, `FALLBACK_COST_PATH`,
`REPLICATE_ONLY_REGION`, `OPAQUE_REGION_UNPRICED`, and
`EXTRAPOLATED_CRITICAL_TERM`. Use `OPTIMIZER_COST_MISSING_FROM_OBJECTIVE` when
full-step measurements isolate an optimizer cost that the sharding objective
did not price. Include source, node, operation, shapes, and the evidence
supporting the diagnosis when available.
When an optimizer-cost envelope is used, report its feature definition, fixed
term, fitted coefficient, units, calibration population, realized kernel
family, regime identifiers, held-out error bound, and whether the prediction
was frozen before the target run. Keep the prediction report linked separately
from the later actual measurement.

For workload and selection diagnostics, use
`SYNTHETIC_UNDERFILLED_WORKLOAD`,
`SELECTED_REPLICATION_WITH_SHARDED_OPTIONS`,
`LAUNCH_FLOOR_DOMINATED_OBJECTIVE`, `BATCH_SENSITIVITY_OBSERVED`, and
`BATCH_CONTRACT_MISMATCH` only when their supporting batch contract, strategy
counts, or cost evidence is present.

The session automatically emits `HIGH_COMMUNICATION_FRACTION`,
`REPLICATED_PARAMETER_STORAGE`,
`SELECTED_REPLICATION_WITH_SHARDED_OPTIONS`, and
`CLUSTERING_GROUPS_SKIPPED` from the selected plan using profile thresholds. It
also emits
`EXTRAPOLATED_CRITICAL_TERM` when NCCL coverage reports an extrapolated or
algorithm-loop term. These are investigation signals, not correctness claims.

Failures include a normalized signature derived from stage, current substage,
exception type, message template, and stable traceback frame identities. Mark
substages before risky operations so capture, solve, lower, compile, and
measure failures remain distinguishable. The full traceback is still retained
in the failure artifact; use the signature for recurrence grouping rather than
discarding diagnostic detail.

## Bundle contract

The writer creates:

```text
<root>/v1/<profile>/<date>/<run-id>/
  manifest.json
  report.json
  summary.md
  artifacts/
```

`report.json` follows [run-report-schema-v1.json](run-report-schema-v1.json).
`manifest.json` hashes every bundled file. Catalog verification rejects
unmanifested files and symbolic links. Finalized bundles are immutable through
the writer; create a linked run for a counterfactual or retry rather than
editing prior evidence.

The standard capture level should include placements, optimizer JSON, metrics,
logs, and structured failures. Full optimizer state, compiler traces, and
execution traces are diagnostic or replay artifacts: they can be large,
version-coupled, sensitive, or unsafe to load from an untrusted source. Add
full optimizer state with `include_optimizer_state=True` only when the profile
policy and user authorization permit it, and load it only from a trusted
bundle. The standard capture level retains a smaller compressed optimizer
replay while paying optimizer construction cost again during replay. Pass
`include_optimizer_replay=False` only when graph-retention policy forbids it;
record that the bundle is not eligible for offline optimizer evaluation.
External artifacts can be registered by URI and digest instead of copied.

Do not place model weights, input values, credentials, or secrets in reports.
Source paths, shapes, tracebacks, and graph metadata may still be sensitive;
honor the destination's access and retention policy.

## Improvement use

Treat observations as immutable and interpretations as revisable. Compare
predicted and observed results only when workload, topology, dtype, compiler,
and scheduling configurations match. An aggregate measurement validates the
aggregate; it does not change an individual cost term's provenance to measured.

Use recurring findings to propose skill fixes, regression cases, coverage work,
or calibration experiments. Validate shared profile or skill changes against
historical and held-out reports before publishing them. Do not automatically
change cluster defaults or cost calibration from one report.

Refresh the privacy-reduced feedback products periodically rather than
rewriting finalized run bundles. A scheduler can invoke the one-shot command:

```bash
python scripts/refresh_report_feedback.py
```

For an environment with a supervised long-running process, the same script can
own the interval directly:

```bash
python scripts/refresh_report_feedback.py --watch
```

Profile defaults control the interval, recurrence threshold, measurement-error
threshold, and representative-bundle limit. CLI arguments override them. Each
refresh takes a non-blocking filesystem lock, verifies complete manifests,
builds all outputs in memory, writes them atomically, and publishes
`feedback-status-v1.json` last. Concurrent refreshes safely report
`skipped_locked`. Report sessions do not spawn an unsupervised background
process; install the one-shot command in the cluster scheduler or supervise the
watch process explicitly.

Feedback, diagnostic, capture, retention, and campaign settings are operational
metadata and are excluded from execution-configuration hashes, so changing a
threshold or suite identifier does not split otherwise comparable measurements.

The refresh produces:

```text
<root>/
  catalog-v1.json
  improvement-queue-v1.json
  improvement-queue-v1.md
  feedback-status-v1.json
```

The catalog indexes status, exact completed evidence, profile, artifact roles,
cost-model status, finding codes, failure substages and signatures, external
source completeness, campaign coverage, privacy-reduced parallelization
decisions, and measurement errors. Parallelization summaries report
AutoParallel outcomes, modeled competitiveness verdicts and evidence, baseline
strategy and feasibility coverage, selected strategies, fallback outcomes, missing
competitiveness records, and incomplete decisions. Campaign summaries report
latest-attempt status, completed evidence, and missing case counts. Full
request and provenance values remain hashed.
Active bundles are pending until their final manifest is written.

Decision summaries use the latest terminal report (`accepted`,
`fallback_selected`, or `failed`) as the authoritative routing state. Later
component measurements or bounded template experiments with
`selected_strategy=none` and `status=inconclusive` remain visible as evidence
but do not erase that terminal state. A later terminal report for the same exact
contract may revise it. Use a new decision ID when a material contract field
changes, including the activation-checkpoint policy, checkpoint implementation,
whether checkpointing was actually realized, or optimizer parameter-group
membership and per-group hyperparameters. State-checkpoint format, included
model/optimizer state, and restore topology are also material when resume is in
scope. Comparison reports must reject
sources that declare different checkpoint policies; an enabled flag with no
graph or memory effect is evidence of a failed realization, not an equivalent
checkpointed run.

The improvement queue groups failures by normalized signature, findings by
stable code, DDP/FSDP selections by fallback strategy and stable reason code,
relative-quality regressions or schedule sensitivity by baseline strategy, and
measured estimate drift by cluster, configuration, metric, and unit. A signal
must affect the configured minimum number of distinct runs; invalid bundles are
immediate integrity candidates. Every candidate has a stable ID, priority,
first and last observation, affected profiles, a bounded set of representative
bundle paths, and a suggested next action. The queue is input for fixes,
regression cases, coverage work, and calibration experiments; it never changes
skill instructions, cluster defaults, or cost calibration by itself.

For optimizer changes, run replayable queue representatives through the
offline corpus evaluator:

```bash
python scripts/evaluate_optimizer_replay_corpus.py REPORT_ROOT \
  --output optimizer-replay-evaluation.json
```

The queue orders replayable representatives first. The evaluator assigns a
stable training/held-out split, reloads each trusted replay, rescales the
recorded and candidate placements under the current evaluator, and reports
strategy-space, placement, objective, storage, and modeled-baseline changes.
Do not compare raw objectives across source revisions. A replay pass is not
lowering, numerical, compilation, or target-runtime evidence; use its held-out
results to choose the bounded candidates that warrant those checks.

`feedback-status-v1.json` exposes the last attempt and success, duration,
expected interval, `stale_after`, pending and invalid bundle counts, and triage
candidate counts. Consumers should treat `failed`, `degraded`, or a current time
after `stale_after` as unhealthy feedback state.

For measured estimate comparisons, include entries in the validation result:

```json
{
  "comparisons": [
    {
      "configuration_hash": "sha256:...",
      "metric": "step_time",
      "unit": "us",
      "estimated": 123.0,
      "observed": 130.0
    }
  ]
}
```

The configuration hash must cover workload, topology, dtype, compiler, and
scheduling configuration. The catalog groups errors only when that hash,
cluster profile, metric, and unit all match.
