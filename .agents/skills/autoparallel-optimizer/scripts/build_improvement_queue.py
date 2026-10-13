from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

TRIAGE_SCHEMA_VERSION = 1
_PRIORITY_ORDER = {"P1": 0, "P2": 1, "P3": 2}
_SEVERITY_ORDER = {"info": 0, "warning": 1, "error": 2}
_FALLBACK_ACTIONS = {
    "CAPTURE_DYNAMIC_OUTPUT_SHAPE_UNSUPPORTED": (
        "Add capture support for the recorded data-dependent shape or control-flow "
        "pattern; keep the measured native fallback until the unmodified path lowers."
    ),
    "CAPTURE_UNSUPPORTED": (
        "Minimize the unsupported capture pattern and test one bounded, semantics-"
        "preserving integration before retaining the measured native fallback."
    ),
    "LOWERING_FAILED": (
        "Minimize the selected placement and add a lowering regression before "
        "reconsidering the measured native fallback."
    ),
    "AUTOPARALLEL_COMPILE_FAILED": (
        "Minimize the compiled lowered graph failure and retain its native fallback as "
        "a regression until target execution succeeds."
    ),
}
_FINDING_ACTIONS = {
    "EXTRAPOLATED_CRITICAL_TERM": (
        "Benchmark the reported term and add a bounded calibration or coverage case."
    ),
    "FALLBACK_COST_PATH": (
        "Restore topology detection or provide an explicit topology before changing "
        "placement defaults."
    ),
    "OPAQUE_REGION_UNPRICED": (
        "Add an explicit cost or targeted benchmark for the opaque region."
    ),
    "REPLICATE_ONLY_REGION": (
        "Inspect unsupported sharding propagation and add a regression workload."
    ),
    "SPMD_MODELING_ASSUMPTION": (
        "Audit the global SPMD formulation and encode the resolved assumption in a test."
    ),
    "HIGH_COMMUNICATION_FRACTION": (
        "Inspect the largest communication edges, then compare alternate placements "
        "or improve the relevant cost and propagation rules."
    ),
    "REPLICATED_PARAMETER_STORAGE": (
        "Inspect why parameter-bearing nodes remain replicated and add a focused "
        "sharding-option or cost-model experiment."
    ),
    "CLUSTERING_GROUPS_SKIPPED": (
        "Inspect skipped clustering groups and add a representative regression case "
        "before changing clustering policy."
    ),
    "AUTOPARALLEL_FALLBACK": (
        "Inspect the linked AutoParallel attempts, group stable fallback reasons, and "
        "add a regression case for the highest-leverage unsupported path."
    ),
    "OPTIMIZER_COST_MISSING_FROM_OBJECTIVE": (
        "Model optimizer update cost as a function of local parameter placements and "
        "validate the objective change on the representative full-step cases."
    ),
    "OPTIMIZER_IMPLEMENTATION_SENSITIVITY": (
        "Preserve the configured optimizer implementation, record the realized kernel "
        "family, and cover ranking changes with a regression suite."
    ),
    "OPTIMIZER_UPDATE_PERFORMANCE_GAP": (
        "Compare partial and full-step profiles to isolate placement-dependent optimizer "
        "work before changing sharding or communication rules."
    ),
    "MEASURED_LAUNCH_VARIANCE_UNMODELED": (
        "Collect at least three independent launches per strategy and compare the "
        "combined launch-level envelope."
    ),
    "OPTIMIZER_STORAGE_COUNTERFACTUAL_RECOVERS_PERFORMANCE": (
        "Add placement-dependent optimizer work to the search objective and use the "
        "recorded storage-constrained cases as performance regressions."
    ),
    "OPTIMIZER_COST_MODEL_REQUIRES_CALIBRATION": (
        "Expand independent-launch held-out coverage before using the optimizer term in "
        "production search or routing."
    ),
    "OPTIMIZER_COST_MODEL_PROSPECTIVE_VALIDATION_PENDING": (
        "Run the pre-registered full-step strategies and compare actual results with the "
        "recorded predictions without refitting first."
    ),
    "OPTIMIZER_COST_MODEL_PROSPECTIVE_VALIDATION_PASSED": (
        "Replicate close strategies and expand prospective architecture coverage before "
        "integrating the optimizer term into production search."
    ),
    "OPTIMIZER_COST_MODEL_PROFILE_CALIBRATION": (
        "Expand held-out all-rank optimizer child-kernel profiles and keep parent "
        "communication tail in the full-step scheduling model."
    ),
    "PROFILED_OPTIMIZER_COST_MODEL_PROSPECTIVE_VALIDATION_PENDING": (
        "Measure every pre-registered strategy without refitting and compare the "
        "winner plus kernel/full-step errors with the frozen prediction."
    ),
    "PROFILED_OPTIMIZER_COST_MODEL_PROSPECTIVE_VALIDATION_PASSED": (
        "Expand pre-registered architecture and independent-launch coverage before "
        "integrating the direct optimizer term into production search."
    ),
    "PROFILED_OPTIMIZER_INCREMENT_MODEL_VALIDATION_PASSED": (
        "Retain the direct child-kernel regression and expand architecture coverage "
        "without conflating increment accuracy with base-step stability."
    ),
    "COMPOSED_FULL_STEP_BASE_DRIFT": (
        "Measure contemporaneous independent-launch partial-step controls and calibrate "
        "a temporal-drift envelope for composed full-step predictions."
    ),
    "OPTIMIZER_COST_MODEL_WINNER_VALIDATION_ONLY": (
        "Do not route on winner accuracy alone; use direct kernel profiles and propagate "
        "base-step plus increment uncertainty into full-step bounds."
    ),
    "BASELINE_PARETO_DOMINATES": (
        "Preserve the Pareto-dominance regression and inspect why search selected a plan "
        "that is worse on latency, memory, and startup."
    ),
    "META_RUNTIME_ESTIMATE_RANKING_ONLY": (
        "Stratify meta-to-target calibration by workload regime and collect held-out "
        "target runs before using meta latency estimates for routing."
    ),
    "GRADIENT_ACCUMULATION_REDUCTION_FREQUENCY_GAP": (
        "Add semantics-preserving suppression of intermediate gradient reductions and "
        "retain the full accumulation-cycle regression."
    ),
    "DDP_STATIC_GRAPH_NO_SYNC_INCOMPATIBLE": (
        "Use and regression-test a static_graph=false compiled DDP template when "
        "alternating no-sync and synchronized accumulation microsteps."
    ),
    "WORLD_SIZE_ROUTING_CROSSOVER": (
        "Condition routing and cost calibration on the exact mesh, and preserve the "
        "crossover as a multi-world-size regression."
    ),
    "SUBALLOCATION_TOPOLOGY_REQUIRES_EXPLICIT_PROFILE": (
        "Provide a versioned topology for the allocated communicator and retain a "
        "sub-allocation planning regression."
    ),
    "FROZEN_PARAMETER_TRAINABILITY_LOST_DURING_LOWERING": (
        "Preserve the source requires-grad mask through lowering and assert optimizer "
        "membership before measuring a fine-tuning workload."
    ),
    "FROZEN_PARAMETER_MEMORY_REQUIRES_TRAINABLE_BYTES": (
        "Separate total persistent parameter bytes from trainable gradient and optimizer "
        "state bytes in feasibility projections."
    ),
    "FSDP_T5_VIEW_HOOK_WARNING_REQUIRES_SAFE_BOUNDARY": (
        "Use a warning-free wrapping boundary or retain DDP as the correctness-qualified "
        "baseline; do not qualify a warned block template from timing alone."
    ),
    "FSDP_T5_SAFE_TEMPLATE_PERFORMANCE_GAP": (
        "Preserve the warning-free template results and search only bounded wrapping or "
        "model adaptations that retain exact T5 semantics."
    ),
    "FSDP_T5_SHARED_PARAMETER_WRAP_CONFLICT": (
        "Account for shared embeddings when defining nested FSDP ownership and reject "
        "overlapping independently wrapped stacks."
    ),
    "BACKEND_ACTIVATION_CHECKPOINTING_NO_EFFECT": (
        "Integrate checkpoint save/recompute decisions into AutoParallel's first "
        "partition or make the replay partitioner honor a validated equivalent policy."
    ),
    "MIXTRAL_DYNAMIC_EXPERT_CAPTURE_UNSUPPORTED": (
        "Add capture and sharding coverage for data-dependent expert token selection; "
        "retain native DDP/FSDP fallback until the unmodified sparse path lowers."
    ),
    "MIXTRAL_DENSE_EXPERT_CAPTURE_DIAGNOSTIC": (
        "Use dense all-expert execution only to isolate lowering coverage, never for "
        "cost calibration or routing decisions."
    ),
    "BATCHNORM_GLOBAL_STATS_REQUIRE_SYNC_BASELINE": (
        "Match AutoParallel's global logical BatchNorm statistics with SyncBatchNorm or "
        "an equivalent reduced-statistics baseline before comparing strategies."
    ),
    "TIME_TO_READY_RUN_LENGTH_REQUIRED": (
        "Collect the intended optimizer-step count and route on total time-to-solution "
        "relative to the reported startup break-even point."
    ),
    "STARTUP_NOT_AMORTIZED": (
        "Preserve the finite-job fallback while the intended run ends before the "
        "measured startup difference is recovered by steady-state throughput."
    ),
    "AUTOPARALLEL_COLLECTIVE_FRAGMENTATION_VS_DDP": (
        "Separate dependency-bound tensor-parallel collectives from compatible gradient "
        "reductions, then coalesce only the latter toward the recorded DDP bucket count."
    ),
    "AUTOPARALLEL_MATMUL_FRAGMENTATION_VS_DDP": (
        "Inspect sharding-induced matmul decomposition and retain larger fused kernels "
        "when the communication saving does not cover launch overhead."
    ),
    "PLACEMENT_DEPENDENT_BUCKETING_INVERTS_COST_RANKING": (
        "Model placement-dependent collective coalescing in search or retain a bounded "
        "candidate shortlist for post-lowering reordered selection."
    ),
    "REPLICATED_AUTOPARALLEL_STACK_GAP_VS_DDP": (
        "Profile remaining DTensor dispatch, matmul decomposition, and optimizer cost "
        "against the versioned native DDP template."
    ),
    "DEPENDENCY_BOUND_TENSOR_PARALLEL_COLLECTIVES": (
        "Price dependency-bound tensor-parallel collective launches and local kernel "
        "shapes during search, or rerank a bounded placement shortlist after lowering."
    ),
    "REORDERING_CONFIGURATION_MISMATCH": (
        "Make the effective reordering configuration match report provenance and cover "
        "the configuration contract with a regression test."
    ),
    "MULTIGROUP_BUCKETING_COVERAGE_GAP": (
        "Discover eligible all-reduce-only process groups on multi-dimensional meshes "
        "and validate per-group ordering with a measured regression."
    ),
    "REORDERED_MODEL_MISSES_MULTIGROUP_BUCKETING_GAIN": (
        "Add collective launch count and bucketed-kernel runtime to reordered graph "
        "estimation, then prospectively rerun the recorded multi-group candidates."
    ),
    "MULTIGROUP_BUCKETING_PROFITABILITY_GAP": (
        "Estimate per-group launch savings, packing cost, and dependency structure "
        "before broadening all-reduce bucketing beyond the primary FSDP group."
    ),
    "SHARDING_SOLVER_TIMEOUT": (
        "Reduce ILP construction and presolve cost, preserve repeated-subgraph "
        "compression, and enforce both CBC and end-to-end attempt budgets."
    ),
    "SHARDING_SOLVER_INFEASIBLE": (
        "Inspect the smallest infeasible graph and separate intrinsic placement "
        "legality from contradictory user constraints before relaxing semantics."
    ),
    "CROSS_GROUP_RANK_SKEW_UNMODELED": (
        "Model communicator-specific launch order and peer-wait skew across ranks, "
        "or use max-rank profile evidence when reranking multi-group candidates."
    ),
}


def build_improvement_queue(
    catalog: Mapping[str, Any],
    *,
    minimum_occurrences: int = 2,
    measurement_relative_error_threshold: float = 0.2,
    max_representatives: int = 3,
) -> dict[str, Any]:
    """Create a privacy-reduced queue of recurring improvement candidates."""

    if catalog.get("schema_version") != 1:
        raise ValueError(
            f"unsupported catalog schema {catalog.get('schema_version')!r}"
        )
    if minimum_occurrences < 2:
        raise ValueError("minimum_occurrences must be at least 2")
    if measurement_relative_error_threshold < 0:
        raise ValueError("measurement_relative_error_threshold must be non-negative")
    if max_representatives < 1:
        raise ValueError("max_representatives must be positive")

    catalog_runs = catalog.get("runs", [])
    if not isinstance(catalog_runs, list):
        raise ValueError("catalog runs must be a list")
    runs = _latest_campaign_case_runs(catalog_runs)

    candidates = []
    candidates.extend(
        _failure_candidates(runs, minimum_occurrences, max_representatives)
    )
    candidates.extend(
        _finding_candidates(runs, minimum_occurrences, max_representatives)
    )
    candidates.extend(
        _parallelization_fallback_candidates(
            runs, minimum_occurrences, max_representatives
        )
    )
    candidates.extend(
        _competitiveness_candidates(runs, minimum_occurrences, max_representatives)
    )
    candidates.extend(
        _measurement_candidates(
            runs,
            minimum_occurrences,
            measurement_relative_error_threshold,
            max_representatives,
        )
    )
    candidates.extend(
        _invalid_bundle_candidates(
            catalog.get("invalid_bundles", []), max_representatives
        )
    )
    candidates.sort(
        key=lambda candidate: (
            _PRIORITY_ORDER[candidate["priority"]],
            -candidate["affected_workload_count"],
            -candidate["occurrence_count"],
            candidate["id"],
        )
    )

    kind_counts = Counter(candidate["kind"] for candidate in candidates)
    priority_counts = Counter(candidate["priority"] for candidate in candidates)
    return {
        "schema_version": TRIAGE_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "source_catalog": {
            "generated_at": catalog.get("generated_at"),
            "sha256": f"sha256:{_canonical_sha256(catalog)}",
        },
        "policy": {
            "minimum_occurrences": minimum_occurrences,
            "measurement_relative_error_threshold": (
                measurement_relative_error_threshold
            ),
            "max_representatives": max_representatives,
            "recurrence_unit": "latest_run_per_cluster_campaign_case",
            "ranking": "priority_then_workload_breadth_then_occurrences",
            "catalog_run_count": len(catalog_runs),
            "deduplicated_run_count": len(runs),
        },
        "summary": {
            "candidate_count": len(candidates),
            "kind_counts": _sorted_counter(kind_counts),
            "priority_counts": _sorted_counter(priority_counts),
        },
        "candidates": candidates,
    }


def write_improvement_queue(
    catalog: Mapping[str, Any],
    *,
    output: str | Path,
    markdown_output: str | Path,
    minimum_occurrences: int = 2,
    measurement_relative_error_threshold: float = 0.2,
    max_representatives: int = 3,
) -> tuple[Path, Path]:
    queue = build_improvement_queue(
        catalog,
        minimum_occurrences=minimum_occurrences,
        measurement_relative_error_threshold=measurement_relative_error_threshold,
        max_representatives=max_representatives,
    )
    output_path = Path(output)
    markdown_path = Path(markdown_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(output_path, queue)
    _atomic_write_text(markdown_path, render_improvement_queue(queue))
    return output_path, markdown_path


def render_improvement_queue(queue: Mapping[str, Any]) -> str:
    lines = [
        "# AutoParallel improvement queue",
        "",
        f"- Generated: {queue['generated_at']}",
        f"- Source catalog: {queue['source_catalog']['generated_at']}",
        f"- Candidates: {queue['summary']['candidate_count']}",
    ]
    candidates = queue["candidates"]
    if not candidates:
        lines.extend(["", "No actionable recurring signals."])
        return "\n".join(lines) + "\n"

    for priority in _PRIORITY_ORDER:
        prioritized = [
            candidate for candidate in candidates if candidate["priority"] == priority
        ]
        if not prioritized:
            continue
        lines.extend(["", f"## {priority}", ""])
        for candidate in prioritized:
            lines.extend(
                [
                    f"### {candidate['title']}",
                    "",
                    f"- ID: `{candidate['id']}`",
                    f"- Kind: {candidate['kind']}",
                    f"- Occurrences: {candidate['occurrence_count']}",
                    f"- Affected runs: {candidate['affected_run_count']}",
                    f"- Affected workloads: {candidate['affected_workload_count']}",
                    f"- Replayable runs: {candidate.get('replayable_run_count', 0)}",
                    f"- Profiles: {', '.join(candidate['cluster_profiles']) or 'unknown'}",
                    f"- Suggested action: {candidate['suggested_action']}",
                ]
            )
            representatives = candidate["representative_bundles"]
            if representatives:
                lines.append("- Representative bundles:")
                lines.extend(f"  - `{bundle}`" for bundle in representatives)
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _failure_candidates(
    runs: list[Mapping[str, Any]],
    minimum_occurrences: int,
    max_representatives: int,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for run in runs:
        failure = run.get("failure")
        if not isinstance(failure, Mapping):
            continue
        if failure.get("classification") == "resource_exhaustion":
            continue
        signature = failure.get("signature")
        location_signature = failure.get("diagnostic_location_signature")
        key: tuple[str, ...]
        if signature and location_signature:
            key = ("signature_location", str(signature), str(location_signature))
        elif signature:
            key = ("signature", str(signature))
        else:
            key = (
                "coarse",
                str(failure.get("stage")),
                str(failure.get("type")),
            )
        groups[key].append(run)

    candidates = []
    for key, records in groups.items():
        if len(records) < minimum_occurrences:
            continue
        failure = records[-1]["failure"]
        failure_substage = failure.get("substage")
        failure_location = failure_substage or failure.get("stage")
        signature = failure.get("signature")
        identifier = (
            f"failure:{signature}"
            if signature
            else f"failure:coarse:{_canonical_sha256(key)[:16]}"
        )
        candidates.append(
            _run_candidate(
                identifier=identifier,
                kind="failure",
                priority="P1",
                title=f"Recurring {failure.get('type')} during {failure_location}",
                records=records,
                occurrence_count=len(records),
                max_representatives=max_representatives,
                details={
                    "failure_type": failure.get("type"),
                    "failure_stage": failure.get("stage"),
                    "failure_substage": failure_substage,
                    "failure_signature": signature,
                    "diagnostic_location_signature": location_signature,
                    "failure_classification": failure.get("classification"),
                    "grouping": key[0],
                },
                suggested_action=(
                    "Reproduce the representative failure at its recorded source state, "
                    "minimize it, fix the responsible layer, and add a regression test."
                ),
            )
        )
    return candidates


def _finding_candidates(
    runs: list[Mapping[str, Any]],
    minimum_occurrences: int,
    max_representatives: int,
) -> list[dict[str, Any]]:
    groups: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(
        list
    )
    for run in runs:
        for finding in run.get("findings", []):
            code = finding.get("code")
            if code:
                groups[str(code)].append((run, finding))

    candidates = []
    for code, observations in groups.items():
        observations = _deduplicate_finding_observations(observations)
        records = [observation[0] for observation in observations]
        if _distinct_run_count(records) < minimum_occurrences:
            continue
        severities = [
            str(observation[1].get("severity", "info")) for observation in observations
        ]
        severity = max(severities, key=lambda value: _SEVERITY_ORDER.get(value, -1))
        priority = {"error": "P1", "warning": "P2", "info": "P3"}.get(severity, "P3")
        candidates.append(
            _run_candidate(
                identifier=f"finding:{code}",
                kind="finding",
                priority=priority,
                title=f"Recurring {code} finding",
                records=records,
                occurrence_count=len(records),
                max_representatives=max_representatives,
                details={
                    "finding_code": code,
                    "highest_severity": severity,
                    "severity_counts": _sorted_counter(Counter(severities)),
                },
                suggested_action=_FINDING_ACTIONS.get(
                    code,
                    "Inspect representative finding contexts and add a focused "
                    "regression or coverage experiment.",
                ),
            )
        )
    return candidates


def _parallelization_fallback_candidates(
    runs: list[Mapping[str, Any]],
    minimum_occurrences: int,
    max_representatives: int,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for run in runs:
        routing = run.get("parallelization")
        if (
            not isinstance(routing, Mapping)
            or routing.get("intent_kind") == "diagnostic_evaluation"
        ):
            continue
        strategy_value = routing.get("selected_strategy")
        if strategy_value not in {"ddp", "fsdp"}:
            continue
        strategy = str(strategy_value)
        reason_codes = routing.get("decision_reason_codes", []) or ["UNSPECIFIED"]
        for reason_code in reason_codes:
            groups[(str(strategy), str(reason_code))].append(run)

    candidates = []
    for (strategy, reason_code), records in groups.items():
        if _distinct_run_count(records) < minimum_occurrences:
            continue
        candidates.append(
            _run_candidate(
                identifier=f"fallback:{strategy}:{reason_code}",
                kind="parallelization_fallback",
                priority="P2",
                title=(
                    f"Recurring AutoParallel fallback to {strategy.upper()}: "
                    f"{reason_code}"
                ),
                records=records,
                occurrence_count=len(records),
                max_representatives=max_representatives,
                details={
                    "selected_strategy": strategy,
                    "reason_code": reason_code,
                },
                suggested_action=(
                    _FALLBACK_ACTIONS.get(
                        reason_code,
                        "Inspect the linked AutoParallel attempts and add a regression "
                        "for this fallback reason.",
                    )
                ),
            )
        )
    return candidates


def _competitiveness_candidates(
    runs: list[Mapping[str, Any]],
    minimum_occurrences: int,
    max_representatives: int,
) -> list[dict[str, Any]]:
    actionable = {
        "robustly_worse": (
            "Inspect the counterfactual placement and largest modeled regression terms, "
            "then add a representative optimization regression."
        ),
        "schedule_sensitive": (
            "Calibrate the baseline-specific scheduling envelope or collect an "
            "authorized representative measurement."
        ),
        "comparison_unavailable": (
            "Add a versioned baseline template or cost coverage for this workload "
            "without silently benchmarking user code."
        ),
    }
    groups: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    resolved_decisions: dict[tuple[str, str], str] = {}
    for run in runs:
        routing = run.get("parallelization")
        if (
            not isinstance(routing, Mapping)
            or routing.get("intent_kind") == "diagnostic_evaluation"
            or routing.get("decision_status")
            not in {"accepted", "fallback_selected"}
        ):
            continue
        decision_id = routing.get("decision_id")
        if not decision_id:
            continue
        key = (str(run.get("cluster_profile")), str(decision_id))
        timestamp = str(run.get("completed_at") or run.get("created_at") or "")
        resolved_decisions[key] = max(resolved_decisions.get(key, ""), timestamp)

    for run in runs:
        routing = run.get("parallelization")
        if (
            not isinstance(routing, Mapping)
            or routing.get("intent_kind") == "diagnostic_evaluation"
        ):
            continue
        verdict = routing.get("competitiveness_verdict")
        if verdict not in actionable:
            continue
        decision_id = routing.get("decision_id")
        timestamp = str(run.get("completed_at") or run.get("created_at") or "")
        if (
            verdict == "comparison_unavailable"
            and decision_id
            and resolved_decisions.get(
                (str(run.get("cluster_profile")), str(decision_id)), ""
            )
            >= timestamp
        ):
            continue
        evidence = str(routing.get("competitiveness_evidence") or "none")
        strategies = routing.get("baseline_strategies", []) or ["unspecified"]
        reason_codes = routing.get("competitiveness_reason_codes", []) or [
            "UNSPECIFIED"
        ]
        for strategy in strategies:
            for reason_code in reason_codes:
                if reason_code == "CALIBRATION_ONLY":
                    continue
                groups[
                    (str(verdict), evidence, str(strategy), str(reason_code))
                ].append(run)

    candidates = []
    for (verdict, evidence, strategy, reason_code), records in groups.items():
        if _distinct_run_count(records) < minimum_occurrences:
            continue
        suggested_action = actionable[verdict]
        if verdict == "robustly_worse":
            title = f"{evidence.capitalize()} AutoParallel regression"
            if evidence == "measured":
                suggested_action = (
                    "Inspect the linked target profiles and placement, localize the "
                    "measured regression, and add a representative optimization "
                    "regression."
                )
        elif verdict == "schedule_sensitive":
            title = "Schedule-sensitive AutoParallel comparison"
        else:
            title = "Unavailable AutoParallel baseline comparison"
        if reason_code == "TARGET_MEASUREMENT_REQUIRED":
            suggested_action = (
                "Collect representative target measurements when authorized, then "
                "complete the baseline comparison."
            )
        candidates.append(
            _run_candidate(
                identifier=(
                    f"competitiveness:{verdict}:{evidence}:{strategy}:{reason_code}"
                ),
                kind="parallelization_competitiveness",
                priority=(
                    "P1"
                    if verdict == "robustly_worse"
                    and evidence in {"measured", "mixed"}
                    else "P2"
                ),
                title=f"{title} against {strategy.upper()}: {reason_code}",
                records=records,
                occurrence_count=len(records),
                max_representatives=max_representatives,
                details={
                    "verdict": verdict,
                    "evidence": evidence,
                    "baseline_strategy": strategy,
                    "reason_code": reason_code,
                },
                suggested_action=suggested_action,
            )
        )
    return candidates


def _measurement_candidates(
    runs: list[Mapping[str, Any]],
    minimum_occurrences: int,
    relative_error_threshold: float,
    max_representatives: int,
) -> list[dict[str, Any]]:
    groups: dict[
        tuple[str, str, str, str],
        list[tuple[Mapping[str, Any], Mapping[str, Any]]],
    ] = defaultdict(list)
    for run in runs:
        for comparison in run.get("measurement_comparisons", []):
            key = (
                str(run.get("cluster_profile")),
                str(comparison.get("configuration_hash")),
                str(comparison.get("metric")),
                str(comparison.get("unit")),
            )
            groups[key].append((run, comparison))

    candidates = []
    for key, observations in groups.items():
        relative_errors = [
            float(comparison["relative_error"])
            for _, comparison in observations
            if comparison.get("relative_error") is not None
        ]
        records = [observation[0] for observation in observations]
        if _distinct_run_count(records) < minimum_occurrences or not relative_errors:
            continue
        mean_relative_error = mean(relative_errors)
        if mean_relative_error < relative_error_threshold:
            continue
        absolute_errors = [
            float(comparison["absolute_error"]) for _, comparison in observations
        ]
        candidates.append(
            _run_candidate(
                identifier=f"measurement:{_canonical_sha256(key)[:16]}",
                kind="measurement_drift",
                priority="P2",
                title=f"Recurring {key[2]} estimate drift on {key[0]}",
                records=records,
                occurrence_count=len(observations),
                max_representatives=max_representatives,
                details={
                    "cluster_profile": key[0],
                    "configuration_hash": key[1],
                    "metric": key[2],
                    "unit": key[3],
                    "mean_absolute_error": mean(absolute_errors),
                    "mean_relative_error": mean_relative_error,
                },
                suggested_action=(
                    "Re-run a held-out calibration measurement, localize the dominant "
                    "estimate error, and validate any cost-model change historically."
                ),
            )
        )
    return candidates


def _invalid_bundle_candidates(
    invalid_bundles: Any, max_representatives: int
) -> list[dict[str, Any]]:
    if not isinstance(invalid_bundles, list):
        raise ValueError("catalog invalid_bundles must be a list")
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for invalid in invalid_bundles:
        error_class = str(invalid.get("error", "unknown invalid bundle")).split(":", 1)[
            0
        ]
        groups[error_class].append(invalid)

    candidates = []
    for error_class, records in groups.items():
        paths = sorted(str(record.get("path")) for record in records)
        candidates.append(
            {
                "id": f"invalid_bundle:{_canonical_sha256(error_class)[:16]}",
                "kind": "invalid_bundle",
                "priority": "P1",
                "title": f"Invalid report bundles: {error_class}",
                "occurrence_count": len(records),
                "affected_run_count": len(records),
                "affected_workload_count": 0,
                "affected_campaign_count": 0,
                "first_seen": None,
                "last_seen": None,
                "cluster_profiles": [],
                "evidence_levels": [],
                "representative_bundles": paths[:max_representatives],
                "details": {"error_class": error_class},
                "suggested_action": (
                    "Inspect the producer and preserve the immutable bundle; fix the "
                    "writer or transport path rather than editing prior evidence."
                ),
            }
        )
    return candidates


def _run_candidate(
    *,
    identifier: str,
    kind: str,
    priority: str,
    title: str,
    records: list[Mapping[str, Any]],
    occurrence_count: int,
    max_representatives: int,
    details: Mapping[str, Any],
    suggested_action: str,
) -> dict[str, Any]:
    unique_runs = {str(record.get("run_id")): record for record in records}
    ordered = sorted(
        unique_runs.values(),
        key=lambda record: str(record.get("completed_at") or record.get("created_at")),
        reverse=True,
    )
    ordered.sort(key=lambda record: not _has_optimizer_replay(record))
    replayable = [record for record in ordered if _has_optimizer_replay(record)]
    timestamps = sorted(
        str(record.get("completed_at") or record.get("created_at"))
        for record in unique_runs.values()
    )
    workload_ids = {
        str(record.get("workload_sha256") or record.get("request_sha256"))
        for record in unique_runs.values()
        if record.get("workload_sha256") is not None
        or record.get("request_sha256") is not None
    }
    campaign_ids = {
        (
            str(record.get("cluster_profile")),
            str(record["campaign"]["id"]),
        )
        for record in unique_runs.values()
        if isinstance(record.get("campaign"), Mapping)
        and record["campaign"].get("id")
    }
    return {
        "id": identifier,
        "kind": kind,
        "priority": priority,
        "title": title,
        "occurrence_count": occurrence_count,
        "affected_run_count": len(unique_runs),
        "affected_workload_count": len(workload_ids),
        "affected_campaign_count": len(campaign_ids),
        "first_seen": timestamps[0] if timestamps else None,
        "last_seen": timestamps[-1] if timestamps else None,
        "cluster_profiles": sorted(
            {
                str(record.get("cluster_profile"))
                for record in unique_runs.values()
                if record.get("cluster_profile") is not None
            }
        ),
        "evidence_levels": sorted(
            {
                str(record.get("evidence"))
                for record in unique_runs.values()
                if record.get("evidence") is not None
            }
        ),
        "replayable_run_count": len(replayable),
        "replayable_representative_bundles": [
            str(record["bundle"])
            for record in replayable[:max_representatives]
            if record.get("bundle") is not None
        ],
        "representative_bundles": [
            str(record["bundle"])
            for record in ordered[:max_representatives]
            if record.get("bundle") is not None
        ],
        "details": dict(details),
        "suggested_action": suggested_action,
    }


def _has_optimizer_replay(record: Mapping[str, Any]) -> bool:
    replay = record.get("optimizer_replay")
    if isinstance(replay, Mapping):
        return replay.get("available") is True
    return "optimizer_replay" in record.get("artifact_roles", [])


def _distinct_run_count(records: list[Mapping[str, Any]]) -> int:
    return len({str(record.get("run_id")) for record in records})


def _latest_campaign_case_runs(
    runs: list[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    latest: dict[tuple[str, ...], Mapping[str, Any]] = {}
    for run in runs:
        campaign = run.get("campaign")
        key: tuple[str, ...]
        if (
            isinstance(campaign, Mapping)
            and campaign.get("id")
            and campaign.get("case_id")
        ):
            key = (
                "campaign_case",
                str(run.get("cluster_profile")),
                str(campaign["id"]),
                str(campaign["case_id"]),
            )
        else:
            key = ("run", str(run.get("run_id")))
        current = latest.get(key)
        timestamp = str(run.get("completed_at") or run.get("created_at") or "")
        current_timestamp = (
            str(current.get("completed_at") or current.get("created_at") or "")
            if current is not None
            else ""
        )
        if current is None or timestamp >= current_timestamp:
            latest[key] = run
    return list(latest.values())


def _deduplicate_finding_observations(
    observations: list[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    latest: dict[tuple[str, ...], tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    for run, finding in observations:
        context_signature = finding.get("context_signature")
        key = (
            (
                "context",
                str(run.get("cluster_profile")),
                str(context_signature),
            )
            if context_signature
            else ("run", str(run.get("run_id")))
        )
        current = latest.get(key)
        timestamp = str(run.get("completed_at") or run.get("created_at") or "")
        current_timestamp = (
            str(current[0].get("completed_at") or current[0].get("created_at") or "")
            if current is not None
            else ""
        )
        if current is None or timestamp >= current_timestamp:
            latest[key] = (run, finding)
    return list(latest.values())


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sorted_counter(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): counter[key] for key in sorted(counter, key=str)}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an AutoParallel improvement queue from a report catalog"
    )
    parser.add_argument("catalog", help="Path to catalog-v1.json")
    parser.add_argument("--output", help="JSON output path")
    parser.add_argument("--markdown-output", help="Markdown output path")
    parser.add_argument("--minimum-occurrences", type=int, default=2)
    parser.add_argument(
        "--measurement-relative-error-threshold", type=float, default=0.2
    )
    parser.add_argument("--max-representatives", type=int, default=3)
    args = parser.parse_args()

    catalog_path = Path(args.catalog).resolve()
    output = (
        Path(args.output)
        if args.output
        else catalog_path.with_name("improvement-queue-v1.json")
    )
    markdown_output = (
        Path(args.markdown_output)
        if args.markdown_output
        else catalog_path.with_name("improvement-queue-v1.md")
    )
    paths = write_improvement_queue(
        _read_json(catalog_path),
        output=output,
        markdown_output=markdown_output,
        minimum_occurrences=args.minimum_occurrences,
        measurement_relative_error_threshold=(
            args.measurement_relative_error_threshold
        ),
        max_representatives=args.max_representatives,
    )
    print("\n".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
