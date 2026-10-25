# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from torch.distributed.tensor.placement_types import Replicate, Shard

ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = ROOT / ".agents/skills/autoparallel-optimizer"


def _load_report_module():
    path = SKILL_ROOT / "scripts/run_report.py"
    spec = importlib.util.spec_from_file_location("autoparallel_skill_run_report", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


report_module = _load_report_module()
AutoParallelRunReport = report_module.AutoParallelRunReport
resolve_report_config = report_module.resolve_report_config


def _load_catalog_module():
    path = SKILL_ROOT / "scripts/build_report_catalog.py"
    spec = importlib.util.spec_from_file_location(
        "autoparallel_skill_build_report_catalog", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


catalog_module = _load_catalog_module()
build_report_catalog = catalog_module.build_report_catalog
write_report_catalog = catalog_module.write_report_catalog


def _load_skill_script(module_name, filename):
    scripts_path = SKILL_ROOT / "scripts"
    sys.path.insert(0, str(scripts_path))
    try:
        path = scripts_path / filename
        spec = importlib.util.spec_from_file_location(module_name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts_path))


triage_module = _load_skill_script(
    "autoparallel_skill_build_improvement_queue", "build_improvement_queue.py"
)
build_improvement_queue = triage_module.build_improvement_queue
render_improvement_queue = triage_module.render_improvement_queue

feedback_module = _load_skill_script(
    "autoparallel_skill_refresh_report_feedback", "refresh_report_feedback.py"
)
refresh_report_feedback = feedback_module.refresh_report_feedback

recovery_module = _load_skill_script(
    "autoparallel_skill_recover_interrupted_report",
    "recover_interrupted_report.py",
)
recover_interrupted_report = recovery_module.recover_interrupted_report

policy_module = _load_skill_script(
    "autoparallel_skill_parallelization_policy", "parallelization_policy.py"
)
default_acceptance_criteria = policy_module.default_acceptance_criteria
default_competitiveness_criteria = policy_module.default_competitiveness_criteria
default_attempt_budget = policy_module.default_attempt_budget
classify_modeled_competitiveness = policy_module.classify_modeled_competitiveness
find_pareto_dominating_baseline = policy_module.find_pareto_dominating_baseline
recommended_fallback = policy_module.recommended_fallback

routing_eval_module = _load_skill_script(
    "autoparallel_skill_evaluate_parallelization_routing",
    "evaluate_parallelization_routing.py",
)
evaluate_routing_responses = routing_eval_module.evaluate_responses

routing_validation_module = _load_skill_script(
    "autoparallel_skill_validate_parallelization_routing",
    "validate_parallelization_routing.py",
)
validate_routing_classification = (
    routing_validation_module.validate_routing_classification
)


def _load_session_module():
    path = SKILL_ROOT / "scripts/report_session.py"
    scripts_path = str(path.parent)
    sys.path.insert(0, scripts_path)
    try:
        spec = importlib.util.spec_from_file_location(
            "autoparallel_skill_report_session", path
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(scripts_path)


session_module = _load_session_module()
AutoParallelReportSession = session_module.AutoParallelReportSession
strategy_space_summary = session_module._strategy_space_summary


def _write_profiles(path, profiles, *, default_profile="local"):
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "default_profile": default_profile,
                "profiles": profiles,
            }
        )
    )


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metrics():
    return {
        "cost_model": {"status": "nccl_explicit", "topology": "test"},
        "reordering": {"collective_bucketing": True},
        "graphs": [
            {
                "phase": "forward",
                "total_time": 10.0,
                "compute_time": 8.0,
                "communication_time": 3.0,
                "exposed_comm_time": 2.0,
                "peak_memory": 1024,
            },
            {
                "phase": "backward",
                "total_time": 20.0,
                "compute_time": 15.0,
                "communication_time": 7.0,
                "exposed_comm_time": 5.0,
                "peak_memory": 2048,
            },
        ],
    }


class _FakeOptimizer:
    def save(self, path):
        Path(path).write_bytes(b"trusted optimizer state")

    def save_replay(self, path):
        Path(path).write_bytes(b"trusted optimizer replay")

    def save_placements(self, path):
        Path(path).write_text('{"mesh": [8]}\n')

    def get_json(self):
        return {
            "summary": {
                "compute": 11.0,
                "comm": 7.0,
                "transition": 1.0,
                "search_objective": 19.0,
                "total": 19.0,
                "parameter_storage": {
                    "global_bytes": 80,
                    "local_bytes": 80,
                    "local_to_global_fraction": 1.0,
                    "tensor_count": 1,
                },
            },
            "nodes": [],
            "edges": [],
        }


class _UnusedMesh:
    shape = (8,)


class _FakeNode:
    def __init__(self, name, target):
        self.name = name
        self.op = "call_function"
        self.target = target
        self.meta = {}


class _UnexpectedOptimizerUse:
    def save_placements(self, _path):
        raise AssertionError("non-reporting ranks must not serialize placements")

    def get_json(self):
        raise AssertionError("non-reporting ranks must not export optimizer JSON")


def test_strategy_space_summary_separates_coverage_from_selection():
    nodes = [
        _FakeNode("replicated_choice", "aten.mm.default"),
        _FakeNode("sharded_choice", "aten.mul.Tensor"),
        _FakeNode("replicate_only", "aten.scalar_tensor.default"),
    ]

    def option(placement):
        spec = SimpleNamespace(placements=(placement,))
        return SimpleNamespace(output_specs=spec)

    replicated = option(Replicate())
    sharded = option(Shard(0))
    replicate_only = option(Replicate())
    optimizer = SimpleNamespace(
        nodes=nodes,
        selected_keys=[(0, 0, 0, 0), (1, 0, 0, 0), (2, 0, 0, 0)],
        strats={
            nodes[0]: SimpleNamespace(strategies=[replicated, sharded]),
            nodes[1]: SimpleNamespace(strategies=[replicated, sharded]),
            nodes[2]: SimpleNamespace(strategies=[replicate_only]),
        },
    )
    selected = {
        optimizer.selected_keys[0]: replicated,
        optimizer.selected_keys[1]: sharded,
        optimizer.selected_keys[2]: replicate_only,
    }
    optimizer._resolve_decision_var = lambda key: SimpleNamespace(
        strategy=selected[key]
    )

    summary = strategy_space_summary(optimizer)

    assert summary["selected_tensor_node_count"] == 3
    assert summary["nodes_with_sharded_output_option"] == 2
    assert summary["selected_replicated_node_count"] == 2
    assert summary["selected_replicated_with_sharded_output_option"] == 1
    assert summary["replicate_only_option_set_count"] == 1
    assert summary["top_selected_replicated_operations"] == [
        {"operation": "aten.mm.default", "count": 1}
    ]


def _write_recurring_failure(config, run_id, message):
    report = AutoParallelRunReport(
        request={"workload": {"mode": "training"}},
        config=config,
        run_id=run_id,
    )
    with pytest.raises(RuntimeError):
        with report:
            report.mark_stage("lowered", evidence_level="lowered")
            report.mark_substage("compile")
            raise RuntimeError(message)
    return report


def _parallelization(decision_id):
    return {
        "decision_id": decision_id,
        "intent": {
            "kind": "distributed_training",
            "requested_strategy": "automatic",
        },
        "acceptance_criteria": default_acceptance_criteria(),
    }


def _latency(optimistic, expected, pessimistic, *, error=0.0):
    return {
        "unit": "us",
        "optimistic": optimistic,
        "expected": expected,
        "pessimistic": pessimistic,
        "calibration_relative_error": error,
        "schedule_model": "test-v1",
    }


def _baseline(
    strategy="ddp",
    *,
    feasibility="feasible",
    latency=None,
    memory=None,
):
    default_memory = {
        "scope": "training_state",
        "fits": feasibility != "infeasible",
    }
    return {
        "strategy": strategy,
        "template_id": f"pytorch-{strategy}",
        "template_version": "1",
        "configuration_hash": f"sha256:{'a' * 64}",
        "feasibility": feasibility,
        "latency": latency,
        "memory": dict(memory or default_memory),
    }


def _autoparallel_estimate(latency=None):
    return {
        "latency": latency or _latency(70, 80, 90),
        "memory": {"scope": "post_reordering_peak", "fits": True},
    }


def _record_robust_competitiveness(report, strategy="ddp"):
    report.record_competitiveness_assessment(
        "robustly_competitive",
        evidence="modeled",
        autoparallel_estimate=_autoparallel_estimate(),
        baselines=[_baseline(strategy, latency=_latency(110, 120, 130))],
        reason_codes=("MODELED_ROBUSTLY_COMPETITIVE",),
    )


def test_report_config_resolves_environment_match_and_precedence(tmp_path):
    profiles_path = tmp_path / "profiles.json"
    _write_profiles(
        profiles_path,
        {
            "production": {
                "match_env": {"TEST_AUTOPARALLEL_SITE": "production"},
                "report_root": "profile-reports",
                "topology_profile": "h100-test-v1",
            }
        },
        default_profile="production",
    )

    matched = resolve_report_config(
        profiles_path=profiles_path,
        environ={"TEST_AUTOPARALLEL_SITE": "production"},
        cwd=tmp_path,
    )
    assert matched.profile_name == "production"
    assert matched.selection_source == "environment_match"
    assert matched.report_root == (tmp_path / "profile-reports").resolve()
    assert matched.config_sources[0] == "builtin"
    assert re.fullmatch(r"external:sha256:[0-9a-f]{64}", matched.config_sources[1])
    assert str(tmp_path) not in " ".join(matched.config_sources)

    report = AutoParallelRunReport(
        request={"workload": {"mode": "training"}},
        config=matched,
        run_id="redacted-config-source",
    )
    with report:
        pass
    payload = json.loads((report.path / "report.json").read_text())
    assert payload["cluster"]["config_sources"] == list(matched.config_sources)
    assert str(tmp_path) not in json.dumps(payload["cluster"])

    environment_override = resolve_report_config(
        profiles_path=profiles_path,
        environ={
            "AUTOPARALLEL_CLUSTER_PROFILE": "production",
            "AUTOPARALLEL_REPORT_ROOT": str(tmp_path / "environment-reports"),
        },
    )
    assert environment_override.selection_source == "environment"
    assert environment_override.report_root_source == "environment"

    explicit_override = resolve_report_config(
        profile_name="local",
        profiles_path=profiles_path,
        report_root=tmp_path / "explicit-reports",
        environ={
            "AUTOPARALLEL_CLUSTER_PROFILE": "production",
            "AUTOPARALLEL_REPORT_ROOT": str(tmp_path / "environment-reports"),
        },
    )
    assert explicit_override.profile_name == "local"
    assert explicit_override.selection_source == "explicit"
    assert explicit_override.report_root_source == "explicit"


def test_report_config_rejects_ambiguous_environment_matches(tmp_path):
    profiles_path = tmp_path / "profiles.json"
    _write_profiles(
        profiles_path,
        {
            "cluster-a": {
                "match_env": {"TEST_AUTOPARALLEL_SITE": "shared"},
                "report_root": "a",
            },
            "cluster-b": {
                "match_env": {"TEST_AUTOPARALLEL_SITE": "shared"},
                "report_root": "b",
            },
        },
        default_profile="cluster-a",
    )

    with pytest.raises(ValueError, match="Multiple AutoParallel cluster profiles"):
        resolve_report_config(
            profiles_path=profiles_path,
            environ={"TEST_AUTOPARALLEL_SITE": "shared"},
            cwd=tmp_path,
        )


def test_parallelization_policy_defaults_and_fallback_choice():
    criteria = default_acceptance_criteria(
        memory_limit_bytes=80 * 1024**3,
        performance_target={"step_time_us": 1000},
    )

    assert criteria["minimum_evidence"] == "executed_on_target"
    assert criteria["memory"]["per_rank_limit_bytes"] == 80 * 1024**3
    assert criteria["memory"]["scope"] == "intended_training_step"
    assert criteria["memory"]["optimizer_state"] == (
        "include_or_conservatively_estimate"
    )
    assert criteria["memory"]["operational_headroom"] == "required"
    assert criteria["performance"] == {
        "required": True,
        "target": {"step_time_us": 1000},
        "scope": "intended_training_step",
        "optimizer_update": "include_or_calibrate_when_training",
        "optimizer_implementation": "preserve_configuration_and_record_realized",
        "time_to_ready": "record_and_amortize_when_workload_duration_known",
    }
    assert criteria["relative_quality"] == default_competitiveness_criteria()
    assert criteria["relative_quality"]["baseline_benchmarking"] == (
        "explicit_authorization_only"
    )
    assert criteria["relative_quality"]["reject_when_baseline_pareto_dominates"]
    assert default_attempt_budget()["targeted_semantics_preserving_retries"] == 1
    assert (
        recommended_fallback(
            replicated_training_state_fits=True,
            parameter_sharding_required=False,
        )
        == "ddp"
    )
    assert (
        recommended_fallback(
            replicated_training_state_fits=False,
            parameter_sharding_required=False,
        )
        == "fsdp"
    )
    with pytest.raises(ValueError, match="memory assessment"):
        recommended_fallback(
            replicated_training_state_fits=None,
            parameter_sharding_required=False,
        )


def test_modeled_competitiveness_uses_calibrated_schedule_envelopes():
    robust = classify_modeled_competitiveness(
        _latency(60, 70, 80, error=0.05),
        [_baseline(latency=_latency(120, 130, 140, error=0.05))],
    )
    assert robust["verdict"] == "robustly_competitive"
    assert robust["reason_codes"] == ["MODELED_ROBUSTLY_COMPETITIVE"]

    worse = classify_modeled_competitiveness(
        _latency(180, 190, 200),
        [_baseline(strategy="fsdp", latency=_latency(100, 110, 120))],
    )
    assert worse["verdict"] == "robustly_worse"

    overlapping = classify_modeled_competitiveness(
        _latency(90, 110, 130),
        [_baseline(latency=_latency(100, 110, 120))],
    )
    assert overlapping["verdict"] == "schedule_sensitive"

    infeasible = classify_modeled_competitiveness(
        _latency(90, 100, 110),
        [_baseline(feasibility="infeasible")],
    )
    assert infeasible["verdict"] == "baseline_infeasible"

    unavailable = classify_modeled_competitiveness(None, [])
    assert unavailable["verdict"] == "comparison_unavailable"


def test_pareto_dominance_requires_robust_no_worse_evidence():
    autoparallel = {
        "latency": _latency(446_000, 448_000, 449_000),
        "memory": {"projected_peak_with_adamw_moments_bytes": 61_000},
        "time_to_ready": {
            "minimum_first_iteration": 80.0,
            "maximum_first_iteration": 81.0,
        },
    }
    fsdp = {
        **_baseline(
            strategy="fsdp",
            latency=_latency(427_000, 428_000, 429_000),
            memory={"projected_peak_with_adamw_moments_bytes": 38_000},
        ),
        "time_to_ready": {
            "minimum_first_iteration": 7.0,
            "maximum_first_iteration": 8.0,
        },
    }

    dominance = find_pareto_dominating_baseline(autoparallel, [fsdp])

    assert dominance["strategy"] == "fsdp"
    assert dominance["reason_code"] == "BASELINE_PARETO_DOMINATES"
    assert set(dominance["comparisons"]) == {
        "steady_state_latency",
        "peak_memory",
        "time_to_ready",
    }
    ddp = {
        **_baseline(
            strategy="ddp",
            latency=_latency(437_000, 438_000, 439_000),
            memory={"projected_peak_with_adamw_moments_bytes": 45_000},
        ),
        "time_to_ready": {
            "minimum_first_iteration": 9.0,
            "maximum_first_iteration": 10.0,
        },
    }
    assert (
        find_pareto_dominating_baseline(autoparallel, [ddp, fsdp])["strategy"] == "fsdp"
    )
    overlapping = {
        **fsdp,
        "latency": _latency(427_000, 440_000, 447_000),
    }
    assert find_pareto_dominating_baseline(autoparallel, [overlapping]) is None


def test_parallelization_routing_behavior_eval_accepts_gold_and_rejects_bad():
    suite = json.loads((SKILL_ROOT / "references/agent-routing-evals.json").read_text())
    gold = {
        "cases": [{"id": case["id"], **case["expected"]} for case in suite["cases"]]
    }

    result = evaluate_routing_responses(suite, gold)

    assert result["passed"]
    assert result["score"] == 1.0
    assert result["field_score"] == 1.0
    assert set(result["field_scores"]) == {
        "invoke_skill",
        "autoparallel_disposition",
        "autoparallel_outcome",
        "competitiveness_verdict",
        "benchmark_action",
        "selected_strategy",
        "report_required",
    }

    for response in gold["cases"]:
        assert validate_routing_classification(response)["valid"]

    gold["cases"][0]["selected_strategy"] = "ddp"
    failed = evaluate_routing_responses(suite, gold)
    assert not failed["passed"]
    assert failed["score"] < 1.0
    assert failed["field_score"] < 1.0

    duplicate = evaluate_routing_responses(
        suite,
        {"cases": [*gold["cases"], gold["cases"][0]]},
    )
    assert not duplicate["passed"]
    assert duplicate["duplicate_case_ids"] == [gold["cases"][0]["id"]]


def test_parallelization_routing_validator_uses_evidence_invariants():
    decision = {
        "invoke_skill": True,
        "autoparallel_disposition": "attempt",
        "autoparallel_outcome": "failed",
        "competitiveness_verdict": "comparison_unavailable",
        "benchmark_action": "none",
        "selected_strategy": "ddp",
        "report_required": True,
    }

    assert validate_routing_classification(
        decision,
        {
            "autoparallel_attempted": True,
            "semantic_violation_observed": True,
            "qualified_fallback": "ddp",
        },
    )["valid"]

    missing_evidence = validate_routing_classification(
        decision,
        {
            "autoparallel_attempted": True,
            "required_evidence_missing": True,
        },
    )
    assert not missing_evidence["valid"]
    assert {error["code"] for error in missing_evidence["errors"]} == {
        "MISSING_EVIDENCE_OUTCOME"
    }


def test_parallelization_routing_validator_preserves_authority_and_pending_work():
    explicit_fsdp = {
        "invoke_skill": True,
        "autoparallel_disposition": "attempt",
        "autoparallel_outcome": "inconclusive",
        "competitiveness_verdict": "comparison_unavailable",
        "benchmark_action": "none",
        "selected_strategy": "ddp",
        "report_required": True,
    }
    result = validate_routing_classification(
        explicit_fsdp,
        {"requested_strategy": "fsdp"},
    )
    assert not result["valid"]
    assert {error["code"] for error in result["errors"]} == {
        "EXPLICIT_STRATEGY_NOT_PRESERVED",
        "EXPLICIT_STRATEGY_REPLACED",
    }

    pending_benchmark = {
        "invoke_skill": True,
        "autoparallel_disposition": "attempt",
        "autoparallel_outcome": "inconclusive",
        "competitiveness_verdict": "schedule_sensitive",
        "benchmark_action": "run_authorized",
        "selected_strategy": "ddp",
        "report_required": True,
    }
    result = validate_routing_classification(
        pending_benchmark,
        {
            "benchmark_needed": True,
            "benchmark_authorized": True,
            "benchmark_pending": True,
        },
    )
    assert not result["valid"]
    assert {error["code"] for error in result["errors"]} == {
        "PENDING_BENCHMARK_DECISION",
        "PENDING_SELECTION",
    }


def test_parallelization_routing_validator_rejects_smoke_workload_selection():
    decision = {
        "invoke_skill": True,
        "autoparallel_disposition": "attempt",
        "autoparallel_outcome": "inconclusive",
        "competitiveness_verdict": "comparison_unavailable",
        "benchmark_action": "none",
        "selected_strategy": "ddp",
        "report_required": True,
    }

    result = validate_routing_classification(
        decision,
        {"workload_representative": False},
    )

    assert not result["valid"]
    assert {error["code"] for error in result["errors"]} == {
        "UNREPRESENTATIVE_WORKLOAD_SELECTION"
    }


def test_parallelization_routing_validator_requires_consistent_decision_card():
    decision = {
        "invoke_skill": True,
        "autoparallel_disposition": "attempt",
        "autoparallel_outcome": "accepted",
        "competitiveness_verdict": "robustly_competitive",
        "benchmark_action": "none",
        "selected_strategy": "autoparallel",
        "report_required": True,
        "decision_card": {
            "evidence_level": "measured",
            "boundary": "complete AdamW step",
            "boundary_matched": True,
            "reason_codes": ["MEASURED_ROBUSTLY_COMPETITIVE"],
            "missing_evidence": [],
            "fallback_strategy": "none",
            "next_control": None,
        },
    }

    assert validate_routing_classification(decision, require_decision_card=True)[
        "valid"
    ]

    decision["decision_card"]["missing_evidence"] = ["optimizer update"]
    result = validate_routing_classification(decision, require_decision_card=True)
    assert not result["valid"]
    assert {error["code"] for error in result["errors"]} == {
        "ACCEPTED_CARD_HAS_EVIDENCE_GAP"
    }


def test_parallelization_routing_validator_checks_recurring_evidence_gaps():
    decision = {
        "invoke_skill": True,
        "autoparallel_disposition": "attempt",
        "autoparallel_outcome": "inconclusive",
        "competitiveness_verdict": "comparison_unavailable",
        "benchmark_action": "none",
        "selected_strategy": "ddp",
        "report_required": True,
        "decision_card": {
            "evidence_level": "measured",
            "boundary": "complete AdamW step",
            "boundary_matched": False,
            "reason_codes": [
                "OPTIMIZER_PERFORMANCE_UNMODELED",
                "COMPOSED_FULL_STEP_BASE_DRIFT",
            ],
            "missing_evidence": ["aligned optimizer update", "current base"],
            "fallback_strategy": "ddp",
            "next_control": "Refresh the partial-step base and measure the update.",
        },
    }
    evidence = {
        "optimizer_update_modeled_or_measured": False,
        "composed_prediction": True,
        "base_measurement_contemporaneous": False,
        "qualified_fallback": "ddp",
    }

    assert validate_routing_classification(
        decision, evidence, require_decision_card=True
    )["valid"]

    decision["decision_card"]["reason_codes"] = []
    result = validate_routing_classification(
        decision, evidence, require_decision_card=True
    )
    assert not result["valid"]
    assert {error["code"] for error in result["errors"]} == {
        "CARD_OMITS_BASE_DRIFT_REASON",
        "CARD_OMITS_OPTIMIZER_BOUNDARY_REASON",
    }


def test_parallelization_routing_validator_rejects_unqualified_fallback_template():
    decision = {
        "invoke_skill": True,
        "autoparallel_disposition": "attempt",
        "autoparallel_outcome": "below_bar",
        "competitiveness_verdict": "robustly_worse",
        "benchmark_action": "none",
        "selected_strategy": "ddp",
        "report_required": True,
        "decision_card": {
            "evidence_level": "measured",
            "boundary": "gradient accumulation cycle",
            "boundary_matched": True,
            "reason_codes": ["MEASURED_ROBUSTLY_WORSE"],
            "missing_evidence": [],
            "fallback_strategy": "ddp",
            "next_control": None,
        },
    }

    result = validate_routing_classification(
        decision,
        {
            "alternating_no_sync": True,
            "ddp_static_graph": True,
            "exact_static_graph_template_validated": False,
        },
        require_decision_card=True,
    )
    assert not result["valid"]
    assert {error["code"] for error in result["errors"]} == {"UNQUALIFIED_DDP_TEMPLATE"}


def test_operational_metadata_does_not_split_execution_configuration_hash(tmp_path):
    hashes = []
    for index, interval in enumerate((60, 600)):
        profiles_path = tmp_path / f"profiles-{index}.json"
        _write_profiles(
            profiles_path,
            {
                "test": {
                    "report_root": str(tmp_path / f"reports-{index}"),
                    "topology_profile": "test-topology",
                    "feedback": {"refresh_interval_seconds": interval},
                    "diagnostics": {"high_communication_fraction": 0.1 * (index + 1)},
                    "routing": {"minutes_without_progress": 5 * (index + 1)},
                }
            },
            default_profile="test",
        )
        config = resolve_report_config(
            profile_name="test",
            profiles_path=profiles_path,
            environ={},
        )
        with AutoParallelRunReport(
            request={
                "workload": {"mode": "training"},
                "campaign": {"id": f"campaign-{index}", "case_id": "same-case"},
            },
            parallelization=_parallelization(f"decision-{index}"),
            provenance={"compiler_config": {"overlap_scheduling": False}},
            config=config,
            run_id=f"configuration-run-{index}",
        ) as report:
            report.record_runtime_context(
                {"status": "nccl_explicit", "source": "explicit"}
            )
            hashes.append(report.report["provenance"]["configuration_sha256"])

    assert hashes[0] == hashes[1]


def test_report_bundle_records_plan_metrics_validation_and_hashes(tmp_path):
    placements = tmp_path / "placements.json"
    placements.write_text('{"mesh": [8]}\n')
    config = resolve_report_config(
        profile_name="local",
        report_root=tmp_path / "reports",
        environ={},
    )

    with AutoParallelRunReport(
        request={
            "workload": {"mode": "training", "global_batch": 64},
            "mesh": {"shape": [8], "names": ["dp"]},
            "constraints": {},
            "spmd_assumptions": [],
        },
        provenance={"git_revision": "test", "pytorch_build": "test"},
        config=config,
        run_id="test-run",
    ) as report:
        report.mark_stage("captured", evidence_level="captured")
        report.record_plan(
            {
                "summary": {
                    "compute": 11.0,
                    "comm": 7.0,
                    "transition": 1.0,
                    "search_objective": 19.0,
                    "parameter_storage": {
                        "global_bytes": 80,
                        "local_bytes": 10,
                        "local_to_global_fraction": 0.125,
                        "tensor_count": 1,
                    },
                },
                "nodes": [],
                "edges": [],
            },
            placements_path=placements,
        )
        report.record_metrics(_metrics())
        report.record_validation(
            "numerical",
            {
                "inputs_hash": "sha256:test",
                "random_seed": 0,
                "tolerances": {"rtol": 1e-5, "atol": 1e-6},
                "outputs_match": True,
                "gradients_match": True,
            },
            evidence_level="numerically_checked",
        )
        report.add_finding(
            "EXTRAPOLATED_CRITICAL_TERM",
            "The dominant collective is outside its calibration range",
            context={"node": "all_gather_1", "share_of_critical_path": 0.4},
            suggested_action="Benchmark this collective configuration",
        )

    payload = json.loads((report.path / "report.json").read_text())
    manifest = json.loads((report.path / "manifest.json").read_text())

    assert payload["status"] == "completed"
    assert payload["stage"] == "numerically_checked"
    assert payload["evidence"]["highest_level"] == "numerically_checked"
    assert payload["cluster"]["profile"] == "local"
    assert {artifact["name"] for artifact in payload["artifacts"]} == {
        "optimizer",
        "placements",
        "metrics",
        "validation-numerical",
    }
    assert payload["evaluation"]["graphs"][0]["phase"] == "forward"
    assert payload["evaluation"]["graphs"][1]["phase"] == "backward"

    for entry in manifest["files"]:
        assert entry["sha256"] == _sha256(report.path / entry["path"])
    assert "EXTRAPOLATED_CRITICAL_TERM" in (report.path / "summary.md").read_text()

    with pytest.raises(RuntimeError, match="already been finalized"):
        report.add_finding("LATE_FINDING", "too late")


def test_parallelization_decisions_and_fallbacks_are_cataloged(tmp_path):
    report_root = tmp_path / "reports"
    config = resolve_report_config(
        profile_name="local",
        report_root=report_root,
        environ={},
    )
    with AutoParallelRunReport(
        request={"workload": {"mode": "training"}},
        config=config,
        run_id="accepted-autoparallel",
        parallelization=_parallelization("decision-accepted"),
    ) as accepted:
        accepted.mark_stage("executed_on_target", evidence_level="executed_on_target")
        _record_robust_competitiveness(accepted)
        accepted.record_autoparallel_outcome("accepted")
        accepted.record_parallelization_decision("autoparallel", "accepted")

    failed_attempt = AutoParallelRunReport(
        request={"workload": {"mode": "training"}},
        config=config,
        run_id="failed-autoparallel",
        parallelization=_parallelization("decision-fallback"),
    )
    error = RuntimeError("capture failed")
    failed_attempt.record_autoparallel_outcome(
        "failed",
        reason_codes=("CAPTURE_UNSUPPORTED",),
    )
    failed_attempt.record_failure(error, error.__traceback__)

    with AutoParallelRunReport(
        request={"workload": {"mode": "training"}},
        config=config,
        run_id="accepted-fsdp",
        parallelization=_parallelization("decision-fallback"),
    ) as fallback:
        fallback.record_fallback_outcome(
            "fsdp",
            "accepted",
            reason_codes=("CAPTURE_UNSUPPORTED",),
            evidence_level="executed_on_target",
            related_run_id="failed-autoparallel",
        )
        fallback.record_parallelization_decision(
            "fsdp",
            "accepted",
            reason_codes=("CAPTURE_UNSUPPORTED",),
        )

    catalog = build_report_catalog(report_root)
    summary = catalog["summary"]["parallelization"]

    assert summary == {
        "report_count": 3,
        "decision_count": 2,
        "diagnostic_evaluation_count": 0,
        "autoparallel_outcome_counts": {"accepted": 1, "failed": 1},
        "autoparallel_reason_code_counts": {"CAPTURE_UNSUPPORTED": 1},
        "competitiveness_verdict_counts": {"robustly_competitive": 1},
        "competitiveness_evidence_counts": {"modeled": 1},
        "competitiveness_reason_code_counts": {"MODELED_ROBUSTLY_COMPETITIVE": 1},
        "baseline_strategy_counts": {"ddp": 1},
        "baseline_feasibility_counts": {"feasible": 1},
        "selected_strategy_counts": {"autoparallel": 1, "fsdp": 1},
        "decision_status_counts": {"accepted": 2},
        "decision_reason_code_counts": {"CAPTURE_UNSUPPORTED": 1},
        "fallback_strategy_counts": {"fsdp": 1},
        "fallback_outcome_counts": {"accepted": 1},
        "fallback_reason_code_counts": {"CAPTURE_UNSUPPORTED": 1},
        "incomplete_decision_count": 0,
        "missing_competitiveness_count": 0,
    }
    fallback_record = next(
        run for run in catalog["runs"] if run["run_id"] == "accepted-fsdp"
    )
    assert fallback_record["parallelization"]["decision_id"] == ("decision-fallback")
    assert fallback_record["parallelization"]["selected_strategy"] == "fsdp"
    assert fallback_record["findings"][0]["code"] == "AUTOPARALLEL_FALLBACK"
    assert fallback_record["findings"][0]["severity"] == "warning"
    assert fallback_record["findings"][0]["context_signature"].startswith("sha256:")
    assert "Fallback: FSDP (accepted)" in (fallback.path / "summary.md").read_text()
    assert (
        "Competitiveness: robustly_competitive"
        in (accepted.path / "summary.md").read_text()
    )


def test_autoparallel_acceptance_requires_competitiveness_gate(tmp_path):
    config = resolve_report_config(
        profile_name="local",
        report_root=tmp_path / "reports",
        environ={},
    )
    report = AutoParallelRunReport(
        request={"workload": {"mode": "training"}},
        config=config,
        run_id="competitiveness-gate",
        parallelization=_parallelization("competitiveness-decision"),
    )
    report.mark_stage("executed_on_target", evidence_level="executed_on_target")
    report.record_autoparallel_outcome("accepted")

    with pytest.raises(ValueError, match="competitiveness assessment"):
        report.record_parallelization_decision("autoparallel", "accepted")

    report.record_competitiveness_assessment(
        "schedule_sensitive",
        evidence="modeled",
        autoparallel_estimate=_autoparallel_estimate(_latency(90, 110, 130)),
        baselines=[_baseline(latency=_latency(100, 110, 120))],
        reason_codes=("MODELED_SCHEDULE_SENSITIVE",),
    )
    with pytest.raises(ValueError, match="does not meet the acceptance bar"):
        report.record_parallelization_decision("autoparallel", "accepted")
    report.complete()

    dominated = AutoParallelRunReport(
        request={"workload": {"mode": "training"}},
        config=config,
        run_id="competitiveness-pareto-dominated",
        parallelization=_parallelization("competitiveness-pareto-decision"),
    )
    dominated.mark_stage("executed_on_target", evidence_level="executed_on_target")
    with pytest.raises(ValueError, match="requires modeled or measured evidence"):
        dominated.record_competitiveness_assessment(
            "pareto_dominated",
            evidence="none",
            reason_codes=("BASELINE_PARETO_DOMINATES",),
        )
    dominated.record_competitiveness_assessment(
        "pareto_dominated",
        evidence="measured",
        autoparallel_estimate=_autoparallel_estimate(_latency(120, 130, 140)),
        baselines=[_baseline(strategy="fsdp", latency=_latency(80, 90, 100))],
        reason_codes=("BASELINE_PARETO_DOMINATES",),
    )
    dominated.record_autoparallel_outcome("below_bar")
    dominated.record_fallback_outcome("fsdp", "accepted")
    dominated.record_parallelization_decision("fsdp", "fallback_selected")
    dominated.complete()

    accepted = AutoParallelRunReport(
        request={"workload": {"mode": "training"}},
        config=config,
        run_id="competitiveness-accepted",
        parallelization=_parallelization("competitiveness-accepted-decision"),
    )
    accepted.mark_stage("executed_on_target", evidence_level="executed_on_target")
    _record_robust_competitiveness(accepted)
    accepted.record_autoparallel_outcome("accepted")
    accepted.record_parallelization_decision("autoparallel", "accepted")
    accepted.complete()


def test_catalog_terminal_decision_survives_later_exploratory_report(tmp_path):
    config = resolve_report_config(
        profile_name="local",
        report_root=tmp_path / "reports",
        environ={},
    )
    parallelization = _parallelization("stable-terminal-decision")
    with AutoParallelRunReport(
        request={"workload": {"mode": "training"}},
        config=config,
        run_id="terminal-decision",
        parallelization=parallelization,
    ) as report:
        report.mark_stage("executed_on_target", evidence_level="executed_on_target")
        _record_robust_competitiveness(report)
        report.record_autoparallel_outcome("accepted")
        report.record_parallelization_decision("autoparallel", "accepted")

    with AutoParallelRunReport(
        request={"workload": {"mode": "training"}},
        config=config,
        run_id="later-exploration",
        parallelization=parallelization,
    ) as report:
        report.record_competitiveness_assessment(
            "comparison_unavailable",
            evidence="none",
            reason_codes=("BASELINE_COMPARISON_UNAVAILABLE",),
        )
        report.record_autoparallel_outcome("inconclusive")
        report.record_parallelization_decision("none", "inconclusive")

    summary = build_report_catalog(config.report_root)["summary"]["parallelization"]

    assert summary["decision_count"] == 1
    assert summary["selected_strategy_counts"] == {"autoparallel": 1}
    assert summary["decision_status_counts"] == {"accepted": 1}
    assert summary["autoparallel_outcome_counts"] == {"accepted": 1}
    assert summary["competitiveness_verdict_counts"] == {"robustly_competitive": 1}


def test_diagnostic_evaluation_is_not_counted_as_routing_gap(tmp_path):
    config = resolve_report_config(
        profile_name="local",
        report_root=tmp_path / "reports",
        environ={},
    )
    for index in range(2):
        diagnostic = _parallelization(f"diagnostic-evaluation-{index}")
        diagnostic["intent"]["kind"] = "diagnostic_evaluation"
        with AutoParallelRunReport(
            request={"workload": {"mode": "training"}},
            config=config,
            run_id=f"diagnostic-evaluation-{index}",
            parallelization=diagnostic,
        ) as report:
            report.record_competitiveness_assessment(
                "comparison_unavailable",
                evidence="none",
                reason_codes=("BASELINE_COMPARISON_UNAVAILABLE",),
            )
            report.record_autoparallel_outcome("inconclusive")
            report.record_parallelization_decision("none", "inconclusive")

    catalog = build_report_catalog(config.report_root)
    summary = catalog["summary"]["parallelization"]
    queue = build_improvement_queue(catalog, minimum_occurrences=2)

    assert summary["decision_count"] == 0
    assert summary["diagnostic_evaluation_count"] == 2
    assert summary["decision_status_counts"] == {}
    assert not any(
        candidate["kind"] == "parallelization_competitiveness"
        for candidate in queue["candidates"]
    )


def test_diagnostic_report_cannot_hide_linked_routing_decision(tmp_path):
    config = resolve_report_config(
        profile_name="local",
        report_root=tmp_path / "reports",
        environ={},
    )
    for run_id, intent_kind in (
        ("routing-decision", "distributed_training"),
        ("linked-diagnostic", "diagnostic_evaluation"),
    ):
        parallelization = _parallelization("shared-decision")
        parallelization["intent"]["kind"] = intent_kind
        with AutoParallelRunReport(
            request={"workload": {"mode": "training"}},
            config=config,
            run_id=run_id,
            parallelization=parallelization,
        ) as report:
            report.record_autoparallel_outcome("inconclusive")
            report.record_parallelization_decision("none", "inconclusive")

    summary = build_report_catalog(config.report_root)["summary"]["parallelization"]

    assert summary["decision_count"] == 1
    assert summary["diagnostic_evaluation_count"] == 0


def test_report_bundle_preserves_uncaught_failure_and_reraises(tmp_path):
    config = resolve_report_config(
        profile_name="local",
        report_root=tmp_path / "reports",
        environ={},
    )
    report = AutoParallelRunReport(
        request={"workload": {"mode": "training"}},
        config=config,
        run_id="failed-run",
    )

    with pytest.raises(ValueError, match="capture failed"):
        with report:
            report.mark_substage("capture")
            raise ValueError("capture failed")

    payload = json.loads((report.path / "report.json").read_text())
    failure_path = report.path / "artifacts/failure.json"
    failure = json.loads(failure_path.read_text())

    assert payload["status"] == "failed"
    assert payload["stage"] == "started"
    assert payload["failure"] == {
        "stage": "started",
        "substage": "capture",
        "type": "ValueError",
        "message": "capture failed",
        "signature": payload["failure"]["signature"],
        "artifact": "failure",
    }
    assert payload["failure"]["signature"].startswith("sha256:")
    assert payload["planning"]["current_substage"] == "capture"
    assert payload["planning"]["substage_history"][0]["status"] == "failed"
    assert payload["planning"]["substage_history"][0]["completed_at"]
    assert failure["signature"]["id"] == payload["failure"]["signature"]
    assert failure["signature"]["basis"]["stage"] == "started"
    assert failure["signature"]["basis"]["substage"] == "capture"
    assert 'raise ValueError("capture failed")' in failure["traceback"]
    assert "ValueError during capture" in (report.path / "summary.md").read_text()


def test_failure_signature_normalizes_dynamic_message_values():
    normalize = report_module._normalize_failure_message

    assert normalize("tensor 123 failed at /tmp/run-a/model.py:17") == normalize(
        "tensor 456 failed at /tmp/run-b/model.py:29"
    )


def test_recover_interrupted_report_finalizes_bundle(tmp_path):
    report_root = tmp_path / "reports"
    config = resolve_report_config(
        profile_name="local",
        report_root=report_root,
        environ={},
    )
    report = AutoParallelRunReport(
        request={"workload": {"mode": "training"}},
        config=config,
        run_id="interrupted-run",
    )
    report.mark_substage("measure")

    recovered = recover_interrupted_report(
        report.path,
        reason="launcher was interrupted during diagnostic measurement",
        source_root=tmp_path,
    )

    payload = json.loads((recovered / "report.json").read_text())
    manifest = json.loads((recovered / "manifest.json").read_text())
    assert payload["status"] == "failed"
    assert payload["failure"]["type"] == "InterruptedRun"
    assert payload["failure"]["substage"] == "measure"
    assert payload["planning"]["substage_history"][-1]["status"] == "failed"
    assert manifest["status"] == "failed"
    catalog = build_report_catalog(report_root)
    assert catalog["summary"]["run_count"] == 1
    assert catalog["invalid_bundles"] == []


def test_recover_interrupted_report_rebuilds_stale_running_manifest(tmp_path):
    report_root = tmp_path / "reports"
    config = resolve_report_config(
        profile_name="local",
        report_root=report_root,
        environ={},
    )
    with AutoParallelRunReport(
        request={"workload": {"mode": "training"}},
        config=config,
        run_id="terminal-report-stale-manifest",
    ) as report:
        report.record_validation(
            "smoke",
            {"passed": True},
            evidence_level="executed_on_target",
        )

    report_path = report.path / "report.json"
    manifest_path = report.path / "manifest.json"
    payload_before = json.loads(report_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "running"
    manifest_path.write_text(json.dumps(manifest))
    (report.path / ".manifest.json.interrupted.tmp").write_text("")

    recovered = recover_interrupted_report(
        report.path,
        reason="manifest writer was interrupted after report finalization",
        source_root=tmp_path,
    )

    assert json.loads(report_path.read_text()) == payload_before
    assert json.loads((recovered / "manifest.json").read_text())["status"] == (
        "completed"
    )
    assert not (report.path / ".manifest.json.interrupted.tmp").exists()
    catalog = build_report_catalog(report_root)
    assert catalog["pending_bundles"] == []
    assert catalog["invalid_bundles"] == []


def test_catalog_verifies_bundles_and_aggregates_improvement_signals(tmp_path):
    report_root = tmp_path / "reports"
    config = resolve_report_config(
        profile_name="local",
        report_root=report_root,
        environ={},
    )

    with AutoParallelRunReport(
        request={"workload": {"mode": "training", "shape": [8, 16]}},
        config=config,
        run_id="measured-run",
    ) as measured_report:
        measured_report.add_finding(
            "FALLBACK_COST_PATH",
            "The fallback communication model was used",
        )
        measured_report.record_validation(
            "target-benchmark",
            {
                "kind": "autoparallel_target_step",
                "boundary": {"optimizer_update": True},
                "timing": {
                    "warmup_wall_time": {
                        "first_iteration_seconds": 12.0,
                        "total_seconds": 14.0,
                        "rank_reduction": "max",
                    }
                },
                "comparisons": [
                    {
                        "configuration_hash": "sha256:configuration",
                        "metric": "step_time",
                        "unit": "us",
                        "estimated": 100.0,
                        "observed": 125.0,
                    }
                ],
            },
            evidence_level="measured",
        )

    failed_report = AutoParallelRunReport(
        request={"workload": {"mode": "training", "shape": [4, 16]}},
        config=config,
        run_id="catalog-failed-run",
    )
    with pytest.raises(RuntimeError, match="compile failed"):
        with failed_report:
            failed_report.mark_stage("lowered", evidence_level="lowered")
            failed_report.mark_substage("compile")
            raise RuntimeError("compile failed")

    catalog = build_report_catalog(report_root)

    assert catalog["summary"]["run_count"] == 2
    assert catalog["summary"]["status_counts"] == {"completed": 1, "failed": 1}
    assert catalog["summary"]["finding_code_counts"] == {"FALLBACK_COST_PATH": 1}
    assert catalog["summary"]["failure_type_counts"] == {"RuntimeError": 1}
    assert catalog["summary"]["failure_stage_counts"] == {"lowered": 1}
    assert catalog["summary"]["failure_substage_counts"] == {"compile": 1}
    assert catalog["summary"]["time_to_ready"] == {
        "run_count": 1,
        "mean_first_iteration_seconds": 12.0,
        "maximum_first_iteration_seconds": 12.0,
    }
    assert catalog["measurement_error"] == [
        {
            "cluster_profile": "local",
            "configuration_hash": "sha256:configuration",
            "metric": "step_time",
            "unit": "us",
            "count": 1,
            "mean_absolute_error": 25.0,
            "mean_relative_error": 0.2,
        }
    ]
    assert all("request" not in run for run in catalog["runs"])
    assert all("request_sha256" in run for run in catalog["runs"])
    assert all("workload_sha256" in run for run in catalog["runs"])
    measured_record = next(
        run for run in catalog["runs"] if run["run_id"] == "measured-run"
    )
    assert measured_record["measurement_comparisons"] == [
        {
            "configuration_hash": "sha256:configuration",
            "metric": "step_time",
            "unit": "us",
            "absolute_error": 25.0,
            "relative_error": 0.2,
        }
    ]
    assert measured_record["time_to_ready"] == {
        "validation_kind": "autoparallel_target_step",
        "first_iteration_seconds": 12.0,
        "total_warmup_seconds": 14.0,
        "optimizer_update": True,
        "rank_reduction": "max",
    }

    output = write_report_catalog(report_root)
    assert output == report_root / "catalog-v1.json"
    assert json.loads(output.read_text())["summary"]["run_count"] == 2

    report_path = measured_report.path / "report.json"
    report_path.write_text(report_path.read_text() + "\n")
    catalog_with_corruption = build_report_catalog(report_root)
    assert catalog_with_corruption["summary"]["run_count"] == 1
    assert catalog_with_corruption["invalid_bundles"] == [
        {
            "path": str(measured_report.path.relative_to(report_root)),
            "error": "manifest hash mismatch: report.json",
        }
    ]


def test_catalog_treats_in_progress_bundle_as_pending(tmp_path):
    report_root = tmp_path / "reports"
    config = resolve_report_config(
        profile_name="local",
        report_root=report_root,
        environ={},
    )
    report = AutoParallelRunReport(
        request={"workload": {"mode": "training"}},
        config=config,
        run_id="running-report",
    )

    catalog = build_report_catalog(report_root)

    assert catalog["summary"]["run_count"] == 0
    assert catalog["invalid_bundles"] == []
    assert catalog["pending_bundles"] == [str(report.path.relative_to(report_root))]
    report.complete()


def test_catalog_summarizes_latest_campaign_attempts_and_missing_cases(tmp_path):
    report_root = tmp_path / "reports"
    config = resolve_report_config(
        profile_name="local",
        report_root=report_root,
        environ={},
    )

    def request(case_id):
        return {
            "workload": {"mode": "training"},
            "campaign": {
                "id": "torchbench-campaign",
                "suite": "torchbench",
                "case_id": case_id,
                "expected_case_count": 3,
            },
        }

    first_attempt = AutoParallelRunReport(
        request=request("case-a"), config=config, run_id="case-a-failed"
    )
    with pytest.raises(RuntimeError):
        with first_attempt:
            first_attempt.mark_substage("capture")
            raise RuntimeError("capture failed")

    with AutoParallelRunReport(
        request=request("case-a"), config=config, run_id="case-a-completed"
    ) as completed:
        completed.mark_stage("lowered", evidence_level="lowered")

    second_case = AutoParallelRunReport(
        request=request("case-b"), config=config, run_id="case-b-failed"
    )
    with pytest.raises(ValueError):
        with second_case:
            second_case.mark_substage("solve")
            raise ValueError("solver failed")

    campaign = build_report_catalog(report_root)["summary"]["campaigns"][0]

    assert campaign["run_count"] == 3
    assert campaign["reported_case_count"] == 2
    assert campaign["expected_case_count"] == 3
    assert campaign["missing_case_count"] == 1
    assert campaign["status_counts"] == {"completed": 1, "failed": 1}
    assert campaign["failure_substage_counts"] == {"solve": 1}
    assert campaign["completed_evidence_counts"]["lowered"] == 1


def test_catalog_rejects_unmanifested_bundle_file(tmp_path):
    report_root = tmp_path / "reports"
    config = resolve_report_config(
        profile_name="local",
        report_root=report_root,
        environ={},
    )
    with AutoParallelRunReport(
        request={"workload": {"mode": "training"}},
        config=config,
        run_id="unexpected-file-run",
    ) as report:
        pass
    (report.path / "placements.json").write_text("{}\n")

    catalog = build_report_catalog(report_root)

    assert catalog["summary"]["run_count"] == 0
    assert catalog["invalid_bundles"] == [
        {
            "path": str(report.path.relative_to(report_root)),
            "error": "unmanifested bundle files: ['placements.json']",
        }
    ]


def test_improvement_queue_groups_recurring_signals(tmp_path):
    report_root = tmp_path / "reports"
    config = resolve_report_config(
        profile_name="local",
        report_root=report_root,
        environ={},
    )
    for index in range(2):
        with AutoParallelRunReport(
            request={"workload": {"mode": "training", "index": index}},
            config=config,
            run_id=f"signal-run-{index}",
        ) as report:
            report.add_finding(
                "FALLBACK_COST_PATH",
                "The fallback communication model was used",
            )
            report.record_validation(
                "target-benchmark",
                {
                    "comparisons": [
                        {
                            "configuration_hash": "sha256:shared-configuration",
                            "metric": "step_time",
                            "unit": "us",
                            "estimated": 100.0,
                            "observed": 200.0,
                        }
                    ]
                },
                evidence_level="measured",
            )
        _write_recurring_failure(
            config,
            f"failure-run-{index}",
            f"compile failed for tensor {index} at /tmp/run-{index}/model.py",
        )

    queue = build_improvement_queue(
        build_report_catalog(report_root),
        max_representatives=1,
    )

    assert queue["summary"] == {
        "candidate_count": 3,
        "kind_counts": {
            "failure": 1,
            "finding": 1,
            "measurement_drift": 1,
        },
        "priority_counts": {"P1": 1, "P2": 2},
    }
    candidates = {candidate["kind"]: candidate for candidate in queue["candidates"]}
    assert candidates["failure"]["occurrence_count"] == 2
    assert candidates["failure"]["details"]["grouping"] == "signature"
    assert candidates["failure"]["details"]["failure_stage"] == "lowered"
    assert candidates["failure"]["details"]["failure_substage"] == "compile"
    assert candidates["finding"]["id"] == "finding:FALLBACK_COST_PATH"
    assert candidates["measurement_drift"]["details"]["mean_relative_error"] == 0.5
    assert all(
        len(candidate["representative_bundles"]) == 1
        for candidate in candidates.values()
    )
    markdown = render_improvement_queue(queue)
    assert "Recurring RuntimeError during compile" in markdown
    assert "Recurring FALLBACK_COST_PATH finding" in markdown


def test_improvement_queue_deduplicates_campaign_case_retries(tmp_path):
    report_root = tmp_path / "reports"
    config = resolve_report_config(
        profile_name="local",
        report_root=report_root,
        environ={},
    )

    def request(case_id):
        return {
            "workload": {"mode": "training"},
            "campaign": {
                "id": "optimizer-comparison",
                "case_id": case_id,
                "expected_case_count": 2,
            },
        }

    for run_id, case_id in (
        ("case-a-old", "case-a"),
        ("case-a-latest", "case-a"),
        ("case-b", "case-b"),
    ):
        with AutoParallelRunReport(
            request=request(case_id),
            config=config,
            run_id=run_id,
        ) as report:
            report.add_finding(
                "OPTIMIZER_COST_MISSING_FROM_OBJECTIVE",
                "Optimizer cost was absent from the search objective",
            )

    queue = build_improvement_queue(build_report_catalog(report_root))
    candidates = {candidate["id"]: candidate for candidate in queue["candidates"]}
    candidate = candidates["finding:OPTIMIZER_COST_MISSING_FROM_OBJECTIVE"]

    assert candidate["occurrence_count"] == 2
    assert candidate["affected_run_count"] == 2
    assert candidate["affected_workload_count"] == 1
    assert candidate["affected_campaign_count"] == 1
    assert all("case-a-old" not in path for path in candidate["representative_bundles"])
    assert queue["policy"]["catalog_run_count"] == 3
    assert queue["policy"]["deduplicated_run_count"] == 2


def test_improvement_queue_deduplicates_identical_finding_contexts(tmp_path):
    report_root = tmp_path / "reports"
    config = resolve_report_config(
        profile_name="local",
        report_root=report_root,
        environ={},
    )
    for index, edge in enumerate(("logits", "logits", "attention")):
        with AutoParallelRunReport(
            request={"workload": {"mode": "training", "index": index}},
            config=config,
            run_id=f"finding-context-{index}",
        ) as report:
            report.add_finding(
                "HIGH_COMMUNICATION_FRACTION",
                "Communication is a large fraction of the selected ILP cost",
                context={"largest_edge": edge, "fraction": 0.3},
            )

    queue = build_improvement_queue(build_report_catalog(report_root))
    candidates = {candidate["id"]: candidate for candidate in queue["candidates"]}
    candidate = candidates["finding:HIGH_COMMUNICATION_FRACTION"]

    assert candidate["occurrence_count"] == 2
    assert candidate["affected_run_count"] == 2


def test_improvement_queue_separates_failure_locations(tmp_path):
    report_root = tmp_path / "reports"
    config = resolve_report_config(
        profile_name="local",
        report_root=report_root,
        environ={},
    )
    messages = (
        'Data-dependent branching\nFile "/models/a/model.py", line 10, in forward',
        'Data-dependent branching\nFile "/models/b/model.py", line 20, in forward',
    )
    for index, message in enumerate(messages):
        _write_recurring_failure(config, f"location-failure-{index}", message)

    catalog = build_report_catalog(report_root)
    failures = [run["failure"] for run in catalog["runs"]]
    queue = build_improvement_queue(catalog)

    assert len({failure["signature"] for failure in failures}) == 1
    assert len({failure["diagnostic_location_signature"] for failure in failures}) == 2
    assert not any(candidate["kind"] == "failure" for candidate in queue["candidates"])


def test_improvement_queue_excludes_resource_exhaustion_failures(tmp_path):
    report_root = tmp_path / "reports"
    config = resolve_report_config(
        profile_name="local",
        report_root=report_root,
        environ={},
    )
    for index in range(2):
        _write_recurring_failure(
            config,
            f"oom-failure-{index}",
            "External launcher terminated a rank after it exhausted device memory",
        )

    catalog = build_report_catalog(report_root)
    queue = build_improvement_queue(catalog)

    assert {run["failure"]["classification"] for run in catalog["runs"]} == {
        "resource_exhaustion"
    }
    assert not any(candidate["kind"] == "failure" for candidate in queue["candidates"])


def test_improvement_queue_groups_fallbacks_by_strategy_and_reason(tmp_path):
    report_root = tmp_path / "reports"
    config = resolve_report_config(
        profile_name="local",
        report_root=report_root,
        environ={},
    )
    for index in range(2):
        with AutoParallelRunReport(
            request={"workload": {"mode": "training"}},
            config=config,
            run_id=f"fallback-signal-{index}",
            parallelization=_parallelization(f"fallback-decision-{index}"),
        ) as report:
            report.record_autoparallel_outcome(
                "failed",
                reason_codes=("CAPTURE_UNSUPPORTED",),
            )
            report.record_fallback_outcome(
                "ddp",
                "accepted",
                reason_codes=("CAPTURE_UNSUPPORTED",),
                evidence_level="executed_on_target",
            )
            report.record_parallelization_decision(
                "ddp",
                "accepted",
                reason_codes=("CAPTURE_UNSUPPORTED",),
            )

    queue = build_improvement_queue(build_report_catalog(report_root))
    candidates = {candidate["id"]: candidate for candidate in queue["candidates"]}

    fallback = candidates["fallback:ddp:CAPTURE_UNSUPPORTED"]
    assert fallback["kind"] == "parallelization_fallback"
    assert fallback["occurrence_count"] == 2
    assert fallback["affected_run_count"] == 2
    assert "semantics-preserving integration" in fallback["suggested_action"]
    assert candidates["finding:AUTOPARALLEL_FALLBACK"]["occurrence_count"] == 2


def test_improvement_queue_groups_schedule_sensitive_comparisons(tmp_path):
    report_root = tmp_path / "reports"
    config = resolve_report_config(
        profile_name="local",
        report_root=report_root,
        environ={},
    )
    for index in range(2):
        with AutoParallelRunReport(
            request={"workload": {"mode": "training"}},
            config=config,
            run_id=f"schedule-sensitive-{index}",
            parallelization=_parallelization(f"schedule-decision-{index}"),
        ) as report:
            report.record_competitiveness_assessment(
                "schedule_sensitive",
                evidence="modeled",
                autoparallel_estimate=_autoparallel_estimate(_latency(90, 110, 130)),
                baselines=[_baseline(latency=_latency(100, 110, 120))],
                reason_codes=("MODELED_SCHEDULE_SENSITIVE",),
            )
            report.record_autoparallel_outcome(
                "below_bar",
                reason_codes=("MODELED_SCHEDULE_SENSITIVE",),
            )
            report.record_fallback_outcome(
                "ddp",
                "accepted",
                reason_codes=("MODELED_SCHEDULE_SENSITIVE",),
                evidence_level="executed_on_target",
            )
            report.record_parallelization_decision(
                "ddp",
                "accepted",
                reason_codes=("MODELED_SCHEDULE_SENSITIVE",),
            )

    queue = build_improvement_queue(build_report_catalog(report_root))
    candidates = {candidate["id"]: candidate for candidate in queue["candidates"]}

    candidate = candidates[
        "competitiveness:schedule_sensitive:modeled:ddp:MODELED_SCHEDULE_SENSITIVE"
    ]
    assert candidate["kind"] == "parallelization_competitiveness"
    assert candidate["occurrence_count"] == 2


def test_improvement_queue_preserves_measured_competitiveness_provenance(tmp_path):
    report_root = tmp_path / "reports"
    config = resolve_report_config(
        profile_name="local",
        report_root=report_root,
        environ={},
    )
    for index in range(2):
        with AutoParallelRunReport(
            request={"workload": {"mode": "training"}},
            config=config,
            run_id=f"measured-regression-{index}",
            parallelization=_parallelization(f"measured-decision-{index}"),
        ) as report:
            report.record_competitiveness_assessment(
                "robustly_worse",
                evidence="measured",
                autoparallel_estimate=_autoparallel_estimate(_latency(120, 130, 140)),
                baselines=[_baseline(latency=_latency(80, 90, 100))],
                reason_codes=("MEASURED_ROBUSTLY_WORSE",),
            )
            report.record_autoparallel_outcome(
                "below_bar",
                reason_codes=("MEASURED_ROBUSTLY_WORSE",),
            )
            report.record_fallback_outcome(
                "ddp",
                "accepted",
                reason_codes=("MEASURED_ROBUSTLY_WORSE",),
                evidence_level="executed_on_target",
            )
            report.record_parallelization_decision(
                "ddp",
                "fallback_selected",
                reason_codes=("MEASURED_ROBUSTLY_WORSE",),
            )

    queue = build_improvement_queue(build_report_catalog(report_root))
    candidates = {candidate["id"]: candidate for candidate in queue["candidates"]}
    candidate = candidates[
        "competitiveness:robustly_worse:measured:ddp:MEASURED_ROBUSTLY_WORSE"
    ]

    assert candidate["title"].startswith("Measured AutoParallel regression")
    assert candidate["priority"] == "P1"
    assert candidate["details"]["evidence"] == "measured"
    assert "linked target profiles" in candidate["suggested_action"]


def test_improvement_queue_suppresses_resolved_comparison_unavailable(tmp_path):
    report_root = tmp_path / "reports"
    config = resolve_report_config(
        profile_name="local",
        report_root=report_root,
        environ={},
    )
    for index in range(2):
        parallelization = _parallelization(f"resolved-decision-{index}")
        with AutoParallelRunReport(
            request={"workload": {"mode": "training"}},
            config=config,
            run_id=f"unavailable-{index}",
            parallelization=parallelization,
        ) as report:
            report.record_competitiveness_assessment(
                "comparison_unavailable",
                evidence="none",
                reason_codes=("BASELINE_COMPARISON_UNAVAILABLE",),
            )
            report.record_autoparallel_outcome("inconclusive")
            report.record_parallelization_decision("none", "inconclusive")

        with AutoParallelRunReport(
            request={"workload": {"mode": "training"}},
            config=config,
            run_id=f"resolved-{index}",
            parallelization=parallelization,
        ) as report:
            report.mark_stage(
                "executed_on_target",
                evidence_level="executed_on_target",
            )
            _record_robust_competitiveness(report)
            report.record_autoparallel_outcome("accepted")
            report.record_parallelization_decision("autoparallel", "accepted")

    queue = build_improvement_queue(build_report_catalog(report_root))

    assert not any(
        candidate["kind"] == "parallelization_competitiveness"
        for candidate in queue["candidates"]
    )


def test_feedback_refresh_writes_health_catalog_and_triage(tmp_path):
    report_root = tmp_path / "reports"
    config = resolve_report_config(
        profile_name="local",
        report_root=report_root,
        environ={},
    )
    with AutoParallelRunReport(
        request={"workload": {"mode": "training"}},
        config=config,
        run_id="feedback-run",
    ) as report:
        pass

    status = refresh_report_feedback(report_root, expected_refresh_interval_seconds=60)

    assert status["status"] == "healthy"
    assert status["catalog"]["run_count"] == 1
    assert status["catalog"]["invalid_bundle_count"] == 0
    assert status["triage"]["candidate_count"] == 0
    assert (report_root / "catalog-v1.json").is_file()
    assert (report_root / "improvement-queue-v1.json").is_file()
    assert (report_root / "improvement-queue-v1.md").is_file()
    assert json.loads((report_root / "feedback-status-v1.json").read_text()) == status

    with feedback_module._refresh_lock(
        report_root / ".feedback-refresh-v1.lock"
    ) as acquired:
        assert acquired
        locked = refresh_report_feedback(report_root)
    assert locked["status"] == "skipped_locked"

    report_path = report.path / "report.json"
    report_path.write_text(report_path.read_text() + "\n")
    degraded = refresh_report_feedback(report_root)
    queue = json.loads((report_root / "improvement-queue-v1.json").read_text())

    assert degraded["status"] == "degraded"
    assert degraded["catalog"]["invalid_bundle_count"] == 1
    assert degraded["triage"]["candidate_count"] == 1
    assert queue["candidates"][0]["kind"] == "invalid_bundle"


def test_feedback_watch_refreshes_until_stopped(tmp_path, monkeypatch, capsys):
    stopped = feedback_module.threading.Event()
    calls = []

    def refresh(report_root, **kwargs):
        calls.append((report_root, kwargs))
        stopped.set()
        return {"schema_version": 1, "status": "healthy"}

    monkeypatch.setattr(feedback_module, "refresh_report_feedback", refresh)
    feedback_module.run_refresh_loop(
        tmp_path,
        interval_seconds=1,
        verify=True,
        minimum_occurrences=2,
        measurement_relative_error_threshold=0.2,
        max_representatives=3,
        stop_event=stopped,
    )

    assert len(calls) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "healthy"


def test_report_session_automates_profile_topology_plan_and_metrics(
    tmp_path, monkeypatch
):
    profiles_path = tmp_path / "profiles.json"
    report_root = tmp_path / "reports"
    _write_profiles(
        profiles_path,
        {
            "test-cluster": {
                "match_env": {"TEST_AUTOPARALLEL_SITE": "test-cluster"},
                "report_root": str(report_root),
                "topology_profile": "runtime_detected_nccl",
                "capture_level": "standard",
                "policies": {"fallback_costs": "reject"},
            }
        },
        default_profile="test-cluster",
    )
    collector_module = sys.modules["collect_reordered_metrics"]
    nccl_module = sys.modules["autoparallel.cost_models.nccl_cost_model"]
    monkeypatch.setattr(
        collector_module, "detect_nccl_topo_config", lambda _mesh: object()
    )
    monkeypatch.setattr(nccl_module, "NCCLTopoConfig", object)
    monkeypatch.setattr(
        nccl_module,
        "nccl_cost_coverage",
        lambda _topology, _mesh_shape: ("algo_loop",),
    )

    with AutoParallelReportSession(
        _UnusedMesh(),
        request={
            "workload": {"mode": "training", "global_batch": 8},
            "mesh": {"shape": [8], "names": ["dp"]},
            "constraints": {},
            "spmd_assumptions": [],
        },
        campaign={
            "id": "session-campaign",
            "suite": "unit-tests",
            "case_id": "session-run",
            "expected_case_count": 1,
        },
        provenance={"git_revision": "test"},
        compiler_config={"overlap_scheduling": False},
        source_root=ROOT,
        external_source_roots={"autoparallel-fixture": ROOT},
        entry_point="tests.test_autoparallel_run_report:session_run",
        launch_config={"launcher": "torchrun", "world_size": 8},
        include_optimizer_state=True,
        profiles_path=profiles_path,
        environ={"TEST_AUTOPARALLEL_SITE": "test-cluster"},
        run_id="session-run",
    ) as session:
        assert session.config.selection_source == "environment_match"
        assert session.collector.cost_model_source == "detected"
        session.record_plan(_FakeOptimizer())
        session.mark_lowered()
        session.collector.records.extend(_metrics()["graphs"])
        session.record_validation(
            "execution",
            {"process_group": "nccl", "world_size": 8},
            evidence_level="executed_on_target",
        )

    assert session.path is not None
    payload = json.loads((session.path / "report.json").read_text())
    assert payload["status"] == "completed"
    assert payload["stage"] == "executed_on_target"
    assert payload["cluster"]["selection_source"] == "environment_match"
    assert payload["provenance"]["cost_model"]["status"] == "nccl_detected"
    assert payload["provenance"]["cost_model"]["source"] == "detected"
    assert payload["provenance"]["configuration_sha256"].startswith("sha256:")
    reproduction = payload["provenance"]["reproduction"]
    assert reproduction["entry_point"] == (
        "tests.test_autoparallel_run_report:session_run"
    )
    assert reproduction["launch_config"] == {
        "launcher": "torchrun",
        "world_size": 8,
    }
    assert reproduction["source"]["git_revision"]
    assert reproduction["source"]["working_tree_state_sha256"].startswith("sha256:")
    assert reproduction["external_sources"]["autoparallel-fixture"]["git_revision"]
    assert reproduction["complete"]
    assert reproduction["missing"] == []
    assert not (session.path / "placements.json").exists()
    assert (session.path / "artifacts/placements.json").is_file()
    assert (session.path / "artifacts/optimizer.ap").read_bytes() == (
        b"trusted optimizer state"
    )
    assert (session.path / "artifacts/optimizer-replay.pt.gz").read_bytes() == (
        b"trusted optimizer replay"
    )
    assert payload["planning"]["optimizer_state_artifact"] == "optimizer-state"
    assert payload["planning"]["optimizer_replay_artifact"] == "optimizer-replay"
    assert [finding["code"] for finding in payload["findings"]] == [
        "EXTRAPOLATED_CRITICAL_TERM",
        "HIGH_COMMUNICATION_FRACTION",
        "REPLICATED_PARAMETER_STORAGE",
    ]
    catalog = build_report_catalog(report_root)
    assert catalog["summary"]["run_count"] == 1
    assert catalog["invalid_bundles"] == []
    assert catalog["runs"][0]["cost_model_source"] == "detected"
    assert catalog["runs"][0]["optimizer_replay"] == {
        "available": True,
        "name": "optimizer-replay",
        "path": "artifacts/optimizer-replay.pt.gz",
        "sha256": _sha256(session.path / "artifacts/optimizer-replay.pt.gz"),
        "size_bytes": len(b"trusted optimizer replay"),
    }
    assert catalog["runs"][0]["external_source_count"] == 1
    assert catalog["runs"][0]["external_sources_complete"]
    assert catalog["summary"]["campaigns"] == [
        {
            "id": "session-campaign",
            "suites": ["unit-tests"],
            "run_count": 1,
            "reported_case_count": 1,
            "expected_case_count": 1,
            "missing_case_count": 0,
            "status_counts": {"completed": 1},
            "evidence_counts": {"executed_on_target": 1},
            "failure_type_counts": {},
            "failure_substage_counts": {},
            "completed_evidence_counts": {
                "captured": 1,
                "ilp_feasible": 1,
                "lowered": 1,
                "compiled": 1,
                "numerically_checked": 0,
                "executed_on_target": 1,
                "measured": 0,
            },
        }
    ]


def test_report_session_measures_autoparallel_step(tmp_path, monkeypatch):
    profiles_path = tmp_path / "profiles.json"
    report_root = tmp_path / "reports"
    _write_profiles(
        profiles_path,
        {"test": {"report_root": str(report_root)}},
        default_profile="test",
    )

    class FakeEvent:
        count = 0

        def __init__(self, *, enable_timing):
            assert enable_timing
            self.index = FakeEvent.count
            FakeEvent.count += 1

        def record(self):
            pass

        def elapsed_time(self, _end):
            return self.index // 2 + 1.0

    monkeypatch.setattr(session_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(session_module.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        session_module.torch.cuda, "synchronize", lambda _device=None: None
    )
    monkeypatch.setattr(
        session_module.torch.cuda, "memory_allocated", lambda _device=None: 1024
    )
    monkeypatch.setattr(
        session_module.torch.cuda, "memory_reserved", lambda _device=None: 2048
    )
    monkeypatch.setattr(
        session_module.torch.cuda, "reset_peak_memory_stats", lambda _device=None: None
    )
    monkeypatch.setattr(
        session_module.torch.cuda,
        "max_memory_allocated",
        lambda _device=None: 4096,
    )
    monkeypatch.setattr(
        session_module.torch.cuda,
        "max_memory_reserved",
        lambda _device=None: 8192,
    )
    monkeypatch.setattr(session_module.torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(
        session_module.torch.cuda,
        "get_device_name",
        lambda _device=None: "Test GPU",
    )
    wall_clock = iter([10.0, 12.0, 13.0])
    monkeypatch.setattr(session_module.time, "perf_counter", wall_clock.__next__)

    calls = []
    with AutoParallelReportSession(
        _UnusedMesh(),
        request={"workload": {"mode": "training"}},
        profiles_path=profiles_path,
        environ={},
        run_id="measurement-run",
        mode="measure",
        nccl_topology=object(),
    ) as session:
        session.collector.records.extend(_metrics()["graphs"])
        result = session.measure_step(
            "target-step",
            lambda: calls.append("step"),
            warmup_iterations=1,
            measured_iterations=2,
            optimizer_update=True,
            boundary_details={
                "loss": True,
                "gradient_clipping": {"kind": "global_l2_norm", "max_norm": 1.0},
            },
        )

    assert calls == ["step", "step", "step"]
    assert result["timing"]["samples_us"] == [1000.0, 2000.0]
    assert result["timing"]["mean_us"] == 1500.0
    assert result["timing"]["warmup_wall_time"] == {
        "method": "host_monotonic_clock_with_device_synchronize",
        "rank_reduction": result["hardware"]["rank_reduction"],
        "first_iteration_seconds": 2.0,
        "total_seconds": 3.0,
        "includes_lazy_compilation_when_applicable": True,
    }
    assert result["memory"]["peak_allocated_bytes"] == 4096
    assert result["boundary"]["optimizer_update"] is True
    assert result["boundary"]["loss"] is True
    assert result["boundary"]["gradient_clipping"]["max_norm"] == 1.0
    assert result["estimate"]["step_time_us"] == 30.0
    assert result["estimate"]["peak_memory_bytes"] == 2048
    assert [comparison["metric"] for comparison in result["comparisons"]] == [
        "step_time",
        "step_peak_allocated_memory",
    ]
    assert result["measurement_configuration_sha256"].startswith("sha256:")
    assert all(
        comparison["configuration_hash"] == result["measurement_configuration_sha256"]
        for comparison in result["comparisons"]
    )
    assert session.path is not None
    payload = json.loads((session.path / "report.json").read_text())
    assert payload["stage"] == "measured"
    assert payload["evidence"]["highest_level"] == "measured"
    assert payload["validation"]["target-step"]["result"] == result
    summary = (session.path / "summary.md").read_text()
    assert "target-step: mean 1500.0 us; peak allocated 4096 bytes" in summary
    catalog = build_report_catalog(report_root)
    assert [comparison["metric"] for comparison in catalog["measurement_error"]] == [
        "step_peak_allocated_memory",
        "step_time",
    ]


def test_report_session_writes_only_on_reporting_rank(tmp_path):
    profiles_path = tmp_path / "profiles.json"
    report_root = tmp_path / "reports"
    _write_profiles(
        profiles_path,
        {"test": {"report_root": str(report_root)}},
        default_profile="test",
    )

    with AutoParallelReportSession(
        _UnusedMesh(),
        request={"workload": {"mode": "training"}},
        profiles_path=profiles_path,
        environ={},
        mode="plan",
        rank=1,
        report_rank=0,
        nccl_topology=object(),
    ) as session:
        assert not session.is_reporter
        assert session.path is None
        session.record_plan(_UnexpectedOptimizerUse())
        session.mark_lowered()

    assert not report_root.exists()


@pytest.mark.parametrize(
    ("declared_name", "override_name"),
    [
        ("metric_collective_bucketing", "collective_bucketing"),
        (
            "metric_bucket_all_all_reduce_groups",
            "bucket_all_all_reduce_groups",
        ),
    ],
)
def test_report_session_rejects_declared_reordering_mismatch(
    tmp_path, declared_name, override_name
):
    profiles_path = tmp_path / "profiles.json"
    _write_profiles(
        profiles_path,
        {"test": {"report_root": str(tmp_path / "reports")}},
        default_profile="test",
    )
    session = AutoParallelReportSession(
        _UnusedMesh(),
        request={"workload": {"mode": "training"}},
        compiler_config={declared_name: True},
        reordering_overrides={override_name: False},
        profiles_path=profiles_path,
        environ={},
        mode="plan",
        nccl_topology=object(),
    )

    with pytest.raises(ValueError, match="does not match effective"):
        with session:
            pass


def test_report_session_finds_dependency_bound_collectives(tmp_path):
    profiles_path = tmp_path / "profiles.json"
    _write_profiles(
        profiles_path,
        {"test": {"report_root": str(tmp_path / "reports")}},
        default_profile="test",
    )
    session = AutoParallelReportSession(
        _UnusedMesh(),
        request={"workload": {"mode": "training"}},
        profiles_path=profiles_path,
        environ={},
        mode="evaluate",
        nccl_topology=object(),
    )
    graphs = _metrics()["graphs"]
    graphs[0]["bucketing_diagnostics"] = {
        "collective_counts_before": {"all_reduce": 10},
        "collective_counts_after": {"all_reduce": 9},
        "greedy_bucket_calls": [
            {
                "collective_type_counts": {"_c10d_functional.all_reduce": 10},
                "dependency_blocked_count": 8,
            }
        ],
    }

    with session:
        session.collector.records.extend(graphs)
        session.record_metrics()

    assert session.path is not None
    payload = json.loads((session.path / "report.json").read_text())
    finding = next(
        finding
        for finding in payload["findings"]
        if finding["code"] == "DEPENDENCY_BOUND_TENSOR_PARALLEL_COLLECTIVES"
    )
    assert finding["context"]["dependency_bound_fraction"] == 0.8


def test_report_session_finds_multigroup_bucketing_coverage_gap(tmp_path):
    profiles_path = tmp_path / "profiles.json"
    _write_profiles(
        profiles_path,
        {"test": {"report_root": str(tmp_path / "reports")}},
        default_profile="test",
    )
    session = AutoParallelReportSession(
        _UnusedMesh(),
        request={
            "workload": {"mode": "training"},
            "mesh": {"shape": [4, 2], "names": ["dp", "tp"]},
        },
        profiles_path=profiles_path,
        environ={},
        mode="evaluate",
        nccl_topology=object(),
    )
    graphs = _metrics()["graphs"]
    graphs[1]["bucketing_diagnostics"] = {
        "collective_counts_before": {"all_reduce": 100},
        "collective_counts_after": {"all_reduce": 30},
        "greedy_bucket_calls": [
            {
                "collective_type_counts": {"_c10d_functional.all_reduce": 75},
                "candidate_count": 75,
                "dependency_blocked_count": 0,
            }
        ],
    }

    with session:
        session.collector.records.extend(graphs)
        session.record_metrics()

    assert session.path is not None
    payload = json.loads((session.path / "report.json").read_text())
    finding = next(
        finding
        for finding in payload["findings"]
        if finding["code"] == "MULTIGROUP_BUCKETING_COVERAGE_GAP"
    )
    assert finding["context"]["omitted_all_reduce_count"] == 25
    assert finding["context"]["omitted_all_reduce_fraction"] == 0.25


def test_report_session_never_silently_accepts_parallelization(tmp_path):
    profiles_path = tmp_path / "profiles.json"
    report_root = tmp_path / "reports"
    _write_profiles(
        profiles_path,
        {"test": {"report_root": str(report_root)}},
        default_profile="test",
    )

    with AutoParallelReportSession(
        _UnusedMesh(),
        request={"workload": {"mode": "training"}},
        parallelization=_parallelization("inconclusive-decision"),
        profiles_path=profiles_path,
        environ={},
        run_id="inconclusive-routing",
        mode="plan",
        nccl_topology=object(),
    ) as inconclusive:
        pass

    assert inconclusive.path is not None
    payload = json.loads((inconclusive.path / "report.json").read_text())
    assert payload["parallelization"]["autoparallel"]["outcome"] == "inconclusive"
    assert payload["parallelization"]["autoparallel"]["reason_codes"] == [
        "ACCEPTANCE_NOT_EVALUATED"
    ]

    failed = AutoParallelReportSession(
        _UnusedMesh(),
        request={"workload": {"mode": "training"}},
        parallelization=_parallelization("failed-decision"),
        profiles_path=profiles_path,
        environ={},
        run_id="failed-routing",
        mode="plan",
        nccl_topology=object(),
    )
    with pytest.raises(ValueError, match="capture failed"):
        with failed:
            raise ValueError("capture failed")

    assert failed.path is not None
    failure_payload = json.loads((failed.path / "report.json").read_text())
    assert failure_payload["parallelization"]["autoparallel"]["outcome"] == ("failed")
    assert failure_payload["parallelization"]["autoparallel"]["reason_codes"] == [
        "AUTOPARALLEL_EXCEPTION"
    ]


@pytest.mark.parametrize(
    ("substage", "message", "reason_code"),
    [
        ("capture", "unsupported Python control flow", "CAPTURE_UNSUPPORTED"),
        (
            "capture",
            "Could not extract specialized integer from data-dependent expression u0",
            "CAPTURE_DYNAMIC_OUTPUT_SHAPE_UNSUPPORTED",
        ),
        (
            "solve",
            "The sharding optimizer could not find a feasible solution",
            "SHARDING_SOLVER_INFEASIBLE",
        ),
        ("lower", "DTensor lowering failed", "LOWERING_FAILED"),
        ("compile", "Inductor failed", "AUTOPARALLEL_COMPILE_FAILED"),
    ],
)
def test_report_session_classifies_known_failure_substages(
    tmp_path, substage, message, reason_code
):
    profiles_path = tmp_path / "profiles.json"
    report_root = tmp_path / "reports"
    _write_profiles(
        profiles_path,
        {"test": {"report_root": str(report_root)}},
        default_profile="test",
    )
    session = AutoParallelReportSession(
        _UnusedMesh(),
        request={"workload": {"mode": "training"}},
        parallelization=_parallelization(f"{substage}-failure"),
        profiles_path=profiles_path,
        environ={},
        run_id=f"{substage}-{reason_code.lower()}",
        mode="plan",
        nccl_topology=object(),
    )
    with pytest.raises(RuntimeError, match=message):
        with session:
            session.mark_substage(substage)
            raise RuntimeError(message)

    payload = json.loads((session.path / "report.json").read_text())
    outcome = payload["parallelization"]["autoparallel"]
    assert outcome["outcome"] == "failed"
    assert outcome["reason_codes"] == [reason_code]
    assert outcome["details"]["substage"] == substage


def test_report_session_classifies_solver_timeout(tmp_path):
    profiles_path = tmp_path / "profiles.json"
    _write_profiles(
        profiles_path,
        {"test": {"report_root": str(tmp_path / "reports")}},
        default_profile="test",
    )
    session = AutoParallelReportSession(
        _UnusedMesh(),
        request={"workload": {"mode": "training"}},
        parallelization=_parallelization("solver-timeout"),
        profiles_path=profiles_path,
        environ={},
        mode="plan",
        nccl_topology=object(),
    )
    with pytest.raises(TimeoutError, match="solver timed out"):
        with session:
            session.mark_substage("solve")
            raise TimeoutError("solver timed out")

    payload = json.loads((session.path / "report.json").read_text())
    outcome = payload["parallelization"]["autoparallel"]
    assert outcome["reason_codes"] == ["SHARDING_SOLVER_TIMEOUT"]


def test_report_session_infers_unmarked_dynamic_branch_capture_failure(tmp_path):
    profiles_path = tmp_path / "profiles.json"
    report_root = tmp_path / "reports"
    _write_profiles(
        profiles_path,
        {"test": {"report_root": str(report_root)}},
        default_profile="test",
    )
    session = AutoParallelReportSession(
        _UnusedMesh(),
        request={"workload": {"mode": "training"}},
        parallelization=_parallelization("unmarked-capture-failure"),
        profiles_path=profiles_path,
        environ={},
        run_id="unmarked-capture-failure",
        mode="plan",
        nccl_topology=object(),
    )
    unsupported = type("Unsupported", (RuntimeError,), {})
    with pytest.raises(unsupported, match="Data-dependent branching"):
        with session:
            raise unsupported("Data-dependent branching in model forward")

    payload = json.loads((session.path / "report.json").read_text())
    outcome = payload["parallelization"]["autoparallel"]
    assert outcome["reason_codes"] == ["CAPTURE_UNSUPPORTED"]
    assert outcome["details"]["substage"] is None
    assert outcome["details"]["inferred_substage"] == "capture"


def test_report_session_enforces_rejected_fallback_policy(tmp_path):
    profiles_path = tmp_path / "profiles.json"
    report_root = tmp_path / "reports"
    _write_profiles(
        profiles_path,
        {
            "strict": {
                "report_root": str(report_root),
                "policies": {"fallback_costs": "reject"},
            }
        },
        default_profile="strict",
    )
    session = AutoParallelReportSession(
        _UnusedMesh(),
        request={"workload": {"mode": "training"}},
        profiles_path=profiles_path,
        environ={},
        run_id="fallback-run",
        nccl_topology=None,
    )

    with pytest.raises(RuntimeError, match="rejects the PyTorch fallback"):
        with session:
            pass

    assert session.path is not None
    payload = json.loads((session.path / "report.json").read_text())
    assert payload["status"] == "failed"
    assert payload["provenance"]["cost_model"] == {
        "status": "pytorch_default_explicit",
        "source": "explicit",
        "topology": "None",
    }
    assert payload["findings"] == [
        {
            "code": "FALLBACK_COST_PATH",
            "severity": "error",
            "message": (
                "No NCCL topology is active; the PyTorch fallback cost model was selected"
            ),
            "context": payload["provenance"]["cost_model"],
            "suggested_action": "Provide or detect an NCCL topology before planning",
        }
    ]
    catalog = build_report_catalog(report_root)
    assert catalog["runs"][0]["cost_model_status"] == "pytorch_default_explicit"
    assert catalog["runs"][0]["cost_model_source"] == "explicit"


def test_report_schema_and_packaged_profiles_match_writer():
    schema = json.loads(
        (SKILL_ROOT / "references/run-report-schema-v1.json").read_text()
    )
    profiles = json.loads((SKILL_ROOT / "references/cluster-profiles.json").read_text())

    assert schema["properties"]["schema_version"]["const"] == 1
    assert schema["properties"]["stage"]["enum"] == list(report_module._STAGES)
    assert "parallelization" in schema["properties"]
    assert "competitiveness" in schema["$defs"]["parallelization"]["properties"]
    assert schema["$defs"]["evidenceLevel"]["enum"] == list(
        report_module._EVIDENCE_LEVELS
    )
    assert profiles["schema_version"] == report_module.PROFILE_SCHEMA_VERSION
    assert profiles["default_profile"] in profiles["profiles"]


def test_packaged_profiles_are_site_neutral():
    profiles_path = SKILL_ROOT / "references/cluster-profiles.json"
    example_path = SKILL_ROOT / "references/cluster-profiles.example.json"
    profiles = json.loads(profiles_path.read_text())
    example = json.loads(example_path.read_text())

    assert set(profiles["profiles"]) == {"local"}
    assert profiles["profiles"]["local"]["report_root"] == (
        "artifacts/autoparallel-reports"
    )
    assert example["schema_version"] == report_module.PROFILE_SCHEMA_VERSION
    assert example["default_profile"] in example["profiles"]
