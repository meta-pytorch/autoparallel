from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping

_FIELDS: dict[str, set[Any]] = {
    "invoke_skill": {True, False},
    "autoparallel_disposition": {"assess_only", "attempt", "skip"},
    "autoparallel_outcome": {
        "accepted",
        "below_bar",
        "failed",
        "inconclusive",
        "ineligible",
    },
    "competitiveness_verdict": {
        "baseline_infeasible",
        "comparison_unavailable",
        "pareto_dominated",
        "robustly_competitive",
        "robustly_worse",
        "schedule_sensitive",
    },
    "benchmark_action": {"none", "run_authorized", "skip"},
    "selected_strategy": {"autoparallel", "ddp", "fsdp", "none"},
    "report_required": {True, False},
}

_DECISION_CARD_FIELDS = {
    "evidence_level",
    "boundary",
    "boundary_matched",
    "reason_codes",
    "missing_evidence",
    "fallback_strategy",
    "next_control",
}
_EVIDENCE_LEVELS = {
    "none",
    "captured",
    "ilp_feasible",
    "lowered",
    "compiled",
    "numerically_checked",
    "executed_on_target",
    "measured",
}
_REASON_CODE_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def validate_routing_classification(
    decision: Mapping[str, Any],
    evidence: Mapping[str, Any] | None = None,
    *,
    require_decision_card: bool = False,
) -> dict[str, Any]:
    errors: list[dict[str, str]] = []

    def reject(code: str, message: str) -> None:
        errors.append({"code": code, "message": message})

    for field, allowed in _FIELDS.items():
        value = decision.get(field)
        if value not in allowed:
            reject("INVALID_FIELD", f"{field} has invalid value {value!r}")

    if errors:
        return {"valid": False, "errors": errors}

    invoke = decision["invoke_skill"]
    disposition = decision["autoparallel_disposition"]
    outcome = decision["autoparallel_outcome"]
    verdict = decision["competitiveness_verdict"]
    benchmark = decision["benchmark_action"]
    strategy = decision["selected_strategy"]
    report_required = decision["report_required"]

    if not invoke and disposition != "skip":
        reject("SKIPPED_SKILL_DISPOSITION", "A skipped skill must use disposition=skip")
    if invoke and disposition == "skip":
        reject(
            "INVOKED_SKILL_DISPOSITION", "An invoked skill cannot use disposition=skip"
        )
    if invoke and not report_required:
        reject(
            "MISSING_ROUTING_REPORT", "An invoked routing decision requires a report"
        )
    if not invoke and report_required:
        reject(
            "UNRELATED_ROUTING_REPORT", "A skipped unrelated request needs no report"
        )
    if disposition == "attempt" and outcome == "ineligible":
        reject("ATTEMPT_MARKED_INELIGIBLE", "An attempted candidate is not ineligible")
    if disposition == "assess_only" and outcome in {"accepted", "below_bar", "failed"}:
        reject(
            "UNATTEMPTED_OUTCOME",
            "accepted, below_bar, and failed require an AutoParallel attempt",
        )
    if outcome == "accepted" and strategy != "autoparallel":
        reject(
            "ACCEPTED_NOT_SELECTED",
            "An accepted AutoParallel candidate must be selected",
        )
    if strategy == "autoparallel" and outcome != "accepted":
        reject(
            "SELECTED_WITHOUT_ACCEPTANCE",
            "Selecting AutoParallel requires outcome=accepted",
        )
    if strategy == "autoparallel" and verdict not in {
        "baseline_infeasible",
        "robustly_competitive",
    }:
        reject(
            "SELECTED_WITHOUT_RELATIVE_BAR",
            "Selecting AutoParallel requires a competitive or infeasible-baseline verdict",
        )
    if outcome == "accepted" and verdict in {
        "comparison_unavailable",
        "pareto_dominated",
        "robustly_worse",
        "schedule_sensitive",
    }:
        reject("ACCEPTED_UNCERTAIN_COMPARISON", "The relative-quality bar is not met")
    if outcome == "accepted" and disposition != "attempt":
        reject("ACCEPTED_WITHOUT_ATTEMPT", "Acceptance requires disposition=attempt")
    if strategy in {"ddp", "fsdp"} and outcome == "accepted":
        reject(
            "FALLBACK_AFTER_ACCEPTANCE",
            "A fallback cannot replace accepted AutoParallel",
        )
    if verdict == "baseline_infeasible" and strategy in {"ddp", "fsdp"}:
        reject(
            "SELECTED_INFEASIBLE_BASELINE", "An infeasible baseline cannot be selected"
        )
    if benchmark == "run_authorized" and (
        outcome != "inconclusive" or strategy != "none"
    ):
        reject(
            "PENDING_BENCHMARK_DECISION",
            "An authorized pending benchmark keeps the outcome inconclusive and selection open",
        )

    facts = dict(evidence or {})
    requested = facts.get("requested_strategy")
    if requested in {"ddp", "fsdp"}:
        if disposition != "assess_only" or outcome != "ineligible":
            reject(
                "EXPLICIT_STRATEGY_NOT_PRESERVED",
                "An explicit DDP/FSDP request assesses AutoParallel without attempting it",
            )
        if strategy != requested:
            reject(
                "EXPLICIT_STRATEGY_REPLACED",
                f"The explicitly requested {requested.upper()} strategy must be selected",
            )
    if facts.get("autoparallel_eligible") is False and outcome != "ineligible":
        reject(
            "INELIGIBLE_OUTCOME", "An ineligible candidate requires outcome=ineligible"
        )
    if facts.get("autoparallel_attempted") is True and disposition != "attempt":
        reject("ATTEMPT_DISPOSITION", "Observed execution requires disposition=attempt")
    if facts.get("attempt_error_observed") is True and outcome != "failed":
        reject(
            "OBSERVED_ATTEMPT_FAILURE",
            "An observed attempt error requires outcome=failed",
        )
    if facts.get("semantic_violation_observed") is True and outcome != "failed":
        reject(
            "OBSERVED_SEMANTIC_FAILURE",
            "An observed semantic invariant violation requires outcome=failed",
        )
    if facts.get("required_evidence_missing") is True and outcome not in {
        "inconclusive",
        "ineligible",
    }:
        reject(
            "MISSING_EVIDENCE_OUTCOME",
            "Missing required evidence cannot prove acceptance, failure, or a missed bar",
        )
    if facts.get("workload_representative") is False:
        if outcome != "inconclusive" or verdict != "comparison_unavailable":
            reject(
                "UNREPRESENTATIVE_WORKLOAD_DECISION",
                "A smoke-only workload cannot establish a production comparison",
            )
        if facts.get("qualified_fallback") is None and strategy != "none":
            reject(
                "UNREPRESENTATIVE_WORKLOAD_SELECTION",
                "Do not select a production strategy without its workload contract",
            )
    benchmark_needed = facts.get("benchmark_needed")
    benchmark_authorized = facts.get("benchmark_authorized")
    benchmark_pending = facts.get("benchmark_pending")
    if benchmark_needed is True:
        expected_action = "run_authorized" if benchmark_authorized is True else "skip"
        if benchmark != expected_action:
            reject(
                "BENCHMARK_AUTHORIZATION",
                f"The benchmark action must be {expected_action!r} for the recorded authorization",
            )
    if benchmark_pending is True and strategy != "none":
        reject(
            "PENDING_SELECTION", "Keep selected_strategy=none while evidence is pending"
        )
    qualified_fallback = facts.get("qualified_fallback")
    if qualified_fallback in {"ddp", "fsdp"} and strategy not in {
        qualified_fallback,
        "autoparallel",
        "none",
    }:
        reject(
            "UNQUALIFIED_FALLBACK",
            f"The recorded qualified fallback is {qualified_fallback.upper()}",
        )

    card = decision.get("decision_card")
    if card is None:
        if require_decision_card:
            reject(
                "MISSING_DECISION_CARD",
                "A final routing decision requires a machine-checkable decision_card",
            )
        return {"valid": not errors, "errors": errors}
    if not isinstance(card, Mapping):
        reject("INVALID_DECISION_CARD", "decision_card must be an object")
        return {"valid": False, "errors": errors}

    missing_fields = sorted(_DECISION_CARD_FIELDS - set(card))
    extra_fields = sorted(set(card) - _DECISION_CARD_FIELDS)
    if missing_fields:
        reject(
            "INCOMPLETE_DECISION_CARD",
            f"decision_card is missing fields: {missing_fields}",
        )
    if extra_fields:
        reject(
            "UNKNOWN_DECISION_CARD_FIELD",
            f"decision_card has unknown fields: {extra_fields}",
        )
    if missing_fields:
        return {"valid": False, "errors": errors}

    evidence_level = card["evidence_level"]
    boundary = card["boundary"]
    boundary_matched = card["boundary_matched"]
    reason_codes = card["reason_codes"]
    missing_evidence = card["missing_evidence"]
    fallback_strategy = card["fallback_strategy"]
    next_control = card["next_control"]

    if evidence_level not in _EVIDENCE_LEVELS:
        reject(
            "INVALID_EVIDENCE_LEVEL",
            f"decision_card evidence_level has invalid value {evidence_level!r}",
        )
    if not isinstance(boundary, str) or not boundary.strip():
        reject("INVALID_DECISION_BOUNDARY", "decision_card boundary must be non-empty")
    if not isinstance(boundary_matched, bool):
        reject(
            "INVALID_BOUNDARY_MATCH",
            "decision_card boundary_matched must be a boolean",
        )
    if not _valid_string_list(reason_codes, reason_codes=True):
        reject(
            "INVALID_REASON_CODES",
            "decision_card reason_codes must contain unique stable codes",
        )
        reason_codes = []
    if not _valid_string_list(missing_evidence):
        reject(
            "INVALID_MISSING_EVIDENCE",
            "decision_card missing_evidence must contain unique non-empty strings",
        )
        missing_evidence = []
    if fallback_strategy not in {"ddp", "fsdp", "none"}:
        reject(
            "INVALID_CARD_FALLBACK",
            f"decision_card fallback_strategy has invalid value {fallback_strategy!r}",
        )
    expected_fallback = strategy if strategy in {"ddp", "fsdp"} else "none"
    if fallback_strategy != expected_fallback:
        reject(
            "CARD_FALLBACK_MISMATCH",
            f"decision_card fallback_strategy must be {expected_fallback!r}",
        )
    if next_control is not None and (
        not isinstance(next_control, str) or not next_control.strip()
    ):
        reject(
            "INVALID_NEXT_CONTROL",
            "decision_card next_control must be null or a non-empty string",
        )

    if outcome == "accepted":
        if boundary_matched is not True or missing_evidence:
            reject(
                "ACCEPTED_CARD_HAS_EVIDENCE_GAP",
                "Acceptance requires a matched boundary and no missing evidence",
            )
        if evidence_level == "none":
            reject(
                "ACCEPTED_WITHOUT_EVIDENCE_LEVEL",
                "Acceptance requires an observed evidence level",
            )
    if outcome == "inconclusive":
        if boundary_matched is not False or not missing_evidence:
            reject(
                "INCONCLUSIVE_CARD_MISSING_GAP",
                "An inconclusive decision requires an unmatched boundary and explicit missing evidence",
            )
        if not isinstance(next_control, str) or not next_control.strip():
            reject(
                "INCONCLUSIVE_CARD_MISSING_CONTROL",
                "An inconclusive decision requires one next control",
            )
    if outcome in {"below_bar", "failed", "ineligible"} and not reason_codes:
        reject(
            "MISSING_OUTCOME_REASON_CODE",
            f"The {outcome} outcome requires a stable reason code",
        )

    if facts.get("required_evidence_missing") is True and not missing_evidence:
        reject(
            "CARD_OMITS_MISSING_EVIDENCE",
            "Observed missing evidence must be named in decision_card",
        )
    if facts.get("workload_representative") is False:
        _require_reason_code(
            reason_codes,
            "SYNTHETIC_UNDERFILLED_WORKLOAD",
            "CARD_OMITS_WORKLOAD_REASON",
            reject,
        )
    if facts.get("optimizer_update_modeled_or_measured") is False:
        _require_reason_code(
            reason_codes,
            "OPTIMIZER_PERFORMANCE_UNMODELED",
            "CARD_OMITS_OPTIMIZER_BOUNDARY_REASON",
            reject,
        )
        if outcome == "accepted":
            reject(
                "UNMODELED_OPTIMIZER_ACCEPTANCE",
                "Missing optimizer-update evidence cannot establish full-step throughput",
            )
    if (
        facts.get("composed_prediction") is True
        and facts.get("base_measurement_contemporaneous") is False
    ):
        _require_reason_code(
            reason_codes,
            "COMPOSED_FULL_STEP_BASE_DRIFT",
            "CARD_OMITS_BASE_DRIFT_REASON",
            reject,
        )
    if facts.get("global_microbatch_matches") is False:
        _require_reason_code(
            reason_codes,
            "BATCH_CONTRACT_MISMATCH",
            "CARD_OMITS_BATCH_MISMATCH_REASON",
            reject,
        )
        if outcome == "accepted":
            reject(
                "BATCH_MISMATCH_ACCEPTANCE",
                "A mismatched global/local batch contract cannot support acceptance",
            )
    if (
        strategy == "ddp"
        and facts.get("alternating_no_sync") is True
        and facts.get("ddp_static_graph") is True
        and facts.get("exact_static_graph_template_validated") is not True
    ):
        reject(
            "UNQUALIFIED_DDP_TEMPLATE",
            "Alternating no_sync requires static_graph=false unless that exact template executed successfully",
        )
    if strategy == "fsdp" and facts.get("fsdp_template_qualified") is False:
        reject(
            "UNQUALIFIED_FSDP_TEMPLATE",
            "The selected FSDP wrapping template is not correctness-qualified",
        )

    return {"valid": not errors, "errors": errors}


def _valid_string_list(value: Any, *, reason_codes: bool = False) -> bool:
    if not isinstance(value, list):
        return False
    if any(not isinstance(item, str) or not item.strip() for item in value):
        return False
    if len(value) != len(set(value)):
        return False
    return not reason_codes or all(_REASON_CODE_RE.fullmatch(item) for item in value)


def _require_reason_code(
    reason_codes: list[str],
    required: str,
    error_code: str,
    reject: Any,
) -> None:
    if required not in reason_codes:
        reject(error_code, f"decision_card must record {required}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("decision", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--require-decision-card", action="store_true")
    args = parser.parse_args()

    decision = json.loads(args.decision.read_text())
    evidence = json.loads(args.evidence.read_text()) if args.evidence else None
    result = validate_routing_classification(
        decision,
        evidence,
        require_decision_card=args.require_decision_card,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
