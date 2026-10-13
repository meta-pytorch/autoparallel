from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, TypeGuard

CATALOG_SCHEMA_VERSION = 1
_FAILURE_LOCATION_PATTERN = re.compile(
    r'File "(?P<path>[^"]+)", line \d+, in (?P<function>[^,\n]+)'
)
_EVIDENCE_LEVELS = (
    "captured",
    "ilp_feasible",
    "lowered",
    "compiled",
    "numerically_checked",
    "executed_on_target",
    "measured",
)


def build_report_catalog(
    report_root: str | Path, *, verify: bool = True
) -> dict[str, Any]:
    """Index v1 run bundles and aggregate diagnostic signals."""

    root = Path(report_root).resolve()
    runs = []
    invalid_bundles = []
    pending_bundles = []
    comparisons = []

    for report_path in sorted(root.glob("v1/*/*/*/report.json")):
        bundle = report_path.parent
        try:
            manifest_path = bundle / "manifest.json"
            if not manifest_path.is_file():
                pending_bundles.append(str(bundle.relative_to(root)))
                continue
            manifest = _read_json(manifest_path)
            if manifest.get("status") == "running":
                pending_bundles.append(str(bundle.relative_to(root)))
                continue
            report = _read_json(report_path)
            if report.get("schema_version") != 1:
                raise ValueError(
                    f"unsupported report schema {report.get('schema_version')!r}"
                )
            if verify:
                _verify_manifest(bundle, manifest, report)
            run_comparisons = _measurement_comparisons(report)
            runs.append(
                _catalog_record(
                    root,
                    bundle,
                    report,
                    measurement_comparisons=run_comparisons,
                )
            )
            comparisons.extend(run_comparisons)
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            json.JSONDecodeError,
        ) as error:
            invalid_bundles.append(
                {
                    "path": str(bundle.relative_to(root)),
                    "error": str(error),
                }
            )

    return {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "report_root": str(root),
        "verification_enabled": verify,
        "summary": _summarize_runs(runs),
        "measurement_error": _summarize_comparisons(comparisons),
        "runs": runs,
        "pending_bundles": pending_bundles,
        "invalid_bundles": invalid_bundles,
    }


def write_report_catalog(
    report_root: str | Path,
    *,
    output: str | Path | None = None,
    verify: bool = True,
) -> Path:
    root = Path(report_root).resolve()
    catalog = build_report_catalog(root, verify=verify)
    output_path = Path(output) if output is not None else root / "catalog-v1.json"
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(output_path, catalog)
    return output_path


def _catalog_record(
    root: Path,
    bundle: Path,
    report: Mapping[str, Any],
    *,
    measurement_comparisons: list[dict[str, Any]],
) -> dict[str, Any]:
    cluster = report.get("cluster", {})
    evidence = report.get("evidence", {})
    planning = report.get("planning", {})
    evaluation = report.get("evaluation", {})
    parameter_storage = planning.get("summary", {}).get("parameter_storage", {})
    cost_model = evaluation.get("cost_model") or report.get("provenance", {}).get(
        "cost_model", {}
    )
    failure = report.get("failure")
    findings = report.get("findings", [])
    artifacts = report.get("artifacts", [])
    optimizer_replay = next(
        (
            artifact
            for artifact in artifacts
            if artifact.get("role") == "optimizer_replay"
        ),
        None,
    )
    reproduction = report.get("provenance", {}).get("reproduction", {})
    external_sources = reproduction.get("external_sources", {})
    request = report.get("request", {})
    campaign = request.get("campaign", {})
    workload_request = {
        key: value for key, value in request.items() if key != "campaign"
    }
    parallelization = report.get("parallelization")

    return {
        "run_id": report["run_id"],
        "status": report["status"],
        "stage": report["stage"],
        "created_at": report["created_at"],
        "completed_at": report.get("completed_at"),
        "cluster_profile": cluster.get("profile"),
        "evidence": evidence.get("highest_level", "none"),
        "completed_evidence": list(evidence.get("completed", [])),
        "request_sha256": _canonical_sha256(request),
        "workload_sha256": _canonical_sha256(workload_request),
        "provenance_sha256": _canonical_sha256(report.get("provenance", {})),
        "reproduction_complete": reproduction.get("complete"),
        "external_source_count": len(external_sources),
        "external_sources_complete": all(
            source.get("git_revision") for source in external_sources.values()
        ),
        "campaign": (
            {
                key: campaign.get(key)
                for key in ("id", "suite", "case_id", "expected_case_count")
                if campaign.get(key) is not None
            }
            if isinstance(campaign, Mapping) and campaign.get("id")
            else None
        ),
        "parallelization": _catalog_parallelization(parallelization),
        "cost_model_status": cost_model.get("status"),
        "cost_model_source": cost_model.get("source"),
        "parameter_local_to_global_fraction": parameter_storage.get(
            "local_to_global_fraction"
        ),
        "findings": [
            {
                "code": finding.get("code"),
                "severity": finding.get("severity"),
                "context_signature": (
                    f"sha256:{_canonical_sha256(finding['context'])}"
                    if isinstance(finding.get("context"), Mapping)
                    and finding["context"]
                    else None
                ),
            }
            for finding in findings
        ],
        "failure": (
            {
                "type": failure.get("type"),
                "stage": failure.get("stage"),
                "substage": failure.get("substage"),
                "signature": failure.get("signature"),
                "diagnostic_location_signature": _failure_location_signature(
                    str(failure.get("message") or "")
                ),
                "classification": _failure_classification(failure),
            }
            if isinstance(failure, Mapping)
            else None
        ),
        "artifact_roles": sorted(
            {
                artifact.get("role")
                for artifact in artifacts
                if artifact.get("role") is not None
            }
        ),
        "optimizer_replay": (
            {
                "available": True,
                "name": optimizer_replay.get("name"),
                "path": optimizer_replay.get("path"),
                "sha256": optimizer_replay.get("sha256"),
                "size_bytes": optimizer_replay.get("size_bytes"),
            }
            if optimizer_replay is not None
            else {"available": False}
        ),
        "measurement_comparisons": [
            {
                key: comparison[key]
                for key in (
                    "configuration_hash",
                    "metric",
                    "unit",
                    "absolute_error",
                    "relative_error",
                )
            }
            for comparison in measurement_comparisons
        ],
        "time_to_ready": _catalog_time_to_ready(report),
        "bundle": str(bundle.relative_to(root)),
    }


def _failure_location_signature(message: str) -> str | None:
    locations = []
    for match in _FAILURE_LOCATION_PATTERN.finditer(message):
        path = Path(match.group("path"))
        locations.append(
            {
                "path_tail": "/".join(path.parts[-3:]),
                "function": match.group("function").strip(),
            }
        )
    if not locations:
        return None
    return f"sha256:{_canonical_sha256(locations[-4:])}"


def _failure_classification(failure: Mapping[str, Any]) -> str:
    message = str(failure.get("message") or "").lower()
    if "out of memory" in message or "exhausted device memory" in message:
        return "resource_exhaustion"
    if failure.get("type") == "Unsupported" and "data-dependent" in message:
        return "capture_unsupported"
    return "execution_failure"


def _summarize_runs(runs: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    run_list = list(runs)
    statuses = Counter(run["status"] for run in run_list)
    evidence = Counter(run["evidence"] for run in run_list)
    profiles = Counter(run["cluster_profile"] for run in run_list)
    finding_codes = Counter(
        finding["code"]
        for run in run_list
        for finding in run["findings"]
        if finding["code"] is not None
    )
    failure_types = Counter(
        run["failure"]["type"]
        for run in run_list
        if run["failure"] is not None and run["failure"]["type"] is not None
    )
    failure_stages = Counter(
        run["failure"]["stage"]
        for run in run_list
        if run["failure"] is not None and run["failure"]["stage"] is not None
    )
    failure_substages = Counter(
        run["failure"]["substage"]
        for run in run_list
        if run["failure"] is not None and run["failure"].get("substage") is not None
    )
    parallelization = _summarize_parallelization(run_list)
    first_warmups = [
        float(run["time_to_ready"]["first_iteration_seconds"])
        for run in run_list
        if run.get("time_to_ready") is not None
    ]
    return {
        "run_count": len(run_list),
        "status_counts": _sorted_counter(statuses),
        "evidence_counts": _sorted_counter(evidence),
        "cluster_profile_counts": _sorted_counter(profiles),
        "finding_code_counts": _sorted_counter(finding_codes),
        "failure_type_counts": _sorted_counter(failure_types),
        "failure_stage_counts": _sorted_counter(failure_stages),
        "failure_substage_counts": _sorted_counter(failure_substages),
        "parallelization": parallelization,
        "time_to_ready": {
            "run_count": len(first_warmups),
            "mean_first_iteration_seconds": (
                mean(first_warmups) if first_warmups else None
            ),
            "maximum_first_iteration_seconds": (
                max(first_warmups) if first_warmups else None
            ),
        },
        "campaigns": _summarize_campaigns(run_list),
    }


def _catalog_time_to_ready(report: Mapping[str, Any]) -> dict[str, Any] | None:
    for validation in report.get("validation", {}).values():
        result = validation.get("result", {})
        timing = result.get("timing", {})
        warmup = timing.get("warmup_wall_time")
        if not isinstance(warmup, Mapping):
            continue
        first = warmup.get("first_iteration_seconds")
        total = warmup.get("total_seconds")
        if not _is_number(first) or not _is_number(total):
            continue
        boundary = result.get("boundary", {})
        return {
            "validation_kind": result.get("kind"),
            "first_iteration_seconds": float(first),
            "total_warmup_seconds": float(total),
            "optimizer_update": (
                boundary.get("optimizer_update")
                if isinstance(boundary, Mapping)
                else None
            ),
            "rank_reduction": warmup.get("rank_reduction"),
        }
    return None


def _catalog_parallelization(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    autoparallel = value.get("autoparallel")
    fallback = value.get("fallback")
    competitiveness = value.get("competitiveness")
    decision = value.get("decision")
    intent = value.get("intent", {})
    return {
        "decision_id": value.get("decision_id"),
        "intent_kind": intent.get("kind") if isinstance(intent, Mapping) else None,
        "autoparallel_outcome": (
            autoparallel.get("outcome") if isinstance(autoparallel, Mapping) else None
        ),
        "autoparallel_reason_codes": (
            list(autoparallel.get("reason_codes", []))
            if isinstance(autoparallel, Mapping)
            else []
        ),
        "competitiveness_verdict": (
            competitiveness.get("verdict")
            if isinstance(competitiveness, Mapping)
            else None
        ),
        "competitiveness_evidence": (
            competitiveness.get("evidence")
            if isinstance(competitiveness, Mapping)
            else None
        ),
        "competitiveness_reason_codes": (
            list(competitiveness.get("reason_codes", []))
            if isinstance(competitiveness, Mapping)
            else []
        ),
        "baseline_strategies": (
            sorted(
                {
                    baseline.get("strategy")
                    for baseline in competitiveness.get("baselines", [])
                    if isinstance(baseline, Mapping)
                    and baseline.get("strategy") is not None
                }
            )
            if isinstance(competitiveness, Mapping)
            else []
        ),
        "baseline_feasibility_counts": (
            _sorted_counter(
                Counter(
                    baseline.get("feasibility")
                    for baseline in competitiveness.get("baselines", [])
                    if isinstance(baseline, Mapping)
                    and baseline.get("feasibility") is not None
                )
            )
            if isinstance(competitiveness, Mapping)
            else {}
        ),
        "fallback_strategy": (
            fallback.get("strategy") if isinstance(fallback, Mapping) else None
        ),
        "fallback_outcome": (
            fallback.get("outcome") if isinstance(fallback, Mapping) else None
        ),
        "fallback_reason_codes": (
            list(fallback.get("reason_codes", []))
            if isinstance(fallback, Mapping)
            else []
        ),
        "selected_strategy": (
            decision.get("selected_strategy") if isinstance(decision, Mapping) else None
        ),
        "decision_status": (
            decision.get("status") if isinstance(decision, Mapping) else None
        ),
        "decision_reason_codes": (
            list(decision.get("reason_codes", []))
            if isinstance(decision, Mapping)
            else []
        ),
    }


def _summarize_parallelization(
    runs: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    report_records = [
        run for run in runs if isinstance(run.get("parallelization"), Mapping)
    ]
    grouped_decisions: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for run in sorted(
        report_records,
        key=lambda record: str(record.get("completed_at") or record.get("created_at")),
    ):
        record = run["parallelization"]
        decision_id = record.get("decision_id")
        if not decision_id:
            continue
        grouped_decisions[str(decision_id)].append(record)

    decisions: dict[str, dict[str, Any]] = {}
    terminal_statuses = {"accepted", "fallback_selected", "failed"}
    for decision_id, decision_records in grouped_decisions.items():
        terminal_indexes = [
            index
            for index, record in enumerate(decision_records)
            if record.get("decision_status") in terminal_statuses
        ]
        if terminal_indexes:
            decision_records = decision_records[: terminal_indexes[-1] + 1]
        combined = decisions.setdefault(decision_id, {})
        for record in decision_records:
            for field in (
                "intent_kind",
                "autoparallel_outcome",
                "autoparallel_reason_codes",
                "competitiveness_verdict",
                "competitiveness_evidence",
                "competitiveness_reason_codes",
                "baseline_strategies",
                "baseline_feasibility_counts",
                "selected_strategy",
                "decision_status",
                "decision_reason_codes",
                "fallback_strategy",
                "fallback_outcome",
                "fallback_reason_codes",
            ):
                value = record.get(field)
                if field.endswith("_reason_codes") or field == "baseline_strategies":
                    if value:
                        combined[field] = value
                elif field == "intent_kind":
                    if (
                        value is not None
                        and (
                            combined.get(field) is None
                            or value != "diagnostic_evaluation"
                        )
                    ):
                        combined[field] = value
                elif value is not None:
                    combined[field] = value
    all_records = list(decisions.values())
    diagnostic_records = [
        record
        for record in all_records
        if record.get("intent_kind") == "diagnostic_evaluation"
    ]
    records = [
        record
        for record in all_records
        if record.get("intent_kind") != "diagnostic_evaluation"
    ]
    return {
        "report_count": len(report_records),
        "decision_count": len(records),
        "diagnostic_evaluation_count": len(diagnostic_records),
        "autoparallel_outcome_counts": _sorted_counter(
            Counter(
                record["autoparallel_outcome"]
                for record in records
                if record.get("autoparallel_outcome") is not None
            )
        ),
        "autoparallel_reason_code_counts": _sorted_counter(
            Counter(
                code
                for record in records
                for code in record.get("autoparallel_reason_codes", [])
            )
        ),
        "competitiveness_verdict_counts": _sorted_counter(
            Counter(
                record["competitiveness_verdict"]
                for record in records
                if record.get("competitiveness_verdict") is not None
            )
        ),
        "competitiveness_evidence_counts": _sorted_counter(
            Counter(
                record["competitiveness_evidence"]
                for record in records
                if record.get("competitiveness_evidence") is not None
            )
        ),
        "competitiveness_reason_code_counts": _sorted_counter(
            Counter(
                code
                for record in records
                for code in record.get("competitiveness_reason_codes", [])
            )
        ),
        "baseline_strategy_counts": _sorted_counter(
            Counter(
                strategy
                for record in records
                for strategy in record.get("baseline_strategies", [])
            )
        ),
        "baseline_feasibility_counts": _sorted_counter(
            sum(
                (
                    Counter(record.get("baseline_feasibility_counts", {}))
                    for record in records
                ),
                Counter(),
            )
        ),
        "selected_strategy_counts": _sorted_counter(
            Counter(
                record["selected_strategy"]
                for record in records
                if record.get("selected_strategy") is not None
            )
        ),
        "decision_status_counts": _sorted_counter(
            Counter(
                record["decision_status"]
                for record in records
                if record.get("decision_status") is not None
            )
        ),
        "decision_reason_code_counts": _sorted_counter(
            Counter(
                code
                for record in records
                for code in record.get("decision_reason_codes", [])
            )
        ),
        "fallback_strategy_counts": _sorted_counter(
            Counter(
                record["fallback_strategy"]
                for record in records
                if record.get("fallback_strategy") is not None
            )
        ),
        "fallback_outcome_counts": _sorted_counter(
            Counter(
                record["fallback_outcome"]
                for record in records
                if record.get("fallback_outcome") is not None
            )
        ),
        "fallback_reason_code_counts": _sorted_counter(
            Counter(
                code
                for record in records
                for code in record.get("fallback_reason_codes", [])
            )
        ),
        "incomplete_decision_count": sum(
            record.get("decision_status") is None for record in records
        ),
        "missing_competitiveness_count": sum(
            record.get("selected_strategy") == "autoparallel"
            and record.get("competitiveness_verdict") is None
            for record in records
        ),
    }


def _summarize_campaigns(
    runs: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for run in runs:
        campaign = run.get("campaign")
        if isinstance(campaign, Mapping) and campaign.get("id"):
            grouped[str(campaign["id"])].append(run)

    summaries = []
    for campaign_id, records in sorted(grouped.items()):
        latest_by_case: dict[str, Mapping[str, Any]] = {}
        for record in records:
            campaign = record["campaign"]
            case_id = str(campaign.get("case_id") or record["run_id"])
            previous = latest_by_case.get(case_id)
            timestamp = str(record.get("completed_at") or record.get("created_at"))
            previous_timestamp = (
                str(previous.get("completed_at") or previous.get("created_at"))
                if previous is not None
                else ""
            )
            if previous is None or timestamp > previous_timestamp:
                latest_by_case[case_id] = record
        latest = list(latest_by_case.values())
        expected_values = {
            int(record["campaign"]["expected_case_count"])
            for record in records
            if record["campaign"].get("expected_case_count") is not None
        }
        expected = max(expected_values) if expected_values else None
        completed_evidence = {
            level: sum(
                level in record.get("completed_evidence", []) for record in latest
            )
            for level in _EVIDENCE_LEVELS
        }
        suites = sorted(
            {
                str(record["campaign"]["suite"])
                for record in records
                if record["campaign"].get("suite") is not None
            }
        )
        summaries.append(
            {
                "id": campaign_id,
                "suites": suites,
                "run_count": len(records),
                "reported_case_count": len(latest),
                "expected_case_count": expected,
                "missing_case_count": (
                    max(0, expected - len(latest)) if expected is not None else None
                ),
                "status_counts": _sorted_counter(
                    Counter(record["status"] for record in latest)
                ),
                "evidence_counts": _sorted_counter(
                    Counter(record["evidence"] for record in latest)
                ),
                "failure_type_counts": _sorted_counter(
                    Counter(
                        record["failure"]["type"]
                        for record in latest
                        if record.get("failure") is not None
                    )
                ),
                "failure_substage_counts": _sorted_counter(
                    Counter(
                        record["failure"]["substage"]
                        for record in latest
                        if record.get("failure") is not None
                        and record["failure"].get("substage") is not None
                    )
                ),
                "completed_evidence_counts": completed_evidence,
            }
        )
    return summaries


def _measurement_comparisons(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    profile = report.get("cluster", {}).get("profile")
    comparisons = []
    for validation in report.get("validation", {}).values():
        if validation.get("evidence_level") != "measured":
            continue
        for comparison in validation.get("result", {}).get("comparisons", []):
            if not isinstance(comparison, Mapping):
                continue
            required = {"configuration_hash", "metric", "unit", "estimated", "observed"}
            if not required <= comparison.keys():
                continue
            if not all(
                isinstance(comparison[field], str)
                for field in ("configuration_hash", "metric", "unit")
            ):
                continue
            estimated = comparison["estimated"]
            observed = comparison["observed"]
            if not _is_number(estimated) or not _is_number(observed):
                continue
            absolute_error = abs(float(estimated) - float(observed))
            comparisons.append(
                {
                    "cluster_profile": profile,
                    "configuration_hash": comparison["configuration_hash"],
                    "metric": comparison["metric"],
                    "unit": comparison["unit"],
                    "absolute_error": absolute_error,
                    "relative_error": (
                        absolute_error / abs(float(observed))
                        if float(observed) != 0.0
                        else None
                    ),
                }
            )
    return comparisons


def _summarize_comparisons(
    comparisons: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for comparison in comparisons:
        key = (
            comparison["cluster_profile"],
            comparison["configuration_hash"],
            comparison["metric"],
            comparison["unit"],
        )
        grouped[key].append(comparison)

    result = []
    for key, records in sorted(grouped.items(), key=lambda item: str(item[0])):
        relative_errors = [
            record["relative_error"]
            for record in records
            if record["relative_error"] is not None
        ]
        result.append(
            {
                "cluster_profile": key[0],
                "configuration_hash": key[1],
                "metric": key[2],
                "unit": key[3],
                "count": len(records),
                "mean_absolute_error": mean(
                    record["absolute_error"] for record in records
                ),
                "mean_relative_error": (
                    mean(relative_errors) if relative_errors else None
                ),
            }
        )
    return result


def _verify_manifest(
    bundle: Path, manifest: Mapping[str, Any], report: Mapping[str, Any]
) -> None:
    if manifest.get("schema_version") != 1:
        raise ValueError(
            f"unsupported manifest schema {manifest.get('schema_version')!r}"
        )
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("manifest has no files list")
    if manifest.get("run_id") != report.get("run_id"):
        raise ValueError("manifest run_id does not match report")
    if manifest.get("status") != report.get("status"):
        raise ValueError("manifest status does not match report")
    if manifest.get("stage") != report.get("stage"):
        raise ValueError("manifest stage does not match report")
    bundle_root = bundle.resolve()
    manifest_paths = set()
    for entry in files:
        relative_path = entry.get("path")
        expected_hash = entry.get("sha256")
        if not isinstance(relative_path, str) or not isinstance(expected_hash, str):
            raise ValueError("manifest file entry is missing path or sha256")
        if relative_path in manifest_paths:
            raise ValueError(f"duplicate manifest file entry: {relative_path}")
        manifest_paths.add(relative_path)
        path = (bundle / relative_path).resolve()
        if not path.is_relative_to(bundle_root):
            raise ValueError(f"manifest path escapes the bundle: {relative_path}")
        if not path.is_file():
            raise ValueError(f"manifest file is missing: {relative_path}")
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(f"manifest hash mismatch: {relative_path}")
    missing_required = {"report.json", "summary.md"} - manifest_paths
    if missing_required:
        raise ValueError(
            f"manifest is missing required files: {sorted(missing_required)}"
        )
    actual_paths = set()
    for path in bundle.rglob("*"):
        if path.is_symlink():
            raise ValueError(
                f"bundle contains a symbolic link: {path.relative_to(bundle)}"
            )
        if path.is_file():
            actual_paths.add(str(path.relative_to(bundle)))
    unexpected = actual_paths - manifest_paths - {"manifest.json"}
    if unexpected:
        raise ValueError(f"unmanifested bundle files: {sorted(unexpected)}")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _is_number(value: Any) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sorted_counter(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): counter[key] for key in sorted(counter, key=str)}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a verified catalog of AutoParallel run reports"
    )
    parser.add_argument("report_root", help="Filesystem report root")
    parser.add_argument("--output", help="Output path; defaults inside report_root")
    parser.add_argument(
        "--no-verify", action="store_true", help="Skip manifest hash verification"
    )
    args = parser.parse_args()
    output = write_report_catalog(
        args.report_root,
        output=args.output,
        verify=not args.no_verify,
    )
    print(output)


if __name__ == "__main__":
    main()
