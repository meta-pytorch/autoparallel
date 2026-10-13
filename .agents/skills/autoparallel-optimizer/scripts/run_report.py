from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Mapping

REPORT_SCHEMA_VERSION = 1
PROFILE_SCHEMA_VERSION = 1

_PROFILE_ENV = "AUTOPARALLEL_CLUSTER_PROFILE"
_PROFILES_PATH_ENV = "AUTOPARALLEL_CLUSTER_PROFILES"
_REPORT_ROOT_ENV = "AUTOPARALLEL_REPORT_ROOT"

_STAGES = (
    "started",
    "captured",
    "planned",
    "lowered",
    "compiled",
    "numerically_checked",
    "executed_on_target",
    "measured",
)
_EVIDENCE_LEVELS = (
    "none",
    "captured",
    "ilp_feasible",
    "lowered",
    "compiled",
    "numerically_checked",
    "executed_on_target",
    "measured",
)
_STAGE_FOR_EVIDENCE = {
    "captured": "captured",
    "ilp_feasible": "planned",
    "lowered": "lowered",
    "compiled": "compiled",
    "numerically_checked": "numerically_checked",
    "executed_on_target": "executed_on_target",
    "measured": "measured",
}
_VALIDATION_EVIDENCE = {
    "compiled",
    "numerically_checked",
    "executed_on_target",
    "measured",
}
_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_FINDING_CODE_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_AUTOPARALLEL_OUTCOMES = {
    "accepted",
    "below_bar",
    "failed",
    "ineligible",
    "inconclusive",
}
_FALLBACK_OUTCOMES = {"accepted", "below_bar", "failed", "inconclusive"}
_PARALLELIZATION_DECISION_STATUSES = {
    "accepted",
    "fallback_selected",
    "failed",
    "inconclusive",
}
_PARALLELIZATION_STRATEGIES = {"autoparallel", "ddp", "fsdp", "none"}
_COMPETITIVENESS_VERDICTS = {
    "robustly_competitive",
    "robustly_worse",
    "pareto_dominated",
    "schedule_sensitive",
    "baseline_infeasible",
    "comparison_unavailable",
    "user_accepted_tradeoff",
}
_COMPETITIVENESS_EVIDENCE = {"none", "modeled", "measured", "mixed"}
_BASELINE_FEASIBILITY = {"feasible", "infeasible", "unknown"}
_OPERATIONAL_PROFILE_FIELDS = {
    "capture_level",
    "diagnostics",
    "feedback",
    "retention_days",
    "routing",
}
_OPERATIONAL_REQUEST_FIELDS = {"campaign"}


@dataclass(frozen=True)
class ResolvedReportConfig:
    profile_name: str
    profile: dict[str, Any]
    report_root: Path
    selection_source: str
    report_root_source: str
    config_sources: tuple[str, ...]


def resolve_report_config(
    *,
    profile_name: str | None = None,
    profiles_path: str | Path | None = None,
    report_root: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
) -> ResolvedReportConfig:
    """Resolve profile and filesystem destination with explicit provenance.

    Precedence is explicit arguments, environment, matched/default profile.
    Profile matching uses declared environment metadata and never device
    visibility.
    """

    env = dict(os.environ if environ is None else environ)
    base = Path.cwd() if cwd is None else Path(cwd)
    builtin_path = (
        Path(__file__).resolve().parents[1] / "references/cluster-profiles.json"
    )
    documents = [_load_profiles(builtin_path)]
    sources = ["builtin"]

    external_path = profiles_path or env.get(_PROFILES_PATH_ENV)
    if external_path is not None:
        external = Path(external_path)
        if not external.is_absolute():
            external = base / external
        external_document = _load_profiles(external)
        documents.append(external_document)
        sources.append(f"external:sha256:{_canonical_sha256(external_document)}")

    profile_document = _merge_profile_documents(documents)
    profiles = profile_document["profiles"]

    requested_profile = profile_name or env.get(_PROFILE_ENV)
    if requested_profile is not None:
        selection_source = "explicit" if profile_name is not None else "environment"
    else:
        matches = [
            name
            for name, profile in profiles.items()
            if _profile_matches_environment(profile, env)
        ]
        if len(matches) > 1:
            raise ValueError(
                "Multiple AutoParallel cluster profiles match the environment: "
                f"{sorted(matches)}; set {_PROFILE_ENV} explicitly"
            )
        if matches:
            requested_profile = matches[0]
            selection_source = "environment_match"
        else:
            requested_profile = profile_document["default_profile"]
            selection_source = "default"

    _validate_label(requested_profile, "cluster profile")
    if requested_profile not in profiles:
        raise ValueError(f"Unknown AutoParallel cluster profile: {requested_profile}")
    profile = _json_copy(profiles[requested_profile])

    if report_root is not None:
        root_value = report_root
        root_source = "explicit"
    elif env.get(_REPORT_ROOT_ENV):
        root_value = env[_REPORT_ROOT_ENV]
        root_source = "environment"
    else:
        root_value = profile.get("report_root")
        root_source = "profile"
    if not root_value:
        raise ValueError(
            f"Cluster profile {requested_profile!r} has no report_root and no "
            f"{_REPORT_ROOT_ENV} override was provided"
        )
    if "://" in str(root_value):
        raise ValueError("The v1 report writer supports filesystem report roots only")
    root = Path(root_value)
    if not root.is_absolute():
        root = base / root

    return ResolvedReportConfig(
        profile_name=requested_profile,
        profile=profile,
        report_root=root.resolve(),
        selection_source=selection_source,
        report_root_source=root_source,
        config_sources=tuple(sources),
    )


class AutoParallelRunReport:
    """Write one self-contained, immutable-on-finalization report bundle."""

    def __init__(
        self,
        *,
        request: Mapping[str, Any],
        provenance: Mapping[str, Any] | None = None,
        config: ResolvedReportConfig | None = None,
        profile_name: str | None = None,
        profiles_path: str | Path | None = None,
        report_root: str | Path | None = None,
        environ: Mapping[str, str] | None = None,
        run_id: str | None = None,
        source_root: str | Path | None = None,
        parallelization: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = config or resolve_report_config(
            profile_name=profile_name,
            profiles_path=profiles_path,
            report_root=report_root,
            environ=environ,
        )
        self.run_id = run_id or _make_run_id()
        _validate_label(self.run_id, "run id")
        self._source_root = Path(source_root or Path.cwd()).resolve()

        created_at = _utc_now()
        day = created_at[:10]
        self.path = (
            self.config.report_root
            / f"v{REPORT_SCHEMA_VERSION}"
            / self.config.profile_name
            / day
            / self.run_id
        )
        self.path.mkdir(parents=True, exist_ok=False)
        self.artifacts_path = self.path / "artifacts"
        self.artifacts_path.mkdir()
        self._sealed = False

        profile_snapshot = {
            key: value
            for key, value in self.config.profile.items()
            if key not in {"match_env", "report_root"}
        }
        self.report: dict[str, Any] = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "status": "running",
            "stage": "started",
            "created_at": created_at,
            "completed_at": None,
            "cluster": {
                "profile": self.config.profile_name,
                "selection_source": self.config.selection_source,
                "report_root_source": self.config.report_root_source,
                "config_sources": list(self.config.config_sources),
                "resolved": profile_snapshot,
            },
            "request": _json_copy(request),
            "provenance": _json_copy(provenance or {}),
            "evidence": {"highest_level": "none", "completed": []},
            "planning": {},
            "evaluation": {},
            "validation": {},
            "findings": [],
            "failure": None,
            "parallelization": None,
            "artifacts": [],
        }
        if parallelization is not None:
            self.report["parallelization"] = _new_parallelization_record(
                parallelization,
                default_decision_id=self.run_id,
            )
        self._flush()

    def __enter__(self) -> "AutoParallelRunReport":
        return self

    @property
    def finalized(self) -> bool:
        return self._sealed

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: TracebackType | None,
    ) -> Literal[False]:
        if self._sealed:
            return False
        if exc_value is not None:
            self.record_failure(exc_value, exc_traceback)
        else:
            self.complete()
        return False

    def mark_stage(self, stage: str, *, evidence_level: str | None = None) -> None:
        self._ensure_open()
        if stage not in _STAGES:
            raise ValueError(f"Unknown report stage: {stage}")
        current_index = _STAGES.index(self.report["stage"])
        if _STAGES.index(stage) < current_index:
            raise ValueError(
                f"Cannot move report stage backward from {self.report['stage']} to {stage}"
            )
        self.report["stage"] = stage
        if evidence_level is not None:
            self._mark_evidence(evidence_level)
        self._flush()

    def record_runtime_context(self, cost_model: Mapping[str, Any]) -> None:
        self._ensure_open()
        provenance = self.report["provenance"]
        provenance["cost_model"] = _json_copy(cost_model)
        fingerprint_provenance = {
            key: value
            for key, value in provenance.items()
            if key != "configuration_sha256"
        }
        fingerprint = {
            "request": {
                key: value
                for key, value in self.report["request"].items()
                if key not in _OPERATIONAL_REQUEST_FIELDS
            },
            "cluster": {
                "profile": self.report["cluster"]["profile"],
                "resolved": {
                    key: value
                    for key, value in self.report["cluster"]["resolved"].items()
                    if key not in _OPERATIONAL_PROFILE_FIELDS
                },
            },
            "provenance": fingerprint_provenance,
        }
        provenance["configuration_sha256"] = f"sha256:{_canonical_sha256(fingerprint)}"
        self._flush()

    def mark_substage(self, substage: str) -> None:
        self._ensure_open()
        _validate_label(substage, "run substage")
        now = _utc_now()
        planning = self.report["planning"]
        history = planning.setdefault("substage_history", [])
        if history and "completed_at" not in history[-1]:
            history[-1]["completed_at"] = now
            history[-1]["status"] = "completed"
        history.append({"name": substage, "started_at": now})
        planning["current_substage"] = substage
        self._flush()

    def configure_parallelization(
        self,
        *,
        intent: Mapping[str, Any],
        acceptance_criteria: Mapping[str, Any],
        decision_id: str | None = None,
    ) -> None:
        self._ensure_open()
        if self.report["parallelization"] is not None:
            raise RuntimeError("Parallelization routing is already configured")
        self.report["parallelization"] = _new_parallelization_record(
            {
                "decision_id": decision_id or self.run_id,
                "intent": intent,
                "acceptance_criteria": acceptance_criteria,
            },
            default_decision_id=self.run_id,
        )
        self._flush()

    def record_autoparallel_outcome(
        self,
        outcome: str,
        *,
        reason_codes: list[str] | tuple[str, ...] = (),
        details: Mapping[str, Any] | None = None,
        related_run_id: str | None = None,
    ) -> None:
        self._ensure_open()
        if outcome not in _AUTOPARALLEL_OUTCOMES:
            raise ValueError(f"Invalid AutoParallel outcome: {outcome}")
        routing = self._require_parallelization()
        if routing["autoparallel"] is not None:
            raise RuntimeError("AutoParallel outcome is already recorded")
        routing["autoparallel"] = {
            "outcome": outcome,
            "reason_codes": _validated_reason_codes(reason_codes),
            "evidence_level": self.report["evidence"]["highest_level"],
            "stage": self.report["stage"],
            "details": _json_copy(details or {}),
            "related_run_id": related_run_id,
            "recorded_at": _utc_now(),
        }
        self._flush()

    def record_fallback_outcome(
        self,
        strategy: str,
        outcome: str,
        *,
        reason_codes: list[str] | tuple[str, ...] = (),
        evidence_level: str = "none",
        details: Mapping[str, Any] | None = None,
        related_run_id: str | None = None,
    ) -> None:
        self._ensure_open()
        if strategy not in {"ddp", "fsdp"}:
            raise ValueError(f"Invalid fallback strategy: {strategy}")
        if outcome not in _FALLBACK_OUTCOMES:
            raise ValueError(f"Invalid fallback outcome: {outcome}")
        if evidence_level not in _EVIDENCE_LEVELS:
            raise ValueError(f"Invalid fallback evidence level: {evidence_level}")
        routing = self._require_parallelization()
        if routing["fallback"] is not None:
            raise RuntimeError("Fallback outcome is already recorded")
        routing["fallback"] = {
            "strategy": strategy,
            "outcome": outcome,
            "reason_codes": _validated_reason_codes(reason_codes),
            "evidence_level": evidence_level,
            "details": _json_copy(details or {}),
            "related_run_id": related_run_id,
            "recorded_at": _utc_now(),
        }
        self._flush()

    def record_competitiveness_assessment(
        self,
        verdict: str,
        *,
        evidence: str,
        autoparallel_estimate: Mapping[str, Any] | None = None,
        baselines: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...] = (),
        reason_codes: list[str] | tuple[str, ...] = (),
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self._ensure_open()
        if verdict not in _COMPETITIVENESS_VERDICTS:
            raise ValueError(f"Invalid competitiveness verdict: {verdict}")
        if evidence not in _COMPETITIVENESS_EVIDENCE:
            raise ValueError(f"Invalid competitiveness evidence: {evidence}")
        routing = self._require_parallelization()
        if routing.get("competitiveness") is not None:
            raise RuntimeError("Competitiveness assessment is already recorded")
        assessment = {
            "verdict": verdict,
            "evidence": evidence,
            "autoparallel_estimate": (
                _json_copy(autoparallel_estimate)
                if autoparallel_estimate is not None
                else None
            ),
            "baselines": _json_copy(list(baselines)),
            "reason_codes": _validated_reason_codes(reason_codes),
            "details": _json_copy(details or {}),
            "recorded_at": _utc_now(),
        }
        _validate_competitiveness_assessment(assessment)
        routing["competitiveness"] = assessment
        self._flush()

    def record_parallelization_decision(
        self,
        selected_strategy: str,
        status: str,
        *,
        reason_codes: list[str] | tuple[str, ...] = (),
    ) -> None:
        self._ensure_open()
        if selected_strategy not in _PARALLELIZATION_STRATEGIES:
            raise ValueError(
                f"Invalid selected parallelization strategy: {selected_strategy}"
            )
        if status not in _PARALLELIZATION_DECISION_STATUSES:
            raise ValueError(f"Invalid parallelization decision status: {status}")
        routing = self._require_parallelization()
        if routing["decision"] is not None:
            raise RuntimeError("Parallelization decision is already recorded")
        reasons = _validated_reason_codes(reason_codes)
        _validate_parallelization_decision(routing, selected_strategy, status)
        routing["decision"] = {
            "selected_strategy": selected_strategy,
            "status": status,
            "reason_codes": reasons,
            "recorded_at": _utc_now(),
        }
        if selected_strategy in {"ddp", "fsdp"}:
            self.add_finding(
                "AUTOPARALLEL_FALLBACK",
                f"AutoParallel was not selected; {selected_strategy.upper()} is the fallback",
                context={
                    "decision_id": routing["decision_id"],
                    "selected_strategy": selected_strategy,
                    "reason_codes": reasons,
                    "autoparallel_outcome": (routing["autoparallel"] or {}).get(
                        "outcome"
                    ),
                },
                suggested_action=(
                    "Inspect the linked AutoParallel attempt before changing fallback policy"
                ),
            )
        self._flush()

    def _require_parallelization(self) -> dict[str, Any]:
        routing = self.report["parallelization"]
        if routing is None:
            raise RuntimeError("Configure parallelization routing before recording it")
        return routing

    def record_plan(
        self,
        optimizer_json: Mapping[str, Any],
        *,
        placements_path: str | Path | None = None,
    ) -> None:
        self._ensure_open()
        summary = optimizer_json.get("summary")
        if not isinstance(summary, Mapping):
            raise ValueError("optimizer_json must contain a summary object")
        optimizer_artifact = self.write_json_artifact(
            "optimizer", optimizer_json, role="optimizer"
        )
        self.report["planning"].update(
            {
                "summary": _json_copy(summary),
                "optimizer_artifact": optimizer_artifact,
            }
        )
        if placements_path is not None:
            placement_artifact = self.add_artifact(
                "placements", placements_path, role="placements"
            )
            self.report["planning"]["placements_artifact"] = placement_artifact
        self.mark_stage("planned", evidence_level="ilp_feasible")

    def record_optimizer_state(self, optimizer_path: str | Path) -> None:
        self._ensure_open()
        if not self.report["planning"].get("optimizer_artifact"):
            raise RuntimeError("Record the selected plan before optimizer state")
        artifact = self.add_artifact(
            "optimizer-state",
            optimizer_path,
            role="optimizer_state",
            filename="optimizer.ap",
            media_type="application/x-python-pickle",
        )
        self.report["planning"]["optimizer_state_artifact"] = artifact
        self._flush()

    def record_optimizer_replay(self, optimizer_path: str | Path) -> None:
        self._ensure_open()
        if not self.report["planning"].get("optimizer_artifact"):
            raise RuntimeError("Record the selected plan before optimizer replay")
        artifact = self.add_artifact(
            "optimizer-replay",
            optimizer_path,
            role="optimizer_replay",
            filename="optimizer-replay.pt.gz",
            media_type="application/gzip",
        )
        self.report["planning"]["optimizer_replay_artifact"] = artifact
        self._flush()

    def record_metrics(self, metrics: Mapping[str, Any]) -> None:
        self._ensure_open()
        graphs = metrics.get("graphs")
        if not isinstance(graphs, list):
            raise ValueError("metrics must contain a graphs list")
        phases = {graph.get("phase") for graph in graphs if isinstance(graph, Mapping)}
        missing = {"forward", "backward"} - phases
        if missing:
            raise ValueError(f"metrics are missing graph phases: {sorted(missing)}")
        metrics_artifact = self.write_json_artifact(
            "metrics", metrics, role="reordered_metrics"
        )
        self.report["evaluation"] = {
            "cost_model": _json_copy(metrics.get("cost_model", {})),
            "reordering": _json_copy(metrics.get("reordering", {})),
            "graphs": _json_copy(graphs),
            "metrics_artifact": metrics_artifact,
        }
        self.mark_stage("compiled", evidence_level="compiled")

    def record_validation(
        self,
        name: str,
        result: Mapping[str, Any],
        *,
        evidence_level: str,
    ) -> None:
        self._ensure_open()
        _validate_label(name, "validation name")
        if evidence_level not in _VALIDATION_EVIDENCE:
            raise ValueError(f"Unsupported validation evidence level: {evidence_level}")
        artifact_name = f"validation-{name}"
        artifact = self.write_json_artifact(
            artifact_name,
            result,
            role="validation",
            filename=f"{artifact_name}.json",
        )
        self.report["validation"][name] = {
            "evidence_level": evidence_level,
            "result": _json_copy(result),
            "artifact": artifact,
        }
        self.mark_stage(
            _STAGE_FOR_EVIDENCE[evidence_level], evidence_level=evidence_level
        )

    def add_finding(
        self,
        code: str,
        message: str,
        *,
        severity: str = "warning",
        context: Mapping[str, Any] | None = None,
        suggested_action: str | None = None,
    ) -> None:
        self._ensure_open()
        if not _FINDING_CODE_RE.fullmatch(code):
            raise ValueError(f"Invalid finding code: {code}")
        if severity not in {"info", "warning", "error"}:
            raise ValueError(f"Invalid finding severity: {severity}")
        finding = {
            "code": code,
            "severity": severity,
            "message": message,
            "context": _json_copy(context or {}),
        }
        if suggested_action is not None:
            finding["suggested_action"] = suggested_action
        self.report["findings"].append(finding)
        self._flush()

    def write_json_artifact(
        self,
        name: str,
        payload: Mapping[str, Any],
        *,
        role: str,
        filename: str | None = None,
    ) -> str:
        self._ensure_open()
        _validate_label(name, "artifact name")
        artifact_filename = filename or f"{name}.json"
        _validate_filename(artifact_filename)
        destination = self.artifacts_path / artifact_filename
        if destination.exists():
            raise FileExistsError(f"Artifact already exists: {destination}")
        _atomic_write_json(destination, payload)
        self._register_bundle_artifact(name, role, destination, "application/json")
        self._flush()
        return name

    def add_artifact(
        self,
        name: str,
        source: str | Path,
        *,
        role: str,
        filename: str | None = None,
        media_type: str = "application/octet-stream",
    ) -> str:
        self._ensure_open()
        _validate_label(name, "artifact name")
        source_path = Path(source)
        if not source_path.is_file():
            raise FileNotFoundError(f"Artifact source does not exist: {source_path}")
        artifact_filename = filename or source_path.name
        _validate_filename(artifact_filename)
        destination = self.artifacts_path / artifact_filename
        if destination.exists():
            raise FileExistsError(f"Artifact already exists: {destination}")
        shutil.copy2(source_path, destination)
        self._register_bundle_artifact(name, role, destination, media_type)
        self._flush()
        return name

    def add_artifact_reference(
        self,
        name: str,
        uri: str,
        *,
        role: str,
        sha256: str | None = None,
        size_bytes: int | None = None,
        media_type: str = "application/octet-stream",
    ) -> str:
        self._ensure_open()
        _validate_label(name, "artifact name")
        self._ensure_unique_artifact(name)
        if not uri:
            raise ValueError("Artifact URI must not be empty")
        artifact: dict[str, Any] = {
            "name": name,
            "role": role,
            "storage": "external",
            "uri": uri,
            "media_type": media_type,
        }
        if sha256 is not None:
            artifact["sha256"] = sha256
        if size_bytes is not None:
            artifact["size_bytes"] = size_bytes
        self.report["artifacts"].append(artifact)
        self._flush()
        return name

    def record_failure(
        self,
        error: BaseException,
        error_traceback: TracebackType | None = None,
    ) -> None:
        self._ensure_open()
        traceback_text = "".join(
            traceback.format_exception(type(error), error, error_traceback)
        )
        substage = self.report["planning"].get("current_substage")
        signature = _failure_signature(
            error,
            error_traceback,
            stage=self.report["stage"],
            substage=substage,
            source_root=self._source_root,
        )
        failure_payload = {
            "stage": self.report["stage"],
            "substage": substage,
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback_text,
            "signature": signature,
        }
        artifact = self.write_json_artifact("failure", failure_payload, role="failure")
        self.report["failure"] = {
            "stage": failure_payload["stage"],
            "substage": failure_payload["substage"],
            "type": failure_payload["type"],
            "message": failure_payload["message"],
            "signature": signature["id"],
            "artifact": artifact,
        }
        self._finalize("failed")

    def complete(self) -> None:
        self._ensure_open()
        self._finalize("completed")

    def _mark_evidence(self, level: str) -> None:
        if level not in _EVIDENCE_LEVELS:
            raise ValueError(f"Unknown evidence level: {level}")
        completed = self.report["evidence"]["completed"]
        if level != "none" and level not in completed:
            completed.append(level)
            completed.sort(key=_EVIDENCE_LEVELS.index)
        current = self.report["evidence"]["highest_level"]
        if _EVIDENCE_LEVELS.index(level) > _EVIDENCE_LEVELS.index(current):
            self.report["evidence"]["highest_level"] = level

    def _register_bundle_artifact(
        self, name: str, role: str, path: Path, media_type: str
    ) -> None:
        self._ensure_unique_artifact(name)
        self.report["artifacts"].append(
            {
                "name": name,
                "role": role,
                "storage": "bundle",
                "path": str(path.relative_to(self.path)),
                "media_type": media_type,
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )

    def _ensure_unique_artifact(self, name: str) -> None:
        if any(artifact["name"] == name for artifact in self.report["artifacts"]):
            raise ValueError(f"Artifact name already exists: {name}")

    def _finalize(self, status: str) -> None:
        history = self.report["planning"].get("substage_history", [])
        if history and "completed_at" not in history[-1]:
            history[-1]["completed_at"] = _utc_now()
            history[-1]["status"] = "failed" if status == "failed" else "completed"
        self.report["status"] = status
        self.report["completed_at"] = _utc_now()
        self._flush()
        self._sealed = True

    def _ensure_open(self) -> None:
        if self._sealed:
            raise RuntimeError("The run report has already been finalized")

    def _flush(self) -> None:
        _validate_report(self.report)
        report_path = self.path / "report.json"
        summary_path = self.path / "summary.md"
        _atomic_write_json(report_path, self.report)
        _atomic_write_text(summary_path, _render_summary(self.report))

        files = [
            _file_manifest_entry(self.path, report_path),
            _file_manifest_entry(self.path, summary_path),
        ]
        for artifact in self.report["artifacts"]:
            if artifact["storage"] == "bundle":
                files.append(
                    _file_manifest_entry(self.path, self.path / artifact["path"])
                )
        manifest = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "status": self.report["status"],
            "stage": self.report["stage"],
            "files": sorted(files, key=lambda entry: entry["path"]),
        }
        _atomic_write_json(self.path / "manifest.json", manifest)


def _load_profiles(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Cluster profile configuration not found: {path}"
        ) from error
    if not isinstance(document, dict):
        raise ValueError(f"Cluster profile configuration must be an object: {path}")
    if document.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported cluster profile schema in {path}: "
            f"{document.get('schema_version')!r}"
        )
    if not isinstance(document.get("profiles"), dict):
        raise ValueError(
            f"Cluster profile configuration has no profiles object: {path}"
        )
    return document


def _merge_profile_documents(documents: list[dict[str, Any]]) -> dict[str, Any]:
    profiles: dict[str, Any] = {}
    default_profile = None
    for document in documents:
        default_profile = document.get("default_profile", default_profile)
        for name, profile in document["profiles"].items():
            _validate_label(name, "cluster profile")
            if not isinstance(profile, dict):
                raise ValueError(f"Cluster profile {name!r} must be an object")
            profiles[name] = {**profiles.get(name, {}), **profile}
    if not default_profile or default_profile not in profiles:
        raise ValueError("Cluster profile configuration has an invalid default_profile")
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "default_profile": default_profile,
        "profiles": profiles,
    }


def _profile_matches_environment(
    profile: Mapping[str, Any], environ: Mapping[str, str]
) -> bool:
    match_env = profile.get("match_env")
    if not match_env:
        return False
    if not isinstance(match_env, Mapping):
        raise ValueError("Cluster profile match_env must be an object")
    return all(environ.get(str(key)) == str(value) for key, value in match_env.items())


def _validate_report(report: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "run_id",
        "status",
        "stage",
        "created_at",
        "completed_at",
        "cluster",
        "request",
        "provenance",
        "evidence",
        "planning",
        "evaluation",
        "validation",
        "findings",
        "failure",
        "parallelization",
        "artifacts",
    }
    missing = required - report.keys()
    if missing:
        raise ValueError(f"Run report is missing fields: {sorted(missing)}")
    if report["schema_version"] != REPORT_SCHEMA_VERSION:
        raise ValueError("Run report schema version does not match the writer")
    if report["status"] not in {"running", "completed", "failed"}:
        raise ValueError(f"Invalid report status: {report['status']}")
    if report["stage"] not in _STAGES:
        raise ValueError(f"Invalid report stage: {report['stage']}")
    evidence = report["evidence"]
    if evidence["highest_level"] not in _EVIDENCE_LEVELS:
        raise ValueError("Invalid highest evidence level")
    parallelization = report["parallelization"]
    if parallelization is not None:
        _validate_parallelization_record(parallelization)
    json.dumps(report)


def _render_summary(report: Mapping[str, Any]) -> str:
    cluster = report["cluster"]
    evidence = report["evidence"]
    lines = [
        f"# AutoParallel run {report['run_id']}",
        "",
        f"- Status: {report['status']}",
        f"- Stage: {report['stage']}",
        f"- Evidence: {evidence['highest_level']}",
        f"- Cluster profile: {cluster['profile']} ({cluster['selection_source']})",
    ]
    reproduction = report["provenance"].get("reproduction", {})
    entry_point = reproduction.get("entry_point")
    if entry_point:
        lines.append(f"- Entry point: `{entry_point}`")
    source = reproduction.get("source", {})
    revision = source.get("git_revision")
    if revision:
        state = "dirty" if source.get("working_tree_dirty") else "clean"
        repository = source.get("repository", "source")
        lines.append(f"- Source: {repository}@{revision[:12]} ({state})")
    measured = [
        (name, validation["result"])
        for name, validation in report["validation"].items()
        if validation.get("evidence_level") == "measured"
        and "timing" in validation["result"]
        and "memory" in validation["result"]
    ]
    if measured:
        lines.extend(["", "## Measurements", ""])
        for name, result in measured:
            timing = result.get("timing", {})
            memory = result.get("memory", {})
            lines.append(
                f"- {name}: mean {timing.get('mean_us')} us; peak allocated "
                f"{memory.get('peak_allocated_bytes')} bytes"
            )
    parallelization = report.get("parallelization")
    if parallelization is not None:
        lines.extend(["", "## Parallelization decision", ""])
        lines.append(f"- Decision ID: `{parallelization['decision_id']}`")
        autoparallel = parallelization.get("autoparallel")
        if autoparallel is not None:
            lines.append(f"- AutoParallel: {autoparallel['outcome']}")
        fallback = parallelization.get("fallback")
        if fallback is not None:
            lines.append(
                f"- Fallback: {fallback['strategy'].upper()} ({fallback['outcome']})"
            )
        competitiveness = parallelization.get("competitiveness")
        if competitiveness is not None:
            strategies = ", ".join(
                sorted(
                    {
                        baseline["strategy"].upper()
                        for baseline in competitiveness["baselines"]
                    }
                )
            )
            baseline_suffix = f" against {strategies}" if strategies else ""
            lines.append(
                "- Competitiveness: "
                f"{competitiveness['verdict']} ({competitiveness['evidence']})"
                f"{baseline_suffix}"
            )
        decision = parallelization.get("decision")
        if decision is not None:
            lines.append(
                f"- Selected: {decision['selected_strategy']} ({decision['status']})"
            )
    findings = report["findings"]
    if findings:
        lines.extend(["", "## Findings", ""])
        lines.extend(
            f"- [{finding['severity']}] {finding['code']}: {finding['message']}"
            for finding in findings
        )
    failure = report["failure"]
    if failure is not None:
        failure_location = failure.get("substage") or failure["stage"]
        lines.extend(
            [
                "",
                "## Failure",
                "",
                f"{failure['type']} during {failure_location}: {failure['message']}",
                f"Signature: `{failure['signature']}`",
            ]
        )
    artifacts = report["artifacts"]
    if artifacts:
        lines.extend(["", "## Artifacts", ""])
        for artifact in artifacts:
            location = artifact.get("path", artifact.get("uri"))
            lines.append(f"- {artifact['name']} ({artifact['role']}): `{location}`")
    return "\n".join(lines) + "\n"


def _validate_label(value: str, kind: str) -> None:
    if not isinstance(value, str) or not _LABEL_RE.fullmatch(value):
        raise ValueError(f"Invalid {kind}: {value!r}")


def _new_parallelization_record(
    value: Mapping[str, Any], *, default_decision_id: str
) -> dict[str, Any]:
    intent = value.get("intent")
    acceptance_criteria = value.get("acceptance_criteria")
    if not isinstance(intent, Mapping) or not intent:
        raise ValueError("parallelization.intent must be a non-empty object")
    if not isinstance(acceptance_criteria, Mapping) or not acceptance_criteria:
        raise ValueError(
            "parallelization.acceptance_criteria must be a non-empty object"
        )
    decision_id = value.get("decision_id", default_decision_id)
    _validate_label(decision_id, "parallelization decision id")
    return {
        "decision_id": decision_id,
        "intent": _json_copy(intent),
        "acceptance_criteria": _json_copy(acceptance_criteria),
        "autoparallel": None,
        "competitiveness": None,
        "fallback": None,
        "decision": None,
    }


def _validated_reason_codes(
    reason_codes: list[str] | tuple[str, ...],
) -> list[str]:
    if isinstance(reason_codes, str):
        raise ValueError("reason_codes must be a sequence of stable codes")
    result = []
    for code in reason_codes:
        if not isinstance(code, str) or not _FINDING_CODE_RE.fullmatch(code):
            raise ValueError(f"Invalid reason code: {code!r}")
        if code not in result:
            result.append(code)
    return result


def _validate_parallelization_record(value: Mapping[str, Any]) -> None:
    required = {
        "decision_id",
        "intent",
        "acceptance_criteria",
        "autoparallel",
        "fallback",
        "decision",
    }
    missing = required - value.keys()
    if missing:
        raise ValueError(f"Parallelization record is missing fields: {sorted(missing)}")
    _validate_label(value["decision_id"], "parallelization decision id")
    if not isinstance(value["intent"], Mapping) or not value["intent"]:
        raise ValueError("Parallelization intent must be a non-empty object")
    if (
        not isinstance(value["acceptance_criteria"], Mapping)
        or not value["acceptance_criteria"]
    ):
        raise ValueError("Parallelization acceptance criteria must be non-empty")
    competitiveness = value.get("competitiveness")
    if competitiveness is not None:
        _validate_competitiveness_assessment(competitiveness)


def _validate_parallelization_decision(
    routing: Mapping[str, Any], selected_strategy: str, status: str
) -> None:
    autoparallel = routing.get("autoparallel")
    fallback = routing.get("fallback")
    if selected_strategy == "autoparallel" and (
        autoparallel is None or autoparallel.get("outcome") != "accepted"
    ):
        raise ValueError("Selecting AutoParallel requires an accepted outcome")
    if selected_strategy == "autoparallel" and status != "accepted":
        raise ValueError(
            "An accepted AutoParallel outcome must be selected as accepted"
        )
    if selected_strategy == "autoparallel":
        relative_quality = routing.get("acceptance_criteria", {}).get(
            "relative_quality", {}
        )
        if relative_quality.get("required"):
            competitiveness = routing.get("competitiveness")
            if competitiveness is None:
                raise ValueError(
                    "Selecting AutoParallel requires a competitiveness assessment"
                )
            acceptable = set(relative_quality.get("acceptable_verdicts", []))
            if competitiveness.get("verdict") not in acceptable:
                raise ValueError(
                    "AutoParallel competitiveness does not meet the acceptance bar"
                )
    if selected_strategy in {"ddp", "fsdp"}:
        if fallback is None or fallback.get("strategy") != selected_strategy:
            raise ValueError("Selecting a fallback requires its recorded outcome")
        if status == "accepted" and fallback.get("outcome") != "accepted":
            raise ValueError("An accepted fallback decision requires accepted evidence")
    if selected_strategy == "none" and status in {"accepted", "fallback_selected"}:
        raise ValueError("A successful decision must select a strategy")


def _validate_competitiveness_assessment(value: Mapping[str, Any]) -> None:
    required = {
        "verdict",
        "evidence",
        "autoparallel_estimate",
        "baselines",
        "reason_codes",
        "details",
        "recorded_at",
    }
    missing = required - value.keys()
    if missing:
        raise ValueError(
            f"Competitiveness assessment is missing fields: {sorted(missing)}"
        )
    verdict = value["verdict"]
    if verdict not in _COMPETITIVENESS_VERDICTS:
        raise ValueError(f"Invalid competitiveness verdict: {verdict}")
    if value["evidence"] not in _COMPETITIVENESS_EVIDENCE:
        raise ValueError(f"Invalid competitiveness evidence: {value['evidence']}")
    _validated_reason_codes(value["reason_codes"])
    if not isinstance(value["details"], Mapping):
        raise ValueError("Competitiveness details must be an object")
    autoparallel_estimate = value["autoparallel_estimate"]
    if autoparallel_estimate is not None:
        _validate_strategy_estimate(autoparallel_estimate, baseline=False)
    baselines = value["baselines"]
    if not isinstance(baselines, list):
        raise ValueError("Competitiveness baselines must be a list")
    for baseline in baselines:
        _validate_strategy_estimate(baseline, baseline=True)

    comparable_verdicts = {
        "robustly_competitive",
        "robustly_worse",
        "pareto_dominated",
        "schedule_sensitive",
    }
    if verdict in comparable_verdicts:
        if (
            value["evidence"] == "none"
            or autoparallel_estimate is None
            or autoparallel_estimate.get("latency") is None
        ):
            raise ValueError(f"{verdict} requires modeled or measured evidence")
        if not any(
            baseline["feasibility"] == "feasible"
            and baseline.get("latency") is not None
            for baseline in baselines
        ):
            raise ValueError(f"{verdict} requires a feasible baseline latency")
    if verdict == "baseline_infeasible" and (
        not baselines
        or any(baseline["feasibility"] != "infeasible" for baseline in baselines)
    ):
        raise ValueError(
            "baseline_infeasible requires every recorded baseline to be infeasible"
        )
    if verdict == "baseline_infeasible" and (
        value["evidence"] == "none"
        or any(baseline["memory"].get("fits") is not False for baseline in baselines)
    ):
        raise ValueError(
            "baseline_infeasible requires memory evidence that every baseline misses"
        )
    if (
        verdict == "user_accepted_tradeoff"
        and value["details"].get("user_authorized") is not True
    ):
        raise ValueError(
            "user_accepted_tradeoff requires explicit user authorization evidence"
        )


def _validate_strategy_estimate(value: Mapping[str, Any], *, baseline: bool) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("Strategy estimate must be an object")
    required = {"latency", "memory"}
    if baseline:
        required |= {
            "strategy",
            "template_id",
            "template_version",
            "configuration_hash",
            "feasibility",
        }
    missing = required - value.keys()
    if missing:
        raise ValueError(f"Strategy estimate is missing fields: {sorted(missing)}")
    if baseline:
        if value["strategy"] not in {"ddp", "fsdp"}:
            raise ValueError(f"Invalid baseline strategy: {value['strategy']}")
        _validate_label(value["template_id"], "baseline template id")
        _validate_label(value["template_version"], "baseline template version")
        configuration_hash = value["configuration_hash"]
        if not isinstance(configuration_hash, str) or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", configuration_hash
        ):
            raise ValueError("Invalid baseline configuration hash")
        if value["feasibility"] not in _BASELINE_FEASIBILITY:
            raise ValueError(f"Invalid baseline feasibility: {value['feasibility']}")
    latency = value["latency"]
    if latency is not None:
        _validate_latency_envelope(latency)
    if not isinstance(value["memory"], Mapping):
        raise ValueError("Strategy memory estimate must be an object")
    if baseline and value["feasibility"] == "feasible":
        if value["memory"].get("fits") is not True:
            raise ValueError("A feasible baseline requires memory fits=true")


def _validate_latency_envelope(value: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("Latency envelope must be an object")
    required = {
        "unit",
        "optimistic",
        "expected",
        "pessimistic",
        "calibration_relative_error",
        "schedule_model",
    }
    missing = required - value.keys()
    if missing:
        raise ValueError(f"Latency envelope is missing fields: {sorted(missing)}")
    estimates = [value[key] for key in ("optimistic", "expected", "pessimistic")]
    if any(
        isinstance(estimate, bool)
        or not isinstance(estimate, (int, float))
        or not math.isfinite(estimate)
        or estimate < 0
        for estimate in estimates
    ):
        raise ValueError("Latency estimates must be finite non-negative numbers")
    if not estimates[0] <= estimates[1] <= estimates[2]:
        raise ValueError(
            "Latency envelope must satisfy optimistic <= expected <= pessimistic"
        )
    calibration_error = value["calibration_relative_error"]
    if (
        isinstance(calibration_error, bool)
        or not isinstance(calibration_error, (int, float))
        or not math.isfinite(calibration_error)
        or calibration_error < 0
    ):
        raise ValueError("Calibration relative error must be finite and non-negative")
    if not isinstance(value["unit"], str) or not value["unit"]:
        raise ValueError("Latency envelope unit must be non-empty")
    _validate_label(value["schedule_model"], "schedule model")


def _validate_filename(filename: str) -> None:
    path = Path(filename)
    if not filename or path.name != filename or filename in {".", ".."}:
        raise ValueError(f"Invalid artifact filename: {filename!r}")


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _make_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:12]}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _failure_signature(
    error: BaseException,
    error_traceback: TracebackType | None,
    *,
    stage: str,
    substage: str | None,
    source_root: Path,
) -> dict[str, Any]:
    frames = []
    for frame in traceback.extract_tb(error_traceback)[-8:]:
        path = Path(frame.filename).resolve()
        try:
            filename = str(path.relative_to(source_root))
        except ValueError:
            filename = "/".join(path.parts[-4:])
        frames.append({"file": filename, "function": frame.name})
    basis = {
        "version": 2,
        "stage": stage,
        "substage": substage,
        "exception_type": type(error).__name__,
        "message_template": _normalize_failure_message(str(error)),
        "frames": frames,
    }
    return {
        "id": f"sha256:{_canonical_sha256(basis)}",
        "basis": basis,
    }


def _normalize_failure_message(message: str) -> str:
    first_line = message.strip().splitlines()[0] if message.strip() else ""
    normalized = re.sub(r"0x[0-9A-Fa-f]+", "<hex>", first_line)
    normalized = re.sub(r"\b[0-9]+(?:\.[0-9]+)?\b", "<number>", normalized)
    normalized = re.sub(r"(?:[A-Za-z]:)?/[^\s:]+", "<path>", normalized)
    return normalized[:512]


def _file_manifest_entry(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content)
    os.replace(temporary, path)
