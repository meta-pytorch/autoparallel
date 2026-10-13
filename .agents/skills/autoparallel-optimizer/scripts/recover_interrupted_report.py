from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_report import (
    REPORT_SCHEMA_VERSION,
    _atomic_write_json,
    _atomic_write_text,
    _failure_signature,
    _file_manifest_entry,
    _render_summary,
    _sha256,
    _utc_now,
    _validate_report,
)


class InterruptedRun(RuntimeError):
    pass


def _remove_stale_atomic_temps(path: Path) -> None:
    for temp_path in path.glob(".*.tmp"):
        if temp_path.is_file():
            temp_path.unlink()


def recover_interrupted_report(
    bundle: str | Path,
    *,
    reason: str,
    source_root: str | Path,
) -> Path:
    path = Path(bundle).resolve()
    report_path = path / "report.json"
    manifest_path = path / "manifest.json"
    report = json.loads(report_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported report schema")
    if report.get("status") in {"completed", "failed"}:
        if manifest.get("status") not in {"running", report["status"]}:
            raise ValueError("terminal report and manifest statuses disagree")
        _validate_report(report)
        summary_path = path / "summary.md"
        _atomic_write_text(summary_path, _render_summary(report))
        files = [
            _file_manifest_entry(path, report_path),
            _file_manifest_entry(path, summary_path),
        ]
        files.extend(
            _file_manifest_entry(path, path / artifact["path"])
            for artifact in report["artifacts"]
            if artifact["storage"] == "bundle"
        )
        _atomic_write_json(
            manifest_path,
            {
                "schema_version": REPORT_SCHEMA_VERSION,
                "run_id": report["run_id"],
                "status": report["status"],
                "stage": report["stage"],
                "files": sorted(files, key=lambda entry: entry["path"]),
            },
        )
        _remove_stale_atomic_temps(path)
        return path
    if manifest.get("status") != "running":
        raise ValueError("report manifest is not running")
    if report.get("status") != "running":
        raise ValueError("report bundle has an unsupported status")
    if any(artifact.get("name") == "failure" for artifact in report["artifacts"]):
        raise ValueError("report bundle already contains a failure artifact")

    error = InterruptedRun(reason)
    substage = report["planning"].get("current_substage")
    signature = _failure_signature(
        error,
        None,
        stage=report["stage"],
        substage=substage,
        source_root=Path(source_root).resolve(),
    )
    failure_payload = {
        "stage": report["stage"],
        "substage": substage,
        "type": type(error).__name__,
        "message": str(error),
        "traceback": "",
        "signature": signature,
    }
    artifact_path = path / "artifacts/failure.json"
    _atomic_write_json(artifact_path, failure_payload)
    report["artifacts"].append(
        {
            "name": "failure",
            "role": "failure",
            "storage": "bundle",
            "path": "artifacts/failure.json",
            "media_type": "application/json",
            "sha256": _sha256(artifact_path),
            "size_bytes": artifact_path.stat().st_size,
        }
    )
    report["failure"] = {
        "stage": report["stage"],
        "substage": substage,
        "type": type(error).__name__,
        "message": str(error),
        "signature": signature["id"],
        "artifact": "failure",
    }
    history = report["planning"].get("substage_history", [])
    completed_at = _utc_now()
    if history and "completed_at" not in history[-1]:
        history[-1]["completed_at"] = completed_at
        history[-1]["status"] = "failed"
    report["status"] = "failed"
    report["completed_at"] = completed_at
    _validate_report(report)
    _atomic_write_json(report_path, report)
    summary_path = path / "summary.md"
    _atomic_write_text(summary_path, _render_summary(report))
    files = [
        _file_manifest_entry(path, report_path),
        _file_manifest_entry(path, summary_path),
    ]
    files.extend(
        _file_manifest_entry(path, path / artifact["path"])
        for artifact in report["artifacts"]
        if artifact["storage"] == "bundle"
    )
    _atomic_write_json(
        manifest_path,
        {
            "schema_version": REPORT_SCHEMA_VERSION,
            "run_id": report["run_id"],
            "status": "failed",
            "stage": report["stage"],
            "files": sorted(files, key=lambda entry: entry["path"]),
        },
    )
    _remove_stale_atomic_temps(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finalize one explicitly selected interrupted report bundle"
    )
    parser.add_argument("bundle")
    parser.add_argument("--reason", required=True)
    parser.add_argument("--source-root", required=True)
    args = parser.parse_args()
    print(
        recover_interrupted_report(
            args.bundle,
            reason=args.reason,
            source_root=args.source_root,
        )
    )


if __name__ == "__main__":
    main()
