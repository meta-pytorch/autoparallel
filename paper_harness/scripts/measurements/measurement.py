#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HARNESS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HARNESS_ROOT))

from harness.dryrun import audit_submitted_definition  # noqa: E402
from harness.sources import manifest_digest, tree_manifest  # noqa: E402

ACTIVE_STATES = {"PENDING", "RUNNING"}
TERMINAL_STATES = {"COMPLETE", "DEAD"}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _absolute(path: Path, name: str) -> Path:
    if not path.is_absolute():
        raise SystemExit(f"{name} must be an absolute path: {path}")
    return path


def _capture(
    command: list[str],
    *,
    cwd: Path,
    record_root: Path,
    name: str,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    record_root.mkdir(parents=True, exist_ok=True)
    _write_json(record_root / f"{name}.command.json", command)
    (record_root / f"{name}.stdout").write_text(completed.stdout)
    (record_root / f"{name}.stderr").write_text(completed.stderr)
    (record_root / f"{name}.returncode").write_text(f"{completed.returncode}\n")
    if check and completed.returncode:
        raise SystemExit(f"{' '.join(command)} exited {completed.returncode}")
    return completed


def _campaign_command(args: argparse.Namespace, action: str) -> list[str]:
    command = [
        str(args.python),
        "-m",
        "harness.cli",
        action,
        str(args.campaign),
    ]
    if args.point:
        command.extend(("--point", args.point))
    command.extend(
        (
            "--mode",
            args.mode,
            "--torchtitan-root",
            str(args.torchtitan_root),
            "--autoparallel-root",
            str(args.autoparallel_root),
            "--python",
            str(args.python),
        )
    )
    if args.baseline_autoparallel_root is not None:
        command.extend(
            (
                "--baseline-autoparallel-root",
                str(args.baseline_autoparallel_root),
            )
        )
    for asset in args.asset_root:
        command.extend(("--asset-root", asset))
    return command


def prepare(args: argparse.Namespace) -> None:
    attempt = _absolute(args.attempt, "--attempt")
    if attempt.exists():
        raise SystemExit(f"refusing to reuse attempt path: {attempt}")
    attempt.parent.mkdir(parents=True, exist_ok=True)
    for path, name in (
        (args.campaign, "--campaign"),
        (args.python, "--python"),
        (args.torchtitan_root, "--torchtitan-root"),
        (args.autoparallel_root, "--autoparallel-root"),
    ):
        _absolute(path, name)
    if args.baseline_autoparallel_root is not None:
        _absolute(
            args.baseline_autoparallel_root,
            "--baseline-autoparallel-root",
        )

    command = _campaign_command(args, "package") + ["--attempt", str(attempt)]
    completed = subprocess.run(
        command,
        cwd=HARNESS_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    record_root = attempt / "measurement"
    _write_json(record_root / "package.command.json", command)
    (record_root / "package.stdout").write_text(completed.stdout)
    (record_root / "package.stderr").write_text(completed.stderr)
    (record_root / "package.returncode").write_text(f"{completed.returncode}\n")
    if completed.returncode:
        raise SystemExit(f"{' '.join(command)} exited {completed.returncode}")

    report = json.loads((attempt / "package_report.json").read_text())
    if report.get("status") != "passed" or not report["validation"].get(
        "application_contract_validated"
    ):
        raise SystemExit(
            "package did not preserve a passing application-contract validation"
        )

    _capture(
        [
            str(args.python),
            "-m",
            "harness.cli",
            "render-mast",
            "--attempt",
            str(attempt),
        ],
        cwd=HARNESS_ROOT,
        record_root=record_root,
        name="render_mast",
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    dryrun = json.loads((attempt / "dryrun_audit.json").read_text())
    if dryrun.get("status") != "passed":
        raise SystemExit("render-mast did not produce a passing dry-run audit")
    print(
        json.dumps(
            {
                "status": "prepared",
                "attempt": str(attempt),
                "payload_tree_sha256": report["payload_tree_sha256"],
                "dryrun_packages": dryrun["packages"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def _walk_values(value: Any, key: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, dict):
        for current_key, current_value in value.items():
            if current_key == key:
                found.append(current_value)
            found.extend(_walk_values(current_value, key))
    elif isinstance(value, list):
        for item in value:
            found.extend(_walk_values(item, key))
    return found


def _package_evidence(
    root: Path, manifest: dict[str, str] | None = None
) -> dict[str, Any]:
    manifest = manifest if manifest is not None else tree_manifest(root)
    files = {}
    for relative, digest in manifest.items():
        path = root / relative
        mode = stat.S_IMODE(path.stat().st_mode)
        files[relative] = {
            "sha256": digest,
            "mode": f"{mode:04o}",
            "executable": bool(mode & 0o111),
        }
    metadata = sorted(
        relative
        for relative in manifest
        if Path(relative).name == "METADATA" or relative.endswith(".CHECKSUMS")
    )
    return {
        "root": str(root),
        "tree_sha256": manifest_digest(manifest),
        "file_count": len(manifest),
        "metadata_files": metadata,
        "files": files,
    }


def _one_package(packages: list[str], prefix: str) -> str:
    matches = sorted({package for package in packages if package.startswith(prefix)})
    if len(matches) != 1:
        raise SystemExit(f"expected one {prefix} package, got {matches}")
    return matches[0]


def _ensure_submitted(attempt: Path, python: Path) -> tuple[str, bool]:
    job_id_path = attempt / "job_id.txt"
    if job_id_path.exists():
        job_id = job_id_path.read_text().strip()
        if not job_id:
            raise SystemExit(f"submitted job ID is empty: {job_id_path}")
        return job_id, False
    record_root = attempt / "measurement"
    _write_json(
        record_root / "submission_started.json",
        {"timestamp_utc": datetime.now(timezone.utc).isoformat()},
    )
    _capture(
        [str(python), "-m", "harness.cli", "submit", "--attempt", str(attempt)],
        cwd=HARNESS_ROOT,
        record_root=record_root,
        name="submit",
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    job_id = job_id_path.read_text().strip()
    if not job_id:
        raise SystemExit(f"submitted job ID is empty: {job_id_path}")
    return job_id, True


def submit(args: argparse.Namespace) -> None:
    attempt = _absolute(args.attempt, "--attempt")
    python = Path(
        json.loads((attempt / "package_report.json").read_text())["validation"][
            "python"
        ]
    )
    job_id, newly_submitted = _ensure_submitted(attempt, python)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    record_root = attempt / "submitted_package_audits" / stamp
    _write_json(
        record_root / "audit_started.json",
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "job_id": job_id,
            "newly_submitted": newly_submitted,
        },
    )

    _capture(
        [
            "mast",
            "update-job-priority",
            job_id,
            "--priority",
            "critical",
            "--sub-priority",
            "99",
        ],
        cwd=HARNESS_ROOT,
        record_root=record_root,
        name="priority_update",
    )
    priority = _capture(
        ["mast", "--output", "json", "get-job-priority", job_id],
        cwd=HARNESS_ROOT,
        record_root=record_root,
        name="priority_readback",
    )
    priority_data = json.loads(priority.stdout)["data"]
    if (priority_data.get("priority"), priority_data.get("sub_priority")) != (
        "CRITICAL",
        99,
    ):
        raise SystemExit(f"priority readback is not CRITICAL/99: {priority_data}")

    definition_result = _capture(
        ["mast", "--output", "json", "get-job-definition", job_id],
        cwd=HARNESS_ROOT,
        record_root=record_root,
        name="submitted_definition",
    )
    definition = json.loads(definition_result.stdout)
    _write_json(attempt / "submitted_definition.json", definition)
    definition_audit = audit_submitted_definition(
        attempt,
        definition,
        launcher_root=HARNESS_ROOT / "launcher",
    )
    _write_json(record_root / "definition_audit.json", definition_audit)
    packages = [str(value) for value in _walk_values(definition, "fbpkgIdentifier")]
    package_ids = {
        "workspace": _one_package(packages, "torchtitan_workspace:"),
        "payload": _one_package(packages, "torchtitan_additional_packages:"),
    }

    package_root = record_root / "packages"
    destinations = {
        "workspace": package_root / "submitted_workspace",
        "payload": package_root / "submitted_payload",
    }
    for name, package in package_ids.items():
        destination = destinations[name]
        if destination.exists():
            raise SystemExit(f"refusing to reuse package destination: {destination}")
        destination.mkdir(parents=True)
        _capture(
            ["fbpkg", "fetch", package, "--dest", str(destination)],
            cwd=HARNESS_ROOT,
            record_root=record_root,
            name=f"fetch_submitted_{name}",
        )

    fetched_payload = destinations["payload"] / "payload"
    preflight_output = record_root / "exact_submitted_package_preflight"
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join(
            str(fetched_payload / component)
            for component in ("harness_repo", "autoparallel", "torchtitan")
        ),
    }
    preflight = _capture(
        [
            str(python),
            "-m",
            "harness.package_preflight",
            str(fetched_payload),
            str(preflight_output),
        ],
        cwd=fetched_payload / "torchtitan",
        record_root=record_root,
        name="exact_submitted_package_preflight",
        env=env,
        check=False,
    )
    dryrun_packages = attempt / "packages"
    stored_evidence = {
        "source_payload": dryrun_packages / "source_payload.manifest.json",
        "dryrun_workspace": dryrun_packages / "fetched_workspace.manifest.json",
        "dryrun_payload_package": dryrun_packages
        / "fetched_payload_package.manifest.json",
    }
    evidence = {
        name: json.loads(path.read_text()) for name, path in stored_evidence.items()
    }
    submitted_roots = {
        "submitted_workspace": destinations["workspace"],
        "submitted_payload_package": destinations["payload"],
    }
    for name, root in submitted_roots.items():
        evidence[name] = _package_evidence(root)
        _write_json(record_root / f"{name}.manifest.json", evidence[name])

    def content_manifest(name: str) -> dict[str, str]:
        return {
            path: row["sha256"]
            for path, row in evidence[name]["files"].items()
            if path != "METADATA" and not path.endswith(".CHECKSUMS")
        }

    def payload_content(name: str) -> dict[str, str]:
        prefix = "payload/"
        return {
            path.removeprefix(prefix): digest
            for path, digest in content_manifest(name).items()
            if path.startswith(prefix)
        }

    expected_payload = payload_content("dryrun_payload_package")
    actual_payload = payload_content("submitted_payload_package")
    expected_workspace = content_manifest("dryrun_workspace")
    actual_workspace = content_manifest("submitted_workspace")

    def has_package_metadata(name: str) -> bool:
        files = evidence[name]["metadata_files"]
        return "METADATA" in files

    submitted_metadata = {
        name: has_package_metadata(name)
        for name in ("submitted_workspace", "submitted_payload_package")
    }
    metadata_identity = {}
    for name, package_name, package_root_path in (
        ("workspace", package_ids["workspace"], destinations["workspace"]),
        ("payload", package_ids["payload"], destinations["payload"]),
    ):
        expected_name, expected_version = package_name.split(":", 1)
        try:
            metadata = json.loads((package_root_path / "METADATA").read_text())
        except (OSError, json.JSONDecodeError):
            metadata = {}
        metadata_identity[name] = {
            "expected_package": expected_name,
            "expected_version": expected_version,
            "actual_package": metadata.get("package"),
            "actual_version": metadata.get("version"),
            "matched": metadata.get("package") == expected_name
            and metadata.get("version") == expected_version,
        }
    permission_matches = {
        "workspace": {
            path: row["mode"]
            for path, row in evidence["dryrun_workspace"]["files"].items()
            if path not in {"METADATA"} and not path.endswith(".CHECKSUMS")
        }
        == {
            path: row["mode"]
            for path, row in evidence["submitted_workspace"]["files"].items()
            if path not in {"METADATA"} and not path.endswith(".CHECKSUMS")
        },
        "payload": {
            path: row["mode"]
            for path, row in evidence["dryrun_payload_package"]["files"].items()
            if path.startswith("payload/")
        }
        == {
            path: row["mode"]
            for path, row in evidence["submitted_payload_package"]["files"].items()
            if path.startswith("payload/")
        },
    }
    executable_checks = {
        "workspace_mount": os.access(destinations["workspace"] / "mount.sh", os.X_OK),
        "payload_runner": os.access(
            fetched_payload / "harness_repo/launcher/run_rank.sh", os.X_OK
        ),
    }
    audit = {
        "status": "passed",
        "audit_root": str(record_root),
        "job_id": job_id,
        "priority": priority_data,
        "packages": package_ids,
        "reference_manifests": {
            name: str(path) for name, path in stored_evidence.items()
        },
        "definition_audit": definition_audit,
        "payload": {
            "matched": expected_payload == actual_payload,
            "source_matched": content_manifest("source_payload") == actual_payload,
            "expected_tree_sha256": manifest_digest(expected_payload),
            "actual_tree_sha256": manifest_digest(actual_payload),
        },
        "workspace": {
            "matched": expected_workspace == actual_workspace,
            "expected_tree_sha256": manifest_digest(expected_workspace),
            "actual_tree_sha256": manifest_digest(actual_workspace),
        },
        "exact_package_preflight_returncode": preflight.returncode,
        "exact_package_preflight_output": str(preflight_output),
        "metadata_present": submitted_metadata,
        "metadata_identity": metadata_identity,
        "checksum_manifests": {
            name: [
                path
                for path in evidence[name]["metadata_files"]
                if path.endswith(".CHECKSUMS")
            ]
            for name in ("submitted_workspace", "submitted_payload_package")
        },
        "permission_matches": permission_matches,
        "executable_checks": executable_checks,
    }
    if not (
        audit["payload"]["matched"]
        and audit["payload"]["source_matched"]
        and audit["workspace"]["matched"]
        and preflight.returncode == 0
        and all(submitted_metadata.values())
        and all(value["matched"] for value in metadata_identity.values())
        and all(permission_matches.values())
        and all(executable_checks.values())
    ):
        audit["status"] = "failed"
    _write_json(attempt / "submitted_package_audit.json", audit)
    if audit["status"] != "passed":
        raise SystemExit(
            "actual submitted package content, metadata, or permissions failed audit"
        )
    print(json.dumps(audit, indent=2, sort_keys=True))


def _task_groups(latest: dict[str, Any]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for attempts in latest.get("taskGroupExecutionAttempts", {}).values():
        groups.extend(attempts)
    return groups


def _arm_markers(attempt: Path, run_root: Path) -> dict[str, Any]:
    resolved = json.loads((attempt / "validation/resolved_campaign.json").read_text())
    if resolved.get("campaign_type") == "planner":
        world_size = int(resolved["world_size"])
        runs = []
        for run in resolved["resolved_planner_runs"]:
            root = (
                run_root
                / "planner"
                / run["point"]
                / f"repeat_{int(run['repeat']):02d}"
                / run["solver"]
            )
            runs.append(
                {
                    "point": run["point"],
                    "repeat": run["repeat"],
                    "solver": run["solver"],
                    "started": (root / "started").is_file(),
                    "completed": (root / "completed").is_file(),
                    "failed": (root / "failed").is_file(),
                    "result": (root / "result.json").is_file(),
                }
            )
        exit_paths = sorted((run_root / "planner/runtime").glob("rank_*.exit_code"))
        exit_codes = []
        for path in exit_paths:
            try:
                exit_codes.append(int(path.read_text().strip()))
            except ValueError:
                exit_codes.append(-1)
        return {
            "run_root": str(run_root),
            "campaign_type": "planner",
            "campaign_completed": (run_root / "planner/completed").is_file(),
            "all_ranks_completed": (run_root / "planner/all_ranks_completed").is_file(),
            "allocation_records": len(
                list(
                    (run_root / "allocation/planner/planner_profile").glob(
                        "rank_*.json"
                    )
                )
            ),
            "expected_ranks": world_size,
            "rank_exit_codes": exit_codes,
            "runs": runs,
        }
    world_size = int(resolved["mast"]["nodes"]) * int(
        resolved["mast"]["nproc_per_node"]
    )
    arms = []
    for phase in resolved["resolved_phases"]:
        for arm in phase["arms"]:
            root = run_root / phase["name"] / arm
            arms.append(
                {
                    "phase": phase["name"],
                    "arm": arm,
                    "started": (root / "started").is_file(),
                    "completed": (root / "completed").is_file(),
                    "failed": (root / "failed").is_file(),
                    "allocation_records": len(
                        list(
                            (run_root / "allocation" / phase["name"] / arm).glob(
                                "rank_*.json"
                            )
                        )
                    ),
                    "expected_ranks": world_size,
                }
            )
    return {"run_root": str(run_root), "arms": arms}


def _scheduler_summary(
    status: dict[str, Any], *, expected_nodes: int | None = None
) -> dict[str, Any]:
    if not isinstance(status, dict) or not isinstance(status.get("data"), dict):
        raise SystemExit("MAST status response has no data object")
    data = status["data"]
    latest = data.get("latestAttempt") or {}
    if not isinstance(latest, dict):
        raise SystemExit("MAST status latestAttempt is not an object")
    groups = _task_groups(latest)
    tasks = [
        task
        for group in groups
        for attempts in group.get("taskExecutionAttempts", {}).values()
        for task in attempts
    ]
    continuity_errors = []
    if status.get("status") != "ok":
        continuity_errors.append("MAST status envelope is not ok")
    for owner, value, required in (
        ("job", data, ("state", "numRestarts", "latestAttempt")),
        ("latest attempt", latest, ("state", "attemptIndex")),
    ):
        missing = [key for key in required if key not in value]
        if missing:
            continuity_errors.append(f"{owner} is missing fields {missing}")
    if latest.get("attemptIndex") != 0:
        continuity_errors.append("scheduler attempt index is not zero")
    if not isinstance(data.get("numRestarts"), int) or data.get("numRestarts") != 0:
        continuity_errors.append("scheduler restart count is not zero")
    for group in groups:
        required = (
            "attemptIndex",
            "attemptEpoch",
            "numTasks",
            "numFailedTasks",
            "numShrunkTasks",
            "onElasticCapacity",
            "state",
        )
        missing = [key for key in required if key not in group]
        if missing:
            continuity_errors.append(f"task group is missing fields {missing}")
        if group.get("attemptIndex") != 0 or group.get("attemptEpoch") != 0:
            continuity_errors.append("task-group attempt/epoch is not zero")
        if group.get("numFailedTasks") != 0:
            continuity_errors.append("task group reports failed tasks")
        if group.get("numShrunkTasks") != 0:
            continuity_errors.append("task group reports shrunk tasks")
        if group.get("onElasticCapacity") is not False:
            continuity_errors.append("task group is on elastic capacity")
    if any("attemptIndex" not in task or "state" not in task for task in tasks):
        continuity_errors.append("task execution is missing required fields")
    if any(task.get("attemptIndex") != 0 for task in tasks):
        continuity_errors.append("task execution attempt index is not zero")
    if data.get("state") == "COMPLETE":
        if expected_nodes is None:
            continuity_errors.append("expected node count was not supplied")
        else:
            if len(groups) != 1:
                continuity_errors.append("terminal job does not have one task group")
            if len(tasks) != expected_nodes:
                continuity_errors.append(
                    f"terminal task count {len(tasks)} != expected nodes {expected_nodes}"
                )
            hosts = [task.get("hostname") for task in tasks]
            if any(not isinstance(host, str) or not host for host in hosts):
                continuity_errors.append("terminal task hostname is missing")
            elif len(set(hosts)) != expected_nodes:
                continuity_errors.append(
                    f"terminal host count {len(set(hosts))} != expected {expected_nodes}"
                )
            if any(task.get("exitCode") != 0 for task in tasks):
                continuity_errors.append(
                    "terminal task exit code is missing or nonzero"
                )
            if any(group.get("numTasks") != expected_nodes for group in groups):
                continuity_errors.append(
                    "terminal task-group size differs from request"
                )
    return {
        "root_state": data.get("state"),
        "latest_attempt_state": latest.get("state"),
        "attempt_index": latest.get("attemptIndex"),
        "num_restarts": int(data.get("numRestarts") or 0),
        "task_groups": [
            {
                "state": group.get("state"),
                "attempt_index": group.get("attemptIndex"),
                "attempt_epoch": group.get("attemptEpoch"),
                "num_tasks": group.get("numTasks"),
                "num_failed_tasks": group.get("numFailedTasks"),
                "num_shrunk_tasks": group.get("numShrunkTasks"),
                "on_elastic_capacity": group.get("onElasticCapacity"),
                "server_subtype": group.get("serverSubType"),
                "host_pool": group.get("hostPoolName"),
            }
            for group in groups
        ],
        "tasks": [
            {
                "state": task.get("state"),
                "attempt_index": task.get("attemptIndex"),
                "hostname": task.get("hostname"),
                "exit_code": task.get("exitCode"),
            }
            for task in tasks
        ],
        "continuity_errors": sorted(set(continuity_errors)),
    }


def _terminal_evidence(
    attempt: Path, job_id: str, stamp: str, *, expected_nodes: int
) -> None:
    root = attempt / "terminal_evidence" / stamp
    errors = []
    parsed = {}
    for name, command in (
        ("status", ["mast", "--output", "json", "get-status", job_id]),
        ("history", ["mast", "--output", "json", "get-job-history", job_id]),
        ("definition", ["mast", "--output", "json", "get-job-definition", job_id]),
        ("priority", ["mast", "--output", "json", "get-job-priority", job_id]),
        (
            "stdout",
            ["mast", "--output", "json", "get-logs", "--file-path", "stdout", job_id],
        ),
        (
            "stderr",
            ["mast", "--output", "json", "get-logs", "--file-path", "stderr", job_id],
        ),
    ):
        completed = _capture(
            command, cwd=HARNESS_ROOT, record_root=root, name=name, check=False
        )
        if completed.returncode:
            errors.append(f"{name} exited {completed.returncode}")
            continue
        try:
            parsed[name] = json.loads(completed.stdout)
        except json.JSONDecodeError:
            errors.append(f"{name} did not return JSON")
    if "status" in parsed:
        summary = _scheduler_summary(parsed["status"], expected_nodes=expected_nodes)
        if not (
            summary["root_state"] == "COMPLETE"
            and summary["latest_attempt_state"] == "COMPLETE"
            and not summary["continuity_errors"]
            and all(group["state"] == "COMPLETE" for group in summary["task_groups"])
            and all(task["state"] == "COMPLETE" for task in summary["tasks"])
        ):
            errors.append("terminal status does not satisfy completion gates")
    if "priority" in parsed:
        priority = parsed["priority"].get("data", {})
        if (priority.get("priority"), priority.get("sub_priority")) != (
            "CRITICAL",
            99,
        ):
            errors.append("terminal priority is not CRITICAL/99")
    if "definition" in parsed:
        try:
            audit_submitted_definition(
                attempt,
                parsed["definition"],
                launcher_root=HARNESS_ROOT / "launcher",
            )
        except Exception as error:
            errors.append(f"terminal definition audit failed: {error}")
    _write_json(
        root / "evidence_audit.json",
        {"status": "passed" if not errors else "failed", "errors": errors},
    )
    if errors:
        raise SystemExit(f"terminal evidence collection failed: {errors}")


def monitor(args: argparse.Namespace) -> None:
    attempt = _absolute(args.attempt, "--attempt")
    run_root = _absolute(args.run_root, "--run-root") if args.run_root else None
    job_id = (attempt / "job_id.txt").read_text().strip()
    resolved = json.loads((attempt / "validation/resolved_campaign.json").read_text())
    expected_nodes = int(resolved["mast"]["nodes"])
    while True:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        poll = attempt / "polls" / stamp
        status_result = _capture(
            ["mast", "--output", "json", "get-status", job_id],
            cwd=HARNESS_ROOT,
            record_root=poll,
            name="status",
            check=False,
        )
        _capture(
            ["mast", "--output", "json", "get-job-history", job_id],
            cwd=HARNESS_ROOT,
            record_root=poll,
            name="history",
            check=False,
        )
        if status_result.returncode:
            if args.once:
                raise SystemExit("MAST status query failed")
            time.sleep(args.interval_seconds)
            continue
        status = json.loads(status_result.stdout)
        summary = _scheduler_summary(status, expected_nodes=expected_nodes)
        if run_root is not None:
            summary["arm_markers"] = _arm_markers(attempt, run_root)
        _write_json(poll / "summary.json", summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
        if summary["continuity_errors"]:
            (attempt / "FAIL_CLOSED_ALERT").write_text(
                json.dumps(summary["continuity_errors"], indent=2) + "\n"
            )

        state = summary["root_state"]
        if state in TERMINAL_STATES:
            _terminal_evidence(attempt, job_id, stamp, expected_nodes=expected_nodes)
            complete = (
                state == "COMPLETE"
                and summary["latest_attempt_state"] == "COMPLETE"
                and bool(summary["task_groups"])
                and bool(summary["tasks"])
                and all(
                    group["state"] == "COMPLETE" for group in summary["task_groups"]
                )
                and all(task["state"] == "COMPLETE" for task in summary["tasks"])
                and not summary["continuity_errors"]
            )
            if complete and run_root is not None:
                markers = summary["arm_markers"]
                if markers.get("campaign_type") == "planner":
                    complete = (
                        markers["campaign_completed"]
                        and markers["all_ranks_completed"]
                        and markers["allocation_records"] == markers["expected_ranks"]
                        and len(markers["rank_exit_codes"]) == markers["expected_ranks"]
                        and not any(markers["rank_exit_codes"])
                        and bool(markers["runs"])
                    )
                    complete = complete and all(
                        run["started"]
                        and run["completed"]
                        and not run["failed"]
                        and run["result"]
                        for run in markers["runs"]
                    )
                else:
                    complete = all(
                        arm["started"]
                        and arm["completed"]
                        and not arm["failed"]
                        and arm["allocation_records"] == arm["expected_ranks"]
                        for arm in markers["arms"]
                    )
            if not complete:
                raise SystemExit("job did not satisfy terminal completion gates")
            return
        if state not in ACTIVE_STATES:
            raise SystemExit(f"unrecognized non-terminal MAST state: {state}")
        if args.once:
            return
        time.sleep(args.interval_seconds)


def _dump_dir(definition: Any) -> str:
    values = sorted({str(value) for value in _walk_values(definition, "DUMP_DIR")})
    if len(values) != 1 or not values[0].startswith("/mnt/wsfuse/"):
        raise SystemExit(f"expected one /mnt/wsfuse DUMP_DIR, got {values}")
    return values[0]


def retrieve(args: argparse.Namespace) -> None:
    attempt = _absolute(args.attempt, "--attempt")
    if (attempt / "run").exists():
        raise SystemExit(f"refusing to overwrite retrieved run: {attempt / 'run'}")
    if not args.oilfs_uri.startswith("ws://"):
        raise SystemExit("--oilfs-uri must be an exact ws:// workspace URI")
    job_id = (attempt / "job_id.txt").read_text().strip()
    status_result = _capture(
        ["mast", "--output", "json", "get-status", job_id],
        cwd=HARNESS_ROOT,
        record_root=attempt / "retrieval" / "terminal_gate",
        name="status",
    )
    resolved = json.loads((attempt / "validation/resolved_campaign.json").read_text())
    summary = _scheduler_summary(
        json.loads(status_result.stdout), expected_nodes=int(resolved["mast"]["nodes"])
    )
    if not (
        summary["root_state"] == "COMPLETE"
        and summary["latest_attempt_state"] == "COMPLETE"
        and summary["task_groups"]
        and summary["tasks"]
        and all(group["state"] == "COMPLETE" for group in summary["task_groups"])
        and all(task["state"] == "COMPLETE" for task in summary["tasks"])
        and not summary["continuity_errors"]
    ):
        raise SystemExit("retrieval requires a fully successful terminal MAST job")
    definition = json.loads((attempt / "submitted_definition.json").read_text())
    packages = [str(value) for value in _walk_values(definition, "fbpkgIdentifier")]
    if args.oilfs_package not in packages:
        raise SystemExit(
            f"submitted definition does not contain {args.oilfs_package}: {packages}"
        )
    dump_dir = _dump_dir(definition)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    verification = attempt / "retrieval" / f"verification_{stamp}"
    if verification.exists():
        raise SystemExit(f"refusing to overwrite retrieval: {verification}")
    verification.mkdir(parents=True)
    oilfs = verification / "oilfs_package"
    oilfs.mkdir()
    _capture(
        ["fbpkg", "fetch", args.oilfs_package, "--dest", str(oilfs)],
        cwd=HARNESS_ROOT,
        record_root=verification,
        name="fetch_oilfs",
    )
    mountpoint = verification / "wsfuse"
    mountpoint.mkdir()
    wrapper = oilfs / "oilfs-wrapper"
    _capture(
        [
            str(wrapper),
            "--profile=genai",
            f"--user={args.oilfs_user}",
            "--log-level=info",
            args.oilfs_uri,
            str(mountpoint),
        ],
        cwd=HARNESS_ROOT,
        record_root=verification,
        name="mount_oilfs",
    )
    if not os.path.ismount(mountpoint):
        raise SystemExit(f"OilFS did not mount at {mountpoint}")

    source_run = mountpoint / Path(dump_dir).relative_to("/mnt/wsfuse") / "run"
    try:
        if not source_run.is_dir():
            raise SystemExit(f"job run output is missing: {source_run}")
        source_a = tree_manifest(source_run)
        _write_json(verification / "source_pass_a.json", source_a)
        time.sleep(5)
        source_b = tree_manifest(source_run)
        _write_json(verification / "source_pass_b.json", source_b)
        if source_a != source_b:
            raise SystemExit("OilFS source changed between manifest passes")
        shutil.copytree(source_run, attempt / "run", symlinks=True)
        local = tree_manifest(attempt / "run")
        _write_json(verification / "local.json", local)
        summary = {
            "status": "passed" if source_b == local else "failed",
            "oilfs_uri": args.oilfs_uri,
            "dump_dir": dump_dir,
            "source_run": str(source_run),
            "local_run": str(attempt / "run"),
            "file_count": len(local),
            "source_tree_sha256": manifest_digest(source_b),
            "local_tree_sha256": manifest_digest(local),
        }
        _write_json(verification / "retrieval_summary.json", summary)
        if summary["status"] != "passed":
            raise SystemExit("local run differs from stable OilFS source manifest")
    finally:
        _capture(
            ["fusermount", "-u", str(mountpoint)],
            cwd=HARNESS_ROOT,
            record_root=verification,
            name="unmount_oilfs",
            check=False,
        )
        if os.path.ismount(mountpoint):
            raise SystemExit(f"OilFS mount remains active: {mountpoint}")
    print(json.dumps(summary, indent=2, sort_keys=True))


def analyze(args: argparse.Namespace) -> None:
    attempt = _absolute(args.attempt, "--attempt")
    output = attempt / "analysis"
    if output.exists():
        raise SystemExit(f"refusing to overwrite analysis: {output}")
    packaged_harness = attempt / "package/payload/harness_repo"
    campaign = args.campaign or attempt / "package/payload/campaign/campaign.toml"
    python = args.python or Path(
        json.loads((attempt / "package_report.json").read_text())["validation"][
            "python"
        ]
    )
    command = [
        str(python),
        "-m",
        "harness.cli",
        "analyze",
        str(campaign),
    ]
    if args.point:
        command.extend(("--point", args.point))
    command.extend(
        (
            "--mode",
            args.mode,
            "--attempt-root",
            str(attempt),
            "--output",
            str(output),
        )
    )
    if args.tlparse_bin is not None:
        command.extend(("--tlparse-bin", str(args.tlparse_bin)))
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(packaged_harness),
    }
    _capture(
        command,
        cwd=packaged_harness,
        record_root=attempt / "measurement",
        name="analyze",
        env=env,
    )
    report = json.loads((output / "analysis.json").read_text())
    if report.get("status") != "passed":
        raise SystemExit(f"canonical analysis is {report.get('status')}")
    print(json.dumps(report, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compose the existing permanent-harness lifecycle without changing its metrics or configs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare", help="validate, seal/package, and perform the MAST dry-run audit"
    )
    prepare_parser.add_argument("--campaign", type=Path, required=True)
    prepare_parser.add_argument("--point")
    prepare_parser.add_argument("--mode", choices=("formal", "gate"), default="formal")
    prepare_parser.add_argument("--attempt", type=Path, required=True)
    prepare_parser.add_argument("--torchtitan-root", type=Path, required=True)
    prepare_parser.add_argument("--autoparallel-root", type=Path, required=True)
    prepare_parser.add_argument("--baseline-autoparallel-root", type=Path)
    prepare_parser.add_argument("--asset-root", action="append", default=[])
    prepare_parser.add_argument("--python", type=Path, default=Path(sys.executable))
    prepare_parser.set_defaults(func=prepare)

    submit_parser = subparsers.add_parser(
        "submit",
        help="submit the sealed attempt, set CRITICAL/99, and audit actual packages",
    )
    submit_parser.add_argument("--attempt", type=Path, required=True)
    submit_parser.set_defaults(func=submit)

    monitor_parser = subparsers.add_parser(
        "monitor", help="save structured scheduler and optional arm-marker snapshots"
    )
    monitor_parser.add_argument("--attempt", type=Path, required=True)
    monitor_parser.add_argument("--run-root", type=Path)
    monitor_parser.add_argument("--interval-seconds", type=int, default=30)
    monitor_parser.add_argument("--once", action="store_true")
    monitor_parser.set_defaults(func=monitor)

    retrieve_parser = subparsers.add_parser(
        "retrieve", help="copy the exact OilFS run after two stable source manifests"
    )
    retrieve_parser.add_argument("--attempt", type=Path, required=True)
    retrieve_parser.add_argument("--oilfs-uri", required=True)
    retrieve_parser.add_argument("--oilfs-user", default=os.environ.get("USER", ""))
    retrieve_parser.add_argument("--oilfs-package", default="oil.oilfs:stable")
    retrieve_parser.set_defaults(func=retrieve)

    analyze_parser = subparsers.add_parser(
        "analyze", help="invoke the sealed canonical analyzer and tlparse"
    )
    analyze_parser.add_argument("--attempt", type=Path, required=True)
    analyze_parser.add_argument("--campaign", type=Path)
    analyze_parser.add_argument("--point")
    analyze_parser.add_argument("--mode", choices=("formal", "gate"), default="formal")
    analyze_parser.add_argument("--python", type=Path)
    analyze_parser.add_argument("--tlparse-bin", type=Path)
    analyze_parser.set_defaults(func=analyze)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if getattr(args, "interval_seconds", 1) <= 0:
        raise SystemExit("--interval-seconds must be positive")
    if getattr(args, "oilfs_user", "valid") == "":
        raise SystemExit("--oilfs-user or USER is required")
    args.func(args)


if __name__ == "__main__":
    main()
