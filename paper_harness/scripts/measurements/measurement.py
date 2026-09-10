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
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HARNESS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HARNESS_ROOT))

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
        "probe_configs"
    ):
        raise SystemExit("package did not preserve a passing ConfigManager validation")

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


def _content_manifest(root: Path) -> dict[str, str]:
    return {
        path: digest
        for path, digest in tree_manifest(root).items()
        if path != "METADATA" and not path.endswith(".CHECKSUMS")
    }


def _one_package(packages: list[str], prefix: str) -> str:
    matches = sorted({package for package in packages if package.startswith(prefix)})
    if len(matches) != 1:
        raise SystemExit(f"expected one {prefix} package, got {matches}")
    return matches[0]


def submit(args: argparse.Namespace) -> None:
    attempt = _absolute(args.attempt, "--attempt")
    if (attempt / "job_id.txt").exists():
        raise SystemExit(f"attempt already has a submitted job: {attempt}")
    record_root = attempt / "measurement"
    python = Path(
        json.loads((attempt / "package_report.json").read_text())["validation"][
            "python"
        ]
    )
    _capture(
        [str(python), "-m", "harness.cli", "submit", "--attempt", str(attempt)],
        cwd=HARNESS_ROOT,
        record_root=record_root,
        name="submit",
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    job_id = (attempt / "job_id.txt").read_text().strip()

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
    packages = [str(value) for value in _walk_values(definition, "fbpkgIdentifier")]
    package_ids = {
        "workspace": _one_package(packages, "torchtitan_workspace:"),
        "payload": _one_package(packages, "torchtitan_additional_packages:"),
    }

    package_root = attempt / "packages"
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
    preflight_output = attempt / "exact_submitted_package_preflight"
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
    )
    expected_payload = _content_manifest(attempt / "package/payload")
    actual_payload = _content_manifest(fetched_payload)
    expected_workspace = _content_manifest(package_root / "dryrun_workspace")
    actual_workspace = _content_manifest(destinations["workspace"])
    audit = {
        "status": "passed",
        "job_id": job_id,
        "priority": priority_data,
        "packages": package_ids,
        "payload": {
            "matched": expected_payload == actual_payload,
            "expected_tree_sha256": manifest_digest(expected_payload),
            "actual_tree_sha256": manifest_digest(actual_payload),
        },
        "workspace": {
            "matched": expected_workspace == actual_workspace,
            "expected_tree_sha256": manifest_digest(expected_workspace),
            "actual_tree_sha256": manifest_digest(actual_workspace),
        },
        "exact_package_preflight_returncode": preflight.returncode,
    }
    if not audit["payload"]["matched"] or not audit["workspace"]["matched"]:
        audit["status"] = "failed"
    _write_json(attempt / "submitted_package_audit.json", audit)
    if audit["status"] != "passed":
        raise SystemExit("actual submitted package content differs from dry-run inputs")
    print(json.dumps(audit, indent=2, sort_keys=True))


def _task_groups(latest: dict[str, Any]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for attempts in latest.get("taskGroupExecutionAttempts", {}).values():
        groups.extend(attempts)
    return groups


def _arm_markers(attempt: Path, run_root: Path) -> dict[str, Any]:
    resolved = json.loads((attempt / "validation/resolved_campaign.json").read_text())
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


def _scheduler_summary(status: dict[str, Any]) -> dict[str, Any]:
    data = status["data"]
    latest = data.get("latestAttempt") or {}
    groups = _task_groups(latest)
    tasks = [
        task
        for group in groups
        for attempts in group.get("taskExecutionAttempts", {}).values()
        for task in attempts
    ]
    continuity_errors = []
    if latest.get("attemptIndex", 0) != 0:
        continuity_errors.append("scheduler attempt index is not zero")
    if int(data.get("numRestarts") or 0) != 0:
        continuity_errors.append("scheduler restart count is not zero")
    for group in groups:
        if group.get("attemptIndex", 0) != 0 or group.get("attemptEpoch", 0) != 0:
            continuity_errors.append("task-group attempt/epoch is not zero")
        if int(group.get("numFailedTasks") or 0) != 0:
            continuity_errors.append("task group reports failed tasks")
        if int(group.get("numShrunkTasks") or 0) != 0:
            continuity_errors.append("task group reports shrunk tasks")
        if group.get("onElasticCapacity") is True:
            continuity_errors.append("task group is on elastic capacity")
    if any(task.get("attemptIndex", 0) != 0 for task in tasks):
        continuity_errors.append("task execution attempt index is not zero")
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


def _terminal_evidence(attempt: Path, job_id: str, stamp: str) -> None:
    root = attempt / "terminal_evidence" / stamp
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
        _capture(command, cwd=HARNESS_ROOT, record_root=root, name=name, check=False)


def monitor(args: argparse.Namespace) -> None:
    attempt = _absolute(args.attempt, "--attempt")
    run_root = _absolute(args.run_root, "--run-root") if args.run_root else None
    job_id = (attempt / "job_id.txt").read_text().strip()
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
        summary = _scheduler_summary(status)
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
            _terminal_evidence(attempt, job_id, stamp)
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
                complete = all(
                    arm["started"]
                    and arm["completed"]
                    and not arm["failed"]
                    and arm["allocation_records"] == arm["expected_ranks"]
                    for arm in summary["arm_markers"]["arms"]
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
    summary = _scheduler_summary(json.loads(status_result.stdout))
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
            "--tlparse-bin",
            str(args.tlparse_bin),
        )
    )
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
    analyze_parser.add_argument("--tlparse-bin", type=Path, required=True)
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
