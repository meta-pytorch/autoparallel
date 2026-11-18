# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

from build_improvement_queue import build_improvement_queue, render_improvement_queue
from build_report_catalog import build_report_catalog
from run_report import resolve_report_config

FEEDBACK_STATUS_SCHEMA_VERSION = 1


def refresh_report_feedback(
    report_root: str | Path,
    *,
    verify: bool = True,
    expected_refresh_interval_seconds: int = 300,
    minimum_occurrences: int = 2,
    measurement_relative_error_threshold: float = 0.2,
    max_representatives: int = 3,
) -> dict[str, Any]:
    """Atomically refresh the catalog, triage queue, and health status."""

    if expected_refresh_interval_seconds < 1:
        raise ValueError("expected_refresh_interval_seconds must be positive")
    root = Path(report_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "feedback-status-v1.json"
    attempted_at = _utc_now()
    started = time.monotonic()

    with _refresh_lock(root / ".feedback-refresh-v1.lock") as acquired:
        if not acquired:
            return {
                "schema_version": FEEDBACK_STATUS_SCHEMA_VERSION,
                "status": "skipped_locked",
                "attempted_at": attempted_at,
                "report_root": str(root),
            }
        previous_status = _read_json_if_present(status_path)
        try:
            catalog = build_report_catalog(root, verify=verify)
            queue = build_improvement_queue(
                catalog,
                minimum_occurrences=minimum_occurrences,
                measurement_relative_error_threshold=(
                    measurement_relative_error_threshold
                ),
                max_representatives=max_representatives,
            )
            catalog_path = root / "catalog-v1.json"
            queue_path = root / "improvement-queue-v1.json"
            queue_markdown_path = root / "improvement-queue-v1.md"
            _atomic_write_json(catalog_path, catalog)
            _atomic_write_json(queue_path, queue)
            _atomic_write_text(
                queue_markdown_path,
                render_improvement_queue(queue),
            )

            completed_at = _utc_now()
            invalid_count = len(catalog["invalid_bundles"])
            status = {
                "schema_version": FEEDBACK_STATUS_SCHEMA_VERSION,
                "status": "degraded" if invalid_count else "healthy",
                "attempted_at": attempted_at,
                "completed_at": completed_at,
                "last_success_at": completed_at,
                "duration_seconds": time.monotonic() - started,
                "report_root": str(root),
                "verification_enabled": verify,
                "expected_refresh_interval_seconds": (
                    expected_refresh_interval_seconds
                ),
                "stale_after": _after_seconds(
                    completed_at, 2 * expected_refresh_interval_seconds
                ),
                "catalog": {
                    "path": catalog_path.name,
                    "generated_at": catalog["generated_at"],
                    "run_count": catalog["summary"]["run_count"],
                    "pending_bundle_count": len(catalog["pending_bundles"]),
                    "invalid_bundle_count": invalid_count,
                },
                "triage": {
                    "path": queue_path.name,
                    "markdown_path": queue_markdown_path.name,
                    "generated_at": queue["generated_at"],
                    "source_catalog_sha256": queue["source_catalog"]["sha256"],
                    "candidate_count": queue["summary"]["candidate_count"],
                    "priority_counts": queue["summary"]["priority_counts"],
                },
                "last_error": None,
            }
            _atomic_write_json(status_path, status)
            return status
        except Exception as error:
            failed_status = {
                "schema_version": FEEDBACK_STATUS_SCHEMA_VERSION,
                "status": "failed",
                "attempted_at": attempted_at,
                "completed_at": _utc_now(),
                "last_success_at": previous_status.get("last_success_at"),
                "duration_seconds": time.monotonic() - started,
                "report_root": str(root),
                "verification_enabled": verify,
                "expected_refresh_interval_seconds": (
                    expected_refresh_interval_seconds
                ),
                "stale_after": previous_status.get("stale_after"),
                "catalog": previous_status.get("catalog"),
                "triage": previous_status.get("triage"),
                "last_error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            }
            _atomic_write_json(status_path, failed_status)
            raise


def run_refresh_loop(
    report_root: str | Path,
    *,
    interval_seconds: int,
    verify: bool,
    minimum_occurrences: int,
    measurement_relative_error_threshold: float,
    max_representatives: int,
    stop_event: threading.Event | None = None,
) -> None:
    if interval_seconds < 1:
        raise ValueError("interval_seconds must be positive")
    stopped = stop_event or threading.Event()
    while not stopped.is_set():
        try:
            status = refresh_report_feedback(
                report_root,
                verify=verify,
                expected_refresh_interval_seconds=interval_seconds,
                minimum_occurrences=minimum_occurrences,
                measurement_relative_error_threshold=(
                    measurement_relative_error_threshold
                ),
                max_representatives=max_representatives,
            )
            print(json.dumps(status, sort_keys=True), flush=True)
        except Exception as error:
            print(
                f"AutoParallel feedback refresh failed: {type(error).__name__}: {error}",
                file=sys.stderr,
                flush=True,
            )
        stopped.wait(interval_seconds)


@contextmanager
def _refresh_lock(path: Path) -> Iterator[bool]:
    with path.open("a+") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _read_json_if_present(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _after_seconds(timestamp: str, seconds: int) -> str:
    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    return (parsed + timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content)
    os.replace(temporary, path)


def _resolved_policy(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    config = resolve_report_config(
        profile_name=args.profile_name,
        profiles_path=args.profiles_path,
        report_root=args.report_root,
    )
    feedback = config.profile.get("feedback", {})
    policy = {
        "interval_seconds": (
            args.interval_seconds
            if args.interval_seconds is not None
            else feedback.get("refresh_interval_seconds", 300)
        ),
        "minimum_occurrences": (
            args.minimum_occurrences
            if args.minimum_occurrences is not None
            else feedback.get("minimum_occurrences", 2)
        ),
        "measurement_relative_error_threshold": (
            args.measurement_relative_error_threshold
            if args.measurement_relative_error_threshold is not None
            else feedback.get("measurement_relative_error_threshold", 0.2)
        ),
        "max_representatives": (
            args.max_representatives
            if args.max_representatives is not None
            else feedback.get("max_representatives", 3)
        ),
    }
    return config.report_root, policy


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh AutoParallel catalog, triage queue, and health status"
    )
    parser.add_argument("report_root", nargs="?", help="Filesystem report root")
    parser.add_argument("--profile-name")
    parser.add_argument("--profiles-path")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval-seconds", type=int)
    parser.add_argument("--minimum-occurrences", type=int)
    parser.add_argument("--measurement-relative-error-threshold", type=float)
    parser.add_argument("--max-representatives", type=int)
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    report_root, policy = _resolved_policy(args)
    refresh_kwargs = {
        "verify": not args.no_verify,
        "minimum_occurrences": policy["minimum_occurrences"],
        "measurement_relative_error_threshold": policy[
            "measurement_relative_error_threshold"
        ],
        "max_representatives": policy["max_representatives"],
    }
    if not args.watch:
        status = refresh_report_feedback(
            report_root,
            expected_refresh_interval_seconds=policy["interval_seconds"],
            **refresh_kwargs,
        )
        print(json.dumps(status, indent=2, sort_keys=True))
        return

    stop_event = threading.Event()

    def stop(_signum: int, _frame: Any) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    run_refresh_loop(
        report_root,
        interval_seconds=policy["interval_seconds"],
        stop_event=stop_event,
        **refresh_kwargs,
    )


if __name__ == "__main__":
    main()
