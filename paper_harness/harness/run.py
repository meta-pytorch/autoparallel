from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .assets import ASSET_LOCK_PATH, load_asset_lock, resolve_assets
from .campaign import CampaignError, load_campaign, write_json
from .experiment_lock import experiment_lock_digest, load_experiment_lock
from .settings import resolve_run_setting

HARNESS_ROOT = Path(__file__).resolve().parents[1]
MEASUREMENT = HARNESS_ROOT / "scripts/measurements/measurement.py"


def _capture(
    command: list[str],
    *,
    cwd: Path,
    record_root: Path,
    name: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    record_root.mkdir(parents=True, exist_ok=True)
    write_json(record_root / f"{name}.command.json", command)
    (record_root / f"{name}.stdout").write_text(completed.stdout)
    (record_root / f"{name}.stderr").write_text(completed.stderr)
    (record_root / f"{name}.returncode").write_text(f"{completed.returncode}\n")
    if check and completed.returncode:
        raise CampaignError(f"{' '.join(command)} exited {completed.returncode}")
    return completed


def _materialize_source(
    name: str,
    spec: dict[str, Any],
    *,
    source_root: Path,
    record_root: Path,
) -> Path:
    target = source_root / name
    if target.exists():
        head = subprocess.run(
            ["git", "-C", str(target), "rev-parse", "HEAD"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        status = subprocess.run(
            [
                "git",
                "-C",
                str(target),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if (
            head.returncode
            or head.stdout.strip() != spec["commit"]
            or status.returncode
            or status.stdout.strip()
        ):
            raise CampaignError(f"existing {name} checkout does not match its lock")
        return target
    staging = source_root / f".{name}.staging.{os.getpid()}"
    if staging.exists():
        raise CampaignError(f"source staging path already exists: {staging}")
    _capture(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            spec["remote"],
            str(staging),
        ],
        cwd=source_root,
        record_root=record_root,
        name=f"clone_{name}",
    )
    _capture(
        ["git", "-C", str(staging), "fetch", "origin", spec["commit"]],
        cwd=source_root,
        record_root=record_root,
        name=f"fetch_{name}",
    )
    _capture(
        ["git", "-C", str(staging), "checkout", "--detach", spec["commit"]],
        cwd=source_root,
        record_root=record_root,
        name=f"checkout_{name}",
    )
    status = subprocess.run(
        [
            "git",
            "-C",
            str(staging),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if status.returncode or status.stdout.strip():
        raise CampaignError(f"new {name} checkout is not clean")
    staging.rename(target)
    return target


def _fetch_package(identifier: str, destination: Path, record_root: Path) -> Path:
    if not destination.exists():
        staging = destination.parent / f".{destination.name}.staging.{os.getpid()}"
        if staging.exists():
            raise CampaignError(f"package staging path already exists: {staging}")
        staging.mkdir(parents=True)
        _capture(
            ["fbpkg", "fetch", identifier, "--dest", str(staging)],
            cwd=HARNESS_ROOT,
            record_root=record_root,
            name=f"fetch_{identifier.replace(':', '_')}",
        )
        _package_python(identifier, staging)
        staging.rename(destination)
    return _package_python(identifier, destination)


def _package_python(identifier: str, destination: Path) -> Path:
    matches = sorted(destination.rglob("conda/bin/python"))
    if len(matches) != 1:
        raise CampaignError(
            f"expected one Python in fetched {identifier}, found {matches}"
        )
    metadata = destination / "METADATA"
    checksums = list(destination.glob("*.CHECKSUMS"))
    if not metadata.is_file() or len(checksums) != 1:
        raise CampaignError(
            f"fetched {identifier} is missing METADATA or its checksum manifest"
        )
    try:
        metadata_value = json.loads(metadata.read_text())
    except json.JSONDecodeError as error:
        raise CampaignError(f"fetched {identifier} has invalid METADATA") from error
    expected_name, expected_version = identifier.split(":", 1)
    if (
        metadata_value.get("package") != expected_name
        or str(metadata_value.get("version")) != expected_version
    ):
        raise CampaignError(f"fetched package metadata does not identify {identifier}")
    return matches[0]


def _mount_workspace(
    task_root: Path, record_root: Path, *, name: str
) -> tuple[Path, Path]:
    lock = load_asset_lock()
    package_root = task_root / "environment/oilfs"
    if not package_root.exists():
        package_root.parent.mkdir(parents=True, exist_ok=True)
        staging = package_root.parent / f".{package_root.name}.staging.{os.getpid()}"
        if staging.exists():
            raise CampaignError(f"OilFS staging path already exists: {staging}")
        staging.mkdir()
        _capture(
            ["fbpkg", "fetch", "oil.oilfs:stable", "--dest", str(staging)],
            cwd=HARNESS_ROOT,
            record_root=record_root,
            name="fetch_oilfs",
        )
        if (
            not (staging / "oilfs-wrapper").is_file()
            or not (staging / "METADATA").is_file()
        ):
            raise CampaignError("fetched oil.oilfs package is incomplete")
        staging.rename(package_root)
    wrapper = package_root / "oilfs-wrapper"
    if not wrapper.is_file() or not (package_root / "METADATA").is_file():
        raise CampaignError(f"OilFS package is incomplete: {package_root}")
    mountpoint = task_root / "mounts" / name
    mountpoint.mkdir(parents=True, exist_ok=True)
    if not os.path.ismount(mountpoint):
        user = os.environ.get("USER")
        if not user:
            raise CampaignError("USER is required for OilFS mounting")
        _capture(
            [
                str(wrapper),
                "--profile=genai",
                f"--user={user}",
                "--log-level=info",
                lock["workspace_uri"],
                str(mountpoint),
            ],
            cwd=HARNESS_ROOT,
            record_root=record_root,
            name=f"mount_{name}",
        )
    if not os.path.ismount(mountpoint):
        raise CampaignError(f"OilFS did not mount at {mountpoint}")
    return mountpoint, package_root


@contextmanager
def _mounted_workspace(task_root: Path, record_root: Path, *, name: str):
    mountpoint, package_root = _mount_workspace(task_root, record_root, name=name)
    try:
        yield mountpoint, package_root
    except BaseException:
        if os.path.ismount(mountpoint):
            try:
                _capture(
                    ["fusermount", "-u", str(mountpoint)],
                    cwd=HARNESS_ROOT,
                    record_root=record_root,
                    name=f"unmount_{name}",
                    check=False,
                )
            except OSError as cleanup_error:
                write_json(
                    record_root / f"unmount_{name}.cleanup_error.json",
                    {"error": f"{type(cleanup_error).__name__}: {cleanup_error}"},
                )
        raise
    else:
        if os.path.ismount(mountpoint):
            _capture(
                ["fusermount", "-u", str(mountpoint)],
                cwd=HARNESS_ROOT,
                record_root=record_root,
                name=f"unmount_{name}",
            )
        if os.path.ismount(mountpoint):
            raise CampaignError(f"OilFS mount remains active at {mountpoint}")


def _measurement(
    action: str,
    *,
    task_root: Path,
    attempt: Path,
    python: Path,
    extra: list[str] | None = None,
) -> None:
    command = [str(python), str(MEASUREMENT), action, "--attempt", str(attempt)]
    command.extend(extra or [])
    _capture(
        command,
        cwd=HARNESS_ROOT,
        record_root=attempt / "run_commands",
        name=action,
    )


def _attempt_path(task_root: Path, attempt_number: int = 1) -> Path:
    if not 1 <= attempt_number <= 999:
        raise CampaignError("--attempt-number must be between 1 and 999")
    legacy = task_root / "attempt"
    if legacy.exists():
        raise CampaignError(
            f"legacy mutable attempt layout exists at {legacy}; refusing to migrate "
            "or overwrite it automatically"
        )
    return task_root / "attempts" / f"{attempt_number:03d}"


def _materialization_root(task_root: Path, lock_digest: str) -> tuple[Path, bool]:
    configured = os.environ.get("HARNESS_CACHE_ROOT")
    if not configured:
        return task_root, False
    cache_root = Path(configured)
    if not cache_root.is_absolute():
        raise CampaignError("HARNESS_CACHE_ROOT must be an absolute path")
    return cache_root.resolve() / lock_digest, True


@contextmanager
def _materialization_lock(root: Path, shared: bool):
    if not shared:
        yield
        return
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".materialize.lock").open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _write_materialization_ready(
    path: Path,
    *,
    lock_digest: str,
    lock: dict[str, Any],
    python: Path,
) -> None:
    value = {
        "schema_version": 1,
        "experiment_lock_sha256": lock_digest,
        "sources": {
            name: spec["commit"] for name, spec in sorted(lock["sources"].items())
        },
        "conda_fbpkg": lock["runtime"]["conda_fbpkg"],
        "python": str(python.resolve()),
    }
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _validate_materialization_ready(
    path: Path,
    *,
    lock_digest: str,
    lock: dict[str, Any],
    python: Path,
) -> None:
    if not path.is_file():
        raise CampaignError(f"shared materialization has no READY record: {path}")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignError(
            f"invalid shared materialization READY record: {path}"
        ) from error
    expected_sources = {
        name: spec["commit"] for name, spec in sorted(lock["sources"].items())
    }
    if (
        value.get("schema_version") != 1
        or value.get("experiment_lock_sha256") != lock_digest
        or value.get("sources") != expected_sources
        or value.get("conda_fbpkg") != lock["runtime"]["conda_fbpkg"]
        or value.get("python") != str(python.resolve())
    ):
        raise CampaignError(f"shared materialization READY record is stale: {path}")


def _passing_json(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text()).get("status") == "passed"
    except (OSError, json.JSONDecodeError, AttributeError):
        return False


def run_reproduction(
    model: str, setting_name: str, *, attempt_number: int = 1
) -> dict[str, Any]:
    setting = resolve_run_setting(model, setting_name)
    workspace_root = Path(
        os.environ.get("HARNESS_WORKSPACE_ROOT", Path.home() / "workspace")
    ).resolve()
    task_root = workspace_root / f"{model}-{setting_name}-reproduction"
    task_root.mkdir(parents=True, exist_ok=True)
    attempt = _attempt_path(task_root, attempt_number)
    record_root = task_root / "bootstrap" / f"{attempt_number:03d}"

    lock = load_experiment_lock()
    lock_digest = experiment_lock_digest()
    materialization_root, shared_cache = _materialization_root(task_root, lock_digest)
    source_root = materialization_root / "source"
    environment_root = materialization_root / "environment"
    with _materialization_lock(materialization_root, shared_cache):
        source_root.mkdir(parents=True, exist_ok=True)
        torchtitan_root = _materialize_source(
            "torchtitan",
            lock["sources"]["torchtitan"],
            source_root=source_root,
            record_root=record_root,
        )
        autoparallel_root = _materialize_source(
            "autoparallel",
            lock["sources"]["autoparallel"],
            source_root=source_root,
            record_root=record_root,
        )
        python = _fetch_package(
            lock["runtime"]["conda_fbpkg"],
            environment_root / "torchtitan_conda_prod",
            record_root,
        )
        ready = materialization_root / "READY.json"
        if ready.exists():
            _validate_materialization_ready(
                ready, lock_digest=lock_digest, lock=lock, python=python
            )
        else:
            _write_materialization_ready(
                ready, lock_digest=lock_digest, lock=lock, python=python
            )
    write_json(
        record_root / "materialization.json",
        {
            "shared_cache": shared_cache,
            "experiment_lock_sha256": lock_digest,
            "materialization_root": str(materialization_root),
            "torchtitan_root": str(torchtitan_root),
            "autoparallel_root": str(autoparallel_root),
            "python": str(python),
        },
    )

    campaign = load_campaign(setting.campaign, point=setting.point, mode="formal")
    required_assets = list(campaign.raw.get("artifacts", {}).get("required_assets", []))
    if not (attempt / "SEALED").is_file():
        prepare_extra = [
            "--campaign",
            str(setting.campaign),
            "--mode",
            "formal",
            "--torchtitan-root",
            str(torchtitan_root),
            "--autoparallel-root",
            str(autoparallel_root),
            "--python",
            str(python),
        ]
        if setting.point:
            prepare_extra.extend(("--point", setting.point))
        if required_assets:
            with _mounted_workspace(task_root, record_root, name="assets") as (
                mountpoint,
                _,
            ):
                assets, asset_evidence = resolve_assets(mountpoint, required_assets)
                write_json(record_root / "asset_evidence.json", asset_evidence)
                for name, root in sorted(assets.items()):
                    prepare_extra.extend(("--asset-root", f"{name}={root}"))
                _measurement(
                    "prepare",
                    task_root=task_root,
                    attempt=attempt,
                    python=python,
                    extra=prepare_extra,
                )
        else:
            _measurement(
                "prepare",
                task_root=task_root,
                attempt=attempt,
                python=python,
                extra=prepare_extra,
            )
    elif not _passing_json(attempt / "dryrun_audit.json"):
        raise CampaignError(
            f"attempt {attempt} is sealed without a passing dry-run; choose a new "
            "--attempt-number explicitly"
        )

    post_submit_error: Exception | None = None
    if not _passing_json(attempt / "submitted_package_audit.json"):
        try:
            _measurement("submit", task_root=task_root, attempt=attempt, python=python)
        except Exception as error:
            if not (attempt / "job_id.txt").is_file():
                raise
            post_submit_error = error
            write_json(
                attempt / "post_submit_failure.json",
                {"status": "failed", "error": str(error)},
            )

    job_id = (attempt / "job_id.txt").read_text().strip()
    monitor_error: Exception | None = None
    try:
        monitorpoint, _ = _mount_workspace(task_root, record_root, name="monitor")
    except Exception as error:
        monitor_error = error
        write_json(
            attempt / "monitor_mount_failure.json",
            {"status": "failed", "error": str(error)},
        )
        _measurement("monitor", task_root=task_root, attempt=attempt, python=python)
    else:
        monitor_failure: BaseException | None = None
        try:
            _measurement(
                "monitor",
                task_root=task_root,
                attempt=attempt,
                python=python,
                extra=[
                    "--run-root",
                    str(monitorpoint / "outputs" / job_id / "run"),
                ],
            )
        except BaseException as error:
            monitor_failure = error
            raise
        finally:
            if os.path.ismount(monitorpoint):
                try:
                    _capture(
                        ["fusermount", "-u", str(monitorpoint)],
                        cwd=HARNESS_ROOT,
                        record_root=record_root,
                        name="unmount_monitor",
                        check=False,
                    )
                except OSError as cleanup_error:
                    write_json(
                        record_root / "unmount_monitor.cleanup_error.json",
                        {"error": (f"{type(cleanup_error).__name__}: {cleanup_error}")},
                    )
            if os.path.ismount(monitorpoint) and monitor_failure is None:
                raise CampaignError(f"OilFS mount remains active at {monitorpoint}")
    if post_submit_error is not None:
        raise CampaignError(
            f"job {job_id} reached a terminal state, but post-submit audit failed: "
            f"{post_submit_error}"
        )
    if monitor_error is not None:
        raise CampaignError(
            f"job {job_id} reached a terminal state without live artifact monitoring: "
            f"{monitor_error}"
        )

    if not (attempt / "run").is_dir():
        _measurement(
            "retrieve",
            task_root=task_root,
            attempt=attempt,
            python=python,
            extra=["--oilfs-uri", load_asset_lock()["workspace_uri"]],
        )
    analysis_root = attempt / "analysis"
    analysis = analysis_root / "analysis.json"
    if analysis_root.exists() and not analysis.is_file():
        raise CampaignError(
            f"attempt {attempt} has an incomplete analysis directory; choose a new "
            "--attempt-number explicitly"
        )
    if analysis.is_file() and not _passing_json(analysis):
        raise CampaignError(
            f"attempt {attempt} has non-passing analysis; choose a new "
            "--attempt-number explicitly"
        )
    if not analysis.is_file():
        sealed_resolved = json.loads(
            (attempt / "validation/resolved_campaign.json").read_text()
        )
        analyze_extra = [
            "--mode",
            "formal",
            "--python",
            str(python),
        ]
        if sealed_resolved.get("campaign_type", "training") != "planner":
            tlparse = shutil.which("tlparse")
            if not tlparse:
                raise CampaignError("tlparse is required for canonical analysis")
            analyze_extra.extend(("--tlparse-bin", tlparse))
        selected_point = sealed_resolved.get("selected_point")
        if selected_point:
            analyze_extra.extend(("--point", str(selected_point)))
        _measurement(
            "analyze",
            task_root=task_root,
            attempt=attempt,
            python=python,
            extra=analyze_extra,
        )
        if not _passing_json(analysis):
            raise CampaignError(f"canonical analysis did not pass: {analysis}")

    result = {
        "status": "complete",
        "model": model,
        "setting": setting_name,
        "attempt_number": attempt_number,
        "task_root": str(task_root),
        "attempt": str(attempt),
        "job_id": (attempt / "job_id.txt").read_text().strip(),
        "analysis": str(analysis),
        "asset_lock": str(ASSET_LOCK_PATH),
    }
    write_json(attempt / "run_result.json", result)
    write_json(task_root / "run_result.json", result)
    return result
