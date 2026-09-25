from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .assets import ASSET_LOCK_PATH, load_asset_lock, resolve_assets
from .campaign import CampaignError, load_campaign, write_json
from .experiment_lock import load_experiment_lock
from .settings import resolve_run_setting

HARNESS_ROOT = Path(__file__).resolve().parents[1]
MEASUREMENT = HARNESS_ROOT / "scripts/measurements/measurement.py"


def _capture(
    command: list[str],
    *,
    cwd: Path,
    record_root: Path,
    name: str,
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
    if completed.returncode:
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
        if head.returncode or head.stdout.strip() != spec["commit"]:
            raise CampaignError(f"existing {name} checkout does not match its lock")
        return target
    _capture(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            spec["remote"],
            str(target),
        ],
        cwd=source_root,
        record_root=record_root,
        name=f"clone_{name}",
    )
    _capture(
        ["git", "-C", str(target), "fetch", "origin", spec["commit"]],
        cwd=source_root,
        record_root=record_root,
        name=f"fetch_{name}",
    )
    _capture(
        ["git", "-C", str(target), "checkout", "--detach", spec["commit"]],
        cwd=source_root,
        record_root=record_root,
        name=f"checkout_{name}",
    )
    return target


def _mount_assets(task_root: Path, record_root: Path) -> tuple[Path, Path]:
    lock = load_asset_lock()
    package_root = task_root / "environment/oilfs"
    if not package_root.exists():
        package_root.mkdir(parents=True)
        _capture(
            ["fbpkg", "fetch", "oil.oilfs:stable", "--dest", str(package_root)],
            cwd=HARNESS_ROOT,
            record_root=record_root,
            name="fetch_oilfs",
        )
    wrapper = package_root / "oilfs-wrapper"
    mountpoint = task_root / "mounts/assets"
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
            name="mount_assets",
        )
    if not os.path.ismount(mountpoint):
        raise CampaignError(f"OilFS did not mount at {mountpoint}")
    return mountpoint, package_root


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
        record_root=task_root / "run_commands",
        name=action,
    )


def run_reproduction(model: str, setting_name: str) -> dict[str, Any]:
    setting = resolve_run_setting(model, setting_name)
    workspace_root = Path(
        os.environ.get("HARNESS_WORKSPACE_ROOT", Path.home() / "workspace")
    ).resolve()
    task_root = workspace_root / f"{model}-{setting_name}-reproduction"
    task_root.mkdir(parents=True, exist_ok=True)
    record_root = task_root / "bootstrap"

    lock = load_experiment_lock()
    source_root = task_root / "source"
    source_root.mkdir(exist_ok=True)
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
    python = Path(sys.executable)

    campaign = load_campaign(setting.campaign, point=setting.point, mode="formal")
    mountpoint, _ = _mount_assets(task_root, record_root)
    required_assets = list(campaign.raw.get("artifacts", {}).get("required_assets", []))
    assets, asset_evidence = resolve_assets(mountpoint, required_assets)
    write_json(record_root / "asset_evidence.json", asset_evidence)

    attempt = task_root / "attempt"
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
    for name, root in sorted(assets.items()):
        prepare_extra.extend(("--asset-root", f"{name}={root}"))

    try:
        if not (attempt / "SEALED").is_file():
            _measurement(
                "prepare",
                task_root=task_root,
                attempt=attempt,
                python=python,
                extra=prepare_extra,
            )
        if not (attempt / "job_id.txt").is_file():
            _measurement("submit", task_root=task_root, attempt=attempt, python=python)
        _measurement("monitor", task_root=task_root, attempt=attempt, python=python)
        if not (attempt / "run").is_dir():
            _measurement(
                "retrieve",
                task_root=task_root,
                attempt=attempt,
                python=python,
                extra=["--oilfs-uri", load_asset_lock()["workspace_uri"]],
            )
        analysis = attempt / "analysis/analysis.json"
        if not analysis.is_file():
            tlparse = shutil.which("tlparse")
            if not tlparse:
                raise CampaignError("tlparse is required for canonical analysis")
            analyze_extra = [
                "--campaign",
                str(setting.campaign),
                "--mode",
                "formal",
                "--python",
                str(python),
                "--tlparse-bin",
                tlparse,
            ]
            if setting.point:
                analyze_extra.extend(("--point", setting.point))
            _measurement(
                "analyze",
                task_root=task_root,
                attempt=attempt,
                python=python,
                extra=analyze_extra,
            )
    finally:
        if os.path.ismount(mountpoint):
            _capture(
                ["fusermount", "-u", str(mountpoint)],
                cwd=HARNESS_ROOT,
                record_root=record_root,
                name="unmount_assets",
            )

    result = {
        "status": "complete",
        "model": model,
        "setting": setting_name,
        "task_root": str(task_root),
        "attempt": str(attempt),
        "job_id": (attempt / "job_id.txt").read_text().strip(),
        "analysis": str(analysis),
        "asset_lock": str(ASSET_LOCK_PATH),
    }
    write_json(task_root / "run_result.json", result)
    return result
