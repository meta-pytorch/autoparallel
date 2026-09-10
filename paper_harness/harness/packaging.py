from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from .campaign import Campaign, CampaignError, write_json
from .sources import manifest_digest, tree_manifest
from .validation import validate_campaign


def _ignore(_directory: str, names: list[str]) -> set[str]:
    ignored = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"}
    return set(names) & ignored


def _copy_tree(source: Path, target: Path) -> None:
    if target.exists():
        raise CampaignError(f"refusing to overwrite package path {target}")
    shutil.copytree(source, target, symlinks=True, ignore=_ignore)


def _copy_harness(repo_root: Path, target: Path) -> None:
    files = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout.split(b"\0")
    for raw in files:
        if not raw:
            continue
        relative = Path(os.fsdecode(raw))
        source = repo_root / relative
        if not source.is_file():
            continue
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def package_campaign(
    campaign: Campaign,
    *,
    torchtitan_root: Path,
    autoparallel_root: Path,
    attempt_root: Path,
    asset_roots: dict[str, Path] | None = None,
    probe_configs: bool = True,
    python: Path | None = None,
) -> dict:
    attempt_root = attempt_root.resolve()
    if attempt_root.exists() and any(attempt_root.iterdir()):
        raise CampaignError(f"refusing to overwrite non-empty attempt directory {attempt_root}")
    attempt_root.mkdir(parents=True, exist_ok=True)
    validation_root = attempt_root / "validation"
    validation = validate_campaign(
        campaign,
        torchtitan_root=torchtitan_root,
        autoparallel_root=autoparallel_root,
        output_dir=validation_root,
        probe_configs=probe_configs,
        asset_roots=asset_roots,
        python=python,
    )
    payload = attempt_root / "package" / "payload"
    _copy_harness(Path(__file__).resolve().parents[1], payload / "harness_repo")
    _copy_tree(torchtitan_root.resolve(), payload / "torchtitan")
    _copy_tree(autoparallel_root.resolve(), payload / "autoparallel")
    (payload / "campaign").mkdir(parents=True)
    shutil.copy2(campaign.path, payload / "campaign" / "campaign.toml")
    shutil.copy2(
        validation_root / "resolved_campaign.json",
        payload / "campaign" / "resolved_campaign.json",
    )
    shutil.copy2(validation_root / "source_lock.json", payload / "campaign/source_lock.json")
    shutil.copy2(
        Path(__file__).resolve().parents[1] / "experiment_lock.toml",
        payload / "campaign/experiment_lock.toml",
    )
    if (validation_root / "serialized_configs").is_dir():
        _copy_tree(
            validation_root / "serialized_configs",
            payload / "campaign" / "serialized_configs",
        )
    for name, root in sorted((asset_roots or {}).items()):
        _copy_tree(root.resolve(), payload / "assets" / name)

    asset_copy_checks = {}
    for name in sorted(asset_roots or {}):
        copied_manifest = tree_manifest(payload / "assets" / name)
        copied_digest = manifest_digest(copied_manifest)
        expected_digest = validation["asset_lock"][name]["tree_sha256"]
        if copied_digest != expected_digest:
            raise CampaignError(
                f"packaged asset {name} differs from validated input tree"
            )
        asset_copy_checks[name] = {
            "status": "passed",
            "tree_sha256": copied_digest,
            "file_count": len(copied_manifest),
        }

    source_copy_checks = {}
    for name in ("torchtitan", "autoparallel"):
        copied_manifest = tree_manifest(payload / name)
        copied_digest = manifest_digest(copied_manifest)
        expected_digest = validation["source_lock"][name]["tree_sha256"]
        if copied_digest != expected_digest:
            raise CampaignError(
                f"packaged {name} tree {copied_digest} differs from validated "
                f"source tree {expected_digest}"
            )
        source_copy_checks[name] = {
            "status": "passed",
            "tree_sha256": copied_digest,
            "file_count": len(copied_manifest),
        }
    payload_manifest = tree_manifest(payload)
    package_report = {
        "status": "passed",
        "payload": str(payload),
        "payload_tree_sha256": manifest_digest(payload_manifest),
        "payload_file_count": len(payload_manifest),
        "validation": validation,
        "source_copy_checks": source_copy_checks,
        "asset_copy_checks": asset_copy_checks,
    }
    write_json(attempt_root / "package_manifest.json", payload_manifest)
    write_json(attempt_root / "package_report.json", package_report)
    (attempt_root / "SEALED").write_text(package_report["payload_tree_sha256"] + "\n")
    return package_report
