from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import tomllib
from pathlib import Path
from typing import Any

from .campaign import CampaignError


LOCK_PATH = Path(__file__).resolve().parents[1] / "experiment_lock.toml"


def load_experiment_lock(path: Path = LOCK_PATH) -> dict[str, Any]:
    with path.open("rb") as stream:
        lock = tomllib.load(stream)
    if lock.get("schema_version") != 1:
        raise CampaignError("experiment_lock.toml schema_version must be 1")
    return lock


def experiment_lock_digest(path: Path = LOCK_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_campaign_lock(raw: dict[str, Any]) -> dict[str, Any]:
    lock = load_experiment_lock()
    mismatches: dict[str, dict[str, Any]] = {}
    for name in ("torchtitan", "autoparallel"):
        expected = lock["sources"][name]
        actual = raw["sources"][name]
        for field in ("remote", "commit"):
            if actual.get(field) != expected[field]:
                mismatches[f"sources.{name}.{field}"] = {
                    "expected": expected[field],
                    "actual": actual.get(field),
                }
        if actual.get("dirty_policy", "forbid") != "forbid":
            mismatches[f"sources.{name}.dirty_policy"] = {
                "expected": "forbid",
                "actual": actual.get("dirty_policy"),
            }
    expected_fbpkg = lock["runtime"]["conda_fbpkg"]
    actual_fbpkg = raw["mast"].get("conda_fbpkg")
    if actual_fbpkg != expected_fbpkg:
        mismatches["mast.conda_fbpkg"] = {
            "expected": expected_fbpkg,
            "actual": actual_fbpkg,
        }
    if mismatches:
        raise CampaignError(f"campaign violates experiment lock: {mismatches}")
    return lock


def validate_runtime_lock(torch_module: Any) -> dict[str, Any]:
    lock = load_experiment_lock()
    expected = lock["runtime"]
    observed: dict[str, Any] = {
        "python_version": platform.python_version(),
        "torch_version": torch_module.__version__,
        "torch_git_version": getattr(torch_module.version, "git_version", None),
        "cuda_version": torch_module.version.cuda,
        "nccl_version": list(torch_module.cuda.nccl.version()),
        "packages": {
            name: importlib.metadata.version(name)
            for name in expected.get("packages", {})
        },
    }
    mismatches = {
        name: {"expected": expected[name], "actual": observed[name]}
        for name in (
            "python_version",
            "torch_version",
            "torch_git_version",
            "cuda_version",
            "nccl_version",
            "packages",
        )
        if observed[name] != expected[name]
    }
    if mismatches:
        raise RuntimeError(f"runtime violates experiment lock: {mismatches}")
    return {
        "status": "passed",
        "harness_version": lock["harness_version"],
        "lock_sha256": experiment_lock_digest(),
        "observed": observed,
    }
