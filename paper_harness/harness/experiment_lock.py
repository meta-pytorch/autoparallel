from __future__ import annotations

import hashlib
import importlib.metadata
import platform
from pathlib import Path
from typing import Any

import tomllib

from .campaign import CampaignError

LOCK_PATH = Path(__file__).resolve().parents[1] / "experiment_lock.toml"


def load_experiment_lock(path: Path = LOCK_PATH) -> dict[str, Any]:
    with path.open("rb") as stream:
        lock = tomllib.load(stream)
    if lock.get("schema_version") != 1:
        raise CampaignError("experiment_lock.toml schema_version must be 1")
    sources = lock.get("sources")
    if not isinstance(sources, dict) or set(sources) != {
        "torchtitan",
        "autoparallel",
    }:
        raise CampaignError("experiment lock requires exactly two source entries")
    for name, source in sources.items():
        if not isinstance(source, dict):
            raise CampaignError(f"experiment lock source {name!r} must be a table")
        if not source.get("remote") or not source.get("commit"):
            raise CampaignError(
                f"experiment lock source {name!r} requires remote and commit"
            )
        if source.get("dirty_policy") != "forbid":
            raise CampaignError(
                f"experiment lock source {name!r} must forbid dirty trees"
            )
    runtime = lock.get("runtime")
    if not isinstance(runtime, dict) or not runtime.get("conda_fbpkg"):
        raise CampaignError("experiment lock requires a runtime conda_fbpkg")
    execution = lock.get("execution")
    if not isinstance(execution, dict) or execution.get("spmd_backend") != "default":
        raise CampaignError("experiment lock requires spmd_backend='default'")
    workspace_fbpkg_id = execution.get("workspace_fbpkg_id")
    if not isinstance(workspace_fbpkg_id, str) or not workspace_fbpkg_id.startswith(
        "torchtitan_workspace:"
    ):
        raise CampaignError(
            "experiment lock requires a pinned torchtitan_workspace fbpkg ID"
        )
    return lock


def experiment_lock_digest(path: Path = LOCK_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
