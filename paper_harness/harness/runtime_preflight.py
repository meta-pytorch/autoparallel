from __future__ import annotations

import importlib
import json
import os
import platform
import sys
from pathlib import Path

import torch

from .campaign import write_json
from .sources import manifest_digest, tree_manifest


def _module_path(name: str) -> Path:
    module = importlib.import_module(name)
    path = getattr(module, "__file__", None)
    if path is None:
        raise RuntimeError(f"module {name!r} has no filesystem path")
    return Path(path).resolve()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root.resolve())
        return True
    except ValueError:
        return False


def run(payload: Path, run_root: Path) -> dict:
    source_lock = json.loads((payload / "campaign/source_lock.json").read_text())
    source_roots = {
        "torchtitan": payload / "torchtitan",
        "autoparallel": payload / "autoparallel",
    }
    source_checks = {}
    for name, root in source_roots.items():
        digest = manifest_digest(tree_manifest(root))
        expected = source_lock[name]["tree_sha256"]
        if digest != expected:
            raise RuntimeError(
                f"packaged {name} tree mismatch: expected {expected}, got {digest}"
            )
        source_checks[name] = {"tree_sha256": digest, "status": "passed"}

    module_paths = {
        "torch": _module_path("torch"),
        "torchtitan": _module_path("torchtitan"),
        "autoparallel": _module_path("autoparallel"),
        "structured_logger": _module_path(
            "torchtitan.observability.structured_logger.jsonl_handler"
        ),
    }
    if not _inside(module_paths["torchtitan"], source_roots["torchtitan"]):
        raise RuntimeError(f"unexpected TorchTitan import: {module_paths['torchtitan']}")
    if not _inside(module_paths["autoparallel"], source_roots["autoparallel"]):
        raise RuntimeError(f"unexpected AutoParallel import: {module_paths['autoparallel']}")
    report = {
        "status": "passed",
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_git_version": getattr(torch.version, "git_version", None),
        "cuda_version": torch.version.cuda,
        "nccl_version": torch.cuda.nccl.version() if torch.cuda.is_available() else None,
        "module_paths": {name: str(path) for name, path in module_paths.items()},
        "source_checks": source_checks,
        "structured_logger_handlers": os.environ.get("TITAN_STRUCT_LOGGER_HANDLERS"),
    }
    write_json(run_root / "runtime/preflight/report.json", report)
    return report
