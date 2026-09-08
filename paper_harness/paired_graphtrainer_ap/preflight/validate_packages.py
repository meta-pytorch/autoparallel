from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path


def manifest(root: Path) -> dict[str, str]:
    result = {}
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [
            name
            for name in dirnames
            if name not in {".fbpkg.tmp", ".git", "__pycache__"}
        ]
        for filename in filenames:
            path = Path(directory) / filename
            relative = path.relative_to(root).as_posix()
            if relative == "METADATA" or relative.endswith(".CHECKSUMS"):
                continue
            value = path.resolve().read_bytes() if path.is_symlink() else path.read_bytes()
            result[relative] = hashlib.sha256(value).hexdigest()
    return result


def compare(source: Path, package: Path) -> dict[str, object]:
    source_manifest = manifest(source)
    package_manifest = manifest(package)
    if source_manifest != package_manifest:
        raise AssertionError(
            f"Package mismatch for {source}: "
            f"source_only={sorted(source_manifest.keys() - package_manifest.keys())}, "
            f"package_only={sorted(package_manifest.keys() - source_manifest.keys())}, "
            "content_mismatch="
            f"{sorted(path for path in source_manifest.keys() & package_manifest.keys() if source_manifest[path] != package_manifest[path])}"
        )
    return {"files": len(source_manifest), "match": True}


def executable(path: Path) -> bool:
    return bool(path.stat().st_mode & stat.S_IXUSR)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--attempt-root", type=Path, required=True)
    parser.add_argument("--package-kind", choices=("dryrun", "submitted"), required=True)
    parser.add_argument("--c4-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    task_root = args.task_root.resolve()
    attempt_root = args.attempt_root.resolve()
    workspace = attempt_root / f"packages/{args.package_kind}_workspace"
    additional = attempt_root / f"packages/{args.package_kind}_additional"
    comparisons = {
        "workspace": compare(task_root / "launcher", workspace),
        "runner": compare(task_root / "runner", additional / "runner"),
        "torchtitan": compare(task_root / "source/torchtitan", additional / "torchtitan"),
        "autoparallel": compare(
            task_root / "source/autoparallel", additional / "autoparallel"
        ),
        "offline_c4": compare(args.c4_cache.resolve(), additional / "c4_hf_cache"),
    }
    workspace_wrapper = workspace / "run_rank.sh"
    full_runner = additional / "runner/run_rank.sh"
    if not executable(workspace_wrapper) or not executable(full_runner):
        raise AssertionError("packaged wrapper or runner is not executable")
    expected_target = "/packages/torchtitan_additional_packages/runner/run_rank.sh"
    if expected_target not in workspace_wrapper.read_text():
        raise AssertionError("workspace wrapper does not invoke the packaged runner")

    report = {
        "valid": True,
        "comparisons": comparisons,
        "workspace_wrapper": str(workspace_wrapper),
        "full_runner": str(full_runner),
        "workspace_wrapper_executable": True,
        "full_runner_executable": True,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
