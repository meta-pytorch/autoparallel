from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from .campaign import CampaignError, write_json


IGNORED_DIRECTORIES = {
    ".git",
    ".fbpkg.tmp",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
}


def _git(root: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and result.returncode:
        raise CampaignError(f"git {' '.join(args)} failed for {root}: {result.stderr.strip()}")
    return result.stdout.strip()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_symlink():
        digest.update(path.resolve().read_bytes())
    else:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def tree_manifest(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(name for name in dirnames if name not in IGNORED_DIRECTORIES)
        for filename in sorted(filenames):
            if filename in IGNORED_DIRECTORIES:
                continue
            path = Path(directory) / filename
            result[path.relative_to(root).as_posix()] = _hash_file(path)
    return result


def manifest_digest(manifest: dict[str, str]) -> str:
    payload = json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _canonical_remote(value: str) -> str:
    remote = value.strip().removesuffix(".git")
    match = re.match(r"(?:https://|ssh://)?(?:[^/@]+@)?github\.com[/:](.+)", remote)
    return f"github.com/{match.group(1)}" if match else str(Path(remote).resolve())


def _remote_chain(root: Path, *, limit: int = 8) -> list[str]:
    chain = []
    current = root
    for _ in range(limit):
        remote = _git(current, "remote", "get-url", "origin", check=False)
        if not remote or remote in chain:
            break
        chain.append(remote)
        candidate = Path(remote).expanduser()
        if not candidate.is_absolute():
            candidate = (current / candidate).resolve()
        if not candidate.exists():
            break
        current = candidate
    return chain


def _configured_remotes(root: Path) -> list[str]:
    urls: list[str] = []
    for name in _git(root, "remote", check=False).splitlines():
        urls.extend(
            line
            for line in _git(root, "remote", "get-url", "--all", name).splitlines()
            if line and line not in urls
        )
    for url in _remote_chain(root):
        if url not in urls:
            urls.append(url)
    return urls


def inspect_source(
    name: str,
    root: Path,
    spec: dict[str, Any],
    *,
    evidence_dir: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    if not root.is_dir():
        raise CampaignError(f"source root does not exist: {root}")
    head = _git(root, "rev-parse", "HEAD")
    expected = str(spec["commit"])
    if head != expected:
        raise CampaignError(f"{name} HEAD {head} does not match campaign pin {expected}")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    dirty = bool(status)
    policy = spec.get("dirty_policy", "forbid")
    if dirty and policy == "forbid":
        raise CampaignError(f"{name} checkout is dirty but dirty_policy='forbid':\n{status}")
    manifest = tree_manifest(root)
    remotes = _configured_remotes(root)
    expected_remote = str(spec["remote"])
    if _canonical_remote(expected_remote) not in {
        _canonical_remote(remote) for remote in remotes
    }:
        raise CampaignError(
            f"{name} remote chain {remotes!r} does not contain expected remote "
            f"{expected_remote!r}"
        )
    upstream_ref = None
    for candidate in ("origin/main", "origin/master"):
        if _git(root, "rev-parse", "--verify", candidate, check=False):
            upstream_ref = candidate
            break
    ahead = behind = None
    upstream_diff = None
    if upstream_ref is not None:
        counts = _git(root, "rev-list", "--left-right", "--count", f"{upstream_ref}...HEAD")
        behind, ahead = (int(value) for value in counts.split())
        upstream_diff = _git(root, "diff", "--stat", f"{upstream_ref}...HEAD")
    record: dict[str, Any] = {
        "name": name,
        "root": str(root),
        "remote": remotes[0] if remotes else None,
        "remote_chain": remotes,
        "expected_remote": expected_remote,
        "head": head,
        "branch": _git(root, "branch", "--show-current") or "<detached>",
        "commit_time": _git(root, "show", "-s", "--format=%cI", "HEAD"),
        "dirty": dirty,
        "dirty_policy": policy,
        "status": status.splitlines(),
        "tree_sha256": manifest_digest(manifest),
        "file_count": len(manifest),
        "upstream_ref": upstream_ref,
        "commits_ahead_of_upstream": ahead,
        "commits_behind_upstream": behind,
        "upstream_diff_stat": upstream_diff,
    }
    if evidence_dir is not None:
        source_dir = evidence_dir / name
        source_dir.mkdir(parents=True, exist_ok=True)
        write_json(source_dir / "tree_manifest.json", manifest)
        if dirty:
            (source_dir / "tracked.diff").write_text(
                _git(root, "diff", "--binary", "HEAD") + "\n"
            )
            untracked = _git(
                root, "ls-files", "--others", "--exclude-standard"
            ).splitlines()
            write_json(
                source_dir / "untracked.json",
                {path: manifest[path] for path in untracked},
            )
    return record
