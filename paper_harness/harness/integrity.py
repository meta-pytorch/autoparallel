from __future__ import annotations

import hashlib
from pathlib import Path

from .campaign import CampaignError


MANIFEST_NAME = "HARNESS_CORE.sha256"
CORE_PATHS = (
    "experiment_lock.toml",
    "harness",
    "launcher",
    "workloads",
    "scripts/measurements",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def core_manifest(root: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for name in CORE_PATHS:
        path = root / name
        candidates = [path] if path.is_file() else sorted(path.rglob("*"))
        for candidate in candidates:
            if not candidate.is_file() or "__pycache__" in candidate.parts:
                continue
            relative = candidate.relative_to(root).as_posix()
            files[relative] = _sha256(candidate)
    return files


def read_core_manifest(root: Path) -> dict[str, str]:
    path = root / MANIFEST_NAME
    if not path.is_file():
        raise CampaignError(f"missing frozen harness manifest: {path}")
    result: dict[str, str] = {}
    for line in path.read_text().splitlines():
        digest, separator, relative = line.partition("  ")
        if (
            not separator
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not relative
            or relative in result
        ):
            raise CampaignError(f"invalid frozen harness manifest line: {line!r}")
        result[relative] = digest
    return result


def validate_harness_integrity(root: Path | None = None) -> dict[str, object]:
    root = (root or Path(__file__).resolve().parents[1]).resolve()
    expected = read_core_manifest(root)
    actual = core_manifest(root)
    if actual != expected:
        changed = sorted(
            path
            for path in expected.keys() | actual.keys()
            if expected.get(path) != actual.get(path)
        )
        raise CampaignError(
            "frozen harness core changed; explicitly version and regenerate "
            f"{MANIFEST_NAME}: {changed}"
        )
    return {"status": "passed", "file_count": len(actual)}
