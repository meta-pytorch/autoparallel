from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


ARMS = ("graph_manual", "graph_autoparallel")
TRACE_ARMS = ARMS
PHASES = ("correctness", "steady", "profile")
_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+(?=>)")


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set")
    return value


def int_env(name: str, *, minimum: int = 1) -> int:
    value = int(required_env(name))
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    if value not in {"0", "1"}:
        raise ValueError(f"{name} must be 0 or 1, got {value!r}")
    return value == "1"


def canonical_json_hash(value: Any) -> str:
    encoded = json.dumps(
        stable_json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def stable_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): stable_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [stable_json_value(item) for item in value]
    if isinstance(value, str):
        return _ADDRESS.sub("", value)
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if any(part in {"__pycache__", ".pytest_cache"} for part in path.parts):
            continue
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(file_sha256(path).encode())
        digest.update(b"\0")
    return digest.hexdigest()
