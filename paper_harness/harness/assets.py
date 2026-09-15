from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path
from typing import Any

from .campaign import CampaignError
from .sources import manifest_digest, tree_manifest

ASSET_LOCK_PATH = Path(__file__).resolve().parents[1] / "asset_lock.toml"


def load_asset_lock(path: Path = ASSET_LOCK_PATH) -> dict[str, Any]:
    with path.open("rb") as stream:
        lock = tomllib.load(stream)
    if lock.get("schema_version") != 1:
        raise CampaignError("asset_lock.toml schema_version must be 1")
    if not str(lock.get("workspace_uri", "")).startswith("ws://"):
        raise CampaignError("asset lock requires an exact ws:// workspace URI")
    root = Path(str(lock.get("root", "")))
    if root.is_absolute() or not root.parts or ".." in root.parts:
        raise CampaignError("asset lock root must be a safe workspace-relative path")
    assets = lock.get("assets")
    if not isinstance(assets, dict) or not assets:
        raise CampaignError("asset lock requires at least one asset")
    for name, spec in assets.items():
        if not isinstance(spec, dict) or not spec.get("relative_path"):
            raise CampaignError(f"asset {name!r} requires relative_path")
        relative = Path(str(spec["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise CampaignError(f"asset {name!r} path must be workspace-relative")
        if not spec.get("tree_sha256") and not spec.get("file_sha256"):
            raise CampaignError(f"asset {name!r} requires a content digest")
    return lock


def asset_lock_digest(path: Path = ASSET_LOCK_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_assets(
    mount_root: Path,
    required: list[str],
    *,
    path: Path = ASSET_LOCK_PATH,
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    lock = load_asset_lock(path)
    unknown = sorted(set(required) - set(lock["assets"]))
    if unknown:
        raise CampaignError(f"asset lock has no entries for {unknown}")
    asset_root = mount_root.resolve() / lock["root"]
    roots: dict[str, Path] = {}
    evidence: dict[str, dict[str, Any]] = {}
    for name in sorted(set(required)):
        spec = lock["assets"][name]
        resolved = asset_root / spec["relative_path"]
        if not resolved.exists():
            raise CampaignError(f"locked asset {name!r} does not exist: {resolved}")
        if resolved.is_file():
            digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
            if digest != spec.get("file_sha256"):
                raise CampaignError(f"locked asset {name!r} file digest differs")
            observed = {"file_count": 1, "file_sha256": digest}
        else:
            manifest = tree_manifest(resolved)
            digest = manifest_digest(manifest)
            if digest != spec.get("tree_sha256"):
                raise CampaignError(f"locked asset {name!r} tree digest differs")
            if len(manifest) != int(spec["file_count"]):
                raise CampaignError(f"locked asset {name!r} file count differs")
            observed = {"file_count": len(manifest), "tree_sha256": digest}
        roots[name] = resolved
        evidence[name] = {"root": str(resolved), **observed}
    return roots, evidence
