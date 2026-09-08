from __future__ import annotations

import re
from typing import Any

from .campaign import CampaignError


IGNORED_PATHS = {
    "dump_folder",
    "debug.save_config_file",
}
ADDRESS_RE = re.compile(r"0x[0-9a-fA-F]+")


def normalized(value: object) -> object:
    """Remove process-local details without hiding configuration differences."""
    if isinstance(value, dict):
        return {key: normalized(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalized(item) for item in value]
    if isinstance(value, str):
        return ADDRESS_RE.sub("0xADDR", value)
    return value


def differences(
    left: object,
    right: object,
    path: tuple[str, ...] = (),
    ignored_paths: set[str] | None = None,
) -> list[dict[str, Any]]:
    ignored_paths = IGNORED_PATHS if ignored_paths is None else ignored_paths
    dotted = ".".join(path)
    if dotted in ignored_paths:
        return []
    if isinstance(left, dict) and isinstance(right, dict):
        result: list[dict[str, Any]] = []
        for key in sorted(set(left) | set(right)):
            result.extend(
                differences(
                    left.get(key, "<missing>"),
                    right.get(key, "<missing>"),
                    (*path, key),
                    ignored_paths,
                )
            )
        return result
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [{"path": dotted, "left": left, "right": right}]
        result = []
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            result.extend(
                differences(left_item, right_item, (*path, str(index)), ignored_paths)
            )
        return result
    return [] if left == right else [{"path": dotted, "left": left, "right": right}]


def _allowed(path: str, patterns: list[str]) -> bool:
    for pattern in patterns:
        if pattern.endswith(".*") and (
            path == pattern[:-2] or path.startswith(pattern[:-1])
        ):
            return True
        if path == pattern:
            return True
    return False


def validate_pair(
    baseline_name: str,
    baseline: dict[str, Any],
    treatment_name: str,
    treatment: dict[str, Any],
    allowed_paths: list[str],
    *,
    ignored_paths: set[str] | None = None,
) -> dict[str, Any]:
    observed = differences(
        normalized(baseline),
        normalized(treatment),
        ignored_paths=ignored_paths,
    )
    unexpected = [row for row in observed if not _allowed(row["path"], allowed_paths)]
    if unexpected:
        raise CampaignError(
            f"{baseline_name}/{treatment_name} differ outside declared variables: "
            f"{unexpected}"
        )
    return {
        "baseline": baseline_name,
        "treatment": treatment_name,
        "allowed_paths": allowed_paths,
        "observed_differences": observed,
        "status": "passed",
    }
