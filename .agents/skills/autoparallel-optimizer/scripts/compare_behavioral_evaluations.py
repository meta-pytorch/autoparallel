from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence


def compare_campaigns(left_root: Path, right_root: Path) -> dict[str, Any]:
    left_config = _read_json(left_root / "evaluation-config.json")
    right_config = _read_json(right_root / "evaluation-config.json")
    left_results = _read_json(left_root / "results.json")
    right_results = _read_json(right_root / "results.json")

    protocol_fields = (
        "model",
        "reasoning_effort",
        "repeats",
        "baseline_enabled",
        "grading_enabled",
        "pairwise_grading_enabled",
        "pairwise_ordering",
        "codex_command",
        "timeout_seconds",
        "case_ids",
        "cases_sha256",
        "fixtures_sha256",
        "rubric_sha256",
        "evaluator_sha256",
    )
    compatibility = {
        field: field in left_config
        and field in right_config
        and left_config[field] == right_config[field]
        for field in protocol_fields
    }
    compatibility["skill_sha256"] = (
        "skill_sha256" in left_config
        and "skill_sha256" in right_config
        and left_config["skill_sha256"] == right_config["skill_sha256"]
    )
    scored = _compare_scored_results(left_results, right_results)
    return {
        "schema_version": 1,
        "left": {
            "path": str(left_root.resolve()),
            "source_revision": left_config.get("source_revision"),
            "tracked_source_dirty": left_config.get("tracked_source_dirty"),
        },
        "right": {
            "path": str(right_root.resolve()),
            "source_revision": right_config.get("source_revision"),
            "tracked_source_dirty": right_config.get("tracked_source_dirty"),
        },
        "compatibility": compatibility,
        "scores_are_directly_comparable": all(
            compatibility[field] for field in protocol_fields
        ),
        **scored,
        "blind_pairwise": {
            "left": _pairwise_headline(left_root),
            "right": _pairwise_headline(right_root),
        },
    }


def _compare_scored_results(
    left_results: Sequence[Mapping[str, Any]],
    right_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    left = _indexed_scores(left_results)
    right = _indexed_scores(right_results)
    shared_keys = sorted(set(left) & set(right))
    variants = {}
    for variant in ("skill", "baseline"):
        keys = [key for key in shared_keys if key[1] == variant]
        if not keys:
            continue
        case_changes: dict[str, list[float]] = {}
        for key in keys:
            case_changes.setdefault(key[0], []).append(
                right[key]["weighted_score"] - left[key]["weighted_score"]
            )
        means = {
            case_id: mean(values) for case_id, values in sorted(case_changes.items())
        }
        variants[variant] = {
            "shared_run_count": len(keys),
            "shared_case_count": len(means),
            "mean_case_score_change": mean(means.values()),
            "case_bootstrap_95": _bootstrap_mean_interval(list(means.values())),
            "case_changes": means,
        }

    left_effect = _paired_case_effect(left)
    right_effect = _paired_case_effect(right)
    shared_cases = sorted(set(left_effect) & set(right_effect))
    effect_changes = {
        case_id: right_effect[case_id] - left_effect[case_id]
        for case_id in shared_cases
    }
    return {
        "shared_scored_run_count": len(shared_keys),
        "variant_changes": variants,
        "paired_effect": {
            "shared_case_count": len(shared_cases),
            "left_mean_delta": (
                mean(left_effect[case_id] for case_id in shared_cases)
                if shared_cases
                else None
            ),
            "right_mean_delta": (
                mean(right_effect[case_id] for case_id in shared_cases)
                if shared_cases
                else None
            ),
            "mean_delta_of_deltas": (
                mean(effect_changes.values()) if effect_changes else None
            ),
            "case_bootstrap_95": _bootstrap_mean_interval(
                list(effect_changes.values())
            ),
            "case_changes": effect_changes,
        },
    }


def _indexed_scores(
    results: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, int], Mapping[str, Any]]:
    return {
        (item["case_id"], item["variant"], item["repeat"]): item["score"]
        for item in results
        if "score" in item
    }


def _paired_case_effect(
    scores: Mapping[tuple[str, str, int], Mapping[str, Any]],
) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for case_id, variant, repeat in scores:
        if variant != "skill":
            continue
        baseline_key = (case_id, "baseline", repeat)
        if baseline_key not in scores:
            continue
        values.setdefault(case_id, []).append(
            scores[(case_id, "skill", repeat)]["weighted_score"]
            - scores[baseline_key]["weighted_score"]
        )
    return {case_id: mean(items) for case_id, items in sorted(values.items())}


def _bootstrap_mean_interval(
    values: Sequence[float], *, samples: int = 20_000, seed: int = 0
) -> dict[str, Any] | None:
    if not values:
        return None
    rng = random.Random(seed)
    means = sorted(mean(rng.choice(values) for _ in values) for _ in range(samples))
    return {
        "lower": means[int(0.025 * samples)],
        "upper": means[int(0.975 * samples)],
        "samples": samples,
        "seed": seed,
        "unit": "case",
    }


def _pairwise_headline(root: Path) -> dict[str, Any] | None:
    path = root / "summary.json"
    if not path.is_file():
        return None
    value = _read_json(path).get("pairwise_comparison")
    if not isinstance(value, Mapping):
        return None
    return {key: item for key, item in value.items() if key != "cases"}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = compare_campaigns(args.left, args.right)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
