# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from modeled_baselines import (
    autoparallel_estimate,
    calibrate_relative_baseline,
    load_baseline_cost_calibration,
)
from parallelization_policy import classify_modeled_competitiveness

SCHEMA_VERSION = 1


def evaluate_corpus(
    report_root: Path,
    bundle_paths: Sequence[str | Path],
    *,
    holdout_fraction: float = 0.25,
    split_seed: str = "autoparallel-replay-v1",
    registration_modules: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    if not 0.0 <= holdout_fraction <= 1.0:
        raise ValueError("holdout_fraction must be between zero and one")
    registrations = registration_modules or {}
    cases = []
    for bundle_path in bundle_paths:
        bundle = Path(bundle_path)
        if not bundle.is_absolute():
            bundle = report_root / bundle
        report = _read_json(bundle / "report.json")
        campaign = report.get("request", {}).get("campaign", {})
        case_id = str(campaign.get("case_id") or report["run_id"])
        for module in registrations.get(case_id, ()):
            importlib.import_module(module)
        split = _split_for_case(case_id, split_seed, holdout_fraction)
        try:
            result = _evaluate_bundle(bundle, report)
        except Exception as error:
            result = {
                "status": "failed",
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            }
        cases.append(
            {
                "case_id": case_id,
                "run_id": report["run_id"],
                "bundle": str(bundle.relative_to(report_root)),
                "split": split,
                **result,
            }
        )

    status_counts = Counter(case["status"] for case in cases)
    split_summary = {}
    for split in ("training", "heldout"):
        selected = [case for case in cases if case["split"] == split]
        split_summary[split] = {
            "case_count": len(selected),
            "status_counts": dict(
                sorted(Counter(case["status"] for case in selected).items())
            ),
            "regression_count": sum(bool(case.get("regression")) for case in selected),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "optimizer-replay-corpus-evaluation",
        "evaluator": _source_state(Path(__file__).resolve().parents[4]),
        "report_root": str(report_root.resolve()),
        "split": {
            "seed": split_seed,
            "holdout_fraction": holdout_fraction,
            "assignment": "sha256(case-id-and-seed)",
        },
        "summary": {
            "case_count": len(cases),
            "status_counts": dict(sorted(status_counts.items())),
            "regression_count": sum(bool(case.get("regression")) for case in cases),
            "splits": split_summary,
        },
        "cases": cases,
        "limitations": [
            "replay validates strategy generation, costs, constraints, and ILP solving only",
            "raw objectives from different revisions are not compared as performance evidence",
            "recorded placements and new placements are rescored under the current evaluator",
            "modeled DDP/FSDP templates are retained as screening evidence, not target measurements",
        ],
    }


def bundles_from_feedback(
    catalog: Mapping[str, Any],
    queue: Mapping[str, Any] | None = None,
    *,
    candidate_ids: Sequence[str] = (),
    campaign_id: str | None = None,
    limit: int | None = None,
) -> list[str]:
    runs_by_bundle = {
        str(run["bundle"]): run for run in catalog.get("runs", []) if _has_replay(run)
    }
    selected: list[str] = []
    if queue is not None and (candidate_ids or campaign_id is None):
        requested = set(candidate_ids)
        for candidate in queue.get("candidates", []):
            if requested and candidate.get("id") not in requested:
                continue
            representatives = candidate.get("replayable_representative_bundles", [])
            if not representatives:
                representatives = candidate.get("representative_bundles", [])
            selected.extend(
                str(bundle)
                for bundle in representatives
                if str(bundle) in runs_by_bundle
            )
    if not selected:
        selected = list(runs_by_bundle)
    if campaign_id is not None:
        selected = [
            bundle
            for bundle in selected
            if runs_by_bundle[bundle].get("campaign", {}).get("id") == campaign_id
        ]
    selected = list(dict.fromkeys(selected))
    latest_by_case: dict[str, str] = {}
    for bundle in selected:
        run = runs_by_bundle[bundle]
        case_id = str(run.get("campaign", {}).get("case_id") or run.get("run_id"))
        previous = latest_by_case.get(case_id)
        if previous is None or str(
            run.get("completed_at") or run.get("created_at") or ""
        ) > str(
            runs_by_bundle[previous].get("completed_at")
            or runs_by_bundle[previous].get("created_at")
            or ""
        ):
            latest_by_case[case_id] = bundle
    selected = list(latest_by_case.values())
    selected.sort(
        key=lambda bundle: (
            str(runs_by_bundle[bundle].get("campaign", {}).get("case_id", "")),
            bundle,
        )
    )
    return selected[:limit] if limit is not None else selected


def _evaluate_bundle(bundle: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    from torch.distributed.tensor._op_schema import _pretty_print_spec

    from autoparallel.optimize_sharding import ShardingOptimizer

    replay_path = _artifact_path(bundle, report, "optimizer_replay")
    placements_path = _artifact_path(bundle, report, "placements")
    optimizer_path = _artifact_path(bundle, report, "optimizer")
    source_plan = _read_json(optimizer_path)

    started = time.perf_counter()
    optimizer = ShardingOptimizer.load_replay(replay_path)
    load_seconds = time.perf_counter() - started

    recorded_solution = optimizer.load_placements(placements_path)
    recorded_common = optimizer._compute_solution_cost(
        optimizer._to_concrete_solution(recorded_solution)
    )
    recorded_storage = optimizer._get_parameter_storage_summary(
        optimizer._to_concrete_solution(recorded_solution)
    )

    started = time.perf_counter()
    candidate_solution = optimizer.get_solution(verbose=False)
    solve_seconds = time.perf_counter() - started
    candidate_plan = optimizer.get_json()
    candidate_common = optimizer._compute_solution_cost(
        optimizer._to_concrete_solution(candidate_solution)
    )
    candidate_storage = optimizer._get_parameter_storage_summary(
        optimizer._to_concrete_solution(candidate_solution)
    )

    recorded_signature = _solution_signature(recorded_solution, _pretty_print_spec)
    candidate_signature = _solution_signature(candidate_solution, _pretty_print_spec)
    changed_nodes = sum(
        recorded_signature.get(name) != candidate_signature.get(name)
        for name in set(recorded_signature) | set(candidate_signature)
    )
    tolerance = max(1e-6, abs(recorded_common["total"]) * 1e-9)
    common_score_regressed = candidate_common["total"] > (
        recorded_common["total"] + tolerance
    )

    modeled_comparison = _modeled_comparison(bundle, report, candidate_plan)
    regression = common_score_regressed or modeled_comparison.get(
        "point_estimate_regressed", False
    )
    return {
        "status": "passed",
        "regression": regression,
        "timing": {
            "load_seconds": load_seconds,
            "solve_seconds": solve_seconds,
        },
        "source": {
            "revision": report.get("provenance", {})
            .get("reproduction", {})
            .get("source", {})
            .get("git_revision"),
            "recorded_search_objective": source_plan.get("summary", {}).get(
                "search_objective"
            ),
            "placement_signature": _signature_digest(recorded_signature),
            "parameter_storage": recorded_storage,
        },
        "candidate": {
            "search_objective": candidate_plan["summary"]["search_objective"],
            "placement_signature": _signature_digest(candidate_signature),
            "parameter_storage": candidate_storage,
            "decision_variable_count": len(optimizer.decision_vars),
            "cluster_link_count": len(optimizer.cluster_links),
        },
        "common_evaluator": {
            "source_placement_cost": recorded_common,
            "candidate_placement_cost": candidate_common,
            "improvement_fraction": (
                (recorded_common["total"] - candidate_common["total"])
                / recorded_common["total"]
                if recorded_common["total"]
                else 0.0
            ),
            "regressed": common_score_regressed,
        },
        "placement_changes": {
            "changed_node_count": changed_nodes,
            "source_node_count": len(recorded_signature),
            "candidate_node_count": len(candidate_signature),
        },
        "modeled_baseline_screen": modeled_comparison,
    }


def _modeled_comparison(
    bundle: Path, report: Mapping[str, Any], candidate_plan: Mapping[str, Any]
) -> dict[str, Any]:
    artifacts = [
        artifact
        for artifact in report.get("artifacts", [])
        if artifact.get("role") == "modeled_comparison"
    ]
    if not artifacts:
        return {"available": False, "point_estimate_regressed": False}
    comparison = _read_json(bundle / artifacts[-1]["path"])
    baselines_by_name = comparison.get("baselines", {})
    raw_baselines: list[Mapping[str, Any]] = [
        {"feasibility": "feasible", **baseline}
        for baseline in baselines_by_name.values()
        if isinstance(baseline, Mapping) and baseline.get("latency") is not None
    ]
    mesh_shape = tuple(report.get("request", {}).get("mesh", {}).get("shape", (1,)))
    candidate = autoparallel_estimate(candidate_plan, mesh_shape=mesh_shape)
    calibration_artifacts = [
        artifact
        for artifact in report.get("artifacts", [])
        if artifact.get("name") == "baseline-cost-calibration"
    ]
    if calibration_artifacts:
        calibration = load_baseline_cost_calibration(
            bundle / calibration_artifacts[-1]["path"]
        )
        parameter_bytes = int(
            candidate_plan["summary"]["parameter_storage"]["global_bytes"]
        )
        baselines: list[Mapping[str, Any]] = [
            calibrate_relative_baseline(
                candidate["latency"], baseline, parameter_bytes, calibration
            )
            for baseline in raw_baselines
        ]
        calibration_kind = calibration["kind"]
    else:
        baselines = raw_baselines
        calibration_kind = None
    verdict = classify_modeled_competitiveness(candidate["latency"], baselines)
    old_expected = comparison.get("autoparallel", {}).get("latency", {}).get("expected")
    new_expected = candidate["latency"]["expected"]
    return {
        "available": True,
        "evidence": "modeled-not-measured",
        "recorded_autoparallel_expected_us": old_expected,
        "candidate_autoparallel_expected_us": new_expected,
        "point_estimate_regressed": (
            old_expected is not None and new_expected > old_expected * (1.0 + 1e-9)
        ),
        "verdict": verdict,
        "relative_calibration": calibration_kind,
        "baseline_templates_reused": [
            {
                "strategy": baseline.get("strategy"),
                "template_id": baseline.get("template_id"),
                "template_version": baseline.get("template_version"),
            }
            for baseline in baselines
        ],
    }


def _artifact_path(bundle: Path, report: Mapping[str, Any], role: str) -> Path:
    matches = [
        artifact
        for artifact in report.get("artifacts", [])
        if artifact.get("role") == role
    ]
    if not matches:
        raise RuntimeError(f"bundle has no {role!r} artifact")
    path = bundle / matches[-1]["path"]
    if not path.is_file():
        raise RuntimeError(f"missing {role!r} artifact: {path}")
    return path


def _solution_signature(solution, pretty_print) -> dict[str, str]:
    return {
        node.name: pretty_print(strategy.output_specs)
        for node, strategy in solution.items()
    }


def _signature_digest(signature: Mapping[str, str]) -> str:
    payload = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(payload.encode()).hexdigest()}"


def _split_for_case(case_id: str, seed: str, holdout_fraction: float) -> str:
    digest = hashlib.sha256(f"{seed}:{case_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return "heldout" if value < holdout_fraction else "training"


def _has_replay(run: Mapping[str, Any]) -> bool:
    replay = run.get("optimizer_replay")
    if isinstance(replay, Mapping):
        return replay.get("available") is True
    return "optimizer_replay" in run.get("artifact_roles", [])


def _source_state(repo_root: Path) -> dict[str, Any]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=repo_root, text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        revision = None
        dirty = None
    return {"git_revision": revision, "tracked_or_untracked_dirty": dirty}


def _registration_modules(values: Sequence[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("registration modules must use CASE=MODULE")
        case_id, module = value.split("=", 1)
        result.setdefault(case_id, []).append(module)
    return result


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report_root", type=Path)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--queue", type=Path)
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--campaign")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument("--split-seed", default="autoparallel-replay-v1")
    parser.add_argument("--register", action="append", default=[])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    catalog_path = args.catalog or args.report_root / "catalog-v1.json"
    queue_path = args.queue or args.report_root / "improvement-queue-v1.json"
    catalog = _read_json(catalog_path)
    queue = _read_json(queue_path) if queue_path.is_file() else None
    bundles = bundles_from_feedback(
        catalog,
        queue,
        candidate_ids=args.candidate,
        campaign_id=args.campaign,
        limit=args.limit,
    )
    if not bundles:
        raise SystemExit("no replayable bundles matched the selection")
    result = evaluate_corpus(
        args.report_root.resolve(),
        bundles,
        holdout_fraction=args.holdout_fraction,
        split_seed=args.split_seed,
        registration_modules=_registration_modules(args.register),
    )
    _write_json_atomic(args.output, result)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
