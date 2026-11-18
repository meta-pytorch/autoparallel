# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_SCRIPTS = REPO_ROOT / ".agents/skills/autoparallel-optimizer/scripts"
sys.path[:0] = [str(REPO_ROOT), str(SKILL_SCRIPTS)]

from parallelization_policy import (  # noqa: E402
    default_acceptance_criteria,
    default_attempt_budget,
)
from report_session import _default_provenance, _request_with_campaign  # noqa: E402
from run_report import AutoParallelRunReport, resolve_report_config  # noqa: E402

from experiments.reported_hf_cost_model_baselines import (  # noqa: E402
    CALIBRATION_CAMPAIGN_ID,
    CALIBRATION_CASES,
)

MODELED_DAY = "2026-10-12"
MEASURED_DAY = "2026-10-07"
MEASURED_RUNS = {
    "gpt2-xl-seq1024": {
        "autoparallel": "20261007T193239Z-13def56d4e4d",
        "ddp": "20261007T194628Z-81832ee8bc65",
    },
    "t5-3b-seq512": {
        "autoparallel": "20261007T193559Z-fd36ff5a0760",
        "ddp": "20261007T194827Z-aa179c113be6",
        "fsdp": "20261007T195055Z-0b869962872d",
    },
    "llama3-8b-seq2048": {
        "autoparallel": "20261007T194143Z-45f7a1318b1e",
        "ddp": "20261007T195211Z-10cf81370f35",
        "fsdp": "20261007T195338Z-c3ede128b5bd",
    },
    "mistral-7b-seq2048": {
        "autoparallel": "20261007T195455Z-dbe147fafc4d",
        "fsdp": "20261007T201029Z-ef418b12dec7",
    },
    "deberta-v2-xxlarge-seq512": {
        "autoparallel": "20261007T200305Z-29fad5319dec",
        "ddp": "20261007T201214Z-00f9d0aa17cf",
        "fsdp": "20261007T201500Z-1664ff345645",
    },
    "qwen2-7b-seq2048": {
        "autoparallel": "20261007T210138Z-1db141519894",
        "ddp": "20261007T210549Z-6d0b647e11c7",
        "fsdp": "20261007T210446Z-6393b1b2b370",
    },
    "bert-large-seq512": {
        "autoparallel": "20261007T210715Z-7914b426d43b",
        "ddp": "20261007T210939Z-f1f8f403c1c9",
        "fsdp": "20261007T211103Z-3b25ecd9cffd",
    },
    "t5-11b-seq512": {
        "autoparallel": "20261007T211229Z-bce342515fdc",
        "ddp": "20261007T212335Z-3db2141b3f09",
        "fsdp": "20261007T212211Z-dac3b8149149",
    },
    "llama3.2-1b-seq2048": {
        "autoparallel": "20261007T213820Z-cbefc7c07f19",
        "ddp": "20261007T214014Z-34679828dce6",
        "fsdp": "20261007T214140Z-75e650c2e65f",
    },
    "llama3.2-3b-seq2048": {
        "autoparallel": "20261007T214244Z-fe555b0b1fef",
        "ddp": "20261007T214539Z-f26d429d7d5f",
        "fsdp": "20261007T214709Z-3827c8131213",
    },
    "t5-large-seq512": {
        "autoparallel": "20261007T214824Z-9b13afbd54c2",
        "ddp": "20261007T215400Z-6e6cd1f4c30a",
        "fsdp": "20261007T215556Z-5e96f752b7f1",
    },
    "bloom-7.1b-seq1024": {
        "autoparallel": "20261007T215716Z-a373f48a0f4f",
        "ddp": "20261007T220001Z-01b0dd5a46a2",
        "fsdp": "20261007T220131Z-f655635131ff",
    },
    "gpt2-medium-seq1024": {
        "autoparallel": "20261007T220714Z-4f2a0f0d1f5c",
        "ddp": "20261007T221020Z-c137b2a3a14a",
        "fsdp": "20261007T221133Z-bf974d9eef40",
    },
    "t5-base-seq512": {
        "autoparallel": "20261007T221219Z-bbcc47d7eac2",
        "ddp": "20261007T221521Z-0a51712ae429",
        "fsdp": "20261007T221701Z-926ba8a04d28",
    },
    "qwen2-1.5b-seq2048": {
        "autoparallel": "20261007T221802Z-5ac450c11770",
        "ddp": "20261007T222052Z-1600c19a75e5",
        "fsdp": "20261007T222214Z-16ab8c61c197",
    },
    "bloom-560m-seq1024": {
        "autoparallel": "20261007T222302Z-478e38dc816e",
        "ddp": "20261007T222526Z-62477e7186c1",
        "fsdp": "20261007T222640Z-cb1e26151e75",
    },
}


def load_json(path):
    with path.open() as file:
        return json.load(file)


def measured_run_ids():
    return {case: MEASURED_RUNS[case] for case in CALIBRATION_CASES}


def modeled_reports(profile_root):
    reports = {}
    for report_path in (profile_root / MODELED_DAY).glob("*/report.json"):
        report = load_json(report_path)
        campaign = report.get("request", {}).get("campaign", {})
        if (
            campaign.get("id") != CALIBRATION_CAMPAIGN_ID
            or report["status"] != "completed"
        ):
            continue
        case = campaign["case_id"]
        current = reports.get(case)
        if current is None or report["created_at"] > current["created_at"]:
            reports[case] = {"bundle": report_path.parent, **report}
    missing = set(CALIBRATION_CASES) - reports.keys()
    if missing:
        raise RuntimeError(f"missing modeled reports: {sorted(missing)}")
    return reports


def measured_latency(profile_root, run_id):
    report = load_json(profile_root / MEASURED_DAY / run_id / "report.json")
    target_step = report.get("validation", {}).get("target-step")
    if report["status"] != "completed" or target_step is None:
        return None
    return target_step["result"]["timing"]["median_us"]


def architecture_family(case):
    if case.startswith("t5-"):
        return "t5"
    if case.startswith("llama"):
        return "llama"
    if case.startswith("qwen"):
        return "qwen"
    if case.startswith("bloom"):
        return "bloom"
    if case.startswith("gpt2-"):
        return "gpt2"
    if case.startswith(("bert-", "deberta-")):
        return "encoder-only"
    return case.split("-", 1)[0]


def build_points(profile_root):
    cost_reports = modeled_reports(profile_root)
    measured_ids = measured_run_ids()
    points = {"ddp": [], "fsdp": []}
    exclusions = []
    for case in CALIBRATION_CASES:
        cost_report = cost_reports[case]
        comparison = load_json(
            cost_report["bundle"] / "artifacts" / "modeled-baseline-comparison.json"
        )
        actual_autoparallel = measured_latency(
            profile_root, measured_ids[case]["autoparallel"]
        )
        if actual_autoparallel is None:
            raise RuntimeError(f"missing measured AutoParallel target step for {case}")
        modeled_autoparallel = comparison["autoparallel"]["latency"]["expected"]
        for strategy in ("ddp", "fsdp"):
            run_id = measured_ids[case].get(strategy)
            if run_id is None:
                exclusions.append(
                    {"case": case, "strategy": strategy, "reason": "no_source_run"}
                )
                continue
            actual_baseline = measured_latency(profile_root, run_id)
            if actual_baseline is None:
                exclusions.append(
                    {
                        "case": case,
                        "strategy": strategy,
                        "reason": "source_run_has_no_completed_target_step",
                        "run_id": run_id,
                    }
                )
                continue
            modeled_baseline = comparison["baselines"][strategy]["latency"]["expected"]
            baseline = comparison["baselines"][strategy]
            point = {
                "case": case,
                "architecture_family": architecture_family(case),
                "modeled_run_id": cost_report["run_id"],
                "measured_run_ids": {
                    "autoparallel": measured_ids[case]["autoparallel"],
                    strategy: run_id,
                },
                "modeled_autoparallel_us": modeled_autoparallel,
                "modeled_baseline_us": modeled_baseline,
                "modeled_ratio": modeled_autoparallel / modeled_baseline,
                "actual_autoparallel_us": actual_autoparallel,
                "actual_baseline_us": actual_baseline,
                "actual_ratio": actual_autoparallel / actual_baseline,
                "parameter_bytes": comparison["autoparallel_proxy_diagnostics"][
                    "replicated_parameter_plan"
                ]["parameter_storage"]["global_bytes"],
                "baseline_collective_us": baseline["communication"][
                    "total_collective_us"
                ],
            }
            if strategy == "ddp":
                point["schedule_unit_count"] = baseline["communication"]["bucket_count"]
            else:
                point["schedule_unit_count"] = baseline["communication"][
                    "wrap_unit_count"
                ]
            points[strategy].append(point)
    return points, exclusions


def fit_line(xs, ys):
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    variance = sum((value - mean_x) ** 2 for value in xs)
    if variance == 0:
        return 0.0, mean_y
    slope = (
        sum((x_value - mean_x) * (y_value - mean_y) for x_value, y_value in zip(xs, ys))
        / variance
    )
    return slope, mean_y - slope * mean_x


def predict(method, training, point):
    x = math.log(point["modeled_ratio"])
    if method == "raw":
        return point["modeled_ratio"]
    if method == "median-log-offset":
        offset = statistics.median(
            math.log(item["actual_ratio"]) - math.log(item["modeled_ratio"])
            for item in training
        )
        return math.exp(x + offset)
    if method == "log-affine":
        slope, intercept = fit_line(
            [math.log(item["modeled_ratio"]) for item in training],
            [math.log(item["actual_ratio"]) for item in training],
        )
        return math.exp(slope * x + intercept)
    if method == "log-structural":
        features = np.asarray(
            [
                [
                    1.0,
                    math.log(item["modeled_ratio"]),
                    math.log(item["parameter_bytes"]),
                    math.log(item["schedule_unit_count"]),
                ]
                for item in training
            ]
        )
        targets = np.asarray([math.log(item["actual_ratio"]) for item in training])
        coefficients = np.linalg.lstsq(features, targets, rcond=None)[0]
        point_features = np.asarray(
            [
                1.0,
                math.log(point["modeled_ratio"]),
                math.log(point["parameter_bytes"]),
                math.log(point["schedule_unit_count"]),
            ]
        )
        return math.exp(float(point_features @ coefficients))
    raise ValueError(f"unknown calibration method: {method}")


def evaluate_method(points, method, *, holdout):
    predictions = []
    for index, point in enumerate(points):
        if holdout == "workload":
            training = points[:index] + points[index + 1 :]
        elif holdout == "architecture_family":
            training = [
                item
                for item in points
                if item["architecture_family"] != point["architecture_family"]
            ]
        else:
            raise ValueError(f"unknown holdout: {holdout}")
        predicted = predict(method, training, point)
        actual = point["actual_ratio"]
        predictions.append(
            {
                "case": point["case"],
                "predicted_ratio": predicted,
                "actual_ratio": actual,
                "absolute_relative_error": abs(predicted - actual) / actual,
                "absolute_log_error": abs(math.log(predicted / actual)),
                "winner_correct": (predicted <= 1.0) == (actual <= 1.0),
                "five_percent_competitiveness_correct": (predicted <= 1.05)
                == (actual <= 1.05),
            }
        )
    relative_errors = [item["absolute_relative_error"] for item in predictions]
    log_errors = [item["absolute_log_error"] for item in predictions]
    return {
        "method": method,
        "holdout": holdout,
        "point_count": len(points),
        "median_absolute_relative_error": statistics.median(relative_errors),
        "maximum_absolute_relative_error": max(relative_errors),
        "maximum_multiplicative_error": math.exp(max(log_errors)),
        "winner_accuracy": statistics.fmean(
            item["winner_correct"] for item in predictions
        ),
        "five_percent_competitiveness_accuracy": statistics.fmean(
            item["five_percent_competitiveness_correct"] for item in predictions
        ),
        "predictions": predictions,
    }


def calibrated_points(points, method, maximum_multiplicative_error):
    calibrated = []
    for point in points:
        predicted = predict(method, points, point)
        lower = predicted / maximum_multiplicative_error
        upper = predicted * maximum_multiplicative_error
        if upper <= 1.0:
            win_verdict = "autoparallel"
        elif lower > 1.0:
            win_verdict = "baseline"
        else:
            win_verdict = "uncertain"
        if upper <= 1.05:
            competitiveness = "modeled_competitive"
        elif lower > 1.05:
            competitiveness = "modeled_worse"
        else:
            competitiveness = "uncertain"
        calibrated.append(
            {
                **point,
                "calibrated_ratio": predicted,
                "calibrated_ratio_lower": lower,
                "calibrated_ratio_upper": upper,
                "winner_verdict": win_verdict,
                "five_percent_competitiveness_verdict": competitiveness,
            }
        )
    return calibrated


def analyze_strategy(strategy, points):
    evaluations = {
        method: evaluate_method(points, method, holdout="workload")
        for method in (
            "raw",
            "median-log-offset",
            "log-affine",
            "log-structural",
        )
    }
    family_holdout_evaluations = {
        method: evaluate_method(points, method, holdout="architecture_family")
        for method in (
            "raw",
            "median-log-offset",
            "log-affine",
            "log-structural",
        )
    }
    majority_winner_accuracy = max(
        statistics.fmean(point["actual_ratio"] <= 1.0 for point in points),
        statistics.fmean(point["actual_ratio"] > 1.0 for point in points),
    )
    eligible_methods = [
        method
        for method in ("median-log-offset", "log-affine", "log-structural")
        if evaluations[method]["winner_accuracy"] > majority_winner_accuracy
        and family_holdout_evaluations[method]["winner_accuracy"]
        > majority_winner_accuracy
    ]
    best_method = (
        min(
            eligible_methods,
            key=lambda method: (
                -min(
                    evaluations[method]["winner_accuracy"],
                    family_holdout_evaluations[method]["winner_accuracy"],
                ),
                family_holdout_evaluations[method]["median_absolute_relative_error"],
            ),
        )
        if eligible_methods
        else None
    )
    selected_method = best_method
    selected = evaluations[selected_method] if selected_method is not None else None
    uncertainty = (
        max(
            selected["maximum_multiplicative_error"],
            family_holdout_evaluations[selected_method]["maximum_multiplicative_error"],
        )
        if selected is not None
        else None
    )
    return {
        "strategy": strategy,
        "selection_rule": (
            "must beat the majority-winner predictor under both workload and "
            "architecture-family holdout; then maximize the weaker winner accuracy "
            "and minimize family-held-out median ratio error"
        ),
        "majority_winner_accuracy": majority_winner_accuracy,
        "selected_calibration": selected_method,
        "selected_maximum_multiplicative_error": uncertainty,
        "evaluations": evaluations,
        "architecture_family_holdout_evaluations": family_holdout_evaluations,
        "calibrated_points": (
            calibrated_points(points, selected_method, uncertainty)
            if selected_method is not None and uncertainty is not None
            else []
        ),
        "points": points,
    }


def build_analysis(profile_root):
    points, exclusions = build_points(profile_root)
    strategies = {
        strategy: analyze_strategy(strategy, strategy_points)
        for strategy, strategy_points in points.items()
    }
    return {
        "kind": "native-baseline-cost-model-calibration",
        "scope": "8xH100 forward-loss-backward-no-optimizer",
        "campaign_id": CALIBRATION_CAMPAIGN_ID,
        "calibration_protocol": (
            "leave-one-workload-out ratio calibration against existing target runs"
        ),
        "strategies": strategies,
        "exclusions": exclusions,
        "interpretation": (
            "Only a held-out calibrated interval that excludes the decision boundary "
            "supports a cost-only ranking. In-sample calibrated points are diagnostics, "
            "not independent acceptance evidence."
        ),
    }


def main():
    config = resolve_report_config()
    profile_root = config.report_root / "v1" / config.profile_name
    analysis = build_analysis(profile_root)
    source_run_ids = sorted(
        {
            run_id
            for strategy in analysis["strategies"].values()
            for point in strategy["points"]
            for run_id in (
                point["modeled_run_id"],
                *point["measured_run_ids"].values(),
            )
        }
    )
    request = {
        "workload": {
            "name": "huggingface-native-baseline-cost-model-calibration-v1",
            "mode": "offline_mixed_evidence_analysis",
        },
        "mesh": {"shape": [8], "names": ["dp"]},
        "source_run_ids": source_run_ids,
    }
    campaign = {
        "id": "huggingface-native-baseline-cost-model-calibration-v1",
        "suite": "huggingface-cost-model-baselines",
        "case_id": "calibration",
        "expected_case_count": 1,
    }
    parallelization = {
        "decision_id": "huggingface-native-baseline-cost-model-calibration-v1",
        "intent": {
            "kind": "distributed_training",
            "requested_strategy": "automatic",
            "attempt_budget": default_attempt_budget(),
        },
        "acceptance_criteria": default_acceptance_criteria(),
    }
    provenance = _default_provenance(
        compiler_config={"analysis": "native-baseline-ratio-calibration-v1"},
        source_root=REPO_ROOT,
        external_source_roots=None,
        entry_point="experiments.analyze_hf_cost_model_baselines:main",
        launch_config={"launcher": "offline-linked-report-analysis"},
    )
    with AutoParallelRunReport(
        request=_request_with_campaign(request, campaign),
        provenance=provenance,
        config=config,
        source_root=REPO_ROOT,
        parallelization=parallelization,
    ) as report:
        report.add_artifact(
            "reproducer",
            Path(__file__),
            role="reproducer",
            media_type="text/x-python",
        )
        report.record_validation(
            "native-baseline-cost-model-calibration",
            analysis,
            evidence_level="measured",
        )
        report.record_competitiveness_assessment(
            "comparison_unavailable",
            evidence="mixed",
            reason_codes=("CALIBRATION_ONLY",),
            details={
                strategy: {
                    "selected_calibration": result["selected_calibration"],
                    "maximum_multiplicative_error": result[
                        "selected_maximum_multiplicative_error"
                    ],
                }
                for strategy, result in analysis["strategies"].items()
            },
        )
        report.record_autoparallel_outcome(
            "inconclusive", reason_codes=("CALIBRATION_ONLY",)
        )
        report.record_parallelization_decision(
            "none", "inconclusive", reason_codes=("CALIBRATION_ONLY",)
        )
        report.add_finding(
            "NATIVE_BASELINE_COST_MODEL_CALIBRATION_ONLY",
            "The linked measurements calibrate cost-only rankings but do not validate a prospective workload",
            context={
                strategy: {
                    method: {
                        key: evaluation[key]
                        for key in (
                            "median_absolute_relative_error",
                            "maximum_absolute_relative_error",
                            "winner_accuracy",
                        )
                    }
                    for method, evaluation in result["evaluations"].items()
                }
                for strategy, result in analysis["strategies"].items()
            },
            suggested_action=(
                "Reserve architecture families or future workloads as prospective "
                "validation before using calibrated cost-only rankings for routing"
            ),
        )
    print(f"report={report.path}")


if __name__ == "__main__":
    main()
