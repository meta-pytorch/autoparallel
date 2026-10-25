# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from typing import Any, Mapping

_COMPETITIVE_VERDICTS = {
    "robustly_competitive",
    "baseline_infeasible",
    "user_accepted_tradeoff",
}


def default_acceptance_criteria(
    *,
    memory_limit_bytes: int | None = None,
    performance_target: Mapping[str, Any] | None = None,
    require_numerical_validation: bool = False,
) -> dict[str, Any]:
    if memory_limit_bytes is not None and memory_limit_bytes <= 0:
        raise ValueError("memory_limit_bytes must be positive")
    return {
        "global_spmd_audit": "required",
        "minimum_evidence": "executed_on_target",
        "memory": {
            "must_fit": True,
            "per_rank_limit_bytes": memory_limit_bytes,
            "scope": "intended_training_step",
            "optimizer_state": "include_or_conservatively_estimate",
            "operational_headroom": "required",
        },
        "performance": {
            "required": performance_target is not None,
            "target": dict(performance_target or {}),
            "scope": "intended_training_step",
            "optimizer_update": "include_or_calibrate_when_training",
            "optimizer_implementation": ("preserve_configuration_and_record_realized"),
            "time_to_ready": "record_and_amortize_when_workload_duration_known",
        },
        "numerical_validation": (
            "required" if require_numerical_validation else "when_requested"
        ),
        "maximum_unresolved_error_findings": 0,
        "relative_quality": default_competitiveness_criteria(),
    }


def default_competitiveness_criteria() -> dict[str, Any]:
    return {
        "required": True,
        "acceptable_verdicts": sorted(_COMPETITIVE_VERDICTS),
        "modeled_latency_tolerance_fraction": 0.05,
        "reject_when_baseline_pareto_dominates": True,
        "baseline_benchmarking": "explicit_authorization_only",
    }


def default_attempt_budget() -> dict[str, Any]:
    return {
        "unmodified_attempts": 1,
        "targeted_semantics_preserving_retries": 1,
        "minutes_without_progress": 10,
        "semantics_changing_rewrites": 0,
    }


def recommended_fallback(
    *,
    replicated_training_state_fits: bool | None,
    parameter_sharding_required: bool,
) -> str:
    if parameter_sharding_required or replicated_training_state_fits is False:
        return "fsdp"
    if replicated_training_state_fits is True:
        return "ddp"
    raise ValueError("Fallback choice needs a replicated-state memory assessment")


def classify_modeled_competitiveness(
    autoparallel_latency: Mapping[str, Any] | None,
    baselines: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
    *,
    relative_tolerance: float = 0.05,
) -> dict[str, Any]:
    if (
        isinstance(relative_tolerance, bool)
        or not isinstance(relative_tolerance, (int, float))
        or not math.isfinite(relative_tolerance)
        or relative_tolerance < 0
    ):
        raise ValueError("relative_tolerance must be finite and non-negative")
    if not baselines:
        return _verdict("comparison_unavailable", "BASELINE_COMPARISON_UNAVAILABLE")

    feasible = [
        baseline for baseline in baselines if baseline.get("feasibility") == "feasible"
    ]
    if not feasible:
        if all(baseline.get("feasibility") == "infeasible" for baseline in baselines):
            return _verdict("baseline_infeasible", "BASELINE_INFEASIBLE")
        return _verdict("comparison_unavailable", "BASELINE_COMPARISON_UNAVAILABLE")
    if autoparallel_latency is None or any(
        not isinstance(baseline.get("latency"), Mapping) for baseline in feasible
    ):
        return _verdict("comparison_unavailable", "BASELINE_COMPARISON_UNAVAILABLE")

    autoparallel_bounds = _effective_bounds(autoparallel_latency)
    baseline_bounds = [
        (str(baseline.get("strategy")), _effective_bounds(baseline["latency"]))
        for baseline in feasible
    ]
    units = {autoparallel_bounds["unit"]} | {
        bounds["unit"] for _, bounds in baseline_bounds
    }
    if len(units) != 1:
        raise ValueError("all competitiveness latency envelopes must use one unit")

    best_lower = min(bounds["lower"] for _, bounds in baseline_bounds)
    best_upper = min(bounds["upper"] for _, bounds in baseline_bounds)
    tolerance_multiplier = 1.0 + relative_tolerance
    details = {
        "relative_tolerance": relative_tolerance,
        "autoparallel_effective_bounds": autoparallel_bounds,
        "baseline_effective_bounds": [
            {"strategy": strategy, **bounds} for strategy, bounds in baseline_bounds
        ],
    }
    if autoparallel_bounds["upper"] <= best_lower * tolerance_multiplier:
        return _verdict("robustly_competitive", "MODELED_ROBUSTLY_COMPETITIVE", details)
    if autoparallel_bounds["lower"] > best_upper * tolerance_multiplier:
        return _verdict("robustly_worse", "MODELED_ROBUSTLY_WORSE", details)
    return _verdict("schedule_sensitive", "MODELED_SCHEDULE_SENSITIVE", details)


def find_pareto_dominating_baseline(
    autoparallel: Mapping[str, Any],
    baselines: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
) -> dict[str, Any] | None:
    autoparallel_latency = autoparallel.get("latency")
    autoparallel_memory = _peak_memory_bytes(autoparallel.get("memory"))
    if not isinstance(autoparallel_latency, Mapping) or autoparallel_memory is None:
        return None
    autoparallel_bounds = _effective_bounds(autoparallel_latency)
    autoparallel_startup = _startup_bounds(autoparallel.get("time_to_ready"))
    feasible_baselines = [
        baseline
        for baseline in baselines
        if baseline.get("feasibility") == "feasible"
        and isinstance(baseline.get("latency"), Mapping)
    ]
    feasible_baselines.sort(
        key=lambda baseline: _effective_bounds(baseline["latency"])["upper"]
    )
    for baseline in feasible_baselines:
        baseline_latency = baseline.get("latency")
        baseline_memory = _peak_memory_bytes(baseline.get("memory"))
        if not isinstance(baseline_latency, Mapping) or baseline_memory is None:
            continue
        baseline_bounds = _effective_bounds(baseline_latency)
        if baseline_bounds["unit"] != autoparallel_bounds["unit"]:
            raise ValueError("Pareto latency envelopes must use one unit")
        comparisons = {
            "steady_state_latency": {
                "autoparallel_lower": autoparallel_bounds["lower"],
                "baseline_upper": baseline_bounds["upper"],
                "no_worse": baseline_bounds["upper"] <= autoparallel_bounds["lower"],
                "strictly_better": (
                    baseline_bounds["upper"] < autoparallel_bounds["lower"]
                ),
            },
            "peak_memory": {
                "autoparallel_bytes": autoparallel_memory,
                "baseline_bytes": baseline_memory,
                "no_worse": baseline_memory <= autoparallel_memory,
                "strictly_better": baseline_memory < autoparallel_memory,
            },
        }
        baseline_startup = _startup_bounds(baseline.get("time_to_ready"))
        if (autoparallel_startup is None) != (baseline_startup is None):
            continue
        if autoparallel_startup is not None and baseline_startup is not None:
            comparisons["time_to_ready"] = {
                "autoparallel_lower_seconds": autoparallel_startup["lower"],
                "baseline_upper_seconds": baseline_startup["upper"],
                "no_worse": (
                    baseline_startup["upper"] <= autoparallel_startup["lower"]
                ),
                "strictly_better": (
                    baseline_startup["upper"] < autoparallel_startup["lower"]
                ),
            }
        if not all(item["no_worse"] for item in comparisons.values()):
            continue
        if not any(item["strictly_better"] for item in comparisons.values()):
            continue
        return {
            "strategy": str(baseline.get("strategy")),
            "reason_code": "BASELINE_PARETO_DOMINATES",
            "comparisons": comparisons,
        }
    return None


def _peak_memory_bytes(value: Any) -> int | float | None:
    if not isinstance(value, Mapping):
        return None
    for field in (
        "projected_peak_with_adamw_moments_bytes",
        "observed_peak_allocated_bytes",
        "peak_allocated_bytes",
    ):
        result = value.get(field)
        if (
            not isinstance(result, bool)
            and isinstance(result, (int, float))
            and math.isfinite(result)
            and result >= 0
        ):
            return result
    return None


def _startup_bounds(value: Any) -> dict[str, float] | None:
    if not isinstance(value, Mapping):
        return None
    lower = value.get("minimum_first_iteration")
    upper = value.get("maximum_first_iteration")
    if (
        isinstance(lower, bool)
        or not isinstance(lower, (int, float))
        or not math.isfinite(lower)
        or lower < 0
        or isinstance(upper, bool)
        or not isinstance(upper, (int, float))
        or not math.isfinite(upper)
        or upper < 0
    ):
        return None
    if lower > upper:
        raise ValueError("time-to-ready bounds must satisfy minimum <= maximum")
    return {"lower": float(lower), "upper": float(upper)}


def _effective_bounds(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "unit",
        "optimistic",
        "expected",
        "pessimistic",
        "calibration_relative_error",
        "schedule_model",
    }
    missing = required - value.keys()
    if missing:
        raise ValueError(f"latency envelope is missing fields: {sorted(missing)}")
    raw_values = [
        value["optimistic"],
        value["expected"],
        value["pessimistic"],
        value["calibration_relative_error"],
    ]
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(item)
        for item in raw_values
    ):
        raise ValueError("latency envelope values must be finite numbers")
    optimistic = float(raw_values[0])
    expected = float(raw_values[1])
    pessimistic = float(raw_values[2])
    calibration_error = float(raw_values[3])
    if min(optimistic, expected, pessimistic) < 0:
        raise ValueError("latency estimates must be non-negative")
    if not optimistic <= expected <= pessimistic:
        raise ValueError(
            "latency envelope must satisfy optimistic <= expected <= pessimistic"
        )
    if calibration_error < 0:
        raise ValueError("calibration_relative_error must be non-negative")
    unit = value["unit"]
    if not isinstance(unit, str) or not unit:
        raise ValueError("latency envelope unit must be a non-empty string")
    return {
        "unit": unit,
        "lower": max(0.0, optimistic * (1.0 - calibration_error)),
        "upper": pessimistic * (1.0 + calibration_error),
    }


def _verdict(
    verdict: str, reason_code: str, details: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "verdict": verdict,
        "reason_codes": [reason_code],
        "details": dict(details or {}),
    }
