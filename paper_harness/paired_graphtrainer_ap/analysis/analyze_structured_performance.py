#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import analyze_formal


WORLD_SIZE = 16
STEADY_STEPS = tuple(range(6, 26))
PHASES = {
    "manual": "01_per_gpu_bs1_graphtrainer",
    "autoparallel": "02_per_gpu_bs1_graphtrainer_ap",
}
TIMER_EVENTS = (
    "step_end",
    "gc_collect_end",
    "fetching_batch_end",
    "post_dataloading_process_end",
    "optim_end",
    "collect_dist_metrics_end",
    "checkpoint_save_end",
)
REQUIRED_TIMER_EVENTS = tuple(
    name for name in TIMER_EVENTS if name != "collect_dist_metrics_end"
)


def summarize(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "samples": values,
    }


def allocation_pair_valid(run_root: Path) -> bool:
    identities = {}
    for arm, phase in PHASES.items():
        identities[arm] = [
            analyze_formal.allocation_identity(
                json.loads(
                    (
                        run_root / "allocation" / phase / f"rank_{rank:03d}.json"
                    ).read_text()
                )
            )
            for rank in range(WORLD_SIZE)
        ]
    return identities["manual"] == identities["autoparallel"]


def structured_phase(phase_root: Path) -> dict[str, Any]:
    paths = sorted(
        (phase_root / "job/structured_logs").glob("training.global_rank_*.jsonl")
    )
    records_by_rank: dict[int, dict[int, dict[str, float]]] = {}
    parse_errors = []
    for path in paths:
        rows = []
        with path.open(errors="replace") as source:
            for line_number, line in enumerate(source, 1):
                try:
                    rows.append(json.loads(line))
                except Exception as error:
                    parse_errors.append(
                        {
                            "path": str(path),
                            "line": line_number,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
        ranks = {
            row.get("global_rank")
            for row in rows
            if isinstance(row.get("global_rank"), int)
        }
        if len(ranks) != 1:
            parse_errors.append(
                {"path": str(path), "error": f"ambiguous ranks: {sorted(ranks)}"}
            )
            continue
        rank = next(iter(ranks))
        timings = records_by_rank.setdefault(rank, {})
        for row in rows:
            step = row.get("step")
            name = row.get("log_type_name")
            value = row.get("value")
            if (
                step in STEADY_STEPS
                and name in TIMER_EVENTS
                and isinstance(value, (int, float))
                and math.isfinite(value)
            ):
                timings.setdefault(step, {})[name] = float(value)

    missing = []
    records = []
    for rank in range(WORLD_SIZE):
        for step in STEADY_STEPS:
            timers = records_by_rank.get(rank, {}).get(step, {})
            absent = [name for name in REQUIRED_TIMER_EVENTS if name not in timers]
            if absent:
                missing.append({"rank": rank, "step": step, "events": absent})
                continue
            record = {
                "rank": rank,
                "step": step,
                **{name: timers.get(name, 0.0) for name in TIMER_EVENTS},
            }
            record["instrumented_noncore_ms"] = sum(
                record[name] for name in TIMER_EVENTS if name != "step_end"
            )
            record["unattributed_training_ms"] = (
                record["step_end"] - record["instrumented_noncore_ms"]
            )
            records.append(record)

    components = {}
    component_names = (
        *TIMER_EVENTS,
        "instrumented_noncore_ms",
        "unattributed_training_ms",
    )
    for name in component_names:
        all_rank_steps = [record[name] for record in records]
        rank0_steps = [record[name] for record in records if record["rank"] == 0]
        per_step_rank_max = [
            max(record[name] for record in records if record["step"] == step)
            for step in STEADY_STEPS
            if any(record["step"] == step for record in records)
        ]
        component = {
            "all_rank_steps": summarize(all_rank_steps) if all_rank_steps else None,
            "per_step_rank_max": (
                summarize(per_step_rank_max) if per_step_rank_max else None
            ),
            "rank0_steps": summarize(rank0_steps) if rank0_steps else None,
        }
        if name == "collect_dist_metrics_end":
            recorded_only = [value for value in all_rank_steps if value]
            component["recorded_only"] = (
                summarize(recorded_only) if recorded_only else None
            )
        components[name] = component

    valid = (
        len(paths) == WORLD_SIZE
        and sorted(records_by_rank) == list(range(WORLD_SIZE))
        and not parse_errors
        and not missing
        and len(records) == WORLD_SIZE * len(STEADY_STEPS)
    )
    return {
        "status": "passed" if valid else "failed",
        "paths": [str(path) for path in paths],
        "observed_ranks": sorted(records_by_rank),
        "steady_steps": list(STEADY_STEPS),
        "rank_step_count": len(records),
        "parse_errors": parse_errors,
        "missing": missing,
        "components_ms": components,
    }


def pair(left: float, right: float) -> dict[str, float | None]:
    return {
        "manual": left,
        "autoparallel": right,
        "ap_minus_manual": right - left,
        "ratio_ap_over_manual": right / left if left else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    frozen_copy_marker = run_root.parent.parent / "run_output.copy_complete"
    case = analyze_formal.CASES["per_gpu_bs1"]
    configs = {}
    phases = {}
    structured = {}
    for arm, phase in PHASES.items():
        phase_root = run_root / phase
        config = analyze_formal.phase_config(phase_root, case, arm)
        configs[arm] = config
        phases[phase] = {
            "execution": analyze_formal.phase_state(run_root, phase),
            "tensorboard": analyze_formal.tensorboard_result(phase_root),
            "numerics": analyze_formal.training_numerics(phase_root),
        }
        structured[arm] = structured_phase(phase_root)

    config_parity = analyze_formal.pair_config_parity(configs)
    allocation_valid = allocation_pair_valid(run_root)
    paired_tensorboard = analyze_formal.paired_performance(
        "per_gpu_bs1",
        phases,
        config_parity,
        {"pair_validity": {"per_gpu_bs1.performance": allocation_valid}},
    )
    component_pairs = {}
    for name in (
        *TIMER_EVENTS,
        "instrumented_noncore_ms",
        "unattributed_training_ms",
    ):
        component_pairs[name] = {}
        for scope in ("all_rank_steps", "per_step_rank_max", "rank0_steps"):
            manual = structured["manual"]["components_ms"][name][scope]
            treatment = structured["autoparallel"]["components_ms"][name][scope]
            component_pairs[name][scope] = pair(manual["mean"], treatment["mean"])

    report = {
        "status": (
            "passed"
            if config_parity["status"] == "passed"
            and allocation_valid
            and all(value["status"] == "passed" for value in structured.values())
            and paired_tensorboard["status"] == "comparable"
            else "failed"
        ),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_state": (
            "immutable_final" if frozen_copy_marker.is_file() else "live_preliminary"
        ),
        "run_root": str(run_root),
        "measurement": {
            "tensorboard": "four rank-0 windows ending at steps 10, 15, 20, 25",
            "structured_logs": (
                "all 16 ranks and all individual steps 6-25; timer values are ms"
            ),
            "collect_dist_metrics_absence": (
                "treated as zero on non-reporting steps; recorded-only summary is "
                "also retained"
            ),
            "unattributed_training_ms": (
                "step_end minus GC, batch fetch, post-data, optimizer, distributed "
                "metrics, and checkpoint timers; not a directly instrumented timer"
            ),
        },
        "config_parity": config_parity,
        "allocation_performance_pair_valid": allocation_valid,
        "phase_states": {
            phase: phases[phase]["execution"] for phase in PHASES.values()
        },
        "tensorboard": paired_tensorboard,
        "structured_logs": structured,
        "structured_component_mean_pairs_ms": component_pairs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
