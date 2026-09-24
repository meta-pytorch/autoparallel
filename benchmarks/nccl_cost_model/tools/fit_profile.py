#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import json
import math
import pprint
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROFILE = "h100_nvswitch_roce_400g"
TRAIN_REPLICATES = (1, 2)
VALIDATION_REPLICATE = 3
LATENCY_MAX_BYTES = 64 << 10
PEAK_MIN_BYTES = 128 << 20
RAMP_POINTS = 24


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round(fraction * (len(ordered) - 1))]


def load_jobs(manifest_path: Path) -> list[str]:
    manifest = json.loads(manifest_path.read_text())
    jobs = [job for group in manifest["formal"].values() for job in group]
    jobs.extend(manifest.get("supplemental", []))
    return jobs


def load_records(
    manifest_path: Path, results_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for job in load_jobs(manifest_path):
        job_root = results_root / job
        if not (job_root / "SUCCESS").is_file():
            raise RuntimeError(f"job has not completed successfully: {job}")
        for metadata_path in sorted(job_root.glob("metadata.*.json")):
            item = json.loads(metadata_path.read_text())
            if item["profile"] != PROFILE or item["gate"]:
                raise RuntimeError(f"unexpected metadata in {metadata_path}")
            metadata.append(item)
        for measurements_path in sorted(job_root.glob("measurements.*.jsonl")):
            mode = measurements_path.name.removeprefix("measurements.").removesuffix(
                ".jsonl"
            )
            mode_metadata = next(
                item
                for item in metadata
                if item["job_id"] == job and item["mode"] == mode
            )
            latest_records = {}
            for line in measurements_path.read_text().splitlines():
                item = json.loads(line)
                item["job_id"] = job
                item["replicate"] = mode_metadata["replicate"]
                key = (
                    item["execution"],
                    item["nodes"],
                    item["ppn"],
                    item["collective"],
                    item["dtype"],
                    item["n_bytes"],
                )
                latest_records[key] = item
            records.extend(latest_records.values())
    return records, metadata


def aggregate(
    records: list[dict[str, Any]], replicates: tuple[int, ...]
) -> dict[tuple[str, str, int, int, int], float]:
    samples: dict[tuple[str, str, int, int, int], list[float]] = defaultdict(list)
    for record in records:
        if (
            record["replicate"] not in replicates
            or record["execution"] != "isolated"
            or record["dtype"] != "torch.float32"
        ):
            continue
        key = (
            record["mode"],
            record["collective"],
            record["nodes"],
            record["ppn"],
            record["n_bytes"],
        )
        samples[key].append(record["rank_p50_max_us"])
    return {key: statistics.median(values) for key, values in samples.items()}


def fit_runtime_tables(
    samples: dict[tuple[str, str, int, int, int], float]
) -> dict[tuple[str, str, int, int], tuple[float, float]]:
    grouped: dict[tuple[str, str, int, int], list[tuple[int, float]]] = defaultdict(
        list
    )
    for (mode, collective, nodes, ppn, n_bytes), time_us in samples.items():
        grouped[(mode, collective, nodes, ppn)].append((n_bytes, time_us))

    result = {}
    for signature, points in grouped.items():
        latency_us = min(
            time_us for n_bytes, time_us in points if n_bytes <= LATENCY_MAX_BYTES
        )
        bandwidths = [
            n_bytes / (1000.0 * (time_us - latency_us))
            for n_bytes, time_us in points
            if n_bytes >= PEAK_MIN_BYTES and time_us > latency_us
        ]
        if not bandwidths:
            raise RuntimeError(f"no peak-bandwidth samples for {signature}")
        result[signature] = latency_us, max(bandwidths)
    return result


def interpolate(points: list[tuple[float, float]], x: float) -> float:
    if x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return points[-1][1]


def fit_ramps(
    samples: dict[tuple[str, str, int, int, int], float],
    tables: dict[tuple[str, str, int, int], tuple[float, float]],
) -> dict[tuple[str, str, int, int], tuple[float, ...]]:
    curves: dict[tuple[str, str, int, int], list[tuple[float, float]]] = defaultdict(
        list
    )
    for (mode, collective, nodes, ppn, n_bytes), time_us in samples.items():
        latency_us, peak_bw = tables[(mode, collective, nodes, ppn)]
        if time_us <= latency_us:
            continue
        effective_bw = n_bytes / (1000.0 * (time_us - latency_us))
        per_rank_bytes = n_bytes / (nodes * ppn)
        x = math.log2(per_rank_bytes / 64.0)
        curves[(mode, collective, nodes, ppn)].append(
            (x, min(1.0, effective_bw / peak_bw))
        )

    result = {}
    for group, curve in curves.items():
        design = []
        targets = []
        for x, value in curve:
            row = np.zeros(RAMP_POINTS)
            if x <= 0:
                row[0] = 1.0
            elif x >= RAMP_POINTS - 1:
                row[-1] = 1.0
            else:
                lower = int(x)
                fraction = x - lower
                row[lower] = 1.0 - fraction
                row[lower + 1] = fraction
            design.append(row)
            targets.append(value)
        smoothness = np.zeros((RAMP_POINTS - 2, RAMP_POINTS))
        for index in range(RAMP_POINTS - 2):
            smoothness[index, index : index + 3] = (1.0, -2.0, 1.0)
        matrix = np.vstack((design, math.sqrt(0.01) * smoothness))
        target = np.concatenate((np.asarray(targets), np.zeros(RAMP_POINTS - 2)))
        fitted, *_ = np.linalg.lstsq(matrix, target, rcond=None)
        result[group] = tuple(float(np.clip(value, 0.01, 1.0)) for value in fitted)
    return result


def runtime_prediction(
    key: tuple[str, str, int, int, int],
    tables: dict[tuple[str, str, int, int], tuple[float, float]],
    ramps: dict[tuple[str, str, int, int], tuple[float, ...]],
) -> float:
    mode, collective, nodes, ppn, n_bytes = key
    latency_us, peak_bw = tables[(mode, collective, nodes, ppn)]
    per_rank_bytes = n_bytes / (nodes * ppn)
    x = math.log2(per_rank_bytes / 64.0)
    ramp = interpolate(list(enumerate(ramps[(mode, collective, nodes, ppn)])), x)
    return latency_us + n_bytes / (1000.0 * peak_bw * ramp)


def validation_metrics(
    validation: dict[tuple[str, str, int, int, int], float],
    tables: dict[tuple[str, str, int, int], tuple[float, float]],
    ramps: dict[tuple[str, str, int, int], tuple[float, ...]],
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    workload_grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for key, observed in validation.items():
        if key[:4] not in tables:
            continue
        error = abs(runtime_prediction(key, tables, ramps) - observed) / observed
        grouped[key[:2]].append(error)
        if key[-1] in (28 << 20, 104 << 20, 128 << 20):
            workload_grouped[key[:2]].append(error)

    result = {}
    for (mode, collective), errors in sorted(grouped.items()):
        result[f"{mode}:{collective}"] = {
            "samples": len(errors),
            "median_absolute_percentage_error": statistics.median(errors) * 100.0,
            "p95_absolute_percentage_error": percentile(errors, 0.95) * 100.0,
            "max_absolute_percentage_error": max(errors) * 100.0,
        }
        workload_errors = workload_grouped[(mode, collective)]
        if workload_errors:
            result[f"{mode}:{collective}"].update(
                {
                    "workload_median_absolute_percentage_error": statistics.median(
                        workload_errors
                    )
                    * 100.0,
                    "workload_p95_absolute_percentage_error": percentile(
                        workload_errors, 0.95
                    )
                    * 100.0,
                    "workload_max_absolute_percentage_error": max(workload_errors)
                    * 100.0,
                }
            )
    return result


def concurrent_slowdowns(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int, int, int, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        if (
            record["mode"] != "auto"
            or record["replicate"] not in TRAIN_REPLICATES
            or record["dtype"] != "torch.float32"
        ):
            continue
        key = (
            record["collective"],
            record["nodes"],
            record["ppn"],
            record["n_bytes"],
            record["dtype"],
        )
        grouped[key][record["execution"]].append(record["rank_p50_max_us"])

    result = {}
    for (collective, nodes, ppn, n_bytes, _), executions in sorted(grouped.items()):
        if set(executions) != {"isolated", "concurrent"}:
            continue
        isolated = statistics.median(executions["isolated"])
        concurrent = statistics.median(executions["concurrent"])
        result[f"{collective}:{nodes},{ppn}:{n_bytes}"] = concurrent / isolated
    return result


def nested_tables(
    tables: dict[tuple[str, str, int, int], tuple[float, float]]
) -> dict[str, dict[str, dict[tuple[int, int], tuple[float, float]]]]:
    result: dict[str, dict[str, dict[tuple[int, int], tuple[float, float]]]] = {}
    for (mode, collective, nodes, ppn), value in sorted(tables.items()):
        result.setdefault(mode, {}).setdefault(collective, {})[(nodes, ppn)] = (
            round(value[0], 6),
            round(value[1], 6),
        )
    return result


def nested_ramps(
    ramps: dict[tuple[str, str, int, int], tuple[float, ...]]
) -> dict[str, dict[str, dict[tuple[int, int], tuple[float, ...]]]]:
    result: dict[str, dict[str, dict[tuple[int, int], tuple[float, ...]]]] = {}
    for (mode, collective, nodes, ppn), values in sorted(ramps.items()):
        result.setdefault(mode, {}).setdefault(collective, {})[(nodes, ppn)] = tuple(
            round(value, 6) for value in values
        )
    return result


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    return value


def render_python(tables: dict[str, Any], ramps: dict[str, Any]) -> str:
    return (
        "# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.\n"
        "#\n"
        "# This source code is licensed under the BSD license found in the\n"
        "# LICENSE file in the root directory of this source tree.\n\n"
        "# Generated by benchmarks/nccl_cost_model/tools/fit_profile.py.\n"
        "H100_NVSWITCH_ROCE_400G_NVLS_TREE_AVAILABLE = False\n\n"
        "H100_NVSWITCH_ROCE_400G_RUNTIME_TABLES = "
        + pprint.pformat(tables["auto"], sort_dicts=True, width=100)
        + "\n\nH100_NVSWITCH_ROCE_400G_RUNTIME_RAMPS = "
        + pprint.pformat(ramps["auto"], sort_dicts=True, width=100)
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--python-output", type=Path, required=True)
    args = parser.parse_args()

    records, metadata = load_records(args.manifest, args.results_root)
    training = aggregate(records, TRAIN_REPLICATES)
    validation = aggregate(records, (VALIDATION_REPLICATE,))
    training_tables = fit_runtime_tables(training)
    training_ramps = fit_ramps(training, training_tables)
    production = aggregate(records, TRAIN_REPLICATES + (VALIDATION_REPLICATE,))
    production_tables = fit_runtime_tables(production)
    production_ramps = fit_ramps(production, production_tables)
    nested_table_values = nested_tables(production_tables)
    nested_ramp_values = nested_ramps(production_ramps)
    profile = {
        "schema_version": 1,
        "profile": PROFILE,
        "training_replicates": TRAIN_REPLICATES,
        "validation_replicate": VALIDATION_REPLICATE,
        "metric": "maximum across ranks of per-rank median CUDA-event latency",
        "runtime_tables": nested_table_values,
        "runtime_ramps": nested_ramp_values,
        "validation": validation_metrics(validation, training_tables, training_ramps),
        "concurrent_slowdowns": concurrent_slowdowns(records),
        "jobs": metadata,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(json_safe(profile), indent=2) + "\n")
    args.python_output.parent.mkdir(parents=True, exist_ok=True)
    args.python_output.write_text(
        render_python(nested_table_values, nested_ramp_values)
    )


if __name__ == "__main__":
    main()
