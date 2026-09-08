#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import shutil
import statistics
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


TASK_ROOT = Path(__file__).resolve().parents[1]
WORLD_SIZE = 16
FSDP_DEGREE = 8
TP_DEGREE = 2
SEQUENCE_LENGTH = 4096
LOCAL_BATCH_SIZE = 2
GLOBAL_BATCH_SIZE = 16
TOKENS_PER_STEP = 65_536
STEADY_TB_STEPS = (10, 15, 20, 25)
STEADY_STRUCTURED_STEPS = tuple(range(6, 26))
PROFILER_STEP = 4
PERFORMANCE_PHASE = "01_per_gpu_bs1_torchtitan_native_inductor"
TRACE_PHASE = "11_trace_per_gpu_bs1_torchtitan_native_inductor"
PHASES = {
    PERFORMANCE_PHASE: {"steps": 25, "kind": "performance"},
    TRACE_PHASE: {"steps": 6, "kind": "trace"},
}
EXPECTED_TORCHTITAN_COMMIT = "c59ce51a6fc4f2340d320fe914a7c2049b747c8f"
EXPECTED_TORCH_COMMIT = "77bc9b492a38ec3122bb510f588a1321f4038e5c"
EXPECTED_TORCH_VERSION = "2.14.0a0+git77bc9b4"
EXPECTED_INPUT_MANIFEST_SHA256 = (
    "4fda638c406cb949694e6007b404f026b1c37a33b94511aefdf2e78b14520f71"
)
EXPECTED_CONFIG = "muse_glimmer_30b_sdpa_c4_torchtitan_4x2"
EXPECTED_MODULE = "graph_trainer.muse_glimmer"
RUN_ROOT_PATTERN = re.compile(
    r"muse_glimmer_30b_native_inductor_lb2_s4096_2x8_"
    r"formal_r\d+_v(?P<version>\d+)_a(?P<attempt_index>\d+)"
)
CALLABLE_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")
SERIALIZED_CONFIG_NORMALIZATIONS = {
    ("dump_folder",): "<phase-output>",
    ("debug", "save_config_file"): "<phase-output>",
    ("hf_assets_path",): "<hf-assets-path>",
}
NUMBER = r"(?:[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[+-]?(?:nan|inf))"
FATAL = re.compile(
    r"CUDA out of memory|OutOfMemoryError|ChildFailedError|"
    r"ProcessExitedException|Segmentation fault|NCCL error|"
    r"Watchdog caught collective operation timeout|AssertionError",
    re.IGNORECASE,
)
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
    event for event in TIMER_EVENTS if event != "collect_dist_metrics_end"
)
ALLOCATION_FIELDS = (
    "hostname",
    "hosts",
    "datacenter",
    "job_id",
    "rank",
    "local_rank",
    "world_size",
    "local_world_size",
    "gpu_ordinal",
    "gpu_name",
    "gpu_total_memory_bytes",
    "gpu_compute_capability",
    "gpu_pci_bus_id",
    "gpu_uuid",
    "cuda_visible_devices",
    "device_network_id",
    "device_backend_network_topology",
)


def summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "sample_variance": statistics.variance(values) if len(values) > 1 else 0.0,
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "samples": values,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def nested(value: dict[str, Any], *path: str) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def capture(section: str, function: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        return function()
    except Exception as error:
        return {
            "status": "failed",
            "section": section,
            "error": f"{type(error).__name__}: {error}",
        }


def discover_run_root(attempt: Path) -> Path:
    base = attempt / "artifacts/run_output"
    candidates = [
        path
        for path in sorted(base.glob("*"))
        if path.is_dir() and RUN_ROOT_PATTERN.fullmatch(path.name)
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one native formal run root under {base}, "
            f"found {[str(path) for path in candidates]}"
        )
    return candidates[0]


def read_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if offset >= len(data):
            raise ValueError("truncated protobuf varint")
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7


def protobuf_fields(data: bytes):
    offset = 0
    while offset < len(data):
        key, offset = read_varint(data, offset)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, offset = read_varint(data, offset)
        elif wire == 1:
            value = data[offset : offset + 8]
            offset += 8
        elif wire == 2:
            length, offset = read_varint(data, offset)
            value = data[offset : offset + length]
            offset += length
        elif wire == 5:
            value = data[offset : offset + 4]
            offset += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        yield field, wire, value


def scalar_values(event_path: Path) -> dict[str, dict[int, float]]:
    result: dict[str, dict[int, float]] = {}
    with event_path.open("rb") as source:
        while header := source.read(12):
            if len(header) != 12:
                raise ValueError(f"{event_path}: truncated TFRecord header")
            record_length = struct.unpack("<Q", header[:8])[0]
            record = source.read(record_length)
            footer = source.read(4)
            if len(record) != record_length or len(footer) != 4:
                raise ValueError(f"{event_path}: truncated TFRecord")
            event_fields = list(protobuf_fields(record))
            step = next(
                (
                    value
                    for field, wire, value in event_fields
                    if field == 2 and wire == 0
                ),
                None,
            )
            summary = next(
                (
                    value
                    for field, wire, value in event_fields
                    if field == 5 and wire == 2
                ),
                None,
            )
            if step is None or summary is None:
                continue
            for field, wire, value in protobuf_fields(summary):
                if field != 1 or wire != 2:
                    continue
                summary_value = list(protobuf_fields(value))
                tag = next(
                    (
                        item
                        for number, kind, item in summary_value
                        if number == 1 and kind == 2
                    ),
                    None,
                )
                scalar = next(
                    (
                        item
                        for number, kind, item in summary_value
                        if number == 2 and kind == 5
                    ),
                    None,
                )
                if tag is not None and scalar is not None:
                    result.setdefault(tag.decode(), {})[int(step)] = float(
                        struct.unpack("<f", scalar)[0]
                    )
    return result


def values_at_steps(
    scalars: dict[str, dict[int, float]], tag: str, steps: tuple[int, ...]
) -> tuple[list[float], list[int]]:
    samples = [scalars.get(tag, {}).get(step) for step in steps]
    missing = [
        step for step, value in zip(steps, samples, strict=True) if value is None
    ]
    return [float(value) for value in samples if value is not None], missing


def tensorboard_summary(phase_root: Path) -> dict[str, Any]:
    paths = sorted((phase_root / "job/tb").glob("**/events.out.tfevents.*"))
    if len(paths) != 1:
        return {
            "status": "failed",
            "error": f"expected one TensorBoard event file, found {len(paths)}",
            "paths": [str(path) for path in paths],
        }
    scalars = scalar_values(paths[0])
    tags = {
        "latency_s": "time_metrics/end_to_end(s)",
        "data_loading_s": "time_metrics/data_loading(s)",
        "data_loading_percent": "time_metrics/data_loading(%)",
        "per_device_tps": "throughput(tps)",
        "tflops_per_device": "tflops",
        "mfu_percent": "mfu(%)",
        "active_memory_gib": "memory/max_active(GiB)",
        "reserved_memory_gib": "memory/max_reserved(GiB)",
        "allocator_retries": "memory/num_alloc_retries",
        "allocator_ooms": "memory/num_ooms",
    }
    measurements: dict[str, Any] = {}
    all_present = True
    for name, tag in tags.items():
        samples, missing = values_at_steps(scalars, tag, STEADY_TB_STEPS)
        measurements[name] = {
            "tag": tag,
            "summary": summarize(samples) if samples else None,
            "missing_steps": missing,
        }
        all_present = all_present and not missing
    latency = measurements["latency_s"]["summary"]
    loading = measurements["data_loading_s"]["summary"]
    per_device = measurements["per_device_tps"]["summary"]
    if latency and loading:
        measurements["non_data_s"] = {
            "summary": summarize(
                [
                    total - data
                    for total, data in zip(
                        latency["samples"], loading["samples"], strict=True
                    )
                ]
            )
        }
    if per_device:
        measurements["aggregate_tps"] = {
            "summary": summarize(
                [value * WORLD_SIZE for value in per_device["samples"]]
            )
        }
    loss = scalars.get("loss_metrics/global_avg_loss", {})
    grad = scalars.get("grad_norm", {})
    finite = all(
        math.isfinite(value) for values in scalars.values() for value in values.values()
    )
    return {
        "status": "passed" if all_present and finite else "failed",
        "path": str(paths[0]),
        "sha256": sha256(paths[0]),
        "scope": "rank 0, four five-step windows ending at 10/15/20/25",
        "available_tags": sorted(scalars),
        "measurements": measurements,
        "final_loss": loss[max(loss)] if loss else None,
        "final_grad_norm": grad[max(grad)] if grad else None,
        "all_values_finite": finite,
    }


def structured_logs_summary(phase_root: Path) -> dict[str, Any]:
    paths = sorted(
        (phase_root / "job/structured_logs").glob("training.global_rank_*.jsonl")
    )
    records_by_rank: dict[int, dict[int, dict[str, float]]] = {}
    parse_errors: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    for path in paths:
        filename_match = re.search(r"global_rank_(\d+)\.", path.name)
        fallback_rank = int(filename_match.group(1)) if filename_match else None
        with path.open(errors="replace") as source:
            for line_number, line in enumerate(source, 1):
                try:
                    row = json.loads(line)
                except Exception as error:
                    parse_errors.append(
                        {
                            "path": str(path),
                            "line": line_number,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                    continue
                rank = row.get("global_rank", fallback_rank)
                step = row.get("step")
                name = row.get("log_type_name")
                value = row.get("value")
                if (
                    not isinstance(rank, int)
                    or step not in STEADY_STRUCTURED_STEPS
                    or name not in TIMER_EVENTS
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    continue
                timers = records_by_rank.setdefault(rank, {}).setdefault(step, {})
                if name in timers:
                    duplicates.append(
                        {"rank": rank, "step": step, "event": name, "path": str(path)}
                    )
                timers[name] = float(value)

    missing: list[dict[str, Any]] = []
    records: list[dict[str, float | int]] = []
    for rank in range(WORLD_SIZE):
        for step in STEADY_STRUCTURED_STEPS:
            timers = records_by_rank.get(rank, {}).get(step, {})
            absent = [event for event in REQUIRED_TIMER_EVENTS if event not in timers]
            if absent:
                missing.append({"rank": rank, "step": step, "events": absent})
                continue
            record: dict[str, float | int] = {"rank": rank, "step": step}
            record.update({event: timers.get(event, 0.0) for event in TIMER_EVENTS})
            record["instrumented_noncore_ms"] = sum(
                float(record[event]) for event in TIMER_EVENTS if event != "step_end"
            )
            record["unattributed_training_core_ms"] = float(record["step_end"]) - float(
                record["instrumented_noncore_ms"]
            )
            records.append(record)

    components: dict[str, Any] = {}
    names = (*TIMER_EVENTS, "instrumented_noncore_ms", "unattributed_training_core_ms")
    for name in names:
        all_rank = [float(record[name]) for record in records]
        rank0 = [float(record[name]) for record in records if record["rank"] == 0]
        per_step_max = [
            max(float(record[name]) for record in records if record["step"] == step)
            for step in STEADY_STRUCTURED_STEPS
            if any(record["step"] == step for record in records)
        ]
        components[name] = {
            "all_rank_steps": summarize(all_rank) if all_rank else None,
            "per_step_rank_max": summarize(per_step_max) if per_step_max else None,
            "rank0_steps": summarize(rank0) if rank0 else None,
        }
        if name == "collect_dist_metrics_end":
            recorded = [value for value in all_rank if value != 0.0]
            components[name]["recorded_only"] = (
                summarize(recorded) if recorded else None
            )

    valid = (
        len(paths) == WORLD_SIZE
        and sorted(records_by_rank) == list(range(WORLD_SIZE))
        and len(records) == WORLD_SIZE * len(STEADY_STRUCTURED_STEPS)
        and not parse_errors
        and not duplicates
        and not missing
    )
    return {
        "status": "passed" if valid else "failed",
        "paths": [str(path) for path in paths],
        "scope": "all 16 ranks, individual steps 6-25; timer values are milliseconds",
        "steady_steps": list(STEADY_STRUCTURED_STEPS),
        "rank_step_count": len(records),
        "observed_ranks": sorted(records_by_rank),
        "parse_errors": parse_errors,
        "duplicates": duplicates,
        "missing": missing,
        "rank_step_records": records,
        "components_ms": components,
    }


def structured_step_inventory(phase_root: Path, expected_steps: int) -> dict[str, Any]:
    paths = sorted(
        (phase_root / "job/structured_logs").glob("training.global_rank_*.jsonl")
    )
    by_rank: dict[int, list[int]] = {}
    parse_errors = []
    for path in paths:
        match = re.search(r"global_rank_(\d+)\.", path.name)
        fallback_rank = int(match.group(1)) if match else None
        with path.open(errors="replace") as source:
            for line_number, line in enumerate(source, 1):
                try:
                    row = json.loads(line)
                except Exception as error:
                    parse_errors.append(
                        {
                            "path": str(path),
                            "line": line_number,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                    continue
                if row.get("log_type_name") != "step_end":
                    continue
                rank = row.get("global_rank", fallback_rank)
                step = row.get("step")
                if isinstance(rank, int) and isinstance(step, int):
                    by_rank.setdefault(rank, []).append(step)
    expected = list(range(1, expected_steps + 1))
    mismatches = {
        str(rank): by_rank.get(rank, [])
        for rank in range(WORLD_SIZE)
        if by_rank.get(rank, []) != expected
    }
    valid = len(paths) == WORLD_SIZE and not parse_errors and not mismatches
    return {
        "status": "passed" if valid else "failed",
        "paths": [str(path) for path in paths],
        "expected_steps": expected,
        "observed_ranks": sorted(by_rank),
        "step_mismatches": mismatches,
        "parse_errors": parse_errors,
    }


def phase_execution(run_root: Path, phase: str) -> dict[str, Any]:
    phase_root = run_root / phase
    runtime = phase_root / "runtime"
    markers = {
        state: (run_root / "runtime" / f"{phase}.{state}").is_file()
        for state in ("completed", "oom", "failed")
    }
    exit_codes: dict[int, int] = {}
    malformed = []
    for path in sorted(runtime.glob("rank_*.exit_code")):
        match = re.fullmatch(r"rank_(\d+)\.exit_code", path.name)
        try:
            if match is None:
                raise ValueError(path.name)
            exit_codes[int(match.group(1))] = int(path.read_text().strip())
        except Exception as error:
            malformed.append({"path": str(path), "error": str(error)})
    logs = sorted(runtime.glob("rank_*.combined.log"))
    log_audits = []
    for path in logs:
        match = re.fullmatch(r"rank_(\d+)\.combined\.log", path.name)
        rank = int(match.group(1)) if match else -1
        text = path.read_text(errors="replace")
        log_audits.append(
            {
                "rank": rank,
                "path": str(path),
                "training_completed_count": text.count("Training completed"),
                "fatal_matches": sorted(set(FATAL.findall(text))),
                "startup_missing_cutlass_warning_count": text.count(
                    "No module named 'nvidia_cutlass_dsl'"
                ),
            }
        )
    valid = (
        markers == {"completed": True, "oom": False, "failed": False}
        and sorted(exit_codes) == list(range(WORLD_SIZE))
        and all(code == 0 for code in exit_codes.values())
        and not malformed
        and len(logs) == WORLD_SIZE
        and all(audit["training_completed_count"] == 1 for audit in log_audits)
        and all(not audit["fatal_matches"] for audit in log_audits)
    )
    return {
        "status": "passed" if valid else "failed",
        "markers": markers,
        "exit_codes": {str(rank): code for rank, code in sorted(exit_codes.items())},
        "malformed_exit_codes": malformed,
        "oom_evidence": [
            str(path) for path in sorted(runtime.glob("rank_*.oom_evidence"))
        ],
        "rank_logs": log_audits,
    }


def config_contract(phase_root: Path, phase: str) -> dict[str, Any]:
    path = phase_root / "job/config.json"
    raw = load_json(path)
    expected_steps = PHASES[phase]["steps"]
    expected_trace = PHASES[phase]["kind"] == "trace"
    expected: dict[tuple[str, ...], Any] = {
        ("model_spec", "name"): "muse_glimmer",
        ("model_spec", "flavor"): "30B",
        ("model_spec", "model", "dim"): 6656,
        ("model_spec", "model", "vocab_size"): 202048,
        ("training", "local_batch_size"): LOCAL_BATCH_SIZE,
        ("training", "global_batch_size"): -1,
        ("training", "seq_len"): SEQUENCE_LENGTH,
        ("training", "steps"): expected_steps,
        ("training", "dtype"): "float32",
        ("training", "mixed_precision_param"): "bfloat16",
        ("training", "mixed_precision_reduce"): "float32",
        ("parallelism", "data_parallel_replicate_degree"): 1,
        ("parallelism", "data_parallel_shard_degree"): FSDP_DEGREE,
        ("parallelism", "tensor_parallel_degree"): TP_DEGREE,
        ("parallelism", "enable_sequence_parallel"): True,
        ("parallelism", "context_parallel_degree"): 1,
        ("parallelism", "pipeline_parallel_degree"): 1,
        ("parallelism", "expert_parallel_degree"): 1,
        ("compile", "enable"): True,
        ("compile", "enable_async_tensor_parallel"): False,
        ("compile", "components"): ["model", "loss"],
        ("compile", "backend"): "inductor",
        ("dataloader", "dataset"): "muse_glimmer_c4_pinned_offline",
        ("dataloader", "num_workers"): 0,
        ("optimizer", "implementation"): "fused",
        ("profiler", "enable_profiling"): expected_trace,
        ("profiler", "enable_memory_snapshot"): False,
        ("metrics", "enable_tensorboard"): True,
        ("metrics", "log_freq"): 1 if expected_trace else 5,
    }
    if expected_trace:
        expected.update(
            {
                ("profiler", "profile_freq"): 6,
                ("profiler", "profiler_warmup"): 0,
                ("profiler", "profiler_active"): 3,
                ("profiler", "profiler_repeat"): 1,
            }
        )
    checks = {
        ".".join(field): {"expected": value, "observed": nested(raw, *field)}
        for field, value in expected.items()
    }
    layers = nested(raw, "model_spec", "model", "layers")
    checks["model_spec.model.layer_count"] = {
        "expected": 52,
        "observed": len(layers) if isinstance(layers, list) else None,
    }
    failures = [
        name for name, check in checks.items() if check["observed"] != check["expected"]
    ]
    return {
        "status": "passed" if not failures else "failed",
        "path": str(path),
        "sha256": sha256(path),
        "checks": checks,
        "failed": failures,
        "serialized": raw,
    }


def normalized_serialized_config(value: Any, path: tuple[str, ...] = ()) -> Any:
    if path in SERIALIZED_CONFIG_NORMALIZATIONS:
        return SERIALIZED_CONFIG_NORMALIZATIONS[path]
    if isinstance(value, dict):
        return {
            key: normalized_serialized_config(item, (*path, key))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [normalized_serialized_config(item, path) for item in value]
    if isinstance(value, str):
        return CALLABLE_ADDRESS.sub(" at 0xADDR", value)
    return value


def normalized_phase_config(raw: dict[str, Any]) -> dict[str, Any]:
    value = normalized_serialized_config(raw)
    value.setdefault("training", {})["steps"] = "<phase-specific>"
    value.setdefault("metrics", {})["log_freq"] = "<phase-specific>"
    value["profiler"] = "<phase-specific>"
    return value


def config_phase_parity(configs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if any(config.get("status") != "passed" for config in configs.values()):
        return {"status": "failed", "error": "one or more phase configs failed"}
    performance = normalized_phase_config(configs[PERFORMANCE_PHASE]["serialized"])
    trace = normalized_phase_config(configs[TRACE_PHASE]["serialized"])
    return {
        "status": "passed" if performance == trace else "failed",
        "equal_after_phase_normalization": performance == trace,
        "normalized_fields": [
            "dump_folder",
            "debug.save_config_file",
            "hf_assets_path",
            "callable repr addresses",
            "training.steps",
            "metrics.log_freq",
            "profiler",
        ],
    }


def normalized_saved_config(raw: dict[str, Any]) -> dict[str, Any]:
    return normalized_serialized_config(raw)


def preflight_config_parity(configs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    expected_paths = {
        PERFORMANCE_PHASE: TASK_ROOT
        / "preflight/native_serialized_configs_v1/world16.native.performance.json",
        TRACE_PHASE: TASK_ROOT
        / "preflight/native_serialized_configs_v1/world16.native.trace.json",
    }
    results = {}
    failed = []
    for phase, expected_path in expected_paths.items():
        actual = configs.get(phase, {})
        if actual.get("status") != "passed" or not expected_path.is_file():
            results[phase] = {
                "status": "failed",
                "actual": actual.get("path"),
                "expected": str(expected_path),
            }
            failed.append(phase)
            continue
        expected = load_json(expected_path)
        equal = normalized_saved_config(
            actual["serialized"]
        ) == normalized_saved_config(expected)
        results[phase] = {
            "status": "passed" if equal else "failed",
            "actual": actual["path"],
            "actual_sha256": actual["sha256"],
            "expected": str(expected_path),
            "expected_sha256": sha256(expected_path),
            "equal_after_output_path_normalization": equal,
        }
        if not equal:
            failed.append(phase)
    return {
        "status": "passed" if not failed else "failed",
        "phases": results,
        "failed": failed,
        "normalized_fields": [
            "dump_folder",
            "debug.save_config_file",
            "hf_assets_path",
            "callable repr addresses",
        ],
    }


def runtime_preflight_summary(run_root: Path) -> dict[str, Any]:
    root = run_root / "runtime_preflight"
    paths = sorted(root.glob("rank_*.json"))
    records: dict[int, dict[str, Any]] = {}
    errors = []
    for path in paths:
        try:
            record = load_json(path)
            rank = int(record["rank"])
            records[rank] = record
        except Exception as error:
            errors.append(
                {"path": str(path), "error": f"{type(error).__name__}: {error}"}
            )
    expected_compile = {
        "enable": True,
        "enable_async_tensor_parallel": False,
        "components": ["model", "loss"],
        "backend": "inductor",
    }
    invalid = []
    for rank in range(WORLD_SIZE):
        record = records.get(rank)
        if record is None:
            invalid.append({"rank": rank, "reason": "missing"})
            continue
        expected_verified_steps = 25 if rank % TP_DEGREE == 0 else 0
        conditions = {
            "status": record.get("status") == "passed",
            "world_size": record.get("world_size") == WORLD_SIZE,
            "node_count": record.get("node_count") == 2,
            "mesh": record.get("mesh") == {"fsdp": FSDP_DEGREE, "tp": TP_DEGREE},
            "batch": record.get("batch")
            == {
                "local_batch_size_per_dp_rank": LOCAL_BATCH_SIZE,
                "global_batch_size": GLOBAL_BATCH_SIZE,
                "per_physical_gpu_effective_batch_size": 1,
                "sequence_length": SEQUENCE_LENGTH,
                "tokens_per_step": TOKENS_PER_STEP,
            },
            "trainer_backend": record.get("trainer_backend") == "native_torchtitan",
            "parallelize_fn": record.get("parallelize_fn")
            == "torchtitan.models.muse_glimmer.parallelize.parallelize_muse_glimmer",
            "compile": record.get("native_compile") == expected_compile,
            "activation_checkpoint": record.get("activation_checkpoint")
            == "SelectiveAC",
            "attention": record.get("attention") == "packed_document_sdpa",
            "model": record.get("model")
            == {"dim": 6656, "layers": 52, "vocab_size": 202048},
            "torchtitan_revision": record.get("torchtitan_revision")
            == EXPECTED_TORCHTITAN_COMMIT,
            "torch_version": record.get("torch") == EXPECTED_TORCH_VERSION,
            "torch_commit": record.get("torch_commit") == EXPECTED_TORCH_COMMIT,
            "input_manifest": nested(record, "input_audit", "manifest_sha256")
            == EXPECTED_INPUT_MANIFEST_SHA256,
            "input_steps": nested(record, "input_audit", "verified_steps")
            == expected_verified_steps,
            "c4_revision": record.get("c4_revision")
            == "1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
        }
        failed = [name for name, passed in conditions.items() if not passed]
        if failed:
            invalid.append({"rank": rank, "failed": failed})
    fingerprints = {
        json.dumps(
            {
                key: record.get(key)
                for key in (
                    "python",
                    "torch",
                    "torch_commit",
                    "cuda",
                    "nccl",
                    "torchtitan_revision",
                    "native_source_hashes",
                    "native_compile",
                )
            },
            sort_keys=True,
        )
        for record in records.values()
    }
    manifest = TASK_ROOT / "runner/c4_input_manifest_lb2_s4096_dp8.jsonl"
    local_manifest_hash = sha256(manifest) if manifest.is_file() else None
    valid = (
        len(paths) == WORLD_SIZE
        and sorted(records) == list(range(WORLD_SIZE))
        and not errors
        and not invalid
        and len(fingerprints) == 1
        and local_manifest_hash == EXPECTED_INPUT_MANIFEST_SHA256
    )
    return {
        "status": "passed" if valid else "failed",
        "paths": [str(path) for path in paths],
        "observed_ranks": sorted(records),
        "parse_errors": errors,
        "invalid_records": invalid,
        "uniform_environment_fingerprint": len(fingerprints) == 1,
        "environment_fingerprint": (
            json.loads(next(iter(fingerprints))) if len(fingerprints) == 1 else None
        ),
        "input_manifest": {
            "path": str(manifest),
            "sha256": local_manifest_hash,
            "expected_sha256": EXPECTED_INPUT_MANIFEST_SHA256,
        },
    }


def allocation_identity(record: dict[str, Any]) -> dict[str, Any]:
    return {field: record.get(field) for field in ALLOCATION_FIELDS}


def allocation_summary(run_root: Path) -> dict[str, Any]:
    phase_records: dict[str, dict[int, dict[str, Any]]] = {}
    errors = []
    for phase in PHASES:
        records = {}
        for rank in range(WORLD_SIZE):
            path = run_root / "allocation" / phase / f"rank_{rank:03d}.json"
            try:
                records[rank] = load_json(path)
            except Exception as error:
                errors.append(
                    {"path": str(path), "error": f"{type(error).__name__}: {error}"}
                )
        phase_records[phase] = records
    mismatches = []
    reference = phase_records[PERFORMANCE_PHASE]
    for phase, records in phase_records.items():
        for rank in range(WORLD_SIZE):
            if rank not in reference or rank not in records:
                continue
            if allocation_identity(records[rank]) != allocation_identity(
                reference[rank]
            ):
                mismatches.append({"phase": phase, "rank": rank})
    reference_values = list(reference.values())
    hosts = sorted({str(record.get("hostname")) for record in reference_values})
    uuids = {record.get("gpu_uuid") for record in reference_values}
    invalid_reference = [
        rank
        for rank, record in reference.items()
        if record.get("rank") != rank
        or record.get("world_size") != WORLD_SIZE
        or record.get("local_world_size") != 8
        or record.get("datacenter") != "pci1"
        or "H100" not in str(record.get("gpu_name"))
    ]
    valid = (
        not errors
        and not mismatches
        and not invalid_reference
        and all(len(records) == WORLD_SIZE for records in phase_records.values())
        and len(hosts) == 2
        and len(uuids) == WORLD_SIZE
    )
    return {
        "status": "passed" if valid else "failed",
        "hosts": hosts,
        "unique_gpu_uuid_count": len(uuids),
        "phase_record_counts": {
            phase: len(records) for phase, records in phase_records.items()
        },
        "cross_phase_mismatches": mismatches,
        "invalid_reference_ranks": invalid_reference,
        "parse_errors": errors,
        "rank_mapping": [
            allocation_identity(reference[rank]) for rank in sorted(reference)
        ],
    }


def scheduler_summary(attempt: Path, mode: str) -> dict[str, Any]:
    root = attempt / "submission" / mode
    status_path = root / "job_status.final.json"
    definition_path = root / "job_definition.final.json"
    if not status_path.is_file() or not definition_path.is_file():
        return {
            "status": "failed",
            "error": "terminal status or final job definition is missing",
            "status_path": str(status_path),
            "definition_path": str(definition_path),
        }
    status_json = load_json(status_path)
    data = status_json.get("data", {})
    latest = data.get("latestAttempt", {})
    groups = [
        group
        for values in latest.get("taskGroupExecutionAttempts", {}).values()
        for group in values
    ]
    tasks = [
        attempts[-1]
        for group in groups
        for attempts in group.get("taskExecutionAttempts", {}).values()
        if attempts
    ]
    definition_json = load_json(definition_path)
    definition = definition_json.get("data", definition_json)
    hpc_groups = definition.get("hpcTaskGroups", [])
    spec = hpc_groups[0].get("spec", {}) if len(hpc_groups) == 1 else {}
    expected_ports = {} if mode == "gate" else {"training_phase_2": 29501}
    expected_tail = [mode, "16", "2", "single_native", "sequential", "1"]
    env = spec.get("env", {})
    checks = {
        "job_complete": data.get("state") == "COMPLETE",
        "latest_attempt_complete": latest.get("state") == "COMPLETE",
        "zero_restarts": data.get("numRestarts") == 0,
        "one_complete_task_group": len(groups) == 1
        and groups[0].get("state") == "COMPLETE"
        and groups[0].get("numFailedTasks") == 0,
        "two_complete_hosts": len(tasks) == 2
        and all(
            task.get("state") == "COMPLETE" and task.get("exitCode") == 0
            for task in tasks
        ),
        "ports": spec.get("ports", {}) == expected_ports,
        "runner_arguments": spec.get("arguments", [])[-6:] == expected_tail,
        "module": env.get("MODULE") == EXPECTED_MODULE,
        "config": env.get("CONFIG") == EXPECTED_CONFIG,
        "two_nodes": (
            hpc_groups[0].get("taskCount") == 2 if len(hpc_groups) == 1 else False
        ),
        "eight_gpus_per_host": nested(spec, "resourceLimit", "compute", "gpu") == 8,
        "zero_job_retries": definition.get("maxJobFailures") == 0
        and nested(spec, "restartPolicy", "maxTotalFailures") == 0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "status": "passed" if not failed else "failed",
        "status_path": str(status_path),
        "definition_path": str(definition_path),
        "checks": checks,
        "failed": failed,
        "job_name": data.get("hpcJobName"),
        "job_version": data.get("version"),
        "attempt_index": latest.get("attemptIndex"),
        "hosts": sorted({task.get("hostname") for task in tasks}),
        "packages": [
            package.get("fbpkgIdentifier")
            for package in spec.get("applicationPackages", [])
        ],
    }


def package_preflight_summary(attempt: Path) -> dict[str, Any]:
    paths = {
        "formal_dryrun": attempt / "preflight/formal_dryrun_validation.json",
        "formal_dryrun_packages": attempt
        / "preflight/formal_dryrun_package_validation.json",
        "formal_submitted_packages": attempt
        / "preflight/formal_submitted_package_validation.json",
        "native_serialized_configs": TASK_ROOT
        / "preflight/native_serialized_configs_v1/validation.json",
    }
    values = {}
    failed = []
    for name, path in paths.items():
        if not path.is_file():
            values[name] = {"status": "missing", "path": str(path)}
            failed.append(name)
            continue
        value = load_json(path)
        passed = value.get("status") == "passed" or value.get("valid") is True
        values[name] = {
            "status": "passed" if passed else "failed",
            "path": str(path),
            "value": value,
        }
        if not passed:
            failed.append(name)
    return {
        "status": "passed" if not failed else "failed",
        "checks": values,
        "failed": failed,
    }


def merge(intervals: list[tuple[float, float]]) -> list[list[float]]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def union_length(intervals: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in merge(intervals))


def subtract_length(
    intervals: list[tuple[float, float]], blockers: list[tuple[float, float]]
) -> float:
    merged_blockers = merge(blockers)
    total = 0.0
    for start, end in merge(intervals):
        cursor = start
        for block_start, block_end in merged_blockers:
            if block_end <= cursor:
                continue
            if block_start >= end:
                break
            total += max(0.0, min(block_start, end) - cursor)
            cursor = max(cursor, block_end)
            if cursor >= end:
                break
        total += max(0.0, end - cursor)
    return total


def kernel_category(name: str, args: dict[str, Any]) -> str:
    lower = name.lower()
    if "Collective name" in args or "nccl" in lower:
        return "communication"
    if "fusedoptimizer" in lower or "fusedadammath" in lower:
        return "optimizer"
    if any(
        token in lower for token in ("catarraybatchedcopy", "pre_bucket", "post_bucket")
    ):
        return "bucket_pack"
    if any(token in lower for token in ("nvjet_", "gemm", "cutlass")):
        return "gemm"
    if any(token in lower for token in ("fmha", "cudnn", "attention")):
        return "attention"
    if any(
        token in lower
        for token in ("copyfunctor", "direct_copy", "convert_element_type")
    ):
        return "cast_copy"
    return "other"


def kineto_summary(phase_root: Path, output_dir: Path) -> dict[str, Any]:
    paths = sorted((phase_root / "job/profiling/traces").glob("**/rank0_trace.json.gz"))
    if len(paths) != 1:
        return {
            "status": "failed",
            "error": f"expected one rank-0 Kineto trace, found {len(paths)}",
            "paths": [str(path) for path in paths],
        }
    path = paths[0]
    with gzip.open(path, "rt") as source:
        trace = json.load(source)
    events = trace.get("traceEvents", [])
    step_name = f"ProfilerStep#{PROFILER_STEP}"
    step_events = [
        event
        for event in events
        if event.get("name") == step_name
        and event.get("cat") in {"user_annotation", "gpu_user_annotation"}
        and event.get("ph") == "X"
        and isinstance(event.get("ts"), (int, float))
        and isinstance(event.get("dur"), (int, float))
    ]
    cpu_steps = [
        event for event in step_events if event.get("cat") == "user_annotation"
    ]
    gpu_steps = [
        event for event in step_events if event.get("cat") == "gpu_user_annotation"
    ]
    if len(cpu_steps) != 1:
        raise RuntimeError(
            f"expected exactly one CPU {step_name}, found {len(cpu_steps)}"
        )
    selected = cpu_steps[0]
    step_start = float(selected["ts"])
    step_end = step_start + float(selected["dur"])
    kernels: list[tuple[float, float, bool]] = []
    collectives: dict[tuple[str, str, int, str], tuple[int, float]] = {}
    categories: dict[str, tuple[int, float]] = {}
    kernel_names: dict[str, tuple[int, float]] = {}
    memory_event_count = 0
    counter_event_count = 0
    memory_counter_event_count = 0
    allocator_event_count = 0
    for event in events:
        name = str(event.get("name", ""))
        if name == "[memory]":
            memory_event_count += 1
        if event.get("ph") == "C":
            counter_event_count += 1
            counter_args = event.get("args", {})
            if "memory" in name.lower() or any(
                key in counter_args
                for key in ("Total Allocated", "Total Reserved", "Bytes")
            ):
                memory_counter_event_count += 1
        if name in {"cudaMalloc", "cudaFree", "cudaMallocAsync", "cudaFreeAsync"}:
            allocator_event_count += 1
        if event.get("cat") != "kernel" or "ts" not in event or "dur" not in event:
            continue
        event_start = float(event["ts"])
        event_end = event_start + float(event["dur"])
        if event_end <= step_start or event_start >= step_end:
            continue
        start = max(step_start, event_start)
        end = min(step_end, event_end)
        duration = end - start
        args = event.get("args", {})
        is_collective = "Collective name" in args or "nccl" in name.lower()
        kernels.append((start, end, is_collective))
        count, raw = kernel_names.get(name, (0, 0.0))
        kernel_names[name] = (count + 1, raw + duration)
        category = kernel_category(name, args)
        count, raw = categories.get(category, (0, 0.0))
        categories[category] = (count + 1, raw + duration)
        if is_collective:
            key = (
                str(args.get("Collective name", "nccl")),
                str(args.get("dtype", "?")),
                int(args.get("Group size", 0) or 0),
                str(args.get("Process Group Description", "?")),
            )
            count, raw = collectives.get(key, (0, 0.0))
            collectives[key] = (count + 1, raw + duration)
    all_intervals = [(start, end) for start, end, _ in kernels]
    comm_intervals = [(start, end) for start, end, comm in kernels if comm]
    compute_intervals = [(start, end) for start, end, comm in kernels if not comm]
    kept = []
    for event in events:
        timestamp = event.get("ts")
        duration = event.get("dur", 0)
        overlaps = (
            isinstance(timestamp, (int, float))
            and isinstance(duration, (int, float))
            and timestamp < step_end
            and timestamp + duration > step_start
        )
        if event.get("ph") == "M" or (
            overlaps
            and (
                event.get("cat") in {"kernel", "gpu_user_annotation", "user_annotation"}
                or event.get("ph") in {"s", "t", "f"}
            )
        ):
            kept.append(event)
    slim = {key: value for key, value in trace.items() if key != "traceEvents"}
    slim["traceEvents"] = kept
    slim_path = output_dir / "slim_traces/native.profiler_step_4.json.gz"
    slim_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(slim_path, "wt") as destination:
        json.dump(slim, destination, separators=(",", ":"))
    category_output = {
        name: {"count": count, "raw_ms": raw / 1000}
        for name, (count, raw) in sorted(categories.items())
    }
    for required in ("attention", "gemm", "communication"):
        category_output.setdefault(required, {"count": 0, "raw_ms": 0.0})
    return {
        "status": "passed",
        "path": str(path),
        "sha256": sha256(path),
        "compressed_bytes": path.stat().st_size,
        "profiler_step": PROFILER_STEP,
        "step_duration_ms": (step_end - step_start) / 1000,
        "cpu_step_annotation_count": len(cpu_steps),
        "gpu_step_projection_count": len(gpu_steps),
        "kernel_count": len(kernels),
        "gpu_busy_union_ms": union_length(all_intervals) / 1000,
        "noncommunication_union_ms": union_length(compute_intervals) / 1000,
        "communication_raw_ms": sum(end - start for start, end in comm_intervals)
        / 1000,
        "communication_union_ms": union_length(comm_intervals) / 1000,
        "communication_exposed_ms": subtract_length(comm_intervals, compute_intervals)
        / 1000,
        "kernel_categories": category_output,
        "collectives": [
            {
                "collective": key[0],
                "dtype": key[1],
                "group_size": key[2],
                "group": key[3],
                "count": count,
                "raw_ms": raw / 1000,
            }
            for key, (count, raw) in sorted(
                collectives.items(), key=lambda item: item[1][1], reverse=True
            )
        ],
        "top_kernels": [
            {"name": name, "count": count, "raw_ms": raw / 1000}
            for name, (count, raw) in sorted(
                kernel_names.items(), key=lambda item: item[1][1], reverse=True
            )[:40]
        ],
        "memory_timeline": {
            "profile_memory_available": bool(
                memory_event_count or memory_counter_event_count
            ),
            "memory_event_count": memory_event_count,
            "counter_event_count": counter_event_count,
            "memory_counter_event_count": memory_counter_event_count,
            "allocator_event_count": allocator_event_count,
            "note": (
                "This trace can localize allocator activity."
                if memory_event_count or memory_counter_event_count
                else "Profiler memory events are absent; use runtime peak scalars for memory."
            ),
        },
        "slim_trace": str(slim_path),
    }


def tlparse_summary(
    phase_root: Path, output_dir: Path, tlparse_bin: Path
) -> dict[str, Any]:
    paths = sorted(
        phase_root.glob("compile_trace/dedicated_log_torch_trace_rank_0_*.log")
    )
    if len(paths) != 1:
        return {
            "status": "failed",
            "error": f"expected one rank-0 TORCH_TRACE log, found {len(paths)}",
            "paths": [str(path) for path in paths],
        }
    trace_path = paths[0]
    if not tlparse_bin.is_file() or not tlparse_bin.stat().st_mode & 0o111:
        return {
            "status": "failed",
            "error": f"tlparse executable unavailable: {tlparse_bin}",
            "torch_trace": str(trace_path),
        }
    root = output_dir / "tlparse" / TRACE_PHASE
    root.parent.mkdir(parents=True, exist_ok=True)
    command = [str(tlparse_bin), "--no-browser", "-p", "-o", str(root), str(trace_path)]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    stdout_path = output_dir / "tlparse" / f"{TRACE_PHASE}.stdout.log"
    stderr_path = output_dir / "tlparse" / f"{TRACE_PHASE}.stderr.log"
    returncode_path = output_dir / "tlparse" / f"{TRACE_PHASE}.returncode"
    stdout_path.write_text(completed.stdout)
    stderr_path.write_text(completed.stderr)
    returncode_path.write_text(f"{completed.returncode}\n")
    version = subprocess.run(
        [str(tlparse_bin), "--version"], text=True, capture_output=True, check=False
    )
    version_path = output_dir / "tlparse/tlparse.version.txt"
    version_path.write_text((version.stdout or version.stderr).strip() + "\n")
    files = (
        [path for path in root.rglob("*") if path.is_file()] if root.is_dir() else []
    )
    kinds = {
        "inductor_output_code": sorted(root.rglob("inductor_output_code_*.txt")),
        "after_joint_graph": sorted(root.rglob("after_joint_graph_*.txt")),
        "before_joint_graph": sorted(root.rglob("before_joint_graph_*.txt")),
        "activation_memory_policy": sorted(
            root.rglob("activation_memory_policy_*.txt")
        ),
        "raw_log": sorted(root.rglob("raw.log")),
        "index": sorted(root.rglob("index.html")),
    }
    valid = completed.returncode == 0 and bool(files) and bool(kinds["index"])
    return {
        "status": "passed" if valid else "failed",
        "command": command,
        "returncode": completed.returncode,
        "torch_trace": {
            "path": str(trace_path),
            "bytes": trace_path.stat().st_size,
            "sha256": sha256(trace_path),
        },
        "tlparse_binary": str(tlparse_bin),
        "version": (version.stdout or version.stderr).strip(),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "returncode_file": str(returncode_path),
        "output_root": str(root),
        "file_count": len(files),
        "artifact_counts": {name: len(paths) for name, paths in kinds.items()},
        "artifacts": {
            name: [str(path) for path in paths] for name, paths in kinds.items()
        },
    }


def git_state(repo: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()

    head = run("rev-parse", "HEAD")
    status = run("status", "--short")
    try:
        upstream = run("rev-parse", "origin/main")
        counts = run("rev-list", "--left-right", "--count", "origin/main...HEAD")
    except subprocess.CalledProcessError:
        upstream = None
        counts = None
    return {
        "path": str(repo),
        "head": head,
        "expected_head": (
            EXPECTED_TORCHTITAN_COMMIT if repo.name == "torchtitan" else None
        ),
        "clean": not status,
        "status_short": status.splitlines(),
        "origin_main": upstream,
        "origin_main_left_right_count": counts,
    }


def render_markdown(report: dict[str, Any]) -> str:
    tb = report.get("tensorboard", {})
    measurements = tb.get("measurements", {})
    structured = report.get("structured_logs", {}).get("components_ms", {})
    kineto = report.get("kineto", {})

    lines = [
        "# Muse Glimmer native TorchTitan formal analysis",
        "",
        f"Generated: {report['generated_utc']}",
        "",
        f"Overall validation: `{report['status']}`",
        "",
        "This report contains absolute native-TorchTitan measurements only. It does not "
        "claim a speedup or memory improvement versus an AutoParallel job on another allocation.",
        "",
        "## Workload",
        "",
        "- Muse Glimmer 30B, packed-document SDPA",
        "- 16 H100 GPUs, FSDP8 x TP2",
        "- sequence length 4096, local batch 2 per DP rank, global batch 16",
        "- BF16 parameters, FP32 reductions, SelectiveAC",
        "- native TorchTitan per-TransformerBlock Inductor compilation",
        "",
        "## Rank-0 TensorBoard steady windows",
        "",
        "| Metric | Mean | Sample std | Min | Max |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, key, scale, unit in (
        ("End-to-end latency", "latency_s", 1000.0, "ms"),
        ("Data loading", "data_loading_s", 1000.0, "ms"),
        ("Non-data time", "non_data_s", 1000.0, "ms"),
        ("Per-device throughput", "per_device_tps", 1.0, "tok/s"),
        ("Aggregate throughput", "aggregate_tps", 1.0, "tok/s"),
    ):
        summary = measurements.get(key, {}).get("summary")
        if summary:
            lines.append(
                f"| {label} | {summary['mean'] * scale:.3f} {unit} | "
                f"{summary['sample_std'] * scale:.3f} | {summary['min'] * scale:.3f} | "
                f"{summary['max'] * scale:.3f} |"
            )
    active = measurements.get("active_memory_gib", {}).get("summary")
    reserved = measurements.get("reserved_memory_gib", {}).get("summary")
    lines.extend(["", "## Rank-0 memory", ""])
    if active and reserved:
        lines.extend(
            [
                f"- Active peak: {active['max']:.6f} GiB",
                f"- Reserved peak: {reserved['max']:.6f} GiB",
                "- Source: CUDA allocator peak scalars emitted by TorchTitan; the profiler "
                "trace is not assumed to contain allocation events.",
            ]
        )
    else:
        lines.append("Memory scalars unavailable.")

    lines.extend(
        [
            "",
            "## All-rank structured timers, steps 6-25",
            "",
            "| Component | All rank-step mean | Sample std | Per-step slowest-rank mean |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in (
        "step_end",
        "fetching_batch_end",
        "post_dataloading_process_end",
        "optim_end",
        "collect_dist_metrics_end",
        "checkpoint_save_end",
        "gc_collect_end",
        "instrumented_noncore_ms",
        "unattributed_training_core_ms",
    ):
        component = structured.get(name, {})
        all_rank = component.get("all_rank_steps")
        critical = component.get("per_step_rank_max")
        if all_rank and critical:
            lines.append(
                f"| `{name}` | {all_rank['mean']:.3f} ms | "
                f"{all_rank['sample_std']:.3f} | {critical['mean']:.3f} ms |"
            )

    lines.extend(["", "## Kineto ProfilerStep#4", ""])
    if kineto.get("status") == "passed":
        lines.extend(
            [
                f"- Step duration: {kineto['step_duration_ms']:.3f} ms",
                f"- GPU busy union: {kineto['gpu_busy_union_ms']:.3f} ms",
                f"- Communication raw/union/exposed: {kineto['communication_raw_ms']:.3f} / "
                f"{kineto['communication_union_ms']:.3f} / {kineto['communication_exposed_ms']:.3f} ms",
                f"- Attention raw: {kineto['kernel_categories']['attention']['raw_ms']:.3f} ms",
                f"- GEMM raw: {kineto['kernel_categories']['gemm']['raw_ms']:.3f} ms",
                "",
                "Largest collective families:",
                "",
            ]
        )
        for collective in kineto.get("collectives", [])[:10]:
            lines.append(
                f"- {collective['collective']} / {collective['dtype']} / group "
                f"{collective['group_size']} `{collective['group']}`: "
                f"{collective['count']} calls, {collective['raw_ms']:.3f} ms raw"
            )
    else:
        lines.append(f"Kineto unavailable: {kineto.get('error', 'unknown error')}")

    tlparse = report.get("tlparse", {})
    lines.extend(["", "## TORCH_TRACE and tlparse", ""])
    lines.append(f"- Status: `{tlparse.get('status', 'missing')}`")
    if tlparse.get("torch_trace"):
        lines.append(f"- TORCH_TRACE: `{tlparse['torch_trace']['path']}`")
    if tlparse.get("output_root"):
        lines.append(f"- tlparse output: `{tlparse['output_root']}`")
    if tlparse.get("artifact_counts"):
        lines.append(
            f"- Artifact counts: `{json.dumps(tlparse['artifact_counts'], sort_keys=True)}`"
        )

    lines.extend(["", "## Evidence", ""])
    for label, path in (
        ("Run root", report.get("run_root")),
        ("Full JSON", report.get("outputs", {}).get("analysis_json")),
        ("Validation JSON", report.get("outputs", {}).get("validation_json")),
    ):
        lines.append(f"- {label}: `{path}`")
    lines.extend(
        [
            "",
            "## Interpretation limit",
            "",
            "This is a single native arm. Cross-job comparisons with the earlier "
            "AutoParallel run are not hardware-paired and must not be reported as a "
            "measured speedup or memory improvement.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attempt",
        type=Path,
        default=TASK_ROOT / "job/attempts/001",
        help="MAST attempt directory",
    )
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tlparse-bin",
        type=Path,
        default=None,
        help="tlparse executable; defaults to tlparse from PATH",
    )
    args = parser.parse_args()

    if args.tlparse_bin is None:
        tlparse_path = shutil.which("tlparse")
        if tlparse_path is None:
            raise SystemExit("tlparse is not on PATH; pass --tlparse-bin")
        args.tlparse_bin = Path(tlparse_path)

    attempt = args.attempt.resolve()
    run_root = args.run_root.resolve() if args.run_root else discover_run_root(attempt)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(
            f"refusing to overwrite existing output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True)

    expected_phases = sorted(PHASES)
    observed_phases = sorted(
        path.name
        for path in run_root.iterdir()
        if path.is_dir() and re.match(r"^\d{2}_", path.name)
    )
    phase_inventory = {
        "status": "passed" if observed_phases == expected_phases else "failed",
        "expected": expected_phases,
        "observed": observed_phases,
        "missing": sorted(set(expected_phases) - set(observed_phases)),
        "unexpected": sorted(set(observed_phases) - set(expected_phases)),
    }

    execution = {
        phase: capture(
            f"execution:{phase}", lambda phase=phase: phase_execution(run_root, phase)
        )
        for phase in PHASES
    }
    step_inventory = {
        phase: capture(
            f"structured_steps:{phase}",
            lambda phase=phase: structured_step_inventory(
                run_root / phase, PHASES[phase]["steps"]
            ),
        )
        for phase in PHASES
    }
    configs = {
        phase: capture(
            f"config:{phase}",
            lambda phase=phase: config_contract(run_root / phase, phase),
        )
        for phase in PHASES
    }

    analysis_path = output_dir / "formal_analysis.json"
    validation_path = output_dir / "formal_validation.json"
    summary_path = output_dir / "summary.md"
    report: dict[str, Any] = {
        "status": "pending",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "scope": "absolute native TorchTitan result; no cross-job performance claim",
        "task_root": str(TASK_ROOT),
        "attempt": str(attempt),
        "run_root": str(run_root),
        "workload": {
            "model": "Muse Glimmer 30B",
            "attention": "packed-document SDPA",
            "world_size": WORLD_SIZE,
            "mesh": {"fsdp": FSDP_DEGREE, "tp": TP_DEGREE},
            "sequence_length": SEQUENCE_LENGTH,
            "local_batch_size_per_dp_rank": LOCAL_BATCH_SIZE,
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "tokens_per_step": TOKENS_PER_STEP,
            "trainer_backend": "native TorchTitan Trainer",
            "compile": "per-TransformerBlock torch.compile, backend=inductor, fullgraph=True",
            "activation_checkpoint": "SelectiveAC",
        },
        "measurement": {
            "tensorboard": "rank 0 windows ending at steps 10, 15, 20, 25",
            "structured_logs": "all ranks and individual steady steps 6-25",
            "kineto": "rank 0 ProfilerStep#4 from the separate six-step trace phase",
            "tlparse": "rank 0 TORCH_TRACE from the separate six-step trace phase",
            "memory": "TorchTitan CUDA allocator active/reserved peak scalars on rank 0",
        },
        "phase_inventory": phase_inventory,
        "phase_execution": execution,
        "structured_step_inventory": step_inventory,
        "configs": configs,
        "config_phase_parity": capture(
            "config_phase_parity", lambda: config_phase_parity(configs)
        ),
        "preflight_config_parity": capture(
            "preflight_config_parity", lambda: preflight_config_parity(configs)
        ),
        "runtime_preflight": capture(
            "runtime_preflight", lambda: runtime_preflight_summary(run_root)
        ),
        "allocation": capture("allocation", lambda: allocation_summary(run_root)),
        "scheduler": {
            "gate": capture(
                "scheduler:gate", lambda: scheduler_summary(attempt, "gate")
            ),
            "formal": capture(
                "scheduler:formal", lambda: scheduler_summary(attempt, "formal")
            ),
        },
        "package_preflight": capture(
            "package_preflight", lambda: package_preflight_summary(attempt)
        ),
        "artifact_copy": {
            "status": (
                "passed"
                if (run_root.parent.parent / "run_output.copy_complete").is_file()
                else "failed"
            ),
            "marker": str(run_root.parent.parent / "run_output.copy_complete"),
        },
        "source": {
            "torchtitan": capture(
                "git:torchtitan", lambda: git_state(TASK_ROOT / "source/torchtitan")
            ),
            "autoparallel": capture(
                "git:autoparallel", lambda: git_state(TASK_ROOT / "source/autoparallel")
            ),
        },
        "tensorboard": capture(
            "tensorboard", lambda: tensorboard_summary(run_root / PERFORMANCE_PHASE)
        ),
        "structured_logs": capture(
            "structured_logs",
            lambda: structured_logs_summary(run_root / PERFORMANCE_PHASE),
        ),
        "kineto": capture(
            "kineto", lambda: kineto_summary(run_root / TRACE_PHASE, output_dir)
        ),
        "tlparse": capture(
            "tlparse",
            lambda: tlparse_summary(
                run_root / TRACE_PHASE, output_dir, args.tlparse_bin.resolve()
            ),
        ),
        "outputs": {
            "analysis_json": str(analysis_path),
            "validation_json": str(validation_path),
            "summary_markdown": str(summary_path),
        },
        "comparison_policy": {
            "cross_job_speedup_allowed": False,
            "reason": (
                "This native result does not share a physical allocation and rank-to-GPU "
                "mapping with the earlier AutoParallel formal run."
            ),
        },
    }

    required_sections = {
        "phase_inventory": report["phase_inventory"],
        **{f"phase_execution.{name}": value for name, value in execution.items()},
        **{
            f"structured_step_inventory.{name}": value
            for name, value in step_inventory.items()
        },
        **{f"configs.{name}": value for name, value in configs.items()},
        "config_phase_parity": report["config_phase_parity"],
        "preflight_config_parity": report["preflight_config_parity"],
        "runtime_preflight": report["runtime_preflight"],
        "allocation": report["allocation"],
        "scheduler.gate": report["scheduler"]["gate"],
        "scheduler.formal": report["scheduler"]["formal"],
        "package_preflight": report["package_preflight"],
        "artifact_copy": report["artifact_copy"],
        "tensorboard": report["tensorboard"],
        "structured_logs": report["structured_logs"],
        "kineto": report["kineto"],
        "tlparse": report["tlparse"],
    }
    checks = {
        name: section.get("status") == "passed"
        for name, section in required_sections.items()
    }
    source_checks = {
        f"source.{name}": value.get("clean") is True
        and (name != "torchtitan" or value.get("head") == EXPECTED_TORCHTITAN_COMMIT)
        for name, value in report["source"].items()
    }
    checks.update(source_checks)
    run_match = RUN_ROOT_PATTERN.fullmatch(run_root.name)
    formal_scheduler = report["scheduler"]["formal"]
    checks["run_root_attempt_matches_scheduler"] = bool(run_match) and (
        int(run_match.group("version")) == formal_scheduler.get("job_version")
        and int(run_match.group("attempt_index"))
        == formal_scheduler.get("attempt_index")
    )
    failed = [name for name, passed in checks.items() if not passed]
    report["status"] = "passed" if not failed else "failed"
    validation = {
        "status": report["status"],
        "checks": checks,
        "failed": failed,
        "analysis_json": str(analysis_path),
        "summary_markdown": str(summary_path),
    }
    analysis_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    summary_path.write_text(render_markdown(report))
    print(
        json.dumps(
            {"status": report["status"], "failed": failed, "output": str(output_dir)}
        )
    )
    raise SystemExit(0 if not failed else 1)


if __name__ == "__main__":
    main()
