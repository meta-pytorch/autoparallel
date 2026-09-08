#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import statistics
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any


TASK_ROOT = Path(__file__).resolve().parents[1]
WORLD_SIZE = 16
FSDP_DEGREE = 8
TP_DEGREE = 2
SEQUENCE_LENGTH = 4096
STEADY_STEPS = (10, 15, 20, 25)
PROFILER_STEP = 4
EXPECTED_TORCHTITAN_COMMIT = "c59ce51a6fc4f2340d320fe914a7c2049b747c8f"
EXPECTED_TORCH_COMMIT = "77bc9b492a38ec3122bb510f588a1321f4038e5c"
EXPECTED_TORCH_VERSION = "2.14.0a0+git77bc9b4"
RUN_ROOT_PATTERN = re.compile(
    r"muse_glimmer_30b_sdpa_fix_sac_aligned_lb2_s4096_2x8_"
    r"formal_r\d+_v(?P<version>\d+)_a(?P<attempt_index>\d+)"
)
EXPECTED_GRAPH_TRAINER_HASHES = {
    "torchtitan/experiments/graph_trainer/configs.py": (
        "ce59b39c4112220d520754c2d01aa8762256f39552d88e0b572fb537cd78edf6"
    ),
    "torchtitan/experiments/graph_trainer/passes.py": (
        "600b0e24216c9c256955fc2b2b079b3679cb64c020dd0026a007644061f71274"
    ),
    "torchtitan/experiments/graph_trainer/fsdp_passes.py": (
        "de73f223323eab6117763c76b2460897cc1c8d65951c362e1eb81766e5458018"
    ),
}
EXPECTED_AP_FIX_HASHES = {
    "autoparallel/shardings/dtensor_sharding_helpers.py": (
        "ff66549a82ebfb158b4c479e429a9a21710f5fae28432c89e94d2390cfba8bbc"
    )
}
EXPECTED_SAC_PREFIX = [
    "eliminate_dead_code_pass",
    "canonicalize_graph_pass",
    "deduplicate_fsdp_unshard_chains_pass",
    "tag_with_memory_policy_pass",
    "apply_cpu_offload_pass",
    "selective_activation_remat_pass",
]
CASES = {
    "per_gpu_bs1": {
        "local_batch_size_per_dp_rank": 2,
        "global_batch_size": 16,
        "per_physical_gpu_effective_batch_size": 1,
        "tokens_per_step": 65_536,
        "performance": {
            "manual": "01_per_gpu_bs1_graphtrainer",
            "autoparallel": "02_per_gpu_bs1_graphtrainer_ap",
        },
        "trace": {
            "manual": "11_trace_per_gpu_bs1_graphtrainer",
            "autoparallel": "12_trace_per_gpu_bs1_graphtrainer_ap",
        },
        "required_sdpa_modes": {
            "manual": "tp_head_shard",
            "autoparallel": None,
        },
    },
}
ALL_PHASES = tuple(
    phase
    for case in CASES.values()
    for phase_kind in ("performance", "trace")
    for phase in case[phase_kind].values()
)
PATH_FIELDS = {("dump_folder",), ("debug", "save_config_file")}
INTERESTING_ARTIFACTS = {
    "activation_memory_policy",
    "after_joint_graph",
    "autoparallel_parallel_graph",
    "autoparallel_solution",
    "fx_bucketing_passes_all_gather_buckets",
    "fx_bucketing_passes_reduce_scatter_buckets",
    "pre_bucketing_fsdp_collectives",
    "overlap_scheduling_graph_before",
    "overlap_scheduling_graph_after",
    "inductor_collective_schedule",
}
REPRESENTATIVE_PARAMS = {
    "tok_embeddings.embedding.weight",
    "layers.0.attention.qkv_linear.wq.weight",
    "layers.0.attention.qkv_linear.wk.weight",
    "layers.0.attention.qkv_linear.wv.weight",
    "layers.0.attention.wo.weight",
    "layers.0.attention.o_gate.weight",
    "layers.0.feed_forward.w1.weight",
    "layers.0.feed_forward.w2.weight",
    "layers.0.feed_forward.w3.weight",
    "layers.0.attention_norm.weight",
    "norm.weight",
}
NUMBER = r"(?:[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[+-]?(?:nan|inf))"


def summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        raise RuntimeError("cannot summarize an empty sample")
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "samples": values,
    }


def normalize_config(value: object, path: tuple[str, ...] = ()) -> object:
    if path in PATH_FIELDS:
        return "<phase-output-path>"
    if isinstance(value, dict):
        return {
            key: normalize_config(child, (*path, key)) for key, child in value.items()
        }
    if isinstance(value, list):
        return [normalize_config(child, path) for child in value]
    if isinstance(value, str):
        return re.sub(r"0x[0-9a-fA-F]+", "0xADDR", value)
    return value


def config_differences(
    left: object, right: object, path: tuple[str, ...] = ()
) -> list[dict[str, object]]:
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(set(left) | set(right)):
            result.extend(
                config_differences(
                    left.get(key, "<missing>"),
                    right.get(key, "<missing>"),
                    (*path, key),
                )
            )
        return result
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [{"path": ".".join(path), "left": left, "right": right}]
        result = []
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            result.extend(
                config_differences(left_item, right_item, (*path, str(index)))
            )
        return result
    if left == right:
        return []
    return [{"path": ".".join(path), "left": left, "right": right}]


def nested(value: dict[str, Any], *path: str) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
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
            raise RuntimeError(f"unsupported protobuf wire type {wire}")
        yield field, wire, value


def scalar_values(event_path: Path) -> dict[str, dict[int, float]]:
    result: dict[str, dict[int, float]] = {}
    with event_path.open("rb") as source:
        while header := source.read(12):
            if len(header) != 12:
                raise RuntimeError(f"{event_path}: truncated TFRecord header")
            record_length = struct.unpack("<Q", header[:8])[0]
            record = source.read(record_length)
            footer = source.read(4)
            if len(record) != record_length or len(footer) != 4:
                raise RuntimeError(f"{event_path}: truncated TFRecord")
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
                tag_bytes = next(
                    (
                        item
                        for number, kind, item in summary_value
                        if number == 1 and kind == 2
                    ),
                    None,
                )
                scalar_bytes = next(
                    (
                        item
                        for number, kind, item in summary_value
                        if number == 2 and kind == 5
                    ),
                    None,
                )
                if tag_bytes is None or scalar_bytes is None:
                    continue
                result.setdefault(tag_bytes.decode(), {})[int(step)] = float(
                    struct.unpack("<f", scalar_bytes)[0]
                )
    return result


def values_at_steps(
    values: dict[str, dict[int, float]], tag: str
) -> tuple[list[float], list[int]]:
    samples = [values.get(tag, {}).get(step) for step in STEADY_STEPS]
    missing = [
        step for step, value in zip(STEADY_STEPS, samples, strict=True) if value is None
    ]
    return [float(value) for value in samples if value is not None], missing


def phase_state(run_root: Path, phase: str) -> dict[str, Any]:
    phase_root = run_root / phase
    runtime = phase_root / "runtime"
    markers = [
        state
        for state in ("completed", "oom", "failed")
        if (run_root / "runtime" / f"{phase}.{state}").exists()
    ]
    exit_paths = sorted(runtime.glob("rank_*.exit_code"))
    exit_codes = {}
    malformed = []
    for path in exit_paths:
        match = re.fullmatch(r"rank_(\d+)\.exit_code", path.name)
        try:
            if match is None:
                raise ValueError(path.name)
            exit_codes[int(match.group(1))] = int(path.read_text().strip())
        except ValueError:
            malformed.append(str(path))
    oom_paths = sorted(runtime.glob("rank_*.oom_evidence"))
    nonzero = sorted(rank for rank, code in exit_codes.items() if code != 0)
    state = markers[0] if len(markers) == 1 else "missing_or_ambiguous"
    consistent = (
        len(exit_codes) == WORLD_SIZE
        and sorted(exit_codes) == list(range(WORLD_SIZE))
        and not malformed
        and len(markers) == 1
        and (
            (state == "completed" and not nonzero and not oom_paths)
            or (state == "oom" and bool(nonzero) and bool(oom_paths))
            or (state == "failed" and bool(nonzero) and not oom_paths)
        )
    )
    return {
        "state": state,
        "consistent": consistent,
        "markers": {
            marker: str(run_root / "runtime" / f"{phase}.{marker}")
            for marker in markers
        },
        "rank_exit_codes": exit_codes,
        "exit_code_paths": [str(path) for path in exit_paths],
        "nonzero_ranks": nonzero,
        "oom_evidence_paths": [str(path) for path in oom_paths],
        "malformed_exit_code_paths": malformed,
        "rank_log_paths": [
            str(path) for path in sorted(runtime.glob("rank_*.combined.log"))
        ],
    }


def phase_inventory(run_root: Path) -> dict[str, Any]:
    observed = sorted(
        path.name
        for path in run_root.iterdir()
        if path.is_dir() and re.match(r"^\d{2}_", path.name)
    )
    expected = sorted(ALL_PHASES)
    return {
        "status": "passed" if observed == expected else "failed",
        "expected": expected,
        "observed": observed,
        "missing": sorted(set(expected) - set(observed)),
        "unexpected": sorted(set(observed) - set(expected)),
    }


def tensorboard_result(phase_root: Path) -> dict[str, Any]:
    event_paths = sorted((phase_root / "job/tb").glob("**/events.out.tfevents.*"))
    if not event_paths:
        return {"status": "not_found", "event_files": []}
    if len(event_paths) != 1:
        return {
            "status": "ambiguous",
            "event_files": [str(path) for path in event_paths],
            "error": "expected exactly one TensorBoard event file",
        }
    try:
        scalars = scalar_values(event_paths[0])
        tags = {}
        for name, tag in {
            "latency_s": "time_metrics/end_to_end(s)",
            "data_loading_s": "time_metrics/data_loading(s)",
            "data_loading_percent": "time_metrics/data_loading(%)",
            "per_device_tps": "throughput(tps)",
        }.items():
            samples, missing = values_at_steps(scalars, tag)
            tags[name] = {
                "summary": summarize(samples) if samples else None,
                "missing_steady_steps": missing,
            }
        per_device_summary = tags["per_device_tps"]["summary"]
        aggregate_samples = (
            [value * WORLD_SIZE for value in per_device_summary["samples"]]
            if per_device_summary
            else []
        )
        tags["aggregate_tps"] = {
            "summary": summarize(aggregate_samples) if aggregate_samples else None,
            "missing_steady_steps": tags["per_device_tps"]["missing_steady_steps"],
        }
        active, active_missing = values_at_steps(scalars, "memory/max_active(GiB)")
        reserved, reserved_missing = values_at_steps(
            scalars, "memory/max_reserved(GiB)"
        )
        loss = scalars.get("loss_metrics/global_avg_loss", {})
        finite = all(
            math.isfinite(value)
            for samples in scalars.values()
            for value in samples.values()
        )
        return {
            "status": "available",
            "event_files": [str(path) for path in event_paths],
            "measurement_scope": "global rank 0 TensorBoard writer",
            "available_tags": sorted(scalars),
            "steady": tags,
            "active_memory_gib": {
                "scope": "rank0_only",
                "peak": max(active) if active else None,
                "summary": summarize(active) if active else None,
                "samples": active,
                "missing_steady_steps": active_missing,
            },
            "reserved_memory_gib": {
                "scope": "rank0_only",
                "peak": max(reserved) if reserved else None,
                "summary": summarize(reserved) if reserved else None,
                "samples": reserved,
                "missing_steady_steps": reserved_missing,
            },
            "loss_by_step": {str(step): value for step, value in sorted(loss.items())},
            "final_loss": loss[max(loss)] if loss else None,
            "all_scalar_values_finite": finite,
        }
    # Preserve malformed evidence instead of aborting an OOM-aware analysis.
    except Exception as error:
        return {
            "status": "parse_error",
            "event_files": [str(path) for path in event_paths],
            "error": f"{type(error).__name__}: {error}",
        }


def training_numerics(phase_root: Path) -> dict[str, Any]:
    pattern = re.compile(
        rf"step:\s*(?P<step>\d+).*?loss:\s*(?P<loss>{NUMBER})"
        rf".*?grad_norm:\s*(?P<grad>{NUMBER})",
        re.IGNORECASE,
    )
    observations = []
    paths = sorted((phase_root / "runtime").glob("rank_*.combined.log"))
    for path in paths:
        rank_match = re.fullmatch(r"rank_(\d+)\.combined\.log", path.name)
        if rank_match is None:
            continue
        rank = int(rank_match.group(1))
        for match in pattern.finditer(path.read_text(errors="replace")):
            observations.append(
                {
                    "rank": rank,
                    "step": int(match.group("step")),
                    "loss": float(match.group("loss")),
                    "grad_norm": float(match.group("grad")),
                }
            )
    nonfinite = [
        value
        for value in observations
        if not math.isfinite(value["loss"]) or not math.isfinite(value["grad_norm"])
    ]
    final_by_rank = {}
    for value in observations:
        rank = value["rank"]
        if rank not in final_by_rank or value["step"] > final_by_rank[rank]["step"]:
            final_by_rank[rank] = value
    return {
        "log_paths": [str(path) for path in paths],
        "observation_count": len(observations),
        "observed_ranks": sorted({value["rank"] for value in observations}),
        "observed_steps": sorted({value["step"] for value in observations}),
        "all_finite": bool(observations) and not nonfinite,
        "nonfinite_observations": nonfinite,
        "final_by_rank": [final_by_rank[rank] for rank in sorted(final_by_rank)],
    }


def setup_timings(phase_root: Path) -> dict[str, Any]:
    patterns = {
        "graph_trace_s": r"Graph tracing took (?P<value>[0-9.]+)s",
        "strategy_enumeration_s": (
            r"ShardingOptimizer: strategy enumeration took (?P<value>[0-9.]+)s"
        ),
        "placement_search_s": r"placement search took (?P<value>[0-9.]+) seconds",
        "apply_placements_s": r"Apply placements took (?P<value>[0-9.]+)s",
        "minimal_fx_trace_s": r"minimal_fx_tracer took (?P<value>[0-9.]+)s",
        "graph_passes_including_full_inductor_s": (
            r"All \d+ graph passes took (?P<value>[0-9.]+)s"
        ),
    }
    per_component: dict[str, dict[int, float]] = {name: {} for name in patterns}
    paths = sorted((phase_root / "runtime").glob("rank_*.combined.log"))
    for path in paths:
        rank_match = re.fullmatch(r"rank_(\d+)\.combined\.log", path.name)
        if rank_match is None:
            continue
        rank = int(rank_match.group(1))
        text = path.read_text(errors="replace")
        for name, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                per_component[name][rank] = float(match.group("value"))
    components = {}
    for name, by_rank in per_component.items():
        values = [by_rank[rank] for rank in sorted(by_rank)]
        components[name] = {
            "summary": summarize(values) if values else None,
            "by_rank": {str(rank): value for rank, value in sorted(by_rank.items())},
            "observed_ranks": len(by_rank),
            "expected_ranks": WORLD_SIZE,
        }
    return {"rank_log_paths": [str(path) for path in paths], "components": components}


def phase_config(phase_root: Path, case: dict[str, Any], arm: str) -> dict[str, Any]:
    path = phase_root / "job/config.json"
    if not path.exists():
        return {"status": "not_found", "path": str(path)}
    try:
        raw = json.loads(path.read_text())
    except Exception as error:
        return {
            "status": "parse_error",
            "path": str(path),
            "error": f"{type(error).__name__}: {error}",
        }
    contract = {
        "local_batch_size_per_dp_rank": nested(raw, "training", "local_batch_size"),
        "serialized_global_batch_size": nested(raw, "training", "global_batch_size"),
        "sequence_length": nested(raw, "training", "seq_len"),
        "fsdp_degree": nested(raw, "parallelism", "data_parallel_shard_degree"),
        "tp_degree": nested(raw, "parallelism", "tensor_parallel_degree"),
        "enable_autoparallel": nested(raw, "compile", "enable_autoparallel"),
        "memory_policy": nested(raw, "compile", "memory_policy"),
    }
    expected = {
        "local_batch_size_per_dp_rank": case["local_batch_size_per_dp_rank"],
        "serialized_global_batch_size": -1,
        "sequence_length": SEQUENCE_LENGTH,
        "fsdp_degree": FSDP_DEGREE,
        "tp_degree": TP_DEGREE,
        "enable_autoparallel": arm == "autoparallel",
        "memory_policy": "eager",
    }
    return {
        "status": "available",
        "path": str(path),
        "contract": contract,
        "expected_contract": expected,
        "contract_valid": contract == expected,
        "serialized": raw,
        "normalized": normalize_config(raw),
    }


def pair_config_parity(
    configs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    paths = {arm: value.get("path") for arm, value in configs.items()}
    if any(value.get("status") != "available" for value in configs.values()):
        return {"status": "insufficient_evidence", "paths": paths, "differences": []}
    differences = config_differences(
        configs["manual"]["normalized"], configs["autoparallel"]["normalized"]
    )
    valid = [value["path"] for value in differences] == [
        "compile.enable_autoparallel"
    ] and all(value["contract_valid"] for value in configs.values())
    return {
        "status": "passed" if valid else "failed",
        "paths": paths,
        "differences": differences,
        "only_declared_ap_toggle_differs": valid,
    }


def allocation_identity(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.get(key)
        for key in (
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
    }


def allocation_check(run_root: Path) -> dict[str, Any]:
    phase_records: dict[str, dict[int, dict[str, Any]]] = {}
    missing = []
    parse_errors = []
    for phase in ALL_PHASES:
        records = {}
        for rank in range(WORLD_SIZE):
            path = run_root / "allocation" / phase / f"rank_{rank:03d}.json"
            if not path.exists():
                missing.append(str(path))
                continue
            try:
                records[rank] = json.loads(path.read_text())
            except Exception as error:
                parse_errors.append(
                    {"path": str(path), "error": f"{type(error).__name__}: {error}"}
                )
        phase_records[phase] = records
    mismatches = []
    for rank in range(WORLD_SIZE):
        reference = None
        reference_phase = None
        for phase in ALL_PHASES:
            if rank not in phase_records[phase]:
                continue
            identity = allocation_identity(phase_records[phase][rank])
            if reference is None:
                reference = identity
                reference_phase = phase
            elif identity != reference:
                mismatches.append(
                    {
                        "rank": rank,
                        "reference_phase": reference_phase,
                        "phase": phase,
                        "reference": reference,
                        "observed": identity,
                    }
                )
    pair_validity = {}
    for case_name, case in CASES.items():
        for phase_kind in ("performance", "trace"):
            manual = case[phase_kind]["manual"]
            treatment = case[phase_kind]["autoparallel"]
            valid = all(
                rank in phase_records[manual]
                and rank in phase_records[treatment]
                and allocation_identity(phase_records[manual][rank])
                == allocation_identity(phase_records[treatment][rank])
                for rank in range(WORLD_SIZE)
            )
            pair_validity[f"{case_name}.{phase_kind}"] = valid
    valid = not missing and not parse_errors and not mismatches
    return {
        "valid": valid,
        "pair_validity": pair_validity,
        "missing_paths": missing,
        "parse_errors": parse_errors,
        "mismatches": mismatches,
        "raw_roots": {
            phase: str(run_root / "allocation" / phase) for phase in ALL_PHASES
        },
    }


def git_head(repo: Path) -> dict[str, Any]:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "path": str(repo),
        "returncode": result.returncode,
        "commit": result.stdout.strip() if result.returncode == 0 else None,
        "stderr": result.stderr.strip(),
    }


def package_audit_summary(attempt: Path) -> dict[str, Any]:
    path = attempt / "package_audit_formal_submitted/package_validation.json"
    if not path.exists():
        return {"status": "not_found", "path": str(path), "valid": False}
    try:
        value = json.loads(path.read_text())
    except Exception as error:
        return {
            "status": "parse_error",
            "path": str(path),
            "valid": False,
            "error": f"{type(error).__name__}: {error}",
        }
    valid = (
        value.get("valid") is True
        and value.get("comparisons", {}).get("torchtitan", {}).get("match") is True
    )
    return {"status": "available", "path": str(path), "valid": valid, "raw": value}


def runtime_environment_fingerprint(record: dict[str, Any]) -> dict[str, Any]:
    input_manifests = {
        case_name: {
            key: audit.get(key)
            for key in ("manifest", "manifest_sha256", "local_batch_size")
        }
        for case_name, audit in record.get("input_audits", {}).items()
    }
    offline = record.get("offline_c4", {})
    return {
        key: record.get(key)
        for key in (
            "python",
            "torch",
            "torch_commit",
            "cuda",
            "nccl",
            "modules",
            "model",
            "mesh",
            "batch_cases",
            "graph_trainer_compile",
            "attention",
            "c4_revision",
            "c4_manifest_hash",
            "tokenizer_hashes",
            "graph_trainer_shared_hashes",
            "autoparallel_sdpa_fix_hashes",
            "torchtitan_revision",
        )
    } | {
        "input_manifests": input_manifests,
        "offline_c4": {
            key: offline.get(key)
            for key in ("cache_root", "manifest_sha256", "revision")
        },
    }


def preflight_summary(attempt: Path, run_root: Path) -> dict[str, Any]:
    paths = sorted((run_root / "runtime_preflight").glob("rank_*.json"))
    records = []
    errors = []
    for path in paths:
        try:
            records.append(json.loads(path.read_text()))
        except Exception as error:
            errors.append(
                {"path": str(path), "error": f"{type(error).__name__}: {error}"}
            )
    expected_batches = {
        "per_gpu_bs1": {
            "local_batch_size_per_dp_rank": 2,
            "global_batch_size": 16,
            "per_physical_gpu_effective_batch_size": 1,
            "sequence_length": 4096,
            "tokens_per_step": 65_536,
        },
    }
    record_checks = []
    for record in records:
        sac = record.get("outer_sac_pass_audit", {})
        input_audits = record.get("input_audits", {})
        record_checks.append(
            record.get("status") == "passed"
            and record.get("world_size") == WORLD_SIZE
            and record.get("mesh") == {"fsdp": FSDP_DEGREE, "tp": TP_DEGREE}
            and record.get("attention") == "packed_document_sdpa"
            and record.get("batch_cases") == expected_batches
            and record.get("torch") == EXPECTED_TORCH_VERSION
            and record.get("torch_commit") == EXPECTED_TORCH_COMMIT
            and record.get("torchtitan_revision") == EXPECTED_TORCHTITAN_COMMIT
            and record.get("graph_trainer_shared_hashes")
            == EXPECTED_GRAPH_TRAINER_HASHES
            and record.get("autoparallel_sdpa_fix_hashes") == EXPECTED_AP_FIX_HASHES
            and sac.get("status") == "shared_outer_prefix_verified"
            and sac.get("memory_policy") == "eager"
            and sac.get("shared_sac_prefix") == EXPECTED_SAC_PREFIX
            and set(input_audits) == set(CASES)
        )
    verified_steps = {
        case_name: sum(
            int(
                record.get("input_audits", {})
                .get(case_name, {})
                .get("verified_steps", 0)
            )
            for record in records
        )
        for case_name in CASES
    }
    unique_sac_audits = {
        json.dumps(record.get("outer_sac_pass_audit"), sort_keys=True)
        for record in records
    }
    environment_fingerprints = [
        runtime_environment_fingerprint(record) for record in records
    ]
    unique_environment_fingerprints = {
        json.dumps(value, sort_keys=True) for value in environment_fingerprints
    }
    observed_ranks = [record.get("rank") for record in records]
    valid_rank_coverage = all(
        isinstance(rank, int) for rank in observed_ranks
    ) and sorted(observed_ranks) == list(range(WORLD_SIZE))
    torchtitan_checkout = git_head(TASK_ROOT / "source/torchtitan")
    package_audit = package_audit_summary(attempt)
    valid = (
        len(paths) == WORLD_SIZE
        and len(records) == WORLD_SIZE
        and valid_rank_coverage
        and not errors
        and all(record_checks)
        and verified_steps == {case_name: 25 * FSDP_DEGREE for case_name in CASES}
        and len(unique_sac_audits) == 1
        and len(unique_environment_fingerprints) == 1
        and torchtitan_checkout["commit"] == EXPECTED_TORCHTITAN_COMMIT
        and package_audit["valid"]
    )
    return {
        "status": "passed" if valid else "failed",
        "paths": [str(path) for path in paths],
        "record_count": len(records),
        "observed_ranks": observed_ranks,
        "rank_coverage_valid": valid_rank_coverage,
        "parse_errors": errors,
        "expected_versions": {
            "torchtitan_checkout_commit": EXPECTED_TORCHTITAN_COMMIT,
            "torchtitan_revision": EXPECTED_TORCHTITAN_COMMIT,
            "torch": EXPECTED_TORCH_VERSION,
            "torch_commit": EXPECTED_TORCH_COMMIT,
        },
        "torchtitan_checkout": torchtitan_checkout,
        "formal_package_audit": package_audit,
        "cross_rank_environment_hash_equality": len(unique_environment_fingerprints)
        == 1,
        "environment_fingerprint_count": len(unique_environment_fingerprints),
        "verified_input_steps_across_dp_groups": verified_steps,
        "outer_sac_pass_audit": (
            records[0].get("outer_sac_pass_audit") if records else None
        ),
        "attention": records[0].get("attention") if records else None,
        "batch_cases": records[0].get("batch_cases") if records else None,
        "environment": (
            {
                key: records[0].get(key)
                for key in (
                    "python",
                    "torch",
                    "torch_commit",
                    "cuda",
                    "nccl",
                    "modules",
                    "torchtitan_revision",
                    "graph_trainer_shared_hashes",
                    "autoparallel_sdpa_fix_hashes",
                )
            }
            if records
            else None
        ),
    }


def merge(intervals: list[tuple[float, float]]) -> list[list[float]]:
    result: list[list[float]] = []
    for start, end in sorted(intervals):
        if not result or start > result[-1][1]:
            result.append([start, end])
        else:
            result[-1][1] = max(result[-1][1], end)
    return result


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
        return "comm"
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


def kineto_summary(
    path: Path, step_number: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    with gzip.open(path, "rt") as source:
        trace = json.load(source)
    events = trace["traceEvents"]
    step_name = f"ProfilerStep#{step_number}"
    step_events = [
        event
        for event in events
        if event.get("name") == step_name
        and event.get("cat") in {"user_annotation", "gpu_user_annotation"}
    ]
    invalid_step_events = [
        event
        for event in step_events
        if event.get("ph") != "X"
        or not isinstance(event.get("ts"), (int, float))
        or not isinstance(event.get("dur"), (int, float))
        or event["dur"] <= 0
    ]
    if invalid_step_events:
        raise RuntimeError(f"invalid complete-event schema for {step_name}")
    cpu_step_events = [
        event for event in step_events if event.get("cat") == "user_annotation"
    ]
    gpu_step_events = [
        event for event in step_events if event.get("cat") == "gpu_user_annotation"
    ]
    if len(cpu_step_events) == 1:
        selected_step = cpu_step_events[0]
        step_window_source = "unique_cpu_user_annotation"
    elif not cpu_step_events and len(gpu_step_events) == 1:
        selected_step = gpu_step_events[0]
        step_window_source = "single_gpu_projection_fallback"
    else:
        raise RuntimeError(
            f"expected one CPU {step_name}, or exactly one GPU projection when "
            f"CPU annotation is absent; got CPU={len(cpu_step_events)}, "
            f"GPU={len(gpu_step_events)}"
        )
    step_start = float(selected_step["ts"])
    step_end = step_start + float(selected_step["dur"])
    gpu_projections = []
    for event in gpu_step_events:
        projection_start = float(event["ts"])
        projection_end = projection_start + float(event["dur"])
        overlap = max(
            0.0,
            min(step_end, projection_end) - max(step_start, projection_start),
        )
        gpu_projections.append(
            {
                "pid": event.get("pid"),
                "tid": event.get("tid"),
                "external_id": event.get("args", {}).get("External id"),
                "start_us": projection_start,
                "duration_ms": float(event["dur"]) / 1000,
                "overlap_ms": overlap / 1000,
                "overlaps_selected_window": overlap > 0,
            }
        )
    if any(
        not projection["overlaps_selected_window"] for projection in gpu_projections
    ):
        raise RuntimeError(f"a GPU projection of {step_name} misses the CPU window")
    kernels = []
    collectives: dict[tuple[str, str, int, str], tuple[int, float]] = {}
    kernel_names: dict[str, tuple[int, float]] = {}
    category_durations: dict[str, float] = {}
    category_counts: dict[str, int] = {}
    for event in events:
        if event.get("cat") != "kernel" or "ts" not in event or "dur" not in event:
            continue
        event_start = float(event["ts"])
        event_end = event_start + float(event["dur"])
        if event_end <= step_start or event_start >= step_end:
            continue
        start = max(event_start, step_start)
        end = min(event_end, step_end)
        clipped_duration = end - start
        args = event.get("args", {})
        name = str(event.get("name", ""))
        is_collective = "Collective name" in args or "nccl" in name.lower()
        kernels.append((start, end, is_collective))
        count, duration = kernel_names.get(name, (0, 0.0))
        kernel_names[name] = (count + 1, duration + clipped_duration)
        category = kernel_category(name, args)
        category_durations[category] = (
            category_durations.get(category, 0.0) + clipped_duration
        )
        category_counts[category] = category_counts.get(category, 0) + 1
        if is_collective:
            key = (
                str(args.get("Collective name", "nccl")),
                str(args.get("dtype", "?")),
                int(args.get("Group size", 0) or 0),
                str(args.get("Process Group Description", "?")),
            )
            count, duration = collectives.get(key, (0, 0.0))
            collectives[key] = (count + 1, duration + clipped_duration)
    all_intervals = [(start, end) for start, end, _ in kernels]
    comm_intervals = [(start, end) for start, end, comm in kernels if comm]
    compute_intervals = [(start, end) for start, end, comm in kernels if not comm]
    summary = {
        "status": "available",
        "path": str(path),
        "compressed_bytes": path.stat().st_size,
        "profiler_step": step_number,
        "step_duration_ms": (step_end - step_start) / 1000,
        "step_window": {
            "source": step_window_source,
            "start_us": step_start,
            "end_us": step_end,
            "cpu_annotation_count": len(cpu_step_events),
            "gpu_projection_count": len(gpu_step_events),
            "gpu_projection_durations_ms": [
                projection["duration_ms"] for projection in gpu_projections
            ],
            "all_gpu_projections_overlap": all(
                projection["overlaps_selected_window"] for projection in gpu_projections
            ),
            "gpu_projections": gpu_projections,
        },
        "gpu_busy_union_ms": union_length(all_intervals) / 1000,
        "noncomm_union_ms": union_length(compute_intervals) / 1000,
        "comm_raw_ms": sum(end - start for start, end in comm_intervals) / 1000,
        "comm_union_ms": union_length(comm_intervals) / 1000,
        "comm_exposed_ms": subtract_length(comm_intervals, compute_intervals) / 1000,
        "kernel_count": len(kernels),
        "comm_kernel_count": len(comm_intervals),
        "kernel_categories": {
            category: {
                "raw_ms": category_durations[category] / 1000,
                "count": category_counts[category],
            }
            for category in sorted(category_durations)
        },
        "collectives": [
            {
                "collective": key[0],
                "dtype": key[1],
                "group_size": key[2],
                "group": key[3],
                "count": value[0],
                "raw_ms": value[1] / 1000,
            }
            for key, value in sorted(
                collectives.items(), key=lambda item: item[1][1], reverse=True
            )
        ],
        "top_kernels": [
            {"name": name, "count": value[0], "raw_ms": value[1] / 1000}
            for name, value in sorted(
                kernel_names.items(), key=lambda item: item[1][1], reverse=True
            )[:30]
        ],
    }
    kept = []
    categories: dict[str, int] = {}
    for event in events:
        timestamp = event.get("ts")
        duration = event.get("dur", 0)
        overlaps = (
            isinstance(timestamp, (int, float))
            and isinstance(duration, (int, float))
            and timestamp < step_end
            and timestamp + duration > step_start
        )
        category = str(event.get("cat", ""))
        keep = event.get("ph") == "M" or (
            overlaps
            and (
                category in {"kernel", "gpu_user_annotation", "user_annotation"}
                or event.get("ph") in {"s", "t", "f"}
            )
        )
        if keep:
            kept.append(event)
            categories[category] = categories.get(category, 0) + 1
    slim = {key: value for key, value in trace.items() if key != "traceEvents"}
    slim["traceEvents"] = kept
    return summary, {
        "trace": slim,
        "categories": categories,
        "input_events": len(events),
    }


def local_shape(shape: list[int], placement: str) -> list[int] | None:
    tokens = re.findall(r"S\(\d+\)|R", placement)
    if len(tokens) != 2:
        return None
    result = list(shape)
    for mesh_size, token in zip((FSDP_DEGREE, TP_DEGREE), tokens, strict=True):
        if token == "R":
            continue
        dimension = int(re.search(r"\d+", token).group())
        if dimension >= len(result) or result[dimension] % mesh_size:
            return None
        result[dimension] //= mesh_size
    return result


def sdpa_mode_contracts(expected_local_batch: int) -> dict[str, dict[str, Any]]:
    contracts = {
        "tp_head_shard": {
            "qkv_placement": "S(0)S(1)",
            "mask_placement": "S(0)R",
            "q_shape": [expected_local_batch, 16, SEQUENCE_LENGTH, 128],
            "kv_shape": [expected_local_batch, 1, SEQUENCE_LENGTH, 128],
            "mask_shape": [
                expected_local_batch,
                1,
                SEQUENCE_LENGTH,
                SEQUENCE_LENGTH,
            ],
            "lse_shape": [expected_local_batch, 16, SEQUENCE_LENGTH, 1],
        }
    }
    if expected_local_batch % TP_DEGREE == 0:
        tp_local_batch = expected_local_batch // TP_DEGREE
        contracts["tp_batch_shard"] = {
            "qkv_placement": "S(0)S(0)",
            "mask_placement": "S(0)S(0)",
            "q_shape": [tp_local_batch, 32, SEQUENCE_LENGTH, 128],
            "kv_shape": [tp_local_batch, 2, SEQUENCE_LENGTH, 128],
            "mask_shape": [tp_local_batch, 1, SEQUENCE_LENGTH, SEQUENCE_LENGTH],
            "lse_shape": [tp_local_batch, 32, SEQUENCE_LENGTH, 1],
        }
    return contracts


def sdpa_phase_specs(contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    qkv_placement = contract["qkv_placement"]
    mask_placement = contract["mask_placement"]
    q_shape = contract["q_shape"]
    kv_shape = contract["kv_shape"]
    mask_shape = contract["mask_shape"]
    lse_shape = contract["lse_shape"]
    return {
        "forward": {
            "op": "aten._scaled_dot_product_cudnn_attention.default",
            "inputs": (
                ("query", 0, qkv_placement, q_shape),
                ("key", 1, qkv_placement, kv_shape),
                ("value", 2, qkv_placement, kv_shape),
                ("mask", 3, mask_placement, mask_shape),
            ),
            "output_placements": (qkv_placement,),
            "first_output_shape": q_shape,
        },
        "backward": {
            "op": "aten._scaled_dot_product_cudnn_attention_backward.default",
            "inputs": (
                ("grad_out", 0, qkv_placement, q_shape),
                ("query", 1, qkv_placement, q_shape),
                ("key", 2, qkv_placement, kv_shape),
                ("value", 3, qkv_placement, kv_shape),
                ("output", 4, qkv_placement, q_shape),
                ("logsumexp", 5, qkv_placement, lse_shape),
                ("philox_seed", 6, "RR", []),
                ("philox_offset", 7, "RR", []),
                ("mask", 8, mask_placement, mask_shape),
            ),
            "output_placements": (qkv_placement,) * 3,
            "first_output_shape": q_shape,
        },
    }


def sdpa_solution_summary(
    raw: dict[str, Any], expected_local_batch: int, required_mode: str | None
) -> dict[str, Any]:
    nodes = raw.get("nodes", [])
    by_name = {node.get("name"): node for node in nodes}
    mode_contracts = sdpa_mode_contracts(expected_local_batch)
    if required_mode is not None and required_mode not in mode_contracts:
        raise RuntimeError(
            f"SDPA mode {required_mode} is invalid for local batch "
            f"{expected_local_batch}"
        )
    all_phase_specs = {
        mode: sdpa_phase_specs(contract) for mode, contract in mode_contracts.items()
    }
    reference_phase_specs = next(iter(all_phase_specs.values()))
    phase_results = {}
    for phase, reference_spec in reference_phase_specs.items():
        sites = []
        for node in nodes:
            if node.get("phase") != phase or node.get("op") != reference_spec["op"]:
                continue
            raw_inputs = node.get("inputs", [])
            semantic_inputs = {}
            for name, index, _, _ in reference_spec["inputs"]:
                value = raw_inputs[index] if index < len(raw_inputs) else {}
                source = by_name.get(value.get("name"), {})
                src_placement = value.get("src_placement")
                dst_placement = value.get("dst_placement")
                shape = source.get("shape")
                observed_local_shape = (
                    local_shape(shape, dst_placement)
                    if isinstance(shape, list) and isinstance(dst_placement, str)
                    else None
                )
                semantic_inputs[name] = {
                    "node": value.get("name"),
                    "global_shape": shape,
                    "local_shape": observed_local_shape,
                    "src_placement": src_placement,
                    "dst_placement": dst_placement,
                    "transition_cost": value.get("transition_cost"),
                }
            placement_text = re.sub(r"\s+", "", str(node.get("placement")))
            placement_fields = tuple(re.findall(r"(?:S\(\d+\)|R){2}", placement_text))
            output_placements = placement_fields[
                : len(reference_spec["output_placements"])
            ]
            first_output_placement = output_placements[0] if output_placements else None
            first_output_local_shape = (
                local_shape(node["shape"], first_output_placement)
                if isinstance(node.get("shape"), list) and first_output_placement
                else None
            )
            selected_modes = []
            for mode, candidate_phase_specs in all_phase_specs.items():
                candidate = candidate_phase_specs[phase]
                if (
                    output_placements == candidate["output_placements"]
                    and first_output_local_shape == candidate["first_output_shape"]
                    and all(
                        semantic_inputs[name]["src_placement"]
                        == semantic_inputs[name]["dst_placement"]
                        == placement
                        and semantic_inputs[name]["local_shape"] == shape
                        for name, _, placement, shape in candidate["inputs"]
                    )
                ):
                    selected_modes.append(mode)
            selected_mode = (
                selected_modes[0]
                if len(selected_modes) == 1
                else "mixed_or_unclassified"
            )
            selected_spec = (
                all_phase_specs[selected_mode][phase]
                if selected_mode in all_phase_specs
                else None
            )
            if selected_spec is not None:
                expected_inputs = {
                    name: (placement, shape)
                    for name, _, placement, shape in selected_spec["inputs"]
                }
                for name, value in semantic_inputs.items():
                    expected_placement, expected_shape = expected_inputs[name]
                    value.update(
                        expected_local_shape=expected_shape,
                        expected_placement=expected_placement,
                        placement_valid=value["src_placement"]
                        == value["dst_placement"]
                        == expected_placement,
                        local_shape_valid=value["local_shape"] == expected_shape,
                    )
            mode_requirement_satisfied = selected_mode in mode_contracts and (
                required_mode is None or selected_mode == required_mode
            )
            layer_match = re.search(r"layers\.(\d+)\.", node.get("module_path") or "")
            sites.append(
                {
                    "name": node.get("name"),
                    "layer": int(layer_match.group(1)) if layer_match else None,
                    "module_path": node.get("module_path"),
                    "inputs": semantic_inputs,
                    "output_placements": list(output_placements),
                    "first_output_local_shape": first_output_local_shape,
                    "selected_output_placements": (
                        list(selected_spec["output_placements"])
                        if selected_spec is not None
                        else None
                    ),
                    "selected_first_output_local_shape": (
                        selected_spec["first_output_shape"]
                        if selected_spec is not None
                        else None
                    ),
                    "selected_parallelism_mode": selected_mode,
                    "required_parallelism_mode": required_mode,
                    "mode_requirement_satisfied": mode_requirement_satisfied,
                    "placements_valid": selected_spec is not None
                    and output_placements == selected_spec["output_placements"]
                    and all(
                        value["placement_valid"] for value in semantic_inputs.values()
                    ),
                    "local_shapes_valid": selected_spec is not None
                    and first_output_local_shape == selected_spec["first_output_shape"]
                    and all(
                        value["local_shape_valid"] for value in semantic_inputs.values()
                    ),
                    "zero_input_transition_cost": all(
                        value["transition_cost"] == 0.0
                        for value in semantic_inputs.values()
                    ),
                }
            )
        layers = sorted(site["layer"] for site in sites if site["layer"] is not None)
        valid = (
            len(sites) == 52
            and layers == list(range(52))
            and all(
                site["placements_valid"]
                and site["local_shapes_valid"]
                and site["zero_input_transition_cost"]
                and site["mode_requirement_satisfied"]
                for site in sites
            )
        )
        selected_modes = sorted({site["selected_parallelism_mode"] for site in sites})
        selected_mode = (
            selected_modes[0] if len(selected_modes) == 1 else "mixed_or_unclassified"
        )
        phase_results[phase] = {
            "status": "passed" if valid else "failed",
            "site_count": len(sites),
            "layers": layers,
            "required_parallelism_mode": required_mode,
            "selected_parallelism_mode": selected_mode,
            "sites": sites,
        }
    selected_modes = sorted(
        {value["selected_parallelism_mode"] for value in phase_results.values()}
    )
    selected_mode = (
        selected_modes[0] if len(selected_modes) == 1 else "mixed_or_unclassified"
    )
    selected_contract = mode_contracts.get(selected_mode)
    valid = (
        selected_contract is not None
        and (required_mode is None or selected_mode == required_mode)
        and all(value["status"] == "passed" for value in phase_results.values())
    )
    return {
        "status": "passed" if valid else "failed",
        "required_parallelism_mode": required_mode,
        "allowed_parallelism_modes": sorted(mode_contracts),
        "selected_parallelism_mode": selected_mode,
        "selected_contract": (
            {
                "forward_site_count": 52,
                "backward_site_count": 52,
                "layers": [0, 51],
                "qkv_output_placement": selected_contract["qkv_placement"],
                "mask_placement": selected_contract["mask_placement"],
                "local_shapes": {
                    key: selected_contract[key]
                    for key in ("q_shape", "kv_shape", "mask_shape", "lse_shape")
                },
                "local_head_extents": {
                    "query": selected_contract["q_shape"][1],
                    "key": selected_contract["kv_shape"][1],
                    "value": selected_contract["kv_shape"][1],
                    "mask": selected_contract["mask_shape"][1],
                },
            }
            if selected_contract is not None
            else None
        ),
        **phase_results,
    }


def solution_summary(
    raw: dict[str, Any], expected_local_batch: int, required_mode: str | None
) -> dict[str, Any]:
    parameters = [
        node for node in raw.get("nodes", []) if node.get("placeholder_kind") == "param"
    ]
    placements: dict[str, int] = {}
    representatives = {}
    for node in parameters:
        placement = str(node.get("placement"))
        placements[placement] = placements.get(placement, 0) + 1
        if node.get("module_path") in REPRESENTATIVE_PARAMS:
            representatives[node["module_path"]] = {
                "shape": node.get("shape"),
                "placement": node.get("placement"),
            }
    mesh = raw.get("mesh")
    return {
        "mesh": mesh,
        "mesh_valid": mesh
        == {"shape": [FSDP_DEGREE, TP_DEGREE], "dim_names": ["fsdp", "tp"]},
        "summary": raw.get("summary"),
        "parameter_count": len(parameters),
        "parameter_placement_counts": dict(sorted(placements.items())),
        "representative_parameters": representatives,
        "inputs": [
            {key: node.get(key) for key in ("shape", "dtype", "placement")}
            for node in raw.get("nodes", [])
            if node.get("placeholder_kind") == "input"
        ],
        "sdpa": sdpa_solution_summary(raw, expected_local_batch, required_mode),
    }


def graph_sdpa_summary(
    text: str, expected_local_batch: int, required_mode: str | None
) -> dict[str, Any]:
    mode_contracts = sdpa_mode_contracts(expected_local_batch)
    if required_mode is not None and required_mode not in mode_contracts:
        raise RuntimeError(
            f"SDPA mode {required_mode} is invalid for local batch "
            f"{expected_local_batch}"
        )
    assignment = re.compile(r"^\s*([A-Za-z_]\w*)(?::\s*\"[^\"]*\")?\s*=\s*(.*)$")
    typed_shape = re.compile(r'^\s*([A-Za-z_]\w*):\s*"[^\"]*?\[([0-9, ]*)\]')
    call_patterns = {
        "forward": re.compile(
            r"torch\.ops\.aten\._scaled_dot_product_cudnn_attention\.\w+\("
            r"\s*([A-Za-z_]\w*)\s*,\s*([A-Za-z_]\w*)\s*,"
            r"\s*([A-Za-z_]\w*)\s*,\s*([A-Za-z_]\w*)"
        ),
        "backward": re.compile(
            r"torch\.ops\.aten\._scaled_dot_product_cudnn_attention_backward\.\w+\("
            r"\s*([A-Za-z_]\w*)\s*,\s*([A-Za-z_]\w*)\s*,"
            r"\s*([A-Za-z_]\w*)\s*,\s*([A-Za-z_]\w*)\s*,"
            r"\s*([A-Za-z_]\w*)\s*,\s*([A-Za-z_]\w*)\s*,"
            r"\s*([A-Za-z_]\w*)\s*,\s*([A-Za-z_]\w*)\s*,"
            r"\s*([A-Za-z_]\w*)"
        ),
    }
    definitions: dict[str, str] = {}
    shapes: dict[str, list[int]] = {}
    phase_sites: dict[str, list[dict[str, Any]]] = {
        "forward": [],
        "backward": [],
    }
    current_layer = None

    def gather_ancestors(variable: str) -> list[dict[str, Any]]:
        pending = [variable]
        visited = set()
        gathers = []
        while pending:
            name = pending.pop()
            if name in visited:
                continue
            visited.add(name)
            expression = definitions.get(name, "")
            if "_c10d_functional.all_gather" in expression:
                group_match = re.search(
                    r"all_gather_into_tensor(?:_out)?\.default\([^,]+,\s*(\d+)",
                    expression,
                )
                gathers.append(
                    {
                        "variable": name,
                        "group_size": (
                            int(group_match.group(1)) if group_match else None
                        ),
                        "shape": shapes.get(name),
                        "expression": expression,
                    }
                )
            pending.extend(
                token
                for token in re.findall(r"\b[A-Za-z_]\w*\b", expression)
                if token in definitions and token not in visited
            )
        return gathers

    for line_number, line in enumerate(text.splitlines(), 1):
        if re.match(r"^\s*def \w+\(", line):
            definitions.clear()
            shapes.clear()
            current_layer = None
        layer_match = re.search(
            r"module_fqn[^\n]*layers\.(\d+)\.attention\.inner_attention", line
        )
        if layer_match:
            current_layer = int(layer_match.group(1))
        shape_match = typed_shape.match(line)
        if shape_match:
            dimensions = shape_match.group(2).strip()
            shapes[shape_match.group(1)] = (
                [int(value.strip()) for value in dimensions.split(",") if value.strip()]
                if dimensions
                else []
            )
        assignment_match = assignment.match(line)
        if assignment_match:
            definitions[assignment_match.group(1)] = assignment_match.group(2).split(
                ";", 1
            )[0]
        phase = next(
            (name for name, pattern in call_patterns.items() if pattern.search(line)),
            None,
        )
        if phase is None:
            continue
        call_match = call_patterns[phase].search(line)
        arguments = list(call_match.groups())
        if phase == "forward":
            semantic_indices = {"query": 0, "key": 1, "value": 2, "mask": 3}
        else:
            semantic_indices = {"query": 1, "key": 2, "value": 3, "mask": 8}
        semantic_arguments = {
            name: arguments[index] for name, index in semantic_indices.items()
        }
        local_shapes = {
            name: shapes.get(argument) for name, argument in semantic_arguments.items()
        }
        selected_modes = []
        for mode, contract in mode_contracts.items():
            candidate_shapes = {
                "query": contract["q_shape"],
                "key": contract["kv_shape"],
                "value": contract["kv_shape"],
                "mask": contract["mask_shape"],
            }
            if local_shapes == candidate_shapes:
                selected_modes.append(mode)
        selected_mode = (
            selected_modes[0] if len(selected_modes) == 1 else "mixed_or_unclassified"
        )
        selected_contract = mode_contracts.get(selected_mode)
        selected_shapes = (
            {
                "query": selected_contract["q_shape"],
                "key": selected_contract["kv_shape"],
                "value": selected_contract["kv_shape"],
                "mask": selected_contract["mask_shape"],
            }
            if selected_contract is not None
            else None
        )
        mode_requirement_satisfied = selected_mode in mode_contracts and (
            required_mode is None or selected_mode == required_mode
        )
        ancestors = {
            name: gather_ancestors(semantic_arguments[name])
            for name in ("query", "key", "value")
        }
        tp_head_gathers = [
            {"input": name, **gather}
            for name, gathers in ancestors.items()
            for gather in gathers
            if gather["group_size"] == TP_DEGREE
            and isinstance(gather["shape"], list)
            and len(gather["shape"]) == 4
            and gather["shape"][-1] == 128
        ]
        phase_sites[phase].append(
            {
                "line": line_number,
                "layer": current_layer,
                "arguments": arguments,
                "semantic_arguments": semantic_arguments,
                "local_shapes": local_shapes,
                "selected_contract_local_shapes": selected_shapes,
                "local_shapes_valid": local_shapes == selected_shapes,
                "selected_parallelism_mode": selected_mode,
                "required_parallelism_mode": required_mode,
                "mode_requirement_satisfied": mode_requirement_satisfied,
                "qkv_all_gather_ancestors": ancestors,
                "tp_head_gather_ancestors": tp_head_gathers,
            }
        )
    phase_results = {}
    for phase, sites in phase_sites.items():
        layers = sorted(site["layer"] for site in sites if site["layer"] is not None)
        valid = (
            len(sites) == 52
            and layers == list(range(52))
            and all(site["local_shapes_valid"] for site in sites)
            and all(site["mode_requirement_satisfied"] for site in sites)
            and all(not site["tp_head_gather_ancestors"] for site in sites)
        )
        selected_modes = sorted({site["selected_parallelism_mode"] for site in sites})
        selected_mode = (
            selected_modes[0] if len(selected_modes) == 1 else "mixed_or_unclassified"
        )
        phase_results[phase] = {
            "status": "passed" if valid else "failed",
            "site_count": len(sites),
            "layers": layers,
            "required_parallelism_mode": required_mode,
            "selected_parallelism_mode": selected_mode,
            "all_local_shapes_valid": bool(sites)
            and all(site["local_shapes_valid"] for site in sites),
            "no_pre_sdpa_tp_head_gather": bool(sites)
            and all(not site["tp_head_gather_ancestors"] for site in sites),
            "sites": sites,
        }
    selected_modes = sorted(
        {value["selected_parallelism_mode"] for value in phase_results.values()}
    )
    selected_mode = (
        selected_modes[0] if len(selected_modes) == 1 else "mixed_or_unclassified"
    )
    selected_contract = mode_contracts.get(selected_mode)
    valid = (
        selected_contract is not None
        and (required_mode is None or selected_mode == required_mode)
        and all(value["status"] == "passed" for value in phase_results.values())
    )
    return {
        "status": "passed" if valid else "failed",
        "required_parallelism_mode": required_mode,
        "allowed_parallelism_modes": sorted(mode_contracts),
        "selected_parallelism_mode": selected_mode,
        "selected_contract": (
            {
                "forward_site_count": 52,
                "backward_site_count": 52,
                "local_shapes": {
                    "query": selected_contract["q_shape"],
                    "key": selected_contract["kv_shape"],
                    "value": selected_contract["kv_shape"],
                    "mask": selected_contract["mask_shape"],
                },
                "local_head_extents": {
                    "query": selected_contract["q_shape"][1],
                    "key": selected_contract["kv_shape"][1],
                    "value": selected_contract["kv_shape"][1],
                    "mask": selected_contract["mask_shape"][1],
                },
            }
            if selected_contract is not None
            else None
        ),
        **phase_results,
    }


def activation_policy_summary(text: str) -> dict[str, Any]:
    policies = {}
    for policy in re.findall(
        r"\b(?:MUST_SAVE|PREFER_RECOMPUTE|MUST_RECOMPUTE|RECOMPUTE|SAVE\*)\b|\bSAVE\b",
        text,
    ):
        policies[policy] = policies.get(policy, 0) + 1
    return {
        "policy_counts": dict(sorted(policies.items())),
        "tag_count": sum(policies.values()),
    }


def sdpa_mode_consistency(
    summaries: dict[str, dict[str, Any] | None],
) -> dict[str, Any]:
    observed = {
        name: summary.get("selected_parallelism_mode")
        for name, summary in summaries.items()
        if summary is not None
    }
    modes = set(observed.values())
    valid = (
        bool(observed)
        and len(modes) == 1
        and "mixed_or_unclassified" not in modes
        and all(
            summary is not None and summary.get("status") == "passed"
            for summary in summaries.values()
            if summary is not None
        )
    )
    return {
        "status": "passed" if valid else "failed",
        "selected_parallelism_mode": next(iter(modes)) if len(modes) == 1 else None,
        "observations": observed,
    }


def compile_trace_summary(
    path: Path, expected_local_batch: int, required_mode: str | None
) -> dict[str, Any]:
    artifact_counts: dict[str, int] = {}
    payloads: dict[str, list[str]] = {
        "activation_memory_policy": [],
        "after_joint_graph": [],
        "autoparallel_parallel_graph": [],
        "autoparallel_solution": [],
    }
    current_name = None
    current_lines: list[str] = []

    def finish_payload() -> None:
        nonlocal current_name, current_lines
        if current_name in payloads:
            payloads[current_name].append("".join(current_lines))
        current_name = None
        current_lines = []

    with path.open(errors="replace") as source:
        for line in source:
            if line.startswith("V"):
                finish_payload()
                match = re.search(r'\{"artifact": \{"name": "([^"]+)"', line)
                if match:
                    current_name = match.group(1)
                    artifact_counts[current_name] = (
                        artifact_counts.get(current_name, 0) + 1
                    )
                continue
            if current_name in payloads:
                current_lines.append(line)
    finish_payload()

    solution = None
    solution_error = None
    if payloads["autoparallel_solution"]:
        try:
            solution = solution_summary(
                json.loads(payloads["autoparallel_solution"][0].lstrip()),
                expected_local_batch,
                required_mode,
            )
        except Exception as error:
            solution_error = f"{type(error).__name__}: {error}"
    parallel_graph = None
    if payloads["autoparallel_parallel_graph"]:
        parallel_graph = graph_sdpa_summary(
            payloads["autoparallel_parallel_graph"][0],
            expected_local_batch,
            required_mode,
        )
    after_joint_graph = None
    if payloads["after_joint_graph"]:
        after_joint_graph = graph_sdpa_summary(
            payloads["after_joint_graph"][0], expected_local_batch, required_mode
        )
    activation_policy = None
    if payloads["activation_memory_policy"]:
        activation_policy = activation_policy_summary(
            payloads["activation_memory_policy"][0]
        )
    mode_consistency = sdpa_mode_consistency(
        {
            "autoparallel_solution": solution.get("sdpa") if solution else None,
            "autoparallel_parallel_graph": parallel_graph,
            "after_joint_graph": after_joint_graph,
        }
    )
    return {
        "status": "available",
        "path": str(path),
        "bytes": path.stat().st_size,
        "required_parallelism_mode": required_mode,
        "sdpa_mode_consistency": mode_consistency,
        "artifact_counts": dict(sorted(artifact_counts.items())),
        "interesting_artifact_counts": {
            name: artifact_counts.get(name, 0) for name in sorted(INTERESTING_ARTIFACTS)
        },
        "solution": solution,
        "solution_error": solution_error,
        "autoparallel_parallel_graph_sdpa": parallel_graph,
        "after_joint_graph_sdpa": after_joint_graph,
        "activation_memory_policy": activation_policy,
    }


def artifact_path(root: Path, pattern: str) -> tuple[Path | None, list[str]]:
    paths = sorted(root.glob(f"**/{pattern}")) if root.exists() else []
    return (paths[0] if len(paths) == 1 else None), [str(path) for path in paths]


def tlparse_summary(
    phase: str,
    arm: str,
    expected_local_batch: int,
    required_mode: str | None,
    tlparse_root: Path,
    compile_trace_paths: list[Path],
) -> dict[str, Any]:
    root = tlparse_root / phase
    if not root.is_dir():
        return {
            "status": "not_found",
            "root": str(root),
            "error": "the explicit phase-bound tlparse directory does not exist",
        }
    solution_path, solution_paths = artifact_path(root, "autoparallel_solution_*.json")
    graph_path, graph_paths = artifact_path(root, "autoparallel_parallel_graph_*.txt")
    joint_path, joint_paths = artifact_path(root, "after_joint_graph_*.txt")
    policy_path, policy_paths = artifact_path(root, "activation_memory_policy_*.txt")
    returncode_candidates = [root / "returncode", tlparse_root / f"{phase}.returncode"]
    returncode_candidates.extend(sorted(root.glob("*.returncode")))
    returncode_paths = sorted(
        {path.resolve() for path in returncode_candidates if path.exists()}
    )
    returncodes: dict[str, int | None] = {}
    for path in returncode_paths:
        try:
            returncodes[str(path)] = int(path.read_text().strip())
        except ValueError:
            returncodes[str(path)] = None
    stderr_candidates = [
        root / "stderr.log",
        root / "tlparse.stderr.log",
        tlparse_root / f"{phase}.stderr.log",
    ]
    stderr_candidates.extend(sorted(root.glob("*.stderr.log")))
    stderr_paths = sorted(
        {path.resolve() for path in stderr_candidates if path.exists()}
    )
    stderr_diagnostics = {}
    unexpected_stderr = {}
    allowed_stderr = re.compile(
        r"^(?:Detected rank: Some\(\d+\)|Stats \{.*\}|"
        r"Unknown fields: \{.*\} \(consider updating tlparse to render these\))$"
    )
    for path in stderr_paths:
        contents = path.read_text(errors="replace")
        if contents.strip():
            lines = [line for line in contents.splitlines() if line.strip()]
            stderr_diagnostics[str(path)] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "lines": lines,
            }
            unexpected = [line for line in lines if not allowed_stderr.fullmatch(line)]
            if unexpected:
                unexpected_stderr[str(path)] = unexpected
    provenance_candidates = [
        root / "compile_trace.sha256",
        tlparse_root / f"{phase}.compile_trace.sha256",
    ]
    provenance_paths = sorted(
        {path.resolve() for path in provenance_candidates if path.exists()}
    )
    recorded_sha256 = None
    if len(provenance_paths) == 1:
        match = re.match(r"^([0-9a-fA-F]{64})(?:\s|$)", provenance_paths[0].read_text())
        if match:
            recorded_sha256 = match.group(1).lower()
    compile_trace_sha256 = (
        sha256(compile_trace_paths[0]) if len(compile_trace_paths) == 1 else None
    )
    expected_artifact_counts = {
        "after_joint_graph": 1,
        "activation_memory_policy": 1,
        "autoparallel_solution": 1 if arm == "autoparallel" else 0,
        "autoparallel_parallel_graph": 1 if arm == "autoparallel" else 0,
    }
    artifact_counts = {
        "after_joint_graph": len(joint_paths),
        "activation_memory_policy": len(policy_paths),
        "autoparallel_solution": len(solution_paths),
        "autoparallel_parallel_graph": len(graph_paths),
    }
    result: dict[str, Any] = {
        "status": "failed",
        "root": str(root),
        "returncodes": returncodes,
        "stderr_paths": [str(path) for path in stderr_paths],
        "stderr_diagnostics": stderr_diagnostics,
        "unexpected_stderr": unexpected_stderr,
        "provenance_paths": [str(path) for path in provenance_paths],
        "recorded_compile_trace_sha256": recorded_sha256,
        "compile_trace_paths": [str(path) for path in compile_trace_paths],
        "observed_compile_trace_sha256": compile_trace_sha256,
        "expected_artifact_counts": expected_artifact_counts,
        "artifact_counts": artifact_counts,
        "solution_paths": solution_paths,
        "parallel_graph_paths": graph_paths,
        "after_joint_graph_paths": joint_paths,
        "activation_memory_policy_paths": policy_paths,
        "solution": None,
        "autoparallel_parallel_graph_sdpa": None,
        "after_joint_graph_sdpa": None,
        "activation_memory_policy": None,
        "errors": [],
    }
    if len(returncode_paths) != 1 or list(returncodes.values()) != [0]:
        result["errors"].append(
            {
                "returncode_paths": [str(path) for path in returncode_paths],
                "values": returncodes,
            }
        )
    if unexpected_stderr:
        result["errors"].append({"unexpected_stderr": unexpected_stderr})
    if len(provenance_paths) != 1 or recorded_sha256 is None:
        result["errors"].append(
            {
                "provenance_paths": [str(path) for path in provenance_paths],
                "error": "expected one valid compile-trace SHA256 provenance file",
            }
        )
    elif recorded_sha256 != compile_trace_sha256:
        result["errors"].append(
            {
                "recorded_compile_trace_sha256": recorded_sha256,
                "observed_compile_trace_sha256": compile_trace_sha256,
            }
        )
    if artifact_counts != expected_artifact_counts:
        result["errors"].append(
            {
                "expected_artifact_counts": expected_artifact_counts,
                "observed_artifact_counts": artifact_counts,
            }
        )
    if solution_path:
        try:
            result["solution"] = solution_summary(
                json.loads(solution_path.read_text()),
                expected_local_batch,
                required_mode,
            )
        except Exception as error:
            result["errors"].append(
                {
                    "path": str(solution_path),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    if graph_path:
        try:
            result["autoparallel_parallel_graph_sdpa"] = graph_sdpa_summary(
                graph_path.read_text(errors="replace"),
                expected_local_batch,
                required_mode,
            )
        except Exception as error:
            result["errors"].append(
                {"path": str(graph_path), "error": f"{type(error).__name__}: {error}"}
            )
    if joint_path:
        try:
            result["after_joint_graph_sdpa"] = graph_sdpa_summary(
                joint_path.read_text(errors="replace"),
                expected_local_batch,
                required_mode,
            )
        except Exception as error:
            result["errors"].append(
                {"path": str(joint_path), "error": f"{type(error).__name__}: {error}"}
            )
    if policy_path:
        result["activation_memory_policy"] = activation_policy_summary(
            policy_path.read_text(errors="replace")
        )
    result["sdpa_mode_consistency"] = sdpa_mode_consistency(
        {
            "autoparallel_solution": (
                result["solution"].get("sdpa") if result["solution"] else None
            ),
            "autoparallel_parallel_graph": result["autoparallel_parallel_graph_sdpa"],
            "after_joint_graph": result["after_joint_graph_sdpa"],
        }
    )
    checks = {
        "returncode_zero": len(returncode_paths) == 1
        and list(returncodes.values()) == [0],
        "compile_trace_provenance": len(provenance_paths) == 1
        and recorded_sha256 is not None
        and recorded_sha256 == compile_trace_sha256,
        "exact_required_artifacts": artifact_counts == expected_artifact_counts,
        "after_joint_forward_backward_sdpa": result.get(
            "after_joint_graph_sdpa", {}
        ).get("status")
        == "passed",
        "activation_memory_policy": result.get("activation_memory_policy", {}).get(
            "tag_count", 0
        )
        > 0,
        "manual_zero_ap_solution_artifacts": arm != "manual"
        or artifact_counts["autoparallel_solution"] == 0,
        "ap_solution": arm != "autoparallel"
        or (
            result.get("solution", {}).get("mesh_valid") is True
            and result.get("solution", {}).get("sdpa", {}).get("status") == "passed"
        ),
        "ap_parallel_graph": arm != "autoparallel"
        or result.get("autoparallel_parallel_graph_sdpa", {}).get("status") == "passed",
        "sdpa_mode_consistent_across_artifacts": result["sdpa_mode_consistency"].get(
            "status"
        )
        == "passed",
        "no_parse_errors": not result["errors"],
    }
    result["checks"] = checks
    if all(checks.values()):
        result["status"] = "passed"
    return result


def trace_result(
    run_root: Path,
    phase: str,
    arm: str,
    expected_local_batch: int,
    required_mode: str | None,
    profiler_step: int,
    output_dir: Path,
    tlparse_root: Path,
) -> dict[str, Any]:
    root = run_root / phase
    kineto_paths = sorted(
        (root / "job/profiling/traces").glob("**/rank0_trace.json.gz")
    )
    compile_paths = sorted(
        (root / "compile_trace").glob("dedicated_log_torch_trace_rank_0_*.log")
    )
    result: dict[str, Any] = {
        "kineto_paths": [str(path) for path in kineto_paths],
        "compile_trace_paths": [str(path) for path in compile_paths],
        "required_parallelism_mode": required_mode,
        "kineto": {"status": "not_found"},
        "compile_trace": {"status": "not_found"},
        "tlparse": tlparse_summary(
            phase,
            arm,
            expected_local_batch,
            required_mode,
            tlparse_root,
            compile_paths,
        ),
    }
    if len(kineto_paths) == 1:
        try:
            kineto, slim = kineto_summary(kineto_paths[0], profiler_step)
            slim_root = output_dir / "slim_traces" / phase
            slim_root.mkdir(parents=True, exist_ok=True)
            slim_path = slim_root / f"rank0.profiler_step_{profiler_step}.json.gz"
            with gzip.open(slim_path, "wt") as destination:
                json.dump(slim.pop("trace"), destination, separators=(",", ":"))
            kineto["slim_trace"] = {
                "path": str(slim_path),
                "compressed_bytes": slim_path.stat().st_size,
                **slim,
            }
            result["kineto"] = kineto
        except Exception as error:
            result["kineto"] = {
                "status": "parse_error",
                "path": str(kineto_paths[0]),
                "error": f"{type(error).__name__}: {error}",
            }
    elif len(kineto_paths) > 1:
        result["kineto"] = {"status": "ambiguous", "paths": result["kineto_paths"]}
    if len(compile_paths) == 1:
        try:
            result["compile_trace"] = compile_trace_summary(
                compile_paths[0], expected_local_batch, required_mode
            )
        except Exception as error:
            result["compile_trace"] = {
                "status": "parse_error",
                "path": str(compile_paths[0]),
                "error": f"{type(error).__name__}: {error}",
            }
    elif len(compile_paths) > 1:
        result["compile_trace"] = {
            "status": "ambiguous",
            "paths": result["compile_trace_paths"],
        }
    return result


def job_summary(attempt: Path) -> dict[str, Any]:
    status_paths = [
        path
        for path in (
            attempt / "job_status.final.json",
            attempt / "submission/formal/job_status.final.json",
        )
        if path.exists()
    ]
    job_id_paths = [
        path
        for path in (attempt / "job_id.txt", attempt / "submission/formal/job_id.txt")
        if path.exists()
    ]
    job_ids = {path.read_text().strip() for path in job_id_paths}
    status_hashes = {sha256(path) for path in status_paths}
    result: dict[str, Any] = {
        "attempt": str(attempt),
        "job_status_paths": [str(path) for path in status_paths],
        "job_id_paths": [str(path) for path in job_id_paths],
        "job_id": next(iter(job_ids)) if len(job_ids) == 1 else None,
        "status": "not_found",
    }
    if not status_paths or not job_id_paths:
        return result
    if len(status_hashes) != 1 or len(job_ids) != 1:
        if status_paths or job_id_paths:
            result["status"] = "ambiguous"
        return result
    status_path = status_paths[0]
    try:
        status = json.loads(status_path.read_text())["data"]
        failed = sum(
            group.get("numFailedTasks", 0)
            for groups in status.get("latestAttempt", {})
            .get("taskGroupExecutionAttempts", {})
            .values()
            for group in groups
        )
        result.update(
            status="available",
            hpc_job_name=status.get("hpcJobName"),
            state=status.get("state"),
            restarts=status.get("numRestarts", 0),
            failed_tasks=failed,
            latest_attempt_index=status.get("latestAttempt", {}).get("attemptIndex"),
        )
    except Exception as error:
        result.update(status="parse_error", error=f"{type(error).__name__}: {error}")
    return result


def final_job_attempt_index(attempt: Path) -> int:
    job = job_summary(attempt)
    attempt_index = job.get("latest_attempt_index")
    if job.get("status") != "available" or type(attempt_index) is not int:
        raise RuntimeError(
            "final job status does not contain an integer "
            f"latestAttempt.attemptIndex: {job}"
        )
    return attempt_index


def run_root_attempt_index(run_root: Path) -> int:
    match = RUN_ROOT_PATTERN.fullmatch(run_root.name)
    if match is None:
        raise RuntimeError(f"unrelated run root: {run_root}")
    return int(match.group("attempt_index"))


def validate_run_root_attempt(attempt: Path, run_root: Path) -> int:
    expected = final_job_attempt_index(attempt)
    observed = run_root_attempt_index(run_root)
    if observed != expected:
        raise RuntimeError(
            "run root belongs to stale MAST attempt "
            f"{observed}; final job status reports attempt {expected}: {run_root}"
        )
    return observed


def discover_run_root(attempt: Path) -> Path:
    run_output = attempt / "artifacts/run_output"
    if not run_output.is_dir():
        raise RuntimeError(f"run-output directory does not exist: {run_output}")
    expected_attempt_index = final_job_attempt_index(attempt)
    candidates = sorted(
        path
        for path in run_output.iterdir()
        if path.is_dir()
        and RUN_ROOT_PATTERN.fullmatch(path.name)
        and run_root_attempt_index(path) == expected_attempt_index
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one run root for final MAST attempt {expected_attempt_index} "
            f"under {attempt}, got {candidates}"
        )
    return candidates[0]


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    if args.run_root:
        run_root = args.run_root.resolve()
        try:
            inferred_attempt = run_root.parents[2]
        except IndexError as error:
            raise RuntimeError(f"unrelated run root: {run_root}") from error
        attempt = args.attempt.resolve() if args.attempt else inferred_attempt
        if attempt != inferred_attempt:
            raise RuntimeError("--attempt and --run-root refer to different attempts")
    elif args.attempt:
        attempt = args.attempt.resolve()
        run_root = discover_run_root(attempt)
    else:
        raise RuntimeError("one of --attempt or --run-root is required")
    attempts_root = (TASK_ROOT / "job/attempts").resolve()
    if attempt.parent != attempts_root or not re.fullmatch(r"\d+", attempt.name):
        raise RuntimeError(f"unrelated attempt path: {attempt}")
    expected_run_parent = (attempt / "artifacts/run_output").resolve()
    if run_root.parent != expected_run_parent or not RUN_ROOT_PATTERN.fullmatch(
        run_root.name
    ):
        raise RuntimeError(f"unrelated run root: {run_root}")
    if not attempt.is_dir() or not run_root.is_dir():
        raise RuntimeError(f"run root does not exist: {run_root}")
    validate_run_root_attempt(attempt, run_root)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tlparse_root = args.tlparse_root.resolve()
    if not tlparse_root.is_dir():
        raise RuntimeError(f"tlparse root does not exist: {tlparse_root}")
    return attempt, run_root, output_dir, tlparse_root


def paired_numerics(
    manual: dict[str, Any], treatment: dict[str, Any]
) -> dict[str, Any]:
    manual_values = {
        (value["rank"], value["step"]): (value["loss"], value["grad_norm"])
        for value in manual["final_by_rank"]
    }
    treatment_values = {
        (value["rank"], value["step"]): (value["loss"], value["grad_norm"])
        for value in treatment["final_by_rank"]
    }
    common = sorted(set(manual_values) & set(treatment_values))
    return {
        "common_final_rank_steps": [list(key) for key in common],
        "loss_absolute_differences": [
            abs(manual_values[key][0] - treatment_values[key][0]) for key in common
        ],
        "grad_norm_absolute_differences": [
            abs(manual_values[key][1] - treatment_values[key][1]) for key in common
        ],
    }


def paired_metric(
    manual_samples: list[float], treatment_samples: list[float]
) -> dict[str, Any]:
    if len(manual_samples) != len(treatment_samples) or not manual_samples:
        raise RuntimeError("paired metric samples must be non-empty and aligned")
    deltas = [
        treatment - manual
        for manual, treatment in zip(manual_samples, treatment_samples, strict=True)
    ]
    manual = summarize(manual_samples)
    treatment = summarize(treatment_samples)
    ratio = treatment["mean"] / manual["mean"] if manual["mean"] != 0 else None
    return {
        "manual": manual,
        "autoparallel": treatment,
        "ap_minus_manual": summarize(deltas),
        "mean_ratio_ap_over_manual": ratio,
        "mean_reduction_fraction": 1 - ratio if ratio is not None else None,
    }


def paired_performance(
    case_name: str,
    phases: dict[str, dict[str, Any]],
    config_parity: dict[str, Any],
    allocation: dict[str, Any],
) -> dict[str, Any]:
    case = CASES[case_name]
    arm_phases = case["performance"]
    reasons = []
    for arm, phase in arm_phases.items():
        if (
            phases[phase]["execution"]["state"] != "completed"
            or not phases[phase]["execution"]["consistent"]
        ):
            reasons.append(
                f"{arm} phase state is {phases[phase]['execution']['state']}"
            )
        if phases[phase]["tensorboard"].get("status") != "available":
            reasons.append(f"{arm} TensorBoard metrics unavailable")
        elif not phases[phase]["tensorboard"].get("all_scalar_values_finite", False):
            reasons.append(f"{arm} TensorBoard contains non-finite values")
        if not phases[phase]["numerics"]["all_finite"]:
            reasons.append(f"{arm} loss or grad norm is unavailable/non-finite")
    if config_parity.get("status") != "passed":
        reasons.append("paired configs are not proven identical except AP toggle")
    if not allocation["pair_validity"].get(f"{case_name}.performance", False):
        reasons.append("paired allocation or rank-to-GPU mapping differs")
    if reasons:
        return {"status": "not_comparable", "reasons": reasons}
    manual = phases[arm_phases["manual"]]["tensorboard"]
    treatment = phases[arm_phases["autoparallel"]]["tensorboard"]
    required = (
        "latency_s",
        "data_loading_s",
        "data_loading_percent",
        "per_device_tps",
        "aggregate_tps",
    )
    for arm, value in (("manual", manual), ("autoparallel", treatment)):
        for metric in required:
            if value["steady"][metric]["missing_steady_steps"]:
                reasons.append(f"{arm} {metric} lacks all steady steps")
    if reasons:
        return {"status": "not_comparable", "reasons": reasons}
    for arm, value in (("manual", manual), ("autoparallel", treatment)):
        for metric in ("active_memory_gib", "reserved_memory_gib"):
            if (
                value[metric]["summary"] is None
                or value[metric]["missing_steady_steps"]
            ):
                reasons.append(f"{arm} {metric} lacks all steady steps")
    if reasons:
        return {"status": "not_comparable", "reasons": reasons}

    def samples(value: dict[str, Any], metric: str) -> list[float]:
        return value["steady"][metric]["summary"]["samples"]

    manual_core = [
        latency - loading
        for latency, loading in zip(
            samples(manual, "latency_s"),
            samples(manual, "data_loading_s"),
            strict=True,
        )
    ]
    treatment_core = [
        latency - loading
        for latency, loading in zip(
            samples(treatment, "latency_s"),
            samples(treatment, "data_loading_s"),
            strict=True,
        )
    ]
    active_memory = paired_metric(
        manual["active_memory_gib"]["samples"],
        treatment["active_memory_gib"]["samples"],
    )
    active_memory["peak"] = {
        "manual": manual["active_memory_gib"]["peak"],
        "autoparallel": treatment["active_memory_gib"]["peak"],
        "ap_minus_manual": treatment["active_memory_gib"]["peak"]
        - manual["active_memory_gib"]["peak"],
    }
    reserved_memory = paired_metric(
        manual["reserved_memory_gib"]["samples"],
        treatment["reserved_memory_gib"]["samples"],
    )
    reserved_memory["peak"] = {
        "manual": manual["reserved_memory_gib"]["peak"],
        "autoparallel": treatment["reserved_memory_gib"]["peak"],
        "ap_minus_manual": treatment["reserved_memory_gib"]["peak"]
        - manual["reserved_memory_gib"]["peak"],
    }
    return {
        "status": "comparable",
        "measurement": {
            "steady_log_windows_ending_at_steps": list(STEADY_STEPS),
            "variance": (
                "sample standard deviation across four five-step logging windows; "
                "one paired allocation"
            ),
            "latency": "TensorBoard time_metrics/end_to_end(s)",
            "data_loading": "TensorBoard time_metrics/data_loading(s) and (%)",
            "aggregate_throughput": (
                f"TensorBoard throughput(tps) multiplied by world size {WORLD_SIZE}"
            ),
            "training_core_residual": (
                "per-step end_to_end(s) minus data_loading(s); not an independently "
                "instrumented timer"
            ),
            "memory_scope": "rank0_only",
        },
        "latency_s": paired_metric(
            samples(manual, "latency_s"), samples(treatment, "latency_s")
        ),
        "data_loading_s": paired_metric(
            samples(manual, "data_loading_s"),
            samples(treatment, "data_loading_s"),
        ),
        "data_loading_percent": paired_metric(
            samples(manual, "data_loading_percent"),
            samples(treatment, "data_loading_percent"),
        ),
        "training_core_residual_s": paired_metric(manual_core, treatment_core),
        "aggregate_tps": paired_metric(
            samples(manual, "aggregate_tps"), samples(treatment, "aggregate_tps")
        ),
        "memory_gib": {
            "scope": "rank0_only",
            "active": active_memory,
            "reserved": reserved_memory,
        },
        "numerics": paired_numerics(
            phases[arm_phases["manual"]]["numerics"],
            phases[arm_phases["autoparallel"]]["numerics"],
        ),
    }


def paired_trace_components(
    case_name: str,
    phases: dict[str, dict[str, Any]],
    config_parity: dict[str, Any],
    allocation: dict[str, Any],
) -> dict[str, Any]:
    trace_phases = CASES[case_name]["trace"]
    reasons = []
    for arm, phase in trace_phases.items():
        value = phases[phase]
        if value["execution"]["state"] != "completed":
            reasons.append(f"{arm} trace state is {value['execution']['state']}")
        if value["trace"]["kineto"].get("status") != "available":
            reasons.append(f"{arm} Kineto trace is unavailable")
        if value["trace"]["compile_trace"].get("status") != "available":
            reasons.append(f"{arm} compile trace is unavailable")
        if value["trace"]["tlparse"].get("status") != "passed":
            reasons.append(f"{arm} tlparse validation did not pass")
    if config_parity.get("status") != "passed":
        reasons.append("paired trace configs are not identical except AP toggle")
    if not allocation["pair_validity"].get(f"{case_name}.trace", False):
        reasons.append("paired trace allocation or rank-to-GPU mapping differs")
    if reasons:
        return {"status": "not_comparable", "reasons": reasons}

    manual_trace = phases[trace_phases["manual"]]["trace"]
    ap_trace = phases[trace_phases["autoparallel"]]["trace"]
    manual = manual_trace["kineto"]
    treatment = ap_trace["kineto"]

    def pair(manual_value: float | int, treatment_value: float | int) -> dict[str, Any]:
        ratio = treatment_value / manual_value if manual_value else None
        return {
            "manual": manual_value,
            "autoparallel": treatment_value,
            "ap_minus_manual": treatment_value - manual_value,
            "ratio_ap_over_manual": ratio,
        }

    scalar_names = (
        "step_duration_ms",
        "gpu_busy_union_ms",
        "noncomm_union_ms",
        "comm_raw_ms",
        "comm_union_ms",
        "comm_exposed_ms",
        "kernel_count",
        "comm_kernel_count",
    )
    categories = {}
    for category in sorted(
        set(manual["kernel_categories"]) | set(treatment["kernel_categories"])
    ):
        manual_category = manual["kernel_categories"].get(
            category, {"raw_ms": 0.0, "count": 0}
        )
        treatment_category = treatment["kernel_categories"].get(
            category, {"raw_ms": 0.0, "count": 0}
        )
        categories[category] = {
            "raw_ms": pair(manual_category["raw_ms"], treatment_category["raw_ms"]),
            "count": pair(manual_category["count"], treatment_category["count"]),
        }
    return {
        "status": "comparable",
        "measurement": {
            "scope": f"rank 0 Kineto ProfilerStep#{PROFILER_STEP}",
            "variance": "single trace step; no variance estimate",
            "category_duration": (
                "sum of clipped kernel durations; categories and raw durations can "
                "overlap across CUDA streams and are not additive wall time"
            ),
        },
        "totals": {name: pair(manual[name], treatment[name]) for name in scalar_names},
        "kernel_categories": categories,
        "sdpa_parallelism_modes": {
            "manual": manual_trace["compile_trace"]["sdpa_mode_consistency"].get(
                "selected_parallelism_mode"
            ),
            "autoparallel": ap_trace["compile_trace"]["sdpa_mode_consistency"].get(
                "selected_parallelism_mode"
            ),
        },
        "collectives": {
            "manual": manual["collectives"],
            "autoparallel": treatment["collectives"],
        },
    }


def validate(
    job: dict[str, Any],
    preflight: dict[str, Any],
    inventory: dict[str, Any],
    allocation: dict[str, Any],
    phases: dict[str, dict[str, Any]],
    cases: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    checks: dict[str, bool] = {
        "job_complete": job.get("state") == "COMPLETE",
        "initial_scheduler_attempt_only": job.get("latest_attempt_index") == 0,
        "zero_restarts": job.get("restarts") == 0,
        "zero_failed_tasks": job.get("failed_tasks") == 0,
        "runtime_preflight": preflight["status"] == "passed",
        "exact_four_phase_inventory": inventory["status"] == "passed",
        "allocation_identical_all_phases": allocation["valid"],
    }
    performance_phases = [
        phase for case in CASES.values() for phase in case["performance"].values()
    ]
    trace_phases = [
        phase for case in CASES.values() for phase in case["trace"].values()
    ]
    checks["all_performance_completed_or_oom"] = all(
        phases[phase]["execution"]["state"] in {"completed", "oom"}
        for phase in performance_phases
    )
    checks["all_trace_phases_completed"] = all(
        phases[phase]["execution"]["state"] == "completed" for phase in trace_phases
    )
    oom_phases = []
    for phase, value in phases.items():
        state = value["execution"]["state"]
        if state == "oom":
            oom_phases.append(phase)
        checks[f"{phase}.execution_markers"] = value["execution"]["consistent"]
        checks[f"{phase}.expected_terminal_state"] = state in {"completed", "oom"}
        if state == "completed":
            checks[f"{phase}.finite_loss_and_grad_norm"] = value["numerics"][
                "all_finite"
            ]
            if value["kind"] == "performance":
                tb = value["tensorboard"]
                checks[f"{phase}.tensorboard"] = (
                    tb.get("status") == "available"
                    and tb.get("all_scalar_values_finite") is True
                    and all(
                        not tb["steady"][metric]["missing_steady_steps"]
                        for metric in (
                            "latency_s",
                            "data_loading_s",
                            "data_loading_percent",
                            "per_device_tps",
                        )
                    )
                    and not tb["active_memory_gib"]["missing_steady_steps"]
                    and not tb["reserved_memory_gib"]["missing_steady_steps"]
                )
            else:
                trace = value["trace"]
                checks[f"{phase}.kineto"] = (
                    trace["kineto"].get("status") == "available"
                    and trace["kineto"].get("profiler_step") == PROFILER_STEP
                    and trace["kineto"].get("kernel_count", 0) > 0
                    and trace["kineto"].get("comm_kernel_count", 0) > 0
                )
                compile_trace = trace["compile_trace"]
                checks[f"{phase}.compile_trace"] = (
                    compile_trace.get("status") == "available"
                )
                expected_artifact_counts = {
                    "activation_memory_policy": 1,
                    "after_joint_graph": 1,
                    "autoparallel_parallel_graph": (
                        1 if value["arm"] == "autoparallel" else 0
                    ),
                    "autoparallel_solution": (
                        1 if value["arm"] == "autoparallel" else 0
                    ),
                }
                checks[f"{phase}.compile_trace_required_artifacts"] = all(
                    compile_trace.get("interesting_artifact_counts", {}).get(name)
                    == count
                    for name, count in expected_artifact_counts.items()
                )
                checks[f"{phase}.activation_memory_policy"] = (
                    compile_trace.get("activation_memory_policy", {}).get(
                        "tag_count", 0
                    )
                    > 0
                )
                checks[f"{phase}.after_joint_forward_backward_sdpa"] = (
                    compile_trace.get("after_joint_graph_sdpa", {}).get("status")
                    == "passed"
                )
                checks[f"{phase}.sdpa_mode_consistent_across_artifacts"] = (
                    compile_trace.get("sdpa_mode_consistency", {}).get("status")
                    == "passed"
                )
                checks[f"{phase}.tlparse"] = trace["tlparse"].get("status") == "passed"
                if value["arm"] == "manual":
                    checks[f"{phase}.manual_has_zero_ap_solution_artifacts"] = (
                        compile_trace.get("interesting_artifact_counts", {}).get(
                            "autoparallel_solution"
                        )
                        == 0
                    )
                else:
                    solution = compile_trace.get("solution") or {}
                    graph = compile_trace.get("autoparallel_parallel_graph_sdpa") or {}
                    checks[f"{phase}.ap_solution_sdpa"] = (
                        solution.get("mesh_valid") is True
                        and solution.get("sdpa", {}).get("status") == "passed"
                    )
                    checks[f"{phase}.ap_parallel_graph_sdpa"] = (
                        graph.get("status") == "passed"
                    )
    for case_name, value in cases.items():
        for phase_kind in ("performance", "trace"):
            parity = value[phase_kind]["config_parity"]
            checks[f"{case_name}.{phase_kind}.config_parity"] = (
                parity["status"] == "passed"
            )
            checks[f"{case_name}.{phase_kind}.allocation_parity"] = allocation[
                "pair_validity"
            ].get(f"{case_name}.{phase_kind}", False)
    checks["oom_only_allowed_for_performance"] = set(oom_phases).issubset(
        performance_phases
    )
    return {
        "status": (
            "failed"
            if not all(checks.values())
            else "passed_with_oom" if oom_phases else "passed"
        ),
        "checks": checks,
        "oom_phases": oom_phases,
        "note": (
            "Either performance arm may be accepted as OOM, and later phases must "
            "still be present. Both trace phases must complete; no paired performance "
            "result is computed unless both performance arms complete and fairness "
            "checks pass."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze the paired 16-GPU Muse Glimmer SDPA-fix formal run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--attempt", type=Path, help="MAST attempt directory")
    parser.add_argument("--run-root", type=Path, help="Extracted run-output directory")
    parser.add_argument(
        "--output-dir", type=Path, default=TASK_ROOT / "analysis/formal"
    )
    parser.add_argument(
        "--tlparse-root",
        type=Path,
        required=True,
        help="Directory containing one provenance-bound tlparse subdirectory per phase",
    )
    parser.add_argument(
        "--profiler-step",
        type=int,
        default=PROFILER_STEP,
        help="Kineto ProfilerStep to analyze (validation requires step 4)",
    )
    args = parser.parse_args()
    attempt, run_root, output_dir, tlparse_root = resolve_paths(args)

    job = job_summary(attempt)
    preflight = preflight_summary(attempt, run_root)
    inventory = phase_inventory(run_root)
    allocation = allocation_check(run_root)
    phases = {}
    cases = {}
    for case_name, case in CASES.items():
        case_result: dict[str, Any] = {
            "contract": {**case, "performance": None, "trace": None}
        }
        case_result["contract"].update(
            world_size=WORLD_SIZE,
            fsdp_degree=FSDP_DEGREE,
            tensor_parallel_degree=TP_DEGREE,
            sequence_length=SEQUENCE_LENGTH,
        )
        for phase_kind in ("performance", "trace"):
            configs = {}
            for arm, phase in case[phase_kind].items():
                phase_root = run_root / phase
                value = {
                    "case": case_name,
                    "kind": phase_kind,
                    "arm": arm,
                    "phase": phase,
                    "required_parallelism_mode": case["required_sdpa_modes"][arm],
                    "root": str(phase_root),
                    "execution": phase_state(run_root, phase),
                    "config": phase_config(phase_root, case, arm),
                    "numerics": training_numerics(phase_root),
                    "setup_timings": setup_timings(phase_root),
                }
                if phase_kind == "performance":
                    value["tensorboard"] = tensorboard_result(phase_root)
                else:
                    value["trace"] = trace_result(
                        run_root,
                        phase,
                        arm,
                        case["local_batch_size_per_dp_rank"],
                        case["required_sdpa_modes"][arm],
                        args.profiler_step,
                        output_dir,
                        tlparse_root,
                    )
                phases[phase] = value
                configs[arm] = value["config"]
            case_result[phase_kind] = {
                "phases": case[phase_kind],
                "config_parity": pair_config_parity(configs),
            }
            for config in configs.values():
                config.pop("normalized", None)
        cases[case_name] = case_result
    for case_name, value in cases.items():
        value["paired_performance"] = paired_performance(
            case_name,
            phases,
            value["performance"]["config_parity"],
            allocation,
        )
        value["paired_trace_components"] = paired_trace_components(
            case_name,
            phases,
            value["trace"]["config_parity"],
            allocation,
        )

    sac_evidence = {
        "runtime_preflight_outer_passes": preflight.get("outer_sac_pass_audit"),
        "trace_activation_memory_policy": {
            phase: value["trace"]["compile_trace"].get("activation_memory_policy")
            for phase, value in phases.items()
            if value["kind"] == "trace"
        },
        "claim": (
            "The shared outer SAC pass prefix and eager memory policy are checked; "
            "tag-count equality is not claimed because the two physical graphs differ."
        ),
    }

    validation = validate(job, preflight, inventory, allocation, phases, cases)
    report = {
        "analysis_contract": {
            "declared_variable": "compile.enable_autoparallel",
            "world_size": WORLD_SIZE,
            "mesh": {"fsdp": FSDP_DEGREE, "tp": TP_DEGREE},
            "sequence_length": SEQUENCE_LENGTH,
            "cases": {
                name: {
                    key: value[key]
                    for key in (
                        "local_batch_size_per_dp_rank",
                        "global_batch_size",
                        "per_physical_gpu_effective_batch_size",
                        "tokens_per_step",
                        "required_sdpa_modes",
                    )
                }
                for name, value in CASES.items()
            },
            "steady_log_windows_ending_at_steps": list(STEADY_STEPS),
            "profiler_step": args.profiler_step,
        },
        "raw_paths": {
            "attempt": str(attempt),
            "run_root": str(run_root),
            "output_dir": str(output_dir),
            "tlparse_root": str(tlparse_root) if tlparse_root else None,
            "launch_script": str(TASK_ROOT / "launch.sh"),
            "mast_launcher": str(TASK_ROOT / "launcher/mast.py"),
            "rank_runner": str(TASK_ROOT / "runner/run_rank.sh"),
        },
        "analysis_invocation": {
            "argv": [sys.executable, *sys.argv],
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256(Path(__file__).resolve()),
        },
        "job": job,
        "runtime_preflight": preflight,
        "phase_inventory": inventory,
        "sac_evidence": sac_evidence,
        "allocation": allocation,
        "cases": cases,
        "phases": phases,
        "validation": validation,
    }
    (output_dir / "formal_analysis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "formal_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n"
    )
    trace_results = {
        phase: value["trace"]
        for phase, value in phases.items()
        if value["kind"] == "trace"
    }
    (output_dir / "trace_results.json").write_text(
        json.dumps(trace_results, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "status": validation["status"],
                "oom_phases": validation["oom_phases"],
                "paired_performance": {
                    name: value["paired_performance"] for name, value in cases.items()
                },
                "report": str(output_dir / "formal_analysis.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    raise SystemExit(0 if validation["status"] in {"passed", "passed_with_oom"} else 1)


if __name__ == "__main__":
    main()
