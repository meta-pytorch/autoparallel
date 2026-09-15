from __future__ import annotations

import hashlib
import json
import math
import statistics
import struct
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .campaign import PLANNER_RESULT_CLASSIFICATION, Campaign, write_json
from .sources import manifest_digest, tree_manifest


TIMER_EVENTS = (
    "step_end",
    "gc_collect_end",
    "fetching_batch_end",
    "post_dataloading_process_end",
    "fwd_bwd_end",
    "optim_end",
    "collect_dist_metrics_end",
    "checkpoint_save_end",
)
ALLOCATION_IDENTITY_FIELDS = (
    "hostname",
    "rank",
    "local_rank",
    "world_size",
    "local_world_size",
    "gpu_ordinal",
    "gpu_name",
    "gpu_pci_bus_id",
    "gpu_uuid",
    "cuda_visible_devices",
    "device_network_id",
    "device_backend_network_topology",
)
MEMORY_TAGS = (
    "memory/max_active(GiB)",
    "memory/max_reserved(GiB)",
    "memory/num_alloc_retries",
    "memory/num_ooms",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(values: list[float]) -> dict[str, Any] | None:
    if not values:
        return None
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "population_sd": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
        "samples": values,
    }


def _structured_records(
    phase_root: Path, steps: list[int], world_size: int
) -> dict[str, dict[int, dict[int, dict[str, float | int]]]]:
    wanted_steps = set(steps)
    values: dict[str, dict[int, dict[int, dict[str, float | int]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    paths = sorted(
        (phase_root / "job/structured_logs").glob("training.global_rank_*.jsonl")
    )
    if len(paths) != world_size:
        raise RuntimeError(
            f"{phase_root}: expected {world_size} structured logs, found {len(paths)}"
        )
    for path in paths:
        with path.open(errors="replace") as stream:
            for line_number, line in enumerate(stream, 1):
                row = json.loads(line)
                step = row.get("step")
                rank = row.get("global_rank", row.get("rank"))
                name = row.get("log_type_name")
                value = row.get("value")
                if step not in wanted_steps or name not in TIMER_EVENTS:
                    continue
                if not isinstance(rank, int) or not isinstance(value, (int, float)):
                    raise RuntimeError(f"{path}:{line_number}: malformed timer record")
                current = values[name][step].get(rank)
                if current is None:
                    values[name][step][rank] = {
                        "duration_ms": float(value),
                        "time_us": int(row["time_us"]),
                    }
                elif name == "step_end":
                    raise RuntimeError(
                        f"{path}:{line_number}: duplicate step_end for step {step}"
                    )
                else:
                    current["duration_ms"] = float(current["duration_ms"]) + float(value)
                    current["time_us"] = max(int(current["time_us"]), int(row["time_us"]))
    expected_ranks = set(range(world_size))
    for event in ("step_end",):
        for step in steps:
            if set(values[event].get(step, {})) != expected_ranks:
                raise RuntimeError(f"{phase_root}: incomplete {event} records at step {step}")
    return values


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7


def _protobuf_fields(data: bytes):
    offset = 0
    while offset < len(data):
        key, offset = _read_varint(data, offset)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, offset = _read_varint(data, offset)
        elif wire == 1:
            value = data[offset : offset + 8]
            offset += 8
        elif wire == 2:
            length, offset = _read_varint(data, offset)
            value = data[offset : offset + length]
            offset += length
        elif wire == 5:
            value = data[offset : offset + 4]
            offset += 4
        else:
            raise RuntimeError(f"unsupported TensorBoard protobuf wire type {wire}")
        yield field, wire, value


def _tensorboard_scalars(event_path: Path) -> dict[str, dict[int, float]]:
    result: dict[str, dict[int, float]] = {}
    with event_path.open("rb") as source:
        while header := source.read(12):
            if len(header) != 12:
                raise RuntimeError(f"{event_path}: truncated TFRecord header")
            length = struct.unpack("<Q", header[:8])[0]
            record = source.read(length)
            footer = source.read(4)
            if len(record) != length or len(footer) != 4:
                raise RuntimeError(f"{event_path}: truncated TFRecord")
            fields = list(_protobuf_fields(record))
            step = next(
                (value for field, wire, value in fields if field == 2 and wire == 0),
                None,
            )
            payload = next(
                (value for field, wire, value in fields if field == 5 and wire == 2),
                None,
            )
            if step is None or payload is None:
                continue
            for field, wire, value in _protobuf_fields(payload):
                if field != 1 or wire != 2:
                    continue
                parts = list(_protobuf_fields(value))
                tag = next(
                    (item for number, kind, item in parts if number == 1 and kind == 2),
                    None,
                )
                scalar = next(
                    (item for number, kind, item in parts if number == 2 and kind == 5),
                    None,
                )
                if tag is not None and scalar is not None:
                    result.setdefault(tag.decode(), {})[int(step)] = float(
                        struct.unpack("<f", scalar)[0]
                    )
    return result


def _all_rank_memory(
    phase_root: Path, warmup_steps: int, world_size: int
) -> dict[str, Any]:
    paths = sorted((phase_root / "job/tb").glob("**/rank_*/events.out.tfevents.*"))
    buffered_paths = sorted(
        (phase_root / "job/benchmark_metrics").glob("rank_*.json")
    )
    if not paths and buffered_paths:
        errors = []
        values: dict[str, list[dict[str, float | int]]] = {
            tag: [] for tag in MEMORY_TAGS
        }
        observed_ranks = set()
        for path in buffered_paths:
            try:
                payload = json.loads(path.read_text())
                rank = int(payload["rank"])
                observed_ranks.add(rank)
                for record in payload["records"]:
                    step = int(record["step"])
                    if step <= warmup_steps:
                        continue
                    metrics = record["metrics"]
                    for tag in MEMORY_TAGS:
                        if tag in metrics:
                            values[tag].append(
                                {"rank": rank, "step": step, "value": float(metrics[tag])}
                            )
            except Exception as error:
                errors.append(f"invalid buffered metric file {path}: {error}")
        if observed_ranks != set(range(world_size)):
            errors.append(
                "buffered metric rank set mismatch: "
                f"expected={list(range(world_size))}, observed={sorted(observed_ranks)}"
            )
        summaries = {}
        for tag, rows in values.items():
            samples = [float(row["value"]) for row in rows]
            if not samples:
                errors.append(f"buffered metrics have no post-warmup samples for {tag}")
            peak_row = max(rows, key=lambda row: float(row["value"])) if rows else None
            summaries[tag] = {
                "all_rank_samples": _summary(samples),
                "peak": peak_row,
            }
        return {
            "status": "passed" if not errors else "failed",
            "measurement_method": (
                "maximum across every rank and post-warmup buffered metric window"
            ),
            "files": [str(path) for path in buffered_paths],
            "metrics": summaries,
            "errors": errors,
        }
    by_rank: dict[int, Path] = {}
    errors = []
    for path in paths:
        try:
            rank = int(path.parent.name.removeprefix("rank_"))
        except ValueError:
            errors.append(f"cannot parse TensorBoard rank from {path}")
            continue
        if rank in by_rank:
            errors.append(f"multiple TensorBoard event files for rank {rank}")
        by_rank[rank] = path

    expected_ranks = set(range(world_size))
    if set(by_rank) != expected_ranks:
        errors.append(
            "TensorBoard rank set mismatch: "
            f"expected={sorted(expected_ranks)}, observed={sorted(by_rank)}"
        )

    values: dict[str, list[dict[str, float | int]]] = {
        tag: [] for tag in MEMORY_TAGS
    }
    for rank, path in sorted(by_rank.items()):
        scalars = _tensorboard_scalars(path)
        for tag in MEMORY_TAGS:
            samples = {
                step: value
                for step, value in scalars.get(tag, {}).items()
                if step > warmup_steps
            }
            if not samples:
                errors.append(f"rank {rank} has no post-warmup samples for {tag}")
                continue
            values[tag].extend(
                {"rank": rank, "step": step, "value": float(value)}
                for step, value in sorted(samples.items())
            )

    summaries = {}
    for tag, rows in values.items():
        samples = [float(row["value"]) for row in rows]
        peak_row = max(rows, key=lambda row: float(row["value"])) if rows else None
        summaries[tag] = {
            "all_rank_samples": _summary(samples),
            "peak": peak_row,
        }
    return {
        "status": "passed" if not errors else "failed",
        "measurement_method": "maximum across every rank and post-warmup log window",
        "files": [str(path) for path in paths],
        "metrics": summaries,
        "errors": errors,
    }


def _structured_metrics(
    phase_root: Path, warmup_steps: int, world_size: int
) -> dict[str, Any]:
    paths = sorted(
        (phase_root / "job/structured_logs").glob("training.global_rank_*.jsonl")
    )
    values: dict[str, dict[int, dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    parse_errors = []
    ranks = set()
    for path in paths:
        with path.open(errors="replace") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                except Exception as error:
                    parse_errors.append(
                        {"path": str(path), "line": line_number, "error": str(error)}
                    )
                    continue
                step = row.get("step")
                rank = row.get("global_rank", row.get("rank"))
                name = row.get("log_type_name")
                value = row.get("value")
                if (
                    isinstance(step, int)
                    and step > warmup_steps
                    and isinstance(rank, int)
                    and name in TIMER_EVENTS
                    and isinstance(value, (int, float))
                    and math.isfinite(value)
                ):
                    ranks.add(rank)
                    if rank in values[name][step] and name == "step_end":
                        parse_errors.append(
                            {
                                "path": str(path),
                                "line": line_number,
                                "error": f"duplicate {name} for step {step}, rank {rank}",
                            }
                        )
                    elif rank in values[name][step]:
                        values[name][step][rank] += float(value)
                    else:
                        values[name][step][rank] = float(value)

    components = {}
    incomplete_steps = {}
    for name in TIMER_EVENTS:
        step_rows = values[name]
        ordered_steps = sorted(step_rows)
        incomplete = [
            step
            for step, rank_values in sorted(step_rows.items())
            if set(rank_values) != set(range(world_size))
        ]
        incomplete_steps[name] = incomplete
        components[name] = {
            "all_rank_samples": _summary(
                [
                    value
                    for rank_values in step_rows.values()
                    for value in rank_values.values()
                ]
            ),
            "rank0_steps": _summary(
                [step_rows[step][0] for step in ordered_steps if 0 in step_rows[step]]
            ),
            "per_step_rank_max": _summary(
                [max(step_rows[step].values()) for step in ordered_steps]
            ),
            "steps": ordered_steps,
        }
    complete = (
        len(paths) == world_size
        and ranks == set(range(world_size))
        and not parse_errors
        and components["step_end"]["per_step_rank_max"] is not None
        and not incomplete_steps["step_end"]
    )
    return {
        "status": "passed" if complete else "failed",
        "files": [str(path) for path in paths],
        "expected_rank_count": world_size,
        "observed_ranks": sorted(ranks),
        "parse_errors": parse_errors,
        "incomplete_steps": incomplete_steps,
        "measurement_method": "per-step maximum across all ranks after warmup",
        "components_ms": components,
    }


def _phase_exit_audit(root: Path, world_size: int) -> dict[str, Any]:
    paths = sorted((root / "runtime").glob("rank_*.exit_code"))
    exit_codes: dict[str, int] = {}
    errors = []
    for path in paths:
        try:
            rank = int(path.stem.removeprefix("rank_"))
            exit_codes[str(rank)] = int(path.read_text().strip())
        except ValueError as error:
            errors.append({"path": str(path), "error": str(error)})
    expected = {str(rank) for rank in range(world_size)}
    passed = (
        set(exit_codes) == expected
        and all(code == 0 for code in exit_codes.values())
        and not errors
        and (root / "completed").is_file()
        and not (root / "failed").exists()
    )
    return {
        "status": "passed" if passed else "failed",
        "exit_codes": exit_codes,
        "errors": errors,
        "completed_marker": (root / "completed").is_file(),
        "failed_marker": (root / "failed").is_file(),
    }


def _allocation_audit(campaign: Campaign, run_root: Path) -> dict[str, Any]:
    reference: dict[int, dict[str, Any]] | None = None
    runs = {}
    errors = []
    for phase in campaign.phases:
        for arm in phase.arms:
            key = f"{phase.name}/{arm}"
            paths = sorted(
                (run_root / "allocation" / phase.name / arm).glob("rank_*.json")
            )
            rows = {}
            for path in paths:
                try:
                    row = json.loads(path.read_text())
                    rows[int(row["rank"])] = row
                except Exception as error:
                    errors.append({"run": key, "path": str(path), "error": str(error)})
            if set(rows) != set(range(campaign.world_size)):
                errors.append(
                    {
                        "run": key,
                        "error": "allocation rank set mismatch",
                        "observed_ranks": sorted(rows),
                    }
                )
            if reference is None:
                reference = rows
            else:
                for rank, row in rows.items():
                    expected = reference.get(rank)
                    if expected is None:
                        continue
                    differences = {
                        field: {"reference": expected.get(field), "current": row.get(field)}
                        for field in ALLOCATION_IDENTITY_FIELDS
                        if expected.get(field) != row.get(field)
                    }
                    if differences:
                        errors.append({"run": key, "rank": rank, "differences": differences})
            runs[key] = {"files": [str(path) for path in paths], "ranks": sorted(rows)}
    return {
        "status": "passed" if not errors else "failed",
        "runs": runs,
        "errors": errors,
    }


def _input_hash_audit(campaign: Campaign, run_root: Path) -> dict[str, Any]:
    required = bool(campaign.raw.get("data", {}).get("runtime_input_hash_audit"))
    if not required:
        return {"status": "passed", "required": False, "pairs": [], "errors": []}
    pairs = []
    errors = []
    for phase in campaign.phases:
        for baseline, treatment in campaign.raw.get("comparison", {}).get("pairs", []):
            if baseline not in phase.arms or treatment not in phase.arms:
                continue
            by_arm = {}
            for arm in (baseline, treatment):
                paths = sorted((run_root / phase.name / arm / "input_audit").glob("rank_*.json"))
                rows = {}
                for path in paths:
                    try:
                        row = json.loads(path.read_text())
                        rows[int(row["rank"])] = {
                            "dp_rank": int(row["dp_rank"]),
                            "batch_count": int(row["batch_count"]),
                            "batch_sha256": list(row["batch_sha256"]),
                        }
                    except Exception as error:
                        errors.append(f"invalid input audit {path}: {error}")
                if set(rows) != set(range(campaign.world_size)):
                    errors.append(
                        f"{phase.name}/{arm}: input-audit rank set mismatch"
                    )
                by_arm[arm] = rows
            matched = by_arm[baseline] == by_arm[treatment]
            if not matched:
                errors.append(
                    f"{phase.name}: input hashes differ for {baseline}/{treatment}"
                )
            pairs.append(
                {
                    "phase": phase.name,
                    "baseline": baseline,
                    "treatment": treatment,
                    "matched": matched,
                    "ranks": sorted(set(by_arm[baseline]) & set(by_arm[treatment])),
                }
            )
    if not pairs:
        errors.append("runtime input-hash audit found no comparison pair")
    return {
        "status": "passed" if not errors else "failed",
        "required": True,
        "pairs": pairs,
        "errors": errors,
    }


def _placement_audit(campaign: Campaign, run_root: Path) -> dict[str, Any]:
    required = campaign.raw.get("artifacts", {}).get("placement_audit") == "replanning"
    if not required:
        return {"status": "passed", "required": False, "runs": {}, "errors": []}
    runs = {}
    errors = []
    for phase in campaign.phases:
        for arm in phase.arms:
            root = run_root / phase.name / arm / "placement_audit"
            paths = sorted(root.glob("rank_*.json"))
            rows = {}
            for path in paths:
                try:
                    row = json.loads(path.read_text())
                    rows[int(row["rank"])] = row
                except Exception as error:
                    errors.append(f"invalid placement audit {path}: {error}")
            key = f"{phase.name}/{arm}"
            if set(rows) != set(range(campaign.world_size)):
                errors.append(f"{key}: placement-audit rank set mismatch")
            modes = {row.get("mode") for row in rows.values()}
            solvers = {row.get("configured_solver") for row in rows.values()}
            invoked = {row.get("solver_invoked") for row in rows.values()}
            placement_digests = {row.get("placement_digest") for row in rows.values()}
            placement_file_hashes = {
                row.get("placement_file_sha256") for row in rows.values()
            }
            if solvers != {"approx"}:
                errors.append(f"{key}: configured solver is not uniformly Approx")
            expected_mode = "replay_2k" if arm == "replay_2k" else "fresh"
            if modes != {expected_mode}:
                errors.append(f"{key}: strategy mode contract mismatch")
            expected_invoked = arm != "replay_2k"
            if invoked != {expected_invoked}:
                errors.append(f"{key}: solver invocation contract mismatch")
            if len(placement_digests) != 1 or None in placement_digests:
                errors.append(f"{key}: ranks selected different placements")
            if len(placement_file_hashes) != 1 or None in placement_file_hashes:
                errors.append(f"{key}: placement file hashes differ across ranks")
            runs[key] = {
                "files": [str(path) for path in paths],
                "modes": sorted(str(value) for value in modes),
                "solver_invoked": sorted(str(value) for value in invoked),
                "placement_digests": sorted(str(value) for value in placement_digests),
                "placement_file_sha256": sorted(
                    str(value) for value in placement_file_hashes
                ),
                "rows": rows,
            }

    canonical = runs.get("canonical/canonical_fresh", {})
    canonical_digests = set(canonical.get("placement_digests", []))
    canonical_hashes = set(canonical.get("placement_file_sha256", []))
    for phase_name in ("performance", "trace"):
        replay = runs.get(f"{phase_name}/replay_2k", {})
        replay_rows = replay.get("rows", {})
        replay_canonical_digests = {
            row.get("canonical_placement_digest") for row in replay_rows.values()
        }
        replay_canonical_hashes = {
            row.get("canonical_file_sha256") for row in replay_rows.values()
        }
        if replay_canonical_digests != canonical_digests:
            errors.append(f"{phase_name}: replay placement differs from canonical")
        if replay_canonical_hashes != canonical_hashes:
            errors.append(f"{phase_name}: replay file hash differs from canonical")
    return {
        "status": "passed" if not errors else "failed",
        "required": True,
        "runs": runs,
        "errors": errors,
    }


def _canonical_parameter_name(name: str) -> str:
    for wrapper in ("._checkpoint_wrapped_module.", "._orig_mod."):
        while wrapper in name:
            name = name.replace(wrapper, ".")
    return name


_BFLOAT16_MOMENT_RTOL = 2**-7


def _parameter_moment_matches(reference: float, current: float) -> bool:
    return math.isclose(
        reference,
        current,
        rel_tol=_BFLOAT16_MOMENT_RTOL,
        abs_tol=0.0,
    )


def _parameter_state_audit(campaign: Campaign, run_root: Path) -> dict[str, Any]:
    """Require and compare initialized-parameter moments for paired runs."""
    runs: dict[str, Any] = {}
    errors: list[dict[str, Any]] = []
    reference_key: str | None = None
    reference: dict[tuple[int, str], dict[str, Any]] | None = None
    reference_stage: str | None = None
    saw_audit = False

    for phase in campaign.phases:
        for arm in phase.arms:
            key = f"{phase.name}/{arm}"
            root = run_root / phase.name / arm / "parameter_audit"
            paths = sorted(root.glob("rank_*.json"))
            if not paths:
                runs[key] = {"files": [], "parameters": {}}
                continue
            saw_audit = True
            if len(paths) != campaign.world_size:
                errors.append(
                    {
                        "run": key,
                        "error": "parameter audit rank count mismatch",
                        "expected": campaign.world_size,
                        "observed": len(paths),
                    }
                )

            values: dict[tuple[int, str], dict[str, Any]] = {}
            stages = set()
            for path in paths:
                try:
                    records = json.loads(path.read_text())
                    if not isinstance(records, list):
                        raise TypeError("parameter audit must be a JSON list")
                    for record in records:
                        stages.add(str(record.get("stage", "legacy_pre_load")))
                        parameter_key = (
                            int(record["part"]),
                            _canonical_parameter_name(str(record["name"])),
                        )
                        current = values.setdefault(
                            parameter_key,
                            {
                                "global_shape": list(record["global_shape"]),
                                "dtype": str(record["dtype"]),
                                "local_numel": 0,
                                "sums": [],
                                "square_sums": [],
                            },
                        )
                        if current["global_shape"] != list(record["global_shape"]):
                            raise ValueError(
                                f"inconsistent global shape for {parameter_key}"
                            )
                        if current["dtype"] != str(record["dtype"]):
                            raise ValueError(f"inconsistent dtype for {parameter_key}")
                        current["local_numel"] += math.prod(record["local_shape"])
                        current["sums"].append(float(record["sum"]))
                        current["square_sums"].append(float(record["square_sum"]))
                except Exception as error:
                    errors.append({"run": key, "path": str(path), "error": str(error)})

            if len(stages) != 1:
                errors.append(
                    {
                        "run": key,
                        "error": "parameter audit stage mismatch",
                        "stages": sorted(stages),
                    }
                )
            stage = next(iter(stages)) if len(stages) == 1 else None

            aggregated = {}
            for parameter_key, value in values.items():
                global_numel = math.prod(value["global_shape"])
                local_numel = value["local_numel"]
                if global_numel <= 0 or local_numel % global_numel:
                    errors.append(
                        {
                            "run": key,
                            "parameter": f"{parameter_key[0]}:{parameter_key[1]}",
                            "error": "parameter audit has a non-integral replication factor",
                            "global_numel": global_numel,
                            "local_numel": local_numel,
                        }
                    )
                    continue
                replication_factor = local_numel // global_numel
                if replication_factor < 1:
                    errors.append(
                        {
                            "run": key,
                            "parameter": f"{parameter_key[0]}:{parameter_key[1]}",
                            "error": "parameter audit has no complete global tensor",
                            "global_numel": global_numel,
                            "local_numel": local_numel,
                        }
                    )
                    continue
                aggregated[parameter_key] = {
                    "global_shape": value["global_shape"],
                    "dtype": value["dtype"],
                    "local_numel": local_numel,
                    "replication_factor": replication_factor,
                    "sum": math.fsum(value["sums"]) / replication_factor,
                    "square_sum": math.fsum(value["square_sums"]) / replication_factor,
                }
            runs[key] = {
                "files": [str(path) for path in paths],
                "stage": stage,
                "parameters": {
                    f"{part}:{name}": value
                    for (part, name), value in sorted(aggregated.items())
                },
            }
            if reference is None:
                reference_key = key
                reference = aggregated
                reference_stage = stage
                continue

            if stage != reference_stage:
                errors.append(
                    {
                        "run": key,
                        "reference_run": reference_key,
                        "error": "parameter audit stage differs",
                        "reference": reference_stage,
                        "current": stage,
                    }
                )

            if set(aggregated) != set(reference):
                errors.append(
                    {
                        "run": key,
                        "reference_run": reference_key,
                        "error": "parameter name set mismatch",
                        "missing": sorted(
                            f"{part}:{name}"
                            for part, name in set(reference) - set(aggregated)
                        ),
                        "extra": sorted(
                            f"{part}:{name}"
                            for part, name in set(aggregated) - set(reference)
                        ),
                    }
                )
            for parameter_key in sorted(set(reference) & set(aggregated)):
                expected = reference[parameter_key]
                current = aggregated[parameter_key]
                differences = {
                    field: {"reference": expected[field], "current": current[field]}
                    for field in ("global_shape", "dtype")
                    if expected[field] != current[field]
                }
                for field in ("sum", "square_sum"):
                    if not _parameter_moment_matches(
                        expected[field], current[field]
                    ):
                        differences[field] = {
                            "reference": expected[field],
                            "current": current[field],
                        }
                if differences:
                    errors.append(
                        {
                            "run": key,
                            "reference_run": reference_key,
                            "parameter": f"{parameter_key[0]}:{parameter_key[1]}",
                            "differences": differences,
                        }
                    )

    required = bool(campaign.raw.get("comparison", {}).get("pairs"))
    if required or saw_audit:
        for key, run in runs.items():
            if not run["files"]:
                errors.append({"run": key, "error": "missing parameter audit files"})
    return {
        "status": "passed" if not errors else "failed",
        "required": required,
        "reference_run": reference_key,
        "runs": runs,
        "errors": errors,
        "limitation": (
            "BF16 aggregate moments use one BF16 machine-epsilon relative tolerance; "
            "matching moments do not prove elementwise tensor identity"
        ),
    }


def _load_audit(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "failed", "error": f"missing {description}: {path}"}
    try:
        value = json.loads(path.read_text())
    except Exception as error:
        return {"status": "failed", "error": f"invalid {description}: {error}"}
    if value.get("status") != "passed":
        return {
            "status": "failed",
            "error": f"{description} did not pass",
            "report": value,
        }
    return {"status": "passed", "path": str(path), "report": value}


def _package_audit(attempt_root: Path) -> dict[str, Any]:
    loaded = _load_audit(attempt_root / "package_report.json", "package report")
    if loaded["status"] != "passed":
        return loaded
    payload = attempt_root / "package/payload"
    seal = attempt_root / "SEALED"
    if not payload.is_dir() or not seal.is_file():
        return {"status": "failed", "error": "package payload or seal is missing"}
    report = loaded["report"]
    expected = report.get("payload_tree_sha256")
    sealed = seal.read_text().strip()
    actual = manifest_digest(tree_manifest(payload))
    if not expected or expected != sealed or expected != actual:
        return {
            "status": "failed",
            "error": "packaged payload differs from its seal",
            "reported": expected,
            "sealed": sealed,
            "actual": actual,
        }
    return {
        "status": "passed",
        "path": loaded["path"],
        "payload_tree_sha256": actual,
    }


def _trace_audit(
    phase_kind: str, root: Path, trace_ranks: tuple[int, ...]
) -> dict[str, Any]:
    kineto = sorted(root.rglob("*.json.gz"))
    torch_trace = sorted(
        (root / "torch_trace").rglob("dedicated_log_torch_trace*.log")
    )
    errors = []
    if phase_kind in {"trace", "kineto"} and not kineto:
        errors.append("missing Kineto trace")
    if phase_kind in {"trace", "torch_trace"} and not torch_trace:
        errors.append("missing TORCH_TRACE log")
    for rank in trace_ranks:
        rank_root = root / "torch_trace" / f"rank_{rank:05d}"
        if phase_kind in {"trace", "torch_trace"} and not rank_root.is_dir():
            errors.append(f"missing TORCH_TRACE rank directory {rank}")
    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "kineto_files": [str(path) for path in kineto],
        "torch_trace_files": [str(path) for path in torch_trace],
    }


def _performance_phase_by_arm(campaign: Campaign) -> dict[str, str]:
    configured = campaign.raw.get("comparison", {}).get(
        "performance_phase_by_arm", {}
    )
    result = dict(configured)
    for arm in campaign.arms:
        if arm.name in result:
            continue
        matches = [
            phase.name
            for phase in campaign.phases
            if phase.kind == "performance" and arm.name in phase.arms
        ]
        if len(matches) == 1:
            result[arm.name] = matches[0]
    return result


def _primary_for_structured(
    root: Path,
    *,
    config: dict[str, Any],
    world_size: int,
) -> dict[str, Any]:
    steps = list(config["steps"])
    records = _structured_records(root, steps, world_size)
    rows = []
    component_samples: dict[str, list[float]] = defaultdict(list)
    for step in steps:
        if config["rank_selector"] == "max_step_end":
            step_rows = records["step_end"][step]
            selected_rank = max(
                step_rows, key=lambda rank: float(step_rows[rank]["duration_ms"])
            )
            latency_ms = float(step_rows[selected_rank]["duration_ms"])
        else:
            data_rows = records["fetching_batch_end"].get(step, {})
            if set(data_rows) != set(range(world_size)):
                raise RuntimeError(
                    f"{root}: incomplete fetching_batch_end records at step {step}"
                )
            selected_rank = max(
                data_rows, key=lambda rank: int(data_rows[rank]["time_us"])
            )
            latency_ms = (
                int(records["step_end"][step][selected_rank]["time_us"])
                - int(data_rows[selected_rank]["time_us"])
            ) / 1000.0
            if latency_ms <= 0:
                raise RuntimeError(f"{root}: nonpositive post-data latency at step {step}")
        components = {}
        for event in TIMER_EVENTS:
            value = float(
                records[event].get(step, {}).get(selected_rank, {}).get(
                    "duration_ms", 0.0
                )
            )
            components[event] = value
            component_samples[event].append(value)
        rows.append(
            {
                "step": step,
                "selected_rank": selected_rank,
                "latency_ms": latency_ms,
                "components_ms": components,
            }
        )
    return {
        "status": "passed",
        "source": "structured_logs",
        "rank_selector": config["rank_selector"],
        "value": config["value"],
        "steps": steps,
        "latency_ms": _summary([row["latency_ms"] for row in rows]),
        "components_ms": {
            event: _summary(samples) for event, samples in component_samples.items()
        },
        "rows": rows,
    }


def _selected_scalars(
    scalars: dict[str, dict[int, float]], tag: str, steps: list[int]
) -> list[float]:
    values = [scalars.get(tag, {}).get(step) for step in steps]
    if any(value is None for value in values):
        missing = [step for step, value in zip(steps, values) if value is None]
        raise RuntimeError(f"missing TensorBoard tag {tag!r} at steps {missing}")
    return [float(value) for value in values]


def _primary_for_tensorboard(
    root: Path,
    *,
    config: dict[str, Any],
    world_size: int,
) -> dict[str, Any]:
    paths = sorted((root / "job/tb").glob("**/rank_0/events.out.tfevents.*"))
    if len(paths) != 1:
        raise RuntimeError(f"{root}: expected one rank-0 TensorBoard event file")
    steps = list(config["steps"])
    scalars = _tensorboard_scalars(paths[0])
    scale = float(config["unit_scale_to_ms"])
    latency = [value * scale for value in _selected_scalars(scalars, config["value"], steps)]
    result = {
        "status": "passed",
        "source": "tensorboard",
        "rank_selector": "rank0",
        "value": config["value"],
        "steps": steps,
        "event_file": str(paths[0]),
        "available_tags": sorted(scalars),
        "latency_ms": _summary(latency),
    }
    throughput_tag = config.get("throughput_tag")
    if throughput_tag:
        per_device = _selected_scalars(scalars, throughput_tag, steps)
        result["throughput_tokens_per_second"] = {
            "per_device": _summary(per_device),
            "aggregate": _summary([value * world_size for value in per_device]),
        }
    for field, output_name in (
        ("active_memory_tag", "active_memory_gib"),
        ("reserved_memory_tag", "reserved_memory_gib"),
    ):
        tag = config.get(field)
        if tag:
            samples = _selected_scalars(scalars, tag, steps)
            result[output_name] = {"peak": max(samples), "summary": _summary(samples)}
    return result


def _primary_measurement(campaign: Campaign, run_root: Path) -> dict[str, Any] | None:
    measurement = campaign.raw.get("measurement", {}).get("primary")
    if not measurement:
        return None
    phase_by_arm = _performance_phase_by_arm(campaign)
    arms = {}
    errors = []
    for arm in campaign.arms:
        phase_name = phase_by_arm.get(arm.name)
        if phase_name is None:
            continue
        root = run_root / phase_name / arm.name
        try:
            if measurement["source"] == "structured_logs":
                result = _primary_for_structured(
                    root, config=measurement, world_size=campaign.world_size
                )
            else:
                result = _primary_for_tensorboard(
                    root, config=measurement, world_size=campaign.world_size
                )
            result["phase"] = phase_name
            arms[arm.name] = result
        except Exception as error:
            errors.append(f"{arm.name}: {type(error).__name__}: {error}")

    acceptance_config = campaign.raw.get("comparison", {}).get("acceptance")
    acceptance = None
    if acceptance_config is not None and not errors:
        baseline = acceptance_config["baseline"]
        treatment = acceptance_config["treatment"]
        baseline_mean = arms[baseline]["latency_ms"]["mean"]
        treatment_mean = arms[treatment]["latency_ms"]["mean"]
        observed = (treatment_mean / baseline_mean - 1.0) * 100.0
        reference = float(acceptance_config["reference_gap_percent"])
        tolerance = float(acceptance_config["tolerance_percentage_points"])
        acceptance = {
            "status": "passed" if abs(observed - reference) <= tolerance else "failed",
            "metric": "primary_latency",
            "baseline": baseline,
            "treatment": treatment,
            "observed_gap_percent": observed,
            "reference_gap_percent": reference,
            "tolerance_percentage_points": tolerance,
            "accepted_range_percent": [reference - tolerance, reference + tolerance],
        }
    status = not errors and (acceptance is None or acceptance["status"] == "passed")
    return {
        "status": "passed" if status else "failed",
        "method": dict(measurement),
        "arms": arms,
        "acceptance": acceptance,
        "errors": errors,
    }


def _pair_summaries(campaign: Campaign, phases: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    pairs = campaign.raw.get("comparison", {}).get("pairs", [])
    training = campaign.raw.get("training", {})
    global_batch = int(training.get("global_batch_size", -1))
    if global_batch < 0:
        parallelism = campaign.raw.get("parallelism", {})
        dp_degree = int(parallelism.get("data_parallel_replicate_degree", 1)) * int(
            parallelism.get("data_parallel_shard_degree", 1)
        )
        global_batch = (
            int(training["local_batch_size"])
            * dp_degree
            * int(training.get("gradient_accumulation_steps", 1))
        )
    tokens_per_step = global_batch * int(training["seq_len"])
    normal_steps = set(
        campaign.raw.get("measurement", {}).get("primary", {}).get("steps", [])
    )
    phase_by_arm = _performance_phase_by_arm(campaign)
    for baseline, treatment in pairs:
        if baseline not in phase_by_arm or treatment not in phase_by_arm:
            continue
        left_phase = phase_by_arm[baseline]
        right_phase = phase_by_arm[treatment]
        left_arm = phases[left_phase]["arms"][baseline]
        right_arm = phases[right_phase]["arms"][treatment]
        phase_label = left_phase if left_phase == right_phase else f"{left_phase} -> {right_phase}"
        components = {}
        errors = []
        for name in TIMER_EVENTS:
            left_component = left_arm["structured_metrics"]["components_ms"][name]
            right_component = right_arm["structured_metrics"]["components_ms"][name]
            left = left_component["per_step_rank_max"]
            right = right_component["per_step_rank_max"]
            left_steps = left_component["steps"]
            right_steps = right_component["steps"]
            if normal_steps and left is not None and right is not None:
                left_values = dict(zip(left_steps, left["samples"] if left else []))
                right_values = dict(zip(right_steps, right["samples"] if right else []))
                if name == "step_end":
                    missing_left = sorted(normal_steps - left_values.keys())
                    missing_right = sorted(normal_steps - right_values.keys())
                    if missing_left:
                        errors.append(
                            f"{name} baseline missing normal steps {missing_left}"
                        )
                    if missing_right:
                        errors.append(
                            f"{name} treatment missing normal steps {missing_right}"
                        )
                selected = sorted(
                    normal_steps & left_values.keys() & right_values.keys()
                )
                left = _summary([left_values[step] for step in selected])
                right = _summary([right_values[step] for step in selected])
                left_steps = selected
                right_steps = selected
            if left is None or right is None:
                components[name] = None
                continue
            if left["count"] != right["count"]:
                errors.append(
                    f"{name} sample count differs: {left['count']} != {right['count']}"
                )
            if left_steps != right_steps:
                errors.append(f"{name} measured step identities differ")
            components[name] = {
                "baseline_ms": left,
                "treatment_ms": right,
                "mean_delta_ms": right["mean"] - left["mean"],
                "mean_speedup_baseline_over_treatment": (
                    left["mean"] / right["mean"] if right["mean"] else None
                ),
            }
        rows.append(
            {
                "phase": phase_label,
                "baseline": baseline,
                "treatment": treatment,
                "measurement_method": (
                    "paired configuration; full-step latency and throughput use the "
                    "declared primary normal steps and per-step rank-max timing"
                ),
                "status": "passed" if not errors else "failed",
                "errors": errors,
                "components_ms": components,
                "throughput_tokens_per_second": {
                    name: _summary(
                        [
                            tokens_per_step * 1000.0 / value
                            for value in components["step_end"][
                                "baseline_ms" if name == baseline else "treatment_ms"
                            ]["samples"]
                        ]
                    )
                    for name, arm in ((baseline, left_arm), (treatment, right_arm))
                    if components.get("step_end") is not None
                },
            }
        )
    return rows


def _legacy_inventory(campaign: Campaign, attempt_root: Path) -> dict[str, Any]:
    legacy = campaign.raw.get("legacy", {})
    globs = legacy.get("evidence_globs", []) if isinstance(legacy, dict) else []
    matches = []
    for pattern in globs:
        for path in sorted(attempt_root.glob(pattern)):
            if path.is_file():
                matches.append(
                    {
                        "path": str(path.resolve()),
                        "size": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                )
    return {
        "status": "found" if matches else "not_found",
        "patterns": globs,
        "files": matches,
    }


def _run_tlparse(
    trace_root: Path, output_root: Path, binary: Path | None
) -> dict[str, Any]:
    traces = sorted(trace_root.rglob("dedicated_log_torch_trace*.log"))
    if not traces:
        return {"status": "not_available", "traces": []}
    if binary is None:
        return {"status": "available", "traces": [str(path) for path in traces]}
    if not binary.is_file() or not binary.stat().st_mode & 0o111:
        return {
            "status": "failed",
            "error": "the supplied tlparse path is not executable",
            "traces": [str(path) for path in traces],
        }
    output_root.mkdir(parents=True, exist_ok=True)
    outputs = []
    for index, trace in enumerate(traces):
        destination = output_root / f"trace_{index:03d}"
        command = [
            str(binary),
            "parse",
            "--no-browser",
            "--overwrite",
            "-o",
            str(destination),
            str(trace),
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        log_root = output_root / "logs"
        log_root.mkdir(parents=True, exist_ok=True)
        (log_root / f"trace_{index:03d}.stdout").write_text(completed.stdout)
        (log_root / f"trace_{index:03d}.stderr").write_text(completed.stderr)
        (log_root / f"trace_{index:03d}.returncode").write_text(
            f"{completed.returncode}\n"
        )
        outputs.append(
            {
                "trace": str(trace),
                "trace_sha256": _sha256(trace),
                "command": command,
                "returncode": completed.returncode,
                "output": str(destination),
                "file_count": (
                    sum(path.is_file() for path in destination.rglob("*"))
                    if destination.is_dir()
                    else 0
                ),
            }
        )
    version = subprocess.run(
        [str(binary), "--version"], text=True, capture_output=True, check=False
    )
    return {
        "status": "passed"
        if all(row["returncode"] == 0 and row["file_count"] for row in outputs)
        else "failed",
        "binary": str(binary),
        "version": (version.stdout or version.stderr).strip(),
        "runs": outputs,
    }


PLANNER_TIMING_FIELDS = (
    "model_and_mesh_setup_s",
    "graph_trace_s",
    "optimizer_init_s",
    "user_constraints_s",
    "factor_build_s",
    "solver_core_s",
    "search_total_s",
)


def _planner_result(
    run_root: Path,
    expected: dict[str, Any],
    *,
    source_lock: dict[str, Any],
    source_lock_sha256: str,
) -> tuple[dict[str, Any], list[str]]:
    root = (
        run_root
        / "planner"
        / expected["point"]
        / f"repeat_{int(expected['repeat']):02d}"
        / expected["solver"]
    )
    errors = []
    for marker in (
        "started",
        "completed",
        "exit_code",
        "command.json",
        "result.json",
        "stdout_stderr.log",
    ):
        if not (root / marker).is_file():
            errors.append(f"missing {marker}: {root}")
    if (root / "failed").exists():
        errors.append(f"failed marker exists: {root}")
    if errors:
        return {"root": str(root)}, errors
    try:
        exit_code = int((root / "exit_code").read_text().strip())
        command = json.loads((root / "command.json").read_text())
        result = json.loads((root / "result.json").read_text())
    except Exception as error:
        return {"root": str(root)}, [f"invalid planner evidence at {root}: {error}"]
    if exit_code != 0:
        errors.append(f"nonzero exit code {exit_code}: {root}")
    if result.get("status") != "success":
        errors.append(f"planner result is not successful: {root}")
    request = result.get("request", {})
    expected_request = {
        "model": "llama8b",
        "mesh": ",".join(str(value) for value in expected["mesh"]),
        "solver": expected["solver"],
        "lazy_costs": (
            str(expected["lazy_costs"]).lower()
            if expected["lazy_costs"] is not None
            else None
        ),
        "seeded": expected["seeded"],
        "revision_label": source_lock["autoparallel"]["head"],
    }
    for name, value in expected_request.items():
        if request.get(name) != value:
            errors.append(
                f"planner request {name} mismatch at {root}: "
                f"{request.get(name)!r} != {value!r}"
            )
    if not isinstance(command, list) or not all(
        isinstance(token, str) for token in command
    ):
        errors.append(f"planner command is not a string array: {root}")
    elif "--source-lock" not in command or "--detailed-solution" not in command:
        errors.append(
            f"planner command lacks locked detailed profiling options: {root}"
        )
    if not request.get("source_lock"):
        errors.append(f"planner request has no source lock: {root}")
    lock_evidence = result.get("source_lock", {})
    if lock_evidence.get("sha256") != source_lock_sha256:
        errors.append(f"source-lock digest mismatch: {root}")
    if lock_evidence.get("autoparallel", {}).get("head") != source_lock.get(
        "autoparallel", {}
    ).get("head"):
        errors.append(f"source-lock AutoParallel head mismatch: {root}")
    if result.get("expanded_config", {}).get("mesh_shape") != expected["mesh"]:
        errors.append(f"expanded planner mesh mismatch: {root}")
    objective = result.get("objective")
    if not isinstance(objective, (int, float)) or not math.isfinite(objective):
        errors.append(f"planner objective is not finite: {root}")
    timings = result.get("timings", {})
    for name in PLANNER_TIMING_FIELDS:
        value = timings.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            errors.append(f"planner timing {name} is invalid: {root}")
    row = {
        **expected,
        "status": "passed" if not errors else "failed",
        "root": str(root),
        "command": command,
        "result": str(root / "result.json"),
        "result_sha256": _sha256(root / "result.json"),
        "log": str(root / "stdout_stderr.log"),
        "objective": objective,
        "placement_sha256": result.get("placement_sha256"),
        "timings_s": {name: timings.get(name) for name in PLANNER_TIMING_FIELDS},
        "environment": result.get("environment"),
        "source_lock": lock_evidence,
        "errors": errors,
    }
    return row, errors


def _planner_rank_audit(campaign: Campaign, run_root: Path) -> dict[str, Any]:
    allocation_paths = sorted(
        (run_root / "allocation/planner/planner_profile").glob("rank_*.json")
    )
    exit_paths = sorted((run_root / "planner/runtime").glob("rank_*.exit_code"))
    errors = []
    allocation_ranks = set()
    hosts = set()
    for path in allocation_paths:
        try:
            row = json.loads(path.read_text())
            allocation_ranks.add(int(row["rank"]))
            hosts.add(row["hostname"])
            if (
                int(row["world_size"]) != campaign.world_size
                or int(row["local_world_size"])
                != int(campaign.raw["mast"]["nproc_per_node"])
            ):
                errors.append(f"allocation size mismatch: {path}")
        except Exception as error:
            errors.append(f"invalid allocation record {path}: {error}")
    expected_ranks = set(range(campaign.world_size))
    if allocation_ranks != expected_ranks:
        errors.append(
            "planner allocation rank set mismatch: "
            f"{sorted(allocation_ranks)} != {sorted(expected_ranks)}"
        )
    if len(hosts) != int(campaign.raw["mast"]["nodes"]):
        errors.append(f"planner allocation host count mismatch: {sorted(hosts)}")
    exit_codes = {}
    for path in exit_paths:
        try:
            rank = int(path.stem.removeprefix("rank_"))
            exit_codes[rank] = int(path.read_text().strip())
        except ValueError as error:
            errors.append(f"invalid planner rank exit record {path}: {error}")
    if set(exit_codes) != expected_ranks or any(exit_codes.values()):
        errors.append(f"planner rank exits are incomplete or nonzero: {exit_codes}")
    if not (run_root / "planner/all_ranks_completed").is_file():
        errors.append("planner all-rank completion marker is missing")
    return {
        "status": "passed" if not errors else "failed",
        "allocation_files": [str(path) for path in allocation_paths],
        "allocation_ranks": sorted(allocation_ranks),
        "hosts": sorted(hosts),
        "rank_exit_codes": exit_codes,
        "errors": errors,
    }


def _analyze_planner_campaign(
    campaign: Campaign,
    *,
    attempt_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    attempt_root = attempt_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_root = attempt_root / "run"
    source_lock_path = attempt_root / "package/payload/campaign/source_lock.json"
    source_lock = json.loads(source_lock_path.read_text())
    source_lock_sha256 = _sha256(source_lock_path)
    rows = []
    errors = []
    for expected in campaign.planner_runs():
        row, run_errors = _planner_result(
            run_root,
            expected,
            source_lock=source_lock,
            source_lock_sha256=source_lock_sha256,
        )
        rows.append(row)
        errors.extend(run_errors)

    points = []
    for point in campaign.raw["planner"]["points"]:
        point_rows = [row for row in rows if row.get("point") == point["name"]]
        candidates = [row for row in point_rows if row.get("role") == "candidate"]
        references = [row for row in point_rows if row.get("role") == "reference"]
        gaps = []
        for repeat in range(1, int(campaign.raw["planner"]["repeats"]) + 1):
            candidate = next(
                (row for row in candidates if row.get("repeat") == repeat), None
            )
            reference = next(
                (row for row in references if row.get("repeat") == repeat), None
            )
            if candidate is None or reference is None:
                errors.append(f"{point['name']}: missing repeat {repeat} result pair")
                continue
            candidate_objective = candidate.get("objective")
            reference_objective = reference.get("objective")
            if not all(
                isinstance(value, (int, float)) and math.isfinite(value) and value > 0
                for value in (candidate_objective, reference_objective)
            ):
                errors.append(
                    f"{point['name']}: invalid objective pair at repeat {repeat}"
                )
                continue
            tolerance = max(abs(reference_objective) * 1e-8, 1e-6)
            if candidate_objective + tolerance < reference_objective:
                errors.append(
                    f"{point['name']}: Approx objective is below the LP lower bound "
                    f"at repeat {repeat}"
                )
            gaps.append(
                {
                    "repeat": repeat,
                    "approx_objective": candidate_objective,
                    "lp_objective": reference_objective,
                    "gap_percent": (candidate_objective - reference_objective)
                    / reference_objective
                    * 100.0,
                }
            )

        def timing_summary(selected: list[dict[str, Any]]) -> dict[str, Any]:
            return {
                name: _summary(
                    [
                        float(row["timings_s"][name])
                        for row in selected
                        if "timings_s" in row
                    ]
                )
                for name in PLANNER_TIMING_FIELDS
            }

        points.append(
            {
                "point": point["name"],
                "mesh": point["mesh"],
                "approx": {
                    "timings_s": timing_summary(candidates),
                    "objectives": _summary(
                        [
                            float(row["objective"])
                            for row in candidates
                            if "objective" in row
                        ]
                    ),
                    "placement_sha256": sorted(
                        {
                            row["placement_sha256"]
                            for row in candidates
                            if row.get("placement_sha256")
                        }
                    ),
                },
                "lp_reference": {
                    "timings_s": timing_summary(references),
                    "objectives": _summary(
                        [
                            float(row["objective"])
                            for row in references
                            if "objective" in row
                        ]
                    ),
                },
                "objective_gap_percent": _summary(
                    [float(row["gap_percent"]) for row in gaps]
                ),
                "paired_repeats": gaps,
            }
        )

    audits = {
        "package": _package_audit(attempt_root),
        "runtime_preflight": _load_audit(
            run_root / "runtime/preflight/report.json", "runtime preflight"
        ),
        "runtime_planner_contract": _load_audit(
            run_root / "runtime/configs/report.json", "runtime planner contract"
        ),
        "allocation_and_rank_completion": _planner_rank_audit(campaign, run_root),
    }
    if not (run_root / "planner/completed").is_file():
        errors.append("planner campaign completion marker is missing")
    valid = (
        bool(rows)
        and not errors
        and all(row.get("status") == "passed" for row in rows)
        and all(audit["status"] == "passed" for audit in audits.values())
    )
    report = {
        "status": "passed" if valid else "incomplete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": campaign.name,
        "campaign_type": "planner",
        "classification": PLANNER_RESULT_CLASSIFICATION,
        "training_performance_evidence": False,
        "performance_conclusion_allowed": False,
        "measurement_method": (
            "three independent Approx lazy seeded searches and three independent "
            "LP relaxations per mesh; solver time is search_total_s; gap is "
            "(Approx objective - LP objective) / LP objective * 100%"
        ),
        "source_lock": str(source_lock_path),
        "source_lock_sha256": source_lock_sha256,
        "points": points,
        "runs": rows,
        "audits": audits,
        "errors": errors,
    }
    write_json(output_dir / "analysis.json", report)
    lines = [
        f"# {campaign.name}",
        "",
        f"Status: `{report['status']}`",
        "",
        f"Classification: {PLANNER_RESULT_CLASSIFICATION}.",
        "",
        report["measurement_method"],
        "",
        "| Mesh | Approx search mean (s) | LP search mean (s) | Gap mean (%) |",
        "|---|---:|---:|---:|",
    ]
    for point in points:
        approx = point["approx"]["timings_s"]["search_total_s"]
        reference = point["lp_reference"]["timings_s"]["search_total_s"]
        gap = point["objective_gap_percent"]
        lines.append(
            f"| {point['point']} {point['mesh']} | "
            f"{approx['mean'] if approx else 'n/a'} | "
            f"{reference['mean'] if reference else 'n/a'} | "
            f"{gap['mean'] if gap else 'n/a'} |"
        )
    lines.extend(
        ["", f"Full machine-readable result: `{output_dir / 'analysis.json'}`"]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")
    return report


def analyze_campaign(
    campaign: Campaign,
    *,
    attempt_root: Path,
    output_dir: Path,
    tlparse_bin: Path | None = None,
) -> dict[str, Any]:
    if campaign.is_planner:
        return _analyze_planner_campaign(
            campaign,
            attempt_root=attempt_root,
            output_dir=output_dir,
        )
    attempt_root = attempt_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_root = attempt_root / "run"
    warmup = int(campaign.raw.get("artifacts", {}).get("warmup_steps", 0))
    phases = {}
    for phase in campaign.phases:
        arms = {}
        for arm in phase.arms:
            root = run_root / phase.name / arm
            exit_audit = _phase_exit_audit(root, campaign.world_size)
            metrics = (
                _structured_metrics(root, warmup, campaign.world_size)
                if phase.kind == "performance"
                else None
            )
            memory = (
                _all_rank_memory(root, warmup, campaign.world_size)
                if phase.kind == "performance"
                else None
            )
            trace = (
                _trace_audit(phase.kind, root, phase.trace_ranks)
                if phase.kind in {"trace", "kineto", "torch_trace"}
                else None
            )
            arm_passed = (
                exit_audit["status"] == "passed"
                and (metrics is None or metrics["status"] == "passed")
                and (memory is None or memory["status"] == "passed")
                and (trace is None or trace["status"] == "passed")
            )
            arms[arm] = {
                "status": "passed" if arm_passed else "failed",
                "exit_audit": exit_audit,
                "structured_metrics": metrics,
                "all_rank_memory": memory,
                "trace_audit": trace,
            }
        phases[phase.name] = {"kind": phase.kind, "arms": arms}

    audits = {
        "package": _package_audit(attempt_root),
        "runtime_preflight": _load_audit(
            run_root / "runtime/preflight/report.json", "runtime preflight"
        ),
        "runtime_configs": _load_audit(
            run_root / "runtime/configs/report.json", "runtime config parity report"
        ),
        "allocation": _allocation_audit(campaign, run_root),
        "runtime_input_hashes": _input_hash_audit(campaign, run_root),
        "placements": _placement_audit(campaign, run_root),
        "parameter_state_moments": _parameter_state_audit(campaign, run_root),
    }
    if campaign.raw.get("data", {}).get("preflight_auditor"):
        audits["input_preflight"] = _load_audit(
            run_root / "runtime/input_preflight/report.json", "input preflight"
        )
    pairs = _pair_summaries(campaign, phases)
    primary = _primary_measurement(campaign, run_root)
    tlparse = _run_tlparse(run_root, output_dir / "tlparse", tlparse_bin)
    all_runs_passed = bool(phases) and all(
        arm["status"] == "passed"
        for phase in phases.values()
        for arm in phase["arms"].values()
    )
    all_audits_passed = all(audit["status"] == "passed" for audit in audits.values())
    is_gate = campaign.raw.get("execution_mode") == "gate"
    all_pairs_passed = bool(pairs) and all(pair["status"] == "passed" for pair in pairs)
    valid = all_runs_passed and all_audits_passed and tlparse["status"] != "failed"
    if not is_gate:
        valid = valid and all_pairs_passed
        if primary is not None:
            valid = valid and primary["status"] == "passed"
    report = {
        "status": "passed" if valid else "incomplete",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": campaign.name,
        "attempt_root": str(attempt_root),
        "measurement_method": (
            f"discard steps 1..{warmup}; summarize each component as the per-step "
            "maximum across all ranks; compare paired arms from one allocation"
        ),
        "phases": phases,
        "primary_measurement": primary,
        "pair_summaries": pairs,
        "audits": audits,
        "legacy_evidence": _legacy_inventory(campaign, attempt_root),
        "tlparse": tlparse,
        "functional_gate_passed": valid if is_gate else None,
        "performance_conclusion_allowed": valid and not is_gate,
    }
    write_json(output_dir / "analysis.json", report)
    lines = [
        f"# {campaign.name}",
        "",
        f"Status: `{report['status']}`",
        "",
        report["measurement_method"],
        "",
        "A comparative claim is allowed only when package, runtime source/config, "
        "parameter-state, same-allocation, per-rank completion, and metric gates "
        "all pass.",
        "",
        f"Full machine-readable result: `{output_dir / 'analysis.json'}`",
    ]
    if primary is not None:
        lines.extend(["", "## Primary measurement", ""])
        for arm, result in primary["arms"].items():
            latency = result["latency_ms"]
            lines.append(
                f"- {arm}: {latency['mean']:.6f} ms ± {latency['sample_sd']:.6f} "
                f"({latency['count']} samples)"
            )
        acceptance = primary.get("acceptance")
        if acceptance is not None:
            lines.append(
                f"- {acceptance['treatment']}/{acceptance['baseline']} gap: "
                f"{acceptance['observed_gap_percent']:+.6f}% "
                f"(`{acceptance['status']}`; expected "
                f"{acceptance['accepted_range_percent'][0]:+.6f}% to "
                f"{acceptance['accepted_range_percent'][1]:+.6f}%)"
            )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")
    return report
