# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from autoparallel.cost_models.nccl_cost_model import (
    NCCLTopoConfig,
    derive_mesh_dim_topo,
    nccl_allgather_cost,
    nccl_allreduce_cost,
    nccl_reduce_scatter_cost,
)

_DTYPE_BYTES = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "int16": 2,
    "float16": 2,
    "bfloat16": 2,
    "int32": 4,
    "float32": 4,
    "int64": 8,
    "float64": 8,
}
_PLACEMENT_RE = re.compile(r"S\((\d+)\)|R|P(?:\([^)]*\))?")


def pack_buckets(sizes: Iterable[int], capacity_bytes: int) -> list[int]:
    if capacity_bytes <= 0:
        raise ValueError("capacity_bytes must be positive")

    buckets = []
    current = 0
    for size in sizes:
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("bucket inputs must be non-negative integer byte sizes")
        if size == 0:
            continue
        if current and current + size > capacity_bytes:
            buckets.append(current)
            current = 0
        current += size
        if current >= capacity_bytes:
            buckets.append(current)
            current = 0
    if current:
        buckets.append(current)
    return buckets


def optimizer_phase_costs(plan: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    phases: dict[str, dict[str, float]] = {}
    for node in plan["nodes"]:
        phase = node.get("phase")
        if phase not in {"forward", "backward"}:
            continue
        costs = phases.setdefault(phase, {"compute_us": 0.0, "communication_us": 0.0})
        costs["compute_us"] += float(node.get("compute_cost", 0.0))
        costs["communication_us"] += float(node.get("transition_cost", 0.0))
        costs["communication_us"] += sum(
            float(edge.get("comm_cost", 0.0)) for edge in node.get("inputs", ())
        )
    for phase in ("forward", "backward"):
        phases.setdefault(phase, {"compute_us": 0.0, "communication_us": 0.0})
    return phases


def activation_liveness_estimate(
    plan: Mapping[str, Any], mesh_shape: tuple[int, ...]
) -> dict[str, Any]:
    nodes = list(plan.get("nodes", ()))
    remaining_uses: dict[str, int] = {}
    for node in nodes:
        for edge in node.get("inputs", ()):
            name = edge.get("name")
            if name is not None:
                remaining_uses[str(name)] = remaining_uses.get(str(name), 0) + 1

    live: dict[str, int] = {}
    current = 0
    peak = 0
    for node in nodes:
        name = str(node.get("name"))
        if node.get("op") not in {"placeholder", "output"}:
            size = _local_tensor_bytes(node, mesh_shape)
            if size:
                live[name] = size
                current += size
                peak = max(peak, current)
        for edge in node.get("inputs", ()):
            source = str(edge.get("name"))
            if source not in remaining_uses:
                continue
            remaining_uses[source] -= 1
            if remaining_uses[source] == 0:
                current -= live.pop(source, 0)
    return {
        "scope": "joint-graph-local-tensor-liveness-upper-bound",
        "peak_bytes": peak,
        "aliasing_modeled": False,
        "allocator_fragmentation_modeled": False,
    }


def optimizer_update_estimate(
    local_parameter_bytes: int,
    calibration: Mapping[str, Any] | None,
    applicability: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if calibration is None:
        return {"status": "unavailable", "reason": "no_calibration"}
    local_parameter_gib = local_parameter_bytes / 2**30
    population = calibration["calibration"]
    if not (
        population["local_parameter_gib_min"]
        <= local_parameter_gib
        <= population["local_parameter_gib_max"]
    ):
        return {
            "status": "unavailable",
            "reason": "scale_out_of_range",
            "local_parameter_gib": local_parameter_gib,
            "calibrated_range_gib": [
                population["local_parameter_gib_min"],
                population["local_parameter_gib_max"],
            ],
        }
    matches = dict(applicability or {})
    unknown = sorted(key for key, value in matches.items() if value == "unknown")
    mismatched = sorted(key for key, value in matches.items() if value == "mismatched")
    if mismatched:
        return {
            "status": "unavailable",
            "reason": "calibration_regime_mismatch",
            "mismatched": mismatched,
            "applicability": matches,
        }
    expected = max(
        0.0,
        float(calibration["intercept_us"])
        + float(calibration["slope_us_per_local_parameter_gib"]) * local_parameter_gib,
    )
    error = float(population["maximum_prospective_kernel_relative_error"])
    return {
        "status": "screening_only" if unknown else "applicable",
        "latency": {
            "unit": "us",
            "optimistic": expected * (1.0 - error),
            "expected": expected,
            "pessimistic": expected * (1.0 + error),
            "calibration_relative_error": error,
            "schedule_model": calibration["kind"],
        },
        "local_parameter_gib": local_parameter_gib,
        "applicability": matches,
        "unknown": unknown,
        "source_run_ids": list(calibration.get("source_run_ids", ())),
    }


def load_optimizer_cost_calibration(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as calibration_file:
        calibration = json.load(calibration_file)
    if calibration.get("schema_version") != 1:
        raise ValueError("unsupported optimizer cost calibration schema")
    return calibration


def calibrate_relative_baseline(
    autoparallel_latency: Mapping[str, Any],
    baseline: Mapping[str, Any],
    parameter_bytes: int,
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    strategy = str(baseline["strategy"])
    strategy_calibration = calibration["strategies"][strategy]
    raw_ratio = float(autoparallel_latency["expected"]) / float(
        baseline["latency"]["expected"]
    )
    method = strategy_calibration["method"]
    if method == "median-log-offset":
        calibrated_ratio = math.exp(
            math.log(raw_ratio) + float(strategy_calibration["offset"])
        )
    elif method == "log-structural":
        communication = baseline["communication"]
        unit_count = communication.get("wrap_unit_count")
        if not isinstance(unit_count, int) or unit_count <= 0:
            raise ValueError("structural calibration requires a positive unit count")
        features = (
            1.0,
            math.log(raw_ratio),
            math.log(parameter_bytes),
            math.log(unit_count),
        )
        calibrated_ratio = math.exp(
            sum(
                float(coefficient) * feature
                for coefficient, feature in zip(
                    strategy_calibration["coefficients"], features
                )
            )
        )
    else:
        raise ValueError(f"unknown relative baseline calibration method: {method}")
    scale = raw_ratio / calibrated_ratio
    latency: dict[str, Any] = {
        key: float(baseline["latency"][key]) * scale
        for key in ("optimistic", "expected", "pessimistic")
    }
    latency.update(
        {
            "unit": baseline["latency"]["unit"],
            "calibration_relative_error": (
                float(
                    strategy_calibration["family_holdout_maximum_multiplicative_error"]
                )
                - 1.0
            ),
            "schedule_model": f"{method}-relative-calibration-v2",
        }
    )
    return {
        **baseline,
        "latency": latency,
        "relative_calibration": {
            "method": method,
            "raw_ratio": raw_ratio,
            "calibrated_ratio": calibrated_ratio,
            "family_holdout_winner_accuracy": strategy_calibration[
                "family_holdout_winner_accuracy"
            ],
            "maximum_multiplicative_error": strategy_calibration[
                "family_holdout_maximum_multiplicative_error"
            ],
            "use": calibration["use"],
        },
    }


def load_baseline_cost_calibration(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as calibration_file:
        calibration = json.load(calibration_file)
    if calibration.get("schema_version") != 2:
        raise ValueError("unsupported baseline cost calibration schema")
    return calibration


def overlap_envelope(
    phases: Mapping[str, Mapping[str, float]],
    *,
    expected_overlap_fraction: float = 0.5,
    schedule_model: str,
) -> dict[str, Any]:
    if not 0.0 <= expected_overlap_fraction <= 1.0:
        raise ValueError("expected_overlap_fraction must be between zero and one")

    optimistic = 0.0
    expected = 0.0
    pessimistic = 0.0
    for phase in ("forward", "backward"):
        compute = float(phases[phase]["compute_us"])
        communication = float(phases[phase]["communication_us"])
        if compute < 0 or communication < 0:
            raise ValueError("phase costs must be non-negative")
        optimistic += max(compute, communication)
        expected += (
            compute
            + communication
            - expected_overlap_fraction * min(compute, communication)
        )
        pessimistic += compute + communication
    return {
        "unit": "us",
        "optimistic": optimistic,
        "expected": expected,
        "pessimistic": pessimistic,
        "calibration_relative_error": 0.0,
        "schedule_model": schedule_model,
        "expected_overlap_fraction": expected_overlap_fraction,
    }


def autoparallel_estimate(
    plan: Mapping[str, Any],
    *,
    expected_overlap_fraction: float = 0.5,
    mesh_shape: tuple[int, ...] = (1,),
    usable_device_bytes: int | None = None,
    operational_reserve_fraction: float = 0.1,
    optimizer_calibration: Mapping[str, Any] | None = None,
    optimizer_applicability: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    phases = optimizer_phase_costs(plan)
    storage = plan["summary"]["parameter_storage"]
    local_parameter_bytes = int(storage["local_bytes"])
    latency = overlap_envelope(
        phases,
        expected_overlap_fraction=expected_overlap_fraction,
        schedule_model="phase-local-overlap-envelope-v1",
    )
    activation = activation_liveness_estimate(plan, mesh_shape)
    persistent = 4 * local_parameter_bytes
    memory = _complete_memory_projection(
        persistent,
        activation["peak_bytes"],
        usable_device_bytes,
        operational_reserve_fraction,
    )
    update = optimizer_update_estimate(
        local_parameter_bytes, optimizer_calibration, optimizer_applicability
    )
    return {
        "latency": latency,
        "full_step_latency": _combine_latency(latency, update),
        "optimizer_update": update,
        "phase_costs": phases,
        "memory": {
            "scope": "analytical-training-state-activation-liveness-and-reserve",
            "local_parameter_bytes": local_parameter_bytes,
            "modeled_local_training_state_bytes": persistent,
            "activation_liveness": activation,
            **memory,
        },
    }


def ddp_estimate(
    replicated_plan: Mapping[str, Any],
    parameter_sizes_fp32: Sequence[int],
    topology: NCCLTopoConfig,
    mesh_shape: tuple[int, ...],
    *,
    mesh_dim: int = 0,
    bucket_capacity_bytes: int = 25 * 1024 * 1024,
    expected_overlap_fraction: float = 0.5,
    usable_device_bytes: int | None = None,
    operational_reserve_fraction: float = 0.1,
    optimizer_calibration: Mapping[str, Any] | None = None,
    optimizer_applicability: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    parameter_sizes = _validated_sizes(parameter_sizes_fp32)
    buckets = pack_buckets(reversed(parameter_sizes), bucket_capacity_bytes)
    mesh_topology = derive_mesh_dim_topo(topology, mesh_shape, mesh_dim)
    bucket_costs = [
        nccl_allreduce_cost(size, mesh_topology, topology) for size in buckets
    ]
    phases = optimizer_phase_costs(replicated_plan)
    phases["forward"]["communication_us"] = 0.0
    phases["backward"]["communication_us"] = sum(bucket_costs)
    global_parameter_bytes = sum(parameter_sizes)
    training_state_bytes = 4 * global_parameter_bytes
    state_fits = (
        training_state_bytes <= usable_device_bytes
        if usable_device_bytes is not None
        else None
    )
    latency = overlap_envelope(
        phases,
        expected_overlap_fraction=expected_overlap_fraction,
        schedule_model="ddp-gradient-bucket-overlap-envelope-v1",
    )
    activation = activation_liveness_estimate(replicated_plan, mesh_shape)
    memory = _complete_memory_projection(
        training_state_bytes,
        activation["peak_bytes"],
        usable_device_bytes,
        operational_reserve_fraction,
    )
    update = optimizer_update_estimate(
        global_parameter_bytes, optimizer_calibration, optimizer_applicability
    )
    return {
        "strategy": "ddp",
        "template_id": "pytorch-compiled-ddp-cost-model",
        "template_version": "1",
        "latency": latency,
        "full_step_latency": _combine_latency(latency, update),
        "optimizer_update": update,
        "phase_costs": phases,
        "communication": {
            "collective": "allreduce",
            "gradient_dtype": "float32",
            "bucket_capacity_bytes": bucket_capacity_bytes,
            "bucket_count": len(buckets),
            "bucket_sizes_bytes": buckets,
            "total_collective_us": sum(bucket_costs),
        },
        "memory": {
            "scope": "analytical-training-state-activation-liveness-and-reserve",
            "modeled_local_training_state_bytes": training_state_bytes,
            "state_fits": state_fits,
            "activation_liveness": activation,
            **memory,
        },
    }


def fsdp_estimate(
    replicated_plan: Mapping[str, Any],
    wrap_unit_sizes_fp32: Sequence[int],
    topology: NCCLTopoConfig,
    mesh_shape: tuple[int, ...],
    *,
    mesh_dim: int = 0,
    parameter_dtype_bytes: int = 2,
    persistent_parameter_dtype_bytes: int = 4,
    gradient_dtype_bytes: int = 4,
    expected_overlap_fraction: float = 0.5,
    usable_device_bytes: int | None = None,
    operational_reserve_fraction: float = 0.1,
    optimizer_calibration: Mapping[str, Any] | None = None,
    optimizer_applicability: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    units_fp32 = _validated_sizes(wrap_unit_sizes_fp32)
    if persistent_parameter_dtype_bytes <= 0:
        raise ValueError("persistent_parameter_dtype_bytes must be positive")
    if parameter_dtype_bytes <= 0 or gradient_dtype_bytes <= 0:
        raise ValueError("parameter and gradient dtype sizes must be positive")

    mesh_topology = derive_mesh_dim_topo(topology, mesh_shape, mesh_dim)
    parameter_units = [
        math.ceil(size * parameter_dtype_bytes / persistent_parameter_dtype_bytes)
        for size in units_fp32
    ]
    gradient_units = [
        math.ceil(size * gradient_dtype_bytes / persistent_parameter_dtype_bytes)
        for size in units_fp32
    ]
    allgather_costs = [
        nccl_allgather_cost(size, mesh_topology, topology) for size in parameter_units
    ]
    reduce_scatter_costs = [
        nccl_reduce_scatter_cost(size, mesh_topology, topology)
        for size in gradient_units
    ]
    forward_allgather_us = sum(allgather_costs)
    backward_allgather_us = forward_allgather_us
    backward_reduce_scatter_us = sum(reduce_scatter_costs)
    phases = optimizer_phase_costs(replicated_plan)
    phases["forward"]["communication_us"] = forward_allgather_us
    phases["backward"]["communication_us"] = (
        backward_allgather_us + backward_reduce_scatter_us
    )

    world_size = math.prod(mesh_shape)
    global_parameter_bytes = sum(units_fp32)
    local_training_state_bytes = math.ceil(4 * global_parameter_bytes / world_size)
    largest_parameter_unit = max(parameter_units, default=0)
    largest_gradient_unit = max(gradient_units, default=0)
    minimum_collective_buffer_bytes = max(largest_parameter_unit, largest_gradient_unit)
    conservative_collective_buffer_bytes = (
        2 * largest_parameter_unit + largest_gradient_unit
    )
    upper_state_and_buffers = (
        local_training_state_bytes + conservative_collective_buffer_bytes
    )
    state_and_buffers_fit = (
        upper_state_and_buffers <= usable_device_bytes
        if usable_device_bytes is not None
        else None
    )
    latency = overlap_envelope(
        phases,
        expected_overlap_fraction=expected_overlap_fraction,
        schedule_model="fsdp2-prefetch-reshard-overlap-envelope-v1",
    )
    activation = activation_liveness_estimate(replicated_plan, mesh_shape)
    memory = _complete_memory_projection(
        upper_state_and_buffers,
        activation["peak_bytes"],
        usable_device_bytes,
        operational_reserve_fraction,
    )
    update = optimizer_update_estimate(
        local_training_state_bytes // 4,
        optimizer_calibration,
        optimizer_applicability,
    )
    return {
        "strategy": "fsdp",
        "template_id": "pytorch-compiled-fsdp2-transformer-block-cost-model",
        "template_version": "1",
        "latency": latency,
        "full_step_latency": _combine_latency(latency, update),
        "optimizer_update": update,
        "phase_costs": phases,
        "communication": {
            "wrap_unit_count": len(units_fp32),
            "wrap_unit_sizes_fp32_bytes": units_fp32,
            "forward_allgather_us": forward_allgather_us,
            "backward_allgather_us": backward_allgather_us,
            "backward_reduce_scatter_us": backward_reduce_scatter_us,
            "total_collective_us": (
                forward_allgather_us
                + backward_allgather_us
                + backward_reduce_scatter_us
            ),
            "parameter_dtype_bytes": parameter_dtype_bytes,
            "gradient_dtype_bytes": gradient_dtype_bytes,
            "reshard_after_forward": True,
        },
        "memory": {
            "scope": (
                "analytical-sharded-training-state-collective-buffers-"
                "activation-liveness-and-reserve"
            ),
            "modeled_local_training_state_bytes": local_training_state_bytes,
            "minimum_state_and_collective_buffers_bytes": (
                local_training_state_bytes + minimum_collective_buffer_bytes
            ),
            "conservative_state_and_collective_buffers_bytes": (
                upper_state_and_buffers
            ),
            "state_and_collective_buffers_fit": state_and_buffers_fit,
            "activation_liveness": activation,
            **memory,
        },
    }


def _combine_latency(
    partial: Mapping[str, Any], update: Mapping[str, Any]
) -> dict[str, Any] | None:
    if update.get("status") not in {"applicable", "screening_only"}:
        return None
    increment = update["latency"]
    return {
        "unit": "us",
        "optimistic": partial["optimistic"] + increment["optimistic"],
        "expected": partial["expected"] + increment["expected"],
        "pessimistic": partial["pessimistic"] + increment["pessimistic"],
        "calibration_relative_error": 0.0,
        "schedule_model": "partial-step-plus-calibrated-optimizer-update-v1",
        "components": {
            "partial_step": partial["schedule_model"],
            "optimizer_update": increment["schedule_model"],
        },
    }


def _complete_memory_projection(
    persistent_bytes: int,
    activation_peak_bytes: int,
    usable_device_bytes: int | None,
    reserve_fraction: float,
) -> dict[str, Any]:
    if not 0.0 <= reserve_fraction < 1.0:
        raise ValueError("operational_reserve_fraction must be in [0, 1)")
    subtotal = persistent_bytes + activation_peak_bytes
    reserve = math.ceil(subtotal * reserve_fraction)
    projected = subtotal + reserve
    return {
        "modeled_activation_peak_bytes": activation_peak_bytes,
        "operational_reserve_fraction": reserve_fraction,
        "operational_reserve_bytes": reserve,
        "projected_complete_step_peak_bytes": projected,
        "usable_device_bytes": usable_device_bytes,
        "complete_step_fits": (
            projected <= usable_device_bytes
            if usable_device_bytes is not None
            else None
        ),
        "evidence": "analytical_projection",
    }


def _local_tensor_bytes(node: Mapping[str, Any], mesh_shape: tuple[int, ...]) -> int:
    shape = node.get("shape")
    dtype = str(node.get("dtype"))
    if not isinstance(shape, Sequence) or dtype not in _DTYPE_BYTES:
        return 0
    local_shape = [int(dim) for dim in shape]
    placements = list(_PLACEMENT_RE.finditer(str(node.get("placement", ""))))
    for mesh_dim, placement in enumerate(placements):
        if mesh_dim >= len(mesh_shape):
            break
        tensor_dim_text = placement.group(1)
        if tensor_dim_text is None:
            continue
        tensor_dim = int(tensor_dim_text)
        if 0 <= tensor_dim < len(local_shape):
            local_shape[tensor_dim] = math.ceil(
                local_shape[tensor_dim] / mesh_shape[mesh_dim]
            )
    return math.prod(local_shape) * _DTYPE_BYTES[dtype]


def _validated_sizes(values: Sequence[int]) -> list[int]:
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("parameter sizes must be non-negative integers")
        if value:
            result.append(value)
    return result
