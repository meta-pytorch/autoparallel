from __future__ import annotations

import copy
import json
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class CampaignError(ValueError):
    pass


PROFILES = {
    "tt_main_manual_jit_v1",
    "gt_manual_aot_v1",
    "apgt_v1",
    "gt_manual_cp_legacy_v1",
    "apgt_cp_legacy_v1",
    "apgt_3d_exact_mesh_flash_v1",
    "ap_backend_legacy_v1",
}
PHASE_KINDS = {"correctness", "performance", "kineto", "torch_trace", "trace"}
CONFIG_SECTIONS = {
    "training",
    "parallelism",
    "compile",
    "activation_checkpoint",
    "optimizer",
    "lr_scheduler",
    "metrics",
    "profiler",
    "debug",
    "checkpoint",
    "validator",
    "comm",
}
HARNESS_ONLY_SETTINGS = {"training.gradient_accumulation_steps"}
RESERVED_ENVIRONMENT_KEYS = {
    "CONDA_DIR",
    "DUMP_DIR",
    "HARNESS_PAYLOAD_ROOT",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "PYTHONHASHSEED",
    "PYTHONPATH",
    "PYTHONDONTWRITEBYTECODE",
    "RUN_ROOT",
    "TITAN_STRUCT_LOGGER_HANDLERS",
    "TMPDIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "TORCHX_RUN_PYTHONPATH",
    "TORCH_TRACE",
    "TRITON_CACHE_DIR",
    "WORLD_SIZE",
}
TOP_LEVEL_KEYS = {
    "schema_version",
    "name",
    "description",
    "workload",
    "sources",
    "model",
    "data",
    "training",
    "parallelism",
    "precision",
    "activation_checkpoint",
    "compile",
    "optimizer",
    "lr_scheduler",
    "metrics",
    "profiler",
    "debug",
    "checkpoint",
    "validator",
    "comm",
    "torchtitan",
    "autoparallel",
    "comparison",
    "measurement",
    "mast",
    "artifacts",
    "trace",
    "arms",
    "phases",
    "legacy",
    "matrix",
    "selected_point",
    "execution_mode",
}


@dataclass(frozen=True)
class Arm:
    name: str
    profile: str
    module: str
    config: str
    overrides: dict[str, Any]
    args: tuple[str, ...]
    environment: dict[str, str]


@dataclass(frozen=True)
class Phase:
    name: str
    kind: str
    arms: tuple[str, ...]
    overrides: dict[str, Any]
    args: tuple[str, ...]
    trace_ranks: tuple[int, ...]
    environment: dict[str, str]


@dataclass(frozen=True)
class Campaign:
    path: Path
    raw: dict[str, Any]
    arms: tuple[Arm, ...]
    phases: tuple[Phase, ...]

    @property
    def name(self) -> str:
        base = str(self.raw["name"])
        point = self.raw.get("selected_point")
        name = f"{base}-{point}" if point else base
        return f"{name}-gate" if self.raw.get("execution_mode") == "gate" else name

    @property
    def world_size(self) -> int:
        mast = self.raw["mast"]
        return int(mast["nodes"]) * int(mast["nproc_per_node"])

    @property
    def arm_names(self) -> set[str]:
        return {arm.name for arm in self.arms}

    def arm(self, name: str) -> Arm:
        return next(arm for arm in self.arms if arm.name == name)

    def source_specs(self) -> dict[str, dict[str, Any]]:
        return copy.deepcopy(self.raw["sources"])

    def common_overrides(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for section in CONFIG_SECTIONS:
            section_values = self.raw.get(section, {})
            if not isinstance(section_values, dict):
                raise CampaignError(f"[{section}] must be a table")
            for key, value in _flatten(section_values, prefix=section).items():
                values[key] = value
        native = self.raw.get("torchtitan", {}).get("overrides", {})
        if not isinstance(native, dict):
            raise CampaignError("[torchtitan.overrides] must be a table")
        for key, value in native.items():
            if key in values:
                raise CampaignError(
                    f"TorchTitan setting {key!r} is set in both a structured section "
                    "and [torchtitan.overrides]"
                )
            values[key] = value
        return values

    def phase_arm_args(self, phase: Phase, arm: Arm) -> list[str]:
        values = self.phase_arm_settings(phase, arm)
        args = ["--module", arm.module, "--config", arm.config]
        args.extend(settings_to_argv(values))
        args.extend(arm.args)
        args.extend(phase.args)
        return args

    def phase_arm_settings(self, phase: Phase, arm: Arm) -> dict[str, Any]:
        values = self.common_overrides()
        _merge_unique(values, arm.overrides, owner=f"arm {arm.name}")
        _merge_unique(values, phase.overrides, owner=f"phase {phase.name}")
        return values

    def phase_arm_environment(self, phase: Phase, arm: Arm) -> dict[str, str]:
        result = dict(arm.environment)
        for key, value in phase.environment.items():
            if key in result:
                raise CampaignError(
                    f"phase {phase.name} overrides arm environment variable {key!r}"
                )
            result[key] = value
        return result

    def resolved_dict(self) -> dict[str, Any]:
        value = copy.deepcopy(self.raw)
        value["name"] = self.name
        value["campaign_file"] = str(self.path.resolve())
        value["world_size"] = self.world_size
        value["resolved_arms"] = [
            {
                "name": arm.name,
                "profile": arm.profile,
                "module": arm.module,
                "config": arm.config,
            }
            for arm in self.arms
        ]
        value["resolved_phases"] = [
            {
                "name": phase.name,
                "kind": phase.kind,
                "arms": list(phase.arms),
                "trace_ranks": list(phase.trace_ranks),
                "argv_by_arm": {
                    arm_name: self.phase_arm_args(phase, self.arm(arm_name))
                    for arm_name in phase.arms
                },
                "environment_by_arm": {
                    arm_name: self.phase_arm_environment(phase, self.arm(arm_name))
                    for arm_name in phase.arms
                },
            }
            for phase in self.phases
        ]
        return value


def _flatten(value: dict[str, Any], *, prefix: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, child in value.items():
        path = f"{prefix}.{key}"
        if isinstance(child, dict):
            result.update(_flatten(child, prefix=path))
        else:
            result[path] = child
    return result


def _merge_unique(target: dict[str, Any], values: dict[str, Any], *, owner: str) -> None:
    for key, value in values.items():
        if not isinstance(key, str) or "." not in key:
            raise CampaignError(f"{owner} setting {key!r} must be a dotted path")
        target[key] = value


def _option_name(path: str, *, enabled: bool | None = None) -> str:
    pieces = path.replace("_", "-").split(".")
    if enabled is False:
        pieces[-1] = f"no-{pieces[-1]}"
    return "--" + ".".join(pieces)


def settings_to_argv(settings: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for path, value in sorted(settings.items()):
        if path in HARNESS_ONLY_SETTINGS:
            continue
        if not path or any(not part for part in path.split(".")):
            raise CampaignError(f"invalid TorchTitan setting path {path!r}")
        if isinstance(value, bool):
            result.append(_option_name(path, enabled=value))
        elif isinstance(value, list):
            result.append(_option_name(path))
            result.append(",".join(str(item) for item in value))
        elif value is not None:
            result.extend((_option_name(path), str(value)))
    return result


def _string_list(value: Any, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise CampaignError(f"{field} must be an array of strings")
    return tuple(value)


def _string_map(value: Any, *, field: str) -> dict[str, str]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, (str, int, float, bool))
        for key, item in value.items()
    ):
        raise CampaignError(f"{field} must be a string-to-scalar table")
    return {key: str(item) for key, item in value.items()}


def _parse_arms(raw: dict[str, Any]) -> tuple[Arm, ...]:
    rows = raw.get("arms")
    if not isinstance(rows, list) or not rows:
        raise CampaignError("at least one [[arms]] entry is required")
    result = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise CampaignError(f"arms[{index}] must be a table")
        required = {"name", "profile", "module", "config"}
        missing = required - row.keys()
        if missing:
            raise CampaignError(f"arms[{index}] is missing {sorted(missing)}")
        profile = str(row["profile"])
        if profile not in PROFILES:
            raise CampaignError(f"unknown profile {profile!r}")
        overrides = row.get("overrides", {})
        if not isinstance(overrides, dict):
            raise CampaignError(f"arms[{index}].overrides must be a table")
        result.append(
            Arm(
                name=str(row["name"]),
                profile=profile,
                module=str(row["module"]),
                config=str(row["config"]),
                overrides=dict(overrides),
                args=_string_list(row.get("args", []), field=f"arms[{index}].args"),
                environment=_string_map(
                    row.get("environment", {}), field=f"arms[{index}].environment"
                ),
            )
        )
    names = [arm.name for arm in result]
    if len(set(names)) != len(names):
        raise CampaignError("arm names must be unique")
    return tuple(result)


def _parse_phases(raw: dict[str, Any], arm_names: set[str]) -> tuple[Phase, ...]:
    rows = raw.get("phases")
    if not isinstance(rows, list) or not rows:
        raise CampaignError("at least one [[phases]] entry is required")
    result = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise CampaignError(f"phases[{index}] must be a table")
        required = {"name", "kind", "arms"}
        missing = required - row.keys()
        if missing:
            raise CampaignError(f"phases[{index}] is missing {sorted(missing)}")
        kind = str(row["kind"])
        if kind not in PHASE_KINDS:
            raise CampaignError(f"unknown phase kind {kind!r}")
        arms = _string_list(row["arms"], field=f"phases[{index}].arms")
        if not arms:
            raise CampaignError(f"phases[{index}].arms cannot be empty")
        unknown = set(arms) - arm_names
        if unknown:
            raise CampaignError(f"phase {row['name']!r} has unknown arms {sorted(unknown)}")
        overrides = row.get("overrides", {})
        if not isinstance(overrides, dict):
            raise CampaignError(f"phases[{index}].overrides must be a table")
        trace_ranks = row.get(
            "trace_ranks", [0] if kind in {"kineto", "torch_trace", "trace"} else []
        )
        if not isinstance(trace_ranks, list) or not all(
            isinstance(rank, int) and rank >= 0 for rank in trace_ranks
        ):
            raise CampaignError(f"phases[{index}].trace_ranks must contain nonnegative integers")
        result.append(
            Phase(
                name=str(row["name"]),
                kind=kind,
                arms=arms,
                overrides=dict(overrides),
                args=_string_list(row.get("args", []), field=f"phases[{index}].args"),
                trace_ranks=tuple(trace_ranks),
                environment=_string_map(
                    row.get("environment", {}), field=f"phases[{index}].environment"
                ),
            )
        )
    names = [phase.name for phase in result]
    if len(set(names)) != len(names):
        raise CampaignError("phase names must be unique")
    return tuple(result)


def _validate_campaign(campaign: Campaign) -> None:
    raw = campaign.raw
    unknown = set(raw) - TOP_LEVEL_KEYS
    if unknown:
        raise CampaignError(f"unknown top-level keys: {sorted(unknown)}")
    if raw.get("schema_version") != 1:
        raise CampaignError("schema_version must be 1")
    if not raw.get("name") or not raw.get("workload"):
        raise CampaignError("name and workload are required")
    sources = raw.get("sources")
    if not isinstance(sources, dict) or set(sources) < {"torchtitan", "autoparallel"}:
        raise CampaignError("[sources.torchtitan] and [sources.autoparallel] are required")
    for name, source in sources.items():
        if not isinstance(source, dict):
            raise CampaignError(f"sources.{name} must be a table")
        if not source.get("commit") or not source.get("remote"):
            raise CampaignError(f"sources.{name} requires remote and commit")
        if source.get("dirty_policy", "forbid") not in {"forbid", "snapshot"}:
            raise CampaignError(f"sources.{name}.dirty_policy must be forbid or snapshot")

    mast = raw.get("mast")
    if not isinstance(mast, dict):
        raise CampaignError("[mast] is required")
    for key in ("nodes", "nproc_per_node", "hardware", "locality"):
        if key not in mast:
            raise CampaignError(f"mast.{key} is required")
    if int(mast["nodes"]) <= 0 or int(mast["nproc_per_node"]) <= 0:
        raise CampaignError("MAST node and process counts must be positive")
    if int(mast.get("retries", 0)) != 0:
        raise CampaignError("paired permanent-harness campaigns require mast.retries = 0")
    locality = str(mast["locality"]).split(";", 1)
    if len(locality) != 2 or locality[0] not in {"dc", "region"} or not locality[1]:
        raise CampaignError(
            "mast.locality must be an explicit 'dc;NAME' or 'region;NAME' constraint"
        )

    environment_maps = [("mast.environment", mast.get("environment", {}))]
    environment_maps.extend(
        (f"arms.{arm.name}.environment", arm.environment) for arm in campaign.arms
    )
    environment_maps.extend(
        (f"phases.{phase.name}.environment", phase.environment)
        for phase in campaign.phases
    )
    for owner, environment in environment_maps:
        if not isinstance(environment, dict):
            raise CampaignError(f"{owner} must be a table")
        reserved = sorted(set(environment) & RESERVED_ENVIRONMENT_KEYS)
        if reserved:
            raise CampaignError(f"{owner} sets harness-reserved keys: {reserved}")

    comparison = raw.get("comparison", {})
    if len(campaign.arms) > 1:
        if not comparison.get("declared_variable"):
            raise CampaignError("multi-arm campaigns require comparison.declared_variable")
        allowed = comparison.get("allowed_config_paths")
        if not isinstance(allowed, list) or not all(isinstance(item, str) for item in allowed):
            raise CampaignError("comparison.allowed_config_paths must be an array of strings")
        allowed_environment = comparison.get("allowed_environment_keys", [])
        if not isinstance(allowed_environment, list) or not all(
            isinstance(item, str) for item in allowed_environment
        ):
            raise CampaignError(
                "comparison.allowed_environment_keys must be an array of strings"
            )
        pairs = comparison.get("pairs")
        if not isinstance(pairs, list) or not pairs:
            raise CampaignError("multi-arm campaigns require comparison.pairs")
        for index, pair in enumerate(pairs):
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or not all(isinstance(name, str) for name in pair)
                or pair[0] == pair[1]
                or not set(pair) <= campaign.arm_names
            ):
                raise CampaignError(f"comparison.pairs[{index}] is invalid: {pair!r}")
        phase_by_arm = comparison.get("performance_phase_by_arm", {})
        if not isinstance(phase_by_arm, dict) or not all(
            isinstance(arm, str) and isinstance(phase, str)
            for arm, phase in phase_by_arm.items()
        ):
            raise CampaignError(
                "comparison.performance_phase_by_arm must be a string table"
            )
        phases_by_name = {phase.name: phase for phase in campaign.phases}
        for arm, phase_name in phase_by_arm.items():
            phase = phases_by_name.get(phase_name)
            if arm not in campaign.arm_names or phase is None:
                raise CampaignError(
                    "comparison.performance_phase_by_arm references an unknown arm or phase"
                )
            if phase.kind != "performance" or arm not in phase.arms:
                raise CampaignError(
                    f"{phase_name!r} is not a performance phase containing {arm!r}"
                )
        acceptance = comparison.get("acceptance")
        if acceptance is not None:
            if not isinstance(acceptance, dict):
                raise CampaignError("comparison.acceptance must be a table")
            required = {
                "metric",
                "baseline",
                "treatment",
                "reference_gap_percent",
                "tolerance_percentage_points",
            }
            if set(acceptance) != required:
                raise CampaignError(
                    f"comparison.acceptance requires exactly {sorted(required)}"
                )
            if acceptance["metric"] != "primary_latency":
                raise CampaignError(
                    "comparison.acceptance.metric must be primary_latency"
                )
            pair = [acceptance["baseline"], acceptance["treatment"]]
            if pair not in pairs:
                raise CampaignError(
                    "comparison.acceptance baseline/treatment must be a declared pair"
                )
            if not isinstance(acceptance["reference_gap_percent"], (int, float)):
                raise CampaignError("reference_gap_percent must be numeric")
            tolerance = acceptance["tolerance_percentage_points"]
            if not isinstance(tolerance, (int, float)) or tolerance < 0:
                raise CampaignError(
                    "tolerance_percentage_points must be a nonnegative number"
                )

    measurement = raw.get("measurement")
    if measurement is not None:
        if not isinstance(measurement, dict) or set(measurement) != {"primary"}:
            raise CampaignError("[measurement] requires only [measurement.primary]")
        primary = measurement["primary"]
        if not isinstance(primary, dict):
            raise CampaignError("measurement.primary must be a table")
        source = primary.get("source")
        steps = primary.get("steps")
        if source not in {"structured_logs", "tensorboard"}:
            raise CampaignError("measurement.primary.source is unsupported")
        if (
            not isinstance(steps, list)
            or not steps
            or not all(isinstance(step, int) and step > 0 for step in steps)
            or len(steps) != len(set(steps))
        ):
            raise CampaignError(
                "measurement.primary.steps must contain unique positive integers"
            )
        rank_selector = primary.get("rank_selector")
        value = primary.get("value")
        supported = {
            ("structured_logs", "max_step_end", "step_end_duration"),
            (
                "structured_logs",
                "latest_fetching_batch_end",
                "step_end_minus_fetching_batch_end",
            ),
            ("tensorboard", "rank0", "time_metrics/end_to_end(s)"),
        }
        if (source, rank_selector, value) not in supported:
            raise CampaignError(
                "measurement.primary source/rank_selector/value combination is unsupported"
            )
        allowed_primary = {"source", "steps", "rank_selector", "value"}
        if source == "tensorboard":
            allowed_primary |= {
                "unit_scale_to_ms",
                "throughput_tag",
                "active_memory_tag",
                "reserved_memory_tag",
            }
            if primary.get("unit_scale_to_ms") != 1000:
                raise CampaignError("TensorBoard latency must declare unit_scale_to_ms = 1000")
        unknown_primary = set(primary) - allowed_primary
        if unknown_primary:
            raise CampaignError(
                f"unknown measurement.primary keys: {sorted(unknown_primary)}"
            )

    data = raw.get("data", {})
    mode = data.get("mode", "streaming")
    if mode not in {"streaming", "replay"}:
        raise CampaignError("data.mode must be streaming or replay")
    identity = data.get("identity")
    allowed_identities = {"index_manifest", "batch_hash_manifest"}
    if mode == "streaming" and identity not in allowed_identities:
        raise CampaignError(
            "streaming comparisons require an index or preflight batch-hash manifest"
        )
    preflight_auditor = data.get("preflight_auditor")
    if preflight_auditor is not None and (
        not isinstance(preflight_auditor, str)
        or not preflight_auditor.startswith("workloads.")
    ):
        raise CampaignError("data.preflight_auditor must name a workloads.* module")

    campaign.common_overrides()
    for phase in campaign.phases:
        if any(rank >= campaign.world_size for rank in phase.trace_ranks):
            raise CampaignError(
                f"phase {phase.name!r} trace rank is outside world size "
                f"{campaign.world_size}"
            )
        for arm_name in phase.arms:
            arm = campaign.arm(arm_name)
            settings = campaign.phase_arm_settings(phase, arm)
            campaign.phase_arm_args(phase, arm)
            degree_names = (
                "data_parallel_replicate_degree",
                "data_parallel_shard_degree",
                "context_parallel_degree",
                "tensor_parallel_degree",
                "pipeline_parallel_degree",
            )
            degrees = [
                int(settings.get(f"parallelism.{name}", 1)) for name in degree_names
            ]
            product = math.prod(degrees)
            if product != campaign.world_size:
                raise CampaignError(
                    f"phase {phase.name!r} arm {arm_name!r} parallel mesh product "
                    f"{product} does not equal world size {campaign.world_size}"
                )
            local_batch = settings.get("training.local_batch_size")
            global_batch = settings.get("training.global_batch_size")
            grad_accum = int(settings.get("training.gradient_accumulation_steps", 1))
            if (
                isinstance(local_batch, int)
                and isinstance(global_batch, int)
                and global_batch > 0
            ):
                expected = local_batch * degrees[0] * degrees[1] * grad_accum
                if global_batch != expected:
                    raise CampaignError(
                        f"phase {phase.name!r} arm {arm_name!r} global batch "
                        f"{global_batch} != local batch {local_batch} * DP "
                        f"{degrees[0] * degrees[1]} * gradient accumulation {grad_accum}"
                    )
                environment = {
                    **mast.get("environment", {}),
                    **campaign.phase_arm_environment(phase, arm),
                }
                expected_environment = {
                    "BENCHMARK_WORLD_SIZE": campaign.world_size,
                    "BENCHMARK_DP_DEGREE": degrees[0] * degrees[1],
                    "BENCHMARK_TP_DEGREE": degrees[3],
                    "BENCHMARK_LOCAL_BATCH_SIZE": local_batch,
                    "BENCHMARK_GLOBAL_BATCH_SIZE": global_batch,
                }
                for key, expected_value in expected_environment.items():
                    if key in environment and int(environment[key]) != expected_value:
                        raise CampaignError(
                            f"phase {phase.name!r} arm {arm_name!r} {key} "
                            f"{environment[key]} != resolved setting {expected_value}"
                        )


def _apply_matrix(raw: dict[str, Any], point: str | None) -> dict[str, Any]:
    matrix = raw.get("matrix")
    if matrix is None:
        if point is not None:
            raise CampaignError("--point was provided for a campaign without [matrix]")
        return raw
    if not isinstance(matrix, dict) or not isinstance(matrix.get("points"), list):
        raise CampaignError("[matrix] requires [[matrix.points]] entries")
    names = [entry.get("name") for entry in matrix["points"]]
    if point is None:
        raise CampaignError(f"campaign requires --point; available values: {names}")
    matches = [entry for entry in matrix["points"] if entry.get("name") == point]
    if len(matches) != 1:
        raise CampaignError(f"unknown or duplicate matrix point {point!r}; available: {names}")
    result = copy.deepcopy(raw)
    result["selected_point"] = point
    settings = matches[0].get("settings", {})
    if not isinstance(settings, dict):
        raise CampaignError(f"matrix point {point!r} settings must be a table")
    for path, value in settings.items():
        current = result
        pieces = path.split(".")
        if any(not piece for piece in pieces):
            raise CampaignError(f"invalid matrix setting path {path!r}")
        for piece in pieces[:-1]:
            child = current.setdefault(piece, {})
            if not isinstance(child, dict):
                raise CampaignError(f"matrix setting {path!r} crosses a non-table value")
            current = child
        current[pieces[-1]] = value
    return result


def _apply_mode(raw: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode not in {"formal", "gate"}:
        raise CampaignError(f"unknown execution mode {mode!r}")
    result = copy.deepcopy(raw)
    result["execution_mode"] = mode
    if mode == "formal":
        return result

    result.pop("measurement", None)
    comparison = result.get("comparison", {})
    comparison.pop("performance_phase_by_arm", None)
    comparison.pop("acceptance", None)

    phases = result.get("phases", [])
    if not phases:
        return result
    base = next((phase for phase in phases if phase.get("kind") == "performance"), phases[0])
    trace = next(
        (phase for phase in phases if phase.get("kind") in {"trace", "kineto"}),
        base,
    )
    functional = copy.deepcopy(base)
    functional["name"] = "functional"
    functional["kind"] = "correctness"
    functional["arms"] = [arm["name"] for arm in result["arms"]]
    functional.setdefault("overrides", {}).update(
        {
            "training.steps": 2,
            "profiler.enable_profiling": False,
            "profiler.enable_memory_snapshot": False,
            "metrics.log_freq": 1,
        }
    )
    functional.setdefault("environment", {}).update(
        {
            "BENCHMARK_TOTAL_STEPS": "2",
            "BENCHMARK_LOG_FREQ": "1",
            "BENCHMARK_ENABLE_KINETO": "0",
            "BENCHMARK_ENABLE_MEMORY_SNAPSHOT": "0",
        }
    )
    trace_smoke = copy.deepcopy(trace)
    trace_smoke["name"] = "trace_smoke"
    trace_smoke["kind"] = "trace"
    trace_smoke["arms"] = [arm["name"] for arm in result["arms"]]
    trace_smoke.setdefault("trace_ranks", [0])
    trace_smoke.setdefault("overrides", {}).update(
        {
            "training.steps": 2,
            "profiler.enable_profiling": True,
            "profiler.enable_memory_snapshot": False,
            "profiler.profile_freq": 2,
            "profiler.profiler_warmup": 0,
            "profiler.profiler_active": 1,
            "profiler.profiler_repeat": 1,
            "metrics.log_freq": 1,
        }
    )
    trace_smoke.setdefault("environment", {}).update(
        {
            "BENCHMARK_TOTAL_STEPS": "2",
            "BENCHMARK_LOG_FREQ": "1",
            "BENCHMARK_ENABLE_KINETO": "1",
            "BENCHMARK_ENABLE_MEMORY_SNAPSHOT": "0",
            "BENCHMARK_PROFILE_FREQ": "2",
            "BENCHMARK_PROFILE_ACTIVE": "1",
            "BENCHMARK_PROFILE_WARMUP": "0",
        }
    )
    result["phases"] = [functional, trace_smoke]
    return result


def load_campaign(
    path: Path, *, point: str | None = None, mode: str = "formal"
) -> Campaign:
    path = path.resolve()
    with path.open("rb") as stream:
        raw = _apply_mode(_apply_matrix(tomllib.load(stream), point), mode)
    arms = _parse_arms(raw)
    phases = _parse_phases(raw, {arm.name for arm in arms})
    campaign = Campaign(path=path, raw=raw, arms=arms, phases=phases)
    _validate_campaign(campaign)
    return campaign


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
