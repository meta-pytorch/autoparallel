from __future__ import annotations

from pathlib import Path
from typing import Any

from .campaign import Arm, CampaignError


def _get(config: dict[str, Any], path: str) -> Any:
    value: Any = config
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise CampaignError(f"serialized config is missing {path!r}")
        value = value[part]
    return value


def _expect(config: dict[str, Any], expected: dict[str, Any]) -> None:
    mismatches = {
        path: {"expected": value, "actual": _get(config, path)}
        for path, value in expected.items()
        if _get(config, path) != value
    }
    if mismatches:
        raise CampaignError(f"profile contract mismatch: {mismatches}")


def validate_profile(arm: Arm, config: dict[str, Any]) -> dict[str, Any]:
    if arm.profile == "tt_main_default_v1":
        model_spec_name = str(_get(config, "model_spec.name"))
        if "graphtrainer" in model_spec_name.replace("_", "").lower():
            raise CampaignError(
                "tt_main_default_v1 cannot use a GraphTrainer model spec"
            )
        _expect(
            config,
            {
                "compile.enable": True,
                "compile.backend": "inductor",
            },
        )
    elif arm.profile == "gt_manual_eager_v1":
        _expect(
            config,
            {
                "compile.enable": True,
                "compile.backend": "aot_eager",
                "compile.mode": "aot_fx_trace",
                "compile.memory_policy": "eager",
                "compile.inductor_compilation": "full",
                "compile.numerics_changing_optim": False,
                "compile.enable_fsdp_ag_rs_overlap": False,
                "compile.enable_fsdp_dense_region_overlap": False,
                "compile.enable_autoparallel": False,
            },
        )
        if set(_get(config, "compile.disable_passes")) != {"cudagraph_pass"}:
            raise CampaignError(
                "gt_manual_eager_v1 requires only cudagraph_pass to be disabled"
            )
    elif arm.profile == "apgt_validated_v1":
        _expect(
            config,
            {
                "compile.enable": True,
                "compile.backend": "aot_eager",
                "compile.mode": "aot_fx_trace",
                "compile.memory_policy": "eager",
                "compile.inductor_compilation": "full",
                "compile.numerics_changing_optim": False,
                "compile.enable_fsdp_ag_rs_overlap": False,
                "compile.enable_fsdp_dense_region_overlap": False,
                "compile.enable_autoparallel": True,
            },
        )
        if set(_get(config, "compile.disable_passes")) != {"cudagraph_pass"}:
            raise CampaignError(
                "apgt_validated_v1 requires only cudagraph_pass to be disabled"
            )
    else:
        raise CampaignError(f"unknown profile {arm.profile!r}")

    if set(_get(config, "compile.components")) != {"model", "loss"}:
        raise CampaignError(f"{arm.profile} must compile model and loss")
    return {"arm": arm.name, "profile": arm.profile, "status": "passed"}


def validate_apgt_source(torchtitan_root: Path) -> dict[str, Any]:
    graph_root = torchtitan_root / "torchtitan/experiments/graph_trainer"
    passes_path = graph_root / "passes.py"
    api_path = graph_root / "autoparallel_api.py"
    trainer_path = graph_root / "trainer.py"
    if not all(path.is_file() for path in (passes_path, api_path, trainer_path)):
        raise CampaignError("TorchTitan source lacks GraphTrainer AutoParallel files")

    evidence = {
        passes_path: (
            "if config.compile.enable_autoparallel:",
            "joint_transformer_block_bucketing_reordering_pass",
            "_autoparallel_inductor_configs",
            "full_inductor_configs=full_inductor_configs",
        ),
        api_path: (
            "aten_distributed_optimizations.enable_overlap_scheduling",
            "aten_distributed_optimizations.collective_bucketing",
            "aten_autobucketing_reordering_pass",
            "autoparallel_manages_context_parallel_input",
        ),
        trainer_path: (
            "autoparallel_manages_context_parallel_input",
            "dist_utils.set_pg_timeouts",
        ),
    }
    missing = [
        token
        for path, tokens in evidence.items()
        for token in tokens
        if token not in path.read_text()
    ]
    if missing:
        raise CampaignError(
            "source does not satisfy apgt_validated_v1; missing evidence: " f"{missing}"
        )
    return {
        "status": "passed",
        "passes_path": str(passes_path.resolve()),
        "autoparallel_api_path": str(api_path.resolve()),
        "trainer_path": str(trainer_path.resolve()),
    }


def validate_parameter_axis_constraint_source(
    autoparallel_root: Path,
) -> dict[str, Any]:
    api_path = autoparallel_root / "autoparallel/api.py"
    optimizer_path = autoparallel_root / "autoparallel/optimize_sharding.py"
    required = {
        api_path: "def add_parameter_axis_constraint",
        optimizer_path: "def add_parameter_axis_constraint",
    }
    missing = [
        str(path)
        for path, token in required.items()
        if not path.is_file() or token not in path.read_text()
    ]
    if missing:
        raise CampaignError(
            "HSDP AutoParallel requires parameter-axis constraints: " f"{missing}"
        )
    return {
        "status": "passed",
        "api_path": str(api_path.resolve()),
        "optimizer_path": str(optimizer_path.resolve()),
    }
