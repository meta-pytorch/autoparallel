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


GRAPH_TRAINER_COMPILE = (
    "torchtitan.experiments.graph_trainer.configs.GraphTrainerCompileConfig"
)
GRAPH_TRAINER_FULL_INDUCTOR = {
    "compile._class": GRAPH_TRAINER_COMPILE,
    "compile.memory_policy": "eager",
    "compile.inductor_compilation": "full",
    "compile.numerics_changing_optim": False,
    "compile.enable_fsdp_ag_rs_overlap": False,
    "compile.enable_fsdp_dense_region_overlap": False,
    "compile.disable_passes": ["cuda_graph_pass"],
}


def validate_profile(arm: Arm, config: dict[str, Any]) -> dict[str, Any]:
    """Check the serialized TorchTitan config against the arm's profile."""
    root = str(_get(config, "_class"))
    graph_trainer = root.startswith("torchtitan.experiments.graph_trainer.")
    if arm.profile in {"tt_eager", "tt_compiled_loss"}:
        if graph_trainer:
            raise CampaignError(f"{arm.profile} cannot use a GraphTrainer config")
        if arm.profile == "tt_eager":
            _expect(config, {"compile": None})
        else:
            _expect(
                config,
                {
                    "compile._class": "torchtitan.config.configs.CompileConfig",
                    "compile.components": ["loss"],
                    "compile.backend": "inductor",
                },
            )
    elif arm.profile in {"gt", "apgt"}:
        if not graph_trainer:
            raise CampaignError(f"{arm.profile} requires a GraphTrainer config")
        _expect(config, GRAPH_TRAINER_FULL_INDUCTOR)
        _expect(config, {"compile.enable_autoparallel": arm.profile == "apgt"})
        if arm.profile == "apgt":
            _expect(config, {"compile.use_autoparallel_defaults": True})
    else:
        raise CampaignError(f"unknown profile {arm.profile!r}")

    _expect(
        config,
        {
            "activation_checkpoint._class": (
                "torchtitan.distributed.activation_checkpoint.SelectiveAC.Config"
            ),
            "training.disable_cuda_graphs": True,
        },
    )
    return {"arm": arm.name, "profile": arm.profile, "status": "passed"}


def validate_apgt_source(torchtitan_root: Path) -> dict[str, Any]:
    graph_root = torchtitan_root / "torchtitan/experiments/graph_trainer"
    passes_path = graph_root / "passes.py"
    api_path = graph_root / "autoparallel_api.py"
    configs_path = graph_root / "configs.py"
    parallelize_path = graph_root / "llama3/parallelize_autoparallel.py"
    paths = (passes_path, api_path, configs_path, parallelize_path)
    if not all(path.is_file() for path in paths):
        raise CampaignError("TorchTitan source lacks GraphTrainer AutoParallel files")

    evidence = {
        passes_path: (
            "if config.compile.enable_autoparallel:",
            "autobucketing_reordering_pass",
            "_autoparallel_inductor_configs",
            "full_inductor_configs=full_inductor_configs",
        ),
        api_path: (
            "aten_distributed_optimizations.enable_overlap_scheduling",
            "aten_distributed_optimizations.collective_bucketing",
            "aten_distributed_optimizations.custom_runtime_estimation",
            "aten_autobucketing_reordering_pass",
        ),
        configs_path: (
            "use_autoparallel_defaults",
            "autoparallel_solver",
        ),
        parallelize_path: (
            "LocalMapVarlenAttention.from_inner(",
            "SplitFeedForward.from_fused(",
            'collectives.all_to_all(x, None, None, "cp")',
            'autop.add_parameter_axis_constraint("dp_replicate", Replicate())',
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
            f"source does not satisfy the apgt profile; missing evidence: {missing}"
        )
    return {
        "status": "passed",
        **{f"{path.stem}_path": str(path.resolve()) for path in paths},
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
