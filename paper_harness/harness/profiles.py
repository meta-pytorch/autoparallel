from __future__ import annotations

from pathlib import Path
from typing import Any

from .campaign import Arm, CampaignError
from .parity import validate_pair


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
    if arm.profile == "tt_main_manual_jit_v1":
        model_spec_name = str(_get(config, "model_spec.name"))
        if "graphtrainer" in model_spec_name.replace("_", "").lower():
            raise CampaignError(
                "tt_main_manual_jit_v1 cannot use a GraphTrainer model spec"
            )
        _expect(
            config,
            {
                "compile.enable": True,
                "compile.backend": "inductor",
            },
        )
        if set(_get(config, "compile.components")) != {"model", "loss"}:
            raise CampaignError("tt_main_manual_jit_v1 must compile model and loss")
    elif arm.profile in {
        "gt_manual_aot_v1",
        "apgt_v1",
        "gt_manual_cp_legacy_v1",
        "apgt_cp_legacy_v1",
        "apgt_3d_exact_mesh_flash_v1",
    }:
        legacy_cp = arm.profile in {"gt_manual_cp_legacy_v1", "apgt_cp_legacy_v1"}
        model_spec_name = _get(config, "model_spec.name")
        if not legacy_cp and not (
            arm.module.startswith("graph_trainer.")
            or "graphtrainer" in str(model_spec_name).replace("_", "").lower()
        ):
            raise CampaignError(
                f"{arm.profile} requires a GraphTrainer model spec, got "
                f"{model_spec_name!r}"
            )
        expected_ap = arm.profile in {
            "apgt_v1",
            "apgt_cp_legacy_v1",
            "apgt_3d_exact_mesh_flash_v1",
        }
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
                "compile.enable_autoparallel": expected_ap,
            },
        )
        if set(_get(config, "compile.components")) != {"model", "loss"}:
            raise CampaignError(f"{arm.profile} must compile model and loss")
        disabled = set(_get(config, "compile.disable_passes"))
        if "cudagraph_pass" not in disabled:
            raise CampaignError(f"{arm.profile} requires cudagraph_pass to be disabled")
        joint = "joint_transformer_block_bucketing_reordering_pass"
        if legacy_cp and joint not in disabled:
            raise CampaignError(f"{arm.profile} requires the historical joint-pass disable")
        if not legacy_cp and joint in disabled:
            raise CampaignError(
                f"{arm.profile} must not disable the joint pass in serialized config; "
                "the source pass builder selects manual joint versus AP scheduling"
            )
        if not legacy_cp and disabled != {"cudagraph_pass"}:
            raise CampaignError(
                f"{arm.profile} requires only cudagraph_pass disabled, got {sorted(disabled)}"
            )
        if arm.profile == "apgt_3d_exact_mesh_flash_v1" and not arm.module.startswith(
            "workloads.llama3_3d_current."
        ):
            raise CampaignError(
                "apgt_3d_exact_mesh_flash_v1 requires the fixed current-head "
                "LLaMA3 3D workload"
            )
    elif arm.profile == "ap_backend_legacy_v1":
        if "autoparallel" not in arm.config and "autoparallel" not in arm.module:
            raise CampaignError("ap_backend_legacy_v1 must name an AutoParallel config/module")
    return {"arm": arm.name, "profile": arm.profile, "status": "passed"}


def validate_profile_pair(
    baseline_arm: Arm,
    baseline: dict[str, Any],
    treatment_arm: Arm,
    treatment: dict[str, Any],
) -> dict[str, Any] | None:
    if baseline_arm.profile == "gt_manual_aot_v1" and treatment_arm.profile in {
        "apgt_v1",
        "apgt_3d_exact_mesh_flash_v1",
    }:
        check = validate_pair(
            baseline_arm.name,
            baseline,
            treatment_arm.name,
            treatment,
            ["compile.enable_autoparallel"],
        )
        check["kind"] = "fixed_apgt_v1_pair_contract"
        return check
    return None


def validate_apgt_source(torchtitan_root: Path) -> dict[str, Any]:
    graph_root = torchtitan_root / "torchtitan/experiments/graph_trainer"
    passes_path = graph_root / "passes.py"
    api_path = graph_root / "autoparallel_api.py"
    trainer_path = graph_root / "trainer.py"
    if not passes_path.is_file() or not api_path.is_file() or not trainer_path.is_file():
        raise CampaignError("TorchTitan source lacks GraphTrainer AutoParallel files")
    passes = passes_path.read_text()
    api = api_path.read_text()
    trainer = trainer_path.read_text()
    pass_needles = (
        "if config.compile.enable_autoparallel:",
        "joint_transformer_block_bucketing_reordering_pass",
        "_autoparallel_inductor_configs",
        "full_inductor_configs=full_inductor_configs",
        "autoparallel_mesh=autoparallel_mesh",
        "AutoParallel full Inductor compilation requires its runtime mesh",
    )
    api_needles = (
        "_graph_trainer_autoparallel_mesh",
        "aten_distributed_optimizations.enable_overlap_scheduling",
        "aten_distributed_optimizations.collective_bucketing",
        "aten_distributed_optimizations.insert_overlap_deps",
        '"aten_distributed_optimizations.max_compute_pre_fetch": 10',
        '"reorder_for_peak_memory": False',
        '"reorder_for_compute_comm_overlap": False',
        '"post_grad_custom_post_pass"',
        "aten_autobucketing_reordering_pass",
    )
    missing = [item for item in pass_needles if item not in passes]
    missing.extend(item for item in api_needles if item not in api)
    trainer_needles = (
        "pipeline_fn is construct_default_graph_passes",
        "autoparallel_mesh=getattr(",
        'model, "_graph_trainer_autoparallel_mesh", None',
    )
    missing.extend(item for item in trainer_needles if item not in trainer)
    if missing:
        raise CampaignError(
            "source does not satisfy apgt_v1; user gate required before changing "
            f"the contract. Missing evidence: {missing}"
        )
    return {
        "status": "passed",
        "passes_path": str(passes_path.resolve()),
        "autoparallel_api_path": str(api_path.resolve()),
        "trainer_path": str(trainer_path.resolve()),
        "manual_joint_pass": "enabled_by_enable_autoparallel_false_branch",
        "ap_joint_pass": "absent_by_enable_autoparallel_true_branch",
        "ap_full_inductor_configs": "verified",
    }
