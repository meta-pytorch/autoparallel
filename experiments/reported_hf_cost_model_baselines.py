#!/usr/bin/env python3

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_SCRIPTS = REPO_ROOT / ".agents/skills/autoparallel-optimizer/scripts"
sys.path[:0] = [str(REPO_ROOT), str(SKILL_SCRIPTS)]

from modeled_baselines import (  # noqa: E402
    autoparallel_estimate,
    calibrate_relative_baseline,
    ddp_estimate,
    fsdp_estimate,
    load_baseline_cost_calibration,
    load_optimizer_cost_calibration,
)
from parallelization_policy import (  # noqa: E402
    classify_modeled_competitiveness,
    default_acceptance_criteria,
    default_attempt_budget,
)
from report_session import AutoParallelReportSession  # noqa: E402

from autoparallel.api import AutoParallel  # noqa: E402
from autoparallel.cost_models.nccl_cost_model import (  # noqa: E402
    GpuArch,
    NCCLTopoConfig,
)
from experiments.hf_benchmark_models import make_inputs, make_model  # noqa: E402
from experiments.hf_meta_sweep_models import SWEEP_CASES  # noqa: E402

PILOT_CAMPAIGN_ID = "huggingface-cost-model-baselines-v1"
CALIBRATION_CAMPAIGN_ID = "huggingface-cost-model-baseline-calibration-v1"
PILOT_CASES = (
    "gpt2-medium-seq1024",
    "t5-base-seq512",
    "llama3.2-1b-seq2048",
)
CALIBRATION_CASES = (
    "gpt2-xl-seq1024",
    "t5-3b-seq512",
    "llama3-8b-seq2048",
    "mistral-7b-seq2048",
    "deberta-v2-xxlarge-seq512",
    "qwen2-7b-seq2048",
    "bert-large-seq512",
    "t5-11b-seq512",
    "llama3.2-1b-seq2048",
    "llama3.2-3b-seq2048",
    "t5-large-seq512",
    "bloom-7.1b-seq1024",
    "gpt2-medium-seq1024",
    "t5-base-seq512",
    "qwen2-1.5b-seq2048",
    "bloom-560m-seq1024",
)
WORLD_SIZE = 8
DEVICE_CAPACITY_BYTES = 85_070_544_896
USABLE_DEVICE_BYTES = round(DEVICE_CAPACITY_BYTES * 0.95)
OPTIMIZER_CALIBRATION_PATH = (
    REPO_ROOT
    / ".agents/skills/autoparallel-optimizer/references"
    / "optimizer-cost-calibration-v1.json"
)
BASELINE_CALIBRATION_PATH = (
    REPO_ROOT
    / ".agents/skills/autoparallel-optimizer/references"
    / "baseline-cost-calibration-v2.json"
)


def configuration_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def unique_parameter_sizes(model):
    return [
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
        if parameter.requires_grad
    ]


def fsdp_wrap_unit_sizes(model, block_cls):
    claimed = set()
    units = []
    unit_names = []
    for name, module in model.named_modules():
        if not isinstance(module, block_cls):
            continue
        parameters = [
            parameter
            for parameter in module.parameters()
            if parameter.requires_grad and id(parameter) not in claimed
        ]
        if not parameters:
            continue
        units.append(
            sum(
                parameter.numel() * parameter.element_size() for parameter in parameters
            )
        )
        unit_names.append(name)
        claimed.update(id(parameter) for parameter in parameters)

    root_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in claimed
    ]
    if root_parameters:
        units.append(
            sum(
                parameter.numel() * parameter.element_size()
                for parameter in root_parameters
            )
        )
        unit_names.append("<root-remainder>")
    return units, unit_names


def plan_summary(plan):
    summary = plan["summary"]
    return {
        key: summary[key]
        for key in (
            "total",
            "compute",
            "comm",
            "transition",
            "search_objective",
            "parameter_storage",
        )
    }


def with_template_metadata(estimate, template):
    estimate.update(
        {
            "template_id": template["template_id"],
            "template_version": template["template_version"],
            "configuration_hash": configuration_hash(template),
        }
    )
    return estimate


def run(case_name, campaign_id, expected_case_count):
    case = SWEEP_CASES[case_name]
    optimizer_calibration = load_optimizer_cost_calibration(OPTIMIZER_CALIBRATION_PATH)
    baseline_calibration = load_baseline_cost_calibration(BASELINE_CALIBRATION_PATH)
    optimizer_applicability = {
        "platform": "matched",
        "optimizer_contract": "matched",
        "realized_kernel_family": "unknown",
        "scale": "matched",
    }
    torch.distributed.init_process_group(
        "fake", store=FakeStore(), rank=0, world_size=WORLD_SIZE
    )
    try:
        mesh = torch.distributed.device_mesh.init_device_mesh(
            "cuda", (WORLD_SIZE,), mesh_dim_names=("dp",)
        )
        topology = NCCLTopoConfig(
            arch=GpuArch.HOPPER,
            num_nodes=1,
            gpus_per_node=WORLD_SIZE,
            bw_intra=320.0,
            bw_inter=50.0,
            num_channels=16,
            has_nvswitch=True,
            has_collnet=False,
            net_latency=0.0,
        )
        with torch.device("meta"):
            model = make_model(case)
        parameter_sizes = unique_parameter_sizes(model)
        wrap_unit_sizes, wrap_unit_names = fsdp_wrap_unit_sizes(model, case.block_cls)
        parameter_bytes = sum(parameter_sizes)
        if sum(wrap_unit_sizes) != parameter_bytes:
            raise RuntimeError("FSDP wrap units do not cover trainable parameters once")

        input_count = len(make_inputs(case, case.global_batch, "cuda"))
        input_placements = [(Shard(0),)] * input_count
        decision_id = f"{campaign_id}-{case.name}"
        request = {
            "workload": {
                "name": case.name,
                "framework": "transformers",
                "task": case.task,
                "mode": "cost_model_only_training_plan",
                "parameter_count": parameter_bytes // 4,
                "global_batch": case.global_batch,
                "modeled_local_batch": case.global_batch // WORLD_SIZE,
                "sequence_length": case.sequence_length,
                "loss": "transformers-native-cross-entropy",
                "step_boundary": "forward-loss-backward-no-optimizer",
                "activation_checkpointing": {"enabled": False},
            },
            "mesh": {"shape": [WORLD_SIZE], "names": ["dp"]},
            "constraints": {
                "inputs": ["Shard(0)"] * input_count,
                "outputs": ["Replicate"],
                "parameters": "unconstrained",
            },
            "decision_id": decision_id,
            "modeling_scope": {
                "kind": "meta_tensor_cost_model_with_native_baseline_templates",
                "target_execution": False,
                "use_for": "solution-quality-screening",
                "not_for": "deployment-acceptance-or-complete-step-memory-fit",
            },
            "optimizer_state_assumptions": {
                "optimizer": "AdamW",
                "persistent_parameter_dtype": "float32",
                "gradient_dtype": "float32",
                "moment_dtype": "float32",
                "optimizer_update_timed": False,
            },
        }
        parallelization = {
            "decision_id": decision_id,
            "intent": {
                "kind": "distributed_training",
                "requested_strategy": "automatic",
                "attempt_budget": default_attempt_budget(),
            },
            "acceptance_criteria": default_acceptance_criteria(),
        }
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
        with AutoParallelReportSession(
            mesh,
            request=request,
            campaign={
                "id": campaign_id,
                "suite": "huggingface-cost-model-baselines",
                "case_id": case.name,
                "expected_case_count": expected_case_count,
            },
            parallelization=parallelization,
            compiler_config={
                "lower": False,
                "compile": False,
                "baseline_model_version": "1",
            },
            source_root=REPO_ROOT,
            entry_point="experiments.reported_hf_cost_model_baselines:main",
            launch_config={
                "launcher": "python",
                "process_group": "fake",
                "processes": 1,
                "modeled_world_size": WORLD_SIZE,
            },
            include_optimizer_replay=True,
            mode="plan",
            rank=0,
            nccl_topology=topology,
        ) as session:
            if session.report is not None:
                session.report.add_artifact(
                    "reproducer",
                    Path(__file__),
                    role="reproducer",
                    media_type="text/x-python",
                )
                session.report.add_artifact(
                    "modeled-baselines",
                    SKILL_SCRIPTS / "modeled_baselines.py",
                    role="cost_model",
                    media_type="text/x-python",
                )
                session.report.add_artifact(
                    "optimizer-cost-calibration",
                    OPTIMIZER_CALIBRATION_PATH,
                    role="cost_model_calibration",
                    media_type="application/json",
                )
                session.report.add_artifact(
                    "baseline-cost-calibration",
                    BASELINE_CALIBRATION_PATH,
                    role="cost_model_calibration",
                    media_type="application/json",
                )
            session.mark_substage("capture")

            def input_fn():
                return make_inputs(case, case.global_batch, "cuda")

            with AutoParallel(
                model,
                input_fn,
                mesh,
                mp_policy=mp_policy,
                cost_model=session.cost_model,
                repeated_subgraphs=True,
            ) as autop:
                autop.add_input_constraints(input_placements)
                autop.add_output_constraints([(Replicate(),)])
                session.mark_substage("solve-autoparallel")
                autop.optimize_placement(verbose=False)
                optimizer = autop.sharding_optimizer
                selected_plan = optimizer.get_json()
                session.record_plan(optimizer)

                session.mark_substage("solve-replicated-proxy")
                optimizer.add_parameter_memory_constraint(1.0, 1.0)
                optimizer.resolve(verbose=False)
                replicated_plan = optimizer.get_json()

                session.mark_substage("solve-full-shard-proxy")
                optimizer.add_parameter_memory_constraint(
                    1.0 / WORLD_SIZE, 1.0 / WORLD_SIZE
                )
                optimizer.resolve(verbose=False)
                full_shard_plan = optimizer.get_json()

            selected_parameter_bytes = selected_plan["summary"]["parameter_storage"][
                "global_bytes"
            ]
            if selected_parameter_bytes != parameter_bytes:
                raise RuntimeError(
                    "model parameters disagree with captured persistent storage"
                )

            autoparallel = autoparallel_estimate(
                selected_plan,
                mesh_shape=(WORLD_SIZE,),
                usable_device_bytes=USABLE_DEVICE_BYTES,
                optimizer_calibration=optimizer_calibration,
                optimizer_applicability=optimizer_applicability,
            )
            ddp_template = {
                "template_id": "pytorch-compiled-ddp-cost-model",
                "template_version": "1",
                "parameters": "replicated",
                "gradient_reduction": "all-reduce",
                "persistent_parameter_dtype": "float32",
                "compute_dtype": "bfloat16",
                "gradient_reduction_dtype": "float32",
                "bucket_cap_mb": 25,
                "gradient_as_bucket_view": True,
                "static_graph": True,
            }
            ddp = with_template_metadata(
                ddp_estimate(
                    replicated_plan,
                    parameter_sizes,
                    topology,
                    (WORLD_SIZE,),
                    usable_device_bytes=USABLE_DEVICE_BYTES,
                    optimizer_calibration=optimizer_calibration,
                    optimizer_applicability=optimizer_applicability,
                ),
                ddp_template,
            )
            fsdp_template = {
                "template_id": ("pytorch-compiled-fsdp2-transformer-block-cost-model"),
                "template_version": "1",
                "sharding_mode": "full-shard",
                "wrapping_policy": case.block_cls.__name__,
                "wrapped_block_count": len(wrap_unit_names) - 1,
                "root_wrap": True,
                "reshard_after_forward": True,
                "persistent_parameter_dtype": "float32",
                "parameter_dtype": "bfloat16",
                "gradient_reduction_dtype": "float32",
            }
            fsdp = with_template_metadata(
                fsdp_estimate(
                    replicated_plan,
                    wrap_unit_sizes,
                    topology,
                    (WORLD_SIZE,),
                    usable_device_bytes=USABLE_DEVICE_BYTES,
                    optimizer_calibration=optimizer_calibration,
                    optimizer_applicability=optimizer_applicability,
                ),
                fsdp_template,
            )
            latency_verdict = classify_modeled_competitiveness(
                autoparallel["latency"],
                [
                    {"strategy": "ddp", "feasibility": "feasible", **ddp},
                    {"strategy": "fsdp", "feasibility": "feasible", **fsdp},
                ],
            )
            calibrated_ddp = calibrate_relative_baseline(
                autoparallel["latency"], ddp, parameter_bytes, baseline_calibration
            )
            calibrated_fsdp = calibrate_relative_baseline(
                autoparallel["latency"], fsdp, parameter_bytes, baseline_calibration
            )
            calibrated_latency_verdict = classify_modeled_competitiveness(
                autoparallel["latency"],
                [
                    {"strategy": "ddp", "feasibility": "feasible", **calibrated_ddp},
                    {
                        "strategy": "fsdp",
                        "feasibility": "feasible",
                        **calibrated_fsdp,
                    },
                ],
            )
            full_step_latencies = {
                "autoparallel": autoparallel["full_step_latency"],
                "ddp": ddp["full_step_latency"],
                "fsdp": fsdp["full_step_latency"],
            }
            full_step_screening_verdict = None
            if all(value is not None for value in full_step_latencies.values()):
                full_step_screening_verdict = classify_modeled_competitiveness(
                    full_step_latencies["autoparallel"],
                    [
                        {
                            "strategy": "ddp",
                            "feasibility": "feasible",
                            "latency": full_step_latencies["ddp"],
                        },
                        {
                            "strategy": "fsdp",
                            "feasibility": "feasible",
                            "latency": full_step_latencies["fsdp"],
                        },
                    ],
                )
            comparison = {
                "kind": "cost-model-only-native-baseline-comparison",
                "scope": "forward-loss-backward-no-optimizer",
                "evidence": "modeled-not-measured",
                "workload_contract": request["workload"],
                "mesh": request["mesh"],
                "topology": {
                    "accelerator": "H100",
                    "nodes": 1,
                    "gpus_per_node": WORLD_SIZE,
                    "nvswitch": True,
                },
                "autoparallel": autoparallel,
                "baselines": {"ddp": ddp, "fsdp": fsdp},
                "latency_only_verdict": latency_verdict,
                "calibrated_latency_screening": {
                    "verdict": calibrated_latency_verdict,
                    "baselines": {
                        "ddp": calibrated_ddp,
                        "fsdp": calibrated_fsdp,
                    },
                    "calibration_path": str(
                        BASELINE_CALIBRATION_PATH.relative_to(REPO_ROOT)
                    ),
                    "use": "screening_only",
                },
                "full_step_screening_verdict": full_step_screening_verdict,
                "optimizer_calibration": {
                    "path": str(OPTIMIZER_CALIBRATION_PATH.relative_to(REPO_ROOT)),
                    "kind": optimizer_calibration["kind"],
                    "applicability": optimizer_applicability,
                    "use": "screening_only",
                },
                "expected_latency_ratios": {
                    "autoparallel_to_ddp": (
                        autoparallel["latency"]["expected"] / ddp["latency"]["expected"]
                    ),
                    "autoparallel_to_fsdp": (
                        autoparallel["latency"]["expected"]
                        / fsdp["latency"]["expected"]
                    ),
                },
                "autoparallel_proxy_diagnostics": {
                    "replicated_parameter_plan": plan_summary(replicated_plan),
                    "full_shard_parameter_plan": plan_summary(full_shard_plan),
                    "labels": {
                        "replicated_parameter_plan": "not-native-ddp",
                        "full_shard_parameter_plan": "not-native-fsdp",
                    },
                },
                "fsdp_wrap_units": {
                    "names": wrap_unit_names,
                    "sizes_fp32_bytes": wrap_unit_sizes,
                },
                "limitations": [
                    "latency envelopes are structural cost-model estimates without current target execution",
                    "optimizer update estimates are screening-only until the realized kernel family is observed",
                    "activation liveness is an analytical upper bound without aliasing or allocator fragmentation",
                    "post-lowering reordered scheduling is not available from optimizer replay alone",
                    "FSDP wrapper runtime compatibility is not established by a meta-tensor run",
                ],
            }
            if session.report is not None:
                session.report.write_json_artifact(
                    "replicated-proxy-optimizer",
                    replicated_plan,
                    role="diagnostic_plan",
                )
                session.report.write_json_artifact(
                    "full-shard-proxy-optimizer",
                    full_shard_plan,
                    role="diagnostic_plan",
                )
                session.report.write_json_artifact(
                    "modeled-baseline-comparison",
                    comparison,
                    role="modeled_comparison",
                )
            session.add_finding(
                "MODELED_BASELINE_COMPARISON_REQUIRES_CALIBRATION",
                "Native DDP and FSDP costs are modeled for solution-quality screening only",
                context={
                    "latency_only_verdict": latency_verdict["verdict"],
                    "autoparallel_to_ddp_expected_ratio": comparison[
                        "expected_latency_ratios"
                    ]["autoparallel_to_ddp"],
                    "autoparallel_to_fsdp_expected_ratio": comparison[
                        "expected_latency_ratios"
                    ]["autoparallel_to_fsdp"],
                },
                suggested_action=(
                    "Validate the structural baseline model against existing target runs "
                    "before using modeled margins for routing"
                ),
            )
            session.record_competitiveness_assessment(
                "comparison_unavailable",
                evidence="modeled",
                reason_codes=("TARGET_MEASUREMENT_REQUIRED",),
                details={
                    "latency_only_modeled_verdict": latency_verdict["verdict"],
                    "calibrated_latency_screening_verdict": (
                        calibrated_latency_verdict["verdict"]
                    ),
                    "full_step_screening_verdict": (
                        full_step_screening_verdict["verdict"]
                        if full_step_screening_verdict is not None
                        else "unavailable"
                    ),
                    "complete_step_memory_fit": {
                        "autoparallel": autoparallel["memory"]["complete_step_fits"],
                        "ddp": ddp["memory"]["complete_step_fits"],
                        "fsdp": fsdp["memory"]["complete_step_fits"],
                    },
                    "missing_evidence": [
                        "post_lowering_reordered_schedule",
                        "observed_optimizer_kernel_family",
                        "target_execution",
                    ],
                },
            )
            session.record_autoparallel_outcome(
                "inconclusive", reason_codes=("TARGET_MEASUREMENT_REQUIRED",)
            )
            session.record_parallelization_decision(
                "none", "inconclusive", reason_codes=("TARGET_MEASUREMENT_REQUIRED",)
            )
        print(f"report={session.path}", flush=True)
    finally:
        torch.distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case", nargs="?", choices=tuple(SWEEP_CASES))
    parser.add_argument("--campaign", choices=("pilot", "calibration"), default="pilot")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    cases = PILOT_CASES if args.campaign == "pilot" else CALIBRATION_CASES
    campaign_id = (
        PILOT_CAMPAIGN_ID if args.campaign == "pilot" else CALIBRATION_CAMPAIGN_ID
    )
    if args.all:
        if args.case is not None:
            parser.error("case and --all are mutually exclusive")
        for case_name in cases:
            print(f"starting={case_name}", flush=True)
            run(case_name, campaign_id, len(cases))
    else:
        if args.case is None:
            parser.error("case is required unless --all is used")
        run(args.case, campaign_id, len(cases))


if __name__ == "__main__":
    main()
