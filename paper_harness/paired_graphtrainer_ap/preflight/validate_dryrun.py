#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path


MARKER = "=== SCHEDULER REQUEST ===\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dryrun", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("gate", "formal"), required=True)
    parser.add_argument("--nodes", type=int, choices=(2,), required=True)
    parser.add_argument("--definition", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    dryrun_path = args.dryrun.resolve()
    text = dryrun_path.read_text()
    stderr_path = dryrun_path.with_name("dryrun.stderr")
    diagnostic_text = text + (stderr_path.read_text() if stderr_path.is_file() else "")
    start = text.index(MARKER) + len(MARKER)
    definition, _ = json.JSONDecoder().raw_decode(text[start:])
    task_root = args.task_root.resolve()
    world_size = args.nodes * 8
    checks: dict[str, bool] = {}

    def check(name: str, condition: bool) -> None:
        checks[name] = bool(condition)

    groups = definition.get("hpcTaskGroups", [])
    check("one_task_group", len(groups) == 1)
    group = groups[0] if groups else {}
    spec = group.get("spec", {})
    arguments = spec.get("arguments", [])
    packages = [
        package.get("fbpkgIdentifier")
        for package in spec.get("applicationPackages", [])
    ]
    env = spec.get("env", {})
    ports = spec.get("ports", {})

    expected_config = task_root / "launcher" / ".torchxconfig"
    check(
        "loaded_task_torchxconfig",
        (
            f"loaded configs from {expected_config}" in diagnostic_text
            or f"loaded configs from `{expected_config}`" in diagnostic_text
        ),
    )
    check("mast_scheduler", '"torchx/scheduler": "mast_conda"' in text)
    check("no_local_scheduler", "local_cwd" not in text)
    check(
        "no_unknown_scheduler_options", "unknown scheduler options" not in text.lower()
    )
    check("host_count", group.get("taskCount") == args.nodes)
    check("one_task_per_host", group.get("taskCountPerHost") == 1)
    check(
        "eight_gpus_per_host",
        spec.get("resourceLimit", {}).get("compute", {}).get("gpu") == 8,
    )
    check(
        "h100_hbm3_roce",
        spec.get("machineConstraints", {}).get("types", {}).get("serverSubTypes")
        == ["LogicalServerSubType.T20_GRAND_TETON_HBM3_ROCE"],
    )
    check("pci1", definition.get("localityConstraints", {}).get("options") == ["pci1"])
    check(
        "torchrun_nodes",
        "--nnodes" in arguments
        and arguments[arguments.index("--nnodes") + 1] == str(args.nodes),
    )
    check(
        "eight_processes_per_host",
        "--nproc-per-node" in arguments
        and arguments[arguments.index("--nproc-per-node") + 1] == "8",
    )
    check(
        "workspace_bootstrap_argument",
        "--no-python" in arguments
        and arguments[arguments.index("--no-python") + 1]
        == "/packages/torchtitan_workspace/run_rank.sh",
    )
    expected_tail = [
        args.mode,
        str(world_size),
        "2",
        "two_arm",
        "sequential",
        "1",
    ]
    check("runner_arguments", arguments[-6:] == expected_tail)
    check("zero_retries", spec.get("restartPolicy", {}).get("maxTotalFailures") == 0)
    check("ttls", spec.get("ttlsConfig", {}).get("enable") is True)
    check("conda_902", "torchtitan_conda_prod:902" in packages)
    check(
        "one_additional_package",
        sum(
            str(package).startswith("torchtitan_additional_packages:")
            for package in packages
        )
        == 1,
    )
    phase_count = 2 if args.mode == "gate" else 4
    expected_phase_ports = {
        f"training_phase_{index}" for index in range(2, phase_count + 1)
    }
    check("phase_ports", set(ports) == expected_phase_ports)
    check("root_user", spec.get("unixUser") == "root")
    check("no_job_retries", definition.get("maxJobFailures") == 0)
    check("module", env.get("MODULE") == "graph_trainer.muse_glimmer")
    check(
        "config",
        env.get("CONFIG") == "graph_trainer_muse_glimmer_30b_sdpa_c4_autoparallel_4x2",
    )
    check(
        "pythonpath_order",
        env.get("TORCHX_RUN_PYTHONPATH")
        == "/packages/torchtitan_additional_packages/autoparallel:/packages/torchtitan_additional_packages/torchtitan",
    )
    check(
        "structured_logger",
        env.get("TITAN_STRUCT_LOGGER_HANDLERS")
        == "torchtitan.observability.structured_logger.jsonl_handler.register_jsonl_handler",
    )
    check("compile_threads", env.get("TORCHINDUCTOR_COMPILE_THREADS") == "8")

    wrapper = task_root / "launcher" / "run_rank.sh"
    runner = task_root / "runner" / "run_rank.sh"
    check("workspace_bootstrap_exists", wrapper.is_file())
    check("workspace_bootstrap_executable", os.access(wrapper, os.X_OK))
    check("runner_exists", runner.is_file())
    check("runner_executable", os.access(runner, os.X_OK))
    runner_text = runner.read_text()
    check("official_entrypoint", "python -m torchtitan.train" in runner_text)
    check(
        "performance_phases",
        all(
            name in runner_text
            for name in (
                "01_per_gpu_bs1_graphtrainer",
                "02_per_gpu_bs1_graphtrainer_ap",
            )
        ),
    )
    check(
        "trace_phases",
        all(
            name in runner_text
            for name in (
                "11_trace_per_gpu_bs1_graphtrainer",
                "12_trace_per_gpu_bs1_graphtrainer_ap",
            )
        ),
    )
    check("gate_two_steps", "--training.steps 2" in runner_text)
    check("formal_performance_25_steps", "--training.steps 25" in runner_text)
    check("formal_trace_six_steps", "--training.steps 6" in runner_text)
    check(
        "formal_trace_first_step_schedule",
        "--profiler.profile-freq 6" in runner_text
        and "--profiler.profiler-warmup 0" in runner_text
        and "--profiler.profiler-active 3" in runner_text
        and "--profiler.profiler-repeat 1" in runner_text,
    )
    check("checkpoint_disabled", "--checkpoint.no-enable" in runner_text)
    check(
        "local_batch_size_two",
        "local_batch_sizes=(2 2)" in runner_text
        and '--training.local-batch-size "${local_batch_size}"' in runner_text,
    )
    check("global_batch_derived", "--training.global-batch-size -1" in runner_text)
    check("sequence_length_4096", "--training.seq-len 4096" in runner_text)
    check(
        "dynamic_fsdp",
        '--parallelism.data-parallel-shard-degree "${fsdp_degree}"' in runner_text,
    )
    check(
        "fixed_tp2",
        '--parallelism.tensor-parallel-degree "${tp_degree}"' in runner_text,
    )
    check("no_pass_disable_override", "--compile.disable" not in runner_text)
    check(
        "rank0_torch_trace_only",
        "[[ ${RANK} == 0 && ${phase_kind} == trace ]]" in runner_text
        and 'phase_env+=(TORCH_TRACE="${output}/compile_trace")' in runner_text,
    )
    check(
        "offline_c4_cache",
        "export HF_HUB_CACHE=${c4_hub_cache}" in runner_text
        and "export HF_HUB_OFFLINE=1" in runner_text
        and runner_text.index("export HF_HUB_CACHE=${c4_hub_cache}")
        < runner_text.index('python "${runner_root}/runtime_preflight.py"'),
    )
    check(
        "input_manifests_present",
        all(
            (
                task_root
                / "runner"
                / f"c4_input_manifest_lb{local_batch}_s4096_dp8.jsonl"
            ).is_file()
            for local_batch in (2,)
        ),
    )
    check(
        "oom_continuation",
        "rank_${RANK}.oom_evidence" in runner_text
        and "${RUN_ROOT}/runtime/${phase}.oom" in runner_text
        and "unexpected_failure=0" in runner_text
        and runner_text.index("rank_${RANK}.oom_evidence")
        < runner_text.index("rank_${RANK}.exit_code.tmp"),
    )
    check(
        "attempt_isolated_output",
        "MAST_HPC_JOB_VERSION is required" in runner_text
        and "MAST_HPC_JOB_ATTEMPT_INDEX is required" in runner_text
        and "_v${MAST_HPC_JOB_VERSION}_a${MAST_HPC_JOB_ATTEMPT_INDEX}" in runner_text,
    )
    check(
        "allocation_validation_fail_closed",
        'if ! python "${runner_root}/validate_allocation.py" "${phase}"; then'
        in runner_text
        and '${RUN_ROOT}/runtime/${phase}.failed' in runner_text,
    )

    failed = [name for name, passed in checks.items() if not passed]
    report = {
        "status": "passed" if not failed else "failed",
        "checks": checks,
        "failed": failed,
        "job_name": definition.get("name"),
        "packages": packages,
        "mode": args.mode,
        "nodes": args.nodes,
        "world_size": world_size,
    }
    args.definition.write_text(json.dumps(definition, indent=2, sort_keys=True) + "\n")
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"checks": len(checks), "failed": failed, "status": report["status"]}
        )
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
