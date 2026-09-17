from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

from .campaign import PLANNER_RESULT_CLASSIFICATION, load_campaign, write_json
from .parity import IGNORED_PATHS, validate_pair
from .profiles import validate_profile
from .validation import _probe_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


def _expand(value: str, *, payload: Path, output: Path) -> str:
    value = value.replace("{output}", str(output))
    value = value.replace("{harness}", str(payload / "harness_repo"))
    while "{asset:" in value:
        start = value.index("{asset:")
        end = value.index("}", start)
        name = value[start + len("{asset:") : end]
        value = value[:start] + str(payload / "assets" / name) + value[end + 1 :]
    return value


def _autoparallel_root(payload: Path, environment: dict[str, str]) -> Path:
    variant = environment.get("BENCHMARK_AP_SOURCE_VARIANT", "candidate")
    if variant == "candidate":
        root = payload / "autoparallel"
    elif variant == "baseline":
        root = payload / "autoparallel_baseline"
    else:
        raise RuntimeError(f"invalid AutoParallel source variant {variant!r}")
    if not root.is_dir():
        raise RuntimeError(
            f"AutoParallel source variant {variant!r} is not packaged: {root}"
        )
    return root


def _arm_pythonpath(payload: Path, autoparallel_root: Path) -> str:
    return os.pathsep.join(
        [
            str(payload / "harness_repo"),
            str(autoparallel_root),
            str(payload / "torchtitan"),
            os.environ.get("TORCHX_RUN_PYTHONPATH", ""),
        ]
    )


def _set_boolean_option(argv: list[str], path: str, enabled: bool) -> list[str]:
    positive = f"--{path.replace('_', '-')}"
    section, name = positive.rsplit(".", 1)
    negative = f"{section}.no-{name}"
    return [token for token in argv if token not in {positive, negative}] + [
        positive if enabled else negative
    ]


def _tee_run(command: list[str], *, cwd: Path, env: dict[str, str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        try:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
            )
        except OSError as error:
            stream.write(f"failed to start command: {error}\n")
            return 127
        assert process.stdout is not None
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            print(line, end="", flush=True)
        return process.wait()


def _wait_for_ranks(runtime: Path, world_size: int, timeout_seconds: int) -> list[int]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        paths = sorted(runtime.glob("rank_*.exit_code"))
        if len(paths) == world_size:
            return [int(path.read_text().strip()) for path in paths]
        time.sleep(1)
    raise RuntimeError(
        f"timed out waiting for {world_size} rank exit files under {runtime}"
    )


def _wait_for_phase(root: Path, timeout_seconds: int) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if (root / "completed").is_file():
            return True
        if (root / "failed").is_file():
            return False
        time.sleep(1)
    raise RuntimeError(f"timed out waiting for phase result under {root}")


def _verify_data(payload: Path, data: dict) -> None:
    manifest = data.get("identity_manifest")
    expected_hash = data.get("identity_manifest_sha256")
    if not manifest or not expected_hash:
        return
    path = Path(_expand(str(manifest), payload=payload, output=payload))
    if not path.is_file() or _sha256(path) != expected_hash:
        raise RuntimeError(f"input identity manifest is missing or mismatched: {path}")


def _audit_inputs(
    payload: Path,
    resolved: dict,
    run_root: Path,
    *,
    environment: dict[str, str],
    rank: int,
    world_size: int,
    timeout_seconds: int,
) -> None:
    module_name = resolved.get("data", {}).get("preflight_auditor")
    if not module_name:
        return
    root = run_root / "runtime/input_preflight"
    root.mkdir(parents=True, exist_ok=True)
    try:
        os.environ.update(environment)
        module = importlib.import_module(module_name)
        result = module.audit(
            payload=payload,
            resolved=resolved,
            output=root,
            environment=environment,
            rank=rank,
        )
        if not isinstance(result, dict) or result.get("status") not in {
            "passed",
            "skipped",
        }:
            raise RuntimeError(f"invalid input-auditor result from {module_name}")
        write_json(root / f"rank_{rank:05d}.json", result)
        status = 0
    except Exception as error:
        write_json(
            root / f"rank_{rank:05d}.json",
            {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            },
        )
        status = 1
    _atomic_text(root / f"rank_{rank:05d}.exit_code", f"{status}\n")
    statuses = _wait_for_ranks(root, world_size, timeout_seconds)
    if rank == 0:
        reports = [
            json.loads((root / f"rank_{item:05d}.json").read_text())
            for item in range(world_size)
        ]
        passed = all(code == 0 for code in statuses)
        write_json(
            root / "report.json",
            {
                "status": "passed" if passed else "failed",
                "auditor": module_name,
                "ranks": reports,
            },
        )
        _atomic_text(root / ("completed" if passed else "failed"), "\n")
    if not _wait_for_phase(root, timeout_seconds):
        raise RuntimeError("input preflight failed")


def _audit_runtime_configs(
    payload: Path,
    resolved: dict,
    run_root: Path,
    base_env: dict[str, str],
) -> None:
    campaign = load_campaign(
        payload / "campaign/campaign.toml",
        point=resolved.get("selected_point"),
        mode=resolved.get("execution_mode", "formal"),
    )
    output_root = run_root / "runtime/configs"
    if campaign.is_planner:
        profile_script = payload / "autoparallel/tests/search_profile.py"
        command = [sys.executable, str(profile_script), "--help"]
        completed = subprocess.run(
            command,
            cwd=payload / "autoparallel",
            env=base_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        expected_runs = campaign.planner_runs()
        checks = {
            "profile_script_exists": profile_script.is_file(),
            "profile_cli_help": completed.returncode == 0,
            "profile_cli_source_lock": "--source-lock" in completed.stdout,
            "resolved_runs_match": resolved.get("resolved_planner_runs")
            == expected_runs,
            "planner_only_classification": resolved.get("result_classification")
            == PLANNER_RESULT_CLASSIFICATION,
        }
        report = {
            "status": "passed" if all(checks.values()) else "failed",
            "campaign_type": "planner",
            "classification": PLANNER_RESULT_CLASSIFICATION,
            "checks": checks,
            "command": command,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "runs": expected_runs,
        }
        write_json(output_root / "report.json", report)
        if report["status"] != "passed":
            raise RuntimeError("packaged planner CLI contract validation failed")
        return

    checks = []
    configs_by_run = {}
    for phase in campaign.phases:
        phase_configs = {}
        for arm_name in phase.arms:
            arm = campaign.arm(arm_name)
            output = run_root / phase.name / arm_name
            argv = [
                _expand(token, payload=payload, output=output)
                for token in campaign.phase_arm_args(phase, arm)
            ]
            extra_env = dict(base_env)
            extra_env.update(
                {
                    "BENCHMARK_PHASE": phase.kind,
                    "BENCHMARK_ARM": arm_name,
                    "BENCHMARK_OUTPUT_DIR": str(output),
                    "BENCHMARK_SOURCE_LOCK_SHA256": _sha256(
                        payload / "campaign/source_lock.json"
                    ),
                }
            )
            extra_env.update(
                {
                    key: _expand(value, payload=payload, output=output)
                    for key, value in campaign.phase_arm_environment(phase, arm).items()
                }
            )
            arm_autoparallel_root = _autoparallel_root(payload, extra_env)
            extra_env["PYTHONPATH"] = _arm_pythonpath(payload, arm_autoparallel_root)
            extra_env["EXPECTED_AP_ROOT"] = str(arm_autoparallel_root)
            path = output_root / phase.name / f"{arm_name}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            config = _probe_config(
                argv,
                torchtitan_root=payload / "torchtitan",
                autoparallel_root=arm_autoparallel_root,
                output=path,
                python=Path(sys.executable),
                extra_env=extra_env,
            )
            checks.append(validate_profile(arm, config))
            expected_path = (
                payload
                / "campaign/serialized_configs"
                / phase.name
                / f"{arm_name}.json"
            )
            expected = json.loads(expected_path.read_text())
            package_check = validate_pair(
                "pre_package",
                expected,
                "runtime_package",
                config,
                [],
                ignored_paths=IGNORED_PATHS
                | {
                    "hf_assets_path",
                    "checkpoint.initial_load_path",
                    "compile.autoparallel_placements_load_path",
                    "compile.autoparallel_placements_save_path",
                },
            )
            package_check["phase"] = phase.name
            package_check["arm"] = arm_name
            package_check["kind"] = "pre_package_vs_runtime"
            checks.append(package_check)
            phase_configs[arm_name] = config
            configs_by_run[f"{phase.name}/{arm_name}"] = config
        comparison = campaign.raw.get("comparison", {})
        pairs = comparison.get("pairs", [])
        for baseline, treatment in pairs:
            if baseline in phase_configs and treatment in phase_configs:
                check = validate_pair(
                    baseline,
                    phase_configs[baseline],
                    treatment,
                    phase_configs[treatment],
                    list(comparison["allowed_config_paths"]),
                )
                check["phase"] = phase.name
                checks.append(check)
    comparison = campaign.raw.get("comparison", {})
    phase_by_arm = comparison.get("performance_phase_by_arm", {})
    for baseline, treatment in comparison.get("pairs", []):
        if not phase_by_arm:
            break
        left_key = f"{phase_by_arm[baseline]}/{baseline}"
        right_key = f"{phase_by_arm[treatment]}/{treatment}"
        phase_label = f"{phase_by_arm[baseline]} -> {phase_by_arm[treatment]}"
        check = validate_pair(
            baseline,
            configs_by_run[left_key],
            treatment,
            configs_by_run[right_key],
            list(comparison["allowed_config_paths"]),
        )
        check["phase"] = phase_label
        checks.append(check)
    write_json(output_root / "report.json", {"status": "passed", "checks": checks})


def _planner_command(
    payload: Path,
    resolved: dict,
    run: dict,
    output: Path,
) -> list[str]:
    source_lock_path = payload / "campaign/source_lock.json"
    source_lock = json.loads(source_lock_path.read_text())
    command = [
        "timeout",
        "--signal=TERM",
        "--kill-after=60s",
        f"{int(resolved['mast'].get('phase_timeout_seconds', 21600))}s",
        sys.executable,
        str(payload / "autoparallel/tests/search_profile.py"),
        "--model",
        str(resolved["planner"]["model"]),
        "--mesh",
        ",".join(str(value) for value in run["mesh"]),
        "--solver",
        str(run["solver"]),
        "--revision-label",
        str(source_lock["autoparallel"]["head"]),
        "--source-lock",
        str(source_lock_path),
        "--detailed-solution",
        "--output",
        str(output),
    ]
    if run["lazy_costs"] is not None:
        command.extend(("--lazy-costs", str(run["lazy_costs"]).lower()))
    if run["seeded"]:
        command.append("--seeded")
    return command


def _run_planner_campaign(
    payload: Path,
    resolved: dict,
    run_root: Path,
    base_env: dict[str, str],
) -> int:
    if int(os.environ["RANK"]) != 0:
        raise RuntimeError("only rank zero may execute planner subprocesses")
    planner_root = run_root / "planner"
    write_json(
        planner_root / "metadata.json",
        {
            "campaign_type": "planner",
            "classification": PLANNER_RESULT_CLASSIFICATION,
            "training_performance_evidence": False,
            "runs": resolved["resolved_planner_runs"],
        },
    )
    for run in resolved["resolved_planner_runs"]:
        output_root = (
            planner_root
            / run["point"]
            / f"repeat_{int(run['repeat']):02d}"
            / run["solver"]
        )
        result_path = output_root / "result.json"
        command = _planner_command(payload, resolved, run, result_path)
        _atomic_text(output_root / "started", "started\n")
        write_json(output_root / "command.json", command)
        status = _tee_run(
            command,
            cwd=payload / "autoparallel",
            env=base_env,
            log=output_root / "stdout_stderr.log",
        )
        _atomic_text(output_root / "exit_code", f"{status}\n")
        try:
            result = (
                json.loads(result_path.read_text()) if result_path.is_file() else {}
            )
        except Exception as error:
            result = {"status": "invalid", "error": str(error)}
        if status or result.get("status") != "success":
            _atomic_text(output_root / "failed", f"{status}\n")
            _atomic_text(planner_root / "failed", f"{run['point']} {run['solver']}\n")
            return status or 1
        _atomic_text(output_root / "completed", "completed\n")
    _atomic_text(planner_root / "completed", "completed\n")
    return 0


def main() -> None:
    payload = Path(os.environ["HARNESS_PAYLOAD_ROOT"]).resolve()
    resolved_path = payload / "campaign/resolved_campaign.json"
    resolved = json.loads(resolved_path.read_text())
    source_lock_sha256 = _sha256(payload / "campaign/source_lock.json")
    run_root = Path(os.environ["DUMP_DIR"]) / "run"
    run_root.mkdir(parents=True, exist_ok=True)
    _verify_data(payload, resolved.get("data", {}))

    mast = resolved["mast"]
    world_size = int(resolved["world_size"])
    if int(os.environ["WORLD_SIZE"]) != world_size:
        raise RuntimeError("runtime WORLD_SIZE differs from resolved campaign")
    rank = int(os.environ["RANK"])
    timeout = int(mast.get("phase_timeout_seconds", 7200))
    base_port = int(mast.get("master_port", 29500))
    torchtitan_root = payload / "torchtitan"
    autoparallel_root = payload / "autoparallel"
    harness_root = payload / "harness_repo"

    base_env = dict(os.environ)
    base_env.update(
        {
            "PYTHONPATH": os.pathsep.join(
                [
                    str(harness_root),
                    str(autoparallel_root),
                    str(torchtitan_root),
                    base_env.get("TORCHX_RUN_PYTHONPATH", ""),
                    base_env.get("PYTHONPATH", ""),
                ]
            ),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONUNBUFFERED": "1",
            "TORCH_DISABLE_ADDR2LINE": "1",
            "TORCHELASTIC_USE_AGENT_STORE": "False",
        }
    )
    for key, value in list(base_env.items()):
        base_env[key] = _expand(value, payload=payload, output=run_root)

    _audit_inputs(
        payload,
        resolved,
        run_root,
        environment=base_env,
        rank=rank,
        world_size=world_size,
        timeout_seconds=timeout,
    )

    preflight_root = run_root / "runtime/preflight"
    if rank == 0:
        try:
            from .runtime_preflight import run as run_preflight

            run_preflight(payload, run_root)
            _audit_runtime_configs(payload, resolved, run_root, base_env)
            _atomic_text(preflight_root / "completed", "completed\n")
        except Exception:
            _atomic_text(preflight_root / "failed", "failed\n")
            raise
    elif not _wait_for_phase(preflight_root, timeout):
        raise RuntimeError("rank-zero package/import preflight failed")

    if resolved.get("campaign_type") == "planner":
        allocation = [
            sys.executable,
            "-m",
            "harness.runtime_allocation",
            "--phase",
            "planner",
            "--arm",
            "planner_profile",
            "--run-root",
            str(run_root),
            "--nodes",
            str(mast["nodes"]),
            "--world-size",
            str(world_size),
            "--nproc-per-node",
            str(mast["nproc_per_node"]),
            "--locality",
            mast["locality"],
            "--gpu-substring",
            mast.get("gpu_name_contains", "H100"),
        ]
        subprocess.run(allocation, cwd=autoparallel_root, env=base_env, check=True)
        planner_root = run_root / "planner"
        planner_timeout = timeout * len(resolved["resolved_planner_runs"])
        if rank == 0:
            try:
                status = _run_planner_campaign(payload, resolved, run_root, base_env)
            except Exception as error:
                write_json(
                    planner_root / "runtime_error.json",
                    {
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(),
                    },
                )
                _atomic_text(planner_root / "failed", "runtime exception\n")
                status = 1
        else:
            status = 0 if _wait_for_phase(planner_root, planner_timeout) else 1
        planner_runtime = planner_root / "runtime"
        _atomic_text(planner_runtime / f"rank_{rank:05d}.exit_code", f"{status}\n")
        statuses = _wait_for_ranks(planner_runtime, world_size, planner_timeout)
        if rank == 0:
            marker = (
                "all_ranks_completed"
                if all(code == 0 for code in statuses)
                else "all_ranks_failed"
            )
            _atomic_text(
                planner_root / marker,
                "\n".join(str(code) for code in statuses) + "\n",
            )
        if any(statuses):
            raise SystemExit(max(statuses))
        return

    phase_index = 0
    for phase in resolved["resolved_phases"]:
        for arm_name in phase["arms"]:
            output = run_root / phase["name"] / arm_name
            runtime = output / "runtime"
            runtime.mkdir(parents=True, exist_ok=True)
            allocation = [
                sys.executable,
                "-m",
                "harness.runtime_allocation",
                "--phase",
                phase["name"],
                "--arm",
                arm_name,
                "--run-root",
                str(run_root),
                "--nodes",
                str(mast["nodes"]),
                "--world-size",
                str(world_size),
                "--nproc-per-node",
                str(mast["nproc_per_node"]),
                "--locality",
                mast["locality"],
                "--gpu-substring",
                mast.get("gpu_name_contains", "H100"),
            ]
            subprocess.run(allocation, cwd=torchtitan_root, env=base_env, check=True)
            if rank == 0:
                _atomic_text(output / "started", "started\n")

            cache = Path(
                tempfile.mkdtemp(
                    prefix=f"permanent-harness-{phase['name']}-{arm_name}-{rank}-"
                )
            )
            env = dict(base_env)
            env.update(
                {
                    "MASTER_PORT": str(base_port + phase_index),
                    "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"),
                    "TRITON_CACHE_DIR": str(cache / "triton"),
                    "TMPDIR": str(cache / "tmp"),
                    "HARNESS_PHASE": phase["name"],
                    "HARNESS_ARM": arm_name,
                    "HARNESS_OUTPUT": str(output),
                    "BENCHMARK_PHASE": phase["kind"],
                    "BENCHMARK_ARM": arm_name,
                    "BENCHMARK_OUTPUT_DIR": str(output),
                    "BENCHMARK_SOURCE_LOCK_SHA256": source_lock_sha256,
                    "RUN_ROOT": str(output),
                    "INPUT_AUDIT_DIR": str(output / "input_audit"),
                    "MODULE_ISOLATION_AUDIT_DIR": str(
                        output / "module_isolation_audit"
                    ),
                    "PARAMETER_AUDIT_DIR": str(output / "parameter_audit"),
                    "PLACEMENT_AUDIT_DIR": str(output / "placement_audit"),
                    "AP_COLLECTIVE_HOOK_AUDIT_DIR": str(
                        output / "collective_hook_audit"
                    ),
                    "EXPECTED_TT_ROOT": str(torchtitan_root),
                    "EXPECTED_AP_ROOT": str(autoparallel_root),
                }
            )
            env.update(
                {
                    key: _expand(value, payload=payload, output=output)
                    for key, value in phase["environment_by_arm"][arm_name].items()
                }
            )
            arm_autoparallel_root = _autoparallel_root(payload, env)
            env["PYTHONPATH"] = _arm_pythonpath(payload, arm_autoparallel_root)
            env["EXPECTED_AP_ROOT"] = str(arm_autoparallel_root)
            for directory in (cache / "inductor", cache / "triton", cache / "tmp"):
                directory.mkdir(parents=True)
            if (
                phase["kind"] in {"trace", "torch_trace"}
                and rank in phase["trace_ranks"]
            ):
                trace_root = output / "torch_trace" / f"rank_{rank:05d}"
                trace_root.mkdir(parents=True, exist_ok=True)
                env["TORCH_TRACE"] = str(trace_root)
            else:
                env.pop("TORCH_TRACE", None)

            argv = [
                _expand(token, payload=payload, output=output)
                for token in phase["argv_by_arm"][arm_name]
            ]
            if phase["kind"] in {"trace", "kineto"}:
                argv = _set_boolean_option(
                    argv,
                    "profiler.enable-profiling",
                    rank in phase["trace_ranks"],
                )
            if not any(token in {"--dump-folder", "--dump_folder"} for token in argv):
                argv.extend(("--dump-folder", str(output / "job")))
            command = [
                "timeout",
                "--signal=TERM",
                "--kill-after=60s",
                f"{timeout}s",
                sys.executable,
                "-m",
                "torchtitan.train",
                *argv,
            ]
            try:
                status = _tee_run(
                    command,
                    cwd=torchtitan_root,
                    env=env,
                    log=runtime / f"rank_{rank:05d}.log",
                )
            finally:
                shutil.rmtree(cache, ignore_errors=True)
            _atomic_text(runtime / f"rank_{rank:05d}.exit_code", f"{status}\n")
            statuses = _wait_for_ranks(runtime, world_size, timeout)
            if rank == 0:
                _atomic_text(
                    output
                    / (
                        "completed" if all(code == 0 for code in statuses) else "failed"
                    ),
                    "\n".join(str(code) for code in statuses) + "\n",
                )
            if not _wait_for_phase(output, timeout):
                raise SystemExit(max(statuses) or 1)
            phase_index += 1


if __name__ == "__main__":
    main()
