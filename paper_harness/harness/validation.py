from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .assets import asset_lock_digest, load_asset_lock
from .campaign import (
    PLANNER_RESULT_CLASSIFICATION,
    Campaign,
    CampaignError,
    write_json,
)
from .experiment_lock import experiment_lock_digest, load_experiment_lock
from .integrity import validate_harness_integrity
from .parity import validate_pair
from .profiles import validate_apgt_source, validate_profile
from .sources import inspect_source, manifest_digest, tree_manifest


def _validate_phase_config(
    phase_kind: str, profile: str, config: dict[str, Any]
) -> None:
    profiler = config.get("profiler", {})
    if phase_kind == "performance" and (
        profiler.get("enable_profiling") or profiler.get("enable_memory_snapshot")
    ):
        raise CampaignError(
            "performance phases must disable Kineto and memory snapshots"
        )
    if phase_kind in {"trace", "kineto"} and not profiler.get("enable_profiling"):
        raise CampaignError("trace/kineto phases must enable the Torch profiler")


def _pythonpath(*paths: Path) -> str:
    values = [str(path.resolve()) for path in paths]
    inherited = os.environ.get("PYTHONPATH")
    if inherited:
        values.append(inherited)
    return os.pathsep.join(values)


def _probe_config(
    argv: list[str],
    *,
    torchtitan_root: Path,
    autoparallel_root: Path,
    output: Path,
    python: Path,
    extra_env: dict[str, str],
) -> dict[str, Any]:
    request = output.with_suffix(".request.json")
    write_json(request, {"argv": argv})
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": _pythonpath(repo_root, autoparallel_root, torchtitan_root),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    env.update(extra_env)
    result = subprocess.run(
        [
            str(python),
            "-m",
            "harness.config_probe",
            "--request",
            str(request),
            "--output",
            str(output),
        ],
        cwd=torchtitan_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    request.unlink(missing_ok=True)
    if result.returncode:
        raise CampaignError(
            "TorchTitan ConfigManager rejected the resolved arguments:\n"
            f"command argv={argv!r}\nstdout={result.stdout}\nstderr={result.stderr}"
        )
    return json.loads(output.read_text())


def _validate_planner_cli(
    campaign: Campaign,
    *,
    autoparallel_root: Path,
    output_dir: Path,
    python: Path,
) -> dict[str, Any]:
    script = autoparallel_root.resolve() / "tests/search_profile.py"
    if not script.is_file():
        raise CampaignError(f"locked AutoParallel tree has no planner CLI: {script}")
    command = [str(python), str(script), "--help"]
    completed = subprocess.run(
        command,
        cwd=autoparallel_root,
        env={
            **os.environ,
            "PYTHONPATH": _pythonpath(autoparallel_root),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    report = {
        "status": "passed",
        "classification": PLANNER_RESULT_CLASSIFICATION,
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "profile_script": str(script),
        "run_count": len(campaign.planner_runs()),
    }
    if completed.returncode or "--source-lock" not in completed.stdout:
        report["status"] = "failed"
        write_json(output_dir / "planner_contract.json", report)
        raise CampaignError(
            "planner CLI validation requires a successful --help response with "
            "the --source-lock option"
        )
    write_json(output_dir / "planner_contract.json", report)
    return report


def validate_campaign(
    campaign: Campaign,
    *,
    torchtitan_root: Path,
    autoparallel_root: Path,
    output_dir: Path,
    probe_configs: bool = True,
    asset_roots: dict[str, Path] | None = None,
    python: Path | None = None,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    harness_integrity = validate_harness_integrity()
    locked_assets = load_asset_lock()
    locked_assets_sha256 = asset_lock_digest()
    asset_roots = asset_roots or {}
    required_assets = campaign.raw.get("artifacts", {}).get("required_assets", [])
    missing_assets = sorted(set(required_assets) - set(asset_roots))
    if missing_assets:
        raise CampaignError(f"missing required --asset-root values: {missing_assets}")
    unknown_assets = sorted(set(asset_roots) - set(locked_assets["assets"]))
    if unknown_assets:
        raise CampaignError(f"asset lock has no entries for {unknown_assets}")
    for name, root in asset_roots.items():
        if not root.resolve().is_dir():
            raise CampaignError(f"asset root {name!r} is not a directory: {root}")
    asset_lock = {}
    for name, root in sorted(asset_roots.items()):
        manifest = tree_manifest(root.resolve())
        observed = {
            "root": str(root.resolve()),
            "tree_sha256": manifest_digest(manifest),
            "file_count": len(manifest),
        }
        expected = locked_assets["assets"][name]
        if observed["tree_sha256"] != expected["tree_sha256"]:
            raise CampaignError(f"locked asset {name!r} tree digest differs")
        if observed["file_count"] != int(expected["file_count"]):
            raise CampaignError(f"locked asset {name!r} file count differs")
        asset_lock[name] = observed
    write_json(output_dir / "asset_lock.json", asset_lock)
    experiment_lock = load_experiment_lock()
    experiment_lock_sha256 = experiment_lock_digest()
    if not campaign.is_planner:
        data = campaign.raw.get("data", {})
        identity_template = data.get("identity_manifest")
        identity_sha256 = data.get("identity_manifest_sha256")
        if not identity_template or not identity_sha256:
            raise CampaignError(
                "data identity_manifest and identity_manifest_sha256 are required"
            )
        identity_text = str(identity_template).replace(
            "{harness}", str(Path(__file__).resolve().parents[1])
        )
        while "{asset:" in identity_text:
            start = identity_text.index("{asset:")
            end = identity_text.index("}", start)
            name = identity_text[start + len("{asset:") : end]
            if name not in asset_roots:
                raise CampaignError(f"missing --asset-root for {name!r}")
            identity_text = (
                identity_text[:start]
                + str(asset_roots[name].resolve())
                + identity_text[end + 1 :]
            )
        identity_path = Path(identity_text)
        if not identity_path.is_file():
            raise CampaignError(
                f"input identity manifest does not exist: {identity_path}"
            )
        observed_identity_hash = hashlib.sha256(identity_path.read_bytes()).hexdigest()
        if observed_identity_hash != identity_sha256:
            raise CampaignError(
                f"input identity manifest hash mismatch: {observed_identity_hash} != "
                f"{identity_sha256}"
            )
    source_evidence = output_dir / "source_evidence"
    specs = campaign.source_specs()
    source_lock = {
        "torchtitan": inspect_source(
            "torchtitan",
            torchtitan_root,
            specs["torchtitan"],
            evidence_dir=source_evidence,
        ),
        "autoparallel": inspect_source(
            "autoparallel",
            autoparallel_root,
            specs["autoparallel"],
            evidence_dir=source_evidence,
        ),
    }
    write_json(output_dir / "source_lock.json", source_lock)
    source_lock_sha256 = hashlib.sha256(
        (output_dir / "source_lock.json").read_bytes()
    ).hexdigest()

    profiles = {arm.profile for arm in campaign.arms}
    contract: dict[str, Any] = {}
    if "apgt_validated_v1" in profiles:
        contract["apgt_validated_v1"] = validate_apgt_source(torchtitan_root)

    resolved = campaign.resolved_dict()
    resolved["experiment_lock"] = experiment_lock
    resolved["experiment_lock_sha256"] = experiment_lock_sha256
    resolved["asset_lock_sha256"] = locked_assets_sha256
    serialized: dict[str, dict[str, Any]] = {}
    serialized_environments: dict[str, dict[str, str]] = {}
    profile_checks = []
    parity_checks = []
    python = python or Path(sys.executable)
    planner_contract = None
    if campaign.is_planner:
        planner_contract = _validate_planner_cli(
            campaign,
            autoparallel_root=autoparallel_root,
            output_dir=output_dir,
            python=python,
        )
    elif probe_configs:
        serialized_root = output_dir / "serialized_configs"
        for phase in campaign.phases:
            phase_configs = {}
            for arm_name in phase.arms:
                arm = campaign.arm(arm_name)
                path = serialized_root / phase.name / f"{arm_name}.json"
                path.parent.mkdir(parents=True, exist_ok=True)

                def expand(value: str) -> str:
                    while "{asset:" in value:
                        start = value.index("{asset:")
                        end = value.index("}", start)
                        name = value[start + len("{asset:") : end]
                        if name not in asset_roots:
                            raise CampaignError(f"missing --asset-root for {name!r}")
                        value = (
                            value[:start]
                            + str(asset_roots[name].resolve())
                            + value[end + 1 :]
                        )
                    return value.replace("{output}", str(output_dir / "probe")).replace(
                        "{harness}", str(Path(__file__).resolve().parents[1])
                    )

                argv = []
                for token in campaign.phase_arm_args(phase, arm):
                    argv.append(expand(token))
                raw_probe_env = dict(campaign.raw["mast"].get("environment", {}))
                raw_probe_env.update(campaign.phase_arm_environment(phase, arm))
                raw_probe_env.setdefault("BENCHMARK_PHASE", phase.kind)
                raw_probe_env.setdefault("BENCHMARK_ARM", arm_name)
                raw_probe_env.setdefault("BENCHMARK_OUTPUT_DIR", "{output}")
                raw_probe_env.setdefault(
                    "RANK",
                    str(phase.trace_ranks[0])
                    if phase.kind in {"trace", "kineto"} and phase.trace_ranks
                    else "0",
                )
                raw_probe_env.setdefault(
                    "BENCHMARK_SOURCE_LOCK_SHA256", source_lock_sha256
                )
                probe_env = {
                    key: expand(str(value)) for key, value in raw_probe_env.items()
                }
                config = _probe_config(
                    argv,
                    torchtitan_root=torchtitan_root,
                    autoparallel_root=autoparallel_root,
                    output=path,
                    python=python,
                    extra_env=probe_env,
                )
                key = f"{phase.name}/{arm_name}"
                serialized[key] = config
                serialized_environments[key] = {
                    **campaign.raw["mast"].get("environment", {}),
                    **campaign.phase_arm_environment(phase, arm),
                }
                profile_checks.append(validate_profile(arm, config))
                _validate_phase_config(phase.kind, arm.profile, config)
                phase_configs[arm_name] = config
            comparison = campaign.raw.get("comparison", {})
            if len(phase_configs) > 1:
                pairs = comparison.get("pairs")
                if pairs is None:
                    names = list(phase_configs)
                    pairs = [[names[0], name] for name in names[1:]]
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
                        parity_checks.append(check)
                        baseline_env = dict(campaign.raw["mast"].get("environment", {}))
                        treatment_env = dict(baseline_env)
                        baseline_env.update(
                            campaign.phase_arm_environment(
                                phase, campaign.arm(baseline)
                            )
                        )
                        treatment_env.update(
                            campaign.phase_arm_environment(
                                phase, campaign.arm(treatment)
                            )
                        )
                        environment_check = validate_pair(
                            baseline,
                            baseline_env,
                            treatment,
                            treatment_env,
                            list(comparison.get("allowed_environment_keys", [])),
                        )
                        environment_check["phase"] = phase.name
                        environment_check["kind"] = "environment"
                        parity_checks.append(environment_check)

        phase_by_arm = campaign.raw.get("comparison", {}).get(
            "performance_phase_by_arm", {}
        )
        if phase_by_arm:
            comparison = campaign.raw["comparison"]
            for baseline, treatment in comparison["pairs"]:
                left_key = f"{phase_by_arm[baseline]}/{baseline}"
                right_key = f"{phase_by_arm[treatment]}/{treatment}"
                check = validate_pair(
                    baseline,
                    serialized[left_key],
                    treatment,
                    serialized[right_key],
                    list(comparison["allowed_config_paths"]),
                )
                check[
                    "phase"
                ] = f"{phase_by_arm[baseline]} -> {phase_by_arm[treatment]}"
                parity_checks.append(check)
                environment_check = validate_pair(
                    baseline,
                    serialized_environments[left_key],
                    treatment,
                    serialized_environments[right_key],
                    list(comparison.get("allowed_environment_keys", [])),
                )
                environment_check["phase"] = check["phase"]
                environment_check["kind"] = "environment"
                parity_checks.append(environment_check)

    resolved["source_lock"] = source_lock
    resolved["profile_contracts"] = contract
    resolved["serialized_config_paths"] = {
        key: f"serialized_configs/{key}.json" for key in serialized
    }
    write_json(output_dir / "resolved_campaign.json", resolved)
    report = {
        "status": "passed",
        "campaign": campaign.name,
        "campaign_type": campaign.campaign_type,
        "application_contract_validated": bool(
            planner_contract or (probe_configs and serialized)
        ),
        "probe_configs": bool(probe_configs and not campaign.is_planner),
        "python": str(python.resolve()),
        "source_lock": source_lock,
        "asset_lock": asset_lock,
        "harness_integrity": harness_integrity,
        "experiment_lock": experiment_lock,
        "experiment_lock_sha256": experiment_lock_sha256,
        "locked_assets": locked_assets,
        "locked_assets_sha256": locked_assets_sha256,
        "source_lock_sha256": source_lock_sha256,
        "profile_contracts": contract,
        "profile_checks": profile_checks,
        "parity_checks": parity_checks,
        "planner_contract": planner_contract,
    }
    write_json(output_dir / "validation_report.json", report)
    return report
