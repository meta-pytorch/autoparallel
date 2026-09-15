from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Any

from .campaign import CampaignError, write_json
from .sources import manifest_digest, tree_manifest

MARKER = "=== SCHEDULER REQUEST ===\n"
WORKSPACE_PACKAGE = "torchtitan_workspace:"
PAYLOAD_PACKAGE = "torchtitan_additional_packages:"
HARDWARE_SERVER_SUBTYPES = {
    "grandteton_80g_roce": "LogicalServerSubType.T20_GRAND_TETON_HBM3_ROCE",
}
LOCALITY_SCOPES = {
    "dc": "Locality.DC",
    "region": "Locality.REGION",
}


def _argument(arguments: list[str], option: str) -> str | None:
    try:
        return arguments[arguments.index(option) + 1]
    except (ValueError, IndexError):
        return None


def _check(name: str, condition: bool, checks: dict[str, bool]) -> None:
    checks[name] = bool(condition)


def _fetch(package: str, destination: Path, log_root: Path) -> None:
    if destination.exists():
        raise CampaignError(f"refusing to reuse package-audit directory {destination}")
    destination.mkdir(parents=True)
    completed = subprocess.run(
        ["fbpkg", "fetch", package, "--dest", str(destination)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    prefix = package.split(":", 1)[0]
    (log_root / f"fetch_{prefix}.stdout").write_text(completed.stdout)
    (log_root / f"fetch_{prefix}.stderr").write_text(completed.stderr)
    (log_root / f"fetch_{prefix}.returncode").write_text(f"{completed.returncode}\n")
    if completed.returncode:
        raise CampaignError(f"fbpkg fetch failed for {package}")


def _without_metadata(manifest: dict[str, str]) -> dict[str, str]:
    return {
        path: digest
        for path, digest in manifest.items()
        if path != "METADATA" and not path.endswith(".CHECKSUMS")
    }


def _package_snapshot(root: Path, manifest: dict[str, str]) -> dict[str, Any]:
    files = {}
    for relative, digest in manifest.items():
        mode = stat.S_IMODE((root / relative).stat().st_mode)
        files[relative] = {"sha256": digest, "mode": f"{mode:04o}"}
    return {
        "root": str(root),
        "tree_sha256": manifest_digest(manifest),
        "file_count": len(manifest),
        "metadata_files": sorted(
            path
            for path in manifest
            if Path(path).name == "METADATA" or path.endswith(".CHECKSUMS")
        ),
        "files": files,
    }


def _metadata_matches(root: Path, identifier: str) -> bool:
    expected_name, expected_version = identifier.split(":", 1)
    try:
        metadata = json.loads((root / "METADATA").read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        metadata.get("package") == expected_name
        and metadata.get("version") == expected_version
    )


def _run_package_preflight(
    *, payload: Path, python: Path, output: Path, log_root: Path
) -> dict[str, Any]:
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(
                [
                    str(payload / "harness_repo"),
                    str(payload / "autoparallel"),
                    str(payload / "torchtitan"),
                ]
            ),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    completed = subprocess.run(
        [str(python), "-m", "harness.package_preflight", str(payload), str(output)],
        cwd=payload / "torchtitan",
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    (log_root / "exact_package_preflight.stdout").write_text(completed.stdout)
    (log_root / "exact_package_preflight.stderr").write_text(completed.stderr)
    (log_root / "exact_package_preflight.returncode").write_text(
        f"{completed.returncode}\n"
    )
    config_report = output / "runtime/configs/report.json"
    source_report = output / "runtime/preflight/report.json"
    return {
        "status": "passed"
        if completed.returncode == 0
        and config_report.is_file()
        and source_report.is_file()
        else "failed",
        "python": str(python),
        "returncode": completed.returncode,
        "config_report": str(config_report),
        "source_report": str(source_report),
    }


def _scheduler_arguments(command: list[str]) -> dict[str, str]:
    values = [
        token.removeprefix("--scheduler_args=")
        for token in command
        if token.startswith("--scheduler_args=")
    ]
    if len(values) != 1:
        return {}
    result = {}
    for item in values[0].split(","):
        if "=" not in item:
            return {}
        key, value = item.split("=", 1)
        result[key] = value
    return result


def _definition_payload(definition: dict[str, Any]) -> dict[str, Any]:
    data = definition.get("data")
    if "hpcTaskGroups" not in definition and isinstance(data, dict):
        return data
    return definition


def _definition_checks(
    definition: dict[str, Any],
    *,
    resolved: dict[str, Any],
    launcher_root: Path,
    command: list[str],
    combined: str | None,
    expected_job_id: str | None = None,
) -> tuple[dict[str, bool], list[str]]:
    definition = _definition_payload(definition)
    mast = resolved["mast"]
    groups = definition.get("hpcTaskGroups", [])
    group = groups[0] if len(groups) == 1 else {}
    spec = group.get("spec", {})
    arguments = spec.get("arguments", [])
    env = spec.get("env", {})
    packages = [
        row.get("fbpkgIdentifier") for row in spec.get("applicationPackages", [])
    ]
    ports = spec.get("ports", {})
    checks: dict[str, bool] = {}
    expected_config = launcher_root / ".torchxconfig"
    if combined is not None:
        _check(
            "loaded_exact_torchxconfig",
            f"loaded configs from `{expected_config}`" in combined
            or f"loaded configs from {expected_config}" in combined,
            checks,
        )
        _check("no_local_scheduler", "local_cwd" not in combined, checks)
        _check(
            "no_unknown_scheduler_options",
            "unknown scheduler options" not in combined.lower(),
            checks,
        )
    _check(
        "mast_scheduler",
        definition.get("applicationMetadata", {}).get("torchx/scheduler")
        == "mast_conda",
        checks,
    )
    scheduler = _scheduler_arguments(command)
    _check(
        "force_single_region_false",
        scheduler.get("forceSingleRegion") == "False",
        checks,
    )
    _check(
        "scheduler_locality",
        scheduler.get("localityConstraints") == str(mast["locality"]),
        checks,
    )
    _check(
        "scheduler_conda",
        scheduler.get("conda_fbpkg_id") == str(mast["conda_fbpkg"]),
        checks,
    )
    _check(
        "genai_cluster",
        definition.get("hpcClusterUuid") == "MastGenAICluster",
        checks,
    )
    job_name = definition.get("name")
    _check(
        "job_name",
        job_name == expected_job_id
        if expected_job_id is not None
        else isinstance(job_name, str) and job_name.startswith(f"{resolved['name']}-"),
        checks,
    )
    _check("one_task_group", len(groups) == 1, checks)
    _check("host_count", group.get("taskCount") == int(mast["nodes"]), checks)
    _check("one_task_per_host", group.get("taskCountPerHost") == 1, checks)
    _check(
        "gpus_per_host",
        spec.get("resourceLimit", {}).get("compute", {}).get("gpu")
        == int(mast["nproc_per_node"]),
        checks,
    )
    expected_subtype = HARDWARE_SERVER_SUBTYPES.get(str(mast["hardware"]))
    _check(
        "hardware_subtype",
        expected_subtype is not None
        and spec.get("machineConstraints", {}).get("types", {}).get("serverSubTypes")
        == [expected_subtype],
        checks,
    )
    _check(
        "torchrun_nodes",
        _argument(arguments, "--nnodes") == str(mast["nodes"]),
        checks,
    )
    _check(
        "torchrun_processes",
        _argument(arguments, "--nproc-per-node") == str(mast["nproc_per_node"]),
        checks,
    )
    single_node = int(mast["nodes"]) == 1
    _check(
        "rendezvous_backend",
        _argument(arguments, "--rdzv_backend") == ("c10d" if single_node else "mast"),
        checks,
    )
    _check(
        "rendezvous_endpoint",
        _argument(arguments, "--rdzv_endpoint") == "localhost:0"
        if single_node
        else "--rdzv_endpoint" not in arguments,
        checks,
    )
    _check(
        "rendezvous_config",
        "--rdzv_conf" not in arguments
        if single_node
        else _argument(arguments, "--rdzv_conf") == "use_libuv=True",
        checks,
    )
    _check("rendezvous_id", _argument(arguments, "--rdzv_id") == job_name, checks)
    _check(
        "role",
        _argument(arguments, "--role") == resolved.get("campaign_type", "training"),
        checks,
    )
    _check(
        "packaged_runner",
        _argument(arguments, "--no-python")
        == "/packages/torchtitan_additional_packages/payload/harness_repo/launcher/run_rank.sh",
        checks,
    )
    _check(
        "bootstrap",
        "$WORKSPACE_DIR/mount.sh" in str(spec.get("command", ""))
        and "/packages/conda_mast_core/tee/torchx_tee.sh"
        in str(spec.get("command", "")),
        checks,
    )
    locality_parts = str(mast["locality"]).split(";", 1)
    expected_scope = LOCALITY_SCOPES.get(locality_parts[0])
    expected_locality = locality_parts[1] if len(locality_parts) == 2 else None
    observed_locality = definition.get("localityConstraints", {})
    _check(
        "locality_scope",
        expected_scope is not None
        and observed_locality.get("locality") == expected_scope,
        checks,
    )
    _check(
        "locality_option",
        expected_locality is not None
        and observed_locality.get("options") == [expected_locality],
        checks,
    )
    _check(
        "zero_role_retries",
        spec.get("restartPolicy", {}).get("maxTotalFailures")
        == int(mast.get("retries", 0)),
        checks,
    )
    _check(
        "zero_job_retries",
        definition.get("maxJobFailures") == int(mast.get("retries", 0)),
        checks,
    )
    _check("ttls", spec.get("ttlsConfig", {}).get("enable") is True, checks)
    locked_fbpkg = resolved["experiment_lock"]["runtime"]["conda_fbpkg"]
    _check(
        "conda",
        mast["conda_fbpkg"] == locked_fbpkg and packages.count(locked_fbpkg) == 1,
        checks,
    )
    _check("oilfs", "oil.oilfs:stable" in packages, checks)
    _check(
        "one_workspace_package",
        sum(str(package).startswith(WORKSPACE_PACKAGE) for package in packages) == 1,
        checks,
    )
    _check(
        "one_payload_package",
        sum(str(package).startswith(PAYLOAD_PACKAGE) for package in packages) == 1,
        checks,
    )
    process_count = sum(len(phase["arms"]) for phase in resolved["resolved_phases"])
    base_port = int(mast.get("master_port", 29500))
    expected_ports = {
        f"training_phase_{index + 1}": base_port + index
        for index in range(1, process_count)
    }
    _check("sequential_process_ports", ports == expected_ports, checks)
    _check("root_user", spec.get("unixUser") == "root", checks)
    _check(
        "payload_root",
        env.get("HARNESS_PAYLOAD_ROOT")
        == "/packages/torchtitan_additional_packages/payload",
        checks,
    )
    _check(
        "output_mount",
        env.get("DUMP_DIR") == f"/mnt/wsfuse/outputs/{expected_job_id}"
        if expected_job_id is not None
        else str(env.get("DUMP_DIR", "")).startswith("/mnt/wsfuse/outputs/"),
        checks,
    )
    _check(
        "structured_logger",
        env.get("TITAN_STRUCT_LOGGER_HANDLERS")
        == "torchtitan.observability.structured_logger.jsonl_handler.register_jsonl_handler",
        checks,
    )
    _check(
        "experiment_task",
        env.get("EXPERIMENT_TASK") == resolved.get("campaign_type", "training"),
        checks,
    )
    _check(
        "launcher_files",
        expected_config.is_file()
        and (launcher_root / "mount.sh").is_file()
        and os.access(launcher_root / "mount.sh", os.X_OK)
        and (launcher_root / "run_rank.sh").is_file()
        and os.access(launcher_root / "run_rank.sh", os.X_OK),
        checks,
    )
    return checks, packages


def audit_submitted_definition(
    attempt: Path, definition: dict[str, Any], *, launcher_root: Path
) -> dict[str, Any]:
    attempt = attempt.resolve()
    resolved = json.loads((attempt / "validation/resolved_campaign.json").read_text())
    command = json.loads((attempt / "submission.command.json").read_text())
    job_id = (attempt / "job_id.txt").read_text().strip()
    checks, packages = _definition_checks(
        definition,
        resolved=resolved,
        launcher_root=launcher_root.resolve(),
        command=command,
        combined=None,
        expected_job_id=job_id,
    )
    failed = [name for name, passed in checks.items() if not passed]
    report = {
        "status": "passed" if not failed else "failed",
        "checks": checks,
        "failed": failed,
        "packages": packages,
    }
    write_json(attempt / "submitted_definition_audit.json", report)
    if failed:
        raise CampaignError(f"submitted MAST definition validation failed: {failed}")
    return report


def audit_dryrun(
    attempt: Path,
    *,
    stdout: str,
    stderr: str,
    launcher_root: Path,
) -> dict[str, Any]:
    attempt = attempt.resolve()
    launcher_root = launcher_root.resolve()
    combined = stdout + "\n" + stderr
    if MARKER not in stdout:
        raise CampaignError("TorchX dry-run did not emit a scheduler request")
    definition, _ = json.JSONDecoder().raw_decode(stdout.split(MARKER, 1)[1])
    resolved = json.loads((attempt / "validation/resolved_campaign.json").read_text())
    command = json.loads((attempt / "dryrun.command.json").read_text())
    checks, packages = _definition_checks(
        definition,
        resolved=resolved,
        launcher_root=launcher_root,
        command=command,
        combined=combined,
    )

    failed = [name for name, passed in checks.items() if not passed]
    write_json(attempt / "job_definition.dryrun.json", definition)
    if failed:
        report = {
            "status": "failed",
            "checks": checks,
            "failed": failed,
            "packages": packages,
        }
        write_json(attempt / "dryrun_audit.json", report)
        raise CampaignError(f"MAST dry-run validation failed: {failed}")

    package_ids = {
        "workspace": next(
            package
            for package in packages
            if str(package).startswith(WORKSPACE_PACKAGE)
        ),
        "payload": next(
            package for package in packages if str(package).startswith(PAYLOAD_PACKAGE)
        ),
    }
    fetch_root = attempt / "packages"
    fetch_root.mkdir(exist_ok=True)
    workspace_fetch = fetch_root / "dryrun_workspace"
    payload_fetch = fetch_root / "dryrun_payload"
    _fetch(package_ids["workspace"], workspace_fetch, attempt)
    _fetch(package_ids["payload"], payload_fetch, attempt)

    source_payload = attempt / "package/payload"
    fetched_payload = payload_fetch / source_payload.name
    source_payload_full = tree_manifest(source_payload)
    fetched_payload_package_full = tree_manifest(payload_fetch)
    fetched_payload_full = {
        path.removeprefix("payload/"): digest
        for path, digest in fetched_payload_package_full.items()
        if path.startswith("payload/")
    }
    source_launcher_full = tree_manifest(launcher_root)
    fetched_launcher_full = tree_manifest(workspace_fetch)
    source_payload_manifest = _without_metadata(source_payload_full)
    fetched_payload_manifest = _without_metadata(fetched_payload_full)
    source_launcher_manifest = _without_metadata(source_launcher_full)
    fetched_launcher_manifest = _without_metadata(fetched_launcher_full)
    snapshots = {
        "source_payload": _package_snapshot(source_payload, source_payload_full),
        "fetched_payload_package": _package_snapshot(
            payload_fetch, fetched_payload_package_full
        ),
        "source_workspace": _package_snapshot(launcher_root, source_launcher_full),
        "fetched_workspace": _package_snapshot(workspace_fetch, fetched_launcher_full),
    }
    for name, snapshot in snapshots.items():
        write_json(attempt / "packages" / f"{name}.manifest.json", snapshot)

    def modes(name: str) -> dict[str, str]:
        return {path: row["mode"] for path, row in snapshots[name]["files"].items()}

    package_checks = {
        "payload_content": source_payload_manifest == fetched_payload_manifest,
        "workspace_content": source_launcher_manifest == fetched_launcher_manifest,
        "payload_permissions": modes("source_payload")
        == {
            path.removeprefix("payload/"): mode
            for path, mode in modes("fetched_payload_package").items()
            if path.startswith("payload/")
        },
        "workspace_permissions": modes("source_workspace")
        == {
            path: mode
            for path, mode in modes("fetched_workspace").items()
            if path != "METADATA" and not path.endswith(".CHECKSUMS")
        },
        "payload_metadata": "METADATA"
        in snapshots["fetched_payload_package"]["metadata_files"],
        "workspace_metadata": "METADATA"
        in snapshots["fetched_workspace"]["metadata_files"],
        "payload_metadata_identity": _metadata_matches(
            payload_fetch, package_ids["payload"]
        ),
        "workspace_metadata_identity": _metadata_matches(
            workspace_fetch, package_ids["workspace"]
        ),
        "fetched_runner_executable": os.access(
            fetched_payload / "harness_repo/launcher/run_rank.sh", os.X_OK
        ),
        "fetched_mount_executable": os.access(workspace_fetch / "mount.sh", os.X_OK),
    }
    python = Path(
        json.loads((attempt / "package_report.json").read_text())["validation"][
            "python"
        ]
    )
    exact_preflight = _run_package_preflight(
        payload=fetched_payload,
        python=python,
        output=attempt / "exact_package_preflight",
        log_root=attempt,
    )
    package_checks["exact_package_import_and_configs"] = (
        exact_preflight["status"] == "passed"
    )
    failed_packages = [name for name, passed in package_checks.items() if not passed]
    report = {
        "status": "passed" if not failed_packages else "failed",
        "checks": checks,
        "failed": failed_packages,
        "packages": package_ids,
        "package_checks": package_checks,
        "exact_package_preflight": exact_preflight,
        "payload_tree_sha256": manifest_digest(source_payload_manifest),
        "fetched_payload_tree_sha256": manifest_digest(fetched_payload_manifest),
        "workspace_tree_sha256": manifest_digest(source_launcher_manifest),
        "fetched_workspace_tree_sha256": manifest_digest(fetched_launcher_manifest),
    }
    write_json(attempt / "dryrun_audit.json", report)
    if failed_packages:
        raise CampaignError(
            f"fetched dry-run package validation failed: {failed_packages}"
        )
    return report
