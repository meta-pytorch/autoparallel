from __future__ import annotations

import json
import os
import shutil
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
    (log_root / f"fetch_{prefix}.returncode").write_text(
        f"{completed.returncode}\n"
    )
    if completed.returncode:
        raise CampaignError(f"fbpkg fetch failed for {package}")


def _without_metadata(root: Path) -> dict[str, str]:
    return {
        path: digest
        for path, digest in tree_manifest(root).items()
        if path != "METADATA" and not path.endswith(".CHECKSUMS")
    }


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
        if completed.returncode == 0 and config_report.is_file() and source_report.is_file()
        else "failed",
        "python": str(python),
        "returncode": completed.returncode,
        "config_report": str(config_report),
        "source_report": str(source_report),
    }


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
    _check(
        "loaded_exact_torchxconfig",
        f"loaded configs from `{expected_config}`" in combined
        or f"loaded configs from {expected_config}" in combined,
        checks,
    )
    _check("mast_scheduler", '"torchx/scheduler": "mast_conda"' in stdout, checks)
    _check(
        "genai_cluster",
        definition.get("hpcClusterUuid") == "MastGenAICluster",
        checks,
    )
    _check("no_local_scheduler", "local_cwd" not in combined, checks)
    _check("no_unknown_scheduler_options", "unknown scheduler options" not in combined.lower(), checks)
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
        and spec.get("machineConstraints", {})
        .get("types", {})
        .get("serverSubTypes")
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
    _check(
        "packaged_runner",
        _argument(arguments, "--no-python")
        == "/packages/torchtitan_additional_packages/payload/harness_repo/launcher/run_rank.sh",
        checks,
    )
    _check(
        "bootstrap",
        "$WORKSPACE_DIR/mount.sh" in str(spec.get("command", ""))
        and "/packages/conda_mast_core/tee/torchx_tee.sh" in str(spec.get("command", "")),
        checks,
    )
    expected_locality = str(mast["locality"]).split(";", 1)[-1]
    _check(
        "locality",
        definition.get("localityConstraints", {}).get("options")
        == [expected_locality],
        checks,
    )
    _check("zero_role_retries", spec.get("restartPolicy", {}).get("maxTotalFailures") == int(mast.get("retries", 0)), checks)
    _check("zero_job_retries", definition.get("maxJobFailures") == int(mast.get("retries", 0)), checks)
    _check("ttls", spec.get("ttlsConfig", {}).get("enable") is True, checks)
    _check("conda", mast["conda_fbpkg"] in packages, checks)
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
    _check("output_mount", str(env.get("DUMP_DIR", "")).startswith("/mnt/wsfuse/outputs/"), checks)
    _check(
        "structured_logger",
        env.get("TITAN_STRUCT_LOGGER_HANDLERS")
        == "torchtitan.observability.structured_logger.jsonl_handler.register_jsonl_handler",
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
            package for package in packages if str(package).startswith(WORKSPACE_PACKAGE)
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
    source_payload_manifest = _without_metadata(source_payload)
    fetched_payload_manifest = _without_metadata(fetched_payload)
    source_launcher_manifest = _without_metadata(launcher_root)
    fetched_launcher_manifest = _without_metadata(workspace_fetch)
    package_checks = {
        "payload_content": source_payload_manifest == fetched_payload_manifest,
        "workspace_content": source_launcher_manifest == fetched_launcher_manifest,
        "fetched_runner_executable": os.access(
            fetched_payload / "harness_repo/launcher/run_rank.sh", os.X_OK
        ),
        "fetched_mount_executable": os.access(workspace_fetch / "mount.sh", os.X_OK),
    }
    python = Path(
        json.loads((attempt / "package_report.json").read_text())["validation"]["python"]
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
        raise CampaignError(f"fetched dry-run package validation failed: {failed_packages}")
    return report
