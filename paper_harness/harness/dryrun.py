from __future__ import annotations

import json
import os
import shutil
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


def _without_metadata(root: Path) -> dict[str, str]:
    return {
        path: digest
        for path, digest in tree_manifest(root).items()
        if path != "METADATA" and not path.endswith(".CHECKSUMS")
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
        mast["conda_fbpkg"] == locked_fbpkg
        and packages.count(locked_fbpkg) == 1,
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
    source_payload_manifest = _without_metadata(attempt / "package/payload")
    source_launcher_manifest = _without_metadata(launcher_root)
    report = {
        "status": "passed",
        "checks": checks,
        "failed": [],
        "packages": package_ids,
        "payload_tree_sha256": manifest_digest(source_payload_manifest),
        "workspace_tree_sha256": manifest_digest(source_launcher_manifest),
    }
    write_json(attempt / "dryrun_audit.json", report)
    return report
