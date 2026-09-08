import getpass
import json
import os
from pathlib import Path
from typing import Dict, Optional

import conda.conda_torchx.components.fb.conda as conda
import conda.conda_torchx.components.fb.conda_transforms as conda_transforms
import torchx.specs as specs


_ADDITIONAL_PACKAGE = "torchtitan_additional_packages"
_MOUNT_SCRIPT = "$WORKSPACE_DIR/mount.sh"
_TEE_SCRIPT = "/packages/conda_mast_core/tee/torchx_tee.sh"


def _make_fbpkg(path: str) -> str:
    from conda.conda_torchx.workspace.fb import fbpkg_utils

    return fbpkg_utils.build_fbpkg(
        fbpkg_name=_ADDITIONAL_PACKAGE,
        paths=[path],
        expiration="4w",
    )


def train(
    *,
    payload_root: str,
    nodes: int,
    nproc_per_node: int = 8,
    name: str = "permanent-harness",
    h: str = "grandteton_80g_roce",
    env: Optional[Dict[str, str]] = None,
    retries: int = 0,
    enable_ttls: bool = True,
) -> specs.AppDef:
    payload = Path(payload_root).resolve()
    resolved = json.loads((payload / "campaign/resolved_campaign.json").read_text())
    if nodes * nproc_per_node != int(resolved["world_size"]):
        raise ValueError("MAST resources differ from the resolved campaign")
    package = _make_fbpkg(str(payload))
    task_env = {
        "NCCL_DEBUG": "INFO,WARN",
        "TORCH_SHOW_CPP_STACKTRACES": "1",
        "TORCH_ADDR2LINE_BINARY": "/packages/folly.symbolizer/folly-addr2line",
        "FUSE_DST": "/mnt/wsfuse",
        "FUSE_SRC_PATH": "checkpoint/infra",
        "TITAN_STRUCT_LOGGER_HANDLERS": (
            "torchtitan.observability.structured_logger.jsonl_handler.register_jsonl_handler"
        ),
        # Compile parallelism is experiment state. Leave PyTorch's default intact
        # unless the campaign explicitly pins TORCHINDUCTOR_COMPILE_THREADS in
        # mast.environment.
        "HARNESS_PAYLOAD_ROOT": f"/packages/{_ADDITIONAL_PACKAGE}/{payload.name}",
        "DUMP_DIR": "/mnt/wsfuse/outputs/${app_id}",
        "EXPERIMENT_TASK": "training",
        "JOB_ID": "${app_id}",
    }
    task_env.update(resolved["mast"].get("environment", {}))
    task_env.update(env or {})
    entrypoint = f"{task_env['HARNESS_PAYLOAD_ROOT']}/harness_repo/launcher/run_rank.sh"
    task_app = conda.torchrun(
        "--tee",
        "3",
        "--nnodes",
        str(nodes),
        "--nproc-per-node",
        str(nproc_per_node),
        "--role",
        "training",
        "--no-python",
        entrypoint,
        name=f"{name}-{getpass.getuser()}",
        h=h,
        env=task_env,
        max_retries=retries,
        run_as_root=True,
        enable_ttls=enable_ttls,
        conda_mast_core_fbpkg_id="conda_mast_core:stable",
    )
    role = task_app.roles[0]
    role.name = "training"
    for index in range(1, sum(len(phase["arms"]) for phase in resolved["resolved_phases"])):
        role.port_map[f"training_phase_{index + 1}"] = int(
            resolved["mast"].get("master_port", 29500)
        ) + index
    job_spec = specs.AppDef(name=task_app.name, roles=[role], metadata=task_app.metadata)
    job_spec = conda_transforms.append_tb_logdir_metadata(job_spec)
    role.entrypoint = f"{_MOUNT_SCRIPT} && {_TEE_SCRIPT} {role.entrypoint}"
    role.image = ";".join(
        [role.image, "folly.symbolizer:stable", "oil.oilfs:stable", package]
    )
    return job_spec
