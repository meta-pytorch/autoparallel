import getpass
import logging
import os
from typing import Dict, List, Optional

import torchx.components.fb.conda as conda
import torchx.components.fb.conda_transforms as conda_transforms
import torchx.specs as specs


logger = logging.getLogger(__name__)
_ADDITIONAL_PACKAGE = "torchtitan_additional_packages"
_MOUNT_SCRIPT = "$WORKSPACE_DIR/mount.sh"
_TEE_SCRIPT = "/packages/conda_mast_core/tee/torchx_tee.sh"

_DEFAULT_ENV = {
    "NCCL_DEBUG": "INFO,WARN",
    "TORCH_SHOW_CPP_STACKTRACES": "1",
    "TORCH_ADDR2LINE_BINARY": "/packages/folly.symbolizer/folly-addr2line",
    "FUSE_DST": "/mnt/wsfuse",
    "FUSE_SRC_PATH": "checkpoint/infra",
    "TITAN_STRUCT_LOGGER_HANDLERS": (
        "torchtitan.observability.structured_logger.jsonl_handler.register_jsonl_handler"
    ),
    "TORCHINDUCTOR_COMPILE_THREADS": "8",
}


def _make_fbpkg(paths: List[str]) -> str:
    from torchx.workspace.fb import fbpkg_utils

    return fbpkg_utils.build_fbpkg(
        fbpkg_name=_ADDITIONAL_PACKAGE,
        paths=paths,
        expiration="4w",
    )


def train(
    *script_args: str,
    nodes: int = 2,
    nproc_per_node: int = 8,
    name: str = "muse-glimmer-30b-sdpa-ap-fix-sac-aligned-s4096",
    h: str = "grandteton_80g_roce",
    env: Optional[Dict[str, str]] = None,
    retries: int = 0,
    module_name: str = "llama3",
    config_name: str = "llama3_8b",
    additional_folders: Optional[List[str]] = None,
    additional_libraries: Optional[List[str]] = None,
    enable_ttls: bool = True,
) -> specs.AppDef:
    if nproc_per_node != 8:
        raise ValueError("This experiment requires exactly eight ranks per host")
    if len(script_args) != 6:
        raise ValueError(
            "Expected: {gate|formal} WORLD_SIZE 2 two_arm sequential REPLICATE"
        )
    mode, world_size_text, tp_degree_text, matrix, order, replicate = script_args
    if mode not in {"gate", "formal"} or matrix != "two_arm" or order != "sequential":
        raise ValueError(f"Invalid experiment arguments: {script_args!r}")
    if not world_size_text.isdecimal() or not tp_degree_text.isdecimal():
        raise ValueError(f"Invalid mesh arguments: {script_args!r}")
    world_size = int(world_size_text)
    tp_degree = int(tp_degree_text)
    if world_size != 16 or nodes != 2 or tp_degree != 2:
        raise ValueError(f"Unsupported mesh: {world_size=} {tp_degree=}")
    if nodes * nproc_per_node != world_size:
        raise ValueError(
            f"Requested {nodes}x{nproc_per_node} ranks but WORLD_SIZE={world_size}"
        )
    if not replicate.isdecimal():
        raise ValueError(f"Invalid replicate: {replicate!r}")

    folders = list(additional_folders or [])
    libraries = list(additional_libraries or [])
    additional_pkg = _make_fbpkg([*folders, *libraries])
    python_paths = [
        f"/packages/{_ADDITIONAL_PACKAGE}/{os.path.basename(path.rstrip('/'))}"
        for path in libraries
    ]

    job_name = f"{name}-{world_size}-{getpass.getuser()}"
    task_env = dict(_DEFAULT_ENV)
    task_env.update(env or {})
    task_env["MODULE"] = module_name
    task_env["CONFIG"] = config_name
    task_env["DUMP_DIR"] = "/mnt/wsfuse/outputs/${app_id}"
    task_env["EXPERIMENT_TASK"] = "training"
    task_env["JOB_ID"] = "${app_id}"
    task_env["TORCHX_RUN_PYTHONPATH"] = ":".join(python_paths)
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
        "${img_root}/run_rank.sh",
        *script_args,
        name=job_name,
        h=h,
        env=task_env,
        max_retries=retries,
        run_as_root=True,
        enable_ttls=enable_ttls,
        conda_mast_core_fbpkg_id="conda_mast_core:stable",
    )
    role = task_app.roles[0]
    role.name = "training"
    num_phases = 2 if mode == "gate" else 4
    for index in range(1, num_phases):
        role.port_map[f"training_phase_{index + 1}"] = 29500 + index

    job_spec = specs.AppDef(
        name=job_name,
        roles=[role],
        metadata=task_app.metadata,
    )
    job_spec = conda_transforms.append_tb_logdir_metadata(job_spec)
    role.entrypoint = f"{_MOUNT_SCRIPT} && {_TEE_SCRIPT} {role.entrypoint}"
    role.image = ";".join(
        [
            role.image,
            "folly.symbolizer:stable",
            "oil.oilfs:stable",
            additional_pkg,
        ]
    )
    return job_spec
