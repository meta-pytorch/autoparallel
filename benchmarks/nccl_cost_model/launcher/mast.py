# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import getpass
from pathlib import Path
from typing import Dict, Optional

import conda.conda_torchx.components.fb.conda as conda
import conda.conda_torchx.components.fb.conda_transforms as conda_transforms
import torchx.specs as specs

_PACKAGE = "torchtitan_additional_packages"
_MOUNT_SCRIPT = "$WORKSPACE_DIR/mount.sh"
_TEE_SCRIPT = "/packages/conda_mast_core/tee/torchx_tee.sh"


def _make_fbpkg(path: str) -> str:
    from conda.conda_torchx.workspace.fb import fbpkg_utils

    return fbpkg_utils.build_fbpkg(
        fbpkg_name=_PACKAGE,
        paths=[path],
        expiration="4w",
    )


def benchmark(
    *,
    payload_root: str,
    nodes: int,
    replicate: int,
    gate: bool = False,
    modes: str = "",
    nproc_per_node: int = 8,
    name: str = "h100-nvswitch-roce-400g-calibration",
    h: str = "grandteton_80g_roce",
    env: Optional[Dict[str, str]] = None,
    retries: int = 0,
    enable_ttls: bool = True,
) -> specs.AppDef:
    allowed_nodes = (1, 2) if gate else (1, 2, 4, 8, 16, 32)
    if nodes not in allowed_nodes or nproc_per_node != 8:
        raise ValueError(f"invalid calibration allocation: {nodes}x{nproc_per_node}")
    if replicate not in (1, 2, 3):
        raise ValueError("replicate must be 1, 2, or 3")
    payload = Path(payload_root).resolve()
    package = _make_fbpkg(str(payload))
    task_env = {
        "NCCL_DEBUG": "INFO",
        "NCCL_DEBUG_SUBSYS": "INIT,GRAPH,TUNING",
        "TORCH_SHOW_CPP_STACKTRACES": "1",
        "TORCH_ADDR2LINE_BINARY": "/packages/folly.symbolizer/folly-addr2line",
        "FUSE_DST": "/mnt/wsfuse",
        "FUSE_SRC_PATH": "checkpoint/infra",
        "DUMP_DIR": "/mnt/wsfuse/outputs/${app_id}",
        "JOB_ID": "${app_id}",
        "CALIBRATION_GATE": "1" if gate else "0",
        "CALIBRATION_REPLICATE": str(replicate),
        "CALIBRATION_ROOT": f"/packages/{_PACKAGE}/{payload.name}",
    }
    if modes:
        task_env["CALIBRATION_MODES"] = modes
    task_env.update(env or {})
    entrypoint = f"/packages/{_PACKAGE}/{payload.name}/runner/run_rank.sh"
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
        name=f"{name}-{nodes}x8-r{replicate}-{getpass.getuser()}",
        h=h,
        env=task_env,
        max_retries=retries,
        run_as_root=True,
        enable_ttls=enable_ttls,
        conda_mast_core_fbpkg_id="conda_mast_core:stable",
    )
    role = task_app.roles[0]
    role.name = "training"
    app = specs.AppDef(name=task_app.name, roles=[role], metadata=task_app.metadata)
    app = conda_transforms.append_tb_logdir_metadata(app)
    role.entrypoint = f"{_MOUNT_SCRIPT} && {_TEE_SCRIPT} {role.entrypoint}"
    role.image = ";".join(
        [role.image, "folly.symbolizer:stable", "oil.oilfs:stable", package]
    )
    return app
