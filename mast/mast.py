# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
import copy
import getpass
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple

import torchx.components.fb.conda as conda
import torchx.components.fb.conda_transforms as conda_transforms
import torchx.specs as specs
from torchx.schedulers.ids import random_id
from torchx.specs import named_resources

logger: logging.Logger = logging.getLogger(__name__)
_DEFAULT_AIRSTORE_PACKAGE = "ws_airstore.client:prod"

_DEFAULT_ENV = {
    "NCCL_DEBUG": "INFO,WARN",
    "TORCH_SHOW_CPP_STACKTRACES": "1",
    "TORCH_ADDR2LINE_BINARY": "/packages/folly.symbolizer/folly-addr2line",
    "FUSE_DST": "/mnt/mffuse",
    "ENABLE_MANIFUSE_OVER_MANIFOLDFS": "1",
    "MANIFOLDFS_BUCKET": "torchtrain_datasets",
    # --- WS-Airstore configuration
    "ENABLE_AIRSTORE": "",
    "AIRSTORE_DECRYPT_SERVER_AFFINITY": "parent",
    "AIRSTORE_DECRYPT_SERVER_PATH": "/packages/ws_airstore.client/decrypt_server",
    "AIRSTORE_LOCAL_MOUNT_ROOT": "/mnt/airstore",
    # WS-AIRStore caches the shuffling, sharding information to enable fast startups
    "AIRSTORE_INTERVAL_CACHE_DIR": "/mnt/airstore/airstore_metadata_cache",
    # For long running llamma4 production training jobs, please
    #  set AIRSTORE_FBPKG_ID env var to ws_airstore.client:prod
    "AIRSTORE_FBPKG_ID": _DEFAULT_AIRSTORE_PACKAGE,
    # --- OilFS
    "WS_SSCV2_THRIFT_CONN_POOL_SIZE": "250000",
    # Only used for pretraining jobs. Perf tweaks for 8k+ gpu jobs
    "OILFS_PROFILE": "pretraining",
}

DEFAULT_ARGS = {
    "test.py": {
        # Required for alerting & getting metrics on a dashboard.
        "enable_ods": True,
    }
}

_MOUNT_SCRIPT = "$WORKSPACE_DIR/mount.sh"
_PY_SPY_SCRIPT = "py_spy_startup.sh"
_TEE_SCRIPT = "/packages/conda_mast_core/tee/torchx_tee.sh"
_RUN_SCRIPT = "run_nothing.sh"
_ADDITIONAL_PACKAGES_FBPKG_NAME = "torchtitan_additional_packages"

WITH_PROXY_ENV_VARS: Final[dict[str, str]] = {
    "https_proxy": "http://fwdproxy:8080",
    "http_proxy": "http://fwdproxy:8080",
    "no_proxy": (
        ".fbcdn.net,.facebook.com,.thefacebook.com,.tfbnw.net,.fb.com,"
        ".fburl.com,.facebook.net,.sb.fbsbx.com,localhost"
    ),
}

# Monarch
# This fbpkg contains image for both client/worker (includes Pytorch) and hyperactor/controller/system
_HYPERACTOR_FBPKG = "monarch_torchtitan"
_HYPERACTOR_MAST_BOOTSTRAP = "/packages/" + _HYPERACTOR_FBPKG + "/hyperactor"
_MONARCH_LIB_LOCATION = f"/packages/{_HYPERACTOR_FBPKG}/lib"

# Default port name for the system port to be used in TW.
_TW_SYSTEM_PORT_NAME = "system"
_TW_MESH_WORKER_PORT_NAME = "mesh"
# Cannot use an auto port (0), because the allocator on the client expects all
# tasks to have the same port.
_MONARCH_PORT = 29500
_MONARCH_RENDEZVOUS_PORT = 29501
# Default named ports for TW for core task group.
# TODO: Use dynamic port assignment to avoid potential host port conflicts.
_MAST_DEFAULT_CORE_PORTS = {_TW_SYSTEM_PORT_NAME: _MONARCH_PORT}
# SMC bridge tier name for service discovery on workers.
_SMC_TIER_NAME_ENV = "MONARCH_SMC_SYSTEM_TIER_NAME"


def train(
    *script_args: str,
    script: str = "test.py",  # wenyin: not really used currently
    module: Optional[str] = None,
    nodes: int = 2,
    nproc_per_node: int = 8,
    name: str = "cpu_nccl_init",
    h: str = "t1",
    env: Optional[Dict[str, str]] = None,
    unset_env: Optional[List[str]] = None,
    retry_policy: Optional[str] = None,
    run_as_root: bool = False,
    dump_dir_id: str = "${app_id}",
    xzone: bool = False,
    dump_logs: bool = True,
    additional_libraries: Optional[List[str]] = None,
    additional_folders: Optional[List[str]] = None,
    additional_python_paths: Optional[List[str]] = None,
    py_spy_startup: bool = False,
    retries: int = 1,
    twtask_bootstrap_script: Optional[str] = None,
    enable_ttls: bool = False,
) -> specs.AppDef:
    """
    Kick off a training job on MAST.
    Sane defaults are specified in the .torchxconfig.

    Args:
        script_args: additional args to pass through to the script
        script: defaults to train.py, but you can run a different script
        module: if provided, run Python module instead of script
        sweep: name of yaml file for parameters
        sweep_index: in case there are mulitple possible runs, choose the run to kick off
        nodes: total hosts to use
        nproc_per_node: processes per node
        name: custom name for this job
        h: hardware to use, eg. t1, tc_any, etc.
        env: custom environment parameters to pass through
        unset_env: environment parameters to unset/delete
        retry_policy: as title
        run_as_root: run the job as root; should be set to true for mounting
        dump_dir_id: Explicitly specify an mast job to continue training (defaults to new job id)
        xzone: enable cross zone jobs
        dump_logs: save logs to dump dir as well under "<dump dir>/logs"
        additional_libraries: copy these folders into torchtitan_additional_packages
            and add them to python path
        additional_folders: copy these folders into the fbpkg torchtitan_additional_packages
        additional_python_paths: add these paths to $PYTHONPATH before executing
        py_spy_startup: trace script startup; see scripts/mast/py_spy_init.sh for configuration
        retries: number of times to retry the job before failing completely
        twtask_bootstrap_script: shell script that is run on each tw task which bootstraps the real training script
        enable_ttls: enable ttls for the job
    """

    if xzone:
        logger.error(
            "The --xzone parameter has been moved from component args to scheduler_args;"
            'Please set it with --scheduler_args="xzone=True"!'
        )
        exit(1)

    # Set up the environment variables
    mast_env = dict(_DEFAULT_ENV)
    username = getpass.getuser()
    run_script = (
        "${img_root}/" + twtask_bootstrap_script
        if twtask_bootstrap_script
        else _RUN_SCRIPT
    )
    gpu_num = nodes * nproc_per_node
    mast_job_name = f"{name}-{gpu_num}-{username}"

    if enable_ttls:
        mast_env.update(WITH_PROXY_ENV_VARS)

    if env:
        mast_env.update(env)

    if unset_env:
        for env_var in unset_env:
            mast_env.pop(env_var, None)

    if not mast_env.get("ENABLE_AIRSTORE"):
        mast_env["FUSE_SRC"] = "ws://ws.ai.nha0genai/checkpoint/infra"
        mast_env["FUSE_SRC_PATH"] = "checkpoint/infra"

    # Ensure that a dump dir is available
    dump_mount = Path(mast_env["FUSE_DST"])
    dump_dir = dump_mount / "outputs" / dump_dir_id

    # Make the dump dir available for shell scripts
    mast_env["DUMP_DIR"] = str(dump_dir)
    mast_env["JOB_ID"] = dump_dir_id

    # Dependencies libraries for picking up latest site package
    additional_python_paths = additional_python_paths or []
    additional_folders = additional_folders or []
    additional_libraries = additional_libraries or []
    additional_pkg = None

    if mast_env["ENABLE_AIRSTORE"]:
        additional_python_paths.append("/packages/ws_airstore.client/lib")

    if additional_libraries or additional_folders:
        additional_folders.extend(additional_libraries)
        additional_pkg = _make_fbpkg(additional_folders)
        for folder in additional_libraries:
            additional_python_paths.append(
                f"/packages/{_ADDITIONAL_PACKAGES_FBPKG_NAME}/{os.path.basename(folder.rstrip('/'))}"
            )

    mast_env["TORCHX_RUN_PYTHONPATH"] = ":".join(additional_python_paths)

    # Set up arguments for fb.conda.torchrun
    kwargs = {
        "name": mast_job_name,
        "h": h,
        "env": mast_env,
        "retry_policy": retry_policy,
        "run_as_root": run_as_root,
        "enable_ttls": enable_ttls,
        "max_retries": retries,
        "conda_mast_core_fbpkg_id": "conda_mast_core:stable",
    }
    for key in list(kwargs.keys()):
        if kwargs[key] is None:
            kwargs.pop(key)

    # Construct arguments per sweep
    full_args = [
        [
            "--tee",
            "3",
            "--nnodes",
            str(nodes),
            "--nproc-per-node",
            str(nproc_per_node),
            "--role",
            "trainer",
            "--no-python",
            run_script,
            # f"-m{module}" if module else script,
            *script_args,
        ]
    ]

    job_spec = conda.torchrun(*full_args[0], **kwargs)
    job_spec = conda_transforms.append_tb_logdir_metadata(job_spec)

    inner_entrypoint = job_spec.roles[0].entrypoint
    tee_script = _TEE_SCRIPT if dump_logs else ""
    py_spy_script = "$WORKSPACE_DIR/" + _PY_SPY_SCRIPT if py_spy_startup else ""
    entrypoint = f"{_MOUNT_SCRIPT} && {tee_script} {py_spy_script} {inner_entrypoint}"

    print(f"{entrypoint=}")

    job_spec.roles[0].entrypoint = entrypoint

    packages = [
        job_spec.roles[0].image,
        "folly.symbolizer:stable",
        "manifold.manifoldfs",
        "oil.oilfs:stable",
        "fb-py-spy:prod",
    ]
    if additional_pkg:
        packages.append(additional_pkg)

    if mast_env.get("ENABLE_AIRSTORE") == "1":
        packages.append(mast_env.get("AIRSTORE_FBPKG_ID", _DEFAULT_AIRSTORE_PACKAGE))

    job_spec.roles[0].image = ";".join(packages)
    return job_spec


def train_monarch(
    *script_args: str,
    script: str = "test.py",
    module: Optional[str] = None,
    nodes: int = 2,
    nproc_per_node: int = 8,
    name: str = "monarch_titan",
    h: str = "t1",
    env: Optional[Dict[str, str]] = None,
    unset_env: Optional[List[str]] = None,
    retry_policy: Optional[str] = None,
    run_as_root: bool = True,  # Need to mount
    dump_dir_id: str = "${app_id}",
    xzone: bool = False,
    dump_logs: bool = True,
    additional_libraries: Optional[List[str]] = None,
    additional_folders: Optional[List[str]] = None,
    additional_python_paths: Optional[List[str]] = None,
    py_spy_startup: bool = False,
    retries: int = 1,
    twtask_bootstrap_script: Optional[str] = None,
) -> specs.AppDef:
    """
    Kick off a mast job that can run a monarch hyperactor that is capable of running torchtitan actors.

    To do an actual monarch+torchtitan training run, it involves running this job on mast and then
    running a monarch controller (which can be done in a bento notebook) to schedule torchtitan trainers
    that are running as actors on this job.

    This can enable very rapid iteration of code without relaunching mast jobs.

    For a full reproducible demo, you can look at the test plan for this diff:
    D77302505
    """
    additional_python_paths = (additional_python_paths or []) + [_MONARCH_LIB_LOCATION]

    additional_python_paths += [
        "/packages/torchtitan_conda_prod/conda/lib/python3.10/site-packages",
        f"/packages/{_HYPERACTOR_FBPKG}/conda/lib/python3.10/site-packages",
        "/packages/torchtitan_additional_packages/torchtitan",
    ]

    job_spec = train(
        *script_args,
        script=script,
        module=module,
        nodes=nodes,
        nproc_per_node=nproc_per_node,
        name=name,
        h=h,
        env=env,
        unset_env=unset_env,
        retry_policy=retry_policy,
        run_as_root=run_as_root,
        dump_dir_id=dump_dir_id,
        xzone=xzone,
        dump_logs=dump_logs,
        additional_libraries=additional_libraries,
        additional_folders=additional_folders,
        additional_python_paths=additional_python_paths,
        py_spy_startup=py_spy_startup,
        retries=retries,
        twtask_bootstrap_script=twtask_bootstrap_script,
    )

    is_actor_model = True

    # Monarch specific additions here.
    packages = [
        "ttls_so:stable",
        "monarch_torchtitan:latest_contbuild",
    ]
    job_spec.roles[0].image += ";" + ";".join(packages)

    role_template = job_spec.roles[0]
    role_template.entrypoint = (
        "$WORKSPACE_DIR/mount.sh &&"
        " ls /packages/torchtitan_additional_packages &&"
        f" MONARCH_DIR=/packages/torchtitan_additional_packages/monarch {_HYPERACTOR_MAST_BOOTSTRAP}"
    )
    role_template.args = []

    if is_actor_model:
        role_template.port_map.update({_TW_MESH_WORKER_PORT_NAME: _get_system_ports()})

    task_groups, num_nodes = _create_monarch_worker_mesh_templates(
        role_template, is_actor_model, task_group_map={"worker_mesh_0": nodes}
    )

    job_spec.roles = task_groups

    return job_spec


def _get_system_ports() -> int:
    return _get_core_ports()[_TW_SYSTEM_PORT_NAME]


def _get_core_ports():
    return _MAST_DEFAULT_CORE_PORTS


def _create_role_args_with_role_name(role, role_name):
    role_args = copy.deepcopy(role.args)
    role_args_name_index = role_args.index("--role") + 1
    role_args[role_args_name_index] = role_name
    return role_args


def _create_monarch_worker_mesh_templates(
    role_template: specs.Role,
    is_actor_model: bool,
    task_group_map: Optional[Dict[str, int]] = None,
    num_meshes: int = -1,
    nproc_per_node: int = 8,
    nodes: int = 1,
    h: str = "gtt_any",
) -> Tuple[List[specs.Role], int]:
    if task_group_map is not None and num_meshes != -1:
        raise RuntimeError("Cannot specify both task_group_map and num_meshes")
    if task_group_map is None:
        if num_meshes == -1:
            logger.info(
                "Assuming num_meshes=2 given that is currently the default for post-training workflows in monarch"
            )
            num_meshes = 2
        task_group_map = {
            f"worker_mesh_{replica_num}": nodes // num_meshes
            for replica_num in range(num_meshes)
        }
    # else: task_group_map provided by the config
    assert task_group_map
    assert (
        len(set(task_group_map.values())) == 1
    ), f"All task groups must set the same number of nodes {task_group_map=}"

    worker_meshes = []
    for task_group_name, num_nodes in task_group_map.items():
        worker_template = copy.deepcopy(role_template)

        # create the worker role
        name_prefix = "[${MAST_HPC_TASK_GROUP_NAME}/${rank}|${local_rank}]:"
        worker_template.name = task_group_name
        worker_template.env["TORCHELASTIC_LOG_LINE_PREFIX_TEMPLATE"] = name_prefix

        # split nodes evenly if we are not specified
        if num_nodes == -1:
            num_nodes = nodes // len(task_group_map)
        assert (
            num_nodes >= 1
        ), f"Must have at least 1 node per task group {task_group_name=}"

        worker_template.args = _get_monarch_worker_task_group_args(
            worker_template.name,
            num_nodes,
            nproc_per_node,
            named_resources[h].gpu == 0,
            is_actor_model,
        )
        worker_template.num_replicas = num_nodes

        worker_meshes.append(worker_template)

    return worker_meshes, num_nodes


def _get_monarch_worker_task_group_args(
    world: str,
    num_hosts: int,
    nproc_per_node: int,
    cpu_worker: bool,
    is_actor_model: bool,
) -> List[str]:
    if is_actor_model:
        args = [
            f"--num-hosts={num_hosts}",
            "mesh-worker",
            f"--port=%port.{_TW_MESH_WORKER_PORT_NAME}%",
            "--program=/packages/torchtitan_additional_packages/monarch/python/monarch/meta/bin/monarch_bootstrap.sh",
        ]
        return args

    args = [f"--num-hosts={num_hosts}", f"--system-port=%port.{_TW_SYSTEM_PORT_NAME}%"]

    args.extend(
        [
            "worker",
            f"--controller-actor-id={world}_controller[0].controller[0]",
            f"--num-procs-per-host={nproc_per_node}",
            f"--world={world}_worker",
            f"--host-world=host{world}_worker",
            "--program=/packages/xlformers_pretrain1/projects/monarch/pretrain/scripts/run_monarch_worker.sh",
        ]
    )

    if cpu_worker:
        logger.warning("Launching monarch worker task group with CPU-only workers")
        args.append("--is-cpu-worker")

    return args


def _get_monarch_core_task_group_args(
    num_hosts: int,
    script: Optional[str],
    script_args: List[str],
    system_port: Optional[int],
) -> List[str]:
    args = [
        f"--num-hosts={num_hosts}",
    ]

    if system_port is not None:
        args.append(f"--system-port={system_port}")

    args.append("system")

    if script:
        args.extend(
            [
                "--main-script",
                "/packages/conda_mast_core/run/torchx_run.sh",
                script,
            ]
        )
        args += script_args

    return args


def _smc_bridge_tier(name):
    return f"mast.monarch.{name}-{_TW_SYSTEM_PORT_NAME}"


def _args_dict_to_args_list(swept_params: Dict[str, Any]) -> List[str]:
    args_list = []

    for key, value in swept_params.items():
        if value is None:  # to avoid setting optional parameters
            continue

        v = (
            json.dumps(value)
            if type(value) is list or type(value) is dict
            else str(value)
        )
        args_list.append(f"--{key}={v}")

    return args_list


def _make_fbpkg(paths: List[str]) -> str:
    """
    Temporarily here until this is natively supported by torchx.
    """
    from torchx.workspace.fb import fbpkg_utils

    return fbpkg_utils.build_fbpkg(
        fbpkg_name=_ADDITIONAL_PACKAGES_FBPKG_NAME,
        paths=paths,
        expiration="4w",
    )


def train_ft(
    *script_args: str,
    groups: int = 2,
    nodes: int = 2,
    nproc_per_node: int = 8,
    name: str = "torchft",
    h: str = "grandteton_8",
    env: Optional[Dict[str, str]] = None,
    retries: int = 3,
    additional_folders: Optional[List[str]] = None,
    twtask_bootstrap_script: Optional[str] = None,
    smc_tier_base: str = "torch.ft.lighthouse",
    num_fragments: int = 1,
    semi_sync_method: Optional[str] = None,  # diloco | local_sgd
    model_name: Optional[str] = None,
    process_group: str = "gloo",
) -> specs.AppDef:
    env = env or {}

    # set SMC tier name for replica groups to find lighthouse
    smc_tier = f"{smc_tier_base}.{random_id()}"
    env["LIGHTHOUSE_SMC_TIER"] = smc_tier

    # use agent store in torchelastic to avoid TCPStore init race condition
    env["TORCH_SHARE_RDZV_TCP_STORE"] = "1"
    env["TORCH_CPP_LOG_LEVEL"] = "INFO"
    env["TORCH_CUDA_SANITIZER=1"] = "1"

    # skipping for faster development
    env["MAST_PRECHECK_SKIP"] = "1"

    # NCCL envs for debugging
    env["NCCL_DEBUG"] = "WARN"
    env["NCCL_DEBUG_SUBSYS"] = "ALL"

    env["TORCHFT_QUORUM_TIMEOUT_SEC"] = "900"
    env["TORCHFT_TIMEOUT_SEC"] = "600"
    env["TORCHFT_QUORUM_RETRIES"] = "0"

    # application log levels
    env["LOGLEVEL"] = "INFO"
    env["RUST_LOGS"] = "INFO"

    app = train(
        h=h,
        nodes=1,
        nproc_per_node=1,
        name=f"{name}-{groups}x{nodes}x{nproc_per_node}",
        env=env,
        retries=retries,
        additional_folders=additional_folders,
        twtask_bootstrap_script="run_lighthouse.sh",
        run_as_root=True,
    )
    app.roles[0].name = "lighthouse"

    lighthouse_role = app.roles[0]
    assert len(lighthouse_role.metadata) == 0

    lighthouse_role.metadata = {
        "mast": {
            "HpcTaskGroupSpec": {
                "smcBridge": {
                    "smcTier": smc_tier,
                    "portName": "lighthouse",
                },
            }
        }
    }
    lighthouse_role.port_map["lighthouse"] = 29510

    # only package the app once
    train_app = train(
        *script_args,
        h=h,
        nodes=nodes,
        nproc_per_node=nproc_per_node,
        name=name,
        env=env,
        additional_folders=additional_folders,
        twtask_bootstrap_script=twtask_bootstrap_script,
        run_as_root=True,
    )
    train_role = train_app.roles[0]
    train_role.image += ";torchft_smcc:stable"
    for i in range(groups):
        role = copy.deepcopy(train_role)
        role.name = f"replica_group_{i}"
        role.retry_policy = specs.RetryPolicy.ROLE
        role.max_retries = retries
        role.args += [
            "--training.seed=1",
            "--comm.trace_buf_size=0",
            "--metrics.log_freq=1",
            "--training.local_batch_size=2",
            "--profiling.enable_profiling",
            "--training.steps=10000",
            "--experimental.custom_args_module=torchtitan.components.ft.config",
        ]

        if model_name:
            role.args += [f"--model.name={model_name}"]

        data_parallel_shard_degree = nproc_per_node * nodes

        role.args += [
            "--fault_tolerance.enable",
            f"--fault_tolerance.process_group={process_group}",
            f"--fault_tolerance.group_size={groups}",
            f"--fault_tolerance.replica_id={i}",
            f"--fault_tolerance.process_group_timeout_ms={600 * 1000}",
            f"--parallelism.data_parallel_shard_degree={data_parallel_shard_degree}",
        ]

        if semi_sync_method is not None:
            role.args += [
                f"--fault_tolerance.semi_sync_method={semi_sync_method}",
            ]

        if semi_sync_method == "diloco":
            role.args += [
                "--fault_tolerance.sync_steps=20",
                "--fault_tolerance.fragment_sync_delay=1",
                f"--fault_tolerance.num_fragments={num_fragments}",
            ]

        app.roles.append(role)

    return app
