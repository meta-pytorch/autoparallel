# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import importlib.util
import logging
import os
import re
import subprocess
from typing import Optional

import git

logger = logging.getLogger(__name__)


def print_tbm(results: dict[str, str]) -> None:
    tbm_str = " ".join(f"{name}:{job_id}" for name, job_id in results.items())
    print(f"tbm {tbm_str}")


def get_git_hash(repo_path):
    try:
        repo = git.Repo(repo_path)
        latest_commit = repo.head.commit
        return latest_commit.hexsha
    except Exception as e:
        logger.error(f"Error accessing Git repository at {repo_path}: {e}")
        return None


def maybe_tabulate(data, headers=()):
    if importlib.util.find_spec("tabulate"):
        from tabulate import tabulate

        return tabulate(data, headers=headers)
    return f"Please pip install `tabulate` for better printing\n{headers}\n{data}"


def is_git_repo_clean(repo_path):
    try:
        repo = git.Repo(repo_path)
        # Check for unstaged changes (modified, added, deleted)
        if repo.is_dirty(untracked_files=True):
            return False
        # Check for staged but uncommitted changes
        if repo.index.diff(None):
            return False
        return True
    except git.InvalidGitRepositoryError:
        logger.error(f"Error: '{repo_path}' is not a valid Git repository.")
        return False
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return False


def find_repo(path: str, name: str) -> str:
    try:
        # error if not a repo or if not a valid path
        _ = git.Repo(path)
        assert os.path.exists(os.path.join(path, name))
        return os.path.abspath(path)
    except Exception:
        logger.error(f"Failed to find {name} repo, pass valid path as argument.")
        raise


def find_torchtitan(maybe_path: Optional[str] = None) -> str:
    return find_repo(maybe_path or "../../torchtitan", "torchtitan")


def find_autoparallel(maybe_path: Optional[str] = None) -> str:
    return find_repo(maybe_path or "../", "autoparallel")


def maybe_find_pulp(maybe_path: Optional[str] = None) -> Optional[str]:

    try:
        return find_repo(maybe_path or "../../pulp", "pulp")
    except Exception:
        logger.error(
            "Failed to find pulp repo, will not include it in the run. "
            "This is OK if the fbpkg itself includes pulp, which should be true for latest nightly"
        )
    return None


llama3_1d = {
    "llama3_FSDP_compile": [
        "--model.name=llama3",
        "--training.compile",
        "--parallelism.tensor_parallel_degree=1",
    ],
    "llama3_autop_1d_compile": [
        "--model.name=llama3_auto_parallel",
        "--training.compile",
        "--parallelism.tensor_parallel_degree=1",
    ],
    "llama3_autop_1d_compile_bucket_reorder": [
        "--model.name=llama3_auto_parallel",
        "--training.compile",
        "--parallelism.tensor_parallel_degree=1",
        "--experimental.bucket_all_gathers_fx=fsdp",
        "--experimental.bucket_reduce_scatters_fx=fsdp",
        "--experimental.reorder_for_compute_comm_overlap",
    ],
}

llama3_2d = {
    "llama3_FSDP_tp_compile": [
        "--model.name=llama3",
        "--training.compile",
        "--parallelism.tensor_parallel_degree=8",
    ],
    "llama3_autop_2d_compile": [
        "--model.name=llama3_auto_parallel",
        "--training.compile",
        "--parallelism.tensor_parallel_degree=8",
    ],
    "llama3_autop_2d_compile_bucket_reorder": [
        "--model.name=llama3_auto_parallel",
        "--training.compile",
        "--parallelism.tensor_parallel_degree=8",
        "--experimental.bucket_all_gathers_fx=fsdp",
        "--experimental.bucket_reduce_scatters_fx=fsdp",
        "--experimental.reorder_for_compute_comm_overlap",
    ],
}

test_run = {
    "FSDP_tp_compile": [
        "--model.name=llama3",
        "--training.compile",
        "--parallelism.tensor_parallel_degree=8",
    ],
}

sweeps = {
    "llama3_1d": llama3_1d,
    "llama3_2d": llama3_2d,
}
all_runs = (
    llama3_1d
    | llama3_2d
    | {
        "llama3_autop_1d_compile_ruisi_bucket_reorder": [
            "--model.name=llama3_auto_parallel",
            "--training.compile",
            "--parallelism.tensor_parallel_degree=1",
            "--experimental.enable_simplefsdp_passes",
        ],
        "llama3_autop_2d_compile_ruisi_bucket_reorder": [
            "--model.name=llama3_auto_parallel",
            "--training.compile",
            "--parallelism.tensor_parallel_degree=8",
            "--experimental.enable_simplefsdp_passes",
        ],
    }
)


def run(args: argparse.Namespace) -> None:

    if args.runs:
        runs = {name: all_runs[name] for name in args.runs}
    else:
        runs = {}
        for sweep in args.sweep:
            runs.update(sweeps[sweep])

    # overrides values in .torchxconfig
    scheduler_args = ",".join([f"conda_fbpkg_id={args.fbpkg}"])

    base_cmd = [
        "torchx",
        "run",
        f"--scheduler_args={scheduler_args}",
        "mast.py:train",
        "--nodes",
        f"{args.nodes}",
        "--additional_folders",
        args.torchtitan_dir,
        "--twtask_bootstrap_script",
        "run_torchtitan.sh",
    ]
    addl_libs_str = ",".join(
        [
            args.autoparallel_dir,
        ]
        + [args.pulp_dir]
        if args.pulp_dir
        else []
    )
    addl_libs = [f"--additional_libraries={addl_libs_str}"]
    llama3_base = [
        "torchtitan/models/llama3/train_configs/llama3_8b.toml",
        "--training.dataset",
        "c4",
    ]

    def launch_job(cmd: list[str]) -> str:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            # todo move the checking to later, print stdout/err first
            check=True,
        )
        job_id_pattern = r".*runs\/mast\/([a-zA-Z0-9\-]+)"
        for line in result.stdout.splitlines() + result.stderr.splitlines():
            if m := re.match(job_id_pattern, line):
                return m.group(1)

        raise RuntimeError(
            f"Failed to find job id in torchx launch output. Full stdout:\n {result.stdout}"
        )

    results = {}
    autoparallel_hash = get_git_hash(args.autoparallel_dir)
    autoparallel_clean = is_git_repo_clean(args.autoparallel_dir)
    torchtitan_hash = get_git_hash(args.torchtitan_dir)
    torchtitan_clean = is_git_repo_clean(args.torchtitan_dir)

    if not torchtitan_clean or not autoparallel_clean:
        logger.warning(
            f"Repo is not clean. Please commit your changes before running the script.  {autoparallel_clean=} {torchtitan_clean=}"
        )

    extra_torchtitan_args = args.extra_torchtitan_args or []
    extra_torchtitan_name = "_".join(extra_torchtitan_args)
    extra_torchtitan_args = ["--" + arg for arg in extra_torchtitan_args]
    for name, sub_cmd in runs.items():
        if extra_torchtitan_name:
            name += "_" + extra_torchtitan_name
        logger.info(f"Launching {name}")
        cmd = base_cmd + addl_libs + llama3_base + sub_cmd + extra_torchtitan_args
        if args.dry_run:
            # TODO configure log levels..
            logger.warning(f"Dry-run: command for {name} is\n" + " ".join(cmd))
            job_id = "dry-run"
        else:
            job_id = launch_job(cmd)
        results[name] = job_id

    print("")
    print(
        maybe_tabulate(
            [
                ["fbpkg", args.fbpkg, "n/a"],
                ["autoparallel", autoparallel_hash, autoparallel_clean],
                ["torchtitan", torchtitan_hash, torchtitan_clean],
            ],
            headers=["Repo", "Hash", "Is Clean"],
        )
    )
    print("")
    print(maybe_tabulate(results.items(), headers=["Name", "Job ID"]))
    print("")
    print("tbm command:\n")
    print_tbm(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch autoparallel runs from a stable configuration. Run from autoparallel/scripts dir."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show the commands that would be run, don't actually run them",
    )
    parser.add_argument(
        "--torchtitan_dir",
        type=find_torchtitan,
        default=find_torchtitan(),
        help="Path to torchtitan repo",
    )
    parser.add_argument(
        "--autoparallel_dir",
        type=find_autoparallel,
        default=find_autoparallel(),
        help="Path to autoparallel repo",
    )
    parser.add_argument(
        "--pulp_dir",
        type=maybe_find_pulp,
        default=maybe_find_pulp(),
        help="Path to pulp repo, not strictly required but recommended since not all fbpkgs include pulp dep",
    )
    parser.add_argument(
        "--fbpkg",
        default="torchtitan_conda_prod:latest_conveyor_build",
        help="Fbpkg to use for job",
    )
    parser.add_argument(
        "--sweep",
        choices=sweeps.keys(),
        default="llama3_1d",
        nargs="+",
        help="Sweep to run, if not specified will run only specified runs",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        choices=all_runs.keys(),
        help="exact list of runs to run, overrides sweep",
    )
    parser.add_argument(
        "--extra_torchtitan_args",
        nargs="+",
        help="arguments to pass to torchtitan, e.g. 'training.batch_size=2'",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=8,
        help="How many nodes to use for the job, defaults to 8.",
    )

    args = parser.parse_args()
    run(args)
