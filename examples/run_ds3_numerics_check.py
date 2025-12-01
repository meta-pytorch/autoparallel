# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
Script to run DS3 numerics check by comparing outputs from local_map and pipeline parallel.
"""
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path


def run_command(cmd, cwd):
    """Run a shell command in the specified directory."""
    print(f"Running: {cmd}")
    print(f"In directory: {cwd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    if result.returncode != 0:
        warnings.warn(f"Command failed with return code {result.returncode}")
    return result


def main(args):
    schedule_name = args.schedule_name

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp(prefix="ds3_numerics_check_")
    print(f"Created temporary directory: {temp_dir}")

    try:
        examples_dir = Path(__file__).parent

        print("\n" + "=" * 80)
        print("Running non-PP example with 4 GPUs...")
        print("=" * 80)
        cmd1 = f"torchrun --standalone --nproc-per-node 4 {examples_dir}/example_ds3_local_map.py --rng-seed 42"
        run_command(cmd1, temp_dir)

        print("\n" + "=" * 80)
        print("Running PP example with 8 GPUs...")
        print("=" * 80)
        cmd2 = f"torchrun --standalone --nproc-per-node 8 {examples_dir}/example_ds3_pp.py --rng-seed 42 --schedule-name={schedule_name}"
        run_command(cmd2, temp_dir)

        out_dir = Path(temp_dir) / "out"
        if not out_dir.exists():
            raise RuntimeError(f"Output directory {out_dir} does not exist")

        print("\n" + "=" * 80)
        print("Comparing weights.log files...")
        print("=" * 80)
        run_command("diff out/0/weights.log out/1/pp_weights.log", temp_dir)

        print("\n" + "=" * 80)
        print("Comparing diff.log files...")
        print("=" * 80)
        run_command("diff out/0/diff.log out/1/diff.log", temp_dir)

        print("\n" + "=" * 80)
        print("Numerics check completed successfully!")
        print(f"Output directory: {temp_dir}/out")
        print("=" * 80)

    except Exception as e:
        print(f"\nError occurred: {e}")
        print(f"Temporary directory preserved at: {temp_dir}")
        raise

    print(f"\nTemporary directory location: {temp_dir}")
    response = input("Do you want to delete the temporary directory? (y/n): ")
    if response.lower() == "y":
        shutil.rmtree(temp_dir)
        print("Temporary directory deleted.")
    else:
        print(f"Temporary directory preserved at: {temp_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run DeepSeek V3 pipeline parallel example"
    )
    parser.add_argument(
        "--schedule-name",
        type=str,
        default="ZBVZeroBubble",
        help="Schedule to use for PP",
    )
    args = parser.parse_args()
    main(args)
