#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
#
# Run the nccl-tests sweep behind a MeasuredCommTable: every collective
# AutoParallel prices, on every (n_nodes, ppn) key in the grid, one process
# per GPU. See docs/measured_comm_tables.md for the full procedure.
#
# Run it inside an exclusive Slurm allocation of the largest node count:
#
#   NCCL_TESTS_BUILD=/path/to/nccl-tests/build sbatch --nodes=32 \
#       --gpus-per-node=8 --ntasks-per-node=8 --exclusive \
#       autoparallel/tools/nccl_tests_sweep.sh
#
# Environment variables:
#   NCCL_TESTS_BUILD  nccl-tests build directory (MPI=1 build), required
#   OUT_DIR           output directory (default: nccl_tests_results)
#   GPUS_PER_NODE     GPUs per node (default: 8)
#   NODE_COUNTS       node counts to measure (default: 1 2 4 ... allocation)
#   PPNS              ranks per node to measure (default: 1 2 4 ... GPUS_PER_NODE)
#   BINARY_SUFFIX     suffix of the nccl-tests binaries (default: _mpi)
#   NCCL_TESTS_ARGS   per-run arguments (default: -b 8 -e 8G -f 2 -w 5 -n 20)
#   RUN_TAG           appended to file names to keep repeated sweeps apart
#   DRY_RUN           if set, print the commands instead of running them

set -euo pipefail

OUT_DIR=${OUT_DIR:-nccl_tests_results}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MAX_NODES=${SLURM_NNODES:-1}
BINARY_SUFFIX=${BINARY_SUFFIX:-_mpi}
NCCL_TESTS_ARGS=${NCCL_TESTS_ARGS:-"-b 8 -e 8G -f 2 -w 5 -n 20"}

powers_of_two_up_to() {
    local n=1 out=""
    while ((n <= $1)); do
        out+="$n "
        n=$((n * 2))
    done
    echo "$out"
}
NODE_COUNTS=${NODE_COUNTS:-$(powers_of_two_up_to "$MAX_NODES")}
PPNS=${PPNS:-$(powers_of_two_up_to "$GPUS_PER_NODE")}

if [[ -z "${DRY_RUN:-}" ]]; then
    : "${NCCL_TESTS_BUILD:?set NCCL_TESTS_BUILD to the nccl-tests build directory}"
    mkdir -p "$OUT_DIR"
fi

for test in all_gather_perf reduce_scatter_perf all_reduce_perf alltoall_perf; do
    for nodes in $NODE_COUNTS; do
        for ppn in $PPNS; do
            # A single rank has nothing to communicate with.
            if ((nodes == 1 && ppn == 1)); then
                continue
            fi
            out="$OUT_DIR/${test%_perf}_${nodes}n_${ppn}ppn${RUN_TAG:+_$RUN_TAG}.txt"
            # Set --ntasks explicitly: the allocation exports its own task count.
            cmd=(srun --nodes="$nodes" --ntasks="$((nodes * ppn))"
                --ntasks-per-node="$ppn" --gpus-per-node="$ppn"
                "${NCCL_TESTS_BUILD:-nccl-tests/build}/${test}${BINARY_SUFFIX}" -g 1
                $NCCL_TESTS_ARGS)
            if [[ -n "${DRY_RUN:-}" ]]; then
                echo "${cmd[*]} > $out"
            else
                echo "$test: $nodes nodes x $ppn ranks/node -> $out"
                "${cmd[@]}" >"$out"
            fi
        done
    done
done
