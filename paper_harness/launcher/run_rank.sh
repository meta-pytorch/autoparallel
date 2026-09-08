#!/bin/bash
set -euo pipefail

: "${CONDA_DIR:?CONDA_DIR is required}"
: "${HARNESS_PAYLOAD_ROOT:?HARNESS_PAYLOAD_ROOT is required}"

set +u
source "${CONDA_DIR}/bin/activate"
set -u

platform=platform010
if [[ "$(uname -m)" == aarch64 ]]; then
    platform=platform010-aarch64
fi
libcuda_dir=/usr/local/fbcode/${platform}/lib
export LIBCUDA_DIR=${libcuda_dir}
export TRITON_LIBCUDA_PATH=${libcuda_dir}
export LD_PRELOAD="${PRELOAD_PATH:-${libcuda_dir}/libcuda.so:${libcuda_dir}/libnvidia-ml.so:${libcuda_dir}/libnvidia-ptxjitcompiler.so}"

nvshmem_lib_dir=
for candidate in "${CONDA_DIR}"/lib/python3.*/site-packages/nvidia/nvshmem/lib; do
    if [[ -e "${candidate}/libnvshmem_host.so.3" ]]; then
        nvshmem_lib_dir=${candidate}
        break
    fi
done
export LD_LIBRARY_PATH="${CONDA_DIR}/lib${nvshmem_lib_dir:+:${nvshmem_lib_dir}}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${HARNESS_PAYLOAD_ROOT}/harness_repo:${HARNESS_PAYLOAD_ROOT}/autoparallel:${HARNESS_PAYLOAD_ROOT}/torchtitan${TORCHX_RUN_PYTHONPATH:+:${TORCHX_RUN_PYTHONPATH}}${PYTHONPATH:+:${PYTHONPATH}}"

exec python -m harness.runtime
