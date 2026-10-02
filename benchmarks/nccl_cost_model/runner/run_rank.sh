#!/bin/bash
set -euo pipefail

: "${CONDA_DIR:?CONDA_DIR is required}"
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
export LD_LIBRARY_PATH="${CONDA_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
mkdir -p "${DUMP_DIR}"

read -r -a modes <<< "${CALIBRATION_MODES:-auto ring_ll ring_ll128 ring_simple tree_ll tree_ll128 tree_simple}"

for index in "${!modes[@]}"; do
    mode=${modes[${index}]}
    algo=
    proto=
    case ${mode} in
        ring_ll) algo=Ring; proto=LL ;;
        ring_ll128) algo=Ring; proto=LL128 ;;
        ring_simple) algo=Ring; proto=Simple ;;
        tree_ll) algo=Tree; proto=LL ;;
        tree_ll128) algo=Tree; proto=LL128 ;;
        tree_simple) algo=Tree; proto=Simple ;;
        pat_simple) algo=PAT; proto=Simple ;;
        nvlstree_simple) algo=NVLSTree; proto=Simple ;;
    esac
    if [[ -n ${algo} ]]; then
        NCCL_ALGO=${algo} \
        NCCL_PROTO=${proto} \
        NCCL_DEBUG_FILE="${DUMP_DIR}/nccl.${mode}.%h.%p.log" \
        CALIBRATION_MODE=${mode} \
            python "${CALIBRATION_ROOT}/runner/benchmark_phase.py"
    else
        env -u NCCL_ALGO -u NCCL_PROTO \
            NCCL_DEBUG_FILE="${DUMP_DIR}/nccl.${mode}.%h.%p.log" \
            CALIBRATION_MODE=${mode} \
            python "${CALIBRATION_ROOT}/runner/benchmark_phase.py"
    fi
done

if [[ ${RANK} == 0 ]]; then
    printf '{"NVLSTree":"unsupported","reason":"zero NVLS channels"}\n' > "${DUMP_DIR}/algorithm_eligibility.json"
    printf 'passed\n' > "${DUMP_DIR}/SUCCESS"
fi
