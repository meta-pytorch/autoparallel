#!/bin/bash

set -euo pipefail

if [[ $# != 6 ]] || [[ $1 != gate && $1 != formal ]] || [[ $4 != single_native ]]; then
    echo "Usage: $0 {gate|formal} 16 2 single_native sequential REPLICATE" >&2
    exit 2
fi
run_mode=$1
world_size=$2
tp_degree=$3
order=$5
replicate=$6
case ${world_size}:${tp_degree} in
    16:2) ;;
    *) echo "Unsupported mesh ${world_size}:${tp_degree}" >&2; exit 2 ;;
esac
[[ ${order} == sequential ]] || exit 2
[[ ${replicate} =~ ^[0-9]+$ ]] || exit 2

fsdp_degree=$((world_size / tp_degree))
node_count=$((world_size / 8))
package_root=/packages/torchtitan_additional_packages
runner_root=${package_root}/runner
tt_root=${package_root}/torchtitan
ap_root=${package_root}/autoparallel
c4_hub_cache=${package_root}/c4_hf_cache

: "${CONDA_DIR:?CONDA_DIR is required}"
: "${DUMP_DIR:?DUMP_DIR is required}"
: "${RANK:?RANK is required}"
: "${LOCAL_RANK:?LOCAL_RANK is required}"
: "${MAST_HPC_JOB_VERSION:?MAST_HPC_JOB_VERSION is required}"
: "${MAST_HPC_JOB_ATTEMPT_INDEX:?MAST_HPC_JOB_ATTEMPT_INDEX is required}"

set +u
source "${CONDA_DIR}/bin/activate"
set -u

platform=platform010
if [[ "$(uname -m)" == aarch64 ]]; then
    platform=platform010-aarch64
fi
libcuda=/usr/local/fbcode/${platform}/lib/libcuda.so
libcuda_dir=${libcuda%/*}
export LIBCUDA_DIR=${libcuda_dir}
export TRITON_LIBCUDA_PATH=${libcuda_dir}
export LD_PRELOAD="${PRELOAD_PATH:-${libcuda}:${libcuda_dir}/libnvidia-ml.so:${libcuda_dir}/libnvidia-ptxjitcompiler.so}"

nvshmem_lib_dir=
for candidate in "${CONDA_DIR}"/lib/python3.*/site-packages/nvidia/nvshmem/lib; do
    if [[ -e "${candidate}/libnvshmem_host.so.3" ]]; then
        nvshmem_lib_dir=${candidate}
        break
    fi
done
export LD_LIBRARY_PATH="${CONDA_DIR}/lib${nvshmem_lib_dir:+:${nvshmem_lib_dir}}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${runner_root}:${ap_root}:${tt_root}:${TORCHX_RUN_PYTHONPATH:?TORCHX_RUN_PYTHONPATH is required}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export TORCH_DISABLE_ADDR2LINE=1
export HF_HUB_DISABLE_XET=1
export LOG_RANK=0
export REQUIRED_PCI_DOMAIN=pci1
export EXPECTED_WORLD_SIZE=${world_size}
export EXPECTED_LOCAL_WORLD_SIZE=8
export EXPECTED_NODE_COUNT=${node_count}
export EXPECTED_FSDP_DEGREE=${fsdp_degree}
export EXPECTED_TP_DEGREE=${tp_degree}
export EXPECTED_AP_ROOT=${ap_root}
export EXPECTED_TT_ROOT=${tt_root}
export EXPECTED_RUNNER_ROOT=${runner_root}
export HF_HUB_CACHE=${c4_hub_cache}
export HF_HUB_OFFLINE=1
export RUN_ROOT=${DUMP_DIR}/muse_glimmer_30b_native_inductor_lb2_s4096_${node_count}x8_${run_mode}_r${replicate}_v${MAST_HPC_JOB_VERSION}_a${MAST_HPC_JOB_ATTEMPT_INDEX}

mkdir -p "${RUN_ROOT}/runtime"
python "${runner_root}/runtime_preflight.py"

configs=(
    muse_glimmer_30b_sdpa_c4_torchtitan_4x2
)
phases=(
    01_per_gpu_bs1_torchtitan_native_inductor
)
phase_kinds=(performance)
local_batch_sizes=(2)

if [[ ${run_mode} == formal ]]; then
    configs+=(muse_glimmer_30b_sdpa_c4_torchtitan_4x2)
    phases+=(
        11_trace_per_gpu_bs1_torchtitan_native_inductor
    )
    phase_kinds+=(trace)
    local_batch_sizes+=(2)
fi

run_phase() {
    local phase=$1
    local config_name=$2
    local master_port=$3
    local phase_kind=$4
    local local_batch_size=$5
    local output=${RUN_ROOT}/${phase}
    local cache_root
    cache_root=$(mktemp -d "/tmp/muse-glimmer-s4096-${phase}-${RANK}-XXXXXX")
    mkdir -p \
        "${output}/job" \
        "${output}/runtime" \
        "${cache_root}/hf" \
        "${cache_root}/datasets" \
        "${cache_root}/inductor" \
        "${cache_root}/triton" \
        "${cache_root}/tmp"

    if ! python "${runner_root}/validate_allocation.py" "${phase}"; then
        if [[ ${RANK} == 0 ]]; then
            touch "${RUN_ROOT}/runtime/${phase}.failed"
        fi
        exit 1
    fi
    if [[ ${RANK} == 0 ]]; then
        touch "${RUN_ROOT}/runtime/${phase}.started"
    fi

    local -a mode_args
    if [[ ${run_mode} == gate ]]; then
        mode_args=(
            --training.steps 2
            --profiler.no-enable-profiling
            --metrics.no-enable-tensorboard
            --metrics.log-freq 1
        )
    elif [[ ${phase_kind} == performance ]]; then
        mode_args=(
            --training.steps 25
            --profiler.no-enable-profiling
            --metrics.enable-tensorboard
            --metrics.log-freq 5
        )
    else
        mode_args=(
            --training.steps 6
            --profiler.profile-freq 6
            --profiler.profiler-warmup 0
            --profiler.profiler-active 3
            --profiler.profiler-repeat 1
            --profiler.no-enable-memory-snapshot
            --metrics.enable-tensorboard
            --metrics.log-freq 1
        )
        if [[ ${RANK} == 0 ]]; then
            mode_args+=(--profiler.enable-profiling)
        else
            mode_args+=(--profiler.no-enable-profiling)
        fi
    fi

    local -a phase_env=(
        HF_HOME="${cache_root}/hf"
        HF_DATASETS_CACHE="${cache_root}/datasets"
        TORCHINDUCTOR_CACHE_DIR="${cache_root}/inductor"
        TRITON_CACHE_DIR="${cache_root}/triton"
        TMPDIR="${cache_root}/tmp"
        MASTER_PORT="${master_port}"
        TORCHELASTIC_USE_AGENT_STORE=False
    )
    if [[ ${RANK} == 0 && ${phase_kind} == trace ]]; then
        phase_env+=(TORCH_TRACE="${output}/compile_trace")
    fi

    cd "${tt_root}"
    set +e
    env "${phase_env[@]}" python -m torchtitan.train \
        --module graph_trainer.muse_glimmer \
        --config "${config_name}" \
        --hf-assets-path "${runner_root}/assets/muse_glimmer" \
        --dump-folder "${output}/job" \
        --debug.save-config-file "${output}/job/config.json" \
        --checkpoint.no-enable \
        --validator.no-enable \
        --debug.seed 42 \
        --comm.init-timeout-seconds 1800 \
        --compile.enable \
        --compile.backend inductor \
        --training.local-batch-size "${local_batch_size}" \
        --training.global-batch-size -1 \
        --training.seq-len 4096 \
        --parallelism.data-parallel-shard-degree "${fsdp_degree}" \
        --parallelism.tensor-parallel-degree "${tp_degree}" \
        "${mode_args[@]}" \
        2>&1 | tee "${output}/runtime/rank_${RANK}.combined.log"
    local status=${PIPESTATUS[0]}
    set -e

    if [[ ${status} != 0 ]] && grep -Fq \
        'torch.OutOfMemoryError: CUDA out of memory' \
        "${output}/runtime/rank_${RANK}.combined.log"; then
        touch "${output}/runtime/rank_${RANK}.oom_evidence"
    fi
    printf '%s\n' "${status}" > "${output}/runtime/rank_${RANK}.exit_code.tmp"
    mv \
        "${output}/runtime/rank_${RANK}.exit_code.tmp" \
        "${output}/runtime/rank_${RANK}.exit_code"

    while [[ $(find "${output}/runtime" -maxdepth 1 -name 'rank_*.exit_code' | wc -l) -lt ${world_size} ]]; do
        sleep 1
    done
    if [[ ${RANK} == 0 ]]; then
        local all_ranks_succeeded=1
        local exit_code_file
        for exit_code_file in "${output}"/runtime/rank_*.exit_code; do
            if [[ $(<"${exit_code_file}") != 0 ]]; then
                all_ranks_succeeded=0
                break
            fi
        done
        if [[ ${all_ranks_succeeded} == 1 ]]; then
            touch "${RUN_ROOT}/runtime/${phase}.completed"
        elif compgen -G "${output}/runtime/rank_*.oom_evidence" > /dev/null; then
            touch "${RUN_ROOT}/runtime/${phase}.oom"
        else
            touch "${RUN_ROOT}/runtime/${phase}.failed"
        fi
    fi
    while [[ ! -e "${RUN_ROOT}/runtime/${phase}.completed" \
        && ! -e "${RUN_ROOT}/runtime/${phase}.oom" \
        && ! -e "${RUN_ROOT}/runtime/${phase}.failed" ]]; do
        sleep 1
    done
    [[ -e "${RUN_ROOT}/runtime/${phase}.completed" ]]
}

for index in "${!configs[@]}"; do
    run_phase \
        "${phases[$index]}" \
        "${configs[$index]}" \
        "$((29500 + index))" \
        "${phase_kinds[$index]}" \
        "${local_batch_sizes[$index]}"
done
