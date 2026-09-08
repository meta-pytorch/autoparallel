#!/bin/bash

set -euo pipefail

dryrun=()
if [[ ${1:-} == --dryrun ]]; then
    dryrun=(--dryrun)
    shift
fi
if [[ $# -lt 2 || $# -gt 3 || ( $1 != gate && $1 != formal ) ]]; then
    echo "Usage: $0 [--dryrun] {gate|formal} 2 [REPLICATE]" >&2
    exit 2
fi

mode=$1
nodes=$2
replicate=${3:-1}
[[ ${nodes} == 2 ]] || { echo "This experiment requires two 8-GPU nodes" >&2; exit 2; }
[[ ${replicate} =~ ^[0-9]+$ ]] || exit 2

world_size=$((nodes * 8))
tp_degree=2
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
task_root=${TASK_ROOT:-${script_dir}}
c4_hf_cache=${C4_HF_CACHE:?C4_HF_CACHE must point to the pinned offline C4 cache}
[[ -d ${c4_hf_cache} ]] || { echo "C4_HF_CACHE is not a directory" >&2; exit 2; }
[[ $(basename "${c4_hf_cache%/}") == c4_hf_cache ]] || {
    echo "C4_HF_CACHE must have basename c4_hf_cache" >&2
    exit 2
}
cd "${task_root}/launcher"

exec torchx run "${dryrun[@]}" \
    --scheduler_args="conda_fbpkg_id=torchtitan_conda_prod:902,localityConstraints=dc;pci1,forceSingleRegion=False" \
    mast.py:train \
    --name "muse-glimmer-30b-sdpa-fix-sac-aligned-lb2-s4096-${nodes}x8-${mode}" \
    --h grandteton_80g_roce \
    --nodes "${nodes}" \
    --nproc_per_node 8 \
    --retries 0 \
    --enable_ttls True \
    --additional_folders "${task_root}/runner,${c4_hf_cache}" \
    --additional_libraries="${task_root}/source/autoparallel,${task_root}/source/torchtitan" \
    --module_name graph_trainer.muse_glimmer \
    --config_name graph_trainer_muse_glimmer_30b_sdpa_c4_autoparallel_4x2 \
    "${mode}" "${world_size}" "${tp_degree}" two_arm sequential "${replicate}"
