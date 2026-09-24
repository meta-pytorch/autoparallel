#!/bin/bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
    echo "Usage: $0 NODES REPLICATE [gate] [modes]" >&2
    exit 2
fi
nodes=$1
replicate=$2
gate=${3:-false}
modes=${4:-}
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
conda_fbpkg_id=${CONDA_FBPKG_ID:-torchtitan_conda_prod:902}
cd "${root}/launcher"
exec torchx run \
    --scheduler_args="conda_fbpkg_id=${conda_fbpkg_id},localityConstraints=dc;pci1,forceSingleRegion=False,use_caf=False" \
    mast.py:benchmark \
    --payload_root "${root}" \
    --nodes "${nodes}" \
    --replicate "${replicate}" \
    --gate "${gate}" \
    --modes "${modes}" \
    --nproc_per_node 8 \
    --name h100-nvswitch-roce-400g-calibration \
    --h grandteton_80g_roce \
    --retries 0 \
    --enable_ttls True
