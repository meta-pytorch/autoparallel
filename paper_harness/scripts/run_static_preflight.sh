#!/bin/bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON_BIN:-python3}
: "${VALIDATION_ROOT:?VALIDATION_ROOT must be a task-specific directory under the workspace}"
mkdir -p "${VALIDATION_ROOT}/pycache"
export PYTHONPYCACHEPREFIX="${VALIDATION_ROOT}/pycache"
export PYTHONDONTWRITEBYTECODE=1
cd "${repo_root}"

"${python_bin}" -m compileall -q harness workloads launcher/mast.py tests
"${python_bin}" -m unittest discover -s tests -v
git diff --check
