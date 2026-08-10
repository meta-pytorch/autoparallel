#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON:-python}
TORCHRUN_BIN=${TORCHRUN:-torchrun}
BASE_REF=${BASE_REF:-origin/main}
RUN_FULL=0
RESULTS_DIR=

usage() {
  cat <<'EOF'
Usage: tests/run_flex_local_map_full_validation.sh [--full] [RESULTS_DIR]

Run the flex_local_map correctness and real four-GPU validation. The optional
--full flag also runs the complete repository pytest suite. RESULTS_DIR must be
empty; when omitted, a temporary directory is created.

Environment overrides:
  PYTHON     Python executable (default: python)
  TORCHRUN   torchrun executable (default: torchrun)
  BASE_REF   Git base used for the saved committed diff (default: origin/main)
EOF
}

while (($#)); do
  case "$1" in
    --full)
      RUN_FULL=1
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    -*)
      printf 'Unknown option: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [[ -n "$RESULTS_DIR" ]]; then
        printf 'Only one RESULTS_DIR may be provided.\n' >&2
        exit 2
      fi
      RESULTS_DIR=$1
      ;;
  esac
  shift
done

"$PYTHON_BIN" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
if torch.cuda.device_count() < 4:
    raise SystemExit(
        f"flex_local_map validation requires at least 4 GPUs; "
        f"found {torch.cuda.device_count()}"
    )
if not torch.distributed.is_nccl_available():
    raise SystemExit("the NCCL backend is unavailable")
PY

RESULTS_DIR=${RESULTS_DIR:-$(mktemp -d /tmp/flex_local_map_validation.XXXXXX)}
mkdir -p "$RESULTS_DIR"
RESULTS_DIR=$(cd -- "$RESULTS_DIR" && pwd)
if find "$RESULTS_DIR" -mindepth 1 -print -quit | grep -q .; then
  printf 'RESULTS_DIR must be empty: %s\n' "$RESULTS_DIR" >&2
  exit 2
fi

mkdir -p "$RESULTS_DIR/source_snapshots"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"

run_stage() {
  local name=$1
  shift
  {
    printf 'command:'
    printf ' %q' "$@"
    printf '\n'
  } | tee "$RESULTS_DIR/${name}.command.txt"
  set +e
  "$@" 2>&1 | tee "$RESULTS_DIR/${name}.log"
  local status=${PIPESTATUS[0]}
  set -e
  printf '%s\n' "$status" > "$RESULTS_DIR/${name}.exit_code"
  return "$status"
}

"$PYTHON_BIN" - "$RESULTS_DIR/environment.json" "$BASE_REF" "$RUN_FULL" <<'PY'
import datetime
import json
import subprocess
import sys

import torch


def git(*args, check=True):
    result = subprocess.run(
        ("git", *args), check=check, capture_output=True, text=True
    )
    return result.stdout.strip()


base = git("rev-parse", sys.argv[2], check=False)
properties = [
    torch.cuda.get_device_properties(index)
    for index in range(torch.cuda.device_count())
]
environment = {
    "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "branch": git("branch", "--show-current"),
    "head": git("rev-parse", "HEAD"),
    "base_ref": sys.argv[2],
    "base_commit": base or None,
    "git_status": git("status", "--short"),
    "python": sys.version.split()[0],
    "pytorch": torch.__version__,
    "cuda": torch.version.cuda,
    "nccl": ".".join(map(str, torch.cuda.nccl.version())),
    "cuda_device_count": torch.cuda.device_count(),
    "gpus": [
        {
            "index": index,
            "name": props.name,
            "total_memory_bytes": props.total_memory,
        }
        for index, props in enumerate(properties)
    ],
    "full_pytest_requested": bool(int(sys.argv[3])),
}
with open(sys.argv[1], "w") as output:
    json.dump(environment, output, indent=2)
    output.write("\n")
PY
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader \
  > "$RESULTS_DIR/environment.txt"
if git rev-parse --verify "$BASE_REF" >/dev/null 2>&1; then
  git diff --binary "$BASE_REF"...HEAD > "$RESULTS_DIR/committed_vs_base.patch"
fi
git diff --binary > "$RESULTS_DIR/working_tree.patch"
git status --short > "$RESULTS_DIR/git_status.txt"
cp \
  tests/compare_flex_local_map_validation.py \
  tests/run_flex_local_map_full_validation.sh \
  examples/example_ds3_local_map.py \
  tests/test_flex_local_map_e2e.py \
  tests/test_optimize_placement.py \
  tests/test_placement_options_utils.py \
  "$RESULTS_DIR/source_snapshots/"
sha256sum "$RESULTS_DIR"/source_snapshots/* > "$RESULTS_DIR/source_checksums.txt"

"$PYTHON_BIN" - "$RESULTS_DIR/execution_config.json" "$RUN_FULL" <<'PY'
import json
import sys

config = {
    "world_size": 4,
    "backend": "nccl",
    "pointwise_mesh": [4],
    "moe_mesh": [2, 2],
    "moe_mesh_names": ["dp", "ep"],
    "rng_seed": 1,
    "sequence_length": 2048,
    "local_batch_size": 8,
    "microbatches": 16,
    "dtype": "bfloat16",
    "reduce_dtype": "float32",
    "full_pytest_requested": bool(int(sys.argv[2])),
    "cases": [
        "plain-sharded",
        "plain-replicated-counts",
        "flex-default",
        "flex-replicated-counts",
    ],
}
with open(sys.argv[1], "w") as output:
    json.dump(config, output, indent=2)
    output.write("\n")
PY

echo "Results directory: $RESULTS_DIR"

run_stage compile \
  "$PYTHON_BIN" -m py_compile \
  tests/test_flex_local_map_e2e.py \
  examples/example_ds3_local_map.py \
  tests/compare_flex_local_map_validation.py \
  tests/test_optimize_placement.py \
  tests/test_placement_options_utils.py

run_stage placement_options \
  "$PYTHON_BIN" -m pytest -q \
  tests/test_placement_options_utils.py::TestFlexLocalMapPlacementOptions

run_stage optimize_placement \
  "$PYTHON_BIN" -m pytest -q \
  tests/test_optimize_placement.py -k flex_local_map

run_stage canonical_graph \
  "$PYTHON_BIN" -m pytest -q \
  tests/test_optimize_placement.py -k grafted_graph_matches_direct_trace

run_stage serialization \
  "$PYTHON_BIN" -m pytest -q \
  tests/test_serialization.py -k flex

run_stage pointwise_4gpu \
  "$PYTHON_BIN" -m pytest -q tests/test_flex_local_map_e2e.py -v

for case in plain-sharded plain-replicated-counts flex-default flex-replicated-counts; do
  run_stage "moe_${case}" \
    "$TORCHRUN_BIN" --standalone --nproc-per-node 4 \
    examples/example_ds3_local_map.py \
    --local-map-case "$case" \
    --rng-seed 1 \
    --logs-dir "$RESULTS_DIR/$case"
done

if ((RUN_FULL)); then
  run_stage full_pytest "$PYTHON_BIN" -m pytest tests
fi

run_stage numerics \
  "$PYTHON_BIN" tests/compare_flex_local_map_validation.py "$RESULTS_DIR"
