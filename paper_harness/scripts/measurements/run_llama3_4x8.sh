#!/usr/bin/env bash
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
harness_root=$(cd -- "$script_dir/../.." && pwd)
driver="$script_dir/measurement.py"
campaign="$harness_root/campaigns/llama3_8b_2d_scaling.toml"

attempt=${ATTEMPT_ROOT:-}
python_bin=${HARNESS_PYTHON:-python3}
torchtitan_root=${TORCHTITAN_ROOT:-}
autoparallel_root=${AUTOPARALLEL_ROOT:-}
tokenizer_root=${LLAMA_TOKENIZER_ROOT:-}
replay_root=${LLAMA_REPLAY_ROOT:-}
seed_checkpoint_root=${LLAMA_SEED_CHECKPOINT_ROOT:-}
oilfs_uri=${OILFS_URI:-}
oilfs_user=${OILFS_USER:-${USER:-}}
tlparse_bin=${TLPARSE_BIN:-}
mode=${MEASUREMENT_MODE:-formal}
interval=${POLL_INTERVAL_SECONDS:-30}
submit=false
dry_run=false

usage() {
  cat <<'EOF'
Usage: run_llama3_4x8.sh [options]

Prepare the canonical LLaMA3 8B 4x8 A/B/C attempt. Preparation is the default
and performs validation, sealed packaging, and MAST dry-run only. Pass --submit
to continue through submission, CRITICAL/99, monitoring, retrieval, and analysis.

Required:
  --attempt PATH                 New absolute immutable attempt path
  --python PATH                  Compatible Python interpreter
  --torchtitan-root PATH         Clean pinned TorchTitan checkout
  --autoparallel-root PATH       Clean pinned AutoParallel checkout
  --tokenizer-root PATH          LLaMA tokenizer asset
  --replay-root PATH             LLaMA replay asset
  --seed-checkpoint-root PATH    LLaMA seed checkpoint asset

Required with --submit:
  --oilfs-uri URI                Exact ws:// URI used by the submitted mount.sh
  --tlparse-bin PATH             Executable canonical tlparse binary

Optional:
  --mode formal|gate             Default: formal
  --interval-seconds N           MAST polling interval; default: 30
  --oilfs-user USER              Default: OILFS_USER or USER
  --submit                       Explicitly authorize job submission
  --dry-run                      Print the composed commands without executing
  -h, --help                     Show this help

Every option also accepts the uppercase environment variable shown in the
README. No source, asset, job ID, or machine-local path is embedded here.
EOF
}

while (($#)); do
  case $1 in
    --attempt) attempt=$2; shift 2 ;;
    --python) python_bin=$2; shift 2 ;;
    --torchtitan-root) torchtitan_root=$2; shift 2 ;;
    --autoparallel-root) autoparallel_root=$2; shift 2 ;;
    --tokenizer-root) tokenizer_root=$2; shift 2 ;;
    --replay-root) replay_root=$2; shift 2 ;;
    --seed-checkpoint-root) seed_checkpoint_root=$2; shift 2 ;;
    --oilfs-uri) oilfs_uri=$2; shift 2 ;;
    --oilfs-user) oilfs_user=$2; shift 2 ;;
    --tlparse-bin) tlparse_bin=$2; shift 2 ;;
    --mode) mode=$2; shift 2 ;;
    --interval-seconds) interval=$2; shift 2 ;;
    --submit) submit=true; shift ;;
    --dry-run) dry_run=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for value in attempt python_bin torchtitan_root autoparallel_root tokenizer_root replay_root seed_checkpoint_root; do
  if [[ -z ${!value} ]]; then
    echo "missing required value: $value" >&2
    exit 2
  fi
done
if [[ $mode != formal && $mode != gate ]]; then
  echo "--mode must be formal or gate" >&2
  exit 2
fi
if [[ $submit == true && ( -z $oilfs_uri || -z $tlparse_bin || -z $oilfs_user ) ]]; then
  echo "--submit requires --oilfs-uri, --oilfs-user, and --tlparse-bin" >&2
  exit 2
fi

prepare=(
  "$python_bin" "$driver" prepare
  --campaign "$campaign"
  --point 32gpu
  --mode "$mode"
  --attempt "$attempt"
  --python "$python_bin"
  --torchtitan-root "$torchtitan_root"
  --autoparallel-root "$autoparallel_root"
  --asset-root "llama_tokenizer=$tokenizer_root"
  --asset-root "llama_replay=$replay_root"
  --asset-root "llama3_3d_seed_checkpoint=$seed_checkpoint_root"
)

if [[ $dry_run == true ]]; then
  printf '%q ' "${prepare[@]}"; printf '\n'
  if [[ $submit == true ]]; then
    printf '%q ' "$python_bin" "$driver" submit --attempt "$attempt"; printf '\n'
    printf '%q ' "$python_bin" "$driver" monitor --attempt "$attempt" --interval-seconds "$interval"; printf '\n'
    printf '%q ' "$python_bin" "$driver" retrieve --attempt "$attempt" --oilfs-uri "$oilfs_uri" --oilfs-user "$oilfs_user"; printf '\n'
    printf '%q ' "$python_bin" "$driver" monitor --attempt "$attempt" --run-root "$attempt/run" --once; printf '\n'
    printf '%q ' "$python_bin" "$driver" analyze --attempt "$attempt" --point 32gpu --mode "$mode" --python "$python_bin" --tlparse-bin "$tlparse_bin"; printf '\n'
  fi
  exit 0
fi

cd -- "$harness_root"
"${prepare[@]}"
if [[ $submit != true ]]; then
  echo "Prepared and dry-run audited $attempt; pass --submit in a new attempt to launch."
  exit 0
fi

"$python_bin" "$driver" submit --attempt "$attempt"
"$python_bin" "$driver" monitor --attempt "$attempt" --interval-seconds "$interval"
"$python_bin" "$driver" retrieve --attempt "$attempt" --oilfs-uri "$oilfs_uri" --oilfs-user "$oilfs_user"
"$python_bin" "$driver" monitor --attempt "$attempt" --run-root "$attempt/run" --once
"$python_bin" "$driver" analyze --attempt "$attempt" --point 32gpu --mode "$mode" --python "$python_bin" --tlparse-bin "$tlparse_bin"
