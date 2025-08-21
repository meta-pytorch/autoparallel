#!/bin/bash

set -x

if [[ $# -lt 1 ]]; then
    echo "Incorrect number of arguments (0)"
    echo "Usage: $0 config_file <overrides>"
    exit 1
fi

# consume config file and leave remaining args to 'overrides'
CONFIG_FILE=${1}
shift

overrides=""
if [ $# -gt 0 ]; then
      overrides="$*"
fi

edir="${DUMP_DIR}"
ename="${JOB_ID}_v${MAST_HPC_JOB_VERSION}_a${MAST_HPC_JOB_ATTEMPT_INDEX}"
dataset_path="/mnt/mffuse/c4"
save_tb_folder="/mnt/wsfuse/outputs/${JOB_ID}/tb"


echo dump_dir=$edir
echo experiment_name=$ename


LIBCUDA="/usr/local/fbcode/platform010/lib/libcuda.so"
export LIBCUDA_DIR="${LIBCUDA%/*}"
export TRITON_LIBCUDA_PATH="/usr/local/fbcode/platform010/lib/"
export LD_PRELOAD="${PRELOAD_PATH:=$LIBCUDA:/usr/local/fbcode/platform010/lib/libnvidia-ml.so}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CONDA_DIR}/lib"
export PYTHONPATH="$PYTHONPATH:$TORCHX_RUN_PYTHONPATH"

source ${CONDA_DIR}/bin/activate

cd /packages/torchtitan_additional_packages/torchtitan

###############
#  do whatever you like below
###############

if [ -n "${WANDB_API_KEY}" ]; then
  wandb login --host=https://meta.wandb.io
fi

if [ -n "$LIGHTHOUSE_SMC_TIER" ]; then
  # Run smcc command until it returns a host:port pair
  while true; do
    service=$(/packages/torchft_smcc/smcc list-services --enabled "$LIGHTHOUSE_SMC_TIER" | head -n 1)
    if [ -n "$service" ]; then
      break
    fi
    sleep 1
  done

  # Set TORCHFT_LIGHTHOUSE environment variable
  export TORCHFT_LIGHTHOUSE="http://$service"
  echo "TORCHFT_LIGHTHOUSE set to $TORCHFT_LIGHTHOUSE"
else
  echo "LIGHTHOUSE_SMC_TIER env not set, skipping..."
fi


PYTORCH_KERNEL_CACHE_PATH="/mnt/mffuse/.cache/torch/kernels" \
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
TORCH_DISABLE_ADDR2LINE=1 \
python torchtitan/train.py \
--job.config_file "${CONFIG_FILE}" \
--job.dump_folder "${edir}" \
--training.dataset_path "${dataset_path}" \
--validation.dataset_path "${dataset_path}" \
--metrics.save_tb_folder "${save_tb_folder}" \
--metrics.disable_color_printing \
--job.print_args \
$overrides
