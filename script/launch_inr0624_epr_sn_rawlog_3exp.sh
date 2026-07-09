#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="results/inr_epr_pipeline/launch_rawlog_3exp_${STAMP}"
mkdir -p "${RUN_ROOT}"

launch_one() {
  local gpu="$1"
  local name="$2"
  local config="$3"
  local per_device_bs="$4"
  local global_bs="$5"
  local run_dir="${RUN_ROOT}/${name}"
  local log_path="${run_dir}/launcher.log"
  mkdir -p "${run_dir}"
  echo "[$(date '+%F %T')] launch ${name} on GPU ${gpu}" | tee -a "${RUN_ROOT}/launch.log"
  setsid env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    CONFIG="${config}" \
    RUN_DIR_OVERRIDE="${run_dir}" \
    BATCH_SIZE_PER_DEVICE="${per_device_bs}" \
    GLOBAL_BATCH_SIZE="${global_bs}" \
    BASE_ASAP_ONLY=1 \
    BASE_NUM_TRAIN_EPOCHS=16 \
    ADAPT_NUM_TRAIN_EPOCHS=0 \
    DET_NUM_SAMPLES=1 \
    SAMPLING_NUM_SAMPLES=1 \
    INFER_BATCH_SIZE_WINDOWS=8 \
    MERGE_MODE=continuation \
    CONTINUATION_DROP_RATIO=0.0 \
    bash script/run_inr_epr_pipeline.sh >"${log_path}" 2>&1 < /dev/null &
  local pid=$!
  echo "${name}: pid=${pid} run_dir=${run_dir} log=${log_path}" | tee -a "${RUN_ROOT}/launch.log"
}

launch_one 0 "exp1_sine_tfmask50" "configs/inr0624_epr_sn_rawlog_sine_tfmask50.json" 32 64
launch_one 1 "exp2_sine_nomus_tfmask50" "configs/inr0624_epr_sn_rawlog_sine_nomus_tfmask50.json" 32 64
launch_one 2 "exp3_splitperf_tfmask50" "configs/inr0624_epr_sn_rawlog_splitperf_tfmask50.json" 32 64

echo "launch_root=${RUN_ROOT}"
