#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

: "${GPU_ID:?GPU_ID is required}"
: "${QUEUE_NAME:?QUEUE_NAME is required}"
: "${CONFIG_1:?CONFIG_1 is required}"
: "${CONFIG_2:?CONFIG_2 is required}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
QUEUE_ROOT="results/slot0710_queues/${STAMP}/gpu${GPU_ID}_${QUEUE_NAME}"
mkdir -p "${QUEUE_ROOT}"

wait_for_cuda() {
  while ! CUDA_VISIBLE_DEVICES="${GPU_ID}" python - <<'PY' >/dev/null 2>&1
import torch
raise SystemExit(0 if torch.cuda.is_available() and torch.cuda.device_count() == 1 else 1)
PY
  do
    echo "[$(date '+%F %T')] GPU ${GPU_ID} CUDA unavailable; retrying in 60s" \
      | tee -a "${QUEUE_ROOT}/queue.log"
    sleep 60
  done
}

run_one() {
  local label="$1"
  local config="$2"
  local run_dir="${QUEUE_ROOT}/${label}"

  echo "[$(date '+%F %T')] START ${label} on GPU ${GPU_ID}" | tee -a "${QUEUE_ROOT}/queue.log"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  CONFIG="${config}" \
  RUN_DIR_OVERRIDE="${run_dir}" \
  BATCH_SIZE_PER_DEVICE=32 \
  GLOBAL_BATCH_SIZE=64 \
  BASE_NUM_TRAIN_EPOCHS="${BASE_NUM_TRAIN_EPOCHS:-8}" \
  ADAPT_NUM_TRAIN_EPOCHS="${ADAPT_NUM_TRAIN_EPOCHS:-16}" \
  bash script/run_inr_epr_pipeline.sh
  echo "[$(date '+%F %T')] END ${label} on GPU ${GPU_ID}" | tee -a "${QUEUE_ROOT}/queue.log"
}

wait_for_cuda
run_one "$(basename "${CONFIG_1}" .json)" "${CONFIG_1}"
run_one "$(basename "${CONFIG_2}" .json)" "${CONFIG_2}"
