#!/usr/bin/env bash
set -u

if [ "$#" -ne 4 ]; then
  echo "Usage: $0 GPU_ID SESSION_NAME EPR_CONFIG CSR_CONFIG" >&2
  exit 2
fi

GPU_ID="$1"
SESSION_NAME="$2"
EPR_CONFIG="$3"
CSR_CONFIG="$4"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs/inr_tmux"
mkdir -p "${LOG_DIR}"

run_with_batch_fallback() {
  local task_name="$1"
  local base_config="$2"
  local batch_size tmp_config log_path

  for batch_size in 32 24 16 8; do
    tmp_config="${LOG_DIR}/${SESSION_NAME}_${task_name}_bs${batch_size}.json"
    log_path="${LOG_DIR}/${SESSION_NAME}_${task_name}_bs${batch_size}.log"
    python - "${base_config}" "${tmp_config}" "${batch_size}" <<'PY'
import json
import sys
from pathlib import Path

base_path, out_path, batch_size = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(base_path, "r", encoding="utf-8") as file:
    config = json.load(file)
config["per_device_train_batch_size"] = batch_size
config["per_device_eval_batch_size"] = min(8, batch_size)
config["gradient_accumulation_steps"] = 1
Path(out_path).write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY

    echo "[$(date '+%F %T')] ${SESSION_NAME} ${task_name}: trying batch_size=${batch_size}" | tee -a "${log_path}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      python "${ROOT_DIR}/src/train/sft_node.py" --config "${tmp_config}" 2>&1 | tee -a "${log_path}"
    status=${PIPESTATUS[0]}
    if [ "${status}" -eq 0 ]; then
      echo "[$(date '+%F %T')] ${SESSION_NAME} ${task_name}: finished with batch_size=${batch_size}" | tee -a "${log_path}"
      return 0
    fi
    echo "[$(date '+%F %T')] ${SESSION_NAME} ${task_name}: failed with batch_size=${batch_size}, status=${status}" | tee -a "${log_path}"
  done

  echo "[$(date '+%F %T')] ${SESSION_NAME} ${task_name}: failed at minimum batch_size=8" >&2
  return 1
}

cd "${ROOT_DIR}" || exit 1
run_with_batch_fallback epr "${EPR_CONFIG}" && run_with_batch_fallback csr "${CSR_CONFIG}"
