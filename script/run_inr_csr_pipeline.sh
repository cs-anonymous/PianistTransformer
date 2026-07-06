#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

: "${CONFIG:?CONFIG is required}"

INFER_NUM_WORKERS="${INFER_NUM_WORKERS:-8}"
INFER_BATCH_SIZE_WINDOWS="${INFER_BATCH_SIZE_WINDOWS:-8}"
BATCH_SIZE_PER_DEVICE="${BATCH_SIZE_PER_DEVICE:-32}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-8}"
RESUME_FROM_LATEST_CHECKPOINT="${RESUME_FROM_LATEST_CHECKPOINT:-1}"

IFS=',' read -ra GPU_LIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
GPU_COUNT="${#GPU_LIST[@]}"
GRADIENT_ACCUMULATION_STEPS=$(( GLOBAL_BATCH_SIZE / (BATCH_SIZE_PER_DEVICE * GPU_COUNT) ))
if [[ "${GRADIENT_ACCUMULATION_STEPS}" -lt 1 || $(( BATCH_SIZE_PER_DEVICE * GPU_COUNT * GRADIENT_ACCUMULATION_STEPS )) -ne "${GLOBAL_BATCH_SIZE}" ]]; then
  echo "Invalid batch setup: per_device=${BATCH_SIZE_PER_DEVICE}, GPUs=${GPU_COUNT}, global=${GLOBAL_BATCH_SIZE}" >&2
  exit 1
fi

DEFAULT_RUN_NAME="${CONFIG##*/}"
DEFAULT_RUN_NAME="${DEFAULT_RUN_NAME%.json}_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR_OVERRIDE:-results/inr_csr_pipeline/${DEFAULT_RUN_NAME}}"
TRAIN_ROOT="${RUN_DIR}/training"
TF_LOG_ROOT="${RUN_DIR}/tf-logs"
RUN_CONFIG="${RUN_DIR}/config.json"
TRAIN_LOG="${RUN_DIR}/train.log"
EVALUATE_LOG="${RUN_DIR}/evaluate.log"
TMP_DIR="${RUN_DIR}/_tmp"
mkdir -p "${RUN_DIR}" "${TRAIN_ROOT}" "${TF_LOG_ROOT}" "${TMP_DIR}"

MASTER_PORT="$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")"
INFER_GPU="${GPU_LIST[0]}"

latest_train_dir() {
  local root="$1" marker="$2"
  find "${root}" -maxdepth 1 -mindepth 1 -type d -name 'inr_*' -newer "${marker}" \
    -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-
}

latest_numeric_checkpoint() {
  local root="$1"
  find "${root}" -path '*/checkpoint-*' -type d 2>/dev/null \
    | awk -F'checkpoint-' '/checkpoint-[0-9]+$/ {print $2 " " $0}' \
    | sort -n | tail -n 1 | cut -d' ' -f2-
}

best_checkpoint() {
  local train_dir="$1"
  if [[ -d "${train_dir}/checkpoint-best" ]]; then
    echo "${train_dir}/checkpoint-best"
  else
    echo "${train_dir}"
  fi
}

python - "${CONFIG}" "${RUN_CONFIG}" "${TRAIN_ROOT}" "${TF_LOG_ROOT}" "${NUM_TRAIN_EPOCHS}" \
  "${BATCH_SIZE_PER_DEVICE}" "${GRADIENT_ACCUMULATION_STEPS}" "${GLOBAL_BATCH_SIZE}" "$(latest_numeric_checkpoint "${TRAIN_ROOT}" || true)" <<'PY'
import json
import sys
from pathlib import Path

src, dst, output_root, log_root, epochs, per_device_bs, grad_accum, global_bs, resume_path = sys.argv[1:10]
cfg = json.loads(Path(src).read_text(encoding="utf-8"))
if cfg.get("task_type", "epr").lower() != "csr":
    raise SystemExit("run_inr_csr_pipeline.sh requires task_type=csr")
cfg["use_style_tokens"] = False
cfg["output_dir"] = output_root
cfg["logging_dir"] = log_root
cfg["num_train_epochs"] = float(epochs)
cfg["max_train_epochs"] = float(epochs)
cfg["per_device_train_batch_size"] = int(per_device_bs)
cfg["per_device_eval_batch_size"] = int(per_device_bs)
cfg["gradient_accumulation_steps"] = int(grad_accum)
cfg["global_batch_size"] = int(global_bs)
cfg["load_best_model_at_end"] = True
cfg["metric_for_best_model"] = "eval_loss"
cfg["greater_is_better"] = False
cfg.setdefault("eval_dataloader_persistent_workers", False)
if resume_path:
    cfg["resume_path"] = resume_path
    cfg["resume_trainer_state"] = True
Path(dst).parent.mkdir(parents=True, exist_ok=True)
Path(dst).write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY

MARKER="${TMP_DIR}/train.marker"
touch "${MARKER}"
if [[ "${GPU_COUNT}" -gt 1 ]]; then
  MASTER_PORT="${MASTER_PORT}" PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo \
    torchrun --nnodes=1 --nproc_per_node="${GPU_COUNT}" \
      --master_addr=127.0.0.1 --master_port="${MASTER_PORT}" \
      src/train/train_inr.py --config "${RUN_CONFIG}" 2>&1 | tee -a "${TRAIN_LOG}"
else
  PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python src/train/train_inr.py --config "${RUN_CONFIG}" 2>&1 | tee -a "${TRAIN_LOG}"
fi

TRAIN_OUTPUT_DIR="$(latest_train_dir "${TRAIN_ROOT}" "${MARKER}")"
CHECKPOINT="$(best_checkpoint "${TRAIN_OUTPUT_DIR}")"
INFER_DIR="${RUN_DIR}/deterministic"
mkdir -p "${INFER_DIR}"

CUDA_VISIBLE_DEVICES="${INFER_GPU}" PYTHONUNBUFFERED=1 python src/inference/infer_inr_testset.py \
  --config "${RUN_CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --split test \
  --num-workers "${INFER_NUM_WORKERS}" \
  --batch-size-windows "${INFER_BATCH_SIZE_WINDOWS}" \
  --device cuda \
  --protocol deterministic \
  --output-dir "${INFER_DIR}" 2>&1 | tee -a "${EVALUATE_LOG}"

rm -rf "${TMP_DIR}"
echo "END $(date '+%F %T')" | tee -a "${EVALUATE_LOG}"
