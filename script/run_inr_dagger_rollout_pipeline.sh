#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

: "${CONFIG:?CONFIG is required}"

RUN_TRAIN="${RUN_TRAIN:-0}"
CHECKPOINT="${CHECKPOINT:-}"
RUN_NAME="${RUN_NAME:-$(basename "${CONFIG}" .json)_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-results/inr_dagger_rollout_pipeline/${RUN_NAME}}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-${RUN_DIR}/training}"
TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-${RUN_DIR}/tf-logs}"
TRAIN_LOG="${RUN_DIR}/train.log"
PIPELINE_LOG="${RUN_DIR}/pipeline.log"

SPLIT="${SPLIT:-test}"
PERFORMANCE_DATASET="${PERFORMANCE_DATASET:-ASAP}"
SCORE_SOURCE_LIST="${SCORE_SOURCE_LIST:-results/psr_oracle/window_style_prefix_enc_add/cheap15_score_sources.txt}"
MAX_WORKS="${MAX_WORKS:-}"

ROLLOUT_KS="${ROLLOUT_KS:-0,1,2,4}"
SAMPLING_STRATEGY="${SAMPLING_STRATEGY:-sample}"
FEEDBACK_STRATEGY="${FEEDBACK_STRATEGY:-sample}"
KPASS_BATCH_SIZE_WINDOWS="${KPASS_BATCH_SIZE_WINDOWS:-8}"
AR_BATCH_SIZE_WINDOWS="${AR_BATCH_SIZE_WINDOWS:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
METRIC_NUM_WORKERS="${METRIC_NUM_WORKERS:-8}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
SEED="${SEED:-2042}"
MERGE_MODE="${MERGE_MODE:-continuation}"
CONTINUATION_DROP_RATIO="${CONTINUATION_DROP_RATIO:-0.0}"
DEVICE="${DEVICE:-cuda}"
PLOT_KIND="${PLOT_KIND:-density}"

BATCH_SIZE_PER_DEVICE="${BATCH_SIZE_PER_DEVICE:-32}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-32}"

mkdir -p "${RUN_DIR}" "${TRAIN_OUTPUT_DIR}" "${TRAIN_LOG_DIR}"

IFS=',' read -ra GPU_LIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
GPU_COUNT="${#GPU_LIST[@]}"
GRADIENT_ACCUMULATION_STEPS=$(( GLOBAL_BATCH_SIZE / (BATCH_SIZE_PER_DEVICE * GPU_COUNT) ))
if [[ "${GRADIENT_ACCUMULATION_STEPS}" -lt 1 || $(( BATCH_SIZE_PER_DEVICE * GPU_COUNT * GRADIENT_ACCUMULATION_STEPS )) -ne "${GLOBAL_BATCH_SIZE}" ]]; then
  echo "Invalid batch setup: per_device=${BATCH_SIZE_PER_DEVICE}, GPUs=${GPU_COUNT}, global=${GLOBAL_BATCH_SIZE}" >&2
  exit 1
fi

latest_numeric_checkpoint() {
  local root="$1"
  find "${root}" -path '*/checkpoint-*' -type d 2>/dev/null \
    | awk -F'checkpoint-' '/checkpoint-[0-9]+$/ {print $2 " " $0}' \
    | sort -n | tail -n 1 | cut -d' ' -f2-
}

best_checkpoint() {
  local root="$1"
  if [[ -d "${root}/checkpoint-best" ]]; then
    echo "${root}/checkpoint-best"
  else
    latest_numeric_checkpoint "${root}"
  fi
}

write_train_config() {
  local src="$1" dst="$2"
  python - "$src" "$dst" "$TRAIN_OUTPUT_DIR" "$TRAIN_LOG_DIR" "$NUM_TRAIN_EPOCHS" \
    "$BATCH_SIZE_PER_DEVICE" "$GRADIENT_ACCUMULATION_STEPS" "$GLOBAL_BATCH_SIZE" <<'PY'
import json
import sys
from pathlib import Path

src, dst, output_dir, logging_dir, epochs, per_device_bs, grad_accum, global_bs = sys.argv[1:9]
cfg = json.loads(Path(src).read_text(encoding="utf-8"))
cfg["output_dir"] = output_dir
cfg["logging_dir"] = logging_dir
cfg["num_train_epochs"] = float(epochs)
cfg["max_train_epochs"] = float(epochs)
cfg["per_device_train_batch_size"] = int(per_device_bs)
cfg["per_device_eval_batch_size"] = int(per_device_bs)
cfg["gradient_accumulation_steps"] = int(grad_accum)
cfg["global_batch_size"] = int(global_bs)
cfg["train_performance_dataset"] = "ASAP"
cfg["eval_performance_dataset"] = "ASAP"
cfg["eval_split"] = "valid"
cfg["load_best_model_at_end"] = True
cfg["metric_for_best_model"] = "eval_loss"
cfg["greater_is_better"] = False
cfg["eval_every_epochs"] = 1.0
cfg["save_every_epochs"] = 1.0
cfg.pop("eval_every_steps", None)
cfg.pop("save_every_steps", None)
Path(dst).parent.mkdir(parents=True, exist_ok=True)
Path(dst).write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

run_train() {
  local train_config="${RUN_DIR}/train_config.json"
  write_train_config "${CONFIG}" "${train_config}"
  echo "[$(date '+%F %T')] train start GPUs=${CUDA_VISIBLE_DEVICES:-0}" | tee -a "${PIPELINE_LOG}"
  if [[ "${GPU_COUNT}" -gt 1 ]]; then
    local master_port
    master_port="$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")"
    PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo \
      torchrun --nnodes=1 --nproc_per_node="${GPU_COUNT}" \
        --master_addr=127.0.0.1 --master_port="${master_port}" \
        src/train/train_inr.py --config "${train_config}" 2>&1 | tee -a "${TRAIN_LOG}"
  else
    PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      python src/train/train_inr.py --config "${train_config}" 2>&1 | tee -a "${TRAIN_LOG}"
  fi
  CHECKPOINT="$(best_checkpoint "${TRAIN_OUTPUT_DIR}")"
  CONFIG="${train_config}"
  echo "[$(date '+%F %T')] train done checkpoint=${CHECKPOINT}" | tee -a "${PIPELINE_LOG}"
}

run_kpass() {
  local out_dir="${RUN_DIR}/kpass_eval"
  mkdir -p "${out_dir}"
  echo "[$(date '+%F %T')] k-pass eval k=${ROLLOUT_KS}" | tee -a "${PIPELINE_LOG}"
  PYTHONUNBUFFERED=1 python src/evaluate/eval_inr_rollout_current.py \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --output-dir "${out_dir}" \
    --split "${SPLIT}" \
    --performance-dataset "${PERFORMANCE_DATASET}" \
    --score-source-list "${SCORE_SOURCE_LIST}" \
    --batch-size-windows "${KPASS_BATCH_SIZE_WINDOWS}" \
    --num-workers "${NUM_WORKERS}" \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    --materialize-strategy "${SAMPLING_STRATEGY}" \
    --feedback-strategy "${FEEDBACK_STRATEGY}" \
    --rollout-ks "${ROLLOUT_KS}" \
    --fast-kpass \
    --plot-distributions \
    --save-distribution-values 2>&1 | tee -a "${PIPELINE_LOG}"
}

run_ar_infer() {
  local out_dir="${RUN_DIR}/ar_infer"
  mkdir -p "${out_dir}"
  echo "[$(date '+%F %T')] real AR infer raw+midi" | tee -a "${PIPELINE_LOG}"
  PYTHONUNBUFFERED=1 python src/inference/infer_inr_testset.py \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --split "${SPLIT}" \
    --performance-dataset "${PERFORMANCE_DATASET}" \
    --score-source-list "${SCORE_SOURCE_LIST}" \
    --num-workers "${NUM_WORKERS}" \
    --batch-size-windows "${AR_BATCH_SIZE_WINDOWS}" \
    --merge-mode "${MERGE_MODE}" \
    --continuation-drop-ratio "${CONTINUATION_DROP_RATIO}" \
    --device "${DEVICE}" \
    --protocol sampling \
    --sampling-strategy "${SAMPLING_STRATEGY}" \
    --num-samples "${NUM_SAMPLES}" \
    --seed "${SEED}" \
    --output-dir "${out_dir}" 2>&1 | tee -a "${PIPELINE_LOG}"
}

run_summary() {
  echo "[$(date '+%F %T')] metrics + plots" | tee -a "${PIPELINE_LOG}"
  PYTHONUNBUFFERED=1 python src/evaluate/summarize_inr_rollout_pipeline.py \
    --kpass-summary "${RUN_DIR}/kpass_eval/summary.json" \
    --ar-manifest "${RUN_DIR}/ar_infer/prediction_manifest.json" \
    --output-dir "${RUN_DIR}/summary" \
    --score-source-list "${SCORE_SOURCE_LIST}" \
    --num-workers "${METRIC_NUM_WORKERS}" \
    --plot-kind "${PLOT_KIND}" 2>&1 | tee -a "${PIPELINE_LOG}"
}

echo "RUN_DIR ${RUN_DIR}" | tee -a "${PIPELINE_LOG}"
if [[ "${RUN_TRAIN}" == "1" ]]; then
  run_train
elif [[ -z "${CHECKPOINT}" ]]; then
  echo "CHECKPOINT is required when RUN_TRAIN=0" >&2
  exit 1
fi

if [[ -n "${MAX_WORKS}" ]]; then
  echo "MAX_WORKS is intentionally ignored here; use SCORE_SOURCE_LIST for fixed cheap subsets." | tee -a "${PIPELINE_LOG}"
fi

run_kpass
run_ar_infer
run_summary

echo "[$(date '+%F %T')] done ${RUN_DIR}" | tee -a "${PIPELINE_LOG}"
