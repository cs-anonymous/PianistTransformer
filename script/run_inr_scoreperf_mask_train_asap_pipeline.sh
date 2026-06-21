#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="${CONFIG:-configs/inr_mln3_scoreperf_mask_startctrl.json}"
GPU_ID="${GPU_ID:-0}"
TRAIN_GPUS="${TRAIN_GPUS:-${GPU_ID}}"
INFER_GPUS="${INFER_GPUS:-${GPU_ID}}"
TRAIN_MASTER_ADDR="${TRAIN_MASTER_ADDR:-127.0.0.1}"
TRAIN_MASTER_PORT="${TRAIN_MASTER_PORT:-29771}"
RUN_NAME="${RUN_NAME:-inr_mln3_scoreperf_mask_startctrl_asap_$(date +%Y%m%d_%H%M%S)}"
INFER_NUM_WORKERS="${INFER_NUM_WORKERS:-8}"
METRIC_NUM_WORKERS="${METRIC_NUM_WORKERS:-8}"
SAMPLING_NUM_SAMPLES="${SAMPLING_NUM_SAMPLES:-1}"
DET_NUM_SAMPLES="${DET_NUM_SAMPLES:-1}"
MAX_GT_PER_SCORE="${MAX_GT_PER_SCORE:-}"
INFER_BATCH_SIZE_WINDOWS="${INFER_BATCH_SIZE_WINDOWS:-8}"
MERGE_MODE="${MERGE_MODE:-continuation}"
CONTINUATION_DROP_RATIO="${CONTINUATION_DROP_RATIO:-0.0}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-}"
TRAIN_LIMIT_WORKS="${TRAIN_LIMIT_WORKS:-}"
TRAIN_LIMIT_PERFORMANCES_PER_WORK="${TRAIN_LIMIT_PERFORMANCES_PER_WORK:-}"
TRAIN_LIMIT_WINDOWS_PER_WORK="${TRAIN_LIMIT_WINDOWS_PER_WORK:-}"
INFER_MAX_WORKS="${INFER_MAX_WORKS:-}"

RUN_DIR="${RUN_DIR:-results/inr_pipeline/${RUN_NAME}}"
TRAIN_LOG="${TRAIN_LOG:-${RUN_DIR}/train.log}"
EVALUATE_LOG="${EVALUATE_LOG:-${RUN_DIR}/evaluate.log}"
SUMMARY_JSON="${SUMMARY_JSON:-${RUN_DIR}/summary.json}"
PLOT_PATH="${PLOT_PATH:-${RUN_DIR}/asap_label_distribution.png}"
RUN_CONFIG="${RUN_CONFIG:-${RUN_DIR}/config.json}"

DET_DIR="${RUN_DIR}/deterministic"
SAMPLING_DIR="${RUN_DIR}/sampling"
TMP_DIR="${RUN_DIR}/_tmp"
TRAIN_ROOT="${RUN_DIR}/training"
TF_LOG_ROOT="${RUN_DIR}/tf-logs"

mkdir -p "${RUN_DIR}" "${TMP_DIR}" "${TRAIN_ROOT}" "${TF_LOG_ROOT}"

csv_count() {
  local csv="$1"
  python - "${csv}" <<'PY'
import sys
items = [item.strip() for item in sys.argv[1].split(",") if item.strip()]
print(len(items))
PY
}

csv_item() {
  local csv="$1"
  local index="$2"
  python - "${csv}" "${index}" <<'PY'
import sys
items = [item.strip() for item in sys.argv[1].split(",") if item.strip()]
idx = int(sys.argv[2])
if not items:
    raise SystemExit("empty GPU list")
print(items[idx] if idx < len(items) else items[0])
PY
}

TRAIN_GPU_COUNT="$(csv_count "${TRAIN_GPUS}")"
INFER_GPU_COUNT="$(csv_count "${INFER_GPUS}")"
if [[ "${TRAIN_GPU_COUNT}" -lt 1 ]]; then
  echo "TRAIN_GPUS must contain at least one GPU id" >&2
  exit 1
fi
if [[ "${INFER_GPU_COUNT}" -lt 1 || "${INFER_GPU_COUNT}" -gt 2 ]]; then
  echo "INFER_GPUS must contain one or two GPU ids, got: ${INFER_GPUS}" >&2
  exit 1
fi
DET_GPU="$(csv_item "${INFER_GPUS}" 0)"
SAMPLING_GPU="$(csv_item "${INFER_GPUS}" 1)"

python - "${CONFIG}" "${RUN_CONFIG}" "${TRAIN_ROOT}" "${TF_LOG_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

src, dst, train_root, tf_log_root = map(Path, sys.argv[1:])
config = json.loads(src.read_text(encoding="utf-8"))
config["output_dir"] = str(train_root)
config["logging_dir"] = str(tf_log_root)
dst.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY

{
echo "START $(date '+%F %T')"
echo "ROOT_DIR ${ROOT_DIR}"
echo "RUN_NAME ${RUN_NAME}"
echo "CONFIG ${CONFIG}"
echo "RUN_CONFIG ${RUN_CONFIG}"
echo "GPU_ID ${GPU_ID}"
echo "TRAIN_GPUS ${TRAIN_GPUS}"
echo "TRAIN_GPU_COUNT ${TRAIN_GPU_COUNT}"
echo "TRAIN_MASTER_ADDR ${TRAIN_MASTER_ADDR}"
echo "TRAIN_MASTER_PORT ${TRAIN_MASTER_PORT}"
echo "INFER_GPUS ${INFER_GPUS}"
echo "INFER_GPU_COUNT ${INFER_GPU_COUNT}"
echo "DET_GPU ${DET_GPU}"
echo "SAMPLING_GPU ${SAMPLING_GPU}"
echo "RUN_DIR ${RUN_DIR}"
echo "TRAIN_LOG ${TRAIN_LOG}"
echo "EVALUATE_LOG ${EVALUATE_LOG}"
echo "SUMMARY_JSON ${SUMMARY_JSON}"
echo "PLOT_PATH ${PLOT_PATH}"
echo "INFER_NUM_WORKERS ${INFER_NUM_WORKERS}"
echo "METRIC_NUM_WORKERS ${METRIC_NUM_WORKERS}"
echo "PERFORMANCE_DATASET ASAP"
} | tee -a "${EVALUATE_LOG}"

TRAIN_ARGS=(--config "${RUN_CONFIG}")
if [[ -n "${TRAIN_MAX_STEPS}" ]]; then
  TRAIN_ARGS+=(--max_steps "${TRAIN_MAX_STEPS}")
fi
if [[ -n "${TRAIN_LIMIT_WORKS}" ]]; then
  TRAIN_ARGS+=(--limit_works "${TRAIN_LIMIT_WORKS}")
fi
if [[ -n "${TRAIN_LIMIT_PERFORMANCES_PER_WORK}" ]]; then
  TRAIN_ARGS+=(--limit_performances_per_work "${TRAIN_LIMIT_PERFORMANCES_PER_WORK}")
fi
if [[ -n "${TRAIN_LIMIT_WINDOWS_PER_WORK}" ]]; then
  TRAIN_ARGS+=(--limit_windows_per_work "${TRAIN_LIMIT_WINDOWS_PER_WORK}")
fi

mkdir -p "${TRAIN_ROOT}"
TRAIN_MARKER="${TMP_DIR}/train_start.marker"
touch "${TRAIN_MARKER}"

if [[ "${TRAIN_GPU_COUNT}" -gt 1 ]]; then
  echo "[$(date '+%F %T')] train: DDP start, GPUs=${TRAIN_GPUS}, nproc=${TRAIN_GPU_COUNT}" | tee -a "${EVALUATE_LOG}"
  CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo \
    TORCH_NCCL_BLOCKING_WAIT=1 NCCL_DEBUG=WARN \
    torchrun \
      --nnodes=1 \
      --nproc_per_node="${TRAIN_GPU_COUNT}" \
      --master_addr="${TRAIN_MASTER_ADDR}" \
      --master_port="${TRAIN_MASTER_PORT}" \
      src/train/train_inr.py "${TRAIN_ARGS[@]}" 2>&1 | tee -a "${TRAIN_LOG}"
else
  echo "[$(date '+%F %T')] train: single GPU start, GPU=${TRAIN_GPUS}" | tee -a "${EVALUATE_LOG}"
  CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" PYTHONUNBUFFERED=1 \
    python src/train/train_inr.py "${TRAIN_ARGS[@]}" 2>&1 | tee -a "${TRAIN_LOG}"
fi
echo "[$(date '+%F %T')] train: finished" | tee -a "${EVALUATE_LOG}"

TRAIN_OUTPUT_DIR="$(
  find "${TRAIN_ROOT}" -maxdepth 1 -mindepth 1 -type d -name 'inr_*' -newer "${TRAIN_MARKER}" \
    -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-
)"
if [[ -z "${TRAIN_OUTPUT_DIR}" ]]; then
  echo "Could not locate this run's training output under ${TRAIN_ROOT}" >&2
  exit 1
fi

if [[ -d "${TRAIN_OUTPUT_DIR}/checkpoint-best" ]]; then
  CHECKPOINT="${TRAIN_OUTPUT_DIR}/checkpoint-best"
else
  CHECKPOINT="${TRAIN_OUTPUT_DIR}"
fi
{
echo "TRAIN_OUTPUT_DIR ${TRAIN_OUTPUT_DIR}"
echo "CHECKPOINT ${CHECKPOINT}"
} | tee -a "${EVALUATE_LOG}"

COMMON_INFER_ARGS=(
  --config "${RUN_CONFIG}"
  --checkpoint "${CHECKPOINT}"
  --split test
  --performance-dataset ASAP
  --num-workers "${INFER_NUM_WORKERS}"
  --batch-size-windows "${INFER_BATCH_SIZE_WINDOWS}"
  --merge-mode "${MERGE_MODE}"
  --continuation-drop-ratio "${CONTINUATION_DROP_RATIO}"
  --device cuda
)
if [[ -n "${MAX_GT_PER_SCORE}" ]]; then
  COMMON_INFER_ARGS+=(--max-gt-per-score "${MAX_GT_PER_SCORE}")
fi
if [[ -n "${INFER_MAX_WORKS}" ]]; then
  COMMON_INFER_ARGS+=(--max-works "${INFER_MAX_WORKS}")
fi

run_infer() {
  local protocol="$1"
  local num_samples="$2"
  local out_dir="$3"
  local infer_gpu="$4"

  echo "[$(date '+%F %T')] infer ${protocol}: GPU=${infer_gpu}, workers=${INFER_NUM_WORKERS}"
  CUDA_VISIBLE_DEVICES="${infer_gpu}" PYTHONUNBUFFERED=1 \
    python src/inference/infer_inr_testset.py \
      "${COMMON_INFER_ARGS[@]}" \
      --protocol "${protocol}" \
      --num-samples "${num_samples}" \
      --output-dir "${out_dir}"
  echo "[$(date '+%F %T')] infer ${protocol}: finished"
}

mkdir -p "${DET_DIR}" "${SAMPLING_DIR}"
run_infer deterministic "${DET_NUM_SAMPLES}" "${DET_DIR}" "${DET_GPU}" 2>&1 | tee -a "${EVALUATE_LOG}" &
DET_PID=$!
run_infer sampling "${SAMPLING_NUM_SAMPLES}" "${SAMPLING_DIR}" "${SAMPLING_GPU}" 2>&1 | tee -a "${EVALUATE_LOG}" &
SAMPLING_PID=$!

DET_STATUS=0
SAMPLING_STATUS=0
wait "${DET_PID}" || DET_STATUS=$?
wait "${SAMPLING_PID}" || SAMPLING_STATUS=$?
if [[ "${DET_STATUS}" -ne 0 || "${SAMPLING_STATUS}" -ne 0 ]]; then
  echo "Inference failed: deterministic=${DET_STATUS}, sampling=${SAMPLING_STATUS}" | tee -a "${EVALUATE_LOG}" >&2
  exit 1
fi

SUMMARY_ARGS=(
  --deterministic-manifest "${DET_DIR}/prediction_manifest.json"
  --sampling-manifest "${SAMPLING_DIR}/prediction_manifest.json"
  --output-json "${SUMMARY_JSON}"
  --output-plot "${PLOT_PATH}"
  --config "${RUN_CONFIG}"
  --checkpoint "${CHECKPOINT}"
  --train-output-dir "${TRAIN_OUTPUT_DIR}"
  --pipeline-log "${TRAIN_LOG}"
  --evaluate-log "${EVALUATE_LOG}"
  --num-workers "${METRIC_NUM_WORKERS}"
)
if [[ -n "${MAX_GT_PER_SCORE}" ]]; then
  SUMMARY_ARGS+=(--max-gt-per-score "${MAX_GT_PER_SCORE}")
fi

echo "[$(date '+%F %T')] summarize: ASAP metrics and distribution plot, workers=${METRIC_NUM_WORKERS}" | tee -a "${EVALUATE_LOG}"
PYTHONUNBUFFERED=1 python src/evaluate/summarize_inr_asap_pipeline.py "${SUMMARY_ARGS[@]}" 2>&1 | tee -a "${EVALUATE_LOG}"

rm -rf "${TMP_DIR}"

{
echo "SUMMARY_JSON ${SUMMARY_JSON}"
echo "PLOT_PATH ${PLOT_PATH}"
echo "TRAIN_LOG ${TRAIN_LOG}"
echo "EVALUATE_LOG ${EVALUATE_LOG}"
echo "RAW_OUTPUT deterministic ${DET_DIR}/raw_outputs ${DET_DIR}/prediction_manifest.json ${DET_DIR}/evaluate_list.json"
echo "RAW_OUTPUT sampling ${SAMPLING_DIR}/raw_outputs ${SAMPLING_DIR}/prediction_manifest.json ${SAMPLING_DIR}/evaluate_list.json"
echo "MIDIS deterministic ${DET_DIR}/midis"
echo "MIDIS sampling ${SAMPLING_DIR}/midis"
echo "END $(date '+%F %T')"
} | tee -a "${EVALUATE_LOG}"
