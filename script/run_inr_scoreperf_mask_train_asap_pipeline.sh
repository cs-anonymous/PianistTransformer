#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Only one env var users need to set
: "${CONFIG:?CONFIG is required (path to json config file)}"
: "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is required (e.g. 0,1)}"

# Hardcoded pipeline settings
INFER_NUM_WORKERS=8
METRIC_NUM_WORKERS=8
DET_NUM_SAMPLES=1
SAMPLING_NUM_SAMPLES=1
INFER_BATCH_SIZE_WINDOWS=8
MERGE_MODE="continuation"
CONTINUATION_DROP_RATIO=0.0

# Run layout
RUN_NAME="run_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="results/inr_pipeline/${RUN_NAME}"
TRAIN_LOG="${RUN_DIR}/train.log"
EVALUATE_LOG="${RUN_DIR}/evaluate.log"
SUMMARY_JSON="${RUN_DIR}/summary.json"
PLOT_PATH="${RUN_DIR}/asap_label_distribution.png"
RUN_CONFIG="${RUN_DIR}/config.json"
DET_DIR="${RUN_DIR}/deterministic"
SAMPLING_DIR="${RUN_DIR}/sampling"
TMP_DIR="${RUN_DIR}/_tmp"
TRAIN_ROOT="${RUN_DIR}/training"
TF_LOG_ROOT="${RUN_DIR}/tf-logs"
mkdir -p "${RUN_DIR}" "${TMP_DIR}" "${TRAIN_ROOT}" "${TF_LOG_ROOT}"

# GPU list (inherited from CUDA_VISIBLE_DEVICES)
IFS=',' read -ra GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
TRAIN_GPU_COUNT=${#GPU_LIST[@]}
[[ ${TRAIN_GPU_COUNT} -ge 1 && ${TRAIN_GPU_COUNT} -le 2 ]] \
  || { echo "CUDA_VISIBLE_DEVICES must be 1 or 2 ids, got: ${CUDA_VISIBLE_DEVICES}" >&2; exit 1; }
DET_GPU="${GPU_LIST[0]}"
SAMPLING_GPU="${GPU_LIST[1]:-${GPU_LIST[0]}}"

# Materialize config (rewrite output/logging dirs into the run dir)
python - "${CONFIG}" "${RUN_CONFIG}" "${TRAIN_ROOT}" "${TF_LOG_ROOT}" <<'PY'
import json, sys
src, dst, train_root, tf_log_root = sys.argv[1:5]
cfg = json.loads(open(src).read())
cfg["output_dir"] = train_root
cfg["logging_dir"] = tf_log_root
open(dst, "w").write(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
PY

{
  echo "START $(date '+%F %T')"
  echo "CONFIG ${CONFIG}"
  echo "CUDA_VISIBLE_DEVICES ${CUDA_VISIBLE_DEVICES}"
  echo "RUN_DIR ${RUN_DIR}"
} | tee -a "${EVALUATE_LOG}"

# --- Train -----------------------------------------------------------------
TRAIN_MARKER="${TMP_DIR}/train_start.marker"
touch "${TRAIN_MARKER}"
if [[ "${TRAIN_GPU_COUNT}" -gt 1 ]]; then
  echo "[$(date '+%F %T')] train: DDP start, GPUs=${CUDA_VISIBLE_DEVICES}, nproc=${TRAIN_GPU_COUNT}" | tee -a "${EVALUATE_LOG}"
  PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo \
    TORCH_NCCL_BLOCKING_WAIT=1 NCCL_DEBUG=WARN \
    torchrun --nnodes=1 --nproc_per_node="${TRAIN_GPU_COUNT}" \
      src/train/train_inr.py --config "${RUN_CONFIG}" 2>&1 | tee -a "${TRAIN_LOG}"
else
  echo "[$(date '+%F %T')] train: single GPU start, GPU=${CUDA_VISIBLE_DEVICES}" | tee -a "${EVALUATE_LOG}"
  PYTHONUNBUFFERED=1 \
    python src/train/train_inr.py --config "${RUN_CONFIG}" 2>&1 | tee -a "${TRAIN_LOG}"
fi
echo "[$(date '+%F %T')] train: finished" | tee -a "${EVALUATE_LOG}"

# Find this run's training output dir + checkpoint
TRAIN_OUTPUT_DIR="$(
  find "${TRAIN_ROOT}" -maxdepth 1 -mindepth 1 -type d -name 'inr_*' -newer "${TRAIN_MARKER}" \
    -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-
)"
[[ -n "${TRAIN_OUTPUT_DIR}" ]] \
  || { echo "Could not locate training output under ${TRAIN_ROOT}" >&2; exit 1; }
CHECKPOINT="${TRAIN_OUTPUT_DIR}"
[[ -d "${TRAIN_OUTPUT_DIR}/checkpoint-best" ]] && CHECKPOINT="${TRAIN_OUTPUT_DIR}/checkpoint-best"
{
  echo "TRAIN_OUTPUT_DIR ${TRAIN_OUTPUT_DIR}"
  echo "CHECKPOINT ${CHECKPOINT}"
} | tee -a "${EVALUATE_LOG}"

# --- Inference: deterministic + sampling in parallel -----------------------
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

run_infer() {
  local protocol="$1" num_samples="$2" out_dir="$3" infer_gpu="$4"
  echo "[$(date '+%F %T')] infer ${protocol}: GPU=${infer_gpu}, workers=${INFER_NUM_WORKERS}" | tee -a "${EVALUATE_LOG}"
  CUDA_VISIBLE_DEVICES="${infer_gpu}" PYTHONUNBUFFERED=1 \
    python src/inference/infer_inr_testset.py "${COMMON_INFER_ARGS[@]}" \
      --protocol "${protocol}" --num-samples "${num_samples}" --output-dir "${out_dir}" \
    2>&1 | tee -a "${EVALUATE_LOG}"
  echo "[$(date '+%F %T')] infer ${protocol}: finished" | tee -a "${EVALUATE_LOG}"
}

mkdir -p "${DET_DIR}" "${SAMPLING_DIR}"
run_infer deterministic "${DET_NUM_SAMPLES}" "${DET_DIR}" "${DET_GPU}" &
DET_PID=$!
run_infer sampling "${SAMPLING_NUM_SAMPLES}" "${SAMPLING_DIR}" "${SAMPLING_GPU}" &
SAMPLING_PID=$!
wait "${DET_PID}" || { echo "deterministic inference failed" >&2; exit 1; }
wait "${SAMPLING_PID}" || { echo "sampling inference failed" >&2; exit 1; }

# --- Summarize -------------------------------------------------------------
echo "[$(date '+%F %T')] summarize: ASAP metrics + plot, workers=${METRIC_NUM_WORKERS}" | tee -a "${EVALUATE_LOG}"
PYTHONUNBUFFERED=1 python src/evaluate/summarize_inr_asap_pipeline.py \
  --deterministic-manifest "${DET_DIR}/prediction_manifest.json" \
  --sampling-manifest      "${SAMPLING_DIR}/prediction_manifest.json" \
  --output-json            "${SUMMARY_JSON}" \
  --output-plot            "${PLOT_PATH}" \
  --config                 "${RUN_CONFIG}" \
  --checkpoint             "${CHECKPOINT}" \
  --train-output-dir       "${TRAIN_OUTPUT_DIR}" \
  --pipeline-log           "${TRAIN_LOG}" \
  --evaluate-log           "${EVALUATE_LOG}" \
  --num-workers            "${METRIC_NUM_WORKERS}" \
  2>&1 | tee -a "${EVALUATE_LOG}"

rm -rf "${TMP_DIR}"
echo "END $(date '+%F %T')" | tee -a "${EVALUATE_LOG}"
