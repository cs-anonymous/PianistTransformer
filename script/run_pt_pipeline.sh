#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

: "${CONFIG:?CONFIG is required (path to json config file)}"
: "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is required (e.g. 3 or 2,3)}"

: "${INFER_NUM_WORKERS:=8}"
: "${METRIC_NUM_WORKERS:=8}"
: "${DET_NUM_SAMPLES:=1}"
: "${SAMPLING_NUM_SAMPLES:=1}"
: "${INFER_BATCH_SIZE:=8}"
: "${PT_DATA_NUM_WORKERS:=30}"
: "${PERFORMANCE_DATASET:=ASAP}"

RUN_NAME="${CONFIG##*/}"
RUN_NAME="${RUN_NAME%.json}_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="results/pt_pipeline/${RUN_NAME}"
TRAIN_LOG="${RUN_DIR}/train.log"
EVALUATE_LOG="${RUN_DIR}/evaluate.log"
SUMMARY_JSON="${RUN_DIR}/summary.json"
PLOT_PATH="${RUN_DIR}/asap_label_distribution.png"
RUN_CONFIG="${RUN_DIR}/config.json"
DET_DIR="${RUN_DIR}/deterministic"
SAMPLING_DIR="${RUN_DIR}/sampling"
TRAIN_ROOT="${RUN_DIR}/training"
TF_LOG_ROOT="${RUN_DIR}/tf-logs"
TMP_DIR="${RUN_DIR}/_tmp"
mkdir -p "${RUN_DIR}" "${DET_DIR}" "${SAMPLING_DIR}" "${TRAIN_ROOT}" "${TF_LOG_ROOT}" "${TMP_DIR}"

IFS=',' read -ra GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
TRAIN_GPU_COUNT=${#GPU_LIST[@]}
[[ ${TRAIN_GPU_COUNT} -ge 1 && ${TRAIN_GPU_COUNT} -le 2 ]] \
  || { echo "CUDA_VISIBLE_DEVICES must be 1 or 2 ids, got: ${CUDA_VISIBLE_DEVICES}" >&2; exit 1; }
DET_GPU="${GPU_LIST[0]}"
SAMPLING_GPU="${GPU_LIST[1]:-${GPU_LIST[0]}}"

MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

python - "${CONFIG}" "${RUN_CONFIG}" "${TRAIN_ROOT}" "${TF_LOG_ROOT}" <<'PY'
import json, sys
src, dst, train_root, tf_log_root = sys.argv[1:5]
cfg = json.loads(open(src, encoding="utf-8").read())
cfg["output_dir"] = train_root
cfg["logging_dir"] = tf_log_root
open(dst, "w", encoding="utf-8").write(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
PY

DATA_FILE="$(python - "${RUN_CONFIG}" <<'PY'
import json, sys
cfg = json.loads(open(sys.argv[1], encoding="utf-8").read())
paths = cfg.get("data_paths") or []
if not paths:
    raise SystemExit("config.data_paths is empty")
print(paths[0])
PY
)"
PROCESSED_RAW_DIR="$(python - "${RUN_CONFIG}" <<'PY'
import json, sys
cfg = json.loads(open(sys.argv[1], encoding="utf-8").read())
print(cfg.get("processed_raw_dir", "../PianoCoRe/processed_raw"))
PY
)"
METADATA_PATH="$(python - "${RUN_CONFIG}" <<'PY'
import json, sys
cfg = json.loads(open(sys.argv[1], encoding="utf-8").read())
print(cfg.get("metadata_path", "../PianoCoRe/metadata.csv"))
PY
)"
PT_DATA_PERFORMANCE_DATASET="$(python - "${RUN_CONFIG}" <<'PY'
import json, sys
cfg = json.loads(open(sys.argv[1], encoding="utf-8").read())
print(cfg.get("pt_data_performance_dataset") or cfg.get("data_performance_dataset") or "")
PY
)"
PT_DATA_SPLIT="$(python - "${RUN_CONFIG}" <<'PY'
import json, sys
cfg = json.loads(open(sys.argv[1], encoding="utf-8").read())
print(cfg.get("pt_data_split") or cfg.get("data_split") or "")
PY
)"

{
  echo "START $(date '+%F %T')"
  echo "CONFIG ${CONFIG}"
  echo "CUDA_VISIBLE_DEVICES ${CUDA_VISIBLE_DEVICES}"
  echo "MASTER_PORT ${MASTER_PORT}"
  echo "RUN_DIR ${RUN_DIR}"
  echo "PT_DATA_FILE ${DATA_FILE}"
  echo "PROCESSED_RAW_DIR ${PROCESSED_RAW_DIR}"
  if [[ -n "${PT_DATA_PERFORMANCE_DATASET}" ]]; then
    echo "PT_DATA_PERFORMANCE_DATASET ${PT_DATA_PERFORMANCE_DATASET}"
  fi
  if [[ -n "${PT_DATA_SPLIT}" ]]; then
    echo "PT_DATA_SPLIT ${PT_DATA_SPLIT}"
  fi
} | tee -a "${EVALUATE_LOG}"

if [[ ! -s "${DATA_FILE}" ]]; then
  echo "[$(date '+%F %T')] data: generating PT SFT jsonl from processed_raw" | tee -a "${EVALUATE_LOG}"
  GEN_ARGS=(
    --processed-dir "${PROCESSED_RAW_DIR}" \
    --output-file "${DATA_FILE}" \
    --time-normalization raw \
    --num-workers "${PT_DATA_NUM_WORKERS}"
  )
  [[ -n "${PT_DATA_PERFORMANCE_DATASET}" ]] && GEN_ARGS+=(--performance-dataset "${PT_DATA_PERFORMANCE_DATASET}")
  [[ -n "${PT_DATA_SPLIT}" ]] && GEN_ARGS+=(--split "${PT_DATA_SPLIT}")
  PYTHONPATH="${ROOT_DIR}" PYTHONUNBUFFERED=1 python src/data_process/generate_pt_sft_from_inr_json_multiprocess.py "${GEN_ARGS[@]}" \
    2>&1 | tee -a "${EVALUATE_LOG}"
else
  echo "[$(date '+%F %T')] data: reuse existing ${DATA_FILE}" | tee -a "${EVALUATE_LOG}"
fi

echo "[$(date '+%F %T')] train: PT SFT start, GPUs=${CUDA_VISIBLE_DEVICES}, nproc=${TRAIN_GPU_COUNT}" | tee -a "${EVALUATE_LOG}"
TRAIN_MARKER="${TMP_DIR}/train_start.marker"
touch "${TRAIN_MARKER}"
if [[ "${TRAIN_GPU_COUNT}" -gt 1 ]]; then
  MASTER_PORT="${MASTER_PORT}" PYTHONPATH="${ROOT_DIR}" PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo \
    TORCH_NCCL_BLOCKING_WAIT=1 NCCL_DEBUG=WARN \
    torchrun --nnodes=1 --nproc_per_node="${TRAIN_GPU_COUNT}" \
      --master_addr=127.0.0.1 --master_port="${MASTER_PORT}" \
      src/train/sft.py --config "${RUN_CONFIG}" 2>&1 | tee -a "${TRAIN_LOG}"
else
  PYTHONPATH="${ROOT_DIR}" PYTHONUNBUFFERED=1 python src/train/sft.py --config "${RUN_CONFIG}" 2>&1 | tee -a "${TRAIN_LOG}"
fi
echo "[$(date '+%F %T')] train: finished" | tee -a "${EVALUATE_LOG}"

TRAIN_OUTPUT_DIR="$(
  find "${TRAIN_ROOT}" -maxdepth 1 -mindepth 1 -type d -name 'sft_*' -newer "${TRAIN_MARKER}" \
    -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-
)"
[[ -n "${TRAIN_OUTPUT_DIR}" ]] || { echo "Could not locate PT training output under ${TRAIN_ROOT}" >&2; exit 1; }
CHECKPOINT="${TRAIN_OUTPUT_DIR}"
[[ -d "${TRAIN_OUTPUT_DIR}/checkpoint-best" ]] && CHECKPOINT="${TRAIN_OUTPUT_DIR}/checkpoint-best"
{
  echo "TRAIN_OUTPUT_DIR ${TRAIN_OUTPUT_DIR}"
  echo "CHECKPOINT ${CHECKPOINT}"
} | tee -a "${EVALUATE_LOG}"

run_infer() {
  local protocol="$1" num_samples="$2" out_dir="$3" infer_gpu="$4"
  echo "[$(date '+%F %T')] infer ${protocol}: GPU=${infer_gpu}, workers=${INFER_NUM_WORKERS}, batch_size=${INFER_BATCH_SIZE}" | tee -a "${EVALUATE_LOG}"
  CUDA_VISIBLE_DEVICES="${infer_gpu}" PYTHONPATH="${ROOT_DIR}" PYTHONUNBUFFERED=1 \
    python src/inference/infer_pt_testset.py \
      --model-path "${CHECKPOINT}" \
      --metadata "${METADATA_PATH}" \
      --midi-root ../PianoCoRe/refined \
      --split test \
      --performance-dataset "${PERFORMANCE_DATASET}" \
      --output-dir "${out_dir}" \
      --device cuda \
      --batch-size "${INFER_BATCH_SIZE}" \
      --num-workers "${INFER_NUM_WORKERS}" \
      --protocol "${protocol}" \
      --num-samples "${num_samples}" \
      --overlap-ratio 0.125 \
      --max-context-length 4096 \
    2>&1 | tee -a "${EVALUATE_LOG}"
  echo "[$(date '+%F %T')] infer ${protocol}: finished" | tee -a "${EVALUATE_LOG}"
}

run_infer deterministic "${DET_NUM_SAMPLES}" "${DET_DIR}" "${DET_GPU}" &
DET_PID=$!
run_infer sampling "${SAMPLING_NUM_SAMPLES}" "${SAMPLING_DIR}" "${SAMPLING_GPU}" &
SAMPLING_PID=$!
wait "${DET_PID}" || { echo "deterministic inference failed" >&2; exit 1; }
wait "${SAMPLING_PID}" || { echo "sampling inference failed" >&2; exit 1; }

echo "[$(date '+%F %T')] eval: PN/PP metrics" | tee -a "${EVALUATE_LOG}"
PYTHONPATH="${ROOT_DIR}" PYTHONUNBUFFERED=1 python src/evaluate/evaluate_inr_saved_midis.py \
  --prediction-manifest "${DET_DIR}/prediction_manifest.json" \
  --output-json "${DET_DIR}/eval_pn_pp_metrics.json" \
  --num-workers "${METRIC_NUM_WORKERS}" \
  2>&1 | tee -a "${EVALUATE_LOG}"
PYTHONPATH="${ROOT_DIR}" PYTHONUNBUFFERED=1 python src/evaluate/evaluate_inr_saved_midis.py \
  --prediction-manifest "${SAMPLING_DIR}/prediction_manifest.json" \
  --output-json "${SAMPLING_DIR}/eval_pn_pp_metrics.json" \
  --num-workers "${METRIC_NUM_WORKERS}" \
  2>&1 | tee -a "${EVALUATE_LOG}"

echo "[$(date '+%F %T')] summarize/statistics" | tee -a "${EVALUATE_LOG}"
PYTHONPATH="${ROOT_DIR}" PYTHONUNBUFFERED=1 python src/evaluate/summarize_inr_asap_pipeline.py \
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
