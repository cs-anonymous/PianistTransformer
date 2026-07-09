#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

: "${CONFIG:?CONFIG is required}"
: "${CHECKPOINT:?CHECKPOINT is required}"

RUN_DIR="${RUN_DIR:-$(dirname "$(dirname "${CHECKPOINT}")")}"
NUM_WORKERS="${NUM_WORKERS:-8}"
METRIC_NUM_WORKERS="${METRIC_NUM_WORKERS:-${NUM_WORKERS}}"
KPASS_BATCH_SIZE_WINDOWS="${KPASS_BATCH_SIZE_WINDOWS:-8}"
AR_BATCH_SIZE_WINDOWS="${AR_BATCH_SIZE_WINDOWS:-8}"
ROLLOUT_KS="${ROLLOUT_KS:-0,1,4,16,full}"
SAMPLING_STRATEGY="${SAMPLING_STRATEGY:-sample}"
FEEDBACK_STRATEGY="${FEEDBACK_STRATEGY:-sample}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
SEED="${SEED:-2042}"
MERGE_MODE="${MERGE_MODE:-continuation}"
CONTINUATION_DROP_RATIO="${CONTINUATION_DROP_RATIO:-0.0}"
DEVICE="${DEVICE:-cuda}"
PLOT_KIND="${PLOT_KIND:-density}"
SPLIT="${SPLIT:-test}"
PERFORMANCE_DATASET="${PERFORMANCE_DATASET:-ASAP}"
SCORE_SOURCE_LIST="${SCORE_SOURCE_LIST:-}"
SKIP_EXISTING_PIPELINE_OUTPUTS="${SKIP_EXISTING_PIPELINE_OUTPUTS:-1}"
PIPELINE_NAME="${PIPELINE_NAME:-sample_rollout_k01416_full_w${NUM_WORKERS}}"
PIPELINE_DIR="${PIPELINE_DIR:-${RUN_DIR}/${PIPELINE_NAME}}"
PIPELINE_LOG="${PIPELINE_LOG:-${PIPELINE_DIR}/pipeline.log}"
KPASS_DIR="${KPASS_DIR:-${PIPELINE_DIR}/kpass_eval}"
AR_DIR="${AR_DIR:-${PIPELINE_DIR}/ar_infer}"
SUMMARY_DIR="${SUMMARY_DIR:-${PIPELINE_DIR}/summary}"

mkdir -p "${PIPELINE_DIR}" "${KPASS_DIR}" "${AR_DIR}" "${SUMMARY_DIR}"

score_source_args=()
if [[ -n "${SCORE_SOURCE_LIST}" ]]; then
  score_source_args+=(--score-source-list "${SCORE_SOURCE_LIST}")
fi

run_kpass() {
  echo "[$(date '+%F %T')] k-pass rollout eval ks=${ROLLOUT_KS}" | tee -a "${PIPELINE_LOG}"
  PYTHONUNBUFFERED=1 python src/evaluate/eval_inr_rollout_current.py \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --output-dir "${KPASS_DIR}" \
    --split "${SPLIT}" \
    --performance-dataset "${PERFORMANCE_DATASET}" \
    "${score_source_args[@]}" \
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
  echo "[$(date '+%F %T')] sampling full AR infer" | tee -a "${PIPELINE_LOG}"
  PYTHONUNBUFFERED=1 python src/inference/infer_inr_testset.py \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --split "${SPLIT}" \
    --performance-dataset "${PERFORMANCE_DATASET}" \
    "${score_source_args[@]}" \
    --num-workers "${NUM_WORKERS}" \
    --batch-size-windows "${AR_BATCH_SIZE_WINDOWS}" \
    --merge-mode "${MERGE_MODE}" \
    --continuation-drop-ratio "${CONTINUATION_DROP_RATIO}" \
    --device "${DEVICE}" \
    --protocol sampling \
    --sampling-strategy "${SAMPLING_STRATEGY}" \
    --num-samples "${NUM_SAMPLES}" \
    --seed "${SEED}" \
    --output-dir "${AR_DIR}" 2>&1 | tee -a "${PIPELINE_LOG}"
}

run_summary() {
  echo "[$(date '+%F %T')] rollout summary + eval/stat" | tee -a "${PIPELINE_LOG}"
  PYTHONUNBUFFERED=1 python src/evaluate/summarize_inr_rollout_pipeline.py \
    --kpass-summary "${KPASS_DIR}/summary.json" \
    --ar-manifest "${AR_DIR}/prediction_manifest.json" \
    --output-dir "${SUMMARY_DIR}" \
    "${score_source_args[@]}" \
    --num-workers "${METRIC_NUM_WORKERS}" \
    --plot-kind "${PLOT_KIND}" 2>&1 | tee -a "${PIPELINE_LOG}"
}

if [[ "${SKIP_EXISTING_PIPELINE_OUTPUTS}" == "1" && -s "${KPASS_DIR}/summary.json" ]]; then
  echo "[$(date '+%F %T')] reuse ${KPASS_DIR}/summary.json" | tee -a "${PIPELINE_LOG}"
else
  run_kpass
fi

if [[ "${SKIP_EXISTING_PIPELINE_OUTPUTS}" == "1" && -s "${AR_DIR}/prediction_manifest.json" ]]; then
  echo "[$(date '+%F %T')] reuse ${AR_DIR}/prediction_manifest.json" | tee -a "${PIPELINE_LOG}"
else
  run_ar_infer
fi

if [[ "${SKIP_EXISTING_PIPELINE_OUTPUTS}" == "1" && -s "${SUMMARY_DIR}/pipeline_summary.json" ]]; then
  echo "[$(date '+%F %T')] reuse ${SUMMARY_DIR}/pipeline_summary.json" | tee -a "${PIPELINE_LOG}"
else
  run_summary
fi

echo "[$(date '+%F %T')] done ${PIPELINE_DIR}" | tee -a "${PIPELINE_LOG}"
