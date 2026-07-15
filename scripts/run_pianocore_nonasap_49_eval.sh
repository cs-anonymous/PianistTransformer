#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/kaititech/EPR/PianistTransformer"
cd "${ROOT_DIR}"

NAME="$1"
GPU="$2"
CONFIG="$3"
CHECKPOINT="$4"

SCORE_LIST="${ROOT_DIR}/configs/local_generated/pianocore_nonasap_test_49works_gt100.txt"
OUT_ROOT="${ROOT_DIR}/results/pianocore_nonasap_49works_compare_20260714/${NAME}"
LOG="${OUT_ROOT}/run.log"
MAX_GT_PER_SCORE="${MAX_GT_PER_SCORE:-20}"
NUM_WORKERS="${NUM_WORKERS:-2}"
BATCH_SIZE_WINDOWS="${BATCH_SIZE_WINDOWS:-8}"

mkdir -p "${OUT_ROOT}"

run_infer() {
  local protocol="$1"
  local out_dir="${OUT_ROOT}/${protocol}"
  mkdir -p "${out_dir}"
  echo "[$(date '+%F %T')] ${NAME} ${protocol} infer on GPU ${GPU}" | tee -a "${LOG}"
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 python src/inference/infer_inr_testset.py \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --split test \
    --exclude-performance-dataset ASAP \
    --score-source-list "${SCORE_LIST}" \
    --max-gt-per-score "${MAX_GT_PER_SCORE}" \
    --num-workers "${NUM_WORKERS}" \
    --batch-size-windows "${BATCH_SIZE_WINDOWS}" \
    --merge-mode continuation \
    --continuation-drop-ratio 0.0 \
    --device cuda \
    --protocol "${protocol}" \
    --num-samples 1 \
    --deterministic-strategy greedy \
    --output-dir "${out_dir}" 2>&1 | tee -a "${LOG}"
}

run_eval() {
  local protocol="$1"
  echo "[$(date '+%F %T')] ${NAME} ${protocol} eval" | tee -a "${LOG}"
  PYTHONUNBUFFERED=1 python src/evaluate/evaluate_inr_saved_midis.py \
    --prediction-manifest "${OUT_ROOT}/${protocol}/prediction_manifest.json" \
    --score-source-list "${SCORE_LIST}" \
    --max-gt-per-score "${MAX_GT_PER_SCORE}" \
    --num-workers 8 \
    --output-json "${OUT_ROOT}/${protocol}_metrics.json" 2>&1 | tee -a "${LOG}"
}

echo "NAME=${NAME}" | tee "${LOG}"
echo "GPU=${GPU}" | tee -a "${LOG}"
echo "CONFIG=${CONFIG}" | tee -a "${LOG}"
echo "CHECKPOINT=${CHECKPOINT}" | tee -a "${LOG}"
echo "SCORE_LIST=${SCORE_LIST}" | tee -a "${LOG}"
echo "MAX_GT_PER_SCORE=${MAX_GT_PER_SCORE}" | tee -a "${LOG}"

run_infer deterministic
run_infer sampling
run_eval deterministic
run_eval sampling

echo "[$(date '+%F %T')] ${NAME} done" | tee -a "${LOG}"
