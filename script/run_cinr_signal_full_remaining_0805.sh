#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUN_ROOT="${RUN_ROOT:-results/cinr_baseline_ckpt_signal_probe_0805_20260805_142256}"
RUN_LABEL="${RUN_LABEL:-cinr_baseline}"
GPU_IDS_CSV="${GPU_IDS:-0,1,2}"
IFS=',' read -ra GPU_IDS_ARR <<< "${GPU_IDS_CSV}"
if [[ "${#GPU_IDS_ARR[@]}" -ne 3 ]]; then
  echo "GPU_IDS must contain exactly 3 ids, got: ${GPU_IDS_CSV}" >&2
  exit 2
fi

RUNTIME_CONFIG="${RUNTIME_CONFIG:-${RUN_ROOT}/runtime_configs/${RUN_LABEL}.json}"
TEST_SCORE_LIST="${TEST_SCORE_LIST:-data/asap_test_score_sources.txt}"
EPOCHS="${EPOCHS:-8,10,12,14,16,18,20,22,24}"
MODE="${MODE:-full_test_s1}"
INFER_NUM_WORKERS="${INFER_NUM_WORKERS:-8}"
INFER_BATCH_SIZE_WINDOWS="${INFER_BATCH_SIZE_WINDOWS:-8}"
SAMPLING_NUM_SAMPLES="${SAMPLING_NUM_SAMPLES:-1}"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "${LOG_DIR}"

train_output_dir() {
  local out
  out="$(find "${RUN_ROOT}/${RUN_LABEL}/training" -mindepth 1 -maxdepth 1 -type d | sort | head -1)"
  if [[ -z "${out}" ]]; then
    echo "Could not find train output under ${RUN_ROOT}/${RUN_LABEL}/training" >&2
    exit 1
  fi
  echo "${out}"
}

run_full_one() {
  local epoch="$1"
  local gpu="$2"
  local train_dir="$3"
  local out_dir="${RUN_ROOT}/${RUN_LABEL}_ep${epoch}/${MODE}"
  local checkpoint="${train_dir}/epoch-${epoch}"
  local log_file="${LOG_DIR}/${MODE}_gpu${gpu}.log"
  if [[ -f "${out_dir}/eval_pn_pp_metrics.json" ]]; then
    echo "[$(date '+%F %T')] skip existing ${MODE} ep${epoch}" | tee -a "${log_file}"
    return
  fi
  if [[ ! -d "${checkpoint}" ]]; then
    echo "Missing checkpoint alias ${checkpoint}" >&2
    exit 1
  fi
  mkdir -p "${out_dir}"
  echo "[$(date '+%F %T')] infer ${MODE} ep${epoch} on gpu ${gpu}" | tee -a "${log_file}"
  local start end
  start="$(date +%s)"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 \
    python src/inference/infer_inr_testset.py \
      --config "${RUNTIME_CONFIG}" \
      --checkpoint "${checkpoint}" \
      --split test \
      --score-source-list "${TEST_SCORE_LIST}" \
      --performance-dataset ASAP \
      --device cuda \
      --protocol sampling \
      --sampling-strategy sample \
      --num-samples "${SAMPLING_NUM_SAMPLES}" \
      --num-workers "${INFER_NUM_WORKERS}" \
      --batch-size-windows "${INFER_BATCH_SIZE_WINDOWS}" \
      --output-dir "${out_dir}" 2>&1 | tee -a "${log_file}"
  end="$(date +%s)"
  echo "$((end - start))" > "${out_dir}/infer.seconds"

  echo "[$(date '+%F %T')] eval ${MODE} ep${epoch}" | tee -a "${log_file}"
  start="$(date +%s)"
  PYTHONUNBUFFERED=1 python src/evaluate/evaluate_inr_saved_midis.py \
    --prediction-manifest "${out_dir}/prediction_manifest.json" \
    --output-json "${out_dir}/eval_pn_pp_metrics.json" \
    --num-workers "${INFER_NUM_WORKERS}" \
    2>&1 | tee -a "${log_file}"
  end="$(date +%s)"
  echo "$((end - start))" > "${out_dir}/eval.seconds"
}

run_queue() {
  local gpu="$1"
  local train_dir="$2"
  shift 2
  for epoch in "$@"; do
    run_full_one "${epoch}" "${gpu}" "${train_dir}"
  done
}

TRAIN_DIR="$(train_output_dir)"
IFS=',' read -ra EPOCHS_ARR <<< "${EPOCHS}"
pids=()
for idx in 0 1 2; do
  epochs_for_gpu=()
  for epoch_idx in "${!EPOCHS_ARR[@]}"; do
    if [[ $((epoch_idx % 3)) -eq "${idx}" ]]; then
      epochs_for_gpu+=("${EPOCHS_ARR[$epoch_idx]}")
    fi
  done
  run_queue "${GPU_IDS_ARR[$idx]}" "${TRAIN_DIR}" "${epochs_for_gpu[@]}" &
  pids+=("$!")
done

failures=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failures=$((failures + 1))
  fi
done
if [[ "${failures}" -ne 0 ]]; then
  echo "${failures} full-test job(s) failed" >&2
  exit 1
fi

PYTHONUNBUFFERED=1 python src/evaluate/summarize_velocity_drift.py \
  --run-root "${RUN_ROOT}" \
  --run-label "${RUN_LABEL}" \
  --mode "${MODE}" \
  2>&1 | tee -a "${LOG_DIR}/velocity_drift_${MODE}.log"

echo "${MODE} velocity-drift experiment finished: ${RUN_ROOT}" | tee -a "${LOG_DIR}/velocity_drift_${MODE}.log"
