#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required, e.g. results/asd_10x2_nomus_3gpu_20260804_132212}"
GPU_IDS_CSV="${GPU_IDS:-0,1,2}"
IFS=',' read -ra GPU_IDS_ARR <<< "${GPU_IDS_CSV}"
if [[ "${#GPU_IDS_ARR[@]}" -ne 3 ]]; then
  echo "GPU_IDS must contain exactly 3 ids, got: ${GPU_IDS_CSV}" >&2
  exit 2
fi
INFER_NUM_WORKERS="${INFER_NUM_WORKERS:-8}"
INFER_BATCH_SIZE_WINDOWS="${INFER_BATCH_SIZE_WINDOWS:-8}"
SAMPLING_NUM_SAMPLES="${SAMPLING_NUM_SAMPLES:-2}"
MAX_GT_PER_SCORE="${MAX_GT_PER_SCORE:-}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

lane_checkpoint() {
  local lane="$1"
  local checkpoint
  checkpoint="$(find "${RUN_ROOT}/${lane}/training" -type d -path '*/checkpoint-best' | sort | head -1)"
  if [[ -z "${checkpoint}" ]]; then
    echo "Could not find checkpoint-best under ${RUN_ROOT}/${lane}/training" >&2
    exit 1
  fi
  echo "${checkpoint}"
}

lane_config() {
  local lane="$1"
  local family="$2"
  case "${lane}" in
    baseline|full|share) echo "${RUN_ROOT}/runtime_configs/${lane}_${family}.json" ;;
    *) echo "Unknown lane ${lane}" >&2; exit 1 ;;
  esac
}

run_infer_eval_stat() {
  local lane="$1"
  local family="$2"
  local gpu="$3"
  local config
  local checkpoint
  local out_dir
  config="$(lane_config "${lane}" "${family}")"
  checkpoint="$(lane_checkpoint "${lane}_${family}")"
  out_dir="${RUN_ROOT}/${lane}_${family}/postprocess"
  mkdir -p "${out_dir}"

  local pred_manifest="${out_dir}/prediction_manifest.json"
  local eval_json="${out_dir}/eval_pn_pp_metrics.json"
  local midi_metrics_json="${out_dir}/midi_pair_metrics.json"

  if [[ "${SKIP_EXISTING}" == "1" && -s "${pred_manifest}" && -s "${eval_json}" && -s "${midi_metrics_json}" ]]; then
    echo "[$(date '+%F %T')] reuse ${lane}_${family}" | tee -a "${RUN_ROOT}/postprocess.log"
    return 0
  fi

  echo "[$(date '+%F %T')] infer ${lane}_${family}: ${checkpoint}" | tee -a "${RUN_ROOT}/postprocess.log"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 \
    python src/inference/infer_inr_testset.py \
      --config "${config}" \
      --checkpoint "${checkpoint}" \
      --split test \
      --performance-dataset ASAP \
      --device cuda \
      --protocol sampling \
      --sampling-strategy sample \
      --num-samples "${SAMPLING_NUM_SAMPLES}" \
      --num-workers "${INFER_NUM_WORKERS}" \
      --batch-size-windows "${INFER_BATCH_SIZE_WINDOWS}" \
      --output-dir "${out_dir}" 2>&1 | tee -a "${RUN_ROOT}/postprocess.log"

  echo "[$(date '+%F %T')] eval PN/PP ${lane}_${family}" | tee -a "${RUN_ROOT}/postprocess.log"
  local max_gt_args=()
  if [[ -n "${MAX_GT_PER_SCORE}" ]]; then
    max_gt_args=(--max-gt-per-score "${MAX_GT_PER_SCORE}")
  fi
  PYTHONUNBUFFERED=1 python src/evaluate/evaluate_inr_saved_midis.py \
    --prediction-manifest "${pred_manifest}" \
    --output-json "${eval_json}" \
    --num-workers "${INFER_NUM_WORKERS}" \
    "${max_gt_args[@]}" \
    2>&1 | tee -a "${RUN_ROOT}/postprocess.log"

  echo "[$(date '+%F %T')] stat MIDI pairs ${lane}_${family}" | tee -a "${RUN_ROOT}/postprocess.log"
  PYTHONUNBUFFERED=1 python src/evaluate/compute_saved_midi_mae_wass.py \
    --evaluate-list "${out_dir}/evaluate_list.json" \
    --output-json "${midi_metrics_json}" \
    --num-workers "${INFER_NUM_WORKERS}" \
    2>&1 | tee -a "${RUN_ROOT}/postprocess.log"
}

echo "RUN_ROOT ${RUN_ROOT}" | tee -a "${RUN_ROOT}/postprocess.log"

run_lane() {
  local lane="$1"
  local gpu="$2"
  echo "[$(date '+%F %T')] lane ${lane}: GPU ${gpu}, CINR then DINR" | tee -a "${RUN_ROOT}/postprocess.log"
  run_infer_eval_stat "${lane}" cinr "${gpu}"
  run_infer_eval_stat "${lane}" dinr "${gpu}"
}

LANES=(baseline full share)
pids=()
for idx in 0 1 2; do
  run_lane "${LANES[$idx]}" "${GPU_IDS_ARR[$idx]}" &
  pids+=("$!")
done

failures=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failures=$((failures + 1))
  fi
done
if [[ "${failures}" -ne 0 ]]; then
  echo "${failures} postprocess lane(s) failed; see ${RUN_ROOT}/postprocess.log" >&2
  exit 1
fi

echo "All postprocess jobs finished." | tee -a "${RUN_ROOT}/postprocess.log"
