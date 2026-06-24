#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

: "${CONFIG:?CONFIG is required}"
: "${BASE_CHECKPOINT:?BASE_CHECKPOINT is required}"
: "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is required}"
: "${RUN_TAG:?RUN_TAG is required}"

INFER_NUM_WORKERS="${INFER_NUM_WORKERS:-8}"
METRIC_NUM_WORKERS="${METRIC_NUM_WORKERS:-8}"
DET_NUM_SAMPLES="${DET_NUM_SAMPLES:-1}"
SAMPLING_NUM_SAMPLES="${SAMPLING_NUM_SAMPLES:-1}"
INFER_BATCH_SIZE_WINDOWS="${INFER_BATCH_SIZE_WINDOWS:-8}"
MERGE_MODE="${MERGE_MODE:-continuation}"
CONTINUATION_DROP_RATIO="${CONTINUATION_DROP_RATIO:-0.0}"
ADAPT_LR="${ADAPT_LR:-0.00003}"
ADAPT_EPOCHS="${ADAPT_EPOCHS:-2}"

RUN_NAME="${RUN_TAG}_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="results/inr_pipeline/${RUN_NAME}"
BASE_NONASAP_DIR="${RUN_DIR}/base_nonasap_test"
ADAPT_DIR="${RUN_DIR}/asap_adapt"
ADAPT_ASAP_DIR="${RUN_DIR}/adapted_asap_test"
TRAIN_ROOT="${ADAPT_DIR}/training"
TF_LOG_ROOT="${ADAPT_DIR}/tf-logs"
TRAIN_LOG="${ADAPT_DIR}/train.log"
EVALUATE_LOG="${RUN_DIR}/evaluate.log"
BASE_CONFIG="${RUN_DIR}/base_config.json"
ADAPT_CONFIG="${ADAPT_DIR}/config.json"
mkdir -p "${BASE_NONASAP_DIR}" "${ADAPT_DIR}" "${ADAPT_ASAP_DIR}" "${TRAIN_ROOT}" "${TF_LOG_ROOT}"

python - "${CONFIG}" "${BASE_CONFIG}" <<'PY'
import json, sys
src, dst = sys.argv[1:3]
cfg = json.loads(open(src, encoding="utf-8").read())
open(dst, "w", encoding="utf-8").write(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
PY

python - "${CONFIG}" "${ADAPT_CONFIG}" "${TRAIN_ROOT}" "${TF_LOG_ROOT}" "${BASE_CHECKPOINT}" "${ADAPT_LR}" "${ADAPT_EPOCHS}" <<'PY'
import json, sys
src, dst, train_root, tf_log_root, checkpoint, lr, epochs = sys.argv[1:8]
cfg = json.loads(open(src, encoding="utf-8").read())
cfg["resume_path"] = checkpoint
cfg["resume_trainer_state"] = False
cfg["output_dir"] = train_root
cfg["logging_dir"] = tf_log_root
cfg["learning_rate"] = float(lr)
cfg["num_train_epochs"] = float(epochs)
cfg["max_steps"] = -1
cfg["train_performance_dataset"] = "ASAP"
cfg["eval_split"] = "train"
cfg["eval_performance_dataset"] = "ASAP"
cfg["eval_include_all_performance_dataset"] = None
cfg["max_eval_non_asap_performances_per_work"] = None
cfg["save_steps"] = 500
cfg["eval_steps"] = 500
cfg["logging_steps"] = 20
cfg["save_total_limit"] = 2
cfg["load_best_model_at_end"] = True
cfg["metric_for_best_model"] = "eval_loss"
cfg["greater_is_better"] = False
open(dst, "w", encoding="utf-8").write(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
PY

{
  echo "START $(date '+%F %T')"
  echo "CONFIG ${CONFIG}"
  echo "BASE_CHECKPOINT ${BASE_CHECKPOINT}"
  echo "CUDA_VISIBLE_DEVICES ${CUDA_VISIBLE_DEVICES}"
  echo "RUN_DIR ${RUN_DIR}"
  echo "ADAPT_LR ${ADAPT_LR}"
  echo "ADAPT_EPOCHS ${ADAPT_EPOCHS}"
} | tee -a "${EVALUATE_LOG}"

run_infer_pair() {
  local checkpoint="$1" config="$2" out_root="$3" dataset_mode="$4"
  local det_dir="${out_root}/deterministic"
  local sampling_dir="${out_root}/sampling"
  mkdir -p "${det_dir}" "${sampling_dir}"
  local common=(
    --config "${config}"
    --checkpoint "${checkpoint}"
    --split test
    --num-workers "${INFER_NUM_WORKERS}"
    --batch-size-windows "${INFER_BATCH_SIZE_WINDOWS}"
    --merge-mode "${MERGE_MODE}"
    --continuation-drop-ratio "${CONTINUATION_DROP_RATIO}"
    --device cuda
  )
  if [[ "${dataset_mode}" == "asap" ]]; then
    common+=(--performance-dataset ASAP)
  elif [[ "${dataset_mode}" == "nonasap" ]]; then
    common+=(--exclude-performance-dataset ASAP)
  else
    echo "Unknown dataset mode: ${dataset_mode}" >&2
    exit 1
  fi

  python src/inference/infer_inr_testset.py "${common[@]}" \
    --protocol deterministic --num-samples "${DET_NUM_SAMPLES}" --output-dir "${det_dir}" \
    2>&1 | tee -a "${EVALUATE_LOG}"
  python src/inference/infer_inr_testset.py "${common[@]}" \
    --protocol sampling --num-samples "${SAMPLING_NUM_SAMPLES}" --output-dir "${sampling_dir}" \
    2>&1 | tee -a "${EVALUATE_LOG}"
  python src/evaluate/summarize_inr_asap_pipeline.py \
    --deterministic-manifest "${det_dir}/prediction_manifest.json" \
    --sampling-manifest "${sampling_dir}/prediction_manifest.json" \
    --output-json "${out_root}/summary.json" \
    --output-plot "${out_root}/label_distribution.png" \
    --config "${config}" \
    --checkpoint "${checkpoint}" \
    --train-output-dir "${TRAIN_ROOT}" \
    --pipeline-log "${TRAIN_LOG}" \
    --evaluate-log "${EVALUATE_LOG}" \
    --num-workers "${METRIC_NUM_WORKERS}" \
    2>&1 | tee -a "${EVALUATE_LOG}"
}

echo "[$(date '+%F %T')] base nonASAP test eval" | tee -a "${EVALUATE_LOG}"
run_infer_pair "${BASE_CHECKPOINT}" "${BASE_CONFIG}" "${BASE_NONASAP_DIR}" nonasap

echo "[$(date '+%F %T')] ASAP adaptation train" | tee -a "${EVALUATE_LOG}"
PYTHONUNBUFFERED=1 python src/train/train_inr.py --config "${ADAPT_CONFIG}" 2>&1 | tee -a "${TRAIN_LOG}"

TRAIN_OUTPUT_DIR="$(
  find "${TRAIN_ROOT}" -maxdepth 1 -mindepth 1 -type d -name 'inr_*' \
    -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-
)"
[[ -n "${TRAIN_OUTPUT_DIR}" ]] || { echo "Could not locate adapted training output" >&2; exit 1; }
ADAPTED_CHECKPOINT="${TRAIN_OUTPUT_DIR}"
[[ -d "${TRAIN_OUTPUT_DIR}/checkpoint-best" ]] && ADAPTED_CHECKPOINT="${TRAIN_OUTPUT_DIR}/checkpoint-best"
echo "ADAPTED_CHECKPOINT ${ADAPTED_CHECKPOINT}" | tee -a "${EVALUATE_LOG}"

echo "[$(date '+%F %T')] adapted ASAP test eval" | tee -a "${EVALUATE_LOG}"
run_infer_pair "${ADAPTED_CHECKPOINT}" "${ADAPT_CONFIG}" "${ADAPT_ASAP_DIR}" asap

echo "END $(date '+%F %T')" | tee -a "${EVALUATE_LOG}"
