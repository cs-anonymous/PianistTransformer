#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

: "${CONFIG:?CONFIG is required}"
: "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is required}"

INFER_NUM_WORKERS="${INFER_NUM_WORKERS:-8}"
METRIC_NUM_WORKERS="${METRIC_NUM_WORKERS:-8}"
DET_NUM_SAMPLES="${DET_NUM_SAMPLES:-1}"
SAMPLING_NUM_SAMPLES="${SAMPLING_NUM_SAMPLES:-1}"
DET_STRATEGY="${DET_STRATEGY:-mean}"
INFER_BATCH_SIZE_WINDOWS="${INFER_BATCH_SIZE_WINDOWS:-8}"
MERGE_MODE="${MERGE_MODE:-continuation}"
CONTINUATION_DROP_RATIO="${CONTINUATION_DROP_RATIO:-0.0}"

DEFAULT_RUN_NAME="${CONFIG##*/}"
DEFAULT_RUN_NAME="${DEFAULT_RUN_NAME%.json}_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR_OVERRIDE:-results/inr0624_head_capacity/${DEFAULT_RUN_NAME}}"
TRAIN_LOG="${RUN_DIR}/train.log"
EVALUATE_LOG="${RUN_DIR}/evaluate.log"
SUMMARY_JSON="${RUN_DIR}/summary.json"
PLOT_PATH="${RUN_DIR}/asap_label_distribution.png"
RUN_CONFIG="${RUN_DIR}/config.json"
HEAD_CONFIG="${RUN_DIR}/head_only/config.json"
FULL_CONFIG="${RUN_DIR}/full_ft/config.json"
HEAD_TRAIN_ROOT="${RUN_DIR}/head_only/training"
HEAD_TF_LOG_ROOT="${RUN_DIR}/head_only/tf-logs"
FULL_TRAIN_ROOT="${RUN_DIR}/full_ft/training"
FULL_TF_LOG_ROOT="${RUN_DIR}/full_ft/tf-logs"
DET_DIR="${RUN_DIR}/deterministic"
SAMPLING_DIR="${RUN_DIR}/sampling"
TARGET_DIAG_DIR="${RUN_DIR}/target_distribution_diagnostic"
TMP_DIR="${RUN_DIR}/_tmp"

mkdir -p \
  "${RUN_DIR}" "${TMP_DIR}" \
  "${HEAD_TRAIN_ROOT}" "${HEAD_TF_LOG_ROOT}" \
  "${FULL_TRAIN_ROOT}" "${FULL_TF_LOG_ROOT}" \
  "${DET_DIR}" "${SAMPLING_DIR}"

latest_train_dir() {
  local root="$1" marker="$2"
  find "${root}" -maxdepth 1 -mindepth 1 -type d -name 'inr_*' -newer "${marker}" \
    -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-
}

best_checkpoint() {
  local train_dir="$1"
  if [[ -d "${train_dir}/checkpoint-best" ]]; then
    echo "${train_dir}/checkpoint-best"
  else
    echo "${train_dir}"
  fi
}

run_train() {
  local config="$1" stage="$2"
  echo "[$(date '+%F %T')] ${stage}: train start on GPU=${CUDA_VISIBLE_DEVICES}" | tee -a "${EVALUATE_LOG}"
  PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python src/train/train_inr.py --config "${config}" 2>&1 | tee -a "${TRAIN_LOG}"
  echo "[$(date '+%F %T')] ${stage}: train finished" | tee -a "${EVALUATE_LOG}"
}

run_infer() {
  local protocol="$1" num_samples="$2" out_dir="$3"
  echo "[$(date '+%F %T')] infer ${protocol}: workers=${INFER_NUM_WORKERS}" | tee -a "${EVALUATE_LOG}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" PYTHONUNBUFFERED=1 \
    python src/inference/infer_inr_testset.py \
      --config "${FULL_CONFIG}" \
      --checkpoint "${FINAL_CHECKPOINT}" \
      --split test \
      --performance-dataset ASAP \
      --num-workers "${INFER_NUM_WORKERS}" \
      --batch-size-windows "${INFER_BATCH_SIZE_WINDOWS}" \
      --merge-mode "${MERGE_MODE}" \
      --continuation-drop-ratio "${CONTINUATION_DROP_RATIO}" \
      --device cuda \
      --protocol "${protocol}" \
      --num-samples "${num_samples}" \
      --output-dir "${out_dir}" \
      --deterministic-strategy "${DET_STRATEGY}" \
      2>&1 | tee -a "${EVALUATE_LOG}"
  echo "[$(date '+%F %T')] infer ${protocol}: finished" | tee -a "${EVALUATE_LOG}"
}

cp "${CONFIG}" "${RUN_CONFIG}"

{
  echo "START $(date '+%F %T')"
  echo "CONFIG ${CONFIG}"
  echo "CUDA_VISIBLE_DEVICES ${CUDA_VISIBLE_DEVICES}"
  echo "RUN_DIR ${RUN_DIR}"
} | tee -a "${EVALUATE_LOG}"

python - "${RUN_CONFIG}" "${HEAD_CONFIG}" "${HEAD_TRAIN_ROOT}" "${HEAD_TF_LOG_ROOT}" <<'PY'
import json, sys
src, dst, train_root, tf_log_root = sys.argv[1:5]
cfg = json.loads(open(src, encoding="utf-8").read())
cfg["output_dir"] = train_root
cfg["logging_dir"] = tf_log_root
cfg["resume_trainer_state"] = False
cfg["reset_output_heads_on_resume"] = True
cfg["ignore_mismatched_resume_shapes"] = True
cfg["freeze_non_output_heads"] = True
cfg["num_train_epochs"] = float(cfg.get("head_only_num_train_epochs", 3))
cfg["max_steps"] = -1
cfg["train_performance_dataset"] = "ASAP"
cfg["eval_split"] = "test"
cfg["eval_performance_dataset"] = "ASAP"
cfg["eval_include_all_performance_dataset"] = None
cfg["max_eval_non_asap_performances_per_work"] = None
cfg["precompute_dataset_items"] = bool(cfg.get("precompute_dataset_items", False))
cfg["precompute_eval_dataset_items"] = bool(cfg.get("precompute_eval_dataset_items", False))
cfg["use_prepared_sidecar"] = bool(cfg.get("use_prepared_sidecar", True))
cfg["save_steps"] = min(int(cfg.get("save_steps", 500)), 500)
cfg["eval_steps"] = min(int(cfg.get("eval_steps", 500)), 500)
cfg["save_total_limit"] = 2
cfg["load_best_model_at_end"] = True
cfg["metric_for_best_model"] = "eval_loss"
cfg["greater_is_better"] = False
cfg.setdefault("eval_dataloader_num_workers", cfg.get("dataloader_num_workers", 0))
cfg.setdefault("eval_dataloader_persistent_workers", bool(int(cfg.get("eval_dataloader_num_workers") or 0) > 0))
cfg.setdefault("eval_dataloader_prefetch_factor", cfg.get("dataloader_prefetch_factor", 2))
cfg.setdefault("loss_component_interval", cfg.get("logging_steps", 20))
open(dst, "w", encoding="utf-8").write(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
PY

HEAD_MARKER="${TMP_DIR}/head_start.marker"
touch "${HEAD_MARKER}"
run_train "${HEAD_CONFIG}" "head_only"
HEAD_OUTPUT_DIR="$(latest_train_dir "${HEAD_TRAIN_ROOT}" "${HEAD_MARKER}")"
[[ -n "${HEAD_OUTPUT_DIR}" ]] || { echo "Missing head-only output" >&2; exit 1; }
HEAD_CHECKPOINT="$(best_checkpoint "${HEAD_OUTPUT_DIR}")"
{
  echo "HEAD_OUTPUT_DIR ${HEAD_OUTPUT_DIR}"
  echo "HEAD_CHECKPOINT ${HEAD_CHECKPOINT}"
} | tee -a "${EVALUATE_LOG}"

python - "${HEAD_CONFIG}" "${FULL_CONFIG}" "${FULL_TRAIN_ROOT}" "${FULL_TF_LOG_ROOT}" "${HEAD_CHECKPOINT}" <<'PY'
import json, sys
src, dst, train_root, tf_log_root, checkpoint = sys.argv[1:6]
cfg = json.loads(open(src, encoding="utf-8").read())
cfg["resume_path"] = checkpoint
cfg["resume_trainer_state"] = False
cfg["reset_output_heads_on_resume"] = False
cfg["freeze_non_output_heads"] = False
cfg.pop("trainable_parameter_regex", None)
cfg["output_dir"] = train_root
cfg["logging_dir"] = tf_log_root
cfg["num_train_epochs"] = float(cfg.get("full_ft_num_train_epochs", 1))
cfg["max_steps"] = -1
cfg["train_performance_dataset"] = "ASAP"
cfg["eval_split"] = "test"
cfg["eval_performance_dataset"] = "ASAP"
cfg["precompute_dataset_items"] = bool(cfg.get("precompute_dataset_items", False))
cfg["precompute_eval_dataset_items"] = bool(cfg.get("precompute_eval_dataset_items", False))
cfg["save_steps"] = min(int(cfg.get("save_steps", 500)), 500)
cfg["eval_steps"] = min(int(cfg.get("eval_steps", 500)), 500)
cfg["save_total_limit"] = 2
cfg["load_best_model_at_end"] = True
cfg["metric_for_best_model"] = "eval_loss"
cfg["greater_is_better"] = False
open(dst, "w", encoding="utf-8").write(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
PY

FULL_MARKER="${TMP_DIR}/full_start.marker"
touch "${FULL_MARKER}"
run_train "${FULL_CONFIG}" "full_ft"
FULL_OUTPUT_DIR="$(latest_train_dir "${FULL_TRAIN_ROOT}" "${FULL_MARKER}")"
[[ -n "${FULL_OUTPUT_DIR}" ]] || { echo "Missing full-ft output" >&2; exit 1; }
FINAL_CHECKPOINT="$(best_checkpoint "${FULL_OUTPUT_DIR}")"
{
  echo "FULL_OUTPUT_DIR ${FULL_OUTPUT_DIR}"
  echo "FINAL_CHECKPOINT ${FINAL_CHECKPOINT}"
} | tee -a "${EVALUATE_LOG}"

run_infer deterministic "${DET_NUM_SAMPLES}" "${DET_DIR}"
run_infer sampling "${SAMPLING_NUM_SAMPLES}" "${SAMPLING_DIR}"

echo "[$(date '+%F %T')] summarize metrics" | tee -a "${EVALUATE_LOG}"
PYTHONUNBUFFERED=1 python src/evaluate/summarize_inr_asap_pipeline.py \
  --deterministic-manifest "${DET_DIR}/prediction_manifest.json" \
  --sampling-manifest "${SAMPLING_DIR}/prediction_manifest.json" \
  --output-json "${SUMMARY_JSON}" \
  --output-plot "${PLOT_PATH}" \
  --config "${FULL_CONFIG}" \
  --checkpoint "${FINAL_CHECKPOINT}" \
  --train-output-dir "${FULL_OUTPUT_DIR}" \
  --pipeline-log "${TRAIN_LOG}" \
  --evaluate-log "${EVALUATE_LOG}" \
  --num-workers "${METRIC_NUM_WORKERS}" \
  2>&1 | tee -a "${EVALUATE_LOG}"

echo "[$(date '+%F %T')] target distribution diagnostic" | tee -a "${EVALUATE_LOG}"
PYTHONUNBUFFERED=1 python src/evaluate/plot_target_distribution_diagnostic.py \
  --config "${FULL_CONFIG}" \
  --det-manifest "${DET_DIR}/prediction_manifest.json" \
  --sampling-manifest "${SAMPLING_DIR}/prediction_manifest.json" \
  --output-dir "${TARGET_DIAG_DIR}" \
  --num-workers "${METRIC_NUM_WORKERS}" \
  2>&1 | tee -a "${EVALUATE_LOG}"

rm -rf "${TMP_DIR}"
echo "END $(date '+%F %T')" | tee -a "${EVALUATE_LOG}"
