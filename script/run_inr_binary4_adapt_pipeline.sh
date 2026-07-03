#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

: "${CONFIG:?CONFIG is required}"
: "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is required}"

INFER_NUM_WORKERS="${INFER_NUM_WORKERS:-16}"
METRIC_NUM_WORKERS="${METRIC_NUM_WORKERS:-8}"
DET_NUM_SAMPLES="${DET_NUM_SAMPLES:-1}"
SAMPLING_NUM_SAMPLES="${SAMPLING_NUM_SAMPLES:-1}"
DET_STRATEGY="${DET_STRATEGY:-greedy}"
INFER_BATCH_SIZE_WINDOWS="${INFER_BATCH_SIZE_WINDOWS:-8}"
MERGE_MODE="${MERGE_MODE:-continuation}"
CONTINUATION_DROP_RATIO="${CONTINUATION_DROP_RATIO:-0.0}"
ADAPT_EPOCHS_LIST="${ADAPT_EPOCHS_LIST:-2 4}"
IFS=',' read -ra GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
GPU_COUNT="${#GPU_LIST[@]}"
if [[ "${GPU_COUNT}" -ne 1 && "${GPU_COUNT}" -ne 2 ]]; then
  if [[ "${GPU_COUNT}" -ne 4 ]]; then
    echo "CUDA_VISIBLE_DEVICES must contain 1, 2, or 4 GPU ids, got: ${CUDA_VISIBLE_DEVICES}" >&2
    exit 1
  fi
fi
BATCH_SIZE_PER_DEVICE=32
GLOBAL_BATCH_SIZE=64
GRADIENT_ACCUMULATION_STEPS=$(( GLOBAL_BATCH_SIZE / (BATCH_SIZE_PER_DEVICE * GPU_COUNT) ))
if [[ "${GRADIENT_ACCUMULATION_STEPS}" -lt 1 || $(( BATCH_SIZE_PER_DEVICE * GPU_COUNT * GRADIENT_ACCUMULATION_STEPS )) -ne "${GLOBAL_BATCH_SIZE}" ]]; then
  echo "Invalid batch setup for GPU_COUNT=${GPU_COUNT}" >&2
  exit 1
fi
TRAIN_GPU_COUNT="${GPU_COUNT}"
MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
DET_GPU="${GPU_LIST[0]}"
if [[ "${GPU_COUNT}" -ge 4 ]]; then
  SAMPLING_GPU="${GPU_LIST[2]}"
elif [[ "${GPU_COUNT}" -ge 2 ]]; then
  SAMPLING_GPU="${GPU_LIST[1]}"
else
  SAMPLING_GPU="${GPU_LIST[0]}"
fi

DEFAULT_RUN_NAME="${CONFIG##*/}"
DEFAULT_RUN_NAME="${DEFAULT_RUN_NAME%.json}_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR_OVERRIDE:-results/inr0624_binary4_prior_ablation/${DEFAULT_RUN_NAME}}"
TRAIN_LOG="${RUN_DIR}/train.log"
EVALUATE_LOG="${RUN_DIR}/evaluate.log"
RUN_CONFIG="${RUN_DIR}/config.json"
TRAIN_ROOT="${RUN_DIR}/training"
TF_LOG_ROOT="${RUN_DIR}/tf-logs"
TMP_DIR="${RUN_DIR}/_tmp"

mkdir -p "${RUN_DIR}" "${TRAIN_ROOT}" "${TF_LOG_ROOT}" "${TMP_DIR}"

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

latest_numeric_checkpoint() {
  local root="$1"
  find "${root}" -path '*/checkpoint-*' -type d 2>/dev/null \
    | awk -F'checkpoint-' '/checkpoint-[0-9]+$/ {print $2 " " $0}' \
    | sort -n \
    | tail -n 1 \
    | cut -d' ' -f2-
}

run_train() {
  local config="$1" stage="$2"
  if [[ "${TRAIN_GPU_COUNT}" -gt 1 ]]; then
    echo "[$(date '+%F %T')] ${stage}: DDP train start, GPUs=${CUDA_VISIBLE_DEVICES}, nproc=${TRAIN_GPU_COUNT}" | tee -a "${EVALUATE_LOG}"
    MASTER_PORT="${MASTER_PORT}" PYTHONUNBUFFERED=1 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo \
      TORCH_NCCL_BLOCKING_WAIT=1 NCCL_DEBUG=WARN \
      torchrun --nnodes=1 --nproc_per_node="${TRAIN_GPU_COUNT}" \
        --master_addr=127.0.0.1 --master_port="${MASTER_PORT}" \
        src/train/train_inr.py --config "${config}" 2>&1 | tee -a "${TRAIN_LOG}"
  else
    echo "[$(date '+%F %T')] ${stage}: single GPU train start, GPU=${CUDA_VISIBLE_DEVICES}" | tee -a "${EVALUATE_LOG}"
    PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      python src/train/train_inr.py --config "${config}" 2>&1 | tee -a "${TRAIN_LOG}"
  fi
  echo "[$(date '+%F %T')] ${stage}: train finished" | tee -a "${EVALUATE_LOG}"
}

run_infer() {
  local config="$1" checkpoint="$2" protocol="$3" num_samples="$4" out_dir="$5" infer_gpu="$6"
  echo "[$(date '+%F %T')] infer ${protocol}: checkpoint=${checkpoint}, gpu=${infer_gpu}, workers=${INFER_NUM_WORKERS}" | tee -a "${EVALUATE_LOG}"
  CUDA_VISIBLE_DEVICES="${infer_gpu}" PYTHONUNBUFFERED=1 \
    python src/inference/infer_inr_testset.py \
      --config "${config}" \
      --checkpoint "${checkpoint}" \
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

summarize_pair() {
  local config="$1" checkpoint="$2" train_output_dir="$3" det_dir="$4" sampling_dir="$5" summary_json="$6" plot_path="$7"
  echo "[$(date '+%F %T')] summarize ${summary_json}" | tee -a "${EVALUATE_LOG}"
  PYTHONUNBUFFERED=1 python src/evaluate/summarize_inr_asap_pipeline.py \
    --deterministic-manifest "${det_dir}/prediction_manifest.json" \
    --sampling-manifest "${sampling_dir}/prediction_manifest.json" \
    --output-json "${summary_json}" \
    --output-plot "${plot_path}" \
    --config "${config}" \
    --checkpoint "${checkpoint}" \
    --train-output-dir "${train_output_dir}" \
    --pipeline-log "${TRAIN_LOG}" \
    --evaluate-log "${EVALUATE_LOG}" \
    --num-workers "${METRIC_NUM_WORKERS}" \
    2>&1 | tee -a "${EVALUATE_LOG}"
}

cp "${CONFIG}" "${RUN_CONFIG}"
BASE_RESUME_CHECKPOINT="$(latest_numeric_checkpoint "${TRAIN_ROOT}")"

python - "${RUN_CONFIG}" "${TRAIN_ROOT}" "${TF_LOG_ROOT}" "${BATCH_SIZE_PER_DEVICE}" "${GRADIENT_ACCUMULATION_STEPS}" "${GLOBAL_BATCH_SIZE}" "${BASE_RESUME_CHECKPOINT}" <<'PY'
import json, sys
path, train_root, tf_log_root, per_device_bs, grad_accum, global_bs, resume_checkpoint = sys.argv[1:8]
cfg = json.loads(open(path, encoding="utf-8").read())
cfg["output_dir"] = train_root
cfg["logging_dir"] = tf_log_root
if resume_checkpoint:
    cfg["resume_path"] = resume_checkpoint
    cfg["resume_trainer_state"] = True
    cfg["reset_output_heads_on_resume"] = False
    cfg["ignore_mismatched_resume_shapes"] = False
else:
    cfg["resume_trainer_state"] = False
cfg["per_device_train_batch_size"] = int(per_device_bs)
cfg["per_device_eval_batch_size"] = int(per_device_bs)
cfg["gradient_accumulation_steps"] = int(grad_accum)
cfg["global_batch_size"] = int(global_bs)
cfg["num_train_epochs"] = min(float(cfg.get("num_train_epochs", 1)), 8.0)
cfg["use_prepared_sidecar"] = True
cfg["precompute_dataset_items"] = False
cfg["precompute_eval_dataset_items"] = False
cfg.pop("prepared_sidecar_tag", None)
cfg["fixed_window_split_scheme"] = cfg.get("fixed_window_split_scheme") or "train_valid_asap3_nonasap1_v1"
cfg["fixed_window_base_split"] = cfg.get("fixed_window_base_split") or "train"
cfg["fixed_window_train_split_name"] = cfg.get("fixed_window_train_split_name") or "train"
cfg["fixed_window_eval_split_name"] = cfg.get("fixed_window_eval_split_name") or "valid"
cfg["fixed_window_split_summary_path"] = cfg.get("fixed_window_split_summary_path") or "data/train_valid_asap3_nonasap1_v1_summary.json"
cfg.pop("train_performance_dataset", None)
cfg["eval_every_steps"] = 1000
cfg.pop("eval_every_epochs", None)
cfg["save_every_steps"] = 1000
cfg["save_total_limit"] = 1
cfg["early_stopping_patience"] = int(cfg.get("early_stopping_patience", 5))
cfg["early_stopping_threshold"] = float(cfg.get("early_stopping_threshold", 0.001))
cfg["max_train_epochs"] = 8.0
cfg["num_train_epochs"] = 8.0
cfg["eval_compute_wass"] = bool(cfg.get("eval_compute_wass", False))
cfg.setdefault("eval_dataloader_num_workers", cfg.get("dataloader_num_workers", 0))
cfg.setdefault("eval_dataloader_persistent_workers", bool(int(cfg.get("eval_dataloader_num_workers") or 0) > 0))
cfg.setdefault("eval_dataloader_prefetch_factor", cfg.get("dataloader_prefetch_factor", 2))
cfg.setdefault("loss_component_interval", cfg.get("logging_steps", 20))
open(path, "w", encoding="utf-8").write(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
PY

{
  echo "START $(date '+%F %T')"
  echo "CONFIG ${CONFIG}"
  echo "CUDA_VISIBLE_DEVICES ${CUDA_VISIBLE_DEVICES}"
  echo "GPU_COUNT ${GPU_COUNT}"
  echo "PER_DEVICE_TRAIN_BATCH_SIZE ${BATCH_SIZE_PER_DEVICE}"
  echo "GRADIENT_ACCUMULATION_STEPS ${GRADIENT_ACCUMULATION_STEPS}"
  echo "GLOBAL_BATCH_SIZE ${GLOBAL_BATCH_SIZE}"
  echo "RUN_DIR ${RUN_DIR}"
  echo "BASE_RESUME_CHECKPOINT ${BASE_RESUME_CHECKPOINT:-none}"
  echo "ADAPT_EPOCHS_LIST ${ADAPT_EPOCHS_LIST}"
} | tee -a "${EVALUATE_LOG}"

BASE_MARKER="${TMP_DIR}/base_start.marker"
touch "${BASE_MARKER}"
run_train "${RUN_CONFIG}" "base"
BASE_OUTPUT_DIR="$(latest_train_dir "${TRAIN_ROOT}" "${BASE_MARKER}")"
[[ -n "${BASE_OUTPUT_DIR}" ]] || { echo "Missing base output under ${TRAIN_ROOT}" >&2; exit 1; }
BASE_CHECKPOINT="$(best_checkpoint "${BASE_OUTPUT_DIR}")"
{
  echo "BASE_OUTPUT_DIR ${BASE_OUTPUT_DIR}"
  echo "BASE_CHECKPOINT ${BASE_CHECKPOINT}"
} | tee -a "${EVALUATE_LOG}"

for epochs in ${ADAPT_EPOCHS_LIST}; do
  ADAPT_DIR="${RUN_DIR}/adapt_${epochs}ep"
  ADAPT_TRAIN_ROOT="${ADAPT_DIR}/training"
  ADAPT_TF_LOG_ROOT="${ADAPT_DIR}/tf-logs"
  ADAPT_CONFIG="${ADAPT_DIR}/config.json"
  DET_DIR="${ADAPT_DIR}/deterministic"
  SAMPLING_DIR="${ADAPT_DIR}/sampling"
  mkdir -p "${ADAPT_TRAIN_ROOT}" "${ADAPT_TF_LOG_ROOT}" "${DET_DIR}" "${SAMPLING_DIR}"

  python - "${RUN_CONFIG}" "${ADAPT_CONFIG}" "${ADAPT_TRAIN_ROOT}" "${ADAPT_TF_LOG_ROOT}" "${BASE_CHECKPOINT}" "${epochs}" "${BATCH_SIZE_PER_DEVICE}" "${GRADIENT_ACCUMULATION_STEPS}" "${GLOBAL_BATCH_SIZE}" <<'PY'
import json, sys
src, dst, train_root, tf_log_root, checkpoint, epochs, per_device_bs, grad_accum, global_bs = sys.argv[1:10]
cfg = json.loads(open(src, encoding="utf-8").read())
cfg["resume_path"] = checkpoint
cfg["resume_trainer_state"] = False
cfg["reset_output_heads_on_resume"] = False
cfg["ignore_mismatched_resume_shapes"] = False
cfg["freeze_non_output_heads"] = False
cfg.pop("trainable_parameter_regex", None)
cfg["output_dir"] = train_root
cfg["logging_dir"] = tf_log_root
cfg["learning_rate"] = float(cfg.get("adapt_learning_rate", cfg.get("learning_rate", 3e-4)))
cfg["num_train_epochs"] = 8.0
cfg["per_device_train_batch_size"] = int(per_device_bs)
cfg["per_device_eval_batch_size"] = int(per_device_bs)
cfg["gradient_accumulation_steps"] = int(grad_accum)
cfg["global_batch_size"] = int(global_bs)
cfg["max_steps"] = -1
cfg["train_performance_dataset"] = "ASAP"
cfg["fixed_window_split_scheme"] = cfg.get("fixed_window_split_scheme") or "train_valid_asap3_nonasap1_v1"
cfg["fixed_window_base_split"] = cfg.get("fixed_window_base_split") or "train"
cfg["fixed_window_train_split_name"] = cfg.get("fixed_window_train_split_name") or "train"
cfg["fixed_window_eval_split_name"] = cfg.get("fixed_window_eval_split_name") or "valid"
cfg["fixed_window_split_summary_path"] = cfg.get("fixed_window_split_summary_path") or "data/train_valid_asap3_nonasap1_v1_summary.json"
cfg["eval_split"] = "valid"
cfg["eval_performance_dataset"] = "ASAP"
cfg["eval_include_all_performance_dataset"] = None
cfg["max_eval_non_asap_performances_per_work"] = None
cfg["use_prepared_sidecar"] = True
cfg["prepared_sidecar_tag"] = cfg.get("prepared_sidecar_tag") or "ASAP"
cfg["precompute_dataset_items"] = False
cfg["precompute_eval_dataset_items"] = False
cfg.pop("eval_every_steps", None)
cfg["eval_every_epochs"] = 0.5
cfg.pop("eval_steps", None)
cfg.pop("save_steps", None)
cfg["save_total_limit"] = 1
cfg["load_best_model_at_end"] = True
cfg["metric_for_best_model"] = "eval_loss"
cfg["greater_is_better"] = False
cfg["early_stopping_patience"] = int(cfg.get("early_stopping_patience", 5))
cfg["early_stopping_threshold"] = float(cfg.get("early_stopping_threshold", 0.001))
cfg["max_train_epochs"] = 8.0
cfg["eval_compute_wass"] = bool(cfg.get("eval_compute_wass", False))
open(dst, "w", encoding="utf-8").write(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
PY

  ADAPT_MARKER="${TMP_DIR}/adapt_${epochs}ep_start.marker"
  touch "${ADAPT_MARKER}"
  run_train "${ADAPT_CONFIG}" "adapt_${epochs}ep"
  ADAPT_OUTPUT_DIR="$(latest_train_dir "${ADAPT_TRAIN_ROOT}" "${ADAPT_MARKER}")"
  [[ -n "${ADAPT_OUTPUT_DIR}" ]] || { echo "Missing adapt output under ${ADAPT_TRAIN_ROOT}" >&2; exit 1; }
  ADAPT_CHECKPOINT="$(best_checkpoint "${ADAPT_OUTPUT_DIR}")"
  {
    echo "ADAPT_${epochs}EP_OUTPUT_DIR ${ADAPT_OUTPUT_DIR}"
    echo "ADAPT_${epochs}EP_CHECKPOINT ${ADAPT_CHECKPOINT}"
  } | tee -a "${EVALUATE_LOG}"

  if [[ "${GPU_COUNT}" -gt 1 ]]; then
    run_infer "${ADAPT_CONFIG}" "${ADAPT_CHECKPOINT}" deterministic "${DET_NUM_SAMPLES}" "${DET_DIR}" "${DET_GPU}" &
    DET_PID=$!
    run_infer "${ADAPT_CONFIG}" "${ADAPT_CHECKPOINT}" sampling "${SAMPLING_NUM_SAMPLES}" "${SAMPLING_DIR}" "${SAMPLING_GPU}" &
    SAMPLING_PID=$!
    wait "${DET_PID}" || { echo "deterministic inference failed" >&2; exit 1; }
    wait "${SAMPLING_PID}" || { echo "sampling inference failed" >&2; exit 1; }
  else
    run_infer "${ADAPT_CONFIG}" "${ADAPT_CHECKPOINT}" deterministic "${DET_NUM_SAMPLES}" "${DET_DIR}" "${DET_GPU}"
    run_infer "${ADAPT_CONFIG}" "${ADAPT_CHECKPOINT}" sampling "${SAMPLING_NUM_SAMPLES}" "${SAMPLING_DIR}" "${SAMPLING_GPU}"
  fi
  summarize_pair \
    "${ADAPT_CONFIG}" \
    "${ADAPT_CHECKPOINT}" \
    "${ADAPT_OUTPUT_DIR}" \
    "${DET_DIR}" \
    "${SAMPLING_DIR}" \
    "${ADAPT_DIR}/summary.json" \
    "${ADAPT_DIR}/asap_label_distribution.png"
done

rm -rf "${TMP_DIR}"
echo "END $(date '+%F %T')" | tee -a "${EVALUATE_LOG}"
