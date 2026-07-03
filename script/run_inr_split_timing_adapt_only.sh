#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

: "${CONFIG:?CONFIG is required}"
: "${BASE_CHECKPOINT:?BASE_CHECKPOINT is required}"
: "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is required}"

INFER_NUM_WORKERS="${INFER_NUM_WORKERS:-4}"
METRIC_NUM_WORKERS="${METRIC_NUM_WORKERS:-4}"
DET_NUM_SAMPLES="${DET_NUM_SAMPLES:-1}"
SAMPLING_NUM_SAMPLES="${SAMPLING_NUM_SAMPLES:-1}"
DET_STRATEGY="${DET_STRATEGY:-greedy}"
INFER_BATCH_SIZE_WINDOWS="${INFER_BATCH_SIZE_WINDOWS:-8}"
MERGE_MODE="${MERGE_MODE:-continuation}"
CONTINUATION_DROP_RATIO="${CONTINUATION_DROP_RATIO:-0.0}"
ADAPT_EPOCHS_LIST="${ADAPT_EPOCHS_LIST:-8 16}"
CHAIN_START_EPOCHS="${CHAIN_START_EPOCHS:-0}"

IFS=',' read -ra GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
GPU_COUNT="${#GPU_LIST[@]}"
if [[ "${GPU_COUNT}" -ne 2 ]]; then
  echo "This split timing adapt runner expects exactly 2 GPUs, got: ${CUDA_VISIBLE_DEVICES}" >&2
  exit 1
fi
DET_GPU="${GPU_LIST[0]}"
SAMPLING_GPU="${GPU_LIST[1]}"
BATCH_SIZE_PER_DEVICE=32
GLOBAL_BATCH_SIZE=64
GRADIENT_ACCUMULATION_STEPS=1
MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

DEFAULT_RUN_NAME="${CONFIG##*/}"
DEFAULT_RUN_NAME="${DEFAULT_RUN_NAME%.json}_split_timing_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR_OVERRIDE:-results/inr0624_split_timing_adapt/${DEFAULT_RUN_NAME}}"
TRAIN_LOG="${RUN_DIR}/train.log"
EVALUATE_LOG="${RUN_DIR}/evaluate.log"
RUN_CONFIG="${RUN_DIR}/config.json"
TMP_DIR="${RUN_DIR}/_tmp"

mkdir -p "${RUN_DIR}" "${TMP_DIR}"

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

run_train_ddp() {
  local config="$1" stage="$2"
  echo "[$(date '+%F %T')] ${stage}: DDP adapt start, GPUs=${CUDA_VISIBLE_DEVICES}, nproc=${GPU_COUNT}" | tee -a "${EVALUATE_LOG}"
  MASTER_PORT="${MASTER_PORT}" PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo \
    TORCH_NCCL_BLOCKING_WAIT=1 NCCL_DEBUG=WARN \
    torchrun --nnodes=1 --nproc_per_node="${GPU_COUNT}" \
      --master_addr=127.0.0.1 --master_port="${MASTER_PORT}" \
      src/train/train_inr.py --config "${config}" 2>&1 | tee -a "${TRAIN_LOG}"
  echo "[$(date '+%F %T')] ${stage}: adapt finished" | tee -a "${EVALUATE_LOG}"
}

run_infer() {
  local config="$1" checkpoint="$2" protocol="$3" num_samples="$4" out_dir="$5" gpu="$6"
  echo "[$(date '+%F %T')] infer ${protocol}: gpu=${gpu}, checkpoint=${checkpoint}" | tee -a "${EVALUATE_LOG}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 \
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

python - "${RUN_CONFIG}" "${BASE_CHECKPOINT}" "${BATCH_SIZE_PER_DEVICE}" "${GRADIENT_ACCUMULATION_STEPS}" "${GLOBAL_BATCH_SIZE}" <<'PY'
import json, sys
path, base_checkpoint, per_device_bs, grad_accum, global_bs = sys.argv[1:6]
cfg = json.loads(open(path, encoding="utf-8").read())
cfg["resume_path"] = base_checkpoint
cfg["resume_trainer_state"] = False
cfg["reset_output_heads_on_resume"] = False
cfg["ignore_mismatched_resume_shapes"] = True
cfg["freeze_non_output_heads"] = False
cfg["split_zero_ioi_head"] = True
cfg["epr_timing_target"] = "log_deviation"
cfg["pedal_representation"] = "binary_4"
cfg["output_continuous_dim"] = 7
cfg["per_device_train_batch_size"] = int(per_device_bs)
cfg["per_device_eval_batch_size"] = int(per_device_bs)
cfg["gradient_accumulation_steps"] = int(grad_accum)
cfg["global_batch_size"] = int(global_bs)
cfg["ddp_find_unused_parameters"] = True
cfg["use_prepared_sidecar"] = True
cfg["precompute_dataset_items"] = False
cfg["precompute_eval_dataset_items"] = False
cfg["prepared_sidecar_tag"] = cfg.get("prepared_sidecar_tag") or "ASAP"
cfg.setdefault("eval_dataloader_num_workers", cfg.get("dataloader_num_workers", 0))
cfg.setdefault("eval_dataloader_persistent_workers", bool(int(cfg.get("eval_dataloader_num_workers") or 0) > 0))
cfg.setdefault("eval_dataloader_prefetch_factor", cfg.get("dataloader_prefetch_factor", 2))
cfg.setdefault("loss_component_interval", cfg.get("logging_steps", 20))
open(path, "w", encoding="utf-8").write(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
PY

{
  echo "START $(date '+%F %T')"
  echo "CONFIG ${CONFIG}"
  echo "BASE_CHECKPOINT ${BASE_CHECKPOINT}"
  echo "CUDA_VISIBLE_DEVICES ${CUDA_VISIBLE_DEVICES}"
  echo "MASTER_PORT ${MASTER_PORT}"
  echo "PER_DEVICE_TRAIN_BATCH_SIZE ${BATCH_SIZE_PER_DEVICE}"
  echo "GRADIENT_ACCUMULATION_STEPS ${GRADIENT_ACCUMULATION_STEPS}"
  echo "GLOBAL_BATCH_SIZE ${GLOBAL_BATCH_SIZE}"
  echo "RUN_DIR ${RUN_DIR}"
  echo "ADAPT_EPOCHS_LIST ${ADAPT_EPOCHS_LIST}"
  echo "CHAIN_START_EPOCHS ${CHAIN_START_EPOCHS}"
} | tee -a "${EVALUATE_LOG}"

CHAIN_RESUME_CHECKPOINT="${BASE_CHECKPOINT}"
PREV_TOTAL_EPOCHS="${CHAIN_START_EPOCHS}"

for epochs in ${ADAPT_EPOCHS_LIST}; do
  if (( epochs <= PREV_TOTAL_EPOCHS )); then
    echo "ADAPT_EPOCHS_LIST must be strictly increasing; got ${epochs} after ${PREV_TOTAL_EPOCHS}" >&2
    exit 1
  fi
  STAGE_EPOCHS=$((epochs - PREV_TOTAL_EPOCHS))
  ADAPT_DIR="${RUN_DIR}/adapt_${epochs}ep"
  ADAPT_TRAIN_ROOT="${ADAPT_DIR}/training"
  ADAPT_TF_LOG_ROOT="${ADAPT_DIR}/tf-logs"
  ADAPT_CONFIG="${ADAPT_DIR}/config.json"
  DET_DIR="${ADAPT_DIR}/deterministic"
  SAMPLING_DIR="${ADAPT_DIR}/sampling"
  mkdir -p "${ADAPT_TRAIN_ROOT}" "${ADAPT_TF_LOG_ROOT}" "${DET_DIR}" "${SAMPLING_DIR}"

  python - "${RUN_CONFIG}" "${ADAPT_CONFIG}" "${ADAPT_TRAIN_ROOT}" "${ADAPT_TF_LOG_ROOT}" "${STAGE_EPOCHS}" "${BATCH_SIZE_PER_DEVICE}" "${GRADIENT_ACCUMULATION_STEPS}" "${GLOBAL_BATCH_SIZE}" "${CHAIN_RESUME_CHECKPOINT}" <<'PY'
import json, sys
src, dst, train_root, tf_log_root, epochs, per_device_bs, grad_accum, global_bs, resume_checkpoint = sys.argv[1:10]
cfg = json.loads(open(src, encoding="utf-8").read())
cfg["resume_path"] = resume_checkpoint
cfg["resume_trainer_state"] = False
cfg["output_dir"] = train_root
cfg["logging_dir"] = tf_log_root
cfg["learning_rate"] = float(cfg.get("adapt_learning_rate", cfg.get("learning_rate", 1e-4)))
cfg["num_train_epochs"] = float(epochs)
cfg["max_steps"] = -1
cfg["train_performance_dataset"] = "ASAP"
cfg["eval_split"] = "test"
cfg["eval_performance_dataset"] = "ASAP"
cfg["eval_include_all_performance_dataset"] = None
cfg["max_eval_non_asap_performances_per_work"] = None
cfg["per_device_train_batch_size"] = int(per_device_bs)
cfg["per_device_eval_batch_size"] = int(per_device_bs)
cfg["gradient_accumulation_steps"] = int(grad_accum)
cfg["global_batch_size"] = int(global_bs)
cfg["save_steps"] = min(int(cfg.get("save_steps", 500)), 500)
cfg["eval_steps"] = min(int(cfg.get("eval_steps", 500)), 500)
cfg["save_total_limit"] = 2
cfg["load_best_model_at_end"] = True
cfg["metric_for_best_model"] = "eval_loss"
cfg["greater_is_better"] = False
open(dst, "w", encoding="utf-8").write(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
PY

  ADAPT_MARKER="${TMP_DIR}/adapt_${epochs}ep_start.marker"
  touch "${ADAPT_MARKER}"
  {
    echo "ADAPT_${epochs}EP_RESUME_CHECKPOINT ${CHAIN_RESUME_CHECKPOINT}"
    echo "ADAPT_${epochs}EP_STAGE_EPOCHS ${STAGE_EPOCHS}"
  } | tee -a "${EVALUATE_LOG}"

  run_train_ddp "${ADAPT_CONFIG}" "adapt_${epochs}ep"
  ADAPT_OUTPUT_DIR="$(latest_train_dir "${ADAPT_TRAIN_ROOT}" "${ADAPT_MARKER}")"
  [[ -n "${ADAPT_OUTPUT_DIR}" ]] || { echo "Missing adapt output under ${ADAPT_TRAIN_ROOT}" >&2; exit 1; }
  ADAPT_CHECKPOINT="$(best_checkpoint "${ADAPT_OUTPUT_DIR}")"
  {
    echo "ADAPT_${epochs}EP_OUTPUT_DIR ${ADAPT_OUTPUT_DIR}"
    echo "ADAPT_${epochs}EP_CHECKPOINT ${ADAPT_CHECKPOINT}"
  } | tee -a "${EVALUATE_LOG}"

  run_infer "${ADAPT_CONFIG}" "${ADAPT_CHECKPOINT}" deterministic "${DET_NUM_SAMPLES}" "${DET_DIR}" "${DET_GPU}" &
  det_pid=$!
  run_infer "${ADAPT_CONFIG}" "${ADAPT_CHECKPOINT}" sampling "${SAMPLING_NUM_SAMPLES}" "${SAMPLING_DIR}" "${SAMPLING_GPU}" &
  sampling_pid=$!
  det_status=0
  sampling_status=0
  wait "${det_pid}" || det_status=$?
  wait "${sampling_pid}" || sampling_status=$?
  if [[ "${det_status}" -ne 0 || "${sampling_status}" -ne 0 ]]; then
    echo "Inference failed: deterministic=${det_status}, sampling=${sampling_status}" >&2
    exit 1
  fi
  summarize_pair \
    "${ADAPT_CONFIG}" \
    "${ADAPT_CHECKPOINT}" \
    "${ADAPT_OUTPUT_DIR}" \
    "${DET_DIR}" \
    "${SAMPLING_DIR}" \
    "${ADAPT_DIR}/summary.json" \
    "${ADAPT_DIR}/asap_label_distribution.png"

  CHAIN_RESUME_CHECKPOINT="${ADAPT_CHECKPOINT}"
  PREV_TOTAL_EPOCHS="${epochs}"
done

rm -rf "${TMP_DIR}"
echo "END $(date '+%F %T')" | tee -a "${EVALUATE_LOG}"
