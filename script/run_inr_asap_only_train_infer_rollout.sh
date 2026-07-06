#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

: "${CONFIG:?CONFIG is required}"
: "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is required}"

INFER_NUM_WORKERS="${INFER_NUM_WORKERS:-16}"
METRIC_NUM_WORKERS="${METRIC_NUM_WORKERS:-8}"
TRAIN_NUM_EPOCHS="${TRAIN_NUM_EPOCHS:-16}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-5}"
EARLY_STOP_THRESHOLD="${EARLY_STOP_THRESHOLD:-0.001}"
DET_NUM_SAMPLES="${DET_NUM_SAMPLES:-1}"
SAMPLING_NUM_SAMPLES="${SAMPLING_NUM_SAMPLES:-1}"
DET_STRATEGY="${DET_STRATEGY:-greedy}"
INFER_BATCH_SIZE_WINDOWS="${INFER_BATCH_SIZE_WINDOWS:-8}"
MERGE_MODE="${MERGE_MODE:-continuation}"
CONTINUATION_DROP_RATIO="${CONTINUATION_DROP_RATIO:-0.0}"
ROLLOUT_DIAG_ENABLE="${ROLLOUT_DIAG_ENABLE:-1}"
ROLLOUT_DIAG_KS="${ROLLOUT_DIAG_KS:-0,1}"
ROLLOUT_DIAG_SPLIT="${ROLLOUT_DIAG_SPLIT:-test}"
ROLLOUT_DIAG_PERFORMANCE_DATASET="${ROLLOUT_DIAG_PERFORMANCE_DATASET:-ASAP}"
ROLLOUT_DIAG_NUM_WORKERS="${ROLLOUT_DIAG_NUM_WORKERS:-16}"
ROLLOUT_DIAG_BATCH_SIZE_WINDOWS="${ROLLOUT_DIAG_BATCH_SIZE_WINDOWS:-8}"
ROLLOUT_DIAG_MATERIALIZE_STRATEGY="${ROLLOUT_DIAG_MATERIALIZE_STRATEGY:-sample}"
ROLLOUT_DIAG_FEEDBACK_STRATEGY="${ROLLOUT_DIAG_FEEDBACK_STRATEGY:-}"
ROLLOUT_DIAG_SCORE_SOURCE_LIST="${ROLLOUT_DIAG_SCORE_SOURCE_LIST:-}"
ROLLOUT_DIAG_MAX_WORKS="${ROLLOUT_DIAG_MAX_WORKS:-}"

IFS=',' read -ra GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
GPU_COUNT="${#GPU_LIST[@]}"
if [[ "${GPU_COUNT}" -ne 1 && "${GPU_COUNT}" -ne 2 ]]; then
  echo "CUDA_VISIBLE_DEVICES must contain 1 or 2 GPU ids, got: ${CUDA_VISIBLE_DEVICES}" >&2
  exit 1
fi

BATCH_SIZE_PER_DEVICE=32
GLOBAL_BATCH_SIZE=128
GRADIENT_ACCUMULATION_STEPS=$(( GLOBAL_BATCH_SIZE / (BATCH_SIZE_PER_DEVICE * GPU_COUNT) ))
if [[ "${GRADIENT_ACCUMULATION_STEPS}" -lt 1 || $(( BATCH_SIZE_PER_DEVICE * GPU_COUNT * GRADIENT_ACCUMULATION_STEPS )) -ne "${GLOBAL_BATCH_SIZE}" ]]; then
  echo "Invalid batch setup for GPU_COUNT=${GPU_COUNT}" >&2
  exit 1
fi

TRAIN_GPU_COUNT="${GPU_COUNT}"
MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
DET_GPU="${GPU_LIST[0]}"
if [[ "${GPU_COUNT}" -ge 2 ]]; then
  SAMPLING_GPU="${GPU_LIST[1]}"
else
  SAMPLING_GPU="${GPU_LIST[0]}"
fi

DEFAULT_RUN_NAME="${CONFIG##*/}"
DEFAULT_RUN_NAME="${DEFAULT_RUN_NAME%.json}_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR_OVERRIDE:-results/inr0624_reprsplit_pair/${DEFAULT_RUN_NAME}}"
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

run_train() {
  local config="$1"
  if [[ "${TRAIN_GPU_COUNT}" -gt 1 ]]; then
    echo "[$(date '+%F %T')] train: DDP start, GPUs=${CUDA_VISIBLE_DEVICES}, nproc=${TRAIN_GPU_COUNT}" | tee -a "${EVALUATE_LOG}"
    MASTER_PORT="${MASTER_PORT}" PYTHONUNBUFFERED=1 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo \
      TORCH_NCCL_BLOCKING_WAIT=1 NCCL_DEBUG=WARN \
      torchrun --nnodes=1 --nproc_per_node="${TRAIN_GPU_COUNT}" \
        --master_addr=127.0.0.1 --master_port="${MASTER_PORT}" \
        src/train/train_inr.py --config "${config}" 2>&1 | tee -a "${TRAIN_LOG}"
  else
    echo "[$(date '+%F %T')] train: single GPU start, GPU=${CUDA_VISIBLE_DEVICES}" | tee -a "${EVALUATE_LOG}"
    PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      python src/train/train_inr.py --config "${config}" 2>&1 | tee -a "${TRAIN_LOG}"
  fi
  echo "[$(date '+%F %T')] train: finished" | tee -a "${EVALUATE_LOG}"
}

run_infer() {
  local config="$1" checkpoint="$2" protocol="$3" num_samples="$4" out_dir="$5" infer_gpu="$6"
  echo "[$(date '+%F %T')] infer ${protocol}: checkpoint=${checkpoint}, gpu=${infer_gpu}" | tee -a "${EVALUATE_LOG}"
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

run_rollout_diag() {
  local config="$1" checkpoint="$2" out_dir="$3" infer_gpu="$4" rollout_ks="${5:-${ROLLOUT_DIAG_KS}}"
  local cmd=(
    python src/evaluate/eval_inr_rollout_k_curve.py
    --config "${config}"
    --checkpoint "${checkpoint}"
    --output-dir "${out_dir}"
    --split "${ROLLOUT_DIAG_SPLIT}"
    --performance-dataset "${ROLLOUT_DIAG_PERFORMANCE_DATASET}"
    --num-workers "${ROLLOUT_DIAG_NUM_WORKERS}"
    --batch-size-windows "${ROLLOUT_DIAG_BATCH_SIZE_WINDOWS}"
    --device cuda
    --materialize-strategy "${ROLLOUT_DIAG_MATERIALIZE_STRATEGY}"
    --rollout-ks "${rollout_ks}"
  )
  if [[ -n "${ROLLOUT_DIAG_FEEDBACK_STRATEGY}" ]]; then
    cmd+=(--feedback-strategy "${ROLLOUT_DIAG_FEEDBACK_STRATEGY}")
  fi
  if [[ -n "${ROLLOUT_DIAG_SCORE_SOURCE_LIST}" ]]; then
    cmd+=(--score-source-list "${ROLLOUT_DIAG_SCORE_SOURCE_LIST}")
  fi
  if [[ -n "${ROLLOUT_DIAG_MAX_WORKS}" ]]; then
    cmd+=(--max-works "${ROLLOUT_DIAG_MAX_WORKS}")
  fi
  echo "[$(date '+%F %T')] rollout diag: checkpoint=${checkpoint}, gpu=${infer_gpu}, ks=${rollout_ks}, out=${out_dir}" | tee -a "${EVALUATE_LOG}"
  CUDA_VISIBLE_DEVICES="${infer_gpu}" PYTHONUNBUFFERED=1 "${cmd[@]}" 2>&1 | tee -a "${EVALUATE_LOG}"
  echo "[$(date '+%F %T')] rollout diag: finished" | tee -a "${EVALUATE_LOG}"
}

merge_rollout_diag_summaries() {
  local out_dir="$1"
  shift
  python - "${out_dir}" "$@" <<'PY'
import json
import pathlib
import sys

out_dir = pathlib.Path(sys.argv[1])
summary_paths = [pathlib.Path(path) for path in sys.argv[2:]]
summaries = [json.loads(path.read_text(encoding="utf-8")) for path in summary_paths]
if not summaries:
    raise SystemExit("No rollout summaries to merge")

merged = dict(summaries[0])
rollout_ks = []
aggregate_by_k = {}
items_by_score = {}

for summary in summaries:
    for rollout_k in summary.get("rollout_ks", []):
        if rollout_k not in rollout_ks:
            rollout_ks.append(rollout_k)
    aggregate_by_k.update(summary.get("aggregate_by_k", {}))
    for item in summary.get("items", []):
        score_source = item.get("score_source")
        merged_item = items_by_score.setdefault(
            score_source,
            {"score_source": score_source, "by_k": {}},
        )
        merged_item["by_k"].update(item.get("by_k", {}))

merged["rollout_ks"] = rollout_ks
merged["aggregate_by_k"] = aggregate_by_k
merged["items"] = list(items_by_score.values())
merged["num_scores"] = len(merged["items"])

out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "summary.json").write_text(
    json.dumps(merged, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
PY
}

run_rollout_diag_suite() {
  local config="$1" checkpoint="$2" out_dir="$3"
  local rollout_k_list=()
  local raw_k rollout_k
  IFS=',' read -ra rollout_k_list <<< "${ROLLOUT_DIAG_KS}"

  if [[ "${GPU_COUNT}" -gt 1 && "${#rollout_k_list[@]}" -gt 1 && "${#rollout_k_list[@]}" -le "${GPU_COUNT}" ]]; then
    local pids=()
    local summary_paths=()
    local idx gpu k_out_dir
    echo "[$(date '+%F %T')] rollout diag: split ks across GPUs, ks=${ROLLOUT_DIAG_KS}, gpus=${CUDA_VISIBLE_DEVICES}" | tee -a "${EVALUATE_LOG}"
    for idx in "${!rollout_k_list[@]}"; do
      raw_k="${rollout_k_list[${idx}]}"
      rollout_k="${raw_k//[[:space:]]/}"
      [[ -n "${rollout_k}" ]] || continue
      gpu="${GPU_LIST[${idx}]}"
      k_out_dir="${out_dir}/k${rollout_k}"
      run_rollout_diag "${config}" "${checkpoint}" "${k_out_dir}" "${gpu}" "${rollout_k}" &
      pids+=("$!")
      summary_paths+=("${k_out_dir}/summary.json")
    done
    for idx in "${!pids[@]}"; do
      wait "${pids[${idx}]}" || { echo "rollout diag failed: ${summary_paths[${idx}]}" >&2; exit 1; }
    done
    merge_rollout_diag_summaries "${out_dir}" "${summary_paths[@]}"
    echo "[$(date '+%F %T')] rollout diag: merged split-GPU summary -> ${out_dir}/summary.json" | tee -a "${EVALUATE_LOG}"
  else
    run_rollout_diag "${config}" "${checkpoint}" "${out_dir}" "${DET_GPU}" "${ROLLOUT_DIAG_KS}"
  fi
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

python - "${RUN_CONFIG}" "${TRAIN_ROOT}" "${TF_LOG_ROOT}" "${BATCH_SIZE_PER_DEVICE}" "${GRADIENT_ACCUMULATION_STEPS}" "${GLOBAL_BATCH_SIZE}" "${TRAIN_NUM_EPOCHS}" "${EARLY_STOP_PATIENCE}" "${EARLY_STOP_THRESHOLD}" <<'PY'
import json, sys

path, train_root, tf_log_root, per_device_bs, grad_accum, global_bs, train_epochs, early_stop_patience, early_stop_threshold = sys.argv[1:10]
cfg = json.loads(open(path, encoding="utf-8").read())
cfg["output_dir"] = train_root
cfg["logging_dir"] = tf_log_root
cfg["resume_path"] = None
cfg["resume_trainer_state"] = False
cfg["reset_output_heads_on_resume"] = False
cfg["ignore_mismatched_resume_shapes"] = False
cfg["per_device_train_batch_size"] = int(per_device_bs)
cfg["per_device_eval_batch_size"] = int(per_device_bs)
cfg["gradient_accumulation_steps"] = int(grad_accum)
cfg["global_batch_size"] = int(global_bs)
cfg["num_train_epochs"] = float(train_epochs)
cfg["max_train_epochs"] = float(train_epochs)
cfg["max_steps"] = -1
cfg["train_performance_dataset"] = "ASAP"
cfg["eval_performance_dataset"] = "ASAP"
cfg["prepared_sidecar_tag"] = "ASAP"
cfg["fixed_window_split_scheme"] = "train_valid_asap3_nonasap05_v1"
cfg["fixed_window_base_split"] = "train"
cfg["fixed_window_train_split_name"] = "train"
cfg["fixed_window_eval_split_name"] = "valid"
cfg["fixed_window_split_summary_path"] = "data/train_valid_asap3_nonasap05_v1_summary.json"
cfg["eval_split"] = "valid"
cfg["eval_include_all_performance_dataset"] = None
cfg["max_eval_non_asap_performances_per_work"] = None
cfg["use_prepared_sidecar"] = True
cfg["precompute_dataset_items"] = False
cfg["precompute_eval_dataset_items"] = False
cfg["load_best_model_at_end"] = True
cfg["metric_for_best_model"] = "eval_loss"
cfg["greater_is_better"] = False
cfg["early_stopping_patience"] = int(early_stop_patience)
cfg["early_stopping_threshold"] = float(early_stop_threshold)
cfg["save_total_limit"] = max(2, int(cfg.get("save_total_limit", 2) or 2))
cfg["eval_every_epochs"] = 0.5
cfg.pop("eval_every_steps", None)
cfg.pop("eval_steps", None)
cfg.pop("save_every_steps", None)
cfg.pop("save_steps", None)
cfg["eval_record_components"] = False
cfg["eval_compute_wass"] = False
cfg["prediction_loss_only"] = True
cfg["loss_component_interval"] = 0
cfg.setdefault("eval_dataloader_num_workers", cfg.get("dataloader_num_workers", 0))
cfg.setdefault("eval_dataloader_persistent_workers", False)
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
  echo "TRAIN_NUM_EPOCHS ${TRAIN_NUM_EPOCHS}"
  echo "EARLY_STOP_PATIENCE ${EARLY_STOP_PATIENCE}"
  echo "ROLLOUT_DIAG_ENABLE ${ROLLOUT_DIAG_ENABLE}"
} | tee -a "${EVALUATE_LOG}"

TRAIN_MARKER="${TMP_DIR}/train_start.marker"
touch "${TRAIN_MARKER}"
run_train "${RUN_CONFIG}"
TRAIN_OUTPUT_DIR="$(latest_train_dir "${TRAIN_ROOT}" "${TRAIN_MARKER}")"
[[ -n "${TRAIN_OUTPUT_DIR}" ]] || { echo "Missing train output under ${TRAIN_ROOT}" >&2; exit 1; }
TRAIN_CHECKPOINT="$(best_checkpoint "${TRAIN_OUTPUT_DIR}")"
{
  echo "TRAIN_OUTPUT_DIR ${TRAIN_OUTPUT_DIR}"
  echo "TRAIN_CHECKPOINT ${TRAIN_CHECKPOINT}"
} | tee -a "${EVALUATE_LOG}"

DET_DIR="${RUN_DIR}/deterministic"
SAMPLING_DIR="${RUN_DIR}/sampling"
mkdir -p "${DET_DIR}" "${SAMPLING_DIR}"
if [[ "${GPU_COUNT}" -gt 1 ]]; then
  run_infer "${RUN_CONFIG}" "${TRAIN_CHECKPOINT}" deterministic "${DET_NUM_SAMPLES}" "${DET_DIR}" "${DET_GPU}" &
  DET_PID=$!
  run_infer "${RUN_CONFIG}" "${TRAIN_CHECKPOINT}" sampling "${SAMPLING_NUM_SAMPLES}" "${SAMPLING_DIR}" "${SAMPLING_GPU}" &
  SAMPLING_PID=$!
  wait "${DET_PID}" || { echo "deterministic inference failed" >&2; exit 1; }
  wait "${SAMPLING_PID}" || { echo "sampling inference failed" >&2; exit 1; }
else
  run_infer "${RUN_CONFIG}" "${TRAIN_CHECKPOINT}" deterministic "${DET_NUM_SAMPLES}" "${DET_DIR}" "${DET_GPU}"
  run_infer "${RUN_CONFIG}" "${TRAIN_CHECKPOINT}" sampling "${SAMPLING_NUM_SAMPLES}" "${SAMPLING_DIR}" "${SAMPLING_GPU}"
fi

summarize_pair \
  "${RUN_CONFIG}" \
  "${TRAIN_CHECKPOINT}" \
  "${TRAIN_OUTPUT_DIR}" \
  "${DET_DIR}" \
  "${SAMPLING_DIR}" \
  "${RUN_DIR}/summary.json" \
  "${RUN_DIR}/asap_label_distribution.png"

if [[ "${ROLLOUT_DIAG_ENABLE}" == "1" ]]; then
  run_rollout_diag_suite \
    "${RUN_CONFIG}" \
    "${TRAIN_CHECKPOINT}" \
    "${RUN_DIR}/rollout_kdiag"
fi

rm -rf "${TMP_DIR}"
echo "END $(date '+%F %T')" | tee -a "${EVALUATE_LOG}"
