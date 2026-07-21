#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

: "${CONFIG:?CONFIG is required}"

INFER_NUM_WORKERS="${INFER_NUM_WORKERS:-8}"
METRIC_NUM_WORKERS="${METRIC_NUM_WORKERS:-8}"
SAMPLING_NUM_SAMPLES="${SAMPLING_NUM_SAMPLES:-1}"
INFER_BATCH_SIZE_WINDOWS="${INFER_BATCH_SIZE_WINDOWS:-8}"
INFER_SCORE_SOURCE_LIST="${INFER_SCORE_SOURCE_LIST:-}"
MERGE_MODE="${MERGE_MODE:-continuation}"
CONTINUATION_DROP_RATIO="${CONTINUATION_DROP_RATIO:-0.0}"
BASE_NUM_TRAIN_EPOCHS="${BASE_NUM_TRAIN_EPOCHS:-8}"
ADAPT_NUM_TRAIN_EPOCHS="${ADAPT_NUM_TRAIN_EPOCHS:-16}"
ADAPT_PREPARED_SIDECAR_TAG="${ADAPT_PREPARED_SIDECAR_TAG:-}"
BASE_PREPARED_SIDECAR_TAG="${BASE_PREPARED_SIDECAR_TAG:-}"
BASE_ASAP_ONLY="${BASE_ASAP_ONLY:-0}"
SKIP_BASE_TRAIN="${SKIP_BASE_TRAIN:-0}"
BASE_CHECKPOINT_OVERRIDE="${BASE_CHECKPOINT_OVERRIDE:-}"
RESUME_CHECKPOINT_OVERRIDE="${RESUME_CHECKPOINT_OVERRIDE:-}"
RESUME_FROM_LATEST_CHECKPOINT="${RESUME_FROM_LATEST_CHECKPOINT:-1}"
BATCH_SIZE_PER_DEVICE="${BATCH_SIZE_PER_DEVICE:-32}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
PIPELINE_STAGE_START="${PIPELINE_STAGE_START:-train}"
SKIP_EXISTING_PIPELINE_OUTPUTS="${SKIP_EXISTING_PIPELINE_OUTPUTS:-1}"
EVAL_CHECKPOINT_MODE="${EVAL_CHECKPOINT_MODE:-best}"

PIPELINE_STAGE_START="$(printf '%s' "${PIPELINE_STAGE_START}" | tr '[:upper:]' '[:lower:]')"
case "${PIPELINE_STAGE_START}" in
  train|adapt|infer) ;;
  *)
    echo "Unsupported PIPELINE_STAGE_START=${PIPELINE_STAGE_START}; expected train, adapt, or infer" >&2
    exit 1
    ;;
esac

case "${EVAL_CHECKPOINT_MODE}" in
  best|latest) ;;
  *)
    echo "Unsupported EVAL_CHECKPOINT_MODE=${EVAL_CHECKPOINT_MODE}; expected best or latest" >&2
    exit 1
    ;;
esac

if [[ -n "${RESUME_CHECKPOINT_OVERRIDE}" && -z "${BASE_CHECKPOINT_OVERRIDE}" ]]; then
  BASE_CHECKPOINT_OVERRIDE="${RESUME_CHECKPOINT_OVERRIDE}"
fi

IFS=',' read -ra GPU_LIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
GPU_COUNT="${#GPU_LIST[@]}"
GRADIENT_ACCUMULATION_STEPS=$(( GLOBAL_BATCH_SIZE / (BATCH_SIZE_PER_DEVICE * GPU_COUNT) ))
if [[ "${GRADIENT_ACCUMULATION_STEPS}" -lt 1 || $(( BATCH_SIZE_PER_DEVICE * GPU_COUNT * GRADIENT_ACCUMULATION_STEPS )) -ne "${GLOBAL_BATCH_SIZE}" ]]; then
  echo "Invalid batch setup: per_device=${BATCH_SIZE_PER_DEVICE}, GPUs=${GPU_COUNT}, global=${GLOBAL_BATCH_SIZE}" >&2
  exit 1
fi

DEFAULT_RUN_NAME="${CONFIG##*/}"
DEFAULT_RUN_NAME="${DEFAULT_RUN_NAME%.json}_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR_OVERRIDE:-results/inr_epr_pipeline/${DEFAULT_RUN_NAME}}"
TRAIN_ROOT="${RUN_DIR}/training"
TF_LOG_ROOT="${RUN_DIR}/tf-logs"
RUN_CONFIG="${RUN_DIR}/config.json"
TRAIN_LOG="${RUN_DIR}/train.log"
EVALUATE_LOG="${RUN_DIR}/evaluate.log"
TMP_DIR="${RUN_DIR}/_tmp"
mkdir -p "${RUN_DIR}" "${TRAIN_ROOT}" "${TF_LOG_ROOT}" "${TMP_DIR}"

MASTER_PORT="${MASTER_PORT:-}"
if [[ "${GPU_COUNT}" -gt 1 && -z "${MASTER_PORT}" ]]; then
  MASTER_PORT="$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")"
fi
SAMPLING_GPU="${GPU_LIST[0]}"

latest_train_dir() {
  local root="$1" marker="$2"
  find "${root}" -maxdepth 2 -mindepth 2 -type f -name 'train_config.json' -newer "${marker}" \
    -printf '%T@ %h\n' | sort -nr | head -n 1 | cut -d' ' -f2-
}

latest_numeric_checkpoint() {
  local root="$1"
  find "${root}" -path '*/checkpoint-*' -type d 2>/dev/null \
    | awk -F'checkpoint-' '/checkpoint-[0-9]+$/ {print $2 " " $0}' \
    | sort -n | tail -n 1 | cut -d' ' -f2-
}

best_checkpoint() {
  local train_dir="$1"
  if [[ -d "${train_dir}/checkpoint-best" ]]; then
    echo "${train_dir}/checkpoint-best"
  else
    echo "${train_dir}"
  fi
}

evaluation_checkpoint() {
  local train_dir="$1"
  if [[ "${EVAL_CHECKPOINT_MODE}" == "latest" ]]; then
    latest_numeric_checkpoint "${train_dir}"
  else
    best_checkpoint "${train_dir}"
  fi
}

train_dir_from_checkpoint() {
  local checkpoint="$1"
  local base_name
  base_name="$(basename "${checkpoint}")"
  if [[ "${base_name}" == checkpoint-* ]]; then
    dirname "${checkpoint}"
  else
    echo "${checkpoint}"
  fi
}

run_train() {
  local config="$1" stage="$2"
  if [[ "${GPU_COUNT}" -gt 1 ]]; then
    echo "[$(date '+%F %T')] ${stage}: DDP train, GPUs=${CUDA_VISIBLE_DEVICES}" | tee -a "${EVALUATE_LOG}"
    MASTER_PORT="${MASTER_PORT}" PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo \
      torchrun --nnodes=1 --nproc_per_node="${GPU_COUNT}" \
        --master_addr=127.0.0.1 --master_port="${MASTER_PORT}" \
        src/train/train_inr.py --config "${config}" 2>&1 | tee -a "${TRAIN_LOG}"
  else
    echo "[$(date '+%F %T')] ${stage}: single GPU train" | tee -a "${EVALUATE_LOG}"
    PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      python src/train/train_inr.py --config "${config}" 2>&1 | tee -a "${TRAIN_LOG}"
  fi
}

write_train_config() {
  local src="$1" dst="$2" output_root="$3" log_root="$4" epochs="$5" resume_path="$6" asap_only="$7"
  python - "$src" "$dst" "$output_root" "$log_root" "$epochs" "$resume_path" "$asap_only" \
    "$BATCH_SIZE_PER_DEVICE" "$GRADIENT_ACCUMULATION_STEPS" "$GLOBAL_BATCH_SIZE" "${ADAPT_PREPARED_SIDECAR_TAG}" "${BASE_PREPARED_SIDECAR_TAG}" <<'PY'
import json
import sys
from pathlib import Path

src, dst, output_root, log_root, epochs, resume_path, asap_only, per_device_bs, grad_accum, global_bs, adapt_sidecar_tag, base_sidecar_tag = sys.argv[1:13]
cfg = json.loads(Path(src).read_text(encoding="utf-8"))
if cfg.get("task_type", "epr").lower() != "epr":
    raise SystemExit("run_inr_epr_pipeline.sh requires task_type=epr")
cfg["use_style_tokens"] = False
cfg.setdefault("pedal_representation", "binary_4")
target = str(cfg.get("epr_timing_target", "floor_log_deviation")).lower()
if target not in {"floor_log_deviation", "floor_log_dev"}:
    raise SystemExit("Only epr_timing_target=floor_log_deviation is supported")
pedal_representation = str(cfg.get("pedal_representation", "binary_4")).lower()
pedal_aliases = {"binary4": "binary_4", "pedal4_binary": "binary_4"}
pedal_representation = pedal_aliases.get(pedal_representation, pedal_representation)
if pedal_representation == "binary_4":
    pedal_dim = 4
else:
    raise SystemExit("Only pedal_representation=binary_4 is supported")
cfg["legacy_dual_timing_head"] = False
default_output_dim = 3 + pedal_dim
cfg.setdefault("output_continuous_dim", default_output_dim)
cfg["output_dir"] = output_root
cfg["logging_dir"] = log_root
cfg["num_train_epochs"] = float(epochs)
cfg["max_train_epochs"] = float(epochs)
cfg["per_device_train_batch_size"] = int(per_device_bs)
cfg["per_device_eval_batch_size"] = int(per_device_bs)
cfg["gradient_accumulation_steps"] = int(grad_accum)
cfg["global_batch_size"] = int(global_bs)
cfg.setdefault("use_prepared_sidecar", True)
cfg["precompute_dataset_items"] = False
cfg["precompute_eval_dataset_items"] = False
cfg["fixed_window_split_scheme"] = cfg.get("fixed_window_split_scheme") or "train_valid_asap3_nonasap05_v1"
cfg["fixed_window_base_split"] = cfg.get("fixed_window_base_split") or "train"
cfg["fixed_window_train_split_name"] = cfg.get("fixed_window_train_split_name") or "train"
cfg["fixed_window_eval_split_name"] = cfg.get("fixed_window_eval_split_name") or "valid"
cfg["fixed_window_split_summary_path"] = cfg.get("fixed_window_split_summary_path") or "data/train_valid_asap3_nonasap05_v1_summary.json"
cfg["load_best_model_at_end"] = True
cfg["metric_for_best_model"] = "eval_loss"
cfg["greater_is_better"] = False
cfg["early_stopping_patience"] = int(cfg.get("early_stopping_patience", 5))
cfg["early_stopping_threshold"] = float(cfg.get("early_stopping_threshold", 0.001))
cfg.setdefault("eval_dataloader_persistent_workers", False)
cfg.setdefault("eval_dataloader_num_workers", cfg.get("dataloader_num_workers", 0))
cfg.setdefault("eval_dataloader_prefetch_factor", cfg.get("dataloader_prefetch_factor", 2))
for key in (
    "eval_every_steps",
    "eval_every_epochs",
    "save_every_steps",
    "eval_steps",
    "save_steps",
    "evaluation_strategy",
    "eval_strategy",
    "save_strategy",
):
    cfg.pop(key, None)
if resume_path:
    cfg["resume_path"] = resume_path
    cfg["resume_trainer_state"] = False
else:
    cfg.pop("resume_path", None)
    cfg["resume_trainer_state"] = False
if asap_only == "1":
    cfg["train_performance_dataset"] = "ASAP"
    cfg["eval_performance_dataset"] = "ASAP"
    cfg["eval_split"] = "valid"
    cfg["prepared_sidecar_tag"] = adapt_sidecar_tag or cfg.get("prepared_sidecar_tag") or "ASAP"
else:
    cfg.pop("train_performance_dataset", None)
    cfg.pop("eval_performance_dataset", None)
    if base_sidecar_tag:
        cfg["prepared_sidecar_tag"] = base_sidecar_tag
    else:
        cfg.pop("prepared_sidecar_tag", None)
Path(dst).parent.mkdir(parents=True, exist_ok=True)
Path(dst).write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

run_infer() {
  local config="$1" checkpoint="$2" protocol="$3" samples="$4" out_dir="$5" gpu="$6"
  mkdir -p "${out_dir}"
  echo "[$(date '+%F %T')] infer ${protocol}: ${checkpoint}" | tee -a "${EVALUATE_LOG}"
  local score_source_args=()
  if [[ -n "${INFER_SCORE_SOURCE_LIST}" ]]; then
    score_source_args=(--score-source-list "${INFER_SCORE_SOURCE_LIST}")
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 python src/inference/infer_inr_testset.py \
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
    --num-samples "${samples}" \
    "${score_source_args[@]}" \
    --output-dir "${out_dir}" 2>&1 | tee -a "${EVALUATE_LOG}"
}

maybe_run_infer() {
  local config="$1" checkpoint="$2" protocol="$3" samples="$4" out_dir="$5" gpu="$6"
  local manifest_path="${out_dir}/prediction_manifest.json"
  if [[ "${SKIP_EXISTING_PIPELINE_OUTPUTS}" == "1" && -s "${manifest_path}" ]]; then
    echo "[$(date '+%F %T')] infer ${protocol}: reuse existing ${manifest_path}" | tee -a "${EVALUATE_LOG}"
    return 0
  fi
  run_infer "${config}" "${checkpoint}" "${protocol}" "${samples}" "${out_dir}" "${gpu}"
}

summarize_pair() {
  local config="$1" checkpoint="$2" train_output_dir="$3" det_dir="$4" sampling_dir="$5" summary_json="$6" plot_path="$7"
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
    --num-workers "${METRIC_NUM_WORKERS}" 2>&1 | tee -a "${EVALUATE_LOG}"
}

BASE_CONFIG="${RUN_CONFIG}"
BASE_RESUME=""
if [[ "${RESUME_FROM_LATEST_CHECKPOINT}" == "1" ]]; then
  BASE_RESUME="$(latest_numeric_checkpoint "${TRAIN_ROOT}" || true)"
fi
write_train_config "${CONFIG}" "${BASE_CONFIG}" "${TRAIN_ROOT}" "${TF_LOG_ROOT}" "${BASE_NUM_TRAIN_EPOCHS}" "${BASE_RESUME}" "${BASE_ASAP_ONLY}"

echo "RUN_DIR ${RUN_DIR}" | tee -a "${EVALUATE_LOG}"
BASE_OUTPUT_DIR=""
BASE_CHECKPOINT=""
if [[ "${PIPELINE_STAGE_START}" == "train" ]]; then
  BASE_MARKER="${TMP_DIR}/base.marker"
  touch "${BASE_MARKER}"
  if [[ "${SKIP_BASE_TRAIN}" == "1" ]]; then
    [[ -n "${BASE_CHECKPOINT_OVERRIDE}" ]] || { echo "BASE_CHECKPOINT_OVERRIDE is required" >&2; exit 1; }
    BASE_OUTPUT_DIR="$(train_dir_from_checkpoint "${BASE_CHECKPOINT_OVERRIDE}")"
    BASE_CHECKPOINT="${BASE_CHECKPOINT_OVERRIDE}"
  else
    run_train "${BASE_CONFIG}" "base"
    BASE_OUTPUT_DIR="$(latest_train_dir "${TRAIN_ROOT}" "${BASE_MARKER}")"
    BASE_CHECKPOINT="$(evaluation_checkpoint "${BASE_OUTPUT_DIR}")"
  fi
else
  if [[ -n "${BASE_CHECKPOINT_OVERRIDE}" ]]; then
    BASE_CHECKPOINT="${BASE_CHECKPOINT_OVERRIDE}"
  elif [[ -n "${BASE_RESUME}" ]]; then
    BASE_CHECKPOINT="${BASE_RESUME}"
  else
    BASE_CHECKPOINT="$(latest_numeric_checkpoint "${TRAIN_ROOT}" || true)"
  fi
  [[ -n "${BASE_CHECKPOINT}" ]] || { echo "Could not locate base checkpoint under ${TRAIN_ROOT}" >&2; exit 1; }
  BASE_OUTPUT_DIR="$(train_dir_from_checkpoint "${BASE_CHECKPOINT}")"
fi

FINAL_CONFIG="${BASE_CONFIG}"
FINAL_OUTPUT_DIR="${BASE_OUTPUT_DIR}"
FINAL_CHECKPOINT="${BASE_CHECKPOINT}"
FINAL_DIR="${RUN_DIR}"

if (( ADAPT_NUM_TRAIN_EPOCHS > 0 )); then
  ADAPT_DIR="${RUN_DIR}/adapt_${ADAPT_NUM_TRAIN_EPOCHS}ep"
  ADAPT_CONFIG="${ADAPT_DIR}/config.json"
  ADAPT_TRAIN_ROOT="${ADAPT_DIR}/training"
  ADAPT_LOG_ROOT="${ADAPT_DIR}/tf-logs"
  mkdir -p "${ADAPT_TRAIN_ROOT}" "${ADAPT_LOG_ROOT}"
  write_train_config "${BASE_CONFIG}" "${ADAPT_CONFIG}" "${ADAPT_TRAIN_ROOT}" "${ADAPT_LOG_ROOT}" "${ADAPT_NUM_TRAIN_EPOCHS}" "${BASE_CHECKPOINT}" "1"

  if [[ "${PIPELINE_STAGE_START}" == "train" || "${PIPELINE_STAGE_START}" == "adapt" ]]; then
    ADAPT_MARKER="${TMP_DIR}/adapt.marker"
    touch "${ADAPT_MARKER}"
    run_train "${ADAPT_CONFIG}" "adapt"
    ADAPT_OUTPUT_DIR="$(latest_train_dir "${ADAPT_TRAIN_ROOT}" "${ADAPT_MARKER}")"
    ADAPT_CHECKPOINT="$(evaluation_checkpoint "${ADAPT_OUTPUT_DIR}")"
  else
    ADAPT_CHECKPOINT="$(latest_numeric_checkpoint "${ADAPT_TRAIN_ROOT}" || true)"
    [[ -n "${ADAPT_CHECKPOINT}" ]] || { echo "Could not locate adapt checkpoint under ${ADAPT_TRAIN_ROOT}" >&2; exit 1; }
    ADAPT_OUTPUT_DIR="$(train_dir_from_checkpoint "${ADAPT_CHECKPOINT}")"
  fi

  FINAL_CONFIG="${ADAPT_CONFIG}"
  FINAL_OUTPUT_DIR="${ADAPT_OUTPUT_DIR}"
  FINAL_CHECKPOINT="${ADAPT_CHECKPOINT}"
  FINAL_DIR="${ADAPT_DIR}"
fi

SAMPLING_DIR="${FINAL_DIR}/sampling"
maybe_run_infer "${FINAL_CONFIG}" "${FINAL_CHECKPOINT}" sampling "${SAMPLING_NUM_SAMPLES}" "${SAMPLING_DIR}" "${SAMPLING_GPU}"

if [[ "${SKIP_EXISTING_PIPELINE_OUTPUTS}" == "1" && -s "${FINAL_DIR}/summary.json" ]]; then
  echo "[$(date '+%F %T')] summarize/statistics: reuse existing ${FINAL_DIR}/summary.json" | tee -a "${EVALUATE_LOG}"
else
  summarize_pair \
    "${FINAL_CONFIG}" \
    "${FINAL_CHECKPOINT}" \
    "${FINAL_OUTPUT_DIR}" \
    "${SAMPLING_DIR}" \
    "${SAMPLING_DIR}" \
    "${FINAL_DIR}/summary.json" \
    "${FINAL_DIR}/asap_label_distribution.png"
fi

rm -rf "${TMP_DIR}"
echo "END $(date '+%F %T')" | tee -a "${EVALUATE_LOG}"
