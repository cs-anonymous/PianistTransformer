#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Only one env var users need to set
: "${CONFIG:?CONFIG is required (path to json config file)}"
: "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is required (e.g. 0,1)}"

# Hardcoded pipeline settings
INFER_NUM_WORKERS=8
METRIC_NUM_WORKERS=8
DET_NUM_SAMPLES=1
SAMPLING_NUM_SAMPLES=1
DET_STRATEGY="greedy"
INFER_BATCH_SIZE_WINDOWS=8
MERGE_MODE="continuation"
CONTINUATION_DROP_RATIO=0.0

# Run layout
DEFAULT_RUN_NAME="${CONFIG##*/}"                # basename only, no .json
DEFAULT_RUN_NAME="${DEFAULT_RUN_NAME%.json}_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR_OVERRIDE:-results/inr_pipeline/${DEFAULT_RUN_NAME}}"
TRAIN_LOG="${RUN_DIR}/train.log"
EVALUATE_LOG="${RUN_DIR}/evaluate.log"
SUMMARY_JSON="${RUN_DIR}/summary.json"
PLOT_PATH="${RUN_DIR}/asap_label_distribution.png"
RUN_CONFIG="${RUN_DIR}/config.json"
DET_DIR="${RUN_DIR}/deterministic"
SAMPLING_DIR="${RUN_DIR}/sampling"
TMP_DIR="${RUN_DIR}/_tmp"
TRAIN_ROOT="${RUN_DIR}/training"
TF_LOG_ROOT="${RUN_DIR}/tf-logs"
ADAPT_RUN_DIR="${RUN_DIR}/asap_adapt"
ADAPT_TRAIN_ROOT="${ADAPT_RUN_DIR}/training"
ADAPT_TF_LOG_ROOT="${ADAPT_RUN_DIR}/tf-logs"
ADAPT_CONFIG="${ADAPT_RUN_DIR}/config.json"
mkdir -p "${RUN_DIR}" "${TMP_DIR}" "${TRAIN_ROOT}" "${TF_LOG_ROOT}" "${ADAPT_RUN_DIR}" "${ADAPT_TRAIN_ROOT}" "${ADAPT_TF_LOG_ROOT}"
PIPELINE_STAGE_START="${PIPELINE_STAGE_START:-train}"

find_resume_checkpoint() {
  local search_root="$1"
  python - "$search_root" <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
candidates = []
for path in root.rglob("checkpoint-*"):
    if not path.is_dir():
        continue
    if path.name == "checkpoint-best":
        continue
    m = re.fullmatch(r"checkpoint-(\d+)", path.name)
    if not m:
        continue
    candidates.append((int(m.group(1)), path.stat().st_mtime, str(path)))

if not candidates:
    sys.exit(1)

candidates.sort(key=lambda item: (item[0], item[1]))
print(candidates[-1][2])
PY
}

RESUME_CHECKPOINT="${RESUME_CHECKPOINT_OVERRIDE:-}"
if [[ -z "${RESUME_CHECKPOINT}" && -n "${RUN_DIR_OVERRIDE:-}" ]]; then
  if RESUME_CHECKPOINT="$(find_resume_checkpoint "${TRAIN_ROOT}" 2>/dev/null)"; then
    :
  else
    RESUME_CHECKPOINT=""
  fi
fi

# GPU list (inherited from CUDA_VISIBLE_DEVICES)
IFS=',' read -ra GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
TRAIN_GPU_COUNT=${#GPU_LIST[@]}
[[ ${TRAIN_GPU_COUNT} -ge 1 && ${TRAIN_GPU_COUNT} -le 2 ]] \
  || { echo "CUDA_VISIBLE_DEVICES must be 1 or 2 ids, got: ${CUDA_VISIBLE_DEVICES}" >&2; exit 1; }
DET_GPU="${GPU_LIST[0]}"
SAMPLING_GPU="${GPU_LIST[1]:-${GPU_LIST[0]}}"

# Auto-pick a free port for DDP master (avoids conflicts between concurrent DDP jobs)
MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

# Materialize config (rewrite output/logging dirs into the run dir)
python - "${CONFIG}" "${RUN_CONFIG}" "${TRAIN_ROOT}" "${TF_LOG_ROOT}" "${RESUME_CHECKPOINT}" <<'PY'
import json, sys
src, dst, train_root, tf_log_root, resume_checkpoint = sys.argv[1:6]
cfg = json.loads(open(src).read())
cfg["output_dir"] = train_root
cfg["logging_dir"] = tf_log_root
cfg.setdefault("eval_dataloader_num_workers", cfg.get("dataloader_num_workers", 0))
cfg.setdefault("eval_dataloader_persistent_workers", bool(int(cfg.get("eval_dataloader_num_workers") or 0) > 0))
cfg.setdefault("eval_dataloader_prefetch_factor", cfg.get("dataloader_prefetch_factor", 2))
cfg.setdefault("loss_component_interval", cfg.get("logging_steps", 20))
cfg["use_prepared_sidecar"] = True
cfg["precompute_dataset_items"] = False
cfg["precompute_eval_dataset_items"] = False
if resume_checkpoint:
    cfg["resume_path"] = resume_checkpoint
    cfg["resume_trainer_state"] = True
open(dst, "w").write(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
PY

ADAPT_ON_ASAP="$(
  python - "${RUN_CONFIG}" <<'PY'
import json, sys
cfg = json.loads(open(sys.argv[1], encoding="utf-8").read())
print("1" if cfg.get("adapt_on_asap_after_train") else "0")
PY
)"
ADAPT_LR="$(
  python - "${RUN_CONFIG}" <<'PY'
import json, sys
cfg = json.loads(open(sys.argv[1], encoding="utf-8").read())
print(cfg.get("adapt_learning_rate", 3e-5))
PY
)"
ADAPT_EPOCHS="$(
  python - "${RUN_CONFIG}" <<'PY'
import json, sys
cfg = json.loads(open(sys.argv[1], encoding="utf-8").read())
print(cfg.get("adapt_num_train_epochs", 2))
PY
)"

{
  echo "START $(date '+%F %T')"
  echo "CONFIG ${CONFIG}"
  echo "CUDA_VISIBLE_DEVICES ${CUDA_VISIBLE_DEVICES}"
  echo "MASTER_PORT ${MASTER_PORT}"
  echo "RUN_DIR ${RUN_DIR}"
  echo "PIPELINE_STAGE_START ${PIPELINE_STAGE_START}"
  echo "DET_STRATEGY ${DET_STRATEGY}"
  echo "RESUME_CHECKPOINT ${RESUME_CHECKPOINT:-NONE}"
  echo "ADAPT_ON_ASAP ${ADAPT_ON_ASAP}"
  echo "ADAPT_LR ${ADAPT_LR}"
  echo "ADAPT_EPOCHS ${ADAPT_EPOCHS}"
} | tee -a "${EVALUATE_LOG}"

# --- Train -----------------------------------------------------------------
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR_OVERRIDE:-}"
CHECKPOINT="${CHECKPOINT_OVERRIDE:-}"
if [[ "${PIPELINE_STAGE_START}" == "train" ]]; then
  TRAIN_MARKER="${TMP_DIR}/train_start.marker"
  touch "${TRAIN_MARKER}"
  if [[ "${TRAIN_GPU_COUNT}" -gt 1 ]]; then
    echo "[$(date '+%F %T')] train: DDP start, GPUs=${CUDA_VISIBLE_DEVICES}, nproc=${TRAIN_GPU_COUNT}" | tee -a "${EVALUATE_LOG}"
    MASTER_PORT="${MASTER_PORT}" PYTHONUNBUFFERED=1 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo \
      TORCH_NCCL_BLOCKING_WAIT=1 NCCL_DEBUG=WARN \
      torchrun --nnodes=1 --nproc_per_node="${TRAIN_GPU_COUNT}" \
        --master_addr=127.0.0.1 --master_port="${MASTER_PORT}" \
        src/train/train_inr.py --config "${RUN_CONFIG}" 2>&1 | tee -a "${TRAIN_LOG}"
  else
    echo "[$(date '+%F %T')] train: single GPU start, GPU=${CUDA_VISIBLE_DEVICES}" | tee -a "${EVALUATE_LOG}"
    PYTHONUNBUFFERED=1 \
      python src/train/train_inr.py --config "${RUN_CONFIG}" 2>&1 | tee -a "${TRAIN_LOG}"
  fi
  echo "[$(date '+%F %T')] train: finished" | tee -a "${EVALUATE_LOG}"

  TRAIN_OUTPUT_DIR="$(
    find "${TRAIN_ROOT}" -maxdepth 1 -mindepth 1 -type d -name 'inr_*' -newer "${TRAIN_MARKER}" \
      -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-
  )"
  [[ -n "${TRAIN_OUTPUT_DIR}" ]] \
    || { echo "Could not locate training output under ${TRAIN_ROOT}" >&2; exit 1; }
  CHECKPOINT="${TRAIN_OUTPUT_DIR}"
  [[ -d "${TRAIN_OUTPUT_DIR}/checkpoint-best" ]] && CHECKPOINT="${TRAIN_OUTPUT_DIR}/checkpoint-best"
  {
    echo "TRAIN_OUTPUT_DIR ${TRAIN_OUTPUT_DIR}"
    echo "CHECKPOINT ${CHECKPOINT}"
  } | tee -a "${EVALUATE_LOG}"
elif [[ "${PIPELINE_STAGE_START}" == "adapt" || "${PIPELINE_STAGE_START}" == "infer" ]]; then
  [[ -n "${TRAIN_OUTPUT_DIR}" ]] || TRAIN_OUTPUT_DIR="$(
    find "${TRAIN_ROOT}" -maxdepth 1 -mindepth 1 -type d -name 'inr_*' \
      -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-
  )"
  [[ -n "${TRAIN_OUTPUT_DIR}" ]] \
    || { echo "Could not locate existing training output under ${TRAIN_ROOT}" >&2; exit 1; }
  if [[ -z "${CHECKPOINT}" ]]; then
    CHECKPOINT="${TRAIN_OUTPUT_DIR}"
    [[ -d "${TRAIN_OUTPUT_DIR}/checkpoint-best" ]] && CHECKPOINT="${TRAIN_OUTPUT_DIR}/checkpoint-best"
  fi
  {
    echo "SKIP_BASE_TRAIN 1"
    echo "TRAIN_OUTPUT_DIR ${TRAIN_OUTPUT_DIR}"
    echo "CHECKPOINT ${CHECKPOINT}"
  } | tee -a "${EVALUATE_LOG}"
else
  echo "Unsupported PIPELINE_STAGE_START=${PIPELINE_STAGE_START}; expected train, adapt, or infer" >&2
  exit 1
fi

ACTIVE_CONFIG="${RUN_CONFIG}"
if [[ "${PIPELINE_STAGE_START}" == "infer" && "${ADAPT_ON_ASAP}" == "1" ]]; then
  [[ -f "${ADAPT_CONFIG}" ]] \
    || { echo "Missing adapted config for infer stage: ${ADAPT_CONFIG}" >&2; exit 1; }
  ADAPT_OUTPUT_DIR="${ADAPT_OUTPUT_DIR_OVERRIDE:-}"
  [[ -n "${ADAPT_OUTPUT_DIR}" ]] || ADAPT_OUTPUT_DIR="$(
    find "${ADAPT_TRAIN_ROOT}" -maxdepth 1 -mindepth 1 -type d -name 'inr_*' \
      -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-
  )"
  [[ -n "${ADAPT_OUTPUT_DIR}" ]] \
    || { echo "Could not locate existing adapted output under ${ADAPT_TRAIN_ROOT}" >&2; exit 1; }
  CHECKPOINT="${ADAPT_OUTPUT_DIR}"
  [[ -d "${ADAPT_OUTPUT_DIR}/checkpoint-best" ]] && CHECKPOINT="${ADAPT_OUTPUT_DIR}/checkpoint-best"
  ACTIVE_CONFIG="${ADAPT_CONFIG}"
  {
    echo "SKIP_ADAPT 1"
    echo "ADAPT_OUTPUT_DIR ${ADAPT_OUTPUT_DIR}"
    echo "ADAPTED_CHECKPOINT ${CHECKPOINT}"
  } | tee -a "${EVALUATE_LOG}"
elif [[ "${ADAPT_ON_ASAP}" == "1" ]]; then
  python - "${RUN_CONFIG}" "${ADAPT_CONFIG}" "${ADAPT_TRAIN_ROOT}" "${ADAPT_TF_LOG_ROOT}" "${CHECKPOINT}" "${ADAPT_LR}" "${ADAPT_EPOCHS}" <<'PY'
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
cfg["eval_split"] = "test"
cfg["eval_performance_dataset"] = "ASAP"
cfg["eval_include_all_performance_dataset"] = None
cfg["max_eval_non_asap_performances_per_work"] = None
cfg["use_prepared_sidecar"] = True
cfg["prepared_sidecar_tag"] = cfg.get("prepared_sidecar_tag") or "ASAP"
cfg["precompute_dataset_items"] = False
cfg["precompute_eval_dataset_items"] = False
cfg["save_steps"] = min(int(cfg.get("save_steps", 2000)), 500)
cfg["eval_steps"] = min(int(cfg.get("eval_steps", 2000)), 500)
cfg["logging_steps"] = int(cfg.get("logging_steps", 20))
cfg["save_total_limit"] = 2
cfg["load_best_model_at_end"] = True
cfg["metric_for_best_model"] = "eval_loss"
cfg["greater_is_better"] = False
cfg.setdefault("eval_dataloader_num_workers", cfg.get("dataloader_num_workers", 0))
cfg.setdefault("eval_dataloader_persistent_workers", bool(int(cfg.get("eval_dataloader_num_workers") or 0) > 0))
cfg.setdefault("eval_dataloader_prefetch_factor", cfg.get("dataloader_prefetch_factor", 2))
cfg.setdefault("loss_component_interval", cfg.get("logging_steps", 20))
cfg["use_prepared_sidecar"] = True
cfg["precompute_dataset_items"] = False
cfg["precompute_eval_dataset_items"] = False
open(dst, "w", encoding="utf-8").write(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
PY

  ADAPT_MARKER="${TMP_DIR}/adapt_start.marker"
  touch "${ADAPT_MARKER}"
  if [[ "${TRAIN_GPU_COUNT}" -gt 1 ]]; then
    echo "[$(date '+%F %T')] adapt: DDP start, GPUs=${CUDA_VISIBLE_DEVICES}, nproc=${TRAIN_GPU_COUNT}" | tee -a "${EVALUATE_LOG}"
    MASTER_PORT="${MASTER_PORT}" PYTHONUNBUFFERED=1 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo \
      TORCH_NCCL_BLOCKING_WAIT=1 NCCL_DEBUG=WARN \
      torchrun --nnodes=1 --nproc_per_node="${TRAIN_GPU_COUNT}" \
        --master_addr=127.0.0.1 --master_port="${MASTER_PORT}" \
        src/train/train_inr.py --config "${ADAPT_CONFIG}" 2>&1 | tee -a "${TRAIN_LOG}"
  else
    echo "[$(date '+%F %T')] adapt: single GPU start, GPU=${CUDA_VISIBLE_DEVICES}" | tee -a "${EVALUATE_LOG}"
    PYTHONUNBUFFERED=1 \
      python src/train/train_inr.py --config "${ADAPT_CONFIG}" 2>&1 | tee -a "${TRAIN_LOG}"
  fi
  echo "[$(date '+%F %T')] adapt: finished" | tee -a "${EVALUATE_LOG}"

  ADAPT_OUTPUT_DIR="$(
    find "${ADAPT_TRAIN_ROOT}" -maxdepth 1 -mindepth 1 -type d -name 'inr_*' -newer "${ADAPT_MARKER}" \
      -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-
  )"
  [[ -n "${ADAPT_OUTPUT_DIR}" ]] \
    || { echo "Could not locate adapted training output under ${ADAPT_TRAIN_ROOT}" >&2; exit 1; }
  CHECKPOINT="${ADAPT_OUTPUT_DIR}"
  [[ -d "${ADAPT_OUTPUT_DIR}/checkpoint-best" ]] && CHECKPOINT="${ADAPT_OUTPUT_DIR}/checkpoint-best"
  ACTIVE_CONFIG="${ADAPT_CONFIG}"
  {
    echo "ADAPT_OUTPUT_DIR ${ADAPT_OUTPUT_DIR}"
    echo "ADAPTED_CHECKPOINT ${CHECKPOINT}"
  } | tee -a "${EVALUATE_LOG}"
fi

# --- Inference: deterministic + sampling in parallel -----------------------
COMMON_INFER_ARGS=(
  --config "${ACTIVE_CONFIG}"
  --checkpoint "${CHECKPOINT}"
  --split test
  --performance-dataset ASAP
  --num-workers "${INFER_NUM_WORKERS}"
  --batch-size-windows "${INFER_BATCH_SIZE_WINDOWS}"
  --merge-mode "${MERGE_MODE}"
  --continuation-drop-ratio "${CONTINUATION_DROP_RATIO}"
  --device cuda
)

run_infer() {
  local protocol="$1" num_samples="$2" out_dir="$3" infer_gpu="$4"
  echo "[$(date '+%F %T')] infer ${protocol}: GPU=${infer_gpu}, workers=${INFER_NUM_WORKERS}" | tee -a "${EVALUATE_LOG}"
  CUDA_VISIBLE_DEVICES="${infer_gpu}" PYTHONUNBUFFERED=1 \
    python src/inference/infer_inr_testset.py "${COMMON_INFER_ARGS[@]}" \
      --protocol "${protocol}" --num-samples "${num_samples}" --output-dir "${out_dir}" \
      --deterministic-strategy "${DET_STRATEGY}" \
    2>&1 | tee -a "${EVALUATE_LOG}"
  echo "[$(date '+%F %T')] infer ${protocol}: finished" | tee -a "${EVALUATE_LOG}"
}

mkdir -p "${DET_DIR}" "${SAMPLING_DIR}"
if [[ "${TRAIN_GPU_COUNT}" -gt 1 ]]; then
  run_infer deterministic "${DET_NUM_SAMPLES}" "${DET_DIR}" "${DET_GPU}" &
  DET_PID=$!
  run_infer sampling "${SAMPLING_NUM_SAMPLES}" "${SAMPLING_DIR}" "${SAMPLING_GPU}" &
  SAMPLING_PID=$!
  wait "${DET_PID}" || { echo "deterministic inference failed" >&2; exit 1; }
  wait "${SAMPLING_PID}" || { echo "sampling inference failed" >&2; exit 1; }
else
  run_infer deterministic "${DET_NUM_SAMPLES}" "${DET_DIR}" "${DET_GPU}"
  run_infer sampling "${SAMPLING_NUM_SAMPLES}" "${SAMPLING_DIR}" "${SAMPLING_GPU}"
fi

# --- Summarize -------------------------------------------------------------
echo "[$(date '+%F %T')] summarize: ASAP metrics + plot, workers=${METRIC_NUM_WORKERS}" | tee -a "${EVALUATE_LOG}"
PYTHONUNBUFFERED=1 python src/evaluate/summarize_inr_asap_pipeline.py \
  --deterministic-manifest "${DET_DIR}/prediction_manifest.json" \
  --sampling-manifest      "${SAMPLING_DIR}/prediction_manifest.json" \
  --output-json            "${SUMMARY_JSON}" \
  --output-plot            "${PLOT_PATH}" \
  --config                 "${ACTIVE_CONFIG}" \
  --checkpoint             "${CHECKPOINT}" \
  --train-output-dir       "${ADAPT_OUTPUT_DIR:-${TRAIN_OUTPUT_DIR}}" \
  --pipeline-log           "${TRAIN_LOG}" \
  --evaluate-log           "${EVALUATE_LOG}" \
  --num-workers            "${METRIC_NUM_WORKERS}" \
  2>&1 | tee -a "${EVALUATE_LOG}"

rm -rf "${TMP_DIR}"
echo "END $(date '+%F %T')" | tee -a "${EVALUATE_LOG}"
