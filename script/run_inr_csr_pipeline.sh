#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

: "${CONFIG:?CONFIG is required (path to json config file)}"
: "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is required (single GPU id, e.g. 0)}"

INFER_NUM_WORKERS="${INFER_NUM_WORKERS:-1}"
INFER_BATCH_SIZE_WINDOWS="${INFER_BATCH_SIZE_WINDOWS:-8}"
MERGE_MODE="${MERGE_MODE:-continuation}"
CONTINUATION_DROP_RATIO="${CONTINUATION_DROP_RATIO:-0.0}"
PIPELINE_STAGE_START="${PIPELINE_STAGE_START:-train}"

RUN_NAME="${CONFIG##*/}"
RUN_NAME="${RUN_NAME%.json}_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR_OVERRIDE:-results/inr_csr_pipeline/${RUN_NAME}}"
TRAIN_LOG="${RUN_DIR}/train.log"
EVALUATE_LOG="${RUN_DIR}/evaluate.log"
RUN_CONFIG="${RUN_DIR}/config.json"
INFER_DIR="${RUN_DIR}/inference"
TMP_DIR="${RUN_DIR}/_tmp"
TRAIN_ROOT="${RUN_DIR}/training"
TF_LOG_ROOT="${RUN_DIR}/tf-logs"
ADAPT_RUN_DIR="${RUN_DIR}/asap_adapt"
ADAPT_TRAIN_ROOT="${ADAPT_RUN_DIR}/training"
ADAPT_TF_LOG_ROOT="${ADAPT_RUN_DIR}/tf-logs"
ADAPT_CONFIG="${ADAPT_RUN_DIR}/config.json"

mkdir -p \
  "${RUN_DIR}" "${INFER_DIR}" "${TMP_DIR}" "${TRAIN_ROOT}" "${TF_LOG_ROOT}" \
  "${ADAPT_RUN_DIR}" "${ADAPT_TRAIN_ROOT}" "${ADAPT_TF_LOG_ROOT}"

IFS=',' read -ra GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
[[ ${#GPU_LIST[@]} -eq 1 ]] || { echo "CSR pipeline expects exactly one GPU id, got: ${CUDA_VISIBLE_DEVICES}" >&2; exit 1; }

find_latest_run_dir() {
  local root="$1"
  find "${root}" -maxdepth 1 -mindepth 1 -type d -name 'inr_*' \
    -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-
}

find_resume_checkpoint() {
  local search_root="$1"
  python - "$search_root" <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
candidates = []
for path in root.rglob("checkpoint-*"):
    if not path.is_dir() or path.name == "checkpoint-best":
        continue
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    if match:
        candidates.append((int(match.group(1)), path.stat().st_mtime, str(path)))
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

python - "${CONFIG}" "${RUN_CONFIG}" "${TRAIN_ROOT}" "${TF_LOG_ROOT}" "${RESUME_CHECKPOINT}" <<'PY'
import json
import sys

src, dst, train_root, tf_log_root, resume_checkpoint = sys.argv[1:6]
cfg = json.loads(open(src, encoding="utf-8").read())
cfg["output_dir"] = train_root
cfg["logging_dir"] = tf_log_root
if resume_checkpoint:
    cfg["resume_path"] = resume_checkpoint
    cfg["resume_trainer_state"] = True
open(dst, "w", encoding="utf-8").write(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
PY

ADAPT_ON_ASAP="$(python - "${RUN_CONFIG}" <<'PY'
import json
import sys
cfg = json.loads(open(sys.argv[1], encoding="utf-8").read())
print("1" if cfg.get("adapt_on_asap_after_train") else "0")
PY
)"
ADAPT_LR="$(python - "${RUN_CONFIG}" <<'PY'
import json
import sys
cfg = json.loads(open(sys.argv[1], encoding="utf-8").read())
print(cfg.get("adapt_learning_rate", 3e-5))
PY
)"
ADAPT_EPOCHS="$(python - "${RUN_CONFIG}" <<'PY'
import json
import sys
cfg = json.loads(open(sys.argv[1], encoding="utf-8").read())
print(cfg.get("adapt_num_train_epochs", 2))
PY
)"

{
  echo "START $(date '+%F %T')"
  echo "CONFIG ${CONFIG}"
  echo "CUDA_VISIBLE_DEVICES ${CUDA_VISIBLE_DEVICES}"
  echo "RUN_DIR ${RUN_DIR}"
  echo "PIPELINE_STAGE_START ${PIPELINE_STAGE_START}"
  echo "RESUME_CHECKPOINT ${RESUME_CHECKPOINT:-NONE}"
  echo "ADAPT_ON_ASAP ${ADAPT_ON_ASAP}"
  echo "ADAPT_LR ${ADAPT_LR}"
  echo "ADAPT_EPOCHS ${ADAPT_EPOCHS}"
} | tee -a "${EVALUATE_LOG}"

TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR_OVERRIDE:-}"
CHECKPOINT="${CHECKPOINT_OVERRIDE:-}"

if [[ "${PIPELINE_STAGE_START}" == "train" ]]; then
  TRAIN_MARKER="${TMP_DIR}/train_start.marker"
  touch "${TRAIN_MARKER}"
  echo "[$(date '+%F %T')] train: CSR start, GPU=${CUDA_VISIBLE_DEVICES}" | tee -a "${EVALUATE_LOG}"
  PYTHONPATH="${ROOT_DIR}" PYTHONUNBUFFERED=1 \
    python src/train/train_inr.py --config "${RUN_CONFIG}" 2>&1 | tee -a "${TRAIN_LOG}"
  echo "[$(date '+%F %T')] train: finished" | tee -a "${EVALUATE_LOG}"

  TRAIN_OUTPUT_DIR="$(
    find "${TRAIN_ROOT}" -maxdepth 1 -mindepth 1 -type d -name 'inr_*' -newer "${TRAIN_MARKER}" \
      -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-
  )"
  [[ -n "${TRAIN_OUTPUT_DIR}" ]] || { echo "Could not locate CSR training output under ${TRAIN_ROOT}" >&2; exit 1; }
  CHECKPOINT="${TRAIN_OUTPUT_DIR}"
  [[ -d "${TRAIN_OUTPUT_DIR}/checkpoint-best" ]] && CHECKPOINT="${TRAIN_OUTPUT_DIR}/checkpoint-best"
elif [[ "${PIPELINE_STAGE_START}" == "adapt" || "${PIPELINE_STAGE_START}" == "infer" ]]; then
  [[ -n "${TRAIN_OUTPUT_DIR}" ]] || TRAIN_OUTPUT_DIR="$(find_latest_run_dir "${TRAIN_ROOT}")"
  [[ -n "${TRAIN_OUTPUT_DIR}" ]] || { echo "Could not locate existing CSR training output under ${TRAIN_ROOT}" >&2; exit 1; }
  if [[ -z "${CHECKPOINT}" ]]; then
    CHECKPOINT="${TRAIN_OUTPUT_DIR}"
    [[ -d "${TRAIN_OUTPUT_DIR}/checkpoint-best" ]] && CHECKPOINT="${TRAIN_OUTPUT_DIR}/checkpoint-best"
  fi
else
  echo "Unsupported PIPELINE_STAGE_START=${PIPELINE_STAGE_START}; expected train, adapt, or infer" >&2
  exit 1
fi

{
  echo "TRAIN_OUTPUT_DIR ${TRAIN_OUTPUT_DIR}"
  echo "CHECKPOINT ${CHECKPOINT}"
} | tee -a "${EVALUATE_LOG}"

ACTIVE_CONFIG="${RUN_CONFIG}"
if [[ "${PIPELINE_STAGE_START}" == "infer" && "${ADAPT_ON_ASAP}" == "1" ]]; then
  [[ -f "${ADAPT_CONFIG}" ]] || { echo "Missing adapted config for infer stage: ${ADAPT_CONFIG}" >&2; exit 1; }
  ADAPT_OUTPUT_DIR="${ADAPT_OUTPUT_DIR_OVERRIDE:-}"
  [[ -n "${ADAPT_OUTPUT_DIR}" ]] || ADAPT_OUTPUT_DIR="$(find_latest_run_dir "${ADAPT_TRAIN_ROOT}")"
  [[ -n "${ADAPT_OUTPUT_DIR}" ]] || { echo "Could not locate adapted CSR output under ${ADAPT_TRAIN_ROOT}" >&2; exit 1; }
  CHECKPOINT="${ADAPT_OUTPUT_DIR}"
  [[ -d "${ADAPT_OUTPUT_DIR}/checkpoint-best" ]] && CHECKPOINT="${ADAPT_OUTPUT_DIR}/checkpoint-best"
  ACTIVE_CONFIG="${ADAPT_CONFIG}"
elif [[ "${ADAPT_ON_ASAP}" == "1" ]]; then
  python - "${RUN_CONFIG}" "${ADAPT_CONFIG}" "${ADAPT_TRAIN_ROOT}" "${ADAPT_TF_LOG_ROOT}" "${CHECKPOINT}" "${ADAPT_LR}" "${ADAPT_EPOCHS}" <<'PY'
import json
import sys

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
cfg["save_steps"] = min(int(cfg.get("save_steps", 2000)), 500)
cfg["eval_steps"] = min(int(cfg.get("eval_steps", 2000)), 500)
cfg["logging_steps"] = int(cfg.get("logging_steps", 20))
cfg["save_total_limit"] = 2
cfg["load_best_model_at_end"] = True
cfg["metric_for_best_model"] = "eval_loss"
cfg["greater_is_better"] = False
open(dst, "w", encoding="utf-8").write(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
PY

  ADAPT_MARKER="${TMP_DIR}/adapt_start.marker"
  touch "${ADAPT_MARKER}"
  echo "[$(date '+%F %T')] adapt: CSR ASAP start, GPU=${CUDA_VISIBLE_DEVICES}" | tee -a "${EVALUATE_LOG}"
  PYTHONPATH="${ROOT_DIR}" PYTHONUNBUFFERED=1 \
    python src/train/train_inr.py --config "${ADAPT_CONFIG}" 2>&1 | tee -a "${TRAIN_LOG}"
  echo "[$(date '+%F %T')] adapt: finished" | tee -a "${EVALUATE_LOG}"

  ADAPT_OUTPUT_DIR="$(
    find "${ADAPT_TRAIN_ROOT}" -maxdepth 1 -mindepth 1 -type d -name 'inr_*' -newer "${ADAPT_MARKER}" \
      -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-
  )"
  [[ -n "${ADAPT_OUTPUT_DIR}" ]] || { echo "Could not locate adapted CSR training output under ${ADAPT_TRAIN_ROOT}" >&2; exit 1; }
  CHECKPOINT="${ADAPT_OUTPUT_DIR}"
  [[ -d "${ADAPT_OUTPUT_DIR}/checkpoint-best" ]] && CHECKPOINT="${ADAPT_OUTPUT_DIR}/checkpoint-best"
  ACTIVE_CONFIG="${ADAPT_CONFIG}"
  {
    echo "ADAPT_OUTPUT_DIR ${ADAPT_OUTPUT_DIR}"
    echo "ADAPTED_CHECKPOINT ${CHECKPOINT}"
  } | tee -a "${EVALUATE_LOG}"
fi

echo "[$(date '+%F %T')] inference: CSR start, GPU=${CUDA_VISIBLE_DEVICES}" | tee -a "${EVALUATE_LOG}"
PYTHONPATH="${ROOT_DIR}" PYTHONUNBUFFERED=1 \
  python src/inference/infer_inr_testset.py \
    --config "${ACTIVE_CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --split test \
    --performance-dataset ASAP \
    --protocol deterministic \
    --num-samples 1 \
    --output-dir "${INFER_DIR}" \
    --device cuda \
    --num-workers "${INFER_NUM_WORKERS}" \
    --batch-size-windows "${INFER_BATCH_SIZE_WINDOWS}" \
    --merge-mode "${MERGE_MODE}" \
    --continuation-drop-ratio "${CONTINUATION_DROP_RATIO}" \
  2>&1 | tee -a "${EVALUATE_LOG}"
echo "[$(date '+%F %T')] inference: finished" | tee -a "${EVALUATE_LOG}"

rm -rf "${TMP_DIR}"
echo "END $(date '+%F %T')" | tee -a "${EVALUATE_LOG}"
