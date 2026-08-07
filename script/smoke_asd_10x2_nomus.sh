#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GPU_ID="${1:-0}"
FAMILY="${2:-cinr}"
STAMP="$(date '+%Y%m%d_%H%M%S')"
SMOKE_ROOT="results/smoke_asd_10x2_nomus/${STAMP}_${FAMILY}"
CONFIG_PATH="${SMOKE_ROOT}/config.json"
TRAIN_ROOT="${SMOKE_ROOT}/training"
LOG_ROOT="${SMOKE_ROOT}/tf-logs"
INFER_ROOT="${SMOKE_ROOT}/infer_sampling"
BASE_CONFIG="configs/inr_epr/cinr__full4_10x2_nomus.json"

if [[ "${FAMILY}" == "dinr" ]]; then
  BASE_CONFIG="configs/inr_epr/dinr__full4_10x2_nomus.json"
elif [[ "${FAMILY}" != "cinr" ]]; then
  echo "FAMILY must be cinr or dinr, got ${FAMILY}" >&2
  exit 2
fi

mkdir -p "${SMOKE_ROOT}" "${TRAIN_ROOT}" "${LOG_ROOT}" "${INFER_ROOT}"

python - "${BASE_CONFIG}" "${CONFIG_PATH}" "${TRAIN_ROOT}" "${LOG_ROOT}" "${FAMILY}" <<'PY'
import json
import sys
from pathlib import Path

src, dst, train_root, log_root, family = sys.argv[1:6]
cfg = json.loads(Path(src).read_text(encoding="utf-8"))
cfg.update(
    {
        "run_name": f"smoke_asd_10x2_nomus_{family}",
        "output_dir": train_root,
        "logging_dir": log_root,
        "prepared_sidecar_tag": "ASAP_MUSICAL51",
        "eval_from_train_fraction": 0.5,
        "max_steps": 1,
        "num_train_epochs": 1.0,
        "max_train_epochs": 1.0,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "global_batch_size": 1,
        "max_train_works": 1,
        "max_eval_works": 1,
        "max_performances_per_work": 1,
        "max_eval_performances_per_work": 1,
        "max_windows_per_work": 1,
        "max_eval_windows_per_work": 1,
        "fixed_window_split_scheme": None,
        "fixed_window_base_split": None,
        "fixed_window_train_split_name": None,
        "fixed_window_eval_split_name": None,
        "fixed_window_split_summary_path": None,
        "dataloader_num_workers": 0,
        "eval_dataloader_num_workers": 0,
        "dataloader_persistent_workers": False,
        "eval_dataloader_persistent_workers": False,
        "save_total_limit": 1,
        "auto_rollout_eval_after_train": False,
        "report_to": "none",
    }
)
Path(dst).write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY

echo "[smoke] train ${FAMILY} ASD on GPU ${GPU_ID}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python src/train/train_inr.py --config "${CONFIG_PATH}" --max_steps 1

TRAIN_DIR="${TRAIN_ROOT}/smoke_asd_10x2_nomus_${FAMILY}"
CHECKPOINT="${TRAIN_DIR}/checkpoint-best"
if [[ ! -d "${CHECKPOINT}" ]]; then
  CHECKPOINT="$(find "${TRAIN_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' | sort | tail -n 1)"
fi
if [[ -z "${CHECKPOINT}" || ! -d "${CHECKPOINT}" ]]; then
  echo "[smoke] could not locate checkpoint under ${TRAIN_DIR}" >&2
  exit 1
fi

echo "[smoke] infer ${FAMILY} ASD sampling from ${CHECKPOINT}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONUNBUFFERED=1 \
  python src/inference/infer_inr_testset.py \
    --config "${CONFIG_PATH}" \
    --checkpoint "${CHECKPOINT}" \
    --split test \
    --performance-dataset ASAP \
    --max-works 1 \
    --batch-size-windows 1 \
    --num-workers 1 \
    --device cuda \
    --protocol sampling \
    --sampling-strategy sample \
    --num-samples 1 \
    --output-dir "${INFER_ROOT}"

test -s "${INFER_ROOT}/prediction_manifest.json"
test -s "${INFER_ROOT}/evaluate_list.json"
echo "[smoke] success: ${SMOKE_ROOT}"
