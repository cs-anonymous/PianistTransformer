#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BASE_CONFIG="${BASE_CONFIG:-results/inr_epr_pipeline/asap_processed_musical51_nolossnorm16_slot_effective_20260723/configs/extra__rep_slot_sum.json}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/inr_epr_pipeline/asap_processed_musical51_cinr_te2n_rep_slot_sum_10_2_1gpu_${STAMP}}"
PYTHON_BIN="${PYTHON_BIN:-/home/kaititech/anaconda3/bin/python}"
GPU="${GPU:-0}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"

RUN_DIR="${RUN_ROOT}/te2n_rep_slot_sum_10_2"
mkdir -p "${RUN_DIR}"

"${PYTHON_BIN}" - "${BASE_CONFIG}" "${RUN_DIR}" "${PER_DEVICE_BATCH}" "${GRAD_ACCUM}" <<'PY'
import json
import sys
from pathlib import Path

base_path, run_dir, per_device_batch, grad_accum = sys.argv[1:]
run_dir = Path(run_dir)
cfg = json.loads(Path(base_path).read_text(encoding="utf-8"))
cfg.update({
    "run_name": "asap_musical51_cinr_te2n_rep_slot_sum_10_2_1gpu",
    "output_dir": str(run_dir / "training"),
    "logging_dir": str(run_dir / "tf-logs"),
    "overwrite_output_dir": True,
    "resume_path": None,
    "resume_trainer_state": False,
    "note_embedding_mode": "slot_attribute",
    "slot_fusion": "sum",
    "slot_dim": 768,
    "decoder_task_token_mode": "timing_expression_2n",
    "output_task_embedding_mode": "none",
    "encoder_layers_num": 10,
    "decoder_layers_num": 2,
    "per_device_train_batch_size": int(per_device_batch),
    "per_device_eval_batch_size": int(per_device_batch),
    "gradient_accumulation_steps": int(grad_accum),
})
run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / "config.json").write_text(
    json.dumps(cfg, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
PY

echo "RUN_DIR=${RUN_DIR}"
CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 "${PYTHON_BIN}" \
  src/train/train_inr.py --config "${RUN_DIR}/config.json" 2>&1 | tee "${RUN_DIR}/train.log"
