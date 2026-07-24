#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BASE_CONFIG="${BASE_CONFIG:-results/inr_epr_pipeline/asap_processed_musical51_nolossnorm16_slot_effective_20260723/configs/extra__rep_slot_sum.json}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
ARCH_SPLIT="${ARCH_SPLIT:-10_2}"
if [[ "${ARCH_SPLIT}" != "10_2" && "${ARCH_SPLIT}" != "9_3" ]]; then
  echo "ARCH_SPLIT must be 10_2 or 9_3, got ${ARCH_SPLIT}" >&2
  exit 1
fi
ENCODER_LAYERS="${ARCH_SPLIT%_*}"
DECODER_LAYERS="${ARCH_SPLIT#*_}"
RUN_ROOT="${RUN_ROOT:-results/inr_epr_pipeline/asap_processed_musical51_cinr_te2n_rep_slot_sum_${ARCH_SPLIT}_${STAMP}}"
PYTHON_BIN="${PYTHON_BIN:-/home/kaititech/anaconda3/bin/python}"
GPUS="${GPUS:-0,1}"
NPROC="${NPROC:-2}"
MASTER_PORT="${MASTER_PORT:-29621}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"

RUN_DIR="${RUN_ROOT}/te2n_rep_slot_sum_${ARCH_SPLIT}"
mkdir -p "${RUN_DIR}"

"${PYTHON_BIN}" - "${BASE_CONFIG}" "${RUN_DIR}" "${PER_DEVICE_BATCH}" "${GRAD_ACCUM}" "${NPROC}" "${ENCODER_LAYERS}" "${DECODER_LAYERS}" "${ARCH_SPLIT}" <<'PY'
import json
import sys
from pathlib import Path

base_path, run_dir, per_device_batch, grad_accum, nproc, encoder_layers, decoder_layers, arch_split = sys.argv[1:]
run_dir = Path(run_dir)
cfg = json.loads(Path(base_path).read_text(encoding="utf-8"))
cfg.update({
    "run_name": f"asap_musical51_cinr_te2n_rep_slot_sum_{arch_split}",
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
    "encoder_layers_num": int(encoder_layers),
    "decoder_layers_num": int(decoder_layers),
    "per_device_train_batch_size": int(per_device_batch),
    "per_device_eval_batch_size": int(per_device_batch),
    "gradient_accumulation_steps": int(grad_accum),
    "global_batch_size": int(per_device_batch) * int(grad_accum) * int(nproc),
})
run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / "config.json").write_text(
    json.dumps(cfg, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
PY

SESSION="cinr_te2n_rep_slot_sum_${ARCH_SPLIT}_${STAMP}"
tmux new-session -d -s "${SESSION}" \
  "cd '${ROOT_DIR}' && CUDA_VISIBLE_DEVICES='${GPUS}' PYTHONUNBUFFERED=1 '${PYTHON_BIN}' -m torch.distributed.run \
     --nproc_per_node='${NPROC}' --master_port='${MASTER_PORT}' \
     src/train/train_inr.py --config '${RUN_DIR}/config.json' > '${RUN_DIR}/train.log' 2>&1"

printf '%s\tARCH=%s\tGPUS=%s\t%s\n' "${SESSION}" "${ARCH_SPLIT}" "${GPUS}" "${RUN_DIR}" | tee "${RUN_ROOT}/sessions.tsv"
echo "RUN_ROOT=${RUN_ROOT}"
