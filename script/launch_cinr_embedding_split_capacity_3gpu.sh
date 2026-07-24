#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BASE_CONFIG="${BASE_CONFIG:-results/inr_epr_pipeline/asap_processed_musical51_nolossnorm16_slot_effective_20260723/configs/cinr__default_dlm_k1_bounded5.json}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/inr_epr_pipeline/asap_processed_musical51_cinr_embedding_split_capacity_${STAMP}}"
PYTHON_BIN="${PYTHON_BIN:-/home/kaititech/anaconda3/bin/python}"
mkdir -p "${RUN_ROOT}"

write_config() {
  local name="$1"
  "${PYTHON_BIN}" - "${BASE_CONFIG}" "${RUN_ROOT}" "${name}" <<'PY'
import json
import sys
from pathlib import Path

base_path, run_root, name = sys.argv[1:]
cfg = json.loads(Path(base_path).read_text(encoding="utf-8"))
run_dir = Path(run_root) / name
cfg.update({
    "run_name": f"asap_musical51_cinr_{name}",
    "output_dir": str(run_dir / "training"),
    "logging_dir": str(run_dir / "tf-logs"),
    "overwrite_output_dir": True,
    "resume_path": None,
    "resume_trainer_state": False,
    "output_task_embedding_mode": "none",
    "output_task_adapter_depth": 2,
    "output_task_adapter_width_multiplier": 0.5,
})

if name == "two_embedding":
    cfg["output_task_embedding_mode"] = "timing_expression"
elif name == "four_embedding":
    cfg["output_task_embedding_mode"] = "four"
elif name == "hidden1024":
    cfg["hidden_size"] = 1024
    cfg["intermediate_size"] = 4096
elif name != "baseline_copy":
    raise ValueError(name)

run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / "config.json").write_text(
    json.dumps(cfg, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
PY
}

launch_one() {
  local gpu="$1"
  local name="$2"
  local session="cinr_repr_${name}_${STAMP}"
  local run_dir="${RUN_ROOT}/${name}"
  write_config "${name}"
  tmux new-session -d -s "${session}" \
    "cd '${ROOT_DIR}' && CUDA_VISIBLE_DEVICES='${gpu}' PYTHONUNBUFFERED=1 '${PYTHON_BIN}' \
       src/train/train_inr.py --config '${run_dir}/config.json' > '${run_dir}/train.log' 2>&1"
  printf '%s\tGPU%s\t%s\n' "${session}" "${gpu}" "${run_dir}" | tee -a "${RUN_ROOT}/sessions.tsv"
}

launch_one 0 two_embedding
launch_one 1 four_embedding
launch_one 2 hidden1024
echo "RUN_ROOT=${RUN_ROOT}"
