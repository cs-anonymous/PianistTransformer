#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BASE_CONFIG="${BASE_CONFIG:-results/inr_epr_pipeline/asap_processed_musical51_nolossnorm16_slot_effective_20260723/configs/cinr__default_dlm_k1_bounded5.json}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/inr_epr_pipeline/asap_processed_musical51_cinr_velocity_w1_objectives_${STAMP}}"
SMOKE="${SMOKE:-0}"
PN_LAMBDA="${PN_LAMBDA:-1.0}"
PP_LAMBDA="${PP_LAMBDA:-1.0}"
GROUP_SIZE="${GROUP_SIZE:-3}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-10}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
PYTHON_BIN="${PYTHON_BIN:-/home/kaititech/anaconda3/bin/python}"
mkdir -p "${RUN_ROOT}"

write_config() {
  local name="$1"
  "${PYTHON_BIN}" - "${BASE_CONFIG}" "${RUN_ROOT}" "${name}" "${SMOKE}" \
    "${PN_LAMBDA}" "${PP_LAMBDA}" "${GROUP_SIZE}" "${PER_DEVICE_BATCH}" "${GRAD_ACCUM}" <<'PY'
import json
import sys
from pathlib import Path

base_path, run_root, name, smoke, pn_lambda, pp_lambda, group_size, per_device_batch, grad_accum = sys.argv[1:]
cfg = json.loads(Path(base_path).read_text(encoding="utf-8"))
run_dir = Path(run_root) / name
pn = float(pn_lambda) if name in {"pn", "pn_pp"} else 0.0
pp = float(pp_lambda) if name in {"pp", "pn_pp"} else 0.0
cfg.update({
    "run_name": f"asap_musical51_cinr_velocity_w1_{name}",
    "output_dir": str(run_dir / "training"),
    "logging_dir": str(run_dir / "tf-logs"),
    "overwrite_output_dir": True,
    "resume_path": None,
    "resume_trainer_state": False,
    "multi_perf_group_size": int(group_size),
    "multi_perf_min_group_size": min(3, int(group_size)),
    "per_device_train_batch_size": int(per_device_batch),
    "gradient_accumulation_steps": int(grad_accum),
    "pn_mean_loss_lambda": 0.0,
    "pn_var_ioi_zero_lambda": 0.0,
    "pn_var_ioi_nonzero_lambda": 0.0,
    "pn_var_duration_lambda": 0.0,
    "pn_var_velocity_lambda": 0.0,
    "pn_velocity_w1_lambda": pn,
    "pp_velocity_w1_lambda": pp,
    "velocity_w1_normalizer": 127.0,
})
if smoke == "1":
    cfg.update({
        "max_steps": 1,
        "num_train_epochs": 1.0,
        "max_train_epochs": 1.0,
        "eval_strategy": "no",
        "save_strategy": "no",
        "logging_steps": 1,
        "dataloader_num_workers": 0,
        "dataloader_persistent_workers": False,
        "precompute_dataset_items": False,
    })
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
  local session="cinr_vw1_${name}_${STAMP}"
  local run_dir="${RUN_ROOT}/${name}"
  write_config "${name}"
  tmux new-session -d -s "${session}" \
    "cd '${ROOT_DIR}' && CUDA_VISIBLE_DEVICES='${gpu}' PYTHONUNBUFFERED=1 '${PYTHON_BIN}' \
       src/train/train_inr.py --config '${run_dir}/config.json' > '${run_dir}/train.log' 2>&1"
  printf '%s\tGPU%s\t%s\n' "${session}" "${gpu}" "${run_dir}" | tee -a "${RUN_ROOT}/sessions.tsv"
}

launch_one 0 pn
launch_one 1 pp
launch_one 2 pn_pp
echo "RUN_ROOT=${RUN_ROOT}"
