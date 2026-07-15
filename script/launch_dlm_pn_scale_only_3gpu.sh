#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BASE_CONFIG="${BASE_CONFIG:-results/dlm_pn_group_ablation_3gpu/20260715_175947/B-group-mean/config.json}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-results/dlm_pn_group_ablation_3gpu/20260715_175947/B-group-mean/training/dlm_pn_group_B-group-mean/checkpoint-best}"
SPLIT_SUMMARY="${SPLIT_SUMMARY:-results/dlm_pn_group_ablation_3gpu/20260715_175947/train_valid_asap3_nonasap05_v1_current_summary.json}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/dlm_pn_scale_only_3gpu/${STAMP}}"
SMOKE="${SMOKE:-0}"
mkdir -p "${RUN_ROOT}"

write_config() {
  local name="$1" lambda="$2"
  python - "${BASE_CONFIG}" "${STAGE1_CHECKPOINT}" "${SPLIT_SUMMARY}" "${RUN_ROOT}" "${name}" "${lambda}" "${SMOKE}" <<'PY'
import json, sys
from pathlib import Path

base_path, checkpoint, split_summary, run_root, name, lam, smoke = sys.argv[1:]
run_dir = Path(run_root) / name
cfg = json.loads(Path(base_path).read_text(encoding="utf-8"))
value = float(lam)
cfg.update({
    "run_name": f"dlm_scale_only_{name}",
    "output_dir": str(run_dir / "training"),
    "logging_dir": str(run_dir / "tf-logs"),
    "overwrite_output_dir": True,
    "resume_path": str(Path(checkpoint).resolve()),
    "resume_trainer_state": False,
    "reset_output_heads_on_resume": False,
    "train_scale_only": True,
    "freeze_non_output_heads": False,
    "trainable_parameter_regex": None,
    "weight_decay": 0.0,
    "learning_rate": 3e-5,
    "num_train_epochs": 3.0,
    "max_train_epochs": 3.0,
    "multi_perf_group_size": 4,
    "multi_perf_min_group_size": 3,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "fixed_window_split_summary_path": str(Path(split_summary).resolve()),
    "pn_mean_loss_lambda": 0.0,
    "pn_var_ioi_zero_lambda": value,
    "pn_var_ioi_nonzero_lambda": value,
    "pn_var_duration_lambda": value,
    "pn_var_velocity_lambda": value,
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
    json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
PY
}

launch_one() {
  local gpu="$1" name="$2" lambda="$3"
  local session="scale_${name//[^A-Za-z0-9]/_}_${STAMP}"
  local run_dir="${RUN_ROOT}/${name}"
  write_config "${name}" "${lambda}"
  tmux new-session -d -s "${session}" \
    "cd '${ROOT_DIR}' && CUDA_VISIBLE_DEVICES='${gpu}' PYTHONUNBUFFERED=1 \
     /home/kaititech/anaconda3/bin/python src/train/train_inr.py \
       --config '${run_dir}/config.json' > '${run_dir}/train.log' 2>&1"
  printf '%s\tGPU%s\tlambda=%s\t%s\n' "${session}" "${gpu}" "${lambda}" "${run_dir}" | tee -a "${RUN_ROOT}/sessions.tsv"
}

launch_one 0 lambda-0p5 0.5
launch_one 1 lambda-1p0 1.0
launch_one 2 lambda-2p0 2.0
echo "RUN_ROOT=${RUN_ROOT}"
