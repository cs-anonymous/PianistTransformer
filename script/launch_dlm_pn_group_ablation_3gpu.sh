#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BASE_CONFIG="${BASE_CONFIG:-results/slot5_nomus_simple_distributions_3gpu/20260714_220814/dlm-k1-free-scale/config.json}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/dlm_pn_group_ablation_3gpu/${STAMP}}"
SMOKE="${SMOKE:-0}"
mkdir -p "${RUN_ROOT}"
SPLIT_SUMMARY="${RUN_ROOT}/train_valid_asap3_nonasap05_v1_current_summary.json"
if [[ ! -s "${SPLIT_SUMMARY}" ]]; then
  /home/kaititech/anaconda3/bin/python src/data_process/create_fixed_window_valid_split.py \
    --metadata-path data/ASAP_processed/metadata.generated_json.csv \
    --refined-dir data/ASAP_processed \
    --scheme-name train_valid_asap3_nonasap05_v1 \
    --selection-seed 42 \
    --output-summary "${SPLIT_SUMMARY}" \
    --skip-sidecars \
    --workers 24
fi

write_config() {
  local name="$1"
  python - "${BASE_CONFIG}" "${RUN_ROOT}" "${name}" "${SMOKE}" "${SPLIT_SUMMARY}" <<'PY'
import json, sys
from pathlib import Path

base_path, run_root, name, smoke, split_summary = sys.argv[1:]
cfg = json.loads(Path(base_path).read_text(encoding="utf-8"))
run_dir = Path(run_root) / name
cfg.update({
    "run_name": f"dlm_pn_group_{name}",
    "output_dir": str(run_dir / "training"),
    "logging_dir": str(run_dir / "tf-logs"),
    "overwrite_output_dir": True,
    "resume_path": None,
    "resume_trainer_state": False,
    "metadata_path": str(Path("data/ASAP_processed/metadata.generated_json.csv").resolve()),
    "refined_dir": str(Path("data/ASAP_processed").resolve()),
    "musical_feature_mode": "musical4slot",
    "disable_musical_features": False,
    "multi_perf_group_size": 4,
    "multi_perf_min_group_size": 3,
    # Each dataset item expands to 3-4 performances in the collator; bs8 keeps
    # the flattened model batch close to the original bs32 baseline.
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "pn_variance_shrinkage_tau": 4.0,
    "pn_variance_epsilon": 1e-4,
    "fixed_window_split_summary_path": str(Path(split_summary).resolve()),
    "pn_mean_loss_lambda": 0.0,
    "pn_var_ioi_zero_lambda": 0.0,
    "pn_var_ioi_nonzero_lambda": 0.0,
    "pn_var_duration_lambda": 0.0,
    "pn_var_velocity_lambda": 0.0,
})
cfg.pop("prepared_sidecar_tag", None)
if name == "B-group-mean":
    cfg["pn_mean_loss_lambda"] = 0.1
elif name == "C-group-mean-var":
    cfg.update({
        "pn_mean_loss_lambda": 0.1,
        "pn_var_ioi_zero_lambda": 0.05,
        "pn_var_ioi_nonzero_lambda": 0.10,
        "pn_var_duration_lambda": 0.10,
        "pn_var_velocity_lambda": 0.10,
    })
elif name != "A-group-only":
    raise ValueError(name)
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
(run_dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

launch_one() {
  local gpu="$1" name="$2"
  local session="pn_${name//[^A-Za-z0-9]/_}_${STAMP}"
  local run_dir="${RUN_ROOT}/${name}"
  write_config "${name}"
  tmux new-session -d -s "${session}" \
    "cd '${ROOT_DIR}' && CUDA_VISIBLE_DEVICES='${gpu}' PYTHONUNBUFFERED=1 \
     /home/kaititech/anaconda3/bin/python src/train/train_inr.py \
       --config '${run_dir}/config.json' > '${run_dir}/train.log' 2>&1"
  printf '%s\tGPU%s\t%s\n' "${session}" "${gpu}" "${run_dir}" | tee -a "${RUN_ROOT}/sessions.tsv"
}

launch_one 0 A-group-only
launch_one 1 B-group-mean
launch_one 2 C-group-mean-var
echo "RUN_ROOT=${RUN_ROOT}"
