#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/floorlog_slot6_musical_2gpu/${STAMP}}"
CONFIG_DIR="${RUN_ROOT}/configs"
BASE_CONFIG="${BASE_CONFIG:-results/floorlog_dlm_2gpu/20260712_asap_test/k8-b256-veldlm/config.json}"
mkdir -p "${CONFIG_DIR}"

python - "${BASE_CONFIG}" "${CONFIG_DIR}" <<'PY'
import json
import sys
from pathlib import Path

base_path = Path(sys.argv[1])
config_dir = Path(sys.argv[2])
base = json.loads(base_path.read_text(encoding="utf-8"))

common = dict(base)
common.update(
    {
        "note_embedding_mode": "slot_attribute",
        "slot_version": "slot6",
        "slot_dim": 128,
        "slot_gates": False,
        "slot_share_role_encoders": True,
        "slot_decoder_mask_mode": "whole_token",
        "musical_feature_mode": "musical51",
        "disable_musical_features": False,
        "continuous_dim": 5,
        "input_continuous_dim": 62,
        "score_input_continuous_dim": 62,
        "decoder_input_continuous_dim": 62,
        "output_continuous_dim": 5,
        "pedal_representation": "start_valley",
        "pedal_valley_pos_weight": 28.0,
        "epr_distribution": "dlm",
        "velocity_distribution": "dlm",
        "epr_timing_target": "floor_log_deviation",
        "timing_control_mode": "floor_log",
        "eval_gt_time_normalization": "score_onset_span",
        "epr_mixture_components": 8,
        "dlm_components": 8,
        "dlm_timing_bins": 256,
        "dlm_velocity_bins": 128,
        "dlm_ioi_zero_min": 0.0,
        "dlm_ioi_zero_max": 5.0,
        "dlm_ioi_nonzero_min": -2.5,
        "dlm_ioi_nonzero_max": 1.5,
        "dlm_duration_min": -3.0,
        "dlm_duration_max": 2.0,
        "dlm_velocity_min": -0.5,
        "dlm_velocity_max": 127.5,
        "dlm_scale_min": 1e-3,
        "dlm_scale_max": 10.0,
        "legacy_dual_timing_head": False,
        "raw_timing_head_type": "none",
        "raw_timing_loss_lambda": 0.0,
        "zero_ioi_transform": "none",
        "zero_ioi_positive_support": False,
        "zero_ioi_dual_distribution_mode": "none",
        "zero_score_ioi_embedding": True,
        "zero_timing_head_condition": True,
        "num_train_epochs": 16.0,
        "max_train_epochs": 16.0,
        "global_batch_size": 64,
        "train_performance_dataset": "ASAP",
        "eval_performance_dataset": "ASAP",
        "eval_split": "valid",
        "prepared_sidecar_tag": "ASAP_FLOORLOG_NOMUS_SCORESPAN",
        "use_prepared_sidecar": True,
        "auto_rollout_eval_after_train": False,
        "adapt_on_asap_after_train": False,
        "ddp_find_unused_parameters": True,
        "resume_trainer_state": False,
    }
)
for key in ("resume_path", "raw_timing_head"):
    common.pop(key, None)

variants = {
    "slot6-musical-mlp": {
        "run_name": "floorlog_slot6_musical_mlp_veldlm",
        "slot_fusion": "mlp",
    },
    "slot6-musical-direct": {
        "run_name": "floorlog_slot6_musical_direct_veldlm",
        "slot_fusion": "direct_concat",
    },
}

for name, overrides in variants.items():
    cfg = dict(common)
    cfg.update(overrides)
    cfg["output_dir"] = f"unused/{name}/training"
    cfg["logging_dir"] = f"unused/{name}/tf-logs"
    (config_dir / f"{name}.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

allowed = {
    "run_name",
    "slot_fusion",
    "output_dir",
    "logging_dir",
}
loaded = {
    name: json.loads((config_dir / f"{name}.json").read_text(encoding="utf-8"))
    for name in variants
}
unexpected = []
for key in sorted(set().union(*(cfg.keys() for cfg in loaded.values()))):
    if len({json.dumps(cfg.get(key), sort_keys=True) for cfg in loaded.values()}) > 1 and key not in allowed:
        unexpected.append(key)
if unexpected:
    raise SystemExit(f"Unexpected cross-experiment differences: {unexpected}")

report = {
    "base_config": str(base_path),
    "purpose": "Floorlog score-span slot6 musical comparison from k8-b256-veldlm",
    "common": {
        "slot_version": common["slot_version"],
        "slot_dim": common["slot_dim"],
        "slot_count": 6,
        "hidden_size": common["hidden_size"],
        "musical_feature_mode": common["musical_feature_mode"],
        "input_continuous_dim": common["input_continuous_dim"],
        "prepared_sidecar_tag": common["prepared_sidecar_tag"],
        "timing_control_mode": common["timing_control_mode"],
        "epr_timing_target": common["epr_timing_target"],
        "eval_gt_time_normalization": common["eval_gt_time_normalization"],
        "velocity_distribution": common["velocity_distribution"],
        "epochs": common["num_train_epochs"],
        "global_batch_size": common["global_batch_size"],
    },
    "allowed_differences": sorted(allowed),
    "variants": variants,
}
(config_dir / "config_report.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print(json.dumps(report, indent=2, ensure_ascii=False))
PY

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1; configs generated under ${CONFIG_DIR}"
  exit 0
fi

launch_one() {
  local gpus="$1"
  local name="$2"
  local session="floorlog_slot6_${name//[^A-Za-z0-9_]/_}_${STAMP}"
  local run_dir="${RUN_ROOT}/${name}"
  local log_path="${run_dir}/launcher.log"
  mkdir -p "${run_dir}"

  tmux new-session -d -s "${session}" \
    "cd '${ROOT_DIR}' && \
     env CUDA_VISIBLE_DEVICES='${gpus}' \
       CONFIG='${CONFIG_DIR}/${name}.json' \
       RUN_DIR_OVERRIDE='${run_dir}' \
       BASE_ASAP_ONLY=1 \
       BASE_NUM_TRAIN_EPOCHS=16 \
       ADAPT_NUM_TRAIN_EPOCHS=0 \
       BATCH_SIZE_PER_DEVICE=32 \
       GLOBAL_BATCH_SIZE=64 \
       DET_NUM_SAMPLES=1 \
       SAMPLING_NUM_SAMPLES=1 \
       INFER_NUM_WORKERS=8 \
       METRIC_NUM_WORKERS=8 \
       INFER_BATCH_SIZE_WINDOWS=8 \
       INFER_SCORE_SOURCE_LIST='data/asap_test_score_sources.txt' \
       EVAL_CHECKPOINT_MODE=latest \
       RESUME_FROM_LATEST_CHECKPOINT=1 \
       MERGE_MODE=continuation \
       CONTINUATION_DROP_RATIO=0.0 \
       bash script/run_inr_epr_pipeline.sh 2>&1 | tee '${log_path}'"

  printf '%s\tGPU%s\t%s\t%s\n' "${session}" "${gpus}" "${run_dir}" "${log_path}" \
    | tee -a "${RUN_ROOT}/sessions.tsv"
}

launch_one "0,1" "slot6-musical-mlp"
launch_one "2,3" "slot6-musical-direct"

echo "RUN_ROOT=${RUN_ROOT}"
