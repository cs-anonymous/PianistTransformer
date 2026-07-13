#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

STAMP="${STAMP:-20260712_floorlog_sn_mln3_trunc}"
RUN_ROOT="${RUN_ROOT:-results/floorlog_sn_mln3_trunc_4gpu/${STAMP}}"
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
        "continuous_dim": 5,
        "input_continuous_dim": 10,
        "score_input_continuous_dim": 10,
        "decoder_input_continuous_dim": 10,
        "output_continuous_dim": 5,
        "pedal_representation": "start_valley",
        "pedal_valley_pos_weight": 28.0,
        "epr_timing_target": "floor_log_deviation",
        "timing_control_mode": "floor_log",
        "eval_gt_time_normalization": "score_onset_span",
        "musical_feature_mode": "none",
        "disable_musical_features": True,
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
        "auto_rollout_eval_after_train": False,
        "adapt_on_asap_after_train": False,
        "timing_sample_truncate_radius": 0.0,
        "timing_sample_truncate_center": "mean",
        "dlm_timing_sample_truncate_radius": 0.0,
        "dlm_timing_sample_truncate_center": "mean",
    }
)
for key in ("resume_path", "raw_timing_head"):
    common.pop(key, None)

variants = {
    "floorlog-skew-normal": {
        "epr_distribution": "skew_normal",
        "velocity_distribution": "skew_normal",
        "epr_mixture_components": 1,
    },
    "floorlog-mln3": {
        "epr_distribution": "mixture_logistic_normal",
        "velocity_distribution": "mixture_logistic_normal",
        "epr_mixture_components": 3,
        "logistic_normal_sigma_min": 1e-3,
        "logistic_normal_sigma_max": 10.0,
    },
}

for name, overrides in variants.items():
    cfg = dict(common)
    cfg.update(overrides)
    cfg["run_name"] = name.replace("-", "_")
    cfg["output_dir"] = f"unused/{name}/training"
    cfg["logging_dir"] = f"unused/{name}/tf-logs"
    (config_dir / f"{name}.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

report = {
    "base_config": str(base_path),
    "purpose": "Floorlog score-span skew-normal vs old mln3 with raw and local-truncated sampling",
    "common": {
        "target": common["epr_timing_target"],
        "timing_control_mode": common["timing_control_mode"],
        "prepared_sidecar_tag": common["prepared_sidecar_tag"],
        "eval_gt_time_normalization": common["eval_gt_time_normalization"],
        "epochs": common["num_train_epochs"],
        "global_batch_size": common["global_batch_size"],
        "musical_feature_mode": common["musical_feature_mode"],
    },
    "variants": variants,
    "inference": {
        "raw": {"timing_sample_truncate_radius": 0.0},
        "trunc-r0p05": {"timing_sample_truncate_radius": 0.05, "timing_sample_truncate_center": "mean"},
        "trunc-r0p10": {"timing_sample_truncate_radius": 0.10, "timing_sample_truncate_center": "mean"},
    },
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
  local session="floorlog_${name//[^A-Za-z0-9_]/_}_${STAMP}"
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
       bash script/run_inr_epr_pipeline.sh 2>&1 | tee '${log_path}' && \
     bash script/run_floorlog_trunc_infer_from_run.sh '${run_dir}' '${CONFIG_DIR}/${name}.json' '${gpus}' '${RUN_ROOT}/trunc_eval/${name}' 2>&1 | tee -a '${log_path}'"

  printf '%s\tGPU%s\t%s\t%s\n' "${session}" "${gpus}" "${run_dir}" "${log_path}" \
    | tee -a "${RUN_ROOT}/sessions.tsv"
}

launch_one "0,1" "floorlog-skew-normal"
launch_one "2,3" "floorlog-mln3"

echo "RUN_ROOT=${RUN_ROOT}"
