#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/slot8_fixed_vs_sine_2gpu/${STAMP}}"
CONFIG_DIR="${RUN_ROOT}/configs"
mkdir -p "${CONFIG_DIR}"

BASE_CONFIG="backup/result0708/inr_epr_pipeline/launch_rawlog_3exp_20260708_235220/exp2_sine_nomus_tfmask50/config.json"

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
        "metadata_path": "/home/sy/EPR/PianoCoRe/metadata.csv",
        "refined_dir": "/home/sy/EPR/PianoCoRe/processed",
        "continuous_dim": 9,
        "input_continuous_dim": 68,
        "output_continuous_dim": 9,
        "musical_feature_mode": "musical51",
        "disable_musical_features": True,
        "epr_distribution": "skew_normal",
        "epr_timing_target": "raw_log_deviation",
        "timing_control_mode": "raw_log",
        "legacy_dual_timing_head": True,
        "raw_timing_head_type": "regression",
        "raw_timing_loss_lambda": 0.25,
        "tf_embedding_mask_keep_prob": 0.5,
        "tf_embedding_mask_score": False,
        "tf_embedding_mask_decoder": True,
        "prior_token_keep_prob": 1.0,
        "prior_token_dropout_mode": "mask",
        "loss_weights": {
            "ioi": 1.0,
            "duration": 1.0,
            "velocity": 1.0,
            "pedal": 0.5,
        },
        "num_train_epochs": 16.0,
        "max_train_epochs": 16.0,
        "per_device_train_batch_size": 32,
        "per_device_eval_batch_size": 32,
        "gradient_accumulation_steps": 1,
        "global_batch_size": 64,
        "learning_rate": 3e-4,
        "warmup_steps": 50,
        "warmup_ratio": 0,
        "logging_steps": 20,
        "seed": 42,
        "train_performance_dataset": "ASAP",
        "eval_performance_dataset": "ASAP",
        "eval_split": "valid",
        "prepared_sidecar_tag": "ASAP",
        "use_prepared_sidecar": True,
        "fixed_window_split_scheme": "train_valid_asap3_nonasap05_v1",
        "fixed_window_base_split": "train",
        "fixed_window_train_split_name": "train",
        "fixed_window_eval_split_name": "valid",
        "fixed_window_split_summary_path": "data/train_valid_asap3_nonasap05_v1_summary.json",
        "early_stopping_patience": 0,
        "auto_rollout_eval_after_train": False,
        "adapt_on_asap_after_train": False,
        "ddp_find_unused_parameters": True,
        "resume_trainer_state": False,
    }
)
for key in (
    "prior_property_dropout_prob",
    "prior_attribute_keep_probs",
    "raw_timing_head",
    "resume_path",
):
    common.pop(key, None)

variants = {
    "sine-control": {
        "note_embedding_mode": "sine",
        "run_name": "slot8fix_sine_control",
    },
    "slot8-fixed": {
        "note_embedding_mode": "slot_attribute",
        "slot_version": "slot8",
        "slot_dim": 128,
        "slot_fusion": "mlp",
        "slot_gates": False,
        "slot_share_role_encoders": True,
        "run_name": "slot8fix_slot8_fixed",
    },
}

for name, overrides in variants.items():
    cfg = dict(common)
    cfg.update(overrides)
    cfg["output_dir"] = f"unused/{name}/training"
    cfg["logging_dir"] = f"unused/{name}/tf-logs"
    path = config_dir / f"{name}.json"
    path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

report = {
    "baseline": str(base_path),
    "purpose": "sine control vs slot8 with role encoder sharing, decoder property PAD mask, score always visible, raw regression aux",
    "common": {
        "epochs": common["num_train_epochs"],
        "per_device_batch_size": common["per_device_train_batch_size"],
        "global_batch_size": common["global_batch_size"],
        "tf_embedding_mask_score": common["tf_embedding_mask_score"],
        "tf_embedding_mask_decoder": common["tf_embedding_mask_decoder"],
        "tf_embedding_mask_keep_prob": common["tf_embedding_mask_keep_prob"],
        "raw_timing_head_type": common["raw_timing_head_type"],
        "raw_timing_loss_lambda": common["raw_timing_loss_lambda"],
        "train_performance_dataset": common["train_performance_dataset"],
    },
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
  local session="slot8fix_${name//[^A-Za-z0-9_]/_}_${STAMP}"
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
       INFER_SCORE_SOURCE_LIST='data/cheap15_score_sources.txt' \
       EVAL_CHECKPOINT_MODE=latest \
       RESUME_FROM_LATEST_CHECKPOINT=1 \
       MERGE_MODE=continuation \
       CONTINUATION_DROP_RATIO=0.0 \
       bash script/run_inr_epr_pipeline.sh 2>&1 | tee '${log_path}'"

  printf '%s\tGPU%s\t%s\t%s\n' "${session}" "${gpus}" "${run_dir}" "${log_path}" \
    | tee -a "${RUN_ROOT}/sessions.tsv"
}

launch_one "0,1" "sine-control"
launch_one "2,3" "slot8-fixed"

echo "RUN_ROOT=${RUN_ROOT}"
