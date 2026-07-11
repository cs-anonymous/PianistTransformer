#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/slot8_mask_stable_2gpu/${STAMP}}"
CONFIG_DIR="${RUN_ROOT}/configs"
BASE_CONFIG="${BASE_CONFIG:-results/slot8_fixed_vs_sine_2gpu/20260710_slot8fix_2gpu_ddpfind/configs/slot8-fixed.json}"
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
        "slot_version": "slot8",
        "slot_dim": 128,
        "slot_fusion": "mlp",
        "slot_gates": False,
        "slot_share_role_encoders": True,
        "tf_embedding_mask_score": False,
        "tf_embedding_mask_decoder": True,
        "tf_embedding_mask_keep_prob": 0.5,
        "prior_token_keep_prob": 1.0,
        "prior_token_dropout_mode": "mask",
        "dagger_prefix_training": False,
        "num_train_epochs": 16.0,
        "max_train_epochs": 16.0,
        "per_device_train_batch_size": 32,
        "per_device_eval_batch_size": 32,
        "gradient_accumulation_steps": 1,
        "global_batch_size": 64,
        "train_performance_dataset": "ASAP",
        "eval_performance_dataset": "ASAP",
        "eval_split": "valid",
        "prepared_sidecar_tag": "ASAP",
        "use_prepared_sidecar": True,
        "ddp_find_unused_parameters": True,
        "resume_trainer_state": False,
    }
)
for key in ("resume_path", "prior_attribute_keep_probs"):
    common.pop(key, None)

variants = {
    "slot8-whole-token-mask": {
        "run_name": "slot8_whole_token_mask50",
        "slot_decoder_mask_mode": "whole_token",
        "prior_property_dropout_prob": None,
        "stable_dynamics_training": False,
        "stable_contract_loss": False,
        "stable_contract_lambda": 0.0,
    },
    "slot8-stable-dynamic": {
        "run_name": "slot8_property_mask50_stable_noise",
        "slot_decoder_mask_mode": "property",
        "prior_property_dropout_prob": 0.5,
        "stable_dynamics_training": True,
        "stable_apply_prob": 0.30,
        "stable_channels": ["ioi", "duration"],
        "stable_noise_modes": {
            "zero_mean": {
                "prob": 0.50,
                "ioi_mu": 0.0,
                "ioi_sigma": 0.010,
                "duration_mu": 0.0,
                "duration_sigma": 0.010,
            },
            "positive_bias": {
                "prob": 0.25,
                "ioi_mu": 0.003,
                "ioi_sigma": 0.010,
                "duration_mu": 0.003,
                "duration_sigma": 0.010,
            },
            "variance_inflation": {
                "prob": 0.25,
                "ioi_mu": 0.0,
                "ioi_sigma": 0.025,
                "duration_mu": 0.0,
                "duration_sigma": 0.020,
            },
        },
        "stable_contract_loss": False,
        "stable_contract_lambda": 0.0,
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

report = {
    "baseline": str(base_path),
    "common": {
        "epochs": common["num_train_epochs"],
        "per_device_batch_size": common["per_device_train_batch_size"],
        "global_batch_size": common["global_batch_size"],
        "raw_timing_head_type": common.get("raw_timing_head_type"),
        "raw_timing_loss_lambda": common.get("raw_timing_loss_lambda"),
        "train_performance_dataset": common["train_performance_dataset"],
        "sampling_set": "cheap15, one sample per score",
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
  local session="slot8mask_${name//[^A-Za-z0-9_]/_}_${STAMP}"
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

launch_one "0,1" "slot8-whole-token-mask"
launch_one "2,3" "slot8-stable-dynamic"

echo "RUN_ROOT=${RUN_ROOT}"
