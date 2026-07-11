#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/slot5_slot6_zeroioi_4gpu/${STAMP}}"
CONFIG_DIR="${RUN_ROOT}/configs"
SLOT5_BASE_CONFIG="${SLOT5_BASE_CONFIG:-results/slot5_width_2gpu/20260710_slot5width/slot5-128-whole-token-pad/config.json}"
SLOT6_BASE_CONFIG="${SLOT6_BASE_CONFIG:-results/slot6_musical_4gpu/20260711_slot6_musical/configs/slot6-128-mlp.json}"
mkdir -p "${CONFIG_DIR}"

python - "${SLOT5_BASE_CONFIG}" "${SLOT6_BASE_CONFIG}" "${CONFIG_DIR}" <<'PY'
import json
import sys
from pathlib import Path

slot5_base_path = Path(sys.argv[1])
slot6_base_path = Path(sys.argv[2])
config_dir = Path(sys.argv[3])
slot5_base = json.loads(slot5_base_path.read_text(encoding="utf-8"))
slot6_base = json.loads(slot6_base_path.read_text(encoding="utf-8"))

common = {
    "note_embedding_mode": "slot_attribute",
    "slot_dim": 128,
    "slot_fusion": "mlp",
    "slot_gates": False,
    "slot_share_role_encoders": True,
    "tf_embedding_mask_score": False,
    "tf_embedding_mask_decoder": True,
    "tf_embedding_mask_keep_prob": 0.5,
    "slot_decoder_mask_mode": "whole_token",
    "prior_token_keep_prob": 1.0,
    "prior_token_dropout_mode": "mask",
    "prior_property_dropout_prob": None,
    "stable_dynamics_training": False,
    "stable_contract_loss": False,
    "stable_contract_lambda": 0.0,
    "dagger_prefix_training": False,
    "num_train_epochs": 16.0,
    "max_train_epochs": 16.0,
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 32,
    "gradient_accumulation_steps": 2,
    "global_batch_size": 64,
    "train_performance_dataset": "ASAP",
    "eval_performance_dataset": "ASAP",
    "eval_split": "valid",
    "prepared_sidecar_tag": "ASAP",
    "use_prepared_sidecar": True,
    "precompute_dataset_items": False,
    "precompute_eval_dataset_items": False,
    "raw_timing_head_type": "regression",
    "raw_timing_loss_lambda": 0.25,
    "zero_ioi_positive_support": False,
    "zero_ioi_residual": False,
    "zero_ioi_support_eps": 1e-6,
    "resume_trainer_state": False,
}

stable_noise_modes = {
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
}

variants = {
    "slot6-128-mlp-decoder-musical-mask": (
        slot6_base,
        {
            "run_name": "slot6_128_mlp_decoder_musical_mask_pad50",
            "slot_version": "slot6",
        },
    ),
    "slot5-128-zero-ioi-positive": (
        slot5_base,
        {
            "run_name": "slot5_128_zero_ioi_positive_pad50",
            "slot_version": "slot5",
            "zero_ioi_positive_support": True,
        },
    ),
    "slot5-128-zero-ioi-positive-residual": (
        slot5_base,
        {
            "run_name": "slot5_128_zero_ioi_positive_residual_pad50",
            "slot_version": "slot5",
            "zero_ioi_positive_support": True,
            "zero_ioi_residual": True,
        },
    ),
    "slot5-128-stable-dynamics": (
        slot5_base,
        {
            "run_name": "slot5_128_whole_token_pad50_stable_dynamics",
            "slot_version": "slot5",
            "stable_dynamics_training": True,
            "stable_apply_prob": 0.30,
            "stable_channels": ["ioi", "duration"],
            "stable_noise_modes": stable_noise_modes,
        },
    ),
}

for name, (base, overrides) in variants.items():
    cfg = dict(base)
    cfg.update(common)
    cfg.update(overrides)
    for key in ("resume_path", "prior_attribute_keep_probs"):
        cfg.pop(key, None)
    cfg["output_dir"] = f"unused/{name}/training"
    cfg["logging_dir"] = f"unused/{name}/tf-logs"
    (config_dir / f"{name}.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

report = {
    "slot5_base": str(slot5_base_path),
    "slot6_base": str(slot6_base_path),
    "common": common,
    "variants": {name: overrides for name, (_, overrides) in variants.items()},
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
  local gpu="$1"
  local name="$2"
  local session="slot_zero_${name//[^A-Za-z0-9_]/_}_${STAMP}"
  local run_dir="${RUN_ROOT}/${name}"
  local log_path="${run_dir}/launcher.log"
  mkdir -p "${run_dir}"

  tmux new-session -d -s "${session}" \
    "cd '${ROOT_DIR}' && \
     env CUDA_VISIBLE_DEVICES='${gpu}' \
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

  printf '%s\tGPU%s\t%s\t%s\n' "${session}" "${gpu}" "${run_dir}" "${log_path}" \
    | tee -a "${RUN_ROOT}/sessions.tsv"
}

launch_one "0" "slot6-128-mlp-decoder-musical-mask"
launch_one "1" "slot5-128-zero-ioi-positive"
launch_one "2" "slot5-128-zero-ioi-positive-residual"
launch_one "3" "slot5-128-stable-dynamics"

echo "RUN_ROOT=${RUN_ROOT}"
