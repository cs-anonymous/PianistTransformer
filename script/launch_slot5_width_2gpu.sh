#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/slot5_width_2gpu/${STAMP}}"
CONFIG_DIR="${RUN_ROOT}/configs"
BASE_CONFIG="${BASE_CONFIG:-results/slot8_mask_stable_2gpu/20260710_slot8_mask_stable_v1/configs/slot8-whole-token-mask.json}"
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
        "slot_version": "slot5",
        "slot_fusion": "mlp",
        "slot_gates": False,
        "slot_share_role_encoders": True,
        "slot_decoder_mask_mode": "whole_token",
        "tf_embedding_mask_score": False,
        "tf_embedding_mask_decoder": True,
        "tf_embedding_mask_keep_prob": 0.5,
        "prior_property_dropout_prob": None,
        "prior_token_keep_prob": 1.0,
        "prior_token_dropout_mode": "mask",
        "dagger_prefix_training": False,
        "stable_dynamics_training": False,
        "stable_contract_loss": False,
        "stable_contract_lambda": 0.0,
        "raw_timing_head_type": "regression",
        "raw_timing_loss_lambda": 0.25,
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
for key in (
    "resume_path",
    "prior_attribute_keep_probs",
    "prior_property_dropout_pattern",
    "prior_property_dropout_replacement",
    "prior_property_visible_prob",
    "prior_property_all_dropout_prob",
    "stable_force_all_properties_visible",
):
    common.pop(key, None)

variants = {
    "slot5-128-whole-token-pad": {
        "run_name": "slot5_128_whole_token_pad50",
        "slot_dim": 128,
    },
    "slot5-256-whole-token-pad": {
        "run_name": "slot5_256_whole_token_pad50",
        "slot_dim": 256,
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

allowed_differences = {"run_name", "slot_dim", "output_dir", "logging_dir"}
loaded = {
    name: json.loads((config_dir / f"{name}.json").read_text(encoding="utf-8"))
    for name in variants
}
unexpected = []
for key in sorted(set().union(*(cfg.keys() for cfg in loaded.values()))):
    values = {json.dumps(cfg.get(key), sort_keys=True) for cfg in loaded.values()}
    if len(values) > 1 and key not in allowed_differences:
        unexpected.append(key)
if unexpected:
    raise SystemExit(f"Unexpected cross-experiment config differences: {unexpected}")

report = {
    "baseline": str(base_path),
    "purpose": "Corrected-chain Slot5 width comparison with whole-token PAD",
    "common": {
        "slot_version": common["slot_version"],
        "slot_fusion": common["slot_fusion"],
        "slot_share_role_encoders": common["slot_share_role_encoders"],
        "slot_decoder_mask_mode": common["slot_decoder_mask_mode"],
        "tf_embedding_mask_score": common["tf_embedding_mask_score"],
        "tf_embedding_mask_decoder": common["tf_embedding_mask_decoder"],
        "tf_embedding_mask_keep_prob": common["tf_embedding_mask_keep_prob"],
        "raw_timing_head_type": common["raw_timing_head_type"],
        "raw_timing_loss_lambda": common["raw_timing_loss_lambda"],
        "epochs": common["num_train_epochs"],
        "per_device_batch_size": common["per_device_train_batch_size"],
        "global_batch_size": common["global_batch_size"],
        "train_performance_dataset": common["train_performance_dataset"],
        "sampling_set": "cheap15, one sample per score",
    },
    "allowed_differences": sorted(allowed_differences),
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
  local session="slot5width_${name//[^A-Za-z0-9_]/_}_${STAMP}"
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

launch_one "0,1" "slot5-128-whole-token-pad"
launch_one "2,3" "slot5-256-whole-token-pad"

echo "RUN_ROOT=${RUN_ROOT}"
