#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BASE_CONFIG="${BASE_CONFIG:-configs/local_generated/floorlog_pianocore_base_adapt_gpu12_20260714.json}"
RUN_ROOT="${RUN_ROOT:-results/inr_epr_pipeline/dinr_3exp_$(date +%Y%m%d_%H%M%S)}"
CONFIG_DIR="${RUN_ROOT}/source_configs"
mkdir -p "${CONFIG_DIR}"

python - "${BASE_CONFIG}" "${CONFIG_DIR}" <<'PY'
import json
import sys
from pathlib import Path

base_path, output_dir = map(Path, sys.argv[1:3])
base = json.loads(base_path.read_text(encoding="utf-8"))
common = {
    "epr_distribution": "dinr",
    "timing_control_mode": "dinr_floor_log",
    "note_embedding_mode": "slot_attribute",
    "slot_version": "slot5",
    "slot_dim": 128,
    "slot_fusion": "mlp",
    "slot_share_role_encoders": True,
    "tf_embedding_mask_decoder": True,
    "tf_embedding_mask_keep_prob": 0.5,
    "slot_decoder_mask_mode": "whole_token",
    "prior_property_dropout_prob": None,
    "epr_mixture_components": 1,
    "dinr_absolute_max_ms": 8000.0,
    "dinr_sampling_temperature": 1.0,
    "dinr_numerical_frequencies": 16,
    "dinr_deviation_min": -2.0,
    "dinr_deviation_max": 2.0,
    "dinr_zero_ioi_min": 0.0,
    "dinr_zero_ioi_max": 5.0,
    "output_continuous_dim": 7,
    "use_prepared_sidecar": True,
}
variants = {
    "unified_log_deviation": {
        "run_name": "DINR-unified-log-deviation-512",
        "epr_timing_target": "floor_log_deviation",
        "dinr_vocabulary_mode": "unified",
        "dinr_timing_bins": 512,
        "dinr_zero_bin": 93,
        "dinr_timing_step": 2.0 / 93.0,
        "dinr_output_timing_bins": 512,
        "dinr_output_zero_bin": 93,
        "dinr_output_timing_step": 2.0 / 93.0,
        "prepared_sidecar_tag": "DINR_READY_ASAP",
    },
    "unified_log_absolute": {
        "run_name": "DINR-unified-log-absolute-512",
        "epr_timing_target": "floor_log_absolute",
        "dinr_vocabulary_mode": "unified",
        "dinr_timing_bins": 512,
        "dinr_zero_bin": 0,
        "dinr_timing_step": 9.0 / 511.0,
        "dinr_output_timing_bins": 512,
        "dinr_output_zero_bin": 0,
        "dinr_output_timing_step": 9.0 / 511.0,
        "prepared_sidecar_tag": "DINR_READY_ASAP",
    },
    "separated_vocab_256_256": {
        "run_name": "DINR-separated-absolute256-deviation256",
        "epr_timing_target": "floor_log_deviation",
        "dinr_vocabulary_mode": "separated",
        "dinr_timing_bins": 256,
        "dinr_zero_bin": 0,
        "dinr_timing_step": 9.0 / 255.0,
        "dinr_output_timing_bins": 256,
        "dinr_output_zero_bin": 128,
        "dinr_output_timing_step": 4.0 / 255.0,
        "prepared_sidecar_tag": "DINR_READY_ASAP",
    },
}
for name, overrides in variants.items():
    cfg = dict(base)
    cfg.update(common)
    cfg.update(overrides)
    (output_dir / f"{name}.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
PY

for spec in \
  "dinr_dev:0:unified_log_deviation" \
  "dinr_abs:1:unified_log_absolute" \
  "dinr_sep:2:separated_vocab_256_256"
do
  IFS=: read -r session gpu variant <<<"${spec}"
  tmux has-session -t "${session}" 2>/dev/null && tmux kill-session -t "${session}"
  command="cd '${ROOT_DIR}' && CUDA_VISIBLE_DEVICES=${gpu} CONFIG='${CONFIG_DIR}/${variant}.json' RUN_DIR_OVERRIDE='${RUN_ROOT}/${variant}' BASE_NUM_TRAIN_EPOCHS=${BASE_NUM_TRAIN_EPOCHS:-4} BASE_ASAP_ONLY=${BASE_ASAP_ONLY:-0} ADAPT_NUM_TRAIN_EPOCHS=${ADAPT_NUM_TRAIN_EPOCHS:-8} BATCH_SIZE_PER_DEVICE=${BATCH_SIZE_PER_DEVICE:-16} GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64} DET_NUM_SAMPLES=${DET_NUM_SAMPLES:-1} SAMPLING_NUM_SAMPLES=${SAMPLING_NUM_SAMPLES:-1} bash script/run_inr_epr_pipeline.sh"
  tmux new-session -d -s "${session}" "bash -lc \"${command}\""
done

echo "RUN_ROOT=${RUN_ROOT}"
tmux list-sessions | grep -E '^dinr_(dev|abs|sep):'
