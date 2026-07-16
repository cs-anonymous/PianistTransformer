#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR_OVERRIDE:-results/inr_epr_pipeline/dinr_asap_no_dev_numeric_gpu0_${STAMP}}"
BASE_CONFIG="results/inr_epr_pipeline/dinr_separated_corrected_20260716_004453/config.json"
CONFIG_PATH="${RUN_DIR}/config.json"
mkdir -p "${RUN_DIR}"

python - "${BASE_CONFIG}" "${CONFIG_PATH}" <<'PY'
import json
import sys
from pathlib import Path

src, dst = map(Path, sys.argv[1:3])
config = json.loads(src.read_text(encoding="utf-8"))
config.update({
    "run_name": "DINR-ASAP-10enc2dec-slot8-no-dev-numeric-m2p1-topp95-t08",
    "epr_distribution": "dinr",
    "epr_timing_target": "floor_log_deviation",
    "dinr_vocabulary_mode": "separated",
    "dinr_timing_bins": 256,
    "dinr_zero_bin": 0,
    "dinr_timing_step": 9.0 / 255.0,
    "dinr_output_timing_bins": 256,
    "dinr_output_zero_bin": 170,
    "dinr_output_timing_step": 3.0 / 255.0,
    "dinr_deviation_min": -2.0,
    "dinr_deviation_max": 1.0,
    "dinr_zero_ioi_min": 0.0,
    "dinr_zero_ioi_max": 5.0,
    "dinr_output_deviation_numerical_coordinates": False,
    "dinr_sampling_temperature": 0.8,
    "dinr_sampling_top_p": 0.95,
    "sampling_top_p": 0.95,
    "train_performance_dataset": "ASAP",
    "eval_performance_dataset": "ASAP",
    "eval_split": "valid",
    "prepared_sidecar_tag": "DINR_READY_ASAP",
    "num_train_epochs": 16.0,
    "max_train_epochs": 16.0,
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "encoder_layers_num": 10,
    "decoder_layers_num": 2,
    "slot_dim": 128,
    "slot_version": "slot8",
    "slot_fusion": "mlp",
})
for key in ("resume_path", "train_performance_dataset_exclude", "eval_performance_dataset_exclude"):
    config.pop(key, None)
dst.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY

env CUDA_VISIBLE_DEVICES=0 \
  CONFIG="${CONFIG_PATH}" RUN_DIR_OVERRIDE="${RUN_DIR}" \
  BASE_ASAP_ONLY=1 BASE_NUM_TRAIN_EPOCHS=16 ADAPT_NUM_TRAIN_EPOCHS=0 \
  ADAPT_PREPARED_SIDECAR_TAG=DINR_READY_ASAP \
  BATCH_SIZE_PER_DEVICE=32 GLOBAL_BATCH_SIZE=64 \
  DET_NUM_SAMPLES=1 SAMPLING_NUM_SAMPLES=1 \
  INFER_NUM_WORKERS=8 METRIC_NUM_WORKERS=8 INFER_BATCH_SIZE_WINDOWS=8 \
  INFER_SCORE_SOURCE_LIST=data/asap_test_score_sources.txt \
  EVAL_CHECKPOINT_MODE=best RESUME_FROM_LATEST_CHECKPOINT=0 \
  MERGE_MODE=continuation CONTINUATION_DROP_RATIO=0.0 \
  bash script/run_inr_epr_pipeline.sh 2>&1 | tee "${RUN_DIR}/launcher.log"
