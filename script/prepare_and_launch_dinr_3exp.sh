#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

COMMON=(
  --metadata-path PianoCoRe/metadata.csv
  --refined-dir PianoCoRe/processed
  --split train
  --performance-dataset ASAP
  --task-type epr
  --input-feature-mode integrated
  --musical-feature-mode none
  --disable-musical-features
  --timing-control-mode dinr_floor_log
  --pedal-representation binary_4
  --ready
  --workers "${SIDECAR_WORKERS:-40}"
  --fixed-window-split-summary-path data/train_valid_asap3_nonasap05_v1_summary.json
)

python src/data_process/prebuild_inr_work_pt.py \
  "${COMMON[@]}" \
  --epr-timing-target floor_log_deviation \
  --sidecar-tag DINR_READY_ASAP

BASE_NUM_TRAIN_EPOCHS=16 \
BASE_ASAP_ONLY=1 \
ADAPT_NUM_TRAIN_EPOCHS=0 \
BATCH_SIZE_PER_DEVICE=32 \
GLOBAL_BATCH_SIZE=64 \
SAMPLING_NUM_SAMPLES=1 \
bash script/launch_dinr_3exp.sh
