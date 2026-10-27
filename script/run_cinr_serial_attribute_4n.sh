#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export CONFIG="${CONFIG:-configs/inr_epr/cinr__serial_attribute_4n.json}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Four decoder steps per note increase decoder attention cost. Keep the
# effective batch at 64 by default while reducing the per-device microbatch.
export BATCH_SIZE_PER_DEVICE="${BATCH_SIZE_PER_DEVICE:-4}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
export BASE_NUM_TRAIN_EPOCHS="${BASE_NUM_TRAIN_EPOCHS:-24}"
export ADAPT_NUM_TRAIN_EPOCHS="${ADAPT_NUM_TRAIN_EPOCHS:-0}"
export PIPELINE_STAGE_START="${PIPELINE_STAGE_START:-train}"
export SAMPLING_NUM_SAMPLES="${SAMPLING_NUM_SAMPLES:-3}"

exec bash script/run_inr_epr_pipeline.sh
