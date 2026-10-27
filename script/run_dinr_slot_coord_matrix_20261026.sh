#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
GPU="${1:?usage: $0 GPU_ID run_name ...}"
shift
[[ "$#" -gt 0 ]] || { echo 'at least one run name required' >&2; exit 2; }
MATRIX_ROOT="results/inr_epr_pipeline/dinr_slot_coord_matrix_20261026"
STATE_FILE="$MATRIX_ROOT/gpu${GPU}.status.tsv"
mkdir -p "$MATRIX_ROOT"
for RUN_NAME in "$@"; do
  CONFIG="configs/inr_epr/dinr_slot_coord_matrix_20261026/${RUN_NAME}.json"
  RUN_DIR="$MATRIX_ROOT/$RUN_NAME"
  LOG="$MATRIX_ROOT/$RUN_NAME.launcher.log"
  [[ -s "$CONFIG" ]] || { echo "missing config: $CONFIG" >&2; exit 1; }
  printf '%s\t%s\tSTARTED\t%s\n' "$GPU" "$RUN_NAME" "$(date -Is)" | tee -a "$STATE_FILE"
  if CUDA_VISIBLE_DEVICES="$GPU" CONFIG="$CONFIG" RUN_DIR_OVERRIDE="$RUN_DIR" \
    BASE_NUM_TRAIN_EPOCHS=16 ADAPT_NUM_TRAIN_EPOCHS=0 RESUME_FROM_LATEST_CHECKPOINT=0 \
    PIPELINE_STAGE_START=train EVAL_CHECKPOINT_MODE=best \
    INFER_NUM_WORKERS=8 METRIC_NUM_WORKERS=8 SAMPLING_NUM_SAMPLES=3 \
    bash script/run_inr_epr_pipeline.sh >"$LOG" 2>&1; then
    printf '%s\t%s\tDONE\t%s\n' "$GPU" "$RUN_NAME" "$(date -Is)" | tee -a "$STATE_FILE"
  else
    RC=$?
    printf '%s\t%s\tFAILED(%s)\t%s\n' "$GPU" "$RUN_NAME" "$RC" "$(date -Is)" | tee -a "$STATE_FILE"
    exit "$RC"
  fi
done
