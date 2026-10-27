#!/usr/bin/env bash
set -euo pipefail

# Serial checkpoint queue on one inference GPU, with eight model processes.
# Training may still be running on the other GPUs.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PILOT_ROOT="${PILOT_ROOT:-results/pn_cached_velocity_longpilot_seed11}"
GPU_LIST="${GPU_LIST:-2}"
TRAIN_SEED="${TRAIN_SEED:-11}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-14400}"
export PILOT_ROOT GPU_LIST TRAIN_SEED
export NUM_WORKERS="${NUM_WORKERS:-8}"
export EVAL_WORKERS="${EVAL_WORKERS:-1}"
export NUM_SAMPLES="${NUM_SAMPLES:-3}"
export SAMPLING_SEED="${SAMPLING_SEED:-11}"
deadline=$((SECONDS + WAIT_TIMEOUT_SECONDS))
for condition in cached baseline; do
  for epoch in 4 8 12; do
    checkpoint="${PILOT_ROOT}/${condition}/seed_${TRAIN_SEED}/training/${condition}_s${TRAIN_SEED}/epoch-${epoch}"
    echo "Waiting for ${checkpoint}"
    until [[ -s "${checkpoint}/trainer_state.json" ]]; do
      if (( SECONDS >= deadline )); then
        echo "Checkpoint wait timed out: ${checkpoint}" >&2
        exit 1
      fi
      sleep 15
    done
    CONDITIONS="${condition}" EPOCH="${epoch}" bash script/run_pn_cached_velocity_eval_3gpu.sh
    echo "Completed ${condition} epoch-${epoch}"
  done
done
