#!/usr/bin/env bash
set -euo pipefail

cd /home/sy/EPR/PianistTransformer

RUN_TRAIN="${RUN_TRAIN:-1}" \
RUN_INFER="${RUN_INFER:-0}" \
RUN_EVAL="${RUN_EVAL:-0}" \
RUN_STATS="${RUN_STATS:-0}" \
  bash scripts/run_chord_asap_full_pipeline.sh
