#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/cinr__default_dlm_k1_bounded5.json}
CHECKPOINT=${CHECKPOINT:-/path/to/checkpoint-best}
OUT_DIR=${OUT_DIR:-results/example_inference}
SCORE_SOURCE_LIST=${SCORE_SOURCE_LIST:-data/asap_test_score_sources.txt}

python src/inference/infer_inr_testset.py   --config "$CONFIG"   --checkpoint "$CHECKPOINT"   --split test   --performance-dataset ASAP   --score-source-list "$SCORE_SOURCE_LIST"   --protocol sampling   --num-samples 1   --output-dir "$OUT_DIR"

python src/evaluate/evaluate_inr_saved_midis.py   --prediction-manifest "$OUT_DIR/prediction_manifest.json"   --output-json "$OUT_DIR/eval_pn_pp_metrics.json"   --score-source-list "$SCORE_SOURCE_LIST"
