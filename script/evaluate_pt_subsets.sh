#!/bin/bash

# Evaluate PT on ASAP and PianoCoRe-only subsets
# Each evaluation computes BOTH binary and continuous pedal metrics
# Usage: bash scripts/evaluate_pt_subsets.sh

set -e

CONFIG="configs/sft_node_config_pianocore.json"
OUTPUT_DIR="results/pt_evaluation_by_subset"
WORKERS_PER_GPU=2  # 每个GPU运行2个worker进程

echo "=========================================="
echo "PT Evaluation on ASAP and PianoCoRe Subsets"
echo "Each run computes BOTH binary and continuous pedal"
echo "Workers per GPU: $WORKERS_PER_GPU"
echo "=========================================="
echo ""

# Run 2 evaluations in parallel (each computes both binary and continuous)
echo "Starting both evaluations in parallel..."

# ASAP subset - computes both binary and continuous
echo "1. Starting ASAP subset evaluation..."
python src/evaluate/evaluate_pt_by_subset.py \
    --config "$CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    --subset asap \
    --workers-per-gpu "$WORKERS_PER_GPU" \
    > logs/pt_eval_asap.log 2>&1 &
PID1=$!

sleep 5  # 稍微等待以避免同时启动冲突

# PianoCoRe-only subset - computes both binary and continuous
echo "2. Starting PianoCoRe-only subset evaluation..."
python src/evaluate/evaluate_pt_by_subset.py \
    --config "$CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    --subset pianocore \
    --workers-per-gpu "$WORKERS_PER_GPU" \
    > logs/pt_eval_pianocore.log 2>&1 &
PID2=$!

echo ""
echo "Both evaluations started in parallel:"
echo "  PID $PID1: ASAP (binary + continuous)"
echo "  PID $PID2: PianoCoRe (binary + continuous)"
echo ""
echo "Monitor progress:"
echo "  tail -f logs/pt_eval_asap.log"
echo "  tail -f logs/pt_eval_pianocore.log"
echo ""

# Wait for both processes to complete
echo "Waiting for both evaluations to complete..."
wait $PID1 $PID2

echo ""
echo "=========================================="
echo "Both evaluations complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "  - pt_results_asap.json (binary + continuous)"
echo "  - pt_results_pianocore.json (binary + continuous)"
echo "=========================================="
