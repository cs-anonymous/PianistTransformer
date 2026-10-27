#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT_DIR"
GPU_LIST="${GPU_LIST:-0,1,2}"; IFS=',' read -r -a GPUS <<< "$GPU_LIST"
[[ ${#GPUS[@]} -eq 3 ]] || { echo 'GPU_LIST must contain 3 ids' >&2; exit 2; }
CONFIG="${CONFIG:-results/baseline0807_epoch_hr_20260807_170343/runtime_configs/baseline0807.json}"
ROOT="${ROOT:-results/variance_matrix_3gpu/M3_training}"
OUT_ROOT="${OUT_ROOT:-results/variance_matrix_3gpu/M3_checkpoint_eval}"
SCORE_SOURCE_LIST="${SCORE_SOURCE_LIST:-data/asap_test_score_sources.txt}"
NUM_WORKERS="${NUM_WORKERS:-1}"; CONCURRENCY_PER_GPU="${CONCURRENCY_PER_GPU:-8}"
EPOCHS="${EPOCHS:-4,8,12}"; SAMPLING_SEEDS="${SAMPLING_SEEDS:-11,22,33}"
mkdir -p "$OUT_ROOT/logs"
run_one() {
  local gpu="$1" job="$2" checkpoint="$3" seed="$4" out="$5"; mkdir -p "$out"
  [[ -s "$out/eval_pn_pp_metrics.json" ]] && return 0
  { CUDA_VISIBLE_DEVICES="$gpu" python src/inference/infer_inr_testset.py --config "$CONFIG" --checkpoint "$checkpoint" --split test --performance-dataset ASAP --score-source-list "$SCORE_SOURCE_LIST" --protocol sampling --num-samples 1 --seed "$seed" --batch-size-windows 8 --num-workers "$NUM_WORKERS" --output-dir "$out"
    CUDA_VISIBLE_DEVICES="$gpu" python src/evaluate/evaluate_inr_saved_midis.py --prediction-manifest "$out/prediction_manifest.json" --output-json "$out/eval_pn_pp_metrics.json" --score-source-list "$SCORE_SOURCE_LIST" --num-workers "$NUM_WORKERS"; } > "$OUT_ROOT/logs/${job}.log" 2>&1
}
declare -a JOBS=(); idx=0
IFS=',' read -r -a seed_arr <<< "$SAMPLING_SEEDS"; IFS=',' read -r -a epoch_arr <<< "$EPOCHS"
for train_seed in 11 22 33 44 55; do
  train_dir="$ROOT/seed_${train_seed}/training/baseline0807"
  for epoch in "${epoch_arr[@]}"; do
    ckpt="$train_dir/epoch-${epoch}"; [[ -d "$ckpt" ]] || { echo "Missing checkpoint $ckpt" >&2; exit 2; }
    for sample_seed in "${seed_arr[@]}"; do gpu="${GPUS[$((idx % 3))]}"; job="train${train_seed}_ep${epoch}_s${sample_seed}"; out="$OUT_ROOT/train_${train_seed}/epoch_${epoch}/sample_${sample_seed}"; JOBS+=("$gpu|$job|$ckpt|$sample_seed|$out"); idx=$((idx+1)); done
  done
done
declare -a PIDS=(); declare -a COUNTS=(0 0 0)
for spec in "${JOBS[@]}"; do
  IFS='|' read -r gpu job ckpt seed out <<< "$spec"; slot=0; [[ "${GPUS[1]}" == "$gpu" ]] && slot=1; [[ "${GPUS[2]}" == "$gpu" ]] && slot=2
  if [[ ${COUNTS[$slot]} -ge $CONCURRENCY_PER_GPU ]]; then for p in "${PIDS[@]}"; do wait "$p"; done; PIDS=(); COUNTS=(0 0 0); fi
  run_one "$gpu" "$job" "$ckpt" "$seed" "$out" & PIDS+=("$!"); COUNTS[$slot]=$((COUNTS[$slot]+1))
done
for p in "${PIDS[@]}"; do wait "$p"; done
echo "completed=$(find "$OUT_ROOT" -name eval_pn_pp_metrics.json | wc -l) total=${#JOBS[@]}"
