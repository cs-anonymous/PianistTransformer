#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
RUN_ROOT="${RUN_ROOT:-results/dlm_pn_scale_only_3gpu/20260715_192251}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SCORE_SOURCE_LIST="${SCORE_SOURCE_LIST:-${RUN_ROOT}/score_sources_19.txt}"

launch_one() {
  local gpu="$1" name="$2"
  local run_dir="${RUN_ROOT}/${name}"
  local model_dir="${run_dir}/training/dlm_scale_only_${name}"
  local checkpoint="${model_dir}/checkpoint-273"
  local out="${run_dir}/sampling19"
  local session="eval_scale_${name//[^A-Za-z0-9]/_}"
  mkdir -p "${out}"
  tmux new-session -d -s "${session}" \
    "cd '${ROOT_DIR}' && CUDA_VISIBLE_DEVICES='${gpu}' PYTHONUNBUFFERED=1 \
     /home/kaititech/anaconda3/bin/python src/inference/infer_inr_testset.py \
       --config '${run_dir}/config.json' --checkpoint '${checkpoint}' \
       --split test --protocol sampling --sampling-strategy sample --num-samples 1 \
       --score-source-list '${SCORE_SOURCE_LIST}' \
       --output-dir '${out}' --device cuda --num-workers '${NUM_WORKERS}' --seed 2042 \
       > '${out}/infer.log' 2>&1 && \
     /home/kaititech/anaconda3/bin/python src/evaluate/evaluate_inr_saved_midis.py \
       --prediction-manifest '${out}/prediction_manifest.json' \
       --output-json '${out}/evaluation.json' --num-workers '${NUM_WORKERS}' \
       --score-source-list '${SCORE_SOURCE_LIST}' \
       > '${out}/evaluate.log' 2>&1"
  printf '%s\tGPU%s\t%s\n' "${session}" "${gpu}" "${out}"
}

launch_one 0 lambda-0p5
launch_one 1 lambda-1p0
launch_one 2 lambda-2p0
