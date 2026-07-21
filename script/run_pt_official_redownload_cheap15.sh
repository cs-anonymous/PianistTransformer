#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODEL_DIR="${MODEL_DIR:-models/official_redownload_20260711/pianist-transformer-rendering}"
RUN_DIR="${RUN_DIR:-results/pt_official_redownload_20260711_cheap15}"
LOG="${RUN_DIR}/pipeline.log"
mkdir -p "${MODEL_DIR}" "${RUN_DIR}"

exec > >(tee -a "${LOG}") 2>&1
echo "START $(date '+%F %T')"

export http_proxy="${http_proxy:-http://127.0.0.1:7890}"
export https_proxy="${https_proxy:-http://127.0.0.1:7890}"
export HTTP_PROXY="${HTTP_PROXY:-${http_proxy}}"
export HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy}}"
echo "PROXY ${https_proxy}"

if [[ ! -f "${MODEL_DIR}/model.safetensors" || "$(stat -c %s "${MODEL_DIR}/model.safetensors" 2>/dev/null || echo 0)" -lt 270000000 ]]; then
  curl -L --fail --retry 20 --retry-delay 5 -C - \
    -o "${MODEL_DIR}/model.safetensors" \
    'https://huggingface.co/yhj137/pianist-transformer-rendering/resolve/main/model.safetensors?download=true'
else
  echo "MODEL_DOWNLOAD reuse complete ${MODEL_DIR}/model.safetensors"
fi

sha256sum "${MODEL_DIR}/model.safetensors" | tee "${RUN_DIR}/model.sha256"

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python src/inference/infer_pt_testset.py \
  --model-path "${MODEL_DIR}" \
  --metadata data/ASAP_processed/metadata.generated_json.csv \
  --midi-root data/ASAP_processed \
  --split test \
  --performance-dataset ASAP \
  --output-dir "${RUN_DIR}/sampling" \
  --device cuda \
  --batch-size 8 \
  --num-workers 8 \
  --protocol sampling \
  --num-samples 1 \
  --temperature 1.0 \
  --top-p 0.95 \
  --overlap-ratio 0.125 \
  --max-context-length 4096 \
  --seed 42

python src/evaluate/evaluate_inr_saved_midis.py \
  --prediction-manifest "${RUN_DIR}/sampling/prediction_manifest.json" \
  --score-source-list data/cheap15_score_sources.txt \
  --num-workers 8 \
  --output-json "${RUN_DIR}/sampling/score_level_pp_pn_cheap15.json"

echo "END $(date '+%F %T')"
