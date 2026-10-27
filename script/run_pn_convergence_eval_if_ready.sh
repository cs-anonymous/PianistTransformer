#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="${ROOT:-${1:?ROOT is required}}"
GPU="${GPU:-2}"
SEEDS="${SEEDS:-11,22,33}"
EPOCHS="${EPOCHS:-36,48,60}"
NUM_SAMPLES="${NUM_SAMPLES:-3}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EVAL_WORKERS="${EVAL_WORKERS:-1}"
SCORE_SOURCE_LIST="${SCORE_SOURCE_LIST:-data/asap_test_score_sources.txt}"
python - "${ROOT}" "${SEEDS}" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1])
for seed in sys.argv[2].split(','):
  for condition in ('baseline','cached'):
    p=root/condition/f'seed_{int(seed)}'/ 'training' / f'{condition}_s{int(seed)}' / 'convergence_result.json'
    if not p.exists(): raise SystemExit(f'missing convergence result: {p}')
    result=json.loads(p.read_text())
    if result.get('status') != 'optimization_plateau':
      raise SystemExit(f'PN/PP evaluation blocked: {p} status={result.get("status")}')
PY
IFS=',' read -r -a seeds <<< "${SEEDS}"
IFS=',' read -r -a epochs <<< "${EPOCHS}"
for seed in "${seeds[@]}"; do
  for condition in baseline cached; do
    config="${ROOT}/configs/${condition}_seed_${seed}.json"
    for epoch in "${epochs[@]}"; do
      checkpoint="${ROOT}/${condition}/seed_${seed}/training/${condition}_s${seed}/epoch-${epoch}"
      out="${ROOT}/${condition}/seed_${seed}/inference_epoch${epoch}_ns${NUM_SAMPLES}"
      [[ -d "${checkpoint}" ]] || { echo "missing checkpoint ${checkpoint}" >&2; exit 1; }
      [[ -s "${out}/eval_pn_pp_metrics.json" ]] && continue
      mkdir -p "${out}"
      CUDA_VISIBLE_DEVICES="${GPU}" python src/inference/infer_inr_testset.py \
        --config "${config}" --checkpoint "${checkpoint}" --split test \
        --performance-dataset ASAP --score-source-list "${SCORE_SOURCE_LIST}" \
        --protocol sampling --num-samples "${NUM_SAMPLES}" --seed "${seed}" \
        --batch-size-windows 8 --num-workers "${NUM_WORKERS}" --output-dir "${out}"
      CUDA_VISIBLE_DEVICES="${GPU}" python src/evaluate/evaluate_inr_saved_midis.py \
        --prediction-manifest "${out}/prediction_manifest.json" \
        --output-json "${out}/eval_pn_pp_metrics.json" \
        --score-source-list "${SCORE_SOURCE_LIST}" --num-workers "${EVAL_WORKERS}"
    done
  done
done
