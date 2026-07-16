#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/inr_epr_pipeline/dinr_sampling_matrix_${STAMP}}"
BASE_RUN="results/inr_epr_pipeline/dinr_separated_corrected_20260716_004453"
BASE_CONFIG="${BASE_RUN}/config.json"
CHECKPOINT="${BASE_RUN}/training/DINR-separated-corrected-absolute256-deviation256-m2p1/checkpoint-best"
DET_MANIFEST="${BASE_RUN}/deterministic/prediction_manifest.json"
SCORE_LIST="data/asap_test_score_sources.txt"
mkdir -p "${RUN_ROOT}/configs"

python - "${BASE_CONFIG}" "${RUN_ROOT}/configs" <<'PY'
import json
import sys
from pathlib import Path

base_path, output_dir = Path(sys.argv[1]), Path(sys.argv[2])
base = json.loads(base_path.read_text(encoding="utf-8"))
for temperature in (0.8, 0.6, 0.4):
    t_name = str(temperature).replace(".", "p")
    for top_p in (0.95, 0.90):
        p_name = str(top_p).replace(".", "p")
        name = f"t{t_name}_p{p_name}"
        cfg = dict(base)
        cfg.update({
            "run_name": f"DINR-infer-{name}",
            "dinr_sampling_temperature": temperature,
            "dinr_sampling_top_p": top_p,
            "sampling_top_p": top_p,
            "dinr_sampling_top_k": 0,
            "sampling_top_k": 0,
        })
        (output_dir / f"{name}.json").write_text(json.dumps(cfg, indent=2) + "\n")
    for top_k in (32, 64):
        name = f"t{t_name}_k{top_k}"
        cfg = dict(base)
        cfg.update({
            "run_name": f"DINR-infer-{name}",
            "dinr_sampling_temperature": temperature,
            "dinr_sampling_top_p": 1.0,
            "sampling_top_p": 1.0,
            "dinr_sampling_top_k": top_k,
            "sampling_top_k": top_k,
        })
        (output_dir / f"{name}.json").write_text(json.dumps(cfg, indent=2) + "\n")
PY

run_one() {
  local gpu="$1" name="$2"
  local config="${RUN_ROOT}/configs/${name}.json"
  local out_dir="${RUN_ROOT}/${name}"
  local sampling_dir="${out_dir}/sampling"
  if [[ -s "${out_dir}/summary.json" ]]; then
    echo "[$(date '+%F %T')] SKIP gpu=${gpu} ${name} summary_exists" | tee -a "${RUN_ROOT}/status.tsv"
    return 0
  fi
  mkdir -p "${sampling_dir}"
  echo "[$(date '+%F %T')] START gpu=${gpu} ${name}" | tee -a "${RUN_ROOT}/status.tsv"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 python src/inference/infer_inr_testset.py \
    --config "${config}" \
    --checkpoint "${CHECKPOINT}" \
    --split test \
    --performance-dataset ASAP \
    --num-workers 8 \
    --batch-size-windows 8 \
    --merge-mode continuation \
    --continuation-drop-ratio 0.0 \
    --device cuda \
    --protocol sampling \
    --num-samples 1 \
    --score-source-list "${SCORE_LIST}" \
    --deterministic-strategy mean \
    --output-dir "${sampling_dir}" 2>&1 | tee "${out_dir}/infer.log"
  PYTHONUNBUFFERED=1 python src/evaluate/summarize_inr_asap_pipeline.py \
    --deterministic-manifest "${DET_MANIFEST}" \
    --sampling-manifest "${sampling_dir}/prediction_manifest.json" \
    --output-json "${out_dir}/summary.json" \
    --output-plot "${out_dir}/asap_distribution.png" \
    --config "${config}" \
    --checkpoint "${CHECKPOINT}" \
    --train-output-dir "$(dirname "${CHECKPOINT}")" \
    --pipeline-log "${out_dir}/infer.log" \
    --evaluate-log "${out_dir}/evaluate.log" \
    --num-workers 8 2>&1 | tee "${out_dir}/evaluate.log"
  echo "[$(date '+%F %T')] DONE gpu=${gpu} ${name}" | tee -a "${RUN_ROOT}/status.tsv"
}

run_top_p() {
  for t in 0p8 0p6 0p4; do
    for p in 0p95 0p9; do
      run_one 0 "t${t}_p${p}"
    done
  done
}

run_top_k() {
  for t in 0p8 0p6 0p4; do
    for k in 32 64; do
      run_one 1 "t${t}_k${k}"
    done
  done
}

case "${1:-all}" in
  top-p) run_top_p ;;
  top-k) run_top_k ;;
  all)
    run_top_p &
    p_pid=$!
    run_top_k &
    k_pid=$!
    wait "${p_pid}"
    wait "${k_pid}"
    ;;
  *) echo "usage: $0 [top-p|top-k|all]" >&2; exit 2 ;;
esac

echo "[$(date '+%F %T')] MATRIX_DONE" | tee -a "${RUN_ROOT}/status.tsv"
