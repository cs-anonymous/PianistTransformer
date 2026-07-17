#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/inr_epr_pipeline/matched_dlm_k1_corrected_topk_t0_${STAMP}}"
BASE_RUN="results/inr_epr_pipeline/asaponly_matched_cinr_dinr_20260716_143813/cinr_dlm_k1"
BASE_CONFIG="${BASE_RUN}/config.json"
CHECKPOINT="${BASE_RUN}/training/CINR-DLM-k1-pianocore3-asap16-noclamp-topp95-t08/checkpoint-best"
DET_MANIFEST="${BASE_RUN}/deterministic/prediction_manifest.json"
TRAIN_OUTPUT_DIR="${BASE_RUN}/training/CINR-DLM-k1-pianocore3-asap16-noclamp-topp95-t08"
SCORE_LIST="${SCORE_LIST:-data/asap_test_score_sources.txt}"
mkdir -p "${RUN_ROOT}/configs"

python - "${BASE_CONFIG}" "${RUN_ROOT}/configs" <<'PY'
import json
import sys
from pathlib import Path

base = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
out = Path(sys.argv[2])

def fmt(value):
    if value == 0:
        return "0"
    return str(value).replace(".", "p").rstrip("0").rstrip("p")

for temperature in (1.0, 0.8, 0.6, 0.4, 0.2, 0.0):
    for top_k in (16, 32, 64):
        name = f"t{fmt(temperature)}_k{top_k}"
        cfg = dict(base)
        cfg.update(
            {
                "run_name": f"matched-dlm-k1-corrected-{name}",
                "dlm_sampling_temperature": temperature,
                "dlm_sampling_top_p": 1.0,
                "sampling_top_p": 1.0,
                "dlm_sampling_top_k": top_k,
                "sampling_top_k": top_k,
            }
        )
        (out / f"{name}.json").write_text(
            json.dumps(cfg, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

for top_p in (0.95, 0.90, 0.80):
    name = f"t0_p{fmt(top_p)}"
    cfg = dict(base)
    cfg.update(
        {
            "run_name": f"matched-dlm-k1-corrected-{name}",
            "dlm_sampling_temperature": 0.0,
            "dlm_sampling_top_p": top_p,
            "sampling_top_p": top_p,
            "dlm_sampling_top_k": 0,
            "sampling_top_k": 0,
        }
    )
    (out / f"{name}.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
PY

run_one() {
  local gpu="$1" name="$2"
  local config="${RUN_ROOT}/configs/${name}.json"
  local out_dir="${RUN_ROOT}/${name}"
  local sampling_dir="${out_dir}/sampling"
  if [[ -s "${out_dir}/summary.json" ]]; then
    echo "[$(date '+%F %T')] SKIP gpu=${gpu} ${name}" | tee -a "${RUN_ROOT}/status.tsv"
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
    --train-output-dir "${TRAIN_OUTPUT_DIR}" \
    --pipeline-log "${out_dir}/infer.log" \
    --evaluate-log "${out_dir}/evaluate.log" \
    --num-workers 8 2>&1 | tee "${out_dir}/evaluate.log"
  echo "[$(date '+%F %T')] DONE gpu=${gpu} ${name}" | tee -a "${RUN_ROOT}/status.tsv"
}

run_gpu0() {
  run_one 0 t0_p0p95
  for t in 1 0p4; do
    for k in 16 32 64; do run_one 0 "t${t}_k${k}"; done
  done
}

run_gpu1() {
  run_one 1 t0_p0p9
  for t in 0p8 0p2; do
    for k in 16 32 64; do run_one 1 "t${t}_k${k}"; done
  done
}

run_gpu2() {
  run_one 2 t0_p0p8
  for t in 0p6 0; do
    for k in 16 32 64; do run_one 2 "t${t}_k${k}"; done
  done
}

case "${1:-all}" in
  gpu0) run_gpu0 ;;
  gpu1) run_gpu1 ;;
  gpu2) run_gpu2 ;;
  all)
    tmux new-session -d -s "mdlm_fix0_${STAMP: -6}" "cd '${ROOT_DIR}' && RUN_ROOT='${RUN_ROOT}' bash '$0' gpu0"
    tmux new-session -d -s "mdlm_fix1_${STAMP: -6}" "cd '${ROOT_DIR}' && RUN_ROOT='${RUN_ROOT}' bash '$0' gpu1"
    tmux new-session -d -s "mdlm_fix2_${STAMP: -6}" "cd '${ROOT_DIR}' && RUN_ROOT='${RUN_ROOT}' bash '$0' gpu2"
    echo "RUN_ROOT=${RUN_ROOT}"
    echo "SESSIONS=mdlm_fix0_${STAMP: -6},mdlm_fix1_${STAMP: -6},mdlm_fix2_${STAMP: -6}"
    ;;
  *) echo "usage: $0 [gpu0|gpu1|gpu2|all]" >&2; exit 2 ;;
esac
