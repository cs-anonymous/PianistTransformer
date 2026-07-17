#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/inr_epr_pipeline/matched_dlm_k1_sampling_matrix_${STAMP}}"
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

base_path = Path(sys.argv[1])
config_dir = Path(sys.argv[2])
base = json.loads(base_path.read_text(encoding="utf-8"))
temps = (1.0, 0.8, 0.6, 0.4, 0.2, 0.0)
top_ps = (0.95, 0.90, 0.80)
top_ks = (16, 32, 64)

def fmt(value):
    if isinstance(value, float):
        if value == 0.0:
            return "0"
        return str(value).replace(".", "p").rstrip("0").rstrip("p")
    return str(value)

manifest = {"top_p": [], "top_k": []}
for temperature in temps:
    t_name = fmt(temperature)
    for top_p in top_ps:
        p_name = fmt(top_p)
        name = f"t{t_name}_p{p_name}"
        cfg = dict(base)
        cfg.update(
            {
                "run_name": f"matched-dlm-k1-{name}",
                "dlm_sampling_temperature": temperature,
                "dlm_sampling_top_p": top_p,
                "sampling_top_p": top_p,
                "dlm_sampling_top_k": 0,
                "sampling_top_k": 0,
            }
        )
        path = config_dir / f"{name}.json"
        path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        manifest["top_p"].append(name)
    for top_k in top_ks:
        name = f"t{t_name}_k{top_k}"
        cfg = dict(base)
        cfg.update(
            {
                "run_name": f"matched-dlm-k1-{name}",
                "dlm_sampling_temperature": temperature,
                "dlm_sampling_top_p": 1.0,
                "sampling_top_p": 1.0,
                "dlm_sampling_top_k": top_k,
                "sampling_top_k": top_k,
            }
        )
        path = config_dir / f"{name}.json"
        path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        manifest["top_k"].append(name)

(config_dir / "manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
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
    --train-output-dir "${TRAIN_OUTPUT_DIR}" \
    --pipeline-log "${out_dir}/infer.log" \
    --evaluate-log "${out_dir}/evaluate.log" \
    --num-workers 8 2>&1 | tee "${out_dir}/evaluate.log"
  echo "[$(date '+%F %T')] DONE gpu=${gpu} ${name}" | tee -a "${RUN_ROOT}/status.tsv"
}

run_top_p() {
  for t in 1 0p8 0p6 0p4 0p2 0; do
    for p in 0p95 0p9 0p8; do
      run_one 0 "t${t}_p${p}"
    done
  done
}

run_top_k() {
  for t in 1 0p8 0p6 0p4 0p2 0; do
    for k in 16 32 64; do
      run_one 1 "t${t}_k${k}"
    done
  done
}

case "${1:-all}" in
  top-p) run_top_p ;;
  top-k) run_top_k ;;
  all)
    tmux new-session -d -s "mdlm_tp_${STAMP: -6}" "cd '${ROOT_DIR}' && RUN_ROOT='${RUN_ROOT}' bash '$0' top-p"
    tmux new-session -d -s "mdlm_tk_${STAMP: -6}" "cd '${ROOT_DIR}' && RUN_ROOT='${RUN_ROOT}' bash '$0' top-k"
    echo "RUN_ROOT=${RUN_ROOT}"
    echo "TOP_P_SESSION=mdlm_tp_${STAMP: -6}"
    echo "TOP_K_SESSION=mdlm_tk_${STAMP: -6}"
    ;;
  *) echo "usage: $0 [top-p|top-k|all]" >&2; exit 2 ;;
esac
