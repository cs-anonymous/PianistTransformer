#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODEL_DIR="models/hf_pianist_transformer_rendering"
RUN_DIR="results/pt_official_redownload_20260711_cheap15"
mkdir -p "${RUN_DIR}/shard0" "${RUN_DIR}/shard1" "${RUN_DIR}/sampling"
exec > >(tee -a "${RUN_DIR}/two_gpu.log") 2>&1

run_shard() {
  local gpu="$1" shard="$2"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 python src/inference/infer_pt_testset.py \
    --model-path "${MODEL_DIR}" \
    --metadata data/ASAP_processed/metadata.generated_json.csv --midi-root data/ASAP_processed \
    --split test --performance-dataset ASAP \
    --output-dir "${RUN_DIR}/shard${shard}" --device cuda \
    --batch-size 8 --num-workers 4 --protocol sampling --num-samples 1 \
    --temperature 1.0 --top-p 0.95 --overlap-ratio 0.125 \
    --max-context-length 4096 --seed 42 --num-shards 2 --shard-index "${shard}"
}

run_shard 0 0 & pid0=$!
run_shard 1 1 & pid1=$!
wait "${pid0}"
wait "${pid1}"

python - "${RUN_DIR}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
parts = [json.loads((root / f"shard{i}" / "prediction_manifest.json").read_text()) for i in range(2)]
merged = dict(parts[0])
merged["num_shards"] = 1
merged["shard_index"] = 0
merged["items"] = sorted(parts[0]["items"] + parts[1]["items"], key=lambda x: x["score_source"])
(root / "sampling" / "prediction_manifest.json").write_text(
    json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
print(f"merged_scores={len(merged['items'])}")
PY

python src/evaluate/evaluate_inr_saved_midis.py \
  --prediction-manifest "${RUN_DIR}/sampling/prediction_manifest.json" \
  --score-source-list data/cheap15_score_sources.txt --num-workers 8 \
  --output-json "${RUN_DIR}/sampling/score_level_pp_pn_cheap15.json"
echo "END $(date '+%F %T')"
