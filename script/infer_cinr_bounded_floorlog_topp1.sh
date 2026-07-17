#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SOURCE_RUN="${SOURCE_RUN:-results/inr_epr_pipeline/cinr_bounded_floorlog_2gpu_20260717_195302}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/inr_epr_pipeline/cinr_bounded_floorlog_topp1_${STAMP}}"
SOURCE_CONFIG="${SOURCE_RUN}/config.json"
CONFIG="${RUN_ROOT}/config.json"
CHECKPOINT="${SOURCE_RUN}/training/cinr_bounded_floorlog_2gpu/checkpoint-best"
mkdir -p "${RUN_ROOT}"

python - "${SOURCE_CONFIG}" "${CONFIG}" "${RUN_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

src, dst, run_root = map(Path, sys.argv[1:4])
cfg = json.loads(src.read_text(encoding="utf-8"))
cfg["output_dir"] = str(run_root / "training")
cfg["logging_dir"] = str(run_root / "tf-logs")
cfg["sampling_top_p"] = 1.0
cfg["sampling_top_k"] = 0
cfg["dlm_sampling_top_p"] = 1.0
cfg["dlm_sampling_top_k"] = 0
cfg["dlm_sampling_temperature"] = 1.0
cfg["dinr_sampling_top_p"] = 1.0
cfg["dinr_sampling_top_k"] = 0
cfg["dinr_sampling_temperature"] = 1.0
dst.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" \
  CONFIG="${CONFIG}" \
  RUN_DIR_OVERRIDE="${RUN_ROOT}" \
  BASE_ASAP_ONLY=1 \
  ADAPT_NUM_TRAIN_EPOCHS=0 \
  PIPELINE_STAGE_START=infer \
  BASE_CHECKPOINT_OVERRIDE="${CHECKPOINT}" \
  DET_NUM_SAMPLES=1 \
  SAMPLING_NUM_SAMPLES=1 \
  INFER_NUM_WORKERS=8 \
  METRIC_NUM_WORKERS=8 \
  INFER_BATCH_SIZE_WINDOWS=8 \
  INFER_SCORE_SOURCE_LIST=data/asap_test_score_sources.txt \
  EVAL_CHECKPOINT_MODE=best \
  RESUME_FROM_LATEST_CHECKPOINT=0 \
  bash script/run_inr_epr_pipeline.sh > "${RUN_ROOT}/launcher.log" 2>&1

echo "RUN_ROOT=${RUN_ROOT}"
