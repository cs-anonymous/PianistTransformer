#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/inr_epr_pipeline/cinr_bounded_floorlog_2gpu_${STAMP}}"
BASE_CONFIG="results/inr_epr_pipeline/unified_musical_baselines_20260717_151510/configs/cinr_bounded_5pct.json"
CONFIG="${RUN_ROOT}/config.json"
mkdir -p "${RUN_ROOT}"

python - "${BASE_CONFIG}" "${CONFIG}" "${RUN_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

src, dst, run_root = map(Path, sys.argv[1:4])
cfg = json.loads(src.read_text(encoding="utf-8"))
cfg["run_name"] = "cinr_bounded_floorlog_2gpu"
cfg["output_dir"] = str(run_root / "training")
cfg["logging_dir"] = str(run_root / "tf-logs")
cfg["timing_control_mode"] = "floor_log"
cfg["seed"] = 42
cfg["num_train_epochs"] = 16.0
cfg["max_train_epochs"] = 16.0
cfg["sampling_top_p"] = 0.90
cfg["sampling_top_k"] = 0
cfg["dlm_sampling_temperature"] = 1.0
cfg["dlm_sampling_top_p"] = 0.90
cfg["dlm_sampling_top_k"] = 0
cfg["dinr_sampling_temperature"] = 1.0
cfg["dinr_sampling_top_p"] = 0.90
cfg["dinr_sampling_top_k"] = 0
cfg.pop("resume_path", None)
cfg.pop("resume_from_checkpoint", None)
dst.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}" \
  CONFIG="${CONFIG}" \
  RUN_DIR_OVERRIDE="${RUN_ROOT}" \
  BASE_ASAP_ONLY=1 \
  BASE_NUM_TRAIN_EPOCHS=16 \
  ADAPT_NUM_TRAIN_EPOCHS=0 \
  BATCH_SIZE_PER_DEVICE=32 \
  GLOBAL_BATCH_SIZE=64 \
  DET_NUM_SAMPLES=1 \
  SAMPLING_NUM_SAMPLES=2 \
  INFER_NUM_WORKERS=8 \
  METRIC_NUM_WORKERS=8 \
  INFER_BATCH_SIZE_WINDOWS=8 \
  INFER_SCORE_SOURCE_LIST=data/asap_test_score_sources.txt \
  EVAL_CHECKPOINT_MODE=best \
  RESUME_FROM_LATEST_CHECKPOINT=0 \
  MERGE_MODE=continuation \
  CONTINUATION_DROP_RATIO=0.0 \
  bash script/run_inr_epr_pipeline.sh > "${RUN_ROOT}/launcher.log" 2>&1

echo "RUN_ROOT=${RUN_ROOT}"
