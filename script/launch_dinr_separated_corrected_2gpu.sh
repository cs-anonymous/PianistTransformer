#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BASE_CONFIG="${BASE_CONFIG:-results/inr_epr_pipeline/dinr_3exp_20260715_211309/separated_vocab_256_256/config.json}"
SOURCE_CONFIG="${SOURCE_CONFIG:-configs/local_generated/dinr_separated_corrected_abs256_dev256.json}"
RUN_DIR="${RUN_DIR_OVERRIDE:-results/inr_epr_pipeline/dinr_separated_corrected_$(date +%Y%m%d_%H%M%S)}"

python - "${BASE_CONFIG}" "${SOURCE_CONFIG}" <<'PY'
import json
import sys
from pathlib import Path

src, dst = map(Path, sys.argv[1:3])
cfg = json.loads(src.read_text(encoding="utf-8"))
cfg.update(
    {
        "run_name": "DINR-separated-corrected-absolute256-deviation256-m2p1",
        "epr_distribution": "dinr",
        "epr_timing_target": "floor_log_deviation",
        "dinr_vocabulary_mode": "separated",
        "dinr_timing_bins": 256,
        "dinr_zero_bin": 0,
        "dinr_timing_step": 9.0 / 255.0,
        "dinr_output_timing_bins": 256,
        "dinr_output_zero_bin": 170,
        "dinr_output_timing_step": 3.0 / 255.0,
        "dinr_deviation_min": -2.0,
        "dinr_deviation_max": 1.0,
        "dinr_zero_ioi_min": 0.0,
        "dinr_zero_ioi_max": 5.0,
        "prepared_sidecar_tag": "DINR_READY_ASAP",
        "use_prepared_sidecar": True,
    }
)
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY

CUDA_VISIBLE_DEVICES=1,2 \
CONFIG="${SOURCE_CONFIG}" \
RUN_DIR_OVERRIDE="${RUN_DIR}" \
BASE_NUM_TRAIN_EPOCHS=16 \
BASE_ASAP_ONLY=1 \
ADAPT_NUM_TRAIN_EPOCHS=0 \
BATCH_SIZE_PER_DEVICE=32 \
GLOBAL_BATCH_SIZE=64 \
DET_NUM_SAMPLES=1 \
SAMPLING_NUM_SAMPLES=1 \
bash script/run_inr_epr_pipeline.sh
