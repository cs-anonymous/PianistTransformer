#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUN_DIR="${1:?RUN_DIR is required}"
BASE_CONFIG="${2:?BASE_CONFIG is required}"
GPUS="${3:-0}"
OUT_ROOT="${4:-${RUN_DIR}/trunc_eval}"

IFS=',' read -ra GPU_LIST <<< "${GPUS}"
DET_GPU="${GPU_LIST[0]}"
SAMPLING_GPU="${GPU_LIST[$(( ${#GPU_LIST[@]} > 1 ? 1 : 0 ))]}"

latest_numeric_checkpoint() {
  local root="$1"
  find "${root}" -path '*/checkpoint-*' -type d 2>/dev/null \
    | awk -F'checkpoint-' '/checkpoint-[0-9]+$/ {print $2 " " $0}' \
    | sort -n | tail -n 1 | cut -d' ' -f2-
}

CHECKPOINT="$(latest_numeric_checkpoint "${RUN_DIR}/training")"
[[ -n "${CHECKPOINT}" ]] || { echo "Could not locate checkpoint under ${RUN_DIR}/training" >&2; exit 1; }
TRAIN_OUTPUT_DIR="$(dirname "${CHECKPOINT}")"
mkdir -p "${OUT_ROOT}/configs"

run_one() {
  local name="$1"
  local radius="$2"
  local out_dir="${OUT_ROOT}/${name}"
  local config="${OUT_ROOT}/configs/${name}.json"
  mkdir -p "${out_dir}"

  python - "${BASE_CONFIG}" "${config}" "${radius}" <<'PY'
import json
import sys
from pathlib import Path

src, dst, radius = sys.argv[1:4]
cfg = json.loads(Path(src).read_text(encoding="utf-8"))
radius = float(radius)
cfg["timing_sample_truncate_radius"] = radius
cfg["timing_sample_truncate_center"] = "mean"
cfg["dlm_timing_sample_truncate_radius"] = radius
cfg["dlm_timing_sample_truncate_center"] = "mean"
cfg["output_dir"] = str(Path(dst).parent.parent / Path(dst).stem / "training")
cfg["logging_dir"] = str(Path(dst).parent.parent / Path(dst).stem / "tf-logs")
Path(dst).write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY

  if [[ ! -s "${out_dir}/deterministic/prediction_manifest.json" ]]; then
    CUDA_VISIBLE_DEVICES="${DET_GPU}" PYTHONUNBUFFERED=1 python src/inference/infer_inr_testset.py \
      --config "${config}" \
      --checkpoint "${CHECKPOINT}" \
      --split test \
      --performance-dataset ASAP \
      --num-workers "${INFER_NUM_WORKERS:-8}" \
      --batch-size-windows "${INFER_BATCH_SIZE_WINDOWS:-8}" \
      --merge-mode "${MERGE_MODE:-continuation}" \
      --continuation-drop-ratio "${CONTINUATION_DROP_RATIO:-0.0}" \
      --device cuda \
      --protocol deterministic \
      --num-samples "${DET_NUM_SAMPLES:-1}" \
      --score-source-list "${INFER_SCORE_SOURCE_LIST:-data/asap_test_score_sources.txt}" \
      --deterministic-strategy "${DET_STRATEGY:-greedy}" \
      --output-dir "${out_dir}/deterministic"
  fi

  if [[ ! -s "${out_dir}/sampling/prediction_manifest.json" ]]; then
    CUDA_VISIBLE_DEVICES="${SAMPLING_GPU}" PYTHONUNBUFFERED=1 python src/inference/infer_inr_testset.py \
      --config "${config}" \
      --checkpoint "${CHECKPOINT}" \
      --split test \
      --performance-dataset ASAP \
      --num-workers "${INFER_NUM_WORKERS:-8}" \
      --batch-size-windows "${INFER_BATCH_SIZE_WINDOWS:-8}" \
      --merge-mode "${MERGE_MODE:-continuation}" \
      --continuation-drop-ratio "${CONTINUATION_DROP_RATIO:-0.0}" \
      --device cuda \
      --protocol sampling \
      --num-samples "${SAMPLING_NUM_SAMPLES:-1}" \
      --score-source-list "${INFER_SCORE_SOURCE_LIST:-data/asap_test_score_sources.txt}" \
      --deterministic-strategy "${DET_STRATEGY:-greedy}" \
      --output-dir "${out_dir}/sampling"
  fi

  if [[ ! -s "${out_dir}/summary.json" ]]; then
    PYTHONUNBUFFERED=1 python src/evaluate/summarize_inr_asap_pipeline.py \
      --deterministic-manifest "${out_dir}/deterministic/prediction_manifest.json" \
      --sampling-manifest "${out_dir}/sampling/prediction_manifest.json" \
      --output-json "${out_dir}/summary.json" \
      --output-plot "${out_dir}/asap_label_distribution.png" \
      --config "${config}" \
      --checkpoint "${CHECKPOINT}" \
      --train-output-dir "${TRAIN_OUTPUT_DIR}" \
      --pipeline-log "${RUN_DIR}/train.log" \
      --evaluate-log "${out_dir}/evaluate.log" \
      --num-workers "${METRIC_NUM_WORKERS:-8}"
  fi
}

run_one trunc-r0p05 0.05
run_one trunc-r0p10 0.10

echo "trunc_eval=${OUT_ROOT}"
