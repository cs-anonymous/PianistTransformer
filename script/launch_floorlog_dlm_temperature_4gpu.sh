#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

STAMP="${STAMP:-20260713_dlm_temperature_sweep}"
RUN_ROOT="${RUN_ROOT:-results/floorlog_dlm_temperature_4gpu/${STAMP}}"
BASE_CONFIG="${BASE_CONFIG:-results/floorlog_dlm_2gpu/20260712_asap_test/k8-b256-veldlm/config.json}"
CHECKPOINT="${CHECKPOINT:-results/floorlog_dlm_2gpu/20260712_asap_test/k8-b256-veldlm/training/floorlog_dlm_k8_b256_veldlm/checkpoint-1680}"
DET_MANIFEST="${DET_MANIFEST:-results/floorlog_dlm_2gpu/20260712_asap_test/k8-b256-veldlm/deterministic/prediction_manifest.json}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-results/floorlog_dlm_2gpu/20260712_asap_test/k8-b256-veldlm/training/floorlog_dlm_k8_b256_veldlm}"
CONFIG_DIR="${RUN_ROOT}/configs"

make_configs() {
  mkdir -p "${CONFIG_DIR}"
  python - "${BASE_CONFIG}" "${CONFIG_DIR}" <<'PY'
import json
import sys
from pathlib import Path

base_path, config_dir = map(Path, sys.argv[1:3])
base = json.loads(base_path.read_text(encoding="utf-8"))
tasks = {
    "temperature-1p0": (1.0, "sample"),
    "temperature-0p75": (0.75, "sample"),
    "temperature-0p5": (0.5, "sample"),
    "temperature-0p35": (0.35, "sample"),
    "temperature-0p25": (0.25, "sample"),
    "temperature-0p1": (0.1, "sample"),
    "mean": (1.0, "mean"),
    "argmax": (1.0, "greedy"),
}
config_dir.mkdir(parents=True, exist_ok=True)
for name, (temperature, strategy) in tasks.items():
    cfg = dict(base)
    cfg.update({
        "dlm_sampling_temperature": temperature,
        "timing_sample_truncate_radius": 0.0,
        "dlm_timing_sample_truncate_radius": 0.0,
        "timing_sample_shrink_mode": "none",
        "timing_sample_shrink_factor": 1.0,
        "timing_sample_shrink_radius": 0.0,
        "run_name": f"k8-b256-veldlm_{name}",
        "output_dir": f"unused/k8-b256-veldlm/{name}/training",
        "logging_dir": f"unused/k8-b256-veldlm/{name}/tf-logs",
    })
    (config_dir / f"{name}.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
(config_dir / "task_map.json").write_text(
    json.dumps(
        {name: {"temperature": t, "sampling_strategy": s} for name, (t, s) in tasks.items()},
        indent=2,
    ) + "\n",
    encoding="utf-8",
)
PY
}

run_task() {
  local gpu="$1"
  local name="$2"
  local strategy="$3"
  local out_dir="${RUN_ROOT}/${name}"
  local config="${CONFIG_DIR}/${name}.json"
  mkdir -p "${out_dir}"

  echo "[$(date '+%F %T')] GPU${gpu} start ${name} strategy=${strategy}"
  if [[ ! -s "${out_dir}/sampling/prediction_manifest.json" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 python src/inference/infer_inr_testset.py \
      --config "${config}" \
      --checkpoint "${CHECKPOINT}" \
      --split test \
      --performance-dataset ASAP \
      --num-workers "${INFER_NUM_WORKERS:-8}" \
      --batch-size-windows "${INFER_BATCH_SIZE_WINDOWS:-8}" \
      --merge-mode continuation \
      --continuation-drop-ratio 0.0 \
      --device cuda \
      --protocol sampling \
      --sampling-strategy "${strategy}" \
      --num-samples 1 \
      --seed "${INFER_SEED:-42}" \
      --score-source-list data/asap_test_score_sources.txt \
      --output-dir "${out_dir}/sampling"
  fi

  if [[ ! -s "${out_dir}/summary.json" ]]; then
    PYTHONUNBUFFERED=1 python src/evaluate/summarize_inr_asap_pipeline.py \
      --deterministic-manifest "${DET_MANIFEST}" \
      --sampling-manifest "${out_dir}/sampling/prediction_manifest.json" \
      --output-json "${out_dir}/summary.json" \
      --output-plot "${out_dir}/asap_label_distribution.png" \
      --config "${config}" \
      --checkpoint "${CHECKPOINT}" \
      --train-output-dir "${TRAIN_OUTPUT_DIR}" \
      --pipeline-log "${RUN_ROOT}/train.reused.log" \
      --evaluate-log "${out_dir}/evaluate.log" \
      --num-workers "${METRIC_NUM_WORKERS:-8}"
  fi
  echo "[$(date '+%F %T')] GPU${gpu} done ${name}"
}

if [[ "${1:-}" == "--worker" ]]; then
  gpu="$2"
  shift 2
  while (( "$#" )); do
    name="$1"
    strategy="$2"
    shift 2
    run_task "${gpu}" "${name}" "${strategy}"
  done
  exit 0
fi

make_configs
mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/pids"
: > "${RUN_ROOT}/workers.tsv"

launch_worker() {
  local gpu="$1"
  shift
  local log="${RUN_ROOT}/logs/gpu${gpu}.log"
  setsid bash "$0" --worker "${gpu}" "$@" > "${log}" 2>&1 < /dev/null &
  local pid=$!
  echo "${pid}" > "${RUN_ROOT}/pids/gpu${gpu}.pid"
  printf 'GPU%s\t%s\t%s\n' "${gpu}" "${pid}" "${log}" | tee -a "${RUN_ROOT}/workers.tsv"
}

launch_worker 0 temperature-1p0 sample temperature-0p75 sample
launch_worker 1 temperature-0p5 sample temperature-0p35 sample
launch_worker 2 temperature-0p25 sample temperature-0p1 sample
launch_worker 3 mean mean argmax greedy

echo "RUN_ROOT=${RUN_ROOT}"
