#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUN_ROOT="${RUN_ROOT:-results/inr0624_epr_logscale_4gpu}"
CONFIG_DIR="${RUN_ROOT}/configs"
LOG_DIR="${RUN_ROOT}/launcher_logs"
mkdir -p "${CONFIG_DIR}" "${LOG_DIR}"

BASE_CINE="configs/inr0624_epr_mln3_cine_mslog.json"
BASE_SINE="configs/inr0624_epr_mln3_sine_mslog.json"

make_config() {
  local src="$1"
  local dst="$2"
  local mode="$3"
  local scale="$4"
  local seed="$5"
  local timing_sampling="$6"
  python - "${src}" "${dst}" "${mode}" "${scale}" "${seed}" "${timing_sampling}" "${RUN_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

src, dst, mode, scale, seed, timing_sampling, run_root = sys.argv[1:8]
scale_value = float(scale)
seed_value = int(seed)
name = f"inr0624_epr_mln3_{mode}_mslog_s{int(scale_value)}_{timing_sampling}_seed{seed_value}"

with open(src, encoding="utf-8") as file:
    cfg = json.load(file)

cfg["note_embedding_mode"] = mode
cfg["input_continuous_dim"] = 23
cfg["output_continuous_dim"] = 5
cfg["epr_timing_target"] = "log_deviation"
cfg["timing_control_mode"] = "log_scaled"
cfg["timing_log_scale"] = scale_value
cfg["use_timing_scale_bit"] = False
cfg["timing_input_normalization"] = f"log1p_t_over_{int(scale_value)}_5000"
cfg["adapt_num_train_epochs"] = 4
cfg["per_device_train_batch_size"] = 16
cfg["per_device_eval_batch_size"] = 16
cfg["gradient_accumulation_steps"] = 2
cfg["timing_sampling_method"] = timing_sampling
cfg["seed"] = seed_value
cfg["output_dir"] = f"./{run_root}/train_ddp/{name}/model/"
cfg["logging_dir"] = f"./{run_root}/train_ddp/{name}/tf-logs/"

Path(dst).parent.mkdir(parents=True, exist_ok=True)
with open(dst, "w", encoding="utf-8") as file:
    json.dump(cfg, file, indent=2, ensure_ascii=False)
    file.write("\n")
PY
}

make_config "${BASE_CINE}" "${CONFIG_DIR}/inr0624_epr_mln3_cine_mslog_s50_bias_correction_seed42.json" cine 50 42 bias_correction
make_config "${BASE_CINE}" "${CONFIG_DIR}/inr0624_epr_mln3_cine_mslog_s50_calibrated_residual_seed43.json" cine 50 43 calibrated_residual
make_config "${BASE_SINE}" "${CONFIG_DIR}/inr0624_epr_mln3_sine_mslog_s50_bias_correction_seed42.json" sine 50 42 bias_correction
make_config "${BASE_SINE}" "${CONFIG_DIR}/inr0624_epr_mln3_sine_mslog_s50_calibrated_residual_seed43.json" sine 50 43 calibrated_residual

CONFIGS=(
  "${CONFIG_DIR}/inr0624_epr_mln3_cine_mslog_s50_bias_correction_seed42.json"
  "${CONFIG_DIR}/inr0624_epr_mln3_cine_mslog_s50_calibrated_residual_seed43.json"
  "${CONFIG_DIR}/inr0624_epr_mln3_sine_mslog_s50_bias_correction_seed42.json"
  "${CONFIG_DIR}/inr0624_epr_mln3_sine_mslog_s50_calibrated_residual_seed43.json"
)
TIMING_SAMPLING_METHODS=(bias_correction calibrated_residual bias_correction calibrated_residual)
GPUS=(0 1 2 3)

echo "Launch layout:"
echo "  GPU0: cine, s=50, bias_correction, seed=42"
echo "  GPU1: cine, s=50, calibrated_residual, seed=43"
echo "  GPU2: sine, s=50, bias_correction, seed=42"
echo "  GPU3: sine, s=50, calibrated_residual, seed=43"
echo "  per_device_train_batch_size=16, gradient_accumulation_steps=2, effective single-process global_bs=32"

for idx in "${!CONFIGS[@]}"; do
  config="${CONFIGS[$idx]}"
  gpu="${GPUS[$idx]}"
  timing_sampling="${TIMING_SAMPLING_METHODS[$idx]}"
  name="$(basename "${config}" .json)"
  run_dir="${RUN_ROOT}/${name}"
  log="${LOG_DIR}/${name}_$(date +%Y%m%d_%H%M%S).log"
  echo "launch ${name} on GPU ${gpu}; log=${log}"
  setsid env CUDA_VISIBLE_DEVICES="${gpu}" CONFIG="${config}" RUN_DIR_OVERRIDE="${run_dir}" TIMING_SAMPLING_METHOD="${timing_sampling}" \
    bash script/run_inr_pipeline.sh >"${log}" 2>&1 < /dev/null &
done

echo "All launches submitted."
