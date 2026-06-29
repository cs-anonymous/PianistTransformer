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
  python - "${src}" "${dst}" "${mode}" "${scale}" "${RUN_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

src, dst, mode, scale, run_root = sys.argv[1:6]
scale_value = float(scale)
name = f"inr0624_epr_mln3_{mode}_mslog_s{int(scale_value)}"

with open(src, encoding="utf-8") as file:
    cfg = json.load(file)

cfg["note_embedding_mode"] = mode
cfg["input_continuous_dim"] = 24
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
cfg["output_dir"] = f"./{run_root}/train_ddp/{name}/model/"
cfg["logging_dir"] = f"./{run_root}/train_ddp/{name}/tf-logs/"

Path(dst).parent.mkdir(parents=True, exist_ok=True)
with open(dst, "w", encoding="utf-8") as file:
    json.dump(cfg, file, indent=2, ensure_ascii=False)
    file.write("\n")
PY
}

make_config "${BASE_CINE}" "${CONFIG_DIR}/inr0624_epr_mln3_cine_mslog_s10.json" cine 10
make_config "${BASE_CINE}" "${CONFIG_DIR}/inr0624_epr_mln3_cine_mslog_s50.json" cine 50
make_config "${BASE_SINE}" "${CONFIG_DIR}/inr0624_epr_mln3_sine_mslog_s10.json" sine 10
make_config "${BASE_SINE}" "${CONFIG_DIR}/inr0624_epr_mln3_sine_mslog_s50.json" sine 50

CONFIGS=(
  "${CONFIG_DIR}/inr0624_epr_mln3_cine_mslog_s10.json"
  "${CONFIG_DIR}/inr0624_epr_mln3_cine_mslog_s50.json"
  "${CONFIG_DIR}/inr0624_epr_mln3_sine_mslog_s10.json"
  "${CONFIG_DIR}/inr0624_epr_mln3_sine_mslog_s50.json"
)
GPUS=(0 1 2 3)

echo "Launch layout:"
echo "  GPU0: cine, s=10"
echo "  GPU1: cine, s=50"
echo "  GPU2: sine, s=10"
echo "  GPU3: sine, s=50"
echo "  per_device_train_batch_size=16, gradient_accumulation_steps=2, effective single-process global_bs=32"

for idx in "${!CONFIGS[@]}"; do
  config="${CONFIGS[$idx]}"
  gpu="${GPUS[$idx]}"
  name="$(basename "${config}" .json)"
  run_dir="${RUN_ROOT}/${name}"
  log="${LOG_DIR}/${name}_$(date +%Y%m%d_%H%M%S).log"
  echo "launch ${name} on GPU ${gpu}; log=${log}"
  setsid env CUDA_VISIBLE_DEVICES="${gpu}" CONFIG="${config}" RUN_DIR_OVERRIDE="${run_dir}" \
    bash script/run_inr_pipeline.sh >"${log}" 2>&1 < /dev/null &
done

echo "All launches submitted."
