#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/floorlog_dlm_constraints_4gpu/${STAMP}}"
CONFIG_DIR="${RUN_ROOT}/configs"
BASE_CONFIG="${BASE_CONFIG:-results/floorlog_dlm_2gpu/20260712_asap_test/k8-b256-veldlm/config.json}"
mkdir -p "${CONFIG_DIR}"

python - "${BASE_CONFIG}" "${CONFIG_DIR}" <<'PY'
import json
import sys
from pathlib import Path

base_path, config_dir = Path(sys.argv[1]), Path(sys.argv[2])
base = json.loads(base_path.read_text(encoding="utf-8"))
for key in ("resume_path", "resume_from_checkpoint"):
    base.pop(key, None)
base.update({
    "pretrained_model": None,
    "load_pianoformer_backbone": False,
    "continuous_dim": 5,
    "input_continuous_dim": 10,
    "score_input_continuous_dim": 10,
    "decoder_input_continuous_dim": 10,
    "output_continuous_dim": 5,
    "pedal_representation": "start_valley",
    "pedal_valley_pos_weight": 28.0,
    "num_train_epochs": 16.0,
    "max_train_epochs": 16.0,
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 32,
    "global_batch_size": 64,
    "gradient_accumulation_steps": 2,
    "overwrite_output_dir": True,
    "resume_trainer_state": False,
    "adapt_on_asap_after_train": False,
    "auto_rollout_eval_after_train": False,
    "timing_sample_shrink_mode": "none",
    "timing_sample_shrink_factor": 1.0,
    "timing_sample_shrink_radius": 0.0,
    "timing_sample_truncate_radius": 0.0,
    "dlm_timing_sample_truncate_radius": 0.0,
    "dlm_tail_radius": 0.05,
})

variants = {
    "dlm-base": {
        "dlm_timing_scale_parameterization": "legacy_clamp",
        "dlm_scale_min": 1e-3,
        "dlm_scale_max": 10.0,
        "dlm_tail_loss_lambda": 0.0,
    },
    "dlm-scale-s001-s02": {
        "dlm_timing_scale_parameterization": "bounded_sigmoid",
        "dlm_scale_min": 1e-3,
        "dlm_scale_max": 10.0,
        "dlm_timing_scale_min": 1e-3,
        "dlm_timing_scale_max": 2e-2,
        "dlm_tail_loss_lambda": 0.0,
    },
    "dlm-tail-r005-l1": {
        "dlm_timing_scale_parameterization": "legacy_clamp",
        "dlm_scale_min": 1e-3,
        "dlm_scale_max": 10.0,
        "dlm_tail_loss_lambda": 1.0,
    },
    "dlm-scale-tail-r005-l1": {
        "dlm_timing_scale_parameterization": "bounded_sigmoid",
        "dlm_scale_min": 1e-3,
        "dlm_scale_max": 10.0,
        "dlm_timing_scale_min": 1e-3,
        "dlm_timing_scale_max": 2e-2,
        "dlm_tail_loss_lambda": 1.0,
    },
}
for name, overrides in variants.items():
    cfg = dict(base)
    cfg.update(overrides)
    cfg["run_name"] = name.replace("-", "_")
    cfg["output_dir"] = f"unused/{name}/training"
    cfg["logging_dir"] = f"unused/{name}/tf-logs"
    (config_dir / f"{name}.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
(config_dir / "manifest.json").write_text(
    json.dumps({"base_config": str(base_path), "variants": variants}, indent=2) + "\n",
    encoding="utf-8",
)
PY

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1 configs=${CONFIG_DIR}"
  exit 0
fi

launch_one() {
  local gpu="$1" name="$2"
  local run_dir="${RUN_ROOT}/${name}"
  local log_path="${run_dir}/launcher.log"
  mkdir -p "${run_dir}"
  setsid env CUDA_VISIBLE_DEVICES="${gpu}" \
    CONFIG="${CONFIG_DIR}/${name}.json" RUN_DIR_OVERRIDE="${run_dir}" \
    BASE_ASAP_ONLY=1 BASE_NUM_TRAIN_EPOCHS=16 ADAPT_NUM_TRAIN_EPOCHS=0 \
    BATCH_SIZE_PER_DEVICE=32 GLOBAL_BATCH_SIZE=64 \
    DET_NUM_SAMPLES=1 SAMPLING_NUM_SAMPLES=1 \
    INFER_NUM_WORKERS=8 METRIC_NUM_WORKERS=8 INFER_BATCH_SIZE_WINDOWS=8 \
    INFER_SCORE_SOURCE_LIST=data/asap_test_score_sources.txt \
    EVAL_CHECKPOINT_MODE=latest RESUME_FROM_LATEST_CHECKPOINT=0 \
    MERGE_MODE=continuation CONTINUATION_DROP_RATIO=0.0 \
    bash script/run_inr_epr_pipeline.sh >"${log_path}" 2>&1 < /dev/null &
  local pid=$!
  printf '%s\tGPU%s\tPID%s\t%s\n' "${name}" "${gpu}" "${pid}" "${log_path}" \
    | tee -a "${RUN_ROOT}/processes.tsv"
}

launch_one 0 dlm-base
launch_one 1 dlm-scale-s001-s02
launch_one 2 dlm-tail-r005-l1
launch_one 3 dlm-scale-tail-r005-l1
echo "RUN_ROOT=${RUN_ROOT}"
