#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/floorlog_tight_inflated_dlm_2gpu/${STAMP}}"
CONFIG_DIR="${RUN_ROOT}/configs"
BASE_CONFIG="${BASE_CONFIG:-results/floorlog_tight_targettail_vs_k1std_2x2gpu/20260713_tight_targettail_vs_k1std_v1/dlm-k1-bounded-std-s012/config.json}"
mkdir -p "${CONFIG_DIR}"

python - "${BASE_CONFIG}" "${CONFIG_DIR}" <<'PY'
import json
import sys
from pathlib import Path

base_path, config_dir = Path(sys.argv[1]), Path(sys.argv[2])
base = json.loads(base_path.read_text(encoding="utf-8"))
for key in ("resume_path", "resume_from_checkpoint"):
    base.pop(key, None)

cfg = dict(base)
loss_weights = dict(cfg.get("loss_weights") or {})
loss_weights.update({"ioi": 1.0, "duration": 1.0, "velocity": 1.0, "pedal": 1.0})
cfg.update(
    {
        "pretrained_model": None,
        "load_pianoformer_backbone": False,
        "pedal_representation": "start",
        "pedal_distribution": "dlm",
        "velocity_distribution": "dlm",
        "dlm_ioi_zero_inflated": True,
        "dlm_pedal_zero_one_inflated": True,
        "dlm_pedal_inflated_eps": 0.5,
        "loss_weights": loss_weights,
        "num_train_epochs": 16.0,
        "max_train_epochs": 16.0,
        "per_device_train_batch_size": 32,
        "per_device_eval_batch_size": 32,
        "global_batch_size": 64,
        "gradient_accumulation_steps": 1,
        "overwrite_output_dir": True,
        "resume_trainer_state": False,
        "adapt_on_asap_after_train": False,
        "auto_rollout_eval_after_train": False,
    }
)

name = "dlm-k1-inflated-ioi-pedalw1"
cfg["run_name"] = "floorlog_tight_dlm_k1_inflated_ioi_pedalw1"
cfg["output_dir"] = f"unused/{name}/training"
cfg["logging_dir"] = f"unused/{name}/tf-logs"
(config_dir / f"{name}.json").write_text(
    json.dumps(cfg, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

(config_dir / "manifest.json").write_text(
    json.dumps(
        {
            "base_config": str(base_path),
            "training": {
                "from_scratch": True,
                "epochs": 16,
                "per_device_batch": 32,
                "global_batch": 64,
                "mask": "tight_mask",
                "pedal_representation": "start",
                "loss_weights": loss_weights,
                "dlm_ioi_zero_inflated": True,
                "dlm_pedal_zero_one_inflated": True,
                "dlm_pedal_inflated_eps": 0.5,
            },
            "variants": [name],
            "gpu_assignment": {name: [0, 1]},
        },
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)
PY

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1 configs=${CONFIG_DIR}"
  exit 0
fi

name="dlm-k1-inflated-ioi-pedalw1"
run_dir="${RUN_ROOT}/${name}"
log_path="${run_dir}/launcher.log"
mkdir -p "${run_dir}"
: > "${RUN_ROOT}/processes.tsv"
setsid env CUDA_VISIBLE_DEVICES="${GPUS:-0,1}" \
  CONFIG="${CONFIG_DIR}/${name}.json" RUN_DIR_OVERRIDE="${run_dir}" \
  BASE_ASAP_ONLY=1 BASE_NUM_TRAIN_EPOCHS=16 ADAPT_NUM_TRAIN_EPOCHS=0 \
  BATCH_SIZE_PER_DEVICE=32 GLOBAL_BATCH_SIZE=64 \
  DET_NUM_SAMPLES=1 SAMPLING_NUM_SAMPLES=1 \
  INFER_NUM_WORKERS=8 METRIC_NUM_WORKERS=8 INFER_BATCH_SIZE_WINDOWS=8 \
  INFER_SCORE_SOURCE_LIST=data/asap_test_score_sources.txt \
  EVAL_CHECKPOINT_MODE=latest RESUME_FROM_LATEST_CHECKPOINT=0 \
  MERGE_MODE=continuation CONTINUATION_DROP_RATIO=0.0 \
  bash script/run_inr_epr_pipeline.sh >"${log_path}" 2>&1 < /dev/null &
pid=$!
printf '%s\tGPUs%s\tPID%s\t%s\t%s\n' "${name}" "${GPUS:-0,1}" "${pid}" "${run_dir}" "${log_path}" \
  | tee -a "${RUN_ROOT}/processes.tsv"
echo "RUN_ROOT=${RUN_ROOT}"
