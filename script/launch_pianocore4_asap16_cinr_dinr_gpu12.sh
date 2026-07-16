#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/inr_epr_pipeline/pianocore3_asap16_cinr_dinr_${STAMP}}"
CONFIG_DIR="${RUN_ROOT}/configs"
BASE_CONFIG="results/inr_epr_pipeline/dinr_separated_corrected_20260716_004453/config.json"
mkdir -p "${CONFIG_DIR}"

python - "${BASE_CONFIG}" "${CONFIG_DIR}" <<'PY'
import json, sys
from pathlib import Path

base = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2])
common = dict(base)
common.update({
    "num_train_epochs": 3.0,
    "max_train_epochs": 3.0,
    "train_performance_dataset": None,
    "eval_performance_dataset": None,
    "prepared_sidecar_tag": None,
    "sampling_top_p": 0.95,
    "timing_sample_shrink_mode": "none",
    "timing_sample_truncate_radius": 0.0,
    "dlm_timing_sample_truncate_radius": 0.0,
    "slot_version": "slot8",
    "slot_fusion": "mlp",
    "dinr_deviation_min": -2.0,
    "dinr_deviation_max": 1.0,
    "dlm_ioi_nonzero_min": -2.0,
    "dlm_ioi_nonzero_max": 1.0,
    "dlm_duration_min": -2.0,
    "dlm_duration_max": 1.0,
    "tail_mask_ioi_min": -2.0,
    "tail_mask_ioi_max": 1.0,
    "tail_mask_duration_min": -2.0,
    "tail_mask_duration_max": 1.0,
})

cinr = dict(common)
cinr.update({
    "run_name": "CINR-DLM-k1-pianocore3-asap16-noclamp-topp95-t08",
    "epr_distribution": "dlm",
    "velocity_distribution": "dlm",
    "dlm_components": 1,
    "epr_mixture_components": 1,
    "dlm_ioi_zero_inflated": False,
    "dlm_pedal_zero_one_inflated": False,
    "dlm_ioi_nonzero_min": -2.0,
    "dlm_ioi_nonzero_max": 1.0,
    "dlm_duration_min": -2.0,
    "dlm_duration_max": 1.0,
    "dlm_ioi_zero_min": 0.0,
    "dlm_ioi_zero_max": 5.0,
    "dlm_timing_scale_parameterization": "softplus_unbounded",
    "dlm_velocity_scale_parameterization": "softplus_unbounded",
    "dlm_sampling_temperature": 0.8,
    "dlm_sampling_top_p": 0.95,
})

dinr = dict(common)
dinr.update({
    "run_name": "DINR-separated256-pianocore3-asap16-topp95-t08",
    "epr_distribution": "dinr",
    "dinr_sampling_temperature": 0.8,
    "dinr_sampling_top_p": 0.95,
})

for name, cfg in [("cinr_dlm_k1_noclamp.json", cinr), ("dinr_separated256.json", dinr)]:
    # The pipeline removes dataset filters for base training and adds ASAP filters for adapt.
    for key in ("train_performance_dataset", "eval_performance_dataset", "prepared_sidecar_tag"):
        if cfg.get(key) is None:
            cfg.pop(key, None)
    (out / name).write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
PY

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1 RUN_ROOT=${RUN_ROOT}"
  exit 0
fi

launch_one() {
  local name="$1" gpu="$2" config="$3"
  local run_dir="${RUN_ROOT}/${name}"
  local session="pc3_a16_${name}_${STAMP: -6}"
  mkdir -p "${run_dir}"
  tmux new-session -d -s "${session}" \
    "cd '${ROOT_DIR}' && env CUDA_VISIBLE_DEVICES='${gpu}' CONFIG='${config}' RUN_DIR_OVERRIDE='${run_dir}' BASE_ASAP_ONLY=0 BASE_NUM_TRAIN_EPOCHS=3 ADAPT_NUM_TRAIN_EPOCHS=16 BASE_PREPARED_SIDECAR_TAG=FLOORLOG_NOMUS_SCORESPAN ADAPT_PREPARED_SIDECAR_TAG=ASAP_FLOORLOG_NOMUS_SCORESPAN BATCH_SIZE_PER_DEVICE=32 GLOBAL_BATCH_SIZE=64 DET_NUM_SAMPLES=1 SAMPLING_NUM_SAMPLES=1 INFER_NUM_WORKERS=8 METRIC_NUM_WORKERS=8 INFER_BATCH_SIZE_WINDOWS=8 INFER_SCORE_SOURCE_LIST=data/asap_test_score_sources.txt EVAL_CHECKPOINT_MODE=best RESUME_FROM_LATEST_CHECKPOINT=0 MERGE_MODE=continuation CONTINUATION_DROP_RATIO=0.0 bash script/run_inr_epr_pipeline.sh > '${run_dir}/launcher.log' 2>&1"
  printf '%s\tGPU%s\t%s\t%s\n' "${name}" "${gpu}" "${session}" "${run_dir}" | tee -a "${RUN_ROOT}/processes.tsv"
}

: > "${RUN_ROOT}/processes.tsv"
launch_one cinr_dlm_k1 1 "${CONFIG_DIR}/cinr_dlm_k1_noclamp.json"
launch_one dinr_logits 2 "${CONFIG_DIR}/dinr_separated256.json"
echo "RUN_ROOT=${RUN_ROOT}"
