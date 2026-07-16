#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/inr_epr_pipeline/asaponly_discrete256_k2_slot8_${STAMP}}"
CONFIG_DIR="${RUN_ROOT}/configs"
BASE_CONFIG="results/inr_epr_pipeline/asaponly_matched_cinr_dinr_20260716_143813/cinr_dlm_k1/config.json"
mkdir -p "${CONFIG_DIR}"

python - "${BASE_CONFIG}" "${CONFIG_DIR}" <<'PY'
import json
import sys
from pathlib import Path

base = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2])
common = dict(base)
common.update({
    "num_train_epochs": 16.0,
    "max_train_epochs": 16.0,
    "resume_from_checkpoint": None,
    "slot_version": "slot8",
    "slot_dim": 128,
    "slot_fusion": "mlp",
    "musical_feature_mode": "musical51",
    "disable_musical_features": False,
    "epr_mixture_components": 2,
    "dlm_components": 2,
    "dlm_timing_bins": 256,
    "dlm_velocity_bins": 256,
    "sampling_top_p": 0.95,
    "dlm_sampling_temperature": 0.8,
    "dlm_sampling_top_p": 0.95,
    "timing_sample_shrink_mode": "none",
    "timing_sample_truncate_radius": 0.0,
    "dlm_timing_sample_truncate_radius": 0.0,
    "bounded_floorlog_support": True,
    "train_performance_dataset": "ASAP",
    "eval_performance_dataset": "ASAP",
    "eval_include_all_performance_dataset": "ASAP",
    "prepared_sidecar_tag": "ASAP_FLOORLOG_NOMUS_SCORESPAN",
    "pedal_distribution": "point",
})

experiments = {
    "discrete_ln_k2": {
        "run_name": "ASAP-slot8-discreteLN2-256-T08p95-safe-scale",
        "epr_distribution": "discrete_logistic_normal",
        "velocity_distribution": "discrete_logistic_normal",
        "logistic_normal_sigma_min": 0.001,
    },
    "discrete_beta_k2": {
        "run_name": "ASAP-slot8-discreteBeta2-256-T08p95-safe-scale",
        "epr_distribution": "discrete_beta",
        "velocity_distribution": "discrete_beta",
        "mixture_beta_parameterization": "mu_kappa",
        "mixture_beta_kappa_min": 0.001,
        "beta_alpha_min": 0.0001,
    },
    "truncated_logistic_k2": {
        "run_name": "ASAP-slot8-truncatedLogistic2-256-T08p95-safe-scale",
        "epr_distribution": "truncated_logistic",
        "velocity_distribution": "truncated_logistic",
        "dlm_scale_min": 0.001,
        "dlm_timing_scale_parameterization": "softplus_unbounded",
        "dlm_velocity_scale_parameterization": "softplus_unbounded",
    },
}

for name, overrides in experiments.items():
    cfg = dict(common)
    cfg.update(overrides)
    (out / f"{name}.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False) + "\n"
    )
PY

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1 RUN_ROOT=${RUN_ROOT}"
  exit 0
fi

launch_one() {
  local name="$1" gpu="$2"
  local config="${CONFIG_DIR}/${name}.json"
  local run_dir="${RUN_ROOT}/${name}"
  local session="d256_${name}_${STAMP: -6}"
  mkdir -p "${run_dir}"
  tmux new-session -d -s "${session}" \
    "cd '${ROOT_DIR}' && env CUDA_VISIBLE_DEVICES='${gpu}' CONFIG='${config}' RUN_DIR_OVERRIDE='${run_dir}' BASE_ASAP_ONLY=1 BASE_NUM_TRAIN_EPOCHS=16 ADAPT_NUM_TRAIN_EPOCHS=0 BASE_PREPARED_SIDECAR_TAG=ASAP_FLOORLOG_NOMUS_SCORESPAN BATCH_SIZE_PER_DEVICE=32 GLOBAL_BATCH_SIZE=64 DET_NUM_SAMPLES=1 SAMPLING_NUM_SAMPLES=1 INFER_NUM_WORKERS=8 METRIC_NUM_WORKERS=8 INFER_BATCH_SIZE_WINDOWS=8 INFER_SCORE_SOURCE_LIST=data/asap_test_score_sources.txt EVAL_CHECKPOINT_MODE=best RESUME_FROM_LATEST_CHECKPOINT=0 MERGE_MODE=continuation CONTINUATION_DROP_RATIO=0.0 bash script/run_inr_epr_pipeline.sh > '${run_dir}/launcher.log' 2>&1"
  printf '%s\tGPU%s\t%s\t%s\n' "${name}" "${gpu}" "${session}" "${run_dir}" | tee -a "${RUN_ROOT}/processes.tsv"
}

: > "${RUN_ROOT}/processes.tsv"
launch_one discrete_ln_k2 0
launch_one discrete_beta_k2 1
launch_one truncated_logistic_k2 2
echo "RUN_ROOT=${RUN_ROOT}"
