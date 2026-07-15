#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
export PATH="/home/kaititech/anaconda3/bin:${PATH}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/slot5_nomus_simple_distributions_3gpu/${STAMP}}"
BASE_CONFIG="${BASE_CONFIG:-results/floorlog_distribution_ablation_single_gpu_20260714/slot5-nomus-k1/config.json}"
EPOCHS="${EPOCHS:-16}"
CONFIG_DIR="${RUN_ROOT}/configs"
mkdir -p "${CONFIG_DIR}"

if [[ ! -f "${CONFIG_DIR}/manifest.json" ]]; then
python - "${BASE_CONFIG}" "${CONFIG_DIR}" "${RUN_ROOT}" "${EPOCHS}" <<'PY'
import json
import sys
from pathlib import Path

base_path = Path(sys.argv[1])
config_dir = Path(sys.argv[2])
run_root = Path(sys.argv[3]).resolve()
epochs = float(sys.argv[4])
base = json.loads(base_path.read_text(encoding="utf-8"))

for key in ("resume_path", "resume_from_checkpoint"):
    base.pop(key, None)

common = dict(base)
common.update(
    {
        "slot_version": "slot5",
        "musical_feature_mode": "none",
        "disable_musical_features": True,
        "pedal_distribution": "point",
        "epr_mixture_components": 1,
        "num_train_epochs": epochs,
        "max_train_epochs": epochs,
        "overwrite_output_dir": True,
        "resume_trainer_state": False,
        "bounded_floorlog_support": True,
        "epr_distribution_eps": 1e-5,
    }
)

# The wide extrema below are numerical safety rails, not modeling constraints.
variants = {
    "dlm-k1-free-scale": {
        "epr_distribution": "dlm",
        "velocity_distribution": "dlm",
        "dlm_components": 1,
        "dlm_timing_scale_parameterization": "legacy_clamp",
        "dlm_scale_min": 1e-5,
        "dlm_scale_max": 1e4,
    },
    "logistic-normal-k1-free-sigma": {
        "epr_distribution": "logistic_normal",
        "velocity_distribution": None,
        "epr_mixture_components": 1,
        "logistic_normal_sigma_min": 1e-5,
        "logistic_normal_sigma_max": 1e4,
    },
    "beta-mukappa-k1-free-kappa": {
        "epr_distribution": "beta_mu_kappa",
        "velocity_distribution": None,
        "epr_mixture_components": 1,
        "beta_kappa_min": 1e-5,
        "beta_eps": 1e-5,
    },
}

for name, overrides in variants.items():
    config = dict(common)
    config.update(overrides)
    config["run_name"] = f"slot5_nomus_{name.replace('-', '_')}"
    config["output_dir"] = str(run_root / name / "training")
    config["logging_dir"] = str(run_root / name / "tf-logs")
    config.pop("dlm_timing_scale_min", None)
    config.pop("dlm_timing_scale_max", None)
    if name != "dlm-k1-free-scale":
        for key in ("dlm_timing_scale_parameterization", "dlm_scale_min", "dlm_scale_max"):
            config.pop(key, None)
    if name != "logistic-normal-k1-free-sigma":
        config.pop("logistic_normal_sigma_min", None)
        config.pop("logistic_normal_sigma_max", None)
    if name != "beta-mukappa-k1-free-kappa":
        config.pop("beta_kappa_min", None)
    (config_dir / f"{name}.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

(config_dir / "manifest.json").write_text(
    json.dumps(
        {
            "base_config": str(base_path),
            "purpose": "Three simple K=1 support-bounded distributions with only numerical safety rails on scale/concentration.",
            "gpu_assignment": {
                "dlm-k1-free-scale": 0,
                "logistic-normal-k1-free-sigma": 1,
                "beta-mukappa-k1-free-kappa": 2,
            },
            "variants": variants,
        },
        indent=2,
        ensure_ascii=False,
    ) + "\n",
    encoding="utf-8",
)
PY
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1 RUN_ROOT=${RUN_ROOT}"
  exit 0
fi

launch_one() {
  local name="$1"
  local gpu="$2"
  local run_dir="${RUN_ROOT}/${name}"
  mkdir -p "${run_dir}"
  env CUDA_VISIBLE_DEVICES="${gpu}" \
    CONFIG="${CONFIG_DIR}/${name}.json" RUN_DIR_OVERRIDE="${run_dir}" \
    BASE_ASAP_ONLY=1 BASE_NUM_TRAIN_EPOCHS="${EPOCHS}" ADAPT_NUM_TRAIN_EPOCHS=0 \
    BATCH_SIZE_PER_DEVICE=32 GLOBAL_BATCH_SIZE=64 \
    DET_NUM_SAMPLES=1 SAMPLING_NUM_SAMPLES=1 \
    INFER_NUM_WORKERS=8 METRIC_NUM_WORKERS=8 INFER_BATCH_SIZE_WINDOWS=8 \
    INFER_SCORE_SOURCE_LIST=results/external_baselines_asap_test_score_sources.txt \
    EVAL_CHECKPOINT_MODE=latest RESUME_FROM_LATEST_CHECKPOINT=0 \
    MERGE_MODE=continuation CONTINUATION_DROP_RATIO=0.0 \
    bash script/run_inr_epr_pipeline.sh
}

launch_one "$1" "$2"
