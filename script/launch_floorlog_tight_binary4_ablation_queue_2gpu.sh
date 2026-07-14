#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/floorlog_tight_binary4_ablation_queue_2gpu/${STAMP}}"
CONFIG_DIR="${RUN_ROOT}/configs"
BASE_CONFIG="${BASE_CONFIG:-results/floorlog_tight_k1_binary4_bce_2gpu/20260714_tight_k1_binary4_bce_v1/configs/dlm-k1-noinfl-ioi-binary4-bce-pedalw1.json}"
GPUS="${GPUS:-2,3}"
EPOCHS="${EPOCHS:-16}"
mkdir -p "${CONFIG_DIR}"

python - "${BASE_CONFIG}" "${CONFIG_DIR}" "${EPOCHS}" <<'PY'
import json
import sys
from pathlib import Path

base_path, config_dir, epochs = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
base = json.loads(base_path.read_text(encoding="utf-8"))
for key in ("resume_path", "resume_from_checkpoint"):
    base.pop(key, None)

loss_weights = dict(base.get("loss_weights") or {})
loss_weights.update({"ioi": 1.0, "duration": 1.0, "velocity": 1.0, "pedal": 1.0})

common = dict(base)
common.update(
    {
        "pretrained_model": None,
        "load_pianoformer_backbone": False,
        "note_embedding_mode": "slot_attribute",
        "slot_dim": 128,
        "slot_fusion": "mlp",
        "slot_gates": False,
        "slot_share_role_encoders": True,
        "slot_decoder_mask_mode": "whole_token",
        "continuous_dim": 7,
        "output_continuous_dim": 7,
        "pedal_representation": "binary_4",
        "pedal_distribution": "point",
        "pedal_output_activation": "linear",
        "dlm_ioi_zero_inflated": False,
        "dlm_pedal_zero_one_inflated": False,
        "loss_weights": loss_weights,
        "num_train_epochs": epochs,
        "max_train_epochs": epochs,
        "per_device_train_batch_size": 32,
        "per_device_eval_batch_size": 32,
        "global_batch_size": 64,
        "gradient_accumulation_steps": 1,
        "overwrite_output_dir": True,
        "resume_trainer_state": False,
        "adapt_on_asap_after_train": False,
        "auto_rollout_eval_after_train": False,
        "ddp_find_unused_parameters": True,
        "tail_mask_enabled": True,
        "tail_mask_tf_clamp": True,
        "tail_mask_ioi_min": -1.0,
        "tail_mask_ioi_max": 1.0,
        "tail_mask_duration_min": -2.0,
        "tail_mask_duration_max": 1.0,
        "dlm_ioi_nonzero_min": -1.0,
        "dlm_ioi_nonzero_max": 1.0,
        "dlm_duration_min": -2.0,
        "dlm_duration_max": 1.0,
        "dlm_timing_weighted_nll_alpha": 0.0,
        "dlm_raw_ms_crps_lambda": 0.0,
        "dlm_tail_loss_lambda": 0.0,
        "dlm_target_tail_loss_lambda": 0.0,
        "dlm_target_tail_radius_frac": 0.0,
        "timing_sample_shrink_mode": "none",
        "timing_sample_shrink_factor": 1.0,
        "timing_sample_shrink_radius": 0.0,
        "timing_sample_truncate_radius": 0.0,
        "dlm_timing_sample_truncate_radius": 0.0,
    }
)

dlm_k1 = {
    "epr_distribution": "dlm",
    "velocity_distribution": "dlm",
    "dlm_components": 1,
    "epr_mixture_components": 1,
    "dlm_timing_scale_parameterization": "bounded_sigmoid",
    "dlm_timing_scale_min": 0.002,
    "dlm_timing_scale_max": 0.12,
    "dlm_scale_min": 0.001,
    "dlm_scale_max": 10.0,
}
beta5 = {
    "epr_distribution": "mixture_beta",
    "pedal_distribution": "point",
    "velocity_distribution": None,
    "epr_mixture_components": 5,
    "bounded_floorlog_support": True,
    "epr_distribution_eps": 1e-5,
    "beta_eps": 1e-5,
    "beta_alpha_min": 1e-4,
    "mixture_beta_parameterization": "mu_kappa",
    "mixture_beta_kappa_min": 1e-3,
}
mln2 = {
    "epr_distribution": "mixture_logistic_normal",
    "pedal_distribution": "point",
    "velocity_distribution": None,
    "epr_mixture_components": 2,
    "bounded_floorlog_support": True,
    "epr_distribution_eps": 1e-5,
    "logistic_normal_sigma_min": 1e-3,
    "logistic_normal_sigma_max": 10.0,
}
bsn = {
    "epr_distribution": "bounded_skew_normal",
    "pedal_distribution": "point",
    "velocity_distribution": None,
    "epr_mixture_components": 1,
    "bounded_floorlog_support": True,
    "epr_distribution_eps": 1e-5,
    "logistic_normal_sigma_min": 1e-3,
    "logistic_normal_sigma_max": 10.0,
    "skew_normal_sigma_min": 1e-3,
    "skew_normal_sigma_max": 5.0,
}

variants = {
    "slot6-full-k1": {
        **dlm_k1,
        "slot_version": "slot6",
        "musical_feature_mode": "musical51_full",
        "disable_musical_features": False,
    },
    "slot6-onset-only-k1": {
        **dlm_k1,
        "slot_version": "slot6",
        "musical_feature_mode": "musical51_onset_only",
        "disable_musical_features": False,
    },
    "slot6-annotation-only-k1": {
        **dlm_k1,
        "slot_version": "slot6",
        "musical_feature_mode": "musical51_annotation_only",
        "disable_musical_features": False,
    },
    "slot6-onset-annotation-k1": {
        **dlm_k1,
        "slot_version": "slot6",
        "musical_feature_mode": "musical51_onset_annotation",
        "disable_musical_features": False,
    },
    "slot6-no-duration-k1": {
        **dlm_k1,
        "slot_version": "slot6",
        "musical_feature_mode": "musical51_no_duration",
        "disable_musical_features": False,
    },
    "slot6-no-length-k1": {
        **dlm_k1,
        "slot_version": "slot6",
        "musical_feature_mode": "musical51_no_length",
        "disable_musical_features": False,
    },
    "slot5-nomus-k1": {
        **dlm_k1,
        "slot_version": "slot5",
        "musical_feature_mode": "none",
        "disable_musical_features": True,
        "dlm_components": 1,
        "epr_mixture_components": 1,
    },
    "slot5-nomus-k4": {
        **dlm_k1,
        "slot_version": "slot5",
        "musical_feature_mode": "none",
        "disable_musical_features": True,
        "dlm_components": 4,
        "epr_mixture_components": 4,
    },
    "slot5-nomus-k8": {
        **dlm_k1,
        "slot_version": "slot5",
        "musical_feature_mode": "none",
        "disable_musical_features": True,
        "dlm_components": 8,
        "epr_mixture_components": 8,
    },
    "slot5-nomus-beta5": {
        **beta5,
        "slot_version": "slot5",
        "musical_feature_mode": "none",
        "disable_musical_features": True,
    },
    "slot5-nomus-mln2": {
        **mln2,
        "slot_version": "slot5",
        "musical_feature_mode": "none",
        "disable_musical_features": True,
    },
    "slot5-nomus-bsn": {
        **bsn,
        "slot_version": "slot5",
        "musical_feature_mode": "none",
        "disable_musical_features": True,
    },
}

order = [
    "slot6-full-k1",
    "slot6-onset-only-k1",
    "slot6-annotation-only-k1",
    "slot6-onset-annotation-k1",
    "slot6-no-duration-k1",
    "slot6-no-length-k1",
    "slot5-nomus-k1",
    "slot5-nomus-k4",
    "slot5-nomus-k8",
    "slot5-nomus-beta5",
    "slot5-nomus-mln2",
    "slot5-nomus-bsn",
]

for name in order:
    cfg = dict(common)
    cfg.update(variants[name])
    cfg["run_name"] = f"floorlog_tight_binary4_{name.replace('-', '_')}"
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
            "purpose": "Sequential ASAP ablation queue: first 6 musical ablations, next 6 no-musical distribution ablations; IOI no-inflated and pedal binary4 BCE throughout.",
            "epochs_each": epochs,
            "shared": {
                "pedal_representation": "binary_4",
                "pedal_loss": "bce",
                "pedal_weight": 1.0,
                "dlm_ioi_zero_inflated": False,
                "dlm_pedal_zero_one_inflated": False,
                "tight_mask": True,
            },
            "order": order,
            "variants": variants,
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

: > "${RUN_ROOT}/processes.tsv"

QUEUE_RUNNER="${RUN_ROOT}/run_queue.sh"
python - "${QUEUE_RUNNER}" <<'PY'
import sys
from pathlib import Path

runner = Path(sys.argv[1])
runner.write_text(
    """#!/usr/bin/env bash
set -euo pipefail

launch_one() {
  local name="$1"
  local run_dir="${RUN_ROOT}/${name}"
  local log_path="${run_dir}/launcher.log"
  mkdir -p "${run_dir}"
  printf '%s\\tGPUs%s\\tSTART\\t%s\\t%s\\n' "${name}" "${GPUS}" "${run_dir}" "${log_path}" \\
    | tee -a "${RUN_ROOT}/processes.tsv"
  env CUDA_VISIBLE_DEVICES="${GPUS}" \\
    CONFIG="${CONFIG_DIR}/${name}.json" RUN_DIR_OVERRIDE="${run_dir}" \\
    BASE_ASAP_ONLY=1 BASE_NUM_TRAIN_EPOCHS="${EPOCHS}" ADAPT_NUM_TRAIN_EPOCHS=0 \\
    BATCH_SIZE_PER_DEVICE=32 GLOBAL_BATCH_SIZE=64 \\
    DET_NUM_SAMPLES=1 SAMPLING_NUM_SAMPLES=1 \\
    INFER_NUM_WORKERS=8 METRIC_NUM_WORKERS=8 INFER_BATCH_SIZE_WINDOWS=8 \\
    INFER_SCORE_SOURCE_LIST=data/asap_test_score_sources.txt \\
    EVAL_CHECKPOINT_MODE=latest RESUME_FROM_LATEST_CHECKPOINT=0 \\
    MERGE_MODE=continuation CONTINUATION_DROP_RATIO=0.0 \\
    bash script/run_inr_epr_pipeline.sh >"${log_path}" 2>&1
  printf '%s\\tGPUs%s\\tDONE\\t%s\\t%s\\n' "${name}" "${GPUS}" "${run_dir}" "${log_path}" \\
    | tee -a "${RUN_ROOT}/processes.tsv"
}

for name in \\
  slot6-full-k1 \\
  slot6-onset-only-k1 \\
  slot6-annotation-only-k1 \\
  slot6-onset-annotation-k1 \\
  slot6-no-duration-k1 \\
  slot6-no-length-k1 \\
  slot5-nomus-k1 \\
  slot5-nomus-k4 \\
  slot5-nomus-k8 \\
  slot5-nomus-beta5 \\
  slot5-nomus-mln2 \\
  slot5-nomus-bsn
do
  launch_one "${name}"
done
""",
    encoding="utf-8",
)
PY
chmod +x "${QUEUE_RUNNER}"

setsid env RUN_ROOT="${RUN_ROOT}" CONFIG_DIR="${CONFIG_DIR}" GPUS="${GPUS}" EPOCHS="${EPOCHS}" \
  bash "${QUEUE_RUNNER}" >"${RUN_ROOT}/queue.log" 2>&1 < /dev/null &

pid=$!
printf 'queue\tGPUs%s\tPID%s\t%s\t%s\n' "${GPUS}" "${pid}" "${RUN_ROOT}" "${RUN_ROOT}/queue.log" \
  | tee -a "${RUN_ROOT}/processes.tsv"
echo "RUN_ROOT=${RUN_ROOT}"
