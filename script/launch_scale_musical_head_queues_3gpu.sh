#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
export PATH="/home/kaititech/anaconda3/bin:${PATH}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/scale_musical_head_queues_3gpu/${STAMP}}"
CONFIG_DIR="${RUN_ROOT}/configs"
BASE_CONFIG="${BASE_CONFIG:-results/floorlog_distribution_ablation_single_gpu_20260714/slot5-nomus-k1/config.json}"
EPOCHS="${EPOCHS:-16}"
mkdir -p "${CONFIG_DIR}"

SPLIT_SUMMARY="${CONFIG_DIR}/train_valid_asap3_nonasap05_v1_current_summary.json"
if [[ ! -f "${SPLIT_SUMMARY}" ]]; then
  python src/data_process/create_fixed_window_valid_split.py \
    --metadata-path PianoCoRe/metadata.csv \
    --refined-dir PianoCoRe/processed \
    --scheme-name train_valid_asap3_nonasap05_v1 \
    --selection-seed 42 \
    --output-summary "${SPLIT_SUMMARY}" \
    --skip-sidecars \
    --workers 24
fi

if [[ ! -f "${CONFIG_DIR}/manifest.json" ]]; then
python - "${BASE_CONFIG}" "${CONFIG_DIR}" "${RUN_ROOT}" "${EPOCHS}" "${SPLIT_SUMMARY}" <<'PY'
import json
import sys
from pathlib import Path

base_path, config_dir, run_root = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]).resolve()
epochs = float(sys.argv[4])
split_summary = str(Path(sys.argv[5]).resolve())
base = json.loads(base_path.read_text(encoding="utf-8"))
for key in ("resume_path", "resume_from_checkpoint"):
    base.pop(key, None)

loss_weights = dict(base.get("loss_weights") or {})
loss_weights.update({"ioi": 1.0, "duration": 1.0, "velocity": 1.0, "pedal": 1.0})
common = dict(base)
common.update({
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
    "dlm_ioi_zero_min": 0.0,
    "dlm_ioi_zero_max": 5.0,
    "dlm_ioi_nonzero_min": -1.0,
    "dlm_ioi_nonzero_max": 1.0,
    "dlm_duration_min": -2.0,
    "dlm_duration_max": 1.0,
    "dlm_velocity_min": -0.5,
    "dlm_velocity_max": 127.5,
    "loss_weights": loss_weights,
    "num_train_epochs": epochs,
    "max_train_epochs": epochs,
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 32,
    "global_batch_size": 64,
    "overwrite_output_dir": True,
    "resume_trainer_state": False,
    "adapt_on_asap_after_train": False,
    "auto_rollout_eval_after_train": False,
    "ddp_find_unused_parameters": True,
    "fixed_window_split_summary_path": split_summary,
    "dlm_timing_weighted_nll_alpha": 0.0,
    "dlm_raw_ms_crps_lambda": 0.0,
    "dlm_tail_loss_lambda": 0.0,
    "dlm_target_tail_loss_lambda": 0.0,
    "timing_sample_shrink_mode": "none",
    "timing_sample_truncate_radius": 0.0,
    "dlm_timing_sample_truncate_radius": 0.0,
})

def nomus():
    return {"slot_version": "slot5", "musical_feature_mode": "none", "disable_musical_features": True}

def musical(mode):
    return {"slot_version": "slot6", "musical_feature_mode": mode, "disable_musical_features": False}

def dlm_k1():
    return {
        "epr_distribution": "dlm", "velocity_distribution": "dlm",
        "dlm_components": 1, "epr_mixture_components": 1,
    }

def percent_scale(frac):
    return {
        "dlm_timing_scale_parameterization": "bounded_sigmoid",
        "dlm_timing_scale_min": 1e-5,
        "dlm_ioi_nonzero_scale_max": 2.0 * frac,
        "dlm_ioi_zero_scale_max": 5.0 * frac,
        "dlm_duration_scale_max": 3.0 * frac,
        "dlm_velocity_scale_parameterization": "bounded_sigmoid",
        "dlm_velocity_scale_min": 1e-5,
        "dlm_velocity_scale_max": 128.0 * frac,
        "scale_window_fraction": frac,
    }

variants = {}
variants["scale-default-current"] = {**nomus(), **dlm_k1()}
for label, frac in (("2p5", .025), ("5", .05), ("10", .10), ("20", .20)):
    variants[f"scale-{label}pct"] = {**nomus(), **dlm_k1(), **percent_scale(frac)}

for name, mode in (
    ("musical-onset", "musical51_onset_only"),
    ("musical-annotation", "musical51_annotation_only"),
    ("musical-onset-annotation", "musical51_onset_annotation"),
    ("musical-duration", "musical51_duration_only"),
    ("musical-full", "musical51_full"),
):
    variants[name] = {**musical(mode), **dlm_k1(), **percent_scale(.05)}

head_common = {**nomus(), **percent_scale(.05), "bounded_floorlog_support": True, "epr_distribution_eps": 1e-5}
variants["head-dlm-k3"] = {**head_common, "epr_distribution": "dlm", "velocity_distribution": "dlm", "dlm_components": 3, "epr_mixture_components": 3}
variants["head-mln-k1"] = {**head_common, "epr_distribution": "logistic_normal", "velocity_distribution": None, "epr_mixture_components": 1, "logistic_normal_sigma_min": 1e-5, "logistic_normal_sigma_max": .2}
variants["head-mln-k3"] = {**head_common, "epr_distribution": "mixture_logistic_normal", "velocity_distribution": None, "epr_mixture_components": 3, "logistic_normal_sigma_min": 1e-5, "logistic_normal_sigma_max": .2}
variants["head-beta-k1"] = {**head_common, "epr_distribution": "mixture_beta", "velocity_distribution": None, "epr_mixture_components": 1, "mixture_beta_parameterization": "mu_kappa", "mixture_beta_kappa_min": 99.0, "beta_alpha_min": 1e-5, "logistic_normal_sigma_min": 1e-5, "logistic_normal_sigma_max": .2}
variants["head-beta-k3"] = {**head_common, "epr_distribution": "mixture_beta", "velocity_distribution": None, "epr_mixture_components": 3, "mixture_beta_parameterization": "mu_kappa", "mixture_beta_kappa_min": 99.0, "beta_alpha_min": 1e-5, "logistic_normal_sigma_min": 1e-5, "logistic_normal_sigma_max": .2}

queues = {
    "gpu0_scale": ["scale-default-current", "scale-2p5pct", "scale-5pct", "scale-10pct", "scale-20pct"],
    "gpu1_musical": ["musical-onset", "musical-annotation", "musical-onset-annotation", "musical-duration", "musical-full"],
    "gpu2_head": ["head-dlm-k3", "head-mln-k1", "head-mln-k3", "head-beta-k1", "head-beta-k3"],
}

for name, overrides in variants.items():
    cfg = dict(common)
    cfg.update(overrides)
    cfg["run_name"] = name.replace("-", "_")
    cfg["output_dir"] = str(run_root / name / "training")
    cfg["logging_dir"] = str(run_root / name / "tf-logs")
    (config_dir / f"{name}.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

manifest = {
    "base_config": str(base_path),
    "queues": queues,
    "musical_nomus_reference": "scale-5pct",
    "support": {"ioi_zero": [0, 5], "ioi_nonzero": [-1, 1], "duration": [-2, 1], "velocity": [-0.5, 127.5]},
    "scale_fraction_rule": "per-feature max scale equals fraction times support width",
    "scalar_head_5pct": {"mln_sigma_max_latent": .2, "beta_kappa_min": 99.0},
    "variants": variants,
}
(config_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
for queue, names in queues.items():
    (config_dir / f"{queue}.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
PY
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1 RUN_ROOT=${RUN_ROOT}"
  exit 0
fi

run_queue() {
  local gpu="$1" queue_file="$2"
  while IFS= read -r name; do
    [[ -n "${name}" ]] || continue
    local run_dir="${RUN_ROOT}/${name}"
    mkdir -p "${run_dir}"
    printf '%s\tGPU%s\tSTART\t%s\n' "$(date '+%F %T')" "${gpu}" "${name}" | tee -a "${RUN_ROOT}/processes.tsv"
    env CUDA_VISIBLE_DEVICES="${gpu}" \
      CONFIG="${CONFIG_DIR}/${name}.json" RUN_DIR_OVERRIDE="${run_dir}" \
      BASE_ASAP_ONLY=1 BASE_NUM_TRAIN_EPOCHS="${EPOCHS}" ADAPT_NUM_TRAIN_EPOCHS=0 \
      BATCH_SIZE_PER_DEVICE=32 GLOBAL_BATCH_SIZE=64 \
      DET_NUM_SAMPLES=1 SAMPLING_NUM_SAMPLES=1 \
      INFER_NUM_WORKERS=8 METRIC_NUM_WORKERS=8 INFER_BATCH_SIZE_WINDOWS=8 \
      INFER_SCORE_SOURCE_LIST=results/external_baselines_asap_test_score_sources.txt \
      EVAL_CHECKPOINT_MODE=latest RESUME_FROM_LATEST_CHECKPOINT=0 \
      MERGE_MODE=continuation CONTINUATION_DROP_RATIO=0.0 \
      bash script/run_inr_epr_pipeline.sh >"${run_dir}/launcher.log" 2>&1
    printf '%s\tGPU%s\tDONE\t%s\n' "$(date '+%F %T')" "${gpu}" "${name}" | tee -a "${RUN_ROOT}/processes.tsv"
  done < "${queue_file}"
}

run_queue "$1" "$2"
