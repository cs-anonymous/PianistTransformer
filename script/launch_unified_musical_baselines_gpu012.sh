#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/inr_epr_pipeline/unified_musical_baselines_${STAMP}}"
CONFIG_DIR="${RUN_ROOT}/configs"
DINR_BASE="results/inr_epr_pipeline/asaponly_matched_cinr_dinr_20260716_143813/dinr_logits/config.json"
CINR_BASE="results/inr_epr_pipeline/asaponly_matched_cinr_dinr_20260716_143813/cinr_dlm_k1/config.json"
mkdir -p "${CONFIG_DIR}"

python - "${DINR_BASE}" "${CINR_BASE}" "${CONFIG_DIR}" "${RUN_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

dinr_base = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
cinr_base = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
config_dir = Path(sys.argv[3])
run_root = Path(sys.argv[4])

def common(cfg, name):
    out = dict(cfg)
    for key in ("resume_path", "resume_from_checkpoint"):
        out.pop(key, None)
    out.update(
        {
            "run_name": f"unified_musical_{name}",
            "output_dir": str(run_root / name / "training"),
            "logging_dir": str(run_root / name / "tf-logs"),
            "slot_version": "slot6",
            "slot_dim": 128,
            "slot_fusion": "mlp",
            "slot_gates": False,
            "slot_share_role_encoders": True,
            "slot_decoder_mask_mode": "whole_token",
            "musical_feature_mode": "musical51_full",
            "disable_musical_features": False,
            "timing_control_mode": "dinr_floor_log",
            "epr_timing_target": "floor_log_deviation",
            "eval_gt_time_normalization": "score_onset_span",
            "prepared_sidecar_tag": "ASAP_DINR_SCORESPAN",
            "train_performance_dataset": "ASAP",
            "eval_performance_dataset": "ASAP",
            "eval_include_all_performance_dataset": "ASAP",
            "num_train_epochs": 16.0,
            "max_train_epochs": 16.0,
            "pedal_representation": "binary_4",
            "pedal_distribution": "point",
            "pedal_output_activation": "linear",
            "dlm_ioi_zero_inflated": False,
            "dlm_pedal_zero_one_inflated": False,
            "dlm_ioi_zero_min": 0.0,
            "dlm_ioi_zero_max": 5.0,
            "dlm_ioi_nonzero_min": -2.0,
            "dlm_ioi_nonzero_max": 1.0,
            "dlm_duration_min": -2.0,
            "dlm_duration_max": 1.0,
            "dlm_velocity_min": -0.5,
            "dlm_velocity_max": 127.5,
            "dlm_velocity_bins": 128,
            "seed": 42,
            "sampling_top_p": 0.90,
            "sampling_top_k": 0,
            "dlm_sampling_temperature": 1.0,
            "dlm_sampling_top_p": 0.90,
            "dlm_sampling_top_k": 0,
            "dinr_sampling_temperature": 1.0,
            "dinr_sampling_top_p": 0.90,
            "dinr_sampling_top_k": 0,
            "timing_sample_shrink_mode": "none",
            "timing_sample_truncate_radius": 0.0,
            "dlm_timing_sample_truncate_radius": 0.0,
        }
    )
    return out

configs = {}

dinr = common(dinr_base, "dinr")
dinr.update(
    {
        "epr_distribution": "dinr",
        "dinr_vocabulary_mode": "separated",
    }
)
configs["dinr"] = dinr

cinr = common(cinr_base, "cinr")
cinr.update(
    {
        "epr_distribution": "dlm",
        "velocity_distribution": "dlm",
        "epr_mixture_components": 1,
        "dlm_components": 1,
        "dlm_timing_scale_parameterization": "softplus_unbounded",
        "dlm_timing_scale_min": 0.002,
        "dlm_timing_scale_max": 0.12,
        "dlm_velocity_scale_parameterization": "softplus_unbounded",
        "dlm_velocity_scale_min": None,
        "dlm_velocity_scale_max": None,
    }
)
for key in ("dlm_ioi_nonzero_scale_max", "dlm_ioi_zero_scale_max", "dlm_duration_scale_max"):
    cinr.pop(key, None)
configs["cinr"] = cinr

bounded = dict(cinr)
bounded.update(
    {
        "run_name": "unified_musical_cinr_bounded_5pct",
        "output_dir": str(run_root / "cinr_bounded_5pct" / "training"),
        "logging_dir": str(run_root / "cinr_bounded_5pct" / "tf-logs"),
        "dlm_timing_scale_parameterization": "bounded_sigmoid",
        "dlm_timing_scale_min": 1e-5,
        "dlm_ioi_nonzero_scale_max": 0.15,
        "dlm_ioi_zero_scale_max": 0.25,
        "dlm_duration_scale_max": 0.15,
        "dlm_velocity_scale_parameterization": "bounded_sigmoid",
        "dlm_velocity_scale_min": 1e-5,
        "dlm_velocity_scale_max": 6.4,
        "scale_window_fraction": 0.05,
    }
)
configs["cinr_bounded_5pct"] = bounded

manifest = []
for name, cfg in configs.items():
    path = config_dir / f"{name}.json"
    path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest.append({"name": name, "config": str(path), "run_dir": str(run_root / name)})

(config_dir / "manifest.json").write_text(
    json.dumps(
        {
            "sampling": {"temperature": 1.0, "top_p": 0.90, "top_k": 0, "num_samples": 2, "seed": 42},
            "shared": {
                "slot_version": "slot6",
                "musical_feature_mode": "musical51_full",
                "timing_control_mode": "dinr_floor_log",
                "ioi_nonzero_support": [-2.0, 1.0],
                "duration_support": [-2.0, 1.0],
                "pedal_representation": "binary_4",
                "epochs": 16.0,
            },
            "runs": manifest,
        },
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)
PY

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1 RUN_ROOT=${RUN_ROOT}"
  exit 0
fi

WORKER="${RUN_ROOT}/run_one.sh"
cat > "${WORKER}" <<SH
#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="${ROOT_DIR}"
RUN_ROOT="${RUN_ROOT}"
CONFIG_DIR="${CONFIG_DIR}"
gpu="\$1"
name="\$2"
cd "\${ROOT_DIR}"
config="\${CONFIG_DIR}/\${name}.json"
run_dir="\${RUN_ROOT}/\${name}"
mkdir -p "\${run_dir}"
printf '%s\tGPU%s\tSTART\t%s\n' "\$(date '+%F %T')" "\${gpu}" "\${name}" | tee -a "\${RUN_ROOT}/processes.tsv"
env CUDA_VISIBLE_DEVICES="\${gpu}" \\
  CONFIG="\${config}" \\
  RUN_DIR_OVERRIDE="\${run_dir}" \\
  BASE_ASAP_ONLY=1 \\
  BASE_NUM_TRAIN_EPOCHS=16 \\
  ADAPT_NUM_TRAIN_EPOCHS=0 \\
  BATCH_SIZE_PER_DEVICE=32 \\
  GLOBAL_BATCH_SIZE=64 \\
  DET_NUM_SAMPLES=1 \\
  SAMPLING_NUM_SAMPLES=2 \\
  INFER_NUM_WORKERS=8 \\
  METRIC_NUM_WORKERS=8 \\
  INFER_BATCH_SIZE_WINDOWS=8 \\
  INFER_SCORE_SOURCE_LIST=data/asap_test_score_sources.txt \\
  EVAL_CHECKPOINT_MODE=best \\
  RESUME_FROM_LATEST_CHECKPOINT=0 \\
  MERGE_MODE=continuation \\
  CONTINUATION_DROP_RATIO=0.0 \\
  bash script/run_inr_epr_pipeline.sh > "\${run_dir}/launcher.log" 2>&1
printf '%s\tGPU%s\tDONE\t%s\n' "\$(date '+%F %T')" "\${gpu}" "\${name}" | tee -a "\${RUN_ROOT}/processes.tsv"
SH
chmod +x "${WORKER}"

tmux new-session -d -s "unified_dinr_${STAMP: -6}" "bash '${WORKER}' 0 dinr"
tmux new-session -d -s "unified_cinr_${STAMP: -6}" "bash '${WORKER}' 1 cinr"
tmux new-session -d -s "unified_bnd_${STAMP: -6}" "bash '${WORKER}' 2 cinr_bounded_5pct"

echo "RUN_ROOT=${RUN_ROOT}"
echo "SESSIONS=unified_dinr_${STAMP: -6},unified_cinr_${STAMP: -6},unified_bnd_${STAMP: -6}"
