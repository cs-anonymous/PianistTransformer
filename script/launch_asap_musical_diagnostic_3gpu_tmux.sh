#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/inr_epr_pipeline/asap_musical_diagnostic_${STAMP}}"
CONFIG_DIR="${RUN_ROOT}/configs"
mkdir -p "${CONFIG_DIR}"

python - "${CONFIG_DIR}" "${RUN_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

config_dir, run_root = map(Path, sys.argv[1:])
base_path = Path("results/inr_epr_pipeline/lossnorm_ep20_baselines_20260718_001238/cinr_bounded_5pct/config.json")
base = json.loads(base_path.read_text(encoding="utf-8"))

common = {
    "metadata_path": str(Path("data/ASAP_processed/metadata.generated_json.csv").resolve()),
    "refined_dir": str(Path("data/ASAP_processed").resolve()),
    "use_prepared_sidecar": True,
    "train_performance_dataset": "ASAP",
    "eval_performance_dataset": "ASAP",
    "eval_include_all_performance_dataset": "ASAP",
    "eval_split": "valid",
    "note_embedding_mode": "slot_attribute",
    "slot_version": "slot6",
    "slot_dim": 128,
    "slot_fusion": "mlp",
    "musical_feature_mode": "musical4slot",
    "musical_slot_fusion": "sum",
    "disable_musical_features": False,
    "encoder_layers_num": 8,
    "decoder_layers_num": 4,
    "loss_normalization": False,
    "gradnorm": False,
    "seed": 42,
    "pedal_representation": "binary_4",
    "timing_control_mode": "dinr_floor_log",
    "epr_timing_target": "floor_log_deviation",
    "epr_distribution": "dlm",
    "velocity_distribution": "dlm",
    "epr_mixture_components": 1,
    "dlm_components": 1,
    "dlm_timing_bins": 256,
    "dlm_velocity_bins": 128,
    "dlm_ioi_nonzero_min": -2.0,
    "dlm_ioi_nonzero_max": 1.0,
    "dlm_ioi_zero_min": 0.0,
    "dlm_ioi_zero_max": 5.0,
    "dlm_duration_min": -2.0,
    "dlm_duration_max": 1.0,
    "dlm_timing_scale_parameterization": "bounded_sigmoid",
    "dlm_timing_scale_min": 1e-5,
    "dlm_ioi_nonzero_scale_max": 0.10,
    "dlm_ioi_zero_scale_max": 0.25,
    "dlm_duration_scale_max": 0.15,
    "dlm_velocity_scale_parameterization": "bounded_sigmoid",
    "dlm_velocity_scale_min": 1e-5,
    "dlm_velocity_scale_max": 6.4,
    "bounded_floorlog_support": True,
    "sampling_top_p": 0.90,
    "dlm_sampling_top_p": 0.90,
    "sampling_top_k": 0,
    "dlm_sampling_top_k": 0,
    "fixed_window_split_scheme": "train_valid_asap3_rebuilt_mask_v1",
    "fixed_window_base_split": "train",
    "fixed_window_train_split_name": "train",
    "fixed_window_eval_split_name": "valid",
    "fixed_window_split_summary_path": "data/train_valid_asap3_rebuilt_mask_v1_summary.json",
}

experiments = {
    "full_musical_forced_mask": {
        "musical_feature_transform": "forced_mask",
        "slot_gates": False,
    },
    "full_musical_random_value": {
        "musical_feature_transform": "random_value",
        "musical_random_seed": 4242,
        "slot_gates": False,
    },
    "full_musical_gated_zero_init": {
        "musical_feature_transform": "none",
        "slot_gates": True,
        "musical_gate_init": 0.0,
    },
}

manifest = {"run_root": str(run_root), "experiments": {}}
for name, overrides in experiments.items():
    cfg = dict(base)
    for key in ("resume_path", "resume_from_checkpoint", "prepared_sidecar_tag"):
        cfg.pop(key, None)
    cfg.update(common)
    cfg.update(overrides)
    cfg["run_name"] = f"asap_diag_{name}"
    cfg["output_dir"] = str(run_root / name / "training")
    cfg["logging_dir"] = str(run_root / name / "tf-logs")
    path = config_dir / f"{name}.json"
    path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest["experiments"][name] = str(path)

(config_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY

cat > "${RUN_ROOT}/run_one.sh" <<'SH'
#!/usr/bin/env bash
set -uo pipefail
gpu="$1"
name="$2"
root="$3"
repo="$4"
run_dir="${root}/${name}"
config="${root}/configs/${name}.json"
status="${root}/queue_status.tsv"
mkdir -p "${run_dir}"
cd "${repo}"
printf '%s\tGPU%s\t%s\tSTART\n' "$(date '+%F %T')" "${gpu}" "${name}" >> "${status}"
env CUDA_VISIBLE_DEVICES="${gpu}" CONFIG="${config}" RUN_DIR_OVERRIDE="${run_dir}" \
  BASE_ASAP_ONLY=1 BASE_NUM_TRAIN_EPOCHS="${BASE_NUM_TRAIN_EPOCHS:-16}" ADAPT_NUM_TRAIN_EPOCHS=0 \
  BATCH_SIZE_PER_DEVICE="${BATCH_SIZE_PER_DEVICE:-32}" GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}" \
  SAMPLING_NUM_SAMPLES="${SAMPLING_NUM_SAMPLES:-1}" INFER_NUM_WORKERS="${INFER_NUM_WORKERS:-8}" \
  METRIC_NUM_WORKERS="${METRIC_NUM_WORKERS:-8}" INFER_BATCH_SIZE_WINDOWS="${INFER_BATCH_SIZE_WINDOWS:-8}" \
  INFER_SCORE_SOURCE_LIST="${INFER_SCORE_SOURCE_LIST:-data/asap_test_score_sources.txt}" \
  EVAL_CHECKPOINT_MODE=best RESUME_FROM_LATEST_CHECKPOINT=0 SKIP_EXISTING_PIPELINE_OUTPUTS=0 \
  MERGE_MODE=continuation CONTINUATION_DROP_RATIO=0.0 \
  bash script/run_inr_epr_pipeline.sh > "${run_dir}/launcher.log" 2>&1
code=$?
if [[ "${code}" -eq 0 ]]; then
  printf '%s\tGPU%s\t%s\tDONE\n' "$(date '+%F %T')" "${gpu}" "${name}" >> "${status}"
else
  printf '%s\tGPU%s\t%s\tFAILED:%s\n' "$(date '+%F %T')" "${gpu}" "${name}" "${code}" >> "${status}"
fi
SH
chmod +x "${RUN_ROOT}/run_one.sh"

declare -A GPUS=(
  [full_musical_forced_mask]=0
  [full_musical_random_value]=1
  [full_musical_gated_zero_init]=2
)

for name in full_musical_forced_mask full_musical_random_value full_musical_gated_zero_init; do
  session="asap_diag_${name}_${STAMP}"
  tmux new-session -d -s "${session}" \
    "bash '${RUN_ROOT}/run_one.sh' '${GPUS[$name]}' '${name}' '${RUN_ROOT}' '${ROOT_DIR}'"
  echo "${session}" >> "${RUN_ROOT}/tmux_sessions.txt"
done

echo "${RUN_ROOT}"
