#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/inr_epr_pipeline/asap_processed_musical51_te2n_slot_effective_${STAMP}}"
CONFIG_DIR="${RUN_ROOT}/configs"
PYTHON_BIN="${PYTHON_BIN:-/home/kaititech/anaconda3/bin/python}"
mkdir -p "${CONFIG_DIR}"

"${PYTHON_BIN}" - "${CONFIG_DIR}" "${RUN_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

config_dir, run_root = map(Path, sys.argv[1:])

base_dir = Path("results/inr_epr_pipeline/asap_processed_musical51_nolossnorm16_slot_effective_20260723/configs")
base_paths = {
    "cinr": base_dir / "cinr__default_dlm_k1_bounded5.json",
    "dinr": base_dir / "dinr__default_no_coord.json",
}
base = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in base_paths.items()}

def clean(cfg):
    cfg = dict(cfg)
    for key in (
        "resume_path",
        "resume_from_checkpoint",
        "prepared_sidecar_tag",
        "musical_component_gates",
        "musical_component_gate_init",
    ):
        cfg.pop(key, None)
    cfg.update({
        "metadata_path": str(Path("data/ASAP_processed/metadata.generated_json.csv").resolve()),
        "refined_dir": str(Path("data/ASAP_processed").resolve()),
        "use_prepared_sidecar": True,
        "prepared_sidecar_tag": "ASAP_MUSICAL51",
        "train_performance_dataset": "ASAP",
        "eval_performance_dataset": "ASAP",
        "eval_include_all_performance_dataset": "ASAP",
        "eval_split": "valid",
        "musical_feature_mode": "musical51_full",
        "disable_musical_features": False,
        "note_embedding_mode": "slot_attribute",
        "slot_version": "slot6",
        "slot_dim": 768,
        "slot_fusion": "sum",
        "slot_gates": True,
        "decoder_task_token_mode": "timing_expression_2n",
        "output_task_embedding_mode": "none",
        "encoder_layers_num": 10,
        "decoder_layers_num": 2,
        "loss_normalization": False,
        "gradnorm": False,
        "seed": 42,
        "pedal_representation": "binary_4",
        "timing_control_mode": "dinr_floor_log",
        "epr_timing_target": "floor_log_deviation",
        "sampling_top_p": 0.90,
        "dlm_sampling_top_p": 0.90,
        "dinr_sampling_top_p": 0.90,
        "sampling_top_k": 0,
        "dlm_sampling_top_k": 0,
        "dinr_sampling_top_k": 0,
        "fixed_window_split_scheme": "train_valid_asap3_nonasap05_v1",
        "fixed_window_base_split": "train",
        "fixed_window_train_split_name": "train",
        "fixed_window_eval_split_name": "valid",
        "fixed_window_split_summary_path": "backup/results0722/dlm_pn_group_ablation_3gpu/20260715_175548/train_valid_asap3_nonasap05_v1_current_summary.json",
        "num_train_epochs": 16.0,
        "max_train_epochs": 16.0,
    })
    return cfg

def bounded_scales(frac):
    return {
        "dlm_timing_scale_parameterization": "bounded_sigmoid",
        "dlm_timing_scale_min": 1e-5,
        "dlm_ioi_nonzero_scale_max": 2.0 * frac,
        "dlm_ioi_zero_scale_max": 5.0 * frac,
        "dlm_duration_scale_max": 3.0 * frac,
        "dlm_velocity_scale_parameterization": "bounded_sigmoid",
        "dlm_velocity_scale_min": 1e-5,
        "dlm_velocity_scale_max": 128.0 * frac,
        "bounded_floorlog_support": True,
    }

def unbounded_scales():
    return {
        "dlm_timing_scale_parameterization": "legacy_clamp",
        "dlm_velocity_scale_parameterization": "legacy_clamp",
        "bounded_floorlog_support": False,
        "dlm_scale_max": 10.0,
        "logistic_normal_sigma_max": 10.0,
    }

def cinr_base(name):
    cfg = clean(base["cinr"])
    cfg.update({
        "run_name": f"asap_musical51_te2n_cinr_{name}",
        "output_dir": str(run_root / "cinr" / name / "training"),
        "logging_dir": str(run_root / "cinr" / name / "tf-logs"),
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
    })
    cfg.update(bounded_scales(0.05))
    return cfg

def dinr_base(name):
    cfg = clean(base["dinr"])
    cfg.update({
        "run_name": f"asap_musical51_te2n_dinr_{name}",
        "output_dir": str(run_root / "dinr" / name / "training"),
        "logging_dir": str(run_root / "dinr" / name / "tf-logs"),
        "epr_distribution": "dinr",
        "velocity_distribution": "dinr",
        "dinr_output_timing_bins": 256,
        "dinr_output_zero_bin": 170,
        "dinr_output_timing_step": 3.0 / 255.0,
        "dinr_deviation_min": -2.0,
        "dinr_deviation_max": 1.0,
        "dinr_zero_ioi_min": 0.0,
        "dinr_zero_ioi_max": 5.0,
        "dinr_vocabulary_mode": "unified",
        "dinr_input_numerical_coordinates": False,
        "dinr_input_velocity_numerical_coordinates": False,
        "dinr_output_deviation_numerical_coordinates": False,
        "dinr_velocity_numerical_coordinates": False,
    })
    return cfg

def representation(overrides=None):
    cfg = {
        "note_embedding_mode": "slot_attribute",
        "slot_version": "slot6",
        "slot_dim": 128,
        "slot_fusion": "mlp",
    }
    if overrides:
        cfg.update(overrides)
    return cfg

queues = {"cinr": [], "dinr": [], "extra": []}
configs = {}

def add(queue, name, cfg):
    cfg = dict(cfg)
    cfg.pop("musical_component_gates", None)
    cfg.pop("musical_component_gate_init", None)
    path = config_dir / f"{queue}__{name}.json"
    path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    queues[queue].append(name)
    configs[f"{queue}/{name}"] = str(path)

for name, overrides in [
    ("default_dlm_k1_bounded5", {}),
    ("bounded_2p5pct", bounded_scales(0.025)),
    ("bounded_10pct", bounded_scales(0.10)),
    ("unbounded", unbounded_scales()),
    ("dist_dlm_k3", {"epr_distribution": "dlm", "velocity_distribution": "dlm", "dlm_components": 3, "epr_mixture_components": 3}),
    ("dist_ln_k1", {"epr_distribution": "logistic_normal", "velocity_distribution": None, "epr_mixture_components": 1}),
    ("dist_ln_k3", {"epr_distribution": "mixture_logistic_normal", "velocity_distribution": None, "epr_mixture_components": 3}),
]:
    cfg = cinr_base(name)
    cfg.update(overrides)
    add("cinr", name, cfg)

for name, overrides in [
    ("default_no_coord", {}),
    ("coord_timing_dev", {
        "dinr_input_numerical_coordinates": True,
        "dinr_input_velocity_numerical_coordinates": False,
        "dinr_output_deviation_numerical_coordinates": True,
        "dinr_velocity_numerical_coordinates": False,
    }),
    ("coord_velocity", {
        "dinr_input_numerical_coordinates": False,
        "dinr_input_velocity_numerical_coordinates": False,
        "dinr_output_deviation_numerical_coordinates": False,
        "dinr_velocity_numerical_coordinates": True,
    }),
    ("coord_all", {
        "dinr_input_numerical_coordinates": True,
        "dinr_input_velocity_numerical_coordinates": True,
        "dinr_output_deviation_numerical_coordinates": True,
        "dinr_velocity_numerical_coordinates": True,
    }),
    ("rep_slot_direct", representation({"slot_fusion": "direct_concat", "slot_dim": 128})),
    ("rep_slot_mlp", representation({"slot_fusion": "mlp", "slot_dim": 128})),
    ("rep_slot256_mlp", representation({"slot_fusion": "mlp", "slot_dim": 256})),
]:
    cfg = dinr_base(name)
    cfg.update(overrides)
    add("dinr", name, cfg)

for name, overrides in [
    ("rep_slot_direct", representation({"slot_fusion": "direct_concat", "slot_dim": 128})),
    ("rep_slot_mlp", representation({"slot_fusion": "mlp", "slot_dim": 128})),
    ("rep_slot256_mlp", representation({"slot_fusion": "mlp", "slot_dim": 256})),
    ("musical_no_duration", {"musical_feature_mode": "musical51_no_duration", "disable_musical_features": False}),
    ("musical_no_onset", {"musical_feature_mode": "musical51_no_onset", "disable_musical_features": False}),
    ("musical_no_annotation", {"musical_feature_mode": "musical51_no_annotation", "disable_musical_features": False}),
    ("musical_no_length", {"musical_feature_mode": "musical51_no_length", "disable_musical_features": False}),
]:
    cfg = cinr_base(name)
    cfg.update(overrides)
    add("extra", name, cfg)

(config_dir / "manifest.json").write_text(json.dumps({
    "run_root": str(run_root),
    "queues": queues,
    "configs": configs,
    "notes": [
        "Current baseline: ASAP_processed musical51 no-lossnorm 16ep, 2N timing/expression decoder, rep_slot_sum, 10+2 backbone.",
        "Only slot_gates is enabled; musical_component_gates is intentionally omitted.",
        "Deleted rep_sine variants and dist_beta variants; DINR deletes coord_input_only.",
        "DINR coord_timing_dev enables timing input coordinates plus timing deviation coordinates; coord_velocity enables velocity coordinates only.",
    ],
}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY

cat > "${RUN_ROOT}/run_queue.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

gpu="$1"
queue="$2"
root="$3"
repo="$4"
status="${root}/queue_status.tsv"
cd "${repo}"

python - "${root}/configs/manifest.json" "${queue}" <<'PY' > "${root}/${queue}.list"
import json
import sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for name in manifest["queues"][sys.argv[2]]:
    print(name)
PY

while IFS= read -r name; do
  [[ -n "${name}" ]] || continue
  run_dir="${root}/${queue}/${name}"
  config="${root}/configs/${queue}__${name}.json"
  mkdir -p "${run_dir}"
  printf '%s\tGPU%s\t%s\t%s\tSTART\n' "$(date '+%F %T')" "${gpu}" "${queue}" "${name}" >> "${status}"
  set +e
  CUDA_VISIBLE_DEVICES="${gpu}" CONFIG="${config}" RUN_DIR_OVERRIDE="${run_dir}" \
    BASE_ASAP_ONLY=1 BASE_NUM_TRAIN_EPOCHS="${BASE_NUM_TRAIN_EPOCHS:-16}" ADAPT_NUM_TRAIN_EPOCHS=0 \
    BATCH_SIZE_PER_DEVICE="${BATCH_SIZE_PER_DEVICE:-16}" GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}" \
    SAMPLING_NUM_SAMPLES="${SAMPLING_NUM_SAMPLES:-1}" INFER_NUM_WORKERS="${INFER_NUM_WORKERS:-8}" \
    METRIC_NUM_WORKERS="${METRIC_NUM_WORKERS:-8}" INFER_BATCH_SIZE_WINDOWS="${INFER_BATCH_SIZE_WINDOWS:-8}" \
    INFER_SCORE_SOURCE_LIST="${INFER_SCORE_SOURCE_LIST:-data/asap_test_score_sources.txt}" \
    EVAL_CHECKPOINT_MODE=best RESUME_FROM_LATEST_CHECKPOINT=0 SKIP_EXISTING_PIPELINE_OUTPUTS=0 \
    MERGE_MODE=continuation CONTINUATION_DROP_RATIO=0.0 \
    bash script/run_inr_epr_pipeline.sh > "${run_dir}/launcher.log" 2>&1
  code=$?
  set -e
  if [[ "${code}" -eq 0 ]]; then
    printf '%s\tGPU%s\t%s\t%s\tDONE\n' "$(date '+%F %T')" "${gpu}" "${queue}" "${name}" >> "${status}"
  else
    printf '%s\tGPU%s\t%s\t%s\tFAILED:%s\n' "$(date '+%F %T')" "${gpu}" "${queue}" "${name}" "${code}" >> "${status}"
  fi
done < "${root}/${queue}.list"

printf '%s\tGPU%s\t%s\t-\tQUEUE_DONE\n' "$(date '+%F %T')" "${gpu}" "${queue}" >> "${status}"
SH
chmod +x "${RUN_ROOT}/run_queue.sh"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "RUN_ROOT=${RUN_ROOT}"
  cat "${CONFIG_DIR}/manifest.json"
  exit 0
fi

: > "${RUN_ROOT}/queue_status.tsv"
tmux new-session -d -s "te2n_slot_cinr_${STAMP}" \
  "bash '${RUN_ROOT}/run_queue.sh' 0 cinr '${RUN_ROOT}' '${ROOT_DIR}' > '${RUN_ROOT}/cinr.queue.log' 2>&1"
tmux new-session -d -s "te2n_slot_dinr_${STAMP}" \
  "bash '${RUN_ROOT}/run_queue.sh' 1 dinr '${RUN_ROOT}' '${ROOT_DIR}' > '${RUN_ROOT}/dinr.queue.log' 2>&1"
tmux new-session -d -s "te2n_slot_extra_${STAMP}" \
  "bash '${RUN_ROOT}/run_queue.sh' 2 extra '${RUN_ROOT}' '${ROOT_DIR}' > '${RUN_ROOT}/extra.queue.log' 2>&1"

{
  echo "te2n_slot_cinr_${STAMP}"
  echo "te2n_slot_dinr_${STAMP}"
  echo "te2n_slot_extra_${STAMP}"
} > "${RUN_ROOT}/tmux_sessions.txt"

echo "RUN_ROOT=${RUN_ROOT}"
echo "TMUX_SESSIONS=$(tr '\n' ' ' < "${RUN_ROOT}/tmux_sessions.txt")"
