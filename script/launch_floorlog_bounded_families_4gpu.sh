#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/floorlog_bounded_families_4gpu/${STAMP}}"
CONFIG_DIR="${RUN_ROOT}/configs"
BASE_CONFIG="${BASE_CONFIG:-results/floorlog_dlm_2gpu/20260712_asap_test/k8-b256-veldlm/config.json}"
SHORT_EPOCHS="${SHORT_EPOCHS:-4}"
mkdir -p "${CONFIG_DIR}"

python - "${BASE_CONFIG}" "${CONFIG_DIR}" "${SHORT_EPOCHS}" <<'PY'
import json
import sys
from pathlib import Path

base_path, config_dir, epochs = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
base = json.loads(base_path.read_text(encoding="utf-8"))
for key in ("resume_path", "resume_from_checkpoint", "pretrained_model"):
    base.pop(key, None)
base.update({
    "pretrained_model": None,
    "load_pianoformer_backbone": False,
    "bounded_floorlog_support": True,
    "velocity_distribution": None,
    "timing_sample_shrink_mode": "none",
    "timing_sample_shrink_factor": 1.0,
    "timing_sample_shrink_radius": 0.0,
    "timing_sample_truncate_radius": 0.0,
    "dlm_timing_sample_truncate_radius": 0.0,
    "num_train_epochs": epochs,
    "max_train_epochs": epochs,
    "overwrite_output_dir": True,
    "resume_trainer_state": False,
    "auto_rollout_eval_after_train": False,
    "adapt_on_asap_after_train": False,
})

families = {
    "beta": [
        ("beta-k2", "mixture_beta", 2, {}),
        ("beta-k3", "mixture_beta", 3, {}),
        ("beta-k5", "mixture_beta", 5, {}),
        ("beta-k8", "mixture_beta", 8, {}),
    ],
    "tanh": [
        ("tanh-smin1e-3", "bounded_tanh_student_t", 1, {"logistic_normal_sigma_min": 1e-3}),
        ("tanh-smin1e-2", "bounded_tanh_student_t", 1, {"logistic_normal_sigma_min": 1e-2}),
        ("tanh-smax2", "bounded_tanh_student_t", 1, {"logistic_normal_sigma_max": 2.0}),
        ("tanh-smax5", "bounded_tanh_student_t", 1, {"logistic_normal_sigma_max": 5.0}),
    ],
    "sn": [
        ("sn-smin1e-4", "bounded_skew_normal", 1, {"skew_normal_sigma_min": 1e-4}),
        ("sn-smin1e-3", "bounded_skew_normal", 1, {"skew_normal_sigma_min": 1e-3}),
        ("sn-smax2", "bounded_skew_normal", 1, {"skew_normal_sigma_max": 2.0}),
        ("sn-smax5", "bounded_skew_normal", 1, {"skew_normal_sigma_max": 5.0}),
    ],
    "ln": [
        ("ln-k1", "logistic_normal", 1, {}),
        ("ln-k2", "mixture_logistic_normal", 2, {}),
        ("ln-k3", "mixture_logistic_normal", 3, {}),
        ("ln-k5", "mixture_logistic_normal", 5, {}),
    ],
}

manifest = {"base_config": str(base_path), "epochs": epochs, "families": {}}
for family, variants in families.items():
    family_dir = config_dir / family
    family_dir.mkdir(parents=True, exist_ok=True)
    manifest["families"][family] = []
    for name, distribution, components, overrides in variants:
        cfg = dict(base)
        cfg.update(overrides)
        cfg["epr_distribution"] = distribution
        cfg["pedal_distribution"] = distribution
        cfg["epr_mixture_components"] = components
        cfg["run_name"] = name.replace("-", "_")
        cfg["output_dir"] = f"unused/{family}/{name}/training"
        cfg["logging_dir"] = f"unused/{family}/{name}/tf-logs"
        path = family_dir / f"{name}.json"
        path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        manifest["families"][family].append({"name": name, "config": str(path)})
(config_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
PY

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1 configs=${CONFIG_DIR}"
  exit 0
fi

launch_family() {
  local gpu="$1" family="$2"
  local family_root="${RUN_ROOT}/${family}"
  local log_path="${family_root}/launcher.log"
  mkdir -p "${family_root}"
  setsid bash -c '
    set -euo pipefail
    family="$1"; config_dir="$2"; family_root="$3"; gpu="$4"; epochs="$5"
    for config in "${config_dir}/${family}"/*.json; do
      name="$(basename "${config}" .json)"
      run_dir="${family_root}/${name}"
      mkdir -p "${run_dir}"
      echo "[$(date +"%F %T")] START ${family}/${name} GPU${gpu}"
      CUDA_VISIBLE_DEVICES="${gpu}" CONFIG="${config}" RUN_DIR_OVERRIDE="${run_dir}" \
        BASE_ASAP_ONLY=1 BASE_NUM_TRAIN_EPOCHS="${epochs}" ADAPT_NUM_TRAIN_EPOCHS=0 BATCH_SIZE_PER_DEVICE=32 GLOBAL_BATCH_SIZE=32 \
        DET_NUM_SAMPLES=1 SAMPLING_NUM_SAMPLES=1 INFER_NUM_WORKERS=8 METRIC_NUM_WORKERS=8 \
        INFER_BATCH_SIZE_WINDOWS=8 INFER_SCORE_SOURCE_LIST=data/cheap15_score_sources.txt \
        EVAL_CHECKPOINT_MODE=latest RESUME_FROM_LATEST_CHECKPOINT=0 \
        MERGE_MODE=continuation CONTINUATION_DROP_RATIO=0.0 \
        bash script/run_inr_epr_pipeline.sh
      echo "[$(date +"%F %T")] DONE ${family}/${name}"
    done
  ' _ "${family}" "${CONFIG_DIR}" "${family_root}" "${gpu}" "${SHORT_EPOCHS}" >"${log_path}" 2>&1 < /dev/null &
  local pid=$!
  printf '%s\tGPU%s\tPID%s\t%s\n' "${family}" "${gpu}" "${pid}" "${log_path}" | tee -a "${RUN_ROOT}/processes.tsv"
}

launch_family 0 beta
launch_family 1 tanh
launch_family 2 sn
launch_family 3 ln
echo "RUN_ROOT=${RUN_ROOT}"
