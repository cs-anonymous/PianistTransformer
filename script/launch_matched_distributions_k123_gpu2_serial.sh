#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
GPU="${GPU:-2}"
RUN_ROOT="${RUN_ROOT:-results/inr_epr_pipeline/asaponly_matched_distributions_k123_gpu2_${STAMP}}"
CONFIG_DIR="${RUN_ROOT}/configs"
BASE_CONFIG="results/inr_epr_pipeline/asaponly_matched_cinr_dinr_20260716_143813/cinr_dlm_k1/config.json"
mkdir -p "${CONFIG_DIR}"

python - "${BASE_CONFIG}" "${CONFIG_DIR}" "${RUN_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

base_path = Path(sys.argv[1])
config_dir = Path(sys.argv[2])
run_root = sys.argv[3]
base = json.loads(base_path.read_text(encoding="utf-8"))

distributions = {
    "discrete_ln": {
        "epr_distribution": "discrete_logistic_normal",
        "velocity_distribution": "discrete_logistic_normal",
        "run_label": "discreteLN",
        "extra": {"logistic_normal_sigma_min": 0.001},
    },
    "discrete_beta": {
        "epr_distribution": "discrete_beta",
        "velocity_distribution": "discrete_beta",
        "run_label": "discreteBeta",
        "extra": {
            "mixture_beta_parameterization": "mu_kappa",
            "mixture_beta_kappa_min": 0.001,
            "beta_alpha_min": 0.0001,
        },
    },
    "truncated_logistic": {
        "epr_distribution": "truncated_logistic",
        "velocity_distribution": "truncated_logistic",
        "run_label": "truncatedLogistic",
        "extra": {},
    },
}

manifest = []
for dist_name, spec in distributions.items():
    for k in (1, 2, 3):
        name = f"{dist_name}_k{k}"
        cfg = dict(base)
        cfg.update(
            {
                "run_name": f"ASAP-matched-{spec['run_label']}-k{k}-slot8-nomus-v128-T08p95",
                "output_dir": f"{run_root}/{name}/training",
                "logging_dir": f"{run_root}/{name}/tf-logs",
                "resume_from_checkpoint": None,
                "num_train_epochs": 16.0,
                "max_train_epochs": 16.0,
                "slot_version": "slot8",
                "slot_dim": 128,
                "slot_fusion": "mlp",
                "musical_feature_mode": "none",
                "disable_musical_features": True,
                "epr_distribution": spec["epr_distribution"],
                "velocity_distribution": spec["velocity_distribution"],
                "epr_mixture_components": k,
                "dlm_components": k,
                "dlm_timing_bins": 256,
                "dlm_velocity_bins": 128,
                "bounded_floorlog_support": True,
                "sampling_top_p": 0.95,
                "dlm_sampling_temperature": 0.8,
                "dlm_sampling_top_p": 0.95,
                "timing_sample_shrink_mode": "none",
                "timing_sample_truncate_radius": 0.0,
                "dlm_timing_sample_truncate_radius": 0.0,
                "pedal_distribution": "point",
                "train_performance_dataset": "ASAP",
                "eval_performance_dataset": "ASAP",
                "eval_include_all_performance_dataset": "ASAP",
                "prepared_sidecar_tag": "ASAP_FLOORLOG_NOMUS_SCORESPAN",
            }
        )
        cfg.update(spec["extra"])
        path = config_dir / f"{name}.json"
        path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        manifest.append({"name": name, "config": str(path), "run_dir": f"{run_root}/{name}"})

(config_dir / "manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
PY

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1 RUN_ROOT=${RUN_ROOT}"
  exit 0
fi

SESSION="${SESSION:-matched_k123_gpu2_${STAMP: -6}}"
QUEUE_SCRIPT="${RUN_ROOT}/run_queue.sh"
cat > "${QUEUE_SCRIPT}" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="__ROOT_DIR__"
RUN_ROOT="__RUN_ROOT__"
GPU="__GPU__"
cd "${ROOT_DIR}"
: > "${RUN_ROOT}/processes.tsv"
while IFS=$'\t' read -r name config run_dir; do
  mkdir -p "${run_dir}"
  printf '%s\tGPU%s\tSTART\t%s\n' "${name}" "${GPU}" "$(date '+%F %T')" | tee -a "${RUN_ROOT}/processes.tsv"
  env CUDA_VISIBLE_DEVICES="${GPU}" \
    CONFIG="${config}" \
    RUN_DIR_OVERRIDE="${run_dir}" \
    BASE_ASAP_ONLY=1 \
    BASE_NUM_TRAIN_EPOCHS=16 \
    ADAPT_NUM_TRAIN_EPOCHS=0 \
    BASE_PREPARED_SIDECAR_TAG=ASAP_FLOORLOG_NOMUS_SCORESPAN \
    BATCH_SIZE_PER_DEVICE=32 \
    GLOBAL_BATCH_SIZE=64 \
    DET_NUM_SAMPLES=1 \
    SAMPLING_NUM_SAMPLES=1 \
    INFER_NUM_WORKERS=8 \
    METRIC_NUM_WORKERS=8 \
    INFER_BATCH_SIZE_WINDOWS=8 \
    INFER_SCORE_SOURCE_LIST=data/asap_test_score_sources.txt \
    EVAL_CHECKPOINT_MODE=best \
    RESUME_FROM_LATEST_CHECKPOINT=0 \
    MERGE_MODE=continuation \
    CONTINUATION_DROP_RATIO=0.0 \
    bash script/run_inr_epr_pipeline.sh > "${run_dir}/launcher.log" 2>&1
  printf '%s\tGPU%s\tEND\t%s\n' "${name}" "${GPU}" "$(date '+%F %T')" | tee -a "${RUN_ROOT}/processes.tsv"
done < "${RUN_ROOT}/queue.tsv"
SH

sed -i \
  -e "s#__ROOT_DIR__#${ROOT_DIR}#g" \
  -e "s#__RUN_ROOT__#${RUN_ROOT}#g" \
  -e "s#__GPU__#${GPU}#g" \
  "${QUEUE_SCRIPT}"
chmod +x "${QUEUE_SCRIPT}"

python - "${CONFIG_DIR}/manifest.json" "${RUN_ROOT}/queue.tsv" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
lines = [f"{row['name']}\t{row['config']}\t{row['run_dir']}" for row in manifest]
Path(sys.argv[2]).write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

tmux new-session -d -s "${SESSION}" "bash '${QUEUE_SCRIPT}'"

echo "RUN_ROOT=${RUN_ROOT}"
echo "SESSION=${SESSION}"
echo "QUEUE=${RUN_ROOT}/queue.tsv"
