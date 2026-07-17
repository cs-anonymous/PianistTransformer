#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/inr_epr_pipeline/bounded_slot_support_probe_${STAMP}}"
BASE_CONFIG="results/inr_epr_pipeline/unified_musical_baselines_20260717_151510/configs/cinr_bounded_5pct.json"
CONFIG_DIR="${RUN_ROOT}/configs"
mkdir -p "${CONFIG_DIR}"

python - "${BASE_CONFIG}" "${CONFIG_DIR}" "${RUN_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

base_path = Path(sys.argv[1])
config_dir = Path(sys.argv[2])
run_root = Path(sys.argv[3])
base = json.loads(base_path.read_text(encoding="utf-8"))

def common(name):
    cfg = dict(base)
    cfg.pop("resume_path", None)
    cfg.pop("resume_from_checkpoint", None)
    cfg.update(
        {
            "run_name": f"bounded_probe_{name}",
            "output_dir": str(run_root / name / "training"),
            "logging_dir": str(run_root / name / "tf-logs"),
            "timing_control_mode": "dinr_floor_log",
            "epr_timing_target": "floor_log_deviation",
            "slot_version": "slot8",
            "musical_feature_mode": "musical51_full",
            "disable_musical_features": False,
            "dlm_ioi_nonzero_min": -2.0,
            "dlm_ioi_nonzero_max": 1.0,
            "dlm_ioi_nonzero_scale_max": 0.15,
            "dlm_duration_min": -2.0,
            "dlm_duration_max": 1.0,
            "dlm_duration_scale_max": 0.15,
            "sampling_top_p": 0.90,
            "sampling_top_k": 0,
            "dlm_sampling_temperature": 1.0,
            "dlm_sampling_top_p": 0.90,
            "dlm_sampling_top_k": 0,
            "dinr_sampling_temperature": 1.0,
            "dinr_sampling_top_p": 0.90,
            "dinr_sampling_top_k": 0,
            "seed": 42,
            "num_train_epochs": 16.0,
            "max_train_epochs": 16.0,
        }
    )
    return cfg

configs = {}

slot6 = common("slot6_only")
slot6["slot_version"] = "slot6"
configs["slot6_only"] = slot6

support = common("support_m1_1_only")
support["dlm_ioi_nonzero_min"] = -1.0
support["dlm_ioi_nonzero_max"] = 1.0
support["dlm_ioi_nonzero_scale_max"] = 0.10
configs["support_m1_1_only"] = support

both = common("slot6_support_m1_1")
both["slot_version"] = "slot6"
both["dlm_ioi_nonzero_min"] = -1.0
both["dlm_ioi_nonzero_max"] = 1.0
both["dlm_ioi_nonzero_scale_max"] = 0.10
configs["slot6_support_m1_1"] = both

manifest = []
for name, cfg in configs.items():
    path = config_dir / f"{name}.json"
    path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest.append({"name": name, "config": str(path), "run_dir": str(run_root / name)})

(config_dir / "manifest.json").write_text(
    json.dumps(
        {
            "base_config": str(base_path),
            "shared": {
                "timing_control_mode": "dinr_floor_log",
                "distribution": "cinr_bounded_5pct",
                "musical_feature_mode": "musical51_full",
                "duration_support": [-2.0, 1.0],
                "sampling": {"temperature": 1.0, "top_p": 0.90, "top_k": 0, "num_samples": 2},
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

tmux new-session -d -s "probe_slot6_${STAMP: -6}" "bash '${WORKER}' 0 slot6_only"
tmux new-session -d -s "probe_sup_${STAMP: -6}" "bash '${WORKER}' 1 support_m1_1_only"
tmux new-session -d -s "probe_both_${STAMP: -6}" "bash '${WORKER}' 2 slot6_support_m1_1"

echo "RUN_ROOT=${RUN_ROOT}"
echo "SESSIONS=probe_slot6_${STAMP: -6},probe_sup_${STAMP: -6},probe_both_${STAMP: -6}"
