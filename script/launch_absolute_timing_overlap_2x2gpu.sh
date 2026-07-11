#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/absolute_timing_overlap_2x2gpu/${STAMP}}"
CONFIG_DIR="${RUN_ROOT}/configs"
BASE_CONFIG="${BASE_CONFIG:-results/zeroioi_loss_source_4gpu/20260711_loss_source/simple-folded-abs-slot-zeroembed/config.json}"
mkdir -p "${CONFIG_DIR}"

python - "${BASE_CONFIG}" "${CONFIG_DIR}" <<'PY'
import json, sys
from pathlib import Path

base_path = Path(sys.argv[1])
config_dir = Path(sys.argv[2])
base = json.loads(base_path.read_text(encoding="utf-8"))

common = {
    "epr_timing_target": "raw_log_absolute",
    "timing_control_mode": "raw_log",
    "timing_input_normalization": "log1p_t_over_50_5000",
    "timing_log_scale": 50.0,
    "legacy_dual_timing_head": False,
    "zero_ioi_transform": "none",
    "zero_ioi_positive_support": False,
    "zero_ioi_dual_distribution_mode": "none",
    "zero_ioi_dual_duration": True,
    "zero_timing_head_condition": False,
    "zero_score_ioi_embedding": True,
    "output_continuous_dim": 9,
    "continuous_dim": 9,
    "resume_trainer_state": False,
}
variants = {
    "B-absolute-log-overlap125": {
        "run_name": "slot5_128_absolute_log_slot_zeroembed_overlap125",
        "overlap_ratio": 0.125,
    },
    "C-absolute-log-overlap50": {
        "run_name": "slot5_128_absolute_log_slot_zeroembed_overlap50",
        "overlap_ratio": 0.5,
    },
}
for name, overrides in variants.items():
    cfg = dict(base)
    cfg.update(common)
    cfg.update(overrides)
    cfg.pop("resume_path", None)
    cfg["output_dir"] = f"unused/{name}/training"
    cfg["logging_dir"] = f"unused/{name}/tf-logs"
    (config_dir / f"{name}.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

allowed = {"run_name", "overlap_ratio", "output_dir", "logging_dir"}
loaded = {n: json.loads((config_dir / f"{n}.json").read_text()) for n in variants}
unexpected = []
for key in sorted(set().union(*(x.keys() for x in loaded.values()))):
    if len({json.dumps(x.get(key), sort_keys=True) for x in loaded.values()}) > 1 and key not in allowed:
        unexpected.append(key)
if unexpected:
    raise SystemExit(f"Unexpected B/C differences: {unexpected}")
(config_dir / "config_report.json").write_text(
    json.dumps({"base": str(base_path), "common": common, "variants": variants}, indent=2) + "\n"
)
print(json.dumps({"base": str(base_path), "common": common, "variants": variants}, indent=2))
PY

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

launch_one() {
  local gpus="$1" name="$2" run_dir="${RUN_ROOT}/$2"
  mkdir -p "${run_dir}"
  setsid env CUDA_VISIBLE_DEVICES="${gpus}" \
    CONFIG="${CONFIG_DIR}/${name}.json" RUN_DIR_OVERRIDE="${run_dir}" \
    BASE_ASAP_ONLY=1 BASE_NUM_TRAIN_EPOCHS=16 ADAPT_NUM_TRAIN_EPOCHS=0 \
    BATCH_SIZE_PER_DEVICE=32 GLOBAL_BATCH_SIZE=64 \
    DET_NUM_SAMPLES=1 SAMPLING_NUM_SAMPLES=1 \
    INFER_NUM_WORKERS=8 METRIC_NUM_WORKERS=8 INFER_BATCH_SIZE_WINDOWS=8 \
    INFER_SCORE_SOURCE_LIST=data/cheap15_score_sources.txt \
    EVAL_CHECKPOINT_MODE=latest RESUME_FROM_LATEST_CHECKPOINT=1 \
    MERGE_MODE=continuation CONTINUATION_DROP_RATIO=0.0 \
    bash script/run_inr_epr_pipeline.sh >"${run_dir}/launcher.log" 2>&1 < /dev/null &
  printf '%s\tGPUs=%s\tPID=%s\t%s\n' "${name}" "${gpus}" "$!" "${run_dir}" | tee -a "${RUN_ROOT}/processes.tsv"
}

launch_one 0,1 B-absolute-log-overlap125
launch_one 2,3 C-absolute-log-overlap50
echo "RUN_ROOT=${RUN_ROOT}"
