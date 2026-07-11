#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/zeroioi_loss_source_4gpu/${STAMP}}"
CONFIG_DIR="${RUN_ROOT}/configs"
BASE_CONFIG="${BASE_CONFIG:-results/slot5_zeroembed_dual_2x2gpu/20260711_dual_dist/slot5-128-zeroembed-dual-zero-folded/config.json}"
mkdir -p "${CONFIG_DIR}"

python - "${BASE_CONFIG}" "${CONFIG_DIR}" <<'PY'
import json
import sys
from pathlib import Path

base_path = Path(sys.argv[1])
config_dir = Path(sys.argv[2])
base = json.loads(base_path.read_text(encoding="utf-8"))

variants = {
    "simple-folded-abs-slot-zeroembed": {
        "run_name": "slot5_128_simple_folded_abs_slot_zeroembed_pad50",
        "zero_ioi_transform": "folded_abs",
        "zero_ioi_dual_distribution_mode": "none",
        "zero_score_ioi_embedding": True,
        "zero_timing_head_condition": False,
        "zero_ioi_dual_duration": True,
    },
    "dual-zero-folded-timing-zeroembed": {
        "run_name": "slot5_128_dual_zero_folded_timing_zeroembed_pad50",
        "zero_ioi_transform": "none",
        "zero_ioi_dual_distribution_mode": "zero_folded",
        "zero_score_ioi_embedding": False,
        "zero_timing_head_condition": True,
        "zero_ioi_dual_duration": True,
    },
    "dual-zero-folded-ioi-only": {
        "run_name": "slot5_128_dual_zero_folded_ioi_only_pad50",
        "zero_ioi_transform": "none",
        "zero_ioi_dual_distribution_mode": "zero_folded",
        "zero_score_ioi_embedding": True,
        "zero_timing_head_condition": False,
        "zero_ioi_dual_duration": False,
    },
    "dual-zero-folded-no-zeroembed": {
        "run_name": "slot5_128_dual_zero_folded_no_zeroembed_pad50",
        "zero_ioi_transform": "none",
        "zero_ioi_dual_distribution_mode": "zero_folded",
        "zero_score_ioi_embedding": False,
        "zero_timing_head_condition": False,
        "zero_ioi_dual_duration": True,
    },
}

for name, overrides in variants.items():
    cfg = dict(base)
    cfg.update(overrides)
    cfg.pop("resume_path", None)
    cfg["resume_trainer_state"] = False
    cfg["output_dir"] = f"unused/{name}/training"
    cfg["logging_dir"] = f"unused/{name}/tf-logs"
    (config_dir / f"{name}.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

allowed = {
    "run_name", "zero_ioi_transform", "zero_ioi_dual_distribution_mode",
    "zero_score_ioi_embedding", "zero_timing_head_condition",
    "zero_ioi_dual_duration", "output_dir", "logging_dir",
}
loaded = {
    name: json.loads((config_dir / f"{name}.json").read_text(encoding="utf-8"))
    for name in variants
}
unexpected = []
for key in sorted(set().union(*(cfg.keys() for cfg in loaded.values()))):
    if len({json.dumps(cfg.get(key), sort_keys=True) for cfg in loaded.values()}) > 1 and key not in allowed:
        unexpected.append(key)
if unexpected:
    raise SystemExit(f"Unexpected cross-experiment differences: {unexpected}")

report = {"base_config": str(base_path), "allowed_differences": sorted(allowed), "variants": variants}
(config_dir / "config_report.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2, ensure_ascii=False))
PY

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1; configs generated under ${CONFIG_DIR}"
  exit 0
fi

launch_one() {
  local gpu="$1" name="$2" run_dir="${RUN_ROOT}/$2"
  local log_path="${run_dir}/launcher.log"
  mkdir -p "${run_dir}"
  setsid env CUDA_VISIBLE_DEVICES="${gpu}" \
    CONFIG="${CONFIG_DIR}/${name}.json" RUN_DIR_OVERRIDE="${run_dir}" \
    BASE_ASAP_ONLY=1 BASE_NUM_TRAIN_EPOCHS=16 ADAPT_NUM_TRAIN_EPOCHS=0 \
    BATCH_SIZE_PER_DEVICE=32 GLOBAL_BATCH_SIZE=64 \
    DET_NUM_SAMPLES=1 SAMPLING_NUM_SAMPLES=1 \
    INFER_NUM_WORKERS=8 METRIC_NUM_WORKERS=8 INFER_BATCH_SIZE_WINDOWS=8 \
    INFER_SCORE_SOURCE_LIST=data/cheap15_score_sources.txt \
    EVAL_CHECKPOINT_MODE=latest RESUME_FROM_LATEST_CHECKPOINT=1 \
    MERGE_MODE=continuation CONTINUATION_DROP_RATIO=0.0 \
    bash script/run_inr_epr_pipeline.sh >"${log_path}" 2>&1 < /dev/null &
  local pid=$!
  printf '%s\tGPU%s\tPID%s\t%s\t%s\n' "${name}" "${gpu}" "${pid}" "${run_dir}" "${log_path}" \
    | tee -a "${RUN_ROOT}/processes.tsv"
}

launch_one 0 simple-folded-abs-slot-zeroembed
launch_one 1 dual-zero-folded-timing-zeroembed
launch_one 2 dual-zero-folded-ioi-only
launch_one 3 dual-zero-folded-no-zeroembed

echo "RUN_ROOT=${RUN_ROOT}"
