#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GPU_IDS_CSV="${GPU_IDS:-0,1,2}"
IFS=',' read -ra GPU_IDS_ARR <<< "${GPU_IDS_CSV}"
if [[ "${#GPU_IDS_ARR[@]}" -ne 3 ]]; then
  echo "GPU_IDS must contain exactly 3 ids, got: ${GPU_IDS_CSV}" >&2
  exit 2
fi

MATRIX_CONFIG="${MATRIX_CONFIG:-configs/inr_epr/rollout_pp_hr_nk_matrix_20260805.json}"
BASE_CONFIG="${BASE_CONFIG:-results/cinr_baseline_ckpt_signal_probe_0805_20260805_142256/runtime_configs/cinr_baseline.json}"
RUN_ROOT="${RUN_ROOT:-results/rollout_pp_hr_nk_matrix_$(date '+%Y%m%d_%H%M%S')}"
MAX_STEPS_ARG=()
if [[ -n "${MAX_STEPS:-}" ]]; then
  MAX_STEPS_ARG=(--max_steps "${MAX_STEPS}")
fi

# Refuse to launch until the Trainer consumes the new objective. This prevents
# an apparently successful matrix that silently trains the old TF-NLL objective.
if ! rg -q 'rollout_hr_batch_fraction' src/train/train_inr.py; then
  echo "Rollout-HR Trainer support is not implemented yet; refusing to launch a false experiment." >&2
  exit 3
fi

mkdir -p "${RUN_ROOT}/runtime_configs" "${RUN_ROOT}/logs"

prepare_configs() {
  python - "${BASE_CONFIG}" "${MATRIX_CONFIG}" "${RUN_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

base_path, matrix_path, run_root = map(Path, sys.argv[1:4])
base = json.loads(base_path.read_text(encoding="utf-8"))
matrix = json.loads(matrix_path.read_text(encoding="utf-8"))

# This experiment is intended to be a loss-only comparison against the paper
# baseline (musical51, 8x4, slot6). Fail loudly if another architecture is
# passed accidentally. Runtime batch/epoch invariants are checked after the
# matrix overrides have been applied below.
expected = {
    "encoder_layers_num": 8,
    "decoder_layers_num": 4,
    "disable_musical_features": False,
    "musical_feature_mode": "musical51_full",
    "slot_version": "slot6",
}
wrong = {
    key: {"expected": value, "actual": base.get(key)}
    for key, value in expected.items()
    if base.get(key) != value
}
if wrong:
    raise SystemExit(f"BASE_CONFIG is not the paper CINR baseline: {wrong}")

for lane, jobs in matrix["lanes"].items():
    for job in jobs:
        cfg = dict(base)
        cfg.update(matrix["shared"])
        cfg.update({key: value for key, value in job.items() if key != "label"})
        runtime_expected = {
            "per_device_train_batch_size": 32,
            "gradient_accumulation_steps": 2,
            "global_batch_size": 64,
            "num_train_epochs": 16.0,
        }
        runtime_wrong = {
            key: {"expected": value, "actual": cfg.get(key)}
            for key, value in runtime_expected.items()
            if cfg.get(key) != value
        }
        if runtime_wrong:
            raise SystemExit(f"Invalid single-GPU baseline runtime for {job['label']}: {runtime_wrong}")
        label = job["label"]
        cfg["run_name"] = label
        cfg["output_dir"] = str(run_root / label / "training")
        cfg["logging_dir"] = str(run_root / label / "tf-logs")
        cfg["auto_rollout_eval_after_train"] = False
        cfg["save_total_limit"] = max(4, int(cfg.get("save_total_limit", 4) or 4))
        out = run_root / "runtime_configs" / f"{label}.json"
        out.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

run_job() {
  local gpu="$1"
  local label="$2"
  local config="${RUN_ROOT}/runtime_configs/${label}.json"
  local log="${RUN_ROOT}/logs/${label}.log"
  echo "[$(date '+%F %T')] GPU ${gpu}: start ${label}" | tee -a "${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python src/train/train_inr.py --config "${config}" "${MAX_STEPS_ARG[@]}" 2>&1 | tee -a "${log}"
  echo "[$(date '+%F %T')] GPU ${gpu}: done ${label}" | tee -a "${log}"
}

run_lane() {
  local gpu="$1"
  shift
  local label
  for label in "$@"; do
    run_job "${gpu}" "${label}"
  done
}

prepare_configs
echo "RUN_ROOT ${RUN_ROOT}"
if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then
  echo "Prepared and validated runtime configs only."
  exit 0
fi
echo "GPU ${GPU_IDS_ARR[0]}: singleton PN, n10/k8"
echo "GPU ${GPU_IDS_ARR[1]}: singleton PP, n10/k8"
echo "GPU ${GPU_IDS_ARR[2]}: singleton PN+PP, n10/k8"

pids=()
run_lane "${GPU_IDS_ARR[0]}" singleton_pn_n10_k8 & pids+=("$!")
run_lane "${GPU_IDS_ARR[1]}" singleton_pp_n10_k8 & pids+=("$!")
run_lane "${GPU_IDS_ARR[2]}" singleton_pnpp_n10_k8 & pids+=("$!")

failures=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failures=$((failures + 1))
  fi
done
if [[ "${failures}" -ne 0 ]]; then
  echo "${failures} GPU lane(s) failed; see ${RUN_ROOT}/logs" >&2
  exit 1
fi
echo "All rollout PP-HR matrix jobs finished: ${RUN_ROOT}"
