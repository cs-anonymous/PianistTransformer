#!/usr/bin/env bash
set -euo pipefail

# Controlled first-stage experiment for PN velocity CDF-W1.
# Conditions: NLL baseline / cached-PN, each with 0/25/50% model-prefix.
# Full runs should use three GPUs and multiple seeds; SMOKE=1 runs one short
# training job per condition to validate data/cache/loss plumbing.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BASE_CONFIG="${BASE_CONFIG:-configs/inr_epr/cinr__default_dlm_k1_bounded5.json}"
OUT_ROOT="${OUT_ROOT:-results/pn_cached_velocity_$(date '+%Y%m%d_%H%M%S')}"
GPU_LIST="${GPU_LIST:-0,1,2}"
SEEDS="${SEEDS:-11,22,33,44,55}"
EPOCHS="${EPOCHS:-12}"
CACHED_LAMBDA="${CACHED_LAMBDA:-0.10}"
MAX_STEPS="${MAX_STEPS:--1}"
LIMIT_WORKS="${LIMIT_WORKS:-}"
LIMIT_PERFORMANCES="${LIMIT_PERFORMANCES:-}"
LIMIT_WINDOWS="${LIMIT_WINDOWS:-}"
SMOKE="${SMOKE:-0}"
WAIT_BETWEEN_BATCHES="${WAIT_BETWEEN_BATCHES:-0}"
CONDITIONS="${CONDITIONS:-baseline baseline_p25 baseline_p50 cached cached_p25 cached_p50}"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
[[ "${#GPUS[@]}" -eq 3 || "${#GPUS[@]}" -eq 1 ]] || {
  echo "GPU_LIST must contain one or three ids" >&2; exit 2;
}
mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/configs" "${OUT_ROOT}/manifests"

record() {
  local condition="$1" seed="$2" gpu="$3" status="$4" run="$5"
  python - "${OUT_ROOT}/manifests/runs.jsonl" "${condition}" "${seed}" "${gpu}" "${status}" "${run}" <<'PY'
import json, platform, subprocess, sys
from datetime import datetime, timezone
path, condition, seed, gpu, status, run = sys.argv[1:]
try:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
except Exception:
    commit = "unknown"
row = {
    "time_utc": datetime.now(timezone.utc).isoformat(),
    "condition": condition, "seed": int(seed), "gpu": gpu,
    "status": status, "run": run, "git_commit": commit,
    "hostname": platform.node(),
}
with open(path, "a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
PY
}

make_config() {
  local condition="$1" seed="$2" out="$3" dst="$4"
  python - "${BASE_CONFIG}" "${dst}" "${out}" "${condition}" "${seed}" "${EPOCHS}" "${MAX_STEPS}" "${LIMIT_WORKS}" "${LIMIT_PERFORMANCES}" "${LIMIT_WINDOWS}" "${CACHED_LAMBDA}" <<'PY'
import json, sys
from pathlib import Path
src, dst, out, condition, seed, epochs, max_steps, limit_works, limit_perfs, limit_windows, cached_lambda = sys.argv[1:]
cfg = json.loads(Path(src).read_text(encoding="utf-8"))
seed = int(seed)
cfg.update({
    "seed": seed, "musical_random_seed": seed, "dagger_seed": seed,
    "dagger_cache_seed": seed, "output_dir": str(Path(out) / "training"),
    "logging_dir": str(Path(out) / "tf-logs"), "run_name": f"{condition}_s{seed}",
    "num_train_epochs": float(epochs), "max_train_epochs": float(epochs),
    "max_steps": int(max_steps), "overwrite_output_dir": True,
    "auto_rollout_eval_after_train": False, "load_best_model_at_end": False,
    "metric_for_best_model": "eval_loss", "greater_is_better": False,
    "pn_velocity_target_cache": condition.startswith("cached"),
    "pn_cached_velocity_cdf_w1_lambda": float(cached_lambda) if condition.startswith("cached") else 0.0,
})
if condition.endswith("_p25") or condition.endswith("_p50"):
    ratio = 0.25 if condition.endswith("_p25") else 0.50
    cfg.update({
        "dagger_prefix_training": True,
        "dagger_cache_type": "tf_pred",
        "dagger_cache_scope": "random",
        "dagger_cache_fraction": ratio,
        "dagger_apply_prob": 1.0,
        "dagger_replacement_weights": {"full": 1.0},
        "dagger_materialize_strategy": "sample",
        "dagger_refresh_at_train_start": True,
    })
else:
    cfg["dagger_prefix_training"] = False
if limit_works:
    # Fixed-window summaries enumerate all works; a smoke subset needs the
    # ordinary manifest path instead of an incomplete fixed-split summary.
    cfg.pop("fixed_window_split_scheme", None)
    cfg.pop("fixed_window_split_summary_path", None)
    cfg.pop("fixed_window_base_split", None)
    cfg["max_train_works"] = int(limit_works)
    cfg["max_eval_works"] = int(limit_works)
if limit_perfs:
    cfg["max_performances_per_work"] = int(limit_perfs)
if limit_windows:
    cfg["max_windows_per_work"] = int(limit_windows)
if str(sys.argv[4]).startswith("cached") and int(max_steps) > 0 and int(epochs) <= 1:
    cfg.update({
        "force_eval_steps": 999999, "force_save_steps": 999999,
        "load_best_model_at_end": False,
    })
Path(dst).write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

run_one() {
  local gpu="$1" condition="$2" seed="$3" run="${OUT_ROOT}/${condition}/seed_${seed}"
  local cfg="${OUT_ROOT}/configs/${condition}_seed_${seed}.json"
  mkdir -p "${run}"
  if [[ -f "${run}/training/train_config.json" ]]; then
    record "${condition}" "${seed}" "${gpu}" skipped "${run}"
    return
  fi
  make_config "${condition}" "${seed}" "${run}" "${cfg}"
  if CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 python src/train/train_inr.py --config "${cfg}" \
      >"${OUT_ROOT}/logs/${condition}_seed_${seed}.log" 2>&1; then
    record "${condition}" "${seed}" "${gpu}" done "${run}"
  else
    record "${condition}" "${seed}" "${gpu}" failed "${run}"
    return 1
  fi
}

read -r -a conditions <<< "${CONDITIONS}"
jobs=()
idx=0
for condition in "${conditions[@]}"; do
  IFS=',' read -r -a seed_array <<< "${SEEDS}"
  for seed in "${seed_array[@]}"; do
    gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
    jobs+=("${gpu}|${condition}|${seed}")
    idx=$((idx + 1))
  done
done

if [[ "${SMOKE}" == "1" ]]; then
  jobs=("${GPUS[0]}|cached|11")
  EPOCHS=1 MAX_STEPS=2 LIMIT_WORKS=2 LIMIT_PERFORMANCES=3 LIMIT_WINDOWS=1
fi

active=()
for spec in "${jobs[@]}"; do
  IFS='|' read -r gpu condition seed <<< "${spec}"
  run_one "${gpu}" "${condition}" "${seed}" &
  active+=("$!")
  if [[ "${#active[@]}" -ge "${#GPUS[@]}" ]]; then
    for pid in "${active[@]}"; do wait "${pid}"; done
    active=()
    [[ "${WAIT_BETWEEN_BATCHES}" -gt 0 ]] && sleep "${WAIT_BETWEEN_BATCHES}"
  fi
done
for pid in "${active[@]}"; do wait "${pid}"; done
echo "completed experiment matrix under ${OUT_ROOT}"
