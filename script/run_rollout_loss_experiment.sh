#!/usr/bin/env bash
set -euo pipefail

# Matched baseline versus teacher-forcing plus one-step model-feedback loss.
# Each condition uses the same seed, data, optimizer, and checkpoint schedule.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BASE_CONFIG="${BASE_CONFIG:-configs/inr_epr/baseline0807.json}"
OUT_ROOT="${OUT_ROOT:-results/rollout_loss_$(date '+%Y%m%d_%H%M%S')}"
GPU_LIST="${GPU_LIST:-0,1,2}"
SEEDS="${SEEDS:-11}"
EPOCHS="${EPOCHS:-12}"
MAX_STEPS="${MAX_STEPS:--1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
LIMIT_WORKS="${LIMIT_WORKS:-}"
LIMIT_PERFORMANCES="${LIMIT_PERFORMANCES:-}"
LIMIT_WINDOWS="${LIMIT_WINDOWS:-}"
SMOKE="${SMOKE:-0}"
CONDITIONS="${CONDITIONS:-baseline rollout10 rollout25}"

# Keep a sufficiently dense, common checkpoint grid for correlation analysis.
# The grid is clipped to the requested training horizon so every condition has
# the same epoch rows.
CHECKPOINT_EPOCHS="${CHECKPOINT_EPOCHS:-4,8,12,16,20}"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
[[ "${#GPUS[@]}" -eq 1 || "${#GPUS[@]}" -eq 3 ]] || {
  echo "GPU_LIST must contain one or three ids" >&2
  exit 2
}
mkdir -p "${OUT_ROOT}"/{configs,logs,manifests}

make_config() {
  local condition="$1" seed="$2" out="$3" dst="$4"
  python - "${BASE_CONFIG}" "${dst}" "${out}" "${condition}" "${seed}" "${EPOCHS}" "${MAX_STEPS}" "${LIMIT_WORKS}" "${LIMIT_PERFORMANCES}" "${LIMIT_WINDOWS}" "${TRAIN_BATCH_SIZE}" "${GRADIENT_ACCUMULATION_STEPS}" "${WARMUP_STEPS}" "${CHECKPOINT_EPOCHS}" <<'PY'
import json, sys
from pathlib import Path

src, dst, out, condition, seed, epochs, max_steps, limit_works, limit_perfs, limit_windows, train_batch, grad_accum, warmup_steps, checkpoint_epochs_text = sys.argv[1:]
cfg = json.loads(Path(src).read_text(encoding="utf-8"))
weights = {"baseline": 0.0, "rollout10": 0.10, "rollout25": 0.25}
if condition not in weights:
    raise ValueError(f"unknown condition: {condition}")
seed = int(seed)
checkpoint_epochs = [int(x) for x in checkpoint_epochs_text.split(",") if x.strip()]
cfg.update({
    "seed": seed,
    "musical_random_seed": seed,
    "dagger_seed": seed,
    "dagger_cache_seed": seed,
    "output_dir": str(Path(out) / "training"),
    "logging_dir": str(Path(out) / "tf-logs"),
    "run_name": f"{condition}_s{seed}",
    "num_train_epochs": float(epochs),
    "max_train_epochs": float(epochs),
    "max_steps": int(max_steps),
    "per_device_train_batch_size": int(train_batch),
    "per_device_eval_batch_size": int(train_batch),
    "gradient_accumulation_steps": int(grad_accum),
    "global_batch_size": int(train_batch) * int(grad_accum),
    "warmup_steps": int(warmup_steps),
    "warmup_ratio": 0.0,
    "logging_steps": 1,
    "logging_strategy": "steps",
    "overwrite_output_dir": True,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "save_epoch_aliases": [x for x in checkpoint_epochs if x <= int(float(epochs))],
    "save_epoch_alias_tolerance": 0.001,
    "load_best_model_at_end": False,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    "auto_rollout_eval_after_train": False,
    "rollout_train_weight": weights[condition],
    "rollout_train_k": 1,
    "rollout_train_feedback_strategy": "sample",
    "report_to": "none",
})
if limit_works:
    cfg.pop("fixed_window_split_scheme", None)
    cfg.pop("fixed_window_split_summary_path", None)
    cfg.pop("fixed_window_base_split", None)
    cfg["max_train_works"] = int(limit_works)
    cfg["max_eval_works"] = int(limit_works)
    # The ordinary capped manifest has no valid split; evaluate checkpoints
    # with the standalone protocol instead of invoking an empty Trainer eval.
    cfg["eval_strategy"] = "no"
    cfg["disable_train_eval"] = True
if limit_perfs:
    cfg["max_performances_per_work"] = int(limit_perfs)
if limit_windows:
    cfg["max_windows_per_work"] = int(limit_windows)
if int(max_steps) > 0 and float(epochs) <= 1:
    cfg.update({
        "force_eval_steps": 999999,
        "force_save_steps": 999999,
        "load_best_model_at_end": False,
    })
Path(dst).write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

run_one() {
  local gpu="$1" condition="$2" seed="$3"
  local out="${OUT_ROOT}/${condition}/seed_${seed}"
  local cfg="${OUT_ROOT}/configs/${condition}_seed_${seed}.json"
  mkdir -p "${out}"
  make_config "${condition}" "${seed}" "${out}" "${cfg}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 python src/train/train_inr.py --config "${cfg}" \
    >"${OUT_ROOT}/logs/${condition}_seed_${seed}.log" 2>&1
}

if [[ "${SMOKE}" == "1" ]]; then
  EPOCHS=1
  MAX_STEPS=2
  LIMIT_WORKS=2
  LIMIT_PERFORMANCES=3
  LIMIT_WINDOWS=1
  WARMUP_STEPS=0
fi

read -r -a conditions <<< "${CONDITIONS}"
read -r -a seeds <<< "${SEEDS//,/ }"
jobs=()
idx=0
for condition in "${conditions[@]}"; do
  for seed in "${seeds[@]}"; do
    jobs+=("${GPUS[$((idx % ${#GPUS[@]}))]}|${condition}|${seed}")
    idx=$((idx + 1))
  done
done

active=()
for spec in "${jobs[@]}"; do
  IFS='|' read -r gpu condition seed <<< "${spec}"
  run_one "${gpu}" "${condition}" "${seed}" &
  active+=("$!")
  if [[ "${#active[@]}" -ge "${#GPUS[@]}" ]]; then
    for pid in "${active[@]}"; do wait "${pid}"; done
    active=()
  fi
done
for pid in "${active[@]}"; do wait "${pid}"; done
echo "completed rollout-loss experiment under ${OUT_ROOT}"
