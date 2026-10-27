#!/usr/bin/env bash
set -euo pipefail

# Run the same test rollout/evaluator for the three pilot conditions, one
# checkpoint per GPU. Inference workers are intentionally replicated inside
# each GPU process to keep a single checkpoint busy on the full card.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GPU_LIST="${GPU_LIST:-0,1,2}"
PILOT_ROOT="${PILOT_ROOT:-results/pn_cached_velocity_pilot_baseline2}"
TRAIN_SEED="${TRAIN_SEED:-11}"
EPOCH="${EPOCH:-4}"
NUM_SAMPLES="${NUM_SAMPLES:-3}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EVAL_WORKERS="${EVAL_WORKERS:-1}"
BATCH_SIZE_WINDOWS="${BATCH_SIZE_WINDOWS:-8}"
SAMPLING_SEED="${SAMPLING_SEED:-11}"
SCORE_SOURCE_LIST="${SCORE_SOURCE_LIST:-data/asap_test_score_sources.txt}"
OUT_SUFFIX="${OUT_SUFFIX:-inference_epoch${EPOCH}_ns${NUM_SAMPLES}}"
CONDITIONS="${CONDITIONS:-baseline cached cached_p25}"
mkdir -p "${PILOT_ROOT}/logs"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
[[ "${#GPUS[@]}" -eq 3 || "${#GPUS[@]}" -eq 1 ]] || { echo "GPU_LIST must contain one or three ids" >&2; exit 2; }

record_eval() {
  python - "${PILOT_ROOT}" "$1" "$2" "$3" "${TRAIN_SEED}" "${EPOCH}" "${SAMPLING_SEED}" "${NUM_SAMPLES}" "${OUT_SUFFIX}" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
root, condition, gpu, status, seed, epoch, sampling_seed, samples, suffix = sys.argv[1:]
root = Path(root)
(root / 'manifests').mkdir(exist_ok=True)
row = dict(time_utc=datetime.now(timezone.utc).isoformat(), task='rollout_eval',
           condition=condition, gpu=gpu, status=status, training_seed=int(seed),
           epoch=int(epoch), sampling_seed=int(sampling_seed), num_samples=int(samples),
           output_dir=str(root / condition / f'seed_{seed}' / suffix),
           config=str(root / 'configs' / f'{condition}_seed_{seed}.json'),
           checkpoint=str(root / condition / f'seed_{seed}' / 'training' / f'{condition}_s{seed}' / f'epoch-{epoch}'))
with (root / 'manifests' / 'eval_runs.jsonl').open('a') as f:
    f.write(json.dumps(row) + '\n')
PY
}

run_one() {
  local gpu="$1" condition="$2"
  local config="${PILOT_ROOT}/configs/${condition}_seed_${TRAIN_SEED}.json"
  local checkpoint="${PILOT_ROOT}/${condition}/seed_${TRAIN_SEED}/training/${condition}_s${TRAIN_SEED}/epoch-${EPOCH}"
  local out="${PILOT_ROOT}/${condition}/seed_${TRAIN_SEED}/${OUT_SUFFIX}"
  mkdir -p "${out}"
  [[ -s "${out}/eval_pn_pp_metrics.json" ]] && return 0
  [[ -f "${config}" && -d "${checkpoint}" ]] || {
    echo "Missing config/checkpoint for ${condition}: ${config} ${checkpoint}" >&2
    return 2
  }
  if [[ ! -s "${out}/prediction_manifest.json" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 python src/inference/infer_inr_testset.py \
      --config "${config}" \
      --checkpoint "${checkpoint}" \
      --split test \
      --performance-dataset ASAP \
      --score-source-list "${SCORE_SOURCE_LIST}" \
      --protocol sampling \
      --num-samples "${NUM_SAMPLES}" \
      --seed "${SAMPLING_SEED}" \
      --batch-size-windows "${BATCH_SIZE_WINDOWS}" \
      --num-workers "${NUM_WORKERS}" \
      --output-dir "${out}"
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 python src/evaluate/evaluate_inr_saved_midis.py \
    --prediction-manifest "${out}/prediction_manifest.json" \
    --output-json "${out}/eval_pn_pp_metrics.json" \
    --score-source-list "${SCORE_SOURCE_LIST}" \
      --num-workers "${EVAL_WORKERS}"
}

run_logged() {
  record_eval "$1" "$2" started
  if run_one "$2" "$1"; then
    record_eval "$1" "$2" done
  else
    record_eval "$1" "$2" failed
    return 1
  fi
}

read -r -a conditions <<< "${CONDITIONS}"
[[ "${#conditions[@]}" -le 3 ]] || { echo "At most three conditions can run concurrently" >&2; exit 2; }
pids=()
if [[ "${#GPUS[@]}" -eq 1 ]]; then
  for condition in "${conditions[@]}"; do
    run_logged "${condition}" "${GPUS[0]}" >"${PILOT_ROOT}/logs/eval_${condition}_s${TRAIN_SEED}_e${EPOCH}_ns${NUM_SAMPLES}.log" 2>&1
  done
  exit 0
fi
for idx in "${!conditions[@]}"; do
  run_logged "${conditions[$idx]}" "${GPUS[$idx]}" >"${PILOT_ROOT}/logs/eval_${conditions[$idx]}_s${TRAIN_SEED}_e${EPOCH}_ns${NUM_SAMPLES}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
exit "${status}"
