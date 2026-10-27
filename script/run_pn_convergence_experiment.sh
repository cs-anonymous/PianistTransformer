#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
BASE_CONFIG="${BASE_CONFIG:-configs/inr_epr/baseline0807.json}"
OUT_ROOT="${OUT_ROOT:-results/pn_convergence_$(date '+%Y%m%d_%H%M%S')}"
GPU_LIST="${GPU_LIST:-0,1,2}"
SEEDS="${SEEDS:-11,22,33}"
EPOCHS="${EPOCHS:-60}"
CACHED_LAMBDA="${CACHED_LAMBDA:-0.10}"
CONDITIONS="${CONDITIONS:-baseline cached}"
IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
mkdir -p "${OUT_ROOT}"/{configs,logs,manifests}

make_config() {
  local condition="$1" seed="$2" out="$3" dst="$4"
  python - "${BASE_CONFIG}" "${dst}" "${out}" "${condition}" "${seed}" "${EPOCHS}" "${CACHED_LAMBDA}" <<'PY'
import json, sys
from pathlib import Path
src, dst, out, condition, seed, epochs, cached_lambda = sys.argv[1:]
cfg = json.loads(Path(src).read_text())
seed = int(seed)
cfg.update({'seed': seed, 'musical_random_seed': seed, 'dagger_seed': seed,
 'dagger_cache_seed': seed, 'output_dir': str(Path(out) / 'training'),
 'logging_dir': str(Path(out) / 'tf-logs'), 'run_name': f'{condition}_s{seed}',
 'overwrite_output_dir': True, 'max_steps': -1, 'num_train_epochs': float(epochs),
 'max_train_epochs': float(epochs), 'per_device_train_batch_size': 32,
 'gradient_accumulation_steps': 3, 'global_batch_size': 96,
 'per_device_eval_batch_size': 32, 'lr_scheduler_type': 'constant',
 'learning_rate': 0.0003, 'warmup_steps': 50, 'warmup_ratio': 0,
 'eval_strategy': 'epoch', 'save_strategy': 'epoch',
 'save_epoch_aliases': [4, 8, 12, 24, 36, 48, 60],
 'save_epoch_alias_tolerance': 0.001, 'save_total_limit': 8,
 'load_best_model_at_end': False, 'metric_for_best_model': 'eval_loss',
 'greater_is_better': False, 'auto_rollout_eval_after_train': False,
 'pn_velocity_target_cache': condition.startswith('cached'),
 'pn_cached_velocity_cdf_w1_lambda': float(cached_lambda) if condition.startswith('cached') else 0.0,
 'convergence_diagnostics': True, 'convergence_min_epochs': 36,
 'convergence_patience': 8, 'convergence_relative_tolerance': 0.002,
 'convergence_probe_examples': 512, 'convergence_full_train_confirmations': 3,
 'convergence_lr_drop_epoch': 24, 'convergence_low_lr_factor': 0.1,
 'retain_optimizer_state': True, 'require_full_resume_state': True,
 'dataloader_num_workers': 4, 'eval_dataloader_num_workers': 2,
 'eval_dataloader_persistent_workers': False, 'report_to': 'none', 'bf16': True,
 'dagger_prefix_training': False})
Path(dst).write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + '\n')
PY
}

record() {
  python - "${OUT_ROOT}/manifests/runs.jsonl" "$@" <<'PY'
import json, sys
from datetime import datetime, timezone
path, condition, seed, gpu, status, run = sys.argv[1:]
row = {'time_utc': datetime.now(timezone.utc).isoformat(), 'experiment': 'convergence',
 'condition': condition, 'seed': int(seed), 'gpu': gpu, 'status': status, 'run': run}
with open(path, 'a') as f: f.write(json.dumps(row) + '\n')
PY
}

run_one() {
  local gpu="$1" condition="$2" seed="$3" run="${OUT_ROOT}/${condition}/seed_${seed}"
  local cfg="${OUT_ROOT}/configs/${condition}_seed_${seed}.json"
  mkdir -p "${run}"
  if [[ -s "${run}/training/convergence_result.json" ]]; then record "${condition}" "${seed}" "${gpu}" skipped "${run}"; return 0; fi
  make_config "${condition}" "${seed}" "${run}" "${cfg}"
  record "${condition}" "${seed}" "${gpu}" started "${run}"
  if CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 python src/train/train_inr.py --config "${cfg}" >"${OUT_ROOT}/logs/${condition}_seed_${seed}.log" 2>&1; then
    record "${condition}" "${seed}" "${gpu}" done "${run}"
  else
    record "${condition}" "${seed}" "${gpu}" failed "${run}"; return 1
  fi
}

read -r -a conditions <<< "${CONDITIONS}"
read -r -a seeds <<< "${SEEDS//,/ }"
jobs=(); index=0
for condition in "${conditions[@]}"; do
  for seed in "${seeds[@]}"; do
    jobs+=("${GPUS[$((index % ${#GPUS[@]}))]}|${condition}|${seed}")
    index=$((index + 1))
  done
done
active=()
for spec in "${jobs[@]}"; do
  IFS='|' read -r gpu condition seed <<< "${spec}"
  run_one "${gpu}" "${condition}" "${seed}" & active+=("$!")
  if [[ "${#active[@]}" -ge "${#GPUS[@]}" ]]; then
    for pid in "${active[@]}"; do wait "${pid}"; done
    active=()
  fi
done
for pid in "${active[@]}"; do wait "${pid}"; done
echo "completed convergence experiment under ${OUT_ROOT}"
