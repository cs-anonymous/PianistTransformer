#!/usr/bin/env bash
set -euo pipefail

# Matched CINR experiment with 50% note-window overlap, edge validation
# windows, and exclusion of every training window intersecting validation.
# One independent job is assigned to each GPU.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OUT_ROOT="${OUT_ROOT:-results/cinr_overlap50_24ep_$(date '+%Y%m%d_%H%M%S')}"
GPU_LIST="${GPU_LIST:-0,1,2}"
EPOCHS="${EPOCHS:-24}"
SAMPLES="${SAMPLING_NUM_SAMPLES:-3}"
CHECKPOINT_EPOCHS="4,8,12,16,20"
SPLIT_SCHEME="train_valid_asap3_nonasap05_edge50_v1"
SPLIT_SUMMARY="data/train_valid_asap3_nonasap05_edge50_v1_summary.json"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
[[ "${#GPUS[@]}" -eq 3 ]] || { echo "GPU_LIST must contain exactly three ids" >&2; exit 2; }
mkdir -p "${OUT_ROOT}"/{configs,logs}

make_config() {
  local name="$1" src="$2" weight="$3"
  python - "${src}" "${OUT_ROOT}/configs/${name}.json" "${OUT_ROOT}/${name}" "${weight}" <<'PY'
import json, sys
from pathlib import Path
src, dst, out, rollout_weight = sys.argv[1:]
cfg = json.loads(Path(src).read_text(encoding='utf-8'))
cfg.update({
    'overlap_ratio': 0.5,
    'max_windows_per_work': None,
    'fixed_window_split_scheme': 'train_valid_asap3_nonasap05_edge50_v1',
    'fixed_window_split_summary_path': 'data/train_valid_asap3_nonasap05_edge50_v1_summary.json',
    'exclude_windows_overlapping_valid': True,
    'num_train_epochs': 24.0,
    'max_train_epochs': 24.0,
    'output_dir': str(Path(out) / 'training'),
    'logging_dir': str(Path(out) / 'tf-logs'),
    'run_name': Path(out).name,
    'overwrite_output_dir': True,
    'eval_strategy': 'epoch',
    'save_strategy': 'epoch',
    'force_epoch_eval': True,
    'logging_strategy': 'steps',
    'logging_steps': 1,
    'save_epoch_aliases': [4, 8, 12, 16, 20],
    'save_epoch_alias_tolerance': 0.001,
    'load_best_model_at_end': True,
    'metric_for_best_model': 'eval_loss',
    'greater_is_better': False,
    'early_stopping_patience': 0,
    'auto_rollout_eval_after_train': False,
    'rollout_train_weight': float(rollout_weight),
    'rollout_train_k': 1,
    'rollout_train_feedback_strategy': 'sample',
    'report_to': 'none',
})
Path(dst).write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
PY
}

make_config cinr_slot_sum configs/inr_epr/extra__rep_slot_sum.json 0.0
make_config cinr_canonical configs/inr_epr/cinr__default_dlm_k1_bounded5.json 0.0
make_config cinr_canonical_rollout10 configs/inr_epr/cinr__default_dlm_k1_bounded5.json 0.10

run_one() {
  local gpu="$1" name="$2"
  local batch_size=32
  if [[ "${name}" == *rollout* ]]; then
    # Rollout adds a differentiable second forward pass.  Keep the same
    # effective global batch while halving per-device activation memory.
    batch_size=16
  fi
  local cfg="${OUT_ROOT}/configs/${name}.json"
  local run_dir="${OUT_ROOT}/${name}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 \
    BASE_NUM_TRAIN_EPOCHS="${EPOCHS}" ADAPT_NUM_TRAIN_EPOCHS=0 \
    RESUME_FROM_LATEST_CHECKPOINT=0 PIPELINE_STAGE_START=train \
    EVAL_CHECKPOINT_MODE=best SAMPLING_NUM_SAMPLES="${SAMPLES}" \
    INFER_NUM_WORKERS=8 METRIC_NUM_WORKERS=8 \
    BATCH_SIZE_PER_DEVICE="${batch_size}" GLOBAL_BATCH_SIZE=64 \
    CONFIG="${cfg}" RUN_DIR_OVERRIDE="${run_dir}" \
    bash script/run_inr_epr_pipeline.sh >"${OUT_ROOT}/logs/${name}.log" 2>&1
}

run_one "${GPUS[0]}" cinr_slot_sum & p0=$!
run_one "${GPUS[1]}" cinr_canonical & p1=$!
run_one "${GPUS[2]}" cinr_canonical_rollout10 & p2=$!
status=0
for p in "$p0" "$p1" "$p2"; do wait "$p" || status=$?; done
exit "${status}"
