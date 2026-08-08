#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="${CONFIG:-configs/inr_epr/baseline0807.json}"
RUN_ROOT="${RUN_ROOT:-results/baseline0807_epoch_hr_$(date '+%Y%m%d_%H%M%S')}"
GPU_IDS="${GPU_IDS:-0}"
EPOCHS="${EPOCHS:-1,2,3,4,5,6,7,8,9,10,11,12}"
M_VALUES="${M_VALUES:-64}"
ROLLOUT_KS="${ROLLOUT_KS:-4}"
SEED="${SEED:-42}"
INFER_NUM_WORKERS="${INFER_NUM_WORKERS:-8}"
INFER_BATCH_SIZE_WINDOWS="${INFER_BATCH_SIZE_WINDOWS:-8}"
VALIDATION_BATCH_SIZE_WINDOWS="${VALIDATION_BATCH_SIZE_WINDOWS:-4}"
TRUE_NUM_SAMPLES="${TRUE_NUM_SAMPLES:-1}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_SUMMARY="${SKIP_SUMMARY:-0}"
DEVICE_COUNT="$(awk -F',' '{print NF}' <<< "${GPU_IDS}")"

if [[ "${DEVICE_COUNT}" -ne 1 && "${DEVICE_COUNT}" -ne 3 ]]; then
  echo "GPU_IDS must contain exactly 1 or 3 ids for bs=32,gbs=96; got ${GPU_IDS}" >&2
  exit 2
fi
if [[ "${SKIP_TRAIN}" != "0" && "${SKIP_TRAIN}" != "1" ]]; then
  echo "SKIP_TRAIN must be 0 or 1" >&2
  exit 2
fi
if [[ "${SKIP_SUMMARY}" != "0" && "${SKIP_SUMMARY}" != "1" ]]; then
  echo "SKIP_SUMMARY must be 0 or 1" >&2
  exit 2
fi

PER_DEVICE_BS=32
if [[ "${DEVICE_COUNT}" -eq 1 ]]; then
  GRAD_ACCUM=3
else
  GRAD_ACCUM=1
fi

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/runtime_configs"
RUNTIME_CONFIG="${RUN_ROOT}/runtime_configs/baseline0807.json"

python - "${CONFIG}" "${RUNTIME_CONFIG}" "${RUN_ROOT}" "${PER_DEVICE_BS}" "${GRAD_ACCUM}" "${DEVICE_COUNT}" <<'PY'
import json
import sys
from pathlib import Path

src, dst, run_root, per_device_bs, grad_accum, device_count = sys.argv[1:7]
cfg = json.loads(Path(src).read_text(encoding="utf-8"))
cfg["output_dir"] = str(Path(run_root) / "training")
cfg["logging_dir"] = str(Path(run_root) / "tf-logs")
cfg["run_name"] = "baseline0807"
cfg["num_train_epochs"] = 12.0
cfg["max_train_epochs"] = 12.0
cfg["per_device_train_batch_size"] = int(per_device_bs)
cfg["per_device_eval_batch_size"] = int(per_device_bs)
cfg["gradient_accumulation_steps"] = int(grad_accum)
cfg["global_batch_size"] = 96
cfg["save_total_limit"] = 12
cfg["save_epoch_aliases"] = list(range(1, 13))
cfg["auto_rollout_eval_after_train"] = False
cfg["resume_trainer_state"] = False
Path(dst).write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY

echo "RUN_ROOT=${RUN_ROOT}" | tee "${RUN_ROOT}/logs/run.log"
echo "RUNTIME_CONFIG=${RUNTIME_CONFIG}" | tee -a "${RUN_ROOT}/logs/run.log"

if [[ "${SKIP_TRAIN}" == "0" ]]; then
  IFS=',' read -ra GPU_ARRAY <<< "${GPU_IDS}"
  if [[ "${DEVICE_COUNT}" -eq 3 ]]; then
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" PYTHONUNBUFFERED=1 \
      torchrun --nnodes=1 --nproc_per_node=3 \
      src/train/train_inr.py --config "${RUNTIME_CONFIG}" \
      2>&1 | tee "${RUN_ROOT}/logs/train.log"
  else
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" PYTHONUNBUFFERED=1 \
      python src/train/train_inr.py --config "${RUNTIME_CONFIG}" \
      2>&1 | tee "${RUN_ROOT}/logs/train.log"
  fi
fi

checkpoint_for_epoch() {
  local epoch="$1"
  local checkpoint="${RUN_ROOT}/training/baseline0807/epoch-${epoch}"
  if [[ ! -d "${checkpoint}" ]]; then
    echo "Missing checkpoint: ${checkpoint}" >&2
    exit 1
  fi
  echo "${checkpoint}"
}

run_epoch() {
  local epoch="$1"
  local gpu="$2"
  local root="${RUN_ROOT}/ep${epoch}"
  local test_dir="${root}/test"
  local valid_json="${root}/validation_window_rollout_hr.json"
  local checkpoint
  checkpoint="$(checkpoint_for_epoch "${epoch}")"
  mkdir -p "${test_dir}"

  echo "[$(date '+%F %T')] epoch=${epoch} test inference on GPU ${gpu}" | tee -a "${RUN_ROOT}/logs/eval.log"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 \
    python src/inference/infer_inr_testset.py \
      --config "${RUNTIME_CONFIG}" \
      --checkpoint "${checkpoint}" \
      --split test \
      --performance-dataset ASAP \
      --protocol sampling \
      --sampling-strategy sample \
      --num-samples "${TRUE_NUM_SAMPLES}" \
      --num-workers "${INFER_NUM_WORKERS}" \
      --batch-size-windows "${INFER_BATCH_SIZE_WINDOWS}" \
      --device cuda \
      --output-dir "${test_dir}" \
      2>&1 | tee -a "${RUN_ROOT}/logs/eval.log"

  python src/evaluate/evaluate_inr_saved_midis.py \
    --prediction-manifest "${test_dir}/prediction_manifest.json" \
    --output-json "${test_dir}/eval_pn_pp_metrics.json" \
    --num-workers "${INFER_NUM_WORKERS}" \
    2>&1 | tee -a "${RUN_ROOT}/logs/eval.log"

  echo "[$(date '+%F %T')] epoch=${epoch} validation m=${M_VALUES} k=${ROLLOUT_KS} on GPU ${gpu}" \
    | tee -a "${RUN_ROOT}/logs/eval.log"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 \
    python src/evaluate/eval_inr_window_rollout_hr.py \
      --config "${RUNTIME_CONFIG}" \
      --checkpoint "${checkpoint}" \
      --output-json "${valid_json}" \
      --m-values "${M_VALUES}" \
      --rollout-ks "${ROLLOUT_KS}" \
      --seed "${SEED}" \
      --batch-size-windows "${VALIDATION_BATCH_SIZE_WINDOWS}" \
      --num-workers 0 \
      --device cuda \
      --materialize-strategy sample \
      --feedback-strategy sample \
      --fail-on-insufficient-works \
      2>&1 | tee -a "${RUN_ROOT}/logs/eval.log"
}

IFS=',' read -ra EPOCH_ARRAY <<< "${EPOCHS}"
IFS=',' read -ra GPU_ARRAY <<< "${GPU_IDS}"
pids=()
for lane in "${!GPU_ARRAY[@]}"; do
  (
    gpu="${GPU_ARRAY[$lane]}"
    for index in "${!EPOCH_ARRAY[@]}"; do
      if [[ $((index % DEVICE_COUNT)) -ne "${lane}" ]]; then
        continue
      fi
      run_epoch "${EPOCH_ARRAY[$index]}" "${gpu}"
    done
  ) &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "${pid}"
done

if [[ "${SKIP_SUMMARY}" == "0" ]]; then
  python src/evaluate/summarize_baseline0807_epoch_hr.py \
    --run-root "${RUN_ROOT}" \
    --epochs "${EPOCHS}" \
    --m-values "${M_VALUES}" \
    --rollout-ks "${ROLLOUT_KS}" \
    --output-json "${RUN_ROOT}/epoch_hr_summary.json" \
    --output-csv "${RUN_ROOT}/epoch_hr_summary.csv"
fi

echo "Completed baseline0807 epoch HR sweep: ${RUN_ROOT}"
