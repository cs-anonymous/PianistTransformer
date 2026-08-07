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

STAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_ROOT="${RUN_ROOT:-results/g2_cinr_epoch_sweep_0804_${STAMP}}"
RUNTIME_CONFIG_DIR="${RUN_ROOT}/runtime_configs"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "${RUNTIME_CONFIG_DIR}" "${LOG_DIR}"

BASE_CONFIG="${BASE_CONFIG:-configs/inr_epr/cinr__baseline_10x2_nomus.json}"
RUNTIME_CONFIG="${RUNTIME_CONFIG_DIR}/g2_cinr_ep24.json"
RUN_LABEL="${RUN_LABEL:-g2_cinr}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-48}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-24}"
SAVE_EPOCHS="${SAVE_EPOCHS:-16,20,24}"
EPOCH_SAVE_STEPS="${EPOCH_SAVE_STEPS:-544}"
INFER_NUM_WORKERS="${INFER_NUM_WORKERS:-8}"
INFER_BATCH_SIZE_WINDOWS="${INFER_BATCH_SIZE_WINDOWS:-8}"
SAMPLING_NUM_SAMPLES="${SAMPLING_NUM_SAMPLES:-2}"
MAX_GT_PER_SCORE="${MAX_GT_PER_SCORE:-}"

python - "${BASE_CONFIG}" "${RUNTIME_CONFIG}" "${RUN_ROOT}" "${RUN_LABEL}" \
  "${PER_DEVICE_BATCH_SIZE}" "${GRADIENT_ACCUMULATION_STEPS}" "${GLOBAL_BATCH_SIZE}" \
  "${NUM_TRAIN_EPOCHS}" "${SAVE_EPOCHS}" "${EPOCH_SAVE_STEPS}" <<'PY'
import json
import sys
from pathlib import Path

src, dst, run_root, label, per_device_bs, grad_accum, global_bs, epochs, save_epochs, epoch_save_steps = sys.argv[1:11]
cfg = json.loads(Path(src).read_text(encoding="utf-8"))
for key in (
    "fixed_window_split_scheme",
    "fixed_window_base_split",
    "fixed_window_train_split_name",
    "fixed_window_eval_split_name",
    "fixed_window_split_summary_path",
):
    cfg.pop(key, None)
epoch_values = [float(part.strip()) for part in save_epochs.split(",") if part.strip()]
cfg["run_name"] = "asap_nomus_10x2_cinr_g2_ep24"
cfg["output_dir"] = str(Path(run_root) / label / "training")
cfg["logging_dir"] = str(Path(run_root) / label / "tf-logs")
cfg["decoder_arch"] = "asd_group2"
cfg["prepared_sidecar_tag"] = "ASAP_MUSICAL51"
cfg["eval_from_train_fraction"] = 0.03
cfg["num_train_epochs"] = float(epochs)
cfg["max_train_epochs"] = float(epochs)
cfg["save_epoch_aliases"] = epoch_values
cfg["force_eval_steps"] = int(epoch_save_steps)
cfg["force_save_steps"] = int(epoch_save_steps)
cfg["save_total_limit"] = max(4, int(cfg.get("save_total_limit", 2) or 2))
cfg["per_device_train_batch_size"] = int(per_device_bs)
cfg["per_device_eval_batch_size"] = int(per_device_bs)
cfg["gradient_accumulation_steps"] = int(grad_accum)
cfg["global_batch_size"] = int(global_bs)
cfg["musical_feature_mode"] = "none"
cfg["disable_musical_features"] = True
cfg["auto_rollout_eval_after_train"] = False
Path(dst).parent.mkdir(parents=True, exist_ok=True)
Path(dst).write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY

echo "RUN_ROOT ${RUN_ROOT}" | tee -a "${LOG_DIR}/run.log"
echo "RUNTIME_CONFIG ${RUNTIME_CONFIG}" | tee -a "${LOG_DIR}/run.log"
echo "[$(date '+%F %T')] train ${RUN_LABEL} with 3-GPU DDP" | tee -a "${LOG_DIR}/train.log"
CUDA_VISIBLE_DEVICES="${GPU_IDS_CSV}" PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  torchrun --nnodes=1 --nproc_per_node=3 \
    src/train/train_inr.py --config "${RUNTIME_CONFIG}" \
    2>&1 | tee -a "${LOG_DIR}/train.log"
echo "[$(date '+%F %T')] train done" | tee -a "${LOG_DIR}/train.log"

train_output_dir() {
  local out
  out="$(find "${RUN_ROOT}/${RUN_LABEL}/training" -mindepth 1 -maxdepth 1 -type d | sort | head -1)"
  if [[ -z "${out}" ]]; then
    echo "Could not find train output under ${RUN_ROOT}/${RUN_LABEL}/training" >&2
    exit 1
  fi
  echo "${out}"
}

checkpoint_for_epoch() {
  local epoch="$1"
  local train_dir="$2"
  local ckpt="${train_dir}/epoch-${epoch}"
  if [[ ! -d "${ckpt}" ]]; then
    echo "Could not find ${ckpt}" >&2
    exit 1
  fi
  echo "${ckpt}"
}

run_eval_one() {
  local epoch="$1"
  local gpu="$2"
  local train_dir="$3"
  local checkpoint
  local out_dir
  checkpoint="$(checkpoint_for_epoch "${epoch}" "${train_dir}")"
  out_dir="${RUN_ROOT}/${RUN_LABEL}_ep${epoch}/postprocess"
  mkdir -p "${out_dir}"

  echo "[$(date '+%F %T')] infer ${RUN_LABEL}_ep${epoch}: ${checkpoint}" | tee -a "${LOG_DIR}/postprocess.log"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 \
    python src/inference/infer_inr_testset.py \
      --config "${RUNTIME_CONFIG}" \
      --checkpoint "${checkpoint}" \
      --split test \
      --performance-dataset ASAP \
      --device cuda \
      --protocol sampling \
      --sampling-strategy sample \
      --num-samples "${SAMPLING_NUM_SAMPLES}" \
      --num-workers "${INFER_NUM_WORKERS}" \
      --batch-size-windows "${INFER_BATCH_SIZE_WINDOWS}" \
      --output-dir "${out_dir}" 2>&1 | tee -a "${LOG_DIR}/postprocess.log"

  echo "[$(date '+%F %T')] eval PN/PP ${RUN_LABEL}_ep${epoch}" | tee -a "${LOG_DIR}/postprocess.log"
  local max_gt_args=()
  if [[ -n "${MAX_GT_PER_SCORE}" ]]; then
    max_gt_args=(--max-gt-per-score "${MAX_GT_PER_SCORE}")
  fi
  PYTHONUNBUFFERED=1 python src/evaluate/evaluate_inr_saved_midis.py \
    --prediction-manifest "${out_dir}/prediction_manifest.json" \
    --output-json "${out_dir}/eval_pn_pp_metrics.json" \
    --num-workers "${INFER_NUM_WORKERS}" \
    "${max_gt_args[@]}" \
    2>&1 | tee -a "${LOG_DIR}/postprocess.log"

  echo "[$(date '+%F %T')] stat MIDI pairs ${RUN_LABEL}_ep${epoch}" | tee -a "${LOG_DIR}/postprocess.log"
  PYTHONUNBUFFERED=1 python src/evaluate/compute_saved_midi_mae_wass.py \
    --evaluate-list "${out_dir}/evaluate_list.json" \
    --output-json "${out_dir}/midi_pair_metrics.json" \
    --num-workers "${INFER_NUM_WORKERS}" \
    2>&1 | tee -a "${LOG_DIR}/postprocess.log"
}

TRAIN_DIR="$(train_output_dir)"
IFS=',' read -ra SAVE_EPOCHS_ARR <<< "${SAVE_EPOCHS}"
pids=()
for idx in 0 1 2; do
  run_eval_one "${SAVE_EPOCHS_ARR[$idx]}" "${GPU_IDS_ARR[$idx]}" "${TRAIN_DIR}" &
  pids+=("$!")
done

failures=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failures=$((failures + 1))
  fi
done
if [[ "${failures}" -ne 0 ]]; then
  echo "${failures} eval job(s) failed; see ${LOG_DIR}/postprocess.log" >&2
  exit 1
fi

echo "All ${RUN_LABEL} epoch-sweep eval jobs finished." | tee -a "${LOG_DIR}/postprocess.log"
