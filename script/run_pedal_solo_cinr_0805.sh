#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GPU_IDS_CSV="${GPU_IDS:-0,1,2}"
IFS=',' read -ra GPU_IDS_ARR <<< "${GPU_IDS_CSV}"
if [[ "${#GPU_IDS_ARR[@]}" -lt 1 ]]; then
  echo "GPU_IDS must contain at least one id, got: ${GPU_IDS_CSV}" >&2
  exit 2
fi

STAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_ROOT="${RUN_ROOT:-results/pedal_solo_cinr_0805_${STAMP}}"
RUNTIME_CONFIG_DIR="${RUN_ROOT}/runtime_configs"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "${RUNTIME_CONFIG_DIR}" "${LOG_DIR}"

BASE_CONFIG="${BASE_CONFIG:-configs/inr_epr/cinr__baseline_10x2_nomus.json}"
RUNTIME_CONFIG="${RUNTIME_CONFIG_DIR}/pedal_solo_cinr_ep16.json"
RUN_LABEL="${RUN_LABEL:-pedal_solo_cinr}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-48}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-16}"
INFER_GPU="${INFER_GPU:-${GPU_IDS_ARR[0]}}"
INFER_NUM_WORKERS="${INFER_NUM_WORKERS:-8}"
INFER_BATCH_SIZE_WINDOWS="${INFER_BATCH_SIZE_WINDOWS:-8}"
SAMPLING_NUM_SAMPLES="${SAMPLING_NUM_SAMPLES:-2}"
MAX_GT_PER_SCORE="${MAX_GT_PER_SCORE:-}"

python - "${BASE_CONFIG}" "${RUNTIME_CONFIG}" "${RUN_ROOT}" "${RUN_LABEL}" \
  "${PER_DEVICE_BATCH_SIZE}" "${GRADIENT_ACCUMULATION_STEPS}" "${GLOBAL_BATCH_SIZE}" \
  "${NUM_TRAIN_EPOCHS}" <<'PY'
import json
import sys
from pathlib import Path

src, dst, run_root, label, per_device_bs, grad_accum, global_bs, epochs = sys.argv[1:9]
cfg = json.loads(Path(src).read_text(encoding="utf-8"))
for key in (
    "fixed_window_split_scheme",
    "fixed_window_base_split",
    "fixed_window_train_split_name",
    "fixed_window_eval_split_name",
    "fixed_window_split_summary_path",
):
    cfg.pop(key, None)
cfg["run_name"] = "asap_nomus_10x2_cinr_pedal_solo_ep16"
cfg["output_dir"] = str(Path(run_root) / label / "training")
cfg["logging_dir"] = str(Path(run_root) / label / "tf-logs")
cfg["decoder_arch"] = "asd_pedal_solo"
cfg["prepared_sidecar_tag"] = "ASAP_MUSICAL51"
cfg["eval_from_train_fraction"] = 0.03
cfg["num_train_epochs"] = float(epochs)
cfg["max_train_epochs"] = float(epochs)
cfg["save_total_limit"] = max(2, int(cfg.get("save_total_limit", 2) or 2))
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

TRAIN_DIR="$(find "${RUN_ROOT}/${RUN_LABEL}/training" -mindepth 1 -maxdepth 1 -type d | sort | head -1)"
if [[ -z "${TRAIN_DIR}" ]]; then
  echo "Could not find train output under ${RUN_ROOT}/${RUN_LABEL}/training" >&2
  exit 1
fi
CHECKPOINT="${TRAIN_DIR}/checkpoint-best"
if [[ ! -d "${CHECKPOINT}" ]]; then
  CHECKPOINT="$(find "${TRAIN_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' | sort | tail -n 1)"
fi
if [[ -z "${CHECKPOINT}" || ! -d "${CHECKPOINT}" ]]; then
  echo "Could not find checkpoint under ${TRAIN_DIR}" >&2
  exit 1
fi

OUT_DIR="${RUN_ROOT}/${RUN_LABEL}/postprocess"
mkdir -p "${OUT_DIR}"

echo "[$(date '+%F %T')] infer ${RUN_LABEL}: ${CHECKPOINT}" | tee -a "${LOG_DIR}/postprocess.log"
CUDA_VISIBLE_DEVICES="${INFER_GPU}" PYTHONUNBUFFERED=1 \
  python src/inference/infer_inr_testset.py \
    --config "${RUNTIME_CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --split test \
    --performance-dataset ASAP \
    --device cuda \
    --protocol sampling \
    --sampling-strategy sample \
    --num-samples "${SAMPLING_NUM_SAMPLES}" \
    --num-workers "${INFER_NUM_WORKERS}" \
    --batch-size-windows "${INFER_BATCH_SIZE_WINDOWS}" \
    --output-dir "${OUT_DIR}" 2>&1 | tee -a "${LOG_DIR}/postprocess.log"

echo "[$(date '+%F %T')] eval PN/PP ${RUN_LABEL}" | tee -a "${LOG_DIR}/postprocess.log"
max_gt_args=()
if [[ -n "${MAX_GT_PER_SCORE}" ]]; then
  max_gt_args=(--max-gt-per-score "${MAX_GT_PER_SCORE}")
fi
PYTHONUNBUFFERED=1 python src/evaluate/evaluate_inr_saved_midis.py \
  --prediction-manifest "${OUT_DIR}/prediction_manifest.json" \
  --output-json "${OUT_DIR}/eval_pn_pp_metrics.json" \
  --num-workers "${INFER_NUM_WORKERS}" \
  "${max_gt_args[@]}" \
  2>&1 | tee -a "${LOG_DIR}/postprocess.log"

echo "[$(date '+%F %T')] stat MIDI pairs ${RUN_LABEL}" | tee -a "${LOG_DIR}/postprocess.log"
PYTHONUNBUFFERED=1 python src/evaluate/compute_saved_midi_mae_wass.py \
  --evaluate-list "${OUT_DIR}/evaluate_list.json" \
  --output-json "${OUT_DIR}/midi_pair_metrics.json" \
  --num-workers "${INFER_NUM_WORKERS}" \
  2>&1 | tee -a "${LOG_DIR}/postprocess.log"

echo "All ${RUN_LABEL} jobs finished." | tee -a "${LOG_DIR}/postprocess.log"
