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
RUN_ROOT="${RUN_ROOT:-results/cinr_baseline_ckpt_signal_probe_0805_${STAMP}}"
RUN_LABEL="${RUN_LABEL:-cinr_baseline}"
BASE_CONFIG="${BASE_CONFIG:-configs/inr_epr/cinr__default_dlm_k1_bounded5.json}"
RUNTIME_CONFIG_DIR="${RUN_ROOT}/runtime_configs"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "${RUNTIME_CONFIG_DIR}" "${LOG_DIR}"

RUNTIME_CONFIG="${RUNTIME_CONFIG_DIR}/${RUN_LABEL}.json"
PROBE_SCORE_LIST="${RUN_ROOT}/probe3_score_sources.txt"
TEST_SCORE_LIST="${TEST_SCORE_LIST:-data/asap_test_score_sources.txt}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-48}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-24}"
SAVE_EPOCHS="${SAVE_EPOCHS:-8,10,12,14,16,18,20,22,24}"
FULL_TEST_EPOCHS="${FULL_TEST_EPOCHS:-16,20,24}"
INFER_NUM_WORKERS="${INFER_NUM_WORKERS:-8}"
INFER_BATCH_SIZE_WINDOWS="${INFER_BATCH_SIZE_WINDOWS:-8}"
SAMPLING_NUM_SAMPLES="${SAMPLING_NUM_SAMPLES:-2}"
MAX_GT_PER_SCORE="${MAX_GT_PER_SCORE:-}"

python - "${BASE_CONFIG}" "${RUNTIME_CONFIG}" "${RUN_ROOT}" "${RUN_LABEL}" \
  "${PER_DEVICE_BATCH_SIZE}" "${GRADIENT_ACCUMULATION_STEPS}" "${GLOBAL_BATCH_SIZE}" \
  "${NUM_TRAIN_EPOCHS}" "${SAVE_EPOCHS}" <<'PY'
import json
import sys
from pathlib import Path

src, dst, run_root, label, per_device_bs, grad_accum, global_bs, epochs, save_epochs = sys.argv[1:10]
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
cfg["run_name"] = f"asap_musical51_cinr_baseline_ckpt_signal_0805"
cfg["output_dir"] = str(Path(run_root) / label / "training")
cfg["logging_dir"] = str(Path(run_root) / label / "tf-logs")
cfg["prepared_sidecar_tag"] = "ASAP_MUSICAL51"
cfg["eval_from_train_fraction"] = 0.03
cfg["num_train_epochs"] = float(epochs)
cfg["max_train_epochs"] = float(epochs)
cfg["save_epoch_aliases"] = epoch_values
cfg["save_epoch_alias_tolerance"] = 0.50
cfg["save_total_limit"] = max(len(epoch_values) + 2, int(cfg.get("save_total_limit", 2) or 2))
cfg["per_device_train_batch_size"] = int(per_device_bs)
cfg["per_device_eval_batch_size"] = int(per_device_bs)
cfg["gradient_accumulation_steps"] = int(grad_accum)
cfg["global_batch_size"] = int(global_bs)
cfg["rollout_eval_enabled"] = True
cfg["rollout_eval_k"] = 1
cfg["rollout_eval_weight"] = 0.0
cfg["rollout_eval_materialize_strategy"] = "sample"
cfg["rollout_eval_feedback_strategy"] = "sample"
cfg["auto_rollout_eval_after_train"] = False
cfg["load_best_model_at_end"] = True
cfg["metric_for_best_model"] = "eval_loss"
cfg["greater_is_better"] = False
Path(dst).parent.mkdir(parents=True, exist_ok=True)
Path(dst).write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY

python - "${TEST_SCORE_LIST}" "${PROBE_SCORE_LIST}" <<'PY'
import sys
from pathlib import Path

src, dst = [Path(item) for item in sys.argv[1:3]]
items = [line.strip() for line in src.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
if len(items) < 3:
    raise SystemExit(f"Need at least 3 score sources in {src}")
selected = [items[0], items[len(items) // 2], items[-1]]
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text("\n".join(selected) + "\n", encoding="utf-8")
PY

echo "RUN_ROOT ${RUN_ROOT}" | tee -a "${LOG_DIR}/run.log"
echo "RUNTIME_CONFIG ${RUNTIME_CONFIG}" | tee -a "${LOG_DIR}/run.log"
echo "PROBE_SCORE_LIST ${PROBE_SCORE_LIST}" | tee -a "${LOG_DIR}/run.log"
echo "SAVE_EPOCHS ${SAVE_EPOCHS}" | tee -a "${LOG_DIR}/run.log"
echo "FULL_TEST_EPOCHS ${FULL_TEST_EPOCHS}" | tee -a "${LOG_DIR}/run.log"

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

run_infer_eval() {
  local epoch="$1"
  local gpu="$2"
  local mode="$3"
  local score_list="$4"
  local train_dir="$5"
  local log_file="$6"
  local checkpoint
  local out_dir
  local start
  local end
  checkpoint="$(checkpoint_for_epoch "${epoch}" "${train_dir}")"
  out_dir="${RUN_ROOT}/${RUN_LABEL}_ep${epoch}/${mode}"
  mkdir -p "${out_dir}"

  echo "[$(date '+%F %T')] infer ${mode} ep${epoch} on gpu ${gpu}: ${checkpoint}" | tee -a "${log_file}"
  start="$(date +%s)"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 \
    python src/inference/infer_inr_testset.py \
      --config "${RUNTIME_CONFIG}" \
      --checkpoint "${checkpoint}" \
      --split test \
      --score-source-list "${score_list}" \
      --performance-dataset ASAP \
      --device cuda \
      --protocol sampling \
      --sampling-strategy sample \
      --num-samples "${SAMPLING_NUM_SAMPLES}" \
      --num-workers "${INFER_NUM_WORKERS}" \
      --batch-size-windows "${INFER_BATCH_SIZE_WINDOWS}" \
      --output-dir "${out_dir}" 2>&1 | tee -a "${log_file}"
  end="$(date +%s)"
  echo "$((end - start))" > "${out_dir}/infer.seconds"

  echo "[$(date '+%F %T')] eval PN/PP ${mode} ep${epoch}" | tee -a "${log_file}"
  local max_gt_args=()
  if [[ -n "${MAX_GT_PER_SCORE}" ]]; then
    max_gt_args=(--max-gt-per-score "${MAX_GT_PER_SCORE}")
  fi
  start="$(date +%s)"
  PYTHONUNBUFFERED=1 python src/evaluate/evaluate_inr_saved_midis.py \
    --prediction-manifest "${out_dir}/prediction_manifest.json" \
    --output-json "${out_dir}/eval_pn_pp_metrics.json" \
    --num-workers "${INFER_NUM_WORKERS}" \
    "${max_gt_args[@]}" \
    2>&1 | tee -a "${log_file}"
  end="$(date +%s)"
  echo "$((end - start))" > "${out_dir}/eval.seconds"
}

run_epoch_queue() {
  local gpu="$1"
  local mode="$2"
  local score_list="$3"
  local train_dir="$4"
  local log_file="$5"
  shift 5
  local epochs=("$@")
  for epoch in "${epochs[@]}"; do
    run_infer_eval "${epoch}" "${gpu}" "${mode}" "${score_list}" "${train_dir}" "${log_file}"
  done
}

TRAIN_DIR="$(train_output_dir)"
IFS=',' read -ra SAVE_EPOCHS_ARR <<< "${SAVE_EPOCHS}"
probe_pids=()
for idx in 0 1 2; do
  epochs_for_gpu=()
  for epoch_idx in "${!SAVE_EPOCHS_ARR[@]}"; do
    if [[ $((epoch_idx % 3)) -eq "${idx}" ]]; then
      epochs_for_gpu+=("${SAVE_EPOCHS_ARR[$epoch_idx]}")
    fi
  done
  run_epoch_queue "${GPU_IDS_ARR[$idx]}" "probe" "${PROBE_SCORE_LIST}" "${TRAIN_DIR}" "${LOG_DIR}/probe_gpu${idx}.log" "${epochs_for_gpu[@]}" &
  probe_pids+=("$!")
done

failures=0
for pid in "${probe_pids[@]}"; do
  if ! wait "${pid}"; then
    failures=$((failures + 1))
  fi
done
if [[ "${failures}" -ne 0 ]]; then
  echo "${failures} probe job(s) failed; see ${LOG_DIR}/probe_gpu*.log" >&2
  exit 1
fi

IFS=',' read -ra FULL_TEST_EPOCHS_ARR <<< "${FULL_TEST_EPOCHS}"
full_pids=()
for idx in 0 1 2; do
  epochs_for_gpu=()
  for epoch_idx in "${!FULL_TEST_EPOCHS_ARR[@]}"; do
    if [[ $((epoch_idx % 3)) -eq "${idx}" ]]; then
      epochs_for_gpu+=("${FULL_TEST_EPOCHS_ARR[$epoch_idx]}")
    fi
  done
  if [[ "${#epochs_for_gpu[@]}" -gt 0 ]]; then
    run_epoch_queue "${GPU_IDS_ARR[$idx]}" "full_test" "${TEST_SCORE_LIST}" "${TRAIN_DIR}" "${LOG_DIR}/full_gpu${idx}.log" "${epochs_for_gpu[@]}" &
    full_pids+=("$!")
  fi
done

failures=0
for pid in "${full_pids[@]}"; do
  if ! wait "${pid}"; then
    failures=$((failures + 1))
  fi
done
if [[ "${failures}" -ne 0 ]]; then
  echo "${failures} full-test job(s) failed; see ${LOG_DIR}/full_gpu*.log" >&2
  exit 1
fi

PYTHONUNBUFFERED=1 python src/evaluate/summarize_ckpt_signal_probe.py \
  --run-root "${RUN_ROOT}" \
  --run-label "${RUN_LABEL}" \
  2>&1 | tee -a "${LOG_DIR}/summary.log"

echo "All checkpoint-signal probe jobs finished: ${RUN_ROOT}" | tee -a "${LOG_DIR}/run.log"
