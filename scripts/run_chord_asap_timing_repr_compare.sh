#!/usr/bin/env bash
set -euo pipefail

cd /home/sy/EPR/PianistTransformer

RAWLOG_EXP_DIR="${RAWLOG_EXP_DIR:-results/chord_rawlog_offsetmask_cine}"
RAWONLY_EXP_DIR="${RAWONLY_EXP_DIR:-results/chord_rawonly_offsetmask_cine}"
RAWLOG_CONFIG="${RAWLOG_CONFIG:-configs/inr0624_chord_asap_offsetmask_rawlog_multihot_nomus_cine.json}"
RAWONLY_CONFIG="${RAWONLY_CONFIG:-configs/inr0624_chord_asap_offsetmask_rawonly_multihot_nomus_cine.json}"
RAWLOG_GPUS="${RAWLOG_GPUS:-0,1}"
RAWONLY_GPUS="${RAWONLY_GPUS:-2,3}"
RAWLOG_PORT="${RAWLOG_PORT:-29611}"
RAWONLY_PORT="${RAWONLY_PORT:-29612}"
INFER_WORKERS="${INFER_WORKERS:-8}"
INFER_BATCH_WINDOWS="${INFER_BATCH_WINDOWS:-4}"
EVAL_WORKERS="${EVAL_WORKERS:-10}"

RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_INFER="${RUN_INFER:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_STATS="${RUN_STATS:-1}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

run_train() {
  local name="$1"
  local exp_dir="$2"
  local cfg="$3"
  local gpus="$4"
  local port="$5"
  local log="${exp_dir}/logs/train.log"

  mkdir -p "${exp_dir}/logs" "${exp_dir}/ckpt" "${exp_dir}/tb"
  echo "[$(date '+%F %T')] train ${name}: ${cfg} on CUDA_VISIBLE_DEVICES=${gpus}" | tee -a "${log}"
  CUDA_VISIBLE_DEVICES="${gpus}" torchrun --nproc_per_node=2 --master_port="${port}" \
    src/train/train_inr.py \
    --config "${cfg}" \
    --run_name model \
    --output_dir "${exp_dir}/ckpt" \
    --logging_dir "${exp_dir}/tb" \
    2>&1 | tee -a "${log}"
  echo "[$(date '+%F %T')] train ${name}: done" | tee -a "${log}"
}

run_infer() {
  local name="$1"
  local exp_dir="$2"
  local cfg="$3"
  local gpu="$4"
  local protocol="$5"
  local ckpt="${exp_dir}/ckpt/model/checkpoint-best"
  local out_dir="${exp_dir}/infer_${protocol}"
  local log="${exp_dir}/logs/infer_${protocol}.log"

  mkdir -p "${out_dir}" "${exp_dir}/logs"
  echo "[$(date '+%F %T')] infer ${name}/${protocol}: checkpoint=${ckpt} on CUDA_VISIBLE_DEVICES=${gpu}, workers=${INFER_WORKERS}" | tee -a "${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" python src/inference/infer_inr_testset.py \
    --config "${cfg}" \
    --checkpoint "${ckpt}" \
    --split test \
    --performance-dataset ASAP \
    --protocol "${protocol}" \
    --num-samples 1 \
    --output-dir "${out_dir}" \
    --batch-size-windows "${INFER_BATCH_WINDOWS}" \
    --num-workers "${INFER_WORKERS}" \
    --merge-mode continuation \
    2>&1 | tee -a "${log}"
}

first_gpu() {
  local gpus="$1"
  printf '%s\n' "${gpus%%,*}"
}

second_gpu() {
  local gpus="$1"
  if [[ "${gpus}" == *,* ]]; then
    local rest="${gpus#*,}"
    printf '%s\n' "${rest%%,*}"
  else
    printf '%s\n' "${gpus}"
  fi
}

run_eval() {
  local name="$1"
  local exp_dir="$2"
  local protocol="$3"
  local manifest="${exp_dir}/infer_${protocol}/prediction_manifest.json"
  local out_json="${exp_dir}/eval/${protocol}_wass.json"
  local log="${exp_dir}/logs/eval_${protocol}.log"

  mkdir -p "${exp_dir}/eval" "${exp_dir}/logs"
  echo "[$(date '+%F %T')] eval ${name}/${protocol}" | tee -a "${log}"
  python src/evaluate/evaluate_inr_saved_midis.py \
    --prediction-manifest "${manifest}" \
    --output-json "${out_json}" \
    --num-workers "${EVAL_WORKERS}" \
    2>&1 | tee -a "${log}"
}

write_eval_summary() {
  local exp_dir="$1"
  python - "${exp_dir}" <<'PY'
import json, sys
from pathlib import Path
exp = Path(sys.argv[1])
runs = []
for protocol in ("deterministic", "sampling"):
    path = exp / "eval" / f"{protocol}_wass.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    runs.append({
        "protocol": protocol,
        "num_scores": payload["num_scores"],
        "num_samples": payload["num_samples"],
        "pp_wass": payload["aggregate"]["pp_wass"],
        "pn_wass": payload["aggregate"]["pn_wass"],
        "path": str(path),
    })
(exp / "eval" / "wass_summary.json").write_text(
    json.dumps({"experiment_dir": str(exp), "runs": runs}, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
PY
}

run_stats() {
  local exp_dir="$1"
  local log="${exp_dir}/logs/stats_distributions.log"
  python scripts/plot_chord_asap_infer_distributions.py \
    --experiment-dir "${exp_dir}" \
    --processed-dir data/processed/chord_asap \
    --manifest "deterministic=${exp_dir}/infer_deterministic/prediction_manifest.json" \
    --manifest "sampling=${exp_dir}/infer_sampling/prediction_manifest.json" \
    2>&1 | tee "${log}"
}

run_one_pipeline() {
  local name="$1"
  local exp_dir="$2"
  local cfg="$3"
  local gpus="$4"
  local port="$5"

  mkdir -p "${exp_dir}/logs"
  if [[ "${RUN_TRAIN}" == "1" ]]; then
    run_train "${name}" "${exp_dir}" "${cfg}" "${gpus}" "${port}"
  fi
  if [[ "${RUN_INFER}" == "1" ]]; then
    local det_gpu
    local sampling_gpu
    det_gpu="$(first_gpu "${gpus}")"
    sampling_gpu="$(second_gpu "${gpus}")"
    run_infer "${name}" "${exp_dir}" "${cfg}" "${det_gpu}" deterministic &
    det_pid=$!
    run_infer "${name}" "${exp_dir}" "${cfg}" "${sampling_gpu}" sampling &
    sampling_pid=$!
    wait "${det_pid}"
    wait "${sampling_pid}"
  fi
  if [[ "${RUN_EVAL}" == "1" ]]; then
    run_eval "${name}" "${exp_dir}" deterministic
    run_eval "${name}" "${exp_dir}" sampling
    write_eval_summary "${exp_dir}"
  fi
  if [[ "${RUN_STATS}" == "1" ]]; then
    run_stats "${exp_dir}"
  fi
  echo "[$(date '+%F %T')] pipeline done: ${exp_dir}"
}

run_one_pipeline rawlog "${RAWLOG_EXP_DIR}" "${RAWLOG_CONFIG}" "${RAWLOG_GPUS}" "${RAWLOG_PORT}" &
rawlog_pid=$!
run_one_pipeline rawonly "${RAWONLY_EXP_DIR}" "${RAWONLY_CONFIG}" "${RAWONLY_GPUS}" "${RAWONLY_PORT}" &
rawonly_pid=$!

wait "${rawlog_pid}"
wait "${rawonly_pid}"

echo "[$(date '+%F %T')] timing representation comparison done"
