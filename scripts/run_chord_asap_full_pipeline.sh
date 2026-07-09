#!/usr/bin/env bash
set -euo pipefail

cd /home/sy/EPR/PianistTransformer

EXP_NAME="${EXP_NAME:-inr0624_chord_asap_sn_rawlog_multihot_nomus}"
EXP_DIR="${EXP_DIR:-results/${EXP_NAME}}"
SINE_CONFIG="${SINE_CONFIG:-configs/inr0624_chord_asap_sn_rawlog_multihot_nomus_sine.json}"
CINE_CONFIG="${CINE_CONFIG:-configs/inr0624_chord_asap_sn_rawlog_multihot_nomus_cine.json}"
SINE_GPUS="${SINE_GPUS:-0,1}"
CINE_GPUS="${CINE_GPUS:-2,3}"
SINE_PORT="${SINE_PORT:-29601}"
CINE_PORT="${CINE_PORT:-29602}"
INFER_WORKERS="${INFER_WORKERS:-2}"
INFER_BATCH_WINDOWS="${INFER_BATCH_WINDOWS:-4}"
EVAL_WORKERS="${EVAL_WORKERS:-10}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_INFER="${RUN_INFER:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_STATS="${RUN_STATS:-1}"
RUN_MODELS="${RUN_MODELS:-sine,cine}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

mkdir -p "${EXP_DIR}/logs" "${EXP_DIR}/train" "${EXP_DIR}/infer" "${EXP_DIR}/eval" "${EXP_DIR}/stats"

run_train_job() {
  local model_name="$1"
  local gpus="$2"
  local cfg="$3"
  local port="$4"
  local out_dir="${EXP_DIR}/train/${model_name}"
  local log="${EXP_DIR}/logs/train_${model_name}.log"

  mkdir -p "${out_dir}"
  echo "[$(date '+%F %T')] train ${model_name}: ${cfg} on CUDA_VISIBLE_DEVICES=${gpus}" | tee -a "${log}"
  CUDA_VISIBLE_DEVICES="${gpus}" torchrun --nproc_per_node=2 --master_port="${port}" \
    src/train/train_inr.py --config "${cfg}" \
    --output_dir "${out_dir}/model" \
    --logging_dir "${out_dir}/tf-logs" \
    2>&1 | tee -a "${log}"
  echo "[$(date '+%F %T')] train ${model_name}: done" | tee -a "${log}"
}

checkpoint_path() {
  local model_name="$1"
  local cfg="$2"
  local run_name
  run_name="$(python - "${cfg}" <<'PY'
import json, sys
from pathlib import Path
cfg = json.loads(Path(sys.argv[1]).read_text())
print(cfg.get("run_name") or "")
PY
)"
  local exp_ckpt="${EXP_DIR}/train/${model_name}/model/${run_name}/checkpoint-best"
  local legacy_ckpt
  legacy_ckpt="$(python - "${cfg}" <<'PY'
import json, sys
from pathlib import Path
cfg = json.loads(Path(sys.argv[1]).read_text())
run_name = cfg.get("run_name")
out = Path(cfg["output_dir"]) / run_name / "checkpoint-best"
print(out)
PY
)"
  if [[ -d "${exp_ckpt}" ]]; then
    printf '%s\n' "${exp_ckpt}"
  elif [[ -d "${legacy_ckpt}" ]]; then
    printf '%s\n' "${legacy_ckpt}"
  else
    printf '%s\n' "${exp_ckpt}"
  fi
}

run_infer_job() {
  local model_name="$1"
  local protocol="$2"
  local gpus="$3"
  local cfg="$4"
  local ckpt="$5"
  local out_dir="${EXP_DIR}/infer/${model_name}/${protocol}"
  local log="${EXP_DIR}/logs/infer_${model_name}_${protocol}.log"

  mkdir -p "${out_dir}"
  echo "[$(date '+%F %T')] infer ${model_name}/${protocol}: checkpoint=${ckpt}" | tee -a "${log}"
  CUDA_VISIBLE_DEVICES="${gpus}" python src/inference/infer_inr_testset.py \
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
  echo "[$(date '+%F %T')] infer ${model_name}/${protocol}: done" | tee -a "${log}"
}

run_eval_job() {
  local model_name="$1"
  local protocol="$2"
  local manifest="${EXP_DIR}/infer/${model_name}/${protocol}/prediction_manifest.json"
  local out_json="${EXP_DIR}/eval/${model_name}_${protocol}_wass.json"
  local log="${EXP_DIR}/logs/eval_${model_name}_${protocol}.log"

  echo "[$(date '+%F %T')] eval ${model_name}/${protocol}" | tee -a "${log}"
  python src/evaluate/evaluate_inr_saved_midis.py \
    --prediction-manifest "${manifest}" \
    --output-json "${out_json}" \
    --num-workers "${EVAL_WORKERS}" \
    2>&1 | tee -a "${log}"
  echo "[$(date '+%F %T')] eval ${model_name}/${protocol}: done" | tee -a "${log}"
}

write_eval_summary() {
  python - "${EXP_DIR}" "${RUN_MODELS}" <<'PY'
import json, sys
from pathlib import Path

exp = Path(sys.argv[1])
models = [item.strip() for item in sys.argv[2].split(",") if item.strip()]
runs = []
for model in models:
    for protocol in ("deterministic", "sampling"):
        path = exp / "eval" / f"{model}_{protocol}_wass.json"
        payload = json.loads(path.read_text())
        runs.append({
            "model": model,
            "protocol": protocol,
            "num_scores": payload["num_scores"],
            "num_samples": payload["num_samples"],
            "pp_wass": payload["aggregate"]["pp_wass"],
            "pn_wass": payload["aggregate"]["pn_wass"],
            "path": str(path),
        })
out = {"experiment_dir": str(exp), "runs": runs}
(exp / "eval" / "wass_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print(exp / "eval" / "wass_summary.json")
PY
}

if [[ "${RUN_TRAIN}" == "1" ]]; then
  pids=()
  if [[ ",${RUN_MODELS}," == *",sine,"* ]]; then
    run_train_job sine "${SINE_GPUS}" "${SINE_CONFIG}" "${SINE_PORT}" &
    pids+=("$!")
  fi
  if [[ ",${RUN_MODELS}," == *",cine,"* ]]; then
    run_train_job cine "${CINE_GPUS}" "${CINE_CONFIG}" "${CINE_PORT}" &
    pids+=("$!")
  fi
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
fi

if [[ ",${RUN_MODELS}," == *",sine,"* ]]; then
  SINE_CKPT="${SINE_CKPT:-$(checkpoint_path sine "${SINE_CONFIG}")}"
fi
if [[ ",${RUN_MODELS}," == *",cine,"* ]]; then
  CINE_CKPT="${CINE_CKPT:-$(checkpoint_path cine "${CINE_CONFIG}")}"
fi

if [[ "${RUN_INFER}" == "1" ]]; then
  pids=()
  if [[ ",${RUN_MODELS}," == *",sine,"* ]]; then
    run_infer_job sine deterministic "${SINE_GPUS}" "${SINE_CONFIG}" "${SINE_CKPT}" &
    pids+=("$!")
  fi
  if [[ ",${RUN_MODELS}," == *",cine,"* ]]; then
    run_infer_job cine deterministic "${CINE_GPUS}" "${CINE_CONFIG}" "${CINE_CKPT}" &
    pids+=("$!")
  fi
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done

  pids=()
  if [[ ",${RUN_MODELS}," == *",sine,"* ]]; then
    run_infer_job sine sampling "${SINE_GPUS}" "${SINE_CONFIG}" "${SINE_CKPT}" &
    pids+=("$!")
  fi
  if [[ ",${RUN_MODELS}," == *",cine,"* ]]; then
    run_infer_job cine sampling "${CINE_GPUS}" "${CINE_CONFIG}" "${CINE_CKPT}" &
    pids+=("$!")
  fi
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
  pids=()
  for model in ${RUN_MODELS//,/ }; do
    run_eval_job "${model}" deterministic &
    pids+=("$!")
    run_eval_job "${model}" sampling &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
  write_eval_summary
fi

if [[ "${RUN_STATS}" == "1" ]]; then
  python scripts/plot_chord_asap_infer_distributions.py \
    --experiment-dir "${EXP_DIR}" \
    --processed-dir data/processed/chord_asap \
    2>&1 | tee "${EXP_DIR}/logs/stats_distributions.log"
fi

echo "[$(date '+%F %T')] pipeline done: ${EXP_DIR}"
