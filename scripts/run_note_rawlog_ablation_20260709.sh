#!/usr/bin/env bash
set -euo pipefail

cd /home/sy/EPR/PianistTransformer

EXP_ROOT="${EXP_ROOT:-results/note_rawlog_ablation_20260709}"
BASE_CONFIG="${BASE_CONFIG:-backup/result0708/inr_epr_pipeline/launch_rawlog_3exp_20260708_235220/exp2_sine_nomus_tfmask50/training/inr_2026-07-09-00-19-58/train_config.json}"
LANE_A_GPUS="${LANE_A_GPUS:-0,1}"
LANE_B_GPUS="${LANE_B_GPUS:-2,3}"
LANE_A_PORT="${LANE_A_PORT:-29701}"
LANE_B_PORT="${LANE_B_PORT:-29702}"
INFER_WORKERS="${INFER_WORKERS:-8}"
INFER_BATCH_WINDOWS="${INFER_BATCH_WINDOWS:-4}"
ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-8}"
ROLLOUT_BATCH_WINDOWS="${ROLLOUT_BATCH_WINDOWS:-8}"
EVAL_WORKERS="${EVAL_WORKERS:-10}"
ROLLOUT_KS="${ROLLOUT_KS:-0,1,4,16,full}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_INFER="${RUN_INFER:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_ROLLOUT="${RUN_ROLLOUT:-1}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"
TRAIN_PARALLEL="${TRAIN_PARALLEL:-0}"
RUNS="${RUNS:-clean,tf_pred,stable,stable_tf_pred}"
GENERATE_CONFIGS="${GENERATE_CONFIGS:-1}"
TF_PRED_RESUME_PATH="${TF_PRED_RESUME_PATH:-}"
TF_PRED_RESUME_TRAINER_STATE="${TF_PRED_RESUME_TRAINER_STATE:-1}"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"

mkdir -p "${EXP_ROOT}/configs" "${EXP_ROOT}/logs" configs

generate_configs() {
  python - "${BASE_CONFIG}" "${EXP_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

base_path = Path(sys.argv[1])
exp_root = Path(sys.argv[2])
base = json.loads(base_path.read_text(encoding="utf-8"))

common_updates = {
    "adapt_on_asap_after_train": False,
    "auto_rollout_eval_after_train": False,
    "resume_path": None,
    "resume_trainer_state": False,
    "tf_embedding_mask_keep_prob": 1.0,
    "tf_embedding_mask_score": False,
    "tf_embedding_mask_decoder": False,
    "dagger_prefix_training": False,
    "stable_dynamics_training": False,
    "stable_contract_loss": False,
    "disable_musical_features": True,
    "note_embedding_mode": "sine",
    "epr_distribution": "skew_normal",
    "epr_timing_target": "raw_log_deviation",
    "timing_control_mode": "raw_log",
    "continuous_dim": 9,
    "input_continuous_dim": 68,
    "output_continuous_dim": 9,
    "block_notes": 512,
    "overlap_ratio": 0.125,
    "num_train_epochs": 16.0,
    "max_train_epochs": 16.0,
    "eval_every_epochs": 1.0,
    "eval_steps": None,
    "save_steps": None,
    "save_every_steps": None,
    "save_total_limit": 3,
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 32,
    "gradient_accumulation_steps": 1,
    "global_batch_size": 64,
    "dataloader_num_workers": 4,
    "eval_dataloader_num_workers": 4,
    "dataloader_persistent_workers": True,
    "eval_dataloader_persistent_workers": False,
    "ddp_find_unused_parameters": True,
    "ddp_broadcast_buffers": False,
    "prediction_loss_only": True,
    "report_to": "none",
    "train_performance_dataset": "ASAP",
    "eval_performance_dataset": "ASAP",
    "eval_split": "valid",
    "eval_include_all_performance_dataset": "ASAP",
    "prepared_sidecar_tag": "ASAP",
    "use_prepared_sidecar": True,
    "precompute_dataset_items": False,
    "precompute_eval_dataset_items": False,
    "early_stopping_patience": 0,
    "early_stopping_threshold": 0.0,
}

stable_updates = {
    "stable_dynamics_training": True,
    "stable_apply_prob": 0.3,
    "stable_channels": ["ioi", "duration"],
    "stable_noise_modes": {
        "zero_mean": {
            "prob": 0.5,
            "ioi_mu": 0.0,
            "ioi_sigma": 0.01,
            "duration_mu": 0.0,
            "duration_sigma": 0.01,
        },
        "positive_bias": {
            "prob": 0.25,
            "ioi_mu": 0.003,
            "ioi_sigma": 0.01,
            "duration_mu": 0.003,
            "duration_sigma": 0.01,
        },
        "variance_inflation": {
            "prob": 0.25,
            "ioi_mu": 0.0,
            "ioi_sigma": 0.025,
            "duration_mu": 0.0,
            "duration_sigma": 0.02,
        },
    },
    "stable_contract_loss": True,
    "stable_contract_alpha": 1.0,
    "stable_contract_lambda": 0.05,
    "stable_contract_eps": 1e-6,
}

tf_pred_updates = {
    "dagger_prefix_training": True,
    "dagger_cache_type": "tf_pred",
    "dagger_cache_scope": "next_interval",
    "dagger_cache_max_interval_fraction": 0.5,
    "dagger_window_curriculum": "linear",
    "dagger_window_curriculum_start": 0.0,
    "dagger_window_curriculum_end": 1.0,
    "dagger_cache_fraction": 0.5,
    "dagger_cache_max_items": None,
    "dagger_cache_batch_size": 32,
    "dagger_cache_num_workers": 4,
    "dagger_materialize_strategy": "sample",
    "dagger_apply_prob": 1.0,
    "dagger_refresh_at_train_start": True,
    "dagger_refresh_on_eval": True,
}

runs = {
    "clean": {},
    "stable": stable_updates,
    "tf_pred": tf_pred_updates,
    "stable_tf_pred": {**stable_updates, **tf_pred_updates},
}

remove_keys = {
    "score_input_continuous_dim",
    "decoder_input_continuous_dim",
}

for name, updates in runs.items():
    cfg = dict(base)
    for key in remove_keys:
        cfg.pop(key, None)
    cfg.update(common_updates)
    cfg.update(updates)
    cfg["run_name"] = "model"
    cfg["output_dir"] = str(exp_root / name / "ckpt")
    cfg["logging_dir"] = str(exp_root / name / "tb")
    cfg["seed"] = int({"clean": 4201, "stable": 4202, "tf_pred": 4203, "stable_tf_pred": 4204}[name])
    cfg["stable_seed"] = cfg["seed"]
    cfg["dagger_seed"] = cfg["seed"]
    run_dir = exp_root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    Path("configs", f"inr0624_note_rawlog_sine_nomus_{name}_20260709.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (exp_root / "configs" / f"{name}.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"{name}\t{run_dir / 'config.json'}")
PY
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

train_one() {
  local name="$1"
  local gpus="$2"
  local port="$3"
  local run_dir="${EXP_ROOT}/${name}"
  local cfg="${run_dir}/config.json"
  local train_cfg="${cfg}"
  local log="${run_dir}/logs/train.log"

  mkdir -p "${run_dir}/logs" "${run_dir}/ckpt" "${run_dir}/tb"
  if [[ "${name}" == "tf_pred" && -n "${TF_PRED_RESUME_PATH}" ]]; then
    train_cfg="${run_dir}/config.resume.json"
    python - "${cfg}" "${train_cfg}" "${TF_PRED_RESUME_PATH}" "${TF_PRED_RESUME_TRAINER_STATE}" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
resume_path = sys.argv[3]
resume_trainer_state = sys.argv[4].strip().lower() not in {"0", "false", "no", "off"}
cfg = json.loads(src.read_text(encoding="utf-8"))
cfg["resume_path"] = resume_path
cfg["resume_trainer_state"] = resume_trainer_state
dst.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(dst)
PY
  fi

  echo "[$(date '+%F %T')] train ${name}: ${train_cfg} on ${gpus}" | tee -a "${log}"
  CUDA_VISIBLE_DEVICES="${gpus}" torchrun --nnodes=1 --nproc_per_node=2 \
    --master_addr=127.0.0.1 --master_port="${port}" \
    src/train/train_inr.py --config "${train_cfg}" \
    --run_name model \
    --output_dir "${run_dir}/ckpt" \
    --logging_dir "${run_dir}/tb" \
    2>&1 | tee -a "${log}"
  echo "[$(date '+%F %T')] train ${name}: done" | tee -a "${log}"
}

infer_one() {
  local name="$1"
  local gpu="$2"
  local protocol="$3"
  local run_dir="${EXP_ROOT}/${name}"
  local cfg="${run_dir}/ckpt/model/train_config.json"
  local ckpt="${run_dir}/ckpt/model/checkpoint-best"
  local out_dir="${run_dir}/infer_${protocol}"
  local log="${run_dir}/logs/infer_${protocol}.log"

  mkdir -p "${out_dir}" "${run_dir}/logs"
  echo "[$(date '+%F %T')] infer ${name}/${protocol}: gpu=${gpu}, workers=${INFER_WORKERS}" | tee -a "${log}"
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

eval_one() {
  local name="$1"
  local protocol="$2"
  local run_dir="${EXP_ROOT}/${name}"
  local manifest="${run_dir}/infer_${protocol}/prediction_manifest.json"
  local out_json="${run_dir}/eval/${protocol}_wass.json"
  local log="${run_dir}/logs/eval_${protocol}.log"

  mkdir -p "${run_dir}/eval" "${run_dir}/logs"
  echo "[$(date '+%F %T')] eval ${name}/${protocol}" | tee -a "${log}"
  python src/evaluate/evaluate_inr_saved_midis.py \
    --prediction-manifest "${manifest}" \
    --output-json "${out_json}" \
    --num-workers "${EVAL_WORKERS}" \
    2>&1 | tee -a "${log}"
}

rollout_one() {
  local name="$1"
  local gpu="$2"
  local run_dir="${EXP_ROOT}/${name}"
  local cfg="${run_dir}/ckpt/model/train_config.json"
  local ckpt="${run_dir}/ckpt/model/checkpoint-best"
  local out_dir="${run_dir}/rollout"
  local log="${run_dir}/logs/rollout.log"

  mkdir -p "${out_dir}" "${run_dir}/logs"
  echo "[$(date '+%F %T')] rollout ${name}: gpu=${gpu}, k=${ROLLOUT_KS}" | tee -a "${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" python src/evaluate/eval_inr_rollout_current.py \
    --config "${cfg}" \
    --checkpoint "${ckpt}" \
    --output-dir "${out_dir}" \
    --split test \
    --performance-dataset ASAP \
    --batch-size-windows "${ROLLOUT_BATCH_WINDOWS}" \
    --num-workers "${ROLLOUT_WORKERS}" \
    --materialize-strategy sample \
    --feedback-strategy sample \
    --rollout-ks "${ROLLOUT_KS}" \
    --fast-kpass \
    --plot-distributions \
    --save-distribution-values \
    2>&1 | tee -a "${log}"
}

summarize_one() {
  local name="$1"
  local run_dir="${EXP_ROOT}/${name}"
  local log="${run_dir}/logs/summary.log"

  mkdir -p "${run_dir}/summary" "${run_dir}/logs"
  echo "[$(date '+%F %T')] summarize ${name}" | tee -a "${log}"
  python src/evaluate/summarize_inr_rollout_pipeline.py \
    --kpass-summary "${run_dir}/rollout/summary.json" \
    --ar-manifest "${run_dir}/infer_sampling/prediction_manifest.json" \
    --output-dir "${run_dir}/summary" \
    --num-workers "${EVAL_WORKERS}" \
    2>&1 | tee -a "${log}"
}

write_eval_summary() {
  python - "${EXP_ROOT}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for name in ["clean", "stable", "tf_pred", "stable_tf_pred"]:
    run_dir = root / name
    for protocol in ["deterministic", "sampling"]:
        path = run_dir / "eval" / f"{protocol}_wass.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        pp = payload["aggregate"]["pp_wass"]
        pn = payload["aggregate"]["pn_wass"]
        rows.append({
            "run": name,
            "protocol": protocol,
            "pp_ioi": pp.get("ioi_wass"),
            "pp_duration": pp.get("duration_wass"),
            "pp_velocity": pp.get("velocity_wass"),
            "pp_pedal": pp.get("pedal_wass"),
            "pn_ioi": pn.get("ioi_wass"),
            "pn_duration": pn.get("duration_wass"),
            "pn_velocity": pn.get("velocity_wass"),
            "pn_pedal": pn.get("pedal_wass"),
            "path": str(path),
        })
    curve = run_dir / "rollout" / "curve.csv"
    if curve.exists():
        with curve.open(newline="", encoding="utf-8") as f:
            for item in csv.DictReader(f):
                if item.get("k") in {"0", "1", "4", "16", "full"}:
                    rows.append({
                        "run": name,
                        "protocol": f"k{item['k']}",
                        "pp_ioi": float(item["pp_ioi_wass"]),
                        "pp_duration": float(item["pp_duration_wass"]),
                        "pp_velocity": float(item["pp_velocity_wass"]),
                        "pp_pedal": float(item["pp_pedal_wass"]),
                        "pn_ioi": float(item["pn_ioi_wass"]),
                        "pn_duration": float(item["pn_duration_wass"]),
                        "pn_velocity": float(item["pn_velocity_wass"]),
                        "pn_pedal": float(item["pn_pedal_wass"]),
                        "path": str(curve),
                    })
out_json = root / "summary.json"
out_csv = root / "summary.csv"
out_json.write_text(json.dumps({"experiment_root": str(root), "rows": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
with out_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "run", "protocol", "pp_ioi", "pp_duration", "pp_velocity", "pp_pedal",
            "pn_ioi", "pn_duration", "pn_velocity", "pn_pedal", "path",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)
print(out_csv)
PY
}

run_pipeline() {
  local name="$1"
  local gpus="$2"
  local port="$3"
  local det_gpu
  local sample_gpu
  det_gpu="$(first_gpu "${gpus}")"
  sample_gpu="$(second_gpu "${gpus}")"

  if [[ "${RUN_TRAIN}" == "1" ]]; then
    train_one "${name}" "${gpus}" "${port}"
  fi

  if [[ "${RUN_INFER}" == "1" ]]; then
    infer_one "${name}" "${det_gpu}" deterministic &
    det_pid=$!
    infer_one "${name}" "${sample_gpu}" sampling &
    sample_pid=$!
    wait "${det_pid}"
    wait "${sample_pid}"
  fi

  if [[ "${RUN_EVAL}" == "1" ]]; then
    eval_one "${name}" deterministic &
    eval_det_pid=$!
    eval_one "${name}" sampling &
    eval_sample_pid=$!
    wait "${eval_det_pid}"
    wait "${eval_sample_pid}"
  fi

  if [[ "${RUN_ROLLOUT}" == "1" ]]; then
    rollout_one "${name}" "${sample_gpu}"
  fi

  if [[ "${RUN_SUMMARY}" == "1" ]]; then
    summarize_one "${name}"
  fi

  echo "[$(date '+%F %T')] pipeline done: ${EXP_ROOT}/${name}"
}

run_selected() {
  local target="$1"
  local name
  IFS=',' read -ra names <<< "${RUNS}"
  for name in "${names[@]}"; do
    if [[ "${name}" == "${target}" ]]; then
      return 0
    fi
  done
  return 1
}

if [[ "${GENERATE_CONFIGS}" == "1" ]]; then
  generate_configs 2>&1 | tee "${EXP_ROOT}/logs/generate_configs.log"
else
  echo "[$(date '+%F %T')] skip config generation: GENERATE_CONFIGS=0" | tee -a "${EXP_ROOT}/logs/generate_configs.log"
fi

if [[ "${TRAIN_PARALLEL}" == "1" ]]; then
  pids=()
  if run_selected clean; then
    run_pipeline clean "${LANE_A_GPUS}" "${LANE_A_PORT}" &
    pids+=("$!")
  fi
  if run_selected tf_pred; then
    run_pipeline tf_pred "${LANE_B_GPUS}" "${LANE_B_PORT}" &
    pids+=("$!")
  fi
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done

  pids=()
  if run_selected stable; then
    run_pipeline stable "${LANE_A_GPUS}" "${LANE_A_PORT}" &
    pids+=("$!")
  fi
  if run_selected stable_tf_pred; then
    run_pipeline stable_tf_pred "${LANE_B_GPUS}" "${LANE_B_PORT}" &
    pids+=("$!")
  fi
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
else
  if run_selected clean; then
    run_pipeline clean "${LANE_A_GPUS}" "${LANE_A_PORT}"
  fi
  if run_selected tf_pred; then
    run_pipeline tf_pred "${LANE_B_GPUS}" "${LANE_B_PORT}"
  fi
  if run_selected stable; then
    run_pipeline stable "${LANE_A_GPUS}" "${LANE_A_PORT}"
  fi
  if run_selected stable_tf_pred; then
    run_pipeline stable_tf_pred "${LANE_B_GPUS}" "${LANE_B_PORT}"
  fi
fi

write_eval_summary 2>&1 | tee "${EXP_ROOT}/logs/write_summary.log"
echo "[$(date '+%F %T')] all done: ${EXP_ROOT}"
