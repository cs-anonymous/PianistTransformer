#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BASE_CONFIG="${BASE_CONFIG:-results/inr_epr_pipeline/asap_processed_musical51_nolossnorm16_slot_effective_20260723/configs/cinr__default_dlm_k1_bounded5.json}"
RUN_ROOT="${RUN_ROOT:-results/inr_epr_pipeline/asap_processed_musical51_allw1_pnpp_lam05_20260725/cinr/default_dlm_k1_bounded5_allw1_pnpp_lam05_24ep}"
PYTHON_BIN="${PYTHON_BIN:-/home/kaititech/anaconda3/bin/python}"
LAMBDA="${LAMBDA:-0.5}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-96}"
GROUP_SIZE="${GROUP_SIZE:-3}"
MASTER_PORT="${MASTER_PORT:-29625}"
SMOKE="${SMOKE:-0}"

CONFIG_PATH="${RUN_ROOT}/config.json"
TRAIN_LOG="${RUN_ROOT}/train.log"
POST_LOG="${RUN_ROOT}/post_eval.log"
TRAIN_DIR="${RUN_ROOT}/training/asap_musical51_cinr_allw1_pnpp_lam05_24ep"

mkdir -p "${RUN_ROOT}" "${RUN_ROOT}/tf-logs" "${RUN_ROOT}/training"

"${PYTHON_BIN}" - "${BASE_CONFIG}" "${CONFIG_PATH}" "${RUN_ROOT}" "${LAMBDA}" \
  "${PER_DEVICE_BATCH}" "${GRAD_ACCUM}" "${GLOBAL_BATCH_SIZE}" "${GROUP_SIZE}" "${SMOKE}" <<'PY'
import json
import sys
from pathlib import Path

base_path, config_path, run_root, lam, per_device, grad_accum, global_bs, group_size, smoke = sys.argv[1:]
cfg = json.loads(Path(base_path).read_text(encoding="utf-8"))
lam = float(lam)
cfg.update({
    "run_name": "asap_musical51_cinr_allw1_pnpp_lam05_24ep",
    "output_dir": str(Path(run_root) / "training"),
    "logging_dir": str(Path(run_root) / "tf-logs"),
    "overwrite_output_dir": True,
    "resume_path": None,
    "resume_trainer_state": False,
    "num_train_epochs": 24.0,
    "max_train_epochs": 24.0,
    "per_device_train_batch_size": int(per_device),
    "per_device_eval_batch_size": int(per_device),
    "gradient_accumulation_steps": int(grad_accum),
    "global_batch_size": int(global_bs),
    "multi_perf_group_size": int(group_size),
    "multi_perf_min_group_size": min(3, int(group_size)),
    "save_total_limit": 32,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    "pn_mean_loss_lambda": 0.0,
    "pn_var_ioi_zero_lambda": 0.0,
    "pn_var_ioi_nonzero_lambda": 0.0,
    "pn_var_duration_lambda": 0.0,
    "pn_var_velocity_lambda": 0.0,
    "pn_ioi_w1_lambda": lam,
    "pp_ioi_w1_lambda": lam,
    "pn_duration_w1_lambda": lam,
    "pp_duration_w1_lambda": lam,
    "pn_velocity_w1_lambda": lam,
    "pp_velocity_w1_lambda": lam,
    "pn_ioi_w1_normalizer": 22.772456164440005,
    "pp_ioi_w1_normalizer": 5.030337670706154,
    "pn_duration_w1_normalizer": 75.08565114668764,
    "pp_duration_w1_normalizer": 20.449834140491266,
    "pn_velocity_w1_normalizer": 10.580359191566208,
    "pp_velocity_w1_normalizer": 3.070947106636113,
    "ioi_w1_bins": 256,
    "duration_w1_bins": 256,
    "ioi_w1_min": 0.0,
    "ioi_w1_max": 8000.0,
    "duration_w1_min": 0.0,
    "duration_w1_max": 8000.0,
})
if smoke == "1":
    cfg.update({
        "max_steps": 1,
        "num_train_epochs": 1.0,
        "max_train_epochs": 1.0,
        "eval_strategy": "no",
        "save_strategy": "no",
        "logging_steps": 1,
        "save_total_limit": 1,
        "dataloader_num_workers": 0,
        "eval_dataloader_num_workers": 0,
        "dataloader_persistent_workers": False,
        "precompute_dataset_items": False,
        "bf16": False,
        "fp16": False,
    })
Path(config_path).write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY

if [[ "${SMOKE}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES%%,*}" PYTHONUNBUFFERED=1 "${PYTHON_BIN}" \
    src/train/train_inr.py --config "${CONFIG_PATH}" 2>&1 | tee "${TRAIN_LOG}"
  exit 0
fi

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo \
  torchrun --nnodes=1 --nproc_per_node=3 --master_addr=127.0.0.1 --master_port="${MASTER_PORT}" \
    src/train/train_inr.py --config "${CONFIG_PATH}" 2>&1 | tee "${TRAIN_LOG}"

"${PYTHON_BIN}" - "${TRAIN_DIR}" "${RUN_ROOT}/selected_checkpoints.tsv" <<'PY'
import json
import sys
from pathlib import Path

train_dir = Path(sys.argv[1])
out_path = Path(sys.argv[2])
rows = []
for ckpt in sorted(train_dir.glob("checkpoint-*")):
    if not ckpt.name.split("checkpoint-", 1)[-1].isdigit():
        continue
    state_path = ckpt / "trainer_state.json"
    if not state_path.exists():
        continue
    state = json.loads(state_path.read_text(encoding="utf-8"))
    epoch = state.get("epoch")
    if epoch is None:
        for item in reversed(state.get("log_history", [])):
            if "epoch" in item:
                epoch = item["epoch"]
                break
    if epoch is not None:
        rows.append((float(epoch), int(ckpt.name.split("-")[-1]), ckpt))

if not rows:
    raise SystemExit(f"No numeric checkpoints with trainer_state.json found under {train_dir}")

lines = []
for target in (16.0, 20.0, 24.0):
    epoch, step, ckpt = min(rows, key=lambda item: (abs(item[0] - target), abs(item[1])))
    lines.append(f"ep{int(target)}\t{epoch:.6f}\t{step}\t{ckpt}\n")
out_path.write_text("".join(lines), encoding="utf-8")
print(out_path)
PY

run_one_eval() {
  local label="$1"
  local ckpt="$2"
  local gpu="$3"
  local eval_dir="${RUN_ROOT}/eval_${label}"
  mkdir -p "${eval_dir}"
  env CUDA_VISIBLE_DEVICES="${gpu}" \
    CONFIG="${CONFIG_PATH}" RUN_DIR_OVERRIDE="${eval_dir}" \
    PIPELINE_STAGE_START=infer BASE_CHECKPOINT_OVERRIDE="${ckpt}" \
    BASE_ASAP_ONLY=1 BASE_NUM_TRAIN_EPOCHS=24 ADAPT_NUM_TRAIN_EPOCHS=0 \
    BATCH_SIZE_PER_DEVICE=16 GLOBAL_BATCH_SIZE=16 \
    SAMPLING_NUM_SAMPLES=1 INFER_NUM_WORKERS=8 METRIC_NUM_WORKERS=8 INFER_BATCH_SIZE_WINDOWS=8 \
    INFER_SCORE_SOURCE_LIST=data/asap_test_score_sources.txt \
    SKIP_EXISTING_PIPELINE_OUTPUTS=0 EVAL_CHECKPOINT_MODE=latest \
    RESUME_FROM_LATEST_CHECKPOINT=0 MERGE_MODE=continuation CONTINUATION_DROP_RATIO=0.0 \
    bash script/run_inr_epr_pipeline.sh > "${eval_dir}/pipeline.log" 2>&1
}

mapfile -t selected < "${RUN_ROOT}/selected_checkpoints.tsv"
: > "${POST_LOG}"
for idx in "${!selected[@]}"; do
  line="${selected[$idx]}"
  label="$(printf '%s' "${line}" | cut -f1)"
  ckpt="$(printf '%s' "${line}" | cut -f4)"
  gpu="${idx}"
  echo "[$(date '+%F %T')] ${label} gpu=${gpu} ckpt=${ckpt}" | tee -a "${POST_LOG}"
  run_one_eval "${label}" "${ckpt}" "${gpu}" &
done
wait
echo "[$(date '+%F %T')] all checkpoint evals finished" | tee -a "${POST_LOG}"
