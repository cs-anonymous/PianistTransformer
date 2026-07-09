#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="results/inr_epr_pipeline/launch_note_rawlog_cine_slot8_slot12_${STAMP}"
mkdir -p "${RUN_ROOT}"

launch_one() {
  local gpu="$1"
  local name="$2"
  local config="$3"
  local session="inr_${name}_${STAMP}"
  local run_dir="${RUN_ROOT}/${name}"
  local log_path="${run_dir}/launcher.log"
  mkdir -p "${run_dir}"

  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "tmux session ${session} already exists" >&2
    return 1
  fi

  echo "[$(date '+%F %T')] launch ${name} on GPU ${gpu}" | tee -a "${RUN_ROOT}/launch.log"
  tmux new-session -d -s "${session}" \
    "cd '${ROOT_DIR}' && export PYTHONPATH='${ROOT_DIR}' && CUDA_VISIBLE_DEVICES='${gpu}' CONFIG='${config}' RUN_DIR_OVERRIDE='${run_dir}' BATCH_SIZE_PER_DEVICE=32 GLOBAL_BATCH_SIZE=64 BASE_ASAP_ONLY=1 BASE_NUM_TRAIN_EPOCHS=16 ADAPT_NUM_TRAIN_EPOCHS=0 DET_NUM_SAMPLES=1 SAMPLING_NUM_SAMPLES=1 INFER_BATCH_SIZE_WINDOWS=8 MERGE_MODE=continuation CONTINUATION_DROP_RATIO=0.0 SKIP_EXISTING_PIPELINE_OUTPUTS=0 bash script/run_inr_epr_pipeline.sh 2>&1 | tee '${log_path}'"

  echo "${name}: session=${session} run_dir=${run_dir} config=${config} log=${log_path}" | tee -a "${RUN_ROOT}/launch.log"
}

launch_one 0 "cine" "configs/inr0624_note_rawlog_cine_nomus_clean_20260709.json"
launch_one 1 "slot8" "configs/inr0624_note_rawlog_slot8_nomus_clean_20260709.json"
launch_one 2 "slot12" "configs/inr0624_note_rawlog_slot12_m51_clean_20260709.json"

echo "launch_root=${RUN_ROOT}"
echo "tmux sessions:"
tmux ls | grep "inr_.*_${STAMP}" || true
