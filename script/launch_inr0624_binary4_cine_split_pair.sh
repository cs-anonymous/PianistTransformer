#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUN_SCRIPT="script/run_inr_binary4_adapt_pipeline.sh"
BASE_DIR="results/inr0624_binary4_prior_ablation_4gpu"

CONFIG_NOSPLIT="${BASE_DIR}/configs/cine_kp05_nosplit.json"
CONFIG_SPLIT="${BASE_DIR}/configs/cine_kp05_split.json"

LOG_DIR="${BASE_DIR}/launcher_logs"
mkdir -p "${LOG_DIR}"

echo "Launching paired 2-GPU runs:"
echo "  GPUs 0,1 -> ${CONFIG_NOSPLIT}"
echo "  GPUs 2,3 -> ${CONFIG_SPLIT}"

tmux new-session -d -s inr0624_cine_kp05_nosplit \
  "cd '${ROOT_DIR}' && \
   CUDA_VISIBLE_DEVICES='0,1' \
   CONFIG='${CONFIG_NOSPLIT}' \
   RUN_DIR_OVERRIDE='${BASE_DIR}/cine_kp05_nosplit' \
   bash '${RUN_SCRIPT}' 2>&1 | tee '${LOG_DIR}/cine_kp05_nosplit.log'"

tmux new-session -d -s inr0624_cine_kp05_split \
  "cd '${ROOT_DIR}' && \
   CUDA_VISIBLE_DEVICES='2,3' \
   CONFIG='${CONFIG_SPLIT}' \
   RUN_DIR_OVERRIDE='${BASE_DIR}/cine_kp05_split' \
   bash '${RUN_SCRIPT}' 2>&1 | tee '${LOG_DIR}/cine_kp05_split.log'"

echo
echo "tmux sessions:"
echo "  inr0624_cine_kp05_nosplit"
echo "  inr0624_cine_kp05_split"
echo
echo "Attach with:"
echo "  tmux attach -t inr0624_cine_kp05_nosplit"
echo "  tmux attach -t inr0624_cine_kp05_split"
