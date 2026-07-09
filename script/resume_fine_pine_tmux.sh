#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"

FINE_SESSION="fine_resume_ddp"
PINE_SESSION="pine_resume_gpu2"

FINE_LOG="${LOG_DIR}/fine_pianocore_full_resume_2000.log"
PINE_LOG="${LOG_DIR}/pine_pianocore_full_resume_2000.log"

FINE_CONFIG="${ROOT_DIR}/configs/fine_pianocore_full_inr_resume_2000.json"
PINE_CONFIG="${ROOT_DIR}/configs/pine_pianocore_full_inr_resume_2000.json"

mkdir -p "${LOG_DIR}"

if tmux has-session -t "${FINE_SESSION}" 2>/dev/null; then
  echo "tmux session ${FINE_SESSION} already exists" >&2
  exit 1
fi

if tmux has-session -t "${PINE_SESSION}" 2>/dev/null; then
  echo "tmux session ${PINE_SESSION} already exists" >&2
  exit 1
fi

tmux new-session -d -s "${FINE_SESSION}" "cd '${ROOT_DIR}' && export PYTHONPATH='${ROOT_DIR}' && CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 src/train/train_inr.py --config '${FINE_CONFIG}' 2>&1 | tee -a '${FINE_LOG}'"
tmux new-session -d -s "${PINE_SESSION}" "cd '${ROOT_DIR}' && export PYTHONPATH='${ROOT_DIR}' && CUDA_VISIBLE_DEVICES=2 python src/train/train_inr.py --config '${PINE_CONFIG}' 2>&1 | tee -a '${PINE_LOG}'"

echo "started ${FINE_SESSION} -> ${FINE_LOG}"
echo "started ${PINE_SESSION} -> ${PINE_LOG}"
