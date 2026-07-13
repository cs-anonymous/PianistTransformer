#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/floorlog_tight_k1_binary4_bce_full_pianocore_2gpu/${STAMP}}"
CONFIG="${CONFIG:-results/floorlog_tight_k1_binary4_bce_2gpu/20260714_tight_k1_binary4_bce_v1/configs/dlm-k1-noinfl-ioi-binary4-bce-pedalw1.json}"
NAME="${NAME:-dlm-k1-noinfl-ioi-binary4-bce-pianocore4-asap8}"
BASE_EPOCHS="${BASE_EPOCHS:-4}"
ADAPT_EPOCHS="${ADAPT_EPOCHS:-8}"
GPUS="${GPUS:-0,1}"

run_dir="${RUN_ROOT}/${NAME}"
log_path="${run_dir}/launcher.log"
mkdir -p "${run_dir}"

cat >"${RUN_ROOT}/manifest.json" <<EOF
{
  "config": "${CONFIG}",
  "variant": "${NAME}",
  "pipeline": [
    "PianoCoRe train",
    "ASAP adapt",
    "ASAP deterministic infer",
    "ASAP sampling infer",
    "summary/eval/statistics"
  ],
  "epochs": {
    "pianocore_base": ${BASE_EPOCHS},
    "asap_adapt": ${ADAPT_EPOCHS}
  },
  "gpus": "${GPUS}"
}
EOF

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1 RUN_ROOT=${RUN_ROOT}"
  exit 0
fi

: > "${RUN_ROOT}/processes.tsv"
setsid env CUDA_VISIBLE_DEVICES="${GPUS}" \
  CONFIG="${CONFIG}" RUN_DIR_OVERRIDE="${run_dir}" \
  BASE_ASAP_ONLY=0 BASE_NUM_TRAIN_EPOCHS="${BASE_EPOCHS}" ADAPT_NUM_TRAIN_EPOCHS="${ADAPT_EPOCHS}" \
  BATCH_SIZE_PER_DEVICE=32 GLOBAL_BATCH_SIZE=64 \
  DET_NUM_SAMPLES=1 SAMPLING_NUM_SAMPLES=1 \
  INFER_NUM_WORKERS=8 METRIC_NUM_WORKERS=8 INFER_BATCH_SIZE_WINDOWS=8 \
  INFER_SCORE_SOURCE_LIST=data/asap_test_score_sources.txt \
  EVAL_CHECKPOINT_MODE=best RESUME_FROM_LATEST_CHECKPOINT=0 \
  MERGE_MODE=continuation CONTINUATION_DROP_RATIO=0.0 \
  bash script/run_inr_epr_pipeline.sh >"${log_path}" 2>&1 < /dev/null &
pid=$!
printf '%s\tGPUs%s\tPID%s\t%s\t%s\n' "${NAME}" "${GPUS}" "${pid}" "${run_dir}" "${log_path}" \
  | tee -a "${RUN_ROOT}/processes.tsv"
echo "RUN_ROOT=${RUN_ROOT}"
