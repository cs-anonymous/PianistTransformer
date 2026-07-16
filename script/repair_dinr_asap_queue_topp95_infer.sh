#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

: "${RUN_ROOT:?RUN_ROOT is required}"
variants=(backbone_8enc_4dec backbone_6enc_6dec hidden1024_10enc_2dec slot_dim256 slot_dim512)
: > "${RUN_ROOT}/topp95_repair_status.tsv"

for name in "${variants[@]}"; do
  run_dir="${RUN_ROOT}/${name}"
  config="${RUN_ROOT}/configs/${name}.json"
  checkpoint="$(python - "${run_dir}/summary.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["checkpoint"])
PY
)"
  printf '%s\tSTART\t%s\n' "$(date '+%F %T')" "${name}" | tee -a "${RUN_ROOT}/topp95_repair_status.tsv"
  env CUDA_VISIBLE_DEVICES=0 \
    CONFIG="${config}" RUN_DIR_OVERRIDE="${run_dir}" \
    PIPELINE_STAGE_START=infer BASE_CHECKPOINT_OVERRIDE="${checkpoint}" \
    BASE_ASAP_ONLY=1 BASE_NUM_TRAIN_EPOCHS=16 ADAPT_NUM_TRAIN_EPOCHS=0 \
    ADAPT_PREPARED_SIDECAR_TAG=DINR_READY_ASAP \
    BATCH_SIZE_PER_DEVICE=32 GLOBAL_BATCH_SIZE=64 \
    DET_NUM_SAMPLES=1 SAMPLING_NUM_SAMPLES=1 \
    INFER_NUM_WORKERS=8 METRIC_NUM_WORKERS=8 INFER_BATCH_SIZE_WINDOWS=8 \
    INFER_SCORE_SOURCE_LIST=data/asap_test_score_sources.txt \
    SKIP_EXISTING_PIPELINE_OUTPUTS=0 EVAL_CHECKPOINT_MODE=best \
    RESUME_FROM_LATEST_CHECKPOINT=0 MERGE_MODE=continuation CONTINUATION_DROP_RATIO=0.0 \
    bash script/run_inr_epr_pipeline.sh >> "${run_dir}/topp95_repair.log" 2>&1
  printf '%s\tDONE\t%s\n' "$(date '+%F %T')" "${name}" | tee -a "${RUN_ROOT}/topp95_repair_status.tsv"
done
