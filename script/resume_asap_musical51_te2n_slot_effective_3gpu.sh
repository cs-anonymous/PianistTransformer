#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUN_ROOT="${RUN_ROOT:-results/inr_epr_pipeline/asap_processed_musical51_te2n_slot_effective_20260725_3queue_v2}"
MANIFEST="${RUN_ROOT}/configs/manifest.json"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Missing manifest: ${MANIFEST}" >&2
  exit 1
fi

cat > "${RUN_ROOT}/resume_queue.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

gpu="$1"
queue="$2"
root="$3"
repo="$4"
status="${root}/resume_status.tsv"
cd "${repo}"

python - "${root}/configs/manifest.json" "${queue}" <<'PY' > "${root}/${queue}.resume.list"
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for name in manifest["queues"][sys.argv[2]]:
    print(name)
PY

best_checkpoint() {
  local run_dir="$1"
  find "${run_dir}/training" -mindepth 2 -maxdepth 2 -type d -name checkpoint-best 2>/dev/null \
    | sort | head -n 1
}

latest_numeric_checkpoint() {
  local run_dir="$1"
  find "${run_dir}/training" -mindepth 2 -maxdepth 2 -type d -name 'checkpoint-[0-9]*' 2>/dev/null \
    | awk -F'checkpoint-' '/checkpoint-[0-9]+$/ {print $2 " " $0}' \
    | sort -n | tail -n 1 | cut -d' ' -f2-
}

while IFS= read -r name; do
  [[ -n "${name}" ]] || continue
  run_dir="${root}/${queue}/${name}"
  config="${root}/configs/${queue}__${name}.json"
  mkdir -p "${run_dir}"

  if [[ -s "${run_dir}/summary.json" ]]; then
    printf '%s\tGPU%s\t%s\t%s\tSKIP_SUMMARY_EXISTS\n' "$(date '+%F %T')" "${gpu}" "${queue}" "${name}" >> "${status}"
    continue
  fi

  checkpoint="$(best_checkpoint "${run_dir}" || true)"
  if [[ -n "${checkpoint}" ]]; then
    printf '%s\tGPU%s\t%s\t%s\tRESUME_INFER\t%s\n' "$(date '+%F %T')" "${gpu}" "${queue}" "${name}" "${checkpoint}" >> "${status}"
    set +e
    CUDA_VISIBLE_DEVICES="${gpu}" CONFIG="${config}" RUN_DIR_OVERRIDE="${run_dir}" \
      PIPELINE_STAGE_START=train SKIP_BASE_TRAIN=1 BASE_CHECKPOINT_OVERRIDE="${checkpoint}" \
      BASE_ASAP_ONLY=1 BASE_NUM_TRAIN_EPOCHS="${BASE_NUM_TRAIN_EPOCHS:-16}" ADAPT_NUM_TRAIN_EPOCHS=0 \
      BATCH_SIZE_PER_DEVICE="${BATCH_SIZE_PER_DEVICE:-16}" GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}" \
      SAMPLING_NUM_SAMPLES="${SAMPLING_NUM_SAMPLES:-1}" INFER_NUM_WORKERS="${INFER_NUM_WORKERS:-8}" \
      METRIC_NUM_WORKERS="${METRIC_NUM_WORKERS:-8}" INFER_BATCH_SIZE_WINDOWS="${INFER_BATCH_SIZE_WINDOWS:-8}" \
      INFER_SCORE_SOURCE_LIST="${INFER_SCORE_SOURCE_LIST:-data/asap_test_score_sources.txt}" \
      EVAL_CHECKPOINT_MODE=best RESUME_FROM_LATEST_CHECKPOINT=0 SKIP_EXISTING_PIPELINE_OUTPUTS=1 \
      MERGE_MODE=average CONTINUATION_DROP_RATIO=0.0 \
      bash script/run_inr_epr_pipeline.sh > "${run_dir}/resume_launcher.log" 2>&1
    code=$?
    set -e
  else
    checkpoint="$(latest_numeric_checkpoint "${run_dir}" || true)"
    printf '%s\tGPU%s\t%s\t%s\tRESUME_TRAIN\t%s\n' "$(date '+%F %T')" "${gpu}" "${queue}" "${name}" "${checkpoint:-from_scratch}" >> "${status}"
    set +e
    CUDA_VISIBLE_DEVICES="${gpu}" CONFIG="${config}" RUN_DIR_OVERRIDE="${run_dir}" \
      BASE_ASAP_ONLY=1 BASE_NUM_TRAIN_EPOCHS="${BASE_NUM_TRAIN_EPOCHS:-16}" ADAPT_NUM_TRAIN_EPOCHS=0 \
      BATCH_SIZE_PER_DEVICE="${BATCH_SIZE_PER_DEVICE:-16}" GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}" \
      SAMPLING_NUM_SAMPLES="${SAMPLING_NUM_SAMPLES:-1}" INFER_NUM_WORKERS="${INFER_NUM_WORKERS:-8}" \
      METRIC_NUM_WORKERS="${METRIC_NUM_WORKERS:-8}" INFER_BATCH_SIZE_WINDOWS="${INFER_BATCH_SIZE_WINDOWS:-8}" \
      INFER_SCORE_SOURCE_LIST="${INFER_SCORE_SOURCE_LIST:-data/asap_test_score_sources.txt}" \
      EVAL_CHECKPOINT_MODE=best RESUME_FROM_LATEST_CHECKPOINT=1 SKIP_EXISTING_PIPELINE_OUTPUTS=1 \
      MERGE_MODE=average CONTINUATION_DROP_RATIO=0.0 \
      bash script/run_inr_epr_pipeline.sh > "${run_dir}/resume_launcher.log" 2>&1
    code=$?
    set -e
  fi

  if [[ "${code}" -eq 0 ]]; then
    printf '%s\tGPU%s\t%s\t%s\tDONE\n' "$(date '+%F %T')" "${gpu}" "${queue}" "${name}" >> "${status}"
  else
    printf '%s\tGPU%s\t%s\t%s\tFAILED:%s\n' "$(date '+%F %T')" "${gpu}" "${queue}" "${name}" "${code}" >> "${status}"
  fi
done < "${root}/${queue}.resume.list"

printf '%s\tGPU%s\t%s\t-\tQUEUE_DONE\n' "$(date '+%F %T')" "${gpu}" "${queue}" >> "${status}"
SH
chmod +x "${RUN_ROOT}/resume_queue.sh"

: > "${RUN_ROOT}/resume_status.tsv"

tmux new-session -d -s "te2n_slot_resume_cinr_${STAMP}" \
  "bash '${RUN_ROOT}/resume_queue.sh' 0 cinr '${RUN_ROOT}' '${ROOT_DIR}' > '${RUN_ROOT}/cinr.resume.log' 2>&1"
tmux new-session -d -s "te2n_slot_resume_dinr_${STAMP}" \
  "bash '${RUN_ROOT}/resume_queue.sh' 1 dinr '${RUN_ROOT}' '${ROOT_DIR}' > '${RUN_ROOT}/dinr.resume.log' 2>&1"
tmux new-session -d -s "te2n_slot_resume_extra_${STAMP}" \
  "bash '${RUN_ROOT}/resume_queue.sh' 2 extra '${RUN_ROOT}' '${ROOT_DIR}' > '${RUN_ROOT}/extra.resume.log' 2>&1"

{
  echo "te2n_slot_resume_cinr_${STAMP}"
  echo "te2n_slot_resume_dinr_${STAMP}"
  echo "te2n_slot_resume_extra_${STAMP}"
} > "${RUN_ROOT}/resume_tmux_sessions.txt"

echo "RUN_ROOT=${RUN_ROOT}"
echo "TMUX_SESSIONS=$(tr '\n' ' ' < "${RUN_ROOT}/resume_tmux_sessions.txt")"
