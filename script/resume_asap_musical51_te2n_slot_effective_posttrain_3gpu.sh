#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUN_ROOT="${RUN_ROOT:-results/inr_epr_pipeline/asap_processed_musical51_te2n_slot_effective_20260725_3queue_v2}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-/home/kaititech/anaconda3/bin/python}"
CONFIG_DIR="${RUN_ROOT}/configs"

if [[ ! -f "${CONFIG_DIR}/manifest.json" ]]; then
  echo "Missing manifest: ${CONFIG_DIR}/manifest.json" >&2
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
  local best
  best="$(find "${run_dir}/training" -maxdepth 2 -mindepth 2 -type d -name checkpoint-best 2>/dev/null | head -n 1 || true)"
  if [[ -n "${best}" ]]; then
    echo "${best}"
    return 0
  fi
  find "${run_dir}/training" -path '*/checkpoint-*' -type d 2>/dev/null \
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
    stage="infer"
    skip_train=1
    printf '%s\tGPU%s\t%s\t%s\tRESUME_INFER\t%s\n' "$(date '+%F %T')" "${gpu}" "${queue}" "${name}" "${checkpoint}" >> "${status}"
  else
    stage="train"
    skip_train=0
    printf '%s\tGPU%s\t%s\t%s\tRESUME_TRAIN\n' "$(date '+%F %T')" "${gpu}" "${queue}" "${name}" >> "${status}"
  fi

  set +e
  env CUDA_VISIBLE_DEVICES="${gpu}" CONFIG="${config}" RUN_DIR_OVERRIDE="${run_dir}" \
    BASE_ASAP_ONLY=1 BASE_NUM_TRAIN_EPOCHS="${BASE_NUM_TRAIN_EPOCHS:-16}" ADAPT_NUM_TRAIN_EPOCHS=0 \
    BATCH_SIZE_PER_DEVICE="${BATCH_SIZE_PER_DEVICE:-16}" GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}" \
    SAMPLING_NUM_SAMPLES="${SAMPLING_NUM_SAMPLES:-1}" INFER_NUM_WORKERS="${INFER_NUM_WORKERS:-8}" \
    METRIC_NUM_WORKERS="${METRIC_NUM_WORKERS:-8}" INFER_BATCH_SIZE_WINDOWS="${INFER_BATCH_SIZE_WINDOWS:-8}" \
    INFER_SCORE_SOURCE_LIST="${INFER_SCORE_SOURCE_LIST:-data/asap_test_score_sources.txt}" \
    EVAL_CHECKPOINT_MODE=best RESUME_FROM_LATEST_CHECKPOINT=0 SKIP_EXISTING_PIPELINE_OUTPUTS=1 \
    PIPELINE_STAGE_START="${stage}" SKIP_BASE_TRAIN="${skip_train}" BASE_CHECKPOINT_OVERRIDE="${checkpoint:-}" \
    MERGE_MODE=average CONTINUATION_DROP_RATIO=0.0 \
    bash script/run_inr_epr_pipeline.sh > "${run_dir}/resume_launcher.log" 2>&1
  code=$?
  set -e

  if [[ "${code}" -eq 0 ]]; then
    printf '%s\tGPU%s\t%s\t%s\tDONE\n' "$(date '+%F %T')" "${gpu}" "${queue}" "${name}" >> "${status}"
  else
    printf '%s\tGPU%s\t%s\t%s\tFAILED:%s\n' "$(date '+%F %T')" "${gpu}" "${queue}" "${name}" "${code}" >> "${status}"
  fi
done < "${root}/${queue}.resume.list"

printf '%s\tGPU%s\t%s\t-\tQUEUE_DONE\n' "$(date '+%F %T')" "${gpu}" "${queue}" >> "${status}"
SH
chmod +x "${RUN_ROOT}/resume_queue.sh"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "RUN_ROOT=${RUN_ROOT}"
  python - "${RUN_ROOT}" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
manifest = json.loads((root / "configs" / "manifest.json").read_text(encoding="utf-8"))
for queue, names in manifest["queues"].items():
    print(f"[{queue}]")
    for name in names:
        run_dir = root / queue / name
        summary = run_dir / "summary.json"
        best = list((run_dir / "training").glob("*/checkpoint-best"))
        ckpts = list((run_dir / "training").glob("*/checkpoint-*"))
        if summary.exists() and summary.stat().st_size:
            state = "summary"
        elif best:
            state = "infer"
        elif ckpts:
            state = "infer_latest"
        else:
            state = "train"
        print(f"{name}\t{state}")
PY
  exit 0
fi

: > "${RUN_ROOT}/resume_status.tsv"

start_session() {
  local session="$1"
  local gpu="$2"
  local queue="$3"
  local cmd="bash '${RUN_ROOT}/resume_queue.sh' '${gpu}' '${queue}' '${RUN_ROOT}' '${ROOT_DIR}' > '${RUN_ROOT}/${queue}.resume.queue.log' 2>&1"
  if command -v tmux >/dev/null 2>&1 && tmux new-session -d -s "${session}" "${cmd}" 2>/dev/null; then
    echo "${session}"
  else
    nohup bash -lc "${cmd}" >/dev/null 2>&1 &
    echo "${session} (nohup pid $!)"
  fi
}

{
  start_session "te2n_resume_cinr_${STAMP}" 0 cinr
  start_session "te2n_resume_dinr_${STAMP}" 1 dinr
  start_session "te2n_resume_extra_${STAMP}" 2 extra
} > "${RUN_ROOT}/resume_sessions.txt"

echo "RUN_ROOT=${RUN_ROOT}"
echo "RESUME_SESSIONS=$(tr '\n' ' ' < "${RUN_ROOT}/resume_sessions.txt")"
echo "STATUS=${RUN_ROOT}/resume_status.tsv"
