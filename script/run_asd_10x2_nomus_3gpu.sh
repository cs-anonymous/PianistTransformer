#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GPU_IDS_CSV="${GPU_IDS:-0,1,2}"
IFS=',' read -ra GPU_IDS_ARR <<< "${GPU_IDS_CSV}"
if [[ "${#GPU_IDS_ARR[@]}" -ne 3 ]]; then
  echo "GPU_IDS must contain exactly 3 ids, got: ${GPU_IDS_CSV}" >&2
  exit 2
fi

RUN_SMOKE="${RUN_SMOKE:-1}"
MAX_STEPS_ARG=()
if [[ -n "${MAX_STEPS:-}" ]]; then
  MAX_STEPS_ARG=(--max_steps "${MAX_STEPS}")
fi

RUN_ROOT="results/asd_10x2_nomus_3gpu_$(date '+%Y%m%d_%H%M%S')"
mkdir -p "${RUN_ROOT}/logs"
RUNTIME_CONFIG_DIR="${RUN_ROOT}/runtime_configs"
mkdir -p "${RUNTIME_CONFIG_DIR}"

PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-3}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-48}"

prepare_runtime_config() {
  local src="$1"
  local dst="$2"
  local label="$3"
  python - "${src}" "${dst}" "${label}" "${RUN_ROOT}" "${PER_DEVICE_BATCH_SIZE}" "${GRADIENT_ACCUMULATION_STEPS}" "${GLOBAL_BATCH_SIZE}" <<'PY'
import json
import sys
from pathlib import Path

src, dst, label, run_root, per_device_bs, grad_accum, global_bs = sys.argv[1:8]
cfg = json.loads(Path(src).read_text(encoding="utf-8"))
for key in (
    "fixed_window_split_scheme",
    "fixed_window_base_split",
    "fixed_window_train_split_name",
    "fixed_window_eval_split_name",
    "fixed_window_split_summary_path",
):
    cfg.pop(key, None)
cfg["run_name"] = f"{cfg.get('run_name', 'asd')}_{label}"
cfg["output_dir"] = str(Path(run_root) / label / "training")
cfg["logging_dir"] = str(Path(run_root) / label / "tf-logs")
cfg["prepared_sidecar_tag"] = "ASAP_MUSICAL51"
cfg["eval_from_train_fraction"] = 0.03
cfg["per_device_train_batch_size"] = int(per_device_bs)
cfg["per_device_eval_batch_size"] = int(per_device_bs)
cfg["gradient_accumulation_steps"] = int(grad_accum)
cfg["global_batch_size"] = int(global_bs)
Path(dst).parent.mkdir(parents=True, exist_ok=True)
Path(dst).write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

run_train() {
  local gpu="$1"
  local config="$2"
  local label="$3"
  local log_path="${RUN_ROOT}/logs/${label}.log"
  echo "[$(date '+%F %T')] GPU ${gpu}: start ${label}" | tee -a "${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python src/train/train_inr.py --config "${config}" "${MAX_STEPS_ARG[@]}" 2>&1 | tee -a "${log_path}"
  echo "[$(date '+%F %T')] GPU ${gpu}: done ${label}" | tee -a "${log_path}"
}

lane_config_paths() {
  local lane="$1"
  case "${lane}" in
    baseline)
      echo "configs/inr_epr/cinr__baseline_10x2_nomus.json configs/inr_epr/dinr__baseline_10x2_nomus.json"
      ;;
    full)
      echo "configs/inr_epr/cinr__full4_10x2_nomus.json configs/inr_epr/dinr__full4_10x2_nomus.json"
      ;;
    share)
      echo "configs/inr_epr/cinr__share_10x2_nomus.json configs/inr_epr/dinr__share_10x2_nomus.json"
      ;;
    *)
      echo "Unknown lane ${lane}" >&2
      exit 2
      ;;
  esac
}

run_lane() {
  local lane="$1"
  local gpu="$2"
  local configs
  read -r cinr_config dinr_config <<< "$(lane_config_paths "${lane}")"
  local runtime_cinr_config="${RUNTIME_CONFIG_DIR}/${lane}_cinr.json"
  local runtime_dinr_config="${RUNTIME_CONFIG_DIR}/${lane}_dinr.json"
  prepare_runtime_config "${cinr_config}" "${runtime_cinr_config}" "${lane}_cinr"
  prepare_runtime_config "${dinr_config}" "${runtime_dinr_config}" "${lane}_dinr"
  run_train "${gpu}" "${runtime_cinr_config}" "${lane}_cinr"
  run_train "${gpu}" "${runtime_dinr_config}" "${lane}_dinr"
}

if [[ "${RUN_SMOKE}" == "1" ]]; then
  bash script/smoke_asd_10x2_nomus.sh "${GPU_IDS_ARR[0]}" cinr
  bash script/smoke_asd_10x2_nomus.sh "${GPU_IDS_ARR[0]}" dinr
fi

echo "RUN_ROOT ${RUN_ROOT}"
LANES=(baseline full share)
pids=()
for idx in 0 1 2; do
  run_lane "${LANES[$idx]}" "${GPU_IDS_ARR[$idx]}" &
  pids+=("$!")
done

failures=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failures=$((failures + 1))
  fi
done

if [[ "${failures}" -ne 0 ]]; then
  echo "${failures} lane(s) failed; see ${RUN_ROOT}/logs" >&2
  exit 1
fi

echo "All baseline/full/share lanes finished. Logs: ${RUN_ROOT}/logs"
