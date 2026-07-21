#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="results/inr_epr_pipeline/launch_rawlog_sine_cine_musical4slot_${STAMP}"
CONFIG_ROOT="${RUN_ROOT}/configs"
mkdir -p "${RUN_ROOT}" "${CONFIG_ROOT}"

BASE_CONFIG="configs/inr0624_epr_sn_rawlog_sine_tfmask50.json"

write_config() {
  local output_path="$1"
  local note_mode="$2"
  local musical_mode="$3"
  local input_dim="$4"
  python - "$BASE_CONFIG" "$output_path" "$note_mode" "$musical_mode" "$input_dim" <<'PY'
import json
import sys
from pathlib import Path

base_path, output_path, note_mode, musical_mode, input_dim = sys.argv[1:6]
cfg = json.loads(Path(base_path).read_text(encoding="utf-8"))
cfg["note_embedding_mode"] = note_mode
cfg["musical_feature_mode"] = musical_mode
cfg["input_continuous_dim"] = int(input_dim)
cfg["metadata_path"] = str(Path("data/ASAP_processed/metadata.generated_json.csv").resolve())
cfg["refined_dir"] = str(Path("data/ASAP_processed").resolve())
cfg["decoder_note_input_schema"] = "integrated"
cfg["score_note_input_schema"] = "integrated"
cfg["disable_musical_features"] = False
cfg["auto_rollout_eval_after_train"] = False
cfg.pop("prepared_sidecar_tag", None)
Path(output_path).write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

launch_one() {
  local gpu="$1"
  local name="$2"
  local note_mode="$3"
  local musical_mode="$4"
  local input_dim="$5"
  local config_path="${CONFIG_ROOT}/${name}.json"
  local run_dir="${RUN_ROOT}/${name}"
  local log_path="${run_dir}/launcher.log"
  mkdir -p "${run_dir}"
  write_config "${config_path}" "${note_mode}" "${musical_mode}" "${input_dim}"
  echo "[$(date '+%F %T')] launch ${name} on GPU ${gpu} (${note_mode}, ${musical_mode})" | tee -a "${RUN_ROOT}/launch.log"
  setsid env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    CONFIG="${config_path}" \
    RUN_DIR_OVERRIDE="${run_dir}" \
    BATCH_SIZE_PER_DEVICE=32 \
    GLOBAL_BATCH_SIZE=64 \
    BASE_ASAP_ONLY=1 \
    BASE_NUM_TRAIN_EPOCHS=16 \
    ADAPT_NUM_TRAIN_EPOCHS=0 \
    DET_NUM_SAMPLES=1 \
    SAMPLING_NUM_SAMPLES=1 \
    INFER_BATCH_SIZE_WINDOWS=8 \
    MERGE_MODE=continuation \
    CONTINUATION_DROP_RATIO=0.0 \
    SKIP_EXISTING_PIPELINE_OUTPUTS=1 \
    bash script/run_inr_epr_pipeline.sh >"${log_path}" 2>&1 < /dev/null &
  local pid=$!
  echo "${name}: pid=${pid} run_dir=${run_dir} config=${config_path} log=${log_path}" | tee -a "${RUN_ROOT}/launch.log"
}

launch_one 0 "exp1_sine_musical4slot_ed_tfmask50" "sine" "musical4slot" 22
launch_one 1 "exp2_cine_musical4slot_ed_tfmask50" "cine" "musical4slot" 22

echo "launch_root=${RUN_ROOT}"
