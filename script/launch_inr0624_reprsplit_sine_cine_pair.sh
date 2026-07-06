#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUN_SCRIPT="script/run_inr_asap_only_train_infer_rollout.sh"
BASE_CONFIG="results/asap_full_compare/prefix_enc_add/config.json"
CHEAP15_LIST="results/psr_oracle/window_style_prefix_enc_add/cheap15_score_sources.txt"
EXP_ROOT="results/inr0624_reprsplit_pair"
CONFIG_DIR="${EXP_ROOT}/configs"
LOG_DIR="${EXP_ROOT}/launcher_logs"

TRAIN_EPOCHS="${TRAIN_EPOCHS:-16}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-5}"
EARLY_STOP_THRESHOLD="${EARLY_STOP_THRESHOLD:-0.001}"
TRAIN_DATALOADER_WORKERS="${TRAIN_DATALOADER_WORKERS:-8}"
EVAL_DATALOADER_WORKERS="${EVAL_DATALOADER_WORKERS:-8}"
INFER_NUM_WORKERS="${INFER_NUM_WORKERS:-18}"
METRIC_NUM_WORKERS="${METRIC_NUM_WORKERS:-10}"
ROLLOUT_DIAG_NUM_WORKERS="${ROLLOUT_DIAG_NUM_WORKERS:-18}"

mkdir -p "${CONFIG_DIR}" "${LOG_DIR}"

python - "${BASE_CONFIG}" "${CONFIG_DIR}" "${TRAIN_EPOCHS}" "${EARLY_STOP_PATIENCE}" "${EARLY_STOP_THRESHOLD}" "${TRAIN_DATALOADER_WORKERS}" "${EVAL_DATALOADER_WORKERS}" <<'PY'
import json
import pathlib
import sys

from src.train.train_inr import decoder_perf_target_input_dim, score_musical_input_dim

base_config_path, config_dir, base_epochs, early_stop_patience, early_stop_threshold, train_workers, eval_workers = sys.argv[1:8]
base_cfg = json.loads(pathlib.Path(base_config_path).read_text(encoding="utf-8"))
config_dir = pathlib.Path(config_dir)
config_dir.mkdir(parents=True, exist_ok=True)

score_dim = score_musical_input_dim(
    timing_control_mode=base_cfg.get("timing_control_mode"),
    use_timing_scale_bit=bool(base_cfg.get("use_timing_scale_bit", True)),
    musical_feature_mode=base_cfg.get("musical_feature_mode", "categorical"),
)
decoder_dim = decoder_perf_target_input_dim()

for mode in ("sine", "cine"):
    cfg = dict(base_cfg)
    cfg["note_embedding_mode"] = mode
    cfg["score_note_input_schema"] = "score_musical"
    cfg["decoder_note_input_schema"] = "perf_target"
    cfg["input_continuous_dim"] = int(score_dim)
    cfg["score_input_continuous_dim"] = int(score_dim)
    cfg["decoder_input_continuous_dim"] = int(decoder_dim)
    cfg["num_train_epochs"] = float(base_epochs)
    cfg["max_train_epochs"] = float(base_epochs)
    cfg["early_stopping_patience"] = int(early_stop_patience)
    cfg["early_stopping_threshold"] = float(early_stop_threshold)
    cfg["dataloader_num_workers"] = int(train_workers)
    cfg["eval_dataloader_num_workers"] = int(eval_workers)
    cfg["run_name"] = f"asap_reprsplit_{mode}"
    cfg["output_dir"] = f"{config_dir.parent}/{mode}/training"
    cfg["logging_dir"] = f"{config_dir.parent}/{mode}/tf-logs"
    out_path = config_dir / f"{mode}_scoremusical_perftarget.json"
    out_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
PY

launch_job() {
  local session_name="$1" gpus="$2" config_path="$3" run_dir="$4" log_path="$5"
  if tmux has-session -t "${session_name}" 2>/dev/null; then
    echo "tmux session already exists: ${session_name}" >&2
    exit 1
  fi
  tmux new-session -d -s "${session_name}" \
    "cd '${ROOT_DIR}' && \
     CUDA_VISIBLE_DEVICES='${gpus}' \
     CONFIG='${config_path}' \
     RUN_DIR_OVERRIDE='${run_dir}' \
     TRAIN_NUM_EPOCHS='${TRAIN_EPOCHS}' \
     EARLY_STOP_PATIENCE='${EARLY_STOP_PATIENCE}' \
     EARLY_STOP_THRESHOLD='${EARLY_STOP_THRESHOLD}' \
     INFER_NUM_WORKERS='${INFER_NUM_WORKERS}' \
     METRIC_NUM_WORKERS='${METRIC_NUM_WORKERS}' \
     ROLLOUT_DIAG_ENABLE='1' \
     ROLLOUT_DIAG_KS='0,1' \
     ROLLOUT_DIAG_SCORE_SOURCE_LIST='${CHEAP15_LIST}' \
     ROLLOUT_DIAG_NUM_WORKERS='${ROLLOUT_DIAG_NUM_WORKERS}' \
     ROLLOUT_DIAG_BATCH_SIZE_WINDOWS='8' \
     ROLLOUT_DIAG_MATERIALIZE_STRATEGY='sample' \
     bash '${RUN_SCRIPT}' 2>&1 | tee '${log_path}'"
}

SINE_CONFIG="${CONFIG_DIR}/sine_scoremusical_perftarget.json"
CINE_CONFIG="${CONFIG_DIR}/cine_scoremusical_perftarget.json"
SINE_RUN_DIR="${EXP_ROOT}/sine"
CINE_RUN_DIR="${EXP_ROOT}/cine"
SINE_LOG="${LOG_DIR}/sine.log"
CINE_LOG="${LOG_DIR}/cine.log"

launch_job "inr_reprsplit_sine" "0,1" "${SINE_CONFIG}" "${SINE_RUN_DIR}" "${SINE_LOG}"
launch_job "inr_reprsplit_cine" "2,3" "${CINE_CONFIG}" "${CINE_RUN_DIR}" "${CINE_LOG}"

echo "Launched:"
echo "  tmux session inr_reprsplit_sine -> GPUs 0,1 -> ${SINE_RUN_DIR}"
echo "  tmux session inr_reprsplit_cine -> GPUs 2,3 -> ${CINE_RUN_DIR}"
echo
echo "Attach:"
echo "  tmux attach -t inr_reprsplit_sine"
echo "  tmux attach -t inr_reprsplit_cine"
