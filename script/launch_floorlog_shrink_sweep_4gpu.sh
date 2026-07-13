#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

STAMP="${STAMP:-20260712_floorlog_shrink_sweep}"
RUN_ROOT="${RUN_ROOT:-results/floorlog_shrink_sweep_4gpu/${STAMP}}"
CONFIG_DIR="${RUN_ROOT}/configs"
mkdir -p "${CONFIG_DIR}"

make_configs() {
  local model="$1"
  local base_config="$2"
  python - "${model}" "${base_config}" "${CONFIG_DIR}" <<'PY'
import json
import sys
from pathlib import Path

model, base_config, config_dir = sys.argv[1:4]
base = json.loads(Path(base_config).read_text(encoding="utf-8"))
strategies = {
    "linear-s0p25": {"timing_sample_shrink_mode": "linear", "timing_sample_shrink_factor": 0.25, "timing_sample_shrink_radius": 0.0},
    "linear-s0p50": {"timing_sample_shrink_mode": "linear", "timing_sample_shrink_factor": 0.50, "timing_sample_shrink_radius": 0.0},
    "tanh-r0p05": {"timing_sample_shrink_mode": "tanh", "timing_sample_shrink_factor": 1.0, "timing_sample_shrink_radius": 0.05},
    "tanh-r0p10": {"timing_sample_shrink_mode": "tanh", "timing_sample_shrink_factor": 1.0, "timing_sample_shrink_radius": 0.10},
}
for name, overrides in strategies.items():
    cfg = dict(base)
    cfg.update(overrides)
    cfg["timing_sample_truncate_radius"] = 0.0
    cfg["timing_sample_truncate_center"] = "mean"
    cfg["dlm_timing_sample_truncate_radius"] = 0.0
    cfg["dlm_timing_sample_truncate_center"] = "mean"
    cfg["eval_gt_time_normalization"] = "score_onset_span"
    cfg["run_name"] = f"{model}_{name}"
    cfg["output_dir"] = f"unused/{model}/{name}/training"
    cfg["logging_dir"] = f"unused/{model}/{name}/tf-logs"
    out = Path(config_dir) / model / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

run_model() {
  local gpu="$1"
  local model="$2"
  local base_config="$3"
  local checkpoint="$4"
  local det_manifest="$5"
  local train_output_dir="$6"
  local session="shrink_${model//[^A-Za-z0-9_]/_}_${STAMP}"
  local model_root="${RUN_ROOT}/${model}"
  local log_path="${model_root}/launcher.log"
  mkdir -p "${model_root}"
  make_configs "${model}" "${base_config}"

  tmux new-session -d -s "${session}" \
    "cd '${ROOT_DIR}' && \
     for strategy in linear-s0p25 linear-s0p50 tanh-r0p05 tanh-r0p10; do \
       out_dir='${model_root}/'\${strategy}; \
       mkdir -p \"\${out_dir}\"; \
       echo \"[\$(date '+%F %T')] ${model} \${strategy}\"; \
       if [ ! -s \"\${out_dir}/sampling/prediction_manifest.json\" ]; then \
         CUDA_VISIBLE_DEVICES='${gpu}' PYTHONUNBUFFERED=1 python src/inference/infer_inr_testset.py \
           --config '${CONFIG_DIR}/${model}/'\${strategy}'.json' \
           --checkpoint '${checkpoint}' \
           --split test \
           --performance-dataset ASAP \
           --num-workers '${INFER_NUM_WORKERS:-8}' \
           --batch-size-windows '${INFER_BATCH_SIZE_WINDOWS:-8}' \
           --merge-mode continuation \
           --continuation-drop-ratio 0.0 \
           --device cuda \
           --protocol sampling \
           --num-samples 1 \
           --score-source-list data/asap_test_score_sources.txt \
           --output-dir \"\${out_dir}/sampling\"; \
       fi; \
       if [ ! -s \"\${out_dir}/summary.json\" ]; then \
         PYTHONUNBUFFERED=1 python src/evaluate/summarize_inr_asap_pipeline.py \
           --deterministic-manifest '${det_manifest}' \
           --sampling-manifest \"\${out_dir}/sampling/prediction_manifest.json\" \
           --output-json \"\${out_dir}/summary.json\" \
           --output-plot \"\${out_dir}/asap_label_distribution.png\" \
           --config '${CONFIG_DIR}/${model}/'\${strategy}'.json' \
           --checkpoint '${checkpoint}' \
           --train-output-dir '${train_output_dir}' \
           --pipeline-log '${model_root}/train.reused.log' \
           --evaluate-log \"\${out_dir}/evaluate.log\" \
           --num-workers '${METRIC_NUM_WORKERS:-8}'; \
       fi; \
     done 2>&1 | tee '${log_path}'"

  printf '%s\tGPU%s\t%s\t%s\n' "${session}" "${gpu}" "${model_root}" "${log_path}" \
    | tee -a "${RUN_ROOT}/sessions.tsv"
}

run_model 0 \
  k8-b256-veldlm \
  results/floorlog_dlm_2gpu/20260712_asap_test/k8-b256-veldlm/config.json \
  results/floorlog_dlm_2gpu/20260712_asap_test/k8-b256-veldlm/training/floorlog_dlm_k8_b256_veldlm/checkpoint-1680 \
  results/floorlog_dlm_2gpu/20260712_asap_test/k8-b256-veldlm/deterministic/prediction_manifest.json \
  results/floorlog_dlm_2gpu/20260712_asap_test/k8-b256-veldlm/training/floorlog_dlm_k8_b256_veldlm

run_model 1 \
  slot6-musical-mlp \
  results/floorlog_slot6_musical_2gpu/20260712_slot6_musical/slot6-musical-mlp/config.json \
  results/floorlog_slot6_musical_2gpu/20260712_slot6_musical/slot6-musical-mlp/training/floorlog_slot6_musical_mlp_veldlm/checkpoint-1680 \
  results/floorlog_slot6_musical_2gpu/20260712_slot6_musical/slot6-musical-mlp/deterministic/prediction_manifest.json \
  results/floorlog_slot6_musical_2gpu/20260712_slot6_musical/slot6-musical-mlp/training/floorlog_slot6_musical_mlp_veldlm

run_model 2 \
  mln3 \
  results/floorlog_sn_mln3_trunc_4gpu/20260712_floorlog_sn_mln3_trunc/floorlog-mln3/config.json \
  results/floorlog_sn_mln3_trunc_4gpu/20260712_floorlog_sn_mln3_trunc/floorlog-mln3/training/floorlog_mln3/checkpoint-1680 \
  results/floorlog_sn_mln3_trunc_4gpu/20260712_floorlog_sn_mln3_trunc/floorlog-mln3/deterministic/prediction_manifest.json \
  results/floorlog_sn_mln3_trunc_4gpu/20260712_floorlog_sn_mln3_trunc/floorlog-mln3/training/floorlog_mln3

run_model 3 \
  sn \
  results/floorlog_sn_mln3_trunc_4gpu/20260712_floorlog_sn_mln3_trunc/floorlog-skew-normal/config.json \
  results/floorlog_sn_mln3_trunc_4gpu/20260712_floorlog_sn_mln3_trunc/floorlog-skew-normal/training/floorlog_skew_normal/checkpoint-1680 \
  results/floorlog_sn_mln3_trunc_4gpu/20260712_floorlog_sn_mln3_trunc/floorlog-skew-normal/deterministic/prediction_manifest.json \
  results/floorlog_sn_mln3_trunc_4gpu/20260712_floorlog_sn_mln3_trunc/floorlog-skew-normal/training/floorlog_skew_normal

echo "RUN_ROOT=${RUN_ROOT}"
