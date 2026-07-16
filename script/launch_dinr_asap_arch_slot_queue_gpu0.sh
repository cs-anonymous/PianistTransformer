#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/inr_epr_pipeline/dinr_asap_arch_slot_queue_gpu0_${STAMP}}"
CONFIG_DIR="${RUN_ROOT}/configs"
BASE_CONFIG="results/inr_epr_pipeline/dinr_separated_corrected_20260716_004453/config.json"
mkdir -p "${CONFIG_DIR}"

python - "${BASE_CONFIG}" "${CONFIG_DIR}" <<'PY'
import json, sys
from pathlib import Path

base_path, out = Path(sys.argv[1]), Path(sys.argv[2])
base = json.loads(base_path.read_text())
base.update({
    "epr_distribution": "dinr",
    "epr_timing_target": "floor_log_deviation",
    "dinr_vocabulary_mode": "separated",
    "dinr_deviation_min": -2.0,
    "dinr_deviation_max": 1.0,
    "dinr_zero_ioi_min": 0.0,
    "dinr_zero_ioi_max": 5.0,
    "dinr_sampling_temperature": 0.8,
    "dinr_sampling_top_p": 0.95,
    "sampling_top_p": 0.95,
    "train_performance_dataset": "ASAP",
    "eval_performance_dataset": "ASAP",
    "eval_split": "valid",
    "prepared_sidecar_tag": "DINR_READY_ASAP",
    "num_train_epochs": 16.0,
    "max_train_epochs": 16.0,
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "encoder_layers_num": 10,
    "decoder_layers_num": 2,
    "slot_dim": 128,
    "slot_version": "slot8",
    "slot_fusion": "mlp",
})
for key in ("resume_path", "train_performance_dataset_exclude", "eval_performance_dataset_exclude"):
    base.pop(key, None)

variants = {
    "backbone_8enc_4dec": {"encoder_layers_num": 8, "decoder_layers_num": 4},
    "backbone_6enc_6dec": {"encoder_layers_num": 6, "decoder_layers_num": 6},
    "hidden1024_10enc_2dec": {"hidden_size": 1024, "intermediate_size": 4096},
    "slot_dim256": {"slot_dim": 256},
    "slot_dim512": {"slot_dim": 512},
    "slot_dim768": {"slot_dim": 768},
    "slot_dim96": {"slot_dim": 96},
    "slot_dim96_no_mlp": {"slot_dim": 96, "slot_fusion": "direct_concat"},
}
report = {"baseline": str(base_path), "variants": {}}
for name, override in variants.items():
    cfg = dict(base)
    cfg.update(override)
    cfg["run_name"] = f"DINR-ASAP-{name}-m2p1-topp95-t08"
    (out / f"{name}.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
    report["variants"][name] = {
        "encoder_layers": cfg["encoder_layers_num"],
        "decoder_layers": cfg["decoder_layers_num"],
        "hidden_size": cfg["hidden_size"],
        "intermediate_size": cfg["intermediate_size"],
        "slot_dim": cfg["slot_dim"],
        "slot_fusion": cfg["slot_fusion"],
    }
(out / "config_report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1 RUN_ROOT=${RUN_ROOT}"
  exit 0
fi

variants=(
  backbone_8enc_4dec
  backbone_6enc_6dec
  hidden1024_10enc_2dec
  slot_dim256
  slot_dim512
  slot_dim768
  slot_dim96
  slot_dim96_no_mlp
)

: > "${RUN_ROOT}/queue_status.tsv"
for name in "${variants[@]}"; do
  run_dir="${RUN_ROOT}/${name}"
  per_device=32
  if [[ "${name}" == hidden1024_10enc_2dec ]]; then per_device=16; fi
  mkdir -p "${run_dir}"
  printf '%s\tSTART\t%s\t%s\n' "$(date '+%F %T')" "${name}" "${run_dir}" | tee -a "${RUN_ROOT}/queue_status.tsv"
  env CUDA_VISIBLE_DEVICES=0 \
    CONFIG="${CONFIG_DIR}/${name}.json" RUN_DIR_OVERRIDE="${run_dir}" \
    BASE_ASAP_ONLY=1 BASE_NUM_TRAIN_EPOCHS=16 ADAPT_NUM_TRAIN_EPOCHS=0 \
    ADAPT_PREPARED_SIDECAR_TAG=DINR_READY_ASAP \
    BATCH_SIZE_PER_DEVICE="${per_device}" GLOBAL_BATCH_SIZE=64 \
    DET_NUM_SAMPLES=1 SAMPLING_NUM_SAMPLES=1 \
    INFER_NUM_WORKERS=8 METRIC_NUM_WORKERS=8 INFER_BATCH_SIZE_WINDOWS=8 \
    INFER_SCORE_SOURCE_LIST=data/asap_test_score_sources.txt \
    EVAL_CHECKPOINT_MODE=best RESUME_FROM_LATEST_CHECKPOINT=0 \
    MERGE_MODE=continuation CONTINUATION_DROP_RATIO=0.0 \
    bash script/run_inr_epr_pipeline.sh 2>&1 | tee "${run_dir}/launcher.log"
  printf '%s\tDONE\t%s\t%s\n' "$(date '+%F %T')" "${name}" "${run_dir}" | tee -a "${RUN_ROOT}/queue_status.tsv"
done

echo "QUEUE_DONE $(date '+%F %T')" | tee -a "${RUN_ROOT}/queue_status.tsv"
