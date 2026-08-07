#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SOURCE_CONFIG="backup/repo_cleanup_20260804/results/inr_epr_pipeline/asap_processed_musical51_3baseline_20260723/configs/cinr_bounded_ep16_nolossnorm.json"
RUN_ROOT="${RUN_ROOT:-results/paper_baseline_strict_repro_3gpu_$(date +%Y%m%d_%H%M%S)}"
CONFIG_DIR="${RUN_ROOT}/configs"
mkdir -p "${CONFIG_DIR}"

python - "${SOURCE_CONFIG}" "${CONFIG_DIR}" <<'PY'
import json
import sys
from pathlib import Path

source, output_dir = map(Path, sys.argv[1:])
base = json.loads(source.read_text(encoding="utf-8"))

for gpu in (0, 1):
    config = dict(base)
    config["run_name"] = f"paper_baseline_repro_gpu{gpu}"
    (output_dir / f"gpu{gpu}.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

pedal_group = dict(base)
pedal_group["metadata_path"] = str(Path("data/ASAP_processed_pedal_group/metadata.generated_json.csv").resolve())
pedal_group["refined_dir"] = str(Path("data/ASAP_processed_pedal_group").resolve())
pedal_group["mask_zero_score_ioi_pedal_loss"] = True
pedal_group["run_name"] = "paper_baseline_repro_gpu2_pedal_group"
(output_dir / "gpu2.json").write_text(
    json.dumps(pedal_group, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
PY

for gpu in 0 1 2; do
  session="paper_baseline_repro_g${gpu}"
  run_dir="${RUN_ROOT}/gpu${gpu}"
  config="${CONFIG_DIR}/gpu${gpu}.json"
  tmux kill-session -t "${session}" 2>/dev/null || true
  command="cd '${ROOT_DIR}' && env CUDA_VISIBLE_DEVICES='${gpu}' CONFIG='${config}' RUN_DIR_OVERRIDE='${run_dir}' BASE_ASAP_ONLY=1 BASE_NUM_TRAIN_EPOCHS=16 ADAPT_NUM_TRAIN_EPOCHS=0 BATCH_SIZE_PER_DEVICE=32 GLOBAL_BATCH_SIZE=64 SAMPLING_NUM_SAMPLES=2 INFER_NUM_WORKERS=8 METRIC_NUM_WORKERS=8 INFER_BATCH_SIZE_WINDOWS=8 INFER_SCORE_SOURCE_LIST=data/asap_test_score_sources.txt EVAL_CHECKPOINT_MODE=best RESUME_FROM_LATEST_CHECKPOINT=0 SKIP_EXISTING_PIPELINE_OUTPUTS=0 MERGE_MODE=continuation CONTINUATION_DROP_RATIO=0.0 bash script/run_inr_epr_pipeline.sh > '${run_dir}/launcher.log' 2>&1"
  mkdir -p "${run_dir}"
  tmux new-session -d -s "${session}" "${command}"
done

printf '%s\n' "${RUN_ROOT}"
