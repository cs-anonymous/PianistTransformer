#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/inr_epr_pipeline/asap_processed_musical4slot_${STAMP}}"
CONFIG_DIR="${RUN_ROOT}/configs"
mkdir -p "${CONFIG_DIR}"

python - "${CONFIG_DIR}" "${RUN_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

config_dir, run_root = map(Path, sys.argv[1:])
sources = {
    "cinr": Path("results/inr_epr_pipeline/lossnorm_ep20_baselines_20260718_001238/cinr/config.json"),
    "dinr": Path("results/inr_epr_pipeline/lossnorm_ep20_baselines_20260718_001238/dinr/config.json"),
    "cinr_bounded": Path("results/inr_epr_pipeline/lossnorm_ep20_baselines_20260718_001238/cinr_bounded_5pct/config.json"),
}

common = {
    "metadata_path": str(Path("data/ASAP_processed/metadata.generated_json.csv").resolve()),
    "refined_dir": str(Path("data/ASAP_processed").resolve()),
    "use_prepared_sidecar": True,
    "prepared_sidecar_tag": "ASAP",
    "train_performance_dataset": "ASAP",
    "eval_performance_dataset": "ASAP",
    "musical_feature_mode": "musical4slot",
    "disable_musical_features": False,
    "note_embedding_mode": "slot_attribute",
    "slot_version": "slot6",
    "slot_dim": 128,
    "slot_fusion": "mlp",
    "loss_normalization": True,
    "gradnorm": False,
    "seed": 42,
    "sampling_top_p": 0.90,
    "dlm_sampling_top_p": 0.90,
    "dinr_sampling_top_p": 0.90,
    "sampling_top_k": 0,
    "dlm_sampling_top_k": 0,
    "dinr_sampling_top_k": 0,
    "fixed_window_split_scheme": "train_valid_asap3_nonasap05_v1",
    "fixed_window_base_split": "train",
    "fixed_window_train_split_name": "train",
    "fixed_window_eval_split_name": "valid",
    "fixed_window_split_summary_path": "data/train_valid_asap3_nonasap05_v1_summary.json",
}

manifest = {}
for name, path in sources.items():
    cfg = json.loads(path.read_text(encoding="utf-8"))
    for key in ("resume_path", "resume_from_checkpoint"):
        cfg.pop(key, None)
    cfg.update(common)
    cfg["run_name"] = f"asap_processed_musical4slot_{name}"
    cfg["output_dir"] = str(run_root / name / "training")
    cfg["logging_dir"] = str(run_root / name / "tf-logs")
    cfg["num_train_epochs"] = 20.0
    cfg["max_train_epochs"] = 20.0
    out = config_dir / f"{name}.json"
    out.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest[name] = str(out)

(config_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY

cat > "${RUN_ROOT}/run_one.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
gpu="$1"
name="$2"
root="$3"
repo="$4"
cd "${repo}"
run_dir="${root}/${name}"
config="${root}/configs/${name}.json"
mkdir -p "${run_dir}"
printf '%s\tGPU%s\tSTART\t%s\n' "$(date '+%F %T')" "${gpu}" "${name}" >> "${root}/processes.tsv"
env CUDA_VISIBLE_DEVICES="${gpu}" CONFIG="${config}" RUN_DIR_OVERRIDE="${run_dir}" \
  BASE_ASAP_ONLY=1 BASE_NUM_TRAIN_EPOCHS=20 ADAPT_NUM_TRAIN_EPOCHS=0 \
  BATCH_SIZE_PER_DEVICE=32 GLOBAL_BATCH_SIZE=64 DET_NUM_SAMPLES=1 SAMPLING_NUM_SAMPLES=1 \
  INFER_NUM_WORKERS=8 METRIC_NUM_WORKERS=8 INFER_BATCH_SIZE_WINDOWS=8 \
  INFER_SCORE_SOURCE_LIST=data/asap_test_score_sources.txt EVAL_CHECKPOINT_MODE=best \
  RESUME_FROM_LATEST_CHECKPOINT=0 SKIP_EXISTING_PIPELINE_OUTPUTS=0 \
  MERGE_MODE=continuation CONTINUATION_DROP_RATIO=0.0 \
  bash script/run_inr_epr_pipeline.sh > "${run_dir}/launcher.log" 2>&1
printf '%s\tGPU%s\tDONE\t%s\n' "$(date '+%F %T')" "${gpu}" "${name}" >> "${root}/processes.tsv"
SH
chmod +x "${RUN_ROOT}/run_one.sh"

: > "${RUN_ROOT}/processes.tsv"
setsid bash "${RUN_ROOT}/run_one.sh" 0 cinr "${RUN_ROOT}" "${ROOT_DIR}" > "${RUN_ROOT}/cinr.queue.log" 2>&1 < /dev/null &
echo "$!" > "${RUN_ROOT}/cinr.pid"
setsid bash "${RUN_ROOT}/run_one.sh" 1 dinr "${RUN_ROOT}" "${ROOT_DIR}" > "${RUN_ROOT}/dinr.queue.log" 2>&1 < /dev/null &
echo "$!" > "${RUN_ROOT}/dinr.pid"
setsid bash "${RUN_ROOT}/run_one.sh" 2 cinr_bounded "${RUN_ROOT}" "${ROOT_DIR}" > "${RUN_ROOT}/cinr_bounded.queue.log" 2>&1 < /dev/null &
echo "$!" > "${RUN_ROOT}/cinr_bounded.pid"

echo "RUN_ROOT=${RUN_ROOT}"
