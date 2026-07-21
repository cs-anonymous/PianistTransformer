#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/inr_epr_pipeline/gradnorm_ablation_${STAMP}}"
CONFIG_DIR="${RUN_ROOT}/configs"
mkdir -p "${CONFIG_DIR}"

python - "${CONFIG_DIR}" "${RUN_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

config_dir, run_root = map(Path, sys.argv[1:])
dinr_base = json.loads(Path(
    "results/inr_epr_pipeline/lossnorm_ep20_baselines_20260718_001238/dinr/config.json"
).read_text())
cinr_base = json.loads(Path(
    "results/inr_epr_pipeline/lossnorm_ep20_baselines_20260718_001238/cinr_bounded_5pct/config.json"
).read_text())

def base(src, name, method):
    cfg = dict(src)
    cfg.update({
        "run_name": f"gradnorm_{name}",
        "output_dir": str(run_root / name / "training"),
        "logging_dir": str(run_root / name / "tf-logs"),
        "num_train_epochs": 16.0,
        "max_train_epochs": 16.0,
        "loss_normalization": False,
        "gradnorm": True,
        "gradnorm_alpha": 0.5,
        "seed": 42,
        "slot_version": "slot6",
        "slot_dim": 128,
        "slot_fusion": "mlp",
        "metadata_path": str(Path("data/ASAP_processed/metadata.generated_json.csv").resolve()),
        "refined_dir": str(Path("data/ASAP_processed").resolve()),
        "musical_feature_mode": "musical4slot",
        "disable_musical_features": False,
        "note_embedding_mode": "slot_attribute",
        "epr_distribution": method,
        "pedal_representation": "binary_4",
        "sampling_top_p": 0.90,
        "dlm_sampling_top_p": 0.90,
        "dinr_sampling_top_p": 0.90,
        "sampling_top_k": 0,
        "dlm_sampling_top_k": 0,
        "dinr_sampling_top_k": 0,
    })
    cfg.pop("prepared_sidecar_tag", None)
    return cfg

configs = {}
queues = {"gpu0": [], "gpu1": [], "gpu2": []}

def add(name, cfg, queue):
    path = config_dir / f"{name}.json"
    path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
    configs[name] = str(path)
    queues[queue].append(name)

# Queue 0: method and T5 backbone comparisons. GPT is intentionally omitted:
# the current implementation has no GPT-specific integrated backbone contract.
add("dinr", base(dinr_base, "dinr", "dinr"), "gpu0")
add("cinr_bounded", base(cinr_base, "cinr_bounded", "dlm"), "gpu0")
add("t5_6_6", {**base(cinr_base, "t5_6_6", "dlm"),
                "encoder_layers_num": 6, "decoder_layers_num": 6}, "gpu0")
add("t5_8_4", {**base(cinr_base, "t5_8_4", "dlm"),
                "encoder_layers_num": 8, "decoder_layers_num": 4}, "gpu0")
add("t5_10_2", {**base(cinr_base, "t5_10_2", "dlm"),
                "encoder_layers_num": 10, "decoder_layers_num": 2}, "gpu0")
add("t5_16_0", {**base(cinr_base, "t5_16_0", "dlm"),
                "encoder_layers_num": 16, "decoder_layers_num": 0}, "gpu0")
add("cinr_bounded_slot256_mlp",
    {**base(cinr_base, "cinr_bounded_slot256_mlp", "dlm"),
     "slot_dim": 256, "slot_fusion": "mlp"},
    "gpu0")

# Queue 1: representation comparisons, four variants x DINR/CINR-bounded.
rep_variants = {
    "sine": {"note_embedding_mode": "sine", "slot_fusion": "mlp"},
    "slot_sum": {"note_embedding_mode": "slot_attribute", "slot_fusion": "sum",
                 "slot_dim": 768},
    "slot_direct": {"note_embedding_mode": "slot_attribute",
                    "slot_fusion": "direct_concat", "slot_dim": 128},
}
for method, src, prefix in (("dinr", dinr_base, "dinr"), ("dlm", cinr_base, "cinr_bounded")):
    for variant, overrides in rep_variants.items():
        cfg = base(src, f"{prefix}_{variant}", method)
        cfg.update(overrides)
        add(f"{prefix}_{variant}", cfg, "gpu1")

# Keep one explicit current-representation control in this queue so all three
# GPU queues contain seven serial experiments.
add(
    "cinr_bounded_current_rep",
    base(cinr_base, "cinr_bounded_current_rep", "dlm"),
    "gpu1",
)

# Queue 2: DINR coordinate/table ablations. Legacy musical ablations are retired.
for label, overrides in (
    ("dinr_no_coord", {"dinr_output_deviation_numerical_coordinates": False}),
    ("dinr_separate_timing_tables", {"dinr_separate_timing_tables": True}),
    ("dinr_timing_dev_no_coord", {"dinr_output_deviation_numerical_coordinates": False}),
):
    cfg = base(dinr_base, label, "dinr")
    cfg.update(overrides)
    add(label, cfg, "gpu2")

(config_dir / "manifest.json").write_text(json.dumps({
    "epochs": 16,
    "loss_normalization": False,
    "gradnorm": True,
    "gpt_omitted": True,
    "queues": queues,
    "configs": configs,
}, indent=2, ensure_ascii=False) + "\n")
PY

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "RUN_ROOT=${RUN_ROOT}"
  cat "${CONFIG_DIR}/manifest.json"
  exit 0
fi

WORKER="${RUN_ROOT}/run_queue.sh"
cat > "${WORKER}" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
gpu="$1"
queue="$2"
root="$3"
root_dir="$4"
cd "${root_dir}"
manifest="${root}/configs/manifest.json"
mapfile -t names < <(python - "$manifest" "$queue" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
for name in m["queues"][sys.argv[2]]:
    print(name)
PY
)
for name in "${names[@]}"; do
  run_dir="${root}/${name}"
  config="${root}/configs/${name}.json"
  mkdir -p "${run_dir}"
  if [[ -s "${run_dir}/summary.json" ]]; then
    printf '%s\tGPU%s\tSKIP\t%s\n' "$(date '+%F %T')" "$gpu" "$name" >> "${root}/processes.tsv"
    continue
  fi
  printf '%s\tGPU%s\tSTART\t%s\n' "$(date '+%F %T')" "$gpu" "$name" >> "${root}/processes.tsv"
  env CUDA_VISIBLE_DEVICES="${gpu}" CONFIG="${config}" RUN_DIR_OVERRIDE="${run_dir}" \
    BASE_ASAP_ONLY=1 BASE_NUM_TRAIN_EPOCHS=16 ADAPT_NUM_TRAIN_EPOCHS=0 \
    BATCH_SIZE_PER_DEVICE=32 GLOBAL_BATCH_SIZE=64 DET_NUM_SAMPLES=1 \
    SAMPLING_NUM_SAMPLES=1 INFER_NUM_WORKERS=8 METRIC_NUM_WORKERS=8 \
    INFER_BATCH_SIZE_WINDOWS=8 INFER_SCORE_SOURCE_LIST=data/asap_test_score_sources.txt \
    EVAL_CHECKPOINT_MODE=best RESUME_FROM_LATEST_CHECKPOINT=0 \
    MERGE_MODE=continuation CONTINUATION_DROP_RATIO=0.0 \
    bash script/run_inr_epr_pipeline.sh > "${run_dir}/launcher.log" 2>&1
  printf '%s\tGPU%s\tDONE\t%s\n' "$(date '+%F %T')" "$gpu" "$name" >> "${root}/processes.tsv"
done
SH
chmod +x "${WORKER}"

for spec in "0 gpu0" "1 gpu1" "2 gpu2"; do
  read -r gpu queue <<< "${spec}"
  tmux new-session -d -s "gradnorm_${queue}_${STAMP: -6}" \
    "bash '${WORKER}' '${gpu}' '${queue}' '${RUN_ROOT}' '${ROOT_DIR}'"
done

echo "RUN_ROOT=${RUN_ROOT}"
echo "GPT omitted; T5/backbone and supported representation variants queued."
