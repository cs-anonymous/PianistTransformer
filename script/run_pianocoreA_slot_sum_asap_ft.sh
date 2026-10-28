#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUN_DIR="${RUN_DIR:-results/pianocoreA_slot_sum_pretrain4_asap_ft8_20261028}"
BASE_CONFIG="${RUN_DIR}/runtime_configs/pianocoreA_slot_sum_pretrain4.json"
ASAP_CONFIG="${RUN_DIR}/runtime_configs/asap_slot_sum_finetune8.json"
SUMMARY_PATH="backup/results0722/dlm_pn_group_ablation_3gpu/20260715_175548/train_valid_asap3_nonasap05_v1_current_summary.json"
RUN_NAME_BASE="pianocoreA_slot_sum_pretrain4_gpu01_global64"
RUN_NAME_ASAP="asap_slot_sum_finetune8_gpu01_global64"
BASE_OUTPUT="${RUN_DIR}/stage1/training/${RUN_NAME_BASE}"
ASAP_OUTPUT="${RUN_DIR}/stage2/training/${RUN_NAME_ASAP}"
BASE_LOG="${RUN_DIR}/logs/stage1_train.log"
ASAP_LOG="${RUN_DIR}/logs/stage2_train.log"

mkdir -p "${RUN_DIR}/runtime_configs" "${RUN_DIR}/logs" "${RUN_DIR}/validation_hr" "${RUN_DIR}/final_asap_test"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

record_status() {
  python - "${RUN_DIR}" "$1" <<'PY'
import datetime, json, sys
from pathlib import Path
p = Path(sys.argv[1])
record = {"timestamp": datetime.datetime.now().astimezone().isoformat(),
          "run": str(p.resolve()), "status": sys.argv[2], "gpu": [0, 1],
          "training_seed": 42, "sampling_seed": 42,
          "configs": [str((p / "runtime_configs" / name).resolve()) for name in
                      ("pianocoreA_slot_sum_pretrain4.json", "asap_slot_sum_finetune8.json")],
          "checkpoints": [str((p / stage / "training" / name / "checkpoint-best").resolve()) for stage, name in
                          (("stage1", "pianocoreA_slot_sum_pretrain4_gpu01_global64"),
                           ("stage2", "asap_slot_sum_finetune8_gpu01_global64"))]}
with (p / "runs.jsonl").open("a") as f:
    f.write(json.dumps(record) + "\n")
PY
}
trap 'status=$?; printf "failed: exit %s at line %s (%s)\n" "$status" "$LINENO" "$(date -Is)" > "${RUN_DIR}/FAILED"; record_status failed; exit "$status"' ERR

python - "${BASE_CONFIG}" "${ASAP_CONFIG}" "${RUN_DIR}" "${SUMMARY_PATH}" <<'PY'
import json
import sys
from pathlib import Path

base_path, asap_path, run_dir, summary_path = map(Path, sys.argv[1:])
root = Path.cwd()
paper_config = root / "backup/repo_cleanup_20260804/results/inr_epr_pipeline/asap_processed_musical51_nolossnorm16_slot_effective_20260723/extra/rep_slot_sum/training/asap_musical51_noloss16_extra_rep_slot_sum/train_config.json"
template = json.loads(paper_config.read_text())

def common(cfg, output_dir, run_name, epochs, grad_accum):
    cfg.update({
        "input_continuous_dim": 64,
        "score_input_continuous_dim": 64,
        "decoder_input_continuous_dim": 64,
        "num_train_epochs": float(epochs),
        "max_train_epochs": float(epochs),
        "per_device_train_batch_size": 32,
        "per_device_eval_batch_size": 32,
        "gradient_accumulation_steps": grad_accum,
        "global_batch_size": 64,
        "learning_rate": 3e-4,
        "run_name": run_name,
        "output_dir": str(output_dir),
        "logging_dir": str(Path(output_dir).parent / "tf-logs"),
        "overwrite_output_dir": False,
        "logging_steps": 20,
        "force_epoch_eval": True,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "save_epoch_aliases": list(range(1, epochs + 1)),
        "save_total_limit": 3,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "resume_trainer_state": False,
        "retain_optimizer_state": True,
        "auto_rollout_eval_after_train": False,
        "eval_split": "valid",
        "fixed_window_split_scheme": "train_valid_asap3_nonasap05_v1",
        "fixed_window_base_split": "train",
        "fixed_window_train_split_name": "train",
        "fixed_window_eval_split_name": "valid",
        "fixed_window_split_summary_path": str(summary_path),
        "block_notes": 512,
        "overlap_ratio": 0.125,
        "min_notes": 64,
        "slot_version": "slot6",
        "slot_dim": 768,
        "slot_fusion": "sum",
        "musical_feature_mode": "musical51_full",
        "disable_musical_features": False,
        "note_embedding_mode": "slot_attribute",
        "epr_distribution": "dlm",
        "dlm_components": 1,
        "pedal_representation": "binary_4",
        "bf16": True,
        "dataloader_num_workers": 4,
        "eval_dataloader_num_workers": 4,
        "eval_dataloader_persistent_workers": False,
        "ddp_find_unused_parameters": True,
        "ddp_broadcast_buffers": False,
        "early_stopping_patience": 0,
        "record_epoch_train_loss": True,
        "seed": 42,
    })
    cfg.pop("max_steps", None)
    cfg["max_steps"] = -1

base = dict(template)
base.update({
    "metadata_path": "backup/PianoCoRe/metadata.csv",
    "refined_dir": "backup/PianoCoRe/processed",
    "prepared_sidecar_tag": "PCOREA_MUSICAL51",
    "eval_include_all_performance_dataset": "ASAP",
    "max_eval_non_asap_performances_per_work": 8,
    "eval_gt_time_normalization": "score_onset_span",
})
base.pop("train_performance_dataset", None)
base.pop("eval_performance_dataset", None)
base.pop("train_exclude_performance_dataset", None)
base.pop("eval_exclude_performance_dataset", None)
base.pop("resume_path", None)
common(base, run_dir / "stage1/training", "pianocoreA_slot_sum_pretrain4_gpu01_global64", 4, 1)
base_path.parent.mkdir(parents=True, exist_ok=True)
base_path.write_text(json.dumps(base, indent=2, ensure_ascii=False) + "\n")

asap = dict(template)
asap.update({
    "metadata_path": "data/ASAP_processed/metadata.generated_json.csv",
    "refined_dir": "data/ASAP_processed",
    "prepared_sidecar_tag": "ASAP_MUSICAL51",
    "train_performance_dataset": "ASAP",
    "eval_performance_dataset": "ASAP",
    "eval_include_all_performance_dataset": None,
    "max_eval_non_asap_performances_per_work": None,
    "eval_gt_time_normalization": "score_onset_span",
    "learning_rate": 3e-5,
    "adapt_learning_rate": 3e-5,
})
asap.pop("resume_path", None)
common(asap, run_dir / "stage2/training", "asap_slot_sum_finetune8_gpu01_global64", 8, 1)
asap["learning_rate"] = 3e-5
asap["resume_path"] = str((run_dir / "stage1/training/pianocoreA_slot_sum_pretrain4_gpu01_global64/checkpoint-best").resolve())
asap_path.parent.mkdir(parents=True, exist_ok=True)
asap_path.write_text(json.dumps(asap, indent=2, ensure_ascii=False) + "\n")
(run_dir / "paper_config_comparison.json").write_text(json.dumps({
    "source": str(paper_config.resolve()), "paper_PP_HR_reference": 2.38,
    "effective_global_batch": "32 per GPU * 2 GPUs * 1 accumulation = 64",
    "stage1_changes": {k: {"paper": template.get(k), "new": base.get(k)} for k in sorted(template.keys() | base.keys()) if template.get(k) != base.get(k)},
    "stage2_changes": {k: {"paper": template.get(k), "new": asap.get(k)} for k in sorted(template.keys() | asap.keys()) if template.get(k) != asap.get(k)},
}, indent=2) + "\n")
PY

if [[ "${CONFIG_ONLY:-0}" == "1" ]]; then
  echo "Runtime configs written under ${RUN_DIR}/runtime_configs"
  exit 0
fi

export CUDA_VISIBLE_DEVICES="0,1"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo

if [[ ! -s "${RUN_DIR}/data_preparation_summary.json" ]]; then
  echo "Missing PianoCoRe-A preparation summary" >&2
  exit 1
fi
python - "${RUN_DIR}/data_preparation_summary.json" <<'PY'
import json, sys
x = json.load(open(sys.argv[1]))
assert x["works"] == 1780 and x["successful_works"] == 1780 and not x["errors"], x
assert x["metadata_performances"] == 141719, x
assert x["performances"] + len(x["excluded_performances"]) == x["metadata_performances"], x
PY

python script/audit_pianocoreA_slot_sum_run.py --run-dir "${RUN_DIR}" --preflight
PCORE_TF_M=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["pianocoreA"]["valid"]["hr_eligible_works"])' "${RUN_DIR}/data_preflight.json")
ASAP_TF_M=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["asap"]["valid"]["hr_eligible_works"])' "${RUN_DIR}/data_preflight.json")

echo "[$(date '+%F %T')] PianoCoRe-A slot-sum pretraining: 4 epochs, global batch 64" | tee -a "${BASE_LOG}"
record_status stage1_training_started
STAGE1_START=$(date +%s)
torchrun --standalone --nnodes=1 --nproc_per_node=2 src/train/train_inr.py --config "${BASE_CONFIG}" 2>&1 | tee -a "${BASE_LOG}"
STAGE1_END=$(date +%s)
printf '%s,%s\n' "$STAGE1_START" "$STAGE1_END" > "${RUN_DIR}/stage1_timing.csv"
touch "${RUN_DIR}/stage1_training_complete"
record_status stage1_training_complete

for epoch in 1 2 3 4; do
  python src/evaluate/eval_inr_window_rollout_hr.py \
    --config "${BASE_OUTPUT}/train_config.json" \
    --checkpoint "${BASE_OUTPUT}/epoch-${epoch}" \
    --output-json "${RUN_DIR}/validation_hr/pianocoreA_pretrain_epoch-${epoch}.json" \
    --split valid --performance-dataset ALL --m-values "${PCORE_TF_M}" --rollout-ks 0 \
    --batch-size-windows 1 --num-workers 0 --device cuda:0 --fail-on-insufficient-works \
    > "${RUN_DIR}/logs/pianocoreA_pretrain_epoch-${epoch}_tf_hr.log" 2>&1
  python script/audit_pianocoreA_slot_sum_run.py --run-dir "${RUN_DIR}" --summarize-validation
done

echo "[$(date '+%F %T')] ASAP slot-sum fine-tuning: 8 epochs, global batch 64" | tee -a "${ASAP_LOG}"
record_status stage2_training_started
STAGE2_START=$(date +%s)
torchrun --standalone --nnodes=1 --nproc_per_node=2 src/train/train_inr.py --config "${ASAP_CONFIG}" 2>&1 | tee -a "${ASAP_LOG}"
STAGE2_END=$(date +%s)
printf '%s,%s\n' "$STAGE2_START" "$STAGE2_END" > "${RUN_DIR}/stage2_timing.csv"
touch "${RUN_DIR}/stage2_training_complete"
record_status stage2_training_complete

for epoch in 1 2 3 4 5 6 7 8; do
  python src/evaluate/eval_inr_window_rollout_hr.py \
    --config "${ASAP_OUTPUT}/train_config.json" \
    --checkpoint "${ASAP_OUTPUT}/epoch-${epoch}" \
    --output-json "${RUN_DIR}/validation_hr/asap_finetune_epoch-${epoch}.json" \
    --split valid --performance-dataset ASAP --m-values "${ASAP_TF_M}" --rollout-ks 0 \
    --batch-size-windows 1 --num-workers 0 --device cuda:0 --fail-on-insufficient-works \
    > "${RUN_DIR}/logs/asap_finetune_epoch-${epoch}_tf_hr.log" 2>&1
  python script/audit_pianocoreA_slot_sum_run.py --run-dir "${RUN_DIR}" --summarize-validation
done

record_status final_test_evaluation_started
for stage in pretrained finetuned; do
if [[ "$stage" == "pretrained" ]]; then
  FINAL_CHECKPOINT="${BASE_OUTPUT}/checkpoint-best"
else
  FINAL_CHECKPOINT="${ASAP_OUTPUT}/checkpoint-best"
fi
INFER_DIR="${RUN_DIR}/final_asap_test/${stage}/sampling"
CUDA_VISIBLE_DEVICES=0 python src/inference/infer_inr_testset.py \
  --config "${ASAP_CONFIG}" \
  --checkpoint "${FINAL_CHECKPOINT}" \
  --split test --performance-dataset ASAP --protocol sampling --num-samples 3 \
  --num-workers 1 --batch-size-windows 4 --device cuda --merge-mode continuation --seed 42 \
  --output-dir "${INFER_DIR}" 2>&1 | tee "${RUN_DIR}/logs/${stage}_asap_test_inference.log"

python src/evaluate/normalize_prediction_manifest_score_span.py \
  --manifest "${INFER_DIR}/prediction_manifest.json" \
  --output-manifest "${INFER_DIR}/score_span_prediction_manifest.json" \
  --output-root "${INFER_DIR}/score_span_normalized"
python script/audit_pianocoreA_slot_sum_run.py --run-dir "${RUN_DIR}" --audit-normalization "${INFER_DIR}/score_span_prediction_manifest.json"
python src/evaluate/evaluate_inr_saved_midis.py \
  --prediction-manifest "${INFER_DIR}/score_span_prediction_manifest.json" \
  --output-json "${RUN_DIR}/final_asap_test/${stage}/score_span_metrics.json" \
  --num-workers 4 2>&1 | tee "${RUN_DIR}/logs/${stage}_asap_test_metrics.log"
done

python script/audit_pianocoreA_slot_sum_run.py --run-dir "${RUN_DIR}" --final
record_status complete

touch "${RUN_DIR}/COMPLETE"
echo "[$(date '+%F %T')] Completed: ${RUN_DIR}"
