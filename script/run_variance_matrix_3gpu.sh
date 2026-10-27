#!/usr/bin/env bash
set -euo pipefail
# Three-GPU scheduler for M0--M4 in v2/INR_EPR五类方差分离实验设计.md.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT_DIR}"
PHASE="${1:-help}"; GPU_LIST="${GPU_LIST:-0,1,2}"
IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
[[ "${#GPUS[@]}" -eq 3 ]] || { echo 'GPU_LIST must contain exactly three ids' >&2; exit 2; }
CONFIG="${CONFIG:-configs/inr_epr/cinr__default_dlm_k1_bounded5.json}"
CHECKPOINT="${CHECKPOINT:-}"; SCORE_SOURCE_LIST="${SCORE_SOURCE_LIST:-data/asap_test_score_sources.txt}"
OUT_ROOT="${OUT_ROOT:-results/variance_matrix_3gpu}"; NUM_WORKERS="${NUM_WORKERS:-1}"; BATCH_SIZE_WINDOWS="${BATCH_SIZE_WINDOWS:-8}"; NUM_SAMPLES="${NUM_SAMPLES:-1}"; CONCURRENCY_PER_GPU="${CONCURRENCY_PER_GPU:-8}"
SEEDS="${SEEDS:-11,22,33,44,55}"; EPOCHS="${EPOCHS:-12}"
mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/manifests"; RUN_MANIFEST="${OUT_ROOT}/manifests/runs.jsonl"; touch "${RUN_MANIFEST}"
need_checkpoint() { [[ -n "${CHECKPOINT}" && -e "${CHECKPOINT}" ]] || { echo 'CHECKPOINT must point to a checkpoint' >&2; exit 2; }; }
record_run() {
  local phase="$1" job="$2" gpu="$3" status="$4" output="$5" seed="$6" protocol="$7"
  python - "$RUN_MANIFEST" "$phase" "$job" "$gpu" "$status" "$output" "$seed" "$protocol" "$CONFIG" <<'PY'
import json,os,platform,subprocess,sys
from datetime import datetime,timezone
p,phase,job,gpu,status,out,seed,protocol,config=sys.argv[1:]
try: commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
except Exception: commit='unknown'
row={'time_utc':datetime.now(timezone.utc).isoformat(),'phase':phase,'job':job,'gpu':gpu,'status':status,'output':out,'sampling_seed':None if seed=='' else int(seed),'rollout_protocol':protocol,'config':os.path.abspath(config),'git_commit':commit,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES',''),'hostname':platform.node()}
with open(p,'a',encoding='utf-8') as f: f.write(json.dumps(row,ensure_ascii=False)+'\n')
PY
}
run_infer_eval() {
  local gpu="$1" job="$2" checkpoint="$3" seed="$4" output="$5" protocol="$6"; mkdir -p "$output"
  if [[ -s "$output/eval_pn_pp_metrics.json" ]]; then record_run "$PHASE" "$job" "$gpu" skipped "$output" "$seed" "$protocol"; return; fi
  { echo "[$(date '+%F %T')] job=$job gpu=$gpu checkpoint=$checkpoint seed=$seed protocol=$protocol"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 python src/inference/infer_inr_testset.py --config "$CONFIG" --checkpoint "$checkpoint" --split test --performance-dataset ASAP --score-source-list "$SCORE_SOURCE_LIST" --protocol sampling --num-samples "$NUM_SAMPLES" --seed "$seed" --batch-size-windows "$BATCH_SIZE_WINDOWS" --num-workers "$NUM_WORKERS" --output-dir "$output"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 python src/evaluate/evaluate_inr_saved_midis.py --prediction-manifest "$output/prediction_manifest.json" --output-json "$output/eval_pn_pp_metrics.json" --score-source-list "$SCORE_SOURCE_LIST" --num-workers "$NUM_WORKERS"
  } > "$OUT_ROOT/logs/${job}.log" 2>&1
  record_run "$PHASE" "$job" "$gpu" done "$output" "$seed" "$protocol"
}
parallel_three() {
  local -a pids=(); local i=0
  for spec in "$@"; do IFS='|' read -r gpu job checkpoint seed output protocol <<< "$spec"; run_infer_eval "$gpu" "$job" "$checkpoint" "$seed" "$output" "$protocol" & pids+=("$!"); i=$((i+1)); if [[ $i -eq 3 ]]; then for pid in "${pids[@]}"; do wait "$pid"; done; pids=(); i=0; fi; done
  for pid in "${pids[@]}"; do wait "$pid"; done
}
parallel_gpu_pool() {
  local -a pids=(); local -a counts=(0 0 0)
  for spec in "$@"; do
    IFS='|' read -r gpu job checkpoint seed output protocol <<< "$spec"
    local slot=-1 i
    for i in 0 1 2; do [[ "${GPUS[$i]}" == "$gpu" ]] && slot="$i"; done
    [[ "$slot" -ge 0 ]] || { echo "GPU $gpu is not in GPU_LIST" >&2; return 2; }
    if [[ "${counts[$slot]}" -ge "$CONCURRENCY_PER_GPU" ]]; then
      for pid in "${pids[@]}"; do wait "$pid"; done
      pids=(); counts=(0 0 0)
    fi
    run_infer_eval "$gpu" "$job" "$checkpoint" "$seed" "$output" "$protocol" &
    pids+=("$!"); counts[$slot]=$((counts[$slot]+1))
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
}
phase_m0() {
  local manifest="${PREDICTION_MANIFEST:-}"; [[ -f "$manifest" ]] || { echo 'M0 requires PREDICTION_MANIFEST=/path/to/prediction_manifest.json' >&2; exit 2; }
  local out="$OUT_ROOT/M0_evaluator"; mkdir -p "$out"
  for workers in 1 2 4 8; do for order in original sorted; do
    local input="$manifest" d="$out/workers${workers}_${order}"; mkdir -p "$d"
    if [[ "$order" == sorted ]]; then input="$d/sorted_manifest.json"; python - "$manifest" "$input" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8')); x['items']=sorted(x['items'],key=lambda z:(z.get('score_source',''),z.get('sample_index',0),z.get('pred',''))); json.dump(x,open(sys.argv[2],'w',encoding='utf-8'),ensure_ascii=False,indent=2)
PY
    fi
    python src/evaluate/evaluate_inr_saved_midis.py --prediction-manifest "$input" --output-json "$d/eval_pn_pp_metrics.json" --num-workers "$workers" --score-source-list "$SCORE_SOURCE_LIST" > "$OUT_ROOT/logs/M0_workers${workers}_${order}.log" 2>&1
  done; done
  record_run M0 evaluator local done "$out" '' saved_manifest
}
phase_m1() {
  need_checkpoint; local -a jobs=(); local idx=0; IFS=',' read -r -a arr <<< "${SAMPLING_SEEDS:-11,22,33,44,55,66,77,88,99,110,121,132,143,154,165,176,187,198,209,220,231,242,253,264}"
  for seed in "${arr[@]}"; do gpu="${GPUS[$((idx%3))]}"; jobs+=("$gpu|M1_s$seed|$CHECKPOINT|$seed|$OUT_ROOT/M1_sampling/seed_$seed|full_score_ar"); idx=$((idx+1)); done; parallel_gpu_pool "${jobs[@]}"
}
phase_m2() {
  need_checkpoint; local -a jobs=(); local idx=0; IFS=',' read -r -a arr <<< "${CHECKPOINTS:-$CHECKPOINT/epoch-4,$CHECKPOINT/epoch-8,$CHECKPOINT/epoch-12,$CHECKPOINT/epoch-16,$CHECKPOINT/epoch-20}"
  for ckpt in "${arr[@]}"; do tag="$(basename "$ckpt")"; gpu="${GPUS[$((idx%3))]}"; jobs+=("$gpu|M2_$tag|$ckpt|42|$OUT_ROOT/M2_checkpoints/$tag|full_score_ar"); idx=$((idx+1)); done; parallel_gpu_pool "${jobs[@]}"
}
phase_m3() {
  local -a arr=(); IFS=',' read -r -a arr <<< "$SEEDS"; local -a pids=(); local i=0
  for seed in "${arr[@]}"; do
    gpu="${GPUS[$((i%3))]}"; run="$OUT_ROOT/M3_training/seed_$seed"; cfg="$run/config.json"; mkdir -p "$run"
    python - "$CONFIG" "$cfg" "$run/training" "$seed" "$EPOCHS" <<'PY'
import json,sys
src,dst,out,seed,epochs=sys.argv[1:]
cfg=json.load(open(src,encoding='utf-8')); cfg.update({'seed':int(seed),'musical_random_seed':int(seed),'dagger_seed':int(seed),'stable_seed':int(seed),'dagger_cache_seed':int(seed),'output_dir':out,'logging_dir':out+'/tf-logs','num_train_epochs':float(epochs),'max_train_epochs':float(epochs),'overwrite_output_dir':True,'auto_rollout_eval_after_train':False,'load_best_model_at_end':False})
json.dump(cfg,open(dst,'w',encoding='utf-8'),ensure_ascii=False,indent=2)
PY
    (CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 python src/train/train_inr.py --config "$cfg" > "$OUT_ROOT/logs/M3_seed$seed.log" 2>&1 && record_run M3 seed_$seed "$gpu" done "$run" "$seed" train) & pids+=("$!"); i=$((i+1)); if [[ $i -eq 3 ]]; then for pid in "${pids[@]}"; do wait "$pid"; done; pids=(); i=0; fi
  done; for pid in "${pids[@]}"; do wait "$pid"; done
}
phase_m4() {
  need_checkpoint; local -a protocols=(teacher_forcing ar_1 window64 window512 continuation full_score_ar); local -a jobs=(); local idx=0
  echo 'M4 labels are recorded for rollout comparison; this entry point runs the current full-score sampler. Use a protocol-specific evaluator when available.' >&2
  for protocol in "${protocols[@]}"; do gpu="${GPUS[$((idx%3))]}"; jobs+=("$gpu|M4_$protocol|$CHECKPOINT|$((900+idx))|$OUT_ROOT/M4_rollout/$protocol|$protocol"); idx=$((idx+1)); done; parallel_gpu_pool "${jobs[@]}"
}
case "$PHASE" in
 M0|m0) phase_m0;; M1|m1) phase_m1;; M2|m2) phase_m2;; M3|m3) phase_m3;; M4|m4) phase_m4;; all) phase_m0; phase_m1; phase_m2; phase_m3; phase_m4;;
 *) cat <<'USAGE'
Usage: script/run_variance_matrix_3gpu.sh M0|M1|M2|M3|M4|all
Common: GPU_LIST=0,1,2 CONFIG=... CHECKPOINT=... SCORE_SOURCE_LIST=... OUT_ROOT=...
M0: PREDICTION_MANIFEST=/path/to/prediction_manifest.json
M1: SAMPLING_SEEDS=11,22,... (default 20 seeds)
M2: CHECKPOINTS=/run/epoch-4,/run/epoch-8,...
M3: SEEDS=11,22,33,44,55 EPOCHS=12
Completed inference outputs are skipped; runs.jsonl records every task.
USAGE
 ;;
esac
