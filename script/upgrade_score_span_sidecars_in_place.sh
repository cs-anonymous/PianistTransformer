#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

common_args=(
  --metadata-path PianoCoRe/metadata.csv
  --refined-dir PianoCoRe/processed
  --block-notes 512
  --overlap-ratio 0.125
  --min-notes 64
  --input-feature-mode integrated
  --timing-input-normalization linear_5000
  --max-time-ms 10000
  --pedal-representation binary_4
  --musical-feature-mode musical51_full
  --epr-timing-target floor_log_deviation
  --timing-control-mode dinr_floor_log
  --timing-log-scale 50
  --sidecar-tag ASAP_DINR_SCORESPAN
  --ready
  --performance-time-normalization score_onset_span
  --performance-dataset ASAP
  --workers 40
)

python src/data_process/prebuild_inr_work_pt.py "${common_args[@]}" --split train
python src/data_process/prebuild_inr_work_pt.py "${common_args[@]}" --split test

python - <<'PY'
import glob
import json
import torch

paths = glob.glob("PianoCoRe/processed/**/*.ASAP_DINR_SCORESPAN.pt", recursive=True)
# ASAP-only training contains 188 works and the held-out ASAP test protocol
# contributes another 19 score works.
assert len(paths) == 207, len(paths)
for path in paths:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    signature = json.loads(payload.get("_cache_signature", "{}"))
    assert signature.get("schema") == 7, path
    assert signature.get("kind") == "inr_multi_target_ready_sidecar", path
    assert payload.get("performance_time_normalization") == "score_onset_span", path
    assert payload.get("score_input") is not None, path
    for perf in payload.get("performances", []):
        targets = set((perf.get("labels_by_target") or {}).keys())
        assert {"floor_log_deviation", "floor_log_absolute"} <= targets, (path, perf.get("performance_source"))
print("VALIDATED_READY_SCORE_SPAN", len(paths), flush=True)
PY

find PianoCoRe/processed -name '*.DINR_READY_ASAP.pt' -delete
remaining="$(find PianoCoRe/processed -name '*.DINR_READY_ASAP.pt' | wc -l)"
printf 'DINR_READY_ASAP_REMAINING %s\n' "$remaining"
