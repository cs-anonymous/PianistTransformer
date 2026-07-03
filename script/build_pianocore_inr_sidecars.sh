#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

PIANOCORE_DIR="${PIANOCORE_DIR:-data/pianocore}"
PROCESSED_DIR="${PROCESSED_DIR:-../PianoCoRe/processed}"
RAW_MIDI_ZIP="${RAW_MIDI_ZIP:-${PIANOCORE_DIR}/PianoCoRe-1.0-raw-midi.zip}"
WORKERS="${WORKERS:-36}"

echo "[1/5] Generate paired INR JSON"
python src/data_process/generate_json_with_paired_midi.py \
  --pianocore-dir "${PIANOCORE_DIR}" \
  --metadata metadata.csv \
  --output-dir "${PROCESSED_DIR}" \
  --summary-path "${PROCESSED_DIR}/processed_raw_summary.json" \
  --num-proc "${WORKERS}" \
  --overwrite

echo "[2/5] Project XML score features"
python src/data_process/update_json_score_feature_with_xml.py \
  --pianocore-dir "${PIANOCORE_DIR}" \
  --raw-midi-zip "${RAW_MIDI_ZIP}" \
  --json-dir "${PROCESSED_DIR}" \
  --subset a \
  --num-proc "${WORKERS}" \
  --summary-path "${PROCESSED_DIR}/processed_score_feature_update_summary.json" \
  --details-path "${PROCESSED_DIR}/processed_score_feature_update_details.jsonl"

echo "[3/5] Write fixed train/valid window split metadata"
python src/data_process/create_fixed_window_valid_split.py \
  --metadata-path "${PIANOCORE_DIR}/metadata.csv" \
  --refined-dir "${PROCESSED_DIR}" \
  --output-summary data/train_valid_asap3_nonasap05_v1_summary.json \
  --skip-sidecars \
  --workers "${WORKERS}"

echo "[4/5] Prebuild base INR sidecars (.pt)"
python src/data_process/prebuild_inr_work_pt.py \
  --metadata-path "${PIANOCORE_DIR}/metadata.csv" \
  --refined-dir "${PROCESSED_DIR}" \
  --split train \
  --workers "${WORKERS}" \
  --sidecar-tag NONE

echo "[5/5] Prebuild ASAP-only INR sidecars (.ASAP.pt)"
python src/data_process/prebuild_inr_work_pt.py \
  --metadata-path "${PIANOCORE_DIR}/metadata.csv" \
  --refined-dir "${PROCESSED_DIR}" \
  --split train \
  --performance-dataset ASAP \
  --workers "${WORKERS}" \
  --sidecar-tag ASAP

echo "Pipeline complete."
echo "Processed dir: ${PROCESSED_DIR}"
echo "Fixed split summary: data/train_valid_asap3_nonasap05_v1_summary.json"
