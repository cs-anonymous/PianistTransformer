#!/usr/bin/env python3
import argparse
import bisect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from miditoolkit import MidiFile

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.inr_midi import (
    _time_at_tick_ms,
    _tick_to_ms_mapping,
    sorted_pedal_controls,
    sorted_piano_notes,
)


def cc_value_at_ms(times_ms, values, query_ms):
    idx = bisect.bisect_right(times_ms, query_ms)
    return 0 if idx == 0 else int(values[idx - 1])


def candidate_requires_staff(candidate_indices, score_feature, has_score_feature):
    if len(candidate_indices) < 3:
        return False
    staffs = []
    for idx in candidate_indices:
        if (
            idx >= len(has_score_feature)
            or int(has_score_feature[idx]) != 1
            or idx >= len(score_feature)
            or len(score_feature[idx]) <= 4
        ):
            return False
        staffs.append(int(round(float(score_feature[idx][4]))))
    return len(set(staffs)) > 1


def chord_groups(score_raw, score_feature, has_score_feature):
    groups = []
    current = []
    for idx, row in enumerate(score_raw):
        if idx == 0 or float(row[0]) > 0.0:
            if current:
                groups.append(current)
            current = [idx]
            continue

        same_duration = float(score_raw[idx][1]) == float(score_raw[idx - 1][1])
        candidate = current + [idx]
        staff_blocks_merge = candidate_requires_staff(candidate, score_feature, has_score_feature)

        if same_duration and not staff_blocks_merge:
            current.append(idx)
        else:
            if current:
                groups.append(current)
            current = [idx]
    if current:
        groups.append(current)
    return groups


def load_score_sidecar(processed_dir, score_source):
    sidecar = (processed_dir / score_source).with_suffix(".ASAP.pt")
    if not sidecar.exists():
        sidecar = (processed_dir / score_source).with_suffix(".pt")
    payload = torch.load(sidecar, map_location="cpu")
    score = payload["score"]

    def to_list(value):
        return value.tolist() if hasattr(value, "tolist") else value

    return {
        "pitch": [int(v) for v in to_list(score["pitch"])],
        "score_raw": to_list(score["score_raw"]),
        "score_feature": to_list(score["score_feature"]),
        "has_score_feature": to_list(score["has_score_feature"]),
    }


def build_score_chords(score_payload):
    pitch = score_payload["pitch"]
    score_raw = score_payload["score_raw"]
    score_feature = score_payload["score_feature"]
    has_score_feature = score_payload["has_score_feature"]
    groups = chord_groups(score_raw, score_feature, has_score_feature)

    chords = []
    previous_high_score_onset = None
    score_onset = 0.0
    score_onsets = []
    for row in score_raw:
        score_onset += float(row[0])
        score_onsets.append(score_onset)

    for chord_idx, indices in enumerate(groups):
        high_idx = max(indices, key=lambda idx: pitch[idx])
        low_idx = min(indices, key=lambda idx: pitch[idx])
        high_onset = score_onsets[high_idx]
        score_ioi = 0.0 if previous_high_score_onset is None else high_onset - previous_high_score_onset
        previous_high_score_onset = high_onset

        high_row = score_raw[high_idx]
        low_row = score_raw[low_idx]
        staff = None
        if high_idx < len(has_score_feature) and int(has_score_feature[high_idx]) == 1:
            staff = int(round(float(score_feature[high_idx][4])))

        chords.append(
            {
                "chord_idx": chord_idx,
                "note_indices": indices,
                "pitch_indices": sorted(int(pitch[idx]) for idx in indices),
                "size": len(indices),
                "staff": staff,
                "anchor": "highest_pitch",
                "anchor_note_idx": int(high_idx),
                "low_note_idx": int(low_idx),
                "score_base": {
                    "ioi_ms": round(score_ioi, 6),
                    "duration_ms": round(float(high_row[1]), 6),
                    "velocity": int(round(float(high_row[2]))),
                },
                "score_offsets_low_minus_high": {
                    "onset_ms": 0.0,
                    "duration_ms": round(float(low_row[1]) - float(high_row[1]), 6),
                    "velocity": int(round(float(low_row[2]) - float(high_row[2]))),
                },
            }
        )
    return chords


def pedal_samples(pedal_times, pedal_values, starts_ms, chord_idx, high_onset_ms):
    next_onset = starts_ms[chord_idx + 1] if chord_idx + 1 < len(starts_ms) else high_onset_ms + 4990.0
    span = max(next_onset - high_onset_ms, 0.0)
    return [
        cc_value_at_ms(pedal_times, pedal_values, high_onset_ms + span * frac)
        for frac in (0.0, 0.25, 0.50, 0.75)
    ]


def build_performance_chords(refined_dir, performance_source, alignment_source, score_chords):
    midi = MidiFile(str(refined_dir / performance_source))
    notes = sorted_piano_notes(midi)
    alignment = np.load(refined_dir / alignment_source)
    perf_idx = alignment["perf_idx"].astype(int)
    tick_to_ms = _tick_to_ms_mapping(midi)
    controls = sorted_pedal_controls(midi)
    pedal_times = [_time_at_tick_ms(tick_to_ms, cc.time) for cc in controls]
    pedal_values = [cc.value for cc in controls]

    starts = [_time_at_tick_ms(tick_to_ms, notes[int(idx)].start) for idx in perf_idx]
    ends = [_time_at_tick_ms(tick_to_ms, notes[int(idx)].end) for idx in perf_idx]
    velocities = [int(notes[int(idx)].velocity) for idx in perf_idx]

    high_starts = []
    for chord in score_chords:
        high_idx = chord["anchor_note_idx"]
        high_starts.append(float(starts[high_idx]))

    previous_high_onset = None
    labels = []
    for chord in score_chords:
        high_idx = chord["anchor_note_idx"]
        low_idx = chord["low_note_idx"]
        high_onset = float(starts[high_idx])
        low_onset = float(starts[low_idx])
        high_duration = float(ends[high_idx] - starts[high_idx])
        low_duration = float(ends[low_idx] - starts[low_idx])
        perf_ioi = 0.0 if previous_high_onset is None else max(high_onset - previous_high_onset, 0.0)
        previous_high_onset = high_onset

        labels.append(
            {
                "chord_idx": chord["chord_idx"],
                "performance_note_order_by_onset": [
                    {
                        "note_idx": int(idx),
                        "pitch": int(notes[int(perf_idx[idx])].pitch),
                        "onset_ms": round(float(starts[idx]), 6),
                        "velocity": int(velocities[idx]),
                    }
                    for idx in sorted(
                        chord["note_indices"],
                        key=lambda note_idx: (
                            float(starts[note_idx]),
                            int(notes[int(perf_idx[note_idx])].pitch),
                        ),
                    )
                ],
                "base": {
                    "ioi_ms": round(perf_ioi, 6),
                    "duration_ms": round(high_duration, 6),
                    "velocity": velocities[high_idx],
                    "pedal_0_25_50_75": pedal_samples(
                        pedal_times,
                        pedal_values,
                        high_starts,
                        chord["chord_idx"],
                        high_onset,
                    ),
                },
                "offsets_low_minus_high": {
                    "onset_ms": round(low_onset - high_onset, 6),
                    "duration_ms": round(low_duration - high_duration, 6),
                    "velocity": int(velocities[low_idx] - velocities[high_idx]),
                },
            }
        )
    return labels


def sliding_windows(length, window_size, stride):
    windows = []
    start = 0
    while start + window_size <= length:
        windows.append({"start": start, "end": start + window_size})
        start += stride
    if not windows or windows[-1]["end"] != length:
        windows.append({"start": start, "end": length})
    return windows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--score-source",
        default="Bach,_Johann_Sebastian/Italian_Concerto,_BWV_971/score_ASAP_refined.mid",
    )
    parser.add_argument("--processed-dir", default="/home/sy/EPR/PianoCoRe/processed")
    parser.add_argument("--refined-dir", default="/home/sy/EPR/PianoCoRe/refined")
    parser.add_argument("--metadata", default="/home/sy/EPR/PianoCoRe/metadata.csv")
    parser.add_argument(
        "--jsonl",
        default="/home/sy/EPR/PianistTransformer/data/processed/sft/pt_sft_asap_all_processed_raw_repro.jsonl",
    )
    parser.add_argument(
        "--out",
        default="/home/sy/EPR/PianistTransformer/results/analysis/chord_json_sample_italian_concerto.json",
    )
    parser.add_argument(
        "--compact-out",
        default="/home/sy/EPR/PianistTransformer/results/analysis/chord_json_sample_italian_concerto_compact.json",
    )
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument(
        "--all-performances-from-note-json",
        action="store_true",
        help="Use every performance listed in the processed note JSON for this score.",
    )
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    refined_dir = Path(args.refined_dir)
    score_source = args.score_source

    note_json_path = (processed_dir / score_source).with_suffix(".json")
    note_json_perf_meta = {}
    if args.all_performances_from_note_json:
        note_payload = json.loads(note_json_path.read_text(encoding="utf-8"))
        wanted_perfs = []
        for perf in note_payload.get("performances", []):
            perf_source = perf.get("performance_source")
            alignment_source = perf.get("alignment_source")
            if perf_source and alignment_source:
                wanted_perfs.append(perf_source)
                note_json_perf_meta[perf_source] = perf
    else:
        wanted_perfs = []
        for line in Path(args.jsonl).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["score_source"] == score_source:
                wanted_perfs.append(row["performance_source"])

    meta_by_perf = {}
    if wanted_perfs:
        meta = pd.read_csv(
            args.metadata,
            usecols=[
                "performance_id",
                "split",
                "performance_dataset",
                "refined_performance_midi_path",
                "refined_alignment_path",
            ],
        )
        meta = meta[meta["refined_performance_midi_path"].isin(wanted_perfs)]
        meta_by_perf = {
            row.refined_performance_midi_path: row
            for row in meta.itertuples(index=False)
        }

    score_payload = load_score_sidecar(processed_dir, score_source)
    score_chords = build_score_chords(score_payload)

    performances = []
    for perf_source in wanted_perfs:
        row = meta_by_perf.get(perf_source)
        perf_meta = note_json_perf_meta.get(perf_source, {})
        alignment_source = (
            row.refined_alignment_path
            if row is not None
            else perf_meta.get("alignment_source")
        )
        performance_id = (
            row.performance_id
            if row is not None
            else perf_meta.get("performance_id")
        )
        performance_dataset = (
            row.performance_dataset
            if row is not None
            else perf_meta.get("performance_dataset")
        )
        split = row.split if row is not None else perf_meta.get("split")
        performances.append(
            {
                "performance_source": perf_source,
                "performance_id": performance_id,
                "performance_dataset": performance_dataset,
                "split": split,
                "alignment_source": alignment_source,
                "chord_labels": build_performance_chords(
                    refined_dir,
                    perf_source,
                    alignment_source,
                    score_chords,
                ),
            }
        )

    output = {
        "schema": "pianocore_chord_work_sample_v1",
        "definition": {
            "chord": "score IOI == 0 + same score duration; require same staff only when candidate chord size>=3 and all notes have valid staff",
            "base": "highest pitch note in the chord",
            "offsets": "low pitch note minus high pitch note; onset offset may be negative",
            "windows": "fixed chord windows, no random cut",
        },
        "score_source": score_source,
        "score": {
            "chord_count": len(score_chords),
            "chords": score_chords,
        },
        "performances": performances,
        "windows": sliding_windows(len(score_chords), args.window_size, args.stride),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    compact_score = {
        "pitch": [chord["pitch_indices"] for chord in score_chords],
        "score_raw": [
            [
                chord["score_base"]["ioi_ms"],
                chord["score_base"]["duration_ms"],
                chord["score_base"]["velocity"],
            ]
            for chord in score_chords
        ],
        "chord_size": [chord["size"] for chord in score_chords],
        "score_offset_raw": [
            [
                chord["score_offsets_low_minus_high"]["onset_ms"],
                chord["score_offsets_low_minus_high"]["duration_ms"],
                chord["score_offsets_low_minus_high"]["velocity"],
            ]
            for chord in score_chords
        ],
        "score_feature": [
            score_payload["score_feature"][chord["anchor_note_idx"]]
            for chord in score_chords
        ],
        "has_score_feature": [
            score_payload["has_score_feature"][chord["anchor_note_idx"]]
            for chord in score_chords
        ],
        "note_count": len(score_chords),
        "score_source": score_source,
    }

    compact_performances = []
    for perf in performances:
        compact_performances.append(
            {
                key: perf[key]
                for key in (
                    "performance_id",
                    "performance_source",
                    "alignment_source",
                    "split",
                    "performance_dataset",
                )
                if key in perf
            }
        )
        labels = perf["chord_labels"]
        compact_performances[-1].update(
            {
                "label_shared_raw": [
                    [
                        item["base"]["ioi_ms"],
                        item["base"]["duration_ms"],
                        item["base"]["velocity"],
                    ]
                    for item in labels
                ],
                "label_pedal4_raw": [
                    item["base"]["pedal_0_25_50_75"]
                    for item in labels
                ],
                "label_offset_raw": [
                    [
                        item["offsets_low_minus_high"]["onset_ms"],
                        item["offsets_low_minus_high"]["duration_ms"],
                        item["offsets_low_minus_high"]["velocity"],
                    ]
                    for item in labels
                ],
                "interpolated": [0 for _ in labels],
            }
        )

    compact = {
        "schema": "pianocore_chord_work_compact_v1",
        "meta": {
            "score_source": score_source,
            "performance_count": len(compact_performances),
            "chord_definition": "score IOI==0 + same score duration; require same staff only when candidate chord size>=3 and all notes have valid staff",
            "base_note": "highest_pitch",
            "pitch_format": "list[int] per chord, sorted ascending",
            "score_raw_keys": ["ioi_ms_high_anchor", "duration_ms_high", "velocity_high"],
            "label_shared_raw_keys": ["ioi_ms_high_anchor", "duration_ms_high", "velocity_high"],
            "label_pedal4_raw_keys": ["pedal_0", "pedal_25", "pedal_50", "pedal_75"],
            "offset_raw_keys": ["onset_ms_low_minus_high", "duration_ms_low_minus_high", "velocity_low_minus_high"],
            "window_size_chords": args.window_size,
            "stride_chords": args.stride,
        },
        "score": compact_score,
        "performances": compact_performances,
        "windows": sliding_windows(len(score_chords), args.window_size, args.stride),
    }
    compact_out = Path(args.compact_out)
    compact_out.parent.mkdir(parents=True, exist_ok=True)
    compact_out.write_text(json.dumps(compact, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")
    print(f"wrote {compact_out}")
    print(
        json.dumps(
            {
                "score_source": score_source,
                "note_count": len(score_payload["pitch"]),
                "chord_count": len(score_chords),
                "performance_count": len(performances),
                "window_count": len(output["windows"]),
                "first_chords": score_chords[:3],
                "first_perf_label": performances[0]["chord_labels"][:3] if performances else [],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
