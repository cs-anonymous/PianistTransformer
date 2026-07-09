#!/usr/bin/env python3
import argparse
import bisect
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from miditoolkit import MidiFile

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.inr_midi import (  # noqa: E402
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


def sliding_windows(length, window_size, stride, min_length):
    windows = []
    start = 0
    while start + window_size <= length:
        windows.append([start, start + window_size])
        start += stride
    if (not windows or windows[-1][1] != length) and length - start >= min_length:
        windows.append([start, length])
    elif not windows and length > 0:
        windows.append([0, length])
    return windows


def load_score_payload(processed_dir, score_source):
    sidecar = (processed_dir / score_source).with_suffix(".ASAP.pt")
    if not sidecar.exists():
        sidecar = (processed_dir / score_source).with_suffix(".pt")
    if not sidecar.exists():
        raise FileNotFoundError(f"missing score sidecar for {score_source}")
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

    score_onsets = []
    onset = 0.0
    for row in score_raw:
        onset += float(row[0])
        score_onsets.append(onset)

    chords = []
    previous_high_onset = None
    for indices in groups:
        high_idx = max(indices, key=lambda idx: pitch[idx])
        low_idx = min(indices, key=lambda idx: pitch[idx])
        high_onset = score_onsets[high_idx]
        score_ioi = 0.0 if previous_high_onset is None else high_onset - previous_high_onset
        previous_high_onset = high_onset
        high_row = score_raw[high_idx]
        low_row = score_raw[low_idx]
        chords.append(
            {
                "note_indices": [int(idx) for idx in indices],
                "pitch": sorted(int(pitch[idx]) for idx in indices),
                "anchor_note_idx": int(high_idx),
                "low_note_idx": int(low_idx),
                "score_raw": [
                    round(float(score_ioi), 6),
                    round(float(high_row[1]), 6),
                    int(round(float(high_row[2]))),
                ],
                "score_offset_raw": [
                    0.0,
                    round(float(low_row[1]) - float(high_row[1]), 6),
                    int(round(float(low_row[2]) - float(high_row[2]))),
                ],
                "score_feature": score_feature[high_idx],
                "has_score_feature": int(has_score_feature[high_idx]),
            }
        )
    return chords


PIANO_PITCH_MIN = 21
PIANO_PITCH_COUNT = 88


def pitch_multihot(pitch_lists):
    arr = np.zeros((len(pitch_lists), PIANO_PITCH_COUNT), dtype=np.uint8)
    for idx, pitches in enumerate(pitch_lists):
        for pitch in pitches:
            piano_idx = int(pitch) - PIANO_PITCH_MIN
            if 0 <= piano_idx < PIANO_PITCH_COUNT:
                arr[idx, piano_idx] = 1
    return arr


def pedal_samples(pedal_times, pedal_values, high_starts, chord_idx, high_onset):
    next_onset = high_starts[chord_idx + 1] if chord_idx + 1 < len(high_starts) else high_onset + 4990.0
    span = max(next_onset - high_onset, 0.0)
    return [
        cc_value_at_ms(pedal_times, pedal_values, high_onset + span * frac)
        for frac in (0.0, 0.25, 0.50, 0.75)
    ]


def build_performance_labels(refined_dir, perf_source, alignment_source, chords):
    midi = MidiFile(str(refined_dir / perf_source))
    notes = sorted_piano_notes(midi)
    alignment = np.load(refined_dir / alignment_source)
    perf_idx = alignment["perf_idx"].astype(int)
    interpolated_note = alignment["interpolated"].astype(bool) if "interpolated" in alignment else None
    tick_to_ms = _tick_to_ms_mapping(midi)
    controls = sorted_pedal_controls(midi)
    pedal_times = [_time_at_tick_ms(tick_to_ms, cc.time) for cc in controls]
    pedal_values = [cc.value for cc in controls]

    starts = [_time_at_tick_ms(tick_to_ms, notes[int(idx)].start) for idx in perf_idx]
    ends = [_time_at_tick_ms(tick_to_ms, notes[int(idx)].end) for idx in perf_idx]
    velocities = [int(notes[int(idx)].velocity) for idx in perf_idx]
    high_starts = [float(starts[chord["anchor_note_idx"]]) for chord in chords]

    shared = []
    pedal4 = []
    offsets = []
    interpolated = []
    previous_high_onset = None
    for chord_idx, chord in enumerate(chords):
        high_idx = chord["anchor_note_idx"]
        low_idx = chord["low_note_idx"]
        high_onset = float(starts[high_idx])
        low_onset = float(starts[low_idx])
        high_duration = float(ends[high_idx] - starts[high_idx])
        low_duration = float(ends[low_idx] - starts[low_idx])
        perf_ioi = 0.0 if previous_high_onset is None else max(high_onset - previous_high_onset, 0.0)
        previous_high_onset = high_onset

        shared.append(
            [
                round(perf_ioi, 6),
                round(high_duration, 6),
                int(velocities[high_idx]),
            ]
        )
        pedal4.append(pedal_samples(pedal_times, pedal_values, high_starts, chord_idx, high_onset))
        offsets.append(
            [
                round(low_onset - high_onset, 6),
                round(low_duration - high_duration, 6),
                int(velocities[low_idx] - velocities[high_idx]),
            ]
        )
        if interpolated_note is None:
            interpolated.append(0)
        else:
            interpolated.append(int(any(bool(interpolated_note[idx]) for idx in chord["note_indices"])))
    return shared, pedal4, offsets, interpolated


def tensor_sidecar(compact):
    score = compact["score"]
    sidecar = {
        "schema": compact["schema"],
        "meta": compact["meta"],
        "windows": torch.tensor(compact["windows"], dtype=torch.long),
        "score": {
            "pitch": score["pitch"],
            "pitch_multihot": torch.tensor(pitch_multihot(score["pitch"]), dtype=torch.uint8),
            "chord_size": torch.tensor(score["chord_size"], dtype=torch.long),
            "score_raw": torch.tensor(score["score_raw"], dtype=torch.float32),
            "score_offset_raw": torch.tensor(score["score_offset_raw"], dtype=torch.float32),
            "score_feature": torch.tensor(score["score_feature"], dtype=torch.float32),
            "has_score_feature": torch.tensor(score["has_score_feature"], dtype=torch.uint8),
            "note_count": int(score["note_count"]),
            "score_source": score["score_source"],
        },
        "performances": [],
        "performances_by_source": {},
    }
    for perf in compact["performances"]:
        perf_sidecar = {
            key: perf.get(key)
            for key in (
                "performance_id",
                "performance_source",
                "alignment_source",
                "split",
                "performance_dataset",
            )
        }
        perf_sidecar.update(
            {
                "label_shared_raw": torch.tensor(perf["label_shared_raw"], dtype=torch.float32),
                "label_pedal4_raw": torch.tensor(perf["label_pedal4_raw"], dtype=torch.float32),
                "label_offset_raw": torch.tensor(perf["label_offset_raw"], dtype=torch.float32),
                "interpolated": torch.tensor(perf["interpolated"], dtype=torch.uint8),
            }
        )
        sidecar["performances"].append(perf_sidecar)
        sidecar["performances_by_source"][perf_sidecar["performance_source"]] = perf_sidecar
    return sidecar


def build_one(args):
    (
        score_source,
        perf_rows,
        processed_dir,
        refined_dir,
        output_dir,
        window_size,
        stride,
        min_length,
        write_json,
        write_pt,
    ) = args
    score_payload = load_score_payload(processed_dir, score_source)
    chords = build_score_chords(score_payload)
    chord_count = len(chords)
    score = {
        "pitch": [chord["pitch"] for chord in chords],
        "score_raw": [chord["score_raw"] for chord in chords],
        "chord_size": [len(chord["pitch"]) for chord in chords],
        "score_offset_raw": [chord["score_offset_raw"] for chord in chords],
        "score_feature": [chord["score_feature"] for chord in chords],
        "has_score_feature": [chord["has_score_feature"] for chord in chords],
        "note_count": chord_count,
        "score_source": score_source,
    }

    performances = []
    for row in perf_rows:
        shared, pedal4, offsets, interpolated = build_performance_labels(
            refined_dir,
            row["performance_source"],
            row["alignment_source"],
            chords,
        )
        performances.append(
            {
                "performance_id": row.get("performance_id"),
                "performance_source": row["performance_source"],
                "alignment_source": row["alignment_source"],
                "split": row.get("split"),
                "performance_dataset": row.get("performance_dataset"),
                "label_shared_raw": shared,
                "label_pedal4_raw": pedal4,
                "label_offset_raw": offsets,
                "interpolated": interpolated,
            }
        )

    compact = {
        "schema": "pianocore_chord_work_compact_v1",
        "meta": {
            "score_source": score_source,
            "performance_count": len(performances),
            "chord_definition": "score IOI==0 + same score duration; require same staff only when candidate chord size>=3 and all notes have valid staff",
            "base_note": "highest_pitch",
            "pitch_format": "list[int] per chord, sorted ascending",
            "pitch_multihot_dim": PIANO_PITCH_COUNT,
            "piano_pitch_min": PIANO_PITCH_MIN,
            "score_raw_keys": ["ioi_ms_high_anchor", "duration_ms_high", "velocity_high"],
            "label_shared_raw_keys": ["ioi_ms_high_anchor", "duration_ms_high", "velocity_high"],
            "label_pedal4_raw_keys": ["pedal_0", "pedal_25", "pedal_50", "pedal_75"],
            "offset_raw_keys": [
                "onset_ms_low_minus_high",
                "duration_ms_low_minus_high",
                "velocity_low_minus_high",
            ],
            "window_size_chords": window_size,
            "stride_chords": stride,
            "min_window_chords": min_length,
        },
        "score": score,
        "performances": performances,
        "windows": sliding_windows(chord_count, window_size, stride, min_length),
    }

    rel_base = Path(score_source).with_suffix("")
    out_base = output_dir / rel_base.parent / f"{rel_base.name}.chord"
    out_base.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_base.with_suffix(".json")
    pt_path = out_base.with_suffix(".pt")
    if write_json:
        json_path.write_text(json.dumps(compact, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    if write_pt:
        torch.save(tensor_sidecar(compact), pt_path)
    return {
        "score_source": score_source,
        "json_path": str(json_path) if write_json else None,
        "pt_path": str(pt_path) if write_pt else None,
        "note_count": len(score_payload["pitch"]),
        "chord_count": chord_count,
        "performance_count": len(performances),
        "window_count": len(compact["windows"]),
    }


def load_subset(jsonl_path, metadata_path):
    grouped = defaultdict(list)
    wanted = set()
    split_by_pair = {}
    for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        score_source = row["score_source"]
        perf_source = row["performance_source"]
        wanted.add(perf_source)
        split_by_pair[(score_source, perf_source)] = row.get("split")
    metadata = pd.read_csv(
        metadata_path,
        usecols=[
            "performance_id",
            "split",
            "performance_dataset",
            "refined_score_midi_path",
            "refined_performance_midi_path",
            "refined_alignment_path",
        ],
    )
    metadata = metadata[metadata["refined_performance_midi_path"].isin(wanted)]
    for row in metadata.itertuples(index=False):
        score_source = row.refined_score_midi_path
        perf_source = row.refined_performance_midi_path
        grouped[score_source].append(
            {
                "performance_id": row.performance_id,
                "performance_source": perf_source,
                "alignment_source": row.refined_alignment_path,
                "split": split_by_pair.get((score_source, perf_source), row.split),
                "performance_dataset": row.performance_dataset,
            }
        )
    return grouped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", default=str(ROOT_DIR / "data/processed/sft/pt_sft_asap_all_processed_raw_repro.jsonl"))
    parser.add_argument("--metadata", default="/home/sy/EPR/PianoCoRe/metadata.csv")
    parser.add_argument("--processed-dir", default="/home/sy/EPR/PianoCoRe/processed")
    parser.add_argument("--refined-dir", default="/home/sy/EPR/PianoCoRe/refined")
    parser.add_argument("--output-dir", default=str(ROOT_DIR / "data/processed/chord_asap"))
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=448)
    parser.add_argument("--min-length", type=int, default=64)
    parser.add_argument("--workers", type=int, default=36)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-json", action="store_true")
    parser.add_argument("--no-pt", action="store_true")
    parser.add_argument("--summary", default=str(ROOT_DIR / "results/analysis/chord_asap_build_summary.json"))
    args = parser.parse_args()

    grouped = load_subset(args.jsonl, args.metadata)
    items = sorted(grouped.items())
    if args.limit is not None:
        items = items[: args.limit]
    output_dir = Path(args.output_dir)
    tasks = [
        (
            score_source,
            perf_rows,
            Path(args.processed_dir),
            Path(args.refined_dir),
            output_dir,
            args.window_size,
            args.stride,
            args.min_length,
            not args.no_json,
            not args.no_pt,
        )
        for score_source, perf_rows in items
    ]
    print(
        json.dumps(
            {
                "event": "chord_asap_build_start",
                "works": len(tasks),
                "pairs": sum(len(rows) for _, rows in items),
                "output_dir": str(output_dir),
                "window_size": args.window_size,
                "stride": args.stride,
                "workers": args.workers,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )

    rows = []
    errors = []
    if args.workers <= 1:
        for idx, task in enumerate(tasks, 1):
            try:
                rows.append(build_one(task))
            except Exception as exc:  # noqa: BLE001
                errors.append({"score_source": task[0], "error": repr(exc)})
            if idx % 10 == 0 or idx == len(tasks):
                print(json.dumps({"event": "chord_asap_build_progress", "done": idx, "works": len(tasks)}), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            future_to_score = {pool.submit(build_one, task): task[0] for task in tasks}
            for idx, future in enumerate(as_completed(future_to_score), 1):
                try:
                    rows.append(future.result())
                except Exception as exc:  # noqa: BLE001
                    errors.append({"score_source": future_to_score[future], "error": repr(exc)})
                if idx % 10 == 0 or idx == len(tasks):
                    print(json.dumps({"event": "chord_asap_build_progress", "done": idx, "works": len(tasks)}), flush=True)

    summary = {
        "event": "chord_asap_build_done",
        "works_requested": len(tasks),
        "works_written": len(rows),
        "errors": errors,
        "output_dir": str(output_dir),
        "json_count": sum(1 for row in rows if row.get("json_path")),
        "pt_count": sum(1 for row in rows if row.get("pt_path")),
        "note_count": sum(row["note_count"] for row in rows),
        "chord_count": sum(row["chord_count"] for row in rows),
        "performance_count": sum(row["performance_count"] for row in rows),
        "window_count": sum(row["window_count"] for row in rows),
        "rows": sorted(rows, key=lambda row: row["score_source"]),
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False, sort_keys=True), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
