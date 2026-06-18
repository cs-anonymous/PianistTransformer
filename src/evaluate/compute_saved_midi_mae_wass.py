import argparse
import json
import sys
from pathlib import Path

import numpy as np
from miditoolkit import MidiFile
from scipy.stats import wasserstein_distance

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def sorted_piano_notes(midi_obj):
    notes = []
    for instrument in midi_obj.instruments:
        if not instrument.is_drum:
            notes.extend(instrument.notes)
    return sorted(notes, key=lambda note: (note.start, note.pitch, note.end, note.velocity))


def sorted_pedal_controls(midi_obj):
    controls = []
    for instrument in midi_obj.instruments:
        if not instrument.is_drum:
            controls.extend(cc for cc in instrument.control_changes if cc.number == 64)
    return sorted(controls, key=lambda cc: (cc.time, cc.value))


def _tick_to_ms_mapping(midi_obj):
    tick_to_time = midi_obj.get_tick_to_time_mapping()
    return [time_sec * 1000.0 for time_sec in tick_to_time]


def _time_at_tick_ms(tick_to_ms, tick):
    if tick < len(tick_to_ms):
        return tick_to_ms[tick]
    if not tick_to_ms:
        return 0.0
    return tick_to_ms[-1]


def _cc_value_at_ms(cc_times_ms, cc_values, query_ms):
    import bisect

    idx = bisect.bisect_right(cc_times_ms, query_ms)
    if idx == 0:
        return 0
    return cc_values[idx - 1]


def midi_to_note_features(midi_obj, normalize=False, force_monotonic_starts=False, max_time_ms=10000.0):
    notes = sorted_piano_notes(midi_obj)
    tick_to_ms = _tick_to_ms_mapping(midi_obj)
    pedal_controls = sorted_pedal_controls(midi_obj)
    pedal_times_ms = [_time_at_tick_ms(tick_to_ms, cc.time) for cc in pedal_controls]
    pedal_values = [cc.value for cc in pedal_controls]

    raw_starts_ms = [_time_at_tick_ms(tick_to_ms, note.start) for note in notes]
    raw_ends_ms = [_time_at_tick_ms(tick_to_ms, note.end) for note in notes]
    starts_ms = sorted(raw_starts_ms) if force_monotonic_starts else raw_starts_ms

    pitches = []
    continuous = []
    last_start_ms = 0.0

    for idx, note in enumerate(notes):
        start_ms = starts_ms[idx]
        next_start_ms = starts_ms[idx + 1] if idx + 1 < len(starts_ms) else start_ms + 4990.0
        next_ioi_ms = max(next_start_ms - start_ms, 0.0)

        ioi_ms = max(start_ms - last_start_ms, 0.0)
        duration_ms = max(raw_ends_ms[idx] - raw_starts_ms[idx], 0.0)
        last_start_ms = start_ms

        pedal_samples = [
            _cc_value_at_ms(pedal_times_ms, pedal_values, start_ms),
            _cc_value_at_ms(pedal_times_ms, pedal_values, start_ms + next_ioi_ms * 0.25),
            _cc_value_at_ms(pedal_times_ms, pedal_values, start_ms + next_ioi_ms * 0.50),
            _cc_value_at_ms(pedal_times_ms, pedal_values, start_ms + next_ioi_ms * 0.75),
        ]

        if normalize:
            raise ValueError("This cleanup script expects raw, non-normalized features only.")

        pitches.append(int(note.pitch))
        continuous.append(
            [
                ioi_ms,
                duration_ms,
                min(max(float(note.velocity) / 127.0, 0.0), 1.0),
                *[min(max(float(value) / 127.0, 0.0), 1.0) for value in pedal_samples],
            ]
        )

    return {"pitch": pitches, "continuous": continuous}


def load_evaluate_list(path: Path):
    return json.loads(path.read_text())


def extract_note_arrays(midi_path: Path):
    midi_obj = MidiFile(str(midi_path))
    payload = midi_to_note_features(
        midi_obj,
        max_time_ms=10000.0,
        normalize=False,
        force_monotonic_starts=False,
    )
    cont = np.asarray(payload["continuous"], dtype=np.float64)
    if cont.ndim != 2 or cont.shape[1] != 7:
        raise ValueError(f"Unexpected feature shape for {midi_path}: {cont.shape}")
    return {
        "ioi": cont[:, 0],
        "duration": cont[:, 1],
        "velocity": cont[:, 2] * 127.0,
        "pedal_0": cont[:, 3] * 127.0,
        "pedal_25": cont[:, 4] * 127.0,
        "pedal_50": cont[:, 5] * 127.0,
        "pedal_75": cont[:, 6] * 127.0,
    }


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target))) if len(pred) else float("nan")


def compute_pair_metrics(pred_path: Path, gt_path: Path):
    pred = extract_note_arrays(pred_path)
    gt = extract_note_arrays(gt_path)

    pointwise = {}
    distro = {}

    feature_names = ["ioi", "duration", "velocity", "pedal_0", "pedal_25", "pedal_50", "pedal_75"]
    for name in feature_names:
        usable = min(len(pred[name]), len(gt[name]))
        pred_slice = pred[name][:usable]
        gt_slice = gt[name][:usable]
        pointwise[name] = mae(pred_slice, gt_slice)
        distro[name] = float(wasserstein_distance(pred_slice, gt_slice))

    pedal_mae = float(np.mean([pointwise[k] for k in feature_names[3:]]))
    pedal_wass = float(np.mean([distro[k] for k in feature_names[3:]]))

    return {
        "ioi_mae": pointwise["ioi"],
        "ioi_wass": distro["ioi"],
        "duration_mae": pointwise["duration"],
        "duration_wass": distro["duration"],
        "velocity_mae": pointwise["velocity"],
        "velocity_wass": distro["velocity"],
        "pedal_mae": pedal_mae,
        "pedal_wass": pedal_wass,
        "note_count_pred": int(len(pred["ioi"])),
        "note_count_gt": int(len(gt["ioi"])),
        "note_count_used": int(min(len(pred["ioi"]), len(gt["ioi"]))),
    }


def aggregate_metrics(rows):
    keys = [
        "ioi_mae",
        "ioi_wass",
        "duration_mae",
        "duration_wass",
        "velocity_mae",
        "velocity_wass",
        "pedal_mae",
        "pedal_wass",
        "note_count_pred",
        "note_count_gt",
        "note_count_used",
    ]
    return {k: float(np.mean([row[k] for row in rows])) for k in keys}


def filter_to_gt_subset(evaluate_list, allowed_gt_paths):
    allowed = {str(Path(p).resolve()) for p in allowed_gt_paths}
    return [item for item in evaluate_list if str(Path(item["gt"]).resolve()) in allowed]


def main():
    parser = argparse.ArgumentParser(description="Compute MAE/Wasserstein from saved MIDI predictions.")
    parser.add_argument("--evaluate-list", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--subset-gt-list", type=Path, default=None,
                        help="Optional evaluate_list.json whose gt paths define the subset to keep.")
    args = parser.parse_args()

    evaluate_list = load_evaluate_list(args.evaluate_list)
    subset_reference = None
    if args.subset_gt_list is not None:
        subset_reference = load_evaluate_list(args.subset_gt_list)
        evaluate_list = filter_to_gt_subset(evaluate_list, [item["gt"] for item in subset_reference])

    pair_rows = []
    for item in evaluate_list:
        pair_metrics = compute_pair_metrics(Path(item["pred"]), Path(item["gt"]))
        pair_rows.append({
            "gt": item["gt"],
            "pred": item["pred"],
            **pair_metrics,
        })

    output = {
        "evaluate_list": str(args.evaluate_list),
        "subset_gt_list": str(args.subset_gt_list) if args.subset_gt_list is not None else None,
        "num_pairs": len(pair_rows),
        "aggregate": aggregate_metrics(pair_rows),
        "pairs": pair_rows,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2))
    print(json.dumps(output["aggregate"], indent=2))


if __name__ == "__main__":
    main()
