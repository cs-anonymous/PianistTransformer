import argparse
import json
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from miditoolkit import MidiFile
from scipy.stats import wasserstein_distance
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

LOG_WASS_SCALE = 50.0
LOG_WASS_MAX_TIME_MS = 5000.0


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


def log_time_values(values, scale=LOG_WASS_SCALE, max_time_ms=LOG_WASS_MAX_TIME_MS):
    values = np.asarray(values, dtype=np.float64)
    values = np.clip(values, 0.0, float(max_time_ms))
    return np.log1p(values / float(scale))


def compute_pair_metrics_from_arrays(pred, gt):
    pointwise = {}
    distro = {}

    feature_names = ["ioi", "duration", "velocity", "pedal_0", "pedal_25", "pedal_50", "pedal_75"]
    for name in feature_names:
        usable = min(len(pred[name]), len(gt[name]))
        pred_slice = pred[name][:usable]
        gt_slice = gt[name][:usable]
        pointwise[name] = mae(pred_slice, gt_slice)
        distro[name] = float(wasserstein_distance(pred_slice, gt_slice))
        if name in {"ioi", "duration"}:
            distro[f"{name}_log50"] = float(
                wasserstein_distance(
                    log_time_values(pred_slice),
                    log_time_values(gt_slice),
                )
            )

    pedal_mae = float(np.mean([pointwise[k] for k in feature_names[3:]]))
    pedal_wass = float(np.mean([distro[k] for k in feature_names[3:]]))

    return {
        "ioi_mae": pointwise["ioi"],
        "ioi_wass": distro["ioi"],
        "ioi_log50_wass": distro["ioi_log50"],
        "duration_mae": pointwise["duration"],
        "duration_wass": distro["duration"],
        "duration_log50_wass": distro["duration_log50"],
        "velocity_mae": pointwise["velocity"],
        "velocity_wass": distro["velocity"],
        "pedal_mae": pedal_mae,
        "pedal_wass": pedal_wass,
        "note_count_pred": int(len(pred["ioi"])),
        "note_count_gt": int(len(gt["ioi"])),
        "note_count_used": int(min(len(pred["ioi"]), len(gt["ioi"]))),
    }


def compute_pair_metrics(pred_path: Path, gt_path: Path):
    pred = extract_note_arrays(pred_path)
    gt = extract_note_arrays(gt_path)
    return compute_pair_metrics_from_arrays(pred, gt)


def aggregate_metrics(rows):
    keys = [
        "ioi_mae",
        "ioi_wass",
        "ioi_log50_wass",
        "duration_mae",
        "duration_wass",
        "duration_log50_wass",
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


def compute_pair_row(item):
    pair_metrics = compute_pair_metrics(Path(item["pred"]), Path(item["gt"]))
    return {
        "gt": item["gt"],
        "pred": item["pred"],
        **pair_metrics,
    }


def extract_feature_row(path: str):
    resolved = str(Path(path).resolve())
    return resolved, extract_note_arrays(Path(resolved))


def normalize_pair_paths(item):
    return {
        "pred": str(Path(item["pred"]).resolve()),
        "gt": str(Path(item["gt"]).resolve()),
    }


def build_feature_cache(evaluate_list, num_workers):
    unique_paths = sorted(
        {item["pred"] for item in evaluate_list} | {item["gt"] for item in evaluate_list}
    )
    print(
        f"Extracting note features for {len(unique_paths)} unique MIDI files "
        f"with {num_workers} worker(s)",
        flush=True,
    )
    if num_workers > 1:
        ctx = get_context("spawn")
        with ctx.Pool(processes=num_workers) as pool:
            rows = list(
                tqdm(
                    pool.imap(extract_feature_row, unique_paths, chunksize=8),
                    total=len(unique_paths),
                    desc="MIDI feature cache",
                )
            )
    else:
        rows = [
            extract_feature_row(path)
            for path in tqdm(unique_paths, total=len(unique_paths), desc="MIDI feature cache")
        ]
    return dict(rows)


def compute_pair_row_from_cache(item, feature_cache):
    pair_metrics = compute_pair_metrics_from_arrays(
        feature_cache[item["pred"]],
        feature_cache[item["gt"]],
    )
    return {
        "gt": item["gt"],
        "pred": item["pred"],
        **pair_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute MAE/Wasserstein from saved MIDI predictions.")
    parser.add_argument("--evaluate-list", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--subset-gt-list", type=Path, default=None,
                        help="Optional evaluate_list.json whose gt paths define the subset to keep.")
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument(
        "--no-feature-cache",
        action="store_true",
        help="Disable two-stage MIDI feature caching and compute each pair by reading MIDI files directly.",
    )
    args = parser.parse_args()
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")

    evaluate_list = load_evaluate_list(args.evaluate_list)
    subset_reference = None
    if args.subset_gt_list is not None:
        subset_reference = load_evaluate_list(args.subset_gt_list)
        evaluate_list = filter_to_gt_subset(evaluate_list, [item["gt"] for item in subset_reference])
    evaluate_list = [normalize_pair_paths(item) for item in evaluate_list]

    print(
        f"Computing MIDI MAE/Wasserstein for {len(evaluate_list)} pairs "
        f"with {args.num_workers} worker(s)",
        flush=True,
    )
    if not args.no_feature_cache:
        feature_cache = build_feature_cache(evaluate_list, args.num_workers)
        pair_rows = [
            compute_pair_row_from_cache(item, feature_cache)
            for item in tqdm(evaluate_list, total=len(evaluate_list), desc="MIDI pair metrics")
        ]
    elif args.num_workers > 1:
        ctx = get_context("spawn")
        with ctx.Pool(processes=args.num_workers) as pool:
            pair_rows = list(
                tqdm(
                    pool.imap(compute_pair_row, evaluate_list, chunksize=8),
                    total=len(evaluate_list),
                    desc="MIDI metrics",
                )
            )
    else:
        pair_rows = [
            compute_pair_row(item)
            for item in tqdm(evaluate_list, total=len(evaluate_list), desc="MIDI metrics")
        ]

    output = {
        "evaluate_list": str(args.evaluate_list),
        "subset_gt_list": str(args.subset_gt_list) if args.subset_gt_list is not None else None,
        "num_workers": args.num_workers,
        "num_pairs": len(pair_rows),
        "aggregate": aggregate_metrics(pair_rows),
        "pairs": pair_rows,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2))
    print(json.dumps(output["aggregate"], indent=2))


if __name__ == "__main__":
    main()
