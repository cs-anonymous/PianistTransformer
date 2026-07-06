import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_process.work_manifest import build_work_manifest
from src.utils.inr_midi import note_features_to_midi


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render listenable same-score-window oracle MIDI baselines for ASAP test scores."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--performance-dataset", type=str, default="ASAP")
    parser.add_argument(
        "--donor-pool",
        choices=["test_asap", "all_asap", "all_processed"],
        default="all_processed",
        help="Performance pool used as donors. The target performance itself is always excluded.",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="nearest_window,full_donor,marginal_sample",
        help="Comma-separated modes: nearest_window, full_donor, random_window, marginal_sample.",
    )
    parser.add_argument(
        "--targets-per-score",
        type=int,
        default=1,
        help="Number of ASAP target performances rendered per score. Use 0 for all targets.",
    )
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--max-works", type=int, default=None)
    parser.add_argument("--write-target", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--copy-source-target", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def safe_name(value, max_len=150):
    text = str(value).replace("\\", "/")
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    stem = "__".join(part for part in Path(text).with_suffix("").parts if part)
    keep = []
    for char in stem:
        keep.append(char if char.isalnum() or char in {"-", "_", "."} else "_")
    compact = "".join(keep).strip("_")
    if len(compact) > max_len:
        compact = compact[-max_len:]
    return f"{compact}__{digest}"


def stable_seed(base_seed, *parts):
    payload = "::".join([str(base_seed), *[str(part) for part in parts]])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def perf_continuous(perf, start=None, end=None):
    shared = np.asarray(perf.get("label_shared_raw", []), dtype=np.float64)
    pedal = np.asarray(perf.get("label_pedal4_raw", []), dtype=np.float64)
    if shared.ndim != 2 or shared.shape[1] < 3:
        raise ValueError(f"Bad label_shared_raw for {perf.get('performance_source')}")
    if pedal.ndim != 2 or pedal.shape[1] < 4:
        raise ValueError(f"Bad label_pedal4_raw for {perf.get('performance_source')}")
    if len(shared) != len(pedal):
        raise ValueError(
            f"label_shared_raw/label_pedal4_raw length mismatch for {perf.get('performance_source')}: "
            f"{len(shared)} vs {len(pedal)}"
        )
    rows = np.concatenate([shared[:, :3], pedal[:, :4]], axis=1)
    if start is None and end is None:
        return rows
    return rows[int(start) : int(end)]


def safe_wass(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("inf")
    return float(wasserstein_distance(a, b))


def log50(values, max_time_ms=5000.0, scale=50.0):
    values = np.clip(np.asarray(values, dtype=np.float64), 0.0, float(max_time_ms))
    return np.log1p(values / float(scale)) / math.log1p(float(max_time_ms) / float(scale))


def window_distance(target_rows, donor_rows):
    if len(target_rows) == 0 or len(donor_rows) == 0:
        return float("inf")
    parts = [
        safe_wass(log50(target_rows[:, 0]), log50(donor_rows[:, 0])),
        safe_wass(log50(target_rows[:, 1]), log50(donor_rows[:, 1])),
        safe_wass(target_rows[:, 2] / 127.0, donor_rows[:, 2] / 127.0),
        safe_wass(target_rows[:, 3:7].reshape(-1) / 127.0, donor_rows[:, 3:7].reshape(-1) / 127.0),
    ]
    return float(np.mean(parts))


def assigned_window_for_notes(note_count, windows):
    assignments = np.full(int(note_count), -1, dtype=np.int64)
    best_margin = np.full(int(note_count), -1, dtype=np.int64)
    for window_idx, (start, end) in enumerate(windows):
        start = int(start)
        end = int(end)
        for note_idx in range(start, end):
            margin = min(note_idx - start, end - 1 - note_idx)
            if margin > best_margin[note_idx]:
                best_margin[note_idx] = margin
                assignments[note_idx] = window_idx
    return assignments


def donor_pool_for_mode(pool_name, all_perfs, target_perfs, performance_dataset):
    if pool_name == "test_asap":
        return list(target_perfs)
    if pool_name == "all_asap":
        return [
            perf
            for perf in all_perfs
            if str(perf.get("performance_dataset") or "") == str(performance_dataset)
        ]
    if pool_name == "all_processed":
        return list(all_perfs)
    raise ValueError(f"Unknown donor pool: {pool_name}")


def render_rows(pitch, rows, output_path):
    midi = note_features_to_midi(
        pitch,
        rows.tolist() if isinstance(rows, np.ndarray) else rows,
        target_ticks_per_beat=500,
        target_tempo=120,
        normalized=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi.dump(str(output_path))


def select_targets(target_perfs, targets_per_score):
    if targets_per_score == 0:
        return list(target_perfs)
    return list(target_perfs[: max(0, targets_per_score)])


def choose_nearest_full_donor(target_rows, donors):
    best = None
    best_distance = float("inf")
    for donor in donors:
        donor_rows = perf_continuous(donor)
        distance = window_distance(target_rows, donor_rows)
        if distance < best_distance:
            best_distance = distance
            best = donor
    return best, best_distance


def build_window_rows(mode, target, donors, windows, assignments, rng):
    target_rows = perf_continuous(target)
    output = np.zeros_like(target_rows)
    donor_choices = []
    donor_rows_cache = {donor.get("performance_source"): perf_continuous(donor) for donor in donors}

    for window_idx, (start, end) in enumerate(windows):
        assigned = np.where(assignments == window_idx)[0]
        if assigned.size == 0:
            continue
        if mode == "random_window":
            donor = rng.choice(donors)
            donor_source = donor.get("performance_source")
            output[assigned] = donor_rows_cache[donor_source][assigned]
            donor_choices.append(
                {
                    "window_idx": int(window_idx),
                    "start": int(start),
                    "end": int(end),
                    "donor": donor_source,
                    "distance": None,
                }
            )
            continue

        if mode == "nearest_window":
            target_window_rows = target_rows[int(start) : int(end)]
            best = None
            best_distance = float("inf")
            for donor in donors:
                donor_source = donor.get("performance_source")
                distance = window_distance(target_window_rows, donor_rows_cache[donor_source][int(start) : int(end)])
                if distance < best_distance:
                    best = donor
                    best_distance = distance
            donor_source = best.get("performance_source")
            output[assigned] = donor_rows_cache[donor_source][assigned]
            donor_choices.append(
                {
                    "window_idx": int(window_idx),
                    "start": int(start),
                    "end": int(end),
                    "donor": donor_source,
                    "distance": float(best_distance),
                }
            )
            continue

        if mode == "marginal_sample":
            donor_window_rows = np.concatenate(
                [rows[int(start) : int(end)] for rows in donor_rows_cache.values()],
                axis=0,
            )
            sampled = rng.integers(0, len(donor_window_rows), size=int(assigned.size))
            output[assigned] = donor_window_rows[sampled]
            donor_choices.append(
                {
                    "window_idx": int(window_idx),
                    "start": int(start),
                    "end": int(end),
                    "donor": "empirical_marginal",
                    "distance": None,
                }
            )
            continue

        raise ValueError(f"Unsupported window mode: {mode}")

    missing = np.where(assignments < 0)[0]
    if missing.size:
        output[missing] = target_rows[missing]
    return output, donor_choices


def main():
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    midi_dir = args.output_dir / "midis"
    choices_dir = args.output_dir / "donor_choices"
    midi_dir.mkdir(parents=True, exist_ok=True)
    choices_dir.mkdir(parents=True, exist_ok=True)

    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    allowed_modes = {"nearest_window", "full_donor", "random_window", "marginal_sample"}
    unknown_modes = sorted(set(modes) - allowed_modes)
    if unknown_modes:
        raise ValueError(f"Unknown modes: {unknown_modes}")

    manifest = build_work_manifest(
        metadata_path=config["metadata_path"],
        refined_dir=config["refined_dir"],
        split=args.split,
        block_notes=config["block_notes"],
        overlap_ratio=config["overlap_ratio"],
        min_notes=config["min_notes"],
        max_works=args.max_works,
        skip_work_paths=config.get("skip_work_paths"),
        performance_dataset=args.performance_dataset,
    )

    rows = []
    skipped = []
    refined_dir = Path(config["refined_dir"])

    for item in tqdm(manifest, desc="render oracle MIDIs"):
        work_path = Path(item["path"])
        work = json.loads(work_path.read_text(encoding="utf-8"))
        pitch = work["score"]["pitch"]
        all_perfs = [
            perf
            for perf in work.get("performances", [])
            if perf.get("label_shared_raw") is not None and perf.get("label_pedal4_raw") is not None
        ]
        by_source = {perf.get("performance_source"): perf for perf in all_perfs}
        target_perfs = [
            by_source[source]
            for source in item.get("selected_performance_sources", [])
            if source in by_source
        ]
        target_perfs = select_targets(target_perfs, args.targets_per_score)
        if not target_perfs:
            skipped.append({"score_source": item["score_source"], "reason": "no_target_performances"})
            continue

        pool = donor_pool_for_mode(args.donor_pool, all_perfs, target_perfs, args.performance_dataset)
        assignments = assigned_window_for_notes(len(pitch), item["windows"])

        for target_idx, target in enumerate(target_perfs):
            target_source = target.get("performance_source")
            donors = [perf for perf in pool if perf.get("performance_source") != target_source]
            if not donors:
                skipped.append(
                    {
                        "score_source": item["score_source"],
                        "target_performance_source": target_source,
                        "reason": "no_donors",
                    }
                )
                continue

            score_name = safe_name(item["score_source"])
            target_name = safe_name(target_source, max_len=80)
            prefix = f"{score_name}__target_{target_idx:02d}"

            if args.write_target:
                target_rows = perf_continuous(target)
                target_out = midi_dir / "target_rebuild" / f"{prefix}__target.mid"
                render_rows(pitch, target_rows, target_out)
                rows.append(
                    {
                        "mode": "target_rebuild",
                        "score_source": item["score_source"],
                        "target_performance_source": target_source,
                        "donor_pool": args.donor_pool,
                        "sample_idx": "",
                        "midi_path": str(target_out.resolve()),
                        "donor_choices_path": "",
                    }
                )
                if args.copy_source_target:
                    src = refined_dir / target_source
                    if src.exists():
                        copy_out = midi_dir / "target_source" / f"{prefix}__target_source.mid"
                        copy_out.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, copy_out)
                        rows.append(
                            {
                                "mode": "target_source",
                                "score_source": item["score_source"],
                                "target_performance_source": target_source,
                                "donor_pool": args.donor_pool,
                                "sample_idx": "",
                                "midi_path": str(copy_out.resolve()),
                                "donor_choices_path": "",
                            }
                        )

            for mode in modes:
                for sample_idx in range(max(1, args.num_samples)):
                    rng = np.random.default_rng(
                        stable_seed(args.seed, item["score_source"], target_source, mode, sample_idx)
                    )
                    py_rng = random.Random(
                        stable_seed(args.seed, item["score_source"], target_source, mode, sample_idx, "py")
                    )
                    if mode == "full_donor":
                        donor, distance = choose_nearest_full_donor(perf_continuous(target), donors)
                        pred_rows = perf_continuous(donor)
                        choices = [
                            {
                                "window_idx": "full",
                                "start": 0,
                                "end": len(pred_rows),
                                "donor": donor.get("performance_source"),
                                "distance": float(distance),
                            }
                        ]
                    else:
                        pred_rows, choices = build_window_rows(
                            mode,
                            target,
                            donors,
                            item["windows"],
                            assignments,
                            rng if mode == "marginal_sample" else py_rng,
                        )

                    out = midi_dir / mode / f"{prefix}__{mode}__sample_{sample_idx:03d}.mid"
                    render_rows(pitch, pred_rows, out)
                    choices_path = choices_dir / mode / f"{prefix}__{mode}__sample_{sample_idx:03d}.json"
                    choices_path.parent.mkdir(parents=True, exist_ok=True)
                    choices_path.write_text(json.dumps(choices, indent=2, ensure_ascii=False), encoding="utf-8")
                    rows.append(
                        {
                            "mode": mode,
                            "score_source": item["score_source"],
                            "target_performance_source": target_source,
                            "target_name": target_name,
                            "donor_pool": args.donor_pool,
                            "sample_idx": int(sample_idx),
                            "midi_path": str(out.resolve()),
                            "donor_choices_path": str(choices_path.resolve()),
                        }
                    )

    manifest_csv = args.output_dir / "manifest.csv"
    fieldnames = [
        "mode",
        "score_source",
        "target_performance_source",
        "target_name",
        "donor_pool",
        "sample_idx",
        "midi_path",
        "donor_choices_path",
    ]
    with manifest_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    summary = {
        "config": str(args.config),
        "split": args.split,
        "performance_dataset": args.performance_dataset,
        "donor_pool": args.donor_pool,
        "modes": modes,
        "num_manifest_scores": int(len(manifest)),
        "targets_per_score": int(args.targets_per_score),
        "num_samples": int(args.num_samples),
        "num_midi_files": int(len(rows)),
        "manifest_csv": str(manifest_csv.resolve()),
        "midi_dir": str(midi_dir.resolve()),
        "skipped": skipped,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
