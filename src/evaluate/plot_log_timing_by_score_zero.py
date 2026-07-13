import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import sys
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from miditoolkit import MidiFile

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.inr_midi import midi_to_note_features, sorted_piano_notes


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot ln timing distributions grouped by score IOI zero/non-zero."
    )
    parser.add_argument("--metadata", type=Path, default=Path("../PianoCoRe/metadata.csv"))
    parser.add_argument("--midi-root", type=Path, default=Path("../PianoCoRe/refined"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--performance-dataset", type=str, default="ASAP")
    parser.add_argument("--split", type=str, default=None, help="Optional split filter, e.g. train or test.")
    parser.add_argument("--score-source-list", type=Path, default=None)
    parser.add_argument("--floor-ms", type=float, default=1.0)
    parser.add_argument("--tau-chord-ms", type=float, default=6.0)
    parser.add_argument("--zero-eps-ms", type=float, default=1e-9)
    parser.add_argument("--bins", type=int, default=160)
    parser.add_argument("--normalize-perf-to-score-onset-span", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def load_score_source_filter(path):
    if path is None:
        return None
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            values.append(line)
    return set(values)


@lru_cache(maxsize=4096)
def load_score_features(path_text):
    midi = MidiFile(path_text)
    notes = sorted_piano_notes(midi)
    return midi_to_note_features(midi, notes=notes, normalize=False)


def load_aligned_perf_features(perf_path, align_path, note_count):
    perf_midi = MidiFile(str(perf_path))
    alignment = np.load(align_path)
    if "perf_idx" not in alignment:
        raise ValueError(f"Missing perf_idx in {align_path}")
    perf_idx = alignment["perf_idx"].astype(int)
    if len(perf_idx) != note_count:
        raise ValueError(f"Alignment length mismatch in {align_path}: {len(perf_idx)} != {note_count}")
    perf_notes_sorted = sorted_piano_notes(perf_midi)
    if len(perf_idx):
        if int(perf_idx.min()) < 0 or int(perf_idx.max()) >= len(perf_notes_sorted):
            raise ValueError(f"perf_idx out of range in {align_path}")
    perf_notes = [perf_notes_sorted[int(index)] for index in perf_idx]
    return midi_to_note_features(
        perf_midi,
        notes=perf_notes,
        normalize=False,
        force_monotonic_starts=True,
    )


def load_pair_job(args):
    score_path, perf_path, align_path, score_rel, perf_rel, normalize = args
    try:
        score_features = load_score_features(str(score_path))
        perf_features = load_aligned_perf_features(perf_path, align_path, len(score_features["pitch"]))
        if perf_features["pitch"] != score_features["pitch"]:
            raise ValueError("aligned_pitch_mismatch")
        score_raw = np.asarray(score_features["continuous"], dtype=np.float64)
        perf_raw = np.asarray(perf_features["continuous"], dtype=np.float64)
        scale = None
        if normalize:
            score_span = float(np.sum(score_raw[1:, 0]))
            perf_span = float(np.sum(perf_raw[1:, 0]))
            if not np.isfinite(score_span) or score_span <= 0.0 or not np.isfinite(perf_span) or perf_span <= 0.0:
                raise ValueError(f"invalid_onset_span(score={score_span}, perf={perf_span})")
            scale = score_span / perf_span
            perf_raw = perf_raw.copy()
            perf_raw[:, 0] *= scale
            perf_raw[:, 1] *= scale
        return score_raw, perf_raw, scale, None
    except Exception as exc:  # noqa: BLE001
        return None, None, None, {"score": score_rel, "performance": perf_rel, "reason": repr(exc)}


def ln_floor(values, floor_ms):
    values = np.asarray(values, dtype=np.float64)
    return np.log(np.maximum(values, float(floor_ms)))


def finite_quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"count": 0}
    quantiles = np.quantile(values, [0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999])
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p0.1": float(quantiles[0]),
        "p1": float(quantiles[1]),
        "p5": float(quantiles[2]),
        "p50": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "p99": float(quantiles[5]),
        "p99.9": float(quantiles[6]),
        "max": float(np.max(values)),
    }


def append_summary(rows, group_name, source_name, feature_name, raw_values, log_values, zero_eps):
    raw = np.asarray(raw_values, dtype=np.float64)
    log = np.asarray(log_values, dtype=np.float64)
    stats = finite_quantiles(log)
    stats.update(
        {
            "group": group_name,
            "source": source_name,
            "feature": feature_name,
            "raw_zero_count": int(np.sum(raw <= float(zero_eps))),
            "raw_zero_rate": float(np.mean(raw <= float(zero_eps))) if len(raw) else float("nan"),
            "raw_positive_count": int(np.sum(raw > float(zero_eps))),
        }
    )
    rows.append(stats)


def append_dev_summary(rows, group_name, feature_name, values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    stats = finite_quantiles(finite)
    stats.update(
        {
            "group": group_name,
            "source": "dev",
            "feature": feature_name,
            "raw_zero_count": int(np.sum(np.isclose(finite, 0.0))) if len(finite) else 0,
            "raw_zero_rate": float(np.mean(np.isclose(finite, 0.0))) if len(finite) else float("nan"),
            "raw_positive_count": int(np.sum(finite > 0.0)) if len(finite) else 0,
        }
    )
    rows.append(stats)


def histogram_range(arrays):
    pooled = np.concatenate([np.asarray(values, dtype=np.float64) for values in arrays if len(values)])
    pooled = pooled[np.isfinite(pooled)]
    if len(pooled) == 0:
        return 0.0, 1.0
    lo, hi = np.quantile(pooled, [0.001, 0.999])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(np.min(pooled)), float(np.max(pooled))
    pad = max((hi - lo) * 0.05, 1e-3)
    return float(lo - pad), float(hi + pad)


def plot_grid(groups, output_path, bins):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    group_order = [("score_ioi_eq_0", "score IOI = 0"), ("score_ioi_gt_0", "score IOI > 0")]
    columns = [
        ("score_ioi_log", "score ln IOI"),
        ("perf_ioi_log", "perf ln IOI"),
        ("score_duration_log", "score ln duration"),
        ("perf_duration_log", "perf ln duration"),
    ]
    colors = {
        "score_ioi_log": "#3b82f6",
        "perf_ioi_log": "#ef4444",
        "score_duration_log": "#0f766e",
        "perf_duration_log": "#a855f7",
    }
    for row_idx, (group_key, group_title) in enumerate(group_order):
        for col_idx, (key, title) in enumerate(columns):
            ax = axes[row_idx][col_idx]
            values = groups[group_key][key]
            lo, hi = histogram_range([values])
            ax.hist(values, bins=bins, range=(lo, hi), density=True, alpha=0.82, color=colors[key])
            ax.set_title(f"{group_title}: {title}")
            ax.set_xlabel("ln(max(ms, 1))")
            ax.set_ylabel("density")
            ax.grid(True, alpha=0.2)
    fig.suptitle("ASAP aligned timing distributions by score IOI group", fontsize=15)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_overlay(groups, output_path, feature, bins):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    group_order = [("score_ioi_eq_0", "score IOI = 0"), ("score_ioi_gt_0", "score IOI > 0")]
    for ax, (group_key, group_title) in zip(axes, group_order):
        if feature == "ioi":
            score_values = groups[group_key]["score_ioi_log"]
            perf_values = groups[group_key]["perf_ioi_log"]
            label = "ln(max(IOI ms, 1))"
        else:
            score_values = groups[group_key]["score_duration_log"]
            perf_values = groups[group_key]["perf_duration_log"]
            label = "ln(max(duration ms, 1))"
        lo, hi = histogram_range([score_values, perf_values])
        ax.hist(score_values, bins=bins, range=(lo, hi), density=True, alpha=0.46, label="score", color="#2563eb")
        ax.hist(perf_values, bins=bins, range=(lo, hi), density=True, alpha=0.46, label="perf", color="#dc2626")
        ax.set_title(group_title)
        ax.set_xlabel(label)
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.2)
        ax.legend()
    fig.suptitle(f"Score vs performance {feature.upper()} log distribution", fontsize=14)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def finite_values(values):
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def plot_dev_grid(groups, output_path, bins):
    fig, axes = plt.subplots(2, 3, figsize=(16, 7.5), constrained_layout=True)
    group_order = [("score_ioi_eq_0", "score IOI = 0"), ("score_ioi_gt_0", "score IOI > 0")]
    columns = [
        ("ioi_dev_floor", "IOI dev floor\nln(max(perf,1)/max(score,1))"),
        ("ioi_dev_positive", "IOI dev positive\nscore>0: ln(perf/score), score=0: ln(perf/tau)"),
        ("duration_dev", "Duration dev\nln(perf/score)"),
    ]
    colors = {
        "ioi_dev_floor": "#f97316",
        "ioi_dev_positive": "#dc2626",
        "duration_dev": "#7c3aed",
    }
    for row_idx, (group_key, group_title) in enumerate(group_order):
        for col_idx, (key, title) in enumerate(columns):
            ax = axes[row_idx][col_idx]
            values = finite_values(groups[group_key][key])
            if len(values) == 0:
                ax.text(0.5, 0.5, "no finite values", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{group_title}: {title}")
                continue
            lo, hi = histogram_range([values])
            ax.hist(values, bins=bins, range=(lo, hi), density=True, alpha=0.82, color=colors[key])
            ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.55)
            ax.set_title(f"{group_title}: {title}")
            ax.set_xlabel("log ratio")
            ax.set_ylabel("density")
            ax.grid(True, alpha=0.2)
    fig.suptitle("ASAP aligned timing dev distributions by score IOI group", fontsize=15)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_scale_distribution(scales, output_path, bins):
    scales = np.asarray(scales, dtype=np.float64)
    log_scales = np.log(scales[np.isfinite(scales) & (scales > 0.0)])
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.hist(log_scales, bins=bins, density=True, alpha=0.84, color="#0f766e")
    ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.65)
    ax.set_xlabel("ln(score onset span / performance onset span)")
    ax.set_ylabel("density")
    ax.set_title("ASAP global performance timing scale")
    ax.grid(True, alpha=0.2)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    columns = [
        "tier_a",
        "split",
        "performance_dataset",
        "refined_score_midi_path",
        "refined_performance_midi_path",
        "refined_alignment_path",
    ]
    df = pd.read_csv(args.metadata, usecols=columns)
    df = df[df["tier_a"].fillna(False).astype(bool)]
    df = df[df["refined_score_midi_path"].notna()]
    df = df[df["refined_performance_midi_path"].notna()]
    df = df[df["refined_alignment_path"].notna()]
    if args.performance_dataset:
        df = df[df["performance_dataset"].fillna("").astype(str) == str(args.performance_dataset)]
    if args.split:
        df = df[df["split"].fillna("").astype(str) == str(args.split)]
    selected = load_score_source_filter(args.score_source_list)
    if selected is not None:
        df = df[df["refined_score_midi_path"].isin(selected)]
    df = df.sort_values(["refined_score_midi_path", "refined_performance_midi_path"], kind="stable")

    groups = {
        "score_ioi_eq_0": {
            "score_ioi": [],
            "perf_ioi": [],
            "score_duration": [],
            "perf_duration": [],
        },
        "score_ioi_gt_0": {
            "score_ioi": [],
            "perf_ioi": [],
            "score_duration": [],
            "perf_duration": [],
        },
    }
    failures = []
    timing_scales = []
    jobs = []
    for _, row in df.iterrows():
        score_path = args.midi_root / row["refined_score_midi_path"]
        perf_path = args.midi_root / row["refined_performance_midi_path"]
        align_path = args.midi_root / row["refined_alignment_path"]
        jobs.append((score_path, perf_path, align_path, row["refined_score_midi_path"], row["refined_performance_midi_path"], args.normalize_perf_to_score_onset_span))

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            loaded_pairs = pool.map(load_pair_job, jobs)
            loaded_pairs = list(loaded_pairs)
    else:
        loaded_pairs = [load_pair_job(job) for job in jobs]

    for score_raw, perf_raw, scale, failure in loaded_pairs:
        if failure is not None:
            failures.append(failure)
            continue
        if scale is not None:
            timing_scales.append(scale)
        score_ioi = score_raw[:, 0]
        mask_zero = score_ioi <= float(args.zero_eps_ms)
        for group_key, mask in (("score_ioi_eq_0", mask_zero), ("score_ioi_gt_0", ~mask_zero)):
            if not np.any(mask):
                continue
            groups[group_key]["score_ioi"].extend(score_raw[mask, 0].tolist())
            groups[group_key]["perf_ioi"].extend(perf_raw[mask, 0].tolist())
            groups[group_key]["score_duration"].extend(score_raw[mask, 1].tolist())
            groups[group_key]["perf_duration"].extend(perf_raw[mask, 1].tolist())

    for group in groups.values():
        for key in ("score_ioi", "perf_ioi", "score_duration", "perf_duration"):
            group[f"{key}_log"] = ln_floor(group[key], args.floor_ms)
            group[key] = np.asarray(group[key], dtype=np.float64)
        group["ioi_dev_floor"] = group["perf_ioi_log"] - group["score_ioi_log"]
        score_ioi = group["score_ioi"]
        perf_ioi = group["perf_ioi"]
        score_base = np.where(score_ioi > args.zero_eps_ms, score_ioi, float(args.tau_chord_ms))
        group["ioi_dev_positive"] = np.full(len(perf_ioi), np.nan, dtype=np.float64)
        positive_mask = perf_ioi > args.zero_eps_ms
        group["ioi_dev_positive"][positive_mask] = (
            np.log(perf_ioi[positive_mask]) - np.log(score_base[positive_mask])
        )
        group["duration_dev"] = (
            ln_floor(group["perf_duration"], args.floor_ms)
            - ln_floor(group["score_duration"], args.floor_ms)
        )

    summary_rows = []
    for group_key, group in groups.items():
        append_summary(
            summary_rows,
            group_key,
            "score",
            "ioi",
            group["score_ioi"],
            group["score_ioi_log"],
            args.zero_eps_ms,
        )
        append_summary(
            summary_rows,
            group_key,
            "perf",
            "ioi",
            group["perf_ioi"],
            group["perf_ioi_log"],
            args.zero_eps_ms,
        )
        append_summary(
            summary_rows,
            group_key,
            "score",
            "duration",
            group["score_duration"],
            group["score_duration_log"],
            args.zero_eps_ms,
        )
        append_summary(
            summary_rows,
            group_key,
            "perf",
            "duration",
            group["perf_duration"],
            group["perf_duration_log"],
            args.zero_eps_ms,
        )
        append_dev_summary(summary_rows, group_key, "ioi_floor", group["ioi_dev_floor"])
        append_dev_summary(summary_rows, group_key, "ioi_positive", group["ioi_dev_positive"])
        append_dev_summary(summary_rows, group_key, "duration", group["duration_dev"])

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(args.output_dir / "log_timing_summary.csv", index=False)
    plot_grid(groups, args.output_dir / "log_timing_by_score_ioi_group.png", args.bins)
    plot_overlay(groups, args.output_dir / "log_ioi_score_vs_perf.png", "ioi", args.bins)
    plot_overlay(groups, args.output_dir / "log_duration_score_vs_perf.png", "duration", args.bins)
    plot_dev_grid(groups, args.output_dir / "log_dev_by_score_ioi_group.png", args.bins)
    if timing_scales:
        plot_scale_distribution(timing_scales, args.output_dir / "global_timing_scale.png", args.bins)

    meta = {
        "metadata": str(args.metadata.resolve()),
        "midi_root": str(args.midi_root.resolve()),
        "performance_dataset": args.performance_dataset,
        "split": args.split,
        "score_source_list": str(args.score_source_list.resolve()) if args.score_source_list else None,
        "floor_ms": args.floor_ms,
        "tau_chord_ms": args.tau_chord_ms,
        "zero_eps_ms": args.zero_eps_ms,
        "performance_time_normalization": "score_onset_span" if args.normalize_perf_to_score_onset_span else "none",
        "global_timing_scale": finite_quantiles(timing_scales),
        "rows_requested": int(len(df)),
        "rows_failed": int(len(failures)),
        "failures": failures[:20],
        "outputs": {
            "summary": str((args.output_dir / "log_timing_summary.csv").resolve()),
            "grid": str((args.output_dir / "log_timing_by_score_ioi_group.png").resolve()),
            "ioi_overlay": str((args.output_dir / "log_ioi_score_vs_perf.png").resolve()),
            "duration_overlay": str((args.output_dir / "log_duration_score_vs_perf.png").resolve()),
            "dev_grid": str((args.output_dir / "log_dev_by_score_ioi_group.png").resolve()),
            "timing_scale": str((args.output_dir / "global_timing_scale.png").resolve()) if timing_scales else None,
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
