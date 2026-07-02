#!/usr/bin/env python3
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.train.train_inr import normalize_duration_dev, normalize_ioi_dev


TARGETS = [
    ("ioi_dev", "IOI dev"),
    ("duration_dev", "Duration dev"),
    ("velocity", "Velocity"),
    ("pedal_start", "Pedal start"),
    ("pedal_ctrl", "Pedal ctrl"),
]


def load_config(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def iter_processed_files(processed_dir, metadata_path=None, performance_dataset=None, split="all"):
    processed_dir = Path(processed_dir)
    if metadata_path:
        meta = pd.read_csv(metadata_path)
        if performance_dataset:
            meta = meta[meta["performance_dataset"].fillna("").astype(str) == performance_dataset]
        if split != "all":
            meta = meta[meta["split"].fillna("").astype(str) == split]
        if "is_refined" in meta.columns:
            meta = meta[meta["is_refined"].fillna(False).astype(bool)]
        paths = []
        for value in meta["refined_score_midi_path"].dropna().astype(str).unique():
            json_path = processed_dir / value
            json_path = json_path.with_suffix(".json")
            if json_path.exists():
                paths.append(json_path)
        yield from sorted(set(paths))
        return
    yield from sorted(processed_dir.rglob("score_*_refined.json"))


def normalized_rows_for_work(path, config, performance_dataset, split, exclude_interpolated, max_notes_per_work):
    data = json.loads(path.read_text(encoding="utf-8"))
    score = data.get("score", {})
    score_raw = score.get("score_raw") or []
    pitch = score.get("pitch") or []
    score_source = score.get("score_source") or str(path)
    rows = []
    max_len = min(
        len(score_raw),
        len(pitch),
        *[
            len(perf.get("label_shared_raw") or [])
            for perf in data.get("performances", [])
            if perf.get("label_shared_raw") is not None
        ] or [len(score_raw)],
    )
    if max_notes_per_work and max_len > max_notes_per_work:
        note_indices = set(np.linspace(0, max_len - 1, max_notes_per_work, dtype=int).tolist())
    else:
        note_indices = None

    for perf in data.get("performances", []):
        if performance_dataset and str(perf.get("performance_dataset", "")) != performance_dataset:
            continue
        if split != "all" and str(perf.get("split", "")) != split:
            continue
        shared = perf.get("label_shared_raw")
        pedal = perf.get("label_pedal2_raw") or perf.get("pedal2_raw")
        if shared is None or pedal is None:
            continue
        interpolated = perf.get("interpolated") or [0] * len(shared)
        limit = min(len(score_raw), len(shared), len(pedal), len(pitch))
        for idx in range(limit):
            if note_indices is not None and idx not in note_indices:
                continue
            if exclude_interpolated and idx < len(interpolated) and int(interpolated[idx]) != 0:
                continue
            score_row = score_raw[idx]
            perf_row = shared[idx]
            pedal_row = pedal[idx]
            rows.append(
                {
                    "score_json": str(path),
                    "score_source": score_source,
                    "note_idx": idx,
                    "pitch": int(pitch[idx]),
                    "score_ioi_ms": float(score_row[0]),
                    "score_duration_ms": float(score_row[1]),
                    "performance_id": perf.get("performance_id") or perf.get("id"),
                    "ioi_dev": normalize_ioi_dev(
                        score_row[0],
                        perf_row[0],
                        epr_timing_target=config.get("epr_timing_target", "log_deviation"),
                        log_scale=float(config.get("timing_log_scale", 50.0)),
                        split_zero_ioi_head=bool(config.get("split_zero_ioi_head", False)),
                        nonzero_scale=float(config.get("ioi_nonzero_dev_scale", 2.0)),
                        zero_scale=float(config.get("ioi_zero_dev_scale", 4.0)),
                    ),
                    "duration_dev": normalize_duration_dev(
                        score_row[1],
                        perf_row[1],
                        epr_timing_target=config.get("epr_timing_target", "log_deviation"),
                        log_scale=float(config.get("timing_log_scale", 50.0)),
                    ),
                    "velocity": min(max(float(perf_row[2]), 0.0), 127.0) / 127.0,
                    "pedal_start": min(max(float(pedal_row[0]), 0.0), 127.0) / 127.0,
                    "pedal_ctrl": min(max(float(pedal_row[1]), 0.0), 127.0) / 127.0,
                }
            )
    return rows


def collect_rows(
    config,
    processed_dir,
    metadata_path,
    performance_dataset,
    split,
    max_works,
    exclude_interpolated,
    max_notes_per_work,
):
    rows = []
    used = 0
    for path in iter_processed_files(processed_dir, metadata_path, performance_dataset, split):
        work_rows = normalized_rows_for_work(
            path,
            config,
            performance_dataset,
            split,
            exclude_interpolated,
            max_notes_per_work,
        )
        if not work_rows:
            continue
        rows.extend(work_rows)
        used += 1
        if max_works and used >= max_works:
            break
    if not rows:
        raise SystemExit("No rows collected.")
    return pd.DataFrame(rows)


def skew(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < 3:
        return np.nan
    std = arr.std(ddof=0)
    if std <= 1e-12:
        return 0.0
    return float(np.mean(((arr - arr.mean()) / std) ** 3))


def summarize_groups(df, min_perfs, boundary_eps):
    records = []
    grouped = df.groupby(["score_json", "score_source", "note_idx"], sort=False)
    for (score_json, score_source, note_idx), group in grouped:
        if len(group) < min_perfs:
            continue
        base = {
            "score_json": score_json,
            "score_source": score_source,
            "note_idx": int(note_idx),
            "n": int(len(group)),
            "pitch": int(group["pitch"].iloc[0]),
            "score_ioi_ms": float(group["score_ioi_ms"].iloc[0]),
            "score_duration_ms": float(group["score_duration_ms"].iloc[0]),
            "is_zero_score_ioi": bool(float(group["score_ioi_ms"].iloc[0]) <= 0.0),
        }
        for key, _ in TARGETS:
            values = group[key].to_numpy(dtype=np.float64)
            base[f"{key}_mean"] = float(values.mean())
            base[f"{key}_std"] = float(values.std(ddof=0))
            base[f"{key}_skew"] = skew(values)
            base[f"{key}_q05"] = float(np.quantile(values, 0.05))
            base[f"{key}_q50"] = float(np.quantile(values, 0.50))
            base[f"{key}_q95"] = float(np.quantile(values, 0.95))
            base[f"{key}_boundary_frac"] = float(np.mean((values <= boundary_eps) | (values >= 1.0 - boundary_eps)))
        records.append(base)
    return pd.DataFrame(records)


def choose_example_groups(stats, per_group):
    chosen = []
    for is_zero in (True, False):
        sub = stats[stats["is_zero_score_ioi"] == is_zero].copy()
        if sub.empty:
            continue
        sub["var_score"] = sub[[f"{key}_std" for key, _ in TARGETS]].mean(axis=1)
        sub = sub.sort_values(["n", "var_score"], ascending=[False, False])
        seen_scores = set()
        for _, row in sub.iterrows():
            if row["score_json"] in seen_scores:
                continue
            chosen.append((row["score_json"], int(row["note_idx"])))
            seen_scores.add(row["score_json"])
            if len(seen_scores) >= per_group:
                break
    return chosen


def short_score_name(path):
    parts = Path(path).parts
    return "/".join(parts[-4:]).replace("_", " ")


def plot_examples(df, stats, groups, output_path):
    if not groups:
        return
    rows = len(groups)
    cols = len(TARGETS)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, max(2.0, rows * 1.35)), sharex=True)
    if rows == 1:
        axes = np.expand_dims(axes, 0)
    rng = np.random.default_rng(42)
    for r, (score_json, note_idx) in enumerate(groups):
        group = df[(df["score_json"] == score_json) & (df["note_idx"] == note_idx)]
        meta = stats[(stats["score_json"] == score_json) & (stats["note_idx"] == note_idx)].iloc[0]
        left_label = (
            f"{'zero' if meta['is_zero_score_ioi'] else 'nonzero'} "
            f"idx={note_idx} n={int(meta['n'])} pitch={int(meta['pitch'])}\n"
            f"{short_score_name(score_json)}"
        )
        for c, (key, title) in enumerate(TARGETS):
            ax = axes[r, c]
            values = group[key].to_numpy(dtype=np.float64)
            bins = np.linspace(0.0, 1.0, 21)
            ax.hist(values, bins=bins, density=True, color="#6B8F71", alpha=0.35, edgecolor="none")
            jitter = rng.normal(loc=0.0, scale=0.018, size=len(values))
            ax.scatter(values, np.full_like(values, 0.02) + jitter, s=8, alpha=0.55, color="#1D3557", linewidths=0)
            ax.axvline(np.median(values), color="#C1121F", linewidth=1.0)
            ax.set_xlim(0.0, 1.0)
            if r == 0:
                ax.set_title(title, fontsize=10)
            if c == 0:
                ax.set_ylabel(left_label, fontsize=7)
            else:
                ax.set_yticklabels([])
            ax.tick_params(axis="both", labelsize=7)
    fig.suptitle("Per-note normalized target distributions across performances", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_stats(stats, output_path):
    fig, axes = plt.subplots(3, len(TARGETS), figsize=(len(TARGETS) * 3.0, 8.0))
    for c, (key, title) in enumerate(TARGETS):
        for r, (suffix, ylabel, color) in enumerate(
            [
                ("std", "per-note std", "#457B9D"),
                ("skew", "per-note skew", "#A23E48"),
                ("boundary_frac", "boundary frac", "#2A9D8F"),
            ]
        ):
            ax = axes[r, c]
            values = stats[f"{key}_{suffix}"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=np.float64)
            if suffix == "skew":
                values = np.clip(values, -5.0, 5.0)
                bins = np.linspace(-5.0, 5.0, 41)
            else:
                bins = np.linspace(0.0, max(1.0 if suffix == "boundary_frac" else 0.5, values.max(initial=0.0)), 36)
            ax.hist(values, bins=bins, color=color, alpha=0.72)
            ax.axvline(np.median(values), color="black", linewidth=1.0)
            if r == 0:
                ax.set_title(title, fontsize=10)
            if c == 0:
                ax.set_ylabel(ylabel)
            ax.tick_params(axis="both", labelsize=7)
    fig.suptitle("Across-note distribution-shape statistics", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--processed-dir", default="../PianoCoRe/processed")
    parser.add_argument("--metadata-path", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--performance-dataset", default="ASAP")
    parser.add_argument("--split", default="all", choices=["all", "train", "test", "val", "valid", "validation"])
    parser.add_argument("--min-perfs", type=int, default=8)
    parser.add_argument("--examples-per-group", type=int, default=8)
    parser.add_argument("--max-works", type=int, default=None)
    parser.add_argument("--max-notes-per-work", type=int, default=None)
    parser.add_argument("--include-interpolated", action="store_true")
    parser.add_argument("--write-rows-csv", action="store_true")
    parser.add_argument("--boundary-eps", type=float, default=1e-4)
    args = parser.parse_args()

    config = load_config(args.config)
    metadata_path = args.metadata_path or config.get("metadata_path")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split = "validation" if args.split == "valid" else args.split
    df = collect_rows(
        config,
        args.processed_dir,
        metadata_path,
        args.performance_dataset,
        split,
        args.max_works,
        exclude_interpolated=not args.include_interpolated,
        max_notes_per_work=args.max_notes_per_work,
    )
    stats = summarize_groups(df, args.min_perfs, args.boundary_eps)
    if stats.empty:
        raise SystemExit(f"No note groups with at least {args.min_perfs} performances.")

    stats.to_csv(out_dir / "per_note_distribution_stats.csv", index=False)
    if args.write_rows_csv:
        df.to_csv(out_dir / "per_note_target_rows.csv", index=False)

    groups = choose_example_groups(stats, args.examples_per_group)
    plot_examples(df, stats, groups, out_dir / "per_note_distribution_examples.png")
    plot_stats(stats, out_dir / "per_note_distribution_stats.png")

    summary = {
        "config": str(Path(args.config).resolve()),
        "performance_dataset": args.performance_dataset,
        "split": split,
        "rows": int(len(df)),
        "note_groups": int(len(stats)),
        "min_perfs": int(args.min_perfs),
        "max_works": args.max_works,
        "max_notes_per_work": args.max_notes_per_work,
        "examples": [
            {"score_json": score_json, "note_idx": note_idx}
            for score_json, note_idx in groups
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
