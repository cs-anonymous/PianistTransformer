#!/usr/bin/env python3
"""Compare floor-log IOI deviations for score-zero and score-nonzero IOIs.

The nonzero group is the only group used to estimate a performance's global
speed factor.  For the zero group, ``log(max(perf_ioi, 1))`` is plotted only as
an arpeggiation/quantisation diagnostic because a ratio to score IOI=0 is not
defined.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


def finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def floor_log_deviation(score_ioi: np.ndarray, perf_ioi: np.ndarray) -> np.ndarray:
    """Return log(max(perf,1))-log(max(score,1)) in ms units."""
    return np.log(np.maximum(perf_ioi.astype(np.float64), 1.0)) - np.log(
        np.maximum(score_ioi.astype(np.float64), 1.0)
    )


def compute_scale_statistics(score_ioi: np.ndarray, perf_ioi: np.ndarray) -> dict:
    """Compute old, direct, and zero-aware aggregated timing scales.

    ``score_ioi`` is note-row IOI.  For a positive score IOI at row ``j``,
    the corresponding performance interval is the sum from the previous
    positive-score row (exclusive) through ``j`` (inclusive).  Thus all
    intermediate score-IOI=0 rows are retained in the performance interval,
    while their individual zero score denominators are never used.
    """
    score_ioi = np.asarray(score_ioi, dtype=np.float64)
    perf_ioi = np.asarray(perf_ioi, dtype=np.float64)
    n = min(len(score_ioi), len(perf_ioi))
    score_ioi = score_ioi[:n]
    perf_ioi = perf_ioi[:n]
    old_span = float(np.sum(score_ioi[1:])) / float(np.sum(perf_ioi[1:])) if n > 1 and np.sum(score_ioi[1:]) > 0 and np.sum(perf_ioi[1:]) > 0 else math.nan

    direct_mask = np.isfinite(score_ioi) & np.isfinite(perf_ioi) & (score_ioi > 0.0) & (perf_ioi > 0.0)
    direct_dev = np.log(perf_ioi[direct_mask]) - np.log(score_ioi[direct_mask])
    direct_scale = float(np.exp(-np.median(direct_dev))) if direct_dev.size else math.nan

    positive_rows = np.flatnonzero(np.isfinite(score_ioi) & (score_ioi > 0.0))
    grouped_devs = []
    grouped_intervals = []
    previous_positive_row = None
    for row in positive_rows:
        # Row zero is the initial onset marker in this representation.  Its
        # performance IOI is not an interval, so the first interval starts at
        # row 1.  Later intervals include every row after the previous
        # positive-score onset through the current row.
        start = 1 if previous_positive_row is None else int(previous_positive_row) + 1
        end = int(row)
        if end < start or end >= n:
            continue
        score_interval = float(score_ioi[row])
        perf_interval = float(np.sum(perf_ioi[start : end + 1]))
        if not np.isfinite(score_interval) or not np.isfinite(perf_interval) or score_interval <= 0.0 or perf_interval <= 0.0:
            previous_positive_row = int(row)
            continue
        dev = math.log(perf_interval) - math.log(score_interval)
        grouped_devs.append(dev)
        grouped_intervals.append(
            {
                "score_row": int(row),
                "performance_rows": [int(start), int(end)],
                "score_ioi_ms": score_interval,
                "performance_ioi_sum_ms": perf_interval,
                "dev": dev,
            }
        )
        previous_positive_row = int(row)
    grouped_devs = np.asarray(grouped_devs, dtype=np.float64)
    grouped_scale = float(np.exp(-np.median(grouped_devs))) if grouped_devs.size else math.nan
    return {
        "old_span_scale": old_span,
        "direct_scale": direct_scale,
        "grouped_scale": grouped_scale,
        "direct_dev": direct_dev,
        "grouped_dev": grouped_devs,
        "grouped_intervals": grouped_intervals,
    }


def collect_pairs(root: Path, pieces: int, performances: int) -> list[dict]:
    candidates = []
    for path in sorted(root.rglob("score_ASAP_refined.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        test_perfs = [
            perf
            for perf in payload.get("performances", [])
            if perf.get("split") == "test" and perf.get("label_shared_raw") is not None
        ]
        if len(test_perfs) < performances:
            continue
        rel = path.relative_to(root).as_posix()
        candidates.append((len(test_perfs), rel, path, test_perfs))

    # Prefer pieces with the largest number of held-out performances, then use
    # a stable path order so the figure is reproducible.
    candidates.sort(key=lambda item: (-item[0], item[1]))
    selected = []
    for _, rel, path, test_perfs in candidates[:pieces]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        score = np.asarray(payload["score"]["score_raw"], dtype=np.float64)
        chosen = []
        for perf in sorted(test_perfs, key=lambda p: (str(p.get("performance_id", "")), str(p.get("performance_source", ""))))[:performances]:
            perf_rows = np.asarray(perf["label_shared_raw"], dtype=np.float64)
            n = min(len(score), len(perf_rows))
            if n == 0 or score.ndim != 2 or perf_rows.ndim != 2 or score.shape[1] < 1 or perf_rows.shape[1] < 1:
                continue
            s_ioi = score[:n, 0]
            p_ioi = perf_rows[:n, 0]
            valid = np.isfinite(s_ioi) & np.isfinite(p_ioi) & (p_ioi >= 0.0)
            zero = valid & (s_ioi <= 0.0)
            # A speed ratio is defined only when both IOIs are strictly
            # positive.  Nonpositive performance IOIs are alignment artifacts
            # and are excluded from the span-median estimate as well.
            nonzero = valid & (s_ioi > 0.0) & (p_ioi > 0.0)
            dev = floor_log_deviation(s_ioi, p_ioi)
            valid_dev = finite(dev[nonzero])
            scales = compute_scale_statistics(s_ioi, p_ioi)
            chosen.append(
                {
                    "piece": rel,
                    "performance_id": str(perf.get("performance_id", perf.get("id", ""))),
                    "performance_source": str(perf.get("performance_source", "")),
                    "zero_dev": finite(dev[zero]),
                    "nonzero_dev": valid_dev,
                    "old_span_scale": scales["old_span_scale"],
                    "new_median_scale": scales["direct_scale"],
                    "grouped_median_scale": scales["grouped_scale"],
                    "grouped_dev": scales["grouped_dev"],
                    "grouped_intervals": scales["grouped_intervals"],
                    "zero_count": int(zero.sum()),
                    "nonzero_count": int(nonzero.sum()),
                }
            )
        if len(chosen) == performances:
            selected.append({"piece": rel, "pairs": chosen})
    if len(selected) < pieces:
        raise RuntimeError(f"Found only {len(selected)} pieces with {performances} usable test performances under {root}")
    return selected


def short_piece_name(relative_path: str) -> str:
    parts = Path(relative_path).parts
    composer = parts[0].split(",_")[0].replace("_", " ") if len(parts) >= 2 else ""
    title = parts[-2] if len(parts) >= 2 else Path(relative_path).stem
    title = title.replace("_", " ").replace(",", ", ")
    if title.startswith("1. ") or title.startswith("2. ") or title.startswith("3. ") or title.startswith("4. "):
        parent = parts[-3] if len(parts) >= 3 else ""
        parent = parent.replace("_", " ").replace(",", ", ")
        title = f"{parent}: {title}"
    full = f"{composer}: {title}" if composer else title
    return textwrap.shorten(full, width=44, placeholder="...")


def summary_rows(selected: list[dict]) -> list[dict]:
    rows = []
    for piece in selected:
        for pair in piece["pairs"]:
            row = {
                "piece": pair["piece"],
                "performance_id": pair["performance_id"],
                "performance_source": pair["performance_source"],
                "zero_count": pair["zero_count"],
                "nonzero_count": pair["nonzero_count"],
                "old_span_scale": pair["old_span_scale"],
                "new_median_scale": pair["new_median_scale"],
                "grouped_median_scale": pair["grouped_median_scale"],
                "grouped_count": int(len(pair["grouped_dev"])),
            }
            for group in ("zero", "nonzero"):
                values = pair[f"{group}_dev"]
                row[f"{group}_median"] = float(np.median(values)) if values.size else math.nan
                row[f"{group}_mean"] = float(np.mean(values)) if values.size else math.nan
                row[f"{group}_p05"] = float(np.quantile(values, 0.05)) if values.size else math.nan
                row[f"{group}_p95"] = float(np.quantile(values, 0.95)) if values.size else math.nan
            grouped = pair["grouped_dev"]
            row["grouped_count"] = int(grouped.size)
            row["grouped_median_dev"] = float(np.median(grouped)) if grouped.size else math.nan
            row["grouped_mean_dev"] = float(np.mean(grouped)) if grouped.size else math.nan
            row["grouped_p05_dev"] = float(np.quantile(grouped, 0.05)) if grouped.size else math.nan
            row["grouped_p95_dev"] = float(np.quantile(grouped, 0.95)) if grouped.size else math.nan
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(selected: list[dict], output: Path, bins: int = 80) -> None:
    plt.rcParams.update({"font.family": "DejaVu Serif", "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6})
    fig, axes = plt.subplots(len(selected), 2, figsize=(7.2, 1.55 * len(selected)), squeeze=False, sharex=False)
    colors = plt.get_cmap("tab10").colors
    for row_idx, piece in enumerate(selected):
        ax_zero, ax_nonzero = axes[row_idx]
        # The right panel uses the zero-aware grouped intervals.  A direct
        # per-row comparison would split one score interval across multiple
        # performance rows and can create a spurious negative deviation.
        for ax, group in ((ax_zero, "zero"), (ax_nonzero, "grouped")):
            all_values = np.concatenate([pair[f"{group}_dev"] for pair in piece["pairs"] if pair[f"{group}_dev"].size])
            lo, hi = np.quantile(all_values, [0.005, 0.995])
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = float(np.min(all_values)), float(np.max(all_values))
            if hi <= lo:
                hi = lo + 1.0
            bins_by_group = np.linspace(lo, hi, max(int(bins), 20) + 1)
            ax._shared_bins = bins_by_group
        for idx, pair in enumerate(piece["pairs"]):
            color = colors[idx % len(colors)]
            label = pair["performance_id"]
            for ax, group in ((ax_zero, "zero"), (ax_nonzero, "grouped")):
                values = pair[f"{group}_dev"]
                if values.size == 0:
                    continue
                ax.hist(values, bins=ax._shared_bins, density=True, histtype="step", linewidth=0.55, alpha=0.35, color=color)
                if values.size >= 2 and np.ptp(values) > 1e-9:
                    grid = np.linspace(ax._shared_bins[0], ax._shared_bins[-1], 300)
                    try:
                        density = gaussian_kde(values)(grid)
                        ax.plot(grid, density, linewidth=0.9, color=color, label=label)
                    except (np.linalg.LinAlgError, ValueError):
                        ax.plot([], [], color=color, label=label)
                else:
                    ax.plot([], [], color=color, label=label)
                median = float(np.median(values))
                ax.axvline(median, color=color, linewidth=0.65, alpha=0.65)
        ax_zero.set_title(short_piece_name(piece["piece"]) + " | score IOI = 0")
        ax_nonzero.set_title("score intervals with zero rows aggregated")
        ax_zero.set_ylabel("density")
        for ax in (ax_zero, ax_nonzero):
            ax.grid(alpha=0.2, linewidth=0.4)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        ax_nonzero.legend(frameon=False, loc="upper right", ncol=2)
    axes[-1, 0].set_xlabel("floor-log diagnostic: log(max(performance IOI, 1)) - log(max(score IOI, 1))")
    axes[-1, 1].set_xlabel("log deviation: aggregated performance interval / score interval")
    fig.tight_layout(h_pad=0.8, w_pad=0.8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_scale_comparison(selected: list[dict], output: Path) -> None:
    """Plot old, direct, and zero-aware median-log speed corrections."""
    plt.rcParams.update({"font.family": "DejaVu Serif", "axes.titlesize": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8})
    rows = summary_rows(selected)
    old = np.asarray([row["old_span_scale"] for row in rows], dtype=np.float64)
    direct = np.asarray([row["new_median_scale"] for row in rows], dtype=np.float64)
    grouped = np.asarray([row["grouped_median_scale"] for row in rows], dtype=np.float64)
    keep = np.isfinite(old) & np.isfinite(direct) & np.isfinite(grouped) & (old > 0.0) & (direct > 0.0) & (grouped > 0.0)
    old, direct, grouped = old[keep], direct[keep], grouped[keep]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw={"width_ratios": [1.25, 1]})
    lo = float(min(np.min(old), np.min(direct), np.min(grouped)))
    hi = float(max(np.max(old), np.max(direct), np.max(grouped)))
    pad = max((hi - lo) * 0.08, 0.01)
    grid = np.linspace(max(0.0, lo - pad), hi + pad, 300)
    for values, label, color in (
        (old, "old onset-span", "#c45a3a"),
        (direct, "direct nonzero median", "#2f6fb0"),
        (grouped, "zero-aware grouped median", "#2f8f5b"),
    ):
        axes[0].hist(values, bins=18, density=True, histtype="step", linewidth=0.6, alpha=0.35, color=color)
        if len(values) >= 2 and np.ptp(values) > 1e-9:
            axes[0].plot(grid, gaussian_kde(values)(grid), color=color, linewidth=1.5, label=label)
        axes[0].axvline(np.median(values), color=color, linestyle="--", linewidth=0.8)
    axes[0].axvline(1.0, color="black", linestyle=":", linewidth=0.8)
    axes[0].set_xlabel("scale applied to performance time")
    axes[0].set_ylabel("density")
    axes[0].set_title("Per-performance scale factors")
    axes[0].legend(frameon=False)
    axes[1].scatter(old, grouped, s=25, color="#2f8f5b", alpha=0.82, label="grouped vs old")
    axes[1].scatter(old, direct, s=18, color="#2f6fb0", alpha=0.55, label="direct vs old")
    axes[1].plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k:", linewidth=0.8)
    axes[1].set_xlabel("old onset-span scale")
    axes[1].set_ylabel("new scale")
    axes[1].set_title("Pairwise comparison")
    axes[1].legend(frameon=False, loc="best")
    for ax in axes:
        ax.grid(alpha=0.2, linewidth=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout(w_pad=1.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/ASAP_processed_pedal_group"))
    parser.add_argument("--pieces", type=int, default=5)
    parser.add_argument("--performances", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("results/score_ioi_zero_nonzero_deviation_5x4.png"))
    parser.add_argument("--summary", type=Path, default=Path("results/score_ioi_zero_nonzero_deviation_5x4.json"))
    parser.add_argument("--csv", type=Path, default=Path("results/score_ioi_zero_nonzero_deviation_5x4.csv"))
    parser.add_argument("--scale-plot", type=Path, default=Path("results/score_span_old_vs_median_log_scale_5x4.png"))
    parser.add_argument("--examples", type=Path, default=Path("results/score_span_grouped_examples_5x4.json"))
    parser.add_argument("--bins", type=int, default=80, help="Number of histogram bins per panel")
    args = parser.parse_args()

    selected = collect_pairs(args.root, args.pieces, args.performances)
    rows = summary_rows(selected)
    plot(selected, args.out, bins=args.bins)
    plot_scale_comparison(selected, args.scale_plot)
    write_csv(args.csv, rows)
    payload = {
        "definition": {
            "nonzero": "log(performance_ioi_ms) - log(score_ioi_ms), requiring score_ioi_ms > 0 and performance_ioi_ms > 0; used for the per-performance span median",
            "zero": "same floor-log expression, diagnostic only; score IOI=0 has no defined ratio",
            "grouped": "for every score IOI > 0 row, sum performance IOIs from the previous score IOI > 0 row (exclusive) through the current row (inclusive); this retains score IOI=0 rows inside the interval",
            "split": "test",
            "pieces": args.pieces,
            "performances_per_piece": args.performances,
            "bins": args.bins,
        },
        "pieces": [piece["piece"] for piece in selected],
        "rows": rows,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")
    examples = []
    for piece in selected:
        for pair in piece["pairs"]:
            intervals = pair["grouped_intervals"]
            # Show the first intervals that actually contain an intermediate
            # score-IOI=0 row, which makes the difference concrete.
            for interval in intervals:
                start, end = interval["performance_rows"]
                if end > start:
                    examples.append({
                        "piece": pair["piece"],
                        "performance_id": pair["performance_id"],
                        **interval,
                    })
                if len([x for x in examples if x["piece"] == pair["piece"] and x["performance_id"] == pair["performance_id"]]) >= 3:
                    break
    args.examples.parent.mkdir(parents=True, exist_ok=True)
    args.examples.write_text(json.dumps({"definition": payload["definition"]["grouped"], "examples": examples}, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")
    print(json.dumps({"plot": str(args.out), "summary": str(args.summary), "csv": str(args.csv), "pairs": len(rows), "pieces": payload["pieces"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
