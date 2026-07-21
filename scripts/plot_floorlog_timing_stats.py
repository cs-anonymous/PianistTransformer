#!/usr/bin/env python3
"""Plot floor-log timing and deviation statistics for the INSPIRE paper."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def floor_log_ms(values: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(values.astype(np.float64), 1.0))


def collect_arrays(root: Path) -> dict[str, np.ndarray]:
    score_ioi_logs = []
    perf_ioi_logs = []
    score_dur_logs = []
    perf_dur_logs = []
    ioi_devs = []
    dur_devs = []

    for path in sorted(root.rglob("score_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        score = payload.get("score") or {}
        score_rows = score.get("score_raw") or payload.get("score_shared_raw")
        if not score_rows:
            continue
        score_arr = np.asarray(score_rows, dtype=np.float64)
        if score_arr.ndim != 2 or score_arr.shape[1] < 2:
            continue
        s_ioi = score_arr[:, 0]
        s_dur = score_arr[:, 1]
        s_ioi_log = floor_log_ms(s_ioi)
        s_dur_log = floor_log_ms(s_dur)

        for perf in payload.get("performances", []):
            perf_rows = perf.get("label_shared_raw")
            if perf_rows is None:
                raw = perf.get("label_raw")
                perf_rows = [row[:3] for row in raw] if raw is not None else None
            if perf_rows is None:
                continue
            perf_arr = np.asarray(perf_rows, dtype=np.float64)
            n = min(len(score_arr), len(perf_arr))
            if n <= 0 or perf_arr.ndim != 2 or perf_arr.shape[1] < 2:
                continue

            p_ioi = perf_arr[:n, 0]
            p_dur = perf_arr[:n, 1]
            p_ioi_log = floor_log_ms(p_ioi)
            p_dur_log = floor_log_ms(p_dur)

            score_ioi_logs.append(s_ioi_log[:n])
            perf_ioi_logs.append(p_ioi_log)
            score_dur_logs.append(s_dur_log[:n])
            perf_dur_logs.append(p_dur_log)

            nz_ioi = s_ioi[:n] > 0.0
            pos_dur = s_dur[:n] > 0.0
            if np.any(nz_ioi):
                ioi_devs.append(p_ioi_log[nz_ioi] - s_ioi_log[:n][nz_ioi])
            if np.any(pos_dur):
                dur_devs.append(p_dur_log[pos_dur] - s_dur_log[:n][pos_dur])

    def cat(parts: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(parts) if parts else np.asarray([], dtype=np.float64)

    return {
        "score_ioi_log": cat(score_ioi_logs),
        "perf_ioi_log": cat(perf_ioi_logs),
        "score_dur_log": cat(score_dur_logs),
        "perf_dur_log": cat(perf_dur_logs),
        "ioi_dev": cat(ioi_devs),
        "dur_dev": cat(dur_devs),
    }


def pct_inside(values: np.ndarray, lo: float = -2.0, hi: float = 1.0) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.mean((values >= lo) & (values <= hi)) * 100.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/ASAP_processed"))
    parser.add_argument("--out", type=Path, default=Path("INSPIRE-AAAI2027/Figures/floorlog_timing_stats.png"))
    parser.add_argument("--summary", type=Path, default=Path("INSPIRE-AAAI2027/Figures/floorlog_timing_stats.json"))
    args = parser.parse_args()

    arrays = collect_arrays(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.6,
    })
    fig, axes = plt.subplots(2, 2, figsize=(3.45, 2.55), dpi=300)
    blue = "#2f6fb0"
    gray = "#8a8f98"

    def hist_overlay(ax, score_values, perf_values, title, xlabel):
        bins = np.linspace(
            np.nanpercentile(np.concatenate([score_values, perf_values]), 0.5),
            np.nanpercentile(np.concatenate([score_values, perf_values]), 99.5),
            60,
        )
        ax.hist(score_values, bins=bins, density=True, histtype="stepfilled", alpha=0.22, color=gray, label="Score")
        ax.hist(perf_values, bins=bins, density=True, histtype="step", linewidth=1.0, color=blue, label="Performance")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.grid(alpha=0.18, linewidth=0.4)

    hist_overlay(axes[0, 0], arrays["score_ioi_log"], arrays["perf_ioi_log"], "Log IOI", r"$\log(\max(t,1))$")
    hist_overlay(axes[0, 1], arrays["score_dur_log"], arrays["perf_dur_log"], "Log duration", r"$\log(\max(t,1))$")
    axes[0, 0].legend(frameon=False, loc="upper left")

    def dev_hist(ax, values, title):
        bins = np.linspace(-4.0, 4.0, 80)
        ax.axvspan(-2.0, 1.0, color=blue, alpha=0.10, linewidth=0)
        ax.axvline(-2.0, color=blue, linestyle="--", linewidth=0.7)
        ax.axvline(1.0, color=blue, linestyle="--", linewidth=0.7)
        ax.hist(values, bins=bins, density=True, histtype="stepfilled", alpha=0.42, color=blue)
        inside = pct_inside(values)
        ax.text(0.98, 0.92, f"{inside:.1f}% in [-2,1]", transform=ax.transAxes, ha="right", va="top", fontsize=6)
        ax.set_xlim(-4.0, 4.0)
        ax.set_title(title)
        ax.set_xlabel("log deviation")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.18, linewidth=0.4)

    dev_hist(axes[1, 0], arrays["ioi_dev"], "IOI dev (score IOI > 0)")
    dev_hist(axes[1, 1], arrays["dur_dev"], "Duration dev")

    for ax in axes.ravel():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=0.35, w_pad=0.45, h_pad=0.55)
    fig.savefig(args.out, bbox_inches="tight")

    summary = {
        key: {
            "count": int(value.size),
            "mean": float(np.mean(value)) if value.size else math.nan,
            "p05": float(np.quantile(value, 0.05)) if value.size else math.nan,
            "p50": float(np.quantile(value, 0.50)) if value.size else math.nan,
            "p95": float(np.quantile(value, 0.95)) if value.size else math.nan,
            "pct_in_-2_1": pct_inside(value) if key.endswith("_dev") else math.nan,
        }
        for key, value in arrays.items()
    }
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "summary": str(args.summary), "ioi_dev_pct": summary["ioi_dev"]["pct_in_-2_1"], "dur_dev_pct": summary["dur_dev"]["pct_in_-2_1"]}, indent=2))


if __name__ == "__main__":
    main()
