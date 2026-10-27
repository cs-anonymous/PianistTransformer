#!/usr/bin/env python3
"""Plot the ordered five-metric chain and its adjacent correlations."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr


CHAIN = [
    ("train_loss", "Train loss"),
    ("valid_loss", "Valid loss"),
    ("valid_tf_hr", "Valid TF HR"),
    ("test_tf_hr", "Test TF HR"),
    ("test_hr", "Test HR"),
]


def zscore(values):
    values = np.asarray(values, dtype=float)
    scale = values.std(ddof=0)
    return (values - values.mean()) / scale if scale > 0 else np.zeros_like(values)


def corr(x, y):
    return (
        float(pearsonr(x, y).statistic),
        float(spearmanr(x, y).statistic),
        float(kendalltau(x, y).statistic),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    rows = payload["rows"]
    epochs = np.asarray([row["epoch"] for row in rows], dtype=float)
    values = {key: np.asarray([row[key] for row in rows], dtype=float) for key, _ in CHAIN}
    same12 = np.asarray([row["test_hr_same12"] for row in rows], dtype=float)
    same12_corr = corr(values["test_tf_hr"], same12)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    fig = plt.figure(figsize=(15.2, 7.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 5, height_ratios=[1.0, 1.25])

    ax_trace = fig.add_subplot(grid[0, :])
    colors = ["#1f4e79", "#4f81bd", "#70ad47", "#ed7d31", "#a64d79"]
    for (key, label), color in zip(CHAIN, colors):
        ax_trace.plot(
            epochs,
            zscore(values[key]),
            marker="o",
            markersize=3.5,
            linewidth=1.8,
            label=label,
            color=color,
        )
    ax_trace.axhline(0.0, color="#999999", linewidth=0.7)
    ax_trace.set_title("Ordered checkpoint chain: lower is better for every variable")
    ax_trace.set_xlabel("Epoch")
    ax_trace.set_ylabel("Per-variable z-score")
    ax_trace.set_xticks(epochs.astype(int))
    ax_trace.grid(axis="y", alpha=0.25)
    ax_trace.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False)

    for idx in range(len(CHAIN) - 1):
        left_key, left_label = CHAIN[idx]
        right_key, right_label = CHAIN[idx + 1]
        ax = fig.add_subplot(grid[1, idx])
        x = values[left_key]
        y = values[right_key]
        p, s, k = corr(x, y)
        ax.scatter(x, y, s=28, color=colors[idx + 1], edgecolor="white", linewidth=0.45, zorder=3)
        for epoch, xv, yv in zip(epochs.astype(int), x, y):
            ax.annotate(str(epoch), (xv, yv), xytext=(3, 3), textcoords="offset points", fontsize=6.5)
        slope, intercept = np.polyfit(x, y, deg=1)
        xx = np.linspace(x.min(), x.max(), 100)
        ax.plot(xx, slope * xx + intercept, color="#555555", linewidth=1.0, linestyle="--")
        ax.set_title(f"{left_label} $\\rightarrow$ {right_label}", fontsize=9.5)
        ax.set_xlabel(left_label)
        ax.set_ylabel(right_label)
        ax.grid(alpha=0.2)
        ax.text(
            0.04,
            0.96,
            f"Pearson = {p:.3f}\nSpearman = {s:.3f}\nKendall = {k:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=7.5,
            bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.86, "pad": 3},
        )

    # The empty fourth slot is used as a compact bottleneck callout.
    ax_note = fig.add_subplot(grid[1, 4])
    ax_note.axis("off")
    ax_note.text(
        0.02,
        0.92,
        "Adjacent-edge diagnosis",
        fontsize=10,
        fontweight="bold",
        va="top",
    )
    ax_note.text(
        0.02,
        0.78,
        "Weakest linear link: valid loss\n"
        "to valid TF HR (P=0.670), while\n"
        "rank agreement remains high\n"
        "(S=0.893).\n\n"
        "The final TF-to-sampled-MIDI\n"
        "link also weakens (P=0.795);\n"
        "same-score subset is stronger.\n\n"
        f"Same 12-score subset: P={same12_corr[0]:.3f},\n"
        f"S={same12_corr[1]:.3f}, K={same12_corr[2]:.3f}.",
        fontsize=9,
        va="top",
        linespacing=1.45,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    print(json.dumps({"output": str(args.output), "rows": len(rows)}))


if __name__ == "__main__":
    main()
