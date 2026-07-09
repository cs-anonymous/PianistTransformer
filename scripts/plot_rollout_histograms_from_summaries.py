import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FEATURES = [
    ("ioi_log_dev", "IOI log-dev", "target"),
    ("duration_log_dev", "Duration log-dev", "target"),
    ("velocity_norm", "Velocity norm", "target"),
    ("ioi_ms", "IOI ms", "raw"),
    ("duration_ms", "Duration ms", "raw"),
    ("velocity", "Velocity raw", "raw"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot direct histograms from rollout summary distribution values.")
    parser.add_argument("--finite-summary", type=Path, required=True)
    parser.add_argument("--full-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--groups", default="GT,k=0,k=1,k=2,k=4,k=8,k=16,full AR")
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--clip-low", type=float, default=0.5)
    parser.add_argument("--clip-high", type=float, default=99.5)
    return parser.parse_args()


def load_summary(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def collect_group(summary, label, domain, feature):
    items = summary.get("items", [])
    if not items:
        return np.asarray([], dtype=np.float64)
    rollout_labels = summary.get("rollout_ks", [])
    if label == "GT":
        source_label = rollout_labels[0]
        kind = "gt"
    elif label == "full AR":
        source_label = "full"
        kind = "pred"
    else:
        source_label = label.replace("k=", "")
        kind = "pred"
    values = []
    for item in items:
        dist = item.get("by_k", {}).get(source_label, {}).get("distributions", {})
        values.extend(dist.get(domain, {}).get(kind, {}).get(feature, []))
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def merged_groups(finite_summary, full_summary, labels, domain, feature):
    output = {}
    for label in labels:
        if label == "full AR":
            output[label] = collect_group(full_summary, label, domain, feature)
        else:
            output[label] = collect_group(finite_summary, label, domain, feature)
    return output


def clipped_edges(groups, bins, low, high):
    pooled = np.concatenate([values for values in groups.values() if len(values)])
    if len(pooled) == 0:
        return np.linspace(0.0, 1.0, bins + 1)
    lo, hi = np.percentile(pooled, [low, high])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo, hi = float(np.nanmin(pooled)), float(np.nanmax(pooled))
    if lo >= hi:
        hi = lo + 1.0
    return np.linspace(float(lo), float(hi), bins + 1)


def plot_overlay(path, groups_by_feature, labels, bins, clip_low, clip_high):
    colors = {
        "GT": "#111111",
        "k=0": "#4c78a8",
        "k=1": "#f58518",
        "k=2": "#7f6dba",
        "k=4": "#54a24b",
        "k=8": "#e45756",
        "k=16": "#72b7b2",
        "full AR": "#b83232",
    }
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes = axes.flatten()
    for axis, (feature, title, _domain) in zip(axes, FEATURES):
        groups = groups_by_feature[feature]
        edges = clipped_edges(groups, bins, clip_low, clip_high)
        for label in labels:
            values = groups.get(label, np.asarray([]))
            values = values[(values >= edges[0]) & (values <= edges[-1])]
            if len(values) == 0:
                continue
            histtype = "step" if label in {"GT", "full AR"} else "bar"
            alpha = 0.95 if label in {"GT", "full AR"} else 0.22
            linewidth = 2.5 if label in {"GT", "full AR"} else 1.0
            axis.hist(
                values,
                bins=edges,
                density=True,
                histtype=histtype,
                alpha=alpha,
                linewidth=linewidth,
                color=colors.get(label),
                label=label,
            )
        axis.set_title(title)
        axis.grid(alpha=0.18)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, legend_labels, loc="upper center", ncol=min(len(legend_labels), 8), frameon=False)
    fig.suptitle("Direct histograms: GT vs sample rollout", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_facets(path, groups_by_feature, labels, bins, clip_low, clip_high):
    fig, axes = plt.subplots(len(FEATURES), len(labels), figsize=(2.8 * len(labels), 10.5), sharey="row")
    for row, (feature, title, _domain) in enumerate(FEATURES):
        groups = groups_by_feature[feature]
        edges = clipped_edges(groups, bins, clip_low, clip_high)
        for col, label in enumerate(labels):
            axis = axes[row, col]
            values = groups.get(label, np.asarray([]))
            values = values[(values >= edges[0]) & (values <= edges[-1])]
            if len(values):
                axis.hist(values, bins=edges, density=True, color="#4c78a8" if label != "GT" else "#111111", alpha=0.78)
            if row == 0:
                axis.set_title(label)
            if col == 0:
                axis.set_ylabel(title)
            axis.grid(alpha=0.14)
    fig.suptitle("Direct histogram facets", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_stats(path, groups_by_feature, labels):
    lines = ["feature,group,n,mean,std,p01,p50,p99"]
    for feature, _title, _domain in FEATURES:
        for label in labels:
            values = groups_by_feature[feature].get(label, np.asarray([]))
            if len(values) == 0:
                lines.append(f"{feature},{label},0,nan,nan,nan,nan,nan")
                continue
            lines.append(
                ",".join(
                    [
                        feature,
                        label,
                        str(int(len(values))),
                        f"{float(np.mean(values)):.8g}",
                        f"{float(np.std(values)):.8g}",
                        f"{float(np.percentile(values, 1)):.8g}",
                        f"{float(np.percentile(values, 50)):.8g}",
                        f"{float(np.percentile(values, 99)):.8g}",
                    ]
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    finite = load_summary(args.finite_summary)
    full = load_summary(args.full_summary)
    labels = [item.strip() for item in args.groups.split(",") if item.strip()]
    groups_by_feature = {
        feature: merged_groups(finite, full, labels, domain, feature)
        for feature, _title, domain in FEATURES
    }
    plot_overlay(args.output_dir / "rollout_hist_overlay.png", groups_by_feature, labels, args.bins, args.clip_low, args.clip_high)
    plot_facets(args.output_dir / "rollout_hist_facets.png", groups_by_feature, labels, args.bins, args.clip_low, args.clip_high)
    write_stats(args.output_dir / "rollout_hist_stats.csv", groups_by_feature, labels)
    print(json.dumps({"output_dir": str(args.output_dir), "labels": labels}, indent=2))


if __name__ == "__main__":
    main()
