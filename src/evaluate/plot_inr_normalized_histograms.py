import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


FEATURES = ("ioi", "duration", "velocity", "pedal")
COLORS = {
    "label": "#222222",
    "fine_teacher": "#2b6cb0",
    "fine_free": "#dd6b20",
    "pine_teacher": "#2f855a",
    "pine_free": "#b83280",
}


def load_feature(path, group, feature):
    data = np.load(path)
    if feature == "pedal":
        chunks = [data[f"{group}__pedal_{idx}"] for idx in (0, 25, 50, 75)]
        return np.concatenate(chunks)
    return data[f"{group}__{feature}"]


def downsample(values, max_points, seed):
    if len(values) <= max_points:
        return values
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(values), size=max_points, replace=False)
    return values[idx]


def plot_histograms(series, output_dir, bins, max_points, seed):
    output_dir.mkdir(parents=True, exist_ok=True)
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    for feature in FEATURES:
        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
        for idx, item in enumerate(series):
            values = load_feature(item["path"], item["group"], feature)
            values = values[np.isfinite(values)]
            values = np.clip(values, 0.0, 1.0)
            values = downsample(values, max_points, seed + idx)
            ax.hist(
                values,
                bins=bin_edges,
                density=True,
                histtype="step",
                linewidth=1.8,
                alpha=0.95,
                color=item["color"],
                label=f"{item['label']} (n={len(values):,})",
            )
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel(f"{feature} normalized value")
        ax.set_ylabel("density")
        ax.set_title(f"Normalized {feature} distribution")
        ax.grid(True, alpha=0.18)
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        out = output_dir / f"{feature}_normalized_hist.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"Wrote {out}")


def parse_series(items):
    series = []
    for item in items:
        parts = item.split(":", 3)
        if len(parts) != 4:
            raise ValueError(
                "--series items must be label:npz_path:group:color_key_or_hex"
            )
        label, path, group, color = parts
        series.append(
            {
                "label": label,
                "path": Path(path),
                "group": group,
                "color": COLORS.get(color, color),
            }
        )
    return series


def parse_args():
    parser = argparse.ArgumentParser(description="Plot normalized INR distribution histograms.")
    parser.add_argument(
        "--series",
        action="append",
        required=True,
        help="label:npz_path:group:color_key_or_hex",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--max-points", type=int, default=500000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    plot_histograms(
        parse_series(args.series),
        output_dir=args.output_dir,
        bins=args.bins,
        max_points=args.max_points,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
