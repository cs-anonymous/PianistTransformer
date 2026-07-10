#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluate.plot_target_distribution_diagnostic import (
    collect_gt,
    load_json,
    quantile_summary,
    sampled_long,
    unique_paths,
)


TARGETS = [
    ("timing_0", "Timing 0"),
    ("timing_1", "Timing 1"),
    ("velocity", "Velocity"),
    ("pedal_0", "Pedal 0"),
    ("pedal_1", "Pedal 25"),
    ("pedal_2", "Pedal 50"),
    ("pedal_3", "Pedal 75"),
]


def collect_pred(manifest_path, source):
    manifest = load_json(manifest_path)
    frames = []
    for path in unique_paths(manifest, "raw_output_paths"):
        payload = load_json(path)
        arr = np.asarray(payload.get("predicted_target7") or [], dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] < 7:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "source": source,
                    "score_source": payload.get("score_source"),
                    "timing_0": arr[:, 0],
                    "timing_1": arr[:, 1],
                    "velocity": arr[:, 2],
                    "pedal_0": arr[:, 3],
                    "pedal_1": arr[:, 4],
                    "pedal_2": arr[:, 5],
                    "pedal_3": arr[:, 6],
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def rename_gt_columns(df):
    if df.empty:
        return df
    renamed = df.rename(
        columns={
            "ioi_dev": "timing_0",
            "duration_dev": "timing_1",
        }
    ).copy()
    keep = ["source", "score_source", *[name for name, _ in TARGETS]]
    return renamed[keep]


def plot_group(long_df, output_prefix, title):
    sources = list(long_df["source"].drop_duplicates())
    palette = dict(zip(sources, sns.color_palette("tab10", n_colors=max(len(sources), 1))))
    for kind in ("hist", "ecdf"):
        fig, axes = plt.subplots(3, 3, figsize=(16, 11), constrained_layout=True)
        axes = axes.flatten()
        for ax, (target, target_title) in zip(axes, TARGETS):
            sub = long_df[long_df["target"] == target]
            if kind == "ecdf":
                sns.ecdfplot(data=sub, x="value", hue="source", palette=palette, ax=ax)
                ax.set_ylabel("ECDF")
            else:
                sns.histplot(
                    data=sub,
                    x="value",
                    hue="source",
                    palette=palette,
                    stat="density",
                    common_norm=False,
                    element="step",
                    fill=False,
                    bins=120,
                    ax=ax,
                )
                ax.set_ylabel("density")
            ax.set_title(target_title)
            ax.grid(alpha=0.2)
        for ax in axes[len(TARGETS):]:
            ax.set_axis_off()
        fig.suptitle(f"{title} ({kind.upper()})")
        fig.savefig(output_prefix.with_name(f"{output_prefix.name}_{kind}.png"), dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", action="append", nargs=2, metavar=("LABEL", "PATH"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--sample-per-source-target", type=int, default=120_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_json(args.config)
    gt_df = collect_gt(
        metadata_path=ROOT_DIR / config["metadata_path"],
        processed_dir=ROOT_DIR / config["refined_dir"],
        epr_timing_target=config.get("epr_timing_target", "log_deviation"),
        log_scale=float(config.get("timing_log_scale", 50.0)),
        num_workers=args.num_workers,
    )
    frames = [rename_gt_columns(gt_df)]
    for label, manifest_path in args.manifest:
        frames.append(collect_pred(Path(manifest_path), label))
    all_df = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    if all_df.empty:
        raise SystemExit("No distribution rows collected")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = quantile_summary(
        all_df.rename(columns={"timing_0": "ioi_dev", "timing_1": "duration_dev"})
    )
    summary.to_csv(args.output_dir / "target7_distribution_summary.csv", index=False)

    long_df = sampled_long(
        all_df.rename(columns={"timing_0": "ioi_dev", "timing_1": "duration_dev"}),
        args.sample_per_source_target,
        args.seed,
    ).replace({"target": {"ioi_dev": "timing_0", "duration_dev": "timing_1"}})
    long_df.to_csv(args.output_dir / "target7_distribution_plot_sample.csv", index=False)
    plot_group(long_df, args.output_dir / "target7_distribution", "slot0710 cheap15 target7 distribution")

    meta = {
        "config": str(args.config),
        "manifests": [{"label": label, "path": path} for label, path in args.manifest],
        "counts": {source: int(len(sub)) for source, sub in all_df.groupby("source", sort=False)},
    }
    (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
