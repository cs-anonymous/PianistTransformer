#!/usr/bin/env python3
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
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

from src.evaluate.plot_target_distribution_diagnostic import (  # noqa: E402
    as_bool,
    load_json,
    quantile_summary,
    sampled_long,
    unique_paths,
)


RAW_TARGETS = [
    ("ioi_dev", "IOI ms"),
    ("duration_dev", "Duration ms"),
    ("velocity", "Velocity"),
    ("pedal_0", "Pedal 0"),
    ("pedal_1", "Pedal 25"),
    ("pedal_2", "Pedal 50"),
    ("pedal_3", "Pedal 75"),
]


def load_asap_metadata(metadata_path):
    meta = pd.read_csv(metadata_path)
    return meta[
        as_bool(meta["tier_a"])
        & as_bool(meta["is_refined"])
        & (meta["performance_dataset"].fillna("").astype(str) == "ASAP")
        & meta["split"].isin(["train", "test"])
        & meta["refined_score_midi_path"].notna()
        & meta["refined_performance_midi_path"].notna()
    ].copy()


def processed_json_path(processed_dir, score_source):
    return Path(processed_dir) / Path(str(score_source)).with_suffix(".json")


def perf_raw7_frame(perf, source, split, score_source, performance_source):
    shared_rows = perf.get("label_shared_raw")
    pedal_rows = perf.get("label_pedal4_raw") or perf.get("pedal4_raw")
    if shared_rows is None or pedal_rows is None:
        raw_rows = perf.get("label_raw")
        if not raw_rows:
            return pd.DataFrame()
        shared_rows = [row[:3] for row in raw_rows]
        pedal_rows = [row[3:7] for row in raw_rows]
    arr = np.asarray(shared_rows, dtype=np.float64)
    pedal = np.asarray(pedal_rows, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 3 or pedal.ndim != 2 or pedal.shape[1] < 4:
        return pd.DataFrame()
    n = min(len(arr), len(pedal))
    arr = arr[:n]
    pedal = (pedal[:n, :4] >= 64.0).astype(np.float64) * 127.0
    return pd.DataFrame(
        {
            "source": source,
            "split": split,
            "score_source": score_source,
            "performance_source": performance_source,
            "ioi_dev": arr[:, 0],
            "duration_dev": arr[:, 1],
            "velocity": arr[:, 2],
            "pedal_0": pedal[:, 0],
            "pedal_1": pedal[:, 1],
            "pedal_2": pedal[:, 2],
            "pedal_3": pedal[:, 3],
        }
    )


def process_score_gt_raw(args):
    score_source, rows, processed_dir = args
    work_path = processed_json_path(processed_dir, score_source)
    if not work_path.exists():
        return []
    work = load_json(work_path)
    perf_by_source = {
        str(perf.get("performance_source")): perf
        for perf in work.get("performances", [])
        if perf.get("performance_source") is not None
    }
    out = []
    for row in rows:
        perf_source = str(row["refined_performance_midi_path"])
        perf = perf_by_source.get(perf_source)
        if perf is None:
            continue
        split = str(row["split"])
        frame = perf_raw7_frame(
            perf,
            source=f"ASAP {split}",
            split=split,
            score_source=str(score_source),
            performance_source=perf_source,
        )
        if not frame.empty:
            out.append(frame)
    return out


def collect_gt_raw(metadata_path, processed_dir, num_workers):
    meta = load_asap_metadata(metadata_path)
    tasks = [
        (score_source, score_df.to_dict("records"), str(processed_dir))
        for score_source, score_df in meta.groupby("refined_score_midi_path", sort=False)
    ]
    frames = []
    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = [pool.submit(process_score_gt_raw, task) for task in tasks]
            for fut in as_completed(futures):
                frames.extend(fut.result())
    else:
        for task in tasks:
            frames.extend(process_score_gt_raw(task))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def collect_pred_raw(manifest_path, source):
    manifest = load_json(manifest_path)
    frames = []
    for path in unique_paths(manifest, "raw_output_paths"):
        payload = load_json(path)
        arr = np.asarray(payload.get("reconstructed_raw7") or [], dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] < 7:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "source": source,
                    "split": "pred",
                    "score_source": payload.get("score_source"),
                    "performance_source": Path(path).name,
                    "ioi_dev": arr[:, 0],
                    "duration_dev": arr[:, 1],
                    "velocity": arr[:, 2],
                    "pedal_0": arr[:, 3],
                    "pedal_1": arr[:, 4],
                    "pedal_2": arr[:, 5],
                    "pedal_3": arr[:, 6],
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_group(long_df, output_prefix, title):
    sources = list(long_df["source"].drop_duplicates())
    palette = dict(zip(sources, sns.color_palette("tab10", n_colors=max(len(sources), 1))))
    for kind in ("hist", "ecdf"):
        fig, axes = plt.subplots(3, 3, figsize=(16, 11), constrained_layout=True)
        axes = axes.flatten()
        for ax, (target, target_title) in zip(axes, RAW_TARGETS):
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
        for ax in axes[len(RAW_TARGETS):]:
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
    gt_df = collect_gt_raw(
        metadata_path=Path(config["metadata_path"]),
        processed_dir=Path(config["refined_dir"]),
        num_workers=args.num_workers,
    )
    frames = [gt_df]
    for label, manifest_path in args.manifest:
        frames.append(collect_pred_raw(Path(manifest_path), label))
    all_df = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    if all_df.empty:
        raise SystemExit("No raw distribution rows collected")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = quantile_summary(all_df)
    summary.to_csv(args.output_dir / "raw7_distribution_summary.csv", index=False)
    long_df = sampled_long(all_df, args.sample_per_source_target, args.seed)
    long_df.to_csv(args.output_dir / "raw7_distribution_plot_sample.csv", index=False)
    plot_group(long_df, args.output_dir / "raw7_distribution", "slot0710 cheap15 raw7 distribution")

    meta = {
        "config": str(args.config),
        "manifests": [{"label": label, "path": path} for label, path in args.manifest],
        "counts": {source: int(len(sub)) for source, sub in all_df.groupby("source", sort=False)},
        "space": "raw7: ioi_ms, duration_ms, velocity_0_127, pedal_binary_0_127",
    }
    (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
