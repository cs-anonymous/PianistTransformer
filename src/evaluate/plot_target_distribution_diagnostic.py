#!/usr/bin/env python
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.train.train_inr import performance_dev_velocity_pedal4_binary_rows


TARGETS = [
    ("ioi_dev", "IOI dev target"),
    ("duration_dev", "Duration dev target"),
    ("velocity", "Velocity target"),
    ("pedal_0", "Pedal binary 0 target"),
    ("pedal_1", "Pedal binary 1 target"),
    ("pedal_2", "Pedal binary 2 target"),
    ("pedal_3", "Pedal binary 3 target"),
]


def as_bool(series):
    if series.dtype == bool:
        return series
    return series.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


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


def process_score_gt(args):
    (
        score_source,
        rows,
        processed_dir,
        epr_timing_target,
        log_scale,
    ) = args
    work_path = processed_json_path(processed_dir, score_source)
    if not work_path.exists():
        return []
    work = load_json(work_path)
    score_raw = (work.get("score") or {}).get("score_raw")
    if not score_raw:
        return []
    perf_by_source = {
        str(perf.get("performance_source")): perf
        for perf in work.get("performances", [])
        if perf.get("performance_source") is not None
    }
    out = []
    for row in rows:
        perf = perf_by_source.get(str(row["refined_performance_midi_path"]))
        if perf is None:
            continue
        try:
            target_rows = performance_dev_velocity_pedal4_binary_rows(
                perf,
                score_raw,
                epr_timing_target=epr_timing_target,
                log_scale=log_scale,
            )
        except Exception:
            continue
        split = str(row["split"])
        arr = np.asarray(target_rows, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] < 7:
            continue
        df = pd.DataFrame(
            {
                "source": f"ASAP {split}",
                "split": split,
                "score_source": str(score_source),
                "performance_source": str(row["refined_performance_midi_path"]),
                "ioi_dev": arr[:, 0],
                "duration_dev": arr[:, 1],
                "velocity": arr[:, 2],
                "pedal_0": arr[:, 3],
                "pedal_1": arr[:, 4],
                "pedal_2": arr[:, 5],
                "pedal_3": arr[:, 6],
            }
        )
        out.append(df)
    return out


def collect_gt(
    metadata_path,
    processed_dir,
    epr_timing_target,
    log_scale,
    num_workers,
):
    meta = load_asap_metadata(metadata_path)
    tasks = []
    for score_source, score_df in meta.groupby("refined_score_midi_path", sort=False):
        tasks.append(
            (
                score_source,
                score_df.to_dict("records"),
                str(processed_dir),
                epr_timing_target,
                float(log_scale),
            )
        )
    frames = []
    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = [pool.submit(process_score_gt, task) for task in tasks]
            for fut in as_completed(futures):
                frames.extend(fut.result())
    else:
        for task in tasks:
            frames.extend(process_score_gt(task))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def unique_paths(manifest, key):
    out = []
    seen = set()
    for item in manifest["items"]:
        for path in item.get(key, []):
            resolved = str(Path(path).resolve())
            if resolved not in seen:
                seen.add(resolved)
                out.append(resolved)
    return out


def collect_pred(raw_output_paths, source):
    frames = []
    for path in raw_output_paths:
        payload = load_json(path)
        rows = payload.get("predicted_target7") or []
        arr = np.asarray(rows, dtype=np.float64)
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


def quantile_summary(df):
    rows = []
    for source, sub in df.groupby("source", sort=False):
        for target, _ in TARGETS:
            vals = sub[target].to_numpy(dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            qs = np.quantile(vals, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "count": int(len(vals)),
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "p01": float(qs[0]),
                    "p05": float(qs[1]),
                    "p25": float(qs[2]),
                    "p50": float(qs[3]),
                    "p75": float(qs[4]),
                    "p95": float(qs[5]),
                    "p99": float(qs[6]),
                    "clip0_frac": float(np.mean(vals <= 1e-8)),
                    "clip1_frac": float(np.mean(vals >= 1.0 - 1e-8)),
                }
            )
    return pd.DataFrame(rows)


def sampled_long(df, sample_per_source_target, seed):
    rng = np.random.default_rng(seed)
    parts = []
    for source, sub in df.groupby("source", sort=False):
        for target, title in TARGETS:
            vals = sub[target].to_numpy(dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if len(vals) > sample_per_source_target:
                vals = rng.choice(vals, size=sample_per_source_target, replace=False)
            parts.append(pd.DataFrame({"source": source, "target": target, "title": title, "value": vals}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def plot_hist_ecdf(long_df, output_prefix):
    palette = {
        "ASAP train": "#222222",
        "ASAP test": "#777777",
        "pred det": "#2f6fed",
        "pred samp": "#d0522b",
    }
    order = ["ASAP train", "ASAP test", "pred det", "pred samp"]
    for kind in ("hist", "ecdf"):
        fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), constrained_layout=True)
        axes = axes.flatten()
        for ax, (target, title) in zip(axes, TARGETS):
            sub = long_df[long_df["target"] == target]
            if kind == "ecdf":
                sns.ecdfplot(data=sub, x="value", hue="source", hue_order=order, palette=palette, ax=ax)
                ax.set_ylabel("ECDF")
            else:
                sns.histplot(
                    data=sub,
                    x="value",
                    hue="source",
                    hue_order=order,
                    palette=palette,
                    stat="density",
                    common_norm=False,
                    element="step",
                    fill=False,
                    bins=120,
                    ax=ax,
                )
                ax.set_ylabel("density")
            ax.set_title(title)
            ax.set_xlim(0.0, 1.0)
            ax.grid(alpha=0.2)
        axes[-1].set_axis_off()
        fig.suptitle(f"Target Distribution Diagnostic ({kind.upper()})")
        fig.savefig(output_prefix.with_name(f"{output_prefix.name}_{kind}.png"), dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot ASAP train/test vs prediction target distributions.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--det-manifest", type=Path, required=True)
    parser.add_argument("--sampling-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--sample-per-source-target", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_json(args.config)
    det_manifest = load_json(args.det_manifest)
    sampling_manifest = load_json(args.sampling_manifest)
    epr_timing_target = config.get("epr_timing_target", "log_deviation")
    log_scale = float(config.get("timing_log_scale", 50.0))

    gt_df = collect_gt(
        metadata_path=ROOT_DIR / config["metadata_path"],
        processed_dir=ROOT_DIR / config["refined_dir"],
        epr_timing_target=epr_timing_target,
        log_scale=log_scale,
        num_workers=args.num_workers,
    )
    det_df = collect_pred(unique_paths(det_manifest, "raw_output_paths"), "pred det")
    sampling_df = collect_pred(unique_paths(sampling_manifest, "raw_output_paths"), "pred samp")
    all_df = pd.concat([gt_df, det_df, sampling_df], ignore_index=True)
    if all_df.empty:
        raise SystemExit("No rows collected.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = quantile_summary(all_df)
    summary.to_csv(args.output_dir / "target_distribution_summary.csv", index=False)
    long_df = sampled_long(all_df, args.sample_per_source_target, args.seed)
    long_df.to_csv(args.output_dir / "target_distribution_plot_sample.csv", index=False)
    plot_hist_ecdf(long_df, args.output_dir / "target_distribution")

    meta = {
        "config": str(args.config.resolve()),
        "det_manifest": str(args.det_manifest.resolve()),
        "sampling_manifest": str(args.sampling_manifest.resolve()),
        "epr_timing_target": epr_timing_target,
        "timing_log_scale": log_scale,
        "counts": {source: int(len(sub)) for source, sub in all_df.groupby("source", sort=False)},
    }
    (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    key = summary[summary["target"].isin(["ioi_dev", "duration_dev", "velocity", "pedal_start", "pedal_ctrl"])]
    print(key[["source", "target", "count", "mean", "std", "p05", "p50", "p95", "clip0_frac", "clip1_frac"]].to_string(index=False))


if __name__ == "__main__":
    main()
