#!/usr/bin/env python
import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from script.prebuild_inr_work_pt import make_manifest, unique_work_paths
from src.train.train_inr import normalize_log_timing_dev


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def collect_rows_from_sidecar(config, split, performance_dataset, sidecar_tag):
    manifest = make_manifest(config, split, performance_dataset_override=performance_dataset)
    paths = [Path(path) for path in unique_work_paths(manifest)]
    rows = []
    for path in paths:
        sidecar = path.with_suffix(f".{sidecar_tag}.pt") if sidecar_tag else path.with_suffix(".pt")
        if not sidecar.exists():
            raise FileNotFoundError(f"Missing sidecar: {sidecar}")
        payload = torch_load(sidecar)
        score_raw = payload["score"]["score_raw"]
        score_ioi = np.asarray([row[0] for row in score_raw], dtype=np.float64)
        score_norm_s50 = np.asarray(
            [normalize_log_timing_dev(0.0, value, scale=50.0, max_time_ms=5000.0) - 0.5 for value in score_ioi],
            dtype=np.float64,
        )
        for perf in payload["performances"]:
            labels = np.asarray(perf["labels"], dtype=np.float64)
            if len(labels) != len(score_ioi):
                continue
            # Convert the stored s=50 target back to perf IOI ms. For raw dev
            # and other s values we want the underlying time, not the stored code.
            perf_norm_s50 = score_norm_s50 + (labels[:, 0] - 0.5)
            perf_ioi = (np.expm1(np.maximum(perf_norm_s50, 0.0) * np.log1p(5000.0 / 50.0)) * 50.0)
            raw_dev = perf_ioi - score_ioi
            group = np.where(score_ioi <= 1e-9, "zero_ioi", "nz_ioi")
            frame = pd.DataFrame(
                {
                    "split": split,
                    "group": group,
                    "score_ioi_ms": score_ioi,
                    "perf_ioi_ms": perf_ioi,
                    "raw_dev_ms": raw_dev,
                }
            )
            rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def add_scaled_columns(df, scales):
    for scale in scales:
        df[f"logdev_s{scale:g}"] = [
            normalize_log_timing_dev(score, perf, scale=scale, max_time_ms=5000.0)
            for score, perf in zip(df["score_ioi_ms"].to_numpy(), df["perf_ioi_ms"].to_numpy())
        ]
    return df


def summarize(df, scales):
    rows = []
    columns = ["raw_dev_ms"] + [f"logdev_s{scale:g}" for scale in scales]
    quantiles = [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]
    for (split, group), sub in df.groupby(["split", "group"], sort=False):
        for column in columns:
            values = sub[column].to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            if len(values) == 0:
                continue
            qs = np.quantile(values, quantiles)
            row = {
                "split": split,
                "group": group,
                "value": column,
                "count": len(values),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "frac_lt_0": float(np.mean(values < 0.0)),
                "frac_lt_0_5": float(np.mean(values < 0.5)),
                "frac_le_0_5001": float(np.mean(values <= 0.5001)),
                "frac_gt_0_9": float(np.mean(values > 0.9)),
            }
            for q, value in zip(quantiles, qs):
                row[f"p{q * 100:g}"] = float(value)
            rows.append(row)
    return pd.DataFrame(rows)


def plot_split(df, split, scales, out_path):
    sub = df[df["split"] == split].copy()
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    groups = [("zero_ioi", "score IOI = 0"), ("nz_ioi", "score IOI > 0")]
    values = [("raw_dev_ms", "raw dev (ms)")] + [
        (f"logdev_s{scale:g}", f"log dev target, s={scale:g}") for scale in scales
    ]
    for row_idx, (group, group_title) in enumerate(groups):
        g = sub[sub["group"] == group]
        for col_idx, (column, title) in enumerate(values):
            ax = axes[row_idx, col_idx]
            vals = g[column].to_numpy(dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                ax.set_axis_off()
                continue
            if column == "raw_dev_ms":
                lo, hi = np.quantile(vals, [0.001, 0.999])
                pad = max((hi - lo) * 0.05, 1.0)
                bins = 120
                ax.hist(vals, bins=bins, range=(lo - pad, hi + pad), density=True, alpha=0.75)
                ax.axvline(0.0, color="black", linewidth=1, alpha=0.55)
                ax.set_xlim(lo - pad, hi + pad)
            else:
                ax.hist(vals, bins=120, range=(0, 1), density=True, alpha=0.75)
                ax.axvline(0.5, color="black", linewidth=1, alpha=0.55)
                ax.set_xlim(0, 1)
            ax.set_title(f"{group_title}\n{title}")
            ax.grid(True, alpha=0.2)
    fig.suptitle(f"ASAP {split}: zero vs non-zero score IOI deviation distributions", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="results/inr0624_head_capacity_4gpu/configs/mln3_cine_s50_headd4_w1_seed42.json")
    parser.add_argument("--output-dir", default="results/analysis/zero_nz_ioi_dev_scales")
    parser.add_argument("--scales", nargs="+", type=float, default=[10.0, 50.0, 100.0])
    parser.add_argument("--performance-dataset", default="ASAP")
    parser.add_argument("--sidecar-tag", default="ASAP")
    args = parser.parse_args()

    config = load_json(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for split in ["train", "test"]:
        print(f"collect {split}", flush=True)
        frames.append(collect_rows_from_sidecar(config, split, args.performance_dataset, args.sidecar_tag))
    df = pd.concat(frames, ignore_index=True)
    df = add_scaled_columns(df, args.scales)
    df.to_parquet(out_dir / "zero_nz_ioi_dev_values.parquet", index=False)

    summary = summarize(df, args.scales)
    summary.to_csv(out_dir / "zero_nz_ioi_dev_summary.csv", index=False)
    (out_dir / "zero_nz_ioi_dev_summary.json").write_text(
        json.dumps(summary.to_dict("records"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for split in ["train", "test"]:
        plot_split(
            df,
            split,
            args.scales,
            out_dir / f"zero_nz_ioi_dev_distributions_{split}_raw_s10_s50_s100.png",
        )
    print(out_dir)


if __name__ == "__main__":
    main()
