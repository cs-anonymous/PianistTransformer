#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


TARGETS = ("ioi_delta", "duration_delta", "duration_ratio", "log_duration_ratio")
GROUPS = ("ASAP", "nonASAP")


def iter_performance_pairs(processed_dir, max_files=None, shuffle_files=False, seed=42):
    paths = sorted(Path(processed_dir).rglob("*.json"))
    if shuffle_files:
        rng = np.random.default_rng(seed)
        paths = list(rng.permutation(paths))
    if max_files is not None:
        paths = paths[:max_files]

    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "score" not in data or "performances" not in data:
            continue
        score_raw = data["score"].get("score_raw")
        if not score_raw:
            continue
        score = np.asarray(score_raw, dtype=np.float32)
        if score.ndim != 2 or score.shape[1] < 2:
            continue
        score_ioi = score[:, 0]
        score_dur = score[:, 1]
        note_count = len(score)

        for perf in data.get("performances", []):
            label_raw = perf.get("label_raw")
            if not label_raw:
                continue
            label = np.asarray(label_raw, dtype=np.float32)
            if label.ndim != 2 or label.shape[1] < 2 or len(label) != note_count:
                continue
            group = "ASAP" if str(perf.get("performance_dataset", "")) == "ASAP" else "nonASAP"
            yield {
                "group": group,
                "score_ioi": score_ioi,
                "score_dur": score_dur,
                "perf_ioi": label[:, 0],
                "perf_dur": label[:, 1],
            }


def collect_samples(processed_dir, max_files=None, shuffle_files=False, sample_per_kind=200_000, chunk_sample_size=4096, seed=42):
    rng = np.random.default_rng(seed)
    buckets = {}
    observed = {}
    pair_count = 0
    note_count = 0

    for row in iter_performance_pairs(processed_dir, max_files=max_files, shuffle_files=shuffle_files, seed=seed):
        pair_count += 1
        n = len(row["score_ioi"])
        note_count += n
        score_dur = np.maximum(row["score_dur"], 1.0)
        values = {
            "ioi_delta": row["perf_ioi"] - row["score_ioi"],
            "duration_delta": row["perf_dur"] - row["score_dur"],
            "duration_ratio": row["perf_dur"] / score_dur,
            "log_duration_ratio": np.log(np.maximum(row["perf_dur"], 1.0) / score_dur),
        }
        for target, arr in values.items():
            key = (row["group"], target)
            arr = np.asarray(arr, dtype=np.float32)
            arr = arr[np.isfinite(arr)]
            if len(arr) == 0:
                continue
            observed[key] = observed.get(key, 0) + len(arr)
            if chunk_sample_size is not None and len(arr) > chunk_sample_size:
                arr = rng.choice(arr, size=chunk_sample_size, replace=False)
            buckets.setdefault(key, []).append(arr.astype(np.float32, copy=False))

    frames = []
    for (group, target), parts in buckets.items():
        values = np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)
        if len(values) > sample_per_kind:
            values = rng.choice(values, size=sample_per_kind, replace=False).astype(np.float32, copy=False)
        frames.append(pd.DataFrame({
            "group": group,
            "target": target,
            "value": values,
            "observed_count": observed.get((group, target), len(values)),
        }))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    meta = {"performance_pairs_seen": pair_count, "note_rows_seen": note_count}
    return df, meta


def summarize(df):
    rows = []
    for (group, target), sub in df.groupby(["group", "target"], sort=True):
        values = sub["value"].to_numpy()
        qs = np.quantile(values, [0, .001, .01, .05, .25, .5, .75, .95, .99, .999, 1])
        rows.append({
            "group": group,
            "target": target,
            "observed_count": int(sub["observed_count"].iloc[0]),
            "sample_count": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(qs[0]),
            "p0_1": float(qs[1]),
            "p1": float(qs[2]),
            "p5": float(qs[3]),
            "p25": float(qs[4]),
            "p50": float(qs[5]),
            "p75": float(qs[6]),
            "p95": float(qs[7]),
            "p99": float(qs[8]),
            "p99_9": float(qs[9]),
            "max": float(qs[10]),
            "abs_gt_100_frac": float(np.mean(np.abs(values) > 100)),
            "abs_gt_500_frac": float(np.mean(np.abs(values) > 500)),
            "abs_gt_1000_frac": float(np.mean(np.abs(values) > 1000)),
            "gt_4_frac": float(np.mean(values > 4)),
            "lt_0_25_frac": float(np.mean(values < 0.25)),
        })
    return pd.DataFrame(rows)


def clipped_for_plot(df, quantile=0.995):
    out = []
    for (_, target), sub in df.groupby(["group", "target"], sort=False):
        if target in {"duration_ratio"}:
            lo, hi = sub["value"].quantile(0.001), sub["value"].quantile(quantile)
        else:
            lo, hi = sub["value"].quantile(1 - quantile), sub["value"].quantile(quantile)
        out.append(sub[(sub["value"] >= lo) & (sub["value"] <= hi)])
    return pd.concat(out, ignore_index=True)


def plot_hist(df, output_path):
    plot_df = clipped_for_plot(df)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for ax, target in zip(axes.flat, TARGETS):
        sub = plot_df[plot_df["target"] == target]
        sns.histplot(
            data=sub,
            x="value",
            hue="group",
            stat="density",
            common_norm=False,
            element="step",
            fill=False,
            bins=180,
            ax=ax,
        )
        ax.set_title(target)
        ax.grid(alpha=0.2)
    fig.suptitle("Timing Target Candidate Distributions (clipped to central 99.5%)")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_ecdf(df, output_path):
    plot_df = clipped_for_plot(df)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for ax, target in zip(axes.flat, TARGETS):
        sub = plot_df[plot_df["target"] == target]
        sns.ecdfplot(data=sub, x="value", hue="group", ax=ax)
        ax.set_title(target)
        ax.grid(alpha=0.2)
    fig.suptitle("Timing Target Candidate ECDF (clipped to central 99.5%)")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze timing delta and ratio target distributions.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/ASAP_processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/plots/timing_delta_ratio"))
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--shuffle-files", action="store_true")
    parser.add_argument("--sample-per-kind", type=int, default=200_000)
    parser.add_argument("--chunk-sample-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sns.set_theme(style="whitegrid")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df, meta = collect_samples(
        args.processed_dir,
        max_files=args.max_files,
        shuffle_files=args.shuffle_files,
        sample_per_kind=args.sample_per_kind,
        chunk_sample_size=args.chunk_sample_size,
        seed=args.seed,
    )
    if df.empty:
        raise SystemExit("No timing target rows collected.")
    summary = summarize(df)
    df.to_parquet(args.output_dir / "timing_delta_ratio_sample.parquet", index=False)
    df.to_csv(args.output_dir / "timing_delta_ratio_sample.csv", index=False)
    summary.to_csv(args.output_dir / "timing_delta_ratio_summary.csv", index=False)
    (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    plot_hist(df, args.output_dir / "timing_delta_ratio_hist.png")
    plot_ecdf(df, args.output_dir / "timing_delta_ratio_ecdf.png")
    print(f"Wrote {args.output_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
