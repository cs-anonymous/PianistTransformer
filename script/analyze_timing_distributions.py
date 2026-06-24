#!/usr/bin/env python
import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


FEATURES = ("ioi", "duration")
SOURCES = ("score", "perf", "deviation")
GROUPS = ("ASAP", "nonASAP")


def timing_log_norm(values, max_ms=5000.0, scale=10.0):
    values = np.asarray(values, dtype=np.float64)
    clipped = np.clip(values, 0.0, max_ms)
    return np.log1p(clipped / scale) / math.log1p(max_ms / scale)


def timing_log_ms(values):
    values = np.asarray(values, dtype=np.float64)
    return np.log1p(np.clip(values, 0.0, None))


def signed_log1p(values):
    values = np.asarray(values, dtype=np.float64)
    return np.sign(values) * np.log1p(np.abs(values))


def iter_rows(processed_dir, max_files=None, shuffle_files=False, seed=42):
    json_paths = sorted(Path(processed_dir).rglob("*.json"))
    if shuffle_files:
        rng = np.random.default_rng(seed)
        json_paths = list(rng.permutation(json_paths))
    if max_files is not None:
        json_paths = json_paths[:max_files]
    for json_path in json_paths:
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "score" not in data or "performances" not in data:
            continue
        score = data["score"]
        score_raw = score.get("score_raw")
        if not score_raw:
            continue
        score_arr = np.asarray(score_raw, dtype=np.float32)
        if score_arr.ndim != 2 or score_arr.shape[1] < 2:
            continue
        score_ioi = score_arr[:, 0]
        score_dur = score_arr[:, 1]
        note_count = len(score_arr)
        score_source = score.get("score_source", "")

        for perf in data.get("performances", []):
            label_raw = perf.get("label_raw")
            if not label_raw:
                continue
            label_arr = np.asarray(label_raw, dtype=np.float32)
            if label_arr.ndim != 2 or label_arr.shape[1] < 2 or len(label_arr) != note_count:
                continue
            group = "ASAP" if str(perf.get("performance_dataset", "")) == "ASAP" else "nonASAP"
            perf_ioi = label_arr[:, 0]
            perf_dur = label_arr[:, 1]
            for feature, score_values, perf_values in (
                ("ioi", score_ioi, perf_ioi),
                ("duration", score_dur, perf_dur),
            ):
                yield {
                    "json_path": str(json_path),
                    "score_source": score_source,
                    "performance_source": perf.get("performance_source", ""),
                    "performance_dataset": perf.get("performance_dataset", ""),
                    "split": perf.get("split", ""),
                    "group": group,
                    "feature": feature,
                    "score": score_values.astype(np.float32, copy=False),
                    "perf": perf_values.astype(np.float32, copy=False),
                }


def collect_long_dataframe(
    processed_dir,
    max_files=None,
    sample_per_kind=250_000,
    chunk_sample_size=4096,
    shuffle_files=False,
    seed=42,
):
    rng = np.random.default_rng(seed)
    buckets = {}
    observed_counts = {}
    total_pairs = 0
    total_notes = 0

    for row in iter_rows(processed_dir, max_files=max_files, shuffle_files=shuffle_files, seed=seed):
        total_pairs += 1
        n = len(row["score"])
        total_notes += n
        values_by_source = {
            "score": row["score"],
            "perf": row["perf"],
            "deviation": row["perf"] - row["score"],
        }
        for source, values in values_by_source.items():
            key = (row["group"], row["feature"], source)
            values = np.asarray(values, dtype=np.float32)
            finite = values[np.isfinite(values)]
            if len(finite) == 0:
                continue
            observed_counts[key] = observed_counts.get(key, 0) + len(finite)
            if chunk_sample_size is not None and len(finite) > chunk_sample_size:
                finite = rng.choice(finite, size=chunk_sample_size, replace=False)
            buckets.setdefault(key, []).append(finite.astype(np.float32, copy=False))

    long_rows = []
    for (group, feature, source), parts in buckets.items():
        values = np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)
        if len(values) > sample_per_kind:
            values = rng.choice(values, size=sample_per_kind, replace=False).astype(np.float32, copy=False)
        log_norm = timing_log_norm(values) if source != "deviation" else (
            timing_log_norm(np.maximum(values, 0.0)) - timing_log_norm(np.maximum(-values, 0.0))
        )
        log_ms = timing_log_ms(values) if source != "deviation" else signed_log1p(values)
        long_rows.append(pd.DataFrame({
            "group": group,
            "feature": feature,
            "source": source,
            "value_ms": values,
            "log1p_ms": log_ms,
            "scaled_log_5000_s10": log_norm,
            "observed_count": observed_counts.get((group, feature, source), len(values)),
        }))

    long_df = pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame()
    summary_df = summarize_dataframe(long_df)
    meta = {"total_feature_pairs": total_pairs, "total_note_feature_rows": total_notes}
    return long_df, summary_df, meta


def summarize_values(group, feature, source, values):
    qs = np.quantile(values, [0, 0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1.0])
    return {
        "group": group,
        "feature": feature,
        "source": source,
        "count": len(values),
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
        "zero_frac": float(np.mean(values == 0)),
        "lt_10_frac": float(np.mean(values < 10)),
        "lt_50_frac": float(np.mean(values < 50)),
        "lt_100_frac": float(np.mean(values < 100)),
        "gt_1000_frac": float(np.mean(values > 1000)),
        "gt_5000_frac": float(np.mean(values > 5000)),
    }


def summarize_dataframe(df):
    if df.empty:
        return pd.DataFrame()
    out = []
    for (group, feature, source), sub in df.groupby(["group", "feature", "source"], sort=True):
        row = summarize_values(group, feature, source, sub["value_ms"].to_numpy())
        row["observed_count"] = int(sub["observed_count"].iloc[0]) if "observed_count" in sub else row["count"]
        out.append(row)
    return pd.DataFrame(out)


def plot_hist_grid(df, value_col, output_path, title, x_label, clip_quantile=None, bins=160):
    plot_df = df.copy()
    if clip_quantile is not None:
        clipped = []
        for _, sub in plot_df.groupby(["feature", "source"], sort=False):
            lo = sub[value_col].quantile(1.0 - clip_quantile) if sub["source"].iloc[0] == "deviation" else sub[value_col].min()
            hi = sub[value_col].quantile(clip_quantile)
            clipped.append(sub[(sub[value_col] >= lo) & (sub[value_col] <= hi)])
        plot_df = pd.concat(clipped, ignore_index=True)

    fig, axes = plt.subplots(len(FEATURES), len(SOURCES), figsize=(17, 8.5), constrained_layout=True)
    for i, feature in enumerate(FEATURES):
        for j, source in enumerate(SOURCES):
            ax = axes[i, j]
            sub = plot_df[(plot_df["feature"] == feature) & (plot_df["source"] == source)]
            if sub.empty:
                ax.set_axis_off()
                continue
            sns.histplot(
                data=sub,
                x=value_col,
                hue="group",
                stat="density",
                common_norm=False,
                element="step",
                fill=False,
                bins=bins,
                ax=ax,
            )
            ax.set_title(f"{feature} / {source}")
            ax.set_xlabel(x_label)
            ax.set_ylabel("density")
            ax.grid(alpha=0.2)
    fig.suptitle(title, fontsize=14)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_ecdf_grid(df, value_col, output_path, title, x_label, clip_quantile=None):
    plot_df = df.copy()
    if clip_quantile is not None:
        clipped = []
        for _, sub in plot_df.groupby(["feature", "source"], sort=False):
            lo = sub[value_col].quantile(1.0 - clip_quantile) if sub["source"].iloc[0] == "deviation" else sub[value_col].min()
            hi = sub[value_col].quantile(clip_quantile)
            clipped.append(sub[(sub[value_col] >= lo) & (sub[value_col] <= hi)])
        plot_df = pd.concat(clipped, ignore_index=True)

    fig, axes = plt.subplots(len(FEATURES), len(SOURCES), figsize=(17, 8.5), constrained_layout=True)
    for i, feature in enumerate(FEATURES):
        for j, source in enumerate(SOURCES):
            ax = axes[i, j]
            sub = plot_df[(plot_df["feature"] == feature) & (plot_df["source"] == source)]
            if sub.empty:
                ax.set_axis_off()
                continue
            sns.ecdfplot(data=sub, x=value_col, hue="group", ax=ax)
            ax.set_title(f"{feature} / {source}")
            ax.set_xlabel(x_label)
            ax.set_ylabel("ECDF")
            ax.grid(alpha=0.2)
    fig.suptitle(title, fontsize=14)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_box_quantiles(summary_df, output_path):
    rows = []
    for _, row in summary_df.iterrows():
        for q in ["p1", "p5", "p25", "p50", "p75", "p95", "p99"]:
            rows.append({
                "group": row["group"],
                "feature": row["feature"],
                "source": row["source"],
                "quantile": q,
                "value_ms": row[q],
            })
    qdf = pd.DataFrame(rows)
    fig, axes = plt.subplots(len(FEATURES), len(SOURCES), figsize=(17, 8.5), constrained_layout=True)
    for i, feature in enumerate(FEATURES):
        for j, source in enumerate(SOURCES):
            ax = axes[i, j]
            sub = qdf[(qdf["feature"] == feature) & (qdf["source"] == source)]
            sns.lineplot(data=sub, x="quantile", y="value_ms", hue="group", marker="o", ax=ax)
            ax.set_title(f"{feature} / {source}")
            ax.set_xlabel("quantile")
            ax.set_ylabel("ms")
            ax.grid(alpha=0.2)
    fig.suptitle("Timing Quantiles in Real Domain", fontsize=14)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze PianoCoRe timing distributions.")
    parser.add_argument("--processed-dir", type=Path, default=Path("../PianoCoRe/processed_raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/plots/timing_distributions"))
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--sample-per-kind", type=int, default=250_000)
    parser.add_argument("--chunk-sample-size", type=int, default=4096)
    parser.add_argument("--shuffle-files", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sns.set_theme(style="whitegrid")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df, summary_df, meta = collect_long_dataframe(
        args.processed_dir,
        max_files=args.max_files,
        sample_per_kind=args.sample_per_kind,
        chunk_sample_size=args.chunk_sample_size,
        shuffle_files=args.shuffle_files,
        seed=args.seed,
    )
    if df.empty:
        raise SystemExit("No timing rows collected.")

    df.to_parquet(args.output_dir / "timing_distribution_sample.parquet", index=False)
    df.to_csv(args.output_dir / "timing_distribution_sample.csv", index=False)
    summary_df.to_csv(args.output_dir / "timing_distribution_summary.csv", index=False)
    (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    plot_hist_grid(
        df,
        "value_ms",
        args.output_dir / "timing_real_ms_hist_p99.png",
        "Timing Distribution in Real Domain (clipped to 99th percentile)",
        "ms",
        clip_quantile=0.99,
    )
    plot_hist_grid(
        df,
        "log1p_ms",
        args.output_dir / "timing_log1p_ms_hist_p99.png",
        "Timing Distribution after log1p(ms) / signed log1p(deviation)",
        "log value",
        clip_quantile=0.99,
    )
    plot_hist_grid(
        df,
        "scaled_log_5000_s10",
        args.output_dir / "timing_scaled_log_5000_s10_hist_p99.png",
        "Timing Distribution in INR scaled_log_5000_s10 Domain",
        "normalized log value",
        clip_quantile=0.99,
    )
    plot_ecdf_grid(
        df,
        "value_ms",
        args.output_dir / "timing_real_ms_ecdf_p99.png",
        "Timing ECDF in Real Domain (clipped to 99th percentile)",
        "ms",
        clip_quantile=0.99,
    )
    plot_ecdf_grid(
        df,
        "scaled_log_5000_s10",
        args.output_dir / "timing_scaled_log_5000_s10_ecdf_p99.png",
        "Timing ECDF in INR scaled_log_5000_s10 Domain",
        "normalized log value",
        clip_quantile=0.99,
    )
    plot_box_quantiles(summary_df, args.output_dir / "timing_real_ms_quantiles.png")

    print(f"Wrote {args.output_dir}")
    print(summary_df.to_string(index=False, max_rows=50))


if __name__ == "__main__":
    main()
