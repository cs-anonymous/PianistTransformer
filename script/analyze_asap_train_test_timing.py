#!/usr/bin/env python
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    import orjson
except Exception:  # pragma: no cover
    orjson = None

try:
    from scipy.stats import wasserstein_distance
except Exception:  # pragma: no cover
    wasserstein_distance = None


RAW_TARGETS = ("score_ioi", "perf_ioi", "score_duration", "perf_duration")
DELTA_TARGETS = (
    "ioi_delta_ms",
    "duration_delta_ms",
    "ioi_log_delta",
    "duration_log_delta",
    "duration_ratio",
    "log_duration_ratio",
)
OTHER_TARGETS = ("velocity",)


def as_bool(series):
    if series.dtype == bool:
        return series
    return series.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})


def timing_log_norm(values, scale=50.0, max_ms=5000.0):
    values = np.asarray(values, dtype=np.float64)
    clipped = np.clip(values, 0.0, max_ms)
    return np.log1p(clipped / scale) / math.log1p(max_ms / scale)


def load_asap_metadata(metadata_path):
    usecols = [
        "split",
        "tier_a",
        "is_refined",
        "performance_dataset",
        "refined_score_midi_path",
        "refined_performance_midi_path",
    ]
    meta = pd.read_csv(metadata_path, usecols=usecols)
    meta = meta[
        as_bool(meta["tier_a"])
        & as_bool(meta["is_refined"])
        & (meta["performance_dataset"].fillna("").astype(str) == "ASAP")
        & meta["split"].isin(["train", "test"])
        & meta["refined_score_midi_path"].notna()
        & meta["refined_performance_midi_path"].notna()
    ].copy()
    return meta


def read_json(path):
    raw = Path(path).read_bytes()
    if orjson is not None:
        return orjson.loads(raw)
    return json.loads(raw.decode("utf-8"))


def quantile_summary(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {}
    qs = np.quantile(values, [0, 0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1.0])
    return {
        "count": int(len(values)),
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
    }


def process_one_score(score_source, score_meta, processed_dir, log_scale=50.0, log_max_ms=5000.0):
    json_path = Path(processed_dir) / Path(str(score_source)).with_suffix(".json")
    if not json_path.exists():
        return [], [], {"missing_score": str(score_source)}
    try:
        data = read_json(json_path)
    except Exception as exc:
        return [], [], {"json_error": f"{score_source}: {exc!r}"}

    score_raw = (data.get("score") or {}).get("score_raw")
    if not score_raw:
        return [], [], {"missing_score_raw": str(score_source)}
    score_arr = np.asarray(score_raw, dtype=np.float64)
    if score_arr.ndim != 2 or score_arr.shape[1] < 2:
        return [], [], {"bad_score_shape": str(score_source)}

    perf_by_source = {
        str(perf.get("performance_source")): perf
        for perf in data.get("performances", [])
        if perf.get("performance_source") is not None
    }
    score_ioi = score_arr[:, 0]
    score_duration = score_arr[:, 1]
    score_ioi_log = timing_log_norm(score_ioi, scale=log_scale, max_ms=log_max_ms)
    score_duration_log = timing_log_norm(score_duration, scale=log_scale, max_ms=log_max_ms)

    note_frames = []
    work_rows = []
    perf_seen = 0
    missing_perfs = []
    for _, meta_row in score_meta.iterrows():
        perf_source = str(meta_row["refined_performance_midi_path"])
        perf = perf_by_source.get(perf_source)
        if perf is None:
            missing_perfs.append(perf_source)
            continue
        label = perf.get("label_shared_raw")
        if not label:
            missing_perfs.append(perf_source)
            continue
        label_arr = np.asarray(label, dtype=np.float64)
        if label_arr.ndim != 2 or label_arr.shape[1] < 3 or len(label_arr) != len(score_arr):
            missing_perfs.append(perf_source)
            continue
        split = str(meta_row["split"])
        perf_ioi = label_arr[:, 0]
        perf_duration = label_arr[:, 1]
        velocity = label_arr[:, 2]
        perf_ioi_log = timing_log_norm(perf_ioi, scale=log_scale, max_ms=log_max_ms)
        perf_duration_log = timing_log_norm(perf_duration, scale=log_scale, max_ms=log_max_ms)
        safe_score_duration = np.maximum(score_duration, 1.0)
        safe_perf_duration = np.maximum(perf_duration, 1.0)

        frame = pd.DataFrame(
            {
                "split": split,
                "score_source": str(score_source),
                "performance_source": perf_source,
                "score_ioi": score_ioi,
                "perf_ioi": perf_ioi,
                "score_duration": score_duration,
                "perf_duration": perf_duration,
                "score_ioi_log50": score_ioi_log,
                "perf_ioi_log50": perf_ioi_log,
                "score_duration_log50": score_duration_log,
                "perf_duration_log50": perf_duration_log,
                "ioi_delta_ms": perf_ioi - score_ioi,
                "duration_delta_ms": perf_duration - score_duration,
                "ioi_log_delta": perf_ioi_log - score_ioi_log,
                "duration_log_delta": perf_duration_log - score_duration_log,
                "duration_ratio": safe_perf_duration / safe_score_duration,
                "log_duration_ratio": np.log(safe_perf_duration / safe_score_duration),
                "velocity": velocity,
            }
        )
        note_frames.append(frame)
        perf_seen += 1

        work_summary = {
            "split": split,
            "score_source": str(score_source),
            "performance_source": perf_source,
            "note_count": int(len(frame)),
        }
        for target in RAW_TARGETS + DELTA_TARGETS + OTHER_TARGETS:
            vals = frame[target].to_numpy()
            work_summary[f"{target}_mean"] = float(np.mean(vals))
            work_summary[f"{target}_p50"] = float(np.quantile(vals, 0.5))
            work_summary[f"{target}_p95"] = float(np.quantile(vals, 0.95))
            work_summary[f"{target}_p99"] = float(np.quantile(vals, 0.99))
        work_rows.append(work_summary)

    meta_out = {
        "matched_performances": int(perf_seen),
        "missing_performances": missing_perfs[:20],
        "missing_performance_count": int(len(missing_perfs)),
    }
    return note_frames, work_rows, meta_out


def collect_asap_rows(metadata_path, processed_dir, log_scale=50.0, log_max_ms=5000.0, num_workers=16, executor="process"):
    meta = load_asap_metadata(metadata_path)
    by_score = meta.groupby("refined_score_midi_path", sort=False)

    note_frames = []
    work_rows = []
    missing_scores = []
    perf_seen = 0
    score_tasks = [(score_source, score_meta) for score_source, score_meta in by_score]
    pool_cls = ProcessPoolExecutor if executor == "process" else ThreadPoolExecutor
    with pool_cls(max_workers=num_workers) as pool:
        futures = [
            pool.submit(process_one_score, score_source, score_meta, processed_dir, log_scale, log_max_ms)
            for score_source, score_meta in score_tasks
        ]
        for fut in as_completed(futures):
            frames, rows, meta_piece = fut.result()
            note_frames.extend(frames)
            work_rows.extend(rows)
            perf_seen += int(meta_piece.get("matched_performances", 0))
            if "missing_score" in meta_piece:
                missing_scores.append(meta_piece["missing_score"])
            if "missing_score_raw" in meta_piece:
                missing_scores.append(meta_piece["missing_score_raw"])
            if "bad_score_shape" in meta_piece:
                missing_scores.append(meta_piece["bad_score_shape"])
    note_df = pd.concat(note_frames, ignore_index=True) if note_frames else pd.DataFrame()
    work_df = pd.DataFrame(work_rows)
    meta_out = {
        "metadata_rows": int(len(meta)),
        "metadata_scores": int(meta["refined_score_midi_path"].nunique()),
        "metadata_performances": int(meta["refined_performance_midi_path"].nunique()),
        "matched_performances": int(perf_seen),
        "missing_scores": missing_scores[:50],
        "missing_score_count": int(len(missing_scores)),
        "log_scale": float(log_scale),
        "log_max_ms": float(log_max_ms),
        "num_workers": int(num_workers),
        "executor": executor,
    }
    return note_df, work_df, meta_out


def summarize_note_level(note_df):
    rows = []
    targets = RAW_TARGETS + (
        "score_ioi_log50",
        "perf_ioi_log50",
        "score_duration_log50",
        "perf_duration_log50",
    ) + DELTA_TARGETS + OTHER_TARGETS
    for split, sub in note_df.groupby("split", sort=True):
        for target in targets:
            vals = sub[target].to_numpy()
            row = {"split": split, "target": target}
            row.update(quantile_summary(vals))
            if target.endswith("_ms") or target in RAW_TARGETS:
                row["abs_gt_100_frac"] = float(np.mean(np.abs(vals) > 100))
                row["abs_gt_500_frac"] = float(np.mean(np.abs(vals) > 500))
                row["gt_1000_frac"] = float(np.mean(vals > 1000))
            if target in {"duration_ratio"}:
                row["gt_1_frac"] = float(np.mean(vals > 1.0))
                row["gt_2_frac"] = float(np.mean(vals > 2.0))
            rows.append(row)
    summary = pd.DataFrame(rows)

    diff_rows = []
    train = note_df[note_df["split"] == "train"]
    test = note_df[note_df["split"] == "test"]
    for target in targets:
        a = train[target].to_numpy()
        b = test[target].to_numpy()
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if len(a) == 0 or len(b) == 0:
            continue
        row = {
            "target": target,
            "train_count": int(len(a)),
            "test_count": int(len(b)),
            "train_mean": float(np.mean(a)),
            "test_mean": float(np.mean(b)),
            "mean_diff_test_minus_train": float(np.mean(b) - np.mean(a)),
            "train_p50": float(np.quantile(a, 0.5)),
            "test_p50": float(np.quantile(b, 0.5)),
            "p50_diff_test_minus_train": float(np.quantile(b, 0.5) - np.quantile(a, 0.5)),
            "train_p95": float(np.quantile(a, 0.95)),
            "test_p95": float(np.quantile(b, 0.95)),
            "p95_diff_test_minus_train": float(np.quantile(b, 0.95) - np.quantile(a, 0.95)),
            "train_p99": float(np.quantile(a, 0.99)),
            "test_p99": float(np.quantile(b, 0.99)),
            "p99_diff_test_minus_train": float(np.quantile(b, 0.99) - np.quantile(a, 0.99)),
        }
        if wasserstein_distance is not None:
            row["global_wasserstein_train_vs_test"] = float(wasserstein_distance(a, b))
        diff_rows.append(row)
    return summary, pd.DataFrame(diff_rows)


def to_long(note_df, targets, sample_per_split_target, seed):
    rng = np.random.default_rng(seed)
    frames = []
    for split, split_df in note_df.groupby("split", sort=True):
        for target in targets:
            vals = split_df[target].to_numpy()
            vals = vals[np.isfinite(vals)]
            if len(vals) > sample_per_split_target:
                vals = rng.choice(vals, size=sample_per_split_target, replace=False)
            frames.append(pd.DataFrame({"split": split, "target": target, "value": vals}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def clipped_long(long_df, quantile=0.995):
    parts = []
    for target, sub in long_df.groupby("target", sort=False):
        lo_q = 1.0 - quantile
        if target in {"score_ioi", "perf_ioi", "score_duration", "perf_duration", "duration_ratio", "velocity"}:
            lo_q = 0.0
        lo = sub["value"].quantile(lo_q)
        hi = sub["value"].quantile(quantile)
        parts.append(sub[(sub["value"] >= lo) & (sub["value"] <= hi)])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def plot_grid(long_df, targets, output_path, title, kind="hist", cols=2):
    plot_df = clipped_long(long_df)
    rows = int(math.ceil(len(targets) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 3.8 * rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for ax, target in zip(axes, targets):
        sub = plot_df[plot_df["target"] == target]
        if kind == "ecdf":
            sns.ecdfplot(data=sub, x="value", hue="split", ax=ax)
            ax.set_ylabel("ECDF")
        else:
            sns.histplot(
                data=sub,
                x="value",
                hue="split",
                stat="density",
                common_norm=False,
                element="step",
                fill=False,
                bins=160,
                ax=ax,
            )
            ax.set_ylabel("density")
        ax.set_title(target)
        ax.grid(alpha=0.2)
    for ax in axes[len(targets):]:
        ax.set_axis_off()
    fig.suptitle(title)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_work_summary(work_df, output_path):
    rows = []
    for target in ("perf_ioi", "perf_duration", "ioi_delta_ms", "duration_delta_ms", "duration_ratio"):
        for stat in ("mean", "p50", "p95"):
            col = f"{target}_{stat}"
            if col in work_df:
                rows.append(work_df[["split", "score_source", "performance_source", col]].rename(columns={col: "value"}).assign(target=f"{target}_{stat}"))
    long_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if long_df.empty:
        return
    targets = list(long_df["target"].unique())
    fig, axes = plt.subplots(3, 5, figsize=(21, 10), constrained_layout=True)
    for ax, target in zip(axes.flat, targets):
        sub = long_df[long_df["target"] == target]
        sns.boxplot(data=sub, x="split", y="value", ax=ax, showfliers=False)
        sns.stripplot(data=sub, x="split", y="value", ax=ax, color="0.2", alpha=0.25, size=2)
        ax.set_title(target)
        ax.grid(alpha=0.2)
    for ax in axes.flat[len(targets):]:
        ax.set_axis_off()
    fig.suptitle("ASAP Train/Test Work-Level Summary Distributions")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze ASAP train/test timing distributions from processed JSON.")
    parser.add_argument("--metadata-path", type=Path, default=Path("../PianoCoRe/metadata.csv"))
    parser.add_argument("--processed-dir", type=Path, default=Path("../PianoCoRe/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/plots/asap_train_test_timing_20260629"))
    parser.add_argument("--log-scale", type=float, default=50.0)
    parser.add_argument("--log-max-ms", type=float, default=5000.0)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--executor", choices=["process", "thread"], default="process")
    parser.add_argument("--sample-per-split-target", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sns.set_theme(style="whitegrid")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    note_df, work_df, meta = collect_asap_rows(
        args.metadata_path,
        args.processed_dir,
        log_scale=args.log_scale,
        log_max_ms=args.log_max_ms,
        num_workers=args.num_workers,
        executor=args.executor,
    )
    if note_df.empty:
        raise SystemExit("No ASAP train/test rows collected.")

    summary, diff = summarize_note_level(note_df)
    # Parquet can be fairly heavy on very wide note-level tables; keep CSV as the
    # reliable artifact and write parquet only if it succeeds quickly.
    note_df.to_csv(args.output_dir / "asap_train_test_note_metrics.csv", index=False)
    try:
        note_df.to_parquet(args.output_dir / "asap_train_test_note_metrics.parquet", index=False)
    except Exception as exc:
        (args.output_dir / "parquet_write_error.txt").write_text(repr(exc), encoding="utf-8")
    work_df.to_csv(args.output_dir / "asap_train_test_work_summary.csv", index=False)
    summary.to_csv(args.output_dir / "asap_train_test_note_summary.csv", index=False)
    diff.to_csv(args.output_dir / "asap_train_test_global_diff.csv", index=False)
    (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    raw_long = to_long(note_df, RAW_TARGETS, args.sample_per_split_target, args.seed)
    log_long = to_long(
        note_df,
        ("score_ioi_log50", "perf_ioi_log50", "score_duration_log50", "perf_duration_log50"),
        args.sample_per_split_target,
        args.seed,
    )
    delta_long = to_long(note_df, DELTA_TARGETS, args.sample_per_split_target, args.seed)
    other_long = to_long(note_df, OTHER_TARGETS, args.sample_per_split_target, args.seed)

    raw_long.to_csv(args.output_dir / "plot_sample_raw.csv", index=False)
    delta_long.to_csv(args.output_dir / "plot_sample_delta.csv", index=False)

    plot_grid(raw_long, RAW_TARGETS, args.output_dir / "timing_raw_ms_hist.png", "ASAP Train/Test Raw Timing Histograms")
    plot_grid(raw_long, RAW_TARGETS, args.output_dir / "timing_raw_ms_ecdf.png", "ASAP Train/Test Raw Timing ECDF", kind="ecdf")
    plot_grid(log_long, ("score_ioi_log50", "perf_ioi_log50", "score_duration_log50", "perf_duration_log50"), args.output_dir / "timing_log50_hist.png", "ASAP Train/Test log50 Timing Histograms")
    plot_grid(log_long, ("score_ioi_log50", "perf_ioi_log50", "score_duration_log50", "perf_duration_log50"), args.output_dir / "timing_log50_ecdf.png", "ASAP Train/Test log50 Timing ECDF", kind="ecdf")
    plot_grid(delta_long, DELTA_TARGETS, args.output_dir / "timing_delta_ratio_hist.png", "ASAP Train/Test Timing Delta and Ratio Histograms")
    plot_grid(delta_long, DELTA_TARGETS, args.output_dir / "timing_delta_ratio_ecdf.png", "ASAP Train/Test Timing Delta and Ratio ECDF", kind="ecdf")
    plot_grid(other_long, OTHER_TARGETS, args.output_dir / "velocity_hist.png", "ASAP Train/Test Velocity Histogram", cols=1)
    plot_work_summary(work_df, args.output_dir / "work_level_summary_box.png")

    print(f"Wrote {args.output_dir}")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print("\nKey train/test global differences:")
    key_targets = [
        "perf_ioi",
        "perf_duration",
        "ioi_delta_ms",
        "duration_delta_ms",
        "ioi_log_delta",
        "duration_log_delta",
        "duration_ratio",
        "velocity",
    ]
    cols = [
        "target",
        "train_mean",
        "test_mean",
        "mean_diff_test_minus_train",
        "train_p50",
        "test_p50",
        "p50_diff_test_minus_train",
        "train_p95",
        "test_p95",
        "p95_diff_test_minus_train",
        "global_wasserstein_train_vs_test",
    ]
    print(diff[diff["target"].isin(key_targets)][cols].to_string(index=False))


if __name__ == "__main__":
    main()
