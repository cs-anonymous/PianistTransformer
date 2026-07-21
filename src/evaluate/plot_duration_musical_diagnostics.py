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

try:
    import orjson
except Exception:  # pragma: no cover
    orjson = None

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def as_bool(series):
    if series.dtype == bool:
        return series
    return series.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})


def read_json(path):
    raw = Path(path).read_bytes()
    if orjson is not None:
        return orjson.loads(raw)
    return json.loads(raw.decode("utf-8"))


def timing_log_norm(values, scale=50.0, max_ms=5000.0):
    values = np.asarray(values, dtype=np.float64)
    clipped = np.clip(values, 0.0, max_ms)
    return np.log1p(clipped / scale) / math.log1p(max_ms / scale)


def processed_json_path(processed_dir, score_source):
    return Path(processed_dir) / Path(str(score_source)).with_suffix(".json")


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
    return meta[
        as_bool(meta["tier_a"])
        & as_bool(meta["is_refined"])
        & (meta["performance_dataset"].fillna("").astype(str) == "ASAP")
        & meta["split"].isin(["train", "test"])
        & meta["refined_score_midi_path"].notna()
        & meta["refined_performance_midi_path"].notna()
    ].copy()


def score_arrays(work, log_scale, log_max_ms):
    score = work.get("score") or {}
    score_raw = np.asarray(score.get("score_raw") or [], dtype=np.float64)
    if score_raw.ndim != 2 or score_raw.shape[1] < 2:
        return None
    n = len(score_raw)
    score_duration_ms = score_raw[:, 1]
    score_duration_norm = timing_log_norm(score_duration_ms, scale=log_scale, max_ms=log_max_ms)

    score_feature = score.get("score_feature") or []
    has_score_feature = np.asarray(score.get("has_score_feature") or [0] * n, dtype=bool)
    md = np.zeros(n, dtype=np.float64)
    for idx in range(min(n, len(score_feature))):
        feature = score_feature[idx]
        if has_score_feature[idx] and len(feature) > 1:
            md[idx] = float(feature[1])
    musical_duration_norm = np.clip(md / 6.0, 0.0, 1.0)
    staccato = np.zeros(n, dtype=bool)
    grace = np.zeros(n, dtype=bool)
    for idx in range(min(n, len(score_feature))):
        feature = score_feature[idx]
        if has_score_feature[idx]:
            grace[idx] = len(feature) > 6 and float(feature[6]) >= 0.5
            staccato[idx] = len(feature) > 7 and float(feature[7]) >= 0.5
    return {
        "score_duration_ms": score_duration_ms,
        "score_duration_norm": score_duration_norm,
        "has_score_feature": has_score_feature,
        "musical_duration_norm": musical_duration_norm,
        "staccato": staccato,
        "grace": grace,
    }


def md_bucket(values):
    values = np.asarray(values, dtype=np.float64)
    labels = np.full(values.shape, "missing", dtype=object)
    finite = np.isfinite(values)
    labels[finite & (values <= 1e-9)] = "0"
    labels[finite & (values > 0.0) & (values <= 0.125)] = "(0,1/8]"
    labels[finite & (values > 0.125) & (values <= 0.25)] = "(1/8,1/4]"
    labels[finite & (values > 0.25) & (values <= 0.5)] = "(1/4,1/2]"
    labels[finite & (values > 0.5) & (values <= 1.0)] = "(1/2,1]"
    labels[finite & (values > 1.0) & (values <= 2.0)] = "(1,2]"
    labels[finite & (values > 2.0)] = ">2"
    return labels


def process_gt_score(task):
    score_source, rows, processed_dir, log_scale, log_max_ms = task
    path = processed_json_path(processed_dir, score_source)
    if not path.exists():
        return []
    try:
        work = read_json(path)
    except Exception:
        return []
    arrays = score_arrays(work, log_scale, log_max_ms)
    if arrays is None:
        return []
    n = len(arrays["score_duration_norm"])
    perf_by_source = {
        str(perf.get("performance_source")): perf
        for perf in work.get("performances", [])
        if perf.get("performance_source") is not None
    }

    frames = []
    for row in rows:
        perf = perf_by_source.get(str(row["refined_performance_midi_path"]))
        if perf is None:
            continue
        label = np.asarray(perf.get("label_shared_raw") or [], dtype=np.float64)
        if label.ndim != 2 or label.shape[1] < 2 or len(label) != n:
            continue
        split = str(row["split"])
        perf_duration_ms = label[:, 1]
        perf_duration_norm = timing_log_norm(perf_duration_ms, scale=log_scale, max_ms=log_max_ms)
        duration_dev = np.clip(perf_duration_norm - arrays["score_duration_norm"] + 0.5, 0.0, 1.0)
        has = arrays["has_score_feature"]
        musical = np.where(has, arrays["musical_duration_norm"], np.nan)
        frames.append(
            pd.DataFrame(
                {
                    "source": f"ASAP {split}",
                    "split": split,
                    "kind": "gt",
                    "score_source": str(score_source),
                    "duration_dev": duration_dev,
                    "score_duration_norm": arrays["score_duration_norm"],
                    "perf_duration_norm": perf_duration_norm,
                    "musical_duration_norm": musical,
                    "md_bucket": md_bucket(musical),
                    "has_score_feature": has,
                    "staccato": arrays["staccato"],
                    "grace": arrays["grace"],
                }
            )
        )
    return frames


def collect_gt(metadata_path, processed_dir, log_scale, log_max_ms, num_workers):
    meta = load_asap_metadata(metadata_path)
    tasks = [
        (score_source, group.to_dict("records"), str(processed_dir), log_scale, log_max_ms)
        for score_source, group in meta.groupby("refined_score_midi_path", sort=False)
    ]
    frames = []
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = [pool.submit(process_gt_score, task) for task in tasks]
        for fut in as_completed(futures):
            frames.extend(fut.result())
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def manifest_raw_paths(manifest_path):
    manifest = read_json(manifest_path)
    paths = []
    for item in manifest.get("items", []):
        paths.extend(item.get("raw_output_paths", []))
    return [Path(path) for path in paths]


def collect_pred(manifest_path, processed_dir, log_scale, log_max_ms, label):
    frames = []
    for raw_path in manifest_raw_paths(manifest_path):
        try:
            raw = read_json(raw_path)
            work = read_json(processed_json_path(processed_dir, raw["score_source"]))
        except Exception:
            continue
        arrays = score_arrays(work, log_scale, log_max_ms)
        if arrays is None:
            continue
        target = np.asarray(raw.get("predicted_target7") or [], dtype=np.float64)
        raw7 = np.asarray(raw.get("reconstructed_raw7") or [], dtype=np.float64)
        n = len(arrays["score_duration_norm"])
        if target.ndim != 2 or target.shape[1] < 2 or len(target) != n:
            continue
        if raw7.ndim == 2 and raw7.shape[1] >= 2 and len(raw7) == n:
            perf_duration_norm = timing_log_norm(raw7[:, 1], scale=log_scale, max_ms=log_max_ms)
        else:
            perf_duration_norm = arrays["score_duration_norm"] + (target[:, 1] - 0.5)
        has = arrays["has_score_feature"]
        musical = np.where(has, arrays["musical_duration_norm"], np.nan)
        frames.append(
            pd.DataFrame(
                {
                    "source": label,
                    "split": "pred",
                    "kind": "pred",
                    "score_source": str(raw.get("score_source")),
                    "duration_dev": target[:, 1],
                    "score_duration_norm": arrays["score_duration_norm"],
                    "perf_duration_norm": perf_duration_norm,
                    "musical_duration_norm": musical,
                    "md_bucket": md_bucket(musical),
                    "has_score_feature": has,
                    "staccato": arrays["staccato"],
                    "grace": arrays["grace"],
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def sample_long(df, value_cols, sample_per_source_col, seed):
    rng = np.random.default_rng(seed)
    frames = []
    for source, source_df in df.groupby("source", sort=False):
        for col in value_cols:
            vals = source_df[col].to_numpy(dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if len(vals) > sample_per_source_col:
                vals = rng.choice(vals, size=sample_per_source_col, replace=False)
            frames.append(pd.DataFrame({"source": source, "feature": col, "value": vals}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def write_summary(df, output_dir):
    rows = []
    for (source, bucket), sub in df.groupby(["source", "md_bucket"], sort=False):
        vals = sub["duration_dev"].to_numpy(dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        rows.append(
            {
                "source": source,
                "md_bucket": bucket,
                "count": int(len(vals)),
                "mean": float(np.mean(vals)),
                "p10": float(np.quantile(vals, 0.10)),
                "p50": float(np.quantile(vals, 0.50)),
                "p90": float(np.quantile(vals, 0.90)),
                "p95": float(np.quantile(vals, 0.95)),
                "right_of_0_5_frac": float(np.mean(vals > 0.5)),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "duration_dev_by_md_bucket_summary.csv", index=False)

    coverage = (
        df.groupby("source", sort=False)["has_score_feature"]
        .agg(total_notes="count", musical_notes="sum")
        .reset_index()
    )
    coverage["musical_coverage"] = coverage["musical_notes"] / coverage["total_notes"].clip(lower=1)
    coverage.to_csv(output_dir / "musical_feature_coverage.csv", index=False)
    return summary, coverage


def plot_duration_norms(df, output_dir, seed, sample_per_source_col):
    long_df = sample_long(
        df,
        ["musical_duration_norm", "score_duration_norm", "perf_duration_norm"],
        sample_per_source_col,
        seed,
    )
    long_df.to_csv(output_dir / "duration_norm_plot_sample.csv", index=False)
    for kind in ("hist", "ecdf"):
        fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), constrained_layout=True)
        for ax, feature in zip(axes, ["musical_duration_norm", "score_duration_norm", "perf_duration_norm"]):
            sub = long_df[long_df["feature"] == feature]
            if kind == "ecdf":
                sns.ecdfplot(data=sub, x="value", hue="source", ax=ax)
            else:
                sns.histplot(
                    data=sub,
                    x="value",
                    hue="source",
                    stat="density",
                    common_norm=False,
                    element="step",
                    fill=False,
                    bins=120,
                    ax=ax,
                )
            ax.set_title(feature)
            ax.set_xlim(0.0, 1.0)
            ax.grid(alpha=0.2)
        fig.suptitle(f"Duration Normalized Distributions ({kind})")
        fig.savefig(output_dir / f"duration_norm_{kind}.png", dpi=180)
        plt.close(fig)


def plot_duration_dev(df, output_dir, seed, sample_per_source_col):
    plot_df = sample_long(df, ["duration_dev"], sample_per_source_col, seed)
    plot_df.to_csv(output_dir / "duration_dev_plot_sample.csv", index=False)
    for kind in ("hist", "ecdf"):
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        if kind == "ecdf":
            sns.ecdfplot(data=plot_df, x="value", hue="source", ax=ax)
        else:
            sns.histplot(
                data=plot_df,
                x="value",
                hue="source",
                stat="density",
                common_norm=False,
                element="step",
                fill=False,
                bins=140,
                ax=ax,
            )
        ax.axvline(0.5, color="black", linewidth=1, alpha=0.5)
        ax.set_xlim(0.0, 1.0)
        ax.set_title(f"Duration Dev Target Distribution ({kind})")
        ax.grid(alpha=0.2)
        fig.savefig(output_dir / f"duration_dev_{kind}.png", dpi=180)
        plt.close(fig)


def plot_bucket_views(df, output_dir, seed, sample_per_bucket_source):
    order = ["0", "(0,1/8]", "(1/8,1/4]", "(1/4,1/2]", "(1/2,1]", "(1,2]", ">2", "missing"]
    parts = []
    rng = np.random.default_rng(seed)
    for (source, bucket), sub in df.groupby(["source", "md_bucket"], sort=False):
        vals = sub[["source", "md_bucket", "duration_dev", "staccato", "grace"]].copy()
        if len(vals) > sample_per_bucket_source:
            vals = vals.iloc[rng.choice(len(vals), size=sample_per_bucket_source, replace=False)]
        parts.append(vals)
    plot_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    plot_df["md_bucket"] = pd.Categorical(plot_df["md_bucket"], categories=order, ordered=True)
    plot_df.to_csv(output_dir / "duration_dev_by_md_bucket_plot_sample.csv", index=False)

    fig, ax = plt.subplots(figsize=(14, 5.5), constrained_layout=True)
    sns.boxplot(
        data=plot_df,
        x="md_bucket",
        y="duration_dev",
        hue="source",
        showfliers=False,
        ax=ax,
    )
    ax.axhline(0.5, color="black", linewidth=1, alpha=0.5)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Duration Dev by Musical Duration Bucket")
    ax.grid(alpha=0.2, axis="y")
    fig.savefig(output_dir / "duration_dev_by_md_bucket_box.png", dpi=180)
    plt.close(fig)

    for flag in ("staccato", "grace"):
        sub = plot_df[plot_df[flag].astype(bool)]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        sns.ecdfplot(data=sub, x="duration_dev", hue="source", ax=ax)
        ax.axvline(0.5, color="black", linewidth=1, alpha=0.5)
        ax.set_xlim(0.0, 1.0)
        ax.set_title(f"Duration Dev ECDF for {flag} notes")
        ax.grid(alpha=0.2)
        fig.savefig(output_dir / f"duration_dev_{flag}_ecdf.png", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot musical/score/performance duration diagnostics.")
    parser.add_argument("--metadata-path", type=Path, default=ROOT_DIR / "data" / "ASAP_processed" / "metadata.generated_json.csv")
    parser.add_argument("--processed-dir", type=Path, default=ROOT_DIR / "data" / "ASAP_processed")
    parser.add_argument("--det-manifest", type=Path, default=None)
    parser.add_argument("--sampling-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log-scale", type=float, default=50.0)
    parser.add_argument("--log-max-ms", type=float, default=5000.0)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--sample-per-source-col", type=int, default=250_000)
    parser.add_argument("--sample-per-bucket-source", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sns.set_theme(style="whitegrid")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gt_df = collect_gt(
        args.metadata_path,
        args.processed_dir,
        args.log_scale,
        args.log_max_ms,
        args.num_workers,
    )
    pred_frames = []
    if args.det_manifest is not None:
        pred_frames.append(collect_pred(args.det_manifest, args.processed_dir, args.log_scale, args.log_max_ms, "pred det"))
    samp_df = collect_pred(args.sampling_manifest, args.processed_dir, args.log_scale, args.log_max_ms, "pred samp")
    pred_frames.append(samp_df)
    all_df = pd.concat([gt_df, *pred_frames], ignore_index=True)
    if all_df.empty:
        raise SystemExit("No rows collected.")

    summary, coverage = write_summary(all_df, args.output_dir)
    plot_duration_norms(all_df, args.output_dir, args.seed, args.sample_per_source_col)
    plot_duration_dev(all_df, args.output_dir, args.seed, args.sample_per_source_col)
    plot_bucket_views(all_df, args.output_dir, args.seed, args.sample_per_bucket_source)

    meta = {
        "det_manifest": str(args.det_manifest.resolve()),
        "sampling_manifest": str(args.sampling_manifest.resolve()),
        "log_scale": args.log_scale,
        "log_max_ms": args.log_max_ms,
        "rows": {source: int(len(sub)) for source, sub in all_df.groupby("source", sort=False)},
    }
    (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print("\nCoverage:")
    print(coverage.to_string(index=False))
    print("\nDuration dev by md bucket:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
