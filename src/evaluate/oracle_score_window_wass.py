import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_process.work_manifest import build_work_manifest


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute same-score-window empirical oracle Wasserstein on ASAP test targets."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--performance-dataset", type=str, default="ASAP")
    parser.add_argument("--max-works", type=int, default=None)
    parser.add_argument(
        "--variants",
        type=str,
        default="test_asap,all_asap,all_processed",
        help="Comma-separated oracle pools: test_asap, all_asap, all_processed.",
    )
    return parser.parse_args()


def log50(values, max_time_ms=5000.0, scale=50.0):
    values = np.clip(np.asarray(values, dtype=np.float64), 0.0, float(max_time_ms))
    return np.log1p(values / float(scale)) / math.log1p(float(max_time_ms) / float(scale))


def safe_wass(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float(wasserstein_distance(a, b))


def perf_raw_arrays(perf, start, end):
    shared = np.asarray(perf.get("label_shared_raw", []), dtype=np.float64)
    pedal = np.asarray(perf.get("label_pedal4_raw", []), dtype=np.float64)
    if shared.ndim != 2 or shared.shape[1] < 3:
        raise ValueError(f"Bad label_shared_raw for {perf.get('performance_source')}")
    if pedal.ndim != 2 or pedal.shape[1] < 4:
        raise ValueError(f"Bad label_pedal4_raw for {perf.get('performance_source')}")
    if len(shared) != len(pedal):
        raise ValueError(
            f"label_shared_raw/label_pedal4_raw length mismatch for {perf.get('performance_source')}: "
            f"{len(shared)} vs {len(pedal)}"
        )
    shared = shared[start:end, :3]
    pedal = pedal[start:end, :4]
    return {
        "ioi": shared[:, 0],
        "duration": shared[:, 1],
        "velocity": shared[:, 2],
        "pedal_0": pedal[:, 0],
        "pedal_25": pedal[:, 1],
        "pedal_50": pedal[:, 2],
        "pedal_75": pedal[:, 3],
        "pedal_avg": pedal.mean(axis=1),
        "ioi_log50": log50(shared[:, 0]),
        "duration_log50": log50(shared[:, 1]),
    }


def concat_perf_arrays(perfs, start, end):
    by_key = {}
    for perf in perfs:
        arrays = perf_raw_arrays(perf, start, end)
        for key, values in arrays.items():
            by_key.setdefault(key, []).append(values)
    return {key: np.concatenate(parts) for key, parts in by_key.items() if parts}


def wasserstein_row(target_arrays, donor_arrays):
    row = {}
    keys = [
        "ioi",
        "duration",
        "velocity",
        "pedal_0",
        "pedal_25",
        "pedal_50",
        "pedal_75",
        "pedal_avg",
        "ioi_log50",
        "duration_log50",
    ]
    for key in keys:
        row[f"{key}_wass"] = safe_wass(donor_arrays.get(key, []), target_arrays.get(key, []))
    pedal_keys = ["pedal_0_wass", "pedal_25_wass", "pedal_50_wass", "pedal_75_wass"]
    row["pedal_wass"] = float(np.nanmean([row[key] for key in pedal_keys]))
    return row


def summarize(df):
    metric_cols = [col for col in df.columns if col.endswith("_wass")]
    summary = {
        "num_rows": int(len(df)),
        "num_scores": int(df["score_source"].nunique()) if len(df) else 0,
        "num_target_performances": int(df["target_performance_source"].nunique()) if len(df) else 0,
        "num_score_windows": int(df["score_window_id"].nunique()) if len(df) else 0,
        "metrics_weighted_by_notes": {},
        "metrics_unweighted_by_window": {},
    }
    if len(df) == 0:
        return summary
    weights = df["note_count"].to_numpy(dtype=np.float64)
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1.0)
    for col in metric_cols:
        values = df[col].to_numpy(dtype=np.float64)
        mask = np.isfinite(values)
        if not mask.any():
            continue
        summary["metrics_weighted_by_notes"][col] = float(np.average(values[mask], weights=weights[mask]))
        summary["metrics_unweighted_by_window"][col] = float(np.mean(values[mask]))
    return summary


def main():
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_work_manifest(
        metadata_path=config["metadata_path"],
        refined_dir=config["refined_dir"],
        split=args.split,
        block_notes=config["block_notes"],
        overlap_ratio=config["overlap_ratio"],
        min_notes=config["min_notes"],
        max_works=args.max_works,
        skip_work_paths=config.get("skip_work_paths"),
        performance_dataset=args.performance_dataset,
    )

    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    allowed_variants = {"test_asap", "all_asap", "all_processed"}
    unknown_variants = sorted(set(variants) - allowed_variants)
    if unknown_variants:
        raise ValueError(f"Unknown variants: {unknown_variants}")
    rows_by_variant = {variant: [] for variant in variants}
    missing_targets = []
    skipped = []

    for item in tqdm(manifest, desc="oracle scores"):
        path = Path(item["path"])
        work = json.loads(path.read_text(encoding="utf-8"))
        all_perfs = [
            perf
            for perf in work.get("performances", [])
            if perf.get("label_shared_raw") is not None and perf.get("label_pedal4_raw") is not None
        ]
        by_source = {perf.get("performance_source"): perf for perf in all_perfs}
        target_sources = list(item.get("selected_performance_sources", []))
        target_perfs = []
        for source in target_sources:
            perf = by_source.get(source)
            if perf is None:
                missing_targets.append({"score_source": item["score_source"], "performance_source": source})
            else:
                target_perfs.append(perf)
        if not target_perfs:
            skipped.append({"score_source": item["score_source"], "reason": "no_target_performances"})
            continue

        donor_pools = {}
        if "test_asap" in rows_by_variant:
            donor_pools["test_asap"] = target_perfs
        if "all_asap" in rows_by_variant:
            donor_pools["all_asap"] = [
                perf for perf in all_perfs if str(perf.get("performance_dataset") or "") == args.performance_dataset
            ]
        if "all_processed" in rows_by_variant:
            donor_pools["all_processed"] = all_perfs

        for window_idx, (start, end) in enumerate(item["windows"]):
            score_window_id = f"{item['score_source']}::{window_idx}:{start}-{end}"
            window_arrays_by_source = {}
            unique_perf_by_source = {}
            for pool in donor_pools.values():
                for perf in pool:
                    source = perf.get("performance_source")
                    if source is not None:
                        unique_perf_by_source[source] = perf
            for perf in unique_perf_by_source.values():
                try:
                    window_arrays_by_source[perf.get("performance_source")] = perf_raw_arrays(perf, start, end)
                except Exception as exc:
                    skipped.append(
                        {
                            "score_source": item["score_source"],
                            "performance_source": perf.get("performance_source"),
                            "window_idx": window_idx,
                            "reason": repr(exc),
                        }
                    )
            for target in target_perfs:
                target_source = target.get("performance_source")
                target_arrays = window_arrays_by_source.get(target_source)
                if target_arrays is None:
                    continue
                for variant, pool in donor_pools.items():
                    donors = [perf for perf in pool if perf.get("performance_source") != target_source]
                    if not donors:
                        continue
                    try:
                        donor_sources = [
                            perf.get("performance_source")
                            for perf in donors
                            if perf.get("performance_source") in window_arrays_by_source
                        ]
                        if not donor_sources:
                            continue
                        donor_arrays = {
                            key: np.concatenate(
                                [
                                    window_arrays_by_source[source][key]
                                    for source in donor_sources
                                ]
                            )
                            for key in target_arrays
                        }
                    except Exception as exc:
                        skipped.append(
                            {
                                "score_source": item["score_source"],
                                "performance_source": target_source,
                                "window_idx": window_idx,
                                "variant": variant,
                                "reason": repr(exc),
                            }
                        )
                        continue
                    row = {
                        "variant": variant,
                        "score_source": item["score_source"],
                        "score_window_id": score_window_id,
                        "window_idx": int(window_idx),
                        "start": int(start),
                        "end": int(end),
                        "note_count": int(end) - int(start),
                        "target_performance_source": target_source,
                        "target_performance_dataset": target.get("performance_dataset"),
                        "num_donor_performances": int(len(donors)),
                        "num_donor_notes": int(sum(max(0, int(end) - int(start)) for _ in donors)),
                    }
                    row.update(wasserstein_row(target_arrays, donor_arrays))
                    rows_by_variant[variant].append(row)

    variant_summaries = {}
    all_frames = []
    for variant, rows in rows_by_variant.items():
        df = pd.DataFrame(rows)
        variant_csv = args.output_dir / f"{variant}_window_oracle.csv"
        df.to_csv(variant_csv, index=False)
        variant_summaries[variant] = summarize(df)
        variant_summaries[variant]["csv"] = str(variant_csv)
        if len(df):
            score_rows = []
            for score_source, group in df.groupby("score_source", sort=True):
                group_summary = summarize(group)
                flat = {"score_source": score_source}
                for key, value in group_summary["metrics_weighted_by_notes"].items():
                    flat[key] = value
                flat["num_rows"] = group_summary["num_rows"]
                flat["num_target_performances"] = group_summary["num_target_performances"]
                flat["num_score_windows"] = group_summary["num_score_windows"]
                score_rows.append(flat)
            score_csv = args.output_dir / f"{variant}_score_summary.csv"
            pd.DataFrame(score_rows).to_csv(score_csv, index=False)
            variant_summaries[variant]["score_csv"] = str(score_csv)
            all_frames.append(df)

    all_df = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    if len(all_df):
        all_df.to_csv(args.output_dir / "all_variants_window_oracle.csv", index=False)

    payload = {
        "config": str(args.config),
        "split": args.split,
        "performance_dataset": args.performance_dataset,
        "num_manifest_scores": int(len(manifest)),
        "variants": variant_summaries,
        "missing_targets": missing_targets,
        "skipped": skipped,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
