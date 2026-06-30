#!/usr/bin/env python3
import argparse
import json
import math
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluate.compute_saved_midi_mae_wass import extract_note_arrays


FEATURES = ("ioi", "duration")
SCALES = (50.0, 100.0)
MAX_TIME_MS = 5000.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Per-score/per-note timing diagnostics for saved prediction manifests."
    )
    parser.add_argument(
        "--manifest",
        action="append",
        nargs=2,
        metavar=("NAME", "PATH"),
        required=True,
        help="Run name and prediction_manifest.json path. Can be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--max-note-rows", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def finite_wasserstein(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    return float(wasserstein_distance(a, b))


def finite_mean(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if len(values) else float("nan")


def finite_quantile(values, q):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, q)) if len(values) else float("nan")


def log_time(values, scale):
    values = np.asarray(values, dtype=np.float64)
    values = np.clip(values, 0.0, MAX_TIME_MS)
    return np.log1p(values / float(scale))


def skew(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return float("nan")
    std = np.std(values)
    if std <= 1e-12:
        return 0.0
    centered = values - np.mean(values)
    return float(np.mean((centered / std) ** 3))


def matrix_skew(values):
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] < 3:
        return np.full(values.shape[1], np.nan, dtype=np.float64)
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    out = np.zeros(values.shape[1], dtype=np.float64)
    valid = std > 1e-12
    out[~valid] = 0.0
    centered = values[:, valid] - mean[valid]
    out[valid] = np.mean((centered / std[valid]) ** 3, axis=0)
    return out


def per_note_wasserstein(pred_values, gt_values):
    pred_values = np.asarray(pred_values, dtype=np.float64)
    gt_values = np.asarray(gt_values, dtype=np.float64)
    if pred_values.shape[0] == 1:
        return np.mean(np.abs(gt_values - pred_values[0:1, :]), axis=0)
    return np.asarray(
        [
            finite_wasserstein(pred_values[:, note_idx], gt_values[:, note_idx])
            for note_idx in range(pred_values.shape[1])
        ],
        dtype=np.float64,
    )


def load_manifest(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def manifest_paths(manifest):
    paths = []
    seen = set()
    for item in manifest["items"]:
        for key in ("prediction_paths", "ground_truth_paths"):
            for path in item.get(key, []):
                resolved = str(Path(path).resolve())
                if resolved not in seen:
                    paths.append(resolved)
                    seen.add(resolved)
    return paths


def load_arrays(path):
    resolved = str(Path(path).resolve())
    return resolved, extract_note_arrays(Path(resolved))


def build_cache(manifests, num_workers):
    paths = []
    seen = set()
    for manifest in manifests:
        for path in manifest_paths(manifest):
            if path not in seen:
                paths.append(path)
                seen.add(path)
    if num_workers and num_workers > 1:
        ctx = get_context("fork")
        with ctx.Pool(processes=num_workers) as pool:
            return dict(pool.imap_unordered(load_arrays, paths, chunksize=4))
    return dict(load_arrays(path) for path in paths)


def note_rows_for_item(run_name, item, cache):
    pred_arrays = [cache[str(Path(path).resolve())] for path in item["prediction_paths"]]
    gt_arrays = [cache[str(Path(path).resolve())] for path in item["ground_truth_paths"]]
    all_arrays = pred_arrays + gt_arrays
    usable = min((len(arr["ioi"]) for arr in all_arrays), default=0)
    feature_stats = {}
    for feature in FEATURES:
        pred = np.stack([arr[feature][:usable] for arr in pred_arrays], axis=0).astype(np.float64)
        gt = np.stack([arr[feature][:usable] for arr in gt_arrays], axis=0).astype(np.float64)
        gt_mean = np.mean(gt, axis=0)
        pred_mean = np.mean(pred, axis=0)
        gt_std = np.std(gt, axis=0)
        pred_std = np.std(pred, axis=0)
        gt_min = np.min(gt, axis=0)
        gt_max = np.max(gt, axis=0)
        stats = {
            "gt_mean": gt_mean,
            "pred_mean": pred_mean,
            "mean_bias": pred_mean - gt_mean,
            "abs_mean_bias": np.abs(pred_mean - gt_mean),
            "gt_std": gt_std,
            "pred_std": pred_std,
            "std_ratio": np.divide(pred_std, gt_std, out=np.full_like(pred_std, np.nan), where=gt_std > 1e-12),
            "gt_skew": matrix_skew(gt),
            "pred_skew": matrix_skew(pred),
            "raw_wass": per_note_wasserstein(pred, gt),
            "pred_above_gt_mean": np.mean(pred > gt_mean[None, :], axis=0),
            "pred_below_gt_mean": np.mean(pred < gt_mean[None, :], axis=0),
            "pred_above_gt_max": np.mean(pred > gt_max[None, :], axis=0),
            "pred_below_gt_min": np.mean(pred < gt_min[None, :], axis=0),
            "gt_p10": np.quantile(gt, 0.10, axis=0),
            "gt_p90": np.quantile(gt, 0.90, axis=0),
            "pred_p10": np.quantile(pred, 0.10, axis=0),
            "pred_p90": np.quantile(pred, 0.90, axis=0),
        }
        for scale in SCALES:
            suffix = f"log{int(scale)}"
            pred_log = log_time(pred, scale)
            gt_log = log_time(gt, scale)
            gt_log_mean = np.mean(gt_log, axis=0)
            pred_log_mean = np.mean(pred_log, axis=0)
            stats[f"{suffix}_wass"] = per_note_wasserstein(pred_log, gt_log)
            stats[f"{suffix}_mean_bias"] = pred_log_mean - gt_log_mean
            stats[f"{suffix}_abs_mean_bias"] = np.abs(pred_log_mean - gt_log_mean)
            stats[f"{suffix}_gt_std"] = np.std(gt_log, axis=0)
            stats[f"{suffix}_pred_std"] = np.std(pred_log, axis=0)
        feature_stats[feature] = stats
    rows = []
    for note_idx in range(usable):
        row = {
            "run": run_name,
            "score_source": item.get("score_source"),
            "note_idx": note_idx,
            "num_pred": len(pred_arrays),
            "num_gt": len(gt_arrays),
        }
        for feature in FEATURES:
            stats = feature_stats[feature]
            for key in (
                "gt_mean",
                "pred_mean",
                "mean_bias",
                "abs_mean_bias",
                "gt_std",
                "pred_std",
                "std_ratio",
                "gt_skew",
                "pred_skew",
                "raw_wass",
                "pred_above_gt_mean",
                "pred_below_gt_mean",
                "pred_above_gt_max",
                "pred_below_gt_min",
                "gt_p10",
                "gt_p90",
                "pred_p10",
                "pred_p90",
            ):
                row[f"{feature}_{key}"] = float(stats[key][note_idx])
            for scale in SCALES:
                suffix = f"log{int(scale)}"
                for key in ("wass", "mean_bias", "abs_mean_bias", "gt_std", "pred_std"):
                    row[f"{feature}_{suffix}_{key}"] = float(stats[f"{suffix}_{key}"][note_idx])
        rows.append(row)
    return rows


def aggregate_rows(note_df):
    summary = []
    for run, group in note_df.groupby("run", sort=False):
        row = {"run": run, "notes": int(len(group))}
        for feature in FEATURES:
            for key in (
                "raw_wass",
                "abs_mean_bias",
                "mean_bias",
                "gt_std",
                "pred_std",
                "std_ratio",
                "pred_above_gt_mean",
                "pred_below_gt_mean",
                "pred_above_gt_max",
                "pred_below_gt_min",
            ):
                col = f"{feature}_{key}"
                row[col] = finite_mean(group[col])
            for scale in SCALES:
                suffix = f"log{int(scale)}"
                for key in ("wass", "abs_mean_bias", "mean_bias", "gt_std", "pred_std"):
                    col = f"{feature}_{suffix}_{key}"
                    row[col] = finite_mean(group[col])
            row[f"{feature}_raw_wass_p50"] = finite_quantile(group[f"{feature}_raw_wass"], 0.50)
            row[f"{feature}_raw_wass_p90"] = finite_quantile(group[f"{feature}_raw_wass"], 0.90)
            row[f"{feature}_log100_wass_p50"] = finite_quantile(group[f"{feature}_log100_wass"], 0.50)
            row[f"{feature}_log100_wass_p90"] = finite_quantile(group[f"{feature}_log100_wass"], 0.90)
        summary.append(row)
    return pd.DataFrame(summary)


def sample_note_rows(note_df, max_rows, seed):
    if max_rows is None or max_rows <= 0 or len(note_df) <= max_rows:
        return note_df
    rng = np.random.default_rng(seed)
    idx = rng.choice(note_df.index.to_numpy(), size=max_rows, replace=False)
    return note_df.loc[np.sort(idx)].reset_index(drop=True)


def sanitize_json(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: sanitize_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_json(v) for v in value]
    return value


def main():
    args = parse_args()
    manifests = [(name, load_manifest(path), Path(path)) for name, path in args.manifest]
    cache = build_cache([manifest for _, manifest, _ in manifests], args.num_workers)

    rows = []
    for run_name, manifest, _ in manifests:
        for item in manifest["items"]:
            rows.extend(note_rows_for_item(run_name, item, cache))
    note_df = pd.DataFrame(rows)
    summary_df = aggregate_rows(note_df)
    sampled_note_df = sample_note_rows(note_df, args.max_note_rows, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.output_dir / "timing_per_note_summary.csv"
    notes_csv = args.output_dir / "timing_per_note_rows_sample.csv"
    summary_json = args.output_dir / "timing_per_note_summary.json"
    summary_df.to_csv(summary_csv, index=False)
    sampled_note_df.to_csv(notes_csv, index=False)
    payload = {
        "manifests": [
            {
                "run": name,
                "path": str(path.resolve()),
                "protocol": manifest.get("protocol"),
                "num_samples": manifest.get("num_samples"),
                "num_scores": len(manifest.get("items", [])),
            }
            for name, manifest, path in manifests
        ],
        "summary": summary_df.to_dict(orient="records"),
        "note_rows": int(len(note_df)),
        "sampled_note_rows": int(len(sampled_note_df)),
    }
    summary_json.write_text(json.dumps(sanitize_json(payload), indent=2, ensure_ascii=False))
    print(summary_df.to_string(index=False))
    print(f"Saved {summary_csv}")
    print(f"Saved {notes_csv}")
    print(f"Saved {summary_json}")


if __name__ == "__main__":
    main()
