import argparse
import json
import math
import sys
from multiprocessing import get_context
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluate.compute_saved_midi_mae_wass import extract_note_arrays
from scipy.stats import wasserstein_distance


FEATURE_KEYS = [
    ("ioi", "ioi"),
    ("duration", "duration"),
    ("velocity", "velocity"),
    ("pedal_0", "pedal_0"),
    ("pedal_25", "pedal_25"),
    ("pedal_50", "pedal_50"),
    ("pedal_75", "pedal_75"),
]

LOG_WASS_SCALE = 50.0
LOG_WASS_MAX_TIME_MS = 5000.0


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate deterministic or sampling INR predictions against multi-reference GT sets.")
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-gt-per-score", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=10)
    return parser.parse_args()


@lru_cache(maxsize=4096)
def cached_note_arrays(path: str):
    return extract_note_arrays(Path(path))


def finite_mean(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if len(finite) else float("nan")


def feature_wasserstein(pred_values, gt_values):
    pred_values = np.asarray(pred_values, dtype=np.float64)
    gt_values = np.asarray(gt_values, dtype=np.float64)
    pred_values = pred_values[np.isfinite(pred_values)]
    gt_values = gt_values[np.isfinite(gt_values)]
    if len(pred_values) == 0 or len(gt_values) == 0:
        return float("nan")
    return float(wasserstein_distance(pred_values, gt_values))


def log_time_values(values, scale=LOG_WASS_SCALE, max_time_ms=LOG_WASS_MAX_TIME_MS):
    values = np.asarray(values, dtype=np.float64)
    values = np.clip(values, 0.0, float(max_time_ms))
    return np.log1p(values / float(scale))


def pp_wass_metrics(prediction_paths, gt_paths):
    pred_arrays = [cached_note_arrays(path) for path in prediction_paths]
    gt_arrays = [cached_note_arrays(path) for path in gt_paths]

    output = {}
    for metric_name, feature_name in FEATURE_KEYS:
        pred_pool = np.concatenate([item[feature_name] for item in pred_arrays]) if pred_arrays else np.asarray([], dtype=np.float64)
        gt_pool = np.concatenate([item[feature_name] for item in gt_arrays]) if gt_arrays else np.asarray([], dtype=np.float64)
        output[f"{metric_name}_wass"] = feature_wasserstein(pred_pool, gt_pool)
        if metric_name in {"ioi", "duration"}:
            output[f"{metric_name}_log50_wass"] = feature_wasserstein(
                log_time_values(pred_pool),
                log_time_values(gt_pool),
            )

    pedal_keys = [f"{name}_wass" for name in ("pedal_0", "pedal_25", "pedal_50", "pedal_75")]
    output["pedal_wass"] = finite_mean([output[key] for key in pedal_keys])
    return output


def pn_wass_metrics(prediction_paths, gt_paths):
    pred_arrays = [cached_note_arrays(path) for path in prediction_paths]
    gt_arrays = [cached_note_arrays(path) for path in gt_paths]
    all_arrays = pred_arrays + gt_arrays

    output = {}
    for metric_name, feature_name in FEATURE_KEYS:
        usable = min((len(item[feature_name]) for item in all_arrays), default=0)
        note_wass = [
            feature_wasserstein(
                [item[feature_name][note_idx] for item in pred_arrays],
                [item[feature_name][note_idx] for item in gt_arrays],
            )
            for note_idx in range(usable)
        ]
        output[f"{metric_name}_wass"] = finite_mean(note_wass)
        if metric_name in {"ioi", "duration"}:
            note_log_wass = [
                feature_wasserstein(
                    [log_time_values([item[feature_name][note_idx]])[0] for item in pred_arrays],
                    [log_time_values([item[feature_name][note_idx]])[0] for item in gt_arrays],
                )
                for note_idx in range(usable)
            ]
            output[f"{metric_name}_log50_wass"] = finite_mean(note_log_wass)

    pedal_keys = [f"{name}_wass" for name in ("pedal_0", "pedal_25", "pedal_50", "pedal_75")]
    output["pedal_wass"] = finite_mean([output[key] for key in pedal_keys])
    return output


def score_level_metrics(item, max_gt_per_score=None):
    prediction_paths = item["prediction_paths"]
    gt_paths = item["ground_truth_paths"]
    if max_gt_per_score is not None:
        gt_paths = gt_paths[:max_gt_per_score]

    pn_wass = pn_wass_metrics(prediction_paths, gt_paths)
    pp_wass = pp_wass_metrics(prediction_paths, gt_paths)

    return {
        "score_source": item["score_source"],
        "num_predictions": len(prediction_paths),
        "num_ground_truth": len(gt_paths),
        "pn_wass": pn_wass,
        "pp_wass": pp_wass,
    }


def score_level_metrics_worker(args):
    item, max_gt_per_score = args
    return score_level_metrics(item, max_gt_per_score=max_gt_per_score)


def aggregate_score_metrics(score_rows, section):
    if not score_rows:
        return {}
    keys = sorted(score_rows[0][section].keys())
    output = {}
    for key in keys:
        values = np.asarray([row[section][key] for row in score_rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        output[key] = float(np.mean(finite)) if len(finite) else float("nan")
    return output


def main():
    args = parse_args()
    manifest = json.loads(args.prediction_manifest.read_text())
    if args.num_workers and args.num_workers > 1:
        ctx = get_context("spawn")
        with ctx.Pool(processes=args.num_workers) as pool:
            score_rows = list(
                pool.imap(
                    score_level_metrics_worker,
                    ((item, args.max_gt_per_score) for item in manifest["items"]),
                    chunksize=1,
                )
            )
    else:
        score_rows = [
            score_level_metrics(item, max_gt_per_score=args.max_gt_per_score)
            for item in manifest["items"]
        ]

    output = {
        "prediction_manifest": str(args.prediction_manifest.resolve()),
        "protocol": manifest["protocol"],
        "num_samples": manifest["num_samples"],
        "num_scores": len(score_rows),
        "log_wass": {
            "scale": LOG_WASS_SCALE,
            "max_time_ms": LOG_WASS_MAX_TIME_MS,
            "features": ["ioi", "duration"],
        },
        "aggregate": {
            "pn_wass": aggregate_score_metrics(score_rows, "pn_wass"),
            "pp_wass": aggregate_score_metrics(score_rows, "pp_wass"),
        },
        "scores": score_rows,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(json.dumps(output["aggregate"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
