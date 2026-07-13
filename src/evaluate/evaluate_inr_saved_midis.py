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
    parser.add_argument("--score-source-list", type=Path, default=None)
    return parser.parse_args()


@lru_cache(maxsize=4096)
def cached_note_arrays(path: str):
    return extract_note_arrays(Path(path))


@lru_cache(maxsize=4096)
def cached_raw_output_pedal_arrays(path: str):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("predicted_target")
    if rows is None:
        rows = payload.get("predicted_target7")
    if rows is None:
        raise KeyError(f"No predicted_target or predicted_target7 in {path}")
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] < 7:
        raise ValueError(f"Unexpected predicted target shape for {path}: {rows.shape}")
    return {
        "pedal_0": rows[:, 3],
        "pedal_25": rows[:, 4],
        "pedal_50": rows[:, 5],
        "pedal_75": rows[:, 6],
    }


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


def binary_pedal_arrays(note_arrays, threshold=64.0):
    return {
        key: (np.asarray(note_arrays[key], dtype=np.float64) >= float(threshold)).astype(np.float64)
        for key in ("pedal_0", "pedal_25", "pedal_50", "pedal_75")
    }


def raw_or_thresholded_pred_pedal_arrays(prediction_path, raw_output_path=None, threshold=64.0):
    if raw_output_path is not None:
        try:
            return cached_raw_output_pedal_arrays(str(Path(raw_output_path).resolve()))
        except Exception:
            pass
    return binary_pedal_arrays(cached_note_arrays(str(Path(prediction_path).resolve())), threshold=threshold)


def log_time_values(values, scale=LOG_WASS_SCALE, max_time_ms=LOG_WASS_MAX_TIME_MS):
    values = np.asarray(values, dtype=np.float64)
    values = np.clip(values, 0.0, float(max_time_ms))
    return np.log1p(values / float(scale))


def pp_wass_metrics(prediction_paths, gt_paths, raw_output_paths=None, pedal_binary_support=False, pedal_binary_threshold=64.0):
    pred_arrays = [cached_note_arrays(path) for path in prediction_paths]
    gt_arrays = [cached_note_arrays(path) for path in gt_paths]
    pred_pedal_arrays = None
    gt_pedal_arrays = None
    if pedal_binary_support:
        pred_pedal_arrays = [
            raw_or_thresholded_pred_pedal_arrays(
                prediction_paths[idx],
                raw_output_path=raw_output_paths[idx] if raw_output_paths is not None and idx < len(raw_output_paths) else None,
                threshold=pedal_binary_threshold,
            )
            for idx in range(len(prediction_paths))
        ]
        gt_pedal_arrays = [binary_pedal_arrays(item, threshold=pedal_binary_threshold) for item in gt_arrays]

    output = {}
    for metric_name, feature_name in FEATURE_KEYS:
        if metric_name.startswith("pedal_") and pedal_binary_support:
            pred_source = pred_pedal_arrays
            gt_source = gt_pedal_arrays
        else:
            pred_source = pred_arrays
            gt_source = gt_arrays
        pred_pool = np.concatenate([item[feature_name] for item in pred_source]) if pred_source else np.asarray([], dtype=np.float64)
        gt_pool = np.concatenate([item[feature_name] for item in gt_source]) if gt_source else np.asarray([], dtype=np.float64)
        output[f"{metric_name}_wass"] = feature_wasserstein(pred_pool, gt_pool)
        if metric_name in {"ioi", "duration"}:
            output[f"{metric_name}_log50_wass"] = feature_wasserstein(
                log_time_values(pred_pool),
                log_time_values(gt_pool),
            )

    pedal_keys = [f"{name}_wass" for name in ("pedal_0", "pedal_25", "pedal_50", "pedal_75")]
    output["pedal_wass"] = finite_mean([output[key] for key in pedal_keys])
    output["pedal_start_wass"] = feature_wasserstein(
        np.concatenate([item["pedal_0"] for item in pred_arrays]) if pred_arrays else np.asarray([], dtype=np.float64),
        np.concatenate([item["pedal_0"] for item in gt_arrays]) if gt_arrays else np.asarray([], dtype=np.float64),
    )
    return output


def pn_wass_metrics(prediction_paths, gt_paths, raw_output_paths=None, pedal_binary_support=False, pedal_binary_threshold=64.0):
    pred_arrays = [cached_note_arrays(path) for path in prediction_paths]
    gt_arrays = [cached_note_arrays(path) for path in gt_paths]
    all_arrays = pred_arrays + gt_arrays
    pred_pedal_arrays = None
    gt_pedal_arrays = None
    if pedal_binary_support:
        pred_pedal_arrays = [
            raw_or_thresholded_pred_pedal_arrays(
                prediction_paths[idx],
                raw_output_path=raw_output_paths[idx] if raw_output_paths is not None and idx < len(raw_output_paths) else None,
                threshold=pedal_binary_threshold,
            )
            for idx in range(len(prediction_paths))
        ]
        gt_pedal_arrays = [binary_pedal_arrays(item, threshold=pedal_binary_threshold) for item in gt_arrays]

    output = {}
    for metric_name, feature_name in FEATURE_KEYS:
        if metric_name.startswith("pedal_") and pedal_binary_support:
            pred_source = pred_pedal_arrays
            gt_source = gt_pedal_arrays
            all_source = pred_pedal_arrays + gt_pedal_arrays
        else:
            pred_source = pred_arrays
            gt_source = gt_arrays
            all_source = all_arrays
        usable = min((len(item[feature_name]) for item in all_source), default=0)
        note_wass = [
            feature_wasserstein(
                [item[feature_name][note_idx] for item in pred_source],
                [item[feature_name][note_idx] for item in gt_source],
            )
            for note_idx in range(usable)
        ]
        output[f"{metric_name}_wass"] = finite_mean(note_wass)
        if metric_name in {"ioi", "duration"}:
            note_log_wass = [
                feature_wasserstein(
                    [log_time_values([item[feature_name][note_idx]])[0] for item in pred_source],
                    [log_time_values([item[feature_name][note_idx]])[0] for item in gt_source],
                )
                for note_idx in range(usable)
            ]
            output[f"{metric_name}_log50_wass"] = finite_mean(note_log_wass)

    pedal_keys = [f"{name}_wass" for name in ("pedal_0", "pedal_25", "pedal_50", "pedal_75")]
    output["pedal_wass"] = finite_mean([output[key] for key in pedal_keys])
    usable = min((len(item["pedal_0"]) for item in all_arrays), default=0)
    output["pedal_start_wass"] = finite_mean(
        [
            feature_wasserstein(
                [item["pedal_0"][note_idx] for item in pred_arrays],
                [item["pedal_0"][note_idx] for item in gt_arrays],
            )
            for note_idx in range(usable)
        ]
    )
    return output


def score_level_metrics(item, max_gt_per_score=None, pedal_binary_support=False, pedal_binary_threshold=64.0):
    prediction_paths = item["prediction_paths"]
    raw_output_paths = item.get("raw_output_paths")
    gt_paths = item["ground_truth_paths"]
    if max_gt_per_score is not None:
        gt_paths = gt_paths[:max_gt_per_score]

    pn_wass = pn_wass_metrics(
        prediction_paths,
        gt_paths,
        raw_output_paths=raw_output_paths,
        pedal_binary_support=pedal_binary_support,
        pedal_binary_threshold=pedal_binary_threshold,
    )
    pp_wass = pp_wass_metrics(
        prediction_paths,
        gt_paths,
        raw_output_paths=raw_output_paths,
        pedal_binary_support=pedal_binary_support,
        pedal_binary_threshold=pedal_binary_threshold,
    )

    return {
        "score_source": item["score_source"],
        "num_predictions": len(prediction_paths),
        "num_ground_truth": len(gt_paths),
        "pn_wass": pn_wass,
        "pp_wass": pp_wass,
    }


def score_level_metrics_worker(args):
    item, max_gt_per_score, pedal_binary_support, pedal_binary_threshold = args
    return score_level_metrics(
        item,
        max_gt_per_score=max_gt_per_score,
        pedal_binary_support=pedal_binary_support,
        pedal_binary_threshold=pedal_binary_threshold,
    )


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


def load_score_source_filter(path):
    if path is None:
        return None
    selected = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        selected.append(line)
    return selected or None


def filter_manifest_items(items, score_sources):
    if not score_sources:
        return items
    wanted = set(score_sources)
    order = {score_source: idx for idx, score_source in enumerate(score_sources)}
    filtered = [item for item in items if item.get("score_source") in wanted]
    found = {item.get("score_source") for item in filtered}
    missing = [score_source for score_source in score_sources if score_source not in found]
    if missing:
        raise ValueError(f"Requested score_source not found in manifest: {missing[0]}")
    filtered.sort(key=lambda item: order[item["score_source"]])
    return filtered


def load_manifest_and_config(manifest_path, score_source_list=None):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["items"] = filter_manifest_items(manifest["items"], score_source_list)
    config_path = manifest.get("config")
    if config_path is None:
        fallback = manifest_path.parent.parent / "config.json"
        if fallback.exists():
            config_path = str(fallback.resolve())
    config = {}
    if config_path:
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    return manifest, config


def evaluate_manifest(manifest, max_gt_per_score=None, num_workers=10, pedal_binary_support=False, pedal_binary_threshold=64.0):
    if num_workers and num_workers > 1:
        ctx = get_context("spawn")
        with ctx.Pool(processes=num_workers) as pool:
            score_rows = list(
                pool.imap(
                    score_level_metrics_worker,
                    (
                        (item, max_gt_per_score, pedal_binary_support, pedal_binary_threshold)
                        for item in manifest["items"]
                    ),
                    chunksize=1,
                )
            )
    else:
        score_rows = [
            score_level_metrics(
                item,
                max_gt_per_score=max_gt_per_score,
                pedal_binary_support=pedal_binary_support,
                pedal_binary_threshold=pedal_binary_threshold,
            )
            for item in manifest["items"]
        ]
    return {
        "prediction_manifest": manifest,
        "score_rows": score_rows,
    }


def main():
    args = parse_args()
    score_source_list = load_score_source_filter(args.score_source_list)
    manifest, config = load_manifest_and_config(args.prediction_manifest, score_source_list=score_source_list)
    pedal_binary_support = str(config.get("pedal_representation", "")).lower() == "binary_4"
    evaluation = evaluate_manifest(
        manifest,
        max_gt_per_score=args.max_gt_per_score,
        num_workers=args.num_workers,
        pedal_binary_support=pedal_binary_support,
        pedal_binary_threshold=float(config.get("pedal_binary_threshold", 64.0)),
    )
    score_rows = evaluation["score_rows"]

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
        "pedal_metric_support": "binary_0_1" if pedal_binary_support else "raw_0_127",
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
