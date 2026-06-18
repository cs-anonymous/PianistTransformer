import argparse
import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluate.compute_saved_midi_mae_wass import compute_pair_metrics, extract_note_arrays
from scipy.stats import wasserstein_distance


PAIRWISE_KEYS = [
    "ioi_mae",
    "ioi_wass",
    "duration_mae",
    "duration_wass",
    "velocity_mae",
    "velocity_wass",
    "pedal_mae",
    "pedal_wass",
]

FEATURE_KEYS = [
    ("ioi", "ioi"),
    ("duration", "duration"),
    ("velocity", "velocity"),
    ("pedal_0", "pedal_0"),
    ("pedal_25", "pedal_25"),
    ("pedal_50", "pedal_50"),
    ("pedal_75", "pedal_75"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate deterministic or sampling INR predictions against multi-reference GT sets.")
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-gt-per-score", type=int, default=None)
    return parser.parse_args()


@lru_cache(maxsize=4096)
def cached_pair_metrics(pred_path: str, gt_path: str):
    return compute_pair_metrics(Path(pred_path), Path(gt_path))


@lru_cache(maxsize=4096)
def cached_note_arrays(path: str):
    return extract_note_arrays(Path(path))


def mean_dict(rows, keys):
    if not rows:
        return {key: float("nan") for key in keys}
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def pooled_distribution_metrics(prediction_paths, gt_paths):
    pred_arrays = [cached_note_arrays(path) for path in prediction_paths]
    gt_arrays = [cached_note_arrays(path) for path in gt_paths]

    output = {}
    for metric_name, feature_name in FEATURE_KEYS:
        pred_pool = np.concatenate([item[feature_name] for item in pred_arrays]) if pred_arrays else np.asarray([], dtype=np.float64)
        gt_pool = np.concatenate([item[feature_name] for item in gt_arrays]) if gt_arrays else np.asarray([], dtype=np.float64)
        if len(pred_pool) == 0 or len(gt_pool) == 0:
            output[f"{metric_name}_pooled_wass"] = float("nan")
        else:
            output[f"{metric_name}_pooled_wass"] = float(wasserstein_distance(pred_pool, gt_pool))

    pedal_keys = [f"{name}_pooled_wass" for name in ("pedal_0", "pedal_25", "pedal_50", "pedal_75")]
    output["pedal_pooled_wass"] = float(np.mean([output[key] for key in pedal_keys]))
    return output


def pairwise_expected_metrics(prediction_paths, gt_paths):
    rows = []
    for pred_path in prediction_paths:
        for gt_path in gt_paths:
            rows.append(cached_pair_metrics(pred_path, gt_path))
    return mean_dict(rows, PAIRWISE_KEYS)


def self_pairwise_metrics(paths):
    rows = []
    for idx in range(len(paths)):
        for jdx in range(idx + 1, len(paths)):
            rows.append(cached_pair_metrics(paths[idx], paths[jdx]))
    return mean_dict(rows, PAIRWISE_KEYS)


def score_level_metrics(item, max_gt_per_score=None):
    prediction_paths = item["prediction_paths"]
    gt_paths = item["ground_truth_paths"]
    if max_gt_per_score is not None:
        gt_paths = gt_paths[:max_gt_per_score]

    pairwise = pairwise_expected_metrics(prediction_paths, gt_paths)
    pooled = pooled_distribution_metrics(prediction_paths, gt_paths)
    model_model = self_pairwise_metrics(prediction_paths)
    human_human = self_pairwise_metrics(gt_paths)

    return {
        "score_source": item["score_source"],
        "num_predictions": len(prediction_paths),
        "num_ground_truth": len(gt_paths),
        "expected_pairwise": pairwise,
        "pooled_distribution": pooled,
        "model_model_diversity": model_model,
        "human_human_diversity": human_human,
    }


def aggregate_score_metrics(score_rows, section):
    if not score_rows:
        return {}
    keys = sorted(score_rows[0][section].keys())
    return {key: float(np.mean([row[section][key] for row in score_rows])) for key in keys}


def main():
    args = parse_args()
    manifest = json.loads(args.prediction_manifest.read_text())
    score_rows = [
        score_level_metrics(item, max_gt_per_score=args.max_gt_per_score)
        for item in manifest["items"]
    ]

    output = {
        "prediction_manifest": str(args.prediction_manifest.resolve()),
        "protocol": manifest["protocol"],
        "num_samples": manifest["num_samples"],
        "num_scores": len(score_rows),
        "aggregate": {
            "expected_pairwise": aggregate_score_metrics(score_rows, "expected_pairwise"),
            "pooled_distribution": aggregate_score_metrics(score_rows, "pooled_distribution"),
            "model_model_diversity": aggregate_score_metrics(score_rows, "model_model_diversity"),
            "human_human_diversity": aggregate_score_metrics(score_rows, "human_human_diversity"),
        },
        "scores": score_rows,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(json.dumps(output["aggregate"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
