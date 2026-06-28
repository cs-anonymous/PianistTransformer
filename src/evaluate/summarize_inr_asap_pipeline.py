import argparse
import json
import math
import sys
from multiprocessing import get_context
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluate.compute_saved_midi_mae_wass import extract_note_arrays
from src.evaluate.evaluate_inr_saved_midis import (
    aggregate_score_metrics,
    score_level_metrics,
    score_level_metrics_worker,
)
from src.train.train_inr import build_work_manifest
from src.utils.inr_midi import normalize_time_ms_for_inr_input


FEATURES = [
    ("ioi", "IOI (ms)"),
    ("duration", "Duration (ms)"),
    ("normal_ioi", "Normalized IOI"),
    ("normal_duration", "Normalized Duration"),
    ("velocity", "Velocity"),
    ("pedal", "Pedal"),
]

DEV_FEATURES = [
    ("dev_ioi", "IOI Deviation"),
    ("dev_duration", "Duration Ratio"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize deterministic/sampling INR ASAP inference into one JSON and one distribution plot."
    )
    parser.add_argument("--deterministic-manifest", type=Path, required=True)
    parser.add_argument("--sampling-manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-plot", type=Path, required=True)
    parser.add_argument("--output-dev-plot", type=Path, default=None)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-output-dir", type=Path, required=True)
    parser.add_argument("--pipeline-log", type=Path, required=True)
    parser.add_argument("--evaluate-log", type=Path, required=True)
    parser.add_argument("--max-gt-per-score", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def load_manifest(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sanitize_for_json(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    return value


def compute_manifest_metrics(manifest, max_gt_per_score=None, num_workers=1):
    items = manifest["items"]
    if num_workers and num_workers > 1:
        ctx = get_context("spawn")
        with ctx.Pool(processes=num_workers) as pool:
            score_rows = list(
                tqdm(
                    pool.imap(
                        score_level_metrics_worker,
                        ((item, max_gt_per_score) for item in items),
                        chunksize=1,
                    ),
                    total=len(items),
                    desc=f"{manifest['protocol']} score metrics",
                )
            )
    else:
        score_rows = [
            score_level_metrics(item, max_gt_per_score=max_gt_per_score)
            for item in tqdm(items, total=len(items), desc=f"{manifest['protocol']} score metrics")
        ]

    return {
        "protocol": manifest["protocol"],
        "num_samples": manifest["num_samples"],
        "num_scores": len(score_rows),
        "aggregate": {
            "pn_wass": aggregate_score_metrics(score_rows, "pn_wass"),
            "pp_wass": aggregate_score_metrics(score_rows, "pp_wass"),
        },
        "scores": score_rows,
    }


def unique_paths(manifest, key):
    paths = []
    seen = set()
    for item in manifest["items"]:
        for path in item[key]:
            resolved = str(Path(path).resolve())
            if resolved not in seen:
                paths.append(resolved)
                seen.add(resolved)
    return paths


def normalize_time_array(values, timing_normalization, max_time_ms):
    return np.asarray(
        [
            normalize_time_ms_for_inr_input(
                value,
                normalization=timing_normalization,
                max_time_ms=max_time_ms,
            )
            for value in values
        ],
        dtype=np.float64,
    )


def enrich_arrays(arrays, timing_normalization, max_time_ms):
    enriched = dict(arrays)
    enriched["normal_ioi"] = normalize_time_array(arrays["ioi"], timing_normalization, max_time_ms)
    enriched["normal_duration"] = normalize_time_array(arrays["duration"], timing_normalization, max_time_ms)
    pedal_arrays = [arrays[f"pedal_{pos}"] for pos in ("0", "25", "50", "75")]
    enriched["pedal"] = np.concatenate(pedal_arrays) if pedal_arrays else np.asarray([], dtype=np.float64)
    return enriched


def load_arrays_worker(args):
    path, timing_normalization, max_time_ms = args
    resolved = str(Path(path).resolve())
    return resolved, enrich_arrays(
        extract_note_arrays(Path(resolved)),
        timing_normalization=timing_normalization,
        max_time_ms=max_time_ms,
    )


def build_array_cache(paths, num_workers, timing_normalization, max_time_ms):
    unique = []
    seen = set()
    for path in paths:
        resolved = str(Path(path).resolve())
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    worker_args = [(path, timing_normalization, max_time_ms) for path in unique]
    if num_workers and num_workers > 1:
        ctx = get_context("spawn")
        with ctx.Pool(processes=num_workers) as pool:
            rows = list(
                tqdm(
                    pool.imap(load_arrays_worker, worker_args, chunksize=8),
                    total=len(unique),
                    desc="distribution MIDI features",
                )
            )
    else:
        rows = [
            load_arrays_worker(args)
            for args in tqdm(worker_args, total=len(worker_args), desc="distribution MIDI features")
        ]
    return dict(rows)


def pooled_feature(array_cache, paths, feature):
    chunks = []
    for path in paths:
        arrays = array_cache[str(Path(path).resolve())]
        values = arrays[feature]
        if len(values):
            chunks.append(values)
    if not chunks:
        return np.asarray([], dtype=np.float64)
    return np.concatenate(chunks).astype(np.float64, copy=False)


def finite_values(values):
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def histogram_range(*arrays):
    non_empty = [array for array in arrays if len(array)]
    if not non_empty:
        return (0.0, 1.0)
    merged = finite_values(np.concatenate(non_empty))
    if len(merged) == 0:
        return (0.0, 1.0)
    low = float(np.percentile(merged, 0.5))
    high = float(np.percentile(merged, 99.5))
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        low = float(np.min(merged))
        high = float(np.max(merged))
    if high <= low:
        high = low + 1.0
    return low, high


def plot_distributions(
    det_manifest,
    sampling_manifest,
    output_plot,
    num_workers,
    timing_normalization,
    max_time_ms,
):
    gt_paths = unique_paths(det_manifest, "ground_truth_paths")
    det_paths = unique_paths(det_manifest, "prediction_paths")
    sampling_paths = unique_paths(sampling_manifest, "prediction_paths")
    array_cache = build_array_cache(
        gt_paths + det_paths + sampling_paths,
        num_workers,
        timing_normalization=timing_normalization,
        max_time_ms=max_time_ms,
    )

    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    colors = {
        "gt": "#222222",
        "det": "#2f6fed",
        "sampling": "#d0522b",
    }

    for idx, (feature, title) in enumerate(FEATURES):
        axis = axes[idx]
        gt = pooled_feature(array_cache, gt_paths, feature)
        det = pooled_feature(array_cache, det_paths, feature)
        sampling = pooled_feature(array_cache, sampling_paths, feature)
        low, high = histogram_range(gt, det, sampling)
        bins = np.linspace(low, high, 80)

        for values, label, color, alpha in [
            (gt, "ground truth", colors["gt"], 0.26),
            (det, "deterministic", colors["det"], 0.32),
            (sampling, "sampling", colors["sampling"], 0.32),
        ]:
            values = finite_values(values)
            values = values[(values >= low) & (values <= high)]
            if len(values):
                axis.hist(values, bins=bins, density=True, alpha=alpha, label=label, color=color)

        axis.set_title(title)
        axis.set_ylabel("density")
        axis.grid(alpha=0.2)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("ASAP Test Label Distribution: Ground Truth vs INR Predictions", y=0.98)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(output_plot, dpi=180)
    plt.close(fig)

    return {
        "ground_truth_midis": len(gt_paths),
        "deterministic_prediction_midis": len(det_paths),
        "sampling_prediction_midis": len(sampling_paths),
    }


def normalize_ioi_dev(score_ioi_ms, perf_ioi_ms):
    dev_ms = float(perf_ioi_ms) - float(score_ioi_ms)
    return min(max((dev_ms + 500.0) / 1000.0, 0.0), 1.0)


def normalize_duration_ratio(score_duration_ms, perf_duration_ms, eps=1e-6):
    ratio = float(perf_duration_ms) / max(float(score_duration_ms), float(eps))
    return min(max(ratio / 5.0, 0.0), 1.0)


def load_score_source_to_work_path(config):
    manifest = build_work_manifest(
        metadata_path=config["metadata_path"],
        refined_dir=config["refined_dir"],
        split="test",
        block_notes=config["block_notes"],
        overlap_ratio=config["overlap_ratio"],
        min_notes=config["min_notes"],
        performance_dataset="ASAP",
    )
    return {
        item["score_source"]: str((ROOT_DIR / item["path"]).resolve())
        for item in manifest
    }


def extract_gt_dev_arrays(score_source_to_work_path):
    dev_ioi = []
    dev_duration = []
    for work_path in score_source_to_work_path.values():
        with open(work_path, "r", encoding="utf-8") as file:
            work = json.load(file)
        score_raw = work["score"]["score_raw"]
        for perf in work.get("performances", []):
            shared_rows = perf.get("label_shared_raw")
            if shared_rows is None:
                if "label_raw" not in perf:
                    continue
                shared_rows = [row[:3] for row in perf["label_raw"]]
            if len(shared_rows) != len(score_raw):
                continue
            for score_row, perf_row in zip(score_raw, shared_rows):
                dev_ioi.append(normalize_ioi_dev(score_row[0], perf_row[0]))
                dev_duration.append(normalize_duration_ratio(score_row[1], perf_row[1]))
    return {
        "dev_ioi": np.asarray(dev_ioi, dtype=np.float64),
        "dev_duration": np.asarray(dev_duration, dtype=np.float64),
    }


def extract_pred_dev_arrays(raw_output_paths):
    dev_ioi = []
    dev_duration = []
    for path in raw_output_paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        target5 = payload.get("predicted_target5", [])
        for row in target5:
            dev_ioi.append(float(row[0]))
            dev_duration.append(float(row[1]))
    return {
        "dev_ioi": np.asarray(dev_ioi, dtype=np.float64),
        "dev_duration": np.asarray(dev_duration, dtype=np.float64),
    }


def analyze_dev_distribution(
    det_manifest,
    sampling_manifest,
    config,
    output_plot,
):
    return plot_dev_distributions(det_manifest, sampling_manifest, config, output_plot)


def plot_dev_distributions(
    det_manifest,
    sampling_manifest,
    config,
    output_plot,
):
    score_source_to_work_path = load_score_source_to_work_path(config)
    gt_arrays = extract_gt_dev_arrays(score_source_to_work_path)
    det_arrays = extract_pred_dev_arrays(unique_paths(det_manifest, "raw_output_paths"))
    sampling_arrays = extract_pred_dev_arrays(unique_paths(sampling_manifest, "raw_output_paths"))

    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = {
        "gt": "#222222",
        "det": "#2f6fed",
        "sampling": "#d0522b",
    }

    for idx, (feature, title) in enumerate(DEV_FEATURES):
        axis = axes[idx]
        gt = finite_values(gt_arrays[feature])
        det = finite_values(det_arrays[feature])
        sampling = finite_values(sampling_arrays[feature])
        bins = np.linspace(0.0, 1.0, 80)

        for values, label, color, alpha in [
            (gt, "ground truth", colors["gt"], 0.26),
            (det, "deterministic", colors["det"], 0.32),
            (sampling, "sampling", colors["sampling"], 0.32),
        ]:
            values = values[(values >= 0.0) & (values <= 1.0)]
            if len(values):
                axis.hist(values, bins=bins, density=True, alpha=alpha, label=label, color=color)
        axis.set_title(title)
        axis.set_xlim(0.0, 1.0)
        axis.set_ylabel("density")
        axis.grid(alpha=0.2)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("ASAP Test Dev Distribution: Ground Truth vs INR Predictions", y=0.98)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(output_plot, dpi=180)
    plt.close(fig)

    return {
        "ground_truth_dev_notes": int(len(gt_arrays["dev_ioi"])),
        "deterministic_dev_notes": int(len(det_arrays["dev_ioi"])),
        "sampling_dev_notes": int(len(sampling_arrays["dev_ioi"])),
    }


def main():
    args = parse_args()
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")

    det_manifest = load_manifest(args.deterministic_manifest)
    sampling_manifest = load_manifest(args.sampling_manifest)
    config = load_manifest(args.config)
    timing_normalization = config.get("timing_input_normalization", "legacy_log1p")
    max_time_ms = float(config.get("max_time_ms", 10000.0))

    det_metrics = compute_manifest_metrics(
        det_manifest,
        max_gt_per_score=args.max_gt_per_score,
        num_workers=args.num_workers,
    )
    sampling_metrics = compute_manifest_metrics(
        sampling_manifest,
        max_gt_per_score=args.max_gt_per_score,
        num_workers=args.num_workers,
    )
    plot_summary = plot_distributions(
        det_manifest,
        sampling_manifest,
        args.output_plot,
        num_workers=args.num_workers,
        timing_normalization=timing_normalization,
        max_time_ms=max_time_ms,
    )
    output_dev_plot = args.output_dev_plot or args.output_plot.with_name("asap_dev_distribution.png")
    dev_plot_summary = plot_dev_distributions(
        det_manifest,
        sampling_manifest,
        config,
        output_dev_plot,
    )

    output = {
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "train_output_dir": str(args.train_output_dir.resolve()),
        "pipeline_log": str(args.pipeline_log.resolve()),
        "evaluate_log": str(args.evaluate_log.resolve()),
        "distribution_plot": str(args.output_plot.resolve()),
        "dev_distribution_plot": str(output_dev_plot.resolve()),
        "dataset": {
            "split": det_manifest.get("split"),
            "performance_dataset": det_manifest.get("performance_dataset"),
            "exclude_performance_dataset": det_manifest.get("exclude_performance_dataset"),
            "timing_normalization": timing_normalization,
            **plot_summary,
            **dev_plot_summary,
        },
        "metrics": {
            "deterministic": det_metrics,
            "sampling": sampling_metrics,
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    output = sanitize_for_json(output)
    args.output_json.write_text(
        json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(sanitize_for_json({
        "output_json": str(args.output_json),
        "distribution_plot": str(args.output_plot),
        "dev_distribution_plot": str(output_dev_plot),
        "deterministic": det_metrics["aggregate"],
        "sampling": sampling_metrics["aggregate"],
    }), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
