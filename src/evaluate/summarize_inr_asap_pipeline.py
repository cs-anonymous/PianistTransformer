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
from src.train.train_inr import (
    normalize_duration_dev as normalize_train_duration_dev,
    normalize_ioi_dev as normalize_train_ioi_dev,
)
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
    ("log_dev_ioi", "Logscale IOI Dev"),
    ("log_dev_duration", "Logscale Duration Dev"),
    ("raw_dev_ioi_s", "Raw IOI Dev (s)"),
    ("raw_dev_duration_s", "Raw Duration Dev (s)"),
]

DEV_GROUPS = [
    ("zero_ioi", "Score IOI = 0"),
    ("nonzero_ioi", "Score IOI > 0"),
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


def compute_manifest_metrics(
    manifest,
    max_gt_per_score=None,
    num_workers=1,
    pedal_binary_support=False,
    pedal_binary_threshold=64.0,
):
    items = manifest["items"]
    if num_workers and num_workers > 1:
        ctx = get_context("spawn")
        with ctx.Pool(processes=num_workers) as pool:
            score_rows = list(
                tqdm(
                    pool.imap(
                        score_level_metrics_worker,
                        (
                            (item, max_gt_per_score, pedal_binary_support, pedal_binary_threshold)
                            for item in items
                        ),
                        chunksize=1,
                    ),
                    total=len(items),
                    desc=f"{manifest['protocol']} score metrics",
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
        gt_paths + sampling_paths,
        num_workers,
        timing_normalization=timing_normalization,
        max_time_ms=max_time_ms,
    )

    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    colors = {
        "gt": "#222222",
        "sampling": "#d0522b",
    }

    for idx, (feature, title) in enumerate(FEATURES):
        axis = axes[idx]
        gt = pooled_feature(array_cache, gt_paths, feature)
        sampling = pooled_feature(array_cache, sampling_paths, feature)
        low, high = histogram_range(gt, sampling)
        bins = np.linspace(low, high, 80)

        for values, label, color, alpha in [
            (gt, "score-span GT", colors["gt"], 0.26),
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
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("ASAP Test Label Distribution: Score-Span GT vs Sampling", y=0.98)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(output_plot, dpi=180)
    plt.close(fig)

    return {
        "ground_truth_midis": len(gt_paths),
        "deterministic_prediction_midis": len(det_paths),
        "sampling_prediction_midis": len(sampling_paths),
    }


def empty_dev_groups():
    return {
        group: {
            "log_dev_ioi": [],
            "log_dev_duration": [],
            "raw_dev_ioi_s": [],
            "raw_dev_duration_s": [],
        }
        for group, _ in DEV_GROUPS
    }


def dev_group_name(score_ioi_ms):
    return "zero_ioi" if float(score_ioi_ms) <= 0.0 else "nonzero_ioi"


def eval_gt_time_normalization(config):
    return str(
        config.get(
            "eval_gt_time_normalization",
            config.get("evaluation_gt_time_normalization", "none"),
        )
        or "none"
    ).lower()


def score_onset_span_scale(score_raw, shared_rows):
    score_span = float(sum(float(row[0]) for row in score_raw[1:]))
    perf_span = float(sum(float(row[0]) for row in shared_rows[1:]))
    if not math.isfinite(score_span) or score_span <= 0.0:
        raise ValueError(f"Invalid score onset span: {score_span}")
    if not math.isfinite(perf_span) or perf_span <= 0.0:
        raise ValueError(f"Invalid performance onset span: {perf_span}")
    return score_span / perf_span


def normalize_target_ioi_dev(score_ioi_ms, perf_ioi_ms, config):
    return normalize_train_ioi_dev(
        score_ioi_ms,
        perf_ioi_ms,
        epr_timing_target=config.get("epr_timing_target", "floor_log_deviation"),
        log_scale=float(config.get("timing_log_scale", 50.0)),
    )


def normalize_target_duration_dev(score_duration_ms, perf_duration_ms, config):
    return normalize_train_duration_dev(
        score_duration_ms,
        perf_duration_ms,
        epr_timing_target=config.get("epr_timing_target", "floor_log_deviation"),
        log_scale=float(config.get("timing_log_scale", 50.0)),
    )


def refined_root_from_config(config):
    refined_dir = (ROOT_DIR / config["refined_dir"]).resolve()
    return refined_dir


def relative_refined_path(path, refined_root):
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(refined_root).as_posix()
    except ValueError:
        return resolved.as_posix()


def score_source_to_work_path_from_manifest(manifest, config):
    return {
        item["score_source"]: str((ROOT_DIR / config["refined_dir"] / Path(item["score_source"]).with_suffix(".json")).resolve())
        for item in manifest["items"]
    }


def selected_gt_sources_from_manifest(manifest, config):
    refined_root = refined_root_from_config(config)
    selected = {}
    for item in manifest["items"]:
        source_paths = item.get("original_ground_truth_paths") or item.get("ground_truth_paths", [])
        selected[item["score_source"]] = {
            relative_refined_path(path, refined_root)
            for path in source_paths
        }
    return selected


def dev_groups_to_arrays(groups):
    return {
        group: {
            feature: np.asarray(values, dtype=np.float64)
            for feature, values in features.items()
        }
        for group, features in groups.items()
    }


def extract_gt_dev_arrays(score_source_to_work_path, selected_gt_sources, config):
    groups = empty_dev_groups()
    for score_source, work_path in score_source_to_work_path.items():
        selected_sources = selected_gt_sources.get(score_source)
        with open(work_path, "r", encoding="utf-8") as file:
            work = json.load(file)
        score_raw = work["score"]["score_raw"]
        for perf in work.get("performances", []):
            if selected_sources is not None and perf.get("performance_source") not in selected_sources:
                continue
            shared_rows = perf.get("label_shared_raw")
            if shared_rows is None:
                if "label_raw" not in perf:
                    continue
                shared_rows = [row[:3] for row in perf["label_raw"]]
            if len(shared_rows) != len(score_raw):
                continue
            normalization = eval_gt_time_normalization(config)
            timing_scale = 1.0
            if normalization == "score_onset_span":
                timing_scale = score_onset_span_scale(score_raw, shared_rows)
            elif normalization not in {"none", "off", "false", "0"}:
                raise ValueError(f"Unsupported eval_gt_time_normalization={normalization}")
            for score_row, perf_row in zip(score_raw, shared_rows):
                perf_ioi_ms = float(perf_row[0]) * timing_scale
                perf_duration_ms = float(perf_row[1]) * timing_scale
                group = dev_group_name(score_row[0])
                groups[group]["log_dev_ioi"].append(normalize_target_ioi_dev(score_row[0], perf_ioi_ms, config))
                groups[group]["log_dev_duration"].append(normalize_target_duration_dev(score_row[1], perf_duration_ms, config))
                groups[group]["raw_dev_ioi_s"].append((perf_ioi_ms - float(score_row[0])) / 1000.0)
                groups[group]["raw_dev_duration_s"].append((perf_duration_ms - float(score_row[1])) / 1000.0)
    return dev_groups_to_arrays(groups)


def extract_pred_dev_arrays(raw_output_paths, score_source_to_work_path, config):
    score_raw_cache = {}
    groups = empty_dev_groups()
    for path in raw_output_paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        score_source = payload["score_source"]
        if score_source not in score_raw_cache:
            with open(score_source_to_work_path[score_source], "r", encoding="utf-8") as file:
                work = json.load(file)
            score_raw_cache[score_source] = work["score"]["score_raw"]
        score_raw = score_raw_cache[score_source]
        target = payload.get("predicted_target7") or []
        reconstructed_raw = payload.get("reconstructed_raw7") or []
        if len(reconstructed_raw) != len(score_raw):
            raise ValueError(
                f"Prediction/score length mismatch for {path}: "
                f"reconstructed_raw7={len(reconstructed_raw)}, score_raw={len(score_raw)}"
            )
        has_target = len(target) == len(score_raw) and all(len(row) >= 2 for row in target)
        for row_idx, (score_row, raw_row) in enumerate(zip(score_raw, reconstructed_raw)):
            group = dev_group_name(score_row[0])
            if has_target:
                log_dev_ioi = float(target[row_idx][0])
                log_dev_duration = float(target[row_idx][1])
            else:
                log_dev_ioi = normalize_target_ioi_dev(score_row[0], raw_row[0], config)
                log_dev_duration = normalize_target_duration_dev(score_row[1], raw_row[1], config)
            groups[group]["log_dev_ioi"].append(log_dev_ioi)
            groups[group]["log_dev_duration"].append(log_dev_duration)
            groups[group]["raw_dev_ioi_s"].append((float(raw_row[0]) - float(score_row[0])) / 1000.0)
            groups[group]["raw_dev_duration_s"].append((float(raw_row[1]) - float(score_row[1])) / 1000.0)
    return dev_groups_to_arrays(groups)


def dev_histogram_range(*arrays):
    non_empty = [finite_values(array) for array in arrays if len(finite_values(array))]
    if not non_empty:
        return (0.0, 1.0)
    merged = np.concatenate(non_empty)
    low = float(np.percentile(merged, 0.5))
    high = float(np.percentile(merged, 99.5))
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        low = float(np.min(merged))
        high = float(np.max(merged))
    if high <= low:
        pad = max(1e-3, abs(low) * 0.05)
        low -= pad
        high += pad
    return low, high


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
    score_source_to_work_path = score_source_to_work_path_from_manifest(det_manifest, config)
    selected_gt_sources = selected_gt_sources_from_manifest(det_manifest, config)
    gt_arrays = extract_gt_dev_arrays(score_source_to_work_path, selected_gt_sources, config)
    det_arrays = extract_pred_dev_arrays(
        unique_paths(det_manifest, "raw_output_paths"),
        score_source_to_work_path,
        config,
    )
    sampling_arrays = extract_pred_dev_arrays(
        unique_paths(sampling_manifest, "raw_output_paths"),
        score_source_to_work_path,
        config,
    )

    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(18, 7), sharex=False)
    colors = {
        "gt": "#222222",
        "det": "#2f6fed",
        "sampling": "#d0522b",
    }

    bins = np.linspace(0.0, 1.0, 80)
    for row_idx, (group, group_title) in enumerate(DEV_GROUPS):
        for col_idx, (feature, title) in enumerate(DEV_FEATURES):
            axis = axes[row_idx, col_idx]
            gt = finite_values(gt_arrays[group][feature])
            det = finite_values(det_arrays[group][feature])
            sampling = finite_values(sampling_arrays[group][feature])

            low, high = dev_histogram_range(gt, sampling)
            bins = np.linspace(low, high, 80)

            for values, label, color, alpha in [
                (gt, "score-span GT", colors["gt"], 0.26),
                (sampling, "sampling", colors["sampling"], 0.32),
            ]:
                values = values[(values >= low) & (values <= high)]
                if len(values):
                    axis.hist(values, bins=bins, density=True, alpha=alpha, label=label, color=color)
            axis.set_title(f"{group_title}: {title}")
            axis.set_xlim(low, high)
            axis.set_ylabel("density")
            axis.grid(alpha=0.2)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("ASAP Test Normalized Dev Distribution: Score-Span GT vs Sampling", y=0.98)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(output_plot, dpi=180)
    plt.close(fig)

    def total_notes(arrays):
        return int(sum(len(arrays[group]["log_dev_ioi"]) for group, _ in DEV_GROUPS))

    return {
        "ground_truth_dev_notes": total_notes(gt_arrays),
        "deterministic_dev_notes": total_notes(det_arrays),
        "sampling_dev_notes": total_notes(sampling_arrays),
        "dev_distribution_groups": {
            group: {
                "ground_truth_notes": int(len(gt_arrays[group]["log_dev_ioi"])),
                "deterministic_notes": int(len(det_arrays[group]["log_dev_ioi"])),
                "sampling_notes": int(len(sampling_arrays[group]["log_dev_ioi"])),
            }
            for group, _ in DEV_GROUPS
        },
    }


def main():
    args = parse_args()
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")

    det_manifest = load_manifest(args.deterministic_manifest)
    sampling_manifest = load_manifest(args.sampling_manifest)
    config = load_manifest(args.config)
    timing_normalization = config.get("timing_input_normalization", "linear_5000")
    max_time_ms = float(config.get("max_time_ms", 10000.0))
    pedal_binary_support = str(config.get("pedal_representation", "")).lower() == "binary_4"
    pedal_binary_threshold = float(config.get("pedal_binary_threshold", 64.0))

    det_metrics = compute_manifest_metrics(
        det_manifest,
        max_gt_per_score=args.max_gt_per_score,
        num_workers=args.num_workers,
        pedal_binary_support=pedal_binary_support,
        pedal_binary_threshold=pedal_binary_threshold,
    )
    sampling_metrics = compute_manifest_metrics(
        sampling_manifest,
        max_gt_per_score=args.max_gt_per_score,
        num_workers=args.num_workers,
        pedal_binary_support=pedal_binary_support,
        pedal_binary_threshold=pedal_binary_threshold,
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
