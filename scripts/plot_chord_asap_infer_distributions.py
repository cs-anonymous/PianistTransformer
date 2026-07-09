#!/usr/bin/env python
import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


DEFAULT_RUNS = (
    ("sine deterministic", "infer/sine/deterministic/prediction_manifest.json"),
    ("sine sampling", "infer/sine/sampling/prediction_manifest.json"),
    ("cine deterministic", "infer/cine/deterministic/prediction_manifest.json"),
    ("cine sampling", "infer/cine/sampling/prediction_manifest.json"),
)

COLORS = {
    "GT": "#111111",
    "sine deterministic": "#2f6fbb",
    "sine sampling": "#c05a2b",
    "cine deterministic": "#2f9d68",
    "cine sampling": "#8d5ac2",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot chord-level ASAP inference distributions.")
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("results/inr0624_chord_asap_sn_rawlog_multihot_nomus"),
    )
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/chord_asap"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--timing-log-scale", type=float, default=50.0)
    parser.add_argument("--lower-percentile", type=float, default=0.5)
    parser.add_argument("--upper-percentile", type=float, default=99.0)
    parser.add_argument(
        "--manifest",
        action="append",
        default=None,
        help="Optional label=path override. May be passed multiple times.",
    )
    return parser.parse_args()


def parse_manifest_specs(args):
    if args.manifest:
        specs = []
        for item in args.manifest:
            if "=" not in item:
                raise ValueError(f"--manifest expects label=path, got {item}")
            label, path = item.split("=", 1)
            specs.append((label.strip(), Path(path).expanduser()))
        return specs
    return [(label, args.experiment_dir / rel_path) for label, rel_path in DEFAULT_RUNS]


def log_timing_code(values, scale=50.0, max_time_ms=5000.0):
    values = np.asarray(values, dtype=np.float64)
    values = np.clip(values, 0.0, float(max_time_ms))
    return np.log1p(values / float(scale))


def empty_feature_dict():
    return {
        "ioi_ms": [],
        "duration_ms": [],
        "ioi_log": [],
        "duration_log": [],
        "ioi_dev_ms": [],
        "duration_dev_ms": [],
        "ioi_log_dev": [],
        "duration_log_dev": [],
        "velocity": [],
        "onset_offset_ms": [],
        "duration_offset_ms": [],
        "velocity_offset": [],
    }


def add_row(store, score_row, perf_row, offset_row, log_scale, include_offset=True):
    score_ioi = float(score_row[0])
    score_duration = float(score_row[1])
    perf_ioi = float(perf_row[0])
    perf_duration = float(perf_row[1])
    score_log = log_timing_code([score_ioi, score_duration], scale=log_scale)
    perf_log = log_timing_code([perf_ioi, perf_duration], scale=log_scale)
    store["ioi_ms"].append(perf_ioi)
    store["duration_ms"].append(perf_duration)
    store["ioi_log"].append(float(perf_log[0]))
    store["duration_log"].append(float(perf_log[1]))
    store["ioi_dev_ms"].append(perf_ioi - score_ioi)
    store["duration_dev_ms"].append(perf_duration - score_duration)
    store["ioi_log_dev"].append(float(perf_log[0] - score_log[0]))
    store["duration_log_dev"].append(float(perf_log[1] - score_log[1]))
    store["velocity"].append(float(perf_row[2]))
    if include_offset:
        store["onset_offset_ms"].append(float(offset_row[0]))
        store["duration_offset_ms"].append(float(offset_row[1]))
        store["velocity_offset"].append(float(offset_row[2]))


def score_json_path(processed_dir, score_source):
    return processed_dir / Path(score_source).with_suffix(".json")


def load_score_payload(processed_dir, score_source, cache):
    if score_source not in cache:
        path = score_json_path(processed_dir, score_source)
        cache[score_source] = json.loads(path.read_text(encoding="utf-8"))
    return cache[score_source]


def gt_source_matches(perf_source, gt_paths):
    return any(str(path).endswith(str(perf_source)) for path in gt_paths)


def collect_gt(manifest_items, processed_dir, score_cache, log_scale):
    store = empty_feature_dict()
    seen = set()
    for item in manifest_items:
        payload = load_score_payload(processed_dir, item["score_source"], score_cache)
        score_rows = payload["score"]["score_raw"]
        chord_sizes = [len(pitches) for pitches in payload["score"]["pitch"]]
        gt_paths = item.get("ground_truth_paths", [])
        for perf in payload.get("performances", []):
            perf_source = perf.get("performance_source")
            if not gt_source_matches(perf_source, gt_paths):
                continue
            key = (item["score_source"], perf_source)
            if key in seen:
                continue
            seen.add(key)
            for idx, (score_row, perf_row, offset_row) in enumerate(
                zip(
                    score_rows,
                    perf["label_shared_raw"],
                    perf["label_offset_raw"],
                )
            ):
                include_offset = bool(chord_sizes[idx] > 1)
                add_row(store, score_row, perf_row, offset_row, log_scale, include_offset=include_offset)
    return store


def collect_prediction(manifest, processed_dir, score_cache, log_scale):
    store = empty_feature_dict()
    for item in manifest.get("items", []):
        payload = load_score_payload(processed_dir, item["score_source"], score_cache)
        score_rows = payload["score"]["score_raw"]
        chord_sizes = [len(pitches) for pitches in payload["score"]["pitch"]]
        for raw_path in item.get("raw_output_paths", []):
            raw = json.loads(Path(raw_path).read_text(encoding="utf-8"))
            perf_rows = raw["reconstructed_raw7"]
            target_rows = (
                raw.get("predicted_target12")
                or raw.get("predicted_target10")
                or raw.get("predicted_target7")
                or raw.get("predicted_target")
            )
            if target_rows is None:
                raise KeyError(f"No predicted target rows in {raw_path}")
            for idx, (score_row, perf_row) in enumerate(zip(score_rows, perf_rows)):
                target_row = target_rows[idx]
                if len(target_row) >= 10:
                    offset_row = [
                        float(target_row[-3]) * 1000.0,
                        float(target_row[-2]) * 1000.0,
                        float(target_row[-1]) * 127.0,
                    ]
                else:
                    offset_row = [0.0, 0.0, 0.0]
                add_row(
                    store,
                    score_row,
                    perf_row,
                    offset_row,
                    log_scale,
                    include_offset=bool(chord_sizes[idx] > 1),
                )
    return store


def to_array(values):
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def feature_stats(values):
    arr = to_array(values)
    if len(arr) == 0:
        return {"n": 0}
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p01": float(np.percentile(arr, 1)),
        "p05": float(np.percentile(arr, 5)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def build_summary(all_data):
    return {
        label: {feature: feature_stats(values) for feature, values in data.items()}
        for label, data in all_data.items()
    }


def pooled_values(all_data, features):
    arrays = []
    for data in all_data.values():
        for feature in features:
            arr = to_array(data[feature])
            if len(arr):
                arrays.append(arr)
    if not arrays:
        return np.asarray([], dtype=np.float64)
    return np.concatenate(arrays)


def quantile_xlim(all_data, features, lower_percentile, upper_percentile, min_width, nonnegative=False):
    merged = pooled_values(all_data, features)
    if len(merged) == 0:
        return (0.0, min_width) if nonnegative else (-min_width * 0.5, min_width * 0.5)
    lo = float(np.percentile(merged, lower_percentile))
    hi = float(np.percentile(merged, upper_percentile))
    if nonnegative:
        lo = max(0.0, lo)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        center = 0.0 if nonnegative else float(np.median(merged))
        lo = center
        hi = center + float(min_width)
        if not nonnegative:
            lo = center - float(min_width) * 0.5
            hi = center + float(min_width) * 0.5
    if hi - lo < float(min_width):
        pad = (float(min_width) - (hi - lo)) * 0.5
        lo -= pad
        hi += pad
        if nonnegative and lo < 0.0:
            hi -= lo
            lo = 0.0
    return (lo, hi)


def symmetric_quantile_xlim(all_data, features, lower_percentile, upper_percentile, min_abs):
    merged = pooled_values(all_data, features)
    if len(merged) == 0:
        return (-min_abs, min_abs)
    lo = float(np.percentile(merged, lower_percentile))
    hi = float(np.percentile(merged, upper_percentile))
    limit = max(abs(lo), abs(hi), float(min_abs))
    return (-limit, limit)


def adaptive_bins(xlim, target_bin_width, min_bins=50, max_bins=180):
    width = max(float(xlim[1]) - float(xlim[0]), 1e-9)
    if target_bin_width <= 0:
        return min_bins
    return int(max(min_bins, min(max_bins, round(width / float(target_bin_width)))))


def density_line(ax, values, label, color, xlim, bins=140, linewidth=1.7):
    arr = to_array(values)
    arr = arr[(arr >= xlim[0]) & (arr <= xlim[1])]
    if len(arr) == 0:
        return None
    edges = np.linspace(xlim[0], xlim[1], bins + 1)
    hist, edges = np.histogram(arr, bins=edges, density=True)
    centers = (edges[:-1] + edges[1:]) * 0.5
    ax.plot(centers, hist, label=label, color=color, linewidth=linewidth, alpha=0.95)
    return hist[np.isfinite(hist)]


def set_readable_ylim(ax, hist_values):
    values = [item for item in hist_values if item is not None and len(item)]
    if not values:
        return
    merged = np.concatenate(values)
    finite = merged[np.isfinite(merged)]
    positive = finite[finite > 0.0]
    if len(positive) == 0:
        return
    cap = float(np.percentile(positive, 99.0)) * 1.15
    if np.isfinite(cap) and cap > 0.0:
        ax.set_ylim(0.0, cap)


def plot_two_feature_chart(all_data, title, features, xlabels, xlims, output_path, bin_widths, tail_note):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), constrained_layout=True)
    for ax, feature, xlabel, xlim, bin_width in zip(axes, features, xlabels, xlims, bin_widths):
        hist_values = []
        bins = adaptive_bins(xlim, bin_width)
        for label, data in all_data.items():
            hist_values.append(
                density_line(
                    ax,
                    data[feature],
                    label,
                    COLORS.get(label, None),
                    xlim,
                    bins=bins,
                    linewidth=2.4 if label == "GT" else 1.7,
                )
            )
        ax.set_title(xlabel)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("density")
        ax.set_xlim(*xlim)
        ax.grid(True, alpha=0.22)
        set_readable_ylim(ax, hist_values)
    axes[0].legend(loc="upper right", fontsize=9)
    fig.suptitle(f"{title} ({tail_note})", fontsize=15)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_single_feature_chart(all_data, title, feature, xlabel, xlim, output_path, bin_width, tail_note):
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 4.8), constrained_layout=True)
    hist_values = []
    bins = adaptive_bins(xlim, bin_width)
    for label, data in all_data.items():
        hist_values.append(
            density_line(
                ax,
                data[feature],
                label,
                COLORS.get(label, None),
                xlim,
                bins=bins,
                linewidth=2.4 if label == "GT" else 1.7,
            )
        )
    if xlim[0] < 0 < xlim[1]:
        ax.axvline(0.0, color="#777777", linewidth=1.0, alpha=0.55)
    ax.set_title(f"{title} ({tail_note})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.set_xlim(*xlim)
    ax.grid(True, alpha=0.22)
    set_readable_ylim(ax, hist_values)
    ax.legend(loc="upper right", fontsize=9)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_index(output_dir, written):
    lines = ["# Chord ASAP inference distributions", ""]
    for path in written:
        lines.append(f"- `{path.name}`")
    lines.append("")
    lines.append("All plots are chord-level distributions over the full ASAP test set; score IOI is not bucketed.")
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    processed_dir = (ROOT_DIR / args.processed_dir).resolve() if not args.processed_dir.is_absolute() else args.processed_dir
    output_dir = args.output_dir or (experiment_dir / "stats" / "distributions")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_specs = parse_manifest_specs(args)
    manifests = []
    for label, path in manifest_specs:
        resolved = path if path.is_absolute() else (ROOT_DIR / path).resolve()
        if not resolved.exists():
            if args.manifest is None:
                continue
            raise FileNotFoundError(f"Missing prediction manifest for {label}: {resolved}")
        manifests.append((label, json.loads(resolved.read_text(encoding="utf-8"))))
    if not manifests:
        raise FileNotFoundError(f"No prediction manifests found under {experiment_dir}")

    score_cache = {}
    all_data = {"GT": collect_gt(manifests[0][1].get("items", []), processed_dir, score_cache, args.timing_log_scale)}
    for label, manifest in manifests:
        all_data[label] = collect_prediction(manifest, processed_dir, score_cache, args.timing_log_scale)

    summary = {
        "experiment_dir": str(experiment_dir),
        "processed_dir": str(processed_dir),
        "timing_log_scale": float(args.timing_log_scale),
        "runs": [label for label, _ in manifests],
        "features": build_summary(all_data),
    }
    (output_dir / "distribution_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lower = float(args.lower_percentile)
    upper = float(args.upper_percentile)
    tail_note = f"pooled q{lower:g}-q{upper:g}, tails clipped"
    timing_xlim = (
        quantile_xlim(all_data, ("ioi_ms",), lower, upper, min_width=100.0, nonnegative=True),
        quantile_xlim(all_data, ("duration_ms",), lower, upper, min_width=100.0, nonnegative=True),
    )
    timing_log_xlim = (
        quantile_xlim(all_data, ("ioi_log",), lower, upper, min_width=0.5, nonnegative=True),
        quantile_xlim(all_data, ("duration_log",), lower, upper, min_width=0.5, nonnegative=True),
    )
    timing_dev_xlim = (
        symmetric_quantile_xlim(all_data, ("ioi_dev_ms",), lower, upper, min_abs=50.0),
        symmetric_quantile_xlim(all_data, ("duration_dev_ms",), lower, upper, min_abs=50.0),
    )
    timing_log_dev_xlim = (
        symmetric_quantile_xlim(all_data, ("ioi_log_dev",), lower, upper, min_abs=0.25),
        symmetric_quantile_xlim(all_data, ("duration_log_dev",), lower, upper, min_abs=0.25),
    )

    written = []
    specs = [
        (
            "timing_raw.png",
            "Chord-level timing distribution",
            ("ioi_ms", "duration_ms"),
            ("IOI (ms)", "Duration (ms)"),
            timing_xlim,
        ),
        (
            "timing_logscale.png",
            f"Chord-level timing logscale distribution, log1p(ms / {args.timing_log_scale:g})",
            ("ioi_log", "duration_log"),
            ("logscale IOI", "logscale duration"),
            timing_log_xlim,
        ),
        (
            "timing_dev.png",
            "Chord-level timing deviation from score",
            ("ioi_dev_ms", "duration_dev_ms"),
            ("IOI dev (ms)", "Duration dev (ms)"),
            timing_dev_xlim,
        ),
        (
            "timing_logscale_dev.png",
            "Chord-level timing logscale deviation from score",
            ("ioi_log_dev", "duration_log_dev"),
            ("logscale IOI dev", "logscale duration dev"),
            timing_log_dev_xlim,
        ),
    ]
    two_feature_bin_widths = {
        "timing_raw.png": (5.0, 5.0),
        "timing_logscale.png": (0.025, 0.025),
        "timing_dev.png": (5.0, 5.0),
        "timing_logscale_dev.png": (0.02, 0.02),
    }
    for filename, title, features, xlabels, xlims in specs:
        path = output_dir / filename
        plot_two_feature_chart(
            all_data,
            title,
            features,
            xlabels,
            xlims,
            path,
            bin_widths=two_feature_bin_widths[filename],
            tail_note=tail_note,
        )
        written.append(path)

    single_specs = [
        (
            "velocity.png",
            "Chord-level velocity distribution",
            "velocity",
            "Velocity",
            quantile_xlim(all_data, ("velocity",), lower, upper, min_width=20.0, nonnegative=True),
        ),
        (
            "onset_offset.png",
            "Chord onset offset distribution",
            "onset_offset_ms",
            "Onset offset low-high (ms)",
            symmetric_quantile_xlim(all_data, ("onset_offset_ms",), lower, upper, min_abs=20.0),
        ),
        (
            "duration_offset.png",
            "Chord duration offset distribution",
            "duration_offset_ms",
            "Duration offset low-high (ms)",
            symmetric_quantile_xlim(all_data, ("duration_offset_ms",), lower, upper, min_abs=50.0),
        ),
        (
            "velocity_offset.png",
            "Chord velocity offset distribution",
            "velocity_offset",
            "Velocity offset low-high",
            symmetric_quantile_xlim(all_data, ("velocity_offset",), lower, upper, min_abs=20.0),
        ),
    ]
    single_bin_widths = {
        "velocity.png": 1.0,
        "onset_offset.png": 1.0,
        "duration_offset.png": 5.0,
        "velocity_offset.png": 1.0,
    }
    for filename, title, feature, xlabel, xlim in single_specs:
        path = output_dir / filename
        plot_single_feature_chart(
            all_data,
            title,
            feature,
            xlabel,
            xlim,
            path,
            bin_width=single_bin_widths[filename],
            tail_note=tail_note,
        )
        written.append(path)

    write_index(output_dir, written)
    for path in written:
        print(path)
    print(output_dir / "distribution_summary.json")


if __name__ == "__main__":
    main()
