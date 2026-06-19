import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluate.compute_saved_midi_mae_wass import build_feature_cache, load_evaluate_list, normalize_pair_paths


FEATURES = ("ioi", "duration", "velocity", "pedal_0", "pedal_25", "pedal_50", "pedal_75")
PEDAL_FEATURES = ("pedal_0", "pedal_25", "pedal_50", "pedal_75")


def summarize(values):
    if len(values) == 0:
        return {}
    qs = np.percentile(values, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p01": float(qs[0]),
        "p05": float(qs[1]),
        "p10": float(qs[2]),
        "p25": float(qs[3]),
        "median": float(qs[4]),
        "p75": float(qs[5]),
        "p90": float(qs[6]),
        "p95": float(qs[7]),
        "p99": float(qs[8]),
        "max": float(np.max(values)),
    }


def collect_arrays(evaluate_list, side, num_workers):
    pairs = [normalize_pair_paths(item) for item in load_evaluate_list(evaluate_list)]
    if side == "both":
        selected = [{"pred": item["pred"], "gt": item["gt"]} for item in pairs]
    elif side == "pred":
        selected = [{"pred": item["pred"], "gt": item["pred"]} for item in pairs]
    elif side == "gt":
        selected = [{"pred": item["gt"], "gt": item["gt"]} for item in pairs]
    else:
        raise ValueError(side)
    cache = build_feature_cache(selected, num_workers)
    unique_paths = sorted(cache)
    arrays = {}
    for feature in FEATURES:
        arrays[feature] = np.concatenate([cache[path][feature] for path in unique_paths])
    arrays["pedal"] = np.concatenate([arrays[feature] for feature in PEDAL_FEATURES])
    return arrays, unique_paths


def plot_one(values, feature, output_dir, bins, max_points, seed):
    rng = np.random.default_rng(seed)
    values = values[np.isfinite(values)]
    if len(values) > max_points:
        values = values[rng.choice(len(values), size=max_points, replace=False)]
    if feature in {"ioi", "duration"}:
        upper = np.percentile(values, 99.5)
        upper = max(upper, 1.0)
        clipped = np.clip(values, 0.0, upper)
        xlabel = f"{feature} raw milliseconds (clipped at p99.5={upper:.1f})"
    else:
        clipped = np.clip(values, 0.0, 127.0)
        xlabel = f"{feature} raw value"
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    ax.hist(clipped, bins=bins, density=True, color="#2b6cb0", alpha=0.72)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.set_title(f"Raw {feature} distribution")
    ax.grid(True, alpha=0.18)
    fig.tight_layout()
    out = output_dir / f"{feature}_raw_hist.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot raw MIDI feature histograms from evaluate_list.")
    parser.add_argument("--evaluate-list", type=Path, required=True)
    parser.add_argument("--side", choices=["pred", "gt", "both"], default="gt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--max-points", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays, paths = collect_arrays(args.evaluate_list, args.side, args.num_workers)
    for feature, values in arrays.items():
        plot_one(values, feature, args.output_dir, args.bins, args.max_points, args.seed)
    summary = {
        "evaluate_list": str(args.evaluate_list),
        "side": args.side,
        "num_files": len(paths),
        "features": {feature: summarize(values) for feature, values in arrays.items()},
    }
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"Wrote {args.summary_json}")


if __name__ == "__main__":
    main()
