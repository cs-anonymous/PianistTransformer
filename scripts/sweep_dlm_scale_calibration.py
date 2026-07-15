#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


FEATURES = ("ioi", "duration", "velocity")
COVERAGES = (0.50, 0.80, 0.90, 0.95)


def sigmoid(x):
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out


def evaluate_feature(path, feature, multiplier, seed, chunk_size, subset):
    data = dict(np.load(path))
    target = data["target"].astype(np.float64)
    loc = data["raw_loc"].astype(np.float64).reshape(len(target), -1)[:, 0]
    scale = data["scale"].astype(np.float64) * float(multiplier)
    zero = data["zero_mask"].astype(bool)
    if feature == "ioi":
        lo = np.where(zero, 0.0, -1.0); hi = np.where(zero, 5.0, 1.0); bins = 256
    elif feature == "duration":
        lo = np.full_like(target, -2.0); hi = np.full_like(target, 1.0); bins = 256
    else:
        lo = np.full_like(target, -0.5); hi = np.full_like(target, 127.5); bins = 128
        loc = loc + 63.5
    keep = np.ones(len(target), dtype=bool)
    if subset in ("in-support", "ioi-zero", "ioi-nonzero"):
        keep &= (target >= lo) & (target <= hi)
    if subset == "ioi-zero":
        keep &= zero
    elif subset == "ioi-nonzero":
        keep &= ~zero
    target, loc, scale, zero, lo, hi = (
        value[keep] for value in (target, loc, scale, zero, lo, hi)
    )
    rng = np.random.default_rng(seed)
    pit_random = rng.random(len(target))
    sums = {"nll": 0.0, "crps": 0.0, "z2": 0.0, "z": 0.0, "pit": 0.0, "pit2": 0.0}
    coverage_sums = {q: 0 for q in COVERAGES}
    total = 0
    for start in range(0, len(target), chunk_size):
        end = min(start + chunk_size, len(target))
        y, l, h, mu, s = target[start:end], lo[start:end], hi[start:end], loc[start:end], scale[start:end]
        t = np.linspace(0.0, 1.0, bins + 1)[None, :]
        edges = l[:, None] + (h - l)[:, None] * t
        cdf_edges = sigmoid((edges - mu[:, None]) / np.maximum(s[:, None], 1e-12))
        probs = np.empty((len(y), bins), dtype=np.float64)
        probs[:, 0] = cdf_edges[:, 1]
        probs[:, 1:-1] = np.maximum(cdf_edges[:, 2:-1] - cdf_edges[:, 1:-2], 1e-12)
        probs[:, -1] = np.maximum(1.0 - cdf_edges[:, -2], 1e-12)
        probs /= probs.sum(axis=1, keepdims=True)
        centers = (edges[:, :-1] + edges[:, 1:]) * 0.5
        target_bin = np.floor((y - l) / np.maximum(h - l, 1e-12) * bins).astype(np.int64)
        target_bin = np.clip(target_bin, 0, bins - 1)
        row = np.arange(len(y))
        target_p = probs[row, target_bin]
        cdf = np.cumsum(probs, axis=1)
        before = cdf[row, target_bin] - target_p
        pit = before + pit_random[start:end] * target_p
        mean = np.sum(probs * centers, axis=1)
        var = np.sum(probs * (centers - mean[:, None]) ** 2, axis=1)
        z = (y - mean) / np.sqrt(np.maximum(var, 1e-12))
        obs = centers >= y[:, None]
        step = (h - l) / bins
        crps = np.sum((cdf - obs) ** 2, axis=1) * step
        sums["nll"] += float(np.sum(-np.log(np.maximum(target_p, 1e-12))))
        sums["crps"] += float(np.sum(crps)); sums["z"] += float(np.sum(z)); sums["z2"] += float(np.sum(z*z))
        sums["pit"] += float(np.sum(pit)); sums["pit2"] += float(np.sum(pit*pit))
        for q in COVERAGES:
            alpha = (1.0 - q) / 2.0
            lower = np.argmax(cdf >= alpha, axis=1); upper = np.argmax(cdf >= 1.0-alpha, axis=1)
            coverage_sums[q] += int(np.sum((target_bin >= lower) & (target_bin <= upper)))
        total += len(y)
    pit_mean = sums["pit"] / total
    return {
        "multiplier": float(multiplier), "n": total,
        "mean_nll": sums["nll"] / total, "mean_crps": sums["crps"] / total,
        "mean_z": sums["z"] / total, "rms_z": np.sqrt(sums["z2"] / total),
        "pit_mean": pit_mean, "pit_variance": sums["pit2"] / total - pit_mean ** 2,
        "coverage": {f"{q:.2f}": coverage_sums[q] / total for q in COVERAGES},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--multipliers", default="0.25,0.5,0.75,1,1.25,1.5,2,2.5,3,4")
    p.add_argument("--seed", type=int, default=20260715)
    p.add_argument("--chunk-size", type=int, default=4096)
    p.add_argument(
        "--subset", choices=("all", "in-support", "ioi-zero", "ioi-nonzero"),
        default="all",
    )
    p.add_argument("--features", default=",".join(FEATURES))
    args = p.parse_args()
    multipliers = [float(x) for x in args.multipliers.split(",")]
    result = {}
    features = tuple(x.strip() for x in args.features.split(",") if x.strip())
    for feature in features:
        if feature not in FEATURES:
            p.error(f"unknown feature: {feature}")
        if args.subset.startswith("ioi-") and feature != "ioi":
            p.error(f"subset {args.subset} is only valid for ioi")
        rows = []
        for multiplier in multipliers:
            row = evaluate_feature(
                args.input_dir / f"{feature}_values.npz", feature, multiplier,
                args.seed, args.chunk_size, args.subset,
            )
            rows.append(row)
            print(json.dumps({"feature": feature, **row}), flush=True)
        result[feature] = {
            "rows": rows,
            "best_nll_multiplier": min(rows, key=lambda x: x["mean_nll"])["multiplier"],
            "best_crps_multiplier": min(rows, key=lambda x: x["mean_crps"])["multiplier"],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
