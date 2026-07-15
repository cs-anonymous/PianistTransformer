#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


FEATURES = ("ioi", "duration", "velocity")


def finite_summary(values):
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    q = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "n": int(len(x)), "mean": float(x.mean()),
        "p05": float(q[0]), "p25": float(q[1]), "p50": float(q[2]),
        "p75": float(q[3]), "p95": float(q[4]),
    }


def analyze(path, min_perfs):
    z = np.load(path)
    frame = pd.DataFrame({
        key: z[key] for key in
        ("score_idx", "perf_idx", "note_idx", "target", "mean", "std", "scale")
    })
    frame["zero_mask"] = z["zero_mask"].astype(bool)
    # Overlapping windows repeat a note. Collapse those copies before any PN statistic.
    per_perf_note = frame.groupby(
        ["score_idx", "perf_idx", "note_idx"], as_index=False
    ).agg({"target": "mean", "mean": "mean", "std": "mean", "scale": "mean", "zero_mask": "first"})
    rows = []
    for keys, group in per_perf_note.groupby(["score_idx", "note_idx"], sort=False):
        if len(group) < min_perfs:
            continue
        gt_std = float(group.target.std(ddof=0))
        within_std = float(np.sqrt(np.mean(np.square(group["std"]))))
        pred_total_std = float(np.sqrt(max(
            np.mean(np.square(group["std"]) + np.square(group["mean"]))
            - np.square(group["mean"].mean()), 0.0
        )))
        rows.append({
            "score_idx": int(keys[0]), "note_idx": int(keys[1]), "n_perfs": len(group),
            "gt_mean": float(group.target.mean()), "gt_std": gt_std,
            "pred_mean": float(group["mean"].mean()),
            "pred_within_std": within_std, "pred_total_std": pred_total_std,
            "pred_scale": float(group.scale.mean()),
            "zero_mask": bool(group.zero_mask.iloc[0]),
        })
    out = pd.DataFrame(rows)
    positive = out.gt_std > 1e-8
    for key in ("pred_within_std", "pred_total_std", "pred_scale"):
        out[f"{key}_over_gt"] = np.where(positive, out[key] / out.gt_std, np.nan)
    mean_error = out.pred_mean - out.gt_mean
    def correlation(a, b):
        valid = np.isfinite(a) & np.isfinite(b)
        return float(spearmanr(np.asarray(a)[valid], np.asarray(b)[valid]).statistic)

    def reliability(frame):
        usable = frame[frame.gt_std > 1e-8].copy()
        if len(usable) < 10:
            return []
        usable["bin"] = pd.qcut(usable.gt_std, 10, labels=False, duplicates="drop")
        rows = []
        for idx, group in usable.groupby("bin"):
            rows.append({
                "bin": int(idx), "n": int(len(group)),
                "gt_std": float(group.gt_std.mean()),
                "pred_within_std": float(group.pred_within_std.mean()),
                "pred_total_std": float(group.pred_total_std.mean()),
                "within_over_gt": float(group.pred_within_std.mean() / group.gt_std.mean()),
            })
        return rows

    result = {
        "note_groups": int(len(out)), "min_perfs": min_perfs,
        "gt_std": finite_summary(out.gt_std),
        "pred_within_std": finite_summary(out.pred_within_std),
        "pred_total_std": finite_summary(out.pred_total_std),
        "pred_scale": finite_summary(out.pred_scale),
        "within_std_over_gt": finite_summary(out.pred_within_std_over_gt),
        "total_std_over_gt": finite_summary(out.pred_total_std_over_gt),
        "scale_over_gt": finite_summary(out.pred_scale_over_gt),
        "fraction_within_std_gt": float(np.mean(out.pred_within_std > out.gt_std)),
        "fraction_total_std_gt": float(np.mean(out.pred_total_std > out.gt_std)),
        "mean_error": finite_summary(mean_error),
        "mean_abs_error": float(np.mean(np.abs(mean_error))),
        "spearman_pred_within_std_vs_gt_std": correlation(out.pred_within_std, out.gt_std),
        "spearman_pred_scale_vs_gt_std": correlation(out.pred_scale, out.gt_std),
        "reliability_by_gt_std_decile": reliability(out),
    }
    if "ioi" in path.name:
        result["zero_ioi"] = summarize_subset(out[out.zero_mask])
        result["nonzero_ioi"] = summarize_subset(out[~out.zero_mask])
    return out, result


def summarize_subset(out):
    positive = out.gt_std > 1e-8
    ratio = out.loc[positive, "pred_within_std"] / out.loc[positive, "gt_std"]
    return {
        "note_groups": int(len(out)),
        "gt_std": finite_summary(out.gt_std),
        "pred_within_std": finite_summary(out.pred_within_std),
        "within_std_over_gt": finite_summary(ratio),
        "fraction_within_std_gt": float(np.mean(out.pred_within_std > out.gt_std)),
        "spearman_pred_within_std_vs_gt_std": float(spearmanr(out.pred_within_std, out.gt_std).statistic),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--min-perfs", type=int, default=3)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"definition": {
        "gt": "empirical distribution across reference performances at identical score note",
        "pred_within_std": "RMS discrete-head std across reference-conditioned predictions",
        "pred_total_std": "total variance of the equally weighted predicted distributions",
    }}
    for feature in FEATURES:
        frame, result = analyze(args.input_dir / f"{feature}_values.npz", args.min_perfs)
        frame.to_csv(args.output_dir / f"{feature}_per_note.csv", index=False)
        summary[feature] = result
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
