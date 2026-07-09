import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_process.work_manifest import build_work_manifest
from src.evaluate.eval_inr_rollout_current import filter_manifest, load_config, labels_for_perf, selected_perfs
from src.train.train_inr import normalize_log_timing_value


FEATURES = ("ioi", "duration")
TARGET_KEYS = {"ioi": "ioi_log_dev", "duration": "duration_log_dev"}
SCORE_RAW_COLS = {"ioi": 0, "duration": 1}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose whether rollout drift is tied to bounded timing representation/support."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--finite-summary", type=Path, required=True)
    parser.add_argument("--full-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--performance-dataset", default="ASAP")
    parser.add_argument("--score-source-list", type=Path, default=None)
    parser.add_argument("--groups", default="GT,k=0,k=1,k=2,k=4,k=8,k=16,full AR")
    parser.add_argument("--low-quantile", type=float, default=0.10)
    parser.add_argument("--near-boundary", type=float, default=0.03)
    return parser.parse_args()


def read_score_source_list(path):
    if path is None:
        return None
    values = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            values.append(line)
    return values


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def finite(values):
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def quantile_row(values):
    values = finite(values)
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "p10": float(np.quantile(values, 0.10)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
    }


def write_csv(path, rows):
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def build_geometry(config, manifest):
    scale = float(config.get("timing_log_scale", 50.0))
    by_score = {}
    pooled = defaultdict(list)
    for item in manifest:
        work = json.loads(Path(item["path"]).read_text(encoding="utf-8"))
        score_raw = np.asarray([row[:3] for row in work["score"]["score_raw"]], dtype=np.float64)
        perfs = selected_perfs(work, item)
        geom = {feature: defaultdict(list) for feature in FEATURES}
        for perf in perfs:
            labels = np.asarray(labels_for_perf(config, perf, score_raw.tolist()), dtype=np.float64)
            for feature in FEATURES:
                score_norm = np.asarray(
                    [
                        normalize_log_timing_value(
                            x,
                            scale=scale,
                            max_time_ms=5000.0,
                        )
                        for x in score_raw[:, SCORE_RAW_COLS[feature]]
                    ],
                    dtype=np.float64,
                )
                lower = np.maximum(0.0, 0.5 - score_norm)
                upper = np.minimum(1.0, 1.5 - score_norm)
                target = labels[:, 0 if feature == "ioi" else 1]
                margin_low = target - lower
                margin_high = upper - target
                score_ms = score_raw[:, SCORE_RAW_COLS[feature]]
                geom[feature]["score_norm"].extend(score_norm.tolist())
                geom[feature]["score_ms"].extend(score_ms.tolist())
                geom[feature]["support_low"].extend(lower.tolist())
                geom[feature]["support_high"].extend(upper.tolist())
                geom[feature]["gt_target"].extend(target.tolist())
                geom[feature]["gt_margin_low"].extend(margin_low.tolist())
                geom[feature]["gt_margin_high"].extend(margin_high.tolist())
                for key, values in (
                    ("score_norm", score_norm),
                    ("score_ms", score_ms),
                    ("support_low", lower),
                    ("support_high", upper),
                    ("gt_target", target),
                    ("gt_margin_low", margin_low),
                    ("gt_margin_high", margin_high),
                ):
                    pooled[f"{feature}_{key}"].extend(np.asarray(values).tolist())
        by_score[item["score_source"]] = geom
    return by_score, pooled


def summary_values(summary, score_source, group, feature):
    item = next((entry for entry in summary.get("items", []) if entry.get("score_source") == score_source), None)
    if item is None:
        return np.asarray([], dtype=np.float64)
    if group == "GT":
        source_label = summary.get("rollout_ks", ["0"])[0]
        kind = "gt"
    elif group == "full AR":
        source_label = "full"
        kind = "pred"
    else:
        source_label = group.replace("k=", "")
        kind = "pred"
    values = (
        item.get("by_k", {})
        .get(source_label, {})
        .get("distributions", {})
        .get("target", {})
        .get(kind, {})
        .get(TARGET_KEYS[feature], [])
    )
    return np.asarray(values, dtype=np.float64)


def raw_summary_values(summary, score_source, group, key):
    item = next((entry for entry in summary.get("items", []) if entry.get("score_source") == score_source), None)
    if item is None:
        return np.asarray([], dtype=np.float64)
    if group == "GT":
        source_label = summary.get("rollout_ks", ["0"])[0]
        kind = "gt"
    elif group == "full AR":
        source_label = "full"
        kind = "pred"
    else:
        source_label = group.replace("k=", "")
        kind = "pred"
    values = (
        item.get("by_k", {})
        .get(source_label, {})
        .get("distributions", {})
        .get("target", {})
        .get(kind, {})
        .get(key, [])
    )
    return np.asarray(values, dtype=np.float64)


def source_for_group(group, finite_summary, full_summary):
    return full_summary if group == "full AR" else finite_summary


def bin_name(score_ms):
    if score_ms < 50:
        return "<50ms"
    if score_ms < 100:
        return "50-100ms"
    if score_ms < 200:
        return "100-200ms"
    if score_ms < 500:
        return "200-500ms"
    return ">=500ms"


def diagnose(args, config, geometry_by_score, finite_summary, full_summary):
    groups = [item.strip() for item in args.groups.split(",") if item.strip()]
    rows = []
    bin_rows = []
    channel_rows = []
    low_thresholds = {
        feature: float(np.quantile(finite(np.concatenate([np.asarray(g[feature]["gt_target"]) for g in geometry_by_score.values()])), args.low_quantile))
        for feature in FEATURES
    }

    for group in groups:
        summary = source_for_group(group, finite_summary, full_summary)
        for feature in FEATURES:
            pred_all = []
            gt_all = []
            lower_all = []
            upper_all = []
            score_ms_all = []
            for score_source, geom in geometry_by_score.items():
                pred = summary_values(summary, score_source, group, feature)
                gt = np.asarray(geom[feature]["gt_target"], dtype=np.float64)
                lower = np.asarray(geom[feature]["support_low"], dtype=np.float64)
                upper = np.asarray(geom[feature]["support_high"], dtype=np.float64)
                score_ms = np.asarray(geom[feature]["score_ms"], dtype=np.float64)
                n = min(len(pred), len(gt), len(lower), len(upper), len(score_ms))
                if n == 0:
                    continue
                pred_all.append(pred[:n])
                gt_all.append(gt[:n])
                lower_all.append(lower[:n])
                upper_all.append(upper[:n])
                score_ms_all.append(score_ms[:n])
            if not pred_all:
                continue
            pred = np.concatenate(pred_all)
            gt = np.concatenate(gt_all)
            lower = np.concatenate(lower_all)
            upper = np.concatenate(upper_all)
            score_ms = np.concatenate(score_ms_all)
            valid = np.isfinite(pred) & np.isfinite(gt) & np.isfinite(lower) & np.isfinite(upper)
            pred = pred[valid]
            gt = gt[valid]
            lower = lower[valid]
            upper = upper[valid]
            score_ms = score_ms[valid]
            pred_margin_low = pred - lower
            gt_margin_low = gt - lower
            pred_margin_high = upper - pred
            low_thr = low_thresholds[feature]
            low_gt_mask = gt <= low_thr
            row = {
                "group": group,
                "feature": feature,
                "n": int(pred.size),
                "target_mean_shift": float(pred.mean() - gt.mean()),
                "target_std_ratio": float(pred.std() / max(gt.std(), 1e-12)),
                "gt_low_threshold": low_thr,
                "gt_low_mass": float(low_gt_mask.mean()),
                "pred_low_mass_same_threshold": float((pred <= low_thr).mean()),
                "low_mass_ratio": float((pred <= low_thr).mean() / max(low_gt_mask.mean(), 1e-12)),
                "gt_near_lower_mass": float((gt_margin_low <= float(args.near_boundary)).mean()),
                "pred_near_lower_mass": float((pred_margin_low <= float(args.near_boundary)).mean()),
                "near_lower_mass_ratio": float(
                    (pred_margin_low <= float(args.near_boundary)).mean()
                    / max((gt_margin_low <= float(args.near_boundary)).mean(), 1e-12)
                ),
                "gt_margin_low_mean": float(gt_margin_low.mean()),
                "pred_margin_low_mean": float(pred_margin_low.mean()),
                "margin_low_mean_shift": float(pred_margin_low.mean() - gt_margin_low.mean()),
                "gt_margin_low_p10": float(np.quantile(gt_margin_low, 0.10)),
                "pred_margin_low_p10": float(np.quantile(pred_margin_low, 0.10)),
                "gt_margin_high_mean": float((upper - gt).mean()),
                "pred_margin_high_mean": float(pred_margin_high.mean()),
            }
            rows.append(row)
            for key, values in {
                "pred_target": pred,
                "gt_target": gt,
                "pred_margin_low": pred_margin_low,
                "gt_margin_low": gt_margin_low,
            }.items():
                qrow = quantile_row(values)
                qrow.update({"group": group, "feature": feature, "quantity": key})
                channel_rows.append(qrow)
            for name in sorted({bin_name(value) for value in score_ms}):
                mask = np.asarray([bin_name(value) == name for value in score_ms], dtype=bool)
                if not mask.any():
                    continue
                bin_rows.append(
                    {
                        "group": group,
                        "feature": feature,
                        "score_ms_bin": name,
                        "n": int(mask.sum()),
                        "gt_mean": float(gt[mask].mean()),
                        "pred_mean": float(pred[mask].mean()),
                        "mean_shift": float(pred[mask].mean() - gt[mask].mean()),
                        "gt_low_mass": float((gt[mask] <= low_thr).mean()),
                        "pred_low_mass": float((pred[mask] <= low_thr).mean()),
                        "gt_near_lower_mass": float((gt_margin_low[mask] <= float(args.near_boundary)).mean()),
                        "pred_near_lower_mass": float((pred_margin_low[mask] <= float(args.near_boundary)).mean()),
                    }
                )

    global_rows = []
    for group in groups:
        summary = source_for_group(group, finite_summary, full_summary)
        for key in ("velocity_norm", "pedal_0", "pedal_25", "pedal_50", "pedal_75"):
            pred_values = []
            gt_values = []
            for score_source in geometry_by_score:
                pred = raw_summary_values(summary, score_source, group, key)
                gt = raw_summary_values(summary, score_source, "GT", key)
                n = min(len(pred), len(gt))
                if n:
                    pred_values.append(pred[:n])
                    gt_values.append(gt[:n])
            if pred_values:
                pred = finite(np.concatenate(pred_values))
                gt = finite(np.concatenate(gt_values))
                n = min(len(pred), len(gt))
                if n:
                    global_rows.append(
                        {
                            "group": group,
                            "feature": key,
                            "n": int(n),
                            "gt_mean": float(gt[:n].mean()),
                            "pred_mean": float(pred[:n].mean()),
                            "mean_shift": float(pred[:n].mean() - gt[:n].mean()),
                            "gt_std": float(gt[:n].std()),
                            "pred_std": float(pred[:n].std()),
                            "std_ratio": float(pred[:n].std() / max(gt[:n].std(), 1e-12)),
                        }
                    )

    return rows, bin_rows, channel_rows, global_rows


def plot_margin(args, rows, channel_rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [item.strip() for item in args.groups.split(",") if item.strip()]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, feature in zip(axes, FEATURES):
        sub = [row for row in rows if row["feature"] == feature]
        x = np.arange(len(groups))
        values = [next((row["pred_near_lower_mass"] for row in sub if row["group"] == group), np.nan) for group in groups]
        gt_values = [next((row["gt_near_lower_mass"] for row in sub if row["group"] == group), np.nan) for group in groups]
        axis.plot(x, gt_values, marker="o", label="GT near lower")
        axis.plot(x, values, marker="o", label="Pred near lower")
        axis.set_title(feature)
        axis.set_xticks(x)
        axis.set_xticklabels(groups, rotation=35, ha="right")
        axis.set_ylim(bottom=0.0)
        axis.grid(alpha=0.18)
    axes[0].legend(frameon=False)
    fig.suptitle(f"Mass within {args.near_boundary:g} of timing lower support")
    fig.tight_layout()
    fig.savefig(args.output_dir / "near_lower_mass.png", dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config, args.checkpoint)
    manifest = build_work_manifest(
        metadata_path=config["metadata_path"],
        refined_dir=config["refined_dir"],
        split=args.split,
        block_notes=config["block_notes"],
        overlap_ratio=config["overlap_ratio"],
        min_notes=config["min_notes"],
        skip_work_paths=config.get("skip_work_paths"),
        performance_dataset=args.performance_dataset,
    )
    manifest = filter_manifest(manifest, read_score_source_list(args.score_source_list))
    geometry_by_score, pooled_geometry = build_geometry(config, manifest)
    finite_summary = load_json(args.finite_summary)
    full_summary = load_json(args.full_summary)
    rows, bin_rows, channel_rows, global_rows = diagnose(args, config, geometry_by_score, finite_summary, full_summary)
    write_csv(args.output_dir / "timing_support_drift.csv", rows)
    write_csv(args.output_dir / "timing_support_drift_by_score_ms_bin.csv", bin_rows)
    write_csv(args.output_dir / "timing_support_quantiles.csv", channel_rows)
    write_csv(args.output_dir / "other_channel_drift.csv", global_rows)
    plot_margin(args, rows, channel_rows)
    geometry_summary = {key: quantile_row(values) for key, values in pooled_geometry.items()}
    summary = {
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint),
        "finite_summary": str(args.finite_summary.resolve()),
        "full_summary": str(args.full_summary.resolve()),
        "groups": [item.strip() for item in args.groups.split(",") if item.strip()],
        "num_scores": len(manifest),
        "low_quantile": args.low_quantile,
        "near_boundary": args.near_boundary,
        "geometry": geometry_summary,
        "timing_support_drift": rows,
        "other_channel_drift": global_rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "num_scores": len(manifest)}, indent=2))


if __name__ == "__main__":
    main()
