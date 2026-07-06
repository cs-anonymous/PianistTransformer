import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


FEATURES = [
    ("ioi", 0),
    ("duration", 1),
    ("velocity", 2),
    ("pedal", slice(3, 7)),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose how AR INR predictions drift from GT over rollout windows."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ar-manifest", type=Path, required=True)
    parser.add_argument("--tf-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=30)
    return parser.parse_args()


def finite_values(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return values[np.isfinite(values)]


def finite_mean(values):
    values = finite_values(values)
    return float(np.mean(values)) if len(values) else float("nan")


def finite_std(values):
    values = finite_values(values)
    return float(np.std(values)) if len(values) else float("nan")


def finite_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    if float(np.std(x[mask])) <= 1e-12 or float(np.std(y[mask])) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def finite_wass(a, b):
    a = finite_values(a)
    b = finite_values(b)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    return float(wasserstein_distance(a, b))


def feature_slice(rows, feature):
    _, idx = feature
    return np.asarray(rows[:, idx], dtype=np.float64).reshape(-1)


def feature_metrics(pred_rows, gt_rows_list, feature, prefix=""):
    pred = feature_slice(pred_rows, feature)
    gt_stack = [feature_slice(rows, feature) for rows in gt_rows_list if len(rows) == len(pred_rows)]
    gt_pool = np.concatenate(gt_stack) if gt_stack else np.asarray([], dtype=np.float64)
    gt_mean_by_note = np.mean(np.stack(gt_stack, axis=0), axis=0) if gt_stack else np.asarray([])

    out = {
        f"{prefix}pp_wass": finite_wass(pred, gt_pool),
        f"{prefix}mean_pred": finite_mean(pred),
        f"{prefix}mean_gt": finite_mean(gt_pool),
        f"{prefix}bias": finite_mean(pred) - finite_mean(gt_pool),
        f"{prefix}std_pred": finite_std(pred),
        f"{prefix}std_gt": finite_std(gt_pool),
    }
    if len(gt_mean_by_note) == len(pred):
        out[f"{prefix}mae_to_gt_mean"] = finite_mean(np.abs(pred - gt_mean_by_note))
        per_note_wass = []
        gt_by_perf = np.stack(gt_stack, axis=0)
        for note_idx in range(gt_by_perf.shape[1]):
            per_note_wass.append(finite_wass([pred[note_idx]], gt_by_perf[:, note_idx]))
        out[f"{prefix}pn_wass"] = finite_mean(per_note_wass)
    else:
        out[f"{prefix}mae_to_gt_mean"] = float("nan")
        out[f"{prefix}pn_wass"] = float("nan")
    return out


def pair_metrics(pred_rows_list, target_rows_list, feature):
    wass_values = []
    mae_values = []
    bias_values = []
    for pred_rows, target_rows in zip(pred_rows_list, target_rows_list):
        if len(pred_rows) == 0 or len(pred_rows) != len(target_rows):
            continue
        pred = feature_slice(pred_rows, feature)
        target = feature_slice(target_rows, feature)
        wass_values.append(finite_wass(pred, target))
        mae_values.append(finite_mean(np.abs(pred - target)))
        bias_values.append(finite_mean(pred - target))
    return {
        "tf_pair_wass": finite_mean(wass_values),
        "tf_pair_mae": finite_mean(mae_values),
        "tf_pair_bias": finite_mean(bias_values),
    }


def build_windows(total_notes, block_notes, overlap_ratio):
    if total_notes <= block_notes:
        return [(0, total_notes)]
    stride = max(1, int(round(block_notes * (1.0 - overlap_ratio))))
    windows = []
    start = 0
    while start + block_notes <= total_notes:
        windows.append((start, start + block_notes))
        start += stride
    if windows[-1][1] != total_notes:
        windows.append((max(0, total_notes - block_notes), total_notes))
    output = []
    seen = set()
    for window in windows:
        if window in seen:
            continue
        output.append(window)
        seen.add(window)
    return output


def rollout_segments(windows):
    segments = []
    for idx, (start, end) in enumerate(windows):
        if idx == 0:
            seg_start = start
        else:
            seg_start = windows[idx - 1][1]
        if end > seg_start:
            segments.append((idx, start, end, seg_start, end))
    return segments


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def index_tf_outputs(tf_manifest):
    targets = defaultdict(list)
    tf_preds = defaultdict(list)
    losses = defaultdict(list)
    perf_sources = defaultdict(list)
    for item in tf_manifest.get("items", []):
        for raw_path in item.get("raw_output_paths", []):
            payload = read_json(raw_path)
            score_source = payload["score_source"]
            targets[score_source].append(np.asarray(payload["target_raw7"], dtype=np.float64))
            tf_preds[score_source].append(np.asarray(payload["reconstructed_raw7"], dtype=np.float64))
            losses[score_source].append(float(payload.get("loss", float("nan"))))
            perf_sources[score_source].append(payload.get("performance_source"))
    return targets, tf_preds, losses, perf_sources


def aggregate_rows(rows, group_key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[group_key]].append(row)
    output = []
    metric_keys = [
        key
        for key, value in rows[0].items()
        if isinstance(value, (int, float)) and key not in {"window_idx", "start", "end", "segment_start", "segment_end"}
    ] if rows else []
    for key in sorted(grouped):
        group_rows = grouped[key]
        out = {group_key: key, "n": len(group_rows)}
        for metric in metric_keys:
            out[metric] = finite_mean([row.get(metric, float("nan")) for row in group_rows])
        output.append(out)
    return output


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_plot(output_dir, step_rows):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"plot_error": str(exc)}

    by_feature = defaultdict(list)
    for row in step_rows:
        by_feature[row["feature"]].append(row)

    plot_paths = {}
    for feature, rows in by_feature.items():
        rows = sorted(rows, key=lambda item: item["window_idx"])
        x = [row["window_idx"] for row in rows]
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.plot(x, [row["segment_pp_wass"] for row in rows], label="AR current PP")
        ax.plot(x, [row["segment_pn_wass"] for row in rows], label="AR current PN")
        ax.plot(x, [row["prefix_pn_wass"] for row in rows], label="AR prefix PN")
        ax.plot(x, [row["tf_pair_wass"] for row in rows], label="TF pair")
        ax.set_title(f"{feature} drift by rollout step")
        ax.set_xlabel("rollout window index")
        ax.set_ylabel("raw-domain error")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path = output_dir / f"{feature}_drift.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        plot_paths[feature] = str(path.resolve())
    return plot_paths


def main():
    args = parse_args()
    config = read_json(args.config)
    ar_manifest = read_json(args.ar_manifest)
    tf_manifest = read_json(args.tf_manifest)
    block_notes = int(config["block_notes"])
    overlap_ratio = float(config["overlap_ratio"])
    targets_by_score, tf_preds_by_score, losses_by_score, perf_sources_by_score = index_tf_outputs(tf_manifest)

    rows = []
    score_rows = []
    missing_scores = []
    for item in ar_manifest.get("items", []):
        if not item.get("raw_output_paths"):
            continue
        raw_payload = read_json(item["raw_output_paths"][0])
        score_source = raw_payload["score_source"]
        ar_rows = np.asarray(raw_payload["reconstructed_raw7"], dtype=np.float64)
        target_rows_list = targets_by_score.get(score_source, [])
        tf_pred_rows_list = tf_preds_by_score.get(score_source, [])
        if not target_rows_list:
            missing_scores.append(score_source)
            continue
        usable_len = min([len(ar_rows), *[len(rows_) for rows_ in target_rows_list]])
        ar_rows = ar_rows[:usable_len]
        target_rows_list = [rows_[:usable_len] for rows_ in target_rows_list]
        tf_pred_rows_list = [rows_[:usable_len] for rows_ in tf_pred_rows_list]
        windows = build_windows(usable_len, block_notes, overlap_ratio)
        segments = rollout_segments(windows)
        score_feature_summary = {
            "score_source": score_source,
            "note_count": usable_len,
            "num_refs": len(target_rows_list),
            "num_windows": len(windows),
            "mean_tf_loss": finite_mean(losses_by_score.get(score_source, [])),
        }
        for feature in FEATURES:
            name, _ = feature
            full_metrics = feature_metrics(ar_rows, target_rows_list, feature, prefix="")
            score_feature_summary[f"{name}_full_pp_wass"] = full_metrics["pp_wass"]
            score_feature_summary[f"{name}_full_pn_wass"] = full_metrics["pn_wass"]
            score_feature_summary[f"{name}_full_bias"] = full_metrics["bias"]
        score_rows.append(score_feature_summary)

        for window_idx, window_start, window_end, segment_start, segment_end in segments:
            ar_segment = ar_rows[segment_start:segment_end]
            gt_segment_list = [rows_[segment_start:segment_end] for rows_ in target_rows_list]
            tf_segment_list = [rows_[segment_start:segment_end] for rows_ in tf_pred_rows_list]
            prefix_len = segment_start
            for feature in FEATURES:
                name, _ = feature
                segment_metrics = feature_metrics(ar_segment, gt_segment_list, feature, prefix="segment_")
                tf_metrics = pair_metrics(tf_segment_list, gt_segment_list, feature)
                row = {
                    "score_source": score_source,
                    "feature": name,
                    "window_idx": window_idx,
                    "window_start": window_start,
                    "window_end": window_end,
                    "segment_start": segment_start,
                    "segment_end": segment_end,
                    "segment_notes": segment_end - segment_start,
                    "prefix_notes": prefix_len,
                    **segment_metrics,
                    **tf_metrics,
                }
                if prefix_len > 0:
                    prefix_metrics = feature_metrics(
                        ar_rows[:prefix_len],
                        [rows_[:prefix_len] for rows_ in target_rows_list],
                        feature,
                        prefix="prefix_",
                    )
                    row.update(prefix_metrics)
                else:
                    for key in [
                        "prefix_pp_wass",
                        "prefix_pn_wass",
                        "prefix_mae_to_gt_mean",
                        "prefix_mean_pred",
                        "prefix_mean_gt",
                        "prefix_bias",
                        "prefix_std_pred",
                        "prefix_std_gt",
                    ]:
                        row[key] = 0.0 if key.endswith("_wass") or key.endswith("_mean") or key.endswith("_bias") else float("nan")
                if segment_start > window_start:
                    local_prefix_metrics = feature_metrics(
                        ar_rows[window_start:segment_start],
                        [rows_[window_start:segment_start] for rows_ in target_rows_list],
                        feature,
                        prefix="local_prefix_",
                    )
                    row.update(local_prefix_metrics)
                else:
                    for key in [
                        "local_prefix_pp_wass",
                        "local_prefix_pn_wass",
                        "local_prefix_mae_to_gt_mean",
                        "local_prefix_mean_pred",
                        "local_prefix_mean_gt",
                        "local_prefix_bias",
                        "local_prefix_std_pred",
                        "local_prefix_std_gt",
                    ]:
                        row[key] = 0.0 if key.endswith("_wass") or key.endswith("_mean") or key.endswith("_bias") else float("nan")
                row["segment_minus_tf_wass"] = row["segment_pp_wass"] - row["tf_pair_wass"]
                row["abs_segment_bias"] = abs(row["segment_bias"])
                row["abs_prefix_bias"] = abs(row["prefix_bias"])
                row["abs_local_prefix_bias"] = abs(row["local_prefix_bias"])
                rows.append(row)

    step_rows = []
    for feature in [name for name, _ in FEATURES]:
        feature_rows = [row for row in rows if row["feature"] == feature]
        by_step = defaultdict(list)
        for row in feature_rows:
            by_step[int(row["window_idx"])].append(row)
        for step in sorted(by_step):
            group = by_step[step]
            step_rows.append(
                {
                    "feature": feature,
                    "window_idx": step,
                    "n": len(group),
                    "segment_pp_wass": finite_mean([row["segment_pp_wass"] for row in group]),
                    "segment_pn_wass": finite_mean([row["segment_pn_wass"] for row in group]),
                    "prefix_pn_wass": finite_mean([row["prefix_pn_wass"] for row in group]),
                    "local_prefix_pn_wass": finite_mean([row["local_prefix_pn_wass"] for row in group]),
                    "segment_bias": finite_mean([row["segment_bias"] for row in group]),
                    "prefix_bias": finite_mean([row["prefix_bias"] for row in group]),
                    "local_prefix_bias": finite_mean([row["local_prefix_bias"] for row in group]),
                    "tf_pair_wass": finite_mean([row["tf_pair_wass"] for row in group]),
                    "segment_minus_tf_wass": finite_mean([row["segment_minus_tf_wass"] for row in group]),
                }
            )

    correlations = {}
    for feature in [name for name, _ in FEATURES]:
        feature_rows = [row for row in rows if row["feature"] == feature and row["prefix_notes"] > 0]
        correlations[feature] = {
            "prefix_pn_vs_segment_pn": finite_corr(
                [row["prefix_pn_wass"] for row in feature_rows],
                [row["segment_pn_wass"] for row in feature_rows],
            ),
            "prefix_bias_vs_segment_bias": finite_corr(
                [row["prefix_bias"] for row in feature_rows],
                [row["segment_bias"] for row in feature_rows],
            ),
            "local_prefix_pn_vs_segment_pn": finite_corr(
                [row["local_prefix_pn_wass"] for row in feature_rows],
                [row["segment_pn_wass"] for row in feature_rows],
            ),
            "local_prefix_bias_vs_segment_bias": finite_corr(
                [row["local_prefix_bias"] for row in feature_rows],
                [row["segment_bias"] for row in feature_rows],
            ),
            "abs_prefix_bias_vs_abs_segment_bias": finite_corr(
                [row["abs_prefix_bias"] for row in feature_rows],
                [row["abs_segment_bias"] for row in feature_rows],
            ),
            "abs_local_prefix_bias_vs_abs_segment_bias": finite_corr(
                [row["abs_local_prefix_bias"] for row in feature_rows],
                [row["abs_segment_bias"] for row in feature_rows],
            ),
            "window_idx_vs_segment_pn": finite_corr(
                [row["window_idx"] for row in feature_rows],
                [row["segment_pn_wass"] for row in feature_rows],
            ),
        }

    worst_rows = sorted(
        rows,
        key=lambda row: (
            -float(row["segment_pn_wass"]) if math.isfinite(float(row["segment_pn_wass"])) else 0.0,
            row["score_source"],
        ),
    )[: args.top_k]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "window_metrics.csv", rows)
    write_csv(args.output_dir / "step_summary.csv", step_rows)
    write_csv(args.output_dir / "score_summary.csv", score_rows)
    write_csv(args.output_dir / "worst_windows.csv", worst_rows)
    plot_paths = maybe_plot(args.output_dir, step_rows)

    summary = {
        "config": str(args.config.resolve()),
        "ar_manifest": str(args.ar_manifest.resolve()),
        "tf_manifest": str(args.tf_manifest.resolve()),
        "num_scores": len(score_rows),
        "num_window_feature_rows": len(rows),
        "missing_scores": missing_scores,
        "correlations": correlations,
        "aggregate_by_feature": aggregate_rows(rows, "feature"),
        "plots": plot_paths,
        "outputs": {
            "window_metrics": str((args.output_dir / "window_metrics.csv").resolve()),
            "step_summary": str((args.output_dir / "step_summary.csv").resolve()),
            "score_summary": str((args.output_dir / "score_summary.csv").resolve()),
            "worst_windows": str((args.output_dir / "worst_windows.csv").resolve()),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary["correlations"], indent=2, ensure_ascii=False))
    print(json.dumps(summary["outputs"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
