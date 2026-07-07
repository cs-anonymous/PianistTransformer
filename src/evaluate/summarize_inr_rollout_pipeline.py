import argparse
import csv
import json
import math
import sys
from multiprocessing import get_context
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluate.evaluate_inr_saved_midis import evaluate_manifest, load_manifest_and_config


TARGET_FEATURES = [
    ("ioi_log_dev", "IOI log-dev", 0),
    ("duration_log_dev", "Duration log-dev", 1),
    ("velocity_norm", "Velocity norm", 2),
    ("pedal_0", "Pedal 0%", 3),
    ("pedal_25", "Pedal 25%", 4),
    ("pedal_50", "Pedal 50%", 5),
    ("pedal_75", "Pedal 75%", 6),
]

RAW_FEATURES = [
    ("ioi_ms", "IOI ms", 0),
    ("duration_ms", "Duration ms", 1),
    ("velocity", "Velocity", 2),
    ("pedal_0", "Pedal 0%", 3),
    ("pedal_25", "Pedal 25%", 4),
    ("pedal_50", "Pedal 50%", 5),
    ("pedal_75", "Pedal 75%", 6),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize INR k-pass eval plus real AR raw/MIDI inference.")
    parser.add_argument("--kpass-summary", type=Path, required=True)
    parser.add_argument("--ar-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-source-list", type=Path, default=None)
    parser.add_argument("--max-gt-per-score", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--plot-kind", choices=["density", "hist"], default="density")
    return parser.parse_args()


def finite_values(values):
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def finite_mean(values):
    values = finite_values(values)
    return float(values.mean()) if len(values) else float("nan")


def feature_wasserstein(pred_values, gt_values):
    pred_values = finite_values(pred_values)
    gt_values = finite_values(gt_values)
    if len(pred_values) == 0 or len(gt_values) == 0:
        return float("nan")
    return float(wasserstein_distance(pred_values, gt_values))


def sanitize(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    return value


def load_score_source_filter(path):
    if path is None:
        return None
    selected = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            selected.append(line)
    return selected or None


def load_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def k_label_sort_key(label):
    if label == "GT":
        return (-1, -1)
    if label in {"AR", "k=full", "full AR"}:
        return (2, 10**9)
    if str(label).startswith("k="):
        return (1, int(str(label).split("=", 1)[1]))
    return (1, 10**8)


def collect_kpass_groups(kpass_summary, domain, feature):
    rollout_labels = list(kpass_summary.get("rollout_ks", []))
    groups = {}
    if rollout_labels:
        gt_values = []
        first_label = rollout_labels[0]
        for item in kpass_summary.get("items", []):
            values = (
                item.get("by_k", {})
                .get(first_label, {})
                .get("distributions", {})
                .get(domain, {})
                .get("gt", {})
                .get(feature, [])
            )
            gt_values.extend(values)
        groups["GT"] = gt_values
    for rollout_label in rollout_labels:
        pred_values = []
        for item in kpass_summary.get("items", []):
            values = (
                item.get("by_k", {})
                .get(rollout_label, {})
                .get("distributions", {})
                .get(domain, {})
                .get("pred", {})
                .get(feature, [])
            )
            pred_values.extend(values)
        groups[f"k={rollout_label}"] = pred_values
    return groups


def target_rows_from_raw(path):
    payload = load_json(path)
    rows = payload.get("predicted_target7")
    if rows is None:
        rows = payload.get("predicted_target")
    if rows is None:
        raise KeyError(f"No predicted_target7/predicted_target in {path}")
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] < 7:
        raise ValueError(f"Unexpected target shape in {path}: {rows.shape}")
    return rows[:, :7]


def raw_rows_from_raw(path):
    payload = load_json(path)
    rows = payload.get("reconstructed_raw7")
    if rows is None:
        return None
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] < 7:
        raise ValueError(f"Unexpected raw shape in {path}: {rows.shape}")
    return rows[:, :7]


def collect_ar_groups(ar_manifest, domain, feature, feature_col):
    rows = []
    for item in ar_manifest.get("items", []):
        for raw_path in item.get("raw_output_paths", []):
            if domain == "target":
                values = target_rows_from_raw(raw_path)[:, feature_col]
            else:
                raw_rows = raw_rows_from_raw(raw_path)
                if raw_rows is None:
                    continue
                values = raw_rows[:, feature_col]
            rows.extend(np.asarray(values, dtype=np.float64).tolist())
    return rows


def clipped_range(groups, low=0.5, high=99.5):
    pooled_parts = [finite_values(values) for values in groups.values() if len(values)]
    if not pooled_parts:
        return 0.0, 1.0
    pooled = np.concatenate(pooled_parts)
    if len(pooled) == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(pooled, [low, high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(pooled))
        hi = float(np.max(pooled))
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def write_distribution_stats(path, all_groups_by_domain):
    rows = []
    for domain, feature_groups in all_groups_by_domain.items():
        for feature, groups in feature_groups.items():
            gt = finite_values(groups.get("GT", []))
            for group, values in sorted(groups.items(), key=lambda item: k_label_sort_key(item[0])):
                values = finite_values(values)
                row = {"domain": domain, "feature": feature, "group": group, "n": int(len(values))}
                if len(values):
                    row.update(
                        {
                            "mean": float(np.mean(values)),
                            "std": float(np.std(values)),
                            "p01": float(np.percentile(values, 1)),
                            "p50": float(np.percentile(values, 50)),
                            "p99": float(np.percentile(values, 99)),
                            "wass_to_gt": feature_wasserstein(values, gt) if group != "GT" else 0.0,
                        }
                    )
                rows.append(row)
    fieldnames = ["domain", "feature", "group", "n", "mean", "std", "p01", "p50", "p99", "wass_to_gt"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_feature_panel(path, groups_by_feature, feature_specs, title, plot_kind):
    colors = {
        "GT": "#111111",
        "k=0": "#2f6fbb",
        "k=1": "#d9822b",
        "k=2": "#6f42c1",
        "k=4": "#2b8a3e",
        "k=8": "#0b7285",
        "AR": "#b83232",
    }
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.5))
    axes = axes.flatten()
    for axis, (feature, feature_title, _) in zip(axes, feature_specs):
        groups = {
            group: finite_values(values)
            for group, values in groups_by_feature.get(feature, {}).items()
            if len(finite_values(values))
        }
        if not groups:
            axis.axis("off")
            continue
        lo, hi = clipped_range(groups)
        bins = np.linspace(lo, hi, 90)
        for group, values in sorted(groups.items(), key=lambda item: k_label_sort_key(item[0])):
            values = values[(values >= lo) & (values <= hi)]
            if len(values) == 0:
                continue
            if plot_kind == "hist":
                axis.hist(
                    values,
                    bins=bins,
                    density=True,
                    alpha=0.18 if group != "GT" else 0.11,
                    label=group,
                    color=colors.get(group),
                    edgecolor="none",
                )
            else:
                hist, edges = np.histogram(values, bins=bins, density=True)
                centers = 0.5 * (edges[:-1] + edges[1:])
                axis.plot(centers, hist, label=group, linewidth=2.0, color=colors.get(group))
        axis.set_title(feature_title)
        axis.set_ylabel("Density")
        axis.grid(alpha=0.18)
    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 7), frameon=False)
    fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def aggregate_feature_wass(all_groups_by_domain):
    output = {}
    for domain, feature_groups in all_groups_by_domain.items():
        output[domain] = {}
        for feature, groups in feature_groups.items():
            gt = groups.get("GT", [])
            output[domain][feature] = {
                group: feature_wasserstein(values, gt)
                for group, values in groups.items()
                if group != "GT"
            }
    return output


def evaluate_ar_manifest(ar_manifest_path, score_source_list, max_gt_per_score, num_workers):
    manifest, config = load_manifest_and_config(ar_manifest_path, score_source_list=score_source_list)
    pedal_binary_support = str(config.get("pedal_representation", "")).lower() == "binary_4"
    evaluation = evaluate_manifest(
        manifest,
        max_gt_per_score=max_gt_per_score,
        num_workers=num_workers,
        pedal_binary_support=pedal_binary_support,
        pedal_binary_threshold=float(config.get("pedal_binary_threshold", 64.0)),
    )
    score_rows = evaluation["score_rows"]
    from src.evaluate.evaluate_inr_saved_midis import aggregate_score_metrics

    return manifest, {
        "prediction_manifest": str(ar_manifest_path.resolve()),
        "protocol": manifest.get("protocol"),
        "sampling_strategy": infer_manifest_sampling_strategy(manifest),
        "num_scores": len(score_rows),
        "aggregate": {
            "pn_wass": aggregate_score_metrics(score_rows, "pn_wass"),
            "pp_wass": aggregate_score_metrics(score_rows, "pp_wass"),
        },
        "scores": score_rows,
    }


def infer_manifest_sampling_strategy(manifest):
    for item in manifest.get("items", []):
        for raw_path in item.get("raw_output_paths", []):
            try:
                strategy = load_json(Path(raw_path)).get("sampling_strategy")
            except Exception:
                strategy = None
            if strategy:
                return strategy
    return None


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_source_list = load_score_source_filter(args.score_source_list)
    kpass_summary = load_json(args.kpass_summary)
    ar_manifest, ar_metrics = evaluate_ar_manifest(
        args.ar_manifest,
        score_source_list=score_source_list,
        max_gt_per_score=args.max_gt_per_score,
        num_workers=args.num_workers,
    )

    all_groups_by_domain = {"target": {}, "raw": {}}
    for domain, feature_specs in (("target", TARGET_FEATURES), ("raw", RAW_FEATURES)):
        for feature, _, col in feature_specs:
            groups = collect_kpass_groups(kpass_summary, domain, feature)
            groups["AR"] = collect_ar_groups(ar_manifest, domain, feature, col)
            all_groups_by_domain[domain][feature] = groups

    write_distribution_stats(args.output_dir / "distribution_stats.csv", all_groups_by_domain)
    plot_feature_panel(
        args.output_dir / "target_distribution_gt_kpass_ar.png",
        all_groups_by_domain["target"],
        TARGET_FEATURES,
        "Target distributions: GT vs k-pass vs real AR",
        args.plot_kind,
    )
    plot_feature_panel(
        args.output_dir / "raw_distribution_gt_kpass_ar.png",
        all_groups_by_domain["raw"],
        RAW_FEATURES,
        "Raw distributions: GT vs k-pass vs real AR",
        args.plot_kind,
    )

    summary = {
        "kpass_summary": str(args.kpass_summary.resolve()),
        "ar_manifest": str(args.ar_manifest.resolve()),
        "score_source_list": str(args.score_source_list.resolve()) if args.score_source_list else None,
        "kpass": {
            "rollout_ks": kpass_summary.get("rollout_ks"),
            "materialize_strategy": kpass_summary.get("materialize_strategy"),
            "feedback_strategy": kpass_summary.get("feedback_strategy"),
            "aggregate_by_k": kpass_summary.get("aggregate_by_k"),
        },
        "ar": ar_metrics,
        "distribution_wass_to_gt": aggregate_feature_wass(all_groups_by_domain),
        "plots": {
            "target": str((args.output_dir / "target_distribution_gt_kpass_ar.png").resolve()),
            "raw": str((args.output_dir / "raw_distribution_gt_kpass_ar.png").resolve()),
            "stats": str((args.output_dir / "distribution_stats.csv").resolve()),
        },
    }
    summary = sanitize(summary)
    (args.output_dir / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary["plots"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
