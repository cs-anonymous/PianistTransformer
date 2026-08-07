import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluate.compute_saved_midi_mae_wass import extract_note_arrays


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize velocity drift from saved INR prediction manifests.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-label", type=str, default="cinr_baseline")
    parser.add_argument("--mode", default="full_test")
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def finite_float(value):
    try:
        value = float(value)
    except Exception:
        return math.nan
    return value if math.isfinite(value) else math.nan


def read_metric(path, section, feature):
    if not path.exists():
        return math.nan
    data = json.loads(path.read_text(encoding="utf-8"))
    return finite_float(data.get("aggregate", {}).get(section, {}).get(f"{feature}_wass"))


def pooled_velocity_stats(manifest_path):
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pred_values = []
    gt_values = []
    for item in manifest.get("items", []):
        for path in item.get("prediction_paths", []):
            pred_values.extend(extract_note_arrays(Path(path))["velocity"])
        for path in item.get("ground_truth_paths", []):
            gt_values.extend(extract_note_arrays(Path(path))["velocity"])
    pred = np.asarray(pred_values, dtype=np.float64)
    gt = np.asarray(gt_values, dtype=np.float64)
    if len(pred) == 0 or len(gt) == 0:
        return {}
    return {
        "pred_velocity_count": int(len(pred)),
        "gt_velocity_count": int(len(gt)),
        "pred_velocity_mean": float(np.mean(pred)),
        "gt_velocity_mean": float(np.mean(gt)),
        "pred_velocity_std": float(np.std(pred)),
        "gt_velocity_std": float(np.std(gt)),
        "pred_velocity_p10": float(np.percentile(pred, 10)),
        "gt_velocity_p10": float(np.percentile(gt, 10)),
        "pred_velocity_p50": float(np.percentile(pred, 50)),
        "gt_velocity_p50": float(np.percentile(gt, 50)),
        "pred_velocity_p90": float(np.percentile(pred, 90)),
        "gt_velocity_p90": float(np.percentile(gt, 90)),
        "velocity_mean_delta": float(np.mean(pred) - np.mean(gt)),
        "velocity_std_delta": float(np.std(pred) - np.std(gt)),
    }


def pearson(xs, ys):
    pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if isinstance(x, (int, float))
        and isinstance(y, (int, float))
        and math.isfinite(float(x))
        and math.isfinite(float(y))
    ]
    if len(pairs) < 2:
        return math.nan
    x = np.asarray([item[0] for item in pairs], dtype=np.float64)
    y = np.asarray([item[1] for item in pairs], dtype=np.float64)
    if np.std(x) <= 0.0 or np.std(y) <= 0.0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def ranks(values):
    output = [math.nan] * len(values)
    finite = [(idx, float(value)) for idx, value in enumerate(values) if math.isfinite(finite_float(value))]
    for rank, (idx, _value) in enumerate(sorted(finite, key=lambda item: item[1]), start=1):
        output[idx] = float(rank)
    return output


def main():
    args = parse_args()
    rows = []
    for path in sorted(args.run_root.glob(f"{args.run_label}_ep*")):
        if not path.is_dir() or "_ep" not in path.name:
            continue
        try:
            epoch = int(path.name.rsplit("_ep", 1)[1])
        except ValueError:
            continue
        mode_dir = path / args.mode
        manifest_path = mode_dir / "prediction_manifest.json"
        metrics_path = mode_dir / "eval_pn_pp_metrics.json"
        if not manifest_path.exists() or not metrics_path.exists():
            continue
        stats = pooled_velocity_stats(manifest_path)
        row = {
            "epoch": epoch,
            "mode": args.mode,
            "pp_velocity_wass": read_metric(metrics_path, "pp_wass", "velocity"),
            "pn_velocity_wass": read_metric(metrics_path, "pn_wass", "velocity"),
            "pp_ioi_wass": read_metric(metrics_path, "pp_wass", "ioi"),
            "pp_duration_wass": read_metric(metrics_path, "pp_wass", "duration"),
            "pp_pedal_wass": read_metric(metrics_path, "pp_wass", "pedal"),
            "pn_ioi_wass": read_metric(metrics_path, "pn_wass", "ioi"),
            "pn_duration_wass": read_metric(metrics_path, "pn_wass", "duration"),
            "pn_pedal_wass": read_metric(metrics_path, "pn_wass", "pedal"),
        }
        row.update(stats)
        rows.append(row)
    rows.sort(key=lambda row: row["epoch"])

    output_csv = args.output_csv or (args.run_root / f"velocity_drift_{args.mode}.csv")
    output_json = args.output_json or (args.run_root / f"velocity_drift_{args.mode}.json")
    fields = [
        "epoch",
        "mode",
        "pp_velocity_wass",
        "pn_velocity_wass",
        "velocity_mean_delta",
        "velocity_std_delta",
        "pred_velocity_mean",
        "gt_velocity_mean",
        "pred_velocity_std",
        "gt_velocity_std",
        "pred_velocity_p10",
        "gt_velocity_p10",
        "pred_velocity_p50",
        "gt_velocity_p50",
        "pred_velocity_p90",
        "gt_velocity_p90",
        "pred_velocity_count",
        "gt_velocity_count",
        "pp_ioi_wass",
        "pp_duration_wass",
        "pp_pedal_wass",
        "pn_ioi_wass",
        "pn_duration_wass",
        "pn_pedal_wass",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    correlations = {}
    signals = {
        "velocity_mean_delta": [row.get("velocity_mean_delta", math.nan) for row in rows],
        "abs_velocity_mean_delta": [abs(row.get("velocity_mean_delta", math.nan)) for row in rows],
        "velocity_std_delta": [row.get("velocity_std_delta", math.nan) for row in rows],
        "abs_velocity_std_delta": [abs(row.get("velocity_std_delta", math.nan)) for row in rows],
    }
    targets = {
        "pp_velocity_wass": [row.get("pp_velocity_wass", math.nan) for row in rows],
        "pn_velocity_wass": [row.get("pn_velocity_wass", math.nan) for row in rows],
    }
    for signal_name, signal_values in signals.items():
        for target_name, target_values in targets.items():
            correlations[f"{signal_name}__vs__{target_name}_pearson"] = pearson(signal_values, target_values)
            correlations[f"{signal_name}__vs__{target_name}_spearman"] = pearson(ranks(signal_values), ranks(target_values))

    output = {"rows": rows, "correlations": correlations}
    output_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_csv": str(output_csv), "correlations": correlations}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
