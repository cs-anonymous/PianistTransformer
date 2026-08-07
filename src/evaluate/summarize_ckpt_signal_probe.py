import argparse
import ast
import csv
import json
import math
from pathlib import Path


METRIC_KEYS = (
    ("pn_ioi", ("aggregate", "pn_wass", "ioi_wass")),
    ("pn_duration", ("aggregate", "pn_wass", "duration_wass")),
    ("pn_velocity", ("aggregate", "pn_wass", "velocity_wass")),
    ("pn_pedal", ("aggregate", "pn_wass", "pedal_wass")),
    ("pp_ioi", ("aggregate", "pp_wass", "ioi_wass")),
    ("pp_duration", ("aggregate", "pp_wass", "duration_wass")),
    ("pp_velocity", ("aggregate", "pp_wass", "velocity_wass")),
    ("pp_pedal", ("aggregate", "pp_wass", "pedal_wass")),
)


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize checkpoint-signal probe outputs.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-label", type=str, default="cinr_baseline")
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def parse_log_payload(line):
    start = line.find("{")
    end = line.rfind("}")
    if start < 0 or end <= start:
        return None
    text = line[start : end + 1]
    try:
        return json.loads(text)
    except Exception:
        try:
            return ast.literal_eval(text)
        except Exception:
            return None


def read_training_signals(train_log):
    train_rows = []
    eval_rows = []
    if not train_log.exists():
        return {}, {}
    for line in train_log.read_text(encoding="utf-8", errors="replace").splitlines():
        payload = parse_log_payload(line)
        if not isinstance(payload, dict):
            continue
        step = payload.get("step")
        epoch = payload.get("epoch")
        if step is None:
            continue
        if "loss" in payload and "train_loss_components_total" in payload:
            train_rows.append(payload)
        if any(key in payload for key in ("eval_loss", "eval_tf_loss", "eval_rollout_k1_loss")):
            row = dict(payload)
            if "eval_tf_loss" not in row and "eval_loss" in row:
                row["eval_tf_loss"] = row["eval_loss"]
            if epoch is not None:
                eval_rows.append(row)
    return train_rows, eval_rows


def nearest_by_step(rows, target_step, prefer_le=False):
    if target_step is None or not rows:
        return {}
    usable = [row for row in rows if row.get("step") is not None]
    if prefer_le:
        le_rows = [row for row in usable if int(row["step"]) <= int(target_step)]
        if le_rows:
            return max(le_rows, key=lambda row: int(row["step"]))
    if not usable:
        return {}
    return min(usable, key=lambda row: abs(int(row["step"]) - int(target_step)))


def alias_source_step(run_root, run_label, epoch):
    matches = sorted((run_root / run_label / "training").glob("*"))
    for train_dir in matches:
        meta = train_dir / f"epoch-{epoch}" / ".epoch_source"
        if not meta.exists():
            continue
        try:
            payload = json.loads(meta.read_text(encoding="utf-8"))
        except Exception:
            continue
        step = payload.get("step")
        if step is not None:
            return int(step)
    return None


def read_metrics(path):
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    output = {}
    for name, keys in METRIC_KEYS:
        cursor = data
        for key in keys:
            cursor = cursor.get(key, {}) if isinstance(cursor, dict) else {}
        output[name] = cursor if isinstance(cursor, (int, float)) else math.nan
    return output


def read_seconds(path):
    if not path.exists():
        return math.nan
    try:
        return float(path.read_text(encoding="utf-8").strip())
    except Exception:
        return math.nan


def score_from_metrics(metrics):
    values = [metrics.get(key, math.nan) for key, _ in METRIC_KEYS]
    finite = [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(value)]
    return sum(finite) / len(finite) if finite else math.nan


def ranks(values):
    finite = [(idx, float(value)) for idx, value in enumerate(values) if isinstance(value, (int, float)) and math.isfinite(value)]
    sorted_items = sorted(finite, key=lambda item: item[1])
    out = [math.nan] * len(values)
    for rank, (idx, _value) in enumerate(sorted_items, start=1):
        out[idx] = float(rank)
    return out


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
    mean_x = sum(x for x, _ in pairs) / len(pairs)
    mean_y = sum(y for _, y in pairs) / len(pairs)
    num = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x, _ in pairs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for _, y in pairs))
    if den_x <= 0.0 or den_y <= 0.0:
        return math.nan
    return num / (den_x * den_y)


def main():
    args = parse_args()
    output_csv = args.output_csv or (args.run_root / "summary_ckpt_signal.csv")
    output_json = args.output_json or (args.run_root / "summary_ckpt_signal.json")
    train_rows, eval_rows = read_training_signals(args.run_root / "logs" / "train.log")

    rows = []
    for path in sorted((args.run_root).glob(f"{args.run_label}_ep*/")):
        suffix = path.name.rsplit("_ep", 1)[-1]
        try:
            epoch = int(suffix)
        except ValueError:
            continue
        source_step = alias_source_step(args.run_root, args.run_label, epoch)
        eval_row = nearest_by_step(eval_rows, source_step, prefer_le=False)
        train_row = nearest_by_step(train_rows, source_step, prefer_le=True)
        probe_metrics = read_metrics(path / "probe" / "eval_pn_pp_metrics.json")
        full_metrics = read_metrics(path / "full_test" / "eval_pn_pp_metrics.json")
        row = {
            "epoch": epoch,
            "source_step": source_step,
            "step": eval_row.get("step"),
            "actual_eval_epoch": eval_row.get("epoch"),
            "train_loss": train_row.get("loss"),
            "train_components_total": train_row.get("train_loss_components_total"),
            "eval_tf_loss": eval_row.get("eval_tf_loss"),
            "eval_rollout_k1_loss": eval_row.get("eval_rollout_k1_loss"),
            "eval_loss_for_selection": eval_row.get("eval_loss"),
            "probe_infer_seconds": read_seconds(path / "probe" / "infer.seconds"),
            "probe_eval_seconds": read_seconds(path / "probe" / "eval.seconds"),
            "full_infer_seconds": read_seconds(path / "full_test" / "infer.seconds"),
            "full_eval_seconds": read_seconds(path / "full_test" / "eval.seconds"),
            "probe_mean_metric": score_from_metrics(probe_metrics),
            "full_mean_metric": score_from_metrics(full_metrics),
        }
        for name, _keys in METRIC_KEYS:
            row[f"probe_{name}"] = probe_metrics.get(name, math.nan)
            row[f"full_{name}"] = full_metrics.get(name, math.nan)
        rows.append(row)

    rows.sort(key=lambda item: item["epoch"])
    fields = [
        "epoch",
        "source_step",
        "step",
        "actual_eval_epoch",
        "train_loss",
        "train_components_total",
        "eval_tf_loss",
        "eval_rollout_k1_loss",
        "eval_loss_for_selection",
        "probe_mean_metric",
        "full_mean_metric",
        "probe_infer_seconds",
        "probe_eval_seconds",
        "full_infer_seconds",
        "full_eval_seconds",
    ]
    fields += [f"probe_{name}" for name, _keys in METRIC_KEYS]
    fields += [f"full_{name}" for name, _keys in METRIC_KEYS]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    correlations = {}
    targets = {
        "probe_mean_metric": [row.get("probe_mean_metric") for row in rows],
        "full_mean_metric": [row.get("full_mean_metric") for row in rows],
    }
    signals = {
        "train_loss": [row.get("train_loss") for row in rows],
        "eval_tf_loss": [row.get("eval_tf_loss") for row in rows],
        "eval_rollout_k1_loss": [row.get("eval_rollout_k1_loss") for row in rows],
        "probe_mean_metric": [row.get("probe_mean_metric") for row in rows],
    }
    for signal_name, signal_values in signals.items():
        for target_name, target_values in targets.items():
            if signal_name == target_name:
                continue
            correlations[f"{signal_name}__vs__{target_name}_pearson"] = pearson(signal_values, target_values)
            correlations[f"{signal_name}__vs__{target_name}_spearman"] = pearson(ranks(signal_values), ranks(target_values))

    output_json.write_text(
        json.dumps({"rows": rows, "correlations": correlations}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({"output_csv": str(output_csv), "correlations": correlations}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
