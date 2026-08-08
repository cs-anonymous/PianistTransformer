#!/usr/bin/env python3
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

from src.evaluate.plot_sampling_matrix_humanrel import HUMAN_PN, HUMAN_PP


FEATURES = ("ioi", "duration", "velocity", "pedal")


def finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def correlation(xs, ys, rank=False):
    pairs = [(finite(x), finite(y)) for x, y in zip(xs, ys)]
    pairs = [(x, y) for x, y in pairs if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return math.nan
    x = np.asarray([item[0] for item in pairs], dtype=float)
    y = np.asarray([item[1] for item in pairs], dtype=float)
    if rank:
        x = rankdata(x)
        y = rankdata(y)
    if np.std(x) == 0 or np.std(y) == 0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def rankdata(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        average_rank = (index + end - 1) / 2.0 + 1.0
        ranks[order[index:end]] = average_rank
        index = end
    return ranks


def hr(aggregate, prefix, human):
    values = []
    for feature in FEATURES:
        value = finite((aggregate.get(f"{prefix}_wass") or {}).get(f"{feature}_wass"))
        if math.isfinite(value):
            values.append(value / float(human[f"{feature}_wass"]))
    return float(np.mean(values)) if values else math.nan


def parse_list(text):
    return [int(item.strip()) for item in str(text).split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--epochs", required=True)
    parser.add_argument("--m-values", required=True)
    parser.add_argument("--rollout-ks", required=True)
    parser.add_argument("--validation-file-name", default="validation_window_rollout_hr.json")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    epochs = parse_list(args.epochs)
    m_values = parse_list(args.m_values)
    rollout_ks = parse_list(args.rollout_ks)
    rows = []
    for epoch in epochs:
        test_path = args.run_root / f"ep{epoch}" / "test" / "eval_pn_pp_metrics.json"
        valid_path = args.run_root / f"ep{epoch}" / args.validation_file_name
        test = json.loads(test_path.read_text(encoding="utf-8"))
        valid = json.loads(valid_path.read_text(encoding="utf-8"))
        test_aggregate = test["aggregate"]
        base = {
            "epoch": epoch,
            "true_pn_hr": hr(test_aggregate, "pn", HUMAN_PN),
            "true_pp_hr": hr(test_aggregate, "pp", HUMAN_PP),
        }
        for m in m_values:
            metrics = valid["m"][str(m)]["metrics"]
            for k in rollout_ks:
                item = metrics[str(k)]
                rows.append(
                    {
                        **base,
                        "m": m,
                        "k": k,
                        "validation_pn_hr": finite(item.get("pn_hr")),
                        "validation_pp_hr": finite(item.get("pp_hr")),
                        "validation_hr": float(
                            np.nanmean([finite(item.get("pn_hr")), finite(item.get("pp_hr"))])
                        ),
                    }
                )

    candidates = []
    for m in m_values:
        for k in rollout_ks:
            subset = [row for row in rows if row["m"] == m and row["k"] == k]
            candidates.append(
                {
                    "m": m,
                    "k": k,
                    "pn_pearson": correlation(
                        [row["validation_pn_hr"] for row in subset],
                        [row["true_pn_hr"] for row in subset],
                    ),
                    "pp_pearson": correlation(
                        [row["validation_pp_hr"] for row in subset],
                        [row["true_pp_hr"] for row in subset],
                    ),
                    "mean_pearson": correlation(
                        [row["validation_hr"] for row in subset],
                        [float(np.nanmean([row["true_pn_hr"], row["true_pp_hr"]])) for row in subset],
                    ),
                    "pn_spearman": correlation(
                        [row["validation_pn_hr"] for row in subset],
                        [row["true_pn_hr"] for row in subset],
                        rank=True,
                    ),
                    "pp_spearman": correlation(
                        [row["validation_pp_hr"] for row in subset],
                        [row["true_pp_hr"] for row in subset],
                        rank=True,
                    ),
                }
            )

    ranked = sorted(
        candidates,
        key=lambda row: (
            -finite(row["mean_pearson"]) if math.isfinite(finite(row["mean_pearson"])) else math.inf,
            -finite(row["pn_pearson"]) if math.isfinite(finite(row["pn_pearson"])) else math.inf,
        ),
    )
    output = {"rows": rows, "candidates": candidates, "recommendation": ranked[0] if ranked else None}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with args.output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=sorted(rows[0]) if rows else ["epoch"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"recommendation": output["recommendation"], "output": str(args.output_json)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
