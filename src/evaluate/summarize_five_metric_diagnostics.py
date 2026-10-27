#!/usr/bin/env python3
"""Align train/validation/test metrics for the baseline checkpoint sweep."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluate.plot_sampling_matrix_humanrel import HUMAN_PN, HUMAN_PP

HUMAN = {
    "pn": {feature: HUMAN_PN[f"{feature}_wass"] for feature in ("ioi", "duration", "velocity")},
    "pp": {feature: HUMAN_PP[f"{feature}_wass"] for feature in ("ioi", "duration", "velocity")},
}
FEATURES = ("ioi", "duration", "velocity")
METRICS = ("train_loss", "valid_loss", "valid_tf_hr", "test_tf_hr", "test_hr")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def hr3(metrics, prefix):
    values = []
    for feature in FEATURES:
        key = f"{prefix}_{feature}_w1_raw"
        if key in metrics:
            values.append(float(metrics[key]) / HUMAN[prefix][feature])
    if not values:
        return math.nan
    return float(np.mean(values))


def aggregate_hr3(score_rows, section):
    values = []
    for row in score_rows:
        section_metrics = row.get(section) or {}
        values.append(
            np.mean(
                [
                    float(section_metrics[f"{feature}_wass"])
                    / HUMAN["pn" if section == "pn_wass" else "pp"][feature]
                    for feature in FEATURES
                    if f"{feature}_wass" in section_metrics
                ]
            )
        )
    return float(np.mean(values)) if values else math.nan


def aggregate_section_hr3(aggregate, section):
    prefix = "pn" if section == "pn_wass" else "pp"
    metrics = aggregate.get(section) or {}
    values = [
        float(metrics[f"{feature}_wass"]) / HUMAN[prefix][feature]
        for feature in FEATURES
        if f"{feature}_wass" in metrics
    ]
    return float(np.mean(values)) if values else math.nan


def trainer_rows(checkpoint_dir):
    state = load_json(Path(checkpoint_dir) / "trainer_state.json")
    train = [row for row in state.get("log_history", []) if "loss" in row and "eval_loss" not in row]
    evals = [row for row in state.get("log_history", []) if "eval_loss" in row]
    return state, train, evals


def latest_train_loss(train, epoch):
    candidates = [row for row in train if float(row.get("epoch", -1.0)) <= float(epoch) + 1e-8]
    if not candidates:
        return math.nan, None
    row = candidates[-1]
    return float(row["loss"]), int(row.get("step", 0))


def correlation(x, y):
    pairs = [(float(a), float(b)) for a, b in zip(x, y) if math.isfinite(float(a)) and math.isfinite(float(b))]
    if len(pairs) < 3:
        return {"n": len(pairs), "pearson": math.nan, "spearman": math.nan, "kendall": math.nan}
    xs, ys = zip(*pairs)
    return {
        "n": len(pairs),
        "pearson": float(pearsonr(xs, ys).statistic),
        "spearman": float(spearmanr(xs, ys).statistic),
        "kendall": float(kendalltau(xs, ys).statistic),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--tf-root", type=Path, default=None)
    parser.add_argument("--epochs", default="1,2,3,4,5,6,7,8,9,10,11,12")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    tf_root = args.tf_root or (args.run_root / "five_metric_tf")
    epochs = [int(item) for item in args.epochs.split(",") if item.strip()]

    rows = []
    for epoch in epochs:
        ep_root = args.run_root / f"ep{epoch}"
        state, train_logs, eval_logs = trainer_rows(
            args.run_root / "training" / "baseline0807" / f"epoch-{epoch}"
        )
        train_loss, train_step = latest_train_loss(train_logs, epoch)
        valid_tf = load_json(tf_root / f"ep{epoch}" / "valid_tf_full.json")["m"]["68"]["metrics"]["0"]
        test_tf_payload = load_json(tf_root / f"ep{epoch}" / "test_tf.json")
        test_tf = test_tf_payload["m"]["12"]["metrics"]["0"]
        test_payload = load_json(ep_root / "test" / "eval_pn_pp_metrics.json")
        test_aggregate = test_payload["aggregate"]
        test_hr_all = float(
            np.mean(
                [
                    aggregate_section_hr3(test_aggregate, "pn_wass"),
                    aggregate_section_hr3(test_aggregate, "pp_wass"),
                ]
            )
        )

        selected_sources = {
            item["score_source"] for item in test_tf_payload["m"]["12"]["selected_windows"]
        }
        selected_scores = [
            item for item in test_payload.get("scores", []) if item.get("score_source") in selected_sources
        ]
        test_hr_same12 = float(
            np.mean([aggregate_hr3(selected_scores, "pn_wass"), aggregate_hr3(selected_scores, "pp_wass")])
        )
        rows.append(
            {
                "epoch": epoch,
                "checkpoint_epoch": float(state.get("epoch", math.nan)),
                "checkpoint_global_step": int(state.get("global_step", 0)),
                "train_loss": train_loss,
                "train_loss_step": train_step,
                # Keep the logged Trainer loss as the primary valid loss. The
                # fresh grouped evaluator loss is intentionally reported
                # separately because its dataset grouping differs.
                "valid_loss": float(eval_logs[-1]["eval_loss"]) if eval_logs else math.nan,
                "valid_loss_recomputed_grouped": float(valid_tf["tf_loss"]),
                "valid_loss_trainer_latest": float(eval_logs[-1]["eval_loss"]) if eval_logs else math.nan,
                "valid_loss_trainer_step": int(eval_logs[-1].get("step", 0)) if eval_logs else None,
                "valid_tf_hr": float(np.mean([hr3(valid_tf, "pn"), hr3(valid_tf, "pp")])),
                "test_tf_hr": float(np.mean([hr3(test_tf, "pn"), hr3(test_tf, "pp")])),
                "test_hr": test_hr_all,
                "test_hr_same12": test_hr_same12,
                "test_tf_num_works": len(selected_sources),
                "test_hr_same12_num_scores": len(selected_scores),
            }
        )

    correlations = {}
    for left in METRICS:
        for right in METRICS:
            if left >= right:
                continue
            correlations[f"{left}__{right}"] = correlation(
                [row[left] for row in rows], [row[right] for row in rows]
            )

    sensitivity = {}
    for excluded in ([], [5], [10], [5, 10]):
        subset = [row for row in rows if row["epoch"] not in excluded]
        tag = "all" if not excluded else "exclude_" + "_".join(map(str, excluded))
        sensitivity[tag] = {
            "excluded_epochs": excluded,
            "n": len(subset),
            "valid_tf_hr__test_hr": correlation(
                [row["valid_tf_hr"] for row in subset], [row["test_hr"] for row in subset]
            ),
            "valid_loss__valid_tf_hr": correlation(
                [row["valid_loss"] for row in subset], [row["valid_tf_hr"] for row in subset]
            ),
        }
    exact_rows = [row for row in rows if row["epoch"] <= 11]
    sensitivity["exact_epochs_1_11"] = {
        "excluded_epochs": [12],
        "n": len(exact_rows),
        "reason": "epoch 12 has no new Trainer eval event; latest logged eval is step 770",
        "valid_loss__valid_tf_hr": correlation(
            [row["valid_loss"] for row in exact_rows], [row["valid_tf_hr"] for row in exact_rows]
        ),
        "valid_tf_hr__test_hr": correlation(
            [row["valid_tf_hr"] for row in exact_rows], [row["test_hr"] for row in exact_rows]
        ),
        "test_tf_hr__test_hr": correlation(
            [row["test_tf_hr"] for row in exact_rows], [row["test_hr"] for row in exact_rows]
        ),
    }

    output = {
        "run_root": str(args.run_root),
        "metric_definition": {
            "train_loss": "latest logged HF train loss at or before checkpoint epoch",
            "valid_loss": "Trainer eval_loss on valid; epoch 12 remains the latest logged step-770 eval",
            "valid_loss_recomputed_grouped": "fresh k=0 loss from the grouped HR evaluator; diagnostic only",
            "valid_tf_hr": "analytic PN/PP CDF-W1 HR3 on all 68 fixed-valid works, k=0",
            "test_tf_hr": "analytic PN/PP CDF-W1 HR3 on fixed 12 test works with >=2 GT performances, k=0",
            "test_hr": "generated-MIDI PN/PP HR3 on the original 19-score test aggregate",
            "test_hr_same12": "generated-MIDI PN/PP HR3 restricted to the same 12 test scores as test_tf_hr",
        },
        "rows": rows,
        "correlations": correlations,
        "sensitivity": sensitivity,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with args.output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": str(args.output_json), "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
