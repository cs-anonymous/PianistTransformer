#!/usr/bin/env python3
"""Collect exact-epoch losses and no-pedal PN HR, with curves and correlations."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

VARIABLES = ("train_loss", "valid_loss", "valid_tf_hr", "test_tf_hr", "test_ar_hr")


def correlations(rows):
    pairs = list(zip(VARIABLES[:-1], VARIABLES[1:])) + [("train_loss", "test_ar_hr")]
    result = []
    for index, (left, right) in enumerate(pairs):
        paired = [(r[left], r[right]) for r in rows if r[left] is not None and r[right] is not None]
        entry = {"from": left, "to": right, "kind": "end_to_end" if index == 4 else "adjacent",
                 "n": len(paired), "pearson": None, "spearman": None}
        if len(paired) >= 3:
            x, y = np.asarray(paired).T
            if np.ptp(x) > 0 and np.ptp(y) > 0:
                entry.update(pearson=float(pearsonr(x, y).statistic),
                             spearman=float(spearmanr(x, y).statistic))
        result.append(entry)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    run = args.run_root.resolve()
    training = run / "training" / "cinr_slot_sum_no_pedal"
    rows = []
    for epoch in range(1, args.epochs + 1):
        row = {"epoch": epoch, **{key: None for key in VARIABLES},
               "train_examples": None, "valid_score_count": None, "test_score_count": None}
        state_path = training / f"epoch-{epoch}" / "trainer_state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if abs(float(state["epoch"])-epoch) > 1e-5:
                raise ValueError(f"Wrong epoch in {state_path}")
            for log in state["log_history"]:
                if abs(float(log.get("epoch", -1))-epoch) > 1e-5:
                    continue
                if "epoch_train_loss" in log:
                    row["train_loss"] = float(log["epoch_train_loss"])
                    row["train_examples"] = int(log["epoch_train_examples"])
                if "eval_loss" in log:
                    row["valid_loss"] = float(log["eval_loss"])
        metrics_path = run / f"epoch-{epoch}" / "metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
            if metrics.get("smoke_test"):
                raise ValueError(f"Smoke results cannot enter the full summary: {metrics_path}")
            if Path(metrics["checkpoint"]).resolve() != state_path.parent.resolve():
                raise ValueError(f"Checkpoint mismatch in {metrics_path}")
            if metrics["feature_set"] != ["ioi", "duration", "velocity"]:
                raise ValueError(f"Unexpected HR feature set: {metrics_path}")
            row.update({key: float(metrics[key]) for key in VARIABLES[2:]})
            row["valid_score_count"] = metrics["valid"]["score_count"]
            row["test_score_count"] = metrics["test"]["score_count"]
            if row["test_score_count"] != 19:
                raise ValueError(f"Incomplete test set in {metrics_path}")
        for key in VARIABLES:
            if row[key] is not None and not np.isfinite(row[key]):
                raise ValueError(f"Nonfinite {key} at epoch {epoch}")
        rows.append(row)
    complete = all(all(r[k] is not None for k in VARIABLES) for r in rows)
    if args.require_complete and not complete:
        missing = [{"epoch": r["epoch"], "missing": [k for k in VARIABLES if r[k] is None]}
                   for r in rows if any(r[k] is None for k in VARIABLES)]
        raise ValueError(f"Missing metrics: {missing}")
    with (run / "five_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {"complete": complete, "expected_epochs": args.epochs, "primary_hr": "PN (IOI, duration, velocity)",
               "rows": rows, "correlations": correlations(rows)}
    (run / "five_metrics.json").write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True, constrained_layout=True)
    colors = ("#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728")
    for index, key in enumerate(VARIABLES):
        ax = axes[0 if index < 2 else 1]
        y = [r[key] if r[key] is not None else np.nan for r in rows]
        ax.plot([r["epoch"] for r in rows], y, marker="o", markersize=4,
                linewidth=1.6, color=colors[index], label=key.replace("_", " "))
    axes[0].set_ylabel("Loss (three attributes)")
    axes[1].set_ylabel("PN HR (no pedal; lower is better)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_xticks(range(1, args.epochs+1))
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    done = sum(all(r[k] is not None for k in VARIABLES) for r in rows)
    fig.suptitle(f"CINR slot-sum without pedal — {done}/{args.epochs} epochs fully evaluated")
    fig.savefig(run / "five_metrics.png", dpi=180)
    fig.savefig(run / "five_metrics.svg")
    plt.close(fig)
    print(json.dumps({"complete": complete, "evaluated_epochs": done, "correlations": payload["correlations"]}))


if __name__ == "__main__":
    main()
