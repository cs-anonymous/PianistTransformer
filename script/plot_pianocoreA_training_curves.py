#!/usr/bin/env python3
"""Plot the live PianoCoRe-A training and validation curves."""
import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_training_log(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    records = []
    for match in re.finditer(r"\{[^{}\n]*\}", text):
        try:
            record = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if "step" not in record or "loss" not in record:
            continue
        if "train_loss_components_total" not in record:
            continue
        records.append(record)
    deduped = {}
    for record in records:
        deduped[int(record["step"])] = record
    return [deduped[step] for step in sorted(deduped)]


def validation_points(output_dir):
    points = []
    for path in sorted(Path(output_dir).glob("epoch-*/trainer_state.json")):
        state = json.loads(path.read_text())
        records = [record for record in state.get("log_history", []) if "eval_loss" in record]
        if records:
            record = records[-1]
            points.append((float(record["epoch"]), float(record["eval_loss"])))
    return sorted({epoch: value for epoch, value in points}.items())


def hr_points(run, name):
    values = []
    for epoch in range(1, 5):
        path = run / "validation_hr" / f"pianocoreA_pretrain_epoch-{epoch}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        selected = next(iter(payload["m"].values()))
        metrics = selected["metrics"]["0"]
        values.append((epoch, float(metrics[name])))
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run = args.run_dir
    output_dir = run / "stage1/training/pianocoreA_slot_sum_pretrain4_gpu01_global64"
    preflight = json.loads((run / "data_preflight.json").read_text())
    config = json.loads((run / "runtime_configs/pianocoreA_slot_sum_pretrain4.json").read_text())
    steps_per_epoch = math.ceil(preflight["pianocoreA"]["train"]["examples"] / config["global_batch_size"])
    records = parse_training_log(run / "logs/stage1_train.log")
    valid = validation_points(output_dir)
    pn_hr = hr_points(run, "pn_hr")
    pp_hr = hr_points(run, "pp_hr")
    if not records:
        raise SystemExit("No training records found")
    output = args.output or (run / "pianocoreA_training_curves.png")

    steps = np.asarray([int(record["step"]) for record in records])
    train_loss = np.asarray([float(record["train_loss_components_total"]) for record in records])
    train_loss_raw = np.asarray([float(record["loss"]) for record in records])
    window = max(1, min(80, len(train_loss) // 20))
    smooth = np.convolve(train_loss, np.ones(window) / window, mode="valid")
    smooth_steps = steps[window - 1 :]
    current_epoch = float(steps.max() / steps_per_epoch)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(steps, train_loss_raw, color="#9ca3af", linewidth=0.6, alpha=0.35, label="step loss")
    ax.plot(smooth_steps, smooth, color="#2563eb", linewidth=2.0, label=f"train loss ({window}-point mean)")
    for epoch, _ in valid:
        ax.axvline(epoch * steps_per_epoch, color="#d1d5db", linewidth=0.8, linestyle="--")
    ax.axvline(steps.max(), color="#6b7280", linewidth=1.0, linestyle=":", label=f"current step {steps.max()}")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.set_title("PianoCoRe-A training loss")
    ax.set_xlim(0, 4 * steps_per_epoch)
    ax.legend(frameon=True)

    ax = axes[1, 0]
    if valid:
        epochs, values = zip(*valid)
        ax.plot(epochs, values, marker="o", color="#dc2626", linewidth=2.0, label="valid loss")
        for epoch, value in valid:
            ax.annotate(f"{value:.3f}", (epoch, value), xytext=(5, 7), textcoords="offset points", fontsize=9)
    ax.axvline(current_epoch, color="#6b7280", linewidth=1.0, linestyle=":", label=f"current {current_epoch:.2f}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Valid loss")
    ax.set_title("Completed validation epochs")
    ax.set_xticks(range(1, 5))
    ax.legend(frameon=True)
    for axis, values, title, color in (
        (axes[0, 1], pn_hr, "Valid teacher-forced PN-HR", "#008b8b"),
        (axes[1, 1], pp_hr, "Valid teacher-forced PP-HR", "#7c3aed"),
    ):
        if values:
            x, y = zip(*values)
            axis.plot(x, y, marker="o", color=color, linewidth=2.0)
            for epoch, value in values:
                axis.annotate(f"{value:.3f}", (epoch, value), xytext=(5, 7), textcoords="offset points", fontsize=9)
        else:
            axis.text(0.5, 0.5, "Checkpoint evaluations pending", ha="center", va="center", transform=axis.transAxes)
        axis.axvline(current_epoch, color="#6b7280", linewidth=1.0, linestyle=":", label=f"current {current_epoch:.2f}")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Human relevance (lower is better)")
        axis.set_title(title + "\nconditional diagnostic, raw timing targets")
        axis.set_xticks(range(1, 5))
        axis.set_xlim(0.8, 4.2)
        axis.legend(frameon=True)
    fig.suptitle("PianoCoRe-A slot-sum pretraining (live)", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    summary = {"output": str(output.resolve()), "training_records": len(records),
               "current_step": int(steps.max()), "current_epoch": current_epoch,
               "steps_per_epoch": steps_per_epoch,
               "validation_loss": valid, "valid_tf_pn_hr": pn_hr, "valid_tf_pp_hr": pp_hr,
               "hr_protocol": "conditional analytic/histogram W1 with raw timing targets; diagnostic only"}
    (run / "pianocoreA_training_curve_data.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
