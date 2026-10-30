#!/usr/bin/env python3
"""Extract epoch losses and plot the outer median overfitting comparison."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/outer_median_overfit_8ep"
MODELS = ("dinr_no_coord", "dinr_coord", "cinr")


def latest_state(model):
    candidates = list((RESULTS / model / "training").glob("*/checkpoint-[0-9]*/trainer_state.json"))
    if not candidates:
        raise FileNotFoundError(f"No numbered checkpoints found for {model}")
    path = max(candidates, key=lambda item: int(item.parent.name.rsplit("-", 1)[1]))
    return path, json.loads(path.read_text(encoding="utf-8"))


def epoch_train_losses(state):
    history = state.get("log_history", [])
    total_steps = int(state.get("max_steps", 0))
    epochs = int(float(state.get("num_train_epochs", 0)))
    steps_per_epoch = total_steps // max(epochs, 1)
    logs = [row for row in history if "loss" in row and "eval_loss" not in row and "step" in row]
    weighted = {epoch: [0.0, 0] for epoch in range(1, epochs + 1)}
    previous_step = 0
    for row in logs:
        step = int(row["step"])
        start, end = previous_step + 1, step
        value = float(row.get("train_loss_components_total", row["loss"]))
        for epoch in range(1, epochs + 1):
            epoch_start = (epoch - 1) * steps_per_epoch + 1
            epoch_end = epoch * steps_per_epoch
            overlap = max(0, min(end, epoch_end) - max(start, epoch_start) + 1)
            if overlap:
                weighted[epoch][0] += value * overlap
                weighted[epoch][1] += overlap
        previous_step = step
    return {
        epoch: (total / count if count else None)
        for epoch, (total, count) in weighted.items()
    }


def main():
    summary = {}
    for model in MODELS:
        path, state = latest_state(model)
        history = state.get("log_history", [])
        train = epoch_train_losses(state)
        eval_by_epoch = {}
        for row in history:
            if "eval_loss" not in row:
                continue
            epoch = int(round(float(row.get("epoch", 0))))
            if abs(float(row.get("epoch", 0)) - epoch) < 1e-7:
                eval_by_epoch[epoch] = row
        summary[model] = {
            "checkpoint": str(path.parent.relative_to(ROOT)),
            "epochs": [
                {
                    "epoch": epoch,
                    "train_loss_epoch_mean": train.get(epoch),
                    "valid_loss": eval_by_epoch[epoch]["eval_loss"] if epoch in eval_by_epoch else None,
                    "valid_ioi": eval_by_epoch[epoch].get("eval_loss_ioi") if epoch in eval_by_epoch else None,
                    "valid_duration": eval_by_epoch[epoch].get("eval_loss_duration") if epoch in eval_by_epoch else None,
                    "valid_velocity": eval_by_epoch[epoch].get("eval_loss_velocity") if epoch in eval_by_epoch else None,
                    "valid_pedal": eval_by_epoch[epoch].get("eval_loss_pedal") if epoch in eval_by_epoch else None,
                }
                for epoch in range(1, int(state.get("num_train_epochs", 0)) + 1)
            ],
        }
    out_json = RESULTS / "loss_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, model in zip(axes, MODELS):
        rows = summary[model]["epochs"]
        epochs = [row["epoch"] for row in rows]
        axis.plot(epochs, [row["train_loss_epoch_mean"] for row in rows], marker="o", label="Train (epoch mean)")
        axis.plot(epochs, [row["valid_loss"] for row in rows], marker="o", label="Valid")
        axis.set(title=model.replace("_", " "), xlabel="Epoch", ylabel="Loss")
        axis.set_xticks(epochs)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    fig.suptitle("Outer INR training on median-scaled ASAP labels (50% overlap)")
    fig.tight_layout()
    fig.savefig(RESULTS / "loss_curves.png", dpi=180)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
