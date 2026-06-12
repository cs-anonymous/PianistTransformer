#!/usr/bin/env python3

import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt


LOG_SPECS = [
    (
        "T5 10+2",
        Path("logs/train_t5_10_2_h1024_l1024_20260608_020427.log"),
    ),
    (
        "T5 6+6",
        Path("logs/train_t5_6_6_h1024_l1024_20260608_020427.log"),
    ),
    (
        "BERT 17",
        Path("logs/train_bert_17_h1024_l1024_20260608_020427.log"),
    ),
]

OUTPUT_PATH = Path("results/loss_curves_h1024_l1024_training.png")


def parse_record(line: str):
    start = line.find("{")
    end = line.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    payload = line[start : end + 1]
    for parser in (json.loads, ast.literal_eval):
        try:
            record = parser(payload)
        except Exception:
            continue
        if isinstance(record, dict):
            return record
    return None


def load_series(log_path: Path):
    train_by_step = {}
    eval_by_step = {}

    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            record = parse_record(raw_line)
            if not record or "step" not in record:
                continue

            step = int(record["step"])
            if "eval_loss" in record:
                eval_by_step[step] = float(record["eval_loss"])
            elif "loss" in record and "train_runtime" not in record:
                train_by_step[step] = {
                    "loss": float(record["loss"]),
                    "loss_ioi": float(record.get("loss_ioi", "nan")),
                    "loss_duration": float(record.get("loss_duration", "nan")),
                    "loss_velocity": float(record.get("loss_velocity", "nan")),
                    "loss_pedal": float(record.get("loss_pedal", "nan")),
                }

    train_steps = sorted(train_by_step)
    eval_steps = sorted(eval_by_step)
    train_series = {
        key: [train_by_step[step][key] for step in train_steps]
        for key in ["loss", "loss_ioi", "loss_duration", "loss_velocity", "loss_pedal"]
    }
    eval_series = [eval_by_step[step] for step in eval_steps]
    return train_steps, train_series, eval_steps, eval_series


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(14, 15), sharex=False)
    colors = {
        "loss": "#111111",
        "loss_ioi": "#1f77b4",
        "loss_duration": "#ff7f0e",
        "loss_velocity": "#2ca02c",
        "loss_pedal": "#d62728",
        "eval_loss": "#9467bd",
    }
    labels = {
        "loss": "Train total",
        "loss_ioi": "Train IOI",
        "loss_duration": "Train Duration",
        "loss_velocity": "Train Velocity",
        "loss_pedal": "Train Pedal",
    }

    for ax, (title, log_path) in zip(axes, LOG_SPECS):
        train_steps, train_series, eval_steps, eval_series = load_series(log_path)

        ax.plot(train_steps, train_series["loss"], color=colors["loss"], linewidth=2.0, label=labels["loss"])
        ax.plot(train_steps, train_series["loss_ioi"], color=colors["loss_ioi"], linewidth=1.5, label=labels["loss_ioi"])
        ax.plot(
            train_steps,
            train_series["loss_duration"],
            color=colors["loss_duration"],
            linewidth=1.5,
            label=labels["loss_duration"],
        )
        ax.plot(
            train_steps,
            train_series["loss_velocity"],
            color=colors["loss_velocity"],
            linewidth=1.5,
            label=labels["loss_velocity"],
        )
        ax.plot(
            train_steps,
            train_series["loss_pedal"],
            color=colors["loss_pedal"],
            linewidth=1.5,
            label=labels["loss_pedal"],
        )

        if eval_steps:
            ax.plot(
                eval_steps,
                eval_series,
                color=colors["eval_loss"],
                linestyle="--",
                linewidth=2.2,
                marker="o",
                markersize=4,
                label="Eval total",
            )

        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", ncol=2, fontsize=9)

    fig.suptitle("Integrated Node Training Curves (h1024, l1024)", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(OUTPUT_PATH, dpi=200)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
