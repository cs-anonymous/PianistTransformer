#!/usr/bin/env python3
"""Plot comparable training loss curves from INR train.log files."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RUNS = {
    "Old slot6": Path(
        "results/inr_epr_pipeline/bounded_slot_support_probe_20260717_210939/"
        "slot6_only/train.log"
    ),
    "DINR": Path(
        "results/inr_epr_pipeline/lossnorm_ep20_baselines_20260718_001238/"
        "dinr/train.log"
    ),
    "CINR": Path(
        "results/inr_epr_pipeline/lossnorm_ep20_baselines_20260718_001238/"
        "cinr/train.log"
    ),
    "CINR bounded 5%": Path(
        "results/inr_epr_pipeline/lossnorm_ep20_baselines_20260718_001238/"
        "cinr_bounded_5pct/train.log"
    ),
}

FIELDS = {
    "loss": "Total loss",
    "train_loss_ioi": "IOI",
    "train_loss_duration": "Duration",
    "train_loss_velocity": "Velocity",
    "train_loss_pedal": "Pedal",
    "train_loss_weighted_ioi": "Weighted IOI",
    "train_loss_weighted_duration": "Weighted Duration",
    "train_loss_weighted_velocity": "Weighted Velocity",
    "train_loss_weighted_pedal": "Weighted Pedal",
}

COLORS = {
    "Old slot6": "#777777",
    "DINR": "#2563eb",
    "CINR": "#d97706",
    "CINR bounded 5%": "#059669",
}


def records(path: Path) -> dict[str, np.ndarray]:
    values: dict[int, dict] = {}
    eval_values: dict[int, float] = {}
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        candidates = []
        if line.startswith("{"):
            candidates.append(line)
        for start in range(len(line)):
            if line[start] != "{":
                continue
            depth = 0
            for end in range(start, len(line)):
                if line[end] == "{":
                    depth += 1
                elif line[end] == "}":
                    depth -= 1
                    if depth == 0:
                        candidates.append(line[start : end + 1])
                        break
        for raw in candidates:
            try:
                obj = json.loads(raw)
            except Exception:
                try:
                    obj = ast.literal_eval(raw)
                except Exception:
                    continue
            if not isinstance(obj, dict) or "step" not in obj:
                continue
            step = int(obj["step"])
            if "loss" in obj and any(k in obj for k in FIELDS if k != "loss"):
                values[step] = obj
            if "eval_loss" in obj:
                eval_values[step] = float(obj["eval_loss"])
    if not values:
        raise RuntimeError(f"No train records found in {path}")
    steps = np.array(sorted(values), dtype=float)
    out = {"step": steps}
    for key in FIELDS:
        out[key] = np.array([float(values[int(s)].get(key, np.nan)) for s in steps])
    out["eval_step"] = np.array(sorted(eval_values), dtype=float)
    out["eval_loss"] = np.array([eval_values[int(s)] for s in out["eval_step"]])
    return out


def smooth(y: np.ndarray, window: int = 9) -> np.ndarray:
    if len(y) < window:
        return y
    # Edge-pad before convolution so the moving average does not fall toward
    # zero at the beginning and end of a finite training log.
    radius = window // 2
    padded = np.pad(y, (radius, radius), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def main() -> None:
    out_dir = Path(
        "results/inr_epr_pipeline/lossnorm_ep20_baselines_20260718_001238/loss_curves"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {name: records(path) for name, path in RUNS.items()}

    # The old run has 16 epochs and current runs have 20; normalize by each
    # run's completed epoch count so the x-axis represents training progress.
    for run in data.values():
        run["epoch"] = run["step"] / run["step"].max() * 20
        run["eval_epoch"] = (
            run["eval_step"] / run["step"].max() * 20 if len(run["eval_step"]) else []
        )

    fig, axes = plt.subplots(3, 2, figsize=(15, 13), constrained_layout=True)
    panels = [
        ("loss", "Total train loss"),
        ("eval_loss", "Eval loss"),
        ("train_loss_ioi", "IOI loss"),
        ("train_loss_duration", "Duration loss"),
        ("train_loss_velocity", "Velocity loss"),
        ("train_loss_pedal", "Pedal loss"),
    ]
    for ax, (key, title) in zip(axes.flat, panels):
        for name, run in data.items():
            x = run["eval_epoch"] if key == "eval_loss" else run["epoch"]
            y = run[key]
            if key == "eval_loss" and len(x) == 0:
                continue
            ax.plot(x, y, color=COLORS[name], alpha=0.18, linewidth=0.7)
            if key != "eval_loss":
                ax.plot(x, smooth(y), color=COLORS[name], linewidth=1.8, label=name)
            else:
                ax.plot(x, y, color=COLORS[name], linewidth=1.8, marker="o", markersize=2)
        ax.set_title(title)
        ax.set_xlabel("Normalized training epoch (20 = run end)")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("Loss comparison: old slot6 vs loss-normalized baselines", fontsize=15)
    fig.savefig(out_dir / "loss_curves_old_slot6_vs_current.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for ax, (key, title) in zip(
        axes.flat,
        [
            ("train_loss_ioi", "IOI"),
            ("train_loss_duration", "Duration"),
            ("train_loss_velocity", "Velocity"),
            ("train_loss_pedal", "Pedal"),
        ],
    ):
        for name, run in data.items():
            ax.plot(
                run["epoch"],
                smooth(run[key]),
                color=COLORS[name],
                linewidth=2,
                label=name,
            )
        ax.set_title(title)
        ax.set_xlabel("Normalized training epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("Training loss components", fontsize=15)
    fig.savefig(out_dir / "loss_curves_components_old_slot6_vs_current.png", dpi=180)
    plt.close(fig)

    print(f"saved plots to {out_dir}")
    for name, run in data.items():
        print(name, "records=", len(run["step"]), "eval=", len(run["eval_step"]))


if __name__ == "__main__":
    main()
