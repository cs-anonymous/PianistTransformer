#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
JSON_START = re.compile(r'\{"(grad_norm|eval_loss|train_runtime|step)"')
TRAIN_KEYS = {
    "loss": "Total",
    "train_loss_ioi": "IOI",
    "train_loss_duration": "Duration",
    "train_loss_velocity": "Velocity",
    "train_loss_pedal": "Pedal",
}
RUNS = {
    "Sine": {
        "color": "#1f77b4",
        "log": ROOT_DIR
        / "results/slot8_fixed_vs_sine_2gpu/20260710_slot8fix_2gpu_ddpfind/sine-control/train.log",
        "summary": ROOT_DIR
        / "results/slot8_fixed_vs_sine_2gpu/20260710_slot8fix_2gpu_ddpfind/sine-control/summary.json",
    },
    "Independent property PAD": {
        "color": "#d62728",
        "log": ROOT_DIR
        / "results/slot8_fixed_vs_sine_2gpu/20260710_slot8fix_2gpu_ddpfind/slot8-fixed/train.log",
        "summary": ROOT_DIR
        / "results/slot8_fixed_vs_sine_2gpu/20260710_slot8fix_2gpu_ddpfind/slot8-fixed/summary.json",
    },
    "Whole-token PAD": {
        "color": "#2ca02c",
        "log": ROOT_DIR
        / "results/slot8_mask_stable_2gpu/20260710_slot8_mask_stable_v1/slot8-whole-token-mask/train.log",
        "summary": ROOT_DIR
        / "results/slot8_mask_stable_2gpu/20260710_slot8_mask_stable_v1/slot8-whole-token-mask/summary.json",
    },
    "Correlated property PAD": {
        "color": "#ff7f0e",
        "log": ROOT_DIR
        / "results/slot8_property_schedule_2gpu/20260710_slot8_property_schedule_v1/slot8-correlated-perf-pad50/train.log",
        "summary": ROOT_DIR
        / "results/slot8_property_schedule_2gpu/20260710_slot8_property_schedule_v1/slot8-correlated-perf-pad50/summary.json",
    },
    "Mixed property MASK + stable": {
        "color": "#9467bd",
        "log": ROOT_DIR
        / "results/slot8_property_schedule_2gpu/20260710_slot8_property_schedule_v1/slot8-mixed-property-mask-stable/train.log",
        "summary": ROOT_DIR
        / "results/slot8_property_schedule_2gpu/20260710_slot8_property_schedule_v1/slot8-mixed-property-mask-stable/summary.json",
    },
}


def extract_json_objects(line: str) -> list[dict]:
    records = []
    pos = 0
    while True:
        match = JSON_START.search(line, pos)
        if match is None:
            break
        start = match.start()
        depth = 0
        end = None
        for index in range(start, len(line)):
            if line[index] == "{":
                depth += 1
            elif line[index] == "}":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break
        if end is None:
            break
        try:
            records.append(json.loads(line[start:end]))
        except json.JSONDecodeError:
            pass
        pos = end
    return records


def parse_log(path: Path) -> dict:
    train_records: dict[int, dict] = {}
    eval_records: dict[int, float] = {}
    text = path.read_text(errors="ignore").replace("\r\n", "\n").replace("\r", "\n")
    for line in text.splitlines():
        for record in extract_json_objects(line):
            if "step" not in record:
                continue
            step = int(record["step"])
            if "eval_loss" in record:
                eval_records.setdefault(step, float(record["eval_loss"]))
            elif "loss" in record and "grad_norm" in record:
                train_records.setdefault(step, record)

    train_steps = np.asarray(sorted(train_records), dtype=float)
    eval_steps = np.asarray(sorted(eval_records), dtype=float)
    return {
        "train_step": train_steps,
        "train": {
            key: np.asarray(
                [float(train_records[int(step)].get(key, np.nan)) for step in train_steps],
                dtype=float,
            )
            for key in TRAIN_KEYS
        },
        "eval_step": eval_steps,
        "eval_loss": np.asarray([eval_records[int(step)] for step in eval_steps], dtype=float),
    }


def moving_average(values: np.ndarray, window: int = 7) -> np.ndarray:
    if len(values) < window:
        return values.copy()
    return np.convolve(values, np.ones(window) / window, mode="valid")


def finite_series(steps: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(values)
    return steps[valid], values[valid]


def late_slope_per_1000(steps: np.ndarray, values: np.ndarray) -> float:
    steps, values = finite_series(steps, values)
    if len(values) < 4:
        return float("nan")
    count = max(8, len(values) // 4)
    return float(np.polyfit(steps[-count:], values[-count:], 1)[0] * 1000.0)


def mean_edge(values: np.ndarray, first: bool) -> float:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    count = min(10, len(values))
    return float(np.mean(values[:count] if first else values[-count:]))


def load_sampling_metrics(path: Path) -> dict:
    summary = json.loads(path.read_text())
    pp = summary["metrics"]["sampling"]["aggregate"]["pp_wass"]
    return {
        "ioi_wass": float(pp["ioi_wass"]),
        "duration_wass": float(pp["duration_wass"]),
        "velocity_wass": float(pp["velocity_wass"]),
        "pedal_wass": float(pp["pedal_wass"]),
    }


def style_axis(axis) -> None:
    axis.grid(True, color="#dedede", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(labelsize=8)


def plot_loss_curves(data: dict, output_path: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    panels = list(TRAIN_KEYS.items())
    for axis, (key, title) in zip(axes.flat[:5], panels):
        for name, item in data.items():
            steps, values = finite_series(item["train_step"], item["train"][key])
            axis.plot(steps, values, color=item["color"], alpha=0.10, linewidth=0.7)
            smoothed = moving_average(values)
            smooth_steps = steps[len(steps) - len(smoothed) :]
            axis.plot(smooth_steps, smoothed, color=item["color"], linewidth=2.0, label=name)
        axis.set_title(f"Train {title} loss")
        axis.set_xlabel("optimizer step")
        style_axis(axis)

    eval_axis = axes.flat[5]
    for name, item in data.items():
        eval_axis.plot(
            item["eval_step"],
            item["eval_loss"],
            color=item["color"],
            marker="o",
            markersize=3.5,
            linewidth=1.8,
            label=name,
        )
    eval_axis.set_title("Validation total loss, no training corruption")
    eval_axis.set_xlabel("optimizer step")
    style_axis(eval_axis)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle("Slot8 decoder masking: loss and convergence comparison", fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_eval_rollout(data: dict, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, metric, title in (
        (axes[0], "ioi_wass", "Sampling PP Wasserstein: IOI"),
        (axes[1], "duration_wass", "Sampling PP Wasserstein: Duration"),
    ):
        for name, item in data.items():
            best_eval = float(np.min(item["eval_loss"]))
            value = item["sampling"][metric]
            axis.scatter(best_eval, value, color=item["color"], s=62)
            axis.annotate(name, (best_eval, value), xytext=(5, 4), textcoords="offset points", fontsize=8)
        axis.set_xlabel("best validation loss")
        axis.set_ylabel(metric.replace("_wass", " Wasserstein"))
        axis.set_title(title)
        style_axis(axis)
    fig.suptitle("Teacher-forced validation loss does not predict sampling rollout quality", fontsize=14)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_report(data: dict) -> dict:
    report = {}
    for name, item in data.items():
        eval_index = int(np.argmin(item["eval_loss"]))
        report[name] = {
            "train_points": int(len(item["train_step"])),
            "eval_points": int(len(item["eval_step"])),
            "best_eval_loss": float(item["eval_loss"][eval_index]),
            "best_eval_step": int(item["eval_step"][eval_index]),
            "final_eval_loss": float(item["eval_loss"][-1]),
            "train": {
                TRAIN_KEYS[key].lower(): {
                    "first_10_mean": mean_edge(item["train"][key], first=True),
                    "last_10_mean": mean_edge(item["train"][key], first=False),
                    "last_quarter_slope_per_1000_steps": late_slope_per_1000(
                        item["train_step"], item["train"][key]
                    ),
                }
                for key in TRAIN_KEYS
            },
            "sampling_pp_wass": item["sampling"],
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "results/slot8_mask_loss_analysis_20260710",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = {}
    for name, spec in RUNS.items():
        parsed = parse_log(spec["log"])
        parsed["color"] = spec["color"]
        parsed["sampling"] = load_sampling_metrics(spec["summary"])
        data[name] = parsed

    plot_loss_curves(data, args.output_dir / "slot8_mask_loss_curves.png")
    plot_eval_rollout(data, args.output_dir / "eval_loss_vs_sampling_rollout.png")
    report = build_report(data)
    (args.output_dir / "convergence_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
