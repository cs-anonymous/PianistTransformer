#!/usr/bin/env python3
"""Parse trainer logs and plot loss curves.

Each train.log contains concatenated output from a multi-stage pipeline:

    stage 1 (train/base, may appear multiple times across separate runs) — denom 12284
    stage 2 (adapt_2ep)                                                  — denom 216
    stage 3 (adapt_4ep)                                                  — denom 432

Stages are identified by the progress-bar denominator (`|0/N[`). Each event is
emitted in two formats — JSON (with `step`) and Python-dict (with `epoch`) — and
we keep only the JSON lines. When the same step appears twice (e.g. DDP
rank-1 echoing the same line, or a stage re-emitted after resume) we keep the
first occurrence.
"""
from __future__ import annotations

import json
import re
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE = Path("/home/sy/EPR/PianistTransformer/results/inr0624_binary4_prior_ablation_4gpu")
RUNS = [
    BASE / "gpu0_cine_kp1_binary4" / "train.log",
    BASE / "cine_kp05_nosplit" / "train.log",
]
RUN_LABELS = ["kp=1.0", "kp=0.5 nosplit"]
SUB_LOSSES = ["loss_ioi", "loss_duration", "loss_velocity", "loss_pedal"]
ALL_LOSSES = ["loss"] + SUB_LOSSES

STAGE_RENAMES = {
    12284: "train",
    216:   "adapt_2ep",
    432:   "adapt_4ep",
}


# --------------------------------------------------------------------------- #
# Parser                                                                      #
# --------------------------------------------------------------------------- #

# Anchors we use to split a line into JSON dicts and progress-bar denominator.
JSON_START = re.compile(r'\{"(grad_norm|eval_loss|train_runtime|step)"')
BAR_DENOM = re.compile(r"\|\s*\d+/(\d+)\s*\[")


def _extract_json_objects(line: str) -> list[str]:
    """Yield JSON-object substrings (start with one of the known keys)."""
    out = []
    pos = 0
    while True:
        m = JSON_START.search(line, pos)
        if not m:
            break
        start = m.start()
        depth = 0
        end = -1
        for k in range(start, len(line)):
            if line[k] == '{':
                depth += 1
            elif line[k] == '}':
                depth -= 1
                if depth == 0:
                    end = k + 1
                    break
        if end < 0:
            break
        out.append(line[start:end])
        pos = end
    return out


def _extract_bar_denom(line: str) -> int | None:
    m = BAR_DENOM.search(line)
    return int(m.group(1)) if m else None


def parse_log(path: Path) -> dict:
    """Return dict: stage_name -> {'step', 'epoch', 'loss', 'sub', 'grad_norm',
    'lr', 'eval_step', 'eval_loss'}. All key stages are merged across multiple
    base training runs."""
    text = path.read_text(errors="ignore")
    # Normalize line endings (file mixes \n, \r\n and \r because tqdm re-uses \r
    # for in-place progress-bar updates).
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    stages: "OrderedDict[str, dict]" = OrderedDict()
    current = "train"        # default: anything before first Loaded marker is still train
    current_denom = 12284
    seen_train: dict[str, set] = {}   # stage -> set of train step values
    seen_eval: dict[str, set] = {}    # stage -> set of eval step values

    def ensure_stage(name: str) -> None:
        if name not in stages:
            stages[name] = {
                "step": [], "epoch": [], "loss": [],
                "sub": {k: [] for k in SUB_LOSSES},
                "grad_norm": [], "lr": [],
                "eval_step": [], "eval_loss": [],
            }
            seen_train[name] = set()
            seen_eval[name] = set()

    ensure_stage("train")

    for line in text.split("\n"):
        denom = _extract_bar_denom(line)
        if denom is not None and denom in STAGE_RENAMES:
            current_denom = denom
            current = STAGE_RENAMES[denom]
            ensure_stage(current)

        for raw in _extract_json_objects(line):
            try:
                rec = json.loads(raw)
            except Exception:
                continue

            s = stages[current]

            if "eval_loss" in rec:
                # Evals share step values with training events at the same step,
                # so we maintain a separate dedup set.
                st = rec.get("step")
                if st is not None and st not in seen_eval[current]:
                    seen_eval[current].add(st)
                    s["eval_step"].append(float(st))
                    s["eval_loss"].append(float(rec["eval_loss"]))
                continue

            # training event — must have step + loss
            if "step" not in rec or "loss" not in rec:
                continue
            st = int(rec["step"])
            if st in seen_train[current]:
                continue
            seen_train[current].add(st)

            s["step"].append(float(st))
            s["epoch"].append(float(rec.get("epoch", np.nan)))
            s["loss"].append(float(rec["loss"]))
            for k in SUB_LOSSES:
                s["sub"][k].append(float(rec[k]) if k in rec else np.nan)
            s["grad_norm"].append(float(rec["grad_norm"]) if "grad_norm" in rec else np.nan)
            s["lr"].append(float(rec["learning_rate"]) if "learning_rate" in rec else np.nan)

    # Cast to ndarrays (preserve insertion order = file order)
    out = {}
    for name, s in stages.items():
        if not s["step"]:
            continue
        out[name] = {
            "step": np.asarray(s["step"], dtype=float),
            "epoch": np.asarray(s["epoch"], dtype=float),
            "loss": np.asarray(s["loss"], dtype=float),
            "sub": {k: np.asarray(s["sub"][k], dtype=float) for k in SUB_LOSSES},
            "grad_norm": np.asarray(s["grad_norm"], dtype=float),
            "lr": np.asarray(s["lr"], dtype=float),
            "eval_step": np.asarray(s["eval_step"], dtype=float),
            "eval_loss": np.asarray(s["eval_loss"], dtype=float),
        }
    return out


# --------------------------------------------------------------------------- #
# Plotting                                                                    #
# --------------------------------------------------------------------------- #

COLOR_GPU0 = "#2a78d6"   # slot 1, blue
COLOR_GPU1 = "#e34948"   # slot 6, red
COLOR_EVAL = "#0b0b0b"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"


def smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(y) < window:
        return y
    return np.convolve(y, np.ones(window) / window, mode="valid")


def smooth_window(n: int, ratio: int = 40) -> int:
    return max(5, n // ratio)


def _style_axes(ax) -> None:
    ax.grid(True, color=GRIDLINE, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)


def plot_stages_grid(runs_data, labels, out_dir):
    """Rows = stages, Cols = losses.  Each panel: gpu0 vs gpu1."""
    stage_order = ["train", "adapt_2ep", "adapt_4ep"]
    n_rows, n_cols = len(stage_order), len(ALL_LOSSES)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.7 * n_cols, 3.0 * n_rows),
                             sharex="col")
    palette = [COLOR_GPU0, COLOR_GPU1]

    for r, stage in enumerate(stage_order):
        for c, key in enumerate(ALL_LOSSES):
            ax = axes[r, c] if n_rows > 1 else axes[c]
            plotted = False
            for run, label, color in zip(runs_data, labels, palette):
                if stage not in run:
                    continue
                s = run[stage]
                y, x = (s["loss"], s["step"]) if key == "loss" else (s["sub"][key], s["step"])
                if len(x) < 2:
                    continue
                ax.plot(x, y, color=color, alpha=0.22, linewidth=0.6)
                sw = smooth_window(len(x))
                ys = smooth(y, sw)
                xs = x[len(x) - len(ys):]
                ax.plot(xs, ys, color=color, linewidth=1.8, label=label)
                plotted = True
            if plotted:
                _style_axes(ax)
                if r == 0:
                    ax.set_title(key, fontsize=10, color=INK_PRIMARY)
                if c == 0:
                    ax.set_ylabel(stage, fontsize=10, color=INK_PRIMARY,
                                  rotation=0, ha="right", va="center")
                if r == n_rows - 1:
                    ax.set_xlabel("step", fontsize=9, color=INK_SECONDARY)
                if r == 0 and c == n_cols - 1:
                    ax.legend(fontsize=8, frameon=False, loc="best")
    fig.suptitle("Loss per stage — gpu0 (kp=1.0) vs gpu1 (kp=0.5)",
                 color=INK_PRIMARY, fontsize=13, y=0.995)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / "loss_per_stage_grid.png", dpi=150,
                facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_stage_overview(run, label, color, out_dir, stage):
    if stage not in run or len(run[stage]["step"]) == 0:
        return
    s = run[stage]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, k in zip(axes.flat, ALL_LOSSES):
        y, x = (s["loss"], s["step"]) if k == "loss" else (s["sub"][k], s["step"])
        if len(x) >= 2:
            ax.plot(x, y, color=color, alpha=0.3, linewidth=0.7)
            sw = smooth_window(len(x))
            ys = smooth(y, sw)
            xs = x[len(x) - len(ys):]
            ax.plot(xs, ys, color=color, linewidth=2.0, label="train")
        if len(s["eval_step"]) > 0:
            ax.plot(s["eval_step"], s["eval_loss"], color=COLOR_EVAL,
                    marker="o", markersize=4, linewidth=0, label="eval")
        ax.set_title(k, color=INK_PRIMARY, fontsize=10)
        ax.set_xlabel("step", color=INK_SECONDARY, fontsize=9)
        _style_axes(ax)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle(f"{label} — {stage}", color=INK_PRIMARY, fontsize=12)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    slug = label.replace(" ", "_").replace("=", "_").replace("(", "").replace(")", "")
    fig.savefig(out_dir / f"loss_{slug}_{stage}.png", dpi=150,
                facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_overall_compare(runs_data, labels, out_dir):
    """Per-loss panel, both runs, all stages concatenated with hairlines."""
    n = len(ALL_LOSSES)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.5))
    palette = [COLOR_GPU0, COLOR_GPU1]
    for ax, key in zip(axes, ALL_LOSSES):
        for run, label, color in zip(runs_data, labels, palette):
            x_all, y_all, boundaries = [], [], []
            cur = 0
            for stage in ("train", "adapt_2ep", "adapt_4ep"):
                if stage not in run or len(run[stage]["step"]) == 0:
                    continue
                s = run[stage]
                y = s["loss"] if key == "loss" else s["sub"][key]
                x = s["step"]
                if len(x) < 2:
                    continue
                sw = smooth_window(len(x))
                ys = smooth(y, sw)
                xs = x[len(x) - len(ys):] + cur
                x_all.append(xs)
                y_all.append(ys)
                boundaries.append((cur + len(xs), stage))
                cur += len(xs)
            if not x_all:
                continue
            x = np.concatenate(x_all)
            y = np.concatenate(y_all)
            ax.plot(x, y, color=color, linewidth=1.6, label=label)
            for b, _ in boundaries[:-1]:
                ax.axvline(b, color=INK_SECONDARY, linewidth=0.4,
                           linestyle="--", alpha=0.5)
        ax.set_title(key, color=INK_PRIMARY, fontsize=10)
        ax.set_xlabel("step (concatenated)", color=INK_SECONDARY, fontsize=9)
        _style_axes(ax)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle("All stages concatenated — train → adapt_2ep → adapt_4ep (smoothed)",
                 color=INK_PRIMARY, fontsize=12)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(out_dir / "loss_all_stages_compare.png", dpi=150,
                facecolor=fig.get_facecolor())
    plt.close(fig)


def print_summary(runs_data, labels):
    for run, label in zip(runs_data, labels):
        print(f"\n=== {label} ===")
        for stage in ("train", "adapt_2ep", "adapt_4ep"):
            if stage not in run:
                continue
            s = run[stage]
            print(f"  [{stage:<10}] points={len(s['step']):4d} "
                  f"step range={s['step'][0]:>5.0f}..{s['step'][-1]:>5.0f}  "
                  f"eval pts={len(s['eval_step']):2d}  "
                  f"loss first/last={s['loss'][0]:.3f}/{s['loss'][-1]:.3f}")
            for k in SUB_LOSSES:
                y = s["sub"][k]
                m = ~np.isnan(y)
                if m.any():
                    print(f"      {k:<14}  first/last={y[m][0]:.3f}/{y[m][-1]:.3f}")


def main():
    runs_data = []
    for p in RUNS:
        print(f"parsing {p.name} ...")
        runs_data.append(parse_log(p))

    out_dir = BASE / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    print_summary(runs_data, RUN_LABELS)
    plot_stages_grid(runs_data, RUN_LABELS, out_dir)
    for run, label in zip(runs_data, RUN_LABELS):
        color = COLOR_GPU0 if label.startswith("kp=1") else COLOR_GPU1
        for stage in ("train", "adapt_2ep", "adapt_4ep"):
            plot_stage_overview(run, label, color, out_dir, stage)
    plot_overall_compare(runs_data, RUN_LABELS, out_dir)

    print(f"\nplots written to: {out_dir}")


if __name__ == "__main__":
    main()
