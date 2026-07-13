#!/usr/bin/env python3
"""Plot per-feature train/eval losses for an INR run directory.

This script is designed for INR0624-style logs where multiple stages are
concatenated into one `train.log`:

    train    -> progress denominator 12284
    adapt_2ep -> progress denominator 216
    adapt_4ep -> progress denominator 432

Train sub-losses (`loss_ioi`, `loss_duration`, `loss_velocity`, `loss_pedal`)
are parsed directly from the log. Eval sub-losses are not logged by Trainer, so
we recompute them offline for the checkpoints that still exist on disk.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT_DIR))

from src.model.integrated_pianoformer import _compute_integrated_loss_components
from src.train.train_inr import (
    NodeSFTDataCollator,
    PianoCoReNodeSFTDataset,
    build_work_manifest,
    create_model,
    default_input_continuous_dim,
    infer_input_feature_mode,
    integrated_csr_output_dim,
    integrated_epr_input_dim,
    load_torch_state_dict,
    resolve_timing_control_mode,
)


SUB_LOSSES = ["loss_ioi", "loss_duration", "loss_velocity", "loss_pedal"]
FEATURE_NAMES = {
    "loss_ioi": "IOI",
    "loss_duration": "Duration",
    "loss_velocity": "Velocity",
    "loss_pedal": "Pedal",
}
STAGE_RENAMES = {
    12284: "train",
    216: "adapt_2ep",
    432: "adapt_4ep",
}
STAGE_TITLES = {
    "train": "Train",
    "adapt_2ep": "Adapt 2ep",
    "adapt_4ep": "Adapt 4ep",
}

JSON_START = re.compile(r'\{"(grad_norm|eval_loss|train_runtime|step)"')
BAR_DENOM = re.compile(r"\|\s*\d+/(\d+)\s*\[")

INK_PRIMARY = "#111111"
INK_SECONDARY = "#5a5956"
GRID = "#dfddd7"
SURFACE = "#fbfaf7"
TRAIN_RAW = "#cfd8ea"
TRAIN_SMOOTH = "#2b6cb0"
EVAL_COLOR = "#d64545"
TOTAL_EVAL_COLOR = "#222222"


@dataclass
class EvalCheckpoint:
    stage: str
    step: int
    checkpoint_dir: Path
    train_config_path: Path


def _extract_json_objects(line: str) -> list[str]:
    out = []
    pos = 0
    while True:
        m = JSON_START.search(line, pos)
        if not m:
            break
        start = m.start()
        depth = 0
        end = -1
        for idx in range(start, len(line)):
            if line[idx] == "{":
                depth += 1
            elif line[idx] == "}":
                depth -= 1
                if depth == 0:
                    end = idx + 1
                    break
        if end < 0:
            break
        out.append(line[start:end])
        pos = end
    return out


def _extract_bar_denom(line: str) -> int | None:
    m = BAR_DENOM.search(line)
    return int(m.group(1)) if m else None


def smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(y) < window:
        return y
    return np.convolve(y, np.ones(window) / window, mode="valid")


def smooth_window(n: int, ratio: int = 35) -> int:
    return max(5, n // ratio)


def style_axes(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK_SECONDARY)
    ax.spines["bottom"].set_color(INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)


def parse_log(log_path: Path) -> dict[str, dict]:
    text = log_path.read_text(errors="ignore").replace("\r\n", "\n").replace("\r", "\n")

    stages: "OrderedDict[str, dict]" = OrderedDict()
    current = "train"
    seen_train: dict[str, set[int]] = defaultdict(set)
    seen_eval_total: dict[str, set[int]] = defaultdict(set)

    def ensure_stage(name: str) -> None:
        if name not in stages:
            stages[name] = {
                "step": [],
                "loss": [],
                "sub": {k: [] for k in SUB_LOSSES},
                "eval_total_step": [],
                "eval_total_loss": [],
            }

    ensure_stage("train")

    for line in text.split("\n"):
        denom = _extract_bar_denom(line)
        if denom is not None and denom in STAGE_RENAMES:
            current = STAGE_RENAMES[denom]
            ensure_stage(current)

        for raw in _extract_json_objects(line):
            try:
                rec = json.loads(raw)
            except Exception:
                continue

            stage = stages[current]
            if "eval_loss" in rec:
                step = int(rec.get("step", -1))
                if step >= 0 and step not in seen_eval_total[current]:
                    seen_eval_total[current].add(step)
                    stage["eval_total_step"].append(step)
                    stage["eval_total_loss"].append(float(rec["eval_loss"]))
                continue

            if "step" not in rec or "loss" not in rec:
                continue
            step = int(rec["step"])
            if step in seen_train[current]:
                continue
            seen_train[current].add(step)

            stage["step"].append(step)
            stage["loss"].append(float(rec["loss"]))
            for key in SUB_LOSSES:
                stage["sub"][key].append(float(rec[key]) if key in rec else np.nan)

    out = {}
    for stage_name, values in stages.items():
        if not values["step"]:
            continue
        out[stage_name] = {
            "step": np.asarray(values["step"], dtype=float),
            "loss": np.asarray(values["loss"], dtype=float),
            "sub": {k: np.asarray(v, dtype=float) for k, v in values["sub"].items()},
            "eval_total_step": np.asarray(values["eval_total_step"], dtype=float),
            "eval_total_loss": np.asarray(values["eval_total_loss"], dtype=float),
        }
    return out


def normalize_train_config(train_config: dict) -> dict:
    cfg = dict(train_config)
    task_type = cfg.get("task_type", "epr").lower()
    input_feature_mode = infer_input_feature_mode(cfg)
    cfg["input_feature_mode"] = input_feature_mode
    timing_control_mode = resolve_timing_control_mode(
        timing_control_mode=cfg.get("timing_control_mode"),
        use_timing_scale_bit=cfg.get("use_timing_scale_bit", False),
    )
    musical_feature_mode = str(
        cfg.get("musical_feature_mode", "continuous" if task_type == "csr" else "categorical")
    ).lower()
    cfg["musical_feature_mode"] = musical_feature_mode
    cfg.setdefault(
        "input_continuous_dim",
        integrated_epr_input_dim(
            timing_control_mode=timing_control_mode,
            use_timing_scale_bit=cfg.get("use_timing_scale_bit", False),
            musical_feature_mode=musical_feature_mode,
            pedal_control_dim=4,
        )
        if task_type in {"epr", "csr"} and input_feature_mode == "integrated"
        else default_input_continuous_dim(
            task_type,
            input_feature_mode,
            score_feature_dim=cfg.get("score_feature_dim", 8),
            continuous_dim=cfg.get("continuous_dim", 7),
            musical_feature_mode=musical_feature_mode,
        ),
    )
    if task_type == "csr":
        cfg.setdefault("output_continuous_dim", integrated_csr_output_dim())
    return cfg


def build_eval_dataset(train_config: dict) -> PianoCoReNodeSFTDataset:
    cfg = normalize_train_config(train_config)
    task_type = cfg.get("task_type", "epr").lower()
    input_feature_mode = cfg["input_feature_mode"]
    musical_feature_mode = cfg["musical_feature_mode"]

    eval_manifest = build_work_manifest(
        metadata_path=cfg["metadata_path"],
        refined_dir=cfg["refined_dir"],
        split=cfg.get("eval_split", "test"),
        block_notes=cfg["block_notes"],
        overlap_ratio=cfg["overlap_ratio"],
        min_notes=cfg["min_notes"],
        max_works=cfg.get("max_eval_works"),
        include_all_performance_dataset=cfg.get("eval_include_all_performance_dataset"),
        max_non_asap_performances_per_work=cfg.get("max_eval_non_asap_performances_per_work"),
        selection_seed=cfg.get("seed", 42),
        skip_work_paths=cfg.get("skip_work_paths"),
        performance_dataset=cfg.get("eval_performance_dataset"),
        exclude_performance_dataset=cfg.get("eval_exclude_performance_dataset"),
    )

    return PianoCoReNodeSFTDataset(
        eval_manifest,
        split=cfg.get("eval_split", "test"),
        task_type=task_type,
        input_feature_mode=input_feature_mode,
        shuffle=False,
        seed=cfg["seed"],
        max_performances_per_work=cfg.get("max_eval_performances_per_work"),
        max_windows_per_work=cfg.get("max_eval_windows_per_work"),
        cache_size=cfg.get("node_cache_size", 16),
        timing_normalization=cfg.get("timing_input_normalization", "scaled_log_5000_s10"),
        max_time_ms=cfg.get("max_time_ms", 10000.0),
        epr_timing_bins=cfg.get("epr_timing_bins", 5000),
        epr_value_bins=cfg.get("epr_value_bins", 128),
        pedal_representation=cfg.get("pedal_representation", "start_valley"),
        musical_feature_mode=musical_feature_mode,
        epr_timing_target=cfg.get("epr_timing_target", "log_deviation"),
        use_timing_scale_bit=cfg.get("use_timing_scale_bit", False),
        timing_control_mode=cfg.get("timing_control_mode", "log_scaled"),
        timing_log_scale=cfg.get("timing_log_scale", 50.0),
        precompute_items=cfg.get(
            "precompute_eval_dataset_items",
            cfg.get("precompute_dataset_items", False),
        ),
        use_prepared_sidecar=cfg.get("use_prepared_sidecar", True),
        prepared_sidecar_tag=cfg.get("prepared_sidecar_tag"),
    )


def collect_eval_checkpoints(run_dir: Path) -> list[EvalCheckpoint]:
    checkpoints: list[EvalCheckpoint] = []

    stage_roots = OrderedDict(
        [
            ("train", run_dir / "training"),
            ("adapt_2ep", run_dir / "adapt_2ep" / "training"),
            ("adapt_4ep", run_dir / "adapt_4ep" / "training"),
        ]
    )

    for stage, root in stage_roots.items():
        if not root.exists():
            continue
        for session_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            train_config_path = session_dir / "train_config.json"
            if not train_config_path.exists():
                continue
            ckpt_dirs = sorted(
                [p for p in session_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")],
                key=lambda p: int(p.name.split("-")[-1]),
            )
            for ckpt_dir in ckpt_dirs:
                step = int(ckpt_dir.name.split("-")[-1])
                checkpoints.append(
                    EvalCheckpoint(
                        stage=stage,
                        step=step,
                        checkpoint_dir=ckpt_dir,
                        train_config_path=train_config_path,
                    )
                )
    checkpoints.sort(key=lambda item: (list(STAGE_TITLES).index(item.stage), item.step))
    return checkpoints


def evaluate_checkpoint(
    checkpoint: EvalCheckpoint,
    device: torch.device,
    dataset_cache: dict[Path, tuple[dict, PianoCoReNodeSFTDataset]],
) -> dict[str, float]:
    if checkpoint.train_config_path not in dataset_cache:
        train_config = json.loads(checkpoint.train_config_path.read_text())
        dataset_cache[checkpoint.train_config_path] = (
            normalize_train_config(train_config),
            build_eval_dataset(train_config),
        )
    train_config, eval_dataset = dataset_cache[checkpoint.train_config_path]

    model_config = dict(train_config)
    model_config["resume_path"] = str(checkpoint.checkpoint_dir)
    model_config["reset_output_heads_on_resume"] = False
    model = create_model(model_config).to(device)
    state_dict = load_torch_state_dict(checkpoint.checkpoint_dir)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(
            f"[warn] {checkpoint.checkpoint_dir.name}: missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
    model.eval()

    batch_size = int(
        train_config.get(
            "per_device_eval_batch_size",
            train_config.get("per_device_train_batch_size", 1),
        )
    )
    collator = NodeSFTDataCollator(
        pitch_pad_id=train_config["pitch_pad_id"],
        task_type=train_config.get("task_type", "epr"),
    )
    loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )

    weighted_total = 0.0
    weighted_components = {key: 0.0 for key in SUB_LOSSES}
    num_examples = 0

    with torch.no_grad():
        for batch in loader:
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            outputs = model(**batch)
            batch_examples = int(batch["pitch_ids"].shape[0])
            weighted_total += float(outputs.loss.detach().float().cpu()) * batch_examples

            loss_mask = batch.get("label_mask", batch["attention_mask"]).detach()
            components = _compute_integrated_loss_components(
                model.config,
                outputs.logits.detach(),
                batch["labels_continuous"].detach(),
                loss_mask,
                labels_epr_bins=batch.get("labels_epr_bins"),
                score_shared_raw=batch.get("score_shared_raw"),
            )
            for key in SUB_LOSSES:
                component_key = key.removeprefix("loss_")
                weighted_components[key] += float(components[component_key].detach().float().cpu()) * batch_examples
            num_examples += batch_examples

    result = {
        "eval_loss": weighted_total / max(num_examples, 1),
        "num_examples": num_examples,
    }
    for key in SUB_LOSSES:
        result[f"eval_{key}"] = weighted_components[key] / max(num_examples, 1)
    return result


def plot_feature_grid(stage_data: dict, out_dir: Path) -> None:
    stage_order = [stage for stage in ("train", "adapt_2ep", "adapt_4ep") if stage in stage_data]
    fig, axes = plt.subplots(
        len(stage_order),
        len(SUB_LOSSES),
        figsize=(4.1 * len(SUB_LOSSES), 3.2 * len(stage_order)),
        squeeze=False,
    )

    for row, stage in enumerate(stage_order):
        stage_item = stage_data[stage]
        x_train = stage_item["step"]
        for col, key in enumerate(SUB_LOSSES):
            ax = axes[row, col]
            y_train = stage_item["sub"][key]
            ax.plot(x_train, y_train, color=TRAIN_RAW, linewidth=0.8, alpha=0.45)
            sw = smooth_window(len(y_train))
            y_s = smooth(y_train, sw)
            x_s = x_train[len(x_train) - len(y_s):]
            ax.plot(x_s, y_s, color=TRAIN_SMOOTH, linewidth=2.0, label="train")

            if len(stage_item["eval_step"]) > 0:
                ax.plot(
                    stage_item["eval_step"],
                    stage_item["eval_sub"][key],
                    color=EVAL_COLOR,
                    marker="o",
                    linewidth=1.6,
                    markersize=4,
                    label="eval",
                )

            if row == 0:
                ax.set_title(FEATURE_NAMES[key], fontsize=11, color=INK_PRIMARY)
            if col == 0:
                ax.set_ylabel(STAGE_TITLES[stage], fontsize=10, color=INK_PRIMARY)
            if row == len(stage_order) - 1:
                ax.set_xlabel("step", fontsize=9, color=INK_SECONDARY)

            style_axes(ax)
            if row == 0 and col == len(SUB_LOSSES) - 1:
                ax.legend(frameon=False, fontsize=8, loc="best")

    fig.suptitle("Per-feature loss by stage", fontsize=13, color=INK_PRIMARY, y=0.995)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / "feature_loss_grid.png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_total_loss(stage_data: dict, out_dir: Path) -> None:
    stage_order = [stage for stage in ("train", "adapt_2ep", "adapt_4ep") if stage in stage_data]
    fig, axes = plt.subplots(len(stage_order), 1, figsize=(9.5, 3.0 * len(stage_order)), squeeze=False)

    for row, stage in enumerate(stage_order):
        ax = axes[row, 0]
        stage_item = stage_data[stage]
        x_train = stage_item["step"]
        y_train = stage_item["loss"]
        ax.plot(x_train, y_train, color=TRAIN_RAW, linewidth=0.8, alpha=0.45)
        sw = smooth_window(len(y_train))
        y_s = smooth(y_train, sw)
        x_s = x_train[len(x_train) - len(y_s):]
        ax.plot(x_s, y_s, color=TRAIN_SMOOTH, linewidth=2.0, label="train")

        if len(stage_item["eval_total_step"]) > 0:
            ax.plot(
                stage_item["eval_total_step"],
                stage_item["eval_total_loss"],
                color=TOTAL_EVAL_COLOR,
                linewidth=1.2,
                linestyle="--",
                marker="x",
                markersize=4,
                label="eval total (logged)",
            )
        if len(stage_item["eval_step"]) > 0:
            ax.plot(
                stage_item["eval_step"],
                stage_item["eval_loss"],
                color=EVAL_COLOR,
                linewidth=1.6,
                marker="o",
                markersize=4,
                label="eval total (recomputed)",
            )

        ax.set_ylabel(STAGE_TITLES[stage], fontsize=10, color=INK_PRIMARY)
        if row == len(stage_order) - 1:
            ax.set_xlabel("step", fontsize=9, color=INK_SECONDARY)
        style_axes(ax)
        ax.legend(frameon=False, fontsize=8, loc="best")

    fig.suptitle("Overall loss by stage", fontsize=13, color=INK_PRIMARY, y=0.995)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / "overall_loss_by_stage.png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def merge_eval_points(log_data: dict, eval_results: list[dict]) -> dict:
    merged = {}
    for stage, stage_item in log_data.items():
        merged[stage] = dict(stage_item)
        stage_points = [item for item in eval_results if item["stage"] == stage]
        stage_points.sort(key=lambda item: item["step"])
        merged[stage]["eval_step"] = np.asarray([item["step"] for item in stage_points], dtype=float)
        merged[stage]["eval_loss"] = np.asarray([item["eval_loss"] for item in stage_points], dtype=float)
        merged[stage]["eval_sub"] = {
            key: np.asarray([item[f"eval_{key}"] for item in stage_points], dtype=float) for key in SUB_LOSSES
        }
    return merged


def print_summary(stage_data: dict) -> None:
    print("\nSummary")
    print("-------")
    for stage in ("train", "adapt_2ep", "adapt_4ep"):
        if stage not in stage_data:
            continue
        item = stage_data[stage]
        print(
            f"{stage:>9}: train_points={len(item['step']):4d} "
            f"eval_total_logged={len(item['eval_total_step']):3d} "
            f"eval_feature_recomputed={len(item['eval_step']):3d}"
        )
        for key in SUB_LOSSES:
            train_first = float(item["sub"][key][0])
            train_last = float(item["sub"][key][-1])
            if len(item["eval_step"]) > 0:
                eval_last = float(item["eval_sub"][key][-1])
                print(
                    f"           {key:<14} train {train_first:8.4f} -> {train_last:8.4f}   "
                    f"eval(last) {eval_last:8.4f}"
                )
            else:
                print(f"           {key:<14} train {train_first:8.4f} -> {train_last:8.4f}   eval(last)     n/a")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path, help="Run directory containing train.log and stage subfolders")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    log_path = run_dir / "train.log"
    if not log_path.exists():
        raise FileNotFoundError(f"Missing train log: {log_path}")

    out_dir = run_dir / "feature_loss_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    log_data = parse_log(log_path)
    checkpoints = collect_eval_checkpoints(run_dir)
    device = torch.device(args.device)
    dataset_cache: dict[Path, tuple[dict, PianoCoReNodeSFTDataset]] = {}
    eval_results = []

    for ckpt in checkpoints:
        print(f"Evaluating {ckpt.stage} step={ckpt.step} from {ckpt.checkpoint_dir}")
        result = evaluate_checkpoint(ckpt, device=device, dataset_cache=dataset_cache)
        result["stage"] = ckpt.stage
        result["step"] = ckpt.step
        result["checkpoint_dir"] = str(ckpt.checkpoint_dir)
        eval_results.append(result)

    stage_data = merge_eval_points(log_data, eval_results)

    with open(out_dir / "eval_feature_points.json", "w", encoding="utf-8") as handle:
        json.dump(eval_results, handle, indent=2, ensure_ascii=False)

    plot_feature_grid(stage_data, out_dir)
    plot_total_loss(stage_data, out_dir)
    print_summary(stage_data)
    print(f"\nWrote plots to: {out_dir}")


if __name__ == "__main__":
    main()
