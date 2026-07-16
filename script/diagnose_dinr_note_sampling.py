#!/usr/bin/env python3
"""Replay selected DINR AR steps and compare their categorical law with ASAP labels."""

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.inference.infer_inr_testset import (
    build_windows,
    load_config,
    load_score_from_node,
)
from src.train.train_inr import create_model
import src.model.integrated_pianoformer as ip


class ReplayComplete(RuntimeError):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()

    payload = json.loads(args.raw.read_text())
    config_path = args.run / "config.json"
    checkpoint = Path(json.loads((args.run / "summary.json").read_text())["checkpoint"])
    config = load_config(config_path, str(checkpoint))
    device = torch.device(args.device)
    model = create_model(config).to(device).eval()

    score_source = payload["score_source"]
    sidecar = Path(config["refined_dir"]) / Path(score_source).with_suffix(
        f".{config['prepared_sidecar_tag']}.pt"
    )
    if not sidecar.exists():
        sidecar = Path(config["refined_dir"]) / Path(score_source).with_suffix(".json")
    pitch, continuous, score_raw, loaded = load_score_from_node(
        sidecar,
        use_timing_scale_bit=config.get("use_timing_scale_bit", False),
        timing_control_mode=config.get("timing_control_mode", "log_scaled"),
        timing_log_scale=config.get("timing_log_scale", 50.0),
        musical_feature_mode=config.get("musical_feature_mode", "categorical"),
        score_note_schema=config.get("score_note_input_schema", "integrated"),
        task_type="epr",
        disable_musical_features=config.get("disable_musical_features", False),
    )
    windows = build_windows(len(pitch), config["block_notes"], config["overlap_ratio"])
    start, end = windows[0]
    wanted = {41: "duration", 128: "ioi"}
    captured = {}
    original = ip._categorical_sample_or_argmax
    recorded_rows = torch.tensor(payload["predicted_target7"], dtype=torch.float32)
    for wanted_step, wanted_feature in wanted.items():
        call_number = 0

        def capture(logits, strategy):
            nonlocal call_number
            result = original(logits, strategy)
            feature = ("ioi", "duration", "velocity")[call_number]
            if feature == wanted_feature:
                captured[(wanted_step, feature)] = {
                    "logits": logits.detach().float().cpu().reshape(-1).numpy(),
                }
                raise ReplayComplete
            call_number += 1
            return result

        random.seed(payload["seed"])
        torch.manual_seed(payload["seed"])
        torch.cuda.manual_seed_all(payload["seed"])
        ip._categorical_sample_or_argmax = capture
        try:
            with torch.no_grad():
                model.predict_performance_continuous(
                    pitch_ids=torch.tensor(pitch[start:end], device=device).long()[None],
                    continuous=torch.tensor(continuous[start:end], device=device).float()[None],
                    score_shared_raw=torch.tensor(score_raw[start:end], device=device).float()[None],
                    attention_mask=torch.ones((1, end - start), device=device).long(),
                    prefix_predictions=recorded_rows[:wanted_step].to(device)[None],
                    sampling_strategy="sample",
                )
        except ReplayComplete:
            pass
        finally:
            ip._categorical_sample_or_argmax = original

    coords = (np.arange(config["dinr_output_timing_bins"]) - config["dinr_output_zero_bin"]) * config["dinr_output_timing_step"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    report = {}
    for ax, (idx, feature) in zip(axes, wanted.items()):
        cap = captured[(idx, feature)]
        logits = cap["logits"]
        probs = np.exp(logits - np.max(logits)); probs /= probs.sum()
        recorded = float(payload["predicted_target7"][idx][0 if feature == "ioi" else 1])
        sampled_bin = int(np.clip(np.rint(recorded / config["dinr_output_timing_step"] + config["dinr_output_zero_bin"]), 0, len(coords) - 1))
        ax.plot(coords, probs, color="#1764ab", lw=1.8, label="model softmax(logits)")
        ax.axvline(coords[sampled_bin], color="#d62828", lw=2, label=f"sampled bin={sampled_bin}, value={coords[sampled_bin]:.3f}")
        feature_idx = 0 if feature == "ioi" else 1
        piece_gt = []
        for perf in loaded.get("performances", []):
            if perf.get("performance_dataset") != "ASAP":
                continue
            shared = perf.get("label_shared_raw") or perf.get("label_raw")
            if shared is None:
                continue
            score_value = float(score_raw[idx][feature_idx])
            perf_value = float(shared[idx][feature_idx])
            value = np.log2(1.0 + max(perf_value, 0.0) / 50.0) - np.log2(1.0 + max(score_value, 0.0) / 50.0)
            piece_gt.append({"coordinate": float(value), "raw_ms": perf_value})
        for gt_idx, item in enumerate(piece_gt):
            ax.axvline(item["coordinate"], color="#2a9d8f", lw=1.4, ls="--", alpha=.9,
                       label="GT performance 1/2" if gt_idx == 0 else "GT performance 2/2")
            ax.scatter([item["coordinate"]], [probs.max() * (0.94 - 0.08 * gt_idx)],
                       color="#2a9d8f", marker="D", s=42, zorder=5)
        ax.set_title(f"note {idx}: MIDI pitch {pitch[idx]} | {feature} | recorded={recorded:.3f}")
        ax.set_ylabel("probability mass")
        ax.grid(alpha=.2); ax.legend()
        rank = int((probs > probs[sampled_bin]).sum() + 1)
        report[f"{idx}_{feature}"] = {
            "sampled_bin": sampled_bin, "sampled_coordinate": float(coords[sampled_bin]),
            "recorded_coordinate": recorded, "sample_probability": float(probs[sampled_bin]),
            "probability_rank": rank, "argmax_bin": int(probs.argmax()),
            "argmax_coordinate": float(coords[probs.argmax()]),
            "tail_mass_at_or_beyond_sample": float(probs[coords <= coords[sampled_bin]].sum()) if coords[sampled_bin] < 0 else float(probs[coords >= coords[sampled_bin]].sum()),
            "piece_gt": piece_gt,
        }
    axes[-1].set_xlabel("floor-log deviation coordinate")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
