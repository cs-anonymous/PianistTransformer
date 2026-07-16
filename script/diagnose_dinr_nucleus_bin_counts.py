#!/usr/bin/env python3
"""Measure how many DINR bins survive nucleus filtering during a real AR window."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.infer_inr_testset import load_config, load_score_from_node
from src.train.train_inr import create_model, pedal_representation_dim
import src.model.integrated_pianoformer as ip


def summary(values):
    x = np.asarray(values, dtype=float)
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p10": float(np.percentile(x, 10)),
        "p90": float(np.percentile(x, 90)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--score-json", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--notes", type=int, default=512)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    cfg = load_config(args.config, str(args.checkpoint))
    model = create_model(cfg).to(args.device).eval()
    pitch, continuous, score_raw, _ = load_score_from_node(
        args.score_json,
        use_timing_scale_bit=cfg.get("use_timing_scale_bit", False),
        timing_control_mode=cfg.get("timing_control_mode", "log_scaled"),
        timing_log_scale=cfg.get("timing_log_scale", 50.0),
        musical_feature_mode=cfg.get("musical_feature_mode", "categorical"),
        score_note_schema=cfg.get("score_note_input_schema", "integrated"),
        task_type="epr",
        disable_musical_features=cfg.get("disable_musical_features", False),
        pedal_control_dim=pedal_representation_dim(cfg.get("pedal_representation", "binary_4")),
    )
    n = min(args.notes, len(pitch))
    counts = {"ioi_zero": [], "ioi_nonzero": [], "duration": [], "velocity": []}
    top32_mass = {key: [] for key in counts}
    p90_counts = {key: [] for key in counts}
    original = ip._nucleus_probs
    call_idx = 0

    def measured(probs, top_p=1.0):
        nonlocal call_idx
        filtered = original(probs, top_p=top_p)
        retained = int((filtered.reshape(-1, filtered.shape[-1])[0] > 0).sum().item())
        step, feature_idx = divmod(call_idx, 3)
        if feature_idx == 0:
            key = "ioi_zero" if abs(float(score_raw[step][0])) <= 1e-6 else "ioi_nonzero"
        else:
            key = ("duration", "velocity")[feature_idx - 1]
        counts[key].append(retained)
        sorted_probs = torch.sort(probs.reshape(-1, probs.shape[-1])[0], descending=True).values
        top32_mass[key].append(float(sorted_probs[:32].sum().item()))
        p90_counts[key].append(int((torch.cumsum(sorted_probs, 0) < 0.9).sum().item() + 1))
        call_idx += 1
        return filtered

    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    ip._nucleus_probs = measured
    try:
        with torch.no_grad():
            model.predict_performance_continuous(
                pitch_ids=torch.tensor(pitch[:n], dtype=torch.long, device=args.device)[None],
                continuous=torch.tensor(continuous[:n], dtype=torch.float32, device=args.device)[None],
                score_shared_raw=torch.tensor(score_raw[:n], dtype=torch.float32, device=args.device)[None],
                attention_mask=torch.ones((1, n), dtype=torch.long, device=args.device),
                sampling_strategy="sample",
            )
    finally:
        ip._nucleus_probs = original

    report = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "score_json": str(args.score_json),
        "notes": n,
        "top_p": float(getattr(model.config, "dinr_sampling_top_p", 1.0)),
        "temperature": float(getattr(model.config, "dinr_sampling_temperature", 1.0)),
        "retained_bins": {key: summary(value) for key, value in counts.items()},
        "top32_probability_mass": {key: summary(value) for key, value in top32_mass.items()},
        "bins_needed_for_p90": {key: summary(value) for key, value in p90_counts.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
