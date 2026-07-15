#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.inference.infer_inr_testset import load_config, load_model, load_score_from_node
from src.model.integrated_pianoformer import (
    _dlm_bin_centers,
    _dlm_log_bin_probs,
    _split_epr_mixture_params,
    _zero_score_ioi_mask,
)


CONFIG_PATH = ROOT / "results/floorlog_distribution_ablation_single_gpu_20260714/slot5-nomus-k1/config.json"
CHECKPOINT = ROOT / (
    "results/floorlog_distribution_ablation_single_gpu_20260714/slot5-nomus-k1/"
    "training/floorlog_distribution_ablation_slot5_nomus_k1/checkpoint-1680"
)
WORK_PATH = ROOT / (
    "PianoCoRe/processed/Glinka,_Mikhail/A_Farewell_to_Saint_Petersburg/"
    "10._The_Lark/score_MS_refined.json"
)
SAMPLE_RAW = ROOT / (
    "results/floorlog_distribution_ablation_single_gpu_20260714/slot5-nomus-k1/"
    "sampling/raw_outputs/"
    "Glinka,_Mikhail__A_Farewell_to_Saint_Petersburg__10._The_Lark__score_MS_refined__sample_000.json"
)
DET_RAW = ROOT / (
    "results/floorlog_distribution_ablation_single_gpu_20260714/slot5-nomus-k1/"
    "deterministic/raw_outputs/"
    "Glinka,_Mikhail__A_Farewell_to_Saint_Petersburg__10._The_Lark__score_MS_refined__sample_000.json"
)
OUT_DIR = ROOT / "results/analysis/glinka_lark_k1_predicted_distributions_20260714"

POINTS = [
    {"label": "0.00s F4 DUR", "idx": 0, "feature": "duration", "raw_col": 1, "score_col": 1},
    {"label": "21.60s G#4 velocity", "idx": 42, "feature": "velocity", "raw_col": 2, "score_col": 2},
    {"label": "47.22s C4 IOI", "idx": 88, "feature": "ioi", "raw_col": 0, "score_col": 0},
]


def floor_log_reconstruct(dev, score_ms):
    return float(np.exp(float(dev) + np.log(max(float(score_ms), 1.0))))


def raw_axis(feature, centers, score_ms):
    values = centers.detach().cpu().numpy().astype(float)
    if feature in {"ioi", "duration"}:
        return np.array([floor_log_reconstruct(v, score_ms) for v in values], dtype=float)
    return values


def perf_gt_values(work, idx, raw_col):
    vals = []
    for perf in work.get("performances", []):
        rows = perf.get("performance_raw") or perf.get("raw") or []
        if idx < len(rows):
            vals.append(float(rows[idx][raw_col]))
    return vals


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else ("cuda:0" if torch.cuda.is_available() else "cpu"))

    config = load_config(CONFIG_PATH, str(CHECKPOINT))
    model = load_model(config, device)
    cfg = SimpleNamespace(**config)

    pitch, continuous, score_shared_raw, work = load_score_from_node(
        WORK_PATH,
        use_timing_scale_bit=config.get("use_timing_scale_bit", False),
        timing_control_mode=config.get("timing_control_mode", "floor_log"),
        timing_log_scale=config.get("timing_log_scale", 50.0),
        musical_feature_mode=config.get("musical_feature_mode", "categorical"),
        score_note_schema=config.get("score_note_schema", "integrated"),
        task_type=config.get("task_type", "epr"),
        disable_musical_features=config.get("disable_musical_features", False),
        pedal_control_dim=config.get("pedal_control_dim", 4),
    )

    block_notes = int(config.get("block_notes", 512))
    pitch_t = torch.tensor([pitch[:block_notes]], dtype=torch.long, device=device)
    continuous_t = torch.tensor([continuous[:block_notes]], dtype=torch.float32, device=device)
    score_raw_t = torch.tensor([score_shared_raw[:block_notes]], dtype=torch.float32, device=device)
    attention_mask = torch.ones_like(pitch_t, dtype=torch.long, device=device)

    captured_raw = []
    original_decoder_forward = model.continuous_decoder.forward

    def capture_decoder_forward(*args, **kwargs):
        out = original_decoder_forward(*args, **kwargs)
        if out.shape[1] == 1:
            captured_raw.append(out.detach().float())
        return out

    model.continuous_decoder.forward = capture_decoder_forward
    try:
        with torch.no_grad():
            model(
                pitch_ids=pitch_t,
                continuous=continuous_t,
                score_shared_raw=score_raw_t,
                attention_mask=attention_mask,
                continuous_sampling_strategy="mean",
            )
    finally:
        model.continuous_decoder.forward = original_decoder_forward
    raw_outputs = torch.cat(captured_raw, dim=1)
    if raw_outputs.shape[1] < max(p["idx"] for p in POINTS) + 1:
        raise RuntimeError(f"Captured only {raw_outputs.shape[1]} raw autoregressive steps")
    params = _split_epr_mixture_params(cfg, raw_outputs)
    zero_mask = _zero_score_ioi_mask(cfg, score_raw_t, attention_mask=attention_mask)

    sample = json.loads(SAMPLE_RAW.read_text(encoding="utf-8"))
    det = json.loads(DET_RAW.read_text(encoding="utf-8"))

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)
    result = {
        "config": str(CONFIG_PATH.relative_to(ROOT)),
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "work": str(WORK_PATH.relative_to(ROOT)),
        "device": str(device),
        "points": [],
    }

    for ax, point in zip(axes, POINTS):
        idx = point["idx"]
        feature = point["feature"]
        score_ms = float(score_shared_raw[idx][point["score_col"]])
        zm = zero_mask[:, :, None] if feature == "ioi" else None
        logp = _dlm_log_bin_probs(
            cfg,
            params[f"{feature}_logits"][:, idx : idx + 1],
            params[f"{feature}_loc"][:, idx : idx + 1],
            params[f"{feature}_log_scale"][:, idx : idx + 1],
            feature,
            zero_mask=zm[:, idx : idx + 1] if zm is not None else None,
        )[0, 0].squeeze()
        probs = torch.softmax(logp, dim=-1)
        centers = _dlm_bin_centers(
            cfg,
            feature,
            logp.view(1, 1, -1),
            zero_mask=(zero_mask[:, idx : idx + 1] if feature == "ioi" else None),
        )[0, 0]
        x = raw_axis(feature, centers, score_ms)
        y = probs.detach().cpu().numpy()

        raw_col = point["raw_col"]
        sample_v = float(sample["reconstructed_raw7"][idx][raw_col])
        det_v = float(det["reconstructed_raw7"][idx][raw_col])
        gt_vals = perf_gt_values(work, idx, raw_col)
        score_v = score_ms
        mean_v = float(np.sum(x * y))

        ax.plot(x, y, color="#2266aa", linewidth=1.8, label="K1 predicted bin prob")
        ax.fill_between(x, 0, y, color="#2266aa", alpha=0.18)
        ax.axvline(score_v, color="#777777", linestyle=":", linewidth=1.2, label="score")
        ax.axvline(det_v, color="#cc7722", linestyle="--", linewidth=1.4, label="det")
        ax.axvline(sample_v, color="#aa2222", linestyle="-.", linewidth=1.4, label="sample")
        ax.axvline(mean_v, color="#222222", linestyle="-", linewidth=1.0, label="prob mean")
        if gt_vals:
            ax.scatter(gt_vals, np.zeros(len(gt_vals)), color="#2a9d55", s=38, zorder=4, label="ASAP GT")
        ax.set_title(f"{point['label']}  | idx={idx}, pitch={pitch[idx]}")
        ax.set_xlabel("raw ms" if feature in {"ioi", "duration"} else "MIDI velocity")
        ax.set_ylabel("probability")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)

        result["points"].append(
            {
                **point,
                "pitch": int(pitch[idx]),
                "score_raw": score_shared_raw[idx],
                "score_value": score_v,
                "sample_value": sample_v,
                "deterministic_value": det_v,
                "probability_mean": mean_v,
                "gt_values": gt_vals,
                "x": x.tolist(),
                "prob": y.tolist(),
            }
        )

    fig.suptitle("K1 predicted DLM distribution shapes: Glinka - The Lark", fontsize=13)
    png = OUT_DIR / "k1_predicted_distribution_shapes.png"
    fig.savefig(png, dpi=180)
    (OUT_DIR / "k1_predicted_distribution_values.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(png)


if __name__ == "__main__":
    main()
