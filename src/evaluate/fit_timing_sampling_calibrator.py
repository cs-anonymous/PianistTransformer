import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wasserstein_distance
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.inference.infer_inr_testset import (
    build_windows,
    continuation_window_predictions,
    load_model,
    load_score_from_node,
)
from src.model.integrated_pianoformer import _target5_to_raw7
from src.train.train_inr import build_work_manifest, infer_input_feature_mode


def parse_args():
    parser = argparse.ArgumentParser(description="Fit timing-only sampling calibrator on ASAP train.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--method", choices=["bias_correction", "calibrated_residual"], required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-works", type=int, default=32)
    parser.add_argument("--batch-size-windows", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-gt-per-work", type=int, default=4)
    parser.add_argument("--b-grid", type=str, default="-0.20,-0.16,-0.13,-0.10,-0.08,-0.065,-0.05,-0.03,0.0,0.03")
    parser.add_argument("--alpha-pos-grid", type=str, default="0.45,0.55,0.65,0.8,1.0")
    parser.add_argument("--alpha-neg-grid", type=str, default="0.5,0.75,1.0,1.25,1.5")
    return parser.parse_args()


def load_config(path: Path, checkpoint: str):
    config = json.loads(path.read_text(encoding="utf-8"))
    config["input_feature_mode"] = infer_input_feature_mode(config)
    config["resume_path"] = checkpoint
    return config


def parse_grid(text):
    return [float(item) for item in str(text).split(",") if item.strip()]


def log_code(values, scale=50.0, max_time_ms=5000.0):
    values = np.clip(np.asarray(values, dtype=np.float64), 0.0, float(max_time_ms))
    return np.log1p(values / float(scale)) / math.log1p(float(max_time_ms) / float(scale))


def inv_log_code(values, scale=50.0, max_time_ms=5000.0):
    values = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    return float(scale) * np.expm1(values * math.log1p(float(max_time_ms) / float(scale)))


def load_asap_train_ground_truth(work_path, max_gt_per_work):
    work = json.loads(Path(work_path).read_text(encoding="utf-8"))
    rows = []
    for perf in work.get("performances", []):
        if perf.get("performance_dataset") != "ASAP":
            continue
        shared = perf.get("label_shared_raw")
        if shared is None:
            continue
        rows.append(np.asarray(shared, dtype=np.float64))
        if max_gt_per_work and len(rows) >= max_gt_per_work:
            break
    return rows


def predict_work(model, device, config, work, args):
    pitch, continuous, score_shared_raw, _ = load_score_from_node(
        Path(work["path"]),
        use_timing_scale_bit=config.get("use_timing_scale_bit", True),
        timing_control_mode=config.get("timing_control_mode"),
        timing_log_scale=config.get("timing_log_scale", 50.0),
        task_type="epr",
    )
    windows = build_windows(len(pitch), config["block_notes"], config["overlap_ratio"])
    common = dict(
        model=model,
        pitch=pitch,
        continuous=continuous,
        score_shared_raw=score_shared_raw,
        windows=windows,
        pitch_pad_id=config["pitch_pad_id"],
        device=device,
        drop_ratio=0.0,
    )
    mean_pred = continuation_window_predictions(**common, sampling_strategy="mean")
    sample_pred = continuation_window_predictions(**common, sampling_strategy="sample")
    score_raw = np.asarray(score_shared_raw, dtype=np.float64)
    gt_rows = load_asap_train_ground_truth(work["path"], args.max_gt_per_work)
    return score_raw, mean_pred.numpy(), sample_pred.numpy(), gt_rows


def collect_arrays(model, device, config, manifest, args):
    score_ioi = []
    score_duration = []
    mean_delta = []
    sample_delta = []
    gt_ioi = []
    gt_duration = []

    for work in tqdm(manifest, desc="calibration predictions"):
        score_raw, mean_pred, sample_pred, gt_rows = predict_work(model, device, config, work, args)
        if not gt_rows:
            continue
        n = min(len(score_raw), mean_pred.shape[0], sample_pred.shape[0])
        score_ioi.append(score_raw[:n, 0])
        score_duration.append(score_raw[:n, 1])
        mean_delta.append(np.clip(mean_pred[:n, :2], 0.0, 1.0) - 0.5)
        sample_delta.append(np.clip(sample_pred[:n, :2], 0.0, 1.0) - 0.5)
        for gt in gt_rows:
            m = min(n, len(gt))
            gt_ioi.append(gt[:m, 0])
            gt_duration.append(gt[:m, 1])

    if not score_ioi:
        raise RuntimeError("No ASAP train calibration rows collected")
    return {
        "score_ioi": np.concatenate(score_ioi),
        "score_duration": np.concatenate(score_duration),
        "mean_delta": np.concatenate(mean_delta, axis=0),
        "sample_delta": np.concatenate(sample_delta, axis=0),
        "gt_ioi": np.concatenate(gt_ioi),
        "gt_duration": np.concatenate(gt_duration),
    }


def predict_ms(score_ms, mean_delta, sample_delta, feature_idx, bias, alpha_pos, alpha_neg, scale):
    residual = sample_delta[:, feature_idx] - mean_delta[:, feature_idx]
    residual = np.where(residual >= 0.0, residual * alpha_pos, residual * alpha_neg)
    delta = np.clip(mean_delta[:, feature_idx] + bias + residual, -0.5, 0.5)
    return inv_log_code(log_code(score_ms, scale=scale) + delta, scale=scale)


def fit_one(feature, arrays, args, scale):
    if feature == "ioi":
        score = arrays["score_ioi"]
        gt = arrays["gt_ioi"]
        idx = 0
    else:
        score = arrays["score_duration"]
        gt = arrays["gt_duration"]
        idx = 1

    b_grid = parse_grid(args.b_grid)
    if args.method == "bias_correction":
        alpha_pos_grid = [1.0]
        alpha_neg_grid = [1.0]
    else:
        alpha_pos_grid = parse_grid(args.alpha_pos_grid)
        alpha_neg_grid = parse_grid(args.alpha_neg_grid)

    best = None
    for bias in b_grid:
        for alpha_pos in alpha_pos_grid:
            for alpha_neg in alpha_neg_grid:
                pred = predict_ms(
                    score,
                    arrays["mean_delta"],
                    arrays["sample_delta"],
                    idx,
                    bias,
                    alpha_pos,
                    alpha_neg,
                    scale,
                )
                metric = float(wasserstein_distance(pred[np.isfinite(pred)], gt[np.isfinite(gt)]))
                row = {
                    "bias": float(bias),
                    "alpha_pos": float(alpha_pos),
                    "alpha_neg": float(alpha_neg),
                    "pp_wass_ms": metric,
                }
                if best is None or metric < best["pp_wass_ms"]:
                    best = row
    return best


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = load_config(args.config, args.checkpoint)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(config, device)
    manifest = build_work_manifest(
        metadata_path=config["metadata_path"],
        refined_dir=config["refined_dir"],
        split="train",
        block_notes=config["block_notes"],
        overlap_ratio=config["overlap_ratio"],
        min_notes=config["min_notes"],
        max_works=args.max_works,
        performance_dataset="ASAP",
        selection_seed=args.seed,
    )
    arrays = collect_arrays(model, device, config, manifest, args)
    scale = float(config.get("timing_log_scale", 50.0))
    output = {
        "method": args.method,
        "fit_dataset": "ASAP train",
        "max_works": args.max_works,
        "max_gt_per_work": args.max_gt_per_work,
        "timing_space": "log_deviation_delta",
        "metric": "pooled_pp_wasserstein_ms",
        "features": {
            "ioi": fit_one("ioi", arrays, args, scale),
            "duration": fit_one("duration", arrays, args, scale),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
