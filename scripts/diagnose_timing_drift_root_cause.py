import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_process.work_manifest import build_work_manifest
from src.evaluate.eval_inr_rollout_current import (
    build_windows,
    filter_manifest,
    labels_for_perf,
    load_config,
    selected_perfs,
)
from src.model.integrated_pianoformer import (
    _logistic_normal_params,
    _shared_scalar_params,
    _split_epr_mixture_params,
)
from src.train.train_inr import (
    build_epr_score_input_rows,
    create_model,
    normalize_log_timing_value,
)


FEATURES = [("ioi", 0), ("duration", 1)]


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose timing drift root causes without full rollout.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--performance-dataset", default="ASAP")
    parser.add_argument("--score-source-list", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size-windows", type=int, default=8)
    parser.add_argument("--mc-samples", type=int, default=32)
    parser.add_argument("--perturb-delta", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=20260708)
    return parser.parse_args()


def read_score_source_list(path):
    if path is None:
        return None
    values = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            values.append(line)
    return values


def quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def append_feature_values(store, feature, values):
    store[feature].extend(np.asarray(values, dtype=np.float64).reshape(-1).tolist())


def summarize_geometry(config, manifest):
    scale = float(config.get("timing_log_scale", 50.0))
    values = defaultdict(list)
    rows = []
    for item in manifest:
        work = json.loads(Path(item["path"]).read_text(encoding="utf-8"))
        score = work["score"]
        score_raw = np.asarray([row[:3] for row in score["score_raw"]], dtype=np.float64)
        score_norm = {
            "ioi": np.asarray([normalize_log_timing_value(x, scale=scale, max_time_ms=5000.0) for x in score_raw[:, 0]]),
            "duration": np.asarray([normalize_log_timing_value(x, scale=scale, max_time_ms=5000.0) for x in score_raw[:, 1]]),
        }
        for perf in selected_perfs(work, item):
            labels = np.asarray(labels_for_perf(config, perf, score_raw.tolist()), dtype=np.float64)
            for feature, col in FEATURES:
                y = labels[:, col]
                s = score_norm[feature]
                dev = y - 0.5
                perf_norm = s + dev
                append_feature_values(values, f"{feature}_target", y)
                append_feature_values(values, f"{feature}_dev", dev)
                append_feature_values(values, f"{feature}_score_norm", s)
                append_feature_values(values, f"{feature}_perf_norm", perf_norm)
                append_feature_values(values, f"{feature}_room_down", s)
                append_feature_values(values, f"{feature}_room_up", 1.0 - s)
                for delta in (0.01, 0.03, 0.05, 0.10):
                    rows.append(
                        {
                            "feature": feature,
                            "delta": delta,
                            "n": int(y.size),
                            "clip_down_if_minus_delta": float(((perf_norm - delta) < 0.0).mean()),
                            "clip_up_if_plus_delta": float(((perf_norm + delta) > 1.0).mean()),
                            "target_below_0p5": float((y < 0.5).mean()),
                            "target_above_0p5": float((y > 0.5).mean()),
                        }
                    )
    summary = {}
    for key, vals in values.items():
        summary[key] = quantiles(vals)
    return summary, rows


def make_examples(config, manifest):
    examples = []
    for item in manifest:
        work = json.loads(Path(item["path"]).read_text(encoding="utf-8"))
        score = work["score"]
        pitch = score["pitch"]
        score_shared_raw = [row[:3] for row in score["score_raw"]]
        score_inputs = build_epr_score_input_rows(
            score,
            use_timing_scale_bit=config.get("use_timing_scale_bit", False),
            timing_control_mode=config.get("timing_control_mode", "log_scaled"),
            log_scale=float(config.get("timing_log_scale", 50.0)),
            musical_feature_mode=config.get("musical_feature_mode", "categorical"),
            score_note_schema=config.get("score_note_input_schema", "integrated"),
        )
        windows = build_windows(len(pitch), int(config["block_notes"]), float(config["overlap_ratio"]))
        for perf in selected_perfs(work, item):
            labels = labels_for_perf(config, perf, score_shared_raw)
            for start, end in windows:
                examples.append(
                    {
                        "pitch": torch.tensor(pitch[start:end], dtype=torch.long),
                        "continuous": torch.tensor(score_inputs[start:end], dtype=torch.float32),
                        "score_raw": torch.tensor(score_shared_raw[start:end], dtype=torch.float32),
                        "labels": torch.tensor(labels[start:end], dtype=torch.float32),
                    }
                )
    return examples


def collate(examples, pitch_pad_id):
    pitch = pad_sequence([item["pitch"] for item in examples], batch_first=True, padding_value=pitch_pad_id)
    continuous = pad_sequence([item["continuous"] for item in examples], batch_first=True, padding_value=0.0)
    score_raw = pad_sequence([item["score_raw"] for item in examples], batch_first=True, padding_value=0.0)
    labels = pad_sequence([item["labels"] for item in examples], batch_first=True, padding_value=0.0)
    attention = torch.zeros(pitch.shape, dtype=torch.long)
    for idx, item in enumerate(examples):
        attention[idx, : item["pitch"].numel()] = 1
    return pitch, continuous, score_raw, labels, attention


def head_values(config, raw_outputs, attention_mask, mc_samples=32):
    params = _split_epr_mixture_params(config, raw_outputs)
    mask = attention_mask.bool()
    output = {}
    for feature, index in FEATURES:
        logits, raw_mu, raw_log_sigma, raw_extra = _shared_scalar_params(config, params, index)
        if raw_extra is not None:
            raise ValueError("Expected mixture_logistic_normal shared scalar heads without extra params")
        mu, sigma = _logistic_normal_params(
            raw_mu,
            raw_log_sigma,
            sigma_min=getattr(config, "logistic_normal_sigma_min", 1e-3),
            sigma_max=getattr(config, "logistic_normal_sigma_max", 10.0),
        )
        probs = torch.softmax(logits.float(), dim=-1)
        loc = torch.sigmoid(mu)
        proxy_mean = torch.sum(probs * loc, dim=-1)
        top_idx = probs.argmax(dim=-1, keepdim=True)
        mode = loc.gather(-1, top_idx).squeeze(-1)
        top_weight = probs.gather(-1, top_idx).squeeze(-1)
        derivative = loc * (1.0 - loc)
        component_var = (derivative * sigma).square()
        proxy_var = torch.sum(probs * (component_var + (loc - proxy_mean.unsqueeze(-1)).square()), dim=-1)
        proxy_std = proxy_var.clamp_min(0.0).sqrt()

        sample_sum = torch.zeros_like(proxy_mean)
        sample_sq_sum = torch.zeros_like(proxy_mean)
        for _ in range(int(mc_samples)):
            comp = torch.distributions.Categorical(probs=probs).sample().unsqueeze(-1)
            sampled_mu = mu.gather(-1, comp).squeeze(-1)
            sampled_sigma = sigma.gather(-1, comp).squeeze(-1)
            sample = torch.sigmoid(torch.distributions.Normal(sampled_mu, sampled_sigma).sample())
            sample_sum = sample_sum + sample
            sample_sq_sum = sample_sq_sum + sample.square()
        sample_mean = sample_sum / float(mc_samples)
        sample_std = (sample_sq_sum / float(mc_samples) - sample_mean.square()).clamp_min(0.0).sqrt()

        output[feature] = {
            "proxy_mean": proxy_mean[mask].detach().cpu().numpy(),
            "mode": mode[mask].detach().cpu().numpy(),
            "proxy_std": proxy_std[mask].detach().cpu().numpy(),
            "sample_mean": sample_mean[mask].detach().cpu().numpy(),
            "sample_std": sample_std[mask].detach().cpu().numpy(),
            "top_weight": top_weight[mask].detach().cpu().numpy(),
            "gt": None,
        }
    return output


def run_model_diagnostics(args, config, manifest):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    model = create_model(config)
    model.to(device)
    model.eval()

    examples = make_examples(config, manifest)
    variants = {
        "base": {},
        "ioi_plus": {0: args.perturb_delta},
        "ioi_minus": {0: -args.perturb_delta},
        "duration_plus": {1: args.perturb_delta},
        "duration_minus": {1: -args.perturb_delta},
        "both_plus": {0: args.perturb_delta, 1: args.perturb_delta},
        "both_minus": {0: -args.perturb_delta, 1: -args.perturb_delta},
    }
    accum = {
        name: {
            feature: defaultdict(list)
            for feature, _ in FEATURES
        }
        for name in variants
    }
    gt_accum = {feature: [] for feature, _ in FEATURES}

    pitch_pad_id = int(config.get("pitch_pad_id", 128))
    with torch.no_grad():
        for start in range(0, len(examples), int(args.batch_size_windows)):
            batch = examples[start : start + int(args.batch_size_windows)]
            pitch, continuous, score_raw, labels, attention = collate(batch, pitch_pad_id)
            pitch = pitch.to(device)
            continuous = continuous.to(device)
            score_raw = score_raw.to(device)
            labels = labels.to(device)
            attention = attention.to(device)
            mask = attention.bool()
            for feature, col in FEATURES:
                gt_accum[feature].extend(labels[..., col][mask].detach().cpu().numpy().tolist())
            for variant_name, perturb in variants.items():
                feedback = labels.clone()
                for col, delta in perturb.items():
                    feedback[..., col] = (feedback[..., col] + float(delta)).clamp(0.0, 1.0)
                outputs = model(
                    pitch_ids=pitch,
                    continuous=continuous,
                    score_shared_raw=score_raw,
                    labels_continuous=labels,
                    decoder_feedback_continuous=feedback,
                    attention_mask=attention,
                    continuous_sampling_strategy="mean",
                )
                values = head_values(model.config, outputs.logits, attention, mc_samples=args.mc_samples)
                for feature, data in values.items():
                    for key, array in data.items():
                        if key == "gt":
                            continue
                        accum[variant_name][feature][key].extend(np.asarray(array, dtype=np.float64).tolist())

    rows = []
    for variant_name in variants:
        for feature, _ in FEATURES:
            row = {"variant": variant_name, "feature": feature}
            gt = np.asarray(gt_accum[feature], dtype=np.float64)
            row["gt_mean"] = float(gt.mean())
            for key, vals in accum[variant_name][feature].items():
                arr = np.asarray(vals, dtype=np.float64)
                row[f"{key}_mean"] = float(arr.mean())
                row[f"{key}_std"] = float(arr.std())
            row["proxy_mean_minus_gt"] = row["proxy_mean_mean"] - row["gt_mean"]
            row["sample_mean_minus_proxy_mean"] = row["sample_mean_mean"] - row["proxy_mean_mean"]
            rows.append(row)

    base = {(row["variant"], row["feature"]): row for row in rows}
    for row in rows:
        b = base[("base", row["feature"])]
        row["proxy_mean_delta_vs_base"] = row["proxy_mean_mean"] - b["proxy_mean_mean"]
        row["sample_mean_delta_vs_base"] = row["sample_mean_mean"] - b["sample_mean_mean"]
        row["proxy_std_delta_vs_base"] = row["proxy_std_mean"] - b["proxy_std_mean"]
        row["top_weight_delta_vs_base"] = row["top_weight_mean"] - b["top_weight_mean"]
    return rows


def write_csv(path, rows):
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config, args.checkpoint)
    manifest = build_work_manifest(
        metadata_path=config["metadata_path"],
        refined_dir=config["refined_dir"],
        split=args.split,
        block_notes=config["block_notes"],
        overlap_ratio=config["overlap_ratio"],
        min_notes=config["min_notes"],
        skip_work_paths=config.get("skip_work_paths"),
        performance_dataset=args.performance_dataset,
    )
    manifest = filter_manifest(manifest, read_score_source_list(args.score_source_list))

    geometry_summary, geometry_clip_rows = summarize_geometry(config, manifest)
    model_rows = run_model_diagnostics(args, config, manifest)

    summary = {
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "performance_dataset": args.performance_dataset,
        "num_scores": len(manifest),
        "perturb_delta": args.perturb_delta,
        "mc_samples": args.mc_samples,
        "geometry": geometry_summary,
        "model_rows": model_rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(args.output_dir / "geometry_clip.csv", geometry_clip_rows)
    write_csv(args.output_dir / "model_perturb_sampling.csv", model_rows)
    print(json.dumps({k: v for k, v in summary.items() if k != "geometry"}, indent=2))


if __name__ == "__main__":
    main()
