#!/usr/bin/env python3
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.diagnose_timing_drift_root_cause import collate, read_score_source_list
from src.data_process.work_manifest import build_work_manifest
from src.evaluate.eval_inr_rollout_current import build_windows, filter_manifest, load_config, selected_perfs
from src.model.integrated_pianoformer import (
    _dlm_bin_centers,
    _dlm_log_bin_probs,
    _dlm_scale,
    _split_epr_mixture_params,
    _zero_score_ioi_mask,
)
from src.train.train_inr import (
    build_epr_score_input_rows,
    create_model,
    performance_dev_velocity_pedal4_binary_rows,
)


FEATURES = (("ioi", 0), ("duration", 1), ("velocity", 2))
COVERAGES = (0.50, 0.80, 0.90, 0.95)


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--score-source-list", type=Path)
    p.add_argument("--split", default="test")
    p.add_argument("--performance-dataset", default="ASAP")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size-windows", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260715)
    return p.parse_args()


def quantiles(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    q = np.quantile(x, [0.01, 0.05, 0.50, 0.95, 0.99])
    return {
        "n": int(len(x)), "mean": float(x.mean()), "std": float(x.std()),
        "p01": float(q[0]), "p05": float(q[1]), "p50": float(q[2]),
        "p95": float(q[3]), "p99": float(q[4]),
        "min": float(x.min()), "max": float(x.max()),
    }


def make_calibration_examples(config, manifest):
    examples = []
    score_sources = []
    for score_idx, item in enumerate(manifest):
        work = json.loads(Path(item["path"]).read_text(encoding="utf-8"))
        score_sources.append(work.get("score", {}).get("score_source", str(item["path"])))
        score = work["score"]
        pitch = score["pitch"]
        score_raw = [row[:3] for row in score["score_raw"]]
        score_inputs = build_epr_score_input_rows(
            score,
            use_timing_scale_bit=config.get("use_timing_scale_bit", False),
            timing_control_mode=config.get("timing_control_mode", "floor_log"),
            log_scale=float(config.get("timing_log_scale", 50.0)),
            musical_feature_mode=config.get("musical_feature_mode", "categorical"),
            score_note_schema=config.get("score_note_input_schema", "integrated"),
        )
        windows = build_windows(len(pitch), int(config["block_notes"]), float(config["overlap_ratio"]))
        for perf_idx, perf in enumerate(selected_perfs(work, item)):
            labels = performance_dev_velocity_pedal4_binary_rows(
                perf,
                score_raw,
                epr_timing_target=config.get("epr_timing_target", "floor_log_deviation"),
                log_scale=float(config.get("timing_log_scale", 50.0)),
                pedal_binary_threshold=float(config.get("pedal_binary_threshold", 64.0)),
                legacy_dual_timing_head=bool(config.get("legacy_dual_timing_head", False)),
                pedal_representation=config.get("pedal_representation", "binary_4"),
            )
            if labels is None:
                continue
            for start, end in windows:
                examples.append({
                    "pitch": torch.tensor(pitch[start:end], dtype=torch.long),
                    "continuous": torch.tensor(score_inputs[start:end], dtype=torch.float32),
                    "score_raw": torch.tensor(score_raw[start:end], dtype=torch.float32),
                    "labels": torch.tensor(labels[start:end], dtype=torch.float32),
                    "score_idx": score_idx,
                    "perf_idx": perf_idx,
                    "note_idx": np.arange(start, end, dtype=np.int64),
                })
    return examples, score_sources


def calibration_values(config, params, labels, score_raw, mask, rng):
    zero_mask = _zero_score_ioi_mask(config, score_raw, attention_mask=mask)
    result = {}
    for feature, col in FEATURES:
        zm = zero_mask if feature == "ioi" else None
        logp = _dlm_log_bin_probs(
            config, params[f"{feature}_logits"], params[f"{feature}_loc"],
            params[f"{feature}_log_scale"], feature, zero_mask=zm,
        )
        probs = logp.exp()
        centers = _dlm_bin_centers(config, feature, logp, zero_mask=zm).expand_as(probs)
        target = labels[..., col].float()
        if feature == "velocity":
            target = target * 127.0
        # Match the exact training target bin without importing private binning twice.
        from src.model.integrated_pianoformer import _dlm_target_bins
        target_bin = _dlm_target_bins(config, target, feature, zero_mask=zm)
        target_p = probs.gather(-1, target_bin.unsqueeze(-1)).squeeze(-1)
        cdf = probs.cumsum(-1)
        cdf_before = (cdf - probs).gather(-1, target_bin.unsqueeze(-1)).squeeze(-1)
        random_u = torch.as_tensor(rng.random(target.shape), device=target.device, dtype=target.dtype)
        pit = cdf_before + random_u * target_p
        mean = (probs * centers).sum(-1)
        var = (probs * (centers - mean.unsqueeze(-1)).square()).sum(-1)
        std = var.clamp_min(1e-12).sqrt()
        z = (target - mean) / std
        nll = -target_p.clamp_min(1e-12).log()
        step = (centers[..., 1] - centers[..., 0]).abs()
        obs_cdf = (centers >= target.unsqueeze(-1)).to(dtype=probs.dtype)
        crps = ((cdf - obs_cdf).square().sum(-1) * step)
        scale = _dlm_scale(config, params[f"{feature}_log_scale"], feature, zero_mask=zm)

        values = {
            "target": target, "mean": mean, "std": std, "z": z, "pit": pit,
            "nll": nll, "crps": crps, "scale": scale.mean(-1),
            "zero_mask": zero_mask,
        }
        for coverage in COVERAGES:
            alpha = (1.0 - coverage) / 2.0
            lower = (cdf >= alpha).to(torch.int64).argmax(-1)
            upper = (cdf >= 1.0 - alpha).to(torch.int64).argmax(-1)
            covered = (target_bin >= lower) & (target_bin <= upper)
            values[f"covered_{coverage:.2f}"] = covered
        result[feature] = {key: value[mask].detach().cpu().numpy() for key, value in values.items()}
        result[feature]["raw_logits"] = params[f"{feature}_logits"][mask].detach().cpu().numpy()
        result[feature]["raw_loc"] = params[f"{feature}_loc"][mask].detach().cpu().numpy()
        result[feature]["raw_log_scale"] = params[f"{feature}_log_scale"][mask].detach().cpu().numpy()
    return result


def summarize(store):
    out = {}
    for feature, chunks in store.items():
        merged = {key: np.concatenate(values, axis=0) for key, values in chunks.items()}
        z, pit = merged["z"], merged["pit"]
        row = {
            "target": quantiles(merged["target"]),
            "pred_mean": quantiles(merged["mean"]),
            "pred_std": quantiles(merged["std"]),
            "pred_scale": quantiles(merged["scale"]),
            "residual": quantiles(merged["target"] - merged["mean"]),
            "standardized_residual": quantiles(z),
            "mean_standardized_residual": float(np.mean(z)),
            "rms_standardized_residual": float(np.sqrt(np.mean(z ** 2))),
            "mean_nll": float(np.mean(merged["nll"])),
            "mean_crps": float(np.mean(merged["crps"])),
            "pit_mean": float(np.mean(pit)),
            "pit_variance": float(np.var(pit)),
            "pit_uniform_variance": 1.0 / 12.0,
            "coverage": {},
        }
        for coverage in COVERAGES:
            row["coverage"][f"{coverage:.2f}"] = float(np.mean(merged[f"covered_{coverage:.2f}"]))
        out[feature] = row
        for key, value in merged.items():
            merged[key] = value
        np.savez_compressed(OUTPUT_DIR / f"{feature}_values.npz", **merged)
    return out


def main():
    global OUTPUT_DIR
    args = args_parser()
    OUTPUT_DIR = args.output_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    config = load_config(args.config, args.checkpoint)
    manifest = build_work_manifest(
        metadata_path=config["metadata_path"], refined_dir=config["refined_dir"],
        split=args.split, block_notes=config["block_notes"], overlap_ratio=config["overlap_ratio"],
        min_notes=config["min_notes"], skip_work_paths=config.get("skip_work_paths"),
        performance_dataset=args.performance_dataset,
    )
    manifest = filter_manifest(manifest, read_score_source_list(args.score_source_list))
    examples, score_sources = make_calibration_examples(config, manifest)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = create_model(config).to(device).eval()
    store = {feature: defaultdict(list) for feature, _ in FEATURES}
    with torch.no_grad():
        for start in range(0, len(examples), args.batch_size_windows):
            batch_examples = examples[start:start + args.batch_size_windows]
            pitch, continuous, score_raw, labels, attention = collate(
                batch_examples, int(config.get("pitch_pad_id", 128))
            )
            pitch, continuous, score_raw, labels, attention = [x.to(device) for x in (pitch, continuous, score_raw, labels, attention)]
            outputs = model(
                pitch_ids=pitch, continuous=continuous, score_shared_raw=score_raw,
                labels_continuous=labels, decoder_feedback_continuous=labels,
                attention_mask=attention, continuous_sampling_strategy="mean",
            )
            params = _split_epr_mixture_params(model.config, outputs.logits)
            batch = calibration_values(model.config, params, labels, score_raw, attention.bool(), rng)
            for feature in store:
                for key, value in batch[feature].items():
                    store[feature][key].append(value)
                store[feature]["score_idx"].append(np.concatenate([
                    np.full(len(example["note_idx"]), example["score_idx"], dtype=np.int32)
                    for example in batch_examples
                ]))
                store[feature]["perf_idx"].append(np.concatenate([
                    np.full(len(example["note_idx"]), example["perf_idx"], dtype=np.int32)
                    for example in batch_examples
                ]))
                store[feature]["note_idx"].append(np.concatenate([
                    example["note_idx"] for example in batch_examples
                ]))
            if start % (args.batch_size_windows * 20) == 0:
                print(json.dumps({"processed_windows": start + len(pitch), "total_windows": len(examples)}), flush=True)
    summary = {
        "config": str(args.config.resolve()), "checkpoint": str(args.checkpoint),
        "mode": "teacher_forced", "num_scores": len(manifest), "num_windows": len(examples),
        "features": summarize(store),
    }
    summary["score_sources"] = score_sources
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
