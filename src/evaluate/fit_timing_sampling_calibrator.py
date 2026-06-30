import argparse
import json
import math
import multiprocessing as mp
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from scipy.optimize import differential_evolution, minimize
from scipy.stats import wasserstein_distance
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.inference.infer_inr_testset import (
    build_windows,
    load_model,
    load_score_from_node,
)
from src.train.train_inr import build_work_manifest, infer_input_feature_mode


def parse_args():
    parser = argparse.ArgumentParser(description="Fit timing-only sampling calibrator on ASAP train.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--method", choices=["bias_correction", "calibrated_residual"], required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-works", type=int, default=None)
    parser.add_argument("--max-windows-per-work", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--batch-size-windows", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-gt-per-work", type=int, default=4)
    parser.add_argument("--cache-npz", type=Path, default=None)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--bias-bounds", type=str, default="-0.30,0.15")
    parser.add_argument("--alpha-pos-bounds", type=str, default="0.05,2.00")
    parser.add_argument("--alpha-neg-bounds", type=str, default="0.05,2.00")
    parser.add_argument("--de-maxiter", type=int, default=80)
    parser.add_argument("--de-popsize", type=int, default=12)
    parser.add_argument("--pooled-weight", type=float, default=0.0)
    return parser.parse_args()


def load_config(path: Path, checkpoint: str):
    config = json.loads(path.read_text(encoding="utf-8"))
    config["input_feature_mode"] = infer_input_feature_mode(config)
    config["resume_path"] = checkpoint
    return config


def parse_bounds(text):
    values = [float(item) for item in str(text).split(",") if item.strip()]
    if len(values) != 2:
        raise ValueError(f"Expected two comma-separated bounds, got {text!r}")
    low, high = values
    if high <= low:
        raise ValueError(f"Invalid bounds {text!r}")
    return low, high


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


def sample_work_windows(work, max_windows_per_work, seed):
    windows = list(work["windows"])
    if max_windows_per_work is not None and max_windows_per_work > 0 and len(windows) > max_windows_per_work:
        rng = random.Random(f"{seed}:{work['score_source']}")
        windows = sorted(rng.sample(windows, max_windows_per_work))
    return windows


def predict_window_batch(model, device, config, pitch, continuous, score_shared_raw, windows, strategy):
    outputs = []
    for batch_start in range(0, len(windows), int(config.get("_calibration_batch_size_windows", 8))):
        batch_windows = windows[batch_start : batch_start + int(config.get("_calibration_batch_size_windows", 8))]
        pitch_tensors = []
        continuous_tensors = []
        score_shared_raw_tensors = []
        lengths = []
        for start, end in batch_windows:
            pitch_tensors.append(torch.tensor(pitch[start:end], dtype=torch.long))
            continuous_tensors.append(torch.tensor(continuous[start:end], dtype=torch.float32))
            score_shared_raw_tensors.append(torch.tensor(score_shared_raw[start:end], dtype=torch.float32))
            lengths.append(end - start)

        pitch_ids = pad_sequence(pitch_tensors, batch_first=True, padding_value=config["pitch_pad_id"]).to(device)
        continuous_tensor = pad_sequence(continuous_tensors, batch_first=True, padding_value=0.0).to(device)
        score_shared_raw_tensor = pad_sequence(score_shared_raw_tensors, batch_first=True, padding_value=0.0).to(device)
        attention_mask = (pitch_ids != config["pitch_pad_id"]).long()
        with torch.no_grad():
            pred = model.predict_performance_continuous(
                pitch_ids=pitch_ids,
                continuous=continuous_tensor,
                score_shared_raw=score_shared_raw_tensor,
                attention_mask=attention_mask,
                sampling_strategy=strategy,
            ).detach().float().cpu()
        for idx, length in enumerate(lengths):
            outputs.append(pred[idx, :length].numpy())
    return outputs


def predict_work_windows(model, device, config, work, args):
    pitch, continuous, score_shared_raw, _ = load_score_from_node(
        Path(work["path"]),
        use_timing_scale_bit=config.get("use_timing_scale_bit", True),
        timing_control_mode=config.get("timing_control_mode"),
        timing_log_scale=config.get("timing_log_scale", 50.0),
        task_type="epr",
    )
    windows = sample_work_windows(work, args.max_windows_per_work, args.seed)
    if not windows:
        return None
    mean_preds = predict_window_batch(model, device, config, pitch, continuous, score_shared_raw, windows, "mean")
    sample_preds = predict_window_batch(model, device, config, pitch, continuous, score_shared_raw, windows, "sample")
    score_raw = np.asarray(score_shared_raw, dtype=np.float64)
    gt_rows = load_asap_train_ground_truth(work["path"], args.max_gt_per_work)
    if not gt_rows:
        return None
    row = {
        "work_id": work["score_source"],
        "score_ioi": [],
        "score_duration": [],
        "mean_delta": [],
        "sample_delta": [],
        "gt_ioi": [],
        "gt_duration": [],
        "num_windows": len(windows),
    }
    for (start, end), mean_pred, sample_pred in zip(windows, mean_preds, sample_preds):
        n = min(end - start, mean_pred.shape[0], sample_pred.shape[0])
        if n <= 0:
            continue
        note_slice = slice(start, start + n)
        row["score_ioi"].append(score_raw[note_slice, 0])
        row["score_duration"].append(score_raw[note_slice, 1])
        row["mean_delta"].append(np.clip(mean_pred[:n, :2], 0.0, 1.0) - 0.5)
        row["sample_delta"].append(np.clip(sample_pred[:n, :2], 0.0, 1.0) - 0.5)
        for ref_idx, gt in enumerate(gt_rows):
            if len(row["gt_ioi"]) <= ref_idx:
                row["gt_ioi"].append([])
                row["gt_duration"].append([])
            m = min(start + n, len(gt)) - start
            if m > 0:
                row["gt_ioi"][ref_idx].append(gt[start : start + m, 0])
                row["gt_duration"][ref_idx].append(gt[start : start + m, 1])
    if not row["score_ioi"]:
        return None
    row["score_ioi"] = np.concatenate(row["score_ioi"])
    row["score_duration"] = np.concatenate(row["score_duration"])
    row["mean_delta"] = np.concatenate(row["mean_delta"], axis=0)
    row["sample_delta"] = np.concatenate(row["sample_delta"], axis=0)
    row["gt_ioi"] = [np.concatenate(chunks) for chunks in row["gt_ioi"] if chunks]
    row["gt_duration"] = [np.concatenate(chunks) for chunks in row["gt_duration"] if chunks]
    return row


def worker_loop(worker_idx, args, config, jobs, result_queue):
    random.seed(args.seed + worker_idx)
    np.random.seed(args.seed + worker_idx)
    torch.manual_seed(args.seed + worker_idx)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + worker_idx)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(config, device)
    for job_idx, work in jobs:
        try:
            row = predict_work_windows(model, device, config, work, args)
            result_queue.put((job_idx, row, None))
        except Exception as exc:  # noqa: BLE001
            result_queue.put((job_idx, None, repr(exc)))


def collect_rows(config, manifest, args):
    config = dict(config)
    config["_calibration_batch_size_windows"] = args.batch_size_windows
    if args.num_workers <= 1:
        device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
        model = load_model(config, device)
        rows = []
        for work in tqdm(manifest, desc="calibration window predictions"):
            row = predict_work_windows(model, device, config, work, args)
            if row is not None:
                rows.append(row)
        if not rows:
            raise RuntimeError("No ASAP train calibration rows collected")
        return rows

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    indexed = list(enumerate(manifest))
    shards = [indexed[idx::args.num_workers] for idx in range(args.num_workers)]
    workers = [
        ctx.Process(target=worker_loop, args=(idx, args, config, shard, result_queue))
        for idx, shard in enumerate(shards)
        if shard
    ]
    for worker in workers:
        worker.start()

    rows = []
    with tqdm(total=len(indexed), desc="calibration window prediction pool") as progress:
        for _ in range(len(indexed)):
            job_idx, row, error = result_queue.get()
            if error is not None:
                for worker in workers:
                    worker.terminate()
                raise RuntimeError(f"Calibration worker failed on job {job_idx}: {error}")
            if row is not None:
                rows.append(row)
            progress.update(1)

    for worker in workers:
        worker.join()
        if worker.exitcode != 0:
            raise RuntimeError(f"Calibration worker {worker.pid} exited with code {worker.exitcode}")

    if not rows:
        raise RuntimeError("No ASAP train calibration rows collected")
    return sorted(rows, key=lambda item: item["work_id"])


def save_rows_npz(path, rows, metadata):
    payload = {"metadata": np.asarray(json.dumps(metadata, ensure_ascii=False))}
    for idx, row in enumerate(rows):
        prefix = f"row_{idx}"
        payload[f"{prefix}_work_id"] = np.asarray(row["work_id"])
        payload[f"{prefix}_score_ioi"] = np.asarray(row["score_ioi"], dtype=np.float64)
        payload[f"{prefix}_score_duration"] = np.asarray(row["score_duration"], dtype=np.float64)
        payload[f"{prefix}_mean_delta"] = np.asarray(row["mean_delta"], dtype=np.float32)
        payload[f"{prefix}_sample_delta"] = np.asarray(row["sample_delta"], dtype=np.float32)
        payload[f"{prefix}_gt_ioi"] = np.asarray(row["gt_ioi"], dtype=object)
        payload[f"{prefix}_gt_duration"] = np.asarray(row["gt_duration"], dtype=object)
        payload[f"{prefix}_num_windows"] = np.asarray(int(row.get("num_windows", 0)))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def load_rows_npz(path):
    data = np.load(path, allow_pickle=True)
    metadata = json.loads(str(data["metadata"].item()))
    rows = []
    idx = 0
    while f"row_{idx}_work_id" in data:
        prefix = f"row_{idx}"
        rows.append(
            {
                "work_id": str(data[f"{prefix}_work_id"].item()),
                "score_ioi": data[f"{prefix}_score_ioi"].astype(np.float64),
                "score_duration": data[f"{prefix}_score_duration"].astype(np.float64),
                "mean_delta": data[f"{prefix}_mean_delta"].astype(np.float64),
                "sample_delta": data[f"{prefix}_sample_delta"].astype(np.float64),
                "gt_ioi": [np.asarray(item, dtype=np.float64) for item in data[f"{prefix}_gt_ioi"]],
                "gt_duration": [np.asarray(item, dtype=np.float64) for item in data[f"{prefix}_gt_duration"]],
                "num_windows": int(data[f"{prefix}_num_windows"].item()),
            }
        )
        idx += 1
    return rows, metadata


def predict_ms(score_ms, mean_delta, sample_delta, feature_idx, bias, alpha_pos, alpha_neg, scale):
    residual = sample_delta[:, feature_idx] - mean_delta[:, feature_idx]
    residual = np.where(residual >= 0.0, residual * alpha_pos, residual * alpha_neg)
    delta = np.clip(mean_delta[:, feature_idx] + bias + residual, -0.5, 0.5)
    return inv_log_code(log_code(score_ms, scale=scale) + delta, scale=scale)


def finite_wasserstein(pred, target):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    pred = pred[np.isfinite(pred)]
    target = target[np.isfinite(target)]
    if len(pred) == 0 or len(target) == 0:
        return float("nan")
    return float(wasserstein_distance(pred, target))


def feature_arrays(row, feature):
    if feature == "ioi":
        return row["score_ioi"], row["gt_ioi"], 0
    if feature == "duration":
        return row["score_duration"], row["gt_duration"], 1
    raise ValueError(f"Unsupported feature {feature}")


def per_piece_wass(feature, rows, bias, alpha_pos, alpha_neg, scale):
    metrics = []
    for row in rows:
        score, gt_rows, idx = feature_arrays(row, feature)
        pred = predict_ms(score, row["mean_delta"], row["sample_delta"], idx, bias, alpha_pos, alpha_neg, scale)
        gt = np.concatenate(gt_rows) if gt_rows else np.asarray([], dtype=np.float64)
        metric = finite_wasserstein(pred, gt)
        if math.isfinite(metric):
            metrics.append(metric)
    return float(np.mean(metrics)) if metrics else float("inf")


def global_pooled_wass(feature, rows, bias, alpha_pos, alpha_neg, scale):
    pred_chunks = []
    gt_chunks = []
    for row in rows:
        score, gt_rows, idx = feature_arrays(row, feature)
        pred_chunks.append(
            predict_ms(score, row["mean_delta"], row["sample_delta"], idx, bias, alpha_pos, alpha_neg, scale)
        )
        gt_chunks.extend(gt_rows)
    if not pred_chunks or not gt_chunks:
        return float("inf")
    return finite_wasserstein(np.concatenate(pred_chunks), np.concatenate(gt_chunks))


def fit_one(feature, rows, args, scale):
    if args.method == "bias_correction":
        bounds = [parse_bounds(args.bias_bounds)]

        def unpack(params):
            return float(params[0]), 1.0, 1.0
    else:
        bounds = [
            parse_bounds(args.bias_bounds),
            parse_bounds(args.alpha_pos_bounds),
            parse_bounds(args.alpha_neg_bounds),
        ]

        def unpack(params):
            return float(params[0]), float(params[1]), float(params[2])

    pooled_weight = max(0.0, float(args.pooled_weight))

    def objective(params):
        bias, alpha_pos, alpha_neg = unpack(params)
        pp = per_piece_wass(feature, rows, bias, alpha_pos, alpha_neg, scale)
        if pooled_weight <= 0.0:
            return pp
        pooled = global_pooled_wass(feature, rows, bias, alpha_pos, alpha_neg, scale)
        return pp + pooled_weight * pooled

    result = differential_evolution(
        objective,
        bounds=bounds,
        seed=args.seed,
        maxiter=args.de_maxiter,
        popsize=args.de_popsize,
        polish=False,
        tol=1e-4,
        atol=1e-4,
    )
    polished = minimize(
        objective,
        result.x,
        method="Powell",
        bounds=bounds,
        options={"maxiter": 200, "xtol": 1e-4, "ftol": 1e-4},
    )
    params = polished.x if polished.success and polished.fun <= result.fun else result.x
    bias, alpha_pos, alpha_neg = unpack(params)
    pp = per_piece_wass(feature, rows, bias, alpha_pos, alpha_neg, scale)
    pooled = global_pooled_wass(feature, rows, bias, alpha_pos, alpha_neg, scale)
    return {
        "bias": float(bias),
        "alpha_pos": float(alpha_pos),
        "alpha_neg": float(alpha_neg),
        "objective_wass_ms": float(pp + pooled_weight * pooled),
        "pp_wass_ms": float(pp),
        "global_pooled_wass_ms": float(pooled),
        "optimizer": "differential_evolution+powell",
    }


def main():
    args = parse_args()
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = load_config(args.config, args.checkpoint)
    cache_metadata = None
    if args.cache_npz and args.cache_npz.exists() and not args.refresh_cache:
        rows, cache_metadata = load_rows_npz(args.cache_npz)
        print(f"Loaded calibration cache: {args.cache_npz} ({len(rows)} works)")
    else:
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
        rows = collect_rows(config, manifest, args)
        cache_metadata = {
            "config": str(args.config.resolve()),
            "checkpoint": args.checkpoint,
            "fit_dataset": "ASAP train",
            "num_works": len(rows),
            "max_works": args.max_works,
            "max_windows_per_work": args.max_windows_per_work,
            "max_gt_per_work": args.max_gt_per_work,
            "seed": args.seed,
            "timing_space": "log_deviation_delta",
        }
        if args.cache_npz:
            save_rows_npz(args.cache_npz, rows, cache_metadata)
            print(f"Saved calibration cache: {args.cache_npz} ({len(rows)} works)")
    scale = float(config.get("timing_log_scale", 50.0))
    output = {
        "method": args.method,
        "fit_dataset": "ASAP train",
        "cache_npz": str(args.cache_npz.resolve()) if args.cache_npz else None,
        "cache_metadata": cache_metadata,
        "num_calibration_works": len(rows),
        "num_calibration_windows": int(sum(int(row.get("num_windows", 0)) for row in rows)),
        "max_works": args.max_works,
        "max_windows_per_work": args.max_windows_per_work,
        "max_gt_per_work": args.max_gt_per_work,
        "timing_space": "log_deviation_delta",
        "metric": "per_piece_wasserstein_ms",
        "pooled_weight": float(args.pooled_weight),
        "features": {
            "ioi": fit_one("ioi", rows, args, scale),
            "duration": fit_one("duration", rows, args, scale),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
