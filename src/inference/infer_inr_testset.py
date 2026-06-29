import argparse
import json
import math
import multiprocessing as mp
import random
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.train.train_inr import (
    build_work_manifest,
    build_csr_performance_input_rows,
    build_epr_score_input_rows,
    build_score_musical_rows,
    create_model,
    infer_input_feature_mode,
)
from src.model.integrated_pianoformer import _target5_to_raw7
from src.utils.inr_midi import note_features_to_midi


def parse_args():
    parser = argparse.ArgumentParser(description="Run INR inference on PianoCoRe test scores.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--protocol", choices=["deterministic", "sampling"], default="deterministic")
    parser.add_argument(
        "--deterministic-strategy",
        choices=["mean", "greedy"],
        default="mean",
        help="How deterministic inference materializes probabilistic outputs.",
    )
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-works", type=int, default=None)
    parser.add_argument("--batch-size-windows", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-gt-per-score", type=int, default=None)
    parser.add_argument("--performance-dataset", type=str, default=None,
                        help="Optional performance_dataset filter, e.g. ASAP. Restricts scores and GT refs.")
    parser.add_argument("--exclude-performance-dataset", type=str, default=None,
                        help="Optional performance_dataset exclusion, e.g. ASAP for non-ASAP evaluation.")
    parser.add_argument("--merge-mode", choices=["continuation", "average"], default="continuation")
    parser.add_argument("--continuation-drop-ratio", type=float, default=0.0)
    return parser.parse_args()


def resolve_sampling_strategy(args):
    if args.protocol == "sampling":
        return "sample"
    return str(args.deterministic_strategy).lower()


def select_device(device_arg):
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path: Path, checkpoint: str | None):
    with open(path, "r", encoding="utf-8") as file:
        config = json.load(file)
    config["input_feature_mode"] = infer_input_feature_mode(config)
    if checkpoint:
        config["resume_path"] = checkpoint
    return config


def score_midi_dir_from_processed(refined_dir: str) -> Path:
    refined_path = Path(refined_dir)
    return refined_path.parent / "refined"


def list_gt_midis(
    metadata_path: str,
    score_source: str,
    split: str,
    limit: int | None = None,
    performance_dataset: str | None = None,
    exclude_performance_dataset: str | None = None,
):
    df = pd.read_csv(
        metadata_path,
        usecols=[
            "tier_a",
            "split",
            "performance_dataset",
            "refined_score_midi_path",
            "refined_performance_midi_path",
        ],
    )
    df = df[df["tier_a"].fillna(False).astype(bool)]
    df = df[df["split"] == split]
    df = df[df["refined_score_midi_path"] == score_source]
    df = df[df["refined_performance_midi_path"].notna()]
    if performance_dataset is not None:
        df = df[df["performance_dataset"].fillna("").astype(str) == str(performance_dataset)]
    if exclude_performance_dataset is not None:
        df = df[df["performance_dataset"].fillna("").astype(str) != str(exclude_performance_dataset)]
    paths = sorted(df["refined_performance_midi_path"].unique().tolist())
    if limit is not None:
        paths = paths[:limit]
    return paths


def load_score_from_node(
    path: Path,
    use_timing_scale_bit=True,
    timing_control_mode=None,
    timing_log_scale=50.0,
    task_type="epr",
    performance_source=None,
):
    with open(path, "r", encoding="utf-8") as file:
        work = json.load(file)
    score = work["score"]
    pitch = score["pitch"]
    if str(task_type).lower() == "csr":
        performances = work.get("performances", [])
        perf = None
        if performance_source is not None:
            for candidate in performances:
                if candidate.get("performance_source") == performance_source:
                    perf = candidate
                    break
        if perf is None:
            perf = performances[0] if performances else None
        if perf is None:
            raise ValueError(f"No performance rows available for CSR inference in {path}")
        continuous = build_csr_performance_input_rows(
            perf,
            use_timing_scale_bit=use_timing_scale_bit,
            timing_control_mode=timing_control_mode,
            log_scale=timing_log_scale,
        )
    else:
        continuous = build_epr_score_input_rows(
            score,
            use_timing_scale_bit=use_timing_scale_bit,
            timing_control_mode=timing_control_mode,
            log_scale=timing_log_scale,
        )
    score_shared_raw = [row[:3] for row in score["score_raw"]]
    return pitch, continuous, score_shared_raw, work


def build_windows(total_notes: int, block_notes: int, overlap_ratio: float):
    if total_notes <= block_notes:
        return [(0, total_notes)]
    stride = max(1, int(round(block_notes * (1.0 - overlap_ratio))))
    windows = []
    start = 0
    while start + block_notes <= total_notes:
        windows.append((start, start + block_notes))
        start += stride
    if windows[-1][1] != total_notes:
        windows.append((max(0, total_notes - block_notes), total_notes))
    deduped = []
    seen = set()
    for window in windows:
        if window not in seen:
            deduped.append(window)
            seen.add(window)
    return deduped


def batch_window_predictions(
    model,
    pitch,
    continuous,
    score_shared_raw,
    windows,
    pitch_pad_id,
    device,
    batch_size,
    sampling_strategy="mean",
):
    total_notes = len(pitch)
    pred_sum = None
    pred_count = torch.zeros(total_notes, 1, dtype=torch.float32)

    for batch_start in range(0, len(windows), batch_size):
        batch_windows = windows[batch_start : batch_start + batch_size]
        pitch_tensors = []
        continuous_tensors = []
        score_shared_raw_tensors = []
        lengths = []

        for start, end in batch_windows:
            pitch_tensors.append(torch.tensor(pitch[start:end], dtype=torch.long))
            continuous_tensors.append(torch.tensor(continuous[start:end], dtype=torch.float32))
            score_shared_raw_tensors.append(torch.tensor(score_shared_raw[start:end], dtype=torch.float32))
            lengths.append(end - start)

        pitch_ids = pad_sequence(pitch_tensors, batch_first=True, padding_value=pitch_pad_id).to(device)
        continuous_tensor = pad_sequence(continuous_tensors, batch_first=True, padding_value=0.0).to(device)
        score_shared_raw_tensor = pad_sequence(score_shared_raw_tensors, batch_first=True, padding_value=0.0).to(device)
        attention_mask = (pitch_ids != pitch_pad_id).long()

        with torch.no_grad():
            outputs = model(
                pitch_ids=pitch_ids,
                continuous=continuous_tensor,
                score_shared_raw=score_shared_raw_tensor,
                attention_mask=attention_mask,
                continuous_sampling_strategy=sampling_strategy,
            )
        logits = outputs.logits.detach().float().cpu()
        if pred_sum is None:
            pred_sum = torch.zeros(total_notes, logits.shape[-1], dtype=torch.float32)

        for idx, (start, end) in enumerate(batch_windows):
            length = lengths[idx]
            pred_sum[start:end] += logits[idx, :length]
            pred_count[start:end] += 1.0

    if pred_sum is None:
        raise ValueError("No windows were processed during batch_window_predictions")
    return pred_sum / pred_count.clamp_min(1.0)


def continuation_window_predictions(
    model,
    pitch,
    continuous,
    score_shared_raw,
    windows,
    pitch_pad_id,
    device,
    sampling_strategy="mean",
    drop_ratio=0.2,
):
    window_predictions = []
    merged = None

    for window_idx, (start, end) in enumerate(windows):
        pitch_ids = torch.tensor(pitch[start:end], dtype=torch.long, device=device).unsqueeze(0)
        continuous_tensor = torch.tensor(continuous[start:end], dtype=torch.float32, device=device).unsqueeze(0)
        score_shared_raw_tensor = torch.tensor(score_shared_raw[start:end], dtype=torch.float32, device=device).unsqueeze(0)
        attention_mask = (pitch_ids != pitch_pad_id).long()

        prefix_predictions = None
        if window_idx > 0:
            last_start, last_end = windows[window_idx - 1]
            overlap_len = max(0, last_end - start)
            keep_prefix_len = max(0, overlap_len - int(overlap_len * drop_ratio))
            if keep_prefix_len > 0:
                prefix_start = max(0, start - last_start)
                prefix_end = prefix_start + keep_prefix_len
                prefix_predictions = window_predictions[-1][:, prefix_start:prefix_end].to(device)

        with torch.no_grad():
            pred = model.predict_performance_continuous(
                pitch_ids=pitch_ids,
                continuous=continuous_tensor,
                score_shared_raw=score_shared_raw_tensor,
                attention_mask=attention_mask,
                prefix_predictions=prefix_predictions,
                sampling_strategy=sampling_strategy,
            ).detach().float().cpu()

        window_predictions.append(pred)
        if window_idx == 0:
            merged = pred
        else:
            last_start, last_end = windows[window_idx - 1]
            overlap_len = max(0, last_end - start)
            keep_prefix_len = max(0, overlap_len - int(overlap_len * drop_ratio))
            append_from = overlap_len
            merged = torch.cat([merged, pred[:, append_from:]], dim=1)

    return merged.squeeze(0)


def maybe_warn_sampling(protocol: str, num_samples: int, checkpoint: str | None):
    if protocol == "sampling" and num_samples > 1:
        print(
            "Warning: sampling protocol is enabled. "
            "For probabilistic INR checkpoints, repeated runs will produce different samples."
        )
    if checkpoint:
        print(f"Loading checkpoint: {checkpoint}")


def load_model(config, device):
    model = create_model(config)
    model.to(device)
    model.eval()
    return model


def musical_rows_to_score_features(rows):
    features = []
    for row in rows:
        mo = float(row[0]) * 6.0
        md = float(row[2]) * 6.0
        ml = float(row[3]) * 6.0
        first = 1.0 if float(row[5]) >= 0.5 else 0.0
        grace = 1.0 if float(row[6]) >= 0.5 else 0.0
        hand = 1.0 if float(row[7]) >= 0.5 else 0.0
        trill = 1.0 if float(row[8]) >= 0.5 else 0.0
        stacc = 1.0 if float(row[9]) >= 0.5 else 0.0
        stem_up = float(row[10]) >= 0.5
        stem_down = float(row[11]) >= 0.5
        if stem_up and not stem_down:
            stem = 1.0
        elif stem_down and not stem_up:
            stem = 2.0
        else:
            stem = 0.0
        features.append([mo, md, ml, first, hand, trill, grace, stacc, stem])
    return features


def csr_metric_summary(pred_rows, target_rows, mask):
    valid = torch.tensor(mask, dtype=torch.bool)
    if not valid.any():
        return {"valid_notes": 0}
    pred = torch.tensor(pred_rows, dtype=torch.float32)[valid]
    target = torch.tensor(target_rows, dtype=torch.float32)[valid]

    def mae(idx, scale=1.0):
        return float((pred[:, idx] * scale - target[:, idx] * scale).abs().mean().item())

    def acc(idx):
        return float(((pred[:, idx] >= 0.5) == (target[:, idx] >= 0.5)).float().mean().item())

    stem_pred = pred[:, 10:12].argmax(dim=-1)
    stem_target = target[:, 10:12].argmax(dim=-1)
    return {
        "valid_notes": int(valid.sum().item()),
        "mo_mae_quarter": mae(0, 6.0),
        "mioi_mae_quarter": mae(1, 6.0),
        "md_mae_quarter": mae(2, 6.0),
        "ml_mae_quarter": mae(3, 6.0),
        "tempo_mae_norm": mae(4, 1.0),
        "first_acc": acc(5),
        "grace_acc": acc(6),
        "hand_acc": acc(7),
        "trill_acc": acc(8),
        "stacc_acc": acc(9),
        "stem_acc": float((stem_pred == stem_target).float().mean().item()),
    }


def predict_one_csr_work(model, device, config, work, args):
    score_source = work["score_source"]
    selected_sources = work.get("selected_performance_sources") or [None]
    performance_source = selected_sources[0]
    pitch, continuous, score_shared_raw, loaded = load_score_from_node(
        Path(work["path"]),
        use_timing_scale_bit=config.get("use_timing_scale_bit", True),
        timing_control_mode=config.get("timing_control_mode"),
        timing_log_scale=config.get("timing_log_scale", 50.0),
        task_type="csr",
        performance_source=performance_source,
    )
    score = loaded["score"]
    windows = build_windows(len(pitch), config["block_notes"], config["overlap_ratio"])

    pred_continuous = batch_window_predictions(
        model=model,
        pitch=pitch,
        continuous=continuous,
        score_shared_raw=score_shared_raw,
        windows=windows,
        pitch_pad_id=config["pitch_pad_id"],
        device=device,
        batch_size=args.batch_size_windows,
        sampling_strategy=resolve_sampling_strategy(args),
    ) if args.merge_mode == "average" else continuation_window_predictions(
        model=model,
        pitch=pitch,
        continuous=continuous,
        score_shared_raw=score_shared_raw,
        windows=windows,
        pitch_pad_id=config["pitch_pad_id"],
        device=device,
        sampling_strategy=resolve_sampling_strategy(args),
        drop_ratio=args.continuation_drop_ratio,
    )

    target_rows = build_score_musical_rows(score)
    has_score_feature = score.get("has_score_feature", [0] * len(pitch))
    pred_rows = pred_continuous.float().cpu().tolist()
    score_features = musical_rows_to_score_features(pred_rows)

    raw_dir = args.output_dir / "raw_outputs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    score_stem = Path(score_source).with_suffix("").as_posix().replace("/", "__")
    raw_path = raw_dir / f"{score_stem}__csr.json"
    payload = {
        "score_source": score_source,
        "performance_source": performance_source,
        "protocol": args.protocol,
        "timing_representation": "inr0624_csr_musical12",
        "pitch": [int(value) for value in pitch],
        "predicted_musical": pred_rows,
        "predicted_score_feature": score_features,
        "target_musical": target_rows,
        "has_score_feature": has_score_feature,
        "metrics": csr_metric_summary(pred_rows, target_rows, has_score_feature),
        "note_count": len(pitch),
        "num_windows": len(windows),
    }
    raw_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "score_source": score_source,
        "performance_source": performance_source,
        "raw_output_paths": [str(raw_path.resolve())],
        "metrics": payload["metrics"],
        "note_count": len(pitch),
        "num_windows": len(windows),
    }


def predict_one_work(model, device, config, work, args, score_midi_dir, midi_dir):
    if str(config.get("task_type", "epr")).lower() == "csr":
        return predict_one_csr_work(model, device, config, work, args)

    score_source = work["score_source"]
    pitch, continuous, score_shared_raw, _ = load_score_from_node(
        Path(work["path"]),
        use_timing_scale_bit=config.get("use_timing_scale_bit", True),
        timing_control_mode=config.get("timing_control_mode"),
        timing_log_scale=config.get("timing_log_scale", 50.0),
        task_type="epr",
    )
    windows = build_windows(len(pitch), config["block_notes"], config["overlap_ratio"])
    gt_rel_paths = list_gt_midis(
        config["metadata_path"],
        score_source=score_source,
        split=args.split,
        limit=args.max_gt_per_score,
        performance_dataset=args.performance_dataset,
        exclude_performance_dataset=args.exclude_performance_dataset,
    )

    raw_dir = args.output_dir / "raw_outputs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    prediction_paths = []
    raw_output_paths = []
    score_stem = Path(score_source).with_suffix("").as_posix().replace("/", "__")
    for sample_idx in range(args.num_samples):
        sample_seed = args.seed + sample_idx
        random.seed(sample_seed)
        torch.manual_seed(sample_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(sample_seed)

        pred_continuous = batch_window_predictions(
            model=model,
            pitch=pitch,
            continuous=continuous,
            score_shared_raw=score_shared_raw,
            windows=windows,
            pitch_pad_id=config["pitch_pad_id"],
            device=device,
            batch_size=args.batch_size_windows,
            sampling_strategy=resolve_sampling_strategy(args),
        ) if args.merge_mode == "average" else continuation_window_predictions(
            model=model,
            pitch=pitch,
            continuous=continuous,
            score_shared_raw=score_shared_raw,
            windows=windows,
            pitch_pad_id=config["pitch_pad_id"],
            device=device,
            sampling_strategy=resolve_sampling_strategy(args),
            drop_ratio=args.continuation_drop_ratio,
        )
        pred_target = pred_continuous
        raw_rows = _target5_to_raw7(
            torch.tensor(score_shared_raw, dtype=torch.float32),
            pred_target.float().cpu(),
            config=config,
        )
        midi_obj = note_features_to_midi(
            pitch=pitch,
            continuous=raw_rows.tolist(),
            target_ticks_per_beat=500,
            target_tempo=120,
            max_time_ms=config["max_time_ms"],
            normalized=False,
        )
        raw_path = raw_dir / f"{score_stem}__sample_{sample_idx:03d}.json"
        raw_payload = {
            "score_source": score_source,
            "protocol": args.protocol,
            "deterministic_strategy": args.deterministic_strategy if args.protocol == "deterministic" else None,
            "sample_idx": sample_idx,
            "seed": sample_seed,
            "timing_representation": "target5_dev_velocity_pedal2",
            "pitch": [int(value) for value in pitch],
            "predicted_target5": pred_target.tolist(),
            "reconstructed_raw7": raw_rows.tolist(),
            "ground_truth_paths": gt_rel_paths,
        }
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(raw_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        raw_output_paths.append(str(raw_path.resolve()))

        pred_path = midi_dir / f"{score_stem}__sample_{sample_idx:03d}.mid"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        midi_obj.dump(str(pred_path))
        prediction_paths.append(str(pred_path.resolve()))

    score_midi_path = score_midi_dir / score_source
    gt_paths = [str((score_midi_dir / gt_rel).resolve()) for gt_rel in gt_rel_paths]
    return {
        "score_source": score_source,
        "score_midi": str(score_midi_path.resolve()),
        "prediction_paths": prediction_paths,
        "raw_output_paths": raw_output_paths,
        "ground_truth_paths": gt_paths,
        "note_count": len(pitch),
        "num_windows": len(windows),
    }


def worker_loop(worker_idx, args, config, score_midi_dir, job_queue, result_queue):
    random.seed(args.seed + worker_idx)
    torch.manual_seed(args.seed + worker_idx)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + worker_idx)
    device = select_device(args.device)
    print(f"Worker {worker_idx} using device: {device}", flush=True)
    model = load_model(config, device)
    midi_dir = args.output_dir / "midis"
    midi_dir.mkdir(parents=True, exist_ok=True)

    while True:
        job = job_queue.get()
        if job is None:
            break
        job_idx, work = job
        try:
            item = predict_one_work(model, device, config, work, args, score_midi_dir, midi_dir)
            result_queue.put((job_idx, item, None))
        except Exception as exc:  # noqa: BLE001
            result_queue.put((job_idx, None, repr(exc)))


def run_dynamic_pool(args, config, manifest, score_midi_dir):
    ctx = mp.get_context("spawn")
    job_queue = ctx.Queue()
    result_queue = ctx.Queue()
    workers = [
        ctx.Process(target=worker_loop, args=(idx, args, config, score_midi_dir, job_queue, result_queue))
        for idx in range(args.num_workers)
    ]
    for worker in workers:
        worker.start()
    for job_idx, work in enumerate(manifest):
        job_queue.put((job_idx, work))
    for _ in workers:
        job_queue.put(None)

    items_by_idx = {}
    with tqdm(total=len(manifest), desc=f"INR inference pool ({args.protocol})") as progress:
        for _ in range(len(manifest)):
            job_idx, item, error = result_queue.get()
            if error is not None:
                for worker in workers:
                    worker.terminate()
                raise RuntimeError(f"Worker failed on job {job_idx}: {error}")
            items_by_idx[job_idx] = item
            progress.update(1)

    for worker in workers:
        worker.join()
        if worker.exitcode != 0:
            raise RuntimeError(f"Worker {worker.pid} exited with code {worker.exitcode}")

    return [items_by_idx[idx] for idx in range(len(manifest))]


def run_single_process(args, config, manifest, score_midi_dir):
    device = select_device(args.device)
    print(f"Using device: {device}")
    model = load_model(config, device)
    midi_dir = args.output_dir / "midis"
    midi_dir.mkdir(parents=True, exist_ok=True)

    items = []
    iterator = tqdm(manifest, desc=f"INR inference ({args.protocol})")
    for work in iterator:
        items.append(predict_one_work(model, device, config, work, args, score_midi_dir, midi_dir))
    return items


def build_pair_list(items):
    return [
        {"pred": pred_path, "gt": gt_path}
        for item in items
        for pred_path in item.get("prediction_paths", [])
        for gt_path in item.get("ground_truth_paths", [])
    ]


def summarize_csr_items(items):
    metric_sums = defaultdict(float)
    metric_counts = defaultdict(int)
    for item in items:
        metrics = item.get("metrics") or {}
        for key, value in metrics.items():
            if key == "valid_notes":
                continue
            metric_sums[key] += float(value)
            metric_counts[key] += 1
    return {
        key: metric_sums[key] / max(metric_counts[key], 1)
        for key in sorted(metric_sums)
    }


def filter_manifest_by_performance_dataset(
    manifest,
    metadata_path,
    split,
    performance_dataset,
    exclude_performance_dataset=None,
):
    if performance_dataset is None and exclude_performance_dataset is None:
        return manifest
    df = pd.read_csv(
        metadata_path,
        usecols=[
            "tier_a",
            "split",
            "performance_dataset",
            "refined_score_midi_path",
            "refined_performance_midi_path",
        ],
    )
    df = df[df["tier_a"].fillna(False).astype(bool)]
    df = df[df["split"] == split]
    dataset = df["performance_dataset"].fillna("").astype(str)
    if performance_dataset is not None:
        df = df[dataset == str(performance_dataset)]
        dataset = df["performance_dataset"].fillna("").astype(str)
    if exclude_performance_dataset is not None:
        df = df[dataset != str(exclude_performance_dataset)]
    df = df[df["refined_score_midi_path"].notna()]
    df = df[df["refined_performance_midi_path"].notna()]
    allowed_scores = set(df["refined_score_midi_path"].unique())
    filtered = []
    for item in manifest:
        if item["score_source"] not in allowed_scores:
            continue
        allowed_sources = set(
            df.loc[
                df["refined_score_midi_path"] == item["score_source"],
                "refined_performance_midi_path",
            ]
        )
        copied = dict(item)
        copied["selected_performance_sources"] = [
            source for source in item.get("selected_performance_sources", [])
            if source in allowed_sources
        ]
        copied["estimated_performances"] = len(copied["selected_performance_sources"])
        copied["estimated_examples"] = len(copied["windows"]) * len(copied["selected_performance_sources"])
        filtered.append(copied)
    return filtered


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")

    config = load_config(args.config, args.checkpoint)
    maybe_warn_sampling(args.protocol, args.num_samples, args.checkpoint)

    manifest = build_work_manifest(
        metadata_path=config["metadata_path"],
        refined_dir=config["refined_dir"],
        split=args.split,
        block_notes=config["block_notes"],
        overlap_ratio=config["overlap_ratio"],
        min_notes=config["min_notes"],
        max_works=None if (args.performance_dataset is not None or args.exclude_performance_dataset is not None) else args.max_works,
        skip_work_paths=config.get("skip_work_paths"),
    )
    manifest = filter_manifest_by_performance_dataset(
        manifest,
        metadata_path=config["metadata_path"],
        split=args.split,
        performance_dataset=args.performance_dataset,
        exclude_performance_dataset=args.exclude_performance_dataset,
    )
    if (args.performance_dataset is not None or args.exclude_performance_dataset is not None) and args.max_works is not None:
        manifest = manifest[: args.max_works]
    score_midi_dir = score_midi_dir_from_processed(config["refined_dir"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.num_workers > 1:
        items = run_dynamic_pool(args, config, manifest, score_midi_dir)
    else:
        items = run_single_process(args, config, manifest, score_midi_dir)

    manifest_path = args.output_dir / "prediction_manifest.json"
    pair_list_path = args.output_dir / "evaluate_list.json"
    manifest_payload = {
        "config": str(args.config.resolve()),
        "checkpoint": args.checkpoint or config.get("resume_path"),
        "task_type": config.get("task_type", "epr"),
        "protocol": args.protocol,
        "num_samples": args.num_samples,
        "num_workers": args.num_workers,
        "split": args.split,
        "performance_dataset": args.performance_dataset,
        "exclude_performance_dataset": args.exclude_performance_dataset,
        "items": items,
    }
    if str(config.get("task_type", "epr")).lower() == "csr":
        manifest_payload["csr_summary"] = summarize_csr_items(items)
    pair_list = build_pair_list(items)
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False))
    pair_list_path.write_text(json.dumps(pair_list, indent=2, ensure_ascii=False))
    print(f"Saved prediction manifest to {manifest_path}")
    print(f"Saved pair list to {pair_list_path}")


if __name__ == "__main__":
    main()
