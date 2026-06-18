import argparse
import json
import math
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
    create_model,
    infer_input_feature_mode,
    make_score_note_input,
)
from src.utils.inr_midi import note_features_to_midi


def parse_args():
    parser = argparse.ArgumentParser(description="Run INR inference on PianoCoRe test scores.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--protocol", choices=["deterministic", "sampling"], default="deterministic")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-works", type=int, default=None)
    parser.add_argument("--batch-size-windows", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-gt-per-score", type=int, default=None)
    parser.add_argument("--merge-mode", choices=["continuation", "average"], default="continuation")
    parser.add_argument("--continuation-drop-ratio", type=float, default=0.2)
    return parser.parse_args()


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


def list_gt_midis(metadata_path: str, score_source: str, split: str, limit: int | None = None):
    df = pd.read_csv(
        metadata_path,
        usecols=[
            "tier_a",
            "split",
            "refined_score_midi_path",
            "refined_performance_midi_path",
        ],
    )
    df = df[df["tier_a"].fillna(False).astype(bool)]
    df = df[df["split"] == split]
    df = df[df["refined_score_midi_path"] == score_source]
    df = df[df["refined_performance_midi_path"].notna()]
    paths = sorted(df["refined_performance_midi_path"].unique().tolist())
    if limit is not None:
        paths = paths[:limit]
    return paths


def load_score_from_node(path: Path, input_feature_mode: str):
    with open(path, "r", encoding="utf-8") as file:
        work = json.load(file)
    score = work["score"]
    pitch = score["pitch"]
    continuous = make_score_note_input(
        score["score_continuous"],
        score.get("score_feature", [[0.0] * 8 for _ in pitch]),
        score.get("has_score_feature", [0] * len(pitch)),
        input_feature_mode,
    )
    return pitch, continuous


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


def batch_window_predictions(model, pitch, continuous, windows, pitch_pad_id, device, batch_size):
    total_notes = len(pitch)
    output_dim = model.config.output_continuous_dim
    pred_sum = torch.zeros(total_notes, output_dim, dtype=torch.float32)
    pred_count = torch.zeros(total_notes, 1, dtype=torch.float32)

    for batch_start in range(0, len(windows), batch_size):
        batch_windows = windows[batch_start : batch_start + batch_size]
        pitch_tensors = []
        continuous_tensors = []
        lengths = []

        for start, end in batch_windows:
            pitch_tensors.append(torch.tensor(pitch[start:end], dtype=torch.long))
            continuous_tensors.append(torch.tensor(continuous[start:end], dtype=torch.float32))
            lengths.append(end - start)

        pitch_ids = pad_sequence(pitch_tensors, batch_first=True, padding_value=pitch_pad_id).to(device)
        continuous_tensor = pad_sequence(continuous_tensors, batch_first=True, padding_value=0.0).to(device)
        attention_mask = (pitch_ids != pitch_pad_id).long()

        with torch.no_grad():
            outputs = model(
                pitch_ids=pitch_ids,
                continuous=continuous_tensor,
                attention_mask=attention_mask,
            )
        logits = outputs.logits.detach().float().cpu()

        for idx, (start, end) in enumerate(batch_windows):
            length = lengths[idx]
            pred_sum[start:end] += logits[idx, :length]
            pred_count[start:end] += 1.0

    return pred_sum / pred_count.clamp_min(1.0)


def continuation_window_predictions(
    model,
    pitch,
    continuous,
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
        attention_mask = (pitch_ids != pitch_pad_id).long()

        prefix_predictions = None
        if window_idx > 0:
            last_start, last_end = windows[window_idx - 1]
            overlap_len = max(0, last_end - start)
            keep_prefix_len = max(0, overlap_len - int(overlap_len * drop_ratio))
            if keep_prefix_len > 0:
                prefix_predictions = window_predictions[-1][:, overlap_len - keep_prefix_len : overlap_len]

        with torch.no_grad():
            pred = model.predict_performance_continuous(
                pitch_ids=pitch_ids,
                continuous=continuous_tensor,
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
            append_from = keep_prefix_len
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


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = load_config(args.config, args.checkpoint)
    maybe_warn_sampling(args.protocol, args.num_samples, args.checkpoint)

    device = select_device(args.device)
    model = create_model(config)
    model.to(device)
    model.eval()

    manifest = build_work_manifest(
        metadata_path=config["metadata_path"],
        refined_dir=config["refined_dir"],
        split=args.split,
        block_notes=config["block_notes"],
        overlap_ratio=config["overlap_ratio"],
        min_notes=config["min_notes"],
        max_works=args.max_works,
    )
    score_midi_dir = score_midi_dir_from_processed(config["refined_dir"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    midi_dir = args.output_dir / "midis"
    midi_dir.mkdir(parents=True, exist_ok=True)

    items = []
    iterator = tqdm(manifest, desc=f"INR inference ({args.protocol})")
    for work in iterator:
        score_source = work["score_source"]
        pitch, continuous = load_score_from_node(Path(work["path"]), config["input_feature_mode"])
        windows = build_windows(len(pitch), config["block_notes"], config["overlap_ratio"])
        gt_rel_paths = list_gt_midis(
            config["metadata_path"],
            score_source=score_source,
            split=args.split,
            limit=args.max_gt_per_score,
        )

        prediction_paths = []
        for sample_idx in range(args.num_samples):
            pred_continuous = batch_window_predictions(
                model=model,
                pitch=pitch,
                continuous=continuous,
                windows=windows,
                pitch_pad_id=config["pitch_pad_id"],
                device=device,
                batch_size=args.batch_size_windows,
            ) if args.merge_mode == "average" else continuation_window_predictions(
                model=model,
                pitch=pitch,
                continuous=continuous,
                windows=windows,
                pitch_pad_id=config["pitch_pad_id"],
                device=device,
                sampling_strategy="sample" if args.protocol == "sampling" else "mean",
                drop_ratio=args.continuation_drop_ratio,
            )
            midi_obj = note_features_to_midi(
                pitch=pitch,
                continuous=pred_continuous.tolist(),
                target_ticks_per_beat=500,
                target_tempo=120,
                max_time_ms=config["max_time_ms"],
                normalized=True,
            )
            pred_path = midi_dir / f"{Path(score_source).with_suffix('').as_posix().replace('/', '__')}__sample_{sample_idx:03d}.mid"
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            midi_obj.dump(str(pred_path))
            prediction_paths.append(str(pred_path.resolve()))

        score_midi_path = score_midi_dir / score_source
        gt_paths = [str((score_midi_dir / gt_rel).resolve()) for gt_rel in gt_rel_paths]
        items.append(
            {
                "score_source": score_source,
                "score_midi": str(score_midi_path.resolve()),
                "prediction_paths": prediction_paths,
                "ground_truth_paths": gt_paths,
                "note_count": len(pitch),
                "num_windows": len(windows),
            }
        )

    manifest_path = args.output_dir / "prediction_manifest.json"
    manifest_payload = {
        "config": str(args.config.resolve()),
        "checkpoint": args.checkpoint or config.get("resume_path"),
        "protocol": args.protocol,
        "num_samples": args.num_samples,
        "split": args.split,
        "items": items,
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False))
    print(f"Saved prediction manifest to {manifest_path}")


if __name__ == "__main__":
    main()
