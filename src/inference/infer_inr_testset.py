import argparse
import hashlib
import json
import math
import multiprocessing as mp
import random
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_process.work_manifest import build_work_manifest
from src.train.train_inr import (
    build_perf_style_prefix_cache,
    build_epr_score_input_rows,
    build_score_musical_rows,
    build_style_vocabs,
    create_model,
    default_epr_output_dim,
    decoder_note_input_schema,
    decoder_perf_target_input_dim,
    enforce_asap_processed_config,
    integrated_epr_input_dim,
    infer_input_feature_mode,
    musical_feature_dim,
    pedal_representation_dim,
    performance_dev_velocity_pedal4_binary_rows,
    perf_style_stats_range_from_cache,
    perf_style_stats_from_cache,
    score_style_stats,
    score_musical_input_dim,
    score_note_input_schema,
)
from src.model.integrated_pianoformer import _target7_to_raw7
from src.utils.inr_midi import note_features_to_midi


def parse_args():
    parser = argparse.ArgumentParser(description="Run INR inference on ASAP processed test scores.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--protocol", choices=["sampling"], default="sampling")
    parser.add_argument(
        "--sampling-strategy",
        choices=["mean", "greedy", "sample"],
        default=None,
        help="Override output materialization.",
    )
    parser.add_argument(
        "--deterministic-strategy",
        choices=["greedy", "mean"],
        default="greedy",
        help="How deterministic inference materializes probabilistic outputs.",
    )
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-works", type=int, default=None)
    parser.add_argument("--batch-size-windows", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dlm-ioi-zero-sample-shrink-factor", type=float, default=None)
    parser.add_argument("--dlm-ioi-nonzero-sample-shrink-factor", type=float, default=None)
    parser.add_argument("--dlm-duration-sample-shrink-factor", type=float, default=None)
    parser.add_argument("--dlm-velocity-sample-shrink-factor", type=float, default=None)
    parser.add_argument("--max-gt-per-score", type=int, default=None)
    parser.add_argument("--performance-dataset", type=str, default=None,
                        help="Optional performance_dataset filter, e.g. ASAP. Restricts scores and GT refs.")
    parser.add_argument("--exclude-performance-dataset", type=str, default=None,
                        help="Optional performance_dataset exclusion, e.g. ASAP for non-ASAP evaluation.")
    parser.add_argument(
        "--score-source",
        action="append",
        default=None,
        help="Restrict inference to this refined score path. May be passed multiple times.",
    )
    parser.add_argument(
        "--score-source-list",
        type=Path,
        default=None,
        help="Text file with one refined score path per line. Empty lines and # comments are ignored.",
    )
    parser.add_argument("--merge-mode", choices=["continuation", "average"], default="continuation")
    parser.add_argument("--continuation-drop-ratio", type=float, default=0.0)
    parser.add_argument("--block-notes", type=int, default=None, help="Override config block_notes for inference only.")
    parser.add_argument("--overlap-ratio", type=float, default=None, help="Override config overlap_ratio for inference only.")
    parser.add_argument(
        "--oracle-gt-prefix-mode",
        choices=["none", "decoder", "decoder_and_style"],
        default="none",
        help=(
            "Diagnostic mode: use GT target rows for the cross-window overlap prefix. "
            "'decoder' replaces only decoder prefix_predictions; 'decoder_and_style' also "
            "uses the GT prefix for style perf statistics."
        ),
    )
    return parser.parse_args()


def resolve_sampling_strategy(args):
    if args.sampling_strategy is not None:
        return str(args.sampling_strategy).lower()
    if args.protocol == "sampling":
        return "sample"
    return str(args.deterministic_strategy).lower()


def stable_seed(base_seed, *parts):
    payload = "::".join([str(base_seed), *[str(part) for part in parts]])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def select_device(device_arg):
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def select_worker_device(device_arg, worker_idx):
    if torch.cuda.is_available():
        device_text = str(device_arg or "").strip().lower()
        if device_text in {"", "cuda"}:
            return torch.device(f"cuda:{worker_idx % torch.cuda.device_count()}")
    return select_device(device_arg)


def load_config(path: Path, checkpoint: str | None):
    with open(path, "r", encoding="utf-8") as file:
        config = json.load(file)
    enforce_asap_processed_config(config)
    config["input_feature_mode"] = infer_input_feature_mode(config)
    if config.get("use_style_tokens", False):
        composer_vocab = config.get("style_composer_vocab")
        source_vocab = config.get("style_source_vocab")
        if composer_vocab is None or source_vocab is None:
            composer_vocab, source_vocab = build_style_vocabs(config["metadata_path"])
            config["style_composer_vocab"] = composer_vocab
            config["style_source_vocab"] = source_vocab
        config["style_creator_vocab_size"] = len(config["style_composer_vocab"])
        config["style_source_vocab_size"] = len(config["style_source_vocab"])
    if checkpoint:
        config["resume_path"] = checkpoint
    config.setdefault("pedal_representation", "binary_4")
    target = str(config.get("epr_timing_target", "floor_log_deviation")).lower()
    if target not in {"floor_log_deviation", "floor_log_dev"}:
        raise ValueError("Only epr_timing_target=floor_log_deviation is supported")
    config["legacy_dual_timing_head"] = False
    if config.get("task_type", "epr").lower() == "epr" and "output_continuous_dim" not in config:
        config["output_continuous_dim"] = default_epr_output_dim(
            target,
            config.get("pedal_representation", "binary_4"),
        )
    if config.get("task_type", "epr").lower() != "epr":
        raise ValueError("Only task_type=epr is supported")
    musical_mode = str(config.get("musical_feature_mode", "musical4slot")).lower()
    config["musical_feature_dim"] = musical_feature_dim(musical_mode)
    config["mask_feature_dim"] = 2 if int(config["musical_feature_dim"]) == 0 else 3
    config["score_control_feature_dim"] = int(config.get("score_control_feature_dim", config.get("control_feature_dim", 5)))
    pedal_dim = pedal_representation_dim(config.get("pedal_representation", "binary_4"))
    config["performance_control_feature_dim"] = int(
        config.get("performance_control_feature_dim", config.get("control_feature_dim", 5) + pedal_dim)
    )
    if config.get("task_type", "epr").lower() == "epr" and config["input_feature_mode"] == "integrated":
        score_schema = score_note_input_schema(config)
        decoder_schema = decoder_note_input_schema(config)
        if config.get("task_type", "epr").lower() == "epr" and score_schema == "score_musical":
            score_dim = score_musical_input_dim(
                timing_control_mode=config.get("timing_control_mode"),
                use_timing_scale_bit=config.get("use_timing_scale_bit", False),
                musical_feature_mode=musical_mode,
            )
        else:
            score_dim = integrated_epr_input_dim(
                timing_control_mode=config.get("timing_control_mode"),
                use_timing_scale_bit=config.get("use_timing_scale_bit", False),
                musical_feature_mode=musical_mode,
                pedal_control_dim=pedal_dim,
            )
        config["input_continuous_dim"] = score_dim
        config["score_input_continuous_dim"] = score_dim
        config["decoder_input_continuous_dim"] = (
            decoder_perf_target_input_dim(config.get("output_continuous_dim", config.get("continuous_dim", 7)))
            if config.get("task_type", "epr").lower() == "epr" and decoder_schema == "perf_target"
            else score_dim
        )
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
    use_timing_scale_bit=False,
    timing_control_mode="dinr_floor_log",
    timing_log_scale=50.0,
    musical_feature_mode="musical4slot",
    score_note_schema="integrated",
    task_type="epr",
    performance_source=None,
    disable_musical_features=False,
    pedal_control_dim=4,
):
    with open(path, "r", encoding="utf-8") as file:
        work = json.load(file)
    score = work["score"]
    pitch = score["pitch"]
    continuous = build_epr_score_input_rows(
        score,
        use_timing_scale_bit=use_timing_scale_bit,
        timing_control_mode=timing_control_mode,
        log_scale=timing_log_scale,
        musical_feature_mode=musical_feature_mode,
        score_note_schema=score_note_schema,
        disable_musical_features=disable_musical_features,
        pedal_control_dim=pedal_control_dim,
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
    score,
    windows,
    pitch_pad_id,
    device,
    batch_size,
    sampling_strategy="mean",
    style_creator_id=None,
    style_source_id=None,
):
    total_notes = len(pitch)
    pred_sum = None
    pred_count = torch.zeros(total_notes, 1, dtype=torch.float32)
    use_style_tokens = bool(getattr(model.config, "use_style_tokens", False))

    for batch_start in range(0, len(windows), batch_size):
        batch_windows = windows[batch_start : batch_start + batch_size]
        pitch_tensors = []
        continuous_tensors = []
        score_shared_raw_tensors = []
        style_score_rows = []
        lengths = []

        for start, end in batch_windows:
            pitch_tensors.append(torch.tensor(pitch[start:end], dtype=torch.long))
            continuous_tensors.append(torch.tensor(continuous[start:end], dtype=torch.float32))
            score_shared_raw_tensors.append(torch.tensor(score_shared_raw[start:end], dtype=torch.float32))
            if use_style_tokens:
                style_score_rows.append(score_style_stats(score, start, end))
            lengths.append(end - start)

        pitch_ids = pad_sequence(pitch_tensors, batch_first=True, padding_value=pitch_pad_id).to(device)
        continuous_tensor = pad_sequence(continuous_tensors, batch_first=True, padding_value=0.0).to(device)
        score_shared_raw_tensor = pad_sequence(score_shared_raw_tensors, batch_first=True, padding_value=0.0).to(device)
        attention_mask = (pitch_ids != pitch_pad_id).long()
        style_kwargs = {}
        if use_style_tokens:
            style_kwargs = {
                "style_creator_ids": torch.full(
                    (len(batch_windows),),
                    int(style_creator_id or 0),
                    dtype=torch.long,
                    device=device,
                ),
                "style_source_ids": torch.full(
                    (len(batch_windows),),
                    int(style_source_id or 0),
                    dtype=torch.long,
                    device=device,
                ),
                "style_score_stats": torch.tensor(style_score_rows, dtype=torch.float32, device=device),
                "style_perf_stats": torch.zeros((len(batch_windows), 18), dtype=torch.float32, device=device),
                "style_perf_is_pad": torch.ones((len(batch_windows),), dtype=torch.bool, device=device),
            }

        with torch.no_grad():
            outputs = model(
                pitch_ids=pitch_ids,
                continuous=continuous_tensor,
                score_shared_raw=score_shared_raw_tensor,
                attention_mask=attention_mask,
                continuous_sampling_strategy=sampling_strategy,
                **style_kwargs,
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
    score,
    windows,
    pitch_pad_id,
    device,
    sampling_strategy="mean",
    drop_ratio=0.2,
    style_creator_id=None,
    style_source_id=None,
    oracle_prefix_targets=None,
    oracle_style_targets=None,
    style_perf_stats_mode="prefix",
):
    window_predictions = []
    merged = None
    use_style_tokens = bool(getattr(model.config, "use_style_tokens", False))
    style_perf_stats_mode = str(style_perf_stats_mode or "prefix").lower()
    oracle_prefix_targets = (
        torch.tensor(oracle_prefix_targets, dtype=torch.float32)
        if oracle_prefix_targets is not None
        else None
    )
    oracle_style_targets = (
        torch.tensor(oracle_style_targets, dtype=torch.float32)
        if oracle_style_targets is not None
        else None
    )

    for window_idx, (start, end) in enumerate(windows):
        pitch_ids = torch.tensor(pitch[start:end], dtype=torch.long, device=device).unsqueeze(0)
        continuous_tensor = torch.tensor(continuous[start:end], dtype=torch.float32, device=device).unsqueeze(0)
        score_shared_raw_tensor = torch.tensor(score_shared_raw[start:end], dtype=torch.float32, device=device).unsqueeze(0)
        attention_mask = (pitch_ids != pitch_pad_id).long()
        style_kwargs = {}
        if use_style_tokens:
            if style_perf_stats_mode == "window" and oracle_style_targets is not None:
                perf_stats = perf_style_stats_range_from_cache(
                    build_perf_style_prefix_cache(oracle_style_targets.float().cpu().numpy()),
                    start,
                    end,
                )
                perf_is_pad = False
            elif start <= 0 or merged is None:
                perf_stats = [0.0] * 18
                perf_is_pad = True
            else:
                if oracle_style_targets is not None:
                    prefix = oracle_style_targets[:start].float().cpu().numpy()
                else:
                    prefix = merged.squeeze(0)[:start].float().cpu().numpy()
                perf_stats = perf_style_stats_from_cache(
                    build_perf_style_prefix_cache(prefix),
                    prefix.shape[0],
                )
                perf_is_pad = False
            style_kwargs = {
                "style_creator_ids": torch.tensor([int(style_creator_id or 0)], dtype=torch.long, device=device),
                "style_source_ids": torch.tensor([int(style_source_id or 0)], dtype=torch.long, device=device),
                "style_score_stats": torch.tensor([score_style_stats(score, start, end)], dtype=torch.float32, device=device),
                "style_perf_stats": torch.tensor([perf_stats], dtype=torch.float32, device=device),
                "style_perf_is_pad": torch.tensor([perf_is_pad], dtype=torch.bool, device=device),
            }

        prefix_predictions = None
        if window_idx > 0:
            last_start, last_end = windows[window_idx - 1]
            overlap_len = max(0, last_end - start)
            keep_prefix_len = max(0, overlap_len - int(overlap_len * drop_ratio))
            if keep_prefix_len > 0:
                prefix_start = max(0, start - last_start)
                prefix_end = prefix_start + keep_prefix_len
                if oracle_prefix_targets is not None:
                    prefix_predictions = oracle_prefix_targets[start : start + keep_prefix_len].unsqueeze(0).to(device)
                else:
                    prefix_predictions = window_predictions[-1][:, prefix_start:prefix_end].to(device)

        with torch.no_grad():
            pred = model.predict_performance_continuous(
                pitch_ids=pitch_ids,
                continuous=continuous_tensor,
                score_shared_raw=score_shared_raw_tensor,
                attention_mask=attention_mask,
                prefix_predictions=prefix_predictions,
                sampling_strategy=sampling_strategy,
                **style_kwargs,
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


def style_ids_for_work(config, loaded_work, performance_dataset=None):
    if not bool(config.get("use_style_tokens", False)):
        return None, None
    composer_vocab = config.get("style_composer_vocab", {})
    source_vocab = config.get("style_source_vocab", {})
    meta = loaded_work.get("meta", {}) if isinstance(loaded_work, dict) else {}
    composer = str(meta.get("composer") or "")
    source = str(performance_dataset or "unknown")
    creator_id = int(composer_vocab.get(composer, composer_vocab.get("<unk>", 0)))
    source_id = int(source_vocab.get(source, source_vocab.get("<unk>", 0)))
    return creator_id, source_id


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


def labels_for_perf(config, perf, score_shared_raw):
    labels = performance_dev_velocity_pedal4_binary_rows(
        perf,
        score_shared_raw,
        epr_timing_target=config.get("epr_timing_target", "floor_log_deviation"),
        log_scale=float(config.get("timing_log_scale", 50.0)),
        pedal_binary_threshold=float(config.get("pedal_binary_threshold", 64.0)),
        legacy_dual_timing_head=bool(config.get("legacy_dual_timing_head", False)),
        pedal_representation=config.get("pedal_representation", "binary_4"),
    )
    if labels is None:
        raise ValueError(f"Could not build labels for {perf.get('performance_source')}")
    return labels


def performance_by_source(work, performance_source):
    for perf in work.get("performances", []):
        if perf.get("performance_source") == performance_source:
            return perf
    raise ValueError(f"Performance source not found in work json: {performance_source}")


def _score_onset_span_ms(score_raw):
    return float(sum(float(row[0]) for row in score_raw[1:]))


def _performance_shared_rows(perf):
    shared_rows = perf.get("label_shared_raw")
    if shared_rows is not None:
        return [list(row[:3]) for row in shared_rows]
    label_raw = perf.get("label_raw")
    if label_raw is not None:
        return [list(row[:3]) for row in label_raw]
    raise ValueError(f"Performance has no shared raw timing rows: {perf.get('performance_source')}")


def _performance_pedal4_rows(perf):
    pedal_rows = perf.get("label_pedal4_raw")
    if pedal_rows is None:
        pedal_rows = perf.get("pedal4_raw")
    if pedal_rows is not None:
        return [list(row[:4]) for row in pedal_rows]
    label_raw = perf.get("label_raw")
    if label_raw is not None:
        return [list(row[3:7]) for row in label_raw]
    raise ValueError(f"Performance has no pedal4 raw rows: {perf.get('performance_source')}")


def _raw7_rows_for_eval_gt(perf, score_raw, normalization="none"):
    shared_rows = _performance_shared_rows(perf)
    pedal_rows = _performance_pedal4_rows(perf)
    if len(shared_rows) != len(score_raw) or len(pedal_rows) != len(score_raw):
        raise ValueError(
            "GT row length mismatch for "
            f"{perf.get('performance_source')}: shared={len(shared_rows)}, "
            f"pedal={len(pedal_rows)}, score={len(score_raw)}"
        )

    mode = str(normalization or "none").lower()
    scale = 1.0
    if mode == "score_onset_span":
        score_span = _score_onset_span_ms(score_raw)
        perf_span = float(sum(float(row[0]) for row in shared_rows[1:]))
        if score_span <= 0.0 or perf_span <= 0.0:
            raise ValueError(
                f"Invalid GT score-span normalization for {perf.get('performance_source')}: "
                f"score_span={score_span}, perf_span={perf_span}"
            )
        scale = score_span / perf_span
    elif mode not in {"none", "off", "false", "0"}:
        raise ValueError(f"Unsupported eval_gt_time_normalization={normalization}")

    rows = []
    for shared, pedal in zip(shared_rows, pedal_rows):
        rows.append(
            [
                float(shared[0]) * scale,
                float(shared[1]) * scale,
                float(shared[2]),
                *[float(value) for value in pedal[:4]],
            ]
        )
    return rows, scale


def build_eval_gt_midis(config, loaded, gt_rel_paths, pitch, output_dir, score_stem):
    normalization = str(
        config.get(
            "eval_gt_time_normalization",
            config.get("evaluation_gt_time_normalization", "none"),
        )
        or "none"
    ).lower()
    if normalization in {"none", "off", "false", "0"}:
        return None

    gt_dir = output_dir / "ground_truth_midis"
    gt_dir.mkdir(parents=True, exist_ok=True)
    score_raw = loaded["score"]["score_raw"]
    output_paths = []
    scales = {}
    for gt_idx, gt_rel_path in enumerate(gt_rel_paths):
        perf = performance_by_source(loaded, gt_rel_path)
        raw7_rows, scale = _raw7_rows_for_eval_gt(perf, score_raw, normalization=normalization)
        midi_obj = note_features_to_midi(
            pitch=pitch,
            continuous=raw7_rows,
            target_ticks_per_beat=500,
            target_tempo=120,
            max_time_ms=config["max_time_ms"],
            normalized=False,
        )
        gt_path = gt_dir / f"{score_stem}__gt_{gt_idx:03d}.mid"
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        midi_obj.dump(str(gt_path))
        output_paths.append(str(gt_path.resolve()))
        scales[gt_rel_path] = scale
    return output_paths, scales


def predict_one_work(model, device, config, work, args, score_midi_dir, midi_dir):
    score_source = work["score_source"]
    pitch, continuous, score_shared_raw, loaded = load_score_from_node(
        Path(work["path"]),
        use_timing_scale_bit=config.get("use_timing_scale_bit", False),
        timing_control_mode=config.get("timing_control_mode", "dinr_floor_log"),
        timing_log_scale=config.get("timing_log_scale", 50.0),
        musical_feature_mode=config.get("musical_feature_mode", "musical4slot"),
        score_note_schema=config.get("score_note_input_schema", "integrated"),
        task_type="epr",
        disable_musical_features=config.get("disable_musical_features", False),
        pedal_control_dim=pedal_representation_dim(config.get("pedal_representation", "binary_4")),
    )
    score = loaded["score"]
    windows = build_windows(len(pitch), config["block_notes"], config["overlap_ratio"])
    style_creator_id, style_source_id = style_ids_for_work(config, loaded, args.performance_dataset)
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
    gt_midi_bundle = build_eval_gt_midis(
        config,
        loaded,
        gt_rel_paths,
        pitch,
        args.output_dir,
        score_stem,
    )
    score_midi_path = score_midi_dir / score_source
    original_gt_paths = [str((score_midi_dir / gt_rel).resolve()) for gt_rel in gt_rel_paths]
    if gt_midi_bundle is None:
        gt_paths = original_gt_paths
        eval_gt_scales = {}
        eval_gt_time_normalization = "none"
    else:
        gt_paths, eval_gt_scales = gt_midi_bundle
        eval_gt_time_normalization = str(
            config.get(
                "eval_gt_time_normalization",
                config.get("evaluation_gt_time_normalization", "none"),
            )
            or "none"
        )
    if args.oracle_gt_prefix_mode != "none" and args.merge_mode != "continuation":
        raise ValueError("--oracle-gt-prefix-mode requires --merge-mode continuation")
    perf_style_stats_mode = str(config.get("perf_style_stats_mode", "prefix") or "prefix").lower()
    requires_gt_window_style = bool(config.get("use_style_tokens", False)) and perf_style_stats_mode == "window"
    if args.oracle_gt_prefix_mode == "none" and not requires_gt_window_style:
        inference_plans = [
            {
                "sample_idx": sample_idx,
                "seed": args.seed + sample_idx,
                "performance_source": None,
                "performance_dataset": args.performance_dataset,
                "oracle_targets": None,
                "suffix": f"sample_{sample_idx:03d}",
            }
            for sample_idx in range(args.num_samples)
        ]
    else:
        inference_plans = []
        for perf_idx, gt_rel_path in enumerate(gt_rel_paths):
            perf = performance_by_source(loaded, gt_rel_path)
            inference_plans.append(
                {
                    "sample_idx": perf_idx,
                    "seed": stable_seed(args.seed, score_source, gt_rel_path, args.oracle_gt_prefix_mode),
                    "performance_source": gt_rel_path,
                    "performance_dataset": perf.get("performance_dataset") or args.performance_dataset,
                    "oracle_targets": labels_for_perf(config, perf, score_shared_raw),
                    "suffix": (
                        f"window_style_{perf_idx:03d}"
                        if args.oracle_gt_prefix_mode == "none"
                        else f"oracle_gt_{perf_idx:03d}"
                    ),
                }
            )

    for plan in inference_plans:
        sample_idx = int(plan["sample_idx"])
        sample_seed = int(plan["seed"])
        random.seed(sample_seed)
        torch.manual_seed(sample_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(sample_seed)

        sampling_strategy = resolve_sampling_strategy(args)
        plan_style_source_id = style_source_id
        if bool(config.get("use_style_tokens", False)) and plan.get("performance_dataset") is not None:
            source_vocab = config.get("style_source_vocab", {})
            plan_style_source_id = int(
                source_vocab.get(
                    str(plan.get("performance_dataset") or ""),
                    source_vocab.get("<unk>", 0),
                )
            )
        pred_continuous = batch_window_predictions(
            model=model,
            pitch=pitch,
            continuous=continuous,
            score_shared_raw=score_shared_raw,
            score=score,
            windows=windows,
            pitch_pad_id=config["pitch_pad_id"],
            device=device,
            batch_size=args.batch_size_windows,
            sampling_strategy=sampling_strategy,
            style_creator_id=style_creator_id,
            style_source_id=plan_style_source_id,
        ) if args.merge_mode == "average" else continuation_window_predictions(
            model=model,
            pitch=pitch,
            continuous=continuous,
            score_shared_raw=score_shared_raw,
            score=score,
            windows=windows,
            pitch_pad_id=config["pitch_pad_id"],
            device=device,
            sampling_strategy=sampling_strategy,
            drop_ratio=args.continuation_drop_ratio,
            style_creator_id=style_creator_id,
            style_source_id=plan_style_source_id,
            oracle_prefix_targets=(
                plan.get("oracle_targets")
                if args.oracle_gt_prefix_mode != "none"
                else None
            ),
            oracle_style_targets=(
                plan.get("oracle_targets")
                if (args.oracle_gt_prefix_mode == "decoder_and_style" or requires_gt_window_style)
                else None
            ),
            style_perf_stats_mode=perf_style_stats_mode,
        )
        raw_rows = _target7_to_raw7(
            torch.tensor(score_shared_raw, dtype=torch.float32),
            pred_continuous.float().cpu(),
            config=config,
        )
        pred_continuous_list = pred_continuous.float().cpu().tolist()
        raw_rows_list = raw_rows.tolist()
        midi_obj = note_features_to_midi(
            pitch=pitch,
            continuous=raw_rows_list,
            target_ticks_per_beat=500,
            target_tempo=120,
            max_time_ms=config["max_time_ms"],
            normalized=False,
        )
        raw_path = raw_dir / f"{score_stem}__{plan['suffix']}.json"
        target_key = "predicted_target7"
        timing_representation = "target7_floor_log_dev_velocity_binary4"
        raw_payload = {
            "score_source": score_source,
            "performance_source": plan.get("performance_source"),
            "protocol": args.protocol,
            "sampling_strategy": sampling_strategy,
            "deterministic_strategy": args.deterministic_strategy if args.protocol == "deterministic" else None,
            "sample_idx": sample_idx,
            "seed": sample_seed,
            "oracle_gt_prefix_mode": args.oracle_gt_prefix_mode,
            "timing_representation": timing_representation,
            "pitch": [int(value) for value in pitch],
            target_key: pred_continuous_list,
            "predicted_target7": pred_continuous_list if pred_continuous.shape[-1] == 7 else None,
            "reconstructed_raw7": raw_rows_list,
            "ground_truth_paths": gt_rel_paths,
            "ground_truth_eval_paths": gt_paths,
            "original_ground_truth_paths": original_gt_paths,
            "eval_gt_time_normalization": eval_gt_time_normalization,
            "eval_gt_time_scales": eval_gt_scales,
        }
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(raw_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        raw_output_paths.append(str(raw_path.resolve()))

        pred_path = midi_dir / f"{score_stem}__{plan['suffix']}.mid"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        midi_obj.dump(str(pred_path))
        prediction_paths.append(str(pred_path.resolve()))

    return {
        "score_source": score_source,
        "score_midi": str(score_midi_path.resolve()),
        "prediction_paths": prediction_paths,
        "raw_output_paths": raw_output_paths,
        "ground_truth_paths": gt_paths,
        "original_ground_truth_paths": original_gt_paths,
        "eval_gt_time_normalization": eval_gt_time_normalization,
        "eval_gt_time_scales": eval_gt_scales,
        "note_count": len(pitch),
        "num_windows": len(windows),
        "oracle_gt_prefix_mode": args.oracle_gt_prefix_mode,
    }


def worker_loop(worker_idx, args, config, score_midi_dir, job_queue, result_queue):
    random.seed(args.seed + worker_idx)
    torch.manual_seed(args.seed + worker_idx)
    if torch.cuda.is_available():
        worker_device = select_worker_device(args.device, worker_idx)
        if worker_device.type == "cuda":
            torch.cuda.set_device(worker_device)
        torch.cuda.manual_seed_all(args.seed + worker_idx)
    device = select_worker_device(args.device, worker_idx)
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


def load_score_source_filter(args):
    selected = []
    if args.score_source:
        selected.extend(str(item).strip() for item in args.score_source)
    if args.score_source_list is not None:
        for line in args.score_source_list.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            selected.append(line)
    selected = [item for item in selected if item]
    return selected or None


def filter_manifest_by_score_sources(manifest, score_sources):
    if not score_sources:
        return manifest
    wanted = set(score_sources)
    order = {score_source: idx for idx, score_source in enumerate(score_sources)}
    filtered = [item for item in manifest if item.get("score_source") in wanted]
    missing = [score_source for score_source in score_sources if score_source not in {item.get("score_source") for item in filtered}]
    if missing:
        raise ValueError(f"Requested score_source not found in manifest: {missing[0]}")
    filtered.sort(key=lambda item: order[item["score_source"]])
    return filtered


def sort_manifest_longest_first(manifest):
    return sorted(
        manifest,
        key=lambda item: (
            int(item.get("estimated_examples") or 0),
            int(item.get("note_count") or 0),
            len(item.get("windows") or []),
        ),
        reverse=True,
    )


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")

    config = load_config(args.config, args.checkpoint)
    for key in (
        "dlm_ioi_zero_sample_shrink_factor",
        "dlm_ioi_nonzero_sample_shrink_factor",
        "dlm_duration_sample_shrink_factor",
        "dlm_velocity_sample_shrink_factor",
    ):
        value = getattr(args, key)
        if value is not None:
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"--{key.replace('_', '-')} must be finite and >= 0")
            config[key] = float(value)
    if args.block_notes is not None:
        config["block_notes"] = int(args.block_notes)
    if args.overlap_ratio is not None:
        config["overlap_ratio"] = float(args.overlap_ratio)
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
        prepared_sidecar_tag=config.get("prepared_sidecar_tag"),
    )
    manifest = filter_manifest_by_performance_dataset(
        manifest,
        metadata_path=config["metadata_path"],
        split=args.split,
        performance_dataset=args.performance_dataset,
        exclude_performance_dataset=args.exclude_performance_dataset,
    )
    manifest = filter_manifest_by_score_sources(manifest, load_score_source_filter(args))
    if (args.performance_dataset is not None or args.exclude_performance_dataset is not None) and args.max_works is not None:
        manifest = manifest[: args.max_works]
    manifest = sort_manifest_longest_first(manifest)
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
    pair_list = build_pair_list(items)
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False))
    pair_list_path.write_text(json.dumps(pair_list, indent=2, ensure_ascii=False))
    print(f"Saved prediction manifest to {manifest_path}")
    print(f"Saved pair list to {pair_list_path}")


if __name__ == "__main__":
    main()
