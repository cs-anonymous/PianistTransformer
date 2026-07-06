import argparse
import hashlib
import json
import multiprocessing as mp
import random
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
from scipy.stats import wasserstein_distance
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_process.work_manifest import build_work_manifest
from src.inference.infer_inr_testset import (
    build_epr_score_input_rows,
    score_midi_dir_from_processed,
    select_device,
    select_worker_device,
)
from src.model.integrated_pianoformer import (
    _build_ar_note_continuous,
    _build_ar_special_note_ids,
    _materialize_epr_prediction,
    _shift_pitch_right,
    _target5_to_raw7,
)
from src.train.train_inr import (
    build_perf_style_prefix_cache,
    build_style_vocabs,
    create_model,
    performance_dev_velocity_pedal2_rows,
    performance_dev_velocity_pedal4_binary_rows,
    perf_style_stats_range_from_cache,
    perf_style_stats_from_cache,
    score_style_stats,
)


FEATURE_KEYS = [
    ("ioi", "ioi"),
    ("duration", "duration"),
    ("velocity", "velocity"),
    ("pedal_0", "pedal_0"),
    ("pedal_25", "pedal_25"),
    ("pedal_50", "pedal_50"),
    ("pedal_75", "pedal_75"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate INR rollout-length sensitivity with mixed GT/model decoder history.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--performance-dataset", type=str, default="ASAP")
    parser.add_argument(
        "--score-source",
        action="append",
        default=None,
        help="Optional score_source relative path to keep. Can be repeated.",
    )
    parser.add_argument(
        "--score-source-list",
        type=Path,
        default=None,
        help="Optional newline-delimited score_source list file.",
    )
    parser.add_argument("--max-works", type=int, default=None)
    parser.add_argument("--batch-size-windows", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--materialize-strategy", choices=["mean", "sample"], default="mean")
    parser.add_argument(
        "--feedback-strategy",
        type=str,
        default=None,
        help=(
            "Optional decoder feedback materialization strategy for partial rollout. "
            "Examples: mean, mode, greedy, sample, oracle-noised, teacher-forcing. "
            "If omitted, reuse --materialize-strategy."
        ),
    )
    parser.add_argument(
        "--feedback-timing-noise-scale",
        type=float,
        default=0.0,
        help=(
            "Optional timing-only noise multiplier applied on top of teacher-forced feedback rows. "
            "Noise is sampled from the model's own empirical residual bank in target space."
        ),
    )
    parser.add_argument(
        "--rollout-ks",
        type=str,
        default="0,1,2,4,8,16,32,64,128,256,full",
        help="Comma-separated rollout history lengths. Use 'full' for standard full AR.",
    )
    parser.add_argument(
        "--feedback-feature-ablation",
        type=str,
        default="none",
        choices=["none", "no_score_control", "timing_only", "timing_velocity"],
        help=(
            "Optional decoder feedback-row ablation for partial rollout. "
            "'no_score_control' removes note-specific score-control leakage from feedback rows; "
            "'timing_only' keeps only perf timing in feedback; "
            "'timing_velocity' keeps perf timing+velocity and replaces pedal with a neutral baseline."
        ),
    )
    return parser.parse_args()


def finite_mean(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if len(finite) else float("nan")


def feature_wasserstein(pred_values, gt_values):
    pred_values = np.asarray(pred_values, dtype=np.float64)
    gt_values = np.asarray(gt_values, dtype=np.float64)
    pred_values = pred_values[np.isfinite(pred_values)]
    gt_values = gt_values[np.isfinite(gt_values)]
    if len(pred_values) == 0 or len(gt_values) == 0:
        return float("nan")
    return float(wasserstein_distance(pred_values, gt_values))


def parse_rollout_ks(text):
    values = []
    seen = set()
    for raw in str(text).split(","):
        token = raw.strip().lower()
        if not token:
            continue
        if token == "full":
            value = None
        else:
            value = int(token)
            if value < 0:
                raise ValueError(f"rollout k must be >= 0, got {value}")
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    if not values:
        raise ValueError("No valid rollout ks were provided")
    return values


def rollout_k_label(value):
    return "full" if value is None else str(int(value))


def normalize_feedback_strategy(value, default_sampling_strategy="mean"):
    token = str(value or default_sampling_strategy).strip().lower().replace("_", "-")
    aliases = {
        "mode": "greedy",
        "argmax": "greedy",
        "median": "greedy",
        "median-or-mode": "greedy",
        "teacher-forcing": "teacher_forcing",
        "teacher_forced": "teacher_forcing",
        "tf": "teacher_forcing",
        "gt": "teacher_forcing",
        "oracle-noised": "oracle_noised",
    }
    return aliases.get(token, token.replace("-", "_"))


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
    found = {item.get("score_source") for item in filtered}
    missing = [score_source for score_source in score_sources if score_source not in found]
    if missing:
        raise ValueError(f"Requested score_source not found in manifest: {missing[0]}")
    filtered.sort(key=lambda item: order[item["score_source"]])
    return filtered


def stable_seed(base_seed, *parts):
    payload = "::".join([str(base_seed), *[str(part) for part in parts]])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def load_config(config_path, checkpoint):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    config["resume_path"] = str(checkpoint)
    if config.get("use_style_tokens", False):
        composer_vocab = config.get("style_composer_vocab")
        source_vocab = config.get("style_source_vocab")
        if composer_vocab is None or source_vocab is None:
            composer_vocab, source_vocab = build_style_vocabs(config["metadata_path"])
            config["style_composer_vocab"] = composer_vocab
            config["style_source_vocab"] = source_vocab
        config["style_creator_vocab_size"] = len(config["style_composer_vocab"])
        config["style_source_vocab_size"] = len(config["style_source_vocab"])
    return config


def selected_perfs(work, item):
    by_source = {perf.get("performance_source"): perf for perf in work.get("performances", [])}
    selected = [
        by_source[source]
        for source in item.get("selected_performance_sources", [])
        if source in by_source
    ]
    return selected


def labels_for_perf(config, perf, score_shared_raw):
    if str(config.get("pedal_representation", "")).lower() == "binary_4":
        labels = performance_dev_velocity_pedal4_binary_rows(
            perf,
            score_shared_raw,
            epr_timing_target=config.get("epr_timing_target", "deviation"),
            log_scale=float(config.get("timing_log_scale", 50.0)),
            split_zero_ioi_head=bool(config.get("split_zero_ioi_head", False)),
            ioi_nonzero_dev_scale=float(config.get("ioi_nonzero_dev_scale", 2.0)),
            ioi_zero_dev_scale=float(config.get("ioi_zero_dev_scale", 4.0)),
            pedal_binary_threshold=float(config.get("pedal_binary_threshold", 64.0)),
        )
    else:
        labels = performance_dev_velocity_pedal2_rows(
            perf,
            score_shared_raw,
            epr_timing_target=config.get("epr_timing_target", "deviation"),
            log_scale=float(config.get("timing_log_scale", 50.0)),
            split_zero_ioi_head=bool(config.get("split_zero_ioi_head", False)),
            ioi_nonzero_dev_scale=float(config.get("ioi_nonzero_dev_scale", 2.0)),
            ioi_zero_dev_scale=float(config.get("ioi_zero_dev_scale", 4.0)),
        )
    if labels is None:
        raise ValueError(f"Could not build labels for {perf.get('performance_source')}")
    return labels


def build_windows(total_notes, block_notes, overlap_ratio):
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
        if window in seen:
            continue
        deduped.append(window)
        seen.add(window)
    return deduped


def raw_arrays_from_rows(rows):
    rows = np.asarray(rows, dtype=np.float64)
    return {
        "ioi": rows[:, 0],
        "duration": rows[:, 1],
        "velocity": rows[:, 2],
        "pedal_0": rows[:, 3],
        "pedal_25": rows[:, 4],
        "pedal_50": rows[:, 5],
        "pedal_75": rows[:, 6],
    }


def continuous_feature_slices(config):
    score_dim = int(getattr(config, "score_control_feature_dim", getattr(config, "control_feature_dim", 5)))
    perf_dim = int(getattr(config, "performance_control_feature_dim", getattr(config, "control_feature_dim", 5) + 2))
    musical_dim = int(getattr(config, "musical_feature_dim", 12))
    mask_dim = int(getattr(config, "mask_feature_dim", 3))
    score_slice = slice(0, score_dim)
    perf_slice = slice(score_slice.stop, score_slice.stop + perf_dim)
    musical_slice = slice(perf_slice.stop, perf_slice.stop + musical_dim)
    mask_slice = slice(musical_slice.stop, musical_slice.stop + mask_dim)
    return score_slice, perf_slice, musical_slice, mask_slice


def masked_row_mean(rows, attention_mask):
    weights = attention_mask.to(dtype=rows.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (rows * weights).sum(dim=1, keepdim=True) / denom


def apply_feedback_feature_ablation(
    config,
    feedback_rows,
    score_rows,
    gt_perf_rows,
    attention_mask,
    ablation_mode,
):
    mode = str(ablation_mode or "none").strip().lower()
    if mode == "none" or feedback_rows.shape[1] == 0:
        return feedback_rows

    score_slice, perf_slice, _, _ = continuous_feature_slices(config)
    adjusted = feedback_rows.clone()
    score_baseline = masked_row_mean(score_rows[..., score_slice], attention_mask)
    perf_baseline = masked_row_mean(gt_perf_rows[..., perf_slice], attention_mask)

    if mode == "no_score_control":
        adjusted[..., score_slice] = score_baseline.expand(-1, adjusted.shape[1], -1)
        return adjusted

    if mode == "timing_only":
        # Keep perf timing dims and replace velocity/pedal with a neutral
        # piece-level baseline rather than an ambiguous literal zero.
        adjusted[..., perf_slice.start + 2 : perf_slice.stop] = perf_baseline[..., 2:].expand(
            -1,
            adjusted.shape[1],
            -1,
        )
        return adjusted

    if mode == "timing_velocity":
        # Keep perf timing+velocity and replace pedal with a neutral
        # piece-level baseline rather than an ambiguous literal zero.
        adjusted[..., perf_slice.start + 3 : perf_slice.stop] = perf_baseline[..., 3:].expand(
            -1,
            adjusted.shape[1],
            -1,
        )
        return adjusted

    raise ValueError(f"Unsupported feedback_feature_ablation={ablation_mode}")


def apply_teacher_forced_timing_noise(
    labels_continuous,
    sampled_noise,
    scale,
):
    scale = float(scale)
    if abs(scale) <= 0.0:
        return labels_continuous
    feedback = labels_continuous.clone()
    feedback[..., :2] = (feedback[..., :2] + float(scale) * sampled_noise[..., :2]).clamp(0.0, 1.0)
    return feedback


def metric_arrays_from_prediction(config, raw_rows, target_rows):
    arrays = raw_arrays_from_rows(raw_rows)
    if str(config.get("pedal_representation", "")).lower() != "binary_4":
        return arrays
    target_rows = np.asarray(target_rows, dtype=np.float64)
    if target_rows.ndim != 2 or target_rows.shape[1] < 7:
        return arrays
    # For binary_4 pedal, evaluate the four binary heads directly instead of the
    # reconstructed pseudo-continuous raw7 pedal columns.
    arrays["pedal_0"] = target_rows[:, 3]
    arrays["pedal_25"] = target_rows[:, 4]
    arrays["pedal_50"] = target_rows[:, 5]
    arrays["pedal_75"] = target_rows[:, 6]
    return arrays


def pp_wass_metrics(pred_arrays, gt_arrays):
    output = {}
    for metric_name, feature_name in FEATURE_KEYS:
        pred_pool = np.concatenate([item[feature_name] for item in pred_arrays]) if pred_arrays else np.asarray([], dtype=np.float64)
        gt_pool = np.concatenate([item[feature_name] for item in gt_arrays]) if gt_arrays else np.asarray([], dtype=np.float64)
        output[f"{metric_name}_wass"] = feature_wasserstein(pred_pool, gt_pool)
    pedal_keys = [f"{name}_wass" for name in ("pedal_0", "pedal_25", "pedal_50", "pedal_75")]
    output["pedal_wass"] = finite_mean([output[key] for key in pedal_keys])
    return output


def pn_wass_metrics(pred_arrays, gt_arrays):
    all_arrays = pred_arrays + gt_arrays
    output = {}
    for metric_name, feature_name in FEATURE_KEYS:
        usable = min((len(item[feature_name]) for item in all_arrays), default=0)
        note_wass = [
            feature_wasserstein(
                [item[feature_name][note_idx] for item in pred_arrays],
                [item[feature_name][note_idx] for item in gt_arrays],
            )
            for note_idx in range(usable)
        ]
        output[f"{metric_name}_wass"] = finite_mean(note_wass)
    pedal_keys = [f"{name}_wass" for name in ("pedal_0", "pedal_25", "pedal_50", "pedal_75")]
    output["pedal_wass"] = finite_mean([output[key] for key in pedal_keys])
    return output


def aggregate_score_metrics(score_rows, section):
    if not score_rows:
        return {}
    keys = sorted(score_rows[0][section].keys())
    output = {}
    for key in keys:
        output[key] = finite_mean([row[section].get(key, float("nan")) for row in score_rows])
    return output


def aggregate_pairwise_rows(rows):
    keys = ["loss", "ioi_wass", "duration_wass", "velocity_wass", "pedal_wass"]
    output = {"num_rows": int(len(rows))}
    for key in keys:
        output[key] = finite_mean([row.get(key, float("nan")) for row in rows])
    return output


def aggregate_pairwise_summaries(rows):
    keys = ["loss", "ioi_wass", "duration_wass", "velocity_wass", "pedal_wass"]
    total_weight = float(sum(max(0, int(row.get("num_rows", 0))) for row in rows))
    output = {"num_rows": int(total_weight)}
    for key in keys:
        weighted_values = []
        weights = []
        for row in rows:
            value = float(row.get(key, float("nan")))
            weight = max(0, int(row.get("num_rows", 0)))
            if weight <= 0 or not np.isfinite(value):
                continue
            weighted_values.append(value)
            weights.append(weight)
        if not weights:
            output[key] = float("nan")
        else:
            output[key] = float(np.average(np.asarray(weighted_values, dtype=np.float64), weights=np.asarray(weights, dtype=np.float64)))
    return output


def style_ids_for_work(config, work, performance_dataset):
    if not bool(config.get("use_style_tokens", False)):
        return None, None
    composer_vocab = config.get("style_composer_vocab", {})
    source_vocab = config.get("style_source_vocab", {})
    meta = work.get("meta", {}) if isinstance(work, dict) else {}
    composer = str(meta.get("composer") or "")
    source = str(performance_dataset or "unknown")
    creator_id = int(composer_vocab.get(composer, composer_vocab.get("<unk>", 0)))
    source_id = int(source_vocab.get(source, source_vocab.get("<unk>", 0)))
    return creator_id, source_id


def score_level_metrics_from_raw(score_source, pred_arrays, gt_arrays):
    return {
        "score_source": score_source,
        "num_predictions": len(pred_arrays),
        "num_ground_truth": len(gt_arrays),
        "pn_wass": pn_wass_metrics(pred_arrays, gt_arrays),
        "pp_wass": pp_wass_metrics(pred_arrays, gt_arrays),
    }


def build_window_style_kwargs(
    config,
    score,
    labels,
    start,
    end,
    style_creator_id,
    style_source_id,
    perf_style_cache,
):
    if not bool(config.get("use_style_tokens", False)):
        return None
    perf_style_stats_mode = str(config.get("perf_style_stats_mode", "prefix") or "prefix").lower()
    if perf_style_stats_mode == "window":
        perf_stats = perf_style_stats_range_from_cache(perf_style_cache, start, end)
        perf_is_pad = False
    else:
        perf_stats = perf_style_stats_from_cache(perf_style_cache, start)
        perf_is_pad = bool(start <= 0)
    return {
        "style_creator_id": int(style_creator_id or 0),
        "style_source_id": int(style_source_id or 0),
        "style_score_stats": score_style_stats(score, start, end),
        "style_perf_stats": perf_stats,
        "style_perf_is_pad": perf_is_pad,
    }


def build_batched_style_kwargs(window_style_rows, device):
    if not window_style_rows:
        return {}
    return {
        "style_creator_ids": torch.tensor(
            [row["style_creator_id"] for row in window_style_rows],
            dtype=torch.long,
            device=device,
        ),
        "style_source_ids": torch.tensor(
            [row["style_source_id"] for row in window_style_rows],
            dtype=torch.long,
            device=device,
        ),
        "style_score_stats": torch.tensor(
            [row["style_score_stats"] for row in window_style_rows],
            dtype=torch.float32,
            device=device,
        ),
        "style_perf_stats": torch.tensor(
            [row["style_perf_stats"] for row in window_style_rows],
            dtype=torch.float32,
            device=device,
        ),
        "style_perf_is_pad": torch.tensor(
            [bool(row["style_perf_is_pad"]) for row in window_style_rows],
            dtype=torch.bool,
            device=device,
        ),
    }


def _partial_rollout_t5(
    model,
    pitch_ids,
    continuous,
    score_shared_raw,
    labels_continuous,
    attention_mask,
    rollout_k,
    sampling_strategy,
    feedback_strategy,
    oracle_noise_bank,
    style_kwargs,
    feedback_feature_ablation="none",
    feedback_timing_noise_scale=0.0,
):
    config = model.config
    batch_size, seq_len = pitch_ids.shape
    output_dim = labels_continuous.shape[-1]

    score_note_embeds = model.note_encoder(pitch_ids, continuous)
    score_note_embeds = model._apply_style_to_note_embeds(score_note_embeds, **style_kwargs)
    score_context_embeds, context_attention_mask, _ = model._prepend_style_tokens(
        score_note_embeds,
        attention_mask,
        **style_kwargs,
    )
    encoder_outputs = model.model.encoder(
        attention_mask=context_attention_mask,
        inputs_embeds=score_context_embeds,
    )

    predictions = labels_continuous.new_zeros((batch_size, seq_len, output_dim))
    feedback_predictions = labels_continuous.new_zeros((batch_size, seq_len, output_dim))
    decoder_pitch_ids = _shift_pitch_right(config, pitch_ids, attention_mask)
    special_note_ids = _build_ar_special_note_ids(config, attention_mask)
    valid_lengths = attention_mask.long().sum(dim=1).clamp_min(1)
    gt_note_continuous = _build_ar_note_continuous(
        config,
        labels_continuous,
        score_shared_raw=score_shared_raw,
        task_type=config.task_type,
    )

    for step in range(seq_len):
        active = attention_mask[:, step].bool()
        if not active.any():
            continue

        prefix_len = step
        step_mask = attention_mask[:, : step + 1]
        if prefix_len <= 0:
            decoder_dim = int(getattr(config, "decoder_input_continuous_dim", config.input_continuous_dim))
            decoder_input_continuous = labels_continuous.new_zeros((batch_size, 1, decoder_dim))
        else:
            mixed_prefix = labels_continuous[:, :prefix_len].clone()
            pred_start = max(0, prefix_len - int(rollout_k))
            if pred_start < prefix_len:
                mixed_prefix[:, pred_start:prefix_len] = feedback_predictions[:, pred_start:prefix_len]
            prefix_note_continuous = _build_ar_note_continuous(
                config,
                mixed_prefix,
                score_shared_raw=score_shared_raw[:, :prefix_len],
                task_type=config.task_type,
            )
            prefix_note_continuous = apply_feedback_feature_ablation(
                config=config,
                feedback_rows=prefix_note_continuous,
                score_rows=continuous[:, :prefix_len],
                gt_perf_rows=gt_note_continuous[:, :prefix_len],
                attention_mask=attention_mask[:, :prefix_len],
                ablation_mode=feedback_feature_ablation,
            )
            decoder_input_continuous = prefix_note_continuous.new_zeros((batch_size, step + 1, prefix_note_continuous.shape[-1]))
            decoder_input_continuous[:, 1:] = prefix_note_continuous

        decoder_inputs_embeds = model.decoder_note_encoder(
            decoder_pitch_ids[:, : step + 1],
            decoder_input_continuous,
            special_note_ids=special_note_ids[:, : step + 1],
        )
        decoder_inputs_embeds = model._apply_style_to_decoder_inputs(
            decoder_inputs_embeds,
            **style_kwargs,
        )
        decoder_outputs = model.model(
            attention_mask=context_attention_mask,
            decoder_attention_mask=step_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=score_context_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
        )
        decoder_hidden = model._apply_style_to_decoder_hidden(
            decoder_outputs.last_hidden_state,
            **style_kwargs,
        )
        step_raw = model.continuous_decoder(decoder_hidden[:, -1:, :])
        step_pred = _materialize_epr_prediction(
            config,
            step_raw,
            sampling_strategy=sampling_strategy,
            score_shared_raw=score_shared_raw[:, step : step + 1],
        )
        if feedback_strategy == "teacher_forcing":
            step_feedback = labels_continuous[:, step : step + 1]
        elif feedback_strategy == "oracle_noised":
            noise_indices = torch.floor(torch.rand(batch_size, device=labels_continuous.device) * valid_lengths.float()).long()
            noise_indices = noise_indices.clamp(min=0, max=max(seq_len - 1, 0))
            sampled_noise = oracle_noise_bank[torch.arange(batch_size, device=labels_continuous.device), noise_indices]
            step_feedback = (labels_continuous[:, step : step + 1] + sampled_noise.unsqueeze(1)).clamp(0.0, 1.0)
        elif feedback_strategy == normalize_feedback_strategy(sampling_strategy, default_sampling_strategy=sampling_strategy):
            step_feedback = step_pred
        else:
            step_feedback = _materialize_epr_prediction(
                config,
                step_raw,
                sampling_strategy=feedback_strategy,
                score_shared_raw=score_shared_raw[:, step : step + 1],
            )
        if feedback_strategy == "teacher_forcing" and float(feedback_timing_noise_scale) > 0.0:
            noise_indices = torch.floor(torch.rand(batch_size, device=labels_continuous.device) * valid_lengths.float()).long()
            noise_indices = noise_indices.clamp(min=0, max=max(seq_len - 1, 0))
            sampled_noise = oracle_noise_bank[torch.arange(batch_size, device=labels_continuous.device), noise_indices]
            step_feedback = apply_teacher_forced_timing_noise(
                step_feedback,
                sampled_noise.unsqueeze(1),
                feedback_timing_noise_scale,
            )
        predictions[:, step : step + 1] = torch.where(
            active.view(batch_size, 1, 1),
            step_pred,
            predictions[:, step : step + 1],
        )
        feedback_predictions[:, step : step + 1] = torch.where(
            active.view(batch_size, 1, 1),
            step_feedback,
            feedback_predictions[:, step : step + 1],
        )

    return predictions


def predict_rollout_batch(
    model,
    pitch_ids,
    continuous,
    score_shared_raw,
    labels_continuous,
    attention_mask,
    rollout_k,
    sampling_strategy,
    feedback_strategy,
    style_kwargs,
    feedback_feature_ablation="none",
    feedback_timing_noise_scale=0.0,
):
    if rollout_k == 0:
        with torch.no_grad():
            outputs = model(
                pitch_ids=pitch_ids,
                continuous=continuous,
                score_shared_raw=score_shared_raw,
                labels_continuous=labels_continuous,
                attention_mask=attention_mask,
                continuous_sampling_strategy=sampling_strategy,
                **style_kwargs,
            )
            predictions = _materialize_epr_prediction(
                model.config,
                outputs.logits,
                sampling_strategy=sampling_strategy,
                score_shared_raw=score_shared_raw,
            )
        return predictions.detach().float().cpu(), float(outputs.loss.detach().float().cpu()) if outputs.loss is not None else float("nan")

    if rollout_k is None:
        with torch.no_grad():
            predictions = model.predict_performance_continuous(
                pitch_ids=pitch_ids,
                continuous=continuous,
                score_shared_raw=score_shared_raw,
                attention_mask=attention_mask,
                sampling_strategy=sampling_strategy,
                **style_kwargs,
            )
        return predictions.detach().float().cpu(), float("nan")

    if not hasattr(model, "model"):
        raise ValueError("Partial rollout diagnostic currently supports T5/T5Gemma INR checkpoints only")

    oracle_noise_bank = None
    needs_oracle_noise_bank = feedback_strategy == "oracle_noised" or float(feedback_timing_noise_scale) > 0.0
    if needs_oracle_noise_bank:
        with torch.no_grad():
            oracle_outputs = model(
                pitch_ids=pitch_ids,
                continuous=continuous,
                score_shared_raw=score_shared_raw,
                labels_continuous=labels_continuous,
                attention_mask=attention_mask,
                continuous_sampling_strategy="mean",
                **style_kwargs,
            )
            oracle_center = _materialize_epr_prediction(
                model.config,
                oracle_outputs.logits,
                sampling_strategy="mean",
                score_shared_raw=score_shared_raw,
            )
        oracle_noise_bank = (oracle_center - labels_continuous).detach()

    with torch.no_grad():
        predictions = _partial_rollout_t5(
            model=model,
            pitch_ids=pitch_ids,
            continuous=continuous,
            score_shared_raw=score_shared_raw,
            labels_continuous=labels_continuous,
            attention_mask=attention_mask,
            rollout_k=rollout_k,
            sampling_strategy=sampling_strategy,
            feedback_strategy=feedback_strategy,
            oracle_noise_bank=oracle_noise_bank,
            style_kwargs=style_kwargs,
            feedback_feature_ablation=feedback_feature_ablation,
            feedback_timing_noise_scale=feedback_timing_noise_scale,
        )
    return predictions.detach().float().cpu(), float("nan")


def predict_score_for_k(
    model,
    device,
    config,
    score,
    pitch,
    score_inputs,
    score_shared_raw,
    labels,
    windows,
    rollout_k,
    batch_size,
    sampling_strategy,
    feedback_strategy,
    style_creator_id,
    style_source_id,
    feedback_feature_ablation="none",
    feedback_timing_noise_scale=0.0,
):
    total_notes = len(pitch)
    pred_sum = None
    pred_count = torch.zeros(total_notes, 1, dtype=torch.float32)
    loss_values = []
    perf_style_cache = build_perf_style_prefix_cache(labels) if config.get("use_style_tokens", False) else None

    for batch_start in range(0, len(windows), batch_size):
        batch_windows = windows[batch_start : batch_start + batch_size]
        pitch_tensors = []
        score_tensors = []
        raw_tensors = []
        label_tensors = []
        style_rows = []
        lengths = []

        for start, end in batch_windows:
            pitch_tensors.append(torch.tensor(pitch[start:end], dtype=torch.long))
            score_tensors.append(torch.tensor(score_inputs[start:end], dtype=torch.float32))
            raw_tensors.append(torch.tensor(score_shared_raw[start:end], dtype=torch.float32))
            label_tensors.append(torch.tensor(labels[start:end], dtype=torch.float32))
            if config.get("use_style_tokens", False):
                style_rows.append(
                    build_window_style_kwargs(
                        config=config,
                        score=score,
                        labels=labels,
                        start=start,
                        end=end,
                        style_creator_id=style_creator_id,
                        style_source_id=style_source_id,
                        perf_style_cache=perf_style_cache,
                    )
                )
            lengths.append(end - start)

        pitch_ids = pad_sequence(pitch_tensors, batch_first=True, padding_value=config["pitch_pad_id"]).to(device)
        continuous = pad_sequence(score_tensors, batch_first=True, padding_value=0.0).to(device)
        score_raw = pad_sequence(raw_tensors, batch_first=True, padding_value=0.0).to(device)
        label_batch = pad_sequence(label_tensors, batch_first=True, padding_value=0.0).to(device)
        attention_mask = (pitch_ids != config["pitch_pad_id"]).long()
        style_kwargs = build_batched_style_kwargs(style_rows, device)

        pred, loss_value = predict_rollout_batch(
            model=model,
            pitch_ids=pitch_ids,
            continuous=continuous,
            score_shared_raw=score_raw,
            labels_continuous=label_batch,
            attention_mask=attention_mask,
            rollout_k=rollout_k,
            sampling_strategy=sampling_strategy,
            feedback_strategy=feedback_strategy,
            style_kwargs=style_kwargs,
            feedback_feature_ablation=feedback_feature_ablation,
            feedback_timing_noise_scale=feedback_timing_noise_scale,
        )
        if np.isfinite(loss_value):
            loss_values.append(loss_value)
        if pred_sum is None:
            pred_sum = torch.zeros(total_notes, pred.shape[-1], dtype=torch.float32)

        for idx, (start, end) in enumerate(batch_windows):
            length = lengths[idx]
            pred_sum[start:end] += pred[idx, :length]
            pred_count[start:end] += 1.0

    if pred_sum is None:
        raise ValueError("No windows were processed")
    return pred_sum / pred_count.clamp_min(1.0), finite_mean(loss_values)


def predict_work(model, device, config, item, args, rollout_ks):
    work = json.loads(Path(item["path"]).read_text(encoding="utf-8"))
    score = work["score"]
    pitch = score["pitch"]
    score_shared_raw = [row[:3] for row in score["score_raw"]]
    score_inputs = build_epr_score_input_rows(
        score,
        use_timing_scale_bit=config.get("use_timing_scale_bit", True),
        timing_control_mode=config.get("timing_control_mode"),
        log_scale=float(config.get("timing_log_scale", 50.0)),
        musical_feature_mode=config.get("musical_feature_mode", "categorical"),
        score_note_schema=config.get("score_note_input_schema", "integrated"),
    )
    windows = build_windows(len(pitch), config["block_notes"], config["overlap_ratio"])
    perfs = selected_perfs(work, item)
    style_creator_id, _ = style_ids_for_work(config, work, args.performance_dataset)
    source_vocab = config.get("style_source_vocab", {})

    by_k = {}
    for rollout_k in rollout_ks:
        k_label = rollout_k_label(rollout_k)
        pred_arrays = []
        gt_arrays = []
        pairwise_rows = []
        for perf_idx, perf in enumerate(perfs):
            labels = labels_for_perf(config, perf, score_shared_raw)
            perf_dataset = perf.get("performance_dataset") or args.performance_dataset
            style_source_id = int(source_vocab.get(str(perf_dataset or ""), source_vocab.get("<unk>", 0)))

            sample_seed = stable_seed(
                args.seed,
                item["score_source"],
                perf.get("performance_source"),
                k_label,
                args.materialize_strategy,
            )
            random.seed(sample_seed)
            torch.manual_seed(sample_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(sample_seed)

            pred_target, mean_loss = predict_score_for_k(
                model=model,
                device=device,
                config=config,
                score=score,
                pitch=pitch,
                score_inputs=score_inputs,
                score_shared_raw=score_shared_raw,
                labels=labels,
                windows=windows,
                rollout_k=rollout_k,
                batch_size=args.batch_size_windows,
                sampling_strategy=args.materialize_strategy,
                feedback_strategy=normalize_feedback_strategy(
                    args.feedback_strategy,
                    default_sampling_strategy=args.materialize_strategy,
                ),
                style_creator_id=style_creator_id,
                style_source_id=style_source_id,
                feedback_feature_ablation=args.feedback_feature_ablation,
                feedback_timing_noise_scale=args.feedback_timing_noise_scale,
            )
            pred_raw = _target5_to_raw7(
                torch.tensor(score_shared_raw, dtype=torch.float32),
                pred_target.float().cpu(),
                config=config,
            ).cpu().numpy()
            target_raw = _target5_to_raw7(
                torch.tensor(score_shared_raw, dtype=torch.float32),
                torch.tensor(labels, dtype=torch.float32),
                config=config,
            ).cpu().numpy()

            pred_note_arrays = metric_arrays_from_prediction(config, pred_raw, pred_target.float().cpu().numpy())
            gt_note_arrays = metric_arrays_from_prediction(config, target_raw, labels)
            pred_arrays.append(pred_note_arrays)
            gt_arrays.append(gt_note_arrays)
            pairwise_rows.append(
                {
                    "performance_source": perf.get("performance_source"),
                    "loss": mean_loss,
                    "ioi_wass": feature_wasserstein(pred_note_arrays["ioi"], gt_note_arrays["ioi"]),
                    "duration_wass": feature_wasserstein(pred_note_arrays["duration"], gt_note_arrays["duration"]),
                    "velocity_wass": feature_wasserstein(pred_note_arrays["velocity"], gt_note_arrays["velocity"]),
                    "pedal_wass": finite_mean(
                        [
                            feature_wasserstein(pred_note_arrays[key], gt_note_arrays[key])
                            for key in ("pedal_0", "pedal_25", "pedal_50", "pedal_75")
                        ]
                    ),
                }
            )

        score_metrics = score_level_metrics_from_raw(item["score_source"], pred_arrays, gt_arrays)
        by_k[k_label] = {
            "score_metrics": score_metrics,
            "pairwise": aggregate_pairwise_rows(pairwise_rows),
            "note_count": len(pitch),
            "num_windows": len(windows),
            "num_performances": len(perfs),
        }

    return {
        "score_source": item["score_source"],
        "by_k": by_k,
    }


def aggregate_items(items, rollout_ks):
    aggregate = {}
    for rollout_k in rollout_ks:
        k_label = rollout_k_label(rollout_k)
        score_rows = [item["by_k"][k_label]["score_metrics"] for item in items]
        pairwise_rows = [item["by_k"][k_label]["pairwise"] for item in items]
        aggregate[k_label] = {
            "num_scores": int(len(score_rows)),
            "pairwise": aggregate_pairwise_summaries(pairwise_rows),
            "pp_wass": aggregate_score_metrics(score_rows, "pp_wass"),
            "pn_wass": aggregate_score_metrics(score_rows, "pn_wass"),
        }
    return aggregate


def write_curve_csv(path, aggregate_by_k, rollout_ks):
    import csv

    rows = []
    for rollout_k in rollout_ks:
        k_label = rollout_k_label(rollout_k)
        aggregate = aggregate_by_k[k_label]
        row = {
            "k": k_label,
            "pairwise_ioi_wass": aggregate["pairwise"].get("ioi_wass"),
            "pairwise_duration_wass": aggregate["pairwise"].get("duration_wass"),
            "pairwise_velocity_wass": aggregate["pairwise"].get("velocity_wass"),
            "pairwise_pedal_wass": aggregate["pairwise"].get("pedal_wass"),
            "pp_ioi_wass": aggregate["pp_wass"].get("ioi_wass"),
            "pp_duration_wass": aggregate["pp_wass"].get("duration_wass"),
            "pp_velocity_wass": aggregate["pp_wass"].get("velocity_wass"),
            "pp_pedal_wass": aggregate["pp_wass"].get("pedal_wass"),
            "pn_ioi_wass": aggregate["pn_wass"].get("ioi_wass"),
            "pn_duration_wass": aggregate["pn_wass"].get("duration_wass"),
            "pn_velocity_wass": aggregate["pn_wass"].get("velocity_wass"),
            "pn_pedal_wass": aggregate["pn_wass"].get("pedal_wass"),
        }
        rows.append(row)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_curve(path, title, aggregate_by_k, rollout_ks, section):
    labels = [rollout_k_label(value) for value in rollout_ks]
    x = np.arange(len(labels))
    features = [
        ("ioi_wass", "IOI"),
        ("duration_wass", "Dur"),
        ("velocity_wass", "Vel"),
        ("pedal_wass", "Pedal"),
    ]

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for key, name in features:
        y = [aggregate_by_k[label][section].get(key, float("nan")) for label in labels]
        ax.plot(x, y, marker="o", linewidth=2.0, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Rollout length k")
    ax.set_ylabel("Wasserstein distance")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def worker_loop(worker_idx, args, config, rollout_ks, job_queue, result_queue):
    random.seed(args.seed + worker_idx)
    torch.manual_seed(args.seed + worker_idx)
    device = select_worker_device(args.device, worker_idx)
    print(f"Worker {worker_idx} using device: {device}", flush=True)
    model = create_model(config)
    model.to(device)
    model.eval()
    while True:
        job = job_queue.get()
        if job is None:
            break
        job_idx, item = job
        try:
            result = predict_work(model, device, config, item, args, rollout_ks)
            result_queue.put((job_idx, result, None))
        except Exception as exc:  # noqa: BLE001
            result_queue.put((job_idx, None, repr(exc)))


def run_dynamic_pool(args, config, manifest, rollout_ks):
    ctx = mp.get_context("spawn")
    job_queue = ctx.Queue()
    result_queue = ctx.Queue()
    workers = [
        ctx.Process(target=worker_loop, args=(idx, args, config, rollout_ks, job_queue, result_queue))
        for idx in range(args.num_workers)
    ]
    for worker in workers:
        worker.start()
    for job_idx, item in enumerate(manifest):
        job_queue.put((job_idx, item))
    for _ in workers:
        job_queue.put(None)

    by_idx = {}
    with tqdm(total=len(manifest), desc="rollout-k pool") as progress:
        for _ in range(len(manifest)):
            job_idx, item, error = result_queue.get()
            if error is not None:
                for worker in workers:
                    worker.terminate()
                raise RuntimeError(f"Worker failed on job {job_idx}: {error}")
            by_idx[job_idx] = item
            progress.update(1)
    for worker in workers:
        worker.join()
        if worker.exitcode != 0:
            raise RuntimeError(f"Worker {worker.pid} exited with code {worker.exitcode}")
    return [by_idx[idx] for idx in range(len(manifest))]


def run_single(args, config, manifest, rollout_ks):
    device = select_device(args.device)
    print(f"Using device: {device}")
    model = create_model(config)
    model.to(device)
    model.eval()
    return [
        predict_work(model, device, config, item, args, rollout_ks)
        for item in tqdm(manifest, desc="rollout-k")
    ]


def main():
    args = parse_args()
    if str(load_config(args.config, args.checkpoint).get("task_type", "epr")).lower() != "epr":
        raise ValueError("This rollout-k diagnostic currently supports EPR checkpoints only")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    rollout_ks = parse_rollout_ks(args.rollout_ks)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config, args.checkpoint)
    manifest = build_work_manifest(
        metadata_path=config["metadata_path"],
        refined_dir=config["refined_dir"],
        split=args.split,
        block_notes=config["block_notes"],
        overlap_ratio=config["overlap_ratio"],
        min_notes=config["min_notes"],
        max_works=args.max_works,
        skip_work_paths=config.get("skip_work_paths"),
        performance_dataset=args.performance_dataset,
    )
    manifest = filter_manifest_by_score_sources(manifest, load_score_source_filter(args))
    _ = score_midi_dir_from_processed(config["refined_dir"])

    if args.num_workers > 1:
        items = run_dynamic_pool(args, config, manifest, rollout_ks)
    else:
        items = run_single(args, config, manifest, rollout_ks)

    aggregate_by_k = aggregate_items(items, rollout_ks)
    summary = {
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "performance_dataset": args.performance_dataset,
        "materialize_strategy": args.materialize_strategy,
        "feedback_strategy": normalize_feedback_strategy(
            args.feedback_strategy,
            default_sampling_strategy=args.materialize_strategy,
        ),
        "feedback_feature_ablation": args.feedback_feature_ablation,
        "feedback_timing_noise_scale": float(args.feedback_timing_noise_scale),
        "rollout_ks": [rollout_k_label(value) for value in rollout_ks],
        "num_scores": int(len(items)),
        "aggregate_by_k": aggregate_by_k,
        "items": items,
    }

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_curve_csv(args.output_dir / "curve.csv", aggregate_by_k, rollout_ks)
    plot_curve(
        args.output_dir / "curve_pp_wass.png",
        "Rollout Length vs Score-Level PP Wass",
        aggregate_by_k,
        rollout_ks,
        section="pp_wass",
    )
    plot_curve(
        args.output_dir / "curve_pn_wass.png",
        "Rollout Length vs Score-Level PN Wass",
        aggregate_by_k,
        rollout_ks,
        section="pn_wass",
    )
    plot_curve(
        args.output_dir / "curve_pairwise.png",
        "Rollout Length vs Pairwise Wass",
        aggregate_by_k,
        rollout_ks,
        section="pairwise",
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "items"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
