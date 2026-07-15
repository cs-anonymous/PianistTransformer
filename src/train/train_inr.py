import argparse
import bisect
import datetime
import gc
import hashlib
import json
import math
import os
import random
import shutil
import re
import subprocess
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.sampler import SequentialSampler
from tqdm.auto import tqdm
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT_DIR))

from src.model.integrated_pianoformer import (
    IntegratedPianoT5Gemma,
    IntegratedPianoT5GemmaConfig,
    IntegratedPianoTransformer,
    _compute_integrated_loss_components,
    _dlm_bin_centers,
    _dlm_log_bin_probs,
    _materialize_epr_prediction,
    _split_epr_mixture_params,
    _zero_score_ioi_mask,
)
from src.data_process.work_manifest import build_work_manifest
from src.utils.func import filter_valid_args
from src.utils.inr_midi import raw_rows_to_epr_bins, raw_rows_to_model_continuous


os.environ["WANDB_PROJECT"] = "pianist-transformer"


def release_cuda_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def print_model_parameters(model):
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-Trainable Parameters: {(total_params - trainable_params):,}")
    print("--------------------------------------------------")
    print(f"Total Parameters (M):     {total_params / 1_000_000:.2f}M")
    print(f"Trainable Parameters (M): {trainable_params / 1_000_000:.2f}M")
    print("--------------------------------------------------")


def load_torch_state_dict(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_dir():
        safetensors_path = checkpoint_path / "model.safetensors"
        pytorch_path = checkpoint_path / "pytorch_model.bin"
        if safetensors_path.exists():
            from safetensors.torch import load_file

            return load_file(str(safetensors_path))
        if pytorch_path.exists():
            checkpoint = torch.load(pytorch_path, map_location="cpu")
        else:
            raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin in {checkpoint_path}")
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint


OUTPUT_HEAD_PREFIXES = (
    "continuous_decoder.ioi_head",
    "continuous_decoder.ioi_zero_head",
    "continuous_decoder.duration_head",
    "continuous_decoder.duration_zero_head",
    "continuous_decoder.velocity_head",
    "continuous_decoder.shared_head",
    "continuous_decoder.shared_extra_head",
    "continuous_decoder.pedal_head",
    "continuous_decoder.generic_head",
)


INPUT_EMBEDDING_PREFIXES = (
    "encoder_note_encoder",
    "decoder_note_encoder",
    "note_encoder",
)


def is_output_head_parameter(name):
    normalized = name.removeprefix("module.")
    return normalized.startswith(OUTPUT_HEAD_PREFIXES)


def is_input_embedding_parameter(name):
    normalized = name.removeprefix("module.")
    if not normalized.startswith(INPUT_EMBEDDING_PREFIXES):
        return False
    return any(
        token in normalized
        for token in (
            "score_control_projection",
            "performance_control_projection",
            "musical_projection",
            "mask_projection",
            "continuous_mlp",
        )
    )


def filter_resume_state_dict(model, state_dict, train_config):
    model_state = model.state_dict()
    filtered = OrderedDict()
    reset_heads = bool(train_config.get("reset_output_heads_on_resume", False))
    ignore_mismatched = bool(train_config.get("ignore_mismatched_resume_shapes", True))
    skipped_heads = []
    skipped_mismatched = []

    for key, value in state_dict.items():
        normalized_key = key.removeprefix("module.")
        if reset_heads and is_output_head_parameter(normalized_key):
            skipped_heads.append(key)
            continue
        target = model_state.get(normalized_key)
        if target is not None and tuple(target.shape) != tuple(value.shape):
            if ignore_mismatched:
                skipped_mismatched.append((key, tuple(value.shape), tuple(target.shape)))
                continue
        filtered[normalized_key] = value

    if skipped_heads:
        print(f"Reset output heads on resume: skipped {len(skipped_heads)} head tensors")
        print(f"  head examples: {skipped_heads[:8]}")
    if skipped_mismatched:
        print(f"Skipped mismatched resume tensors: {len(skipped_mismatched)}")
        for key, src_shape, dst_shape in skipped_mismatched[:12]:
            print(f"  {key}: checkpoint{src_shape} -> model{dst_shape}")
    return filtered


def apply_trainable_parameter_policy(model, train_config):
    if train_config.get("train_scale_only", False):
        distribution = str(train_config.get("epr_distribution", "dlm")).lower()
        components = int(train_config.get("epr_mixture_components", 1))
        velocity_distribution = str(
            train_config.get("velocity_distribution", distribution)
        ).lower()
        velocity_components = int(
            train_config.get("velocity_mixture_components") or components
        )
        if distribution != "dlm" or components != 1:
            raise ValueError(
                "train_scale_only currently requires epr_distribution=dlm and "
                "epr_mixture_components=1"
            )
        if velocity_distribution != "dlm" or velocity_components != 1:
            raise ValueError(
                "train_scale_only currently requires velocity DLM K=1"
            )

        for param in model.parameters():
            param.requires_grad = False

        # A DLM K=1 final projection has rows [mixture_logit, loc, log_scale].
        # Keep the original module/state-dict layout, but mask gradients so only
        # log_scale is optimized. Stage-2 configs must set weight_decay=0 to
        # prevent decoupled AdamW decay from moving the two frozen rows.
        hook_handles = []
        trainable = []
        for head_name in ("ioi_head", "duration_head", "velocity_head"):
            head = getattr(model.continuous_decoder, head_name)
            linear = next(
                (module for module in reversed(list(head.modules())) if isinstance(module, torch.nn.Linear)),
                None,
            )
            if linear is None or linear.out_features != 3:
                raise ValueError(
                    f"Expected {head_name} final projection to have 3 DLM K=1 rows; "
                    f"got {None if linear is None else linear.out_features}"
                )

            def scale_row_only(grad):
                masked = torch.zeros_like(grad)
                masked[2] = grad[2]
                return masked

            linear.weight.requires_grad = True
            hook_handles.append(linear.weight.register_hook(scale_row_only))
            trainable.append(f"continuous_decoder.{head_name}.final.weight[2]")
            if linear.bias is not None:
                linear.bias.requires_grad = True
                hook_handles.append(linear.bias.register_hook(scale_row_only))
                trainable.append(f"continuous_decoder.{head_name}.final.bias[2]")
        model._scale_only_gradient_hook_handles = hook_handles
        print(f"Freeze policy: DLM K=1 scale rows only ({len(trainable)} row slices trainable)")
        print(f"  trainable slices: {trainable}")
        if float(train_config.get("weight_decay", 0.0) or 0.0) != 0.0:
            raise ValueError("train_scale_only requires weight_decay=0")
        return

    if train_config.get("freeze_non_output_heads", False):
        trainable = []
        train_input_embedding = bool(train_config.get("freeze_train_input_embedding", False))
        for name, param in model.named_parameters():
            param.requires_grad = is_output_head_parameter(name) or (
                train_input_embedding and is_input_embedding_parameter(name)
            )
            if param.requires_grad:
                trainable.append(name)
        detail = "output heads + input embedding projections" if train_input_embedding else "output heads only"
        print(f"Freeze policy: {detail} ({len(trainable)} tensors trainable)")
        print(f"  trainable examples: {trainable[:12]}")
        return

    trainable_regex = train_config.get("trainable_parameter_regex")
    if trainable_regex:
        pattern = re.compile(trainable_regex)
        trainable = []
        for name, param in model.named_parameters():
            param.requires_grad = bool(pattern.search(name))
            if param.requires_grad:
                trainable.append(name)
        print(f"Freeze policy: regex={trainable_regex!r} ({len(trainable)} tensors trainable)")
        print(f"  trainable examples: {trainable[:12]}")


def default_input_continuous_dim(
    task_type,
    input_feature_mode,
    score_feature_dim=8,
    continuous_dim=5,
    musical_feature_mode="categorical",
):
    if input_feature_mode == "integrated":
        if task_type == "epr":
            return integrated_epr_input_dim(musical_feature_mode=musical_feature_mode)
        if task_type == "csr":
            return integrated_epr_input_dim(musical_feature_mode="continuous")
    return continuous_dim


def infer_input_feature_mode(config):
    mode = config.get("input_feature_mode")
    if mode is not None:
        return str(mode).lower()
    return "integrated"


def resolve_timing_control_mode(timing_control_mode="log_scaled", use_timing_scale_bit=False):
    if timing_control_mode is None:
        return "log_scaled"
    mode = str(timing_control_mode).lower()
    valid_modes = {
        "piecewise_scale_bit",
        "piecewise_single",
        "dual_log_linear",
        "dual_clip_linear",
        "log_scaled",
        "floor_log",
        "dinr_floor_log",
        "raw_log",
    }
    if mode not in valid_modes:
        raise ValueError(f"Unsupported timing_control_mode={timing_control_mode}")
    return mode


def timing_control_feature_dim(timing_control_mode="log_scaled", use_timing_scale_bit=False):
    mode = resolve_timing_control_mode(
        timing_control_mode=timing_control_mode,
        use_timing_scale_bit=use_timing_scale_bit,
    )
    if mode == "raw_log":
        return 5
    return 3 if mode in {"piecewise_single", "log_scaled", "floor_log", "dinr_floor_log"} else 5


def musical_feature_dim(musical_feature_mode="categorical"):
    mode = str(musical_feature_mode).lower()
    if mode in {"none", "off", "disabled", "no_musical", "nomus"}:
        return 0
    if mode == "continuous":
        return 12
    if mode in {
        "categorical",
        "categorical51",
        "musical51",
        "musical51_full",
        "musical51_onset_only",
        "musical51_annotation_only",
        "musical51_duration_only",
        "musical51_onset_annotation",
        "musical51_no_duration",
        "musical51_no_length",
        "musical51_no_duration_length",
    }:
        return 51
    if mode in {"musical145_onset_annotation", "onset145_annotation"}:
        return 151
    if mode in {"categorical62", "musical62"}:
        return 62
    raise ValueError(f"Unsupported musical_feature_mode={musical_feature_mode}")


def score_note_input_schema(config_or_value=None):
    if isinstance(config_or_value, dict):
        return str(config_or_value.get("score_note_input_schema", "integrated")).lower()
    if hasattr(config_or_value, "score_note_input_schema"):
        return str(getattr(config_or_value, "score_note_input_schema", "integrated")).lower()
    if config_or_value is None:
        return "integrated"
    return str(config_or_value).lower()


def decoder_note_input_schema(config_or_value=None):
    if isinstance(config_or_value, dict):
        return str(config_or_value.get("decoder_note_input_schema", "integrated")).lower()
    if hasattr(config_or_value, "decoder_note_input_schema"):
        return str(getattr(config_or_value, "decoder_note_input_schema", "integrated")).lower()
    if config_or_value is None:
        return "integrated"
    return str(config_or_value).lower()


def dagger_target_columns(mode, output_dim=7):
    mode = str(mode or "full").lower()
    output_dim = int(output_dim)
    if mode == "full":
        return list(range(output_dim))
    if mode == "timing":
        return [0, 1]
    if mode == "ioi":
        return [0]
    if mode == "duration":
        return [1]
    if mode == "velocity":
        return [4] if output_dim >= 9 else [2]
    if mode == "pedal":
        return list(range(5, min(output_dim, 9))) if output_dim >= 9 else list(range(3, min(output_dim, 7)))
    raise ValueError(f"Unsupported DAgger replacement mode: {mode}")


def normalize_dagger_replacement_weights(weights=None):
    default = {
        "full": 0.30,
        "timing": 0.20,
        "ioi": 0.10,
        "duration": 0.10,
        "velocity": 0.15,
        "pedal": 0.15,
    }
    raw = dict(default if weights is None else weights)
    cleaned = {str(key).lower(): float(value) for key, value in raw.items() if float(value) > 0.0}
    total = sum(cleaned.values())
    if total <= 0.0:
        raise ValueError("DAgger replacement weights must contain at least one positive value")
    return {key: value / total for key, value in cleaned.items()}


def normalize_stable_noise_modes(modes=None):
    default = {
        "zero_mean": {
            "prob": 0.50,
            "ioi_mu": 0.0,
            "ioi_sigma": 0.010,
            "duration_mu": 0.0,
            "duration_sigma": 0.010,
        },
        "positive_bias": {
            "prob": 0.25,
            "ioi_mu": 0.003,
            "ioi_sigma": 0.010,
            "duration_mu": 0.003,
            "duration_sigma": 0.010,
        },
        "variance_inflation": {
            "prob": 0.25,
            "ioi_mu": 0.0,
            "ioi_sigma": 0.025,
            "duration_mu": 0.0,
            "duration_sigma": 0.020,
        },
    }
    raw = default if modes is None else modes
    cleaned = {}
    for name, spec in dict(raw).items():
        spec = dict(spec or {})
        prob = float(spec.get("prob", 0.0))
        if prob <= 0.0:
            continue
        cleaned[str(name)] = {
            "prob": prob,
            "ioi_mu": float(spec.get("ioi_mu", 0.0)),
            "ioi_sigma": max(0.0, float(spec.get("ioi_sigma", 0.0))),
            "duration_mu": float(spec.get("duration_mu", spec.get("dur_mu", 0.0))),
            "duration_sigma": max(0.0, float(spec.get("duration_sigma", spec.get("dur_sigma", 0.0)))),
        }
    total = sum(item["prob"] for item in cleaned.values())
    if total <= 0.0:
        raise ValueError("stable_noise_modes must contain at least one positive-probability mode")
    for item in cleaned.values():
        item["prob"] = item["prob"] / total
    return cleaned


def stable_target_columns(channels=None):
    if channels is None:
        channels = ["ioi", "duration"]
    mapping = {"ioi": 0, "duration": 1, "dur": 1}
    cols = []
    for channel in channels:
        key = str(channel).lower()
        if key not in mapping:
            raise ValueError(f"Unsupported stable dynamics channel: {channel}")
        cols.append(mapping[key])
    return sorted(set(cols))


def integrated_epr_input_dim(
    timing_control_mode="log_scaled",
    use_timing_scale_bit=False,
    musical_feature_mode="categorical",
    pedal_control_dim=2,
):
    control_dim = timing_control_feature_dim(
        timing_control_mode=timing_control_mode,
        use_timing_scale_bit=use_timing_scale_bit,
    )
    score_control_dim = control_dim
    performance_control_dim = control_dim + int(pedal_control_dim)
    musical_dim = musical_feature_dim(musical_feature_mode)
    mask_dim = 2 if musical_dim == 0 else 3
    return score_control_dim + performance_control_dim + musical_dim + mask_dim


def normalize_pedal_representation(pedal_representation="start_valley"):
    value = str(pedal_representation or "start_valley").lower()
    aliases = {
        "binary4": "binary_4",
        "pedal4_binary": "binary_4",
        "start-valley": "start_valley",
        "pedal_start_valley": "start_valley",
        "pedal_start_has_valley": "start_valley",
        "pedal_start": "start",
        "start_only": "start",
        "start_dlm": "start",
    }
    value = aliases.get(value, value)
    if value not in {"binary_4", "start_valley", "start"}:
        raise ValueError(
            f"Unsupported pedal_representation={pedal_representation}; use binary_4, start_valley, or start"
        )
    return value


def pedal_representation_dim(pedal_representation="start_valley"):
    representation = normalize_pedal_representation(pedal_representation)
    if representation == "start_valley":
        return 2
    if representation == "start":
        return 1
    return 4


def default_epr_output_dim(epr_timing_target="log_deviation", pedal_representation="start_valley", legacy_dual_timing_head=False):
    target = str(epr_timing_target or "log_deviation").lower()
    pedal_dim = pedal_representation_dim(pedal_representation)
    if target in {"raw_log_absolute", "absolute_raw_log"}:
        return 5 + pedal_dim
    if target in {"raw_log_deviation", "raw_log_dev"} and bool(legacy_dual_timing_head):
        return 5 + pedal_dim
    return 3 + pedal_dim


def score_musical_input_dim(timing_control_mode="log_scaled", use_timing_scale_bit=False, musical_feature_mode="categorical"):
    control_dim = timing_control_feature_dim(
        timing_control_mode=timing_control_mode,
        use_timing_scale_bit=use_timing_scale_bit,
    )
    return control_dim + musical_feature_dim(musical_feature_mode) + 1


def decoder_perf_target_input_dim(output_dim=5):
    return int(output_dim) + 3


def integrated_csr_output_dim():
    return 12


def rows_to_model_continuous(rows, timing_normalization="scaled_log_5000_s10", max_time_ms=10000.0, rows_are_raw=False):
    if rows_are_raw:
        return raw_rows_to_model_continuous(
            rows,
            timing_normalization=timing_normalization,
            max_time_ms=max_time_ms,
        )
    return rows


def score_shared_rows(score, timing_normalization="scaled_log_5000_s10", max_time_ms=10000.0):
    if "score_raw" in score:
        return rows_to_model_continuous(
            score["score_raw"],
            timing_normalization=timing_normalization,
            max_time_ms=max_time_ms,
            rows_are_raw=True,
        )
    return score["score_continuous"]


def _raw_value_rows(perf, *keys):
    for key in keys:
        if key in perf:
            return perf[key]
    return None


def _compose_label_raw_rows(perf, pedal_representation="start_valley", pedal_binary_threshold=64.0):
    representation = normalize_pedal_representation(pedal_representation)
    shared_rows = perf.get("label_shared_raw")
    if shared_rows is None:
        if "label_raw" in perf:
            shared_rows = [row[:3] for row in perf["label_raw"]]
        else:
            return None

    preferred_pedal_key = "label_pedal_start_valley_raw" if representation in {"start_valley", "start"} else "label_pedal4_raw"
    pedal_rows = _raw_value_rows(perf, preferred_pedal_key, "label_pedal4_raw", "pedal4_raw")
    if pedal_rows is None:
        if "label_raw" in perf:
            pedal_rows = [row[3:7] for row in perf["label_raw"]]
        else:
            return None
    if len(shared_rows) != len(pedal_rows):
        raise ValueError(f"label_shared_raw/label_pedal4_raw length mismatch: {len(shared_rows)} vs {len(pedal_rows)}")
    threshold = float(pedal_binary_threshold)
    if representation in {"start_valley", "start"}:
        rows = []
        for index, (shared, pedal) in enumerate(zip(shared_rows, pedal_rows)):
            if len(pedal) >= 2 and "label_pedal_start_valley_raw" in perf:
                start = min(max(float(pedal[0]), 0.0), 127.0)
                valley = 127.0 if float(pedal[1]) >= 0.5 else 0.0
            else:
                values = list(pedal[:4])
                start = min(max(float(values[0]), 0.0), 127.0)
                next_start = float(pedal_rows[index + 1][0]) if index + 1 < len(pedal_rows) else start
                valley = 127.0 if start >= threshold and next_start >= threshold and min(float(v) for v in values[1:]) < threshold else 0.0
            pedal_values = [start] if representation == "start" else [start, valley]
            rows.append(list(shared[:3]) + pedal_values)
        return rows
    return [
        list(shared[:3]) + [127.0 if float(value) >= threshold else 0.0 for value in list(pedal[:4])]
        for shared, pedal in zip(shared_rows, pedal_rows)
    ]


def performance_label_rows_for_representation(
    perf,
    pedal_representation="start_valley",
    timing_normalization="scaled_log_5000_s10",
    max_time_ms=10000.0,
):
    raw_rows = _compose_label_raw_rows(perf, pedal_representation=pedal_representation)
    if raw_rows is not None:
        return raw_rows_to_model_continuous(
            raw_rows,
            timing_normalization=timing_normalization,
            max_time_ms=max_time_ms,
        )
    raise KeyError(f"Missing raw labels for pedal_representation={pedal_representation}")


def performance_label_bins_for_representation(
    perf,
    pedal_representation="start_valley",
    timing_bins=5000,
    value_bins=128,
):
    raw_rows = _compose_label_raw_rows(perf, pedal_representation=pedal_representation)
    if raw_rows is None:
        return None
    return raw_rows_to_epr_bins(raw_rows, timing_bins=timing_bins, value_bins=value_bins)


def performance_label_rows(perf, timing_normalization="scaled_log_5000_s10", max_time_ms=10000.0):
    return performance_label_rows_for_representation(
        perf,
        pedal_representation="start_valley",
        timing_normalization=timing_normalization,
        max_time_ms=max_time_ms,
    )


def performance_label_bins(perf, timing_bins=5000, value_bins=128):
    return performance_label_bins_for_representation(
        perf,
        pedal_representation="start_valley",
        timing_bins=timing_bins,
        value_bins=value_bins,
    )


def timing_log_scale(config_or_scale=None):
    if isinstance(config_or_scale, dict):
        return float(config_or_scale.get("timing_log_scale", 50.0))
    if config_or_scale is None:
        return 50.0
    return float(config_or_scale)


def normalize_log_timing_value(time_ms, scale=1.0, max_time_ms=5000.0):
    value = min(max(float(time_ms), 0.0), float(max_time_ms))
    scale = max(float(scale), 1e-12)
    return math.log1p(value / scale) / math.log1p(float(max_time_ms) / scale)


def raw_log_timing_value(time_ms, scale=50.0, max_time_ms=5000.0):
    value = min(max(float(time_ms), 0.0), float(max_time_ms))
    seconds = value / 1000.0
    factor = 1000.0 / max(float(scale), 1e-12)
    return math.log1p(factor * seconds)


def denormalize_log_timing_value(time_norm, scale=50.0, max_time_ms=5000.0):
    clipped = min(max(float(time_norm), 0.0), 1.0)
    scale = max(float(scale), 1e-12)
    return scale * math.expm1(clipped * math.log1p(float(max_time_ms) / scale))


def normalize_log_timing_dev(score_time_ms, perf_time_ms, scale=50.0, max_time_ms=5000.0):
    score_norm = normalize_log_timing_value(score_time_ms, scale=scale, max_time_ms=max_time_ms)
    perf_norm = normalize_log_timing_value(perf_time_ms, scale=scale, max_time_ms=max_time_ms)
    return min(max(perf_norm - score_norm + 0.5, 0.0), 1.0)


def _uses_log_deviation_target(epr_timing_target):
    return str(epr_timing_target or "").lower() in {
        "log_deviation",
        "log_dev",
        "raw_log_deviation",
        "raw_log_dev",
        "floor_log_deviation",
        "floor_log_dev",
        "pure_log_deviation",
        "pure_log_dev",
    }


def _uses_raw_log_deviation_target(epr_timing_target):
    return str(epr_timing_target or "").lower() in {"raw_log_deviation", "raw_log_dev"}


def _uses_floor_log_deviation_target(epr_timing_target):
    return str(epr_timing_target or "").lower() in {
        "floor_log_deviation",
        "floor_log_dev",
        "pure_log_deviation",
        "pure_log_dev",
    }


def _uses_raw_deviation_target(epr_timing_target):
    return str(epr_timing_target or "").lower() in {"raw_deviation", "raw_dev", "raw_seconds_deviation", "raw_seconds_dev"}


def _uses_absolute_target(epr_timing_target):
    return str(epr_timing_target or "").lower() in {
        "absolute",
        "absolute_log",
        "log_absolute",
        "raw_log_absolute",
        "absolute_raw_log",
        "floor_log_absolute",
    }


def _uses_raw_log_absolute_target(epr_timing_target):
    return str(epr_timing_target or "").lower() in {"raw_log_absolute", "absolute_raw_log"}


def normalize_ioi_dev(
    score_ioi_ms,
    perf_ioi_ms,
    epr_timing_target="log_deviation",
    log_scale=50.0,
):
    if _uses_log_deviation_target(epr_timing_target):
        if _uses_floor_log_deviation_target(epr_timing_target):
            return math.log(max(float(perf_ioi_ms), 1.0)) - math.log(max(float(score_ioi_ms), 1.0))
        if _uses_raw_log_deviation_target(epr_timing_target):
            return raw_log_timing_value(perf_ioi_ms, scale=log_scale) - raw_log_timing_value(
                score_ioi_ms,
                scale=log_scale,
            )
        score_norm = normalize_log_timing_value(score_ioi_ms, scale=log_scale, max_time_ms=5000.0)
        perf_norm = normalize_log_timing_value(perf_ioi_ms, scale=log_scale, max_time_ms=5000.0)
        delta = perf_norm - score_norm
        # Keep one target coordinate system for split and non-split heads.
        # The score-IOI mask chooses a specialized head; it should not also
        # change the meaning of the normalized target value.
        return min(max(delta + 0.5, 0.0), 1.0)
    dev_ms = float(perf_ioi_ms) - float(score_ioi_ms)
    return min(max((dev_ms + 500.0) / 1000.0, 0.0), 1.0)


def normalize_duration_dev(score_duration_ms, perf_duration_ms, epr_timing_target="deviation", log_scale=50.0):
    if _uses_log_deviation_target(epr_timing_target):
        if _uses_floor_log_deviation_target(epr_timing_target):
            return math.log(max(float(perf_duration_ms), 1.0)) - math.log(max(float(score_duration_ms), 1.0))
        if _uses_raw_log_deviation_target(epr_timing_target):
            return raw_log_timing_value(perf_duration_ms, scale=log_scale) - raw_log_timing_value(
                score_duration_ms,
                scale=log_scale,
            )
        return normalize_log_timing_dev(score_duration_ms, perf_duration_ms, scale=log_scale, max_time_ms=5000.0)
    dev_ms = float(perf_duration_ms) - float(score_duration_ms)
    return min(max((dev_ms + 500.0) / 1000.0, 0.0), 1.0)


def performance_dev_velocity_pedal4_binary_rows(
    perf,
    score_shared_raw,
    epr_timing_target="log_deviation",
    log_scale=50.0,
    pedal_binary_threshold=64.0,
    legacy_dual_timing_head=False,
    pedal_representation="start_valley",
):
    representation = normalize_pedal_representation(pedal_representation)
    shared_rows = perf.get("label_shared_raw")
    preferred_pedal_key = "label_pedal_start_valley_raw" if representation in {"start_valley", "start"} else "label_pedal4_raw"
    pedal_rows = _raw_value_rows(perf, preferred_pedal_key, "label_pedal4_raw", "pedal4_raw")
    if shared_rows is None or pedal_rows is None:
        if "label_raw" not in perf:
            return None
        shared_rows = [row[:3] for row in perf["label_raw"]]
        pedal_rows = [row[3:7] for row in perf["label_raw"]]

    if len(score_shared_raw) != len(shared_rows):
        raise ValueError(
            f"score_raw/label_shared_raw length mismatch: {len(score_shared_raw)} vs {len(shared_rows)}"
        )
    if len(shared_rows) != len(pedal_rows):
        raise ValueError(
            f"label_shared_raw/label_pedal4_raw length mismatch: {len(shared_rows)} vs {len(pedal_rows)}"
        )

    threshold = float(pedal_binary_threshold)
    rows = []
    for index, (score_row, perf_row, pedal_row) in enumerate(zip(score_shared_raw, shared_rows, pedal_rows)):
        log_ioi = normalize_ioi_dev(
            score_row[0],
            perf_row[0],
            epr_timing_target=epr_timing_target,
            log_scale=log_scale,
        )
        log_duration = normalize_duration_dev(
            score_row[1],
            perf_row[1],
            epr_timing_target=epr_timing_target,
            log_scale=log_scale,
        )
        velocity = min(max(float(perf_row[2]), 0.0), 127.0) / 127.0
        if representation in {"start_valley", "start"}:
            if len(pedal_row) >= 2 and "label_pedal_start_valley_raw" in perf:
                start = min(max(float(pedal_row[0]), 0.0), 127.0) / 127.0
                pedal = [start] if representation == "start" else [start, 1.0 if float(pedal_row[1]) >= 0.5 else 0.0]
            else:
                values = list(pedal_row[:4])
                start = min(max(float(values[0]), 0.0), 127.0)
                next_start = float(pedal_rows[index + 1][0]) if index + 1 < len(pedal_rows) else start
                valley = start >= threshold and next_start >= threshold and min(float(v) for v in values[1:]) < threshold
                pedal = [start / 127.0] if representation == "start" else [start / 127.0, 1.0 if valley else 0.0]
        else:
            pedal = [1.0 if float(value) >= threshold else 0.0 for value in pedal_row[:4]]
        if _uses_raw_log_absolute_target(epr_timing_target):
            rows.append(
                [
                    raw_log_timing_value(perf_row[0], scale=log_scale),
                    raw_log_timing_value(perf_row[1], scale=log_scale),
                    float(perf_row[0]) / 1000.0,
                    float(perf_row[1]) / 1000.0,
                    velocity,
                    *pedal,
                ]
            )
        elif _uses_absolute_target(epr_timing_target):
            if str(epr_timing_target).lower() == "floor_log_absolute":
                absolute_ioi = math.log(max(float(perf_row[0]), 1.0))
                absolute_duration = math.log(max(float(perf_row[1]), 1.0))
            else:
                absolute_ioi = raw_log_timing_value(perf_row[0], scale=log_scale)
                absolute_duration = raw_log_timing_value(perf_row[1], scale=log_scale)
            rows.append(
                [
                    absolute_ioi,
                    absolute_duration,
                    velocity,
                    *pedal,
                ]
            )
        elif _uses_raw_log_deviation_target(epr_timing_target):
            row = [log_ioi, log_duration]
            if legacy_dual_timing_head:
                row.extend(
                    [
                        (float(perf_row[0]) - float(score_row[0])) / 1000.0,
                        (float(perf_row[1]) - float(score_row[1])) / 1000.0,
                    ]
                )
            rows.append([*row, velocity, *pedal])
        else:
            rows.append([log_ioi, log_duration, velocity, *pedal])
    return rows


def normalize_piecewise_time_value(time_ms):
    max_ms = 8000.0 if mode == "dinr_floor_log" else 5000.0
    value = min(max(float(time_ms), 0.0), max_ms)
    if value <= 500.0:
        return value / 500.0
    return value / 5000.0


def encode_timing_control_features(time_ms, timing_control_mode="log_scaled", use_timing_scale_bit=False, log_scale=50.0):
    mode = resolve_timing_control_mode(
        timing_control_mode=timing_control_mode,
        use_timing_scale_bit=use_timing_scale_bit,
    )
    value = min(max(float(time_ms), 0.0), 5000.0)
    if mode == "piecewise_scale_bit":
        return [
            1.0 if value > 500.0 else 0.0,
            normalize_piecewise_time_value(value),
        ]
    if mode == "piecewise_single":
        return [normalize_piecewise_time_value(value)]
    if mode == "dual_log_linear":
        return [
            normalize_log_timing_value(value, scale=1.0, max_time_ms=5000.0),
            value / 5000.0,
        ]
    if mode == "log_scaled":
        return [normalize_log_timing_value(value, scale=log_scale, max_time_ms=5000.0)]
    if mode in {"floor_log", "dinr_floor_log"}:
        return [math.log(max(value, 1.0))]
    if mode == "dual_clip_linear":
        return [
            min(value / 500.0, 1.0),
            value / 5000.0,
        ]
    if mode == "raw_log":
        return [
            value / 1000.0,
            raw_log_timing_value(value, scale=log_scale),
        ]
    raise ValueError(f"Unsupported timing_control_mode={mode}")


def encode_shared_control_row(raw_shared_row, use_timing_scale_bit=False, timing_control_mode="log_scaled", log_scale=50.0):
    ioi_ms = float(raw_shared_row[0])
    duration_ms = float(raw_shared_row[1])
    velocity = min(max(float(raw_shared_row[2]), 0.0), 127.0) / 127.0
    return [
        *encode_timing_control_features(
            ioi_ms,
            timing_control_mode=timing_control_mode,
            use_timing_scale_bit=use_timing_scale_bit,
            log_scale=log_scale,
        ),
        *encode_timing_control_features(
            duration_ms,
            timing_control_mode=timing_control_mode,
            use_timing_scale_bit=use_timing_scale_bit,
            log_scale=log_scale,
        ),
        velocity,
    ]


def _one_hot_bucket(value, edges):
    bucket = 0
    for edge in edges:
        if value <= edge + 1e-12:
            return bucket
        bucket += 1
    return bucket


MUSICAL_MD_CATEGORIES = [
    0.5,
    0.25,
    1.0,
    1.0 / 3.0,
    0.125,
    1.0 / 6.0,
    2.0,
    1.5,
    0.75,
    3.0,
    4.0,
    0.0,
    1.0 / 12.0,
    0.375,
    0.0625,
    2.0 / 3.0,
]

MUSICAL_ML_CATEGORIES = [3.0, 4.0, 2.0, 1.5, 6.0, 1.0, 0.5, 0.25]

MUSICAL_MO_PHASE_CATEGORIES = [
    0.0,
    0.5,
    0.25,
    2.0 / 3.0,
    1.0 / 3.0,
    0.75,
    1.0 / 6.0,
    0.125,
    0.375,
    0.2,
    5.0 / 6.0,
    0.875,
    0.4,
    1.0 / 7.0,
    1.0 / 12.0,
    0.625,
]

MUSICAL51_ABLATION_MODES = {
    "musical51_full",
    "musical51_onset_only",
    "musical51_annotation_only",
    "musical51_duration_only",
    "musical51_onset_annotation",
    "musical51_no_duration",
    "musical51_no_length",
    "musical51_no_duration_length",
}


def _musical51_base_mode(mode):
    mode = str(mode).lower()
    return "musical51" if mode in MUSICAL51_ABLATION_MODES else mode


def _apply_musical51_ablation(row, mode):
    mode = str(mode).lower()
    if mode in {"musical51", "categorical", "categorical51", "musical51_full"}:
        return row
    if mode not in MUSICAL51_ABLATION_MODES:
        return row
    masked = [0.0] * 51
    if mode == "musical51_no_duration":
        spans = [(17, 51)]
    elif mode == "musical51_no_length":
        spans = [(0, 17), (27, 51)]
    elif mode == "musical51_onset_only":
        spans = [(27, 44)]
    elif mode == "musical51_annotation_only":
        spans = [(45, 51)]
    elif mode == "musical51_duration_only":
        spans = [(17, 27)]
    elif mode in {"musical51_onset_annotation", "musical51_no_duration_length"}:
        spans = [(27, 44), (45, 51)]
    else:
        spans = [(0, 51)]
    for start, end in spans:
        masked[start:end] = row[start:end]
    return masked


def _one_hot_exact(value, categories, tol=1e-4):
    return [1.0 if abs(float(value) - float(category)) <= tol else 0.0 for category in categories]


def build_score_musical_rows(score, musical_feature_mode="continuous"):
    score_feature = score.get("score_feature", [])
    has_score_feature = score.get("has_score_feature", [0] * len(score.get("pitch", [])))
    score_raw = score.get("score_raw", [])
    mode = str(musical_feature_mode).lower()
    base_mode = _musical51_base_mode(mode)

    rows = []
    measure_start = 0.0
    current_measure_length = 4.0
    prev_q = None
    prev_ms_per_quarter = 500.0
    seen_any_measure = False

    for idx, has_feature in enumerate(has_score_feature):
        if not bool(has_feature):
            rows.append([0.0] * musical_feature_dim(mode))
            continue

        feature = score_feature[idx]
        mo = float(feature[0]) if len(feature) > 0 else 0.0
        md = float(feature[1]) if len(feature) > 1 else 0.0
        raw_ml = float(feature[2]) if len(feature) > 2 else 0.0
        first = 1.0 if len(feature) > 3 and float(feature[3]) >= 0.5 else 0.0
        hand = 1.0 if len(feature) > 4 and float(feature[4]) >= 0.5 else 0.0
        trill = 1.0 if len(feature) > 5 and float(feature[5]) >= 0.5 else 0.0
        grace = 1.0 if len(feature) > 6 and float(feature[6]) >= 0.5 else 0.0
        stacc = 1.0 if len(feature) > 7 and float(feature[7]) >= 0.5 else 0.0
        stem_code = int(round(float(feature[8]))) if len(feature) > 8 else 0

        if not seen_any_measure:
            seen_any_measure = True
            if raw_ml > 0.0:
                current_measure_length = raw_ml
        elif first >= 0.5:
            measure_start += max(current_measure_length, 0.0)
            if raw_ml > 0.0:
                current_measure_length = raw_ml
        ml_eff = current_measure_length
        ml_present = 1.0 if raw_ml > 0.0 else 0.0

        q = measure_start + mo
        mioi = 0.0 if prev_q is None else max(q - prev_q, 0.0)
        prev_q = q

        candidates = []
        if idx < len(score_raw):
            score_ioi_ms = float(score_raw[idx][0])
            score_duration_ms = float(score_raw[idx][1])
            if mioi > 1e-6:
                candidates.append(score_ioi_ms / mioi)
            if md > 1e-6:
                candidates.append(score_duration_ms / md)
        if candidates:
            prev_ms_per_quarter = sum(candidates) / len(candidates)
        tempo_bpm = 60000.0 / max(prev_ms_per_quarter, 1e-6)
        tempo_norm = min(max(tempo_bpm, 0.0), 300.0) / 300.0

        if base_mode == "continuous":
            score_ioi_ms = float(score_raw[idx][0]) if idx < len(score_raw) else 0.0
            score_ioi_is_zero = 1.0 if score_ioi_ms <= 0.0 else 0.0
            continuous = [
                mo,
                score_ioi_is_zero,
                md,
                raw_ml,
                tempo_norm,
                first,
                grace,
                hand,
                trill,
                stacc,
                1.0 if stem_code == 1 else 0.0,
                1.0 if stem_code == 2 else 0.0,
            ]
            rows.append(continuous)
            continue

        if base_mode in {"musical145_onset_annotation", "onset145_annotation"}:
            onset_idx = int(round(mo * 24.0))
            onset_idx = min(max(onset_idx, 0), 144)
            onset = [0.0] * 145
            onset[onset_idx] = 1.0
            rows.append(
                [
                    *onset,
                    hand,
                    trill,
                    grace,
                    stacc,
                    1.0 if stem_code == 1 else 0.0,
                    1.0 if stem_code == 2 else 0.0,
                ]
            )
            continue

        if base_mode in {"categorical62", "musical62"}:
            d_bins = _one_hot_bucket(md, [0.0, 1 / 16, 1 / 12, 1 / 8, 1 / 6, 1 / 4, 1 / 3, 3 / 8, 1 / 2, 2 / 3, 3 / 4, 1.0, 1.5, 2.0, 3.0])
            i_bins = _one_hot_bucket(mioi, [0.0, 1 / 16, 1 / 12, 1 / 8, 1 / 6, 1 / 4, 1 / 3, 3 / 8, 1 / 2, 2 / 3, 3 / 4, 1.0, 1.5, 2.0, 3.0])
            l_bins = _one_hot_bucket(raw_ml, [0.0, 1.0, 1.5, 2.0, 4.0])
            phase = (mo / max(raw_ml, 1e-6)) % 1.0 if raw_ml > 0 else (mo / 4.0) % 1.0
            o_phase = min(15, int(math.floor(phase * 16.0)))
            o_scalar = mo

            rows.append(
                [
                    *([1.0 if d_bins == k else 0.0 for k in range(16)]),
                    *([1.0 if i_bins == k else 0.0 for k in range(16)]),
                    *([1.0 if l_bins == k else 0.0 for k in range(6)]),
                    *([1.0 if o_phase == k else 0.0 for k in range(16)]),
                    o_scalar,
                    first,
                    grace,
                    hand,
                    trill,
                    stacc,
                    1.0 if stem_code == 1 else 0.0,
                    1.0 if stem_code == 2 else 0.0,
                ]
            )
            continue

        mo_phase = (mo / max(ml_eff, 1e-6)) % 1.0 if ml_eff > 0 else 0.0
        row = [
            *_one_hot_exact(md, MUSICAL_MD_CATEGORIES),
            md,
            *_one_hot_exact(ml_eff, MUSICAL_ML_CATEGORIES),
            ml_eff,
            ml_present,
            *_one_hot_exact(mo_phase, MUSICAL_MO_PHASE_CATEGORIES),
            min(max(mo_phase, 0.0), 1.0),
            tempo_norm,
            hand,
            trill,
            grace,
            stacc,
            1.0 if stem_code == 1 else 0.0,
            1.0 if stem_code == 2 else 0.0,
        ]
        rows.append(_apply_musical51_ablation(row, mode))
    return rows


def build_epr_score_input_rows(
    score,
    use_timing_scale_bit=False,
    timing_control_mode="log_scaled",
    log_scale=50.0,
    musical_feature_mode="categorical",
    score_note_schema="integrated",
    disable_musical_features=False,
    pedal_control_dim=4,
):
    score_raw = score["score_raw"]
    has_score_feature = score.get("has_score_feature", [0] * len(score["pitch"]))
    musical_dim = musical_feature_dim(musical_feature_mode)
    musical_rows = (
        build_score_musical_rows(score, musical_feature_mode=musical_feature_mode)
        if musical_dim > 0
        else [[] for _ in score["pitch"]]
    )
    control_dim = timing_control_feature_dim(
        timing_control_mode=timing_control_mode,
        use_timing_scale_bit=use_timing_scale_bit,
    )
    schema = score_note_input_schema(score_note_schema)
    rows = []
    for raw_shared, musical, has_feature in zip(score_raw, musical_rows, has_score_feature):
        score_control = encode_shared_control_row(
            raw_shared[:3],
            use_timing_scale_bit=use_timing_scale_bit,
            timing_control_mode=timing_control_mode,
            log_scale=log_scale,
        )
        if musical_dim == 0:
            m_musical = 0.0
        elif disable_musical_features:
            m_musical = 0.0
            musical = [0.0] * len(musical)
        else:
            m_musical = 1.0 if bool(has_feature) else 0.0
        masked_musical = [value * m_musical for value in musical]
        if schema == "score_musical":
            rows.append(score_control + masked_musical + [m_musical])
            continue
        if schema != "integrated":
            raise ValueError(f"Unsupported score_note_schema={score_note_schema}")
        perf_control = [0.0] * (control_dim + int(pedal_control_dim))
        masks = [1.0, 0.0] if musical_dim == 0 else [1.0, 0.0, m_musical]
        rows.append(score_control + perf_control + masked_musical + masks)
    return rows


STYLE_STAT_DIM = 18


def _mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0, 0.0
    mean = float(values.mean())
    std = float(values.std())
    return mean, std


def score_style_stats(score, start, end):
    pitch = np.asarray(score.get("pitch", [])[start:end], dtype=np.float64)
    raw = np.asarray(score.get("score_raw", [])[start:end], dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] < 3:
        raw = np.zeros((0, 3), dtype=np.float64)
    valid_count = max(int(end) - int(start), 0)
    total_count = max(len(score.get("pitch", [])), 1)
    length_norm = min(valid_count / 512.0, 1.0)
    pos_norm = min(max(float(start) / float(total_count), 0.0), 1.0)

    if raw.shape[0] > 0:
        ioi = np.clip(raw[:, 0], 0.0, 5000.0)
        dur = np.clip(raw[:, 1], 0.0, 5000.0)
        vel = np.clip(raw[:, 2], 0.0, 127.0)
    else:
        ioi = dur = vel = np.zeros((0,), dtype=np.float64)
    pitch_valid = pitch[(pitch >= 0.0) & (pitch < 128.0)]

    ioi_mean, ioi_std = _mean_std(ioi / 5000.0)
    dur_mean, dur_std = _mean_std(dur / 5000.0)
    vel_mean, vel_std = _mean_std(vel / 127.0)
    pitch_mean, pitch_std = _mean_std(pitch_valid / 127.0)
    zero_ioi_ratio = float((ioi <= 0.0).mean()) if ioi.size else 0.0
    density = min(float((ioi > 0.0).sum()) / max(float(ioi.sum()) / 1000.0, 1e-6), 20.0) / 20.0 if ioi.size else 0.0

    has_feature = score.get("has_score_feature", [])
    score_feature = score.get("score_feature", [])
    feature_rows = [
        score_feature[idx]
        for idx in range(int(start), min(int(end), len(score_feature), len(has_feature)))
        if bool(has_feature[idx])
    ]
    feature_ratio = len(feature_rows) / max(valid_count, 1)
    first_ratio = grace_ratio = hand_ratio = trill_ratio = stacc_ratio = 0.0
    if feature_rows:
        features = np.asarray(feature_rows, dtype=np.float64)
        first_ratio = float((features[:, 3] >= 0.5).mean()) if features.shape[1] > 3 else 0.0
        hand_ratio = float((features[:, 4] >= 0.5).mean()) if features.shape[1] > 4 else 0.0
        trill_ratio = float((features[:, 5] >= 0.5).mean()) if features.shape[1] > 5 else 0.0
        grace_ratio = float((features[:, 6] >= 0.5).mean()) if features.shape[1] > 6 else 0.0
        stacc_ratio = float((features[:, 7] >= 0.5).mean()) if features.shape[1] > 7 else 0.0

    return [
        length_norm,
        pos_norm,
        ioi_mean,
        ioi_std,
        dur_mean,
        dur_std,
        vel_mean,
        vel_std,
        pitch_mean,
        pitch_std,
        zero_ioi_ratio,
        density,
        feature_ratio,
        first_ratio,
        grace_ratio,
        hand_ratio,
        trill_ratio,
        stacc_ratio,
    ]


def _label_style_base_values(labels):
    arr = np.asarray(labels, dtype=np.float64)
    if arr.ndim != 2:
        arr = np.zeros((0, 5), dtype=np.float64)
    dim = arr.shape[1] if arr.size else 5
    if dim < 5:
        padded = np.zeros((arr.shape[0], 5), dtype=np.float64)
        if arr.shape[0] > 0 and dim > 0:
            padded[:, :dim] = arr[:, :dim]
        arr = padded
    return np.clip(arr[:, :5], 0.0, 1.0)


def build_perf_style_prefix_cache(labels):
    values = _label_style_base_values(labels)
    n = values.shape[0]
    sums = np.zeros((n + 1, 5), dtype=np.float64)
    sums_sq = np.zeros((n + 1, 5), dtype=np.float64)
    deltas = np.zeros((n, 1), dtype=np.float64)
    if n > 1:
        deltas[1:, 0] = np.abs(values[1:, 2] - values[:-1, 2])
    pedal_changes = np.zeros((n, 1), dtype=np.float64)
    if n > 1:
        pedal_changes[1:, 0] = np.abs(values[1:, 3] - values[:-1, 3])
    extra = np.concatenate([deltas, pedal_changes], axis=1) if n > 0 else np.zeros((0, 2), dtype=np.float64)
    extra_sums = np.zeros((n + 1, 2), dtype=np.float64)
    if n > 0:
        sums[1:] = np.cumsum(values, axis=0)
        sums_sq[1:] = np.cumsum(values * values, axis=0)
        extra_sums[1:] = np.cumsum(extra, axis=0)
    pedal_on = np.zeros((n + 1,), dtype=np.float64)
    if n > 0:
        pedal_on[1:] = np.cumsum((values[:, 3] >= 0.5).astype(np.float64))
    return {
        "count": n,
        "sum": sums,
        "sum_sq": sums_sq,
        "extra_sum": extra_sums,
        "pedal_on": pedal_on,
        "start_cache": {},
    }


def perf_style_stats_from_cache(cache, start):
    start = int(max(0, min(int(start), int(cache["count"]))))
    if start in cache["start_cache"]:
        return cache["start_cache"][start]
    if start <= 0:
        stats = [0.0] * STYLE_STAT_DIM
        cache["start_cache"][start] = stats
        return stats
    count = float(start)
    mean = cache["sum"][start] / count
    var = np.maximum(cache["sum_sq"][start] / count - mean * mean, 0.0)
    std = np.sqrt(var)
    extra_mean = cache["extra_sum"][start] / count
    pedal_on_ratio = float(cache["pedal_on"][start] / count)
    stats = [
        min(count / 2048.0, 1.0),
        float(mean[0]),
        float(std[0]),
        float(mean[1]),
        float(std[1]),
        float(mean[2]),
        float(std[2]),
        float(mean[3]),
        float(std[3]),
        float(mean[4]),
        float(std[4]),
        float(extra_mean[0]),
        float(extra_mean[1]),
        pedal_on_ratio,
        float(np.clip(mean[1] - mean[0], -1.0, 1.0) * 0.5 + 0.5),
        float(np.clip(mean[2] - 0.5, -0.5, 0.5) + 0.5),
        float(np.clip(std[0] + std[1], 0.0, 1.0)),
        float(np.clip(std[2] + std[3], 0.0, 1.0)),
    ]
    cache["start_cache"][start] = stats
    return stats


def perf_style_stats_range_from_cache(cache, start, end):
    start = int(max(0, min(int(start), int(cache["count"]))))
    end = int(max(start, min(int(end), int(cache["count"]))))
    range_cache = cache.setdefault("range_cache", {})
    key = (start, end)
    if key in range_cache:
        return range_cache[key]
    count_int = end - start
    if count_int <= 0:
        stats = [0.0] * STYLE_STAT_DIM
        range_cache[key] = stats
        return stats
    count = float(count_int)
    sum_values = cache["sum"][end] - cache["sum"][start]
    sum_sq_values = cache["sum_sq"][end] - cache["sum_sq"][start]
    mean = sum_values / count
    var = np.maximum(sum_sq_values / count - mean * mean, 0.0)
    std = np.sqrt(var)
    extra_mean = (cache["extra_sum"][end] - cache["extra_sum"][start]) / count
    pedal_on_ratio = float((cache["pedal_on"][end] - cache["pedal_on"][start]) / count)
    stats = [
        min(count / 512.0, 1.0),
        float(mean[0]),
        float(std[0]),
        float(mean[1]),
        float(std[1]),
        float(mean[2]),
        float(std[2]),
        float(mean[3]),
        float(std[3]),
        float(mean[4]),
        float(std[4]),
        float(extra_mean[0]),
        float(extra_mean[1]),
        pedal_on_ratio,
        float(np.clip(mean[1] - mean[0], -1.0, 1.0) * 0.5 + 0.5),
        float(np.clip(mean[2] - 0.5, -0.5, 0.5) + 0.5),
        float(np.clip(std[0] + std[1], 0.0, 1.0)),
        float(np.clip(std[2] + std[3], 0.0, 1.0)),
    ]
    range_cache[key] = stats
    return stats


def build_csr_performance_input_rows(
    perf,
    use_timing_scale_bit=False,
    timing_control_mode="log_scaled",
    log_scale=50.0,
    pedal_binary_threshold=64.0,
):
    shared_rows = perf.get("label_shared_raw")
    pedal_rows = _raw_value_rows(perf, "label_pedal4_raw", "pedal4_raw")
    if shared_rows is None or pedal_rows is None:
        if "label_raw" not in perf:
            return None
        shared_rows = [row[:3] for row in perf["label_raw"]]
        pedal_rows = [row[3:7] for row in perf["label_raw"]]
    if len(shared_rows) != len(pedal_rows):
        raise ValueError(
            f"label_shared_raw/label_pedal4_raw length mismatch: {len(shared_rows)} vs {len(pedal_rows)}"
        )

    threshold = float(pedal_binary_threshold)
    rows = []
    for raw_shared, pedal in zip(shared_rows, pedal_rows):
        score_control = [0.0] * timing_control_feature_dim(
            timing_control_mode=timing_control_mode,
            use_timing_scale_bit=use_timing_scale_bit,
        )
        perf_control = (
            encode_shared_control_row(
                raw_shared[:3],
                use_timing_scale_bit=use_timing_scale_bit,
                timing_control_mode=timing_control_mode,
                log_scale=log_scale,
            )
            + [
                *[1.0 if float(value) >= threshold else 0.0 for value in pedal[:4]],
            ]
        )
        masks = [0.0, 1.0, 0.0]
        rows.append(
            score_control
            + perf_control
            + [0.0] * 12
            + masks
        )
    return rows


def distributed_info():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    return 0, 1


def configure_eval_schedule(train_config, train_examples):
    asap_only = str(train_config.get("train_performance_dataset", "")).strip().upper() == "ASAP"
    if asap_only:
        world_size = int(os.environ.get("WORLD_SIZE", "1") or 1)
        per_device = int(train_config.get("per_device_train_batch_size", 1) or 1)
        grad_accum = int(train_config.get("gradient_accumulation_steps", 1) or 1)
        global_batch_size = per_device * grad_accum * world_size
        steps_per_epoch = max(1, math.ceil(int(train_examples) / max(global_batch_size, 1)))
        eval_steps = steps_per_epoch
    else:
        eval_steps = 1000

    train_config["eval_strategy"] = "steps"
    train_config["save_strategy"] = "steps"
    train_config["eval_steps"] = eval_steps
    train_config["save_steps"] = eval_steps
    train_config.setdefault("load_best_model_at_end", True)
    train_config.setdefault("metric_for_best_model", "eval_loss")
    train_config.setdefault("greater_is_better", False)


def build_style_vocabs(metadata_path):
    header = pd.read_csv(metadata_path, nrows=0).columns.tolist()
    usecols = [col for col in ("composer", "performance_dataset") if col in header]
    df = pd.read_csv(metadata_path, usecols=usecols) if usecols else pd.DataFrame()
    composer_vocab = {"<unk>": 0}
    source_vocab = {"<unk>": 0}
    if "composer" in df:
        for value in sorted(df["composer"].dropna().astype(str).unique()):
            if value and value not in composer_vocab:
                composer_vocab[value] = len(composer_vocab)
    if "performance_dataset" in df:
        for value in sorted(df["performance_dataset"].dropna().astype(str).unique()):
            if value and value not in source_vocab:
                source_vocab[value] = len(source_vocab)
    return composer_vocab, source_vocab


class PianoCoReNodeSFTDataset(Dataset):
    def __init__(
        self,
        manifest,
        split,
        task_type="epr",
        input_feature_mode="integrated",
        shuffle=True,
        seed=42,
        max_performances_per_work=None,
        max_windows_per_work=None,
        cache_size=2,
        timing_normalization="scaled_log_5000_s10",
        max_time_ms=10000.0,
        epr_timing_bins=5000,
        epr_value_bins=128,
        pedal_representation="start_valley",
        musical_feature_mode="categorical",
        score_note_schema="integrated",
        epr_timing_target="log_deviation",
        disable_musical_features=False,
        use_timing_scale_bit=False,
        timing_control_mode="log_scaled",
        timing_log_scale=50.0,
        precompute_items=False,
        use_prepared_sidecar=True,
        prepared_sidecar_tag=None,
        use_style_tokens=False,
        composer_vocab=None,
        source_vocab=None,
        perf_style_stats_mode="prefix",
        legacy_dual_timing_head=False,
        multi_perf_group_size=1,
        multi_perf_min_group_size=3,
    ):
        super().__init__()
        self.split = split
        self.task_type = task_type
        self.input_feature_mode = input_feature_mode
        self.timing_normalization = timing_normalization
        self.max_time_ms = max_time_ms
        self.epr_timing_bins = epr_timing_bins
        self.epr_value_bins = epr_value_bins
        self.pedal_representation = pedal_representation
        self.musical_feature_mode = str(musical_feature_mode).lower()
        self.score_note_schema = score_note_input_schema(score_note_schema)
        self.epr_timing_target = str(epr_timing_target or "log_deviation").lower()
        self.disable_musical_features = bool(disable_musical_features)
        self.use_timing_scale_bit = bool(use_timing_scale_bit)
        self.timing_control_mode = resolve_timing_control_mode(
            timing_control_mode=timing_control_mode,
            use_timing_scale_bit=use_timing_scale_bit,
        )
        self.timing_log_scale = float(timing_log_scale)
        self.legacy_dual_timing_head = bool(legacy_dual_timing_head)
        self.multi_perf_group_size = max(1, int(multi_perf_group_size or 1))
        self.multi_perf_min_group_size = max(2, int(multi_perf_min_group_size or 3))
        self.use_style_tokens = bool(use_style_tokens)
        self.perf_style_stats_mode = str(perf_style_stats_mode or "prefix").lower()
        if self.perf_style_stats_mode not in {"prefix", "window"}:
            raise ValueError(f"Unsupported perf_style_stats_mode={perf_style_stats_mode}")
        self.composer_vocab = dict(composer_vocab or {})
        self.source_vocab = dict(source_vocab or {})
        self.unknown_composer_id = int(self.composer_vocab.get("<unk>", 0))
        self.unknown_source_id = int(self.source_vocab.get("<unk>", 0))
        items = list(manifest)
        if shuffle:
            random.Random(seed).shuffle(items)

        self.items = []
        self.cumulative_sizes = []
        total = 0
        for item in items:
            windows = list(item["windows"])
            if max_windows_per_work is not None:
                windows = windows[:max_windows_per_work]
            selected_sources = item.get("selected_performance_sources")
            if self.multi_perf_group_size > 1:
                # Manifest estimates can include metadata rows absent from the
                # current processed work. Resolve against the actual aligned
                # performances so group counts and __getitem__ stay consistent.
                work_payload = json.loads(Path(item["path"]).read_text(encoding="utf-8"))
                available_sources = {
                    perf.get("performance_source")
                    for perf in work_payload.get("performances", [])
                    if perf.get("split", self.split) == self.split
                    and perf.get("performance_source") is not None
                }
                if selected_sources is not None:
                    selected_sources = [source for source in selected_sources if source in available_sources]
                    performance_count = len(selected_sources)
                else:
                    performance_count = len(available_sources)
            else:
                performance_count = None
            if performance_count is None:
                performance_count = len(selected_sources) if selected_sources is not None else int(item["estimated_performances"])
            if max_performances_per_work is not None:
                performance_count = min(performance_count, max_performances_per_work)
                if selected_sources is not None:
                    selected_sources = selected_sources[:performance_count]
            if not windows or performance_count <= 0:
                continue

            item = dict(item)
            item["windows"] = windows
            if selected_sources is not None:
                item["selected_performance_sources"] = selected_sources
            item["effective_performances"] = performance_count
            if self.multi_perf_group_size > 1:
                full_groups, remainder = divmod(performance_count, self.multi_perf_group_size)
                group_count = full_groups + int(remainder >= self.multi_perf_min_group_size)
                if group_count <= 0:
                    continue
                item["multi_perf_group_count"] = group_count
                item["effective_examples"] = group_count * len(windows)
            else:
                item["effective_examples"] = performance_count * len(windows)
            self.items.append(item)
            total += item["effective_examples"]
            self.cumulative_sizes.append(total)

        self.total_examples = total
        self.precompute_items = bool(precompute_items)
        self.cache_size = max(int(cache_size), len(self.items)) if self.precompute_items else int(cache_size)
        self.use_prepared_sidecar = bool(use_prepared_sidecar)
        self.prepared_sidecar_tag = str(prepared_sidecar_tag) if prepared_sidecar_tag else None
        self._derived_feature_cache_signature = json.dumps(
            {
                "task_type": self.task_type,
                "input_feature_mode": self.input_feature_mode,
                "musical_feature_mode": self.musical_feature_mode,
                "score_note_input_schema": self.score_note_schema,
                "disable_musical_features": self.disable_musical_features,
                "timing_control_mode": self.timing_control_mode,
                "timing_log_scale": self.timing_log_scale,
                "use_timing_scale_bit": self.use_timing_scale_bit,
                "pedal_representation": normalize_pedal_representation(self.pedal_representation),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        self._prepared_sidecar_signature = self._build_prepared_sidecar_signature()
        self._cache = OrderedDict()
        self._prepared_cache = OrderedDict()
        self._dagger_prefix_cache = {}
        self._dagger_mask_cache = set()
        self._dagger_cache_version = 0
        if self.precompute_items:
            self._precompute_items()

    def set_dagger_prefix_cache(self, cache):
        self._dagger_prefix_cache.clear()
        self._dagger_prefix_cache = dict(cache or {})
        self._dagger_mask_cache.clear()
        self._dagger_cache_version += 1

    def set_dagger_mask_cache(self, indices):
        self._dagger_prefix_cache.clear()
        self._dagger_mask_cache = {int(index) for index in (indices or [])}
        self._dagger_cache_version += 1

    def clear_dagger_prefix_cache(self):
        self.set_dagger_prefix_cache({})

    def _build_prepared_sidecar_signature(self):
        signature = {
            "schema": 5,
            "kind": "inr_raw_sidecar",
        }
        return json.dumps(signature, sort_keys=True, separators=(",", ":"))

    def _build_ready_sidecar_signature(self):
        return json.dumps(
            {
                "schema": 7,
                "kind": "inr_multi_target_ready_sidecar",
                "feature_signature": self._derived_feature_cache_signature,
                "legacy_dual_timing_head": self.legacy_dual_timing_head,
                "pedal_representation": normalize_pedal_representation(self.pedal_representation),
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def _is_raw_sidecar_signature(self, signature):
        if signature == self._prepared_sidecar_signature:
            return True
        try:
            payload = json.loads(signature)
        except (TypeError, json.JSONDecodeError):
            return False
        return payload.get("schema") == 5 and payload.get("kind") == "inr_raw_sidecar"

    def _has_raw_sidecar_payload(self, prepared):
        score = prepared.get("score")
        if not isinstance(score, dict):
            return False
        if "pitch" not in score or "score_raw" not in score:
            return False
        if self.task_type.lower() in {"epr", "csr"}:
            if "score_feature" not in score or "has_score_feature" not in score:
                return False

        performances = prepared.get("performances")
        if not isinstance(performances, list):
            return False
        for perf in performances:
            if not isinstance(perf, dict) or "interpolated" not in perf:
                return False
            if self.task_type.lower() == "epr" and self.epr_timing_target in {
                "log_deviation",
                "log_dev",
                "raw_log_deviation",
                "raw_log_dev",
                "floor_log_deviation",
                "floor_log_dev",
                "pure_log_deviation",
                "pure_log_dev",
                "raw_deviation",
                "raw_dev",
                "raw_seconds_deviation",
                "raw_seconds_dev",
                "absolute",
                "absolute_log",
                "log_absolute",
                "raw_log_absolute",
                "absolute_raw_log",
            }:
                has_shared = "label_shared_raw" in perf or "label_raw" in perf
                if normalize_pedal_representation(self.pedal_representation) == "start_valley":
                    has_pedal = "label_pedal_start_valley_raw" in perf or "label_pedal4_raw" in perf or "pedal4_raw" in perf or "label_raw" in perf
                else:
                    has_pedal = "label_pedal4_raw" in perf or "pedal4_raw" in perf or "label_raw" in perf
                if not (has_shared and has_pedal):
                    return False
        return True

    def _normalize_loaded_raw_sidecar(self, prepared):
        normalized = dict(prepared)
        normalized.pop("score_input", None)
        normalized.pop("score_musical", None)
        normalized.pop("has_score_feature", None)
        normalized.pop("_derived_score_input_cache", None)
        normalized.pop("_derived_score_musical", None)
        normalized["label_cache"] = {}
        normalized["perf_style_cache"] = {}
        if "performances_by_source" not in normalized:
            normalized["performances_by_source"] = {
                perf.get("performance_source"): perf
                for perf in normalized.get("performances", [])
                if perf.get("performance_source") is not None
            }
        return normalized

    def _source_identity(self, path):
        source = Path(path)
        try:
            stat = source.stat()
            return {
                "path": str(source.resolve()),
                "mtime_ns": stat.st_mtime_ns,
                "size": stat.st_size,
            }
        except FileNotFoundError:
            return {"path": str(source)}

    def _prepared_sidecar_paths(self, path):
        source = Path(path)
        explicit_tags = []
        inferred_tags = []
        if self.prepared_sidecar_tag:
            explicit_tags.append(self.prepared_sidecar_tag)
        elif "ASAP" in source.stem.upper():
            inferred_tags.append("ASAP")
        candidates = []
        for tag in explicit_tags:
            candidates.append(source.with_suffix(f".{tag}.pt"))
        candidates.append(source.with_suffix(".pt"))
        for tag in inferred_tags:
            candidates.append(source.with_suffix(f".{tag}.pt"))
        deduped = []
        seen = set()
        for candidate in candidates:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
        return deduped

    def _prepared_disk_cache_path(self, path):
        if not self.use_prepared_sidecar:
            return None
        return self._prepared_sidecar_paths(path)[0]

    def _torch_load_prepared(self, cache_path):
        try:
            return torch.load(cache_path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(cache_path, map_location="cpu")

    def _load_prepared_from_disk(self, path):
        for cache_path in self._prepared_sidecar_paths(path):
            if not cache_path.exists():
                continue
            prepared = self._torch_load_prepared(cache_path)
            signature = prepared.get("_cache_signature")
            if prepared.get("_source_identity") != self._source_identity(path):
                continue
            if signature == self._build_ready_sidecar_signature():
                target = self.epr_timing_target
                selected = dict(prepared)
                selected_performances = []
                for perf in prepared.get("performances", []):
                    labels_by_target = perf.get("labels_by_target") or {}
                    if target not in labels_by_target:
                        break
                    selected_perf = dict(perf)
                    selected_perf["labels"] = labels_by_target[target]
                    selected_performances.append(selected_perf)
                else:
                    selected["performances"] = selected_performances
                    selected["performances_by_source"] = {
                        perf.get("performance_source"): perf
                        for perf in selected_performances
                        if perf.get("performance_source") is not None
                    }
                    return selected
            if self._is_raw_sidecar_signature(signature):
                if self._has_raw_sidecar_payload(prepared):
                    return self._normalize_loaded_raw_sidecar(prepared)
                continue
        return None

    def _save_prepared_to_disk(self, path, prepared):
        cache_path = self._prepared_disk_cache_path(path)
        if cache_path is None:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
        payload = dict(prepared)
        payload.pop("_derived_score_input_cache", None)
        payload.pop("_derived_score_musical", None)
        payload["_cache_signature"] = self._prepared_sidecar_signature
        payload["_source_identity"] = self._source_identity(path)
        torch.save(payload, tmp_path)
        os.replace(tmp_path, cache_path)

    def _load_or_prepare_work(self, path):
        if path in self._prepared_cache:
            self._prepared_cache.move_to_end(path)
            return self._prepared_cache[path]

        prepared = None
        cache_path = self._prepared_disk_cache_path(path) if self.use_prepared_sidecar else None
        if cache_path is not None:
            prepared = self._load_prepared_from_disk(path)
            if prepared is None:
                sidecar_candidates = ", ".join(str(p) for p in self._prepared_sidecar_paths(path))
                raise FileNotFoundError(
                    "Prepared sidecar missing or invalid during training. "
                    "Sidecars must be prebuilt in data_process before training and are treated as read-only at runtime. "
                    f"Source: {path}. Candidates: {sidecar_candidates}"
                )

        if prepared is None:
            work = self._load_work(path)
            prepared = self._prepare_work(
                path,
                work,
                eager_labels=False,
                slim_performances=False,
                split_filter=True,
                force_rebuild=False,
            )

        self._prepared_cache[path] = prepared
        self._prepared_cache.move_to_end(path)
        while len(self._prepared_cache) > self.cache_size:
            evicted_path, _ = self._prepared_cache.popitem(last=False)
            self._cache.pop(evicted_path, None)
        return prepared

    def __len__(self):
        return self.total_examples

    def _load_work(self, path):
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]

        with open(path, "r", encoding="utf-8") as file:
            work = json.load(file)
        self._cache[path] = work
        self._cache.move_to_end(path)
        while len(self._cache) > self.cache_size:
            evicted_path, _ = self._cache.popitem(last=False)
            self._prepared_cache.pop(evicted_path, None)
        return work

    def _prepare_work(
        self,
        path,
        work,
        eager_labels=False,
        slim_performances=False,
        split_filter=True,
        force_rebuild=False,
        derive_features=True,
    ):
        if not force_rebuild and path in self._prepared_cache:
            self._prepared_cache.move_to_end(path)
            return self._prepared_cache[path]

        score = work["score"]
        performances = list(work["performances"])
        if split_filter:
            performances = [
                perf for perf in performances
                if perf.get("split", self.split) == self.split
            ]
        by_source = {perf.get("performance_source"): perf for perf in performances}

        task_type = self.task_type.lower()

        # Cache score inputs and raw-label conversions per dataloader worker.
        # The new raw schema would otherwise rebuild the same score/performance
        # tensors for every overlapping window.
        if slim_performances:
            score_payload = {
                "pitch": score["pitch"],
                "score_raw": score["score_raw"],
            }
            for key in ("score_feature", "has_score_feature"):
                if key in score:
                    score_payload[key] = score[key]
        else:
            score_payload = score
        prepared = {
            "score": score_payload,
            "performances": performances,
            "performances_by_source": by_source,
            "label_cache": {},
            "perf_style_cache": {},
        }
        if isinstance(work.get("meta"), dict):
            prepared["meta"] = dict(work["meta"])

        if derive_features and task_type == "epr":
            prepared["score_input"] = build_epr_score_input_rows(
                score,
                use_timing_scale_bit=self.use_timing_scale_bit,
                timing_control_mode=self.timing_control_mode,
                log_scale=self.timing_log_scale,
                musical_feature_mode=self.musical_feature_mode,
                score_note_schema=self.score_note_schema,
                disable_musical_features=self.disable_musical_features,
                pedal_control_dim=pedal_representation_dim(self.pedal_representation),
            )
        elif derive_features and task_type == "csr":
            prepared["score_musical"] = build_score_musical_rows(score, musical_feature_mode="continuous")
            prepared["has_score_feature"] = score["has_score_feature"]

        if eager_labels:
            slimmed = []
            label_prepared = dict(prepared)
            label_prepared["score"] = score
            for perf in performances:
                labels, label_bins = self._compute_performance_labels(label_prepared, perf)
                slim_perf = {
                    "performance_source": perf.get("performance_source"),
                    "performance_id": perf.get("performance_id", "unknown"),
                    "performance_dataset": perf.get("performance_dataset", "unknown"),
                    "split": perf.get("split", self.split),
                    "interpolated": perf["interpolated"],
                    "labels": labels,
                    "label_bins": label_bins,
                }
                slimmed.append(slim_perf if slim_performances else perf)
                if not slim_performances:
                    prepared["label_cache"][self._performance_cache_key(slim_perf)] = (labels, label_bins)
            if slim_performances:
                prepared["performances"] = slimmed
                prepared["performances_by_source"] = {
                    perf.get("performance_source"): perf
                    for perf in slimmed
                    if perf.get("performance_source") is not None
                }
                prepared["label_cache"] = {}

        self._prepared_cache[path] = prepared
        self._prepared_cache.move_to_end(path)
        while len(self._prepared_cache) > self.cache_size:
            evicted_path, _ = self._prepared_cache.popitem(last=False)
            self._cache.pop(evicted_path, None)
        return prepared

    def _selected_performances(self, prepared, item):
        selected_sources = item.get("selected_performance_sources")
        if selected_sources is None:
            selected = prepared["performances"]
        else:
            by_source = prepared["performances_by_source"]
            selected = [by_source[source] for source in selected_sources if source in by_source]
        return [
            perf for perf in selected
            if perf.get("split", self.split) == self.split
        ]

    def _performance_cache_key(self, perf):
        return (
            perf.get("performance_source")
            or perf.get("performance_id")
            or id(perf)
        )

    def _compute_performance_labels(self, prepared, perf):
        if self.task_type.lower() == "epr" and self.epr_timing_target in {
            "log_deviation",
            "log_dev",
            "raw_log_deviation",
            "raw_log_dev",
            "floor_log_deviation",
            "floor_log_dev",
            "pure_log_deviation",
            "pure_log_dev",
            "raw_deviation",
            "raw_dev",
            "raw_seconds_deviation",
            "raw_seconds_dev",
            "absolute",
            "absolute_log",
            "log_absolute",
            "raw_log_absolute",
            "absolute_raw_log",
            "floor_log_absolute",
        }:
            labels = performance_dev_velocity_pedal4_binary_rows(
                perf,
                prepared["score"]["score_raw"],
                epr_timing_target=self.epr_timing_target,
                log_scale=self.timing_log_scale,
                legacy_dual_timing_head=self.legacy_dual_timing_head,
                pedal_representation=self.pedal_representation,
            )
            missing = f"Missing score_raw/label_shared_raw/pedal rows for {self.pedal_representation} deviation EPR targets"
            if labels is None:
                raise KeyError(missing)
            return labels, None
        if self.task_type.lower() == "csr":
            labels = build_csr_performance_input_rows(
                perf,
                use_timing_scale_bit=self.use_timing_scale_bit,
                timing_control_mode=self.timing_control_mode,
                log_scale=self.timing_log_scale,
            )
            if labels is None:
                raise KeyError("Missing label_shared_raw/label_pedal4_raw for CSR inputs")
            return labels, None

        labels = performance_label_rows_for_representation(
            perf,
            pedal_representation=self.pedal_representation,
            timing_normalization=self.timing_normalization,
            max_time_ms=self.max_time_ms,
        )
        label_bins = performance_label_bins_for_representation(
            perf,
            pedal_representation=self.pedal_representation,
            timing_bins=self.epr_timing_bins,
            value_bins=self.epr_value_bins,
        )
        return labels, label_bins

    def _performance_labels(self, prepared, perf):
        if "labels" in perf:
            return perf["labels"], perf.get("label_bins")
        label_cache = prepared["label_cache"]
        cache_key = self._performance_cache_key(perf)
        if cache_key in label_cache:
            return label_cache[cache_key]
        labels, label_bins = self._compute_performance_labels(prepared, perf)
        label_cache[cache_key] = (labels, label_bins)
        return labels, label_bins

    def _style_creator_id(self, prepared):
        meta = prepared.get("meta")
        composer = str(meta.get("composer") or "") if isinstance(meta, dict) else ""
        return int(self.composer_vocab.get(composer, self.unknown_composer_id))

    def _style_source_id(self, perf):
        source = str(perf.get("performance_dataset") or "unknown")
        return int(self.source_vocab.get(source, self.unknown_source_id))

    def _perf_style_stats(self, prepared, perf, labels, start, end=None):
        perf_cache = prepared.setdefault("perf_style_cache", {})
        cache_key = self._performance_cache_key(perf)
        if cache_key not in perf_cache:
            perf_cache[cache_key] = build_perf_style_prefix_cache(labels)
        if self.perf_style_stats_mode == "window":
            if end is None:
                raise ValueError("end is required for perf_style_stats_mode=window")
            return perf_style_stats_range_from_cache(perf_cache[cache_key], start, end)
        return perf_style_stats_from_cache(perf_cache[cache_key], start)

    def _score_input_rows(self, prepared):
        cache = prepared.setdefault("_derived_score_input_cache", {})
        cache_key = self._derived_feature_cache_signature
        if cache_key not in cache:
            cache[cache_key] = build_epr_score_input_rows(
                prepared["score"],
                use_timing_scale_bit=self.use_timing_scale_bit,
                timing_control_mode=self.timing_control_mode,
                log_scale=self.timing_log_scale,
                musical_feature_mode=self.musical_feature_mode,
                score_note_schema=self.score_note_schema,
                disable_musical_features=self.disable_musical_features,
                pedal_control_dim=pedal_representation_dim(self.pedal_representation),
            )
        return cache[cache_key]

    def _score_musical_rows(self, prepared):
        if "_derived_score_musical" not in prepared:
            prepared["_derived_score_musical"] = build_score_musical_rows(
                prepared["score"],
                musical_feature_mode="continuous",
            )
        return prepared["_derived_score_musical"]

    def _score_feature_mask(self, prepared):
        if "has_score_feature" in prepared:
            return prepared["has_score_feature"]
        return prepared["score"]["has_score_feature"]

    def _precompute_items(self):
        rank, _ = distributed_info()
        if rank == 0:
            print(
                json.dumps(
                    {
                        "event": "dataset_precompute_start",
                        "split": self.split,
                        "works": len(self.items),
                        "examples": self.total_examples,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
        total_performances = 0
        for item in self.items:
            prepared = self._load_or_prepare_work(item["path"])
            performances = self._selected_performances(prepared, item)
            for perf in performances:
                self._performance_labels(prepared, perf)
                total_performances += 1
        if rank == 0:
            print(
                json.dumps(
                    {
                        "event": "dataset_precompute_done",
                        "split": self.split,
                        "works": len(self._prepared_cache),
                        "performances": total_performances,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )

    def _sample_for_performance(self, prepared, perf, start, end, index):
        score = prepared["score"]
        labels, label_bins = self._performance_labels(prepared, perf)
        interpolated = perf["interpolated"]
        task_type = self.task_type.lower()
        if task_type == "epr":
            continuous = self._score_input_rows(prepared)[start:end]
            labels_continuous = labels[start:end]
            labels_epr_bins = label_bins[start:end] if label_bins is not None else None
            label_mask = None
        elif task_type == "csr":
            continuous = labels[start:end]
            labels_continuous = self._score_musical_rows(prepared)[start:end]
            labels_epr_bins = None
            label_mask = self._score_feature_mask(prepared)[start:end]
        else:
            raise ValueError(f"Unsupported task_type: {self.task_type}")

        sample = {
            "example_index": index,
            "pitch_ids": score["pitch"][start:end],
            "continuous": continuous,
            "labels_continuous": labels_continuous,
            "score_shared_raw": [row[:3] for row in score["score_raw"][start:end]],
            "interpolated": interpolated[start:end],
            "performance_dataset": perf.get("performance_dataset", "unknown"),
            "performance_id": perf.get("performance_id", "unknown"),
        }
        if self.use_style_tokens:
            sample.update(
                {
                    "style_creator_id": self._style_creator_id(prepared),
                    "style_source_id": self._style_source_id(perf),
                    "style_score_stats": score_style_stats(score, start, end),
                    "style_perf_stats": self._perf_style_stats(prepared, perf, labels, start, end),
                    "style_perf_is_pad": bool(start <= 0 and self.perf_style_stats_mode == "prefix"),
                }
            )
        if labels_epr_bins is not None:
            sample["labels_epr_bins"] = labels_epr_bins
        if label_mask is not None:
            sample["label_mask"] = label_mask
        return sample

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        item_idx = bisect.bisect_right(self.cumulative_sizes, index)
        prev_size = 0 if item_idx == 0 else self.cumulative_sizes[item_idx - 1]
        local_index = index - prev_size
        item = self.items[item_idx]

        windows = item["windows"]
        window_count = len(windows)
        perf_slot = local_index // window_count
        window_slot = local_index % window_count
        start, end = windows[window_slot]

        prepared = self._load_or_prepare_work(item["path"])
        performances = self._selected_performances(prepared, item)
        if not performances:
            raise IndexError(f"No performances for split={self.split} in {item['path']}")

        if self.multi_perf_group_size > 1:
            group_start = int(perf_slot) * self.multi_perf_group_size
            group = performances[group_start : group_start + self.multi_perf_group_size]
            if len(group) < self.multi_perf_min_group_size:
                raise IndexError(
                    f"Multi-perf group has only {len(group)} performances in {item['path']}"
                )
            return {
                "pn_group_examples": [
                    self._sample_for_performance(prepared, perf, start, end, index)
                    for perf in group
                ]
            }

        # A tiny number of PianoCoRe-A rows were skipped for pitch mismatch. If
        # metadata counted one of those rows, wrap to a valid performance instead.
        perf = performances[int(perf_slot) % len(performances)]
        sample = self._sample_for_performance(prepared, perf, start, end, index)
        dagger_prefix = self._dagger_prefix_cache.get(index)
        if dagger_prefix is not None:
            sample["dagger_prefix_continuous"] = dagger_prefix
        if index in self._dagger_mask_cache:
            sample["dagger_feedback_mask_enabled"] = True
        return sample


def _pn_group_calibration_loss(config, logits, labels, score_shared_raw, attention_mask, group_index):
    """PN moment supervision over aligned performances of the same score window."""
    params = _split_epr_mixture_params(config, logits)
    valid = attention_mask.bool()
    zero_mask = _zero_score_ioi_mask(config, score_shared_raw, attention_mask=valid)
    moments = {}
    for feature, target_col in (("ioi", 0), ("duration", 1), ("velocity", 2)):
        feature_zero = zero_mask if feature == "ioi" else None
        log_probs = _dlm_log_bin_probs(
            config,
            params[f"{feature}_logits"],
            params[f"{feature}_loc"],
            params[f"{feature}_log_scale"],
            feature,
            zero_mask=feature_zero,
        )
        probs = log_probs.exp()
        centers = _dlm_bin_centers(
            config, feature, log_probs, zero_mask=feature_zero
        ).expand_as(probs)
        mean = (probs * centers).sum(-1)
        variance = (probs * (centers - mean.unsqueeze(-1)).square()).sum(-1)
        target = labels[..., target_col].float()
        if feature == "velocity":
            target = target * 127.0
        moments[feature] = (mean, variance, target)

    tau = max(float(getattr(config, "pn_variance_shrinkage_tau", 4.0)), 0.0)
    eps = max(float(getattr(config, "pn_variance_epsilon", 1e-4)), 1e-12)
    mean_terms = []
    variance_terms = {"ioi_zero": [], "ioi_nonzero": [], "duration": [], "velocity": []}
    unique_groups = torch.unique(group_index, sorted=True)
    for group_id in unique_groups:
        rows = group_index == group_id
        count = int(rows.sum().item())
        if count < 2:
            continue
        group_valid = valid[rows].all(dim=0)
        if not group_valid.any():
            continue
        group_zero = zero_mask[rows][0]
        rho = float(count) / float(count + tau) if tau > 0.0 else 1.0
        for feature, (pred_mean, pred_var, target) in moments.items():
            pm = pred_mean[rows]
            pv = pred_var[rows]
            yt = target[rows]
            gt_mean = yt.mean(dim=0)
            gt_var = yt.var(dim=0, unbiased=True)
            model_mean = pm.mean(dim=0)
            model_var = (pv + pm.square()).mean(dim=0) - model_mean.square()
            model_var = model_var.clamp_min(0.0)
            width = 128.0 if feature == "velocity" else 3.0 if feature == "duration" else 5.0
            mean_terms.append(
                torch.nn.functional.smooth_l1_loss(
                    model_mean[group_valid] / width,
                    gt_mean[group_valid] / width,
                    reduction="mean",
                )
            )
            if feature == "ioi":
                feature_groups = (("ioi_zero", group_valid & group_zero), ("ioi_nonzero", group_valid & ~group_zero))
            else:
                feature_groups = ((feature, group_valid),)
            for name, feature_valid in feature_groups:
                if not feature_valid.any():
                    continue
                # Batch-group prior gives a stable target when only 3-4 references exist.
                prior = gt_var[feature_valid].mean().detach()
                shrunk_gt_var = rho * gt_var[feature_valid] + (1.0 - rho) * prior
                variance_terms[name].append(
                    torch.nn.functional.smooth_l1_loss(
                        torch.log(model_var[feature_valid] + eps),
                        torch.log(shrunk_gt_var + eps),
                        reduction="mean",
                    )
                )

    zero = logits.sum() * 0.0
    pn_mean = torch.stack(mean_terms).mean() if mean_terms else zero
    result = {"pn_mean": pn_mean}
    for name, terms in variance_terms.items():
        result[f"pn_var_{name}"] = torch.stack(terms).mean() if terms else zero
    return result


class NodeSFTTrainer(Trainer):
    def _model_config(self, model):
        return model.module.config if hasattr(model, "module") else model.config

    def _dagger_enabled(self):
        return bool(getattr(self, "dagger_prefix_training", False))

    def _dagger_log(self, payload):
        if self.is_world_process_zero():
            data = {"step": self.state.global_step}
            data.update(payload)
            print(json.dumps(data, ensure_ascii=False, sort_keys=True), flush=True)

    def _dagger_next_interval_indices(self, total):
        total = int(total)
        if total <= 0:
            return []
        rank, world_size = distributed_info()
        interval_steps = getattr(self.args, "eval_steps", None)
        if interval_steps is None:
            interval_steps = getattr(self, "dagger_cache_interval_steps", None)
        interval_steps = max(1, int(interval_steps or 1))
        per_device = int(getattr(self.args, "per_device_train_batch_size", 1) or 1)
        grad_accum = int(getattr(self.args, "gradient_accumulation_steps", 1) or 1)
        global_batch_size = int(
            getattr(self, "dagger_global_batch_size", 0)
            or per_device * max(1, int(world_size)) * max(1, grad_accum)
        )
        global_batch_size = max(1, global_batch_size)
        steps_per_epoch = max(1, math.ceil(total / global_batch_size))
        start_step = int(self.state.global_step)

        indices = []
        seen = set()
        for offset in range(interval_steps):
            step_in_epoch = (start_step + offset) % steps_per_epoch
            start = min(step_in_epoch * global_batch_size, total)
            end = min(start + global_batch_size, total)
            for index in range(start, end):
                if index not in seen:
                    seen.add(index)
                    indices.append(index)
        self._dagger_last_scope_info = {
            "interval_steps": interval_steps,
            "global_batch_size": global_batch_size,
            "steps_per_epoch": steps_per_epoch,
            "start_step": start_step,
            "selected_examples": len(indices),
        }
        return indices

    def _estimated_total_train_steps(self, steps_per_epoch=None):
        state_max = int(getattr(getattr(self, "state", None), "max_steps", 0) or 0)
        if state_max > 0:
            return state_max
        args_max = int(getattr(self.args, "max_steps", 0) or 0)
        if args_max > 0:
            return args_max
        if steps_per_epoch is None:
            total = len(self.train_dataset) if self.train_dataset is not None else 0
            per_device = int(getattr(self.args, "per_device_train_batch_size", 1) or 1)
            grad_accum = int(getattr(self.args, "gradient_accumulation_steps", 1) or 1)
            _, world_size = distributed_info()
            steps_per_epoch = max(1, math.ceil(int(total) / max(1, per_device * grad_accum * world_size)))
        return max(1, int(round(float(getattr(self.args, "num_train_epochs", 1.0) or 1.0) * int(steps_per_epoch))))

    def _dagger_window_curriculum_ratio(self, steps_per_epoch=None):
        mode = str(getattr(self, "dagger_window_curriculum", "none") or "none").lower()
        if mode in {"", "none", "off", "false"}:
            return 1.0
        if mode not in {"linear", "linear_window", "window_linear"}:
            raise ValueError(f"Unsupported dagger_window_curriculum={mode}")
        start = float(getattr(self, "dagger_window_curriculum_start", 0.0))
        end = float(getattr(self, "dagger_window_curriculum_end", 1.0))
        total_steps = int(
            getattr(self, "dagger_window_curriculum_steps", 0)
            or self._estimated_total_train_steps(steps_per_epoch=steps_per_epoch)
        )
        progress = min(max(float(int(self.state.global_step)) / max(float(total_steps), 1.0), 0.0), 1.0)
        return min(max(start + (end - start) * progress, 0.0), 1.0)

    def _dagger_training_progress(self, steps_per_epoch=None):
        total_steps = int(
            getattr(self, "dagger_schedule_total_steps", 0)
            or getattr(self, "dagger_window_curriculum_steps", 0)
            or self._estimated_total_train_steps(steps_per_epoch=steps_per_epoch)
        )
        return min(max(float(int(self.state.global_step)) / max(float(total_steps), 1.0), 0.0), 1.0)

    def _dagger_scheduled_window_ratio(self, steps_per_epoch=None):
        schedule = str(getattr(self, "dagger_cache_schedule", "window_curriculum") or "window_curriculum").lower()
        if schedule in {"", "none", "window_curriculum"}:
            return self._dagger_window_curriculum_ratio(steps_per_epoch=steps_per_epoch)
        progress = self._dagger_training_progress(steps_per_epoch=steps_per_epoch)
        if schedule in {"two_stage_tf50", "tf50"}:
            return min(0.5, progress)
        if schedule in {"two_stage_tf50_k1mix50", "tf50_k1mix50"}:
            return min(0.5, progress)
        raise ValueError(f"Unsupported dagger_cache_schedule={schedule}")

    def _dagger_scheduled_k1_fraction(self, steps_per_epoch=None):
        schedule = str(getattr(self, "dagger_cache_schedule", "window_curriculum") or "window_curriculum").lower()
        if schedule not in {"two_stage_tf50_k1mix50", "tf50_k1mix50"}:
            return 0.0
        progress = self._dagger_training_progress(steps_per_epoch=steps_per_epoch)
        if progress <= 0.5:
            return 0.0
        return min(0.5, (progress - 0.5) / 0.5 * 0.5)

    def _apply_dagger_window_curriculum(self, indices, scope_info):
        ratio = self._dagger_scheduled_window_ratio(steps_per_epoch=scope_info.get("steps_per_epoch"))
        mode = str(getattr(self, "dagger_window_curriculum", "none") or "none").lower()
        schedule = str(getattr(self, "dagger_cache_schedule", "window_curriculum") or "window_curriculum").lower()
        scope_info["window_curriculum"] = mode
        scope_info["cache_schedule"] = schedule
        scope_info["window_curriculum_ratio"] = ratio
        if schedule in {"", "none", "window_curriculum"} and mode in {"", "none", "off", "false"}:
            return indices
        keep_count = int(round(len(indices) * ratio))
        scope_info["window_curriculum_requested_examples_before"] = len(indices)
        scope_info["window_curriculum_kept_examples"] = keep_count
        if keep_count <= 0:
            return []
        if keep_count >= len(indices):
            return indices
        seed = int(getattr(self, "dagger_cache_seed", 20260707)) + int(self.state.global_step) + 9173
        rng = random.Random(seed)
        selected = list(indices)
        rng.shuffle(selected)
        return sorted(selected[:keep_count])

    def _split_dagger_cache_indices(self, indices, scope_info):
        k1_fraction = self._dagger_scheduled_k1_fraction(steps_per_epoch=scope_info.get("steps_per_epoch"))
        scope_info["k1_twopass_fraction"] = k1_fraction
        if k1_fraction <= 0.0:
            return list(indices), []
        count = int(round(len(indices) * min(max(k1_fraction, 0.0), 1.0)))
        if count <= 0:
            return list(indices), []
        seed = int(getattr(self, "dagger_cache_seed", 20260707)) + int(self.state.global_step) + 23117
        rng = random.Random(seed)
        shuffled = list(indices)
        rng.shuffle(shuffled)
        k1_set = set(shuffled[:count])
        tf_indices = [index for index in indices if index not in k1_set]
        k1_indices = [index for index in indices if index in k1_set]
        scope_info["tf_pred_examples"] = len(tf_indices)
        scope_info["k1_twopass_examples"] = len(k1_indices)
        return tf_indices, k1_indices

    def refresh_dagger_prefix_cache(self, reason="manual"):
        if not self._dagger_enabled():
            return
        dataset = self.train_dataset
        if dataset is None or not hasattr(dataset, "set_dagger_prefix_cache"):
            self._dagger_log({"event": "dagger_cache_skip", "reason": "unsupported_dataset"})
            return
        cache_type = str(getattr(self, "dagger_cache_type", "tf_pred")).lower()
        if cache_type not in {"tf_pred", "k1_twopass", "mask"}:
            raise ValueError(
                f"Unsupported dagger_cache_type={cache_type}; expected tf_pred, k1_twopass, or mask"
            )
        if str(getattr(self.args, "prediction_loss_only", False)).lower() == "true":
            # The cache path calls the model directly and still obtains logits;
            # this guard only documents that Trainer prediction settings are not used here.
            pass

        total = len(dataset)
        cache_scope = str(getattr(self, "dagger_cache_scope", "random")).lower()
        max_items = getattr(self, "dagger_cache_max_items", None)
        max_interval_fraction = getattr(self, "dagger_cache_max_interval_fraction", None)
        scope_info = {}
        if cache_scope == "next_interval":
            indices = self._dagger_next_interval_indices(total)
            scope_info = getattr(self, "_dagger_last_scope_info", {})
            if max_interval_fraction is not None:
                fraction = max(0.0, min(1.0, float(max_interval_fraction)))
                cap = int(round(len(indices) * fraction))
                scope_info["max_interval_fraction"] = fraction
                scope_info["max_interval_fraction_cap"] = cap
                if cap < len(indices):
                    seed = int(getattr(self, "dagger_cache_seed", 20260707)) + int(self.state.global_step) + 6131
                    rng = random.Random(seed)
                    capped_indices = list(indices)
                    rng.shuffle(capped_indices)
                    indices = sorted(capped_indices[:cap])
            if max_items is not None:
                indices = indices[: int(max_items)]
        elif cache_scope == "random":
            fraction = max(0.0, min(1.0, float(getattr(self, "dagger_cache_fraction", 0.5))))
            count = int(round(total * fraction))
            if max_items is not None:
                count = min(count, int(max_items))
            seed = int(getattr(self, "dagger_cache_seed", 20260707)) + int(self.state.global_step)
            rng = random.Random(seed)
            indices = list(range(total))
            rng.shuffle(indices)
            indices = sorted(indices[:count])
            scope_info = {"fraction": fraction}
        else:
            raise ValueError(f"Unsupported dagger_cache_scope={cache_scope}; expected random or next_interval")

        if len(indices) <= 0:
            dataset.clear_dagger_prefix_cache()
            release_cuda_cache()
            self._dagger_log({"event": "dagger_cache_cleared", "reason": reason, **scope_info})
            return

        indices = self._apply_dagger_window_curriculum(indices, scope_info)
        if len(indices) <= 0:
            dataset.clear_dagger_prefix_cache()
            release_cuda_cache()
            self._dagger_log({"event": "dagger_cache_cleared", "reason": reason, **scope_info})
            return

        tf_indices, k1_indices = self._split_dagger_cache_indices(indices, scope_info)
        cache_type_by_index = {}
        if k1_indices:
            cache_type_by_index.update({int(index): "k1_twopass" for index in k1_indices})
            cache_type_by_index.update({int(index): cache_type for index in tf_indices})

        rank, world_size = distributed_info()
        dist_ready = torch.distributed.is_available() and torch.distributed.is_initialized()
        cache_rank = rank if dist_ready else 0
        cache_world_size = world_size if dist_ready else 1
        local_indices = indices[cache_rank::cache_world_size]
        self._dagger_log(
            {
                "event": "dagger_cache_start",
                "reason": reason,
                "cache_type": cache_type,
                "cache_scope": cache_scope,
                "requested_examples": len(indices),
                "local_examples": len(local_indices),
                "world_size": cache_world_size,
                **scope_info,
            }
        )
        if cache_type == "mask":
            cache_indices = set(int(index) for index in local_indices)
            if dist_ready and cache_world_size > 1:
                gathered = [None for _ in range(cache_world_size)]
                torch.distributed.all_gather_object(gathered, cache_indices)
                merged_indices = set()
                for shard in gathered:
                    if shard:
                        merged_indices.update(int(index) for index in shard)
                cache_indices = merged_indices
            if hasattr(dataset, "set_dagger_mask_cache"):
                dataset.set_dagger_mask_cache(cache_indices)
            else:
                dataset.clear_dagger_prefix_cache()
            release_cuda_cache()
            self._dagger_log(
                {
                    "event": "dagger_cache_refreshed",
                    "reason": reason,
                    "cache_type": cache_type,
                    "cache_scope": cache_scope,
                    "requested_examples": len(indices),
                    "local_examples": len(local_indices),
                    "cached_examples": len(cache_indices),
                    "world_size": cache_world_size,
                    "seconds": 0.0,
                    **scope_info,
                }
            )
            return

        batch_size = int(
            getattr(
                self,
                "dagger_cache_batch_size",
                max(1, int(getattr(self.args, "per_device_eval_batch_size", 1) or 1)),
            )
        )
        num_workers = int(getattr(self, "dagger_cache_num_workers", 0) or 0)
        collator = NodeSFTDataCollator(
            pitch_pad_id=int(self._model_config(self.model).pitch_pad_id),
            task_type=str(getattr(self._model_config(self.model), "task_type", "epr")).lower(),
            use_style_tokens=bool(getattr(self._model_config(self.model), "use_style_tokens", False)),
            dagger_prefix_training=False,
        )
        loader = DataLoader(
            Subset(dataset, local_indices),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=bool(getattr(self.args, "dataloader_pin_memory", False)),
        )

        cache = {}
        was_training = self.model.training
        self.model.eval()
        strategy = str(getattr(self, "dagger_materialize_strategy", "sample")).lower()
        started = time.time()
        try:
            progress = tqdm(
                loader,
                desc=f"dagger {cache_type} cache r{rank}",
                total=len(loader),
                dynamic_ncols=True,
                leave=False,
                disable=(not self.is_world_process_zero()) or bool(getattr(self.args, "disable_tqdm", False)),
            )
            for batch in progress:
                example_indices = batch["example_index"].detach().cpu().tolist()
                batch = self._prepare_inputs(batch)
                with torch.no_grad(), self.autocast_smart_context_manager():
                    outputs = self.model(
                        pitch_ids=batch["pitch_ids"],
                        continuous=batch["continuous"],
                        score_shared_raw=batch["score_shared_raw"],
                        labels_continuous=batch["labels_continuous"],
                        label_valid_mask=batch.get("label_valid_mask"),
                        labels_epr_bins=batch.get("labels_epr_bins"),
                        label_mask=batch.get("label_mask"),
                        attention_mask=batch["attention_mask"],
                        continuous_sampling_strategy=strategy,
                    )
                    pred_tf = _materialize_epr_prediction(
                        self._model_config(self.model),
                        outputs.logits,
                        sampling_strategy=strategy,
                        score_shared_raw=batch["score_shared_raw"],
                    )
                    effective_cache_types = [cache_type_by_index.get(int(index), cache_type) for index in example_indices]
                    needs_k1 = any(item == "k1_twopass" for item in effective_cache_types)
                    if needs_k1:
                        outputs = self.model(
                            pitch_ids=batch["pitch_ids"],
                            continuous=batch["continuous"],
                            score_shared_raw=batch["score_shared_raw"],
                            labels_continuous=batch["labels_continuous"],
                            decoder_feedback_continuous=pred_tf.detach(),
                            label_valid_mask=batch.get("label_valid_mask"),
                            labels_epr_bins=batch.get("labels_epr_bins"),
                            label_mask=batch.get("label_mask"),
                            attention_mask=batch["attention_mask"],
                            continuous_sampling_strategy=strategy,
                        )
                        pred = _materialize_epr_prediction(
                            self._model_config(self.model),
                            outputs.logits,
                            sampling_strategy=strategy,
                            score_shared_raw=batch["score_shared_raw"],
                        )
                        if any(item != "k1_twopass" for item in effective_cache_types):
                            keep_k1 = torch.tensor(
                                [item == "k1_twopass" for item in effective_cache_types],
                                dtype=torch.bool,
                                device=pred.device,
                            ).view(-1, 1, 1)
                            pred = torch.where(keep_k1, pred, pred_tf)
                    else:
                        pred = pred_tf
                lengths = batch["attention_mask"].detach().sum(dim=1).cpu().tolist()
                labels_cpu = batch["labels_continuous"].detach().float().cpu()
                pred_cpu = pred.detach().float().cpu()
                for item_index, length, values, label_values in zip(example_indices, lengths, pred_cpu, labels_cpu):
                    length = int(length)
                    if label_values.shape[-1] >= 9 and values.shape[-1] == 7:
                        expanded = label_values[:length].clone()
                        expanded[:, 0:2] = values[:length, 0:2]
                        expanded[:, 4:5] = values[:length, 2:3]
                        expanded[:, 5:9] = values[:length, 3:7]
                        cache[int(item_index)] = expanded.contiguous()
                    else:
                        cache[int(item_index)] = values[:length].contiguous()
        finally:
            if was_training:
                self.model.train()

        if dist_ready and cache_world_size > 1:
            gathered = [None for _ in range(cache_world_size)]
            torch.distributed.all_gather_object(gathered, cache)
            merged_cache = {}
            for shard in gathered:
                if shard:
                    merged_cache.update(shard)
            cache = merged_cache

        dataset.set_dagger_prefix_cache(cache)
        release_cuda_cache()
        self._dagger_log(
            {
                "event": "dagger_cache_refreshed",
                "reason": reason,
                "cache_type": cache_type,
                "cache_scope": cache_scope,
                "materialize_strategy": strategy,
                "requested_examples": len(indices),
                "local_examples": len(local_indices),
                "cached_examples": len(cache),
                "world_size": cache_world_size,
                "seconds": round(time.time() - started, 3),
                **scope_info,
            }
        )

    def _accumulate_train_loss_components(self, inputs, outputs):
        if not self.model.training or not self.is_world_process_zero():
            return
        labels = inputs.get("labels_continuous")
        attention_mask = inputs.get("attention_mask")
        if labels is None or attention_mask is None or getattr(outputs, "logits", None) is None:
            return
        try:
            loss_mask = inputs.get("label_mask") if str(getattr(self._model_config(self.model), "task_type", "epr")).lower() == "csr" and inputs.get("label_mask") is not None else attention_mask
            components = _compute_integrated_loss_components(
                self._model_config(self.model),
                outputs.logits,
                labels,
                loss_mask,
                labels_epr_bins=inputs.get("labels_epr_bins"),
                score_shared_raw=inputs.get("score_shared_raw"),
                label_valid_mask=inputs.get("label_valid_mask"),
            )
            values = {}
            for name, value in components.items():
                if torch.is_tensor(value):
                    values[name] = float(value.detach().float().cpu().item())
            weights = getattr(self._model_config(self.model), "loss_weights", {}) or {}
            weighted_total = 0.0
            for name in ("ioi", "duration", "velocity", "pedal"):
                if name not in values:
                    continue
                contribution = float(weights.get(name, 1.0)) * values[name]
                values[f"weighted_{name}"] = contribution
                weighted_total += contribution
            for name in ("predictive_variance", "dlm_tail", "dlm_target_tail", "dlm_raw_ms_crps"):
                if name in values:
                    weighted_total += values[name]
            values["components_total"] = weighted_total
            accumulator = getattr(self, "_train_loss_component_sums", None)
            if accumulator is None:
                accumulator = {}
                self._train_loss_component_sums = accumulator
            for name, value in values.items():
                accumulator[name] = accumulator.get(name, 0.0) + value
            self._train_loss_component_count = int(
                getattr(self, "_train_loss_component_count", 0)
            ) + 1
        except Exception as exc:  # noqa: BLE001
            print(
                json.dumps(
                    {
                        "step": int(getattr(self.state, "global_step", 0) or 0),
                        "event": "train_loss_components_failed",
                        "reason": str(exc),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        group_index = inputs.pop("pn_group_index", None)
        inputs.pop("pn_group_sizes", None)
        outputs = model(**inputs)
        loss = outputs.loss
        pn_components = None
        if group_index is not None:
            config = self._model_config(model)
            pn_components = _pn_group_calibration_loss(
                config,
                outputs.logits,
                inputs["labels_continuous"],
                inputs["score_shared_raw"],
                inputs["attention_mask"],
                group_index,
            )
            weights = {
                "pn_mean": float(getattr(config, "pn_mean_loss_lambda", 0.0)),
                "pn_var_ioi_zero": float(getattr(config, "pn_var_ioi_zero_lambda", 0.0)),
                "pn_var_ioi_nonzero": float(getattr(config, "pn_var_ioi_nonzero_lambda", 0.0)),
                "pn_var_duration": float(getattr(config, "pn_var_duration_lambda", 0.0)),
                "pn_var_velocity": float(getattr(config, "pn_var_velocity_lambda", 0.0)),
            }
            loss = loss + sum(weights[name] * value for name, value in pn_components.items())
            if self.model.training and self.is_world_process_zero():
                accumulator = getattr(self, "_train_loss_component_sums", None)
                if accumulator is None:
                    accumulator = {}
                    self._train_loss_component_sums = accumulator
                for name, value in pn_components.items():
                    accumulator[name] = accumulator.get(name, 0.0) + float(value.detach().cpu())
                # _accumulate_train_loss_components below increments the shared count.
        self._accumulate_train_loss_components(inputs, outputs)
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        if "loss" in logs:
            count = int(getattr(self, "_train_loss_component_count", 0) or 0)
            sums = getattr(self, "_train_loss_component_sums", {}) or {}
            if count > 0:
                for name, value in sums.items():
                    logs[f"train_loss_{name}"] = round(float(value) / count, 6)
                self._train_loss_component_sums = {}
                self._train_loss_component_count = 0
        if self.is_world_process_zero():
            printable_logs = {"step": self.state.global_step}
            printable_logs.update(logs)
            print(json.dumps(printable_logs, ensure_ascii=False, sort_keys=True), flush=True)
        return super().log(logs, *args, **kwargs)

    def _get_train_sampler(self, train_dataset=None):
        train_dataset = train_dataset if train_dataset is not None else self.train_dataset
        if train_dataset is None or not hasattr(train_dataset, "__len__"):
            return None
        # Let Trainer/Accelerate shard this sampler for DDP. Returning a
        # DistributedSampler here causes a second split and silently halves the
        # epoch length.
        return SequentialSampler(train_dataset)

    def _clear_eval_dataloader_cache(self):
        if hasattr(self, "_eval_dataloaders"):
            delattr(self, "_eval_dataloaders")

    def _eval_loader_settings(self):
        num_workers = int(getattr(self, "eval_dataloader_num_workers", 0) or 0)
        persistent_workers = bool(getattr(self, "eval_dataloader_persistent_workers", False)) and num_workers > 0
        prefetch_factor = getattr(self, "eval_dataloader_prefetch_factor", None)
        if num_workers <= 0:
            prefetch_factor = None
        pin_memory = bool(
            getattr(
                self,
                "eval_dataloader_pin_memory",
                getattr(self.args, "dataloader_pin_memory", False),
            )
        )
        return {
            "num_workers": num_workers,
            "persistent_workers": persistent_workers,
            "prefetch_factor": prefetch_factor,
            "pin_memory": pin_memory,
        }

    def get_eval_dataloader(self, eval_dataset=None):
        settings = self._eval_loader_settings()
        original_num_workers = self.args.dataloader_num_workers
        original_persistent_workers = self.args.dataloader_persistent_workers
        original_prefetch_factor = self.args.dataloader_prefetch_factor
        original_pin_memory = self.args.dataloader_pin_memory
        self.args.dataloader_num_workers = settings["num_workers"]
        self.args.dataloader_persistent_workers = settings["persistent_workers"]
        self.args.dataloader_prefetch_factor = settings["prefetch_factor"]
        self.args.dataloader_pin_memory = settings["pin_memory"]
        try:
            return super().get_eval_dataloader(eval_dataset=eval_dataset)
        finally:
            self.args.dataloader_num_workers = original_num_workers
            self.args.dataloader_persistent_workers = original_persistent_workers
            self.args.dataloader_prefetch_factor = original_prefetch_factor
            self.args.dataloader_pin_memory = original_pin_memory

    def _list_checkpoint_dirs(self):
        out = Path(self.args.output_dir)
        if not out.exists():
            return []
        dirs = [p for p in out.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
        # exclude checkpoint-best special folder
        dirs = [d for d in dirs if d.name != "checkpoint-best"]
        def step_of(d):
            m = re.match(r"checkpoint-(\d+)$", d.name)
            return int(m.group(1)) if m else -1
        dirs.sort(key=step_of)
        return dirs

    def _cleanup_checkpoints(self, keep_paths):
        out = Path(self.args.output_dir)
        if not out.exists():
            return
        for p in out.iterdir():
            if not p.is_dir():
                continue
            if p.name == "checkpoint-best":
                # keep if requested
                if str(p) in keep_paths:
                    continue
                # otherwise remove
                shutil.rmtree(p)
                continue
            if p.name.startswith("checkpoint-"):
                if str(p) in keep_paths:
                    continue
                shutil.rmtree(p)

    def _sync_checkpoint_best_alias(self):
        if not self.is_world_process_zero():
            return
        best_checkpoint = getattr(getattr(self, "state", None), "best_model_checkpoint", None)
        if not best_checkpoint:
            return

        source = Path(best_checkpoint)
        if not source.exists() or not source.is_dir():
            return

        output_dir = Path(self.args.output_dir)
        best_dir = output_dir / "checkpoint-best"
        marker_path = best_dir / ".best_source"
        try:
            source_marker = str(source.resolve())
        except FileNotFoundError:
            source_marker = str(source)
        try:
            if source.resolve() == best_dir.resolve():
                return
        except FileNotFoundError:
            pass
        if best_dir.exists() and marker_path.exists():
            try:
                if marker_path.read_text().strip() == source_marker:
                    return
            except OSError:
                pass

        tmp_dir = output_dir / f".checkpoint-best.tmp-{os.getpid()}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        try:
            shutil.copytree(source, tmp_dir)
            (tmp_dir / ".best_source").write_text(source_marker + "\n")
            if best_dir.exists():
                shutil.rmtree(best_dir)
            tmp_dir.rename(best_dir)
            print(
                json.dumps(
                    {
                        "step": self.state.global_step,
                        "event": "checkpoint_best_synced",
                        "source": str(source),
                        "alias": str(best_dir),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
        except Exception as exc:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            print(
                json.dumps(
                    {
                        "step": self.state.global_step,
                        "event": "checkpoint_best_sync_failed",
                        "source": str(source),
                        "alias": str(best_dir),
                        "reason": str(exc),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        self._sync_checkpoint_best_alias()

    def _rollout_eval_enabled(self):
        return bool(getattr(self, "rollout_eval_enabled", False))

    def _materialize_rollout_feedback(self, logits, score_shared_raw=None):
        strategy = str(getattr(self, "rollout_eval_feedback_strategy", "sample")).lower()
        return _materialize_epr_prediction(
            self._model_config(self.model),
            logits,
            sampling_strategy=strategy,
            score_shared_raw=score_shared_raw,
        )

    def _rollout_eval_loss_for_k(self, rollout_k):
        rollout_k = int(rollout_k)
        if rollout_k <= 0:
            return None
        loader = self.get_eval_dataloader()
        was_training = self.model.training
        self.model.eval()
        total_loss = 0.0
        total_weight = 0.0
        try:
            for batch in loader:
                batch = self._prepare_inputs(batch)
                attention_mask = batch["attention_mask"]
                batch_weight = float(attention_mask.detach().sum().float().cpu().item())
                labels = batch["labels_continuous"]
                feedback = None
                step_loss = None
                with torch.no_grad(), self.autocast_smart_context_manager():
                    for pass_idx in range(rollout_k + 1):
                        outputs = self.model(
                            pitch_ids=batch["pitch_ids"],
                            continuous=batch["continuous"],
                            score_shared_raw=batch["score_shared_raw"],
                            labels_continuous=labels,
                            decoder_feedback_continuous=feedback,
                            labels_epr_bins=batch.get("labels_epr_bins"),
                            label_mask=batch.get("label_mask"),
                            attention_mask=attention_mask,
                            continuous_sampling_strategy=str(getattr(self, "rollout_eval_materialize_strategy", "sample")),
                        )
                        step_loss = outputs.loss
                        if pass_idx < rollout_k:
                            pred = self._materialize_rollout_feedback(
                                outputs.logits,
                                score_shared_raw=batch["score_shared_raw"],
                            )
                            feedback = pred.detach()
                if step_loss is not None and batch_weight > 0.0:
                    total_loss += float(step_loss.detach().float().cpu().item()) * batch_weight
                    total_weight += batch_weight
        finally:
            if was_training:
                self.model.train()

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
            values = torch.tensor([total_loss, total_weight], dtype=torch.float64, device=device)
            torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
            total_loss = float(values[0].item())
            total_weight = float(values[1].item())
        if total_weight <= 0.0:
            return None
        return total_loss / total_weight

    def _inject_rollout_eval_metrics(self, metrics):
        if not self._rollout_eval_enabled():
            return metrics
        if metrics is None:
            metrics = {}
        rollout_k = int(getattr(self, "rollout_eval_k", 1) or 1)
        rollout_started = time.time()
        if self.is_world_process_zero():
            print(
                json.dumps(
                    {"step": self.state.global_step, "event": "rollout_eval_start", "rollout_k": rollout_k},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
        rollout_loss = self._rollout_eval_loss_for_k(rollout_k)
        if rollout_loss is None:
            return metrics
        tf_loss = float(metrics.get("eval_loss", 0.0))
        weight = float(getattr(self, "rollout_eval_weight", 1.0))
        combined = tf_loss + weight * float(rollout_loss)
        metrics[f"eval_rollout_k{rollout_k}_loss"] = float(rollout_loss)
        metrics["eval_tf_loss"] = tf_loss
        metrics["eval_loss"] = float(combined)
        if self.is_world_process_zero():
            print(
                json.dumps(
                    {
                        "step": self.state.global_step,
                        "event": "rollout_eval_done",
                        "rollout_k": rollout_k,
                        "seconds": round(time.time() - rollout_started, 3),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
        return metrics

    def evaluate(self, *args, **kwargs):
        eval_started = time.time()
        if self.is_world_process_zero():
            print(
                json.dumps(
                    {"step": self.state.global_step, "event": "eval_start"},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
        self._clear_eval_dataloader_cache()
        try:
            try:
                metrics = super().evaluate(*args, **kwargs)
            finally:
                self._clear_eval_dataloader_cache()
        except RuntimeError as exc:
            message = str(exc)
            can_retry = (
                getattr(self, "eval_dataloader_num_workers", 0) not in (None, 0)
                and "DataLoader worker" in message
                and "exited unexpectedly" in message
            )
            if not can_retry:
                raise

            original_num_workers = self.eval_dataloader_num_workers
            original_persistent_workers = getattr(self, "eval_dataloader_persistent_workers", False)
            original_prefetch_factor = getattr(self, "eval_dataloader_prefetch_factor", None)
            if self.is_world_process_zero():
                print(
                    json.dumps(
                        {
                            "step": self.state.global_step,
                            "event": "eval_dataloader_retry",
                            "reason": message,
                            "retry_num_workers": 0,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    flush=True,
                )
            self.eval_dataloader_num_workers = 0
            self.eval_dataloader_persistent_workers = False
            self.eval_dataloader_prefetch_factor = None
            self._clear_eval_dataloader_cache()
            try:
                metrics = super().evaluate(*args, **kwargs)
            finally:
                self.eval_dataloader_num_workers = original_num_workers
                self.eval_dataloader_persistent_workers = original_persistent_workers
                self.eval_dataloader_prefetch_factor = original_prefetch_factor
                self._clear_eval_dataloader_cache()

        metrics = self._inject_rollout_eval_metrics(metrics)
        if self._dagger_enabled() and bool(getattr(self, "dagger_refresh_on_eval", True)):
            self.refresh_dagger_prefix_cache(reason="eval")
        if not self.is_world_process_zero():
            return metrics
        print(
            json.dumps(
                {
                    "step": self.state.global_step,
                    "event": "eval_done",
                    "seconds": round(time.time() - eval_started, 3),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        printable_metrics = {"step": self.state.global_step}
        printable_metrics.update(metrics)
        print(json.dumps(printable_metrics, ensure_ascii=False, sort_keys=True), flush=True)
        return metrics

class NodeSFTDataCollator:
    def __init__(
        self,
        pitch_pad_id=128,
        task_type="epr",
        use_style_tokens=False,
        dagger_prefix_training=False,
        dagger_apply_prob=1.0,
        dagger_replacement_weights=None,
        dagger_seed=42,
        stable_dynamics_training=False,
        stable_apply_prob=0.30,
        stable_channels=None,
        stable_noise_modes=None,
        stable_seed=42,
        stable_zero_ioi_nonnegative_feedback=False,
        epr_timing_target="log_deviation",
        tail_mask_enabled=False,
        tail_mask_tf_clamp=True,
        tail_mask_ioi_min=-1.5,
        tail_mask_ioi_max=1.5,
        tail_mask_duration_min=-2.0,
        tail_mask_duration_max=2.0,
    ):
        self.pitch_pad_id = pitch_pad_id
        self.task_type = task_type
        self.use_style_tokens = bool(use_style_tokens)
        self.dagger_prefix_training = bool(dagger_prefix_training)
        self.dagger_apply_prob = float(dagger_apply_prob)
        self.dagger_replacement_weights = normalize_dagger_replacement_weights(dagger_replacement_weights)
        self._dagger_rng = random.Random(int(dagger_seed))
        self.stable_dynamics_training = bool(stable_dynamics_training)
        self.stable_apply_prob = float(stable_apply_prob)
        self.stable_columns = stable_target_columns(stable_channels)
        self.stable_noise_modes = normalize_stable_noise_modes(stable_noise_modes)
        self._stable_rng = random.Random(int(stable_seed))
        self.stable_zero_ioi_nonnegative_feedback = bool(stable_zero_ioi_nonnegative_feedback)
        self.epr_timing_target = str(epr_timing_target).lower()
        self.tail_mask_enabled = bool(tail_mask_enabled)
        self.tail_mask_tf_clamp = bool(tail_mask_tf_clamp)
        self.tail_mask_ioi_min = float(tail_mask_ioi_min)
        self.tail_mask_ioi_max = float(tail_mask_ioi_max)
        self.tail_mask_duration_min = float(tail_mask_duration_min)
        self.tail_mask_duration_max = float(tail_mask_duration_max)

    def _sample_dagger_mode(self):
        draw = self._dagger_rng.random()
        running = 0.0
        last_key = None
        for key, weight in self.dagger_replacement_weights.items():
            running += float(weight)
            last_key = key
            if draw <= running:
                return key
        return last_key or "full"

    def _sample_stable_noise_mode(self):
        draw = self._stable_rng.random()
        running = 0.0
        last_spec = None
        for spec in self.stable_noise_modes.values():
            running += float(spec["prob"])
            last_spec = spec
            if draw <= running:
                return spec
        return last_spec

    def __call__(self, examples):
        pn_group_index = None
        pn_group_sizes = None
        if examples and all("pn_group_examples" in example for example in examples):
            grouped = [example["pn_group_examples"] for example in examples]
            pn_group_sizes = torch.tensor([len(group) for group in grouped], dtype=torch.long)
            pn_group_index = torch.tensor(
                [group_idx for group_idx, group in enumerate(grouped) for _ in group],
                dtype=torch.long,
            )
            examples = [sample for group in grouped for sample in group]
        pitch_tensors = [torch.tensor(example["pitch_ids"], dtype=torch.long) for example in examples]
        continuous_tensors = [
            torch.tensor(example["continuous"], dtype=torch.float32) for example in examples
        ]
        label_tensors = [
            torch.tensor(example["labels_continuous"], dtype=torch.float32) for example in examples
        ]
        score_shared_raw_tensors = [
            torch.tensor(example["score_shared_raw"], dtype=torch.float32) for example in examples
        ]
        interpolated_tensors = [
            torch.tensor(example["interpolated"], dtype=torch.bool) for example in examples
        ]
        labels_epr_bins_tensors = None
        if all("labels_epr_bins" in example for example in examples):
            labels_epr_bins_tensors = [
                torch.tensor(example["labels_epr_bins"], dtype=torch.long) for example in examples
            ]
        dagger_prefix_tensors = None
        if self.dagger_prefix_training:
            dagger_prefix_tensors = []
            for example in examples:
                if "dagger_prefix_continuous" in example:
                    prefix = example["dagger_prefix_continuous"]
                    if torch.is_tensor(prefix):
                        dagger_prefix_tensors.append(prefix.detach().clone().to(dtype=torch.float32))
                    else:
                        dagger_prefix_tensors.append(torch.tensor(prefix, dtype=torch.float32))
                else:
                    dagger_prefix_tensors.append(torch.empty(0, dtype=torch.float32))

        pitch_ids = pad_sequence(pitch_tensors, batch_first=True, padding_value=self.pitch_pad_id)
        continuous = pad_sequence(continuous_tensors, batch_first=True, padding_value=0.0)
        labels_continuous = pad_sequence(label_tensors, batch_first=True, padding_value=0.0)
        score_shared_raw = pad_sequence(score_shared_raw_tensors, batch_first=True, padding_value=0.0)
        interpolated = pad_sequence(interpolated_tensors, batch_first=True, padding_value=False)
        attention_mask = (pitch_ids != self.pitch_pad_id).long()
        label_valid_mask = torch.ones_like(labels_continuous, dtype=torch.bool)
        decoder_feedback_continuous = None
        if (
            self.tail_mask_enabled
            and self.task_type == "epr"
            and labels_continuous.shape[-1] >= 2
            and self.epr_timing_target in {
                "floor_log_deviation",
                "floor_log_dev",
                "pure_log_deviation",
                "pure_log_dev",
            }
        ):
            valid_rows = attention_mask.bool()
            nonzero_ioi = valid_rows & (score_shared_raw[..., 0] > 1e-8)
            duration_rows = valid_rows & (score_shared_raw[..., 1] > 1e-8)
            ioi_invalid = nonzero_ioi & (
                (labels_continuous[..., 0] < self.tail_mask_ioi_min)
                | (labels_continuous[..., 0] > self.tail_mask_ioi_max)
            )
            duration_invalid = duration_rows & (
                (labels_continuous[..., 1] < self.tail_mask_duration_min)
                | (labels_continuous[..., 1] > self.tail_mask_duration_max)
            )
            label_valid_mask[..., 0] = label_valid_mask[..., 0] & ~ioi_invalid
            label_valid_mask[..., 1] = label_valid_mask[..., 1] & ~duration_invalid
            if self.tail_mask_tf_clamp:
                decoder_feedback_continuous = labels_continuous.clone()
                decoder_feedback_continuous[..., 0] = torch.where(
                    nonzero_ioi,
                    decoder_feedback_continuous[..., 0].clamp(self.tail_mask_ioi_min, self.tail_mask_ioi_max),
                    decoder_feedback_continuous[..., 0],
                )
                decoder_feedback_continuous[..., 1] = torch.where(
                    duration_rows,
                    decoder_feedback_continuous[..., 1].clamp(
                        self.tail_mask_duration_min,
                        self.tail_mask_duration_max,
                    ),
                    decoder_feedback_continuous[..., 1],
                )
        label_mask = None
        if self.task_type == "csr":
            label_mask_tensors = [
                torch.tensor(example["label_mask"], dtype=torch.long) for example in examples
            ]
            label_mask = pad_sequence(label_mask_tensors, batch_first=True, padding_value=0)

        batch = {
            "example_index": torch.tensor([int(example["example_index"]) for example in examples], dtype=torch.long),
            "pitch_ids": pitch_ids,
            "continuous": continuous,
            "labels_continuous": labels_continuous,
            "score_shared_raw": score_shared_raw,
            "attention_mask": attention_mask,
            "interpolated": interpolated,
        }
        if pn_group_index is not None:
            batch["pn_group_index"] = pn_group_index
            batch["pn_group_sizes"] = pn_group_sizes
        if self.tail_mask_enabled:
            batch["label_valid_mask"] = label_valid_mask
        if decoder_feedback_continuous is not None:
            batch["decoder_feedback_continuous"] = decoder_feedback_continuous
        if dagger_prefix_tensors is not None:
            decoder_feedback_continuous = batch.get("decoder_feedback_continuous")
            if decoder_feedback_continuous is None:
                decoder_feedback_continuous = labels_continuous.clone()
            else:
                decoder_feedback_continuous = decoder_feedback_continuous.clone()
            decoder_feedback_mask = torch.zeros_like(labels_continuous)
            used_any = False
            for row_idx, prefix_tensor in enumerate(dagger_prefix_tensors):
                mask_enabled = bool(examples[row_idx].get("dagger_feedback_mask_enabled", False))
                if prefix_tensor.numel() == 0 and not mask_enabled:
                    continue
                if self._dagger_rng.random() > self.dagger_apply_prob:
                    continue
                prefix_len = int(prefix_tensor.shape[0]) if prefix_tensor.numel() > 0 else int(attention_mask[row_idx].sum().item())
                length = min(prefix_len, int(attention_mask[row_idx].sum().item()))
                if length <= 0:
                    continue
                mode = self._sample_dagger_mode()
                cols = dagger_target_columns(mode, output_dim=labels_continuous.shape[-1])
                if not cols:
                    continue
                if mask_enabled:
                    decoder_feedback_mask[row_idx, :length, cols] = 1.0
                else:
                    decoder_feedback_continuous[row_idx, :length, cols] = prefix_tensor[:length, cols]
                used_any = True
            if used_any:
                batch["decoder_feedback_continuous"] = decoder_feedback_continuous
                if decoder_feedback_mask.any():
                    batch["decoder_feedback_mask"] = decoder_feedback_mask
        if self.stable_dynamics_training:
            if self.task_type != "epr":
                raise ValueError("stable_dynamics_training currently supports task_type=epr only")
            decoder_feedback_continuous = batch.get("decoder_feedback_continuous")
            if decoder_feedback_continuous is None:
                decoder_feedback_continuous = labels_continuous.clone()
            else:
                decoder_feedback_continuous = decoder_feedback_continuous.clone()
            stable_feedback_mask = torch.zeros_like(labels_continuous)
            valid = attention_mask.bool()
            apply_mask = torch.rand(valid.shape, dtype=torch.float32) < self.stable_apply_prob
            apply_mask = apply_mask & valid
            if apply_mask.any():
                for row_idx in range(labels_continuous.shape[0]):
                    row_mask = apply_mask[row_idx]
                    if not row_mask.any():
                        continue
                    spec = self._sample_stable_noise_mode()
                    for col in self.stable_columns:
                        if col == 0:
                            mu = float(spec["ioi_mu"])
                            sigma = float(spec["ioi_sigma"])
                        elif col == 1:
                            mu = float(spec["duration_mu"])
                            sigma = float(spec["duration_sigma"])
                        else:
                            continue
                        noise = torch.randn_like(decoder_feedback_continuous[row_idx, :, col]) * sigma + mu
                        updated = decoder_feedback_continuous[row_idx, row_mask, col] + noise[row_mask]
                        if _uses_raw_log_deviation_target(self.epr_timing_target):
                            if (
                                col == 0
                                and self.stable_zero_ioi_nonnegative_feedback
                                and score_shared_raw is not None
                            ):
                                score_zero_mask = score_shared_raw[row_idx, row_mask, 0] <= 1e-8
                                if score_zero_mask.any():
                                    updated = updated.clone()
                                    updated[score_zero_mask] = updated[score_zero_mask].clamp_min(0.0)
                            decoder_feedback_continuous[row_idx, row_mask, col] = updated
                        else:
                            decoder_feedback_continuous[row_idx, row_mask, col] = updated.clamp(0.0, 1.0)
                        stable_feedback_mask[row_idx, row_mask, col] = 1.0
                batch["decoder_feedback_continuous"] = decoder_feedback_continuous
                batch["stable_feedback_mask"] = stable_feedback_mask
        if self.use_style_tokens:
            batch["style_creator_ids"] = torch.tensor(
                [int(example["style_creator_id"]) for example in examples],
                dtype=torch.long,
            )
            batch["style_source_ids"] = torch.tensor(
                [int(example["style_source_id"]) for example in examples],
                dtype=torch.long,
            )
            batch["style_score_stats"] = torch.tensor(
                [example["style_score_stats"] for example in examples],
                dtype=torch.float32,
            )
            batch["style_perf_stats"] = torch.tensor(
                [example["style_perf_stats"] for example in examples],
                dtype=torch.float32,
            )
            batch["style_perf_is_pad"] = torch.tensor(
                [bool(example["style_perf_is_pad"]) for example in examples],
                dtype=torch.bool,
            )
        if label_mask is not None:
            batch["label_mask"] = label_mask
        if labels_epr_bins_tensors is not None:
            batch["labels_epr_bins"] = pad_sequence(
                labels_epr_bins_tensors,
                batch_first=True,
                padding_value=0,
            )
        return batch


def create_model(train_config):
    dtype = torch.bfloat16 if train_config.get("bf16", False) and torch.cuda.is_available() else torch.float32
    backbone_type = train_config.get("backbone_type", "t5").lower()
    task_type = train_config.get("task_type", "epr").lower()
    input_feature_mode = infer_input_feature_mode(train_config)
    epr_timing_target = str(train_config.get("epr_timing_target", "log_deviation")).lower()
    timing_control_mode = resolve_timing_control_mode(
        timing_control_mode=train_config.get("timing_control_mode"),
        use_timing_scale_bit=train_config.get("use_timing_scale_bit", False),
    )
    if task_type in {"epr", "csr"} and timing_control_mode not in {"log_scaled", "floor_log", "dinr_floor_log", "raw_log", "raw", "raw_seconds"}:
        raise ValueError("Integrated INR requires timing_control_mode=log_scaled, floor_log, dinr_floor_log, raw_log, or raw")
    if task_type == "epr" and epr_timing_target not in {
        "log_deviation",
        "log_dev",
        "raw_log_deviation",
        "raw_log_dev",
        "floor_log_deviation",
        "floor_log_dev",
        "pure_log_deviation",
        "pure_log_dev",
        "raw_deviation",
        "raw_dev",
        "raw_seconds_deviation",
        "raw_seconds_dev",
        "absolute",
        "absolute_log",
        "log_absolute",
        "raw_log_absolute",
        "absolute_raw_log",
        "floor_log_absolute",
    }:
        raise ValueError("EPR requires a supported deviation or absolute timing target")
    use_timing_scale_bit = timing_control_mode == "piecewise_scale_bit"
    note_embedding_mode = str(train_config.get("note_embedding_mode", "sine")).lower()
    slot_share_role_encoders = train_config.get("slot_share_role_encoders")
    resume_path = train_config.get("resume_path")
    if slot_share_role_encoders is None:
        checkpoint_config = None
        if resume_path and Path(resume_path).is_dir():
            checkpoint_config_path = Path(resume_path) / "config.json"
            if checkpoint_config_path.exists():
                checkpoint_config = json.loads(checkpoint_config_path.read_text(encoding="utf-8"))
        slot_share_role_encoders = (
            bool(checkpoint_config.get("slot_share_role_encoders", False))
            if checkpoint_config is not None
            else True
        )
        train_config["slot_share_role_encoders"] = slot_share_role_encoders
    score_note_schema = score_note_input_schema(train_config)
    decoder_note_schema = decoder_note_input_schema(train_config)
    if input_feature_mode != "integrated":
        raise ValueError(f"INR0624 only supports input_feature_mode=integrated, got {input_feature_mode}")
    if note_embedding_mode not in {"sine", "cine", "slot_attribute"}:
        raise ValueError(
            f"INR0624 only supports note_embedding_mode in {{'sine', 'cine', 'slot_attribute'}}, "
            f"got {note_embedding_mode}"
        )
    if task_type == "epr":
        train_config.setdefault("pedal_representation", "start_valley")
        configured_output_dim = int(
            train_config.get(
                "output_continuous_dim",
                train_config.get(
                    "continuous_dim",
                    default_epr_output_dim(
                        epr_timing_target,
                        train_config.get("pedal_representation", "start_valley"),
                        legacy_dual_timing_head=False,
                    ),
                ),
            )
        )
        legacy_dual_timing_head = bool(
            train_config.get(
                "legacy_dual_timing_head",
                epr_timing_target in {"raw_log_deviation", "raw_log_dev"}
                and configured_output_dim >= 9,
            )
        )
        train_config["legacy_dual_timing_head"] = legacy_dual_timing_head
        if "epr_distribution" not in train_config:
            raise ValueError("EPR config must set epr_distribution explicitly")
        distribution = str(train_config["epr_distribution"]).lower()
        supported_distributions = {
            "point",
            "huber",
            "deterministic_huber",
            "beta_mu_kappa",
            "categorical",
            "hard_categorical",
            "soft_categorical",
            "dinr",
            "dinr_categorical",
            "metric_categorical",
            "lan",
            "can",
            "ican",
            "iln",
            "logistic_normal",
            "mixture_logistic_normal",
            "mixture_beta",
            "dlm",
            "discretized_logistic_mixture",
            "sn",
            "skew_normal",
            "bounded_skew_normal",
            "bounded_sn",
            "tanh_student_t",
            "bounded_tanh_student_t",
        }
        if distribution not in supported_distributions:
            raise ValueError(f"Unsupported epr_distribution={distribution}")
        scalar_distributions = {
            "lan",
            "can",
            "ican",
            "iln",
            "logistic_normal",
            "mixture_logistic_normal",
            "mixture_beta",
            "sn",
            "skew_normal",
            "bounded_skew_normal",
            "bounded_sn",
            "tanh_student_t",
            "bounded_tanh_student_t",
        }
        pedal_distribution = str(train_config.get("pedal_distribution", distribution)).lower()
        if pedal_distribution not in supported_distributions:
            raise ValueError(f"Unsupported pedal_distribution={pedal_distribution}")
        if pedal_distribution in scalar_distributions and distribution not in scalar_distributions:
            raise ValueError(
                "pedal_distribution can only override another scalar EPR distribution; "
                f"got epr_distribution={distribution}, pedal_distribution={pedal_distribution}"
            )
        if distribution in scalar_distributions or pedal_distribution in scalar_distributions:
            missing_keys = [
                key
                for key in (
                    "epr_mixture_components",
                    "epr_distribution_eps",
                    "logistic_normal_sigma_min",
                    "logistic_normal_sigma_max",
                )
                if key not in train_config
            ]
            if distribution == "mixture_beta" or pedal_distribution == "mixture_beta":
                missing_keys.append("beta_alpha_min") if "beta_alpha_min" not in train_config else None
            if missing_keys:
                raise ValueError(f"EPR {distribution} config is missing required keys: {missing_keys}")
            components = int(train_config["epr_mixture_components"])
            if components < 1:
                raise ValueError(f"epr_mixture_components must be >= 1, got {components}")
            if distribution in {"sn", "skew_normal", "bounded_skew_normal", "bounded_sn", "tanh_student_t", "bounded_tanh_student_t", "lan", "can", "ican", "iln", "logistic_normal"} and components != 1:
                raise ValueError(f"epr_distribution={distribution} requires epr_mixture_components=1")
        if distribution in {"dlm", "discretized_logistic_mixture"}:
            dlm_components = int(train_config.get("dlm_components", train_config.get("epr_mixture_components", 8)))
            if dlm_components < 1:
                raise ValueError(f"dlm_components must be >= 1, got {dlm_components}")
            velocity_distribution = str(train_config.get("velocity_distribution", "skew_normal")).lower()
            if velocity_distribution not in {"sn", "skew_normal", "dlm", "discretized_logistic_mixture"}:
                raise ValueError(
                    "DLM supports velocity_distribution=skew_normal or dlm, "
                    f"got {velocity_distribution}"
                )
        if distribution == "beta_mu_kappa":
            missing_beta_keys = [
                key for key in ("beta_eps", "beta_kappa_min")
                if key not in train_config
            ]
            if missing_beta_keys:
                raise ValueError(f"EPR beta_mu_kappa config is missing required keys: {missing_beta_keys}")
        if epr_timing_target in {"log_deviation", "log_dev"}:
            pedal_representation = normalize_pedal_representation(train_config.get("pedal_representation", "start_valley"))
            expected_output_dim = 3 + pedal_representation_dim(pedal_representation)
        elif epr_timing_target in {
            "raw_log_deviation",
            "raw_log_dev",
            "floor_log_deviation",
            "floor_log_dev",
            "pure_log_deviation",
            "pure_log_dev",
        }:
            pedal_representation = normalize_pedal_representation(train_config.get("pedal_representation", "start_valley"))
            expected_output_dim = (5 if legacy_dual_timing_head else 3) + pedal_representation_dim(pedal_representation)
        elif epr_timing_target in {"raw_log_absolute", "absolute_raw_log"}:
            pedal_representation = normalize_pedal_representation(train_config.get("pedal_representation", "start_valley"))
            expected_output_dim = 5 + pedal_representation_dim(pedal_representation)
        elif epr_timing_target in {"absolute", "absolute_log", "log_absolute"}:
            pedal_representation = normalize_pedal_representation(train_config.get("pedal_representation", "start_valley"))
            expected_output_dim = 3 + pedal_representation_dim(pedal_representation)
        elif epr_timing_target in {"raw_deviation", "raw_dev", "raw_seconds_deviation", "raw_seconds_dev"}:
            pedal_representation = normalize_pedal_representation(train_config.get("pedal_representation", "start_valley"))
            expected_output_dim = 3 + pedal_representation_dim(pedal_representation)
        else:
            expected_output_dim = None
        if expected_output_dim is not None and "output_continuous_dim" not in train_config:
            train_config["output_continuous_dim"] = expected_output_dim
        if expected_output_dim is not None and int(train_config.get("output_continuous_dim", train_config["continuous_dim"])) != expected_output_dim:
            raise ValueError(
                f"EPR with pedal_representation={pedal_representation} "
                f"requires output_continuous_dim={expected_output_dim}"
            )
        if expected_output_dim is not None and distribution not in {
            "point",
            "huber",
            "deterministic_huber",
            "lan",
            "can",
            "ican",
            "iln",
            "logistic_normal",
            "mixture_logistic_normal",
            "beta_mu_kappa",
            "mixture_beta",
            "sn",
            "skew_normal",
            "bounded_skew_normal",
            "bounded_sn",
            "tanh_student_t",
            "bounded_tanh_student_t",
            "dlm",
            "discretized_logistic_mixture",
            "dinr",
            "dinr_categorical",
            "metric_categorical",
        }:
            raise ValueError(
                "deviation EPR currently supports point/huber, SN, lan/can/ican, LN/MLN, beta, and mixture_beta, "
                f"got epr_distribution={distribution}"
            )
    elif task_type == "csr":
        expected_output_dim = integrated_csr_output_dim()
        actual_output_dim = int(train_config.get("output_continuous_dim", train_config["continuous_dim"]))
        if actual_output_dim != expected_output_dim:
            raise ValueError(
                f"Integrated INR0624 CSR expects output_continuous_dim={expected_output_dim}, got {actual_output_dim}"
            )
    score_feature_dim = train_config.get("score_feature_dim", 8)
    pedal_dim = pedal_representation_dim(train_config.get("pedal_representation", "start_valley"))
    musical_feature_mode = str(
        train_config.get(
            "musical_feature_mode",
            "continuous" if task_type == "csr" else "categorical",
        )
    ).lower()
    default_score_input_dim = (
        score_musical_input_dim(
            timing_control_mode=timing_control_mode,
            use_timing_scale_bit=use_timing_scale_bit,
            musical_feature_mode=musical_feature_mode,
        )
        if task_type == "epr" and score_note_schema == "score_musical"
        else integrated_epr_input_dim(
            timing_control_mode=timing_control_mode,
            use_timing_scale_bit=use_timing_scale_bit,
            musical_feature_mode=musical_feature_mode,
            pedal_control_dim=pedal_dim,
        )
        if task_type in {"epr", "csr"} and input_feature_mode == "integrated"
        else default_input_continuous_dim(
            task_type,
            input_feature_mode,
            score_feature_dim=score_feature_dim,
            continuous_dim=train_config.get("continuous_dim", 7),
            musical_feature_mode=musical_feature_mode,
        )
    )
    score_input_continuous_dim = default_score_input_dim
    decoder_input_continuous_dim = (
        decoder_perf_target_input_dim(
            train_config.get("output_continuous_dim", train_config.get("continuous_dim", 7))
        )
        if task_type == "epr" and decoder_note_schema == "perf_target"
        else score_input_continuous_dim
    )
    input_continuous_dim = score_input_continuous_dim
    train_config["input_continuous_dim"] = input_continuous_dim
    train_config["score_input_continuous_dim"] = score_input_continuous_dim
    train_config["decoder_input_continuous_dim"] = decoder_input_continuous_dim
    if task_type in {"epr", "csr"} and input_feature_mode == "integrated":
        if task_type == "epr" and score_note_schema == "score_musical":
            expected_score_dim = score_musical_input_dim(
                timing_control_mode=timing_control_mode,
                use_timing_scale_bit=use_timing_scale_bit,
                musical_feature_mode=musical_feature_mode,
            )
        else:
            expected_score_dim = integrated_epr_input_dim(
                timing_control_mode=timing_control_mode,
                use_timing_scale_bit=use_timing_scale_bit,
                musical_feature_mode=musical_feature_mode,
                pedal_control_dim=pedal_dim,
            )
        if int(score_input_continuous_dim) != expected_score_dim:
            raise ValueError(
                f"Integrated INR0624 {task_type.upper()} expects score_input_continuous_dim={expected_score_dim} "
                f"for score_note_schema={score_note_schema}, got {score_input_continuous_dim}"
            )
        if task_type == "epr" and decoder_note_schema == "perf_target":
            expected_decoder_dim = decoder_perf_target_input_dim(
                train_config.get("output_continuous_dim", train_config.get("continuous_dim", 7))
            )
            if int(decoder_input_continuous_dim) != expected_decoder_dim:
                raise ValueError(
                    f"Integrated INR0624 EPR expects decoder_input_continuous_dim={expected_decoder_dim} "
                    f"for decoder_note_schema={decoder_note_schema}, got {decoder_input_continuous_dim}"
                )
    model_config = IntegratedPianoT5GemmaConfig(
        backbone_type=backbone_type,
        hidden_size=train_config["hidden_size"],
        intermediate_size=train_config["intermediate_size"],
        num_attention_heads=train_config["num_attention_heads"],
        num_key_value_heads=train_config["num_key_value_heads"],
        head_dim=train_config["head_dim"],
        encoder_layers_num=train_config["encoder_layers_num"],
        decoder_layers_num=train_config["decoder_layers_num"],
        gpt_layers_num=train_config.get("gpt_layers_num"),
        bert_layers_num=train_config.get("bert_layers_num"),
        max_position_embeddings=train_config.get("max_position_embeddings", 4096),
        attention_dropout=train_config.get("attention_dropout", 0.0),
        attn_implementation=train_config.get("attn_implementation", "sdpa"),
        continuous_dim=train_config["continuous_dim"],
        input_continuous_dim=input_continuous_dim,
        score_input_continuous_dim=score_input_continuous_dim,
        decoder_input_continuous_dim=decoder_input_continuous_dim,
        output_continuous_dim=train_config.get("output_continuous_dim", train_config["continuous_dim"]),
        score_feature_dim=score_feature_dim,
        max_time_ms=train_config["max_time_ms"],
        pedal_output_activation=train_config.get("pedal_output_activation", "sigmoid"),
        task_type=task_type,
        time_loss_type=train_config["time_loss_type"],
        value_loss_type=train_config["value_loss_type"],
        csr_grid_loss_type=train_config.get("csr_grid_loss_type", "huber"),
        csr_grid_step=train_config.get("csr_grid_step", 1.0 / 24.0),
        csr_grid_soft_ce_tau=train_config.get("csr_grid_soft_ce_tau", 1.5),
        csr_mo_max=train_config.get("csr_mo_max", 6.0),
        csr_md_max=train_config.get("csr_md_max", 6.0),
        csr_ml_max=train_config.get("csr_ml_max", 6.0),
        huber_delta=train_config["huber_delta"],
        loss_weights=train_config["loss_weights"],
        csr_loss_weights=train_config.get("csr_loss_weights"),
        decoder_input_mode=train_config["decoder_input_mode"],
        input_feature_mode=input_feature_mode,
        note_embedding_mode=note_embedding_mode,
        score_note_input_schema=score_note_schema,
        decoder_note_input_schema=decoder_note_schema,
        special_note_vocab_size=train_config.get("special_note_vocab_size", 5),
        special_note_ids=train_config.get("special_note_ids"),
        use_full_type_embedding=train_config.get("use_full_type_embedding", True),
        use_group_presence_mask=train_config.get("use_group_presence_mask", True),
        head_input_mode=train_config.get("head_input_mode", "full"),
        embedding_depth=train_config.get("embedding_depth", 2),
        head_depth=train_config.get("head_depth", 2),
        head_width_multiplier=train_config.get("head_width_multiplier", 1.0),
        head_activation=train_config.get("head_activation", "gelu"),
        slot_version=train_config.get("slot_version"),
        slot_dim=train_config.get("slot_dim"),
        slot_fusion=train_config.get("slot_fusion", "mlp"),
        slot_gates=train_config.get("slot_gates", False),
        slot_share_role_encoders=slot_share_role_encoders,
        decoder_head_layout=train_config.get("decoder_head_layout", "pyramid4"),
        decoder_head_expand_ratio=train_config.get("decoder_head_expand_ratio", 2.0),
        decoder_head_shrink_ratio=train_config.get("decoder_head_shrink_ratio", 0.5),
        epr_distribution=train_config.get("epr_distribution", "point"),
        pedal_distribution=train_config.get("pedal_distribution"),
        epr_mixture_components=train_config.get("epr_mixture_components", 1),
        epr_distribution_eps=train_config.get("epr_distribution_eps"),
        logistic_normal_sigma_min=train_config.get("logistic_normal_sigma_min", 1e-3),
        logistic_normal_sigma_max=train_config.get("logistic_normal_sigma_max", 10.0),
        skew_normal_sigma_min=train_config.get("skew_normal_sigma_min", train_config.get("logistic_normal_sigma_min", 1e-4)),
        skew_normal_sigma_max=train_config.get("skew_normal_sigma_max", train_config.get("logistic_normal_sigma_max", 1e4)),
        bounded_floorlog_support=train_config.get("bounded_floorlog_support", False),
        velocity_distribution=train_config.get("velocity_distribution"),
        dlm_components=train_config.get("dlm_components"),
        dlm_timing_bins=train_config.get("dlm_timing_bins", 256),
        dlm_velocity_bins=train_config.get("dlm_velocity_bins", 128),
        dlm_ioi_zero_min=train_config.get("dlm_ioi_zero_min", 0.0),
        dlm_ioi_zero_max=train_config.get("dlm_ioi_zero_max", 5.0),
        dlm_ioi_nonzero_min=train_config.get("dlm_ioi_nonzero_min", -2.5),
        dlm_ioi_nonzero_max=train_config.get("dlm_ioi_nonzero_max", 1.5),
        dlm_duration_min=train_config.get("dlm_duration_min", -3.0),
        dlm_duration_max=train_config.get("dlm_duration_max", 2.0),
        dlm_velocity_min=train_config.get("dlm_velocity_min", -0.5),
        dlm_velocity_max=train_config.get("dlm_velocity_max", 127.5),
        dlm_scale_min=train_config.get("dlm_scale_min", 1e-3),
        dlm_scale_max=train_config.get("dlm_scale_max", 10.0),
        dlm_timing_scale_min=train_config.get("dlm_timing_scale_min"),
        dlm_timing_scale_max=train_config.get("dlm_timing_scale_max"),
        dlm_timing_scale_parameterization=train_config.get("dlm_timing_scale_parameterization", "legacy_clamp"),
        dlm_ioi_nonzero_scale_max=train_config.get("dlm_ioi_nonzero_scale_max"),
        dlm_ioi_zero_scale_max=train_config.get("dlm_ioi_zero_scale_max"),
        dlm_duration_scale_max=train_config.get("dlm_duration_scale_max"),
        dlm_velocity_scale_min=train_config.get("dlm_velocity_scale_min"),
        dlm_velocity_scale_max=train_config.get("dlm_velocity_scale_max"),
        dlm_velocity_scale_parameterization=train_config.get("dlm_velocity_scale_parameterization", "legacy_clamp"),
        dlm_tail_loss_lambda=train_config.get("dlm_tail_loss_lambda", 0.0),
        dlm_tail_radius=train_config.get("dlm_tail_radius", 0.05),
        dlm_target_tail_loss_lambda=train_config.get("dlm_target_tail_loss_lambda", 0.0),
        dlm_target_tail_radius_frac=train_config.get("dlm_target_tail_radius_frac", 0.0),
        dlm_target_tail_ioi_radius=train_config.get("dlm_target_tail_ioi_radius"),
        dlm_target_tail_duration_radius=train_config.get("dlm_target_tail_duration_radius"),
        dlm_timing_weighted_nll_alpha=train_config.get("dlm_timing_weighted_nll_alpha", 0.0),
        dlm_timing_weight_min=train_config.get("dlm_timing_weight_min", 0.5),
        dlm_timing_weight_max=train_config.get("dlm_timing_weight_max", 4.0),
        dlm_raw_ms_crps_lambda=train_config.get("dlm_raw_ms_crps_lambda", 0.0),
        dlm_raw_ms_crps_scale_ms=train_config.get("dlm_raw_ms_crps_scale_ms", 1000.0),
        dlm_sampling_temperature=train_config.get("dlm_sampling_temperature", 1.0),
        dlm_ioi_zero_inflated=train_config.get("dlm_ioi_zero_inflated", False),
        dlm_pedal_zero_one_inflated=train_config.get("dlm_pedal_zero_one_inflated", False),
        dlm_pedal_inflated_eps=train_config.get("dlm_pedal_inflated_eps", 0.5),
        dlm_timing_sample_truncate_radius=train_config.get("dlm_timing_sample_truncate_radius", 0.0),
        dlm_timing_sample_truncate_center=train_config.get("dlm_timing_sample_truncate_center", "mean"),
        timing_sample_truncate_radius=train_config.get("timing_sample_truncate_radius"),
        timing_sample_truncate_center=train_config.get("timing_sample_truncate_center"),
        timing_sample_shrink_mode=train_config.get("timing_sample_shrink_mode", "none"),
        timing_sample_shrink_factor=train_config.get("timing_sample_shrink_factor", 1.0),
        timing_sample_shrink_radius=train_config.get("timing_sample_shrink_radius", 0.0),
        dlm_ioi_zero_sample_shrink_factor=train_config.get("dlm_ioi_zero_sample_shrink_factor"),
        dlm_ioi_nonzero_sample_shrink_factor=train_config.get("dlm_ioi_nonzero_sample_shrink_factor"),
        dlm_duration_sample_shrink_factor=train_config.get("dlm_duration_sample_shrink_factor"),
        dlm_velocity_sample_shrink_factor=train_config.get("dlm_velocity_sample_shrink_factor"),
        pn_mean_loss_lambda=train_config.get("pn_mean_loss_lambda", 0.0),
        pn_var_ioi_zero_lambda=train_config.get("pn_var_ioi_zero_lambda", 0.0),
        pn_var_ioi_nonzero_lambda=train_config.get("pn_var_ioi_nonzero_lambda", 0.0),
        pn_var_duration_lambda=train_config.get("pn_var_duration_lambda", 0.0),
        pn_var_velocity_lambda=train_config.get("pn_var_velocity_lambda", 0.0),
        pn_variance_shrinkage_tau=train_config.get("pn_variance_shrinkage_tau", 4.0),
        pn_variance_epsilon=train_config.get("pn_variance_epsilon", 1e-4),
        raw_timing_loss_lambda=train_config.get("raw_timing_loss_lambda", 0.5),
        legacy_dual_timing_head=train_config.get("legacy_dual_timing_head", False),
        raw_timing_head_type=train_config.get("raw_timing_head_type"),
        beta_eps=train_config.get("beta_eps", 1e-5),
        beta_kappa_min=train_config.get("beta_kappa_min", 1e-3),
        beta_alpha_min=train_config.get("beta_alpha_min", 1e-4),
        mixture_beta_parameterization=train_config.get("mixture_beta_parameterization", "alpha_beta"),
        mixture_beta_kappa_min=train_config.get("mixture_beta_kappa_min", 1e-3),
        predictive_variance_lambda=train_config.get("predictive_variance_lambda", 0.0),
        predictive_timing_radius=train_config.get("predictive_timing_radius", 0.05),
        predictive_velocity_radius=train_config.get("predictive_velocity_radius", 0.05),
        epr_inflated_features=train_config.get("epr_inflated_features"),
        epr_timing_bins=train_config.get("epr_timing_bins", 5000),
        epr_value_bins=train_config.get("epr_value_bins", 128),
        dinr_timing_bins=train_config.get("dinr_timing_bins", 512),
        dinr_zero_bin=train_config.get("dinr_zero_bin", 93),
        dinr_timing_step=train_config.get("dinr_timing_step", 2.0 / 93.0),
        dinr_output_timing_bins=train_config.get("dinr_output_timing_bins"),
        dinr_output_zero_bin=train_config.get("dinr_output_zero_bin"),
        dinr_output_timing_step=train_config.get("dinr_output_timing_step"),
        dinr_vocabulary_mode=train_config.get("dinr_vocabulary_mode", "unified"),
        dinr_absolute_max_ms=train_config.get("dinr_absolute_max_ms", 8000.0),
        dinr_deviation_min=train_config.get("dinr_deviation_min", -2.0),
        dinr_deviation_max=train_config.get("dinr_deviation_max", 2.0),
        dinr_zero_ioi_min=train_config.get("dinr_zero_ioi_min", 0.0),
        dinr_zero_ioi_max=train_config.get("dinr_zero_ioi_max", 5.0),
        dinr_sampling_temperature=train_config.get("dinr_sampling_temperature", 1.0),
        dinr_numerical_frequencies=train_config.get("dinr_numerical_frequencies", 16),
        epr_timing_target=epr_timing_target,
        timing_control_mode=timing_control_mode,
        timing_log_scale=train_config.get("timing_log_scale", 50.0),
        use_timing_scale_bit=use_timing_scale_bit,
        soft_ce_tau=train_config.get("soft_ce_tau"),
        timing_input_normalization=train_config.get("timing_input_normalization", "scaled_log_5000_s10"),
        musical_feature_mode=musical_feature_mode,
        musical_gate_init=train_config.get("musical_gate_init", 1.0),
        prior_token_keep_prob=train_config.get("prior_token_keep_prob", 1.0),
        prior_token_dropout_mode=train_config.get("prior_token_dropout_mode", "mask"),
        prior_attribute_keep_probs=train_config.get("prior_attribute_keep_probs"),
        prior_attribute_noise_std=train_config.get("prior_attribute_noise_std", 0.05),
        prior_property_dropout_prob=train_config.get("prior_property_dropout_prob"),
        prior_property_dropout_pattern=train_config.get("prior_property_dropout_pattern", "independent"),
        prior_property_dropout_replacement=train_config.get("prior_property_dropout_replacement", "pad"),
        prior_property_visible_prob=train_config.get("prior_property_visible_prob", 0.50),
        prior_property_all_dropout_prob=train_config.get("prior_property_all_dropout_prob", 0.25),
        stable_force_all_properties_visible=train_config.get("stable_force_all_properties_visible", False),
        tf_embedding_mask_keep_prob=train_config.get("tf_embedding_mask_keep_prob", 1.0),
        tf_embedding_mask_score=train_config.get("tf_embedding_mask_score", False),
        tf_embedding_mask_decoder=train_config.get("tf_embedding_mask_decoder", False),
        slot_decoder_mask_mode=train_config.get("slot_decoder_mask_mode", "property"),
        stable_contract_loss=train_config.get("stable_contract_loss", False),
        stable_contract_alpha=train_config.get("stable_contract_alpha", 1.0),
        stable_contract_lambda=train_config.get("stable_contract_lambda", 0.0),
        stable_contract_ioi_alpha=train_config.get("stable_contract_ioi_alpha"),
        stable_contract_duration_alpha=train_config.get("stable_contract_duration_alpha"),
        stable_contract_ioi_lambda=train_config.get("stable_contract_ioi_lambda"),
        stable_contract_duration_lambda=train_config.get("stable_contract_duration_lambda"),
        stable_contract_eps=train_config.get("stable_contract_eps", 1e-6),
        zero_ioi_transform=train_config.get("zero_ioi_transform"),
        zero_ioi_positive_support=train_config.get("zero_ioi_positive_support", False),
        zero_ioi_support_eps=train_config.get("zero_ioi_support_eps", 1e-6),
        zero_ioi_residual=train_config.get("zero_ioi_residual", False),
        zero_ioi_residual_targets=train_config.get("zero_ioi_residual_targets"),
        zero_score_ioi_embedding=train_config.get("zero_score_ioi_embedding", False),
        zero_timing_head_condition=train_config.get("zero_timing_head_condition", False),
        zero_ioi_dual_distribution_mode=train_config.get("zero_ioi_dual_distribution_mode", "none"),
        zero_ioi_dual_duration=train_config.get("zero_ioi_dual_duration", True),
        piano_pitch_min=train_config.get("piano_pitch_min", 21),
        pedal_representation=train_config.get("pedal_representation", "start_valley"),
        use_style_tokens=train_config.get("use_style_tokens", False),
        style_creator_vocab_size=train_config.get("style_creator_vocab_size", 1),
        style_source_vocab_size=train_config.get("style_source_vocab_size", 1),
        style_score_stat_dim=train_config.get("style_score_stat_dim", STYLE_STAT_DIM),
        style_perf_stat_dim=train_config.get("style_perf_stat_dim", STYLE_STAT_DIM),
        style_integration_mode=train_config.get("style_integration_mode", "prepend"),
        torch_dtype=dtype,
    )

    if resume_path:
        model = IntegratedPianoT5Gemma(model_config) if backbone_type in {"t5", "t5gemma"} else IntegratedPianoTransformer(model_config)
        state_dict = load_torch_state_dict(resume_path)
        state_dict = filter_resume_state_dict(model, state_dict, train_config)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded Integrated {backbone_type} weights from {resume_path}")
        print(f"Missing keys: {len(missing)}")
        print(f"Unexpected keys: {len(unexpected)}")
        return model

    if backbone_type in {"t5", "t5gemma"}:
        model = IntegratedPianoT5Gemma(model_config)
    elif backbone_type in {"bert", "gpt"}:
        model = IntegratedPianoTransformer(model_config)
    else:
        raise ValueError(f"Unsupported backbone_type: {backbone_type}")

    pretrained_model = train_config.get("pretrained_model")
    if pretrained_model and train_config.get("load_pianoformer_backbone", True):
        if backbone_type not in {"t5", "t5gemma"}:
            raise ValueError("load_pianoformer_backbone is only supported for t5 backbones")
        incompatible = model.load_pianoformer_backbone(pretrained_model, torch_dtype=dtype)
        print(f"Loaded PianistTransformer backbone from {pretrained_model}")
        print(f"Missing keys: {len(incompatible.missing_keys)}")
        print(f"Unexpected keys: {len(incompatible.unexpected_keys)}")
    return model


def enable_eval_best_checkpointing(train_config):
    eval_strategy = train_config.get("eval_strategy", train_config.get("evaluation_strategy", "no"))
    save_strategy = train_config.get("save_strategy", "steps")
    if eval_strategy == "no" or save_strategy == "no":
        return

    train_config.setdefault("load_best_model_at_end", True)
    train_config.setdefault("metric_for_best_model", "eval_loss")
    train_config.setdefault("greater_is_better", False)
    if train_config["load_best_model_at_end"]:
        train_config["save_total_limit"] = max(2, int(train_config.get("save_total_limit", 2) or 2))

    if train_config["load_best_model_at_end"] and eval_strategy == "steps" and save_strategy == "steps":
        eval_steps = train_config.get("eval_steps")
        if eval_steps:
            train_config["save_steps"] = eval_steps


def _visible_cuda_devices_for_child():
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw:
        return [item.strip() for item in raw.split(",") if item.strip()]
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    return [str(index) for index in range(count)]


def _as_bool(value, default=False):
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_auto_rollout_score_source_list(train_config):
    configured = train_config.get("auto_rollout_score_source_list") or train_config.get("cheap15_score_source_list")
    candidates = []
    if configured:
        candidates.append(Path(configured))
    candidates.append(ROOT_DIR / "results/current_inr_asap_simplified_20260706/cheap15_score_sources.txt")
    for path in candidates:
        if path.exists():
            return path
    return None


def _auto_rollout_root_dir(output_dir):
    output_dir = Path(output_dir)
    if output_dir.parent.name == "training":
        return output_dir.parent.parent
    return output_dir.parent


def _auto_rollout_device_label(device):
    text = str(device or "cpu").strip()
    if not text:
        return "cpu"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def run_auto_rollout_eval_after_train(train_config, output_dir):
    if not _as_bool(train_config.get("auto_rollout_eval_after_train", True), default=True):
        print(json.dumps({"event": "auto_rollout_eval_skip", "reason": "disabled"}), flush=True)
        return

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if int(torch.distributed.get_rank()) != 0:
            return

    checkpoint = output_dir / "checkpoint-best"
    if not checkpoint.exists():
        checkpoint = output_dir
    config_path = output_dir / "train_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"auto rollout eval requires {config_path}")

    devices = _visible_cuda_devices_for_child()
    workers = int(train_config.get("auto_rollout_workers_per_gpu", 8) or 8)
    eval_split = train_config.get("auto_rollout_split", train_config.get("eval_split", "test"))
    performance_dataset = train_config.get(
        "auto_rollout_performance_dataset",
        train_config.get("eval_performance_dataset", train_config.get("performance_dataset", "ASAP")),
    )
    score_source_list = _resolve_auto_rollout_score_source_list(train_config)

    base_cmd = [
        os.sys.executable,
        "src/evaluate/eval_inr_rollout_current.py",
        "--config",
        str(config_path),
        "--checkpoint",
        str(checkpoint),
        "--split",
        str(eval_split),
        "--performance-dataset",
        str(performance_dataset),
        "--batch-size-windows",
        str(int(train_config.get("auto_rollout_batch_size_windows", 8) or 8)),
        "--num-workers",
        str(workers),
        "--materialize-strategy",
        str(train_config.get("auto_rollout_materialize_strategy", "sample")),
        "--feedback-strategy",
        str(train_config.get("auto_rollout_feedback_strategy", "sample")),
    ]
    if score_source_list is not None:
        base_cmd.extend(["--score-source-list", str(score_source_list)])

    eval_root = _auto_rollout_root_dir(output_dir)
    fast_device = devices[0] if devices else ""
    ar_device = devices[1] if len(devices) > 1 else fast_device
    fast_suffix = str(
        train_config.get(
            "auto_rollout_fast_output_suffix",
            f"auto_g{_auto_rollout_device_label(fast_device)}w{workers}",
        )
    )
    ar_suffix = str(
        train_config.get(
            "auto_rollout_ar_output_suffix",
            f"auto_g{_auto_rollout_device_label(ar_device)}w{workers}",
        )
    )

    jobs = [
        {
            "name": "fast_kpass",
            "device": fast_device,
            "output_dir": eval_root / f"cheap15_fast_kpass_{fast_suffix}",
            "cmd_extra": ["--rollout-ks", str(train_config.get("auto_rollout_fast_ks", "0,1,2")), "--fast-kpass"],
            "log": eval_root / f"eval_fast_kpass_{fast_suffix}.log",
        },
        {
            "name": "full_ar",
            "device": ar_device,
            "output_dir": eval_root / f"cheap15_ar_sample_{ar_suffix}",
            "cmd_extra": ["--rollout-ks", "full"],
            "log": eval_root / f"eval_ar_sample_{ar_suffix}.log",
        },
    ]

    procs = []
    for job in jobs:
        cmd = [*base_cmd, "--output-dir", str(job["output_dir"]), *job["cmd_extra"]]
        env = os.environ.copy()
        if job["device"]:
            env["CUDA_VISIBLE_DEVICES"] = str(job["device"])
        log_path = Path(job["log"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w", encoding="utf-8")
        print(
            json.dumps(
                {
                    "event": "auto_rollout_eval_start",
                    "job": job["name"],
                    "device": job["device"],
                    "child_cuda_visible_devices": str(job["device"] or os.environ.get("CUDA_VISIBLE_DEVICES", "")),
                    "workers": workers,
                    "output_dir": str(job["output_dir"]),
                    "log": str(log_path),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        proc = subprocess.Popen(cmd, cwd=str(ROOT_DIR), env=env, stdout=log_file, stderr=subprocess.STDOUT)
        procs.append((job, proc, log_file))

        # With one visible GPU, avoid two eval pools fighting on the same card.
        if len(devices) <= 1:
            code = proc.wait()
            log_file.close()
            job["closed_log"] = True
            if code != 0:
                raise RuntimeError(f"auto rollout eval {job['name']} failed with exit code {code}; see {job['log']}")
            print(
                json.dumps(
                    {"event": "auto_rollout_eval_done", "job": job["name"], "output_dir": str(job["output_dir"])},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )

    failures = []
    for job, proc, log_file in procs:
        if len(devices) <= 1 and proc.poll() is not None:
            continue
        code = proc.wait()
        if not job.get("closed_log"):
            log_file.close()
        if code != 0:
            failures.append((job["name"], code, str(job["log"])))
        else:
            print(
                json.dumps(
                    {"event": "auto_rollout_eval_done", "job": job["name"], "output_dir": str(job["output_dir"])},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
    if failures:
        raise RuntimeError(f"auto rollout eval failures: {failures}")


def main():
    current_datetime = datetime.datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/inr_config_pianocore.json")
    parser.add_argument("--deepspeed", type=str, help="Path to DeepSpeed config")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--limit_works", type=int, default=None)
    parser.add_argument("--limit_performances_per_work", type=int, default=None)
    parser.add_argument("--limit_windows_per_work", type=int, default=None)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    if torch.cuda.is_available():
        if local_rank >= 0:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    with open(args.config, "r", encoding="utf-8") as file:
        train_config = json.load(file)
    seed = int(train_config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    outname = str(train_config.get("run_name") or ("inr_" + current_datetime.strftime("%Y-%m-%d-%H-%M-%S")))
    task_type = train_config.get("task_type", "epr").lower()
    input_feature_mode = infer_input_feature_mode(train_config)
    train_config["input_feature_mode"] = input_feature_mode
    timing_control_mode = resolve_timing_control_mode(
        timing_control_mode=train_config.get("timing_control_mode"),
        use_timing_scale_bit=train_config.get("use_timing_scale_bit", False),
    )
    if task_type in {"epr", "csr"} and timing_control_mode not in {"log_scaled", "floor_log", "dinr_floor_log", "raw_log"}:
        raise ValueError("Integrated INR requires timing_control_mode=log_scaled, floor_log, dinr_floor_log, or raw_log")
    if task_type == "epr" and str(train_config.get("epr_timing_target", "log_deviation")).lower() not in {
        "log_deviation",
        "log_dev",
        "raw_log_deviation",
        "raw_log_dev",
        "floor_log_deviation",
        "floor_log_dev",
        "pure_log_deviation",
        "pure_log_dev",
        "raw_deviation",
        "raw_dev",
        "raw_seconds_deviation",
        "raw_seconds_dev",
        "absolute",
        "absolute_log",
        "log_absolute",
        "raw_log_absolute",
        "absolute_raw_log",
        "floor_log_absolute",
    }:
        raise ValueError("EPR requires a supported deviation or absolute timing target")
    train_config["timing_control_mode"] = timing_control_mode
    train_config["use_timing_scale_bit"] = timing_control_mode == "piecewise_scale_bit"
    musical_feature_mode = str(
        train_config.get(
            "musical_feature_mode",
            "continuous" if task_type == "csr" else "categorical",
        )
    ).lower()
    train_config["musical_feature_mode"] = musical_feature_mode
    if task_type == "epr":
        train_config.setdefault("pedal_representation", "start_valley")
        epr_timing_target = str(train_config.get("epr_timing_target", "log_deviation")).lower()
        configured_output_dim = int(
            train_config.get(
                "output_continuous_dim",
                train_config.get(
                    "continuous_dim",
                    default_epr_output_dim(
                        epr_timing_target,
                        train_config.get("pedal_representation", "start_valley"),
                        legacy_dual_timing_head=False,
                    ),
                ),
            )
        )
        legacy_dual_timing_head = bool(
            train_config.get(
                "legacy_dual_timing_head",
                epr_timing_target in {"raw_log_deviation", "raw_log_dev"}
                and configured_output_dim >= 9,
            )
        )
        base_output_dim = default_epr_output_dim(
            epr_timing_target,
            train_config.get("pedal_representation", "start_valley"),
            legacy_dual_timing_head=legacy_dual_timing_head,
        )
        train_config["legacy_dual_timing_head"] = legacy_dual_timing_head
        train_config["continuous_dim"] = base_output_dim
        train_config["output_continuous_dim"] = base_output_dim
    if train_config.get("use_style_tokens", False):
        raise ValueError("use_style_tokens is disabled for the simplified EPR/CSR pipelines")
    if task_type in {"epr", "csr"} and input_feature_mode == "integrated":
        if task_type == "epr" and score_note_input_schema(train_config) == "score_musical":
            inferred_input_dim = score_musical_input_dim(
                timing_control_mode=timing_control_mode,
                use_timing_scale_bit=train_config.get("use_timing_scale_bit", False),
                musical_feature_mode=musical_feature_mode,
            )
        else:
            inferred_input_dim = integrated_epr_input_dim(
                timing_control_mode=timing_control_mode,
                use_timing_scale_bit=train_config.get("use_timing_scale_bit", False),
                musical_feature_mode=musical_feature_mode,
                pedal_control_dim=pedal_representation_dim(train_config.get("pedal_representation", "start_valley")),
            )
        train_config["input_continuous_dim"] = inferred_input_dim
        train_config["score_input_continuous_dim"] = inferred_input_dim
        train_config["decoder_input_continuous_dim"] = (
            decoder_perf_target_input_dim(
                train_config.get("output_continuous_dim", train_config.get("continuous_dim", 7))
            )
            if task_type == "epr" and decoder_note_input_schema(train_config) == "perf_target"
            else inferred_input_dim
        )
    else:
        train_config.setdefault(
            "input_continuous_dim",
            default_input_continuous_dim(
                task_type,
                input_feature_mode,
                score_feature_dim=train_config.get("score_feature_dim", 8),
                continuous_dim=train_config.get("continuous_dim", 7),
                musical_feature_mode=musical_feature_mode,
            ),
        )
    if task_type == "csr":
        train_config.setdefault("output_continuous_dim", integrated_csr_output_dim())

    if args.max_steps is not None:
        train_config["max_steps"] = args.max_steps
    if args.limit_works is not None:
        train_config["max_train_works"] = args.limit_works
        train_config["max_eval_works"] = min(args.limit_works, train_config.get("max_eval_works") or args.limit_works)
    if args.limit_performances_per_work is not None:
        train_config["max_performances_per_work"] = args.limit_performances_per_work
    if args.limit_windows_per_work is not None:
        train_config["max_windows_per_work"] = args.limit_windows_per_work

    enable_eval_best_checkpointing(train_config)

    train_config["output_dir"] = os.path.join(train_config["output_dir"], outname)
    train_config["run_name"] = outname
    train_config["logging_dir"] = os.path.join(train_config["logging_dir"], outname)

    output_dir = Path(train_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "train_config.json", "w", encoding="utf-8") as file:
        json.dump(train_config, file, indent=2, ensure_ascii=False)

    fixed_window_split_scheme = train_config.get("fixed_window_split_scheme")
    fixed_window_split_summary_path = train_config.get("fixed_window_split_summary_path")
    fixed_window_base_split = train_config.get("fixed_window_base_split", "train")
    train_dataset_split = "train"
    eval_dataset_split = train_config.get("eval_split", "test")

    if fixed_window_split_scheme:
        train_dataset_split = fixed_window_base_split
        eval_dataset_split = fixed_window_base_split
        train_manifest = build_work_manifest(
            metadata_path=train_config["metadata_path"],
            refined_dir=train_config["refined_dir"],
            split=fixed_window_base_split,
            block_notes=train_config["block_notes"],
            overlap_ratio=train_config["overlap_ratio"],
            min_notes=train_config["min_notes"],
            max_works=train_config.get("max_train_works"),
            skip_work_paths=train_config.get("skip_work_paths"),
            performance_dataset=train_config.get("train_performance_dataset"),
            exclude_performance_dataset=train_config.get("train_exclude_performance_dataset"),
            window_split_scheme=fixed_window_split_scheme,
            window_split_name=train_config.get("fixed_window_train_split_name", "train"),
            window_split_summary_path=fixed_window_split_summary_path,
            prepared_sidecar_tag=train_config.get("prepared_sidecar_tag"),
        )
    else:
        train_manifest = build_work_manifest(
            metadata_path=train_config["metadata_path"],
            refined_dir=train_config["refined_dir"],
            split="train",
            block_notes=train_config["block_notes"],
            overlap_ratio=train_config["overlap_ratio"],
            min_notes=train_config["min_notes"],
            max_works=train_config.get("max_train_works"),
            skip_work_paths=train_config.get("skip_work_paths"),
            performance_dataset=train_config.get("train_performance_dataset"),
            exclude_performance_dataset=train_config.get("train_exclude_performance_dataset"),
            prepared_sidecar_tag=train_config.get("prepared_sidecar_tag"),
        )
    if fixed_window_split_scheme:
        eval_manifest = build_work_manifest(
            metadata_path=train_config["metadata_path"],
            refined_dir=train_config["refined_dir"],
            split=fixed_window_base_split,
            block_notes=train_config["block_notes"],
            overlap_ratio=train_config["overlap_ratio"],
            min_notes=train_config["min_notes"],
            max_works=train_config.get("max_eval_works"),
            include_all_performance_dataset=train_config.get("eval_include_all_performance_dataset"),
            max_non_asap_performances_per_work=train_config.get("max_eval_non_asap_performances_per_work"),
            selection_seed=train_config.get("seed", 42),
            skip_work_paths=train_config.get("skip_work_paths"),
            performance_dataset=train_config.get("eval_performance_dataset"),
            exclude_performance_dataset=train_config.get("eval_exclude_performance_dataset"),
            window_split_scheme=fixed_window_split_scheme,
            window_split_name=train_config.get("fixed_window_eval_split_name", "valid"),
            window_split_summary_path=fixed_window_split_summary_path,
            prepared_sidecar_tag=train_config.get("prepared_sidecar_tag"),
        )
        train_config["eval_split"] = train_config.get("fixed_window_eval_split_name", "valid")
    else:
        eval_manifest = build_work_manifest(
            metadata_path=train_config["metadata_path"],
            refined_dir=train_config["refined_dir"],
            split=train_config.get("eval_split", "test"),
            block_notes=train_config["block_notes"],
            overlap_ratio=train_config["overlap_ratio"],
            min_notes=train_config["min_notes"],
            max_works=train_config.get("max_eval_works"),
            include_all_performance_dataset=train_config.get("eval_include_all_performance_dataset"),
            max_non_asap_performances_per_work=train_config.get("max_eval_non_asap_performances_per_work"),
            selection_seed=train_config.get("seed", 42),
            skip_work_paths=train_config.get("skip_work_paths"),
            performance_dataset=train_config.get("eval_performance_dataset"),
            exclude_performance_dataset=train_config.get("eval_exclude_performance_dataset"),
            prepared_sidecar_tag=train_config.get("prepared_sidecar_tag"),
        )

    max_train_epochs = float(train_config.get("max_train_epochs", 8.0))
    train_config["num_train_epochs"] = min(float(train_config.get("num_train_epochs", 1.0)), max_train_epochs)
    if "adapt_num_train_epochs" in train_config:
        train_config["adapt_num_train_epochs"] = min(float(train_config["adapt_num_train_epochs"]), max_train_epochs)

    estimated_train_examples = sum(item["estimated_examples"] for item in train_manifest)
    configure_eval_schedule(train_config, estimated_train_examples)
    print(f"Train works: {len(train_manifest)}")
    print(f"Eval works: {len(eval_manifest)}")
    print(f"Estimated train examples: {estimated_train_examples:,}")
    print(f"Estimated eval examples: {sum(item['estimated_examples'] for item in eval_manifest):,}")

    train_dataset = PianoCoReNodeSFTDataset(
        train_manifest,
        split=train_dataset_split,
        task_type=task_type,
        input_feature_mode=input_feature_mode,
        shuffle=True,
        seed=train_config["seed"],
        max_performances_per_work=train_config.get("max_performances_per_work"),
        max_windows_per_work=train_config.get("max_windows_per_work"),
        cache_size=train_config.get("node_cache_size", 16),
        timing_normalization=train_config.get("timing_input_normalization", "scaled_log_5000_s10"),
        max_time_ms=train_config.get("max_time_ms", 10000.0),
        epr_timing_bins=train_config.get("epr_timing_bins", 5000),
        epr_value_bins=train_config.get("epr_value_bins", 128),
        pedal_representation=train_config.get("pedal_representation", "start_valley"),
        musical_feature_mode=musical_feature_mode,
        score_note_schema=train_config.get("score_note_input_schema", "integrated"),
        epr_timing_target=train_config.get("epr_timing_target", "log_deviation"),
        disable_musical_features=train_config.get("disable_musical_features", False),
        use_timing_scale_bit=train_config.get("use_timing_scale_bit", False),
        timing_control_mode=train_config.get("timing_control_mode"),
        timing_log_scale=train_config.get("timing_log_scale", 50.0),
        precompute_items=train_config.get("precompute_dataset_items", False),
        use_prepared_sidecar=train_config.get("use_prepared_sidecar", True),
        prepared_sidecar_tag=train_config.get("prepared_sidecar_tag"),
        use_style_tokens=train_config.get("use_style_tokens", False),
        composer_vocab=train_config.get("style_composer_vocab"),
        source_vocab=train_config.get("style_source_vocab"),
        perf_style_stats_mode=train_config.get("perf_style_stats_mode", "prefix"),
        legacy_dual_timing_head=train_config.get("legacy_dual_timing_head", False),
        multi_perf_group_size=train_config.get("multi_perf_group_size", 1),
        multi_perf_min_group_size=train_config.get("multi_perf_min_group_size", 3),
    )
    eval_dataset = PianoCoReNodeSFTDataset(
        eval_manifest,
        split=eval_dataset_split,
        task_type=task_type,
        input_feature_mode=input_feature_mode,
        shuffle=False,
        seed=train_config["seed"],
        max_performances_per_work=train_config.get("max_eval_performances_per_work"),
        max_windows_per_work=train_config.get("max_eval_windows_per_work"),
        cache_size=train_config.get("node_cache_size", 16),
        timing_normalization=train_config.get("timing_input_normalization", "scaled_log_5000_s10"),
        max_time_ms=train_config.get("max_time_ms", 10000.0),
        epr_timing_bins=train_config.get("epr_timing_bins", 5000),
        epr_value_bins=train_config.get("epr_value_bins", 128),
        pedal_representation=train_config.get("pedal_representation", "start_valley"),
        musical_feature_mode=musical_feature_mode,
        score_note_schema=train_config.get("score_note_input_schema", "integrated"),
        epr_timing_target=train_config.get("epr_timing_target", "log_deviation"),
        disable_musical_features=train_config.get("disable_musical_features", False),
        use_timing_scale_bit=train_config.get("use_timing_scale_bit", False),
        timing_control_mode=train_config.get("timing_control_mode"),
        timing_log_scale=train_config.get("timing_log_scale", 50.0),
        precompute_items=train_config.get("precompute_eval_dataset_items", train_config.get("precompute_dataset_items", False)),
        use_prepared_sidecar=train_config.get("use_prepared_sidecar", True),
        prepared_sidecar_tag=train_config.get("prepared_sidecar_tag"),
        use_style_tokens=train_config.get("use_style_tokens", False),
        composer_vocab=train_config.get("style_composer_vocab"),
        source_vocab=train_config.get("style_source_vocab"),
        perf_style_stats_mode=train_config.get("perf_style_stats_mode", "prefix"),
        legacy_dual_timing_head=train_config.get("legacy_dual_timing_head", False),
    )

    model = create_model(train_config)
    apply_trainable_parameter_policy(model, train_config)
    model.to(device)
    print_model_parameters(model)

    training_args_dict = filter_valid_args(train_config, TrainingArguments)
    if str(train_config.get("note_embedding_mode", "")).lower() == "slot_attribute":
        training_args_dict["save_safetensors"] = False
    if args.deepspeed:
        training_args_dict["deepspeed"] = args.deepspeed
    if "accelerator_config" in train_config:
        training_args_dict["accelerator_config"] = train_config["accelerator_config"]
    # Integrated INR uses custom continuous labels instead of the standard
    # `labels` field. Tell Trainer explicitly so eval computes `eval_loss`.
    label_names = list(training_args_dict.get("label_names") or ["labels_continuous"])
    if bool(train_config.get("tail_mask_enabled", False)) and "label_valid_mask" not in label_names:
        label_names.append("label_valid_mask")
    training_args_dict["label_names"] = label_names
    if int(train_config.get("dataloader_num_workers", 0) or 0) > 0:
        # Keep workers alive after dataloader warmup; this reduces CPU/input
        # stalls for the bs32/acc1 DDP recipe used by the pedal2 experiments.
        training_args_dict.setdefault("dataloader_persistent_workers", True)
    dagger_prefix_training = bool(train_config.get("dagger_prefix_training", False))
    if dagger_prefix_training:
        training_args_dict["dataloader_persistent_workers"] = False
        train_config["dataloader_persistent_workers"] = False
    if int(training_args_dict.get("dataloader_num_workers", 0) or 0) <= 0:
        training_args_dict["dataloader_prefetch_factor"] = None
    training_args_dict.setdefault("dataloader_pin_memory", torch.cuda.is_available())
    if torch.cuda.device_count() > 1:
        training_args_dict.setdefault("ddp_find_unused_parameters", False)
        training_args_dict.setdefault("ddp_broadcast_buffers", False)
    training_args = TrainingArguments(**training_args_dict)

    trainer = NodeSFTTrainer(
        model=model,
        args=training_args,
        data_collator=NodeSFTDataCollator(
            pitch_pad_id=train_config["pitch_pad_id"],
            task_type=task_type,
            use_style_tokens=train_config.get("use_style_tokens", False),
            dagger_prefix_training=dagger_prefix_training,
            dagger_apply_prob=train_config.get("dagger_apply_prob", 1.0),
            dagger_replacement_weights=train_config.get("dagger_replacement_weights"),
            dagger_seed=train_config.get("dagger_seed", train_config.get("seed", 42)),
            stable_dynamics_training=train_config.get("stable_dynamics_training", False),
            stable_apply_prob=train_config.get("stable_apply_prob", 0.30),
            stable_channels=train_config.get("stable_channels", ["ioi", "duration"]),
            stable_noise_modes=train_config.get("stable_noise_modes"),
            stable_seed=train_config.get("stable_seed", train_config.get("seed", 42)),
            stable_zero_ioi_nonnegative_feedback=train_config.get(
                "stable_zero_ioi_nonnegative_feedback",
                False,
            ),
            epr_timing_target=train_config.get("epr_timing_target", "log_deviation"),
            tail_mask_enabled=train_config.get("tail_mask_enabled", False),
            tail_mask_tf_clamp=train_config.get("tail_mask_tf_clamp", True),
            tail_mask_ioi_min=train_config.get("tail_mask_ioi_min", -1.5),
            tail_mask_ioi_max=train_config.get("tail_mask_ioi_max", 1.5),
            tail_mask_duration_min=train_config.get("tail_mask_duration_min", -2.0),
            tail_mask_duration_max=train_config.get("tail_mask_duration_max", 2.0),
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.dagger_prefix_training = dagger_prefix_training
    trainer.dagger_cache_type = train_config.get("dagger_cache_type", "tf_pred")
    trainer.dagger_cache_scope = train_config.get("dagger_cache_scope", "random")
    trainer.dagger_cache_fraction = train_config.get("dagger_cache_fraction", 0.5)
    trainer.dagger_cache_max_items = train_config.get("dagger_cache_max_items")
    trainer.dagger_cache_max_interval_fraction = train_config.get("dagger_cache_max_interval_fraction")
    trainer.dagger_cache_seed = train_config.get("dagger_cache_seed", train_config.get("seed", 42))
    trainer.dagger_global_batch_size = train_config.get("global_batch_size")
    trainer.dagger_window_curriculum = train_config.get("dagger_window_curriculum", "none")
    trainer.dagger_window_curriculum_start = train_config.get("dagger_window_curriculum_start", 0.0)
    trainer.dagger_window_curriculum_end = train_config.get("dagger_window_curriculum_end", 1.0)
    trainer.dagger_window_curriculum_steps = train_config.get("dagger_window_curriculum_steps", 0)
    trainer.dagger_cache_schedule = train_config.get("dagger_cache_schedule", "window_curriculum")
    trainer.dagger_schedule_total_steps = train_config.get("dagger_schedule_total_steps", 0)
    trainer.dagger_cache_batch_size = train_config.get(
        "dagger_cache_batch_size",
        train_config.get("per_device_eval_batch_size", train_config.get("per_device_train_batch_size", 1)),
    )
    trainer.dagger_cache_num_workers = train_config.get("dagger_cache_num_workers", 0)
    trainer.dagger_materialize_strategy = train_config.get("dagger_materialize_strategy", "sample")
    trainer.dagger_refresh_on_eval = train_config.get("dagger_refresh_on_eval", True)
    trainer.rollout_eval_enabled = bool(train_config.get("rollout_eval_enabled", False))
    trainer.rollout_eval_k = int(train_config.get("rollout_eval_k", 1) or 1)
    trainer.rollout_eval_weight = float(train_config.get("rollout_eval_weight", 1.0))
    trainer.rollout_eval_materialize_strategy = train_config.get("rollout_eval_materialize_strategy", "sample")
    trainer.rollout_eval_feedback_strategy = train_config.get("rollout_eval_feedback_strategy", "sample")
    trainer.loss_component_logging_steps = train_config.get(
        "loss_component_logging_steps",
        train_config.get("logging_steps", 0),
    )
    if "eval_dataloader_num_workers" not in train_config:
        train_config["eval_dataloader_num_workers"] = train_config.get("dataloader_num_workers", 0)
    if "eval_dataloader_persistent_workers" not in train_config:
        train_config["eval_dataloader_persistent_workers"] = False
    elif train_config.get("eval_dataloader_persistent_workers", False) and not train_config.get(
        "allow_eval_persistent_workers",
        False,
    ):
        print(
            "Forcing eval_dataloader_persistent_workers=False to avoid accumulating eval workers. "
            "Set allow_eval_persistent_workers=true to override.",
            flush=True,
        )
        train_config["eval_dataloader_persistent_workers"] = False
    if "eval_dataloader_prefetch_factor" not in train_config:
        train_config["eval_dataloader_prefetch_factor"] = train_config.get("dataloader_prefetch_factor", 2)
    if "eval_dataloader_pin_memory" not in train_config:
        train_config["eval_dataloader_pin_memory"] = training_args.dataloader_pin_memory
    trainer.eval_dataloader_num_workers = int(train_config.get("eval_dataloader_num_workers", 0) or 0)
    trainer.eval_dataloader_persistent_workers = bool(train_config.get("eval_dataloader_persistent_workers", False))
    trainer.eval_dataloader_prefetch_factor = train_config.get("eval_dataloader_prefetch_factor")
    trainer.eval_dataloader_pin_memory = bool(
        train_config.get("eval_dataloader_pin_memory", training_args.dataloader_pin_memory)
    )
    early_stopping_patience = train_config.get("early_stopping_patience")
    if early_stopping_patience is not None and int(early_stopping_patience) > 0:
        trainer.add_callback(
            EarlyStoppingCallback(
                early_stopping_patience=int(early_stopping_patience),
                early_stopping_threshold=float(train_config.get("early_stopping_threshold", 0.0)),
            )
        )

    resume_path = train_config.get("resume_path")
    resume_trainer_state = bool(train_config.get("resume_trainer_state", True))
    if dagger_prefix_training and bool(train_config.get("dagger_refresh_at_train_start", True)):
        trainer.refresh_dagger_prefix_cache(reason="train_start")
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
    trainer.train(resume_from_checkpoint=resume_path if resume_path and resume_trainer_state else None)
    trainer.save_model()
    dist_ready = torch.distributed.is_available() and torch.distributed.is_initialized()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
        rank = int(torch.distributed.get_rank())
    else:
        rank = 0
    if dist_ready:
        try:
            torch.distributed.destroy_process_group()
        except Exception as exc:  # noqa: BLE001
            if rank == 0:
                print(f"Warning: failed to destroy process group before auto eval: {exc}", flush=True)
    if rank != 0:
        return
    del trainer
    del model
    release_cuda_cache()
    run_auto_rollout_eval_after_train(train_config, output_dir)


if __name__ == "__main__":
    main()
