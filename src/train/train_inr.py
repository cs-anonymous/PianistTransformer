import argparse
import bisect
import datetime
import json
import math
import os
import random
import shutil
import re
from collections import OrderedDict
from pathlib import Path

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data.sampler import SequentialSampler
from transformers import Trainer, TrainingArguments

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT_DIR))

from src.model.integrated_pianoformer import (
    IntegratedPianoT5Gemma,
    IntegratedPianoT5GemmaConfig,
    IntegratedPianoTransformer,
    _compute_integrated_loss_components,
)
from src.utils.func import filter_valid_args
from src.utils.inr_midi import raw_rows_to_epr_bins, raw_rows_to_model_continuous


os.environ["WANDB_PROJECT"] = "pianist-transformer"


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


def score_json_path(refined_dir, score_rel_path):
    score_path = Path(refined_dir) / score_rel_path
    candidates = [
        score_path.with_suffix(".json"),
        score_path.parent / f"{score_path.stem}.node_a.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def make_windows(total_notes, block_notes, overlap_ratio, min_notes):
    total_notes = int(total_notes)
    if total_notes < min_notes:
        return []
    if total_notes <= block_notes:
        return [(0, total_notes)]

    stride = max(1, int(block_notes * (1.0 - overlap_ratio)))
    windows = []
    start = 0
    while start + block_notes <= total_notes:
        windows.append((start, start + block_notes))
        start += stride
    if windows[-1][1] != total_notes and total_notes - start >= min_notes:
        windows.append((total_notes - block_notes, total_notes))

    deduped = []
    seen = set()
    for window in windows:
        if window not in seen:
            deduped.append(window)
            seen.add(window)
    return deduped


def default_input_continuous_dim(task_type, input_feature_mode, score_feature_dim=8, continuous_dim=7):
    if input_feature_mode == "integrated":
        if task_type == "epr":
            return integrated_epr_input_dim()
        if task_type == "csr":
            return integrated_epr_input_dim()
    return continuous_dim


def infer_input_feature_mode(config):
    mode = config.get("input_feature_mode")
    if mode is not None:
        return str(mode).lower()
    return "integrated"


def resolve_timing_control_mode(timing_control_mode=None, use_timing_scale_bit=True):
    if timing_control_mode is None:
        return "piecewise_scale_bit" if bool(use_timing_scale_bit) else "piecewise_single"
    mode = str(timing_control_mode).lower()
    valid_modes = {
        "piecewise_scale_bit",
        "piecewise_single",
        "dual_log_linear",
        "dual_clip_linear",
        "log_scaled",
    }
    if mode not in valid_modes:
        raise ValueError(f"Unsupported timing_control_mode={timing_control_mode}")
    return mode


def timing_control_feature_dim(timing_control_mode=None, use_timing_scale_bit=True):
    mode = resolve_timing_control_mode(
        timing_control_mode=timing_control_mode,
        use_timing_scale_bit=use_timing_scale_bit,
    )
    return 3 if mode in {"piecewise_single", "log_scaled"} else 5


def integrated_epr_input_dim(timing_control_mode=None, use_timing_scale_bit=True):
    control_dim = timing_control_feature_dim(
        timing_control_mode=timing_control_mode,
        use_timing_scale_bit=use_timing_scale_bit,
    )
    score_control_dim = control_dim
    performance_control_dim = control_dim + 2
    musical_dim = 12
    mask_dim = 3
    return score_control_dim + performance_control_dim + musical_dim + mask_dim


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


def _compose_label_raw_rows(perf, pedal_representation="continuous_4"):
    representation = str(pedal_representation or "continuous_4").lower()
    shared_rows = perf.get("label_shared_raw")
    if shared_rows is None:
        if "label_raw" in perf:
            shared_rows = [row[:3] for row in perf["label_raw"]]
        else:
            return None

    if representation == "start_ctrl":
        pedal_rows = _raw_value_rows(perf, "label_pedal2_raw", "pedal2_raw")
        if pedal_rows is None:
            if "label_raw" in perf:
                pedal_rows = [[row[3], row[5]] for row in perf["label_raw"]]
            else:
                return None
        if len(shared_rows) != len(pedal_rows):
            raise ValueError(f"label_shared_raw/label_pedal2_raw length mismatch: {len(shared_rows)} vs {len(pedal_rows)}")
        return [
            list(shared[:3]) + [pedal[0], pedal[1], pedal[1], pedal[0]]
            for shared, pedal in zip(shared_rows, pedal_rows)
        ]

    pedal_rows = _raw_value_rows(perf, "label_pedal4_raw", "pedal4_raw")
    if pedal_rows is None:
        if "label_raw" in perf:
            pedal_rows = [row[3:7] for row in perf["label_raw"]]
        else:
            return None
    if len(shared_rows) != len(pedal_rows):
        raise ValueError(f"label_shared_raw/label_pedal4_raw length mismatch: {len(shared_rows)} vs {len(pedal_rows)}")
    return [
        list(shared[:3]) + list(pedal[:4])
        for shared, pedal in zip(shared_rows, pedal_rows)
    ]


def performance_label_rows_for_representation(
    perf,
    pedal_representation="continuous_4",
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
    if str(pedal_representation or "continuous_4").lower() != "continuous_4":
        raise KeyError(f"Missing raw labels for pedal_representation={pedal_representation}")
    return perf["label_continuous"]


def performance_label_bins_for_representation(
    perf,
    pedal_representation="continuous_4",
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
        pedal_representation="continuous_4",
        timing_normalization=timing_normalization,
        max_time_ms=max_time_ms,
    )


def performance_label_bins(perf, timing_bins=5000, value_bins=128):
    return performance_label_bins_for_representation(
        perf,
        pedal_representation="continuous_4",
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


def denormalize_log_timing_value(time_norm, scale=50.0, max_time_ms=5000.0):
    clipped = min(max(float(time_norm), 0.0), 1.0)
    scale = max(float(scale), 1e-12)
    return scale * math.expm1(clipped * math.log1p(float(max_time_ms) / scale))


def normalize_log_timing_dev(score_time_ms, perf_time_ms, scale=50.0, max_time_ms=5000.0):
    score_norm = normalize_log_timing_value(score_time_ms, scale=scale, max_time_ms=max_time_ms)
    perf_norm = normalize_log_timing_value(perf_time_ms, scale=scale, max_time_ms=max_time_ms)
    return min(max(perf_norm - score_norm + 0.5, 0.0), 1.0)


def _uses_log_deviation_target(epr_timing_target):
    return str(epr_timing_target or "").lower() in {"log_deviation", "log_dev", "log_deviation_ratio", "log_dev_ratio"}


def normalize_ioi_dev(score_ioi_ms, perf_ioi_ms, epr_timing_target="deviation", log_scale=50.0):
    if _uses_log_deviation_target(epr_timing_target):
        return normalize_log_timing_dev(score_ioi_ms, perf_ioi_ms, scale=log_scale, max_time_ms=5000.0)
    dev_ms = float(perf_ioi_ms) - float(score_ioi_ms)
    return min(max((dev_ms + 500.0) / 1000.0, 0.0), 1.0)


def normalize_duration_dev(score_duration_ms, perf_duration_ms, epr_timing_target="deviation", log_scale=50.0):
    if _uses_log_deviation_target(epr_timing_target):
        return normalize_log_timing_dev(score_duration_ms, perf_duration_ms, scale=log_scale, max_time_ms=5000.0)
    dev_ms = float(perf_duration_ms) - float(score_duration_ms)
    return min(max((dev_ms + 500.0) / 1000.0, 0.0), 1.0)


def performance_dev_velocity_pedal2_rows(perf, score_shared_raw, epr_timing_target="deviation", log_scale=50.0):
    shared_rows = perf.get("label_shared_raw")
    pedal_rows = _raw_value_rows(perf, "label_pedal2_raw", "pedal2_raw")
    if shared_rows is None or pedal_rows is None:
        if "label_raw" not in perf:
            return None
        shared_rows = [row[:3] for row in perf["label_raw"]]
        pedal_rows = [[row[3], row[5]] for row in perf["label_raw"]]

    if len(score_shared_raw) != len(shared_rows):
        raise ValueError(
            f"score_raw/label_shared_raw length mismatch: {len(score_shared_raw)} vs {len(shared_rows)}"
        )
    if len(shared_rows) != len(pedal_rows):
        raise ValueError(
            f"label_shared_raw/label_pedal2_raw length mismatch: {len(shared_rows)} vs {len(pedal_rows)}"
        )

    rows = []
    for score_row, perf_row, pedal_row in zip(score_shared_raw, shared_rows, pedal_rows):
        rows.append(
            [
                normalize_ioi_dev(
                    score_row[0],
                    perf_row[0],
                    epr_timing_target=epr_timing_target,
                    log_scale=log_scale,
                ),
                normalize_duration_dev(
                    score_row[1],
                    perf_row[1],
                    epr_timing_target=epr_timing_target,
                    log_scale=log_scale,
                ),
                min(max(float(perf_row[2]), 0.0), 127.0) / 127.0,
                min(max(float(pedal_row[0]), 0.0), 127.0) / 127.0,
                min(max(float(pedal_row[1]), 0.0), 127.0) / 127.0,
            ]
        )
    return rows


def normalize_piecewise_time_value(time_ms):
    value = min(max(float(time_ms), 0.0), 5000.0)
    if value <= 500.0:
        return value / 500.0
    return value / 5000.0


def encode_timing_control_features(time_ms, timing_control_mode=None, use_timing_scale_bit=True, log_scale=50.0):
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
    if mode == "dual_clip_linear":
        return [
            min(value / 500.0, 1.0),
            value / 5000.0,
        ]
    raise ValueError(f"Unsupported timing_control_mode={mode}")


def encode_shared_control_row(raw_shared_row, use_timing_scale_bit=True, timing_control_mode=None, log_scale=50.0):
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


def build_score_musical_rows(score):
    score_feature = score.get("score_feature", [])
    has_score_feature = score.get("has_score_feature", [0] * len(score.get("pitch", [])))
    score_raw = score.get("score_raw", [])

    rows = []
    measure_start = 0.0
    current_measure_length = 4.0
    prev_q = None
    prev_ms_per_quarter = 500.0
    seen_any_measure = False

    for idx, has_feature in enumerate(has_score_feature):
        if not bool(has_feature):
            rows.append([0.0] * 12)
            continue

        feature = score_feature[idx]
        mo = float(feature[0]) if len(feature) > 0 else 0.0
        md = float(feature[1]) if len(feature) > 1 else 0.0
        ml = float(feature[2]) if len(feature) > 2 else current_measure_length
        first = 1.0 if len(feature) > 3 and float(feature[3]) >= 0.5 else 0.0
        hand = 1.0 if len(feature) > 4 and float(feature[4]) >= 0.5 else 0.0
        trill = 1.0 if len(feature) > 5 and float(feature[5]) >= 0.5 else 0.0
        grace = 1.0 if len(feature) > 6 and float(feature[6]) >= 0.5 else 0.0
        stacc = 1.0 if len(feature) > 7 and float(feature[7]) >= 0.5 else 0.0
        stem_code = int(round(float(feature[8]))) if len(feature) > 8 else 0

        if not seen_any_measure:
            seen_any_measure = True
            if ml > 0.0:
                current_measure_length = ml
        elif first >= 0.5:
            measure_start += max(current_measure_length, 0.0)
            if ml > 0.0:
                current_measure_length = ml

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

        rows.append(
            [
                min(max(mo / 6.0, 0.0), 1.0),
                min(max(mioi / 6.0, 0.0), 1.0),
                min(max(md / 6.0, 0.0), 1.0),
                min(max(ml / 6.0, 0.0), 1.0),
                # Local musical tempo in BPM; log timing is only used for timing controls/targets.
                tempo_norm,
                first,
                grace,
                hand,
                trill,
                stacc,
                1.0 if stem_code == 1 else 0.0,
                1.0 if stem_code == 2 else 0.0,
            ]
        )
    return rows


def build_epr_score_input_rows(score, use_timing_scale_bit=True, timing_control_mode=None, log_scale=50.0):
    score_raw = score["score_raw"]
    has_score_feature = score.get("has_score_feature", [0] * len(score["pitch"]))
    musical_rows = build_score_musical_rows(score)
    control_dim = timing_control_feature_dim(
        timing_control_mode=timing_control_mode,
        use_timing_scale_bit=use_timing_scale_bit,
    )
    rows = []
    for raw_shared, musical, has_feature in zip(score_raw, musical_rows, has_score_feature):
        score_control = encode_shared_control_row(
            raw_shared[:3],
            use_timing_scale_bit=use_timing_scale_bit,
            timing_control_mode=timing_control_mode,
            log_scale=log_scale,
        )
        perf_control = [0.0] * (control_dim + 2)
        m_musical = 1.0 if bool(has_feature) else 0.0
        masks = [1.0, 0.0, m_musical]
        rows.append(
            score_control
            + perf_control
            + [value * m_musical for value in musical]
            + masks
        )
    return rows


def build_csr_performance_input_rows(perf, use_timing_scale_bit=True, timing_control_mode=None, log_scale=50.0):
    shared_rows = perf.get("label_shared_raw")
    pedal_rows = _raw_value_rows(perf, "label_pedal2_raw", "pedal2_raw")
    if shared_rows is None or pedal_rows is None:
        if "label_raw" not in perf:
            return None
        shared_rows = [row[:3] for row in perf["label_raw"]]
        pedal_rows = [[row[3], row[5]] for row in perf["label_raw"]]
    if len(shared_rows) != len(pedal_rows):
        raise ValueError(
            f"label_shared_raw/label_pedal2_raw length mismatch: {len(shared_rows)} vs {len(pedal_rows)}"
        )

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
                min(max(float(pedal[0]), 0.0), 127.0) / 127.0,
                min(max(float(pedal[1]), 0.0), 127.0) / 127.0,
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


def build_work_manifest(
    metadata_path,
    refined_dir,
    split,
    block_notes,
    overlap_ratio,
    min_notes,
    max_works=None,
    include_all_performance_dataset=None,
    max_non_asap_performances_per_work=None,
    selection_seed=42,
    skip_work_paths=None,
    performance_dataset=None,
    exclude_performance_dataset=None,
):
    columns = [
        "tier_a",
        "split",
        "refined_score_midi_path",
        "refined_performance_midi_path",
        "refined_alignment_path",
        "refined_score_note_count",
        "performance_dataset",
    ]
    df = pd.read_csv(metadata_path, usecols=columns)
    df = df[df["tier_a"].fillna(False).astype(bool)]
    df = df[df["split"] == split]
    df = df[df["refined_score_midi_path"].notna()]
    df = df[df["refined_performance_midi_path"].notna()]
    df = df[df["refined_alignment_path"].notna()]
    dataset = df["performance_dataset"].fillna("").astype(str)
    if performance_dataset is not None:
        df = df[dataset == str(performance_dataset)]
        dataset = df["performance_dataset"].fillna("").astype(str)
    if exclude_performance_dataset is not None:
        df = df[dataset != str(exclude_performance_dataset)]
    df = df.sort_values(["refined_score_midi_path", "refined_performance_midi_path"], kind="stable")

    manifest = []
    skip_work_paths = set(skip_work_paths or [])
    for score_rel_path, group in df.groupby("refined_score_midi_path", sort=True):
        selected_group = group
        if (
            include_all_performance_dataset is not None
            and max_non_asap_performances_per_work is not None
        ):
            dataset = group["performance_dataset"].fillna("").astype(str)
            always_mask = dataset == str(include_all_performance_dataset)
            always = group[always_mask]
            other = group[~always_mask]
            if len(other) > max_non_asap_performances_per_work:
                rng = random.Random(f"{selection_seed}:{score_rel_path}")
                sampled_indices = rng.sample(list(other.index), max_non_asap_performances_per_work)
                other = other.loc[sampled_indices]
            selected_group = pd.concat([always, other], axis=0).sort_values(
                ["refined_performance_midi_path"],
                kind="stable",
            )

        path = score_json_path(refined_dir, score_rel_path)
        if not path.exists():
            continue
        if str(path) in skip_work_paths or score_rel_path in skip_work_paths:
            print(f"Skipping configured work JSON: {path}", flush=True)
            continue
        note_count = int(group["refined_score_note_count"].iloc[0])
        windows = make_windows(note_count, block_notes, overlap_ratio, min_notes)
        if not windows:
            continue
        selected_sources = selected_group["refined_performance_midi_path"].tolist()
        manifest.append(
            {
                "path": str(path),
                "score_source": score_rel_path,
                "note_count": note_count,
                "windows": windows,
                "selected_performance_sources": selected_sources,
                "estimated_performances": int(len(selected_sources)),
                "estimated_examples": int(len(windows) * len(selected_sources)),
            }
        )
    if max_works is not None:
        manifest = manifest[:max_works]
    return manifest


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
        pedal_representation="continuous_4",
        epr_timing_target="absolute",
        use_timing_scale_bit=True,
        timing_control_mode=None,
        timing_log_scale=50.0,
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
        self.epr_timing_target = str(epr_timing_target or "absolute").lower()
        self.use_timing_scale_bit = bool(use_timing_scale_bit)
        self.timing_control_mode = resolve_timing_control_mode(
            timing_control_mode=timing_control_mode,
            use_timing_scale_bit=use_timing_scale_bit,
        )
        self.timing_log_scale = float(timing_log_scale)
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
            item["effective_examples"] = performance_count * len(windows)
            self.items.append(item)
            total += item["effective_examples"]
            self.cumulative_sizes.append(total)

        self.total_examples = total
        self.cache_size = cache_size
        self._cache = OrderedDict()
        self._prepared_cache = OrderedDict()

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

    def _prepare_work(self, path, work):
        if path in self._prepared_cache:
            self._prepared_cache.move_to_end(path)
            return self._prepared_cache[path]

        score = work["score"]
        performances = [
            perf for perf in work["performances"]
            if perf.get("split", self.split) == self.split
        ]
        by_source = {perf.get("performance_source"): perf for perf in performances}

        # Cache score inputs and raw-label conversions per dataloader worker.
        # The new raw schema would otherwise rebuild the same score/performance
        # tensors for every overlapping window.
        prepared = {
            "score": score,
            "performances": performances,
            "performances_by_source": by_source,
            "label_cache": {},
        }

        task_type = self.task_type.lower()
        if task_type == "epr":
            prepared["score_input"] = build_epr_score_input_rows(
                score,
                use_timing_scale_bit=self.use_timing_scale_bit,
                timing_control_mode=self.timing_control_mode,
                log_scale=self.timing_log_scale,
            )
        elif task_type == "csr":
            prepared["score_musical"] = build_score_musical_rows(score)
            prepared["has_score_feature"] = score["has_score_feature"]

        self._prepared_cache[path] = prepared
        self._prepared_cache.move_to_end(path)
        while len(self._prepared_cache) > self.cache_size:
            evicted_path, _ = self._prepared_cache.popitem(last=False)
            self._cache.pop(evicted_path, None)
        return prepared

    def _selected_performances(self, prepared, item):
        selected_sources = item.get("selected_performance_sources")
        if selected_sources is None:
            return prepared["performances"]
        by_source = prepared["performances_by_source"]
        return [by_source[source] for source in selected_sources if source in by_source]

    def _performance_cache_key(self, perf):
        return (
            perf.get("performance_source")
            or perf.get("performance_id")
            or id(perf)
        )

    def _performance_labels(self, prepared, perf):
        label_cache = prepared["label_cache"]
        cache_key = self._performance_cache_key(perf)
        if cache_key in label_cache:
            return label_cache[cache_key]

        if self.task_type.lower() == "epr" and self.epr_timing_target in {
            "deviation",
            "dev",
            "log_deviation",
            "log_dev",
        }:
            labels = performance_dev_velocity_pedal2_rows(
                perf,
                prepared["score"]["score_raw"],
                epr_timing_target=self.epr_timing_target,
                log_scale=self.timing_log_scale,
            )
            if labels is None:
                raise KeyError("Missing score_raw/label_shared_raw/label_pedal2_raw for deviation EPR targets")
            label_cache[cache_key] = (labels, None)
            return labels, None
        if self.task_type.lower() == "epr" and self.epr_timing_target in {"deviation_ratio", "dev_ratio"}:
            labels = performance_dev_velocity_pedal2_rows(
                perf,
                prepared["score"]["score_raw"],
                epr_timing_target=self.epr_timing_target,
                log_scale=self.timing_log_scale,
            )
            if labels is None:
                raise KeyError("Missing score_raw/label_shared_raw/label_pedal2_raw for deviation EPR targets")
            label_cache[cache_key] = (labels, None)
            return labels, None
        if self.task_type.lower() == "csr":
            labels = build_csr_performance_input_rows(
                perf,
                use_timing_scale_bit=self.use_timing_scale_bit,
                timing_control_mode=self.timing_control_mode,
                log_scale=self.timing_log_scale,
            )
            if labels is None:
                raise KeyError("Missing label_shared_raw/label_pedal2_raw for CSR inputs")
            label_cache[cache_key] = (labels, None)
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
        label_cache[cache_key] = (labels, label_bins)
        return labels, label_bins

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

        work = self._load_work(item["path"])
        prepared = self._prepare_work(item["path"], work)
        score = prepared["score"]
        performances = self._selected_performances(prepared, item)
        if not performances:
            raise IndexError(f"No performances for split={self.split} in {item['path']}")

        # A tiny number of PianoCoRe-A rows were skipped for pitch mismatch. If
        # metadata counted one of those rows, wrap to a valid performance instead
        # of making DistributedSampler lengths uneven.
        perf = performances[int(perf_slot) % len(performances)]
        labels, label_bins = self._performance_labels(prepared, perf)
        interpolated = perf["interpolated"]
        task_type = self.task_type.lower()
        if task_type == "epr":
            continuous = prepared["score_input"][start:end]
            labels_continuous = labels[start:end]
            labels_epr_bins = label_bins[start:end] if label_bins is not None else None
            label_mask = None
        elif task_type == "csr":
            continuous = labels[start:end]
            labels_continuous = prepared["score_musical"][start:end]
            labels_epr_bins = None
            label_mask = prepared["has_score_feature"][start:end]
        else:
            raise ValueError(f"Unsupported task_type: {self.task_type}")

        sample = {
            "pitch_ids": score["pitch"][start:end],
            "continuous": continuous,
            "labels_continuous": labels_continuous,
            "score_shared_raw": [row[:3] for row in score["score_raw"][start:end]],
            "interpolated": interpolated[start:end],
            "performance_dataset": perf.get("performance_dataset", "unknown"),
            "performance_id": perf.get("performance_id", "unknown"),
        }
        if labels_epr_bins is not None:
            sample["labels_epr_bins"] = labels_epr_bins
        if label_mask is not None:
            sample["label_mask"] = label_mask
        return sample


class NodeSFTTrainer(Trainer):
    def _model_config(self, model):
        return model.module.config if hasattr(model, "module") else model.config

    def _record_loss_components(self, model, outputs, inputs):
        if not hasattr(outputs, "logits"):
            return
        if "labels_continuous" not in inputs or "attention_mask" not in inputs:
            return
        loss_mask = inputs.get("label_mask", inputs["attention_mask"]).detach()
        components = _compute_integrated_loss_components(
            self._model_config(model),
            outputs.logits.detach(),
            inputs["labels_continuous"].detach(),
            loss_mask,
            labels_epr_bins=inputs.get("labels_epr_bins"),
        )
        if not getattr(self, "_loss_component_sums", None):
            self._loss_component_sums = {name: 0.0 for name in components}
            self._loss_component_count = 0
        for name, value in components.items():
            self._loss_component_sums[name] += float(value.detach().float().cpu())
        self._loss_component_count += 1

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss
        self._record_loss_components(model, outputs, inputs)
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        count = getattr(self, "_loss_component_count", 0)
        if count and "loss" in logs:
            for name, total in self._loss_component_sums.items():
                logs[f"loss_{name}"] = total / count
            self._loss_component_sums = {}
            self._loss_component_count = 0
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

    def evaluate(self, *args, **kwargs):
        try:
            metrics = super().evaluate(*args, **kwargs)
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

        if not self.is_world_process_zero():
            return metrics

        # determine metric key
        metric_key = getattr(self.args, "metric_for_best_model", None)
        if metric_key is None:
            metric_key = "eval_loss"

        # possible keys in returned metrics
        candidate_keys = [metric_key, f"eval_{metric_key}", "loss", "eval_loss"]
        metric_value = None
        for k in candidate_keys:
            if k in metrics:
                metric_value = metrics[k]
                break

        # find latest checkpoint (highest step)
        ckpts = self._list_checkpoint_dirs()
        latest_ckpt = str(ckpts[-1]) if ckpts else None

        # init best tracking
        if not hasattr(self, "_best_metric"):
            self._best_metric = None
            self._best_ckpt = None

        # compare metrics
        is_better = False
        if metric_value is not None:
            greater_is_better = getattr(self.args, "greater_is_better", False)
            if self._best_metric is None:
                is_better = True
            else:
                if greater_is_better:
                    is_better = metric_value > self._best_metric
                else:
                    is_better = metric_value < self._best_metric

        # if we have a new best, copy latest checkpoint to checkpoint-best
        out = Path(self.args.output_dir)
        best_dir = out / "checkpoint-best"
        if is_better and latest_ckpt is not None:
            # remove previous best if exists and different
            if self._best_ckpt and best_dir.exists():
                try:
                    shutil.rmtree(best_dir)
                except Exception:
                    pass
            try:
                # copy latest to checkpoint-best
                if best_dir.exists():
                    shutil.rmtree(best_dir)
                shutil.copytree(latest_ckpt, best_dir)
                self._best_metric = metric_value
                self._best_ckpt = str(best_dir)
            except Exception:
                # fallback: just record path
                self._best_metric = metric_value
                self._best_ckpt = latest_ckpt

        # determine keep paths: latest and best (if exist)
        keep = set()
        if latest_ckpt:
            keep.add(str(latest_ckpt))
        if self._best_ckpt:
            keep.add(str(self._best_ckpt))

        # cleanup other checkpoints
        self._cleanup_checkpoints(keep_paths=keep)

        return metrics

class NodeSFTDataCollator:
    def __init__(self, pitch_pad_id=128, task_type="epr"):
        self.pitch_pad_id = pitch_pad_id
        self.task_type = task_type

    def __call__(self, examples):
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

        pitch_ids = pad_sequence(pitch_tensors, batch_first=True, padding_value=self.pitch_pad_id)
        continuous = pad_sequence(continuous_tensors, batch_first=True, padding_value=0.0)
        labels_continuous = pad_sequence(label_tensors, batch_first=True, padding_value=0.0)
        score_shared_raw = pad_sequence(score_shared_raw_tensors, batch_first=True, padding_value=0.0)
        interpolated = pad_sequence(interpolated_tensors, batch_first=True, padding_value=False)
        attention_mask = (pitch_ids != self.pitch_pad_id).long()
        label_mask = None
        if self.task_type == "csr":
            label_mask_tensors = [
                torch.tensor(example["label_mask"], dtype=torch.long) for example in examples
            ]
            label_mask = pad_sequence(label_mask_tensors, batch_first=True, padding_value=0)

        batch = {
            "pitch_ids": pitch_ids,
            "continuous": continuous,
            "labels_continuous": labels_continuous,
            "score_shared_raw": score_shared_raw,
            "attention_mask": attention_mask,
            "interpolated": interpolated,
        }
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
    epr_timing_target = str(train_config.get("epr_timing_target", "absolute")).lower()
    timing_control_mode = resolve_timing_control_mode(
        timing_control_mode=train_config.get("timing_control_mode"),
        use_timing_scale_bit=train_config.get("use_timing_scale_bit", True),
    )
    use_timing_scale_bit = timing_control_mode == "piecewise_scale_bit"
    note_embedding_mode = str(train_config.get("note_embedding_mode", "sine")).lower()
    if input_feature_mode != "integrated":
        raise ValueError(f"INR0624 only supports input_feature_mode=integrated, got {input_feature_mode}")
    if note_embedding_mode not in {"sine", "cine"}:
        raise ValueError(f"INR0624 only supports note_embedding_mode in {{'sine', 'cine'}}, got {note_embedding_mode}")
    if task_type == "epr":
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
            "logistic_normal",
            "mixture_logistic_normal",
            "inflated_mixture_logistic_normal",
            "mixture_beta",
        }
        if distribution not in supported_distributions:
            raise ValueError(f"Unsupported epr_distribution={distribution}")
        mixture_distributions = {
            "logistic_normal",
            "mixture_logistic_normal",
            "inflated_mixture_logistic_normal",
            "mixture_beta",
        }
        if distribution in mixture_distributions:
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
            if distribution == "mixture_beta":
                missing_keys.append("beta_alpha_min") if "beta_alpha_min" not in train_config else None
            if distribution == "inflated_mixture_logistic_normal" and "epr_inflated_features" not in train_config:
                missing_keys.append("epr_inflated_features")
            if missing_keys:
                raise ValueError(f"EPR {distribution} config is missing required keys: {missing_keys}")
            components = int(train_config["epr_mixture_components"])
            if components < 1:
                raise ValueError(f"epr_mixture_components must be >= 1, got {components}")
            if distribution == "logistic_normal" and components != 1:
                raise ValueError("epr_distribution=logistic_normal requires epr_mixture_components=1")
            if distribution in {"mixture_logistic_normal", "inflated_mixture_logistic_normal", "mixture_beta"} and components < 2:
                raise ValueError(f"epr_distribution={distribution} requires epr_mixture_components >= 2")
            if distribution == "inflated_mixture_logistic_normal":
                expected = {"ioi": "zero", "pedal": "zero_one"}
                if train_config.get("epr_inflated_features") != expected:
                    raise ValueError(
                        "inflated_mixture_logistic_normal currently requires "
                        f"epr_inflated_features={expected}"
                    )
        if distribution == "beta_mu_kappa":
            missing_beta_keys = [
                key for key in ("beta_eps", "beta_kappa_min")
                if key not in train_config
            ]
            if missing_beta_keys:
                raise ValueError(f"EPR beta_mu_kappa config is missing required keys: {missing_beta_keys}")
        if epr_timing_target in {"deviation", "dev", "deviation_ratio", "dev_ratio", "log_deviation", "log_dev"}:
            if int(train_config.get("output_continuous_dim", train_config["continuous_dim"])) != 5:
                raise ValueError("deviation EPR requires output_continuous_dim=5")
            if str(train_config.get("pedal_representation", "continuous_4")).lower() != "start_ctrl":
                raise ValueError("deviation EPR currently requires pedal_representation=start_ctrl")
            if distribution not in {
                "point",
                "huber",
                "deterministic_huber",
                "logistic_normal",
                "mixture_logistic_normal",
                "beta_mu_kappa",
                "mixture_beta",
            }:
                raise ValueError(
                    "deviation EPR currently supports point/huber, mln, mln3, beta, and mixture_beta, "
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
    input_continuous_dim = train_config.get(
        "input_continuous_dim",
        integrated_epr_input_dim(
            timing_control_mode=timing_control_mode,
            use_timing_scale_bit=use_timing_scale_bit,
        )
        if task_type == "epr" and input_feature_mode == "integrated"
        else default_input_continuous_dim(
            task_type,
            input_feature_mode,
            score_feature_dim=score_feature_dim,
            continuous_dim=train_config.get("continuous_dim", 7),
        ),
    )
    if task_type in {"epr", "csr"} and input_feature_mode == "integrated":
        expected_input_dim = integrated_epr_input_dim(
            timing_control_mode=timing_control_mode,
            use_timing_scale_bit=use_timing_scale_bit,
        )
        if int(input_continuous_dim) != expected_input_dim:
            raise ValueError(
                f"Integrated INR0624 {task_type.upper()} expects input_continuous_dim={expected_input_dim} "
                f"for timing_control_mode={timing_control_mode}, got {input_continuous_dim}"
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
        continuous_dim=train_config["continuous_dim"],
        input_continuous_dim=input_continuous_dim,
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
        csr_mioi_max=train_config.get("csr_mioi_max", 6.0),
        csr_md_max=train_config.get("csr_md_max", 6.0),
        csr_ml_max=train_config.get("csr_ml_max", 6.0),
        huber_delta=train_config["huber_delta"],
        loss_weights=train_config["loss_weights"],
        csr_loss_weights=train_config.get("csr_loss_weights"),
        decoder_input_mode=train_config["decoder_input_mode"],
        input_feature_mode=input_feature_mode,
        note_embedding_mode=note_embedding_mode,
        special_note_vocab_size=train_config.get("special_note_vocab_size", 5),
        special_note_ids=train_config.get("special_note_ids"),
        use_full_type_embedding=train_config.get("use_full_type_embedding", True),
        use_group_presence_mask=train_config.get("use_group_presence_mask", True),
        head_input_mode=train_config.get("head_input_mode", "full"),
        embedding_depth=train_config.get("embedding_depth", 2),
        head_depth=train_config.get("head_depth", 2),
        head_activation=train_config.get("head_activation", "gelu"),
        epr_distribution=train_config.get("epr_distribution", "point"),
        epr_mixture_components=train_config.get("epr_mixture_components", 1),
        epr_distribution_eps=train_config.get("epr_distribution_eps"),
        logistic_normal_sigma_min=train_config.get("logistic_normal_sigma_min", 1e-3),
        logistic_normal_sigma_max=train_config.get("logistic_normal_sigma_max", 10.0),
        beta_eps=train_config.get("beta_eps", 1e-5),
        beta_kappa_min=train_config.get("beta_kappa_min", 1e-3),
        beta_alpha_min=train_config.get("beta_alpha_min", 1e-4),
        epr_inflated_features=train_config.get("epr_inflated_features"),
        epr_timing_bins=train_config.get("epr_timing_bins", 5000),
        epr_value_bins=train_config.get("epr_value_bins", 128),
        epr_timing_target=epr_timing_target,
        timing_control_mode=timing_control_mode,
        timing_log_scale=train_config.get("timing_log_scale", 50.0),
        use_timing_scale_bit=use_timing_scale_bit,
        soft_ce_tau=train_config.get("soft_ce_tau"),
        timing_input_normalization=train_config.get("timing_input_normalization", "scaled_log_5000_s10"),
        prior_token_keep_prob=train_config.get("prior_token_keep_prob", 1.0),
        prior_token_dropout_mode=train_config.get("prior_token_dropout_mode", "mask"),
        piano_pitch_min=train_config.get("piano_pitch_min", 21),
        pedal_representation=train_config.get("pedal_representation", "continuous_4"),
        pedal_start_loss_weight=train_config.get("pedal_start_loss_weight", 1.0),
        pedal_ctrl_loss_weight=train_config.get("pedal_ctrl_loss_weight", 1.0),
        torch_dtype=dtype,
    )

    resume_path = train_config.get("resume_path")
    if resume_path:
        model = IntegratedPianoT5Gemma(model_config) if backbone_type in {"t5", "t5gemma"} else IntegratedPianoTransformer(model_config)
        state_dict = load_torch_state_dict(resume_path)
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

    if train_config["load_best_model_at_end"] and eval_strategy == "steps" and save_strategy == "steps":
        eval_steps = train_config.get("eval_steps")
        if eval_steps:
            train_config["save_steps"] = eval_steps


def main():
    current_datetime = datetime.datetime.now()
    outname = "inr_" + current_datetime.strftime("%Y-%m-%d-%H-%M-%S")

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
    task_type = train_config.get("task_type", "epr").lower()
    input_feature_mode = infer_input_feature_mode(train_config)
    train_config["input_feature_mode"] = input_feature_mode
    timing_control_mode = resolve_timing_control_mode(
        timing_control_mode=train_config.get("timing_control_mode"),
        use_timing_scale_bit=train_config.get("use_timing_scale_bit", True),
    )
    train_config.setdefault(
        "input_continuous_dim",
        integrated_epr_input_dim(
            timing_control_mode=timing_control_mode,
            use_timing_scale_bit=train_config.get("use_timing_scale_bit", True),
        )
        if task_type in {"epr", "csr"} and input_feature_mode == "integrated"
        else default_input_continuous_dim(
            task_type,
            input_feature_mode,
            score_feature_dim=train_config.get("score_feature_dim", 8),
            continuous_dim=train_config.get("continuous_dim", 7),
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
    )
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
    )
    print(f"Train works: {len(train_manifest)}")
    print(f"Eval works: {len(eval_manifest)}")
    print(f"Estimated train examples: {sum(item['estimated_examples'] for item in train_manifest):,}")
    print(f"Estimated eval examples: {sum(item['estimated_examples'] for item in eval_manifest):,}")

    train_dataset = PianoCoReNodeSFTDataset(
        train_manifest,
        split="train",
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
        pedal_representation=train_config.get("pedal_representation", "continuous_4"),
        epr_timing_target=train_config.get("epr_timing_target", "absolute"),
        use_timing_scale_bit=train_config.get("use_timing_scale_bit", True),
        timing_control_mode=train_config.get("timing_control_mode"),
        timing_log_scale=train_config.get("timing_log_scale", 50.0),
    )
    eval_dataset = PianoCoReNodeSFTDataset(
        eval_manifest,
        split=train_config.get("eval_split", "test"),
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
        pedal_representation=train_config.get("pedal_representation", "continuous_4"),
        epr_timing_target=train_config.get("epr_timing_target", "absolute"),
        use_timing_scale_bit=train_config.get("use_timing_scale_bit", True),
        timing_control_mode=train_config.get("timing_control_mode"),
        timing_log_scale=train_config.get("timing_log_scale", 50.0),
    )

    model = create_model(train_config)
    model.to(device)
    print_model_parameters(model)

    training_args_dict = filter_valid_args(train_config, TrainingArguments)
    if args.deepspeed:
        training_args_dict["deepspeed"] = args.deepspeed
    if "accelerator_config" in train_config:
        training_args_dict["accelerator_config"] = train_config["accelerator_config"]
    # Integrated INR uses custom continuous labels instead of the standard
    # `labels` field. Tell Trainer explicitly so eval computes `eval_loss`.
    training_args_dict.setdefault("label_names", ["labels_continuous"])
    if int(train_config.get("dataloader_num_workers", 0) or 0) > 0:
        # Keep workers alive after dataloader warmup; this reduces CPU/input
        # stalls for the bs32/acc1 DDP recipe used by the pedal2 experiments.
        training_args_dict.setdefault("dataloader_persistent_workers", True)
    training_args_dict.setdefault("dataloader_pin_memory", torch.cuda.is_available())
    if torch.cuda.device_count() > 1:
        training_args_dict.setdefault("ddp_find_unused_parameters", True)
        training_args_dict.setdefault("ddp_broadcast_buffers", False)
    training_args = TrainingArguments(**training_args_dict)

    trainer = NodeSFTTrainer(
        model=model,
        args=training_args,
        data_collator=NodeSFTDataCollator(
            pitch_pad_id=train_config["pitch_pad_id"],
            task_type=task_type,
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    if "eval_dataloader_num_workers" not in train_config:
        train_config["eval_dataloader_num_workers"] = 0
    if "eval_dataloader_persistent_workers" not in train_config:
        train_config["eval_dataloader_persistent_workers"] = False
    if "eval_dataloader_prefetch_factor" not in train_config:
        train_config["eval_dataloader_prefetch_factor"] = None
    if "eval_dataloader_pin_memory" not in train_config:
        train_config["eval_dataloader_pin_memory"] = training_args.dataloader_pin_memory
    trainer.eval_dataloader_num_workers = int(train_config.get("eval_dataloader_num_workers", 0) or 0)
    trainer.eval_dataloader_persistent_workers = bool(train_config.get("eval_dataloader_persistent_workers", False))
    trainer.eval_dataloader_prefetch_factor = train_config.get("eval_dataloader_prefetch_factor")
    trainer.eval_dataloader_pin_memory = bool(
        train_config.get("eval_dataloader_pin_memory", training_args.dataloader_pin_memory)
    )

    resume_path = train_config.get("resume_path")
    resume_trainer_state = bool(train_config.get("resume_trainer_state", True))
    trainer.train(resume_from_checkpoint=resume_path if resume_path and resume_trainer_state else None)
    trainer.save_model()


if __name__ == "__main__":
    main()
