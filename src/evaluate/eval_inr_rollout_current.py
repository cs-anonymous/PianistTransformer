import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import random
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wasserstein_distance
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_process.work_manifest import build_work_manifest
from src.inference.infer_inr_testset import select_device, select_worker_device
from src.model.integrated_pianoformer import (
    _build_ar_note_continuous,
    _build_ar_special_note_ids,
    _decode_mixture_value,
    _logistic_normal_params,
    _materialize_epr_prediction,
    _shift_pitch_right,
    _shift_pitch_multihot_right,
    _shared_scalar_params,
    _split_epr_mixture_params,
    _target7_to_raw7,
    _target_predictions_to_feedback7,
)
from src.train.train_inr import (
    build_epr_score_input_rows,
    create_model,
    infer_input_feature_mode,
    performance_dev_velocity_pedal4_binary_rows,
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

RAW_DISTRIBUTION_KEYS = [
    ("ioi_ms", "IOI raw"),
    ("duration_ms", "Duration raw"),
    ("velocity", "Velocity"),
    ("pedal_0", "Pedal 0%"),
    ("pedal_25", "Pedal 25%"),
    ("pedal_50", "Pedal 50%"),
    ("pedal_75", "Pedal 75%"),
    ("onset_offset_ms", "Onset offset"),
    ("duration_offset_ms", "Duration offset"),
    ("velocity_offset", "Velocity offset"),
]

TARGET_DISTRIBUTION_KEYS = [
    ("ioi_log_dev", "IOI log-dev"),
    ("duration_log_dev", "Duration log-dev"),
    ("velocity_norm", "Velocity norm"),
    ("pedal_0", "Pedal 0%"),
    ("pedal_25", "Pedal 25%"),
    ("pedal_50", "Pedal 50%"),
    ("pedal_75", "Pedal 75%"),
    ("onset_offset_s", "Onset offset s"),
    ("duration_offset_s", "Duration offset s"),
    ("velocity_offset_norm", "Velocity offset norm"),
]

TIMING_HEAD_FEATURES = [("ioi", 0), ("duration", 1)]

POSITION_FEATURES = [
    ("ioi_log_dev", 0),
    ("duration_log_dev", 1),
    ("velocity_norm", 2),
    ("pedal_0", 3),
    ("pedal_25", 4),
    ("pedal_50", 5),
    ("pedal_75", 6),
]


def has_raw_log_layout(config):
    return str(config.get("epr_timing_target", "log_deviation")).lower() in {
        "raw_log_deviation",
        "raw_log_dev",
    }


def offset_dim(config):
    return 3 if bool(config.get("chord_mode", False)) and bool(config.get("include_chord_offsets", True)) else 0


def target_layout(config, target_rows):
    target_rows = np.asarray(target_rows)
    raw_log = has_raw_log_layout(config)
    if raw_log:
        velocity_col = 4
        pedal_start = 5
    else:
        velocity_col = 2
        pedal_start = 3
    off_dim = offset_dim(config)
    offset_start = pedal_start + 4 if off_dim > 0 and target_rows.shape[-1] >= pedal_start + 4 + off_dim else None
    return velocity_col, pedal_start, offset_start, off_dim


def _temperature_mixture_value(config, logits, raw_mu, raw_log_sigma, sampling_strategy="mean", temperature=1.0):
    temperature = float(temperature)
    mode = str(sampling_strategy).lower()
    if temperature == 1.0 or mode not in {"sample", "sampling", "stochastic"}:
        return _decode_mixture_value(
            config,
            logits,
            raw_mu,
            raw_log_sigma,
            None,
            sampling_strategy=sampling_strategy,
        )

    mu, sigma = _logistic_normal_params(
        raw_mu,
        raw_log_sigma,
        sigma_min=getattr(config, "logistic_normal_sigma_min", 1e-3),
        sigma_max=getattr(config, "logistic_normal_sigma_max", 10.0),
    )
    probs = torch.softmax(logits.float(), dim=-1)
    if temperature <= 0.0:
        return torch.sum(probs * torch.sigmoid(mu), dim=-1)

    component_probs = torch.softmax(logits.float() / max(temperature, 1e-6), dim=-1)
    index = torch.distributions.Categorical(probs=component_probs).sample().unsqueeze(-1)
    sampled_mu = mu.gather(dim=-1, index=index).squeeze(-1)
    sampled_sigma = sigma.gather(dim=-1, index=index).squeeze(-1)
    return torch.sigmoid(torch.distributions.Normal(sampled_mu, sampled_sigma * temperature).sample())


def materialize_epr_prediction_eval(
    config,
    raw_outputs,
    sampling_strategy="mean",
    score_shared_raw=None,
    continuous_temperature=1.0,
):
    temperature = float(continuous_temperature)
    if temperature == 1.0:
        return _materialize_epr_prediction(
            config,
            raw_outputs,
            sampling_strategy=sampling_strategy,
            score_shared_raw=score_shared_raw,
        )

    distribution = str(getattr(config, "epr_distribution", "point")).lower()
    if distribution != "mixture_logistic_normal":
        return _materialize_epr_prediction(
            config,
            raw_outputs,
            sampling_strategy=sampling_strategy,
            score_shared_raw=score_shared_raw,
        )

    params = _split_epr_mixture_params(config, raw_outputs)
    shared_values = []
    for index in range(3):
        logits, raw_mu, raw_log_sigma, raw_extra = _shared_scalar_params(config, params, index)
        if raw_extra is not None:
            return _materialize_epr_prediction(
                config,
                raw_outputs,
                sampling_strategy=sampling_strategy,
                score_shared_raw=score_shared_raw,
            )
        shared_values.append(
            _temperature_mixture_value(
                config,
                logits,
                raw_mu,
                raw_log_sigma,
                sampling_strategy=sampling_strategy,
                temperature=temperature,
            )
        )
    shared = torch.stack(shared_values, dim=-1)
    pedal = _materialize_epr_prediction(
        config,
        raw_outputs,
        sampling_strategy=sampling_strategy,
        score_shared_raw=score_shared_raw,
    )[..., 3:7]
    return torch.cat([shared, pedal], dim=-1)


def empty_head_stats(config):
    components = int(config.get("epr_mixture_components", 1)) if isinstance(config, dict) else int(getattr(config, "epr_mixture_components", 1))
    base_keys = [
        "pred_mean_norm",
        "pred_mode_norm",
        "pred_std_norm",
        "weighted_mu_z",
        "weighted_sigma_z",
        "top_weight",
        "weight_entropy",
    ]
    store = {}
    for feature, _ in TIMING_HEAD_FEATURES:
        stats = {"n": 0.0, **{key: 0.0 for key in base_keys}}
        for idx in range(components):
            stats[f"comp{idx}_weight"] = 0.0
            stats[f"comp{idx}_mu_z"] = 0.0
            stats[f"comp{idx}_sigma_z"] = 0.0
            stats[f"comp{idx}_loc_norm"] = 0.0
        store[feature] = stats
    return store


def update_head_stats(store, config, raw_outputs, attention_mask):
    if store is None:
        return
    distribution = str(getattr(config, "epr_distribution", "point")).lower()
    if distribution != "mixture_logistic_normal":
        return

    params = _split_epr_mixture_params(config, raw_outputs)
    mask = attention_mask.detach().bool()
    if mask.ndim == 3:
        mask = mask.squeeze(-1)
    if not mask.any():
        return

    for feature, index in TIMING_HEAD_FEATURES:
        logits, raw_mu, raw_log_sigma, raw_extra = _shared_scalar_params(config, params, index)
        if raw_extra is not None:
            continue
        mu, sigma = _logistic_normal_params(
            raw_mu,
            raw_log_sigma,
            sigma_min=getattr(config, "logistic_normal_sigma_min", 1e-3),
            sigma_max=getattr(config, "logistic_normal_sigma_max", 10.0),
        )
        probs = torch.softmax(logits.float(), dim=-1)
        loc_norm = torch.sigmoid(mu)
        pred_mean = torch.sum(probs * loc_norm, dim=-1)
        top_index = probs.argmax(dim=-1, keepdim=True)
        pred_mode = loc_norm.gather(dim=-1, index=top_index).squeeze(-1)
        top_weight = probs.gather(dim=-1, index=top_index).squeeze(-1)
        derivative = loc_norm * (1.0 - loc_norm)
        component_var = (derivative * sigma).square()
        pred_var = torch.sum(probs * (component_var + (loc_norm - pred_mean.unsqueeze(-1)).square()), dim=-1)
        pred_std = pred_var.clamp_min(0.0).sqrt()
        weighted_mu = torch.sum(probs * mu, dim=-1)
        weighted_sigma = torch.sum(probs * sigma, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs.clamp_min(1e-12)), dim=-1)

        masked_count = float(mask.sum().detach().cpu().item())
        stats = store[feature]
        stats["n"] += masked_count
        for key, values in (
            ("pred_mean_norm", pred_mean),
            ("pred_mode_norm", pred_mode),
            ("pred_std_norm", pred_std),
            ("weighted_mu_z", weighted_mu),
            ("weighted_sigma_z", weighted_sigma),
            ("top_weight", top_weight),
            ("weight_entropy", entropy),
        ):
            stats[key] += float(values.detach()[mask].float().sum().cpu().item())
        for comp_idx in range(probs.shape[-1]):
            for key, values in (
                (f"comp{comp_idx}_weight", probs[..., comp_idx]),
                (f"comp{comp_idx}_mu_z", mu[..., comp_idx]),
                (f"comp{comp_idx}_sigma_z", sigma[..., comp_idx]),
                (f"comp{comp_idx}_loc_norm", loc_norm[..., comp_idx]),
            ):
                stats[key] += float(values.detach()[mask].float().sum().cpu().item())


def finalize_head_stats(store):
    if store is None:
        return None
    output = {}
    for feature, stats in store.items():
        n = float(stats.get("n", 0.0))
        if n <= 0.0:
            output[feature] = {"n": 0}
            continue
        output[feature] = {key: (int(n) if key == "n" else float(value) / n) for key, value in stats.items()}
    return output


def aggregate_head_stats(items, rollout_ks):
    rows = []
    for rollout_k in rollout_ks:
        label = k_label(rollout_k)
        by_feature = {feature: {} for feature, _ in TIMING_HEAD_FEATURES}
        for item in items:
            head_stats = item.get("by_k", {}).get(label, {}).get("head_stats") or {}
            for feature, stats in head_stats.items():
                n = float(stats.get("n", 0.0))
                if n <= 0.0:
                    continue
                accum = by_feature.setdefault(feature, {})
                accum["n"] = accum.get("n", 0.0) + n
                for key, value in stats.items():
                    if key == "n":
                        continue
                    accum[key] = accum.get(key, 0.0) + float(value) * n
        for feature, accum in by_feature.items():
            n = float(accum.get("n", 0.0))
            if n <= 0.0:
                continue
            row = {"k": label, "feature": feature, "n": int(n)}
            row.update({key: float(value) / n for key, value in sorted(accum.items()) if key != "n"})
            rows.append(row)
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Current INR target7 rollout-k evaluation.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--performance-dataset", default="ASAP")
    parser.add_argument("--score-source-list", type=Path, default=None)
    parser.add_argument("--score-source", action="append", default=None)
    parser.add_argument("--max-works", type=int, default=None)
    parser.add_argument("--batch-size-windows", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument(
        "--materialize-strategy",
        choices=["mean", "greedy", "sample"],
        default="sample",
    )
    parser.add_argument(
        "--feedback-strategy",
        choices=[
            "mean",
            "greedy",
            "sample",
            "soft",
        ],
        default=None,
    )
    parser.add_argument(
        "--feedback-mode",
        choices=["pollute", "protect"],
        default="pollute",
        help="pollute replaces selected feedback targets with predictions; protect keeps selected targets as GT.",
    )
    parser.add_argument(
        "--feedback-targets",
        choices=["all", "ioi", "duration", "velocity", "pedal", "timing"],
        default="all",
        help="Target dimensions affected by feedback-mode for k>0 rollouts.",
    )
    parser.add_argument("--rollout-ks", default="0,1,full")
    parser.add_argument(
        "--fast-kpass",
        action="store_true",
        help="For finite k values, compute iterative TF feedback passes instead of per-note partial rollout.",
    )
    parser.add_argument("--plot-distributions", action="store_true")
    parser.add_argument("--save-distribution-values", action="store_true")
    parser.add_argument("--window-position-stats", action="store_true")
    parser.add_argument(
        "--head-stats",
        action="store_true",
        help="Aggregate timing-head distribution parameters by rollout k.",
    )
    parser.add_argument(
        "--continuous-temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for EPR shared continuous heads; 0 uses deterministic means.",
    )
    return parser.parse_args()


def finite_mean(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if len(finite) else float("nan")


def feature_wasserstein(pred_values, gt_values):
    pred_values = np.asarray(pred_values, dtype=np.float64)
    gt_values = np.asarray(gt_values, dtype=np.float64)
    pred_values = pred_values[np.isfinite(pred_values)]
    gt_values = gt_values[np.isfinite(gt_values)]
    if len(pred_values) == 0 or len(gt_values) == 0:
        return float("nan")
    return float(wasserstein_distance(pred_values, gt_values))


def finite_std(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.std(values)) if len(values) else float("nan")


def finite_mean_shift(pred_values, gt_values):
    pred_values, gt_values = finite_pairs(pred_values, gt_values)
    if len(pred_values) == 0:
        return float("nan")
    return float(np.mean(pred_values - gt_values))


def finite_std_ratio(pred_values, gt_values):
    pred_std = finite_std(pred_values)
    gt_std = finite_std(gt_values)
    if not np.isfinite(pred_std) or not np.isfinite(gt_std) or gt_std <= 1e-12:
        return float("nan")
    return float(pred_std / gt_std)


def parse_rollout_ks(text):
    out = []
    seen = set()
    for raw in str(text).split(","):
        token = raw.strip().lower()
        if not token:
            continue
        value = None if token == "full" else int(token)
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    if not out:
        raise ValueError("No rollout k values were provided")
    return out


def k_label(value):
    return "full" if value is None else str(int(value))


def feedback_target_columns(name, output_dim):
    name = str(name or "all").lower()
    if name == "all":
        return list(range(output_dim))
    if name == "ioi":
        return [0]
    if name == "duration":
        return [1]
    if name == "velocity":
        return [2]
    if name == "pedal":
        return list(range(3, min(output_dim, 7)))
    if name == "timing":
        return [0, 1]
    raise ValueError(f"Unsupported feedback target set: {name}")


def load_score_source_filter(args):
    selected = []
    if args.score_source:
        selected.extend(str(item).strip() for item in args.score_source)
    if args.score_source_list is not None:
        for line in args.score_source_list.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                selected.append(line)
    return selected or None


def filter_manifest(manifest, score_sources):
    if not score_sources:
        return manifest
    wanted = set(score_sources)
    order = {score: idx for idx, score in enumerate(score_sources)}
    filtered = [item for item in manifest if item.get("score_source") in wanted]
    found = {item.get("score_source") for item in filtered}
    missing = [score for score in score_sources if score not in found]
    if missing:
        raise ValueError(f"Requested score_source not found: {missing[0]}")
    return sorted(filtered, key=lambda item: order[item["score_source"]])


def stable_seed(base_seed, *parts):
    payload = "::".join([str(base_seed), *[str(part) for part in parts]])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def load_config(config_path, checkpoint):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    config["resume_path"] = str(checkpoint)
    config["resume_trainer_state"] = False
    config["input_feature_mode"] = infer_input_feature_mode(config)
    config["use_style_tokens"] = False
    if str(config.get("task_type", "epr")).lower() != "epr":
        raise ValueError("This script only supports EPR")
    if str(config.get("pedal_representation", "binary_4")).lower() != "binary_4":
        raise ValueError("This script only supports pedal_representation=binary_4")
    if str(config.get("epr_timing_target", "log_deviation")).lower() not in {
        "log_deviation",
        "log_dev",
        "raw_log_deviation",
        "raw_log_dev",
        "raw_deviation",
        "raw_dev",
        "raw_seconds_deviation",
        "raw_seconds_dev",
    }:
        raise ValueError("This script only supports log_deviation/raw_log_deviation/raw_deviation timing targets")
    return config


def selected_perfs(work, item):
    by_source = {perf.get("performance_source"): perf for perf in work.get("performances", [])}
    return [
        by_source[source]
        for source in item.get("selected_performance_sources", [])
        if source in by_source
    ]


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
        if window not in seen:
            deduped.append(window)
            seen.add(window)
    return deduped


def labels_for_perf(config, perf, score_shared_raw):
    labels = performance_dev_velocity_pedal4_binary_rows(
        perf,
        score_shared_raw,
        epr_timing_target=config.get("epr_timing_target", "log_deviation"),
        log_scale=float(config.get("timing_log_scale", 50.0)),
        pedal_binary_threshold=float(config.get("pedal_binary_threshold", 64.0)),
        include_chord_offsets=bool(config.get("chord_mode", False))
        and bool(config.get("include_chord_offsets", True)),
    )
    if labels is None:
        raise ValueError(f"Could not build target labels for {perf.get('performance_source')}")
    return labels


def pitch_ids_from_pitch_values(pitches, pitch_pad_id):
    pitch_ids = []
    for value in pitches:
        if isinstance(value, (list, tuple)):
            pitch_ids.append(max(int(pitch) for pitch in value) if value else int(pitch_pad_id))
        else:
            pitch_ids.append(int(value))
    return pitch_ids


def pitch_multihot_rows(config, score, start, end):
    dim = int(config.get("pitch_multihot_dim", 88))
    piano_min = int(config.get("piano_pitch_min", 21))
    if "pitch_multihot" in score:
        rows = score["pitch_multihot"][start:end]
        if rows and len(rows[0]) == dim:
            return rows
        if rows and len(rows[0]) == 128 and dim == 88:
            return [row[piano_min : piano_min + dim] for row in rows]
        raise ValueError(f"Unsupported pitch_multihot dim {len(rows[0]) if rows else 0}; expected {dim}")
    rows = []
    for value in score["pitch"][start:end]:
        row = [0.0] * dim
        pitches = value if isinstance(value, (list, tuple)) else [value]
        for pitch in pitches:
            idx = int(pitch) - piano_min
            if 0 <= idx < dim:
                row[idx] = 1.0
        rows.append(row)
    return rows


def chord_mask_from_pitch_values(pitches):
    return np.asarray(
        [
            len(value) > 1 if isinstance(value, (list, tuple)) else False
            for value in pitches
        ],
        dtype=bool,
    )


def target_offsets3(config, target_rows):
    target_rows = torch.as_tensor(target_rows).float()
    off_dim = offset_dim(config)
    if off_dim <= 0 or target_rows.shape[-1] < off_dim:
        return target_rows.new_empty(*target_rows.shape[:-1], 0)
    return target_rows[..., -off_dim:]


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


def metric_arrays(raw_rows, target_rows):
    arrays = raw_arrays_from_rows(raw_rows)
    target_rows = np.asarray(target_rows, dtype=np.float64)
    arrays["pedal_0"] = target_rows[:, 3]
    arrays["pedal_25"] = target_rows[:, 4]
    arrays["pedal_50"] = target_rows[:, 5]
    arrays["pedal_75"] = target_rows[:, 6]
    return arrays


def distribution_arrays(config, raw_rows, target_rows, chord_mask=None):
    raw_rows = np.asarray(raw_rows, dtype=np.float64)
    target_rows = np.asarray(target_rows, dtype=np.float64)
    velocity_col, pedal_start, offset_start, off_dim = target_layout(config, target_rows)
    raw = {
        "ioi_ms": raw_rows[:, 0],
        "duration_ms": raw_rows[:, 1],
        "velocity": raw_rows[:, 2],
        "pedal_0": target_rows[:, pedal_start],
        "pedal_25": target_rows[:, pedal_start + 1],
        "pedal_50": target_rows[:, pedal_start + 2],
        "pedal_75": target_rows[:, pedal_start + 3],
    }
    target = {
        "ioi_log_dev": target_rows[:, 0],
        "duration_log_dev": target_rows[:, 1],
        "velocity_norm": target_rows[:, velocity_col],
        "pedal_0": target_rows[:, pedal_start],
        "pedal_25": target_rows[:, pedal_start + 1],
        "pedal_50": target_rows[:, pedal_start + 2],
        "pedal_75": target_rows[:, pedal_start + 3],
    }
    if offset_start is not None and off_dim >= 3:
        mask = np.ones(target_rows.shape[0], dtype=bool) if chord_mask is None else np.asarray(chord_mask, dtype=bool)
        raw["onset_offset_ms"] = target_rows[mask, offset_start] * 1000.0
        raw["duration_offset_ms"] = target_rows[mask, offset_start + 1] * 1000.0
        raw["velocity_offset"] = target_rows[mask, offset_start + 2] * 127.0
        target["onset_offset_s"] = target_rows[mask, offset_start]
        target["duration_offset_s"] = target_rows[mask, offset_start + 1]
        target["velocity_offset_norm"] = target_rows[mask, offset_start + 2]
    else:
        raw["onset_offset_ms"] = np.asarray([], dtype=np.float64)
        raw["duration_offset_ms"] = np.asarray([], dtype=np.float64)
        raw["velocity_offset"] = np.asarray([], dtype=np.float64)
        target["onset_offset_s"] = np.asarray([], dtype=np.float64)
        target["duration_offset_s"] = np.asarray([], dtype=np.float64)
        target["velocity_offset_norm"] = np.asarray([], dtype=np.float64)
    return {"raw": raw, "target": target}


def empty_distributions():
    return {
        "raw": {
            "pred": {key: [] for key, _ in RAW_DISTRIBUTION_KEYS},
            "gt": {key: [] for key, _ in RAW_DISTRIBUTION_KEYS},
        },
        "target": {
            "pred": {key: [] for key, _ in TARGET_DISTRIBUTION_KEYS},
            "gt": {key: [] for key, _ in TARGET_DISTRIBUTION_KEYS},
        },
    }


def extend_distributions(store, config, pred_raw, gt_raw, pred_target, gt_target, chord_mask=None):
    pred = distribution_arrays(config, pred_raw, pred_target, chord_mask=chord_mask)
    gt = distribution_arrays(config, gt_raw, gt_target, chord_mask=chord_mask)
    for domain in ("raw", "target"):
        for key, values in pred[domain].items():
            store[domain]["pred"][key].extend(np.asarray(values, dtype=np.float64).tolist())
        for key, values in gt[domain].items():
            store[domain]["gt"][key].extend(np.asarray(values, dtype=np.float64).tolist())


def empty_position_values():
    return {
        scheme: {
            bucket: {name: {"pred": [], "gt": []} for name, _ in POSITION_FEATURES}
            for bucket in buckets
        }
        for scheme, buckets in (
            ("half", ("front", "back")),
            ("quarter", ("q1", "q2", "q3", "q4")),
        )
    }


def extend_position_values(store, pred_target, gt_target):
    pred_target = np.asarray(pred_target, dtype=np.float64)
    gt_target = np.asarray(gt_target, dtype=np.float64)
    if pred_target.ndim != 2 or len(pred_target) == 0:
        return
    length = len(pred_target)
    for idx in range(length):
        half_bucket = "front" if idx < (length / 2.0) else "back"
        quarter_bucket = f"q{min(3, int((idx * 4) / max(1, length))) + 1}"
        for name, col in POSITION_FEATURES:
            pred_value = float(pred_target[idx, col])
            gt_value = float(gt_target[idx, col])
            store["half"][half_bucket][name]["pred"].append(pred_value)
            store["half"][half_bucket][name]["gt"].append(gt_value)
            store["quarter"][quarter_bucket][name]["pred"].append(pred_value)
            store["quarter"][quarter_bucket][name]["gt"].append(gt_value)


def merge_position_values(dst, src):
    if src is None:
        return
    for scheme, buckets in src.items():
        for bucket, features in buckets.items():
            for name, values in features.items():
                dst[scheme][bucket][name]["pred"].extend(values["pred"])
                dst[scheme][bucket][name]["gt"].extend(values["gt"])


def pp_wass_metrics(pred_arrays, gt_arrays):
    output = {}
    for metric_name, feature_name in FEATURE_KEYS:
        pred_pool = np.concatenate([item[feature_name] for item in pred_arrays]) if pred_arrays else np.asarray([])
        gt_pool = np.concatenate([item[feature_name] for item in gt_arrays]) if gt_arrays else np.asarray([])
        output[f"{metric_name}_wass"] = feature_wasserstein(pred_pool, gt_pool)
    output["pedal_wass"] = finite_mean([output[f"pedal_{pos}_wass"] for pos in ("0", "25", "50", "75")])
    return output


def pn_wass_metrics(pred_arrays, gt_arrays):
    all_arrays = pred_arrays + gt_arrays
    output = {}
    for metric_name, feature_name in FEATURE_KEYS:
        usable = min((len(item[feature_name]) for item in all_arrays), default=0)
        note_wass = [
            feature_wasserstein(
                [item[feature_name][idx] for item in pred_arrays],
                [item[feature_name][idx] for item in gt_arrays],
            )
            for idx in range(usable)
        ]
        output[f"{metric_name}_wass"] = finite_mean(note_wass)
    output["pedal_wass"] = finite_mean([output[f"pedal_{pos}_wass"] for pos in ("0", "25", "50", "75")])
    return output


def score_metrics(score_source, pred_arrays, gt_arrays):
    return {
        "score_source": score_source,
        "num_predictions": len(pred_arrays),
        "num_ground_truth": len(gt_arrays),
        "pp_wass": pp_wass_metrics(pred_arrays, gt_arrays),
        "pn_wass": pn_wass_metrics(pred_arrays, gt_arrays),
    }


def aggregate_score_metrics(rows, section):
    if not rows:
        return {}
    keys = sorted(rows[0][section].keys())
    return {key: finite_mean([row[section].get(key, float("nan")) for row in rows]) for key in keys}


def aggregate_pairwise(rows):
    keys = [
        "loss",
        "ioi_wass",
        "duration_wass",
        "velocity_wass",
        "pedal_wass",
        "onset_offset_wass",
        "duration_offset_wass",
        "velocity_offset_wass",
        "ioi_ms_mean_shift",
        "duration_ms_mean_shift",
        "ioi_ms_std_ratio",
        "duration_ms_std_ratio",
        "ioi_logdev_mean_shift",
        "duration_logdev_mean_shift",
        "ioi_logdev_std_ratio",
        "duration_logdev_std_ratio",
        "ioi_logdev_pred_mean",
        "duration_logdev_pred_mean",
        "ioi_logdev_gt_mean",
        "duration_logdev_gt_mean",
        "ioi_logdev_pred_std",
        "duration_logdev_pred_std",
        "ioi_logdev_gt_std",
        "duration_logdev_gt_std",
        "onset_offset_ms_mean_shift",
        "duration_offset_ms_mean_shift",
        "velocity_offset_mean_shift",
        "onset_offset_ms_std_ratio",
        "duration_offset_ms_std_ratio",
        "velocity_offset_std_ratio",
    ]
    total_weight = float(sum(max(0, int(row.get("num_rows", 0))) for row in rows))
    output = {"num_rows": int(total_weight)}
    for key in keys:
        values = []
        weights = []
        for row in rows:
            weight = max(0, int(row.get("num_rows", 0)))
            value = float(row.get(key, float("nan")))
            if weight > 0 and np.isfinite(value):
                values.append(value)
                weights.append(weight)
        output[key] = float(np.average(values, weights=weights)) if weights else float("nan")
    return output


def pairwise_summary(rows):
    timing_keys = [
        "ioi_ms_mean_shift",
        "duration_ms_mean_shift",
        "ioi_ms_std_ratio",
        "duration_ms_std_ratio",
        "ioi_logdev_mean_shift",
        "duration_logdev_mean_shift",
        "ioi_logdev_std_ratio",
        "duration_logdev_std_ratio",
        "ioi_logdev_pred_mean",
        "duration_logdev_pred_mean",
        "ioi_logdev_gt_mean",
        "duration_logdev_gt_mean",
        "ioi_logdev_pred_std",
        "duration_logdev_pred_std",
        "ioi_logdev_gt_std",
        "duration_logdev_gt_std",
        "onset_offset_ms_mean_shift",
        "duration_offset_ms_mean_shift",
        "velocity_offset_mean_shift",
        "onset_offset_ms_std_ratio",
        "duration_offset_ms_std_ratio",
        "velocity_offset_std_ratio",
    ]
    return {
        "num_rows": int(len(rows)),
        "loss": finite_mean([row.get("loss", float("nan")) for row in rows]),
        "ioi_wass": finite_mean([row.get("ioi_wass", float("nan")) for row in rows]),
        "duration_wass": finite_mean([row.get("duration_wass", float("nan")) for row in rows]),
        "velocity_wass": finite_mean([row.get("velocity_wass", float("nan")) for row in rows]),
        "pedal_wass": finite_mean([row.get("pedal_wass", float("nan")) for row in rows]),
        "onset_offset_wass": finite_mean([row.get("onset_offset_wass", float("nan")) for row in rows]),
        "duration_offset_wass": finite_mean([row.get("duration_offset_wass", float("nan")) for row in rows]),
        "velocity_offset_wass": finite_mean([row.get("velocity_offset_wass", float("nan")) for row in rows]),
        **{key: finite_mean([row.get(key, float("nan")) for row in rows]) for key in timing_keys},
    }


def predict_tf_batch(
    model,
    pitch_ids,
    pitch_multihot,
    continuous,
    score_shared_raw,
    labels_continuous,
    attention_mask,
    sampling_strategy,
    continuous_temperature=1.0,
    head_stats=None,
):
    with torch.no_grad():
        outputs = model(
            pitch_ids=pitch_ids,
            pitch_multihot=pitch_multihot,
            continuous=continuous,
            score_shared_raw=score_shared_raw,
            labels_continuous=labels_continuous,
            attention_mask=attention_mask,
            continuous_sampling_strategy=sampling_strategy,
        )
        update_head_stats(head_stats, model.config, outputs.logits, attention_mask)
        pred = materialize_epr_prediction_eval(
            model.config,
            outputs.logits,
            sampling_strategy=sampling_strategy,
            score_shared_raw=score_shared_raw,
            continuous_temperature=continuous_temperature,
        )
    loss_value = float(outputs.loss.detach().float().cpu()) if outputs.loss is not None else float("nan")
    return pred.detach().float().cpu(), loss_value


def predict_full_ar_batch(model, pitch_ids, pitch_multihot, continuous, score_shared_raw, attention_mask, sampling_strategy):
    with torch.no_grad():
        pred = model.predict_performance_continuous(
            pitch_ids=pitch_ids,
            pitch_multihot=pitch_multihot,
            continuous=continuous,
            score_shared_raw=score_shared_raw,
            attention_mask=attention_mask,
            sampling_strategy=sampling_strategy,
        )
    return pred.detach().float().cpu(), float("nan")


def predict_partial_t5_batch(
    model,
    pitch_ids,
    pitch_multihot,
    continuous,
    score_shared_raw,
    labels_continuous,
    attention_mask,
    rollout_k,
    sampling_strategy,
    feedback_strategy,
    feedback_mode,
    feedback_targets,
    continuous_temperature=1.0,
    head_stats=None,
):
    config = model.config
    batch_size, seq_len = pitch_ids.shape
    output_dim = labels_continuous.shape[-1]
    score_note_embeds = model.note_encoder(pitch_ids, continuous, pitch_multihot=pitch_multihot)
    score_context_embeds, context_attention_mask, _ = model._prepend_style_tokens(score_note_embeds, attention_mask)
    encoder_outputs = model.model.encoder(attention_mask=context_attention_mask, inputs_embeds=score_context_embeds)
    decoder_pitch_ids = _shift_pitch_right(config, pitch_ids, attention_mask)
    decoder_pitch_multihot = _shift_pitch_multihot_right(pitch_multihot, attention_mask)
    special_note_ids = _build_ar_special_note_ids(config, attention_mask)
    predictions = labels_continuous.new_zeros((batch_size, seq_len, output_dim))
    feedback_predictions = labels_continuous.new_zeros((batch_size, seq_len, output_dim))
    decoder_dim = int(getattr(config, "decoder_input_continuous_dim", config.input_continuous_dim))
    feedback_cols = feedback_target_columns(feedback_targets, output_dim)

    for step in range(seq_len):
        active = attention_mask[:, step].bool()
        if not active.any():
            continue
        if step == 0:
            decoder_input_continuous = labels_continuous.new_zeros((batch_size, 1, decoder_dim))
        else:
            mixed_prefix = labels_continuous[:, :step].clone()
            effective_k = step if rollout_k is None else int(rollout_k)
            pred_start = max(0, step - effective_k)
            if str(feedback_mode).lower() == "protect":
                mixed_prefix[:, pred_start:step] = feedback_predictions[:, pred_start:step]
                if feedback_cols:
                    mixed_prefix[:, pred_start:step, feedback_cols] = labels_continuous[:, pred_start:step, feedback_cols]
            elif feedback_cols:
                mixed_prefix[:, pred_start:step, feedback_cols] = feedback_predictions[:, pred_start:step, feedback_cols]
            prefix_rows = _build_ar_note_continuous(
                config,
                mixed_prefix,
                score_shared_raw=score_shared_raw[:, :step],
                task_type=config.task_type,
            )
            decoder_input_continuous = prefix_rows.new_zeros((batch_size, step + 1, decoder_dim))
            decoder_input_continuous[:, 1:] = prefix_rows

        decoder_inputs_embeds = model.decoder_note_encoder(
            decoder_pitch_ids[:, : step + 1],
            decoder_input_continuous,
            special_note_ids=special_note_ids[:, : step + 1],
            pitch_multihot=(
                decoder_pitch_multihot[:, : step + 1]
                if decoder_pitch_multihot is not None
                else None
            ),
        )
        decoder_outputs = model.model(
            attention_mask=context_attention_mask,
            decoder_attention_mask=attention_mask[:, : step + 1],
            encoder_outputs=encoder_outputs,
            inputs_embeds=score_context_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
        )
        step_raw = model.continuous_decoder(decoder_outputs.last_hidden_state[:, -1:, :])
        update_head_stats(head_stats, config, step_raw, attention_mask[:, step : step + 1])
        step_pred = materialize_epr_prediction_eval(
            config,
            step_raw,
            sampling_strategy=sampling_strategy,
            score_shared_raw=score_shared_raw[:, step : step + 1],
            continuous_temperature=continuous_temperature,
        )
        if feedback_strategy == sampling_strategy:
            step_feedback = step_pred
        else:
            step_feedback = materialize_epr_prediction_eval(
                config,
                step_raw,
                sampling_strategy=feedback_strategy,
                score_shared_raw=score_shared_raw[:, step : step + 1],
                continuous_temperature=continuous_temperature,
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
    return predictions.detach().float().cpu(), float("nan")


def predict_batch(
    model,
    pitch_ids,
    pitch_multihot,
    continuous,
    score_shared_raw,
    labels_continuous,
    attention_mask,
    rollout_k,
    sampling_strategy,
    feedback_strategy,
    feedback_mode,
    feedback_targets,
    continuous_temperature=1.0,
    head_stats=None,
):
    if rollout_k == 0:
        return predict_tf_batch(
            model,
            pitch_ids,
            pitch_multihot,
            continuous,
            score_shared_raw,
            labels_continuous,
            attention_mask,
            sampling_strategy,
            continuous_temperature=continuous_temperature,
            head_stats=head_stats,
        )
    if (
        rollout_k is None
        and float(continuous_temperature) == 1.0
        and head_stats is None
        and str(feedback_mode).lower() == "pollute"
        and str(feedback_targets).lower() == "all"
        and str(feedback_strategy).lower() == str(sampling_strategy).lower()
    ):
        return predict_full_ar_batch(
            model,
            pitch_ids,
            pitch_multihot,
            continuous,
            score_shared_raw,
            attention_mask,
            sampling_strategy,
        )
    if not hasattr(model, "model"):
        raise ValueError("Partial rollout currently supports T5/T5Gemma only")
    with torch.no_grad():
        return predict_partial_t5_batch(
            model,
            pitch_ids,
            pitch_multihot,
            continuous,
            score_shared_raw,
            labels_continuous,
            attention_mask,
            rollout_k,
            sampling_strategy,
            feedback_strategy,
            feedback_mode,
            feedback_targets,
            continuous_temperature=continuous_temperature,
            head_stats=head_stats,
        )


def predict_tf_feedback_batch(
    model,
    pitch_ids,
    pitch_multihot,
    continuous,
    score_shared_raw,
    labels_continuous,
    attention_mask,
    decoder_feedback_continuous,
    sampling_strategy,
    feedback_strategy,
):
    with torch.no_grad():
        outputs = model(
            pitch_ids=pitch_ids,
            pitch_multihot=pitch_multihot,
            continuous=continuous,
            score_shared_raw=score_shared_raw,
            labels_continuous=labels_continuous,
            decoder_feedback_continuous=decoder_feedback_continuous,
            attention_mask=attention_mask,
            continuous_sampling_strategy=sampling_strategy,
        )
        pred = materialize_epr_prediction_eval(
            model.config,
            outputs.logits,
            sampling_strategy=sampling_strategy,
            score_shared_raw=score_shared_raw,
        )
        feedback_strategy_name = str(feedback_strategy).lower()
        if feedback_strategy_name == str(sampling_strategy).lower():
            feedback_pred = pred
        else:
            feedback_pred = materialize_epr_prediction_eval(
                model.config,
                outputs.logits,
                sampling_strategy=feedback_strategy,
                score_shared_raw=score_shared_raw,
            )
        loss_value = float(outputs.loss.detach().float().cpu()) if outputs.loss is not None else float("nan")
    return pred.detach().float().cpu(), feedback_pred.detach().float().cpu(), loss_value


def predict_kpass_batches(
    model,
    pitch_ids,
    pitch_multihot,
    continuous,
    score_shared_raw,
    labels_continuous,
    attention_mask,
    finite_ks,
    sampling_strategy,
    feedback_strategy,
):
    finite_ks = sorted({int(k) for k in finite_ks})
    if not finite_ks:
        return {}
    max_k = max(finite_ks)
    out = {}
    pred, loss_value = predict_tf_batch(
        model,
        pitch_ids,
        pitch_multihot,
        continuous,
        score_shared_raw,
        labels_continuous,
        attention_mask,
        sampling_strategy,
    )
    if 0 in finite_ks:
        out[0] = (pred, loss_value)
    feedback = pred.to(device=labels_continuous.device, dtype=labels_continuous.dtype)
    for k in range(1, max_k + 1):
        pred, feedback_pred, loss_value = predict_tf_feedback_batch(
            model,
            pitch_ids,
            pitch_multihot,
            continuous,
            score_shared_raw,
            labels_continuous,
            attention_mask,
            feedback,
            sampling_strategy,
            feedback_strategy,
        )
        if k in finite_ks:
            out[k] = (pred, loss_value)
        feedback = feedback_pred.to(device=labels_continuous.device, dtype=labels_continuous.dtype)
    return out


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
    feedback_mode,
    feedback_targets,
    collect_position_stats=False,
    continuous_temperature=1.0,
    head_stats=None,
):
    total_notes = len(pitch)
    pred_sum = None
    pred_count = torch.zeros(total_notes, 1, dtype=torch.float32)
    losses = []
    position_values = empty_position_values() if collect_position_stats else None
    for batch_start in range(0, len(windows), batch_size):
        batch_windows = windows[batch_start : batch_start + batch_size]
        pitch_tensors = []
        pitch_multihot_tensors = []
        score_tensors = []
        raw_tensors = []
        label_tensors = []
        lengths = []
        for start, end in batch_windows:
            pitch_tensors.append(torch.tensor(pitch_ids_from_pitch_values(pitch[start:end], config["pitch_pad_id"]), dtype=torch.long))
            pitch_multihot_tensors.append(torch.tensor(pitch_multihot_rows(config, score, start, end), dtype=torch.float32))
            score_tensors.append(torch.tensor(score_inputs[start:end], dtype=torch.float32))
            raw_tensors.append(torch.tensor(score_shared_raw[start:end], dtype=torch.float32))
            label_tensors.append(torch.tensor(labels[start:end], dtype=torch.float32))
            lengths.append(end - start)
        pitch_ids = pad_sequence(pitch_tensors, batch_first=True, padding_value=config["pitch_pad_id"]).to(device)
        pitch_multihot = pad_sequence(pitch_multihot_tensors, batch_first=True, padding_value=0.0).to(device)
        continuous = pad_sequence(score_tensors, batch_first=True, padding_value=0.0).to(device)
        score_raw = pad_sequence(raw_tensors, batch_first=True, padding_value=0.0).to(device)
        label_batch = pad_sequence(label_tensors, batch_first=True, padding_value=0.0).to(device)
        attention_mask = (pitch_ids != config["pitch_pad_id"]).long()
        pred, loss_value = predict_batch(
            model,
            pitch_ids,
            pitch_multihot,
            continuous,
            score_raw,
            label_batch,
            attention_mask,
            rollout_k,
            sampling_strategy,
            feedback_strategy,
            feedback_mode,
            feedback_targets,
            continuous_temperature=continuous_temperature,
            head_stats=head_stats,
        )
        if np.isfinite(loss_value):
            losses.append(loss_value)
        if pred_sum is None:
            pred_sum = torch.zeros(total_notes, pred.shape[-1], dtype=torch.float32)
        for idx, (start, end) in enumerate(batch_windows):
            length = lengths[idx]
            pred_sum[start:end] += pred[idx, :length]
            pred_count[start:end] += 1.0
            if position_values is not None:
                extend_position_values(position_values, pred[idx, :length].numpy(), label_tensors[idx].numpy())
    if pred_sum is None:
        raise ValueError("No windows were processed")
    return pred_sum / pred_count.clamp_min(1.0), finite_mean(losses), position_values


def predict_scores_for_kpass(
    model,
    device,
    config,
    score,
    pitch,
    score_inputs,
    score_shared_raw,
    labels,
    windows,
    finite_ks,
    batch_size,
    sampling_strategy,
    feedback_strategy,
    collect_position_stats=False,
):
    total_notes = len(pitch)
    finite_ks = sorted({int(k) for k in finite_ks})
    pred_sums = {}
    pred_counts = {k: torch.zeros(total_notes, 1, dtype=torch.float32) for k in finite_ks}
    losses = {k: [] for k in finite_ks}
    position_values = {k: empty_position_values() for k in finite_ks} if collect_position_stats else {}
    for batch_start in range(0, len(windows), batch_size):
        batch_windows = windows[batch_start : batch_start + batch_size]
        pitch_tensors = []
        pitch_multihot_tensors = []
        score_tensors = []
        raw_tensors = []
        label_tensors = []
        lengths = []
        for start, end in batch_windows:
            pitch_tensors.append(torch.tensor(pitch_ids_from_pitch_values(pitch[start:end], config["pitch_pad_id"]), dtype=torch.long))
            pitch_multihot_tensors.append(torch.tensor(pitch_multihot_rows(config, score, start, end), dtype=torch.float32))
            score_tensors.append(torch.tensor(score_inputs[start:end], dtype=torch.float32))
            raw_tensors.append(torch.tensor(score_shared_raw[start:end], dtype=torch.float32))
            label_tensors.append(torch.tensor(labels[start:end], dtype=torch.float32))
            lengths.append(end - start)
        pitch_ids = pad_sequence(pitch_tensors, batch_first=True, padding_value=config["pitch_pad_id"]).to(device)
        pitch_multihot = pad_sequence(pitch_multihot_tensors, batch_first=True, padding_value=0.0).to(device)
        continuous = pad_sequence(score_tensors, batch_first=True, padding_value=0.0).to(device)
        score_raw = pad_sequence(raw_tensors, batch_first=True, padding_value=0.0).to(device)
        label_batch = pad_sequence(label_tensors, batch_first=True, padding_value=0.0).to(device)
        attention_mask = (pitch_ids != config["pitch_pad_id"]).long()
        batch_preds = predict_kpass_batches(
            model,
            pitch_ids,
            pitch_multihot,
            continuous,
            score_raw,
            label_batch,
            attention_mask,
            finite_ks,
            sampling_strategy,
            feedback_strategy,
        )
        for k, (pred, loss_value) in batch_preds.items():
            if np.isfinite(loss_value):
                losses[k].append(loss_value)
            if k not in pred_sums:
                pred_sums[k] = torch.zeros(total_notes, pred.shape[-1], dtype=torch.float32)
            for idx, (start, end) in enumerate(batch_windows):
                length = lengths[idx]
                pred_sums[k][start:end] += pred[idx, :length]
                pred_counts[k][start:end] += 1.0
                if collect_position_stats:
                    extend_position_values(position_values[k], pred[idx, :length].numpy(), label_tensors[idx].numpy())
    if set(pred_sums) != set(finite_ks):
        missing = sorted(set(finite_ks) - set(pred_sums))
        raise ValueError(f"No windows were processed for k={missing}")
    return {
        k: (pred_sums[k] / pred_counts[k].clamp_min(1.0), finite_mean(losses[k]), position_values.get(k))
        for k in finite_ks
    }


def predict_work(model, device, config, item, args, rollout_ks):
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
        disable_musical_features=bool(config.get("disable_musical_features", False)),
        include_score_chord_offset=bool(config.get("include_score_chord_offset", False)),
    )
    windows = build_windows(len(pitch), int(config["block_notes"]), float(config["overlap_ratio"]))
    perfs = selected_perfs(work, item)
    by_k = {}
    kpass_values = [
        int(rollout_k)
        for rollout_k in rollout_ks
        if rollout_k is not None and bool(getattr(args, "fast_kpass", False))
    ]
    kpass_results = {}
    if kpass_values:
        for perf in perfs:
            labels = labels_for_perf(config, perf, score_shared_raw)
            seed = stable_seed(args.seed, item["score_source"], perf.get("performance_source"), "kpass", args.materialize_strategy)
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            kpass_results[perf.get("performance_source")] = predict_scores_for_kpass(
                model,
                device,
                config,
                score,
                pitch,
                score_inputs,
                score_shared_raw,
                labels,
                windows,
                kpass_values,
                int(args.batch_size_windows),
                args.materialize_strategy,
                args.feedback_strategy or args.materialize_strategy,
                collect_position_stats=args.window_position_stats,
            )
    for rollout_k in rollout_ks:
        label = k_label(rollout_k)
        pred_arrays = []
        gt_arrays = []
        pair_rows = []
        distributions = empty_distributions() if (args.plot_distributions or args.save_distribution_values) else None
        window_position = empty_position_values() if args.window_position_stats else None
        head_stats = empty_head_stats(config) if args.head_stats else None
        for perf in perfs:
            labels = labels_for_perf(config, perf, score_shared_raw)
            if rollout_k is not None and int(rollout_k) in kpass_results.get(perf.get("performance_source"), {}):
                pred_target, mean_loss, position_values = kpass_results[perf.get("performance_source")][int(rollout_k)]
            else:
                seed = stable_seed(args.seed, item["score_source"], perf.get("performance_source"), label, args.materialize_strategy)
                random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                pred_target, mean_loss, position_values = predict_score_for_k(
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
                    int(args.batch_size_windows),
                    args.materialize_strategy,
                    args.feedback_strategy or args.materialize_strategy,
                    args.feedback_mode,
                    args.feedback_targets,
                    collect_position_stats=args.window_position_stats,
                    continuous_temperature=args.continuous_temperature,
                    head_stats=head_stats,
                )
            score_raw_tensor = torch.tensor(score_shared_raw, dtype=torch.float32)
            target_tensor = torch.tensor(labels, dtype=torch.float32)
            pred_feedback = _target_predictions_to_feedback7(config, pred_target.float())
            target_feedback = _target_predictions_to_feedback7(config, target_tensor)
            pred_raw = _target7_to_raw7(score_raw_tensor, pred_feedback.float(), config=config).cpu().numpy()
            target_raw = _target7_to_raw7(score_raw_tensor, target_feedback.float(), config=config).cpu().numpy()
            pred_feedback_np = pred_feedback.float().cpu().numpy()
            gt_feedback_np = target_feedback.float().cpu().numpy()
            pred_target_np = pred_target.float().cpu().numpy()
            gt_target_np = target_tensor.float().cpu().numpy()
            pred_note_arrays = metric_arrays(pred_raw, pred_feedback_np)
            gt_note_arrays = metric_arrays(target_raw, gt_feedback_np)
            pred_arrays.append(pred_note_arrays)
            gt_arrays.append(gt_note_arrays)
            chord_mask = chord_mask_from_pitch_values(pitch)
            pred_offsets = target_offsets3(config, pred_target.float()).cpu().numpy()
            gt_offsets = target_offsets3(config, target_tensor).cpu().numpy()
            pred_offsets_chord = pred_offsets[chord_mask] if pred_offsets.shape[-1] >= 3 else np.zeros((0, 3))
            gt_offsets_chord = gt_offsets[chord_mask] if gt_offsets.shape[-1] >= 3 else np.zeros((0, 3))
            if distributions is not None:
                extend_distributions(
                    distributions,
                    config,
                    pred_raw,
                    target_raw,
                    pred_target_np,
                    gt_target_np,
                    chord_mask=chord_mask,
                )
            if position_values is not None:
                merge_position_values(window_position, position_values)
            pair_rows.append(
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
                    "onset_offset_wass": (
                        feature_wasserstein(pred_offsets_chord[:, 0] * 1000.0, gt_offsets_chord[:, 0] * 1000.0)
                        if len(pred_offsets_chord) and len(gt_offsets_chord)
                        else float("nan")
                    ),
                    "duration_offset_wass": (
                        feature_wasserstein(pred_offsets_chord[:, 1] * 1000.0, gt_offsets_chord[:, 1] * 1000.0)
                        if len(pred_offsets_chord) and len(gt_offsets_chord)
                        else float("nan")
                    ),
                    "velocity_offset_wass": (
                        feature_wasserstein(pred_offsets_chord[:, 2] * 127.0, gt_offsets_chord[:, 2] * 127.0)
                        if len(pred_offsets_chord) and len(gt_offsets_chord)
                        else float("nan")
                    ),
                    "ioi_ms_mean_shift": finite_mean_shift(pred_raw[:, 0], target_raw[:, 0]),
                    "duration_ms_mean_shift": finite_mean_shift(pred_raw[:, 1], target_raw[:, 1]),
                    "ioi_ms_std_ratio": finite_std_ratio(pred_raw[:, 0], target_raw[:, 0]),
                    "duration_ms_std_ratio": finite_std_ratio(pred_raw[:, 1], target_raw[:, 1]),
                    "ioi_logdev_mean_shift": finite_mean_shift(pred_feedback[:, 0], target_feedback[:, 0]),
                    "duration_logdev_mean_shift": finite_mean_shift(pred_feedback[:, 1], target_feedback[:, 1]),
                    "ioi_logdev_std_ratio": finite_std_ratio(pred_feedback[:, 0], target_feedback[:, 0]),
                    "duration_logdev_std_ratio": finite_std_ratio(pred_feedback[:, 1], target_feedback[:, 1]),
                    "ioi_logdev_pred_mean": finite_mean(pred_feedback[:, 0]),
                    "duration_logdev_pred_mean": finite_mean(pred_feedback[:, 1]),
                    "ioi_logdev_gt_mean": finite_mean(target_feedback[:, 0]),
                    "duration_logdev_gt_mean": finite_mean(target_feedback[:, 1]),
                    "ioi_logdev_pred_std": finite_std(pred_feedback[:, 0]),
                    "duration_logdev_pred_std": finite_std(pred_feedback[:, 1]),
                    "ioi_logdev_gt_std": finite_std(target_feedback[:, 0]),
                    "duration_logdev_gt_std": finite_std(target_feedback[:, 1]),
                    "onset_offset_ms_mean_shift": (
                        finite_mean_shift(pred_offsets_chord[:, 0] * 1000.0, gt_offsets_chord[:, 0] * 1000.0)
                        if len(pred_offsets_chord) and len(gt_offsets_chord)
                        else float("nan")
                    ),
                    "duration_offset_ms_mean_shift": (
                        finite_mean_shift(pred_offsets_chord[:, 1] * 1000.0, gt_offsets_chord[:, 1] * 1000.0)
                        if len(pred_offsets_chord) and len(gt_offsets_chord)
                        else float("nan")
                    ),
                    "velocity_offset_mean_shift": (
                        finite_mean_shift(pred_offsets_chord[:, 2] * 127.0, gt_offsets_chord[:, 2] * 127.0)
                        if len(pred_offsets_chord) and len(gt_offsets_chord)
                        else float("nan")
                    ),
                    "onset_offset_ms_std_ratio": (
                        finite_std_ratio(pred_offsets_chord[:, 0] * 1000.0, gt_offsets_chord[:, 0] * 1000.0)
                        if len(pred_offsets_chord) and len(gt_offsets_chord)
                        else float("nan")
                    ),
                    "duration_offset_ms_std_ratio": (
                        finite_std_ratio(pred_offsets_chord[:, 1] * 1000.0, gt_offsets_chord[:, 1] * 1000.0)
                        if len(pred_offsets_chord) and len(gt_offsets_chord)
                        else float("nan")
                    ),
                    "velocity_offset_std_ratio": (
                        finite_std_ratio(pred_offsets_chord[:, 2] * 127.0, gt_offsets_chord[:, 2] * 127.0)
                        if len(pred_offsets_chord) and len(gt_offsets_chord)
                        else float("nan")
                    ),
                }
            )
        by_k[label] = {
            "score_metrics": score_metrics(item["score_source"], pred_arrays, gt_arrays),
            "pairwise": pairwise_summary(pair_rows),
            "note_count": len(pitch),
            "num_windows": len(windows),
            "num_performances": len(perfs),
        }
        if distributions is not None:
            by_k[label]["distributions"] = distributions
        if window_position is not None:
            by_k[label]["window_position"] = window_position
        if head_stats is not None:
            by_k[label]["head_stats"] = finalize_head_stats(head_stats)
    return {"score_source": item["score_source"], "by_k": by_k}


def aggregate_items(items, rollout_ks):
    output = {}
    for rollout_k in rollout_ks:
        label = k_label(rollout_k)
        score_rows = [item["by_k"][label]["score_metrics"] for item in items]
        pair_rows = [item["by_k"][label]["pairwise"] for item in items]
        output[label] = {
            "num_scores": len(score_rows),
            "pairwise": aggregate_pairwise(pair_rows),
            "pp_wass": aggregate_score_metrics(score_rows, "pp_wass"),
            "pn_wass": aggregate_score_metrics(score_rows, "pn_wass"),
        }
    return output


def write_curve_csv(path, aggregate_by_k, rollout_ks):
    rows = []
    for rollout_k in rollout_ks:
        label = k_label(rollout_k)
        agg = aggregate_by_k[label]
        rows.append(
            {
                "k": label,
                "pairwise_ioi_wass": agg["pairwise"].get("ioi_wass"),
                "pairwise_duration_wass": agg["pairwise"].get("duration_wass"),
                "pairwise_velocity_wass": agg["pairwise"].get("velocity_wass"),
                "pairwise_pedal_wass": agg["pairwise"].get("pedal_wass"),
                "pairwise_onset_offset_wass": agg["pairwise"].get("onset_offset_wass"),
                "pairwise_duration_offset_wass": agg["pairwise"].get("duration_offset_wass"),
                "pairwise_velocity_offset_wass": agg["pairwise"].get("velocity_offset_wass"),
                "pairwise_ioi_ms_mean_shift": agg["pairwise"].get("ioi_ms_mean_shift"),
                "pairwise_duration_ms_mean_shift": agg["pairwise"].get("duration_ms_mean_shift"),
                "pairwise_ioi_ms_std_ratio": agg["pairwise"].get("ioi_ms_std_ratio"),
                "pairwise_duration_ms_std_ratio": agg["pairwise"].get("duration_ms_std_ratio"),
                "pairwise_onset_offset_ms_mean_shift": agg["pairwise"].get("onset_offset_ms_mean_shift"),
                "pairwise_duration_offset_ms_mean_shift": agg["pairwise"].get("duration_offset_ms_mean_shift"),
                "pairwise_velocity_offset_mean_shift": agg["pairwise"].get("velocity_offset_mean_shift"),
                "pairwise_onset_offset_ms_std_ratio": agg["pairwise"].get("onset_offset_ms_std_ratio"),
                "pairwise_duration_offset_ms_std_ratio": agg["pairwise"].get("duration_offset_ms_std_ratio"),
                "pairwise_velocity_offset_std_ratio": agg["pairwise"].get("velocity_offset_std_ratio"),
                "pairwise_ioi_logdev_mean_shift": agg["pairwise"].get("ioi_logdev_mean_shift"),
                "pairwise_duration_logdev_mean_shift": agg["pairwise"].get("duration_logdev_mean_shift"),
                "pairwise_ioi_logdev_std_ratio": agg["pairwise"].get("ioi_logdev_std_ratio"),
                "pairwise_duration_logdev_std_ratio": agg["pairwise"].get("duration_logdev_std_ratio"),
                "pairwise_ioi_logdev_pred_mean": agg["pairwise"].get("ioi_logdev_pred_mean"),
                "pairwise_duration_logdev_pred_mean": agg["pairwise"].get("duration_logdev_pred_mean"),
                "pairwise_ioi_logdev_gt_mean": agg["pairwise"].get("ioi_logdev_gt_mean"),
                "pairwise_duration_logdev_gt_mean": agg["pairwise"].get("duration_logdev_gt_mean"),
                "pairwise_ioi_logdev_pred_std": agg["pairwise"].get("ioi_logdev_pred_std"),
                "pairwise_duration_logdev_pred_std": agg["pairwise"].get("duration_logdev_pred_std"),
                "pairwise_ioi_logdev_gt_std": agg["pairwise"].get("ioi_logdev_gt_std"),
                "pairwise_duration_logdev_gt_std": agg["pairwise"].get("duration_logdev_gt_std"),
                "pp_ioi_wass": agg["pp_wass"].get("ioi_wass"),
                "pp_duration_wass": agg["pp_wass"].get("duration_wass"),
                "pp_velocity_wass": agg["pp_wass"].get("velocity_wass"),
                "pp_pedal_wass": agg["pp_wass"].get("pedal_wass"),
                "pn_ioi_wass": agg["pn_wass"].get("ioi_wass"),
                "pn_duration_wass": agg["pn_wass"].get("duration_wass"),
                "pn_velocity_wass": agg["pn_wass"].get("velocity_wass"),
                "pn_pedal_wass": agg["pn_wass"].get("pedal_wass"),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_head_stats_csv(path, rows):
    if not rows:
        return
    fieldnames = ["k", "feature", "n"]
    extra = sorted({key for row in rows for key in row.keys() if key not in set(fieldnames)})
    fieldnames.extend(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def distribution_group_label(k):
    return "full AR" if k == "full" else f"k={k}"


def collect_distribution_groups(items, rollout_ks, domain, feature):
    labels = [k_label(value) for value in rollout_ks]
    groups = {}
    if labels:
        gt_values = []
        for item in items:
            dist = item["by_k"][labels[0]].get("distributions", {})
            gt_values.extend(dist.get(domain, {}).get("gt", {}).get(feature, []))
        groups["GT"] = np.asarray(gt_values, dtype=np.float64)
    for label in labels:
        pred_values = []
        for item in items:
            dist = item["by_k"][label].get("distributions", {})
            pred_values.extend(dist.get(domain, {}).get("pred", {}).get(feature, []))
        groups[distribution_group_label(label)] = np.asarray(pred_values, dtype=np.float64)
    return groups


def finite_values(values):
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def clipped_range(groups, low=0.5, high=99.5):
    pooled = np.concatenate([finite_values(values) for values in groups.values() if len(values)])
    if len(pooled) == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(pooled, [low, high])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.nanmin(pooled))
        hi = float(np.nanmax(pooled))
    if lo == hi:
        hi = lo + 1.0
    return float(lo), float(hi)


def write_distribution_stats(path, items, rollout_ks):
    rows = []
    for domain, keys in (("raw", RAW_DISTRIBUTION_KEYS), ("target", TARGET_DISTRIBUTION_KEYS)):
        for feature, _ in keys:
            groups = collect_distribution_groups(items, rollout_ks, domain, feature)
            for group, values in groups.items():
                values = finite_values(values)
                if len(values) == 0:
                    rows.append({"domain": domain, "feature": feature, "group": group, "n": 0})
                    continue
                rows.append(
                    {
                        "domain": domain,
                        "feature": feature,
                        "group": group,
                        "n": int(len(values)),
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "p01": float(np.percentile(values, 1)),
                        "p50": float(np.percentile(values, 50)),
                        "p99": float(np.percentile(values, 99)),
                    }
                )
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["domain", "feature", "group", "n", "mean", "std", "p01", "p50", "p99"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_distribution_panel(path, items, rollout_ks, domain, keys):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    continuous_keys = keys[:3]
    pedal_keys = keys[3:]
    labels = ["GT", *[distribution_group_label(k_label(value)) for value in rollout_ks]]
    base_palette = [
        "#111111",
        "#2f6fbb",
        "#d9822b",
        "#756bb1",
        "#31a354",
        "#e6550d",
        "#3182bd",
        "#b83232",
    ]
    colors = {label: base_palette[idx % len(base_palette)] for idx, label in enumerate(labels)}
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    axes = axes.flatten()
    for axis, (feature, title) in zip(axes[:3], continuous_keys):
        groups = collect_distribution_groups(items, rollout_ks, domain, feature)
        lo, hi = clipped_range(groups)
        bins = np.linspace(lo, hi, 90)
        for group in labels:
            values = groups.get(group, [])
            values = finite_values(values)
            values = values[(values >= lo) & (values <= hi)]
            if len(values) == 0:
                continue
            hist, edges = np.histogram(values, bins=bins, density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            linewidth = 2.4 if group in {"GT", "full AR"} else 1.7
            alpha = 0.95 if group in {"GT", "full AR"} else 0.82
            axis.plot(centers, hist, label=group, linewidth=linewidth, alpha=alpha, color=colors.get(group))
        axis.set_title(title)
        axis.set_ylabel("Density")
        axis.grid(alpha=0.18)
    axis = axes[3]
    x = np.arange(len(pedal_keys), dtype=np.float64)
    width = min(0.12, 0.8 / max(1, len(labels)))
    labels = [label for label in labels if label in collect_distribution_groups(items, rollout_ks, domain, pedal_keys[0][0])]
    offsets = (np.arange(len(labels)) - (len(labels) - 1) / 2.0) * width
    for offset, group in zip(offsets, labels):
        means = []
        for feature, _ in pedal_keys:
            groups = collect_distribution_groups(items, rollout_ks, domain, feature)
            means.append(float(np.mean(finite_values(groups.get(group, [])))) if len(groups.get(group, [])) else float("nan"))
        axis.bar(x + offset, means, width=width, label=group, color=colors.get(group))
    axis.set_title("Pedal binary mean")
    axis.set_xticks(x)
    axis.set_xticklabels([title for _, title in pedal_keys], rotation=20, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.grid(axis="y", alpha=0.18)
    axes[4].axis("off")
    axes[5].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False)
    fig.suptitle(f"{domain.capitalize()} distributions: GT vs rollout outputs", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_distribution_plots(output_dir, items, rollout_ks):
    write_distribution_stats(output_dir / "distribution_stats.csv", items, rollout_ks)
    plot_distribution_panel(output_dir / "distribution_raw.png", items, rollout_ks, "raw", RAW_DISTRIBUTION_KEYS)
    plot_distribution_panel(output_dir / "distribution_target.png", items, rollout_ks, "target", TARGET_DISTRIBUTION_KEYS)


def collect_position_feature(items, k, scheme, bucket, feature):
    pred_values = []
    gt_values = []
    for item in items:
        values = (
            item.get("by_k", {})
            .get(k, {})
            .get("window_position", {})
            .get(scheme, {})
            .get(bucket, {})
            .get(feature, {})
        )
        pred_values.extend(values.get("pred", []))
        gt_values.extend(values.get("gt", []))
    return np.asarray(pred_values, dtype=np.float64), np.asarray(gt_values, dtype=np.float64)


def finite_pairs(pred_values, gt_values):
    pred_values = np.asarray(pred_values, dtype=np.float64)
    gt_values = np.asarray(gt_values, dtype=np.float64)
    mask = np.isfinite(pred_values) & np.isfinite(gt_values)
    return pred_values[mask], gt_values[mask]


def write_window_position_stats(path, items, rollout_ks):
    rows = []
    for rollout_k in rollout_ks:
        label = k_label(rollout_k)
        for scheme, buckets in (("half", ("front", "back")), ("quarter", ("q1", "q2", "q3", "q4"))):
            for bucket in buckets:
                for feature, _ in POSITION_FEATURES:
                    pred_values, gt_values = collect_position_feature(items, label, scheme, bucket, feature)
                    pred_values, gt_values = finite_pairs(pred_values, gt_values)
                    diff = pred_values - gt_values
                    rows.append(
                        {
                            "k": label,
                            "scheme": scheme,
                            "bucket": bucket,
                            "feature": feature,
                            "n": int(len(pred_values)),
                            "pred_mean": float(np.mean(pred_values)) if len(pred_values) else float("nan"),
                            "gt_mean": float(np.mean(gt_values)) if len(gt_values) else float("nan"),
                            "mean_shift": float(np.mean(diff)) if len(diff) else float("nan"),
                            "mae": float(np.mean(np.abs(diff))) if len(diff) else float("nan"),
                            "wass": feature_wasserstein(pred_values, gt_values),
                        }
                    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["k", "scheme", "bucket", "feature", "n", "pred_mean", "gt_mean", "mean_shift", "mae", "wass"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def strip_distributions(items):
    for item in items:
        for by_k in item.get("by_k", {}).values():
            by_k.pop("distributions", None)


def strip_window_positions(items):
    for item in items:
        for by_k in item.get("by_k", {}).values():
            by_k.pop("window_position", None)


def worker_loop(worker_idx, args, config, rollout_ks, job_queue, result_queue):
    random.seed(args.seed + worker_idx)
    torch.manual_seed(args.seed + worker_idx)
    device = select_worker_device(args.device, worker_idx)
    print(f"worker {worker_idx} device={device}", flush=True)
    model = create_model(config)
    model.to(device)
    model.eval()
    while True:
        job = job_queue.get()
        if job is None:
            break
        job_idx, item = job
        try:
            result_queue.put((job_idx, predict_work(model, device, config, item, args, rollout_ks), None))
        except Exception as exc:  # noqa: BLE001
            result_queue.put((job_idx, None, repr(exc)))


def run_pool(args, config, manifest, rollout_ks):
    ctx = mp.get_context("spawn")
    job_queue = ctx.Queue()
    result_queue = ctx.Queue()
    workers = [
        ctx.Process(target=worker_loop, args=(idx, args, config, rollout_ks, job_queue, result_queue))
        for idx in range(int(args.num_workers))
    ]
    for worker in workers:
        worker.start()
    for job_idx, item in enumerate(manifest):
        job_queue.put((job_idx, item))
    for _ in workers:
        job_queue.put(None)
    by_idx = {}
    with tqdm(total=len(manifest), desc="rollout current") as progress:
        for _ in range(len(manifest)):
            job_idx, result, error = result_queue.get()
            if error is not None:
                for worker in workers:
                    worker.terminate()
                raise RuntimeError(f"worker failed on job {job_idx}: {error}")
            by_idx[job_idx] = result
            progress.update(1)
    for worker in workers:
        worker.join()
        if worker.exitcode != 0:
            raise RuntimeError(f"worker {worker.pid} exited with {worker.exitcode}")
    return [by_idx[idx] for idx in range(len(manifest))]


def run_single(args, config, manifest, rollout_ks):
    device = select_device(args.device)
    model = create_model(config)
    model.to(device)
    model.eval()
    return [predict_work(model, device, config, item, args, rollout_ks) for item in tqdm(manifest, desc="rollout current")]


def main():
    args = parse_args()
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
        prepared_sidecar_tag=config.get("prepared_sidecar_tag"),
    )
    manifest = filter_manifest(manifest, load_score_source_filter(args))
    items = run_pool(args, config, manifest, rollout_ks) if int(args.num_workers) > 1 else run_single(args, config, manifest, rollout_ks)
    aggregate_by_k = aggregate_items(items, rollout_ks)
    head_stats_rows = aggregate_head_stats(items, rollout_ks) if args.head_stats else []
    if args.plot_distributions:
        write_distribution_plots(args.output_dir, items, rollout_ks)
    if args.window_position_stats:
        write_window_position_stats(args.output_dir / "window_position_stats.csv", items, rollout_ks)
    if args.head_stats:
        write_head_stats_csv(args.output_dir / "timing_head_stats.csv", head_stats_rows)
    if args.plot_distributions and not args.save_distribution_values:
        strip_distributions(items)
    if args.window_position_stats:
        strip_window_positions(items)
    summary = {
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "performance_dataset": args.performance_dataset,
        "materialize_strategy": args.materialize_strategy,
        "feedback_strategy": args.feedback_strategy or args.materialize_strategy,
        "feedback_mode": args.feedback_mode,
        "feedback_targets": args.feedback_targets,
        "continuous_temperature": args.continuous_temperature,
        "rollout_ks": [k_label(value) for value in rollout_ks],
        "num_scores": len(items),
        "aggregate_by_k": aggregate_by_k,
        "head_stats": head_stats_rows,
        "items": items,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_curve_csv(args.output_dir / "curve.csv", aggregate_by_k, rollout_ks)
    print(json.dumps({key: value for key, value in summary.items() if key != "items"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
