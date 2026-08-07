import argparse
import csv
import itertools
import json
import math
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wasserstein_distance
from torch.utils.data import DataLoader, Subset

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_process.work_manifest import build_work_manifest
from src.model.integrated_pianoformer import (
    _compute_integrated_loss_components,
    _materialize_epr_prediction,
    _target7_to_raw7,
)
from src.train.train_inr import (
    NodeSFTDataCollator,
    PianoCoReNodeSFTDataset,
    apply_default_asap_sidecar_tag,
    create_model,
    default_epr_output_dim,
    enforce_asap_processed_config,
    filter_resume_state_dict,
    infer_input_feature_mode,
    load_torch_state_dict,
    split_manifest_windows_for_eval,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Probe validation-window rollout metrics against test PN/PP metrics.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-label", type=str, default="cinr_baseline")
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=str, default="8,10,12,14,16,18,20,22,24")
    parser.add_argument("--test-mode", type=str, default="full_test_s1")
    parser.add_argument("--num-examples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--worker-window-batch", type=int, default=2)
    parser.add_argument("--rollout-ks", type=str, default="0,1,2,4")
    parser.add_argument("--window-ar-strides", type=str, default="")
    parser.add_argument("--window-wass-samples", type=int, default=0)
    parser.add_argument("--window-wass-stride", type=int, default=1)
    parser.add_argument("--window-wass-pedal-support", choices=["raw", "binary"], default="binary")
    parser.add_argument("--pedal-binary-threshold", type=float, default=64.0)
    parser.add_argument("--sampling-strategy", type=str, default="sample")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--human-baseline-only", action="store_true")
    parser.add_argument("--human-baseline-output", type=Path, default=None)
    parser.add_argument("--require-human-baseline", action="store_true")
    return parser.parse_args()


def finite(value):
    try:
        value = float(value)
    except Exception:
        return math.nan
    return value if math.isfinite(value) else math.nan


def pearson(xs, ys):
    pairs = [(finite(x), finite(y)) for x, y in zip(xs, ys)]
    pairs = [(x, y) for x, y in pairs if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return math.nan
    x = np.asarray([p[0] for p in pairs], dtype=np.float64)
    y = np.asarray([p[1] for p in pairs], dtype=np.float64)
    if np.std(x) <= 0.0 or np.std(y) <= 0.0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def ranks(values):
    output = [math.nan] * len(values)
    finite_items = [(idx, finite(value)) for idx, value in enumerate(values)]
    finite_items = [(idx, value) for idx, value in finite_items if math.isfinite(value)]
    for rank, (idx, _value) in enumerate(sorted(finite_items, key=lambda item: item[1]), start=1):
        output[idx] = float(rank)
    return output


def normalize_train_config(config):
    enforce_asap_processed_config(config)
    apply_default_asap_sidecar_tag(config)
    task_type = str(config.get("task_type", "epr")).lower()
    if task_type != "epr":
        raise ValueError("Only EPR configs are supported")
    config["input_feature_mode"] = infer_input_feature_mode(config)
    base_output_dim = default_epr_output_dim(
        str(config.get("epr_timing_target", "floor_log_deviation")).lower(),
        config.get("pedal_representation", "binary_4"),
    )
    config["legacy_dual_timing_head"] = False
    config["continuous_dim"] = base_output_dim
    config["output_continuous_dim"] = base_output_dim
    return config


def build_eval_dataset(config, num_examples, seed, require_human_baseline=False):
    fixed_scheme = config.get("fixed_window_split_scheme")
    if fixed_scheme:
        eval_manifest = build_work_manifest(
            metadata_path=config["metadata_path"],
            refined_dir=config["refined_dir"],
            split=config.get("fixed_window_base_split", "train"),
            block_notes=config["block_notes"],
            overlap_ratio=config["overlap_ratio"],
            min_notes=config["min_notes"],
            max_works=config.get("max_eval_works"),
            include_all_performance_dataset=config.get("eval_include_all_performance_dataset"),
            max_non_asap_performances_per_work=config.get("max_eval_non_asap_performances_per_work"),
            selection_seed=config.get("seed", seed),
            skip_work_paths=config.get("skip_work_paths"),
            performance_dataset=config.get("eval_performance_dataset"),
            exclude_performance_dataset=config.get("eval_exclude_performance_dataset"),
            window_split_scheme=fixed_scheme,
            window_split_name=config.get("fixed_window_eval_split_name", "valid"),
            window_split_summary_path=config.get("fixed_window_split_summary_path"),
            prepared_sidecar_tag=config.get("prepared_sidecar_tag"),
        )
        eval_split = config.get("fixed_window_eval_split_name", "valid")
    else:
        train_manifest = build_work_manifest(
            metadata_path=config["metadata_path"],
            refined_dir=config["refined_dir"],
            split="train",
            block_notes=config["block_notes"],
            overlap_ratio=config["overlap_ratio"],
            min_notes=config["min_notes"],
            max_works=config.get("max_train_works"),
            skip_work_paths=config.get("skip_work_paths"),
            performance_dataset=config.get("train_performance_dataset"),
            exclude_performance_dataset=config.get("train_exclude_performance_dataset"),
            prepared_sidecar_tag=config.get("prepared_sidecar_tag"),
        )
        fraction = config.get("eval_from_train_fraction", 0.03)
        _train_manifest, eval_manifest = split_manifest_windows_for_eval(
            train_manifest,
            fraction=fraction,
            seed=config.get("seed", seed),
        )
        eval_split = "train"

    dataset = PianoCoReNodeSFTDataset(
        eval_manifest,
        split=eval_split,
        task_type=config.get("task_type", "epr").lower(),
        input_feature_mode=config["input_feature_mode"],
        shuffle=False,
        seed=config.get("seed", seed),
        max_performances_per_work=config.get("max_eval_performances_per_work"),
        max_windows_per_work=config.get("max_eval_windows_per_work"),
        cache_size=config.get("node_cache_size", 16),
        timing_normalization=config.get("timing_input_normalization", "linear_5000"),
        max_time_ms=config.get("max_time_ms", 10000.0),
        epr_timing_bins=config.get("epr_timing_bins", 5000),
        epr_value_bins=config.get("epr_value_bins", 128),
        pedal_representation=config.get("pedal_representation", "binary_4"),
        musical_feature_mode=str(config.get("musical_feature_mode", "musical4slot")).lower(),
        score_note_schema=config.get("score_note_input_schema", "integrated"),
        epr_timing_target=config.get("epr_timing_target", "floor_log_deviation"),
        disable_musical_features=config.get("disable_musical_features", False),
        musical_feature_transform=config.get("musical_feature_transform", "none"),
        musical_random_seed=config.get("musical_random_seed", config.get("seed", seed)),
        use_timing_scale_bit=config.get("use_timing_scale_bit", False),
        timing_control_mode=config.get("timing_control_mode"),
        timing_log_scale=config.get("timing_log_scale", 50.0),
        precompute_items=False,
        use_prepared_sidecar=config.get("use_prepared_sidecar", True),
        prepared_sidecar_tag=config.get("prepared_sidecar_tag"),
        use_style_tokens=config.get("use_style_tokens", False),
        composer_vocab=config.get("style_composer_vocab"),
        source_vocab=config.get("style_source_vocab"),
        perf_style_stats_mode=config.get("perf_style_stats_mode", "prefix"),
        legacy_dual_timing_head=config.get("legacy_dual_timing_head", False),
    )
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    if require_human_baseline:
        indices = [
            index
            for index in indices
            if int(_dataset_example_location(dataset, index)[0].get("effective_performances", 0)) >= 2
        ]
    return Subset(dataset, indices[: min(num_examples, len(indices))])


def _dataset_example_location(dataset, index):
    item_idx = int(np.searchsorted(dataset.cumulative_sizes, int(index), side="right"))
    previous = 0 if item_idx == 0 else int(dataset.cumulative_sizes[item_idx - 1])
    local_index = int(index) - previous
    item = dataset.items[item_idx]
    window_count = len(item["windows"])
    performance_slot = local_index // window_count
    window_slot = local_index % window_count
    return item, performance_slot, item["windows"][window_slot]


def evaluate_validation_human_baseline(dataset_subset, config, num_samples, pedal_support, threshold):
    """Matched human-human baseline for the exact selected validation examples.

    Each model example uses ``num_samples`` generated performances against one
    target performance.  Here the target stays fixed and every combination of
    ``num_samples`` other human performances is used as the pseudo-prediction
    set, so PP/PN support and per-window weighting match the model metric.
    """
    base = dataset_subset.dataset
    example_rows = []
    skipped = []
    for selected_position, dataset_index in enumerate(dataset_subset.indices):
        item, performance_slot, (start, end) = _dataset_example_location(base, dataset_index)
        prepared = base._load_or_prepare_work(item["path"])
        performances = base._selected_performances(prepared, item)
        if not performances:
            skipped.append({"selected_position": selected_position, "reason": "no_performances"})
            continue
        target_index = int(performance_slot) % len(performances)
        target = performances[target_index]
        other_indices = [idx for idx in range(len(performances)) if idx != target_index]
        group_size = int(num_samples)
        if not other_indices:
            skipped.append(
                {
                    "selected_position": selected_position,
                    "reason": "insufficient_other_performances",
                    "num_performances": len(performances),
                    "required_other_performances": 1,
                }
            )
            continue

        score_shared = torch.as_tensor(
            [row[:3] for row in prepared["score"]["score_raw"][start:end]], dtype=torch.float32
        ).unsqueeze(0)

        def performance_arrays(perf):
            labels, _ = base._performance_labels(prepared, perf)
            labels = torch.as_tensor(labels[start:end], dtype=torch.float32).unsqueeze(0)
            raw = _target7_to_raw7(score_shared, labels, config=config)[0].detach().cpu().numpy()
            return raw_window_arrays(raw, np.ones(len(raw), dtype=bool), pedal_support, threshold)

        target_arrays = [performance_arrays(target)]
        human_arrays = {idx: performance_arrays(performances[idx]) for idx in other_indices}
        comparison_rows = []
        groups = list(itertools.combinations(other_indices, group_size))
        used_repeated_single_human = False
        if not groups and len(other_indices) == 1:
            # Some validation windows have only two human performances total.
            # Repeating the sole held-out human preserves the requested sample
            # cardinality and is Wasserstein-equivalent to a 1-vs-1 comparison.
            groups = [tuple(other_indices * group_size)]
            used_repeated_single_human = True
        for group in groups:
            pred_arrays = [human_arrays[idx] for idx in group]
            comparison_rows.append(
                {
                    "pp_wass": pp_wass_from_arrays(pred_arrays, target_arrays),
                    "pn_wass": pn_wass_from_arrays(pred_arrays, target_arrays),
                }
            )
        example_rows.append(
            {
                "selected_position": selected_position,
                "dataset_index": int(dataset_index),
                "work_path": str(item["path"]),
                "window": [int(start), int(end)],
                "target_performance": base._performance_cache_key(target),
                "num_performances": len(performances),
                "num_comparisons": len(comparison_rows),
                "used_repeated_single_human": used_repeated_single_human,
                "pp_wass": {
                    feature: finite_mean([row["pp_wass"][feature] for row in comparison_rows])
                    for feature in ("ioi", "duration", "velocity", "pedal")
                },
                "pn_wass": {
                    feature: finite_mean([row["pn_wass"][feature] for row in comparison_rows])
                    for feature in ("ioi", "duration", "velocity", "pedal")
                },
            }
        )

    aggregate = {}
    for section in ("pp_wass", "pn_wass"):
        aggregate[section] = {
            feature: finite_mean([row[section][feature] for row in example_rows])
            for feature in ("ioi", "duration", "velocity", "pedal")
        }
    return {
        "protocol": "matched_validation_window_human_holdout",
        "num_pseudo_samples": int(num_samples),
        "num_selected_examples": len(dataset_subset),
        "num_usable_examples": len(example_rows),
        "num_skipped_examples": len(skipped),
        "aggregate": aggregate,
        "examples": example_rows,
        "skipped": skipped,
    }


def move_batch(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def masked_stats(values, labels, mask):
    mask = mask.bool()
    out = {}
    for name, col in (("ioi", 0), ("duration", 1), ("velocity", 2), ("pedal0", 3)):
        if col >= values.shape[-1]:
            continue
        valid = mask
        pred_col = values[..., col].detach().float()[valid]
        label_col = labels[..., col].detach().float()[valid]
        if pred_col.numel() == 0:
            continue
        out[f"{name}_pred_mean"] = float(pred_col.mean().cpu())
        out[f"{name}_label_mean"] = float(label_col.mean().cpu())
        out[f"{name}_mean_delta"] = float((pred_col.mean() - label_col.mean()).cpu())
        out[f"{name}_pred_std"] = float(pred_col.std(unbiased=False).cpu())
        out[f"{name}_label_std"] = float(label_col.std(unbiased=False).cpu())
        out[f"{name}_std_delta"] = float((pred_col.std(unbiased=False) - label_col.std(unbiased=False)).cpu())
    return out


def evaluate_checkpoint(model, config, loader, device, rollout_ks, sampling_strategy):
    model.eval()
    accum = {
        k: {
            "weight": 0.0,
            "loss": 0.0,
            "components": {},
            "stats_sums": {},
            "stats_count": 0,
            "wass_sums": {},
            "wass_count": 0,
        }
        for k in rollout_ks
    }
    started = time.time()
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            labels = batch["labels_continuous"]
            mask = batch["attention_mask"].bool()
            max_k = max(rollout_ks)
            feedback = None
            pred_for_pass = {}
            outputs_for_pass = {}
            for pass_idx in range(max_k + 1):
                outputs = model(
                    pitch_ids=batch["pitch_ids"],
                    continuous=batch["continuous"],
                    score_shared_raw=batch["score_shared_raw"],
                    labels_continuous=labels,
                    decoder_feedback_continuous=feedback,
                    labels_epr_bins=batch.get("labels_epr_bins"),
                    label_mask=batch.get("label_mask"),
                    label_valid_mask=batch.get("label_valid_mask"),
                    attention_mask=batch["attention_mask"],
                    continuous_sampling_strategy=sampling_strategy,
                )
                pred = _materialize_epr_prediction(
                    model.config,
                    outputs.logits,
                    sampling_strategy=sampling_strategy,
                    score_shared_raw=batch["score_shared_raw"],
                )
                pred_for_pass[pass_idx] = pred.detach()
                outputs_for_pass[pass_idx] = outputs
                if pass_idx < max_k:
                    feedback = pred.detach()
            batch_weight = float(mask.sum().detach().cpu().item())
            gt_raw = _target7_to_raw7(batch["score_shared_raw"], labels, config=config).detach().cpu().numpy()
            mask_np = mask.detach().cpu().numpy()
            for k in rollout_ks:
                outputs = outputs_for_pass[k]
                pred = pred_for_pass[k]
                slot = accum[k]
                slot["weight"] += batch_weight
                slot["loss"] += float(outputs.loss.detach().float().cpu().item()) * batch_weight
                comps = _compute_integrated_loss_components(
                    model.config,
                    outputs.logits,
                    labels,
                    batch["attention_mask"],
                    labels_epr_bins=batch.get("labels_epr_bins"),
                    score_shared_raw=batch["score_shared_raw"],
                    label_valid_mask=batch.get("label_valid_mask"),
                )
                for name, value in comps.items():
                    if value.ndim != 0:
                        continue
                    slot["components"][name] = slot["components"].get(name, 0.0) + float(value.detach().cpu()) * batch_weight
                stats = masked_stats(pred, labels, mask)
                for name, value in stats.items():
                    slot["stats_sums"][name] = slot["stats_sums"].get(name, 0.0) + float(value)
                slot["stats_count"] += 1
                pred_raw = _target7_to_raw7(
                    batch["score_shared_raw"], pred, config=config
                ).detach().cpu().numpy()
                for row_idx in range(mask_np.shape[0]):
                    pred_arrays = [
                        raw_window_arrays(pred_raw[row_idx], mask_np[row_idx], "binary", 64.0)
                    ]
                    gt_arrays = [raw_window_arrays(gt_raw[row_idx], mask_np[row_idx], "binary", 64.0)]
                    metrics = {
                        "pp": pp_wass_from_arrays(pred_arrays, gt_arrays),
                        "pn": pn_wass_from_arrays(pred_arrays, gt_arrays),
                    }
                    for section, values in metrics.items():
                        for feature, value in values.items():
                            key = f"{section}_wass_{feature}"
                            slot["wass_sums"][key] = slot["wass_sums"].get(key, 0.0) + float(value)
                    slot["wass_count"] += 1
    rows = {}
    for k, slot in accum.items():
        weight = max(slot["weight"], 1.0)
        prefix = f"k{k}"
        row = {f"{prefix}_loss": slot["loss"] / weight}
        for name, value in slot["components"].items():
            row[f"{prefix}_{name}_loss"] = value / weight
        count = max(slot["stats_count"], 1)
        for name, value in slot["stats_sums"].items():
            row[f"{prefix}_{name}"] = value / count
        wass_count = max(slot["wass_count"], 1)
        for name, value in slot["wass_sums"].items():
            row[f"{prefix}_{name}"] = value / wass_count
        rows.update(row)
    rows["probe_seconds"] = time.time() - started
    return rows


def evaluate_window_ar_checkpoint(model, loader, device, strides, sampling_strategy):
    model.eval()
    accum = {
        stride: {"weight": 0.0, "loss": 0.0, "stats_sums": {}, "stats_count": 0}
        for stride in strides
    }
    started = time.time()
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            labels = batch["labels_continuous"]
            mask = batch["attention_mask"].bool()
            valid_lengths = batch["attention_mask"].sum(dim=1).long()
            max_len = int(valid_lengths.max().detach().cpu().item())
            batch_weight = float(mask.sum().detach().cpu().item())
            for stride in strides:
                feedback = labels.clone()
                for start in range(0, max_len, int(stride)):
                    outputs = model(
                        pitch_ids=batch["pitch_ids"],
                        continuous=batch["continuous"],
                        score_shared_raw=batch["score_shared_raw"],
                        labels_continuous=labels,
                        decoder_feedback_continuous=feedback,
                        labels_epr_bins=batch.get("labels_epr_bins"),
                        label_mask=batch.get("label_mask"),
                        label_valid_mask=batch.get("label_valid_mask"),
                        attention_mask=batch["attention_mask"],
                        continuous_sampling_strategy=sampling_strategy,
                    )
                    pred = _materialize_epr_prediction(
                        model.config,
                        outputs.logits,
                        sampling_strategy=sampling_strategy,
                        score_shared_raw=batch["score_shared_raw"],
                    ).detach()
                    end = min(start + int(stride), max_len)
                    row_active = valid_lengths > start
                    if row_active.any():
                        feedback[row_active, start:end, :] = pred[row_active, start:end, :]
                final_outputs = model(
                    pitch_ids=batch["pitch_ids"],
                    continuous=batch["continuous"],
                    score_shared_raw=batch["score_shared_raw"],
                    labels_continuous=labels,
                    decoder_feedback_continuous=feedback,
                    labels_epr_bins=batch.get("labels_epr_bins"),
                    label_mask=batch.get("label_mask"),
                    label_valid_mask=batch.get("label_valid_mask"),
                    attention_mask=batch["attention_mask"],
                    continuous_sampling_strategy=sampling_strategy,
                )
                slot = accum[stride]
                slot["weight"] += batch_weight
                slot["loss"] += float(final_outputs.loss.detach().float().cpu().item()) * batch_weight
                stats = masked_stats(feedback, labels, mask)
                abs_error = (feedback.detach().float() - labels.detach().float()).abs()
                for name, col in (("ioi", 0), ("duration", 1), ("velocity", 2), ("pedal0", 3)):
                    if col >= abs_error.shape[-1]:
                        continue
                    values = abs_error[..., col][mask]
                    if values.numel() > 0:
                        stats[f"{name}_mae"] = float(values.mean().cpu())
                for name, value in stats.items():
                    slot["stats_sums"][name] = slot["stats_sums"].get(name, 0.0) + float(value)
                slot["stats_count"] += 1
    rows = {}
    for stride, slot in accum.items():
        prefix = f"war_s{stride}"
        weight = max(slot["weight"], 1.0)
        rows[f"{prefix}_loss"] = slot["loss"] / weight
        count = max(slot["stats_count"], 1)
        for name, value in slot["stats_sums"].items():
            rows[f"{prefix}_{name}"] = value / count
    rows["window_ar_seconds"] = time.time() - started
    return rows


def finite_mean(values):
    values = np.asarray([finite(value) for value in values], dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else math.nan


def feature_wass(pred_values, gt_values):
    pred = np.asarray(pred_values, dtype=np.float64)
    gt = np.asarray(gt_values, dtype=np.float64)
    pred = pred[np.isfinite(pred)]
    gt = gt[np.isfinite(gt)]
    if len(pred) == 0 or len(gt) == 0:
        return math.nan
    return float(wasserstein_distance(pred, gt))


def pedal_values(raw_rows, pedal_support, threshold):
    pedal = np.asarray(raw_rows[..., 3:7], dtype=np.float64)
    if pedal_support == "binary":
        return (pedal >= float(threshold)).astype(np.float64)
    return pedal


def raw_window_arrays(raw_rows, mask, pedal_support, threshold):
    valid = np.asarray(mask, dtype=bool)
    rows = np.asarray(raw_rows, dtype=np.float64)[valid]
    pedal = pedal_values(rows, pedal_support, threshold)
    return {
        "ioi": rows[:, 0],
        "duration": rows[:, 1],
        "velocity": rows[:, 2],
        "pedal_0": pedal[:, 0],
        "pedal_25": pedal[:, 1],
        "pedal_50": pedal[:, 2],
        "pedal_75": pedal[:, 3],
    }


def pp_wass_from_arrays(pred_arrays, gt_arrays):
    output = {}
    for name in ("ioi", "duration", "velocity"):
        pred_pool = np.concatenate([item[name] for item in pred_arrays]) if pred_arrays else np.asarray([])
        gt_pool = np.concatenate([item[name] for item in gt_arrays]) if gt_arrays else np.asarray([])
        output[name] = feature_wass(pred_pool, gt_pool)
    output["pedal"] = finite_mean(
        [
            feature_wass(
                np.concatenate([item[name] for item in pred_arrays]) if pred_arrays else np.asarray([]),
                np.concatenate([item[name] for item in gt_arrays]) if gt_arrays else np.asarray([]),
            )
            for name in ("pedal_0", "pedal_25", "pedal_50", "pedal_75")
        ]
    )
    return output


def pn_wass_from_arrays(pred_arrays, gt_arrays):
    output = {}
    all_arrays = pred_arrays + gt_arrays
    for name in ("ioi", "duration", "velocity"):
        usable = min((len(item[name]) for item in all_arrays), default=0)
        output[name] = finite_mean(
            [
                feature_wass(
                    [item[name][note_idx] for item in pred_arrays],
                    [item[name][note_idx] for item in gt_arrays],
                )
                for note_idx in range(usable)
            ]
        )
    output["pedal"] = finite_mean(
        [
            finite_mean(
                [
                    feature_wass(
                        [item[name][note_idx] for item in pred_arrays],
                        [item[name][note_idx] for item in gt_arrays],
                    )
                    for note_idx in range(min((len(item[name]) for item in all_arrays), default=0))
                ]
            )
            for name in ("pedal_0", "pedal_25", "pedal_50", "pedal_75")
        ]
    )
    return output


def evaluate_window_sampling_wass_checkpoint(
    model,
    config,
    loader,
    device,
    num_samples,
    stride,
    sampling_strategy,
    pedal_support,
    pedal_binary_threshold,
    seed,
):
    model.eval()
    started = time.time()
    score_rows = []
    stride = max(1, int(stride))
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = move_batch(batch, device)
            labels = batch["labels_continuous"]
            mask = batch["attention_mask"].bool()
            valid_lengths = batch["attention_mask"].sum(dim=1).long()
            max_len = int(valid_lengths.max().detach().cpu().item())
            gt_raw = _target7_to_raw7(batch["score_shared_raw"], labels, config=config).detach().cpu().numpy()
            pred_raw_samples = []
            for sample_idx in range(int(num_samples)):
                torch.manual_seed(int(seed) + batch_idx * 1009 + sample_idx)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(seed) + batch_idx * 1009 + sample_idx)
                feedback = labels.clone()
                for start in range(0, max_len, stride):
                    outputs = model(
                        pitch_ids=batch["pitch_ids"],
                        continuous=batch["continuous"],
                        score_shared_raw=batch["score_shared_raw"],
                        labels_continuous=labels,
                        decoder_feedback_continuous=feedback,
                        labels_epr_bins=batch.get("labels_epr_bins"),
                        label_mask=batch.get("label_mask"),
                        label_valid_mask=batch.get("label_valid_mask"),
                        attention_mask=batch["attention_mask"],
                        continuous_sampling_strategy=sampling_strategy,
                    )
                    pred = _materialize_epr_prediction(
                        model.config,
                        outputs.logits,
                        sampling_strategy=sampling_strategy,
                        score_shared_raw=batch["score_shared_raw"],
                    ).detach()
                    end = min(start + stride, max_len)
                    row_active = valid_lengths > start
                    if row_active.any():
                        feedback[row_active, start:end, :] = pred[row_active, start:end, :]
                pred_raw = _target7_to_raw7(batch["score_shared_raw"], feedback, config=config).detach().cpu().numpy()
                pred_raw_samples.append(pred_raw)

            mask_np = mask.detach().cpu().numpy()
            for row_idx in range(mask_np.shape[0]):
                pred_arrays = [
                    raw_window_arrays(sample[row_idx], mask_np[row_idx], pedal_support, pedal_binary_threshold)
                    for sample in pred_raw_samples
                ]
                gt_arrays = [
                    raw_window_arrays(gt_raw[row_idx], mask_np[row_idx], pedal_support, pedal_binary_threshold)
                ]
                score_rows.append(
                    {
                        "pp_wass": pp_wass_from_arrays(pred_arrays, gt_arrays),
                        "pn_wass": pn_wass_from_arrays(pred_arrays, gt_arrays),
                    }
                )
    rows = {
        "valid_window_wass_seconds": time.time() - started,
        "valid_window_wass_samples": int(num_samples),
        "valid_window_wass_count": len(score_rows),
    }
    for section in ("pp_wass", "pn_wass"):
        prefix = "valid_pp_wass" if section == "pp_wass" else "valid_pn_wass"
        for feature in ("ioi", "duration", "velocity", "pedal"):
            rows[f"{prefix}_{feature}"] = finite_mean([row[section][feature] for row in score_rows])
    return rows


def build_probe_loader(config, dataset, batch_size):
    collator = NodeSFTDataCollator(
        pitch_pad_id=config["pitch_pad_id"],
        task_type=config.get("task_type", "epr").lower(),
        use_style_tokens=config.get("use_style_tokens", False),
        epr_timing_target=config.get("epr_timing_target", "floor_log_deviation"),
        tail_mask_enabled=config.get("tail_mask_enabled", False),
        tail_mask_tf_clamp=config.get("tail_mask_tf_clamp", True),
        tail_mask_ioi_min=config.get("tail_mask_ioi_min", -1.5),
        tail_mask_ioi_max=config.get("tail_mask_ioi_max", 1.5),
        tail_mask_duration_min=config.get("tail_mask_duration_min", -2.0),
        tail_mask_duration_max=config.get("tail_mask_duration_max", 2.0),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator, num_workers=0)


def load_probe_model(config, checkpoint, device):
    model = create_model(dict(config))
    state = load_torch_state_dict(checkpoint)
    model.load_state_dict(filter_resume_state_dict(model, state, config), strict=False)
    model.to(device)
    return model


def evaluate_probe_shard(payload):
    worker_idx = int(payload["worker_idx"])
    config = payload["config"]
    checkpoint = Path(payload["checkpoint"])
    device_arg = payload["device"]
    seed = int(payload["seed"]) + worker_idx
    num_examples = int(payload["num_examples"])
    shard_start = int(payload["shard_start"])
    shard_end = int(payload["shard_end"])
    batch_size = int(payload["batch_size"])
    rollout_ks = [int(value) for value in payload["rollout_ks"]]
    window_ar_strides = [int(value) for value in payload["window_ar_strides"]]
    window_wass_samples = int(payload["window_wass_samples"])
    sampling_strategy = payload["sampling_strategy"]

    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available() and str(device_arg).startswith("cuda"):
        device = torch.device("cuda:0" if str(device_arg) == "cuda" else device_arg)
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(seed)
    else:
        device = torch.device("cpu")

    dataset = build_eval_dataset(
        config,
        num_examples,
        int(payload["selection_seed"]),
        require_human_baseline=bool(payload.get("require_human_baseline", False)),
    )
    shard = Subset(dataset, range(shard_start, min(shard_end, len(dataset))))
    loader = build_probe_loader(config, shard, batch_size)
    model = load_probe_model(config, checkpoint, device)
    row = {
        "worker_idx": worker_idx,
        "worker_examples": len(shard),
        "shard_start": shard_start,
        "shard_end": min(shard_end, len(dataset)),
    }
    if rollout_ks:
        row.update(evaluate_checkpoint(model, config, loader, device, rollout_ks, sampling_strategy))
    if window_ar_strides:
        row.update(evaluate_window_ar_checkpoint(model, loader, device, window_ar_strides, sampling_strategy))
    if window_wass_samples > 0:
        row.update(
            evaluate_window_sampling_wass_checkpoint(
                model,
                config,
                loader,
                device,
                window_wass_samples,
                int(payload["window_wass_stride"]),
                sampling_strategy,
                payload["window_wass_pedal_support"],
                float(payload["pedal_binary_threshold"]),
                seed,
            )
        )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def combine_worker_rows(worker_rows, started):
    total_examples = sum(float(row.get("worker_examples", 0.0)) for row in worker_rows)
    total_examples = max(total_examples, 1.0)
    output = {
        "parallel_workers": len(worker_rows),
        "parallel_worker_wall_seconds": time.time() - started,
        "parallel_worker_examples": int(total_examples),
    }
    numeric_keys = sorted(
        {
            key
            for row in worker_rows
            for key, value in row.items()
            if key
            not in {
                "worker_idx",
                "worker_examples",
                "shard_start",
                "shard_end",
                "probe_seconds",
                "window_ar_seconds",
            }
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
        }
    )
    for key in numeric_keys:
        if key.endswith("_count"):
            output[key] = sum(
                float(row[key])
                for row in worker_rows
                if isinstance(row.get(key), (int, float)) and math.isfinite(float(row[key]))
            )
            continue
        weighted = 0.0
        weight = 0.0
        for row in worker_rows:
            value = row.get(key)
            examples = float(row.get("worker_examples", 0.0))
            if isinstance(value, (int, float)) and math.isfinite(float(value)) and examples > 0:
                weighted += float(value) * examples
                weight += examples
        if weight > 0:
            output[key] = weighted / weight
    for key in ("probe_seconds", "window_ar_seconds", "valid_window_wass_seconds"):
        values = [float(row[key]) for row in worker_rows if key in row and math.isfinite(float(row[key]))]
        if values:
            output[f"{key}_max"] = max(values)
            output[f"{key}_sum"] = sum(values)
    return output


def evaluate_checkpoint_parallel(config, checkpoint, args, rollout_ks, window_ar_strides):
    num_workers = max(1, int(args.num_workers))
    worker_batch = max(1, int(args.worker_window_batch))
    num_examples = min(int(args.num_examples), num_workers * worker_batch)
    started = time.time()
    jobs = []
    for worker_idx in range(num_workers):
        start = worker_idx * worker_batch
        end = min(start + worker_batch, num_examples)
        if start >= end:
            continue
        jobs.append(
            {
                "worker_idx": worker_idx,
                "config": config,
                "checkpoint": str(checkpoint),
                "device": args.device,
                "seed": args.seed,
                "selection_seed": args.seed,
                "require_human_baseline": args.require_human_baseline,
                "num_examples": num_examples,
                "shard_start": start,
                "shard_end": end,
                "batch_size": worker_batch,
                "rollout_ks": rollout_ks,
                "window_ar_strides": window_ar_strides,
                "window_wass_samples": int(args.window_wass_samples),
                "window_wass_stride": int(args.window_wass_stride),
                "window_wass_pedal_support": args.window_wass_pedal_support,
                "pedal_binary_threshold": float(args.pedal_binary_threshold),
                "sampling_strategy": args.sampling_strategy,
            }
        )
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(jobs)) as pool:
        worker_rows = pool.map(evaluate_probe_shard, jobs)
    return combine_worker_rows(worker_rows, started)


def load_test_metrics(run_root, run_label, epoch, mode):
    path = run_root / f"{run_label}_ep{epoch}" / mode / "eval_pn_pp_metrics.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8")).get("aggregate", {})
    row = {}
    for section in ("pp_wass", "pn_wass"):
        for feature in ("ioi", "duration", "velocity", "pedal"):
            row[f"test_{section}_{feature}"] = finite(data.get(section, {}).get(f"{feature}_wass"))
    return row


def main():
    args = parse_args()
    config = normalize_train_config(json.loads(args.config.read_text(encoding="utf-8")))
    if args.human_baseline_only:
        dataset = build_eval_dataset(
            config, args.num_examples, args.seed, require_human_baseline=args.require_human_baseline
        )
        baseline = evaluate_validation_human_baseline(
            dataset,
            config,
            int(args.window_wass_samples),
            args.window_wass_pedal_support,
            float(args.pedal_binary_threshold),
        )
        output_path = args.human_baseline_output or (args.run_root / "validation_window_human_baseline.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(baseline, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"human_baseline_output": str(output_path), **baseline["aggregate"]}, ensure_ascii=False))
        return
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    dataset = None
    loader = None
    if int(args.num_workers) <= 1:
        dataset = build_eval_dataset(
            config, args.num_examples, args.seed, require_human_baseline=args.require_human_baseline
        )
        loader = build_probe_loader(config, dataset, args.batch_size)
    epochs = [int(part.strip()) for part in args.epochs.split(",") if part.strip()]
    rollout_ks = [int(part.strip()) for part in args.rollout_ks.split(",") if part.strip()]
    window_ar_strides = [int(part.strip()) for part in args.window_ar_strides.split(",") if part.strip()]

    rows = []
    for epoch in epochs:
        checkpoint = args.train_dir / f"epoch-{epoch}"
        if int(args.num_workers) > 1:
            max_examples = int(args.num_workers) * max(1, int(args.worker_window_batch))
            row = {"epoch": epoch, "checkpoint": str(checkpoint), "num_examples": min(int(args.num_examples), max_examples)}
            row.update(evaluate_checkpoint_parallel(config, checkpoint, args, rollout_ks, window_ar_strides))
        else:
            model = load_probe_model(config, checkpoint, device)
            row = {"epoch": epoch, "checkpoint": str(checkpoint), "num_examples": len(dataset)}
            if rollout_ks:
                row.update(evaluate_checkpoint(model, config, loader, device, rollout_ks, args.sampling_strategy))
            if window_ar_strides:
                row.update(evaluate_window_ar_checkpoint(model, loader, device, window_ar_strides, args.sampling_strategy))
            if int(args.window_wass_samples) > 0:
                row.update(
                    evaluate_window_sampling_wass_checkpoint(
                        model,
                        config,
                        loader,
                        device,
                        int(args.window_wass_samples),
                        int(args.window_wass_stride),
                        args.sampling_strategy,
                        args.window_wass_pedal_support,
                        float(args.pedal_binary_threshold),
                        int(args.seed),
                    )
                )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        row.update(load_test_metrics(args.run_root, args.run_label, epoch, args.test_mode))
        rows.append(row)
        print(
            json.dumps(
                {
                    "event": "probe_epoch_done",
                    "epoch": epoch,
                    "probe_seconds": row.get(
                        "probe_seconds",
                        row.get("parallel_worker_wall_seconds", row.get("valid_window_wass_seconds")),
                    ),
                }
            ),
            flush=True,
        )

    output_csv = args.output_csv or (args.run_root / "validation_window_rollout_probe.csv")
    output_json = args.output_json or (args.run_root / "validation_window_rollout_probe.json")
    fieldnames = sorted({key for row in rows for key in row})
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    targets = [
        "test_pp_wass_velocity",
        "test_pp_wass_duration",
        "test_pp_wass_ioi",
        "test_pn_wass_velocity",
        "test_pn_wass_duration",
        "test_pn_wass_ioi",
    ]
    proxy_keys = [
        key
        for key in fieldnames
        if (
            (
                (key.startswith("k") or key.startswith("war_s"))
                and (key.endswith("_loss") or key.endswith("_mean_delta") or key.endswith("_std_delta") or key.endswith("_mae"))
            )
            or key.startswith("valid_pp_wass_")
            or key.startswith("valid_pn_wass_")
        )
    ]
    correlations = {}
    for proxy in proxy_keys:
        values = [row.get(proxy, math.nan) for row in rows]
        for target in targets:
            target_values = [row.get(target, math.nan) for row in rows]
            correlations[f"{proxy}__vs__{target}_pearson"] = pearson(values, target_values)
            correlations[f"{proxy}__vs__{target}_spearman"] = pearson(ranks(values), ranks(target_values))

    output = {"rows": rows, "correlations": correlations}
    output_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_csv": str(output_csv), "output_json": str(output_json)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
