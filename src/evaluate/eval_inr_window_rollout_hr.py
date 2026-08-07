#!/usr/bin/env python3
import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluate.plot_sampling_matrix_humanrel import HUMAN_PN, HUMAN_PP
from src.inference.infer_inr_testset import load_config, load_model
from src.data_process.work_manifest import build_work_manifest
from src.train.train_inr import (
    NodeSFTDataCollator,
    PianoCoReNodeSFTDataset,
    _materialize_epr_prediction,
    _normalizer,
    _pn_pp_metric_w1_loss,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Low-cost INR validation on random distinct-work windows with rollout feedback.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--split", default="valid")
    parser.add_argument("--performance-dataset", default="ASAP")
    parser.add_argument("--m-values", default="32,64")
    parser.add_argument("--rollout-ks", default="8,16,32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size-windows", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--materialize-strategy", default="sample")
    parser.add_argument("--feedback-strategy", default="sample")
    parser.add_argument("--fail-on-insufficient-works", action="store_true")
    return parser.parse_args()


def parse_int_list(text):
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError("Expected at least one integer")
    return values


def filter_manifest_performance_dataset(manifest, performance_dataset):
    if performance_dataset is None:
        return manifest
    out = []
    wanted = str(performance_dataset)
    for item in manifest:
        copied = dict(item)
        selected = [
            source for source in item.get("selected_performance_sources", [])
            if f"/{wanted}_" in f"/{source}" or Path(source).name.startswith(f"{wanted}_")
        ]
        # Prefer metadata-provided performance sources when source names encode the dataset.
        if selected:
            copied["selected_performance_sources"] = selected
            copied["estimated_performances"] = len(selected)
            copied["estimated_examples"] = len(copied.get("windows") or []) * len(selected)
        out.append(copied)
    return out


def choose_distinct_work_windows(manifest, m, seed):
    candidates = [
        item for item in manifest
        if item.get("windows") and int(item.get("estimated_performances", 0) or 0) >= 2
    ]
    if len(candidates) < m:
        raise ValueError(
            f"Cannot select m={m} distinct-work windows from only {len(candidates)} works. "
            "Use a smaller m or a larger candidate split."
        )
    rng = random.Random(seed)
    selected_items = rng.sample(candidates, m)
    out = []
    for item in selected_items:
        copied = dict(item)
        copied["windows"] = [list(rng.choice(list(item["windows"])))]
        copied["estimated_examples"] = int(copied.get("estimated_performances", 1) or 1)
        out.append(copied)
    return out


def make_dataset(config, manifest, split):
    group_size = int(config.get("validation_multi_perf_group_size", 1024) or 1024)
    min_group = int(config.get("validation_multi_perf_min_group_size", 2) or 2)
    return PianoCoReNodeSFTDataset(
        manifest,
        split=split,
        task_type=config.get("task_type", "epr"),
        input_feature_mode=config.get("input_feature_mode", "integrated"),
        shuffle=False,
        seed=int(config.get("seed", 42)),
        cache_size=int(config.get("node_cache_size", 16) or 16),
        timing_normalization=config.get("timing_input_normalization", "linear_5000"),
        max_time_ms=config.get("max_time_ms", 10000.0),
        epr_timing_bins=config.get("epr_timing_bins", 5000),
        epr_value_bins=config.get("epr_value_bins", 128),
        pedal_representation=config.get("pedal_representation", "binary_4"),
        musical_feature_mode=config.get("musical_feature_mode", "musical4slot"),
        score_note_schema=config.get("score_note_input_schema", "integrated"),
        epr_timing_target=config.get("epr_timing_target", "floor_log_deviation"),
        disable_musical_features=config.get("disable_musical_features", False),
        musical_feature_transform=config.get("musical_feature_transform", "none"),
        musical_random_seed=config.get("musical_random_seed", config.get("seed", 42)),
        use_timing_scale_bit=config.get("use_timing_scale_bit", False),
        timing_control_mode=config.get("timing_control_mode"),
        timing_log_scale=config.get("timing_log_scale", 50.0),
        use_prepared_sidecar=config.get("use_prepared_sidecar", True),
        prepared_sidecar_tag=config.get("prepared_sidecar_tag"),
        use_style_tokens=config.get("use_style_tokens", False),
        composer_vocab=config.get("style_composer_vocab"),
        source_vocab=config.get("style_source_vocab"),
        perf_style_stats_mode=config.get("perf_style_stats_mode", "prefix"),
        legacy_dual_timing_head=config.get("legacy_dual_timing_head", False),
        multi_perf_group_size=group_size,
        multi_perf_min_group_size=min_group,
    )


def materialize_feedback(model_config, logits, strategy, score_shared_raw):
    return _materialize_epr_prediction(
        model_config,
        logits,
        sampling_strategy=strategy,
        score_shared_raw=score_shared_raw,
    )


def evaluate_k(model, model_config, loader, device, rollout_k, materialize_strategy, feedback_strategy):
    totals = {}
    total_weight = 0.0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"k={rollout_k}", leave=False):
            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            attention_mask = batch["attention_mask"]
            labels = batch["labels_continuous"]
            group_index = batch.get("pn_group_index")
            if group_index is None:
                raise ValueError("Validation dataset did not produce pn_group_index; need grouped performance windows.")
            feedback = None
            outputs = None
            for pass_idx in range(int(rollout_k) + 1):
                outputs = model(
                    pitch_ids=batch["pitch_ids"],
                    continuous=batch["continuous"],
                    score_shared_raw=batch["score_shared_raw"],
                    labels_continuous=labels,
                    decoder_feedback_continuous=feedback,
                    labels_epr_bins=batch.get("labels_epr_bins"),
                    label_mask=batch.get("label_mask"),
                    attention_mask=attention_mask,
                    continuous_sampling_strategy=materialize_strategy,
                )
                if pass_idx < int(rollout_k):
                    feedback = materialize_feedback(
                        model_config,
                        outputs.logits,
                        feedback_strategy,
                        batch["score_shared_raw"],
                    ).detach()

            metrics = _pn_pp_metric_w1_loss(
                model_config,
                outputs.logits,
                labels,
                batch["score_shared_raw"],
                attention_mask,
                group_index,
                compute_pn=("ioi", "duration", "velocity"),
                compute_pp=("ioi", "duration", "velocity"),
            )
            weight = float(len(torch.unique(group_index).detach().cpu()))
            total_weight += weight
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * weight
    if total_weight <= 0:
        raise ValueError("No grouped validation windows were evaluated")
    averaged = {key: value / total_weight for key, value in totals.items()}
    raw = {}
    for prefix, human in (("pn", HUMAN_PN), ("pp", HUMAN_PP)):
        rel_terms = []
        for feature in ("ioi", "duration", "velocity"):
            key = f"{prefix}_{feature}_w1"
            raw_key = f"{prefix}_{feature}_w1_raw"
            raw_value = averaged[key] * _normalizer(model_config, prefix, feature)
            raw[raw_key] = raw_value
            rel_terms.append(raw_value / human[f"{feature}_wass"])
        raw[f"{prefix}_hr"] = sum(rel_terms) / len(rel_terms)
    return {**averaged, **raw}


def main():
    args = parse_args()
    m_values = parse_int_list(args.m_values)
    rollout_ks = parse_int_list(args.rollout_ks)
    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    config = load_config(args.config, args.checkpoint)
    fixed_scheme = config.get("fixed_window_split_scheme")
    manifest_split = config.get("fixed_window_base_split", "train") if fixed_scheme else args.split
    window_split_name = config.get("fixed_window_eval_split_name", args.split) if fixed_scheme else None
    manifest = build_work_manifest(
        metadata_path=config["metadata_path"],
        refined_dir=config["refined_dir"],
        split=manifest_split,
        block_notes=config["block_notes"],
        overlap_ratio=config["overlap_ratio"],
        min_notes=config["min_notes"],
        max_works=None,
        performance_dataset=args.performance_dataset,
        skip_work_paths=config.get("skip_work_paths"),
        prepared_sidecar_tag=config.get("prepared_sidecar_tag"),
        window_split_scheme=fixed_scheme,
        window_split_name=window_split_name,
        window_split_summary_path=config.get("fixed_window_split_summary_path"),
    )
    available_works = len([item for item in manifest if item.get("windows")])
    model = load_model(config, device)
    model_config = model.module.config if hasattr(model, "module") else model.config

    results = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "performance_dataset": args.performance_dataset,
        "available_distinct_works": available_works,
        "feature_set": ["ioi", "duration", "velocity"],
        "m": {},
    }
    errors = {}
    for m in m_values:
        if available_works < m:
            message = (
                f"Cannot evaluate m={m}: only {available_works} distinct works are available "
                f"for split={args.split}."
            )
            errors[str(m)] = message
            if args.fail_on_insufficient_works:
                raise ValueError(message)
            continue
        selected = choose_distinct_work_windows(manifest, m=m, seed=args.seed + m)
        dataset = make_dataset(config, selected, split=args.split)
        collator = NodeSFTDataCollator(
            pitch_pad_id=config["pitch_pad_id"],
            task_type=config.get("task_type", "epr"),
            use_style_tokens=config.get("use_style_tokens", False),
            epr_timing_target=config.get("epr_timing_target", "floor_log_deviation"),
            tail_mask_enabled=config.get("tail_mask_enabled", False),
            tail_mask_tf_clamp=config.get("tail_mask_tf_clamp", True),
            tail_mask_ioi_min=config.get("tail_mask_ioi_min", -1.5),
            tail_mask_ioi_max=config.get("tail_mask_ioi_max", 1.5),
            tail_mask_duration_min=config.get("tail_mask_duration_min", -2.0),
            tail_mask_duration_max=config.get("tail_mask_duration_max", 2.0),
        )
        loader = DataLoader(
            dataset,
            batch_size=int(args.batch_size_windows),
            shuffle=False,
            num_workers=int(args.num_workers),
            collate_fn=collator,
        )
        results["m"][str(m)] = {}
        for k in rollout_ks:
            results["m"][str(m)][str(k)] = evaluate_k(
                model,
                model_config,
                loader,
                device,
                rollout_k=k,
                materialize_strategy=args.materialize_strategy,
                feedback_strategy=args.feedback_strategy,
            )
    if errors:
        results["errors"] = errors
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_json), "available_distinct_works": available_works}, sort_keys=True))


if __name__ == "__main__":
    main()
