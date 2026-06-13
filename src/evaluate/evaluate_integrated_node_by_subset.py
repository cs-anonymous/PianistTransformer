"""
Evaluate Integrated Node models on ASAP and PianoCoRe-only subsets in one pass.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.evaluate.epr_metrics import EPRMetrics, extract_features_from_continuous
from src.evaluate.epr_metrics_extended import ExtendedEPRMetrics
from src.evaluate.evaluate_integrated_node import load_model, predict_batch
from src.train.sft_node import (
    PianoCoReNodeSFTDataset,
    build_work_manifest,
    NodeSFTDataCollator,
    infer_input_feature_mode,
)


def get_subset_name(performance_dataset: str) -> str:
    return "ASAP" if performance_dataset == "ASAP" else "PianoCoRe-only"


def init_feature_store():
    return {"velocity": [], "duration": [], "ioi": [], "pedal": []}


def append_features(store, features):
    for key in store:
        store[key].append(features[key])


def finalize_feature_store(store):
    finalized = {}
    for key, values in store.items():
        if values:
            finalized[key] = np.concatenate(values)
        else:
            finalized[key] = np.array([], dtype=np.float32)
    return finalized


def compute_subset_metrics(pred_features, target_features, pedal_is_joint_config):
    dist_metrics = EPRMetrics(bins=100)
    dist_results = dist_metrics.compute_metrics(
        pred_features,
        target_features,
        pedal_is_joint_config=pedal_is_joint_config,
    )
    extended_metrics = ExtendedEPRMetrics(bins=100)
    pedal_method = "binary" if pedal_is_joint_config else "continuous"
    extended_results = extended_metrics.compute_metrics_for_features(
        pred_features,
        target_features,
        pedal_method=pedal_method,
    )
    return {
        "distribution_metrics": {
            "velocity_js": float(dist_results["velocity_js"]),
            "velocity_ia": float(dist_results["velocity_ia"]),
            "duration_js": float(dist_results["duration_js"]),
            "duration_ia": float(dist_results["duration_ia"]),
            "ioi_js": float(dist_results["ioi_js"]),
            "ioi_ia": float(dist_results["ioi_ia"]),
            "pedal_js": float(dist_results["pedal_js"]),
            "pedal_ia": float(dist_results["pedal_ia"]),
            "overall_js": float(dist_results["overall_js"]),
            "overall_ia": float(dist_results["overall_ia"]),
        },
        "fine_grained_metrics": json.loads(json.dumps(extended_results, default=float)),
    }


def add_feature_stats(result_dict, pred_features, target_features):
    result_dict["feature_stats"] = {
        "pred": {
            "velocity_mean": float(np.mean(pred_features["velocity"])),
            "velocity_std": float(np.std(pred_features["velocity"])),
            "duration_mean": float(np.mean(pred_features["duration"])),
            "duration_std": float(np.std(pred_features["duration"])),
            "ioi_mean": float(np.mean(pred_features["ioi"])),
            "ioi_std": float(np.std(pred_features["ioi"])),
            "pedal_mean": float(np.mean(pred_features["pedal"])),
            "pedal_std": float(np.std(pred_features["pedal"])),
        },
        "target": {
            "velocity_mean": float(np.mean(target_features["velocity"])),
            "velocity_std": float(np.std(target_features["velocity"])),
            "duration_mean": float(np.mean(target_features["duration"])),
            "duration_std": float(np.std(target_features["duration"])),
            "ioi_mean": float(np.mean(target_features["ioi"])),
            "ioi_std": float(np.std(target_features["ioi"])),
            "pedal_mean": float(np.mean(target_features["pedal"])),
            "pedal_std": float(np.std(target_features["pedal"])),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Training config JSON")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for output JSON files")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    print("=" * 60)
    print("Integrated Node Evaluation by Subset")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output dir: {args.output_dir}")
    print()

    test_manifest = build_work_manifest(
        metadata_path=config["metadata_path"],
        refined_dir=config["refined_dir"],
        split="test",
        block_notes=config["block_notes"],
        overlap_ratio=config["overlap_ratio"],
        min_notes=config["min_notes"],
        max_works=config.get("max_eval_works"),
    )
    test_dataset = PianoCoReNodeSFTDataset(
        test_manifest,
        split="test",
        input_feature_mode=infer_input_feature_mode(config),
        shuffle=False,
        seed=config["seed"],
        max_performances_per_work=config.get("max_eval_performances_per_work"),
        max_windows_per_work=config.get("max_eval_windows_per_work"),
    )
    print(f"Test dataset size: {len(test_dataset)} samples")

    model = load_model(args.checkpoint, config).to(args.device)
    model.eval()
    collator = NodeSFTDataCollator(pitch_pad_id=config.get("pitch_pad_id", 128))
    batch_size = config.get("per_device_eval_batch_size", 2)
    num_batches = (len(test_dataset) + batch_size - 1) // batch_size

    subset_sample_counts = defaultdict(int)
    subset_note_counts = defaultdict(int)

    pred_binary_by_subset = defaultdict(init_feature_store)
    target_binary_by_subset = defaultdict(init_feature_store)
    pred_cont_by_subset = defaultdict(init_feature_store)
    target_cont_by_subset = defaultdict(init_feature_store)

    for batch_idx in tqdm(range(num_batches), desc="Running inference"):
        batch_samples = []
        for sample_idx in range(batch_size):
            dataset_idx = batch_idx * batch_size + sample_idx
            if dataset_idx >= len(test_dataset):
                break
            batch_samples.append(test_dataset[dataset_idx])

        if not batch_samples:
            break

        batch = collator(batch_samples)
        pred_continuous, mask_batch = predict_batch(model, batch, args.device)
        target_continuous = batch["labels_continuous"].numpy()

        for sample_offset, sample in enumerate(batch_samples):
            subset_name = get_subset_name(sample.get("performance_dataset", "unknown"))
            sample_mask = mask_batch[sample_offset : sample_offset + 1]
            sample_pred = pred_continuous[sample_offset : sample_offset + 1]
            sample_target = target_continuous[sample_offset : sample_offset + 1]

            pred_binary = extract_features_from_continuous(sample_pred, sample_mask, pedal_as_joint_config=True)
            target_binary = extract_features_from_continuous(sample_target, sample_mask, pedal_as_joint_config=True)
            pred_cont = extract_features_from_continuous(sample_pred, sample_mask, pedal_as_joint_config=False)
            target_cont = extract_features_from_continuous(sample_target, sample_mask, pedal_as_joint_config=False)

            append_features(pred_binary_by_subset[subset_name], pred_binary)
            append_features(target_binary_by_subset[subset_name], target_binary)
            append_features(pred_cont_by_subset[subset_name], pred_cont)
            append_features(target_cont_by_subset[subset_name], target_cont)

            subset_sample_counts[subset_name] += 1
            subset_note_counts[subset_name] += int(sample_mask.sum())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_binary = {
        "pedal_method": "binary",
        "checkpoint": args.checkpoint,
        "subsets": {},
    }
    results_continuous = {
        "pedal_method": "continuous",
        "checkpoint": args.checkpoint,
        "subsets": {},
    }

    for subset_name in ["ASAP", "PianoCoRe-only"]:
        pred_binary = finalize_feature_store(pred_binary_by_subset[subset_name])
        target_binary = finalize_feature_store(target_binary_by_subset[subset_name])
        pred_cont = finalize_feature_store(pred_cont_by_subset[subset_name])
        target_cont = finalize_feature_store(target_cont_by_subset[subset_name])

        subset_result_binary = compute_subset_metrics(pred_binary, target_binary, pedal_is_joint_config=True)
        subset_result_cont = compute_subset_metrics(pred_cont, target_cont, pedal_is_joint_config=False)
        add_feature_stats(subset_result_binary, pred_binary, target_binary)
        add_feature_stats(subset_result_cont, pred_cont, target_cont)

        subset_result_binary["num_samples"] = int(subset_sample_counts[subset_name])
        subset_result_binary["num_notes"] = int(subset_note_counts[subset_name])
        subset_result_cont["num_samples"] = int(subset_sample_counts[subset_name])
        subset_result_cont["num_notes"] = int(subset_note_counts[subset_name])

        results_binary["subsets"][subset_name] = subset_result_binary
        results_continuous["subsets"][subset_name] = subset_result_cont

    with open(output_dir / "results_binary.json", "w", encoding="utf-8") as handle:
        json.dump(results_binary, handle, indent=2)
    with open(output_dir / "results_continuous.json", "w", encoding="utf-8") as handle:
        json.dump(results_continuous, handle, indent=2)

    print(f"Saved binary results to {output_dir / 'results_binary.json'}")
    print(f"Saved continuous results to {output_dir / 'results_continuous.json'}")


if __name__ == "__main__":
    main()
