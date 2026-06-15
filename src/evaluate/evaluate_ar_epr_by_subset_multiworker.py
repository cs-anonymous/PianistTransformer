"""
Multi-worker version of AR EPR evaluation.

Each worker processes a disjoint subset of samples (rank / world_size stride),
writes its own partial JSON, then a small merge step combines them.

Usage:
  # Worker 0 of 2 on GPU 0
  CUDA_VISIBLE_DEVICES=0 python evaluate_ar_epr_by_subset_multiworker.py \
      --config ... --checkpoint ... --output-dir ... \
      --worker-id 0 --world-size 2

  # After all workers finish, merge:
  python evaluate_ar_epr_by_subset_multiworker.py --merge \
      --output-dir results/inr_epr_ar_subset_eval/gpt_17 --world-size 2
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
from src.evaluate.evaluate_integrated_node import load_model
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


def predict_batch_ar(model, batch, device):
    with torch.no_grad():
        pitch_ids = batch["pitch_ids"].to(device)
        continuous = batch["continuous"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(
            pitch_ids=pitch_ids,
            continuous=continuous,
            attention_mask=attention_mask,
        )

        continuous_pred = outputs.logits
        return continuous_pred.cpu().numpy(), attention_mask.cpu().numpy()


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


def run_worker(args):
    with open(args.config, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    decoder_mode = config.get("decoder_input_mode", "score").lower()
    if decoder_mode != "ar":
        print(f"[WARN] decoder_input_mode='{decoder_mode}' is not 'ar'; "
              "this script is designed for autoregressive models.")

    print("=" * 60)
    print(f"AR EPR Multi-Worker Evaluation  [worker {args.worker_id}/{args.world_size}]")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output dir: {args.output_dir}")
    print(f"Backbone: {config.get('backbone_type', '?')}")
    print(f"Decoder mode: {decoder_mode}")
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

    # This worker's slice: indices [worker_id, worker_id + world_size, ...]
    worker_indices = list(range(args.worker_id, len(test_dataset), args.world_size))
    print(f"Worker {args.worker_id} handles {len(worker_indices)} samples")

    model = load_model(args.checkpoint, config).to(args.device)
    model.eval()
    collator = NodeSFTDataCollator(pitch_pad_id=config.get("pitch_pad_id", 128))

    batch_size = args.batch_size if args.batch_size is not None else 1
    num_batches = (len(worker_indices) + batch_size - 1) // batch_size
    print(f"Eval batch size: {batch_size}  (num_batches: {num_batches})")
    print()

    subset_sample_counts = defaultdict(int)
    subset_note_counts = defaultdict(int)

    pred_binary_by_subset = defaultdict(init_feature_store)
    target_binary_by_subset = defaultdict(init_feature_store)
    pred_cont_by_subset = defaultdict(init_feature_store)
    target_cont_by_subset = defaultdict(init_feature_store)

    for batch_idx in tqdm(range(num_batches), desc=f"Worker {args.worker_id} AR inference"):
        batch_samples = []
        batch_indices = []
        for sample_offset in range(batch_size):
            idx_in_worker = batch_idx * batch_size + sample_offset
            if idx_in_worker >= len(worker_indices):
                break
            dataset_idx = worker_indices[idx_in_worker]
            batch_samples.append(test_dataset[dataset_idx])
            batch_indices.append(dataset_idx)

        if not batch_samples:
            break

        batch = collator(batch_samples)
        pred_continuous, mask_batch = predict_batch_ar(model, batch, args.device)
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

    worker_result = {
        "worker_id": args.worker_id,
        "world_size": args.world_size,
        "checkpoint": args.checkpoint,
        "subsets": {},
    }

    for subset_name in ["ASAP", "PianoCoRe-only"]:
        pred_binary = finalize_feature_store(pred_binary_by_subset[subset_name])
        target_binary = finalize_feature_store(target_binary_by_subset[subset_name])
        pred_cont = finalize_feature_store(pred_cont_by_subset[subset_name])
        target_cont = finalize_feature_store(target_cont_by_subset[subset_name])

        worker_result["subsets"][subset_name] = {
            "num_samples": int(subset_sample_counts[subset_name]),
            "num_notes": int(subset_note_counts[subset_name]),
            "pred_binary": {k: v.tolist() for k, v in pred_binary.items()},
            "target_binary": {k: v.tolist() for k, v in target_binary.items()},
            "pred_continuous": {k: v.tolist() for k, v in pred_cont.items()},
            "target_continuous": {k: v.tolist() for k, v in target_cont.items()},
        }

    worker_path = output_dir / f"worker_{args.worker_id}_of_{args.world_size}.json"
    with open(worker_path, "w", encoding="utf-8") as handle:
        json.dump(worker_result, handle, indent=2)
    print(f"Worker {args.worker_id} saved partial result to {worker_path}")


def merge_workers(args):
    output_dir = Path(args.output_dir)
    print(f"Merging {args.world_size} worker results from {output_dir}")

    merged = {
        "checkpoint": None,
        "subsets": {
            "ASAP": {
                "num_samples": 0,
                "num_notes": 0,
                "pred_binary": init_feature_store(),
                "target_binary": init_feature_store(),
                "pred_continuous": init_feature_store(),
                "target_continuous": init_feature_store(),
            },
            "PianoCoRe-only": {
                "num_samples": 0,
                "num_notes": 0,
                "pred_binary": init_feature_store(),
                "target_binary": init_feature_store(),
                "pred_continuous": init_feature_store(),
                "target_continuous": init_feature_store(),
            },
        },
    }

    for worker_id in range(args.world_size):
        worker_path = output_dir / f"worker_{worker_id}_of_{args.world_size}.json"
        if not worker_path.exists():
            print(f"[WARN] Missing worker result: {worker_path}")
            continue

        with open(worker_path, "r", encoding="utf-8") as handle:
            worker_data = json.load(handle)

        if merged["checkpoint"] is None:
            merged["checkpoint"] = worker_data["checkpoint"]

        for subset_name in ["ASAP", "PianoCoRe-only"]:
            subset = worker_data["subsets"][subset_name]
            merged["subsets"][subset_name]["num_samples"] += subset["num_samples"]
            merged["subsets"][subset_name]["num_notes"] += subset["num_notes"]

            for key in ["velocity", "duration", "ioi", "pedal"]:
                merged["subsets"][subset_name]["pred_binary"][key].extend(subset["pred_binary"][key])
                merged["subsets"][subset_name]["target_binary"][key].extend(subset["target_binary"][key])
                merged["subsets"][subset_name]["pred_continuous"][key].extend(subset["pred_continuous"][key])
                merged["subsets"][subset_name]["target_continuous"][key].extend(subset["target_continuous"][key])

    # Compute final metrics
    results_binary = {
        "pedal_method": "binary",
        "decoder_mode": "ar",
        "checkpoint": merged["checkpoint"],
        "subsets": {},
    }
    results_continuous = {
        "pedal_method": "continuous",
        "decoder_mode": "ar",
        "checkpoint": merged["checkpoint"],
        "subsets": {},
    }

    for subset_name in ["ASAP", "PianoCoRe-only"]:
        subset = merged["subsets"][subset_name]

        pred_binary = {k: np.array(v, dtype=np.float32) for k, v in subset["pred_binary"].items()}
        target_binary = {k: np.array(v, dtype=np.float32) for k, v in subset["target_binary"].items()}
        pred_cont = {k: np.array(v, dtype=np.float32) for k, v in subset["pred_continuous"].items()}
        target_cont = {k: np.array(v, dtype=np.float32) for k, v in subset["target_continuous"].items()}

        subset_result_binary = compute_subset_metrics(pred_binary, target_binary, pedal_is_joint_config=True)
        subset_result_cont = compute_subset_metrics(pred_cont, target_cont, pedal_is_joint_config=False)
        add_feature_stats(subset_result_binary, pred_binary, target_binary)
        add_feature_stats(subset_result_cont, pred_cont, target_cont)

        subset_result_binary["num_samples"] = subset["num_samples"]
        subset_result_binary["num_notes"] = subset["num_notes"]
        subset_result_cont["num_samples"] = subset["num_samples"]
        subset_result_cont["num_notes"] = subset["num_notes"]

        results_binary["subsets"][subset_name] = subset_result_binary
        results_continuous["subsets"][subset_name] = subset_result_cont

    with open(output_dir / "results_binary.json", "w", encoding="utf-8") as handle:
        json.dump(results_binary, handle, indent=2)
    with open(output_dir / "results_continuous.json", "w", encoding="utf-8") as handle:
        json.dump(results_continuous, handle, indent=2)

    print(f"Merged results saved to:")
    print(f"  {output_dir / 'results_binary.json'}")
    print(f"  {output_dir / 'results_continuous.json'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge", action="store_true", help="Merge worker results instead of running")
    parser.add_argument("--config", type=str, help="Training config JSON")
    parser.add_argument("--checkpoint", type=str, help="Model checkpoint path")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for output JSON files")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--batch-size", type=int, default=1, help="Eval batch size (default=1)")
    parser.add_argument("--worker-id", type=int, help="Worker rank (0-indexed)")
    parser.add_argument("--world-size", type=int, help="Total number of workers")
    args = parser.parse_args()

    if args.merge:
        if args.world_size is None:
            raise ValueError("--world-size required for --merge")
        merge_workers(args)
    else:
        if args.worker_id is None or args.world_size is None:
            raise ValueError("--worker-id and --world-size required")
        if args.config is None or args.checkpoint is None:
            raise ValueError("--config and --checkpoint required")
        run_worker(args)


if __name__ == "__main__":
    main()
