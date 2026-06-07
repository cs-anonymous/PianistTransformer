"""
Evaluate PT on ASAP and PianoCoRe-only subsets separately.
Uses the correct Score → Performance generation method.
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

PT_PATH = Path("/home/kaititech/EPR/third_party/epr/PianistTransformer")
sys.path.insert(0, str(PT_PATH))

from src.evaluate.epr_metrics import EPRMetrics, extract_features_from_continuous
from src.evaluate.epr_metrics_extended import ExtendedEPRMetrics


def load_pt_model_and_config():
    """Load PT model and config."""
    from src.model.pianoformer import PianoT5Gemma, PianoT5GemmaConfig

    model_path = PT_PATH / "models/sft/model.safetensors"
    config_path = PT_PATH / "models/sft/config.json"

    with open(config_path) as f:
        config_dict = json.load(f)

    config = PianoT5GemmaConfig(
        encoder_layers_num=config_dict['encoder']['num_hidden_layers'],
        decoder_layers_num=config_dict['decoder']['num_hidden_layers'],
        hidden_size=config_dict['hidden_size'],
        intermediate_size=config_dict['encoder']['intermediate_size'],
        num_attention_heads=config_dict['encoder']['num_attention_heads'],
        num_key_value_heads=config_dict['encoder']['num_key_value_heads'],
        head_dim=config_dict['encoder']['head_dim'],
    )

    model = PianoT5Gemma(config)

    from safetensors.torch import load_file
    state_dict = load_file(str(model_path))
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model, config


def should_include_sample(performance_dataset, source_filter):
    """
    Check if a sample should be included based on source filter.

    Args:
        performance_dataset: Dataset name from sample (e.g., 'ASAP', 'GiantMIDI-Piano', etc.)
        source_filter: 'asap', 'pianocore', or None

    Returns:
        bool: True if sample should be included
    """
    if source_filter is None:
        return True
    elif source_filter == 'asap':
        return performance_dataset == 'ASAP'
    else:  # pianocore
        return performance_dataset != 'ASAP'


def create_score_input(pitch_ids, score_continuous, config):
    """Create score input tokens from pitch and score_continuous."""
    score_tokens = []
    max_time_ms = 10000.0

    for i, pitch in enumerate(pitch_ids):
        pitch_token = int(config.pitch_start + pitch)

        cont = score_continuous[i]
        ioi_norm, dur_norm, vel_norm = cont[0], cont[1], cont[2]
        pedal_norm = cont[3:7]

        ioi_ms = int(np.clip(np.expm1(ioi_norm * np.log1p(max_time_ms)), 0, 4990))
        dur_ms = int(np.clip(np.expm1(dur_norm * np.log1p(max_time_ms)), 0, 4990))
        vel = int(np.clip(vel_norm * 127, 0, 127))
        pedal_vals = np.clip((np.array(pedal_norm) * 127).astype(int), 0, 127)

        ioi_token = np.clip(config.timing_start + ioi_ms, config.valid_id_range[1][0], config.valid_id_range[1][1] - 1)
        dur_token = np.clip(config.timing_start + dur_ms, config.valid_id_range[3][0], config.valid_id_range[3][1] - 1)
        vel_token = np.clip(config.velocity_start + vel, config.valid_id_range[2][0], config.valid_id_range[2][1] - 1)
        pedal_tokens = [int(np.clip(config.pedal_start + p, config.valid_id_range[4+j][0], config.valid_id_range[4+j][1] - 1)) for j, p in enumerate(pedal_vals)]

        score_tokens.extend([int(pitch_token), int(ioi_token), int(vel_token), int(dur_token)] + pedal_tokens)

    return score_tokens


def pt_tokens_to_continuous_features(token_ids, config):
    """Convert PT's output tokens to continuous features."""
    num_notes = len(token_ids) // 8
    tokens = np.array(token_ids[:num_notes * 8]).reshape(num_notes, 8)

    ioi_tokens = tokens[:, 1] - config.timing_start
    duration_tokens = tokens[:, 3] - config.timing_start
    velocity_tokens = tokens[:, 2] - config.velocity_start
    pedal_tokens = tokens[:, 4:8] - config.pedal_start

    max_time_ms = 10000.0

    ioi_ms = np.clip(ioi_tokens, 0, max_time_ms)
    ioi_normalized = np.log1p(ioi_ms) / np.log1p(max_time_ms)

    duration_ms = np.clip(duration_tokens, 0, max_time_ms)
    duration_normalized = np.log1p(duration_ms) / np.log1p(max_time_ms)

    velocity_normalized = np.clip(velocity_tokens, 0, 127) / 127.0
    pedal_normalized = np.clip(pedal_tokens, 0, 127) / 127.0

    continuous = np.concatenate([
        ioi_normalized.reshape(-1, 1),
        duration_normalized.reshape(-1, 1),
        velocity_normalized.reshape(-1, 1),
        pedal_normalized
    ], axis=1)

    return continuous


def worker_process(worker_id, gpu_id, sample_indices, dataset, config, source_filter, result_queue):
    """Worker process for one GPU."""
    device = torch.device(f'cuda:{gpu_id}')

    # Load model
    model, _ = load_pt_model_and_config()
    model = model.to(device)
    model.eval()

    results = []

    for idx in sample_indices:
        try:
            sample = dataset[idx]

            # Filter by source if specified
            performance_dataset = sample.get('performance_dataset', 'unknown')
            if not should_include_sample(performance_dataset, source_filter):
                continue

            pitch_ids = sample['pitch_ids']
            score_continuous = sample['continuous']
            target_continuous = sample['labels_continuous']

            if len(pitch_ids) > 512:
                continue

            score_input = create_score_input(pitch_ids, score_continuous, config)
            input_ids = torch.tensor([score_input], dtype=torch.long).to(device)

            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=len(pitch_ids) * 8,
                    do_sample=False,
                    pad_token_id=config.pad_token_id,
                    eos_token_id=config.eos_token_id,
                )

            pred_tokens = generated_ids[0].cpu().numpy().tolist()
            pred_cont = pt_tokens_to_continuous_features(pred_tokens, config)

            target_cont = np.array(target_continuous)

            min_len = min(len(pred_cont), len(target_cont))
            pred_cont = pred_cont[:min_len]
            target_cont = target_cont[:min_len]

            mask = np.ones(min_len, dtype=bool)

            results.append({
                'pred_cont': pred_cont,
                'target_cont': target_cont,
                'mask': mask,
                'performance_dataset': performance_dataset,
                'performance_id': sample.get('performance_id', 'unknown'),
            })

        except Exception as e:
            continue

    result_queue.put(results)


def evaluate_pt_multiworker(model, config, dataset, source_filter=None,
                            max_samples=None, workers_per_gpu=3):
    """
    Multi-worker evaluation with optional source filtering.

    Args:
        source_filter: 'asap', 'pianocore', or None (all)
    """
    num_gpus = torch.cuda.device_count()
    total_workers = num_gpus * workers_per_gpu
    print(f"Using {num_gpus} GPUs with {workers_per_gpu} workers per GPU = {total_workers} total workers")

    if source_filter:
        print(f"Filtering to {source_filter.upper()} samples only")

    if max_samples and max_samples < len(dataset):
        dataset = torch.utils.data.Subset(dataset, range(max_samples))

    num_samples = len(dataset)
    samples_per_worker = (num_samples + total_workers - 1) // total_workers

    # Assign samples to workers
    worker_assignments = []
    for worker_id in range(total_workers):
        gpu_id = worker_id % num_gpus
        start_idx = worker_id * samples_per_worker
        end_idx = min(start_idx + samples_per_worker, num_samples)
        if start_idx < num_samples:
            indices = list(range(start_idx, end_idx))
            worker_assignments.append((worker_id, gpu_id, indices))

    print(f"Samples per worker: {[len(w[2]) for w in worker_assignments]}")

    # Create result queue
    mp.set_start_method('spawn', force=True)
    result_queue = mp.Queue()

    # Launch workers
    processes = []
    for worker_id, gpu_id, indices in worker_assignments:
        p = mp.Process(target=worker_process, args=(worker_id, gpu_id, indices, dataset, config, source_filter, result_queue))
        p.start()
        processes.append(p)

    print(f"Launched {len(processes)} worker processes")

    # Collect results
    all_results = []
    for _ in tqdm(range(len(processes)), desc="Collecting results"):
        worker_results = result_queue.get()
        all_results.extend(worker_results)

    # Wait for all processes
    for p in processes:
        p.join()

    print(f"\nSuccessfully processed {len(all_results)} samples")

    # Count by dataset
    dataset_counts = defaultdict(int)
    for result in all_results:
        dataset_counts[result['performance_dataset']] += 1

    print("\nSamples by dataset:")
    for dataset_name, count in sorted(dataset_counts.items()):
        print(f"  {dataset_name}: {count}")

    # Extract features - compute BOTH binary and continuous
    print("\nExtracting features (binary and continuous pedal methods)...")

    # Binary pedal
    pred_features_binary_list = {'velocity': [], 'duration': [], 'ioi': [], 'pedal': []}
    target_features_binary_list = {'velocity': [], 'duration': [], 'ioi': [], 'pedal': []}

    # Continuous pedal
    pred_features_continuous_list = {'velocity': [], 'duration': [], 'ioi': [], 'pedal': []}
    target_features_continuous_list = {'velocity': [], 'duration': [], 'ioi': [], 'pedal': []}

    for result in all_results:
        pred_cont = result['pred_cont']
        target_cont = result['target_cont']
        mask = result['mask']

        pred_batch = pred_cont.reshape(1, -1, 7)
        target_batch = target_cont.reshape(1, -1, 7)
        mask_batch = mask.reshape(1, -1)

        # Extract binary pedal features
        pred_feats_binary = extract_features_from_continuous(pred_batch, mask_batch, pedal_as_joint_config=True)
        target_feats_binary = extract_features_from_continuous(target_batch, mask_batch, pedal_as_joint_config=True)

        for key in pred_features_binary_list.keys():
            pred_features_binary_list[key].append(pred_feats_binary[key])
            target_features_binary_list[key].append(target_feats_binary[key])

        # Extract continuous pedal features
        pred_feats_continuous = extract_features_from_continuous(pred_batch, mask_batch, pedal_as_joint_config=False)
        target_feats_continuous = extract_features_from_continuous(target_batch, mask_batch, pedal_as_joint_config=False)

        for key in pred_features_continuous_list.keys():
            pred_features_continuous_list[key].append(pred_feats_continuous[key])
            target_features_continuous_list[key].append(target_feats_continuous[key])

    # Concatenate
    pred_features_binary = {k: np.concatenate(v) for k, v in pred_features_binary_list.items()}
    target_features_binary = {k: np.concatenate(v) for k, v in target_features_binary_list.items()}

    pred_features_continuous = {k: np.concatenate(v) for k, v in pred_features_continuous_list.items()}
    target_features_continuous = {k: np.concatenate(v) for k, v in target_features_continuous_list.items()}

    print(f"Collected {len(pred_features_binary['velocity'])} notes")

    return (pred_features_binary, target_features_binary,
            pred_features_continuous, target_features_continuous,
            len(all_results))


def compute_and_print_metrics(pred_features_binary, target_features_binary,
                              pred_features_continuous, target_features_continuous):
    """Compute and print metrics for both binary and continuous pedal methods."""

    results = {}

    # Binary pedal
    print("\n" + "="*90)
    print("Binary Pedal Method (BPedal)")
    print("="*90)

    dist_metrics_binary = EPRMetrics(bins=100)
    dist_results_binary = dist_metrics_binary.compute_metrics(
        pred_features_binary, target_features_binary,
        pedal_is_joint_config=True
    )

    extended_metrics_binary = ExtendedEPRMetrics()
    extended_results_binary = extended_metrics_binary.compute_metrics_for_features(
        pred_features_binary, target_features_binary,
        pedal_method='binary'
    )

    if 'pedal' in extended_results_binary:
        extended_results_binary['BPedal'] = extended_results_binary.pop('pedal')

    overall_binary = {}
    feature_keys = ['velocity', 'duration', 'ioi', 'BPedal']
    for metric in ['js', 'ia', 'mae', 'mse', 'rmse', 'pearson']:
        values = [extended_results_binary[f][metric] for f in feature_keys if f in extended_results_binary and metric in extended_results_binary[f]]
        if values:
            overall_binary[metric] = np.mean(values)
    extended_results_binary['overall'] = overall_binary

    # Print binary results
    print()
    header = f"{'Feature':<12} {'JS↓':<10} {'IA↑':<10} {'MAE↓':<10} {'RMSE↓':<10} {'Pearson↑':<10}"
    print(header)
    print("-"*90)
    for feature in ['velocity', 'duration', 'ioi', 'BPedal']:
        if feature in extended_results_binary:
            r = extended_results_binary[feature]
            print(f"{feature:<12} {r['js']:<10.4f} {r['ia']:<10.4f} {r['mae']:<10.2f} {r['rmse']:<10.2f} {r['pearson']:<10.4f}")
    print("-"*90)
    if 'overall' in extended_results_binary:
        r = extended_results_binary['overall']
        print(f"{'Overall':<12} {r['js']:<10.4f} {r['ia']:<10.4f} {r['mae']:<10.2f} {r['rmse']:<10.2f} {r['pearson']:<10.4f}")
    print("="*90)

    results['binary'] = {
        'evaluation_type': 'score_to_performance_generation',
        'pedal_method': 'binary',
        'num_notes': len(pred_features_binary['velocity']),
        'distribution_metrics': dist_results_binary,
        'fine_grained_metrics': extended_results_binary,
    }

    # Continuous pedal
    print("\n" + "="*90)
    print("Continuous Pedal Method (CPedal)")
    print("="*90)

    dist_metrics_continuous = EPRMetrics(bins=100)
    dist_results_continuous = dist_metrics_continuous.compute_metrics(
        pred_features_continuous, target_features_continuous,
        pedal_is_joint_config=False
    )

    extended_metrics_continuous = ExtendedEPRMetrics()
    extended_results_continuous = extended_metrics_continuous.compute_metrics_for_features(
        pred_features_continuous, target_features_continuous,
        pedal_method='continuous'
    )

    if 'pedal' in extended_results_continuous:
        extended_results_continuous['CPedal'] = extended_results_continuous.pop('pedal')

    overall_continuous = {}
    feature_keys = ['velocity', 'duration', 'ioi', 'CPedal']
    for metric in ['js', 'ia', 'mae', 'mse', 'rmse', 'pearson']:
        values = [extended_results_continuous[f][metric] for f in feature_keys if f in extended_results_continuous and metric in extended_results_continuous[f]]
        if values:
            overall_continuous[metric] = np.mean(values)
    extended_results_continuous['overall'] = overall_continuous

    # Print continuous results
    print()
    header = f"{'Feature':<12} {'JS↓':<10} {'IA↑':<10} {'MAE↓':<10} {'RMSE↓':<10} {'Pearson↑':<10}"
    print(header)
    print("-"*90)
    for feature in ['velocity', 'duration', 'ioi', 'CPedal']:
        if feature in extended_results_continuous:
            r = extended_results_continuous[feature]
            print(f"{feature:<12} {r['js']:<10.4f} {r['ia']:<10.4f} {r['mae']:<10.2f} {r['rmse']:<10.2f} {r['pearson']:<10.4f}")
    print("-"*90)
    if 'overall' in extended_results_continuous:
        r = extended_results_continuous['overall']
        print(f"{'Overall':<12} {r['js']:<10.4f} {r['ia']:<10.4f} {r['mae']:<10.2f} {r['rmse']:<10.2f} {r['pearson']:<10.4f}")
    print("="*90)

    results['continuous'] = {
        'evaluation_type': 'score_to_performance_generation',
        'pedal_method': 'continuous',
        'num_notes': len(pred_features_continuous['velocity']),
        'distribution_metrics': dist_results_continuous,
        'fine_grained_metrics': extended_results_continuous,
    }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='results/pt_evaluation_by_subset')
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--workers-per-gpu', type=int, default=3)
    parser.add_argument('--subset', type=str, choices=['asap', 'pianocore', 'all'], default='all',
                       help='Which subset to evaluate: asap, pianocore, or all')
    args = parser.parse_args()

    print("="*70)
    print("PT Evaluation by Subset")
    print("="*70)
    print(f"Subset: {args.subset}")
    print(f"Computing BOTH binary and continuous pedal metrics")
    print(f"Workers per GPU: {args.workers_per_gpu}")
    print()

    with open(args.config) as f:
        our_config = json.load(f)

    pt_model, pt_config = load_pt_model_and_config()
    print("✅ PT model loaded successfully")

    from src.train.sft_node import PianoCoReNodeSFTDataset, build_work_manifest

    print("\nBuilding test dataset...")

    test_manifest = build_work_manifest(
        metadata_path=our_config.get('metadata_path', 'data/pianocore/metadata.csv'),
        refined_dir=our_config.get('refined_dir', 'data/pianocore/PianoCoRe/refined'),
        split='test',
        block_notes=our_config.get('block_notes', 512),
        overlap_ratio=our_config.get('overlap_ratio', 0.5),
        min_notes=our_config.get('min_notes', 64),
        max_works=our_config.get('max_eval_works'),
    )

    test_dataset = PianoCoReNodeSFTDataset(
        test_manifest,
        split='test',
        shuffle=False,
        seed=our_config['seed'],
        max_performances_per_work=our_config.get('max_eval_performances_per_work'),
        max_windows_per_work=our_config.get('max_eval_windows_per_work'),
    )

    print(f"Test dataset: {len(test_dataset)} samples")

    try:
        if args.subset == 'all':
            # Evaluate on full test set
            source_filter = None
            print("\n" + "="*70)
            print("Evaluating on FULL TEST SET")
            print("="*70)

            (pred_features_binary, target_features_binary,
             pred_features_continuous, target_features_continuous,
             num_samples) = evaluate_pt_multiworker(
                pt_model, pt_config, test_dataset,
                source_filter=source_filter,
                max_samples=args.max_samples,
                workers_per_gpu=args.workers_per_gpu
            )

            results = compute_and_print_metrics(
                pred_features_binary, target_features_binary,
                pred_features_continuous, target_features_continuous
            )
            results['subset'] = 'all'
            results['num_samples'] = num_samples

            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            output_file = output_dir / f"pt_results_all.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)

            print(f"\nResults saved to {output_file}")

        else:
            # Evaluate on specific subset
            print("\n" + "="*70)
            print(f"Evaluating on {args.subset.upper()} SUBSET")
            print("="*70)

            (pred_features_binary, target_features_binary,
             pred_features_continuous, target_features_continuous,
             num_samples) = evaluate_pt_multiworker(
                pt_model, pt_config, test_dataset,
                source_filter=args.subset,
                max_samples=args.max_samples,
                workers_per_gpu=args.workers_per_gpu
            )

            results = compute_and_print_metrics(
                pred_features_binary, target_features_binary,
                pred_features_continuous, target_features_continuous
            )
            results['subset'] = args.subset
            results['num_samples'] = num_samples

            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            output_file = output_dir / f"pt_results_{args.subset}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)

            print(f"\nResults saved to {output_file}")

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
