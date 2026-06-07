"""
Complete evaluation with split test sets and extended metrics.
Evaluates on ASAP subset and PianoCoRe-only subset separately.
Reports both JS/IA and MAE/MSE/RMSE/Pearson for all dimensions.
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.evaluate.epr_metrics import EPRMetrics, extract_features_from_continuous
from src.evaluate.epr_metrics_extended import ExtendedEPRMetrics
from src.model.hybrid_pianoformer import HybridPianoT5Gemma, HybridPianoT5GemmaConfig
from src.train.sft_node import PianoCoReNodeSFTDataset, build_work_manifest, NodeSFTDataCollator


def load_model(checkpoint_path, config):
    """Load trained model from checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    print(f"Loading model from {checkpoint_path}")

    model_config = HybridPianoT5GemmaConfig(
        continuous_dim=config.get('continuous_dim', 7),
        max_time_ms=config.get('max_time_ms', 10000.0),
        pitch_pad_id=config.get('pitch_pad_id', 128),
        encoder_layers_num=config.get('encoder_layers_num', 10),
        decoder_layers_num=config.get('decoder_layers_num', 2),
        hidden_size=config.get('hidden_size', 768),
        intermediate_size=config.get('intermediate_size', 3072),
        num_attention_heads=config.get('num_attention_heads', 8),
        num_key_value_heads=config.get('num_key_value_heads', 4),
        head_dim=config.get('head_dim', 128),
    )

    model = HybridPianoT5Gemma(model_config)

    if checkpoint_path.is_dir():
        safetensors_path = checkpoint_path / "model.safetensors"
        pytorch_path = checkpoint_path / "pytorch_model.bin"

        if safetensors_path.exists():
            print(f"Loading from safetensors: {safetensors_path}")
            from safetensors.torch import load_file
            state_dict = load_file(str(safetensors_path))
            model.load_state_dict(state_dict)
        elif pytorch_path.exists():
            print(f"Loading from pytorch_model.bin: {pytorch_path}")
            checkpoint = torch.load(pytorch_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            raise FileNotFoundError(f"No model file found in {checkpoint_path}")
    else:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

    model.eval()
    return model


def predict_batch(model, batch, device):
    """Run model inference on a batch."""
    with torch.no_grad():
        pitch_ids = batch['pitch_ids'].to(device)
        continuous = batch['continuous'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(
            pitch_ids=pitch_ids,
            continuous=continuous,
            attention_mask=attention_mask
        )

        continuous_pred = outputs.logits
        return continuous_pred.cpu().numpy(), attention_mask.cpu().numpy()


def identify_asap_sample(sample):
    """Check if a sample is from ASAP dataset."""
    performance_dataset = sample.get('performance_dataset', '')
    score_dataset = sample.get('score_dataset', '')

    return ('asap' in performance_dataset.lower() or
            'asap' in score_dataset.lower())


def evaluate_subset(model, dataset, config, device, pedal_method, subset_name):
    """Evaluate model on a dataset subset."""
    model = model.to(device)
    model.eval()

    all_pred_continuous = []
    all_target_continuous = []
    all_masks = []

    print(f"\nEvaluating {subset_name} subset ({len(dataset)} samples)...")

    collator = NodeSFTDataCollator(pitch_pad_id=config.get('pitch_pad_id', 128))
    batch_size = config.get('per_device_eval_batch_size', 2)
    num_batches = (len(dataset) + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc=f"{subset_name} inference"):
        batch_samples = []
        for j in range(batch_size):
            idx = i * batch_size + j
            if idx >= len(dataset):
                break
            batch_samples.append(dataset[idx])

        if not batch_samples:
            break

        batch = collator(batch_samples)
        pred_continuous, mask = predict_batch(model, batch, device)
        target_continuous = batch['labels_continuous'].numpy()

        all_pred_continuous.append(pred_continuous)
        all_target_continuous.append(target_continuous)
        all_masks.append(mask)

    print(f"Extracting features for {subset_name}...")

    pred_features_list = {'velocity': [], 'duration': [], 'ioi': [], 'pedal': []}
    target_features_list = {'velocity': [], 'duration': [], 'ioi': [], 'pedal': []}

    use_joint_config = (pedal_method == 'binary')

    for pred_batch, target_batch, mask_batch in zip(all_pred_continuous, all_target_continuous, all_masks):
        pred_feats = extract_features_from_continuous(pred_batch, mask_batch, pedal_as_joint_config=use_joint_config)
        target_feats = extract_features_from_continuous(target_batch, mask_batch, pedal_as_joint_config=use_joint_config)

        for key in pred_features_list.keys():
            pred_features_list[key].append(pred_feats[key])
            target_features_list[key].append(target_feats[key])

    pred_features = {k: np.concatenate(v) for k, v in pred_features_list.items()}
    target_features = {k: np.concatenate(v) for k, v in target_features_list.items()}

    print(f"Collected {len(pred_features['velocity'])} notes from {subset_name}")

    return pred_features, target_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Training config JSON')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--output-dir', type=str, default='results/complete_evaluation', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--pedal-method', type=str, default='continuous',
                       choices=['binary', 'continuous'],
                       help='Pedal evaluation method')
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    print("=" * 70)
    print("COMPLETE EPR EVALUATION - Split Test Sets + Extended Metrics")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print(f"Pedal method: {args.pedal_method}")
    print()

    # Build full test dataset
    print("Building test dataset...")
    test_manifest = build_work_manifest(
        metadata_path=config['metadata_path'],
        refined_dir=config['refined_dir'],
        split='test',
        block_notes=config['block_notes'],
        overlap_ratio=config['overlap_ratio'],
        min_notes=config['min_notes'],
        max_works=config.get('max_eval_works'),
    )

    full_test_dataset = PianoCoReNodeSFTDataset(
        test_manifest,
        split='test',
        shuffle=False,
        seed=config['seed'],
        max_performances_per_work=config.get('max_eval_performances_per_work'),
        max_windows_per_work=config.get('max_eval_windows_per_work'),
    )

    print(f"Full test dataset: {len(full_test_dataset)} samples")

    # Split dataset into ASAP and PianoCoRe-only
    # We need to check the work metadata, not individual samples
    asap_indices = []
    pianocore_indices = []

    print("\nSplitting test set by data source...")

    # Load metadata to identify ASAP sources
    import pandas as pd
    metadata_df = pd.read_csv(config['metadata_path'])

    for idx in range(len(full_test_dataset)):
        # Get the work item this sample belongs to
        item_idx = 0
        for i, size in enumerate(full_test_dataset.cumulative_sizes):
            if idx < size:
                item_idx = i
                break

        work_item = full_test_dataset.items[item_idx]
        score_source = work_item['score_source']

        # Check if this work is from ASAP
        work_metadata = metadata_df[metadata_df['refined_score_midi_path'] == score_source]
        if len(work_metadata) > 0:
            perf_dataset = work_metadata['performance_dataset'].iloc[0] if 'performance_dataset' in work_metadata.columns else ''
            score_dataset = work_metadata['score_dataset'].iloc[0] if 'score_dataset' in work_metadata.columns else ''

            is_asap = ('asap' in str(perf_dataset).lower() or 'asap' in str(score_dataset).lower())

            if is_asap:
                asap_indices.append(idx)
            else:
                pianocore_indices.append(idx)
        else:
            pianocore_indices.append(idx)

    print(f"ASAP subset: {len(asap_indices)} samples")
    print(f"PianoCoRe-only subset: {len(pianocore_indices)} samples")

    # Load model
    model = load_model(args.checkpoint, config)

    # Evaluate both subsets
    results = {
        'pedal_method': args.pedal_method,
        'checkpoint': str(args.checkpoint),
        'subsets': {}
    }

    use_joint_config = (args.pedal_method == 'binary')

    for subset_name, indices in [('ASAP', asap_indices), ('PianoCoRe-only', pianocore_indices)]:
        if len(indices) == 0:
            print(f"\nSkipping {subset_name} (no samples)")
            continue

        # Create subset
        subset_dataset = torch.utils.data.Subset(full_test_dataset, indices)

        # Evaluate
        pred_features, target_features = evaluate_subset(
            model, subset_dataset, config, args.device,
            args.pedal_method, subset_name
        )

        # Compute distribution metrics (JS/IA)
        print(f"\nComputing distribution metrics for {subset_name}...")
        dist_metrics = EPRMetrics(bins=100)
        dist_results = dist_metrics.compute_metrics(
            pred_features, target_features,
            pedal_is_joint_config=use_joint_config
        )

        # Compute fine-grained metrics (MAE/MSE/RMSE/Pearson)
        print(f"Computing fine-grained metrics for {subset_name}...")
        extended_metrics = ExtendedEPRMetrics()

        # For binary pedal method, use 16 bins; for continuous, use default
        pedal_method_arg = 'binary' if use_joint_config else 'continuous'
        extended_results = extended_metrics.compute_metrics_for_features(
            pred_features, target_features,
            pedal_method=pedal_method_arg
        )

        # Combine results
        results['subsets'][subset_name] = {
            'num_samples': len(indices),
            'num_notes': len(pred_features['velocity']),
            'distribution_metrics': dist_results,
            'fine_grained_metrics': extended_results,
        }

        # Print summary
        print(f"\n{'='*70}")
        print(f"{subset_name} RESULTS")
        print(f"{'='*70}")
        print(dist_metrics.format_results(dist_results))
        print()
        print(extended_metrics.format_results(extended_results))
        print()

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"results_{args.pedal_method}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
