"""
Evaluate Hybrid Node model with breakdown by data source.
Specifically extract ASAP subset for comparison with PT paper.
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.evaluate.epr_metrics import EPRMetrics, extract_features_from_continuous
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

    # Load checkpoint
    if checkpoint_path.is_dir():
        safetensors_path = checkpoint_path / "model.safetensors"
        pytorch_path = checkpoint_path / "pytorch_model.bin"

        if safetensors_path.exists():
            from safetensors.torch import load_file
            state_dict = load_file(str(safetensors_path))
            model.load_state_dict(state_dict)
        elif pytorch_path.exists():
            checkpoint = torch.load(pytorch_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
    else:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

    model.eval()
    return model


def load_metadata_mapping(metadata_path):
    """
    Load PianoCoRe metadata and create mapping from score path to source dataset.

    Returns:
        dict: {score_path: 'ASAP' or 'Other'}
    """
    df = pd.read_csv(metadata_path)
    df_a = df[df['tier_a'].fillna(False).astype(bool)]
    df_test = df_a[df_a['split'] == 'test']

    mapping = {}
    for _, row in df_test.iterrows():
        score_path = row['refined_score_midi_path']
        source = row.get('performance_dataset', 'Unknown')
        if score_path not in mapping:
            mapping[score_path] = source

    return mapping


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

        continuous_pred = outputs['continuous_pred']
        return continuous_pred.cpu().numpy(), attention_mask.cpu().numpy()


def evaluate_by_source(model, dataset, metadata_mapping, config, device='cuda'):
    """
    Evaluate model and group results by data source.

    Returns:
        dict: {source: {'pred_features': {...}, 'target_features': {...}}}
    """
    model = model.to(device)
    model.eval()

    # Group predictions by source
    source_data = defaultdict(lambda: {'pred': [], 'target': [], 'mask': []})

    # Create data collator
    collator = NodeSFTDataCollator(pitch_pad_id=config.get('pitch_pad_id', 128))

    print(f"Evaluating on {len(dataset)} samples...")
    batch_size = config.get('per_device_eval_batch_size', 2)
    num_batches = (len(dataset) + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="Running inference"):
        batch_samples = []
        batch_sources = []

        for j in range(batch_size):
            idx = i * batch_size + j
            if idx >= len(dataset):
                break

            sample = dataset[idx]
            batch_samples.append(sample)

            # Get source from work path
            work_item = dataset.items[dataset.sample_to_work[idx]]
            score_path = work_item['score_source']
            source = metadata_mapping.get(score_path, 'Unknown')
            # Simplify: ASAP vs Other
            source_label = 'ASAP' if source == 'ASAP' else 'Other'
            batch_sources.append(source_label)

        if not batch_samples:
            break

        # Collate batch using collator
        batch = collator(batch_samples)

        # Predict
        pred_continuous, mask = predict_batch(model, batch, device)
        target_continuous = batch['labels_continuous'].numpy()

        # Group by source
        for b_idx, source in enumerate(batch_sources):
            source_data[source]['pred'].append(pred_continuous[b_idx:b_idx+1])
            source_data[source]['target'].append(target_continuous[b_idx:b_idx+1])
            source_data[source]['mask'].append(mask[b_idx:b_idx+1])

    # Concatenate and extract features for each source
    results = {}
    for source, data in source_data.items():
        pred = np.concatenate(data['pred'], axis=0)
        target = np.concatenate(data['target'], axis=0)
        mask = np.concatenate(data['mask'], axis=0)

        print(f"\n{source}: {pred.shape[0]} samples")

        pred_features = extract_features_from_continuous(pred, mask)
        target_features = extract_features_from_continuous(target, mask)

        results[source] = {
            'pred_features': pred_features,
            'target_features': target_features,
            'num_samples': pred.shape[0]
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='results/eval_by_source.json')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = json.load(f)

    print("=" * 70)
    print("Hybrid Node Evaluation - Breakdown by Data Source")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print()

    # Load metadata mapping
    print("Loading metadata mapping...")
    metadata_mapping = load_metadata_mapping(config['metadata_path'])
    asap_count = sum(1 for v in metadata_mapping.values() if v == 'ASAP')
    print(f"Test set: {len(metadata_mapping)} unique scores")
    print(f"  - ASAP: {asap_count}")
    print(f"  - Other: {len(metadata_mapping) - asap_count}")
    print()

    # Build test dataset
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

    test_dataset = PianoCoReNodeSFTDataset(
        test_manifest,
        split='test',
        shuffle=False,
        seed=config['seed'],
        max_performances_per_work=config.get('max_eval_performances_per_work'),
        max_windows_per_work=config.get('max_eval_windows_per_work'),
    )

    print(f"Test dataset size: {len(test_dataset)} samples")
    print()

    # Load model
    model = load_model(args.checkpoint, config)

    # Evaluate by source
    source_results = evaluate_by_source(model, test_dataset, metadata_mapping, config, device=args.device)

    # Compute metrics for each source
    metrics_calculator = EPRMetrics(bins=100)
    final_results = {}

    for source, data in source_results.items():
        print(f"\n{'=' * 70}")
        print(f"Results for: {source}")
        print('=' * 70)

        metrics = metrics_calculator.compute_metrics(
            data['pred_features'],
            data['target_features']
        )

        print(metrics_calculator.format_results(metrics))

        final_results[source] = {
            'metrics': metrics,
            'num_samples': data['num_samples']
        }

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"\n\nResults saved to {output_path}")

    # Print summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY: ASAP vs Other")
    print("=" * 70)
    if 'ASAP' in final_results and 'Other' in final_results:
        print(f"\n{'Metric':<20} {'ASAP':<15} {'Other':<15} {'Difference':<15}")
        print("-" * 70)
        for metric in ['overall_js', 'overall_ia', 'velocity_js', 'duration_js', 'ioi_js', 'pedal_js']:
            asap_val = final_results['ASAP']['metrics'].get(metric, 0)
            other_val = final_results['Other']['metrics'].get(metric, 0)
            diff = asap_val - other_val
            print(f"{metric:<20} {asap_val:<15.4f} {other_val:<15.4f} {diff:+.4f}")


if __name__ == '__main__':
    main()
