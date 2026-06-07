"""
Evaluate Hybrid Node model on test set using PT metrics.
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

    # Load checkpoint - try multiple formats
    if checkpoint_path.is_dir():
        # Try safetensors first
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
        # Direct file path
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

        # Extract predictions - model returns Seq2SeqLMOutput with logits
        continuous_pred = outputs.logits  # (B, N, 7)

        return continuous_pred.cpu().numpy(), attention_mask.cpu().numpy()


def evaluate_model(model, dataset, config, device='cuda', max_batches=None, pedal_method='binary'):
    """
    Evaluate model on dataset.

    Returns:
        pred_features: dict of concatenated predictions
        target_features: dict of concatenated targets
    """
    model = model.to(device)
    model.eval()

    # Collect all predictions and targets
    all_pred_continuous = []
    all_target_continuous = []
    all_masks = []

    print(f"Evaluating on {len(dataset)} samples...")

    # Create data collator
    collator = NodeSFTDataCollator(pitch_pad_id=config.get('pitch_pad_id', 128))

    batch_size = config.get('per_device_eval_batch_size', 2)
    num_batches = (len(dataset) + batch_size - 1) // batch_size

    if max_batches:
        num_batches = min(num_batches, max_batches)

    for i in tqdm(range(num_batches), desc="Running inference"):
        # Get batch
        batch_samples = []
        for j in range(batch_size):
            idx = i * batch_size + j
            if idx >= len(dataset):
                break
            batch_samples.append(dataset[idx])

        if not batch_samples:
            break

        # Collate batch using collator
        batch = collator(batch_samples)

        # Predict
        pred_continuous, mask = predict_batch(model, batch, device)
        target_continuous = batch['labels_continuous'].numpy()

        all_pred_continuous.append(pred_continuous)
        all_target_continuous.append(target_continuous)
        all_masks.append(mask)

    # Extract features batch by batch to avoid dimension mismatch
    print("Extracting features...")

    pred_features_list = {'velocity': [], 'duration': [], 'ioi': [], 'pedal': []}
    target_features_list = {'velocity': [], 'duration': [], 'ioi': [], 'pedal': []}

    for pred_batch, target_batch, mask_batch in zip(all_pred_continuous, all_target_continuous, all_masks):
        # Extract features for this batch
        use_joint_config = (pedal_method == 'binary')
        pred_feats = extract_features_from_continuous(pred_batch, mask_batch, pedal_as_joint_config=use_joint_config)
        target_feats = extract_features_from_continuous(target_batch, mask_batch, pedal_as_joint_config=use_joint_config)

        # Append to lists
        for key in pred_features_list.keys():
            pred_features_list[key].append(pred_feats[key])
            target_features_list[key].append(target_feats[key])

    # Concatenate all features
    pred_features = {k: np.concatenate(v) for k, v in pred_features_list.items()}
    target_features = {k: np.concatenate(v) for k, v in target_features_list.items()}

    print(f"Collected {len(pred_features['velocity'])} total notes")

    return pred_features, target_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Training config JSON')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--output', type=str, default='evaluation_results.json', help='Output JSON file')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--max-samples', type=int, default=None, help='Max samples to evaluate (for quick test)')
    parser.add_argument('--pedal-method', type=str, default='binary', choices=['binary', 'continuous'],
                       help='Pedal evaluation method: binary (PT-style 16 configs) or continuous (128 bins)')
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = json.load(f)

    print("=" * 60)
    print("Hybrid Node EPR Evaluation")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print(f"Pedal method: {args.pedal_method}")
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

    # Determine max batches if max_samples specified
    max_batches = None
    if args.max_samples:
        batch_size = config.get('per_device_eval_batch_size', 2)
        max_batches = (args.max_samples + batch_size - 1) // batch_size
        print(f"Limiting to {args.max_samples} samples ({max_batches} batches)")

    # Evaluate
    pred_features, target_features = evaluate_model(
        model, test_dataset, config,
        device=args.device,
        max_batches=max_batches,
        pedal_method=args.pedal_method
    )

    print()
    print("Computing metrics...")

    # Compute metrics
    metrics_calculator = EPRMetrics(bins=100)
    use_joint_config = (args.pedal_method == 'binary')
    results = metrics_calculator.compute_metrics(pred_features, target_features, pedal_is_joint_config=use_joint_config)

    # Print results
    print()
    print(metrics_calculator.format_results(results))
    print()

    # Add feature statistics
    results['feature_stats'] = {
        'pred': {
            'velocity_mean': float(np.mean(pred_features['velocity'])),
            'velocity_std': float(np.std(pred_features['velocity'])),
            'duration_mean': float(np.mean(pred_features['duration'])),
            'duration_std': float(np.std(pred_features['duration'])),
            'ioi_mean': float(np.mean(pred_features['ioi'])),
            'ioi_std': float(np.std(pred_features['ioi'])),
            'pedal_mean': float(np.mean(pred_features['pedal'])),
            'pedal_std': float(np.std(pred_features['pedal'])),
        },
        'target': {
            'velocity_mean': float(np.mean(target_features['velocity'])),
            'velocity_std': float(np.std(target_features['velocity'])),
            'duration_mean': float(np.mean(target_features['duration'])),
            'duration_std': float(np.std(target_features['duration'])),
            'ioi_mean': float(np.mean(target_features['ioi'])),
            'ioi_std': float(np.std(target_features['ioi'])),
            'pedal_mean': float(np.mean(target_features['pedal'])),
            'pedal_std': float(np.std(target_features['pedal'])),
        }
    }

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_path}")
    print()
    print("Evaluation complete!")


if __name__ == '__main__':
    main()
