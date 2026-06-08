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
from src.model.hybrid_pianoformer import HybridPianoT5Gemma, HybridPianoT5GemmaConfig, HybridPianoTransformer
from src.train.sft_node import PianoCoReNodeSFTDataset, build_work_manifest, NodeSFTDataCollator


def load_model(checkpoint_path, config):
    """Load trained model from checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    print(f"Loading model from {checkpoint_path}")

    backbone_type = config.get('backbone_type', 't5').lower()
    model_config = HybridPianoT5GemmaConfig(
        backbone_type=backbone_type,
        continuous_dim=config.get('continuous_dim', 7),
        max_time_ms=config.get('max_time_ms', 10000.0),
        pedal_output_activation=config.get('pedal_output_activation', 'sigmoid'),
        pitch_pad_id=config.get('pitch_pad_id', 128),
        encoder_layers_num=config.get('encoder_layers_num', 10),
        decoder_layers_num=config.get('decoder_layers_num', 2),
        gpt_layers_num=config.get('gpt_layers_num'),
        bert_layers_num=config.get('bert_layers_num'),
        max_position_embeddings=config.get('max_position_embeddings', 4096),
        attention_dropout=config.get('attention_dropout', 0.0),
        hidden_size=config.get('hidden_size', 768),
        intermediate_size=config.get('intermediate_size', 3072),
        num_attention_heads=config.get('num_attention_heads', 8),
        num_key_value_heads=config.get('num_key_value_heads', 4),
        head_dim=config.get('head_dim', 128),
    )

    if backbone_type in {'t5', 't5gemma'}:
        model = HybridPianoT5Gemma(model_config)
    elif backbone_type in {'bert', 'gpt'}:
        model = HybridPianoTransformer(model_config)
    else:
        raise ValueError(f"Unsupported backbone_type: {backbone_type}")

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


def evaluate_model(model, dataset, config, device='cuda', max_batches=None):
    """
    Evaluate model on dataset.
    Computes BOTH binary and continuous pedal metrics in one pass.

    Returns:
        pred_features_binary, target_features_binary,
        pred_features_continuous, target_features_continuous
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

    # Extract features batch by batch - compute BOTH binary and continuous
    print("Extracting features (binary and continuous pedal methods)...")

    # Binary pedal
    pred_features_binary_list = {'velocity': [], 'duration': [], 'ioi': [], 'pedal': []}
    target_features_binary_list = {'velocity': [], 'duration': [], 'ioi': [], 'pedal': []}

    # Continuous pedal
    pred_features_continuous_list = {'velocity': [], 'duration': [], 'ioi': [], 'pedal': []}
    target_features_continuous_list = {'velocity': [], 'duration': [], 'ioi': [], 'pedal': []}

    for pred_batch, target_batch, mask_batch in zip(all_pred_continuous, all_target_continuous, all_masks):
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

    # Concatenate all features
    pred_features_binary = {k: np.concatenate(v) for k, v in pred_features_binary_list.items()}
    target_features_binary = {k: np.concatenate(v) for k, v in target_features_binary_list.items()}

    pred_features_continuous = {k: np.concatenate(v) for k, v in pred_features_continuous_list.items()}
    target_features_continuous = {k: np.concatenate(v) for k, v in target_features_continuous_list.items()}

    print(f"Collected {len(pred_features_binary['velocity'])} total notes")

    return (pred_features_binary, target_features_binary,
            pred_features_continuous, target_features_continuous)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Training config JSON')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--output', type=str, default='evaluation_results.json', help='Output JSON file')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--max-samples', type=int, default=None, help='Max samples to evaluate (for quick test)')
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
    print("Computing BOTH binary and continuous pedal metrics")
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

    # Evaluate - gets both binary and continuous features
    (pred_features_binary, target_features_binary,
     pred_features_continuous, target_features_continuous) = evaluate_model(
        model, test_dataset, config,
        device=args.device,
        max_batches=max_batches
    )

    print()
    print("Computing metrics for binary and continuous pedal methods...")

    # Binary pedal metrics
    print("\n" + "=" * 60)
    print("Binary Pedal Method (BPedal)")
    print("=" * 60)

    metrics_calculator_binary = EPRMetrics(bins=100)
    results_binary = metrics_calculator_binary.compute_metrics(
        pred_features_binary, target_features_binary,
        pedal_is_joint_config=True
    )
    print(metrics_calculator_binary.format_results(results_binary))

    # Continuous pedal metrics
    print("\n" + "=" * 60)
    print("Continuous Pedal Method (CPedal)")
    print("=" * 60)

    metrics_calculator_continuous = EPRMetrics(bins=100)
    results_continuous = metrics_calculator_continuous.compute_metrics(
        pred_features_continuous, target_features_continuous,
        pedal_is_joint_config=False
    )
    print(metrics_calculator_continuous.format_results(results_continuous))

    # Combine results
    results = {
        'binary': {
            'pedal_method': 'binary',
            'metrics': results_binary,
            'feature_stats': {
                'pred': {
                    'velocity_mean': float(np.mean(pred_features_binary['velocity'])),
                    'velocity_std': float(np.std(pred_features_binary['velocity'])),
                    'duration_mean': float(np.mean(pred_features_binary['duration'])),
                    'duration_std': float(np.std(pred_features_binary['duration'])),
                    'ioi_mean': float(np.mean(pred_features_binary['ioi'])),
                    'ioi_std': float(np.std(pred_features_binary['ioi'])),
                    'pedal_mean': float(np.mean(pred_features_binary['pedal'])),
                    'pedal_std': float(np.std(pred_features_binary['pedal'])),
                },
                'target': {
                    'velocity_mean': float(np.mean(target_features_binary['velocity'])),
                    'velocity_std': float(np.std(target_features_binary['velocity'])),
                    'duration_mean': float(np.mean(target_features_binary['duration'])),
                    'duration_std': float(np.std(target_features_binary['duration'])),
                    'ioi_mean': float(np.mean(target_features_binary['ioi'])),
                    'ioi_std': float(np.std(target_features_binary['ioi'])),
                    'pedal_mean': float(np.mean(target_features_binary['pedal'])),
                    'pedal_std': float(np.std(target_features_binary['pedal'])),
                }
            }
        },
        'continuous': {
            'pedal_method': 'continuous',
            'metrics': results_continuous,
            'feature_stats': {
                'pred': {
                    'velocity_mean': float(np.mean(pred_features_continuous['velocity'])),
                    'velocity_std': float(np.std(pred_features_continuous['velocity'])),
                    'duration_mean': float(np.mean(pred_features_continuous['duration'])),
                    'duration_std': float(np.std(pred_features_continuous['duration'])),
                    'ioi_mean': float(np.mean(pred_features_continuous['ioi'])),
                    'ioi_std': float(np.std(pred_features_continuous['ioi'])),
                    'pedal_mean': float(np.mean(pred_features_continuous['pedal'])),
                    'pedal_std': float(np.std(pred_features_continuous['pedal'])),
                },
                'target': {
                    'velocity_mean': float(np.mean(target_features_continuous['velocity'])),
                    'velocity_std': float(np.std(target_features_continuous['velocity'])),
                    'duration_mean': float(np.mean(target_features_continuous['duration'])),
                    'duration_std': float(np.std(target_features_continuous['duration'])),
                    'ioi_mean': float(np.mean(target_features_continuous['ioi'])),
                    'ioi_std': float(np.std(target_features_continuous['ioi'])),
                    'pedal_mean': float(np.mean(target_features_continuous['pedal'])),
                    'pedal_std': float(np.std(target_features_continuous['pedal'])),
                }
            }
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
