"""
Analyze the distribution of model-predicted pedal values.
This helps understand whether the model predicts binary patterns or continuous half-pedal.
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

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
        if safetensors_path.exists():
            from safetensors.torch import load_file
            state_dict = load_file(str(safetensors_path))
            model.load_state_dict(state_dict)
        else:
            pytorch_path = checkpoint_path / "pytorch_model.bin"
            checkpoint = torch.load(pytorch_path, map_location='cpu')
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


def collect_pedal_predictions(model, dataset, config, device, max_samples=None):
    """Collect all pedal predictions and targets."""
    model = model.to(device)
    model.eval()

    all_pred_pedal = []
    all_target_pedal = []

    collator = NodeSFTDataCollator(pitch_pad_id=config.get('pitch_pad_id', 128))
    batch_size = config.get('per_device_eval_batch_size', 2)

    num_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    num_batches = (num_samples + batch_size - 1) // batch_size

    print(f"Collecting pedal predictions from {num_samples} samples...")

    for i in tqdm(range(num_batches), desc="Inference"):
        batch_samples = []
        for j in range(batch_size):
            idx = i * batch_size + j
            if idx >= num_samples:
                break
            batch_samples.append(dataset[idx])

        if not batch_samples:
            break

        batch = collator(batch_samples)
        pred_continuous, mask = predict_batch(model, batch, device)
        target_continuous = batch['labels_continuous'].numpy()

        # Extract pedal (indices 3:7)
        pred_pedal = pred_continuous[:, :, 3:7]  # (B, N, 4)
        target_pedal = target_continuous[:, :, 3:7]  # (B, N, 4)
        mask_np = mask  # (B, N)

        # Flatten and filter by mask
        for b in range(pred_pedal.shape[0]):
            valid_mask = mask_np[b] == 1
            pred_valid = pred_pedal[b][valid_mask].flatten()
            target_valid = target_pedal[b][valid_mask].flatten()

            all_pred_pedal.append(pred_valid)
            all_target_pedal.append(target_valid)

    # Concatenate all
    all_pred_pedal = np.concatenate(all_pred_pedal)
    all_target_pedal = np.concatenate(all_target_pedal)

    return all_pred_pedal, all_target_pedal


def analyze_distribution(values, name):
    """Analyze value distribution."""
    # Clamp to [0, 1] range
    values = np.clip(values, 0, 1)

    # Convert to [0, 127] scale for analysis
    values_scaled = values * 127.0

    # Count binary vs half-pedal
    binary_mask = np.isin(np.round(values_scaled), [0, 127])
    binary_count = np.sum(binary_mask)
    half_pedal_count = len(values) - binary_count

    # Statistics
    unique_vals = len(np.unique(np.round(values_scaled)))

    stats = {
        'total_samples': len(values),
        'binary_count': int(binary_count),
        'binary_pct': float(binary_count / len(values) * 100),
        'half_pedal_count': int(half_pedal_count),
        'half_pedal_pct': float(half_pedal_count / len(values) * 100),
        'unique_values': int(unique_vals),
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'median': float(np.median(values)),
        'percentiles': {
            '25': float(np.percentile(values, 25)),
            '50': float(np.percentile(values, 50)),
            '75': float(np.percentile(values, 75)),
            '90': float(np.percentile(values, 90)),
            '95': float(np.percentile(values, 95)),
            '99': float(np.percentile(values, 99)),
        }
    }

    return stats


def plot_distributions(pred_values, target_values, output_path):
    """Plot pedal value distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Histogram - Target
    ax = axes[0, 0]
    ax.hist(target_values * 127, bins=128, range=(0, 127), alpha=0.7, color='blue', edgecolor='black')
    ax.set_xlabel('Pedal Value (0-127)')
    ax.set_ylabel('Count')
    ax.set_title('Target Pedal Distribution')
    ax.grid(True, alpha=0.3)

    # Histogram - Prediction
    ax = axes[0, 1]
    ax.hist(pred_values * 127, bins=128, range=(0, 127), alpha=0.7, color='red', edgecolor='black')
    ax.set_xlabel('Pedal Value (0-127)')
    ax.set_ylabel('Count')
    ax.set_title('Predicted Pedal Distribution')
    ax.grid(True, alpha=0.3)

    # Scatter plot (sampled)
    ax = axes[1, 0]
    sample_size = min(50000, len(pred_values))
    indices = np.random.choice(len(pred_values), sample_size, replace=False)
    ax.scatter(target_values[indices] * 127, pred_values[indices] * 127,
               alpha=0.1, s=1, color='purple')
    ax.plot([0, 127], [0, 127], 'k--', linewidth=2, label='Perfect prediction')
    ax.set_xlabel('Target Pedal Value')
    ax.set_ylabel('Predicted Pedal Value')
    ax.set_title(f'Target vs Predicted (n={sample_size})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 127)
    ax.set_ylim(0, 127)

    # CDF comparison
    ax = axes[1, 1]
    target_sorted = np.sort(target_values * 127)
    pred_sorted = np.sort(pred_values * 127)
    target_cdf = np.arange(1, len(target_sorted) + 1) / len(target_sorted)
    pred_cdf = np.arange(1, len(pred_sorted) + 1) / len(pred_sorted)
    ax.plot(target_sorted, target_cdf, label='Target', linewidth=2, color='blue')
    ax.plot(pred_sorted, pred_cdf, label='Predicted', linewidth=2, color='red')
    ax.set_xlabel('Pedal Value (0-127)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Distribution Functions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Training config JSON')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--output-dir', type=str, default='results/pedal_analysis', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--max-samples', type=int, default=None, help='Max samples to analyze')
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    print("=" * 70)
    print("Pedal Prediction Distribution Analysis")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
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

    print(f"Test dataset: {len(test_dataset)} samples")

    # Load model
    model = load_model(args.checkpoint, config)

    # Collect predictions
    pred_pedal, target_pedal = collect_pedal_predictions(
        model, test_dataset, config, args.device, args.max_samples
    )

    print(f"\nCollected {len(pred_pedal)} pedal values")

    # Analyze distributions
    print("\nAnalyzing target distribution...")
    target_stats = analyze_distribution(target_pedal, "Target")

    print("\nAnalyzing prediction distribution...")
    pred_stats = analyze_distribution(pred_pedal, "Prediction")

    # Print results
    print("\n" + "=" * 70)
    print("TARGET PEDAL DISTRIBUTION")
    print("=" * 70)
    print(f"Total samples: {target_stats['total_samples']:,}")
    print(f"Binary (0 or 127): {target_stats['binary_count']:,} ({target_stats['binary_pct']:.1f}%)")
    print(f"Half-pedal: {target_stats['half_pedal_count']:,} ({target_stats['half_pedal_pct']:.1f}%)")
    print(f"Unique values: {target_stats['unique_values']}")
    print(f"Mean: {target_stats['mean']:.3f}, Std: {target_stats['std']:.3f}")
    print(f"Range: [{target_stats['min']:.3f}, {target_stats['max']:.3f}]")

    print("\n" + "=" * 70)
    print("PREDICTED PEDAL DISTRIBUTION")
    print("=" * 70)
    print(f"Total samples: {pred_stats['total_samples']:,}")
    print(f"Binary (0 or 127): {pred_stats['binary_count']:,} ({pred_stats['binary_pct']:.1f}%)")
    print(f"Half-pedal: {pred_stats['half_pedal_count']:,} ({pred_stats['half_pedal_pct']:.1f}%)")
    print(f"Unique values: {pred_stats['unique_values']}")
    print(f"Mean: {pred_stats['mean']:.3f}, Std: {pred_stats['std']:.3f}")
    print(f"Range: [{pred_stats['min']:.3f}, {pred_stats['max']:.3f}]")

    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)

    if pred_stats['binary_pct'] > target_stats['binary_pct'] + 10:
        print("⚠️  Model predictions are MORE binary than target")
        print(f"   Prediction: {pred_stats['binary_pct']:.1f}% binary")
        print(f"   Target: {target_stats['binary_pct']:.1f}% binary")
        print("   → Model tends to predict extreme values (0 or 1)")
    elif pred_stats['binary_pct'] < target_stats['binary_pct'] - 10:
        print("✅ Model predictions are MORE continuous than target")
        print(f"   Prediction: {pred_stats['binary_pct']:.1f}% binary")
        print(f"   Target: {target_stats['binary_pct']:.1f}% binary")
        print("   → Model learned to predict half-pedal values")
    else:
        print("✅ Model binary ratio matches target distribution")
        print(f"   Prediction: {pred_stats['binary_pct']:.1f}% binary")
        print(f"   Target: {target_stats['binary_pct']:.1f}% binary")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'target_stats': target_stats,
        'prediction_stats': pred_stats,
    }

    output_json = output_dir / 'pedal_distribution_analysis.json'
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_json}")

    # Generate plots
    print("\nGenerating plots...")
    plot_path = output_dir / 'pedal_distributions.png'
    plot_distributions(pred_pedal, target_pedal, plot_path)

    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
