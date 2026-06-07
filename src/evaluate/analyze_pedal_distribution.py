"""
Analyze pedal value distribution in test set.
Check if pedal values are predominantly binary or continuous.
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.train.sft_node import PianoCoReNodeSFTDataset, build_work_manifest


def analyze_pedal_distribution(dataset, dataset_name):
    """Analyze pedal value distribution in dataset."""
    print(f"\n{'='*60}")
    print(f"Analyzing Pedal Distribution: {dataset_name}")
    print(f"{'='*60}\n")

    all_pedal_values = []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        # IMPORTANT: Use labels_continuous (target/performance), not continuous (input/score)
        labels_continuous = np.array(sample['labels_continuous'])  # (N, 7)
        pedal = labels_continuous[:, 3:7]  # (N, 4)

        # Denormalize: [0, 1] -> [0, 127]
        pedal_denorm = pedal * 127.0

        all_pedal_values.extend(pedal_denorm.flatten().tolist())

    all_pedal_values = np.array(all_pedal_values)

    # Statistics
    print(f"Total pedal samples: {len(all_pedal_values):,}")
    print(f"Min: {all_pedal_values.min():.2f}")
    print(f"Max: {all_pedal_values.max():.2f}")
    print(f"Mean: {all_pedal_values.mean():.2f}")
    print(f"Std: {all_pedal_values.std():.2f}")
    print()

    # Check binary ratio
    binary_threshold = 10  # Values < 10 or > 117 considered "near binary"
    near_zero = np.sum(all_pedal_values < binary_threshold)
    near_max = np.sum(all_pedal_values > (127 - binary_threshold))
    binary_count = near_zero + near_max
    binary_ratio = binary_count / len(all_pedal_values)

    print(f"Near-zero (<{binary_threshold}): {near_zero:,} ({near_zero/len(all_pedal_values)*100:.1f}%)")
    print(f"Near-max (>{127-binary_threshold}): {near_max:,} ({near_max/len(all_pedal_values)*100:.1f}%)")
    print(f"Binary ratio: {binary_ratio*100:.1f}%")
    print()

    # Exact value counts (top 10)
    value_counts = Counter(np.round(all_pedal_values).astype(int))
    print("Top 10 most common values:")
    for value, count in value_counts.most_common(10):
        print(f"  {value:3d}: {count:8,} ({count/len(all_pedal_values)*100:.2f}%)")
    print()

    # Unique value count
    unique_values = len(np.unique(np.round(all_pedal_values)))
    print(f"Unique values (rounded): {unique_values}")
    print()

    # Histogram
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Full range histogram
    axes[0].hist(all_pedal_values, bins=128, range=(0, 127), edgecolor='black', linewidth=0.5)
    axes[0].set_xlabel('Pedal Value')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'{dataset_name} - Full Range')
    axes[0].grid(alpha=0.3)

    # Middle range histogram (10-117) to see half-pedal usage
    middle_values = all_pedal_values[(all_pedal_values >= binary_threshold) &
                                     (all_pedal_values <= 127 - binary_threshold)]
    if len(middle_values) > 0:
        axes[1].hist(middle_values, bins=50, edgecolor='black', linewidth=0.5)
        axes[1].set_xlabel('Pedal Value')
        axes[1].set_ylabel('Count')
        axes[1].set_title(f'{dataset_name} - Middle Range ({binary_threshold}-{127-binary_threshold})')
        axes[1].grid(alpha=0.3)

        print(f"Half-pedal usage (middle range): {len(middle_values):,} ({len(middle_values)/len(all_pedal_values)*100:.2f}%)")
    else:
        axes[1].text(0.5, 0.5, 'No half-pedal values', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title(f'{dataset_name} - Middle Range (empty)')

    plt.tight_layout()
    output_path = ROOT_DIR / 'results' / f'pedal_distribution_{dataset_name.lower().replace(" ", "_")}.png'
    plt.savefig(output_path, dpi=150)
    print(f"Histogram saved to {output_path}")

    # Conclusion
    print(f"\n{'='*60}")
    if binary_ratio > 0.95:
        print("CONCLUSION: Pedal is predominantly BINARY")
        print("  -> PT-style binary evaluation is appropriate")
    elif binary_ratio > 0.80:
        print("CONCLUSION: Pedal is mostly binary with some half-pedal")
        print("  -> Should report both binary and continuous evaluation")
    else:
        print("CONCLUSION: Pedal contains significant HALF-PEDAL usage")
        print("  -> Continuous evaluation is more appropriate")
    print(f"{'='*60}\n")

    return {
        'total_samples': len(all_pedal_values),
        'mean': float(all_pedal_values.mean()),
        'std': float(all_pedal_values.std()),
        'binary_ratio': float(binary_ratio),
        'unique_values': int(unique_values),
        'half_pedal_ratio': float(len(middle_values) / len(all_pedal_values)) if len(middle_values) > 0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

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

    # Analyze full test set
    stats = analyze_pedal_distribution(test_dataset, "Full Test Set")

    # Save stats
    output_file = ROOT_DIR / 'results' / 'pedal_distribution_stats.json'
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nStatistics saved to {output_file}")


if __name__ == '__main__':
    main()
