"""
Compare Hybrid Node results with Pianist Transformer paper (Table 1).
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


# Pianist Transformer Table 1 results (ASAP test set)
PT_PAPER_RESULTS = {
    'Score MIDI': {
        'velocity_js': 0.3569,
        'duration_js': 0.3880,
        'ioi_js': 0.3424,
        'pedal_js': 0.1925,
        'overall_js': 0.3200,
        'overall_ia': 0.5582,
    },
    'VirtuosoNet-ISGN': {
        'velocity_js': 0.2395,
        'duration_js': 0.4049,
        'ioi_js': 0.2929,
        'pedal_js': 0.0829,
        'overall_js': 0.2791,
        'overall_ia': 0.6556,
    },
    'Pianist Transformer': {
        'velocity_js': 0.1417,
        'duration_js': 0.1879,
        'ioi_js': 0.1740,
        'pedal_js': 0.1111,
        'overall_js': 0.1634,
        'overall_ia': 0.8501,
    },
    'Human': {
        'velocity_js': 0.0000,
        'duration_js': 0.0000,
        'ioi_js': 0.0000,
        'pedal_js': 0.0000,
        'overall_js': 0.0000,
        'overall_ia': 1.0000,
    }
}


def load_results(result_path):
    """Load evaluation results from JSON."""
    with open(result_path) as f:
        return json.load(f)


def create_comparison_table(hybrid_results, source='ASAP'):
    """
    Create comparison table with PT paper results.

    Args:
        hybrid_results: Results dict from evaluate_by_source.py
        source: 'ASAP' or 'Other'
    """
    if source not in hybrid_results:
        print(f"Warning: {source} not found in results")
        return

    hybrid_metrics = hybrid_results[source]['metrics']

    print("=" * 100)
    print(f"Comparison with Pianist Transformer Paper (Table 1) - {source} Subset")
    print("=" * 100)
    print()

    # Header
    print(f"{'Model':<25} {'Vel JS↓':<12} {'Dur JS↓':<12} {'IOI JS↓':<12} {'Pedal JS↓':<12} {'Overall JS↓':<12} {'Overall IA↑':<12}")
    print("-" * 100)

    # PT paper results
    for model_name in ['Score MIDI', 'VirtuosoNet-ISGN', 'Pianist Transformer']:
        metrics = PT_PAPER_RESULTS[model_name]
        print(f"{model_name:<25} "
              f"{metrics['velocity_js']:<12.4f} "
              f"{metrics['duration_js']:<12.4f} "
              f"{metrics['ioi_js']:<12.4f} "
              f"{metrics['pedal_js']:<12.4f} "
              f"{metrics['overall_js']:<12.4f} "
              f"{metrics['overall_ia']:<12.4f}")

    print("-" * 100)

    # Our results
    print(f"{'Hybrid Node (Ours)':<25} "
          f"{hybrid_metrics.get('velocity_js', 0):<12.4f} "
          f"{hybrid_metrics.get('duration_js', 0):<12.4f} "
          f"{hybrid_metrics.get('ioi_js', 0):<12.4f} "
          f"{hybrid_metrics.get('pedal_js', 0):<12.4f} "
          f"{hybrid_metrics.get('overall_js', 0):<12.4f} "
          f"{hybrid_metrics.get('overall_ia', 0):<12.4f}")

    print("-" * 100)

    # Human (upper bound)
    metrics = PT_PAPER_RESULTS['Human']
    print(f"{'Human':<25} "
          f"{metrics['velocity_js']:<12.4f} "
          f"{metrics['duration_js']:<12.4f} "
          f"{metrics['ioi_js']:<12.4f} "
          f"{metrics['pedal_js']:<12.4f} "
          f"{metrics['overall_js']:<12.4f} "
          f"{metrics['overall_ia']:<12.4f}")

    print("=" * 100)
    print()

    # Analysis
    print("Key Observations:")
    pt_overall_js = PT_PAPER_RESULTS['Pianist Transformer']['overall_js']
    hybrid_overall_js = hybrid_metrics.get('overall_js', 0)

    if hybrid_overall_js < pt_overall_js:
        improvement = (pt_overall_js - hybrid_overall_js) / pt_overall_js * 100
        print(f"✓ Hybrid Node achieves {improvement:.1f}% better Overall JS than PT")
    else:
        gap = (hybrid_overall_js - pt_overall_js) / pt_overall_js * 100
        print(f"✗ Hybrid Node is {gap:.1f}% worse on Overall JS than PT")

    print()
    print("⚠️ Important Notes:")
    print(f"  - Hybrid Node evaluated on {source} subset from PianoCoRe-A test set")
    if source == 'ASAP':
        print(f"  - Sample size: {hybrid_results[source]['num_samples']} samples (19 works)")
        print(f"  - PT was trained on ASAP with 10B token pretraining")
        print(f"  - Hybrid Node was trained on PianoCoRe-A without pretraining")
        print(f"  - Not a perfectly fair comparison but shows relative performance")
    else:
        print(f"  - These are non-ASAP works, showing model generalization")


def plot_comparison(hybrid_results, output_path='results/comparison_plot.png'):
    """Create visualization comparing with PT paper."""

    if 'ASAP' not in hybrid_results:
        print("No ASAP results to plot")
        return

    hybrid_metrics = hybrid_results['ASAP']['metrics']

    # Prepare data for plotting
    features = ['Velocity', 'Duration', 'IOI', 'Pedal', 'Overall']
    metric_keys = ['velocity_js', 'duration_js', 'ioi_js', 'pedal_js', 'overall_js']

    # Extract values
    score_vals = [PT_PAPER_RESULTS['Score MIDI'][k] for k in metric_keys]
    virtuoso_vals = [PT_PAPER_RESULTS['VirtuosoNet-ISGN'][k] for k in metric_keys]
    pt_vals = [PT_PAPER_RESULTS['Pianist Transformer'][k] for k in metric_keys]
    hybrid_vals = [hybrid_metrics.get(k, 0) for k in metric_keys]

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(features))
    width = 0.2

    ax.bar(x - 1.5*width, score_vals, width, label='Score MIDI', alpha=0.8)
    ax.bar(x - 0.5*width, virtuoso_vals, width, label='VirtuosoNet-ISGN', alpha=0.8)
    ax.bar(x + 0.5*width, pt_vals, width, label='Pianist Transformer', alpha=0.8)
    ax.bar(x + 1.5*width, hybrid_vals, width, label='Hybrid Node (Ours)', alpha=0.8, color='red')

    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('JS Divergence (↓ lower is better)', fontsize=12)
    ax.set_title('EPR Performance Comparison: Hybrid Node vs Pianist Transformer\n(ASAP Subset)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(features)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, required=True, help='Results JSON from evaluate_by_source.py')
    parser.add_argument('--output-plot', type=str, default='results/comparison_with_pt.png')
    args = parser.parse_args()

    # Load results
    results = load_results(args.results)

    # Print comparison table
    print()
    create_comparison_table(results, source='ASAP')
    print()

    # Also show full test set
    if 'Other' in results:
        print()
        create_comparison_table(results, source='Other')
        print()

    # Create plot
    plot_comparison(results, args.output_plot)


if __name__ == '__main__':
    main()
