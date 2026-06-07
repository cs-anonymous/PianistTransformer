"""
EPR Evaluation Metrics
Implements objective metrics for Expressive Performance Rendering (EPR) evaluation.
Based on Pianist Transformer (ICML 2025) evaluation protocol.
"""

import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path


class EPRMetrics:
    """
    Compute EPR evaluation metrics following Pianist Transformer paper.

    Metrics:
    - Jensen-Shannon Divergence (JS Div): Distribution similarity, lower is better
    - Intersection Area (IA): Distribution overlap, higher is better
    """

    def __init__(self, bins: int = 100):
        """
        Args:
            bins: Number of bins for histogram computation
        """
        self.bins = bins

    def _normalize_histogram(self, data: np.ndarray, bins: int, range_min: float = None, range_max: float = None) -> np.ndarray:
        """
        Create normalized histogram from data.

        Args:
            data: 1D array of values
            bins: Number of bins
            range_min: Minimum value for histogram range
            range_max: Maximum value for histogram range

        Returns:
            Normalized histogram (sums to 1)
        """
        if len(data) == 0:
            return np.zeros(bins)

        # Determine range
        if range_min is None:
            range_min = np.min(data)
        if range_max is None:
            range_max = np.max(data)

        # Handle edge case where all values are the same
        if range_max == range_min:
            hist = np.zeros(bins)
            hist[bins // 2] = 1.0
            return hist

        # Create histogram
        hist, _ = np.histogram(data, bins=bins, range=(range_min, range_max), density=False)

        # Normalize to probability distribution
        hist = hist.astype(float)
        total = np.sum(hist)
        if total > 0:
            hist = hist / total

        # Add small epsilon to avoid log(0) in KL divergence
        hist = hist + 1e-10
        hist = hist / np.sum(hist)

        return hist

    def js_divergence(self, pred: np.ndarray, target: np.ndarray,
                      range_min: float = None, range_max: float = None) -> float:
        """
        Compute Jensen-Shannon Divergence between two distributions.

        JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M), where M = 0.5 * (P + Q)

        Args:
            pred: Predicted values (1D array)
            target: Target values (1D array)
            range_min: Minimum value for histogram range (uses data min if None)
            range_max: Maximum value for histogram range (uses data max if None)

        Returns:
            JS divergence (0 = identical, higher = more different)
        """
        # Determine common range from both distributions
        if range_min is None:
            range_min = min(np.min(pred), np.min(target))
        if range_max is None:
            range_max = max(np.max(pred), np.max(target))

        # Create histograms with common range
        p_hist = self._normalize_histogram(pred, self.bins, range_min, range_max)
        q_hist = self._normalize_histogram(target, self.bins, range_min, range_max)

        # Compute JS divergence using scipy
        js = jensenshannon(p_hist, q_hist, base=2)

        # Return squared value (standard formulation)
        return js ** 2

    def intersection_area(self, pred: np.ndarray, target: np.ndarray,
                          range_min: float = None, range_max: float = None) -> float:
        """
        Compute intersection area between two distributions.

        IA(P, Q) = sum(min(P[i], Q[i])) for all bins i

        Args:
            pred: Predicted values (1D array)
            target: Target values (1D array)
            range_min: Minimum value for histogram range
            range_max: Maximum value for histogram range

        Returns:
            Intersection area (1 = identical, 0 = no overlap)
        """
        # Determine common range
        if range_min is None:
            range_min = min(np.min(pred), np.min(target))
        if range_max is None:
            range_max = max(np.max(pred), np.max(target))

        # Create histograms
        p_hist = self._normalize_histogram(pred, self.bins, range_min, range_max)
        q_hist = self._normalize_histogram(target, self.bins, range_min, range_max)

        # Compute intersection
        intersection = np.sum(np.minimum(p_hist, q_hist))

        return float(intersection)

    def compute_metrics(self, pred_features: Dict[str, np.ndarray],
                       target_features: Dict[str, np.ndarray],
                       feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
                       pedal_is_joint_config: bool = True) -> Dict[str, float]:
        """
        Compute all metrics for EPR evaluation.

        Args:
            pred_features: Dictionary with keys ['velocity', 'duration', 'ioi', 'pedal']
                          Each value is a 1D numpy array
            target_features: Dictionary with same keys as pred_features
            feature_ranges: Optional dict of (min, max) tuples for each feature
            pedal_is_joint_config: If True, pedal is joint configs [0,15], use bins=16
                                  If False, pedal is flattened values [0,127], use bins=100

        Returns:
            Dictionary with metrics:
            - 'velocity_js', 'duration_js', 'ioi_js', 'pedal_js', 'overall_js'
            - 'velocity_ia', 'duration_ia', 'ioi_ia', 'pedal_ia', 'overall_ia'
        """
        features = ['velocity', 'duration', 'ioi', 'pedal']
        results = {}

        js_scores = []
        ia_scores = []

        for feature in features:
            if feature not in pred_features or feature not in target_features:
                print(f"Warning: {feature} not found in inputs, skipping")
                continue

            pred = pred_features[feature]
            target = target_features[feature]

            # Get range if provided
            range_min, range_max = None, None
            if feature_ranges and feature in feature_ranges:
                range_min, range_max = feature_ranges[feature]

            # Special handling for pedal with joint configs
            if feature == 'pedal' and pedal_is_joint_config:
                # For joint configs [0, 15], use fixed range and 16 bins
                range_min, range_max = 0, 16
                old_bins = self.bins
                self.bins = 16
                js = self.js_divergence(pred, target, range_min, range_max)
                ia = self.intersection_area(pred, target, range_min, range_max)
                self.bins = old_bins
            else:
                # Compute metrics with default bins
                js = self.js_divergence(pred, target, range_min, range_max)
                ia = self.intersection_area(pred, target, range_min, range_max)

            results[f'{feature}_js'] = js
            results[f'{feature}_ia'] = ia

            js_scores.append(js)
            ia_scores.append(ia)

        # Compute overall metrics (average across features)
        if js_scores:
            results['overall_js'] = np.mean(js_scores)
        if ia_scores:
            results['overall_ia'] = np.mean(ia_scores)

        return results

    def format_results(self, results: Dict[str, float]) -> str:
        """
        Format results as a readable table.

        Args:
            results: Dictionary from compute_metrics()

        Returns:
            Formatted string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("EPR Evaluation Results (Pianist Transformer Metrics)")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"{'Feature':<15} {'JS Div (↓)':<15} {'IA (↑)':<15}")
        lines.append("-" * 60)

        features = ['velocity', 'duration', 'ioi', 'pedal']
        for feature in features:
            js_key = f'{feature}_js'
            ia_key = f'{feature}_ia'
            if js_key in results and ia_key in results:
                lines.append(f"{feature.capitalize():<15} {results[js_key]:<15.4f} {results[ia_key]:<15.4f}")

        lines.append("-" * 60)
        if 'overall_js' in results and 'overall_ia' in results:
            lines.append(f"{'Overall':<15} {results['overall_js']:<15.4f} {results['overall_ia']:<15.4f}")
        lines.append("=" * 60)

        return "\n".join(lines)


def extract_features_from_continuous(continuous: np.ndarray,
                                     attention_mask: Optional[np.ndarray] = None,
                                     pedal_as_joint_config: bool = True) -> Dict[str, np.ndarray]:
    """
    Extract EPR features from continuous tensor (B, N, 7).

    Continuous format: [ioi_norm, duration_norm, velocity_norm, pedal_0, pedal_25, pedal_50, pedal_75]

    Args:
        continuous: (B, N, 7) or (N, 7) array
        attention_mask: (B, N) or (N,) boolean mask, True for valid notes
        pedal_as_joint_config: If True, convert pedal to joint configurations (PT-style)
                              If False, flatten pedal samples (old-style)

    Returns:
        Dictionary with denormalized features:
        - velocity: [0, 127]
        - duration: milliseconds
        - ioi: milliseconds
        - pedal: [0, 15] (joint configs) or [0, 127] (flattened)
    """
    # Handle batch dimension
    if continuous.ndim == 3:
        B, N, D = continuous.shape
        continuous = continuous.reshape(-1, D)
        if attention_mask is not None:
            attention_mask = attention_mask.reshape(-1)
    elif continuous.ndim == 2:
        N, D = continuous.shape
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {continuous.shape}")

    # Apply mask if provided
    if attention_mask is not None:
        continuous = continuous[attention_mask > 0]

    # Extract normalized features
    ioi_norm = continuous[:, 0]
    duration_norm = continuous[:, 1]
    velocity_norm = continuous[:, 2]
    pedal_norm = continuous[:, 3:7]  # 4 pedal values

    # Denormalize
    # Time: log1p normalization with max_time_ms = 10000
    max_time_ms = 10000.0
    ioi_ms = np.expm1(ioi_norm * np.log1p(max_time_ms))
    duration_ms = np.expm1(duration_norm * np.log1p(max_time_ms))

    # Velocity and pedal: [0, 1] -> [0, 127]
    velocity = velocity_norm * 127.0
    pedal = pedal_norm * 127.0

    # Process pedal according to PT paper method
    if pedal_as_joint_config:
        # PT-style: binarize and convert to joint configurations (16 classes)
        # Binarize: > 64 -> 1, <= 64 -> 0
        pedal_binary = (pedal > 64.0).astype(np.int32)  # (N, 4) with values {0, 1}

        # Convert 4-bit pattern to config index: [b0,b1,b2,b3] -> b0*8 + b1*4 + b2*2 + b3
        pedal_configs = (
            pedal_binary[:, 0] * 8 +
            pedal_binary[:, 1] * 4 +
            pedal_binary[:, 2] * 2 +
            pedal_binary[:, 3]
        )  # (N,) with values in [0, 15]

        pedal_result = pedal_configs
    else:
        # Old-style: flatten all 4 samples
        pedal_result = pedal.flatten()

    return {
        'velocity': velocity,
        'duration': duration_ms,
        'ioi': ioi_ms,
        'pedal': pedal_result
    }


if __name__ == "__main__":
    # Test the metrics with synthetic data
    print("Testing EPR Metrics Implementation...")
    print()

    # Create synthetic ground truth
    np.random.seed(42)
    n_notes = 1000

    gt_velocity = np.random.normal(80, 15, n_notes).clip(0, 127)
    gt_duration = np.random.lognormal(5.5, 0.8, n_notes).clip(50, 5000)
    gt_ioi = np.random.lognormal(5.3, 0.9, n_notes).clip(50, 5000)
    gt_pedal = np.random.uniform(0, 127, n_notes * 4)

    target_features = {
        'velocity': gt_velocity,
        'duration': gt_duration,
        'ioi': gt_ioi,
        'pedal': gt_pedal
    }

    # Test Case 1: Perfect prediction (should get JS~0, IA~1)
    print("Test 1: Perfect Prediction")
    metrics = EPRMetrics(bins=100)
    results = metrics.compute_metrics(target_features, target_features)
    print(metrics.format_results(results))
    print()

    # Test Case 2: Slightly noisy prediction
    print("Test 2: Slightly Noisy Prediction")
    pred_velocity = gt_velocity + np.random.normal(0, 5, n_notes)
    pred_duration = gt_duration * np.random.normal(1.0, 0.1, n_notes)
    pred_ioi = gt_ioi * np.random.normal(1.0, 0.1, n_notes)
    pred_pedal = gt_pedal + np.random.normal(0, 10, n_notes * 4)

    pred_features = {
        'velocity': pred_velocity.clip(0, 127),
        'duration': pred_duration.clip(50, 5000),
        'ioi': pred_ioi.clip(50, 5000),
        'pedal': pred_pedal.clip(0, 127)
    }

    results = metrics.compute_metrics(pred_features, target_features)
    print(metrics.format_results(results))
    print()

    # Test Case 3: Poor prediction (uniform random)
    print("Test 3: Poor Prediction (Random)")
    pred_features_random = {
        'velocity': np.random.uniform(0, 127, n_notes),
        'duration': np.random.uniform(50, 5000, n_notes),
        'ioi': np.random.uniform(50, 5000, n_notes),
        'pedal': np.random.uniform(0, 127, n_notes * 4)
    }

    results = metrics.compute_metrics(pred_features_random, target_features)
    print(metrics.format_results(results))
    print()

    print("✓ Metrics implementation test complete!")
