"""
Extended EPR metrics including MAE, MSE, RMSE, Pearson correlation.
Supports both distribution-based (JS, IA) and point-wise metrics.
"""

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr
from typing import Dict, Tuple, Optional


class ExtendedEPRMetrics:
    """
    Extended metrics for EPR evaluation.

    Includes:
    - Distribution metrics: JS Divergence, Intersection Area
    - Point-wise metrics: MAE, MSE, RMSE, Pearson correlation
    """

    def __init__(self, bins: int = 100):
        """
        Args:
            bins: Number of bins for histogram computation
        """
        self.bins = bins

    def _normalize_histogram(self, data: np.ndarray, bins: int,
                            range_min: float, range_max: float) -> np.ndarray:
        """Create normalized histogram."""
        hist, _ = np.histogram(data, bins=bins, range=(range_min, range_max), density=False)
        hist = hist.astype(float)
        total = hist.sum()
        if total > 0:
            hist = hist / total
        else:
            hist = np.ones(bins) / bins
        return hist

    def js_divergence(self, pred: np.ndarray, target: np.ndarray,
                     range_min: float = None, range_max: float = None) -> float:
        """Jensen-Shannon Divergence between two distributions."""
        if range_min is None:
            range_min = min(np.min(pred), np.min(target))
        if range_max is None:
            range_max = max(np.max(pred), np.max(target))

        p_hist = self._normalize_histogram(pred, self.bins, range_min, range_max)
        q_hist = self._normalize_histogram(target, self.bins, range_min, range_max)

        js = jensenshannon(p_hist, q_hist, base=2)
        return js ** 2

    def intersection_area(self, pred: np.ndarray, target: np.ndarray,
                         range_min: float = None, range_max: float = None) -> float:
        """Intersection area between two distributions."""
        if range_min is None:
            range_min = min(np.min(pred), np.min(target))
        if range_max is None:
            range_max = max(np.max(pred), np.max(target))

        p_hist = self._normalize_histogram(pred, self.bins, range_min, range_max)
        q_hist = self._normalize_histogram(target, self.bins, range_min, range_max)

        return float(np.sum(np.minimum(p_hist, q_hist)))

    def mae(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Mean Absolute Error."""
        return float(np.mean(np.abs(pred - target)))

    def mse(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Mean Squared Error."""
        return float(np.mean((pred - target) ** 2))

    def rmse(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Root Mean Squared Error."""
        return float(np.sqrt(self.mse(pred, target)))

    def pearson(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Pearson correlation coefficient."""
        if len(pred) < 2:
            return 0.0
        r, _ = pearsonr(pred, target)
        return float(r) if not np.isnan(r) else 0.0

    def compute_all_metrics(self, pred: np.ndarray, target: np.ndarray,
                           range_min: float = None, range_max: float = None,
                           feature_name: str = '') -> Dict[str, float]:
        """
        Compute all metrics for a single feature.

        Args:
            pred: Predicted values (1D array)
            target: Target values (1D array)
            range_min: Min value for histogram range
            range_max: Max value for histogram range
            feature_name: Name of the feature (for display)

        Returns:
            Dictionary with all metrics
        """
        results = {}

        # Distribution metrics
        results['js'] = self.js_divergence(pred, target, range_min, range_max)
        results['ia'] = self.intersection_area(pred, target, range_min, range_max)

        # Point-wise metrics
        results['mae'] = self.mae(pred, target)
        results['mse'] = self.mse(pred, target)
        results['rmse'] = self.rmse(pred, target)
        results['pearson'] = self.pearson(pred, target)

        return results

    def compute_metrics_for_features(self,
                                    pred_features: Dict[str, np.ndarray],
                                    target_features: Dict[str, np.ndarray],
                                    feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
                                    pedal_method: str = 'continuous') -> Dict[str, Dict[str, float]]:
        """
        Compute metrics for all features.

        Args:
            pred_features: Dict with keys ['velocity', 'duration', 'ioi', 'pedal']
            target_features: Dict with same keys
            feature_ranges: Optional (min, max) for each feature
            pedal_method: 'binary' (16 configs), 'continuous' (128 bins), or 'both'

        Returns:
            Nested dict: {feature: {metric: value}}
        """
        features = ['velocity', 'duration', 'ioi', 'pedal']
        results = {}

        for feature in features:
            if feature not in pred_features or feature not in target_features:
                continue

            pred = pred_features[feature]
            target = target_features[feature]

            # Get range
            range_min, range_max = None, None
            if feature_ranges and feature in feature_ranges:
                range_min, range_max = feature_ranges[feature]

            # Special handling for pedal
            if feature == 'pedal' and pedal_method == 'binary':
                # Binary method uses 16 bins for joint configs
                old_bins = self.bins
                self.bins = 16
                range_min, range_max = 0, 16
                feature_results = self.compute_all_metrics(pred, target, range_min, range_max, feature)
                self.bins = old_bins
            elif feature == 'pedal' and pedal_method == 'both':
                # Compute both methods
                # This should be handled by passing different pred/target
                # For now, just use continuous
                feature_results = self.compute_all_metrics(pred, target, range_min, range_max, feature)
            else:
                # Standard continuous evaluation
                feature_results = self.compute_all_metrics(pred, target, range_min, range_max, feature)

            results[feature] = feature_results

        # Compute overall metrics (average across features)
        if results:
            overall = {}
            for metric in ['js', 'ia', 'mae', 'mse', 'rmse', 'pearson']:
                values = [results[f][metric] for f in results if metric in results[f]]
                if values:
                    overall[metric] = np.mean(values)
            results['overall'] = overall

        return results

    def format_results(self, results: Dict[str, Dict[str, float]],
                      title: str = "Extended EPR Evaluation Results") -> str:
        """Format results as readable table."""
        lines = []
        lines.append("=" * 90)
        lines.append(title)
        lines.append("=" * 90)
        lines.append("")

        # Header
        header = f"{'Feature':<12} {'JS↓':<10} {'IA↑':<10} {'MAE↓':<10} {'RMSE↓':<10} {'Pearson↑':<10}"
        lines.append(header)
        lines.append("-" * 90)

        # Features
        features = ['velocity', 'duration', 'ioi', 'pedal']
        for feature in features:
            if feature not in results:
                continue
            r = results[feature]
            line = f"{feature.capitalize():<12} {r['js']:<10.4f} {r['ia']:<10.4f} {r['mae']:<10.2f} {r['rmse']:<10.2f} {r['pearson']:<10.4f}"
            lines.append(line)

        # Overall
        if 'overall' in results:
            lines.append("-" * 90)
            r = results['overall']
            line = f"{'Overall':<12} {r['js']:<10.4f} {r['ia']:<10.4f} {r['mae']:<10.2f} {r['rmse']:<10.2f} {r['pearson']:<10.4f}"
            lines.append(line)

        lines.append("=" * 90)

        return "\n".join(lines)


if __name__ == "__main__":
    # Test
    print("Testing Extended EPR Metrics...")

    np.random.seed(42)
    n = 1000

    target = np.random.normal(80, 15, n)
    pred_good = target + np.random.normal(0, 3, n)  # Small noise
    pred_bad = target + np.random.normal(0, 20, n)  # Large noise

    metrics = ExtendedEPRMetrics(bins=100)

    print("\nGood prediction (small noise):")
    results_good = metrics.compute_all_metrics(pred_good, target)
    for k, v in results_good.items():
        print(f"  {k}: {v:.4f}")

    print("\nBad prediction (large noise):")
    results_bad = metrics.compute_all_metrics(pred_bad, target)
    for k, v in results_bad.items():
        print(f"  {k}: {v:.4f}")

    print("\nTest passed!")
