"""Detect divergence between stated and internal temporal horizons."""

import logging
import numpy as np
from scipy.spatial.distance import cosine
from scipy import stats

logger = logging.getLogger(__name__)


class DivergenceDetector:
    """Detect mismatches between stated and internal horizons."""

    def __init__(self, probe_evaluator):
        """Initialize detector."""
        self.evaluator = probe_evaluator

    def detect_divergence(
        self, activations: np.ndarray, stated_labels: np.ndarray, threshold: float = 0.7
    ) -> dict:
        """Detect divergence between stated and predicted horizons."""
        # Get probe predictions
        predicted_probs = self.evaluator.predict_proba(activations)
        predicted_labels = np.argmax(predicted_probs, axis=1)
        
        # Find mismatches
        mismatches = predicted_labels != stated_labels
        mismatch_rate = mismatches.mean()
        
        # High confidence mismatches
        max_probs = np.max(predicted_probs, axis=1)
        high_conf_mismatches = mismatches & (max_probs > threshold)
        
        return {
            "mismatch_rate": float(mismatch_rate),
            "num_mismatches": int(mismatches.sum()),
            "high_confidence_mismatches": int(high_conf_mismatches.sum()),
            "mismatch_indices": np.where(mismatches)[0].tolist(),
        }

    def compute_activation_similarity(
        self, act1: np.ndarray, act2: np.ndarray
    ) -> float:
        """Compute cosine similarity between activations."""
        return 1 - cosine(act1.flatten(), act2.flatten())

    def statistical_test(
        self, short_activations: np.ndarray, long_activations: np.ndarray
    ) -> dict:
        """Perform statistical tests for distribution differences."""
        # Flatten activations
        short_flat = short_activations.reshape(len(short_activations), -1)
        long_flat = long_activations.reshape(len(long_activations), -1)
        
        # Compute mean activations
        short_mean = short_flat.mean(axis=0)
        long_mean = long_flat.mean(axis=0)
        
        # T-test
        t_stat, p_value = stats.ttest_ind(short_flat, long_flat)
        
        return {
            "mean_difference": float(np.linalg.norm(short_mean - long_mean)),
            "p_value": float(np.mean(p_value)),
            "significant_dimensions": int((p_value < 0.05).sum()),
        }
