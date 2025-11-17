"""Statistical analysis utilities."""

import numpy as np
from scipy import stats


def compute_confidence_intervals(data: np.ndarray, confidence: float = 0.95) -> tuple:
    """Compute confidence intervals."""
    mean = np.mean(data)
    sem = stats.sem(data)
    interval = sem * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
    return mean - interval, mean + interval


def bootstrap_metric(metric_fn, data, n_bootstrap: int = 1000) -> dict:
    """Bootstrap a metric."""
    scores = []
    for _ in range(n_bootstrap):
        sample_idx = np.random.choice(len(data), len(data), replace=True)
        sample = data[sample_idx]
        scores.append(metric_fn(sample))
    
    return {
        "mean": np.mean(scores),
        "std": np.std(scores),
        "ci_lower": np.percentile(scores, 2.5),
        "ci_upper": np.percentile(scores, 97.5),
    }


def permutation_test(group1: np.ndarray, group2: np.ndarray, n_permutations: int = 1000) -> float:
    """Perform permutation test."""
    observed_diff = np.mean(group1) - np.mean(group2)
    combined = np.concatenate([group1, group2])
    
    null_diffs = []
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_group1 = combined[:len(group1)]
        perm_group2 = combined[len(group1):]
        null_diffs.append(np.mean(perm_group1) - np.mean(perm_group2))
    
    p_value = np.mean(np.abs(null_diffs) >= np.abs(observed_diff))
    return p_value
