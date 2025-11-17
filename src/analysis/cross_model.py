"""Cross-model validation utilities."""

import logging
import numpy as np

logger = logging.getLogger(__name__)


def compare_models(results1: dict, results2: dict) -> dict:
    """Compare results across models."""
    return {
        "accuracy_diff": results1.get("accuracy", 0) - results2.get("accuracy", 0),
        "correlation": 0.85,  # Placeholder
    }


def compute_transfer_score(probe1, probe2, test_data) -> float:
    """Compute how well probe transfers across models."""
    # Placeholder for transfer learning evaluation
    return 0.75
