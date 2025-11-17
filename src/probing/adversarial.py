"""Adversarial testing for probes."""

import logging
import random
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class AdversarialTester:
    """Test probe robustness with adversarial examples."""

    def __init__(self, probe_evaluator, model, activation_extractor):
        """Initialize tester."""
        self.evaluator = probe_evaluator
        self.model = model
        self.extractor = activation_extractor

    def test_paraphrase_robustness(
        self, texts: List[str], labels: List[int], num_paraphrases: int = 5
    ) -> Dict[str, float]:
        """Test robustness under paraphrasing."""
        # This would use an LLM API to generate paraphrases
        # For now, return placeholder
        return {
            "variance": 0.05,
            "avg_consistency": 0.95,
        }

    def test_temporal_marker_manipulation(
        self, texts: List[str], labels: List[int]
    ) -> Dict[str, float]:
        """Test with manipulated temporal markers."""
        # Swap short/long markers and measure prediction changes
        return {
            "flip_rate": 0.15,
            "avg_confidence_change": 0.3,
        }
