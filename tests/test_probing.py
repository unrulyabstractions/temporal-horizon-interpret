"""Tests for probing module."""

import pytest
import numpy as np
from src.probing.probe import LinearProbe
from src.probing.evaluator import ProbeEvaluator


def test_probe_evaluator():
    """Test probe evaluator."""
    probe = LinearProbe(768)
    evaluator = ProbeEvaluator(probe)
    
    # Test prediction
    x = np.random.randn(10, 768)
    preds = evaluator.predict(x)
    assert preds.shape == (10,)
    assert all(p in [0, 1] for p in preds)


def test_evaluation_metrics():
    """Test evaluation metrics."""
    probe = LinearProbe(768)
    evaluator = ProbeEvaluator(probe)
    
    x = np.random.randn(10, 768)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    
    metrics = evaluator.evaluate(x, y)
    assert "accuracy" in metrics
    assert "f1" in metrics
    assert 0 <= metrics["accuracy"] <= 1
