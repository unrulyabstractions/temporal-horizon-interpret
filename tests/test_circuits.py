"""Tests for circuits module."""

import pytest
from src.circuits.attribution import compute_head_importance


def test_compute_head_importance():
    """Test head importance computation."""
    results = {
        "layer_10_head_0": {"effect_size": 0.5},
        "layer_10_head_1": {"effect_size": 0.3},
        "layer_11_head_0": {"effect_size": 0.8},
    }
    
    important = compute_head_importance(results, top_k=2)
    assert len(important) == 2
    assert important[0][1] > important[1][1]  # Sorted by importance
