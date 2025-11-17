"""Tests for models module."""

import pytest
import torch
from src.models.config import get_model_config, list_available_models
from src.probing.probe import LinearProbe, MLPProbe, create_probe


def test_model_config():
    """Test model configuration."""
    config = get_model_config("gpt2")
    assert config.num_layers == 12
    assert config.hidden_size == 768


def test_list_models():
    """Test model listing."""
    models = list_available_models()
    assert "gpt2" in models
    assert "pythia-160m" in models


def test_linear_probe():
    """Test linear probe."""
    probe = LinearProbe(hidden_size=768)
    x = torch.randn(10, 768)
    logits = probe(x)
    assert logits.shape == (10, 2)


def test_mlp_probe():
    """Test MLP probe."""
    probe = MLPProbe(hidden_size=768, num_layers=2)
    x = torch.randn(10, 768)
    logits = probe(x)
    assert logits.shape == (10, 2)


def test_create_probe():
    """Test probe factory."""
    probe = create_probe("linear", hidden_size=768)
    assert isinstance(probe, LinearProbe)
