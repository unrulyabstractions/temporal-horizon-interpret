"""Tests for dataset module."""

import pytest
from src.dataset.templates import get_templates, format_prompt, BUSINESS_TEMPLATES
from src.dataset.validator import DatasetValidator


def test_get_templates():
    """Test template retrieval."""
    # Get all templates
    all_templates = get_templates()
    assert len(all_templates) > 0
    
    # Get business templates
    business_templates = get_templates("business")
    assert len(business_templates) > 0


def test_format_prompt():
    """Test prompt formatting."""
    template = BUSINESS_TEMPLATES[0]
    prompt = format_prompt(template, "product launch", use_long_horizon=False)
    
    assert "product launch" in prompt
    assert len(prompt) > 0


def test_validator():
    """Test dataset validator."""
    validator = DatasetValidator()
    
    pair = {
        "short_prompt": "Plan for next month",
        "long_prompt": "Plan for next decade"
    }
    
    result = validator.validate_pair(pair)
    assert result["is_valid"] == True
    assert isinstance(result["similarity_score"], float)
