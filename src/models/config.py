"""Model configurations for supported transformer models.

This module defines configuration for GPT-2 and Pythia models used
in temporal horizon detection experiments.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ModelConfig:
    """Configuration for a transformer model.

    Attributes:
        name: Model name/identifier
        family: Model family (gpt2, pythia)
        num_layers: Number of transformer layers
        hidden_size: Hidden dimension size
        num_heads: Number of attention heads
        vocab_size: Vocabulary size
        max_position_embeddings: Maximum sequence length
        hub_name: HuggingFace Hub model name
    """

    name: str
    family: str
    num_layers: int
    hidden_size: int
    num_heads: int
    vocab_size: int
    max_position_embeddings: int
    hub_name: str


# GPT-2 model configurations
GPT2_SMALL = ModelConfig(
    name="gpt2-small",
    family="gpt2",
    num_layers=12,
    hidden_size=768,
    num_heads=12,
    vocab_size=50257,
    max_position_embeddings=1024,
    hub_name="gpt2",
)

GPT2_MEDIUM = ModelConfig(
    name="gpt2-medium",
    family="gpt2",
    num_layers=24,
    hidden_size=1024,
    num_heads=16,
    vocab_size=50257,
    max_position_embeddings=1024,
    hub_name="gpt2-medium",
)

GPT2_LARGE = ModelConfig(
    name="gpt2-large",
    family="gpt2",
    num_layers=36,
    hidden_size=1280,
    num_heads=20,
    vocab_size=50257,
    max_position_embeddings=1024,
    hub_name="gpt2-large",
)

GPT2_XL = ModelConfig(
    name="gpt2-xl",
    family="gpt2",
    num_layers=48,
    hidden_size=1600,
    num_heads=25,
    vocab_size=50257,
    max_position_embeddings=1024,
    hub_name="gpt2-xl",
)

# Pythia model configurations
PYTHIA_70M = ModelConfig(
    name="pythia-70m",
    family="pythia",
    num_layers=6,
    hidden_size=512,
    num_heads=8,
    vocab_size=50304,
    max_position_embeddings=2048,
    hub_name="EleutherAI/pythia-70m",
)

PYTHIA_160M = ModelConfig(
    name="pythia-160m",
    family="pythia",
    num_layers=12,
    hidden_size=768,
    num_heads=12,
    vocab_size=50304,
    max_position_embeddings=2048,
    hub_name="EleutherAI/pythia-160m",
)

PYTHIA_410M = ModelConfig(
    name="pythia-410m",
    family="pythia",
    num_layers=24,
    hidden_size=1024,
    num_heads=16,
    vocab_size=50304,
    max_position_embeddings=2048,
    hub_name="EleutherAI/pythia-410m",
)

PYTHIA_1B = ModelConfig(
    name="pythia-1b",
    family="pythia",
    num_layers=16,
    hidden_size=2048,
    num_heads=8,
    vocab_size=50304,
    max_position_embeddings=2048,
    hub_name="EleutherAI/pythia-1b",
)

# Model registry
MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "gpt2": GPT2_SMALL,
    "gpt2-small": GPT2_SMALL,
    "gpt2-medium": GPT2_MEDIUM,
    "gpt2-large": GPT2_LARGE,
    "gpt2-xl": GPT2_XL,
    "pythia-70m": PYTHIA_70M,
    "pythia-160m": PYTHIA_160M,
    "pythia-410m": PYTHIA_410M,
    "pythia-1b": PYTHIA_1B,
}


def get_model_config(model_name: str) -> ModelConfig:
    """Get configuration for a model.

    Args:
        model_name: Name of the model

    Returns:
        ModelConfig object.

    Raises:
        ValueError: If model_name is not recognized

    Example:
        >>> config = get_model_config("gpt2")
        >>> config.num_layers
        12
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name]


def list_available_models() -> List[str]:
    """List all available model names.

    Returns:
        List of model names.

    Example:
        >>> models = list_available_models()
        >>> "gpt2" in models
        True
    """
    return list(MODEL_REGISTRY.keys())


def get_layer_range(model_name: str, layer_subset: Optional[str] = None) -> List[int]:
    """Get layer indices based on subset specification.

    Args:
        model_name: Name of the model
        layer_subset: Subset specification ("all", "early", "middle", "late", or None)

    Returns:
        List of layer indices.

    Example:
        >>> layers = get_layer_range("gpt2", "early")
        >>> max(layers) < 4
        True
    """
    config = get_model_config(model_name)
    total_layers = config.num_layers

    if layer_subset is None or layer_subset == "all":
        return list(range(total_layers))
    elif layer_subset == "early":
        # First 1/3 of layers
        return list(range(0, total_layers // 3))
    elif layer_subset == "middle":
        # Middle 1/3 of layers
        start = total_layers // 3
        end = 2 * total_layers // 3
        return list(range(start, end))
    elif layer_subset == "late":
        # Last 1/3 of layers
        return list(range(2 * total_layers // 3, total_layers))
    else:
        raise ValueError(
            f"Unknown layer subset: {layer_subset}. "
            "Valid options: ['all', 'early', 'middle', 'late']"
        )
