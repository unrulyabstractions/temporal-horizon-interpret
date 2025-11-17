"""Model loading utilities using TransformerLens.

This module provides functions to load GPT-2 and Pythia models using
the TransformerLens library for mechanistic interpretability.
"""

import logging
from typing import Optional, Union

import torch
from transformer_lens import HookedTransformer

from .config import MODEL_REGISTRY, get_model_config

logger = logging.getLogger(__name__)


def load_model(
    model_name: str,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
    **kwargs,
) -> HookedTransformer:
    """Load a transformer model using TransformerLens.

    Args:
        model_name: Name of model (e.g., "gpt2", "pythia-160m")
        device: Device to load model on ("cpu", "cuda", "cuda:0", etc.)
               If None, automatically selects CUDA if available
        dtype: Data type for model parameters
        **kwargs: Additional arguments passed to HookedTransformer.from_pretrained

    Returns:
        HookedTransformer model instance.

    Raises:
        ValueError: If model_name is not recognized
        RuntimeError: If model loading fails

    Example:
        >>> # model = load_model("gpt2", device="cpu")
        >>> # model.cfg.n_layers
        12
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    config = get_model_config(model_name)

    # Auto-select device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading {model_name} on {device} with dtype {dtype}")

    try:
        model = HookedTransformer.from_pretrained(
            config.hub_name,
            device=device,
            dtype=dtype,
            **kwargs,
        )

        logger.info(
            f"Successfully loaded {model_name}: "
            f"{config.num_layers} layers, {config.hidden_size} hidden size"
        )

        return model

    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise RuntimeError(f"Model loading failed: {e}")


def load_tokenizer(model_name: str):
    """Load tokenizer for a model.

    Args:
        model_name: Name of model

    Returns:
        Tokenizer instance.

    Example:
        >>> # tokenizer = load_tokenizer("gpt2")
        >>> # tokenizer is not None
        True
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")

    config = get_model_config(model_name)

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config.hub_name)
        logger.info(f"Loaded tokenizer for {model_name}")
        return tokenizer

    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise


def get_model_info(model: HookedTransformer) -> dict:
    """Get information about a loaded model.

    Args:
        model: HookedTransformer model instance

    Returns:
        Dictionary with model information.

    Example:
        >>> # model = load_model("gpt2", device="cpu")
        >>> # info = get_model_info(model)
        >>> # "num_layers" in info
        True
    """
    return {
        "model_name": model.cfg.model_name,
        "num_layers": model.cfg.n_layers,
        "hidden_size": model.cfg.d_model,
        "num_heads": model.cfg.n_heads,
        "head_size": model.cfg.d_head,
        "vocab_size": model.cfg.d_vocab,
        "max_position_embeddings": model.cfg.n_ctx,
        "device": str(next(model.parameters()).device),
        "dtype": str(next(model.parameters()).dtype),
        "num_parameters": sum(p.numel() for p in model.parameters()),
    }


def prepare_input(
    text: Union[str, list],
    model: HookedTransformer,
    max_length: Optional[int] = None,
    add_bos: bool = True,
) -> torch.Tensor:
    """Prepare text input for model.

    Args:
        text: Input text (string or list of strings)
        model: HookedTransformer model
        max_length: Maximum sequence length (truncates if longer)
        add_bos: Whether to add BOS token

    Returns:
        Tokenized input tensor.

    Example:
        >>> # model = load_model("gpt2", device="cpu")
        >>> # tokens = prepare_input("Hello world", model)
        >>> # tokens.shape[0] == 1  # batch size
        True
    """
    # Get tokenizer from model
    if isinstance(text, str):
        text = [text]

    # Tokenize
    tokens = model.to_tokens(text, prepend_bos=add_bos)

    # Truncate if needed
    if max_length is not None and tokens.shape[1] > max_length:
        tokens = tokens[:, :max_length]

    return tokens


def count_parameters(model: HookedTransformer) -> dict:
    """Count parameters in model.

    Args:
        model: HookedTransformer model

    Returns:
        Dictionary with parameter counts.

    Example:
        >>> # model = load_model("gpt2", device="cpu")
        >>> # counts = count_parameters(model)
        >>> # counts["total"] > 0
        True
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": total_params - trainable_params,
        "total_millions": total_params / 1e6,
    }


def set_model_device(
    model: HookedTransformer, device: Union[str, torch.device]
) -> HookedTransformer:
    """Move model to specified device.

    Args:
        model: HookedTransformer model
        device: Target device

    Returns:
        Model on new device.

    Example:
        >>> # model = load_model("gpt2", device="cpu")
        >>> # model = set_model_device(model, "cpu")
        >>> # str(next(model.parameters()).device) == "cpu"
        True
    """
    logger.info(f"Moving model to {device}")
    model = model.to(device)
    return model


def enable_grad_checkpointing(model: HookedTransformer) -> HookedTransformer:
    """Enable gradient checkpointing for memory efficiency.

    Args:
        model: HookedTransformer model

    Returns:
        Model with gradient checkpointing enabled.

    Note:
        This trades compute for memory by not storing all intermediate
        activations during backward pass.

    Example:
        >>> # model = load_model("gpt2", device="cpu")
        >>> # model = enable_grad_checkpointing(model)
    """
    logger.info("Enabling gradient checkpointing")

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    else:
        logger.warning("Model does not support gradient checkpointing")

    return model
