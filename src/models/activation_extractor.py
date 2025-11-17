"""Activation extraction from transformer models.

This module provides utilities to extract and cache activations from
specific layers and attention heads using TransformerLens hooks.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import h5py
import numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

logger = logging.getLogger(__name__)


class ActivationExtractor:
    """Extract activations from transformer models.

    Supports extraction from specific layers, attention heads, and token positions
    with efficient caching using HDF5 format.

    Attributes:
        model: HookedTransformer model instance
        device: Device model is on
    """

    def __init__(self, model: HookedTransformer):
        """Initialize activation extractor.

        Args:
            model: HookedTransformer model to extract from

        Example:
            >>> from src.models.model_loader import load_model
            >>> # model = load_model("gpt2", device="cpu")
            >>> # extractor = ActivationExtractor(model)
        """
        self.model = model
        self.device = next(model.parameters()).device
        self.activations_cache = {}

    def extract(
        self,
        text: Union[str, List[str]],
        layers: Optional[List[int]] = None,
        positions: Optional[List[int]] = None,
        component: str = "residual",
    ) -> Dict[str, torch.Tensor]:
        """Extract activations from model.

        Args:
            text: Input text (string or list of strings)
            layers: Layer indices to extract from. If None, extracts from all
            positions: Token positions to extract. If None, extracts all
            component: Component to extract ("residual", "attn", "mlp", "attn_out")

        Returns:
            Dictionary mapping layer names to activation tensors.

        Example:
            >>> # extractor = ActivationExtractor(model)
            >>> # acts = extractor.extract("Hello world", layers=[0, 1])
            >>> # len(acts) == 2
            True
        """
        if layers is None:
            layers = list(range(self.model.cfg.n_layers))

        # Tokenize input
        if isinstance(text, str):
            text = [text]

        tokens = self.model.to_tokens(text, prepend_bos=True)

        # Define hook points based on component
        hook_names = []
        for layer in layers:
            if component == "residual":
                hook_names.append(f"blocks.{layer}.hook_resid_post")
            elif component == "attn":
                hook_names.append(f"blocks.{layer}.attn.hook_result")
            elif component == "attn_out":
                hook_names.append(f"blocks.{layer}.attn.hook_attn_out")
            elif component == "mlp":
                hook_names.append(f"blocks.{layer}.hook_mlp_out")
            else:
                raise ValueError(f"Unknown component: {component}")

        # Storage for activations
        activations = {}

        def hook_fn(activation, hook):
            """Hook function to capture activations."""
            layer_name = hook.name
            if positions is not None:
                # Extract specific positions
                activation = activation[:, positions, :]
            activations[layer_name] = activation.detach().cpu()

        # Run model with hooks
        with torch.no_grad():
            self.model.run_with_hooks(
                tokens,
                fwd_hooks=[(name, hook_fn) for name in hook_names],
            )

        # Reorganize by layer
        layer_activations = {}
        for layer in layers:
            for hook_name, act in activations.items():
                if f"blocks.{layer}." in hook_name:
                    layer_activations[f"layer_{layer}"] = act

        return layer_activations

    def extract_batch(
        self,
        texts: List[str],
        layers: Optional[List[int]] = None,
        batch_size: int = 8,
        positions: Optional[List[int]] = None,
        component: str = "residual",
        position_strategy: str = "last",
    ) -> Dict[str, np.ndarray]:
        """Extract activations for a batch of texts.

        Args:
            texts: List of input texts
            layers: Layer indices to extract from
            batch_size: Batch size for processing
            positions: Token positions to extract
            component: Component to extract
            position_strategy: How to select position ("last", "mean", "first")

        Returns:
            Dictionary mapping layer names to activation arrays [num_texts, hidden_size].

        Example:
            >>> # extractor = ActivationExtractor(model)
            >>> # acts = extractor.extract_batch(["text1", "text2"], layers=[10, 11])
            >>> # "layer_10" in acts
            True
        """
        if layers is None:
            layers = list(range(self.model.cfg.n_layers))

        # Initialize storage
        all_activations = {f"layer_{layer}": [] for layer in layers}

        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting activations"):
            batch_texts = texts[i : i + batch_size]

            # Extract for this batch
            batch_acts = self.extract(
                batch_texts,
                layers=layers,
                positions=positions,
                component=component,
            )

            # Apply position strategy
            for layer_name, acts in batch_acts.items():
                # acts shape: [batch, seq_len, hidden_size]
                if position_strategy == "last":
                    # Take last token
                    acts = acts[:, -1, :]
                elif position_strategy == "mean":
                    # Average over sequence
                    acts = acts.mean(dim=1)
                elif position_strategy == "first":
                    # Take first token
                    acts = acts[:, 0, :]
                else:
                    raise ValueError(f"Unknown position strategy: {position_strategy}")

                all_activations[layer_name].append(acts.numpy())

        # Concatenate all batches
        for layer_name in all_activations:
            all_activations[layer_name] = np.concatenate(
                all_activations[layer_name], axis=0
            )

        logger.info(
            f"Extracted activations for {len(texts)} texts across "
            f"{len(layers)} layers"
        )

        return all_activations

    def save_activations(
        self,
        activations: Dict[str, np.ndarray],
        save_path: Union[str, Path],
        metadata: Optional[Dict] = None,
    ) -> None:
        """Save activations to HDF5 file.

        Args:
            activations: Dictionary of activations
            save_path: Path to save file
            metadata: Optional metadata to save

        Example:
            >>> # extractor = ActivationExtractor(model)
            >>> # acts = {"layer_0": np.random.randn(10, 768)}
            >>> # extractor.save_activations(acts, "/tmp/acts.h5")
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(save_path, "w") as f:
            # Save activations
            for layer_name, acts in activations.items():
                f.create_dataset(layer_name, data=acts, compression="gzip")

            # Save metadata
            if metadata:
                meta_group = f.create_group("metadata")
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float)):
                        meta_group.attrs[key] = value
                    else:
                        # Convert to string for complex types
                        meta_group.attrs[key] = str(value)

        logger.info(f"Saved activations to {save_path}")

    def load_activations(
        self, load_path: Union[str, Path]
    ) -> Dict[str, np.ndarray]:
        """Load activations from HDF5 file.

        Args:
            load_path: Path to activation file

        Returns:
            Dictionary of activations.

        Example:
            >>> # extractor = ActivationExtractor(model)
            >>> # acts = extractor.load_activations("/tmp/acts.h5")
            >>> # isinstance(acts, dict)
            True
        """
        load_path = Path(load_path)

        if not load_path.exists():
            raise FileNotFoundError(f"Activation file not found: {load_path}")

        activations = {}

        with h5py.File(load_path, "r") as f:
            for key in f.keys():
                if key != "metadata":
                    activations[key] = f[key][:]

        logger.info(f"Loaded activations from {load_path}")
        return activations

    def extract_head_activations(
        self,
        texts: List[str],
        layers: List[int],
        heads: Optional[List[int]] = None,
        batch_size: int = 8,
    ) -> Dict[str, np.ndarray]:
        """Extract activations from specific attention heads.

        Args:
            texts: List of input texts
            layers: Layer indices
            heads: Head indices. If None, extracts all heads
            batch_size: Batch size for processing

        Returns:
            Dictionary mapping "layer_{l}_head_{h}" to activations.

        Example:
            >>> # extractor = ActivationExtractor(model)
            >>> # acts = extractor.extract_head_activations(
            ... #     ["text"], layers=[10], heads=[0, 1]
            ... # )
            >>> # "layer_10_head_0" in acts
            True
        """
        if heads is None:
            heads = list(range(self.model.cfg.n_heads))

        all_activations = {}

        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting head acts"):
            batch_texts = texts[i : i + batch_size]
            tokens = self.model.to_tokens(batch_texts, prepend_bos=True)

            for layer in layers:
                hook_name = f"blocks.{layer}.attn.hook_result"

                activations = {}

                def hook_fn(activation, hook):
                    # activation shape: [batch, seq, n_heads, d_head]
                    activations[hook.name] = activation.detach().cpu()

                with torch.no_grad():
                    self.model.run_with_hooks(
                        tokens,
                        fwd_hooks=[(hook_name, hook_fn)],
                    )

                # Extract specific heads
                for head in heads:
                    key = f"layer_{layer}_head_{head}"
                    if key not in all_activations:
                        all_activations[key] = []

                    # Get head output: [batch, seq, d_head]
                    head_out = activations[hook_name][:, -1, head, :]
                    all_activations[key].append(head_out.numpy())

        # Concatenate batches
        for key in all_activations:
            all_activations[key] = np.concatenate(all_activations[key], axis=0)

        return all_activations


def extract_and_save(
    model: HookedTransformer,
    texts: List[str],
    labels: List[int],
    save_path: Union[str, Path],
    layers: Optional[List[int]] = None,
    batch_size: int = 8,
) -> None:
    """Extract activations and save with labels.

    Convenience function for full extraction pipeline.

    Args:
        model: HookedTransformer model
        texts: List of texts
        labels: List of labels
        save_path: Path to save activations
        layers: Layers to extract from
        batch_size: Batch size

    Example:
        >>> # from src.models.model_loader import load_model
        >>> # model = load_model("gpt2", device="cpu")
        >>> # extract_and_save(
        ... #     model, ["text1"], [0], "/tmp/acts.h5", layers=[10, 11]
        ... # )
    """
    extractor = ActivationExtractor(model)

    activations = extractor.extract_batch(
        texts, layers=layers, batch_size=batch_size
    )

    # Add labels
    metadata = {
        "num_samples": len(texts),
        "num_layers": len(layers) if layers else model.cfg.n_layers,
        "model_name": model.cfg.model_name,
    }

    extractor.save_activations(activations, save_path, metadata)

    # Save labels separately
    save_path = Path(save_path)
    labels_path = save_path.parent / f"{save_path.stem}_labels.npy"
    np.save(labels_path, np.array(labels))

    logger.info(f"Saved labels to {labels_path}")
