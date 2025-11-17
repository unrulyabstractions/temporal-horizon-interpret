#!/usr/bin/env python
"""Extract activations from temporal horizon dataset using TransformerLens.

This script extracts activations from GPT-2 or Pythia models for the temporal
horizon detection task. It uses TransformerLens hooks to capture activations
from specific layers and saves them in HDF5 format for probe training.

Usage:
    python scripts/extract_activations.py \\
        --dataset data/raw/prompts.jsonl \\
        --model gpt2 \\
        --layers 8 9 10 11 \\
        --output data/processed/activations_gpt2.h5
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.loader import load_dataset, expand_dataset
from src.models.model_loader import load_model
from src.utils.logging_utils import setup_logger

logger = setup_logger("extract_activations")


def verify_transformerlens_installation():
    """Verify TransformerLens is properly installed."""
    try:
        from transformer_lens import HookedTransformer
        from transformer_lens.utils import get_act_name

        logger.info("✓ TransformerLens is properly installed")
        return True
    except ImportError as e:
        logger.error(f"✗ TransformerLens not found: {e}")
        logger.error("Install with: pip install transformer-lens")
        return False


def extract_activations_transformerlens(
    model,
    prompts: list,
    layers: list,
    batch_size: int = 8,
    component: str = "resid_post",
    position: str = "last",
) -> dict:
    """Extract activations using TransformerLens.

    Args:
        model: HookedTransformer model
        prompts: List of text prompts
        layers: List of layer indices to extract from
        batch_size: Batch size for processing
        component: Component to extract (resid_post, attn_out, mlp_out)
        position: Which position to extract (last, mean, all)

    Returns:
        Dictionary mapping layer_N to activations [num_prompts, hidden_size]
    """
    from transformer_lens.utils import get_act_name

    all_activations = {f"layer_{layer}": [] for layer in layers}

    logger.info(f"Extracting {component} activations from layers {layers}")
    logger.info(f"Processing {len(prompts)} prompts in batches of {batch_size}")

    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc="Extracting activations"):
            batch_prompts = prompts[i : i + batch_size]

            # Tokenize
            tokens = model.to_tokens(batch_prompts, prepend_bos=True)

            # Run model with cache
            _, cache = model.run_with_cache(tokens, remove_batch_dim=False)

            # Extract activations for each layer
            for layer in layers:
                # Get the activation name using TransformerLens utils
                act_name = get_act_name(component, layer)

                if act_name not in cache:
                    logger.warning(f"Activation {act_name} not found in cache")
                    continue

                acts = cache[act_name]  # Shape: [batch, seq_len, hidden]

                # Apply position strategy
                if position == "last":
                    # Take last non-padding token
                    acts = acts[:, -1, :]
                elif position == "mean":
                    # Average over sequence
                    acts = acts.mean(dim=1)
                elif position == "all":
                    # Keep all positions (will need to handle variable lengths)
                    pass
                else:
                    raise ValueError(f"Unknown position strategy: {position}")

                all_activations[f"layer_{layer}"].append(acts.cpu().numpy())

    # Concatenate batches
    for layer_name in all_activations:
        if all_activations[layer_name]:
            all_activations[layer_name] = np.concatenate(
                all_activations[layer_name], axis=0
            )
        else:
            logger.warning(f"No activations collected for {layer_name}")

    return all_activations


def save_activations_hdf5(activations: dict, labels: np.ndarray, metadata: dict, output_path: str):
    """Save activations to HDF5 file with metadata.

    Args:
        activations: Dictionary of activations per layer
        labels: Label array
        metadata: Metadata dictionary
        output_path: Path to save HDF5 file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        # Save activations
        for layer_name, acts in activations.items():
            f.create_dataset(
                layer_name,
                data=acts,
                compression="gzip",
                compression_opts=4,
            )
            logger.info(f"Saved {layer_name}: shape {acts.shape}")

        # Save labels
        f.create_dataset("labels", data=labels, compression="gzip")

        # Save metadata
        meta_group = f.create_group("metadata")
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                meta_group.attrs[key] = value
            elif isinstance(value, list):
                meta_group.attrs[key] = json.dumps(value)
            else:
                meta_group.attrs[key] = str(value)

    logger.info(f"✓ Saved activations to {output_path}")


def main():
    """Main extraction function."""
    parser = argparse.ArgumentParser(
        description="Extract activations for temporal horizon detection"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to JSONL dataset",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Model name (gpt2, gpt2-medium, pythia-160m, etc.)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[8, 9, 10, 11],
        help="Layer indices to extract from",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output HDF5 file path",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--component",
        type=str,
        default="resid_post",
        choices=["resid_post", "attn_out", "mlp_out", "resid_pre"],
        help="Component to extract",
    )
    parser.add_argument(
        "--position",
        type=str,
        default="last",
        choices=["last", "mean"],
        help="Position strategy (last token or mean pooling)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)",
    )

    args = parser.parse_args()

    # Verify TransformerLens
    if not verify_transformerlens_installation():
        sys.exit(1)

    # Load dataset
    logger.info(f"Loading dataset from {args.dataset}")
    dataset = load_dataset(args.dataset)

    if args.max_samples:
        dataset = dataset[: args.max_samples]
        logger.info(f"Limited to {len(dataset)} samples for testing")

    # Expand dataset into individual prompts
    prompts, labels, metadata = expand_dataset(dataset)
    logger.info(f"Expanded to {len(prompts)} prompts (short + long for each pair)")

    # Load model
    logger.info(f"Loading {args.model} on {args.device}")
    model = load_model(args.model, device=args.device)

    # Log model info
    logger.info(f"Model config:")
    logger.info(f"  Layers: {model.cfg.n_layers}")
    logger.info(f"  Hidden size: {model.cfg.d_model}")
    logger.info(f"  Attention heads: {model.cfg.n_heads}")

    # Verify layers are valid
    if max(args.layers) >= model.cfg.n_layers:
        logger.error(
            f"Requested layer {max(args.layers)} but model only has "
            f"{model.cfg.n_layers} layers"
        )
        sys.exit(1)

    # Extract activations
    activations = extract_activations_transformerlens(
        model,
        prompts,
        layers=args.layers,
        batch_size=args.batch_size,
        component=args.component,
        position=args.position,
    )

    # Prepare metadata
    extraction_metadata = {
        "model": args.model,
        "num_samples": len(prompts),
        "num_pairs": len(dataset),
        "layers": args.layers,
        "component": args.component,
        "position": args.position,
        "hidden_size": model.cfg.d_model,
        "device": args.device,
    }

    # Save to HDF5
    save_activations_hdf5(
        activations,
        np.array(labels),
        extraction_metadata,
        args.output,
    )

    # Print summary statistics
    logger.info("\n=== Extraction Summary ===")
    logger.info(f"Model: {args.model}")
    logger.info(f"Prompts processed: {len(prompts)}")
    logger.info(f"Labels: {np.unique(labels, return_counts=True)}")
    for layer_name, acts in activations.items():
        logger.info(f"{layer_name}: {acts.shape}")
    logger.info(f"Output: {args.output}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
