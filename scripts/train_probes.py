#!/usr/bin/env python
"""CLI script for training probes."""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.activation_extractor import ActivationExtractor
from src.probing.probe import create_probe
from src.probing.trainer import ProbeTrainer
from src.utils.logging_utils import setup_logger

logger = setup_logger("train_probes")


def main():
    """Main function for probe training."""
    parser = argparse.ArgumentParser(description="Train temporal horizon probes")
    parser.add_argument("--config", type=str, help="Config file path")
    parser.add_argument("--activations", type=str, required=True, help="Activations file")
    parser.add_argument("--labels", type=str, required=True, help="Labels file")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--probe-type", type=str, default="mlp", choices=["linear", "mlp"])
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)

    args = parser.parse_args()

    # Load config if provided
    config = {}
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)

    # Load activations and labels
    logger.info("Loading activations and labels...")
    extractor = ActivationExtractor(None)  # Dummy for loading
    activations = extractor.load_activations(args.activations)
    labels = np.load(args.labels)

    # For simplicity, train on first layer
    layer_key = list(activations.keys())[0]
    X = activations[layer_key]

    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, labels, test_size=0.2, random_state=42
    )

    # Create probe
    logger.info(f"Creating {args.probe_type} probe...")
    probe = create_probe(args.probe_type, hidden_size=args.hidden_size)

    # Train
    logger.info("Training probe...")
    trainer = ProbeTrainer(probe, learning_rate=args.learning_rate)
    history = trainer.train(
        X_train, y_train, X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_dir=args.output_dir,
    )

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
