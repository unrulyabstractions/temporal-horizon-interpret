#!/usr/bin/env python
"""CLI script for model evaluation."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.probing.probe import create_probe
from src.probing.evaluator import ProbeEvaluator
from src.utils.logging_utils import setup_logger

logger = setup_logger("evaluate_model")


def main():
    """Main function for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate trained probe")
    parser.add_argument("--probe", type=str, required=True, help="Probe checkpoint")
    parser.add_argument("--activations", type=str, required=True, help="Test activations")
    parser.add_argument("--labels", type=str, required=True, help="Test labels")
    parser.add_argument("--output", type=str, required=True, help="Output file")

    args = parser.parse_args()

    logger.info("Loading probe...")
    checkpoint = torch.load(args.probe, map_location="cpu")
    probe = create_probe("mlp", hidden_size=768)
    probe.load_state_dict(checkpoint["model_state_dict"])

    logger.info("Loading test data...")
    from src.models.activation_extractor import ActivationExtractor
    extractor = ActivationExtractor(None)
    activations = extractor.load_activations(args.activations)
    labels = np.load(args.labels)

    layer_key = list(activations.keys())[0]
    X_test = activations[layer_key]

    logger.info("Evaluating...")
    evaluator = ProbeEvaluator(probe)
    metrics = evaluator.evaluate(X_test, labels)

    logger.info(f"Results: {metrics}")

    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
