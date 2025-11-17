#!/usr/bin/env python
"""CLI script for circuit analysis."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.model_loader import load_model
from src.circuits.activation_patching import ActivationPatcher
from src.circuits.ablation import AblationAnalyzer
from src.utils.logging_utils import setup_logger

logger = setup_logger("circuit_analysis")


def main():
    """Main function for circuit analysis."""
    parser = argparse.ArgumentParser(description="Run circuit analysis")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name")
    parser.add_argument("--method", type=str, default="activation_patching",
                       choices=["activation_patching", "ablation"])
    parser.add_argument("--output-dir", type=str, required=True)

    args = parser.parse_args()

    logger.info(f"Loading model {args.model}...")
    model = load_model(args.model, device="cpu")

    if args.method == "activation_patching":
        logger.info("Running activation patching...")
        patcher = ActivationPatcher(model)
        results = patcher.patch_heads(
            "Plan for next quarter",
            "Plan for next decade",
            layers=[10, 11],
            heads=[0, 1, 2],
        )
        logger.info(f"Patching results: {results}")

    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()
