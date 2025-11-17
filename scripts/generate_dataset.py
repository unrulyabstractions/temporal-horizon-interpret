#!/usr/bin/env python
"""CLI script for generating temporal horizon datasets."""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.generator import DatasetGenerator
from src.utils.logging_utils import setup_logger

logger = setup_logger("generate_dataset")


def main():
    """Main function for dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate temporal horizon detection dataset"
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=300,
        help="Number of prompt pairs to generate",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["business", "science", "personal"],
        help="Domains to include",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path (JSONL format)",
    )
    parser.add_argument(
        "--api-provider",
        type=str,
        default="openai",
        choices=["openai", "anthropic"],
        help="LLM API provider",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="API key (or set via environment variable)",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name to use for generation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--no-variations",
        action="store_true",
        help="Disable LLM-generated task variations",
    )

    args = parser.parse_args()

    logger.info("Initializing dataset generator...")
    generator = DatasetGenerator(
        api_provider=args.api_provider,
        api_key=args.api_key,
        model=args.model,
    )

    logger.info(f"Generating {args.num_pairs} prompt pairs...")
    dataset = generator.generate(
        num_pairs=args.num_pairs,
        domains=args.domains,
        save_path=args.output,
        use_variations=not args.no_variations,
        seed=args.seed,
    )

    logger.info(f"Successfully generated {len(dataset)} pairs")
    logger.info(f"Dataset saved to {args.output}")


if __name__ == "__main__":
    main()
