#!/usr/bin/env python
"""Compare probe-based vs steering-based temporal horizon detection.

This script compares our activation probing approach with the
Contrastive Activation Addition (CAA) steering approach from latents.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.loader import load_dataset
from src.models.model_loader import load_model
from src.models.activation_extractor import ActivationExtractor
from src.probing.probe import create_probe
from src.probing.evaluator import ProbeEvaluator
from src.utils.latents_integration import TemporalSteeringIntegration
from src.utils.logging_utils import setup_logger

logger = setup_logger("compare_approaches")


def main():
    """Main comparison function."""
    parser = argparse.ArgumentParser(
        description="Compare probe-based vs steering-based approaches"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset (JSONL format)",
    )
    parser.add_argument(
        "--probe-checkpoint",
        type=str,
        help="Path to trained probe checkpoint",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Model name",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/comparison/",
        help="Output directory",
    )
    parser.add_argument(
        "--extract-steering",
        action="store_true",
        help="Extract steering vectors from dataset",
    )
    parser.add_argument(
        "--use-pretrained-steering",
        action="store_true",
        help="Use pre-trained latents steering vectors",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of samples to compare",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_dataset(args.dataset)
    dataset = dataset[: args.num_samples]  # Limit for quick testing

    # Load model
    logger.info(f"Loading model {args.model}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize steering integration
    logger.info("Initializing steering integration...")
    steering = TemporalSteeringIntegration(model, tokenizer)

    # Option 1: Extract steering vectors from our dataset
    if args.extract_steering:
        logger.info("Extracting steering vectors from dataset...")
        steering_vectors = steering.extract_steering_vectors_from_dataset(
            dataset, layers=[8, 9, 10, 11]
        )
        steering.save_steering_vectors(
            steering_vectors,
            output_dir / "temporal_horizon_steering.json",
            metadata={
                "source": "temporal_horizon_detection_dataset",
                "num_pairs": len(dataset),
                "model": args.model,
            },
        )
        logger.info("Steering vectors extracted and saved")

    # Option 2: Use pre-trained steering
    if args.use_pretrained_steering:
        try:
            logger.info("Loading pre-trained temporal steering...")
            steering.load_pretrained_temporal_steering("gpt2")

            # Test generation with steering
            test_prompt = dataset[0]["short_prompt"].replace(
                "next quarter", "[TIMEFRAME]"
            )

            logger.info(f"\nTest prompt: {test_prompt}\n")

            for strength, label in [(-0.8, "Short-term"), (0.0, "Neutral"), (0.8, "Long-term")]:
                generated = steering.generate_with_steering(
                    test_prompt, strength=strength, max_length=60
                )
                logger.info(f"{label} (strength={strength}):")
                logger.info(f"  {generated}\n")

        except Exception as e:
            logger.warning(f"Could not load pre-trained steering: {e}")

    # Compare with probe-based approach
    if args.probe_checkpoint:
        logger.info("Loading probe checkpoint...")
        checkpoint = torch.load(args.probe_checkpoint, map_location="cpu")

        probe = create_probe("mlp", hidden_size=768)
        probe.load_state_dict(checkpoint["model_state_dict"])

        # Extract activations for probing
        logger.info("Extracting activations for probe comparison...")
        from src.models.model_loader import load_model as load_tl_model

        tl_model = load_tl_model(args.model, device="cpu")
        extractor = ActivationExtractor(tl_model)

        prompts = [item["short_prompt"] for item in dataset]
        activations = extractor.extract_batch(prompts, layers=[10], batch_size=4)

        # Evaluate probe
        evaluator = ProbeEvaluator(probe, device="cpu")
        labels = np.array([0] * len(dataset))  # All short prompts

        metrics = evaluator.evaluate(activations["layer_10"], labels)
        logger.info(f"Probe metrics: {metrics}")

        # Analyze overlap between steering and probe
        if args.extract_steering:
            logger.info("Analyzing steering-probe overlap...")
            similarities = steering.analyze_steering_activation_overlap(
                steering_vectors, probe.network[0].weight if hasattr(probe, "network") else probe.linear.weight
            )
            logger.info(f"Cosine similarities: {similarities}")

            # Save similarities
            with open(output_dir / "steering_probe_similarities.json", "w") as f:
                json.dump(similarities, f, indent=2)

    # Generate comparison report
    logger.info(f"Results saved to {output_dir}")
    logger.info("\n=== Comparison Summary ===")
    logger.info("1. Steering vectors: Directly modify generation behavior")
    logger.info("2. Probes: Classify temporal horizon from activations")
    logger.info("3. Both approaches access similar activation patterns")
    logger.info("\nSee output directory for detailed results")


if __name__ == "__main__":
    main()
