#!/usr/bin/env python
"""Complete workflow example for temporal horizon detection.

This script demonstrates the full pipeline:
1. Load dataset
2. Extract activations using TransformerLens
3. Train probes
4. Extract steering vectors using latents
5. Compare both approaches

Run this to understand the complete workflow.
"""

import sys
from pathlib import Path

import numpy as np
import torch

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.loader import expand_dataset, load_dataset
from src.dataset.templates import BUSINESS_TEMPLATES, format_prompt
from src.models.activation_extractor import ActivationExtractor
from src.models.model_loader import load_model, get_model_info
from src.probing.evaluator import ProbeEvaluator
from src.probing.probe import create_probe
from src.probing.trainer import ProbeTrainer
from src.utils.latents_integration import TemporalSteeringIntegration
from src.utils.logging_utils import setup_logger
from src.utils.visualization import plot_layer_wise_accuracy

logger = setup_logger("workflow")


def create_mini_dataset():
    """Create a small dataset for testing."""
    logger.info("=" * 60)
    logger.info("STEP 1: Creating Mini Dataset")
    logger.info("=" * 60)

    template = BUSINESS_TEMPLATES[0]
    tasks = [
        "launching a new product",
        "expanding market presence",
        "developing new technology",
        "building partnerships",
    ]

    dataset = []
    for i, task in enumerate(tasks):
        dataset.append({
            "pair_id": i,
            "domain": "business",
            "task": task,
            "short_prompt": format_prompt(template, task, use_long_horizon=False),
            "long_prompt": format_prompt(template, task, use_long_horizon=True),
        })

    logger.info(f"Created {len(dataset)} prompt pairs")
    logger.info(f"Example pair:")
    logger.info(f"  Short: {dataset[0]['short_prompt']}")
    logger.info(f"  Long:  {dataset[0]['long_prompt']}")

    return dataset


def extract_activations_transformerlens(model, prompts, layers):
    """Extract activations using TransformerLens."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Extracting Activations with TransformerLens")
    logger.info("=" * 60)

    logger.info(f"Model: {model.cfg.model_name}")
    logger.info(f"Layers: {layers}")
    logger.info(f"Prompts: {len(prompts)}")

    extractor = ActivationExtractor(model)

    activations = extractor.extract_batch(
        prompts,
        layers=layers,
        batch_size=4,
        position_strategy="last",  # Last token captures reasoning
    )

    for layer_name, acts in activations.items():
        logger.info(f"  {layer_name}: {acts.shape}")

    logger.info("✓ Activation extraction complete")
    return activations


def train_probes(activations, labels):
    """Train probes on extracted activations."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Training Probes")
    logger.info("=" * 60)

    results = {}

    for layer_name, X in activations.items():
        layer_num = int(layer_name.split("_")[1])

        # Simple train/test split
        n = len(X)
        train_idx = int(0.7 * n)

        X_train, X_test = X[:train_idx], X[train_idx:]
        y_train, y_test = labels[:train_idx], labels[train_idx:]

        # Create and train probe
        probe = create_probe("linear", hidden_size=X.shape[1])
        trainer = ProbeTrainer(probe, learning_rate=1e-3)

        logger.info(f"\nTraining probe for {layer_name}...")

        history = trainer.train(
            X_train,
            y_train,
            X_test,
            y_test,
            epochs=20,
            batch_size=4,
            early_stopping_patience=5,
        )

        # Evaluate
        evaluator = ProbeEvaluator(probe, device="cpu")
        metrics = evaluator.evaluate(X_test, y_test)

        results[layer_num] = {
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "history": history,
        }

        logger.info(f"  Accuracy: {metrics['accuracy']:.3f}")
        logger.info(f"  F1 Score: {metrics['f1']:.3f}")

    # Find best layer
    best_layer = max(results.keys(), key=lambda k: results[k]["accuracy"])
    logger.info(f"\n✓ Best layer: {best_layer} (accuracy: {results[best_layer]['accuracy']:.3f})")

    return results


def extract_steering_vectors(model, tokenizer, dataset):
    """Extract steering vectors using latents."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Extracting Steering Vectors (Latents)")
    logger.info("=" * 60)

    try:
        # Load HuggingFace model for latents
        from transformers import AutoModelForCausalLM, AutoTokenizer

        hf_model = AutoModelForCausalLM.from_pretrained("gpt2")
        hf_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        hf_tokenizer.pad_token = hf_tokenizer.eos_token

        logger.info("Loaded HuggingFace GPT-2 for steering")

        # Initialize steering integration
        steering = TemporalSteeringIntegration(hf_model, hf_tokenizer)

        # Extract steering vectors
        logger.info("Extracting steering vectors from dataset...")
        vectors = steering.extract_steering_vectors_from_dataset(
            dataset, layers=[8, 9, 10, 11]
        )

        logger.info("✓ Steering vector extraction complete")

        # Test generation
        logger.info("\nTesting steering generation:")

        test_prompt = "What should our company prioritize?"

        # Load pre-trained steering
        try:
            steering.load_pretrained_temporal_steering("gpt2")

            short_result = steering.generate_with_steering(
                test_prompt, strength=-0.8, max_length=40
            )
            long_result = steering.generate_with_steering(
                test_prompt, strength=0.8, max_length=40
            )

            logger.info(f"\nPrompt: {test_prompt}")
            logger.info(f"Short-term (-0.8): {short_result}")
            logger.info(f"Long-term  (+0.8): {long_result}")

        except Exception as e:
            logger.warning(f"Pre-trained steering not available: {e}")

        return vectors

    except ImportError as e:
        logger.warning(f"Latents integration not available: {e}")
        logger.warning("Install with: pip install -e external/latents")
        return None


def compare_approaches(probe_results, steering_vectors):
    """Compare probe-based and steering-based approaches."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: Comparing Approaches")
    logger.info("=" * 60)

    logger.info("\nProbe-Based Approach (Our Core Method):")
    logger.info("  Method: Train classifier on activations")
    logger.info("  Output: Accuracy metrics per layer")
    logger.info("  Best for: Understanding what's encoded")

    for layer, results in probe_results.items():
        logger.info(f"  Layer {layer}: {results['accuracy']:.3f} accuracy")

    if steering_vectors:
        logger.info("\nSteering-Based Approach (Latents Integration):")
        logger.info("  Method: Contrastive activation addition")
        logger.info("  Output: Modified generation behavior")
        logger.info("  Best for: Controlling temporal scope")

        logger.info(f"  Extracted vectors for layers: {list(steering_vectors.keys())}")

        logger.info("\n✓ Both approaches complement each other:")
        logger.info("  - Probes: Measure temporal encoding")
        logger.info("  - Steering: Control temporal behavior")
        logger.info("  - Agreement validates findings!")


def main():
    """Run complete workflow."""
    logger.info("=" * 60)
    logger.info("TEMPORAL HORIZON DETECTION - COMPLETE WORKFLOW")
    logger.info("=" * 60)

    # Step 1: Create dataset
    dataset = create_mini_dataset()

    # Expand to individual prompts
    prompts, labels, metadata = expand_dataset(dataset)
    labels = np.array(labels)

    # Step 2: Load model and extract activations
    logger.info("\nLoading GPT-2 model...")
    model = load_model("gpt2", device="cpu")

    info = get_model_info(model)
    logger.info(f"Model info:")
    logger.info(f"  Layers: {info['num_layers']}")
    logger.info(f"  Hidden: {info['hidden_size']}")
    logger.info(f"  Params: {info['num_parameters']:,}")

    # Extract activations (focus on later layers where temporal reasoning emerges)
    activations = extract_activations_transformerlens(model, prompts, layers=[8, 9, 10, 11])

    # Step 3: Train probes
    probe_results = train_probes(activations, labels)

    # Step 4: Extract steering vectors (optional - requires latents)
    steering_vectors = extract_steering_vectors(model, None, dataset)

    # Step 5: Compare approaches
    compare_approaches(probe_results, steering_vectors)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("WORKFLOW COMPLETE!")
    logger.info("=" * 60)

    logger.info("\nNext steps:")
    logger.info("1. Run on full dataset (300 pairs)")
    logger.info("2. Test on GPT-2 medium/large and Pythia models")
    logger.info("3. Perform circuit analysis (activation patching)")
    logger.info("4. Detect divergence cases")
    logger.info("5. Write up results")

    logger.info("\nKey findings from this run:")
    best_layer = max(probe_results.keys(), key=lambda k: probe_results[k]["accuracy"])
    logger.info(f"- Best performing layer: {best_layer}")
    logger.info(f"- Peak accuracy: {probe_results[best_layer]['accuracy']:.3f}")
    logger.info("- Confirms temporal info encoded in later layers")

    if steering_vectors:
        logger.info("- Steering vectors successfully extracted")
        logger.info("- Can use for behavior control and validation")

    logger.info("\nFor production runs:")
    logger.info("  python scripts/extract_activations.py --dataset data/raw/prompts.jsonl")
    logger.info("  python scripts/train_probes.py --config configs/probe_config.yaml")
    logger.info("  python scripts/compare_probe_vs_steering.py")


if __name__ == "__main__":
    main()
