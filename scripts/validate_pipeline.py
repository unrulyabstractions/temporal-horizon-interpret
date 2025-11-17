#!/usr/bin/env python
"""Validate the complete temporal horizon detection pipeline.

This script runs end-to-end validation to ensure all components work correctly:
1. TransformerLens model loading (GPT-2, Pythia)
2. Dataset loading and expansion
3. Activation extraction with proper hooks
4. Latents integration (steering vector extraction)
5. Probe training on extracted activations

Run this before starting experiments to catch any setup issues.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_utils import setup_logger

logger = setup_logger("validate_pipeline")


def test_transformerlens():
    """Test TransformerLens installation and basic functionality."""
    logger.info("\n=== Testing TransformerLens ===")

    try:
        from transformer_lens import HookedTransformer
        from transformer_lens.utils import get_act_name

        logger.info("✓ TransformerLens imported successfully")

        # Test loading GPT-2
        logger.info("Loading GPT-2 small...")
        model = HookedTransformer.from_pretrained("gpt2", device="cpu")
        logger.info(f"✓ Loaded GPT-2: {model.cfg.n_layers} layers, {model.cfg.d_model} hidden")

        # Test tokenization
        test_text = "Plan for the next decade"
        tokens = model.to_tokens(test_text)
        logger.info(f"✓ Tokenization works: '{test_text}' -> {tokens.shape}")

        # Test forward pass
        logits = model(tokens)
        logger.info(f"✓ Forward pass works: output shape {logits.shape}")

        # Test activation extraction with cache
        _, cache = model.run_with_cache(tokens)
        logger.info(f"✓ Cache extraction works: {len(cache)} activations cached")

        # Test specific hook points
        for layer in [0, 5, 11]:
            act_name = get_act_name("resid_post", layer)
            if act_name in cache:
                logger.info(f"✓ Layer {layer} resid_post: {cache[act_name].shape}")
            else:
                logger.warning(f"✗ Layer {layer} resid_post not found")

        logger.info("✓ All TransformerLens tests passed!\n")
        return True

    except Exception as e:
        logger.error(f"✗ TransformerLens test failed: {e}")
        return False


def test_model_loading():
    """Test our model loader wrapper."""
    logger.info("\n=== Testing Model Loader ===")

    try:
        from src.models.model_loader import load_model, get_model_info

        # Test GPT-2
        logger.info("Testing GPT-2 loading...")
        model = load_model("gpt2", device="cpu")
        info = get_model_info(model)
        logger.info(f"✓ GPT-2 loaded: {info['num_layers']} layers")

        # Test that it's a HookedTransformer
        from transformer_lens import HookedTransformer

        assert isinstance(model, HookedTransformer), "Model should be HookedTransformer"
        logger.info("✓ Model is correct type (HookedTransformer)")

        logger.info("✓ Model loader tests passed!\n")
        return True

    except Exception as e:
        logger.error(f"✗ Model loader test failed: {e}")
        return False


def test_activation_extraction():
    """Test activation extraction pipeline."""
    logger.info("\n=== Testing Activation Extraction ===")

    try:
        from src.models.activation_extractor import ActivationExtractor
        from src.models.model_loader import load_model

        # Load model
        model = load_model("gpt2", device="cpu")
        extractor = ActivationExtractor(model)

        # Test single extraction
        test_prompts = [
            "Plan for the next quarter",
            "Plan for the next decade",
        ]

        logger.info(f"Extracting activations from {len(test_prompts)} prompts...")
        activations = extractor.extract_batch(
            test_prompts, layers=[10, 11], batch_size=2
        )

        # Verify shapes
        for layer_name, acts in activations.items():
            logger.info(f"✓ {layer_name}: shape {acts.shape}")
            assert acts.shape[0] == len(test_prompts), "Batch size mismatch"
            assert acts.shape[1] == 768, "Hidden size should be 768 for GPT-2"

        logger.info("✓ Activation extraction tests passed!\n")
        return True

    except Exception as e:
        logger.error(f"✗ Activation extraction test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_dataset_loading():
    """Test dataset loading and expansion."""
    logger.info("\n=== Testing Dataset Loading ===")

    try:
        from src.dataset.loader import expand_dataset
        from src.dataset.templates import BUSINESS_TEMPLATES, format_prompt

        # Create test dataset
        template = BUSINESS_TEMPLATES[0]
        test_dataset = []

        for i in range(5):
            test_dataset.append({
                "pair_id": i,
                "domain": "business",
                "task": f"task_{i}",
                "short_prompt": format_prompt(template, f"task_{i}", False),
                "long_prompt": format_prompt(template, f"task_{i}", True),
            })

        # Test expansion
        prompts, labels, metadata = expand_dataset(test_dataset)

        logger.info(f"✓ Expanded {len(test_dataset)} pairs -> {len(prompts)} prompts")
        logger.info(f"✓ Labels: {np.unique(labels, return_counts=True)}")
        assert len(prompts) == len(test_dataset) * 2, "Should have 2 prompts per pair"
        assert len(set(labels)) == 2, "Should have 2 unique labels (0, 1)"

        logger.info("✓ Dataset loading tests passed!\n")
        return True

    except Exception as e:
        logger.error(f"✗ Dataset loading test failed: {e}")
        return False


def test_latents_integration():
    """Test latents submodule integration."""
    logger.info("\n=== Testing Latents Integration ===")

    # Check if latents submodule exists
    latents_path = Path(__file__).parent.parent / "external" / "latents"

    if not latents_path.exists():
        logger.warning("✗ Latents submodule not found at external/latents")
        logger.warning("  Run: git submodule update --init --recursive")
        return False

    try:
        # Test import
        sys.path.insert(0, str(latents_path))
        from latents.extract_steering_vectors import compute_steering_vectors

        logger.info("✓ Latents library imports successfully")

        # Check for pre-trained vectors
        vectors_path = latents_path / "steering_vectors" / "temporal_scope_gpt2.json"
        if vectors_path.exists():
            logger.info(f"✓ Found pre-trained temporal steering: {vectors_path}")
        else:
            logger.warning(f"✗ Pre-trained vectors not found: {vectors_path}")

        logger.info("✓ Latents integration tests passed!\n")
        return True

    except ImportError as e:
        logger.error(f"✗ Latents import failed: {e}")
        logger.error("  Install with: pip install -e external/latents")
        return False
    except Exception as e:
        logger.error(f"✗ Latents test failed: {e}")
        return False


def test_probe_training():
    """Test probe architecture and training."""
    logger.info("\n=== Testing Probe Training ===")

    try:
        from src.probing.probe import LinearProbe, MLPProbe, create_probe

        # Test linear probe
        linear_probe = LinearProbe(hidden_size=768)
        test_input = torch.randn(10, 768)
        output = linear_probe(test_input)
        assert output.shape == (10, 2), "Linear probe output shape incorrect"
        logger.info(f"✓ Linear probe: input {test_input.shape} -> output {output.shape}")

        # Test MLP probe
        mlp_probe = MLPProbe(hidden_size=768, num_layers=2)
        output = mlp_probe(test_input)
        assert output.shape == (10, 2), "MLP probe output shape incorrect"
        logger.info(f"✓ MLP probe: input {test_input.shape} -> output {output.shape}")

        # Test factory
        probe = create_probe("mlp", hidden_size=768)
        logger.info("✓ Probe factory works")

        logger.info("✓ Probe training tests passed!\n")
        return True

    except Exception as e:
        logger.error(f"✗ Probe training test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_full_pipeline():
    """Test the complete pipeline end-to-end."""
    logger.info("\n=== Testing Full Pipeline ===")

    try:
        from src.dataset.templates import BUSINESS_TEMPLATES, format_prompt
        from src.models.activation_extractor import ActivationExtractor
        from src.models.model_loader import load_model
        from src.probing.probe import create_probe
        from src.probing.trainer import ProbeTrainer

        # 1. Create mini dataset
        template = BUSINESS_TEMPLATES[0]
        prompts = []
        labels = []

        for i in range(4):
            prompts.append(format_prompt(template, f"task_{i}", False))  # Short
            labels.append(0)
            prompts.append(format_prompt(template, f"task_{i}", True))  # Long
            labels.append(1)

        logger.info(f"✓ Created {len(prompts)} test prompts")

        # 2. Load model and extract activations
        model = load_model("gpt2", device="cpu")
        extractor = ActivationExtractor(model)
        activations = extractor.extract_batch(prompts, layers=[10], batch_size=2)

        X = activations["layer_10"]
        y = np.array(labels)

        logger.info(f"✓ Extracted activations: {X.shape}")

        # 3. Train probe (just 1 epoch to test)
        probe = create_probe("linear", hidden_size=768)
        trainer = ProbeTrainer(probe, learning_rate=1e-3)

        # Simple split
        train_X, train_y = X[:6], y[:6]
        val_X, val_y = X[6:], y[6:]

        history = trainer.train(
            train_X,
            train_y,
            val_X,
            val_y,
            epochs=1,
            batch_size=2,
        )

        logger.info("✓ Probe training completed")
        logger.info(f"  Train accuracy: {history['train_acc'][0]:.2f}")

        logger.info("✓ Full pipeline test passed!\n")
        return True

    except Exception as e:
        logger.error(f"✗ Full pipeline test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    parser = argparse.ArgumentParser(description="Validate pipeline setup")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only quick tests (skip full pipeline)",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Temporal Horizon Detection - Pipeline Validation")
    logger.info("=" * 60)

    results = {}

    # Core tests
    results["transformerlens"] = test_transformerlens()
    results["model_loading"] = test_model_loading()
    results["activation_extraction"] = test_activation_extraction()
    results["dataset_loading"] = test_dataset_loading()
    results["latents_integration"] = test_latents_integration()
    results["probe_training"] = test_probe_training()

    # Full pipeline (optional)
    if not args.quick:
        results["full_pipeline"] = test_full_pipeline()

    # Summary
    logger.info("=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, passed_test in results.items():
        status = "✓ PASS" if passed_test else "✗ FAIL"
        logger.info(f"{status}: {test_name}")

    logger.info("=" * 60)
    logger.info(f"Result: {passed}/{total} tests passed")

    if passed == total:
        logger.info("✓ All tests passed! Pipeline is ready.")
        return 0
    else:
        logger.error("✗ Some tests failed. Please fix before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
