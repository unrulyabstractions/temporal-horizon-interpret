"""Integration with latents library for steering-based temporal horizon analysis.

This module integrates the latents library (https://github.com/justinshenk/latents)
for Contrastive Activation Addition (CAA) steering experiments.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Add latents submodule to path
LATENTS_PATH = Path(__file__).parent.parent.parent / "external" / "latents"
sys.path.insert(0, str(LATENTS_PATH))

try:
    from latents.extract_steering_vectors import (
        compute_steering_vectors,
        save_steering_vectors,
    )
    from latents.core import SteeringFramework

    LATENTS_AVAILABLE = True
except ImportError:
    LATENTS_AVAILABLE = False
    logging.warning("Latents library not available. Install with: pip install -e external/latents")

logger = logging.getLogger(__name__)


class TemporalSteeringIntegration:
    """Integration of latents library for temporal horizon steering.

    This class bridges our probe-based approach with the steering-based
    approach from the latents library.

    Attributes:
        model: Transformer model
        tokenizer: Model tokenizer
        steering_framework: Loaded SteeringFramework (if available)
    """

    def __init__(self, model, tokenizer):
        """Initialize steering integration.

        Args:
            model: HuggingFace or TransformerLens model
            tokenizer: Model tokenizer

        Example:
            >>> from transformers import GPT2LMHeadModel, GPT2Tokenizer
            >>> model = GPT2LMHeadModel.from_pretrained("gpt2")
            >>> tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            >>> integration = TemporalSteeringIntegration(model, tokenizer)
        """
        if not LATENTS_AVAILABLE:
            raise ImportError(
                "Latents library not available. "
                "Install with: pip install -e external/latents"
            )

        self.model = model
        self.tokenizer = tokenizer
        self.steering_framework = None

    def extract_steering_vectors_from_dataset(
        self, dataset: List[Dict], layers: Optional[List[int]] = None
    ) -> Dict:
        """Extract steering vectors from our paired prompt dataset.

        Args:
            dataset: List of prompt pair dictionaries with 'short_prompt' and 'long_prompt'
            layers: Layers to extract from (None = all layers)

        Returns:
            Dictionary of steering vectors per layer.

        Example:
            >>> from src.dataset.loader import load_dataset
            >>> dataset = load_dataset("data/raw/prompts.jsonl")
            >>> vectors = integration.extract_steering_vectors_from_dataset(dataset)
        """
        # Convert our dataset format to latents format
        prompt_pairs = []
        for item in dataset:
            prompt_pairs.append({
                "negative": item["short_prompt"],  # Short horizon is "negative"
                "positive": item["long_prompt"],  # Long horizon is "positive"
            })

        logger.info(f"Extracting steering vectors from {len(prompt_pairs)} pairs")

        # Extract steering vectors using latents
        steering_vectors = compute_steering_vectors(
            self.model, self.tokenizer, prompt_pairs, layers=layers
        )

        return steering_vectors

    def save_steering_vectors(
        self, steering_vectors: Dict, save_path: str, metadata: Optional[Dict] = None
    ) -> None:
        """Save extracted steering vectors.

        Args:
            steering_vectors: Steering vectors dictionary
            save_path: Path to save JSON file
            metadata: Optional metadata to include

        Example:
            >>> vectors = integration.extract_steering_vectors_from_dataset(dataset)
            >>> integration.save_steering_vectors(
            ...     vectors, "steering_vectors/temporal_horizon_custom.json"
            ... )
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Add metadata
        if metadata:
            steering_vectors["metadata"] = metadata

        save_steering_vectors(steering_vectors, str(save_path))
        logger.info(f"Saved steering vectors to {save_path}")

    def load_pretrained_temporal_steering(self, variant: str = "gpt2") -> None:
        """Load pre-trained temporal steering vectors from latents.

        Args:
            variant: Which variant to load ("gpt2", "deconfounded", or "base")

        Example:
            >>> integration.load_pretrained_temporal_steering("gpt2")
            >>> result = integration.generate_with_steering("Plan for...", strength=0.8)
        """
        vector_file = {
            "gpt2": "temporal_scope_gpt2.json",
            "deconfounded": "temporal_scope_deconfounded.json",
            "base": "temporal_scope.json",
        }[variant]

        vector_path = LATENTS_PATH / "steering_vectors" / vector_file

        if not vector_path.exists():
            raise FileNotFoundError(f"Steering vectors not found: {vector_path}")

        self.steering_framework = SteeringFramework.load(
            self.model, self.tokenizer, str(vector_path)
        )

        logger.info(f"Loaded pre-trained temporal steering: {variant}")

    def generate_with_steering(
        self,
        prompt: str,
        strength: float = 0.8,
        temperature: float = 0.7,
        max_length: int = 100,
    ) -> str:
        """Generate text with temporal steering.

        Args:
            prompt: Input prompt
            strength: Steering strength (-1.0 = short-term, +1.0 = long-term)
            temperature: Sampling temperature
            max_length: Maximum generation length

        Returns:
            Generated text.

        Example:
            >>> integration.load_pretrained_temporal_steering()
            >>> result = integration.generate_with_steering(
            ...     "How should we address climate change?",
            ...     strength=0.8  # Long-term thinking
            ... )
        """
        if self.steering_framework is None:
            raise ValueError("Load steering vectors first with load_pretrained_temporal_steering()")

        result = self.steering_framework.generate(
            prompt=prompt,
            steerings=[("temporal_scope", strength)],
            temperature=temperature,
            max_length=max_length,
        )

        return result

    def compare_steering_vs_probe(
        self,
        prompts: List[str],
        probe_evaluator,
        activations: np.ndarray,
        steering_strengths: List[float] = [-0.8, 0.0, 0.8],
    ) -> Dict:
        """Compare steering-based and probe-based approaches.

        Args:
            prompts: List of test prompts
            probe_evaluator: Our ProbeEvaluator instance
            activations: Extracted activations for probing
            steering_strengths: Steering strengths to test

        Returns:
            Dictionary with comparison results.

        Example:
            >>> from src.probing.evaluator import ProbeEvaluator
            >>> results = integration.compare_steering_vs_probe(
            ...     prompts, probe_evaluator, activations
            ... )
        """
        # Get probe predictions
        probe_preds = probe_evaluator.predict(activations)
        probe_probs = probe_evaluator.predict_proba(activations)

        # Generate with different steering strengths
        steering_results = {}
        for strength in steering_strengths:
            generations = []
            for prompt in prompts:
                generated = self.generate_with_steering(
                    prompt, strength=strength, max_length=50
                )
                generations.append(generated)

            steering_results[strength] = generations

        return {
            "probe_predictions": probe_preds.tolist(),
            "probe_probabilities": probe_probs.tolist(),
            "steering_generations": steering_results,
            "prompts": prompts,
        }

    def analyze_steering_activation_overlap(
        self, steering_vectors: Dict, probe_weights: torch.Tensor
    ) -> Dict[str, float]:
        """Analyze overlap between steering vectors and probe weights.

        Computes cosine similarity between steering direction and probe
        decision boundary to see if they capture the same information.

        Args:
            steering_vectors: Steering vectors dictionary
            probe_weights: Linear probe weights [hidden_size, num_classes]

        Returns:
            Dictionary of similarity scores per layer.

        Example:
            >>> vectors = integration.extract_steering_vectors_from_dataset(dataset)
            >>> probe = LinearProbe(768)
            >>> similarities = integration.analyze_steering_activation_overlap(
            ...     vectors, probe.linear.weight
            ... )
        """
        from torch.nn.functional import cosine_similarity

        similarities = {}

        # Extract probe direction (difference between class weights)
        if probe_weights.shape[0] == 2:
            probe_direction = probe_weights[1] - probe_weights[0]
        else:
            probe_direction = probe_weights[0]

        probe_direction = probe_direction.detach().cpu()

        # Compare with steering vectors for each layer
        for layer_key, vector_data in steering_vectors.items():
            if layer_key.startswith("layer_") or layer_key.isdigit():
                steering_vector = torch.tensor(vector_data, dtype=torch.float32)

                # Ensure same dimensionality
                if steering_vector.shape[0] == probe_direction.shape[0]:
                    similarity = cosine_similarity(
                        probe_direction.unsqueeze(0), steering_vector.unsqueeze(0)
                    )
                    similarities[layer_key] = float(similarity.item())

        return similarities

    def extract_steered_activations(
        self, prompts: List[str], steering_strength: float, layers: List[int]
    ) -> Dict[str, np.ndarray]:
        """Extract activations under steering intervention.

        Args:
            prompts: Input prompts
            steering_strength: Steering strength to apply
            layers: Layers to extract activations from

        Returns:
            Dictionary of activations per layer.

        Example:
            >>> acts = integration.extract_steered_activations(
            ...     ["Plan for the future"], strength=0.8, layers=[10, 11]
            ... )
        """
        # This would require modifying the model forward pass
        # Placeholder for now
        raise NotImplementedError(
            "Extracting activations under steering requires "
            "hooking into model forward pass"
        )


def create_latents_compatible_dataset(
    input_dataset_path: str, output_path: str
) -> None:
    """Convert our dataset format to latents-compatible format.

    Args:
        input_dataset_path: Path to our JSONL dataset
        output_path: Path to save latents-compatible JSON

    Example:
        >>> create_latents_compatible_dataset(
        ...     "data/raw/prompts.jsonl",
        ...     "data/processed/latents_format.json"
        ... )
    """
    from src.dataset.loader import load_dataset

    dataset = load_dataset(input_dataset_path)

    # Convert to latents format
    latents_format = []
    for item in dataset:
        latents_format.append({
            "positive": item["long_prompt"],
            "negative": item["short_prompt"],
            "domain": item.get("domain"),
            "task": item.get("task"),
        })

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(latents_format, f, indent=2)

    logger.info(f"Saved latents-compatible dataset to {output_path}")


def compare_approaches_report(
    probe_results: Dict, steering_results: Dict, output_path: str
) -> None:
    """Generate comparison report between probe and steering approaches.

    Args:
        probe_results: Results from probe-based analysis
        steering_results: Results from steering-based analysis
        output_path: Path to save report

    Example:
        >>> compare_approaches_report(probe_res, steering_res, "results/comparison.md")
    """
    report = f"""# Probe vs Steering Comparison Report

## Probe-Based Approach

- Accuracy: {probe_results.get('accuracy', 'N/A')}
- F1 Score: {probe_results.get('f1', 'N/A')}
- Method: Linear/MLP classifier on activations

## Steering-Based Approach (CAA)

- Uses contrastive activation addition
- Directly modifies generation behavior
- Pre-trained on temporal scope dimension

## Key Findings

### Activation Overlap
- Cosine similarity between probe weights and steering vectors: [TODO]

### Behavioral Comparison
- [Analysis of generation quality under steering]

### Circuit Analysis Agreement
- [Comparison of important heads identified by both methods]

## Conclusions

[Summary of which approach is more effective for temporal horizon detection]
"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(report)

    logger.info(f"Saved comparison report to {output_path}")
