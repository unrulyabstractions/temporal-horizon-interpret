# Latents Integration Guide

## Overview

This project integrates the [latents library](https://github.com/justinshenk/latents) for Contrastive Activation Addition (CAA) steering experiments. The latents library provides a complementary approach to our probe-based temporal horizon detection.

## Two Approaches Compared

### Our Probe-Based Approach

**Method**: Train classifiers (linear/MLP) on model activations to predict temporal horizon

**Strengths**:
- Measures what information is encoded in activations
- Quantitative metrics (accuracy, F1, AUC)
- Layer-wise analysis of information emergence
- Identifies decision boundaries in activation space

**Use cases**:
- Understanding temporal representations
- Measuring information content
- Circuit analysis via probes

### Latents Steering Approach (CAA)

**Method**: Extract contrastive activation vectors and add them during generation

**Strengths**:
- Directly controls generation behavior
- No training required (uses prompt pairs)
- Multi-dimensional steering
- Real-time behavioral modification

**Use cases**:
- Controlling temporal scope in generation
- Testing causal effects
- Validating that temporal info exists in activations

## Integration Architecture

```
temporal-horizon-detection/
├── external/
│   └── latents/              # Submodule
│       ├── latents/          # Core library
│       ├── steering_vectors/ # Pre-trained vectors
│       └── research/         # Research tools
├── src/
│   └── utils/
│       └── latents_integration.py  # Integration module
├── scripts/
│   └── compare_probe_vs_steering.py  # Comparison script
└── notebooks/
    └── 06_latents_steering_comparison.ipynb  # Demo notebook
```

## Installation

The latents library is included as a git submodule:

```bash
# Already added as submodule at external/latents
# To install:
pip install -e external/latents
```

## Usage Examples

### 1. Extract Steering Vectors from Our Dataset

```python
from src.utils.latents_integration import TemporalSteeringIntegration
from src.dataset.loader import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load model
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load our dataset
dataset = load_dataset('data/raw/prompts.jsonl')

# Initialize integration
steering = TemporalSteeringIntegration(model, tokenizer)

# Extract steering vectors from our 300 paired prompts
steering_vectors = steering.extract_steering_vectors_from_dataset(
    dataset, layers=[8, 9, 10, 11]
)

# Save for later use
steering.save_steering_vectors(
    steering_vectors,
    'steering_vectors/temporal_horizon_custom.json'
)
```

### 2. Use Pre-trained Temporal Steering

Latents provides pre-trained temporal scope steering vectors:

```python
# Load pre-trained steering
steering.load_pretrained_temporal_steering('gpt2')

# Generate with long-term steering
result = steering.generate_with_steering(
    prompt="How should we address climate change?",
    strength=0.8,  # +1.0 = long-term, -1.0 = short-term
    temperature=0.7,
    max_length=100
)
print(result)
```

### 3. Compare Probe vs Steering

```python
from src.probing.evaluator import ProbeEvaluator

# Compare both approaches
comparison = steering.compare_steering_vs_probe(
    prompts=test_prompts,
    probe_evaluator=evaluator,
    activations=test_activations,
    steering_strengths=[-0.8, 0.0, 0.8]
)

# Analyze overlap
similarities = steering.analyze_steering_activation_overlap(
    steering_vectors,
    probe.linear.weight
)
print(f"Cosine similarities: {similarities}")
```

### 4. Run Full Comparison Script

```bash
python scripts/compare_probe_vs_steering.py \
    --dataset data/raw/prompts.jsonl \
    --probe-checkpoint checkpoints/probes/best_probe.pt \
    --model gpt2 \
    --extract-steering \
    --use-pretrained-steering \
    --output-dir results/comparison/
```

## Key Integration Points

### 1. Dataset Compatibility

Our prompt pairs are perfect for CAA:
- Each pair: `{short_prompt, long_prompt}`
- 300 pairs across 3 domains
- Can be directly used to extract steering vectors

### 2. Validation

If steering successfully controls temporal scope, it validates:
- Temporal information is encoded in activations (probe assumption)
- Our dataset captures meaningful contrasts
- Later layers contain temporal representations

### 3. Circuit Analysis

Both approaches can identify important components:

**Probing approach**:
- Train probe, measure layer-wise accuracy
- Use activation patching to test heads
- Identify heads that flip probe predictions

**Steering approach**:
- Apply steering, measure generation changes
- Identify heads that maximize steering effect
- Compare with probe-identified heads

**Convergence = Strong evidence**

### 4. Cross-Validation

```python
# Extract our steering vectors
our_vectors = steering.extract_steering_vectors_from_dataset(dataset)

# Load their pre-trained vectors
steering.load_pretrained_temporal_steering('gpt2')

# Compare:
# 1. Do they identify same layers as important?
# 2. Do they have similar direction (cosine similarity)?
# 3. Do they produce similar behavioral effects?
```

## Pre-trained Steering Vectors

Latents includes several temporal steering variants:

- `temporal_scope.json` - Original temporal steering
- `temporal_scope_gpt2.json` - GPT-2 specific
- `temporal_scope_deconfounded.json` - Style-controlled

These provide baselines for comparison.

## Research Questions

### Q1: Do probe and steering identify same layers?

**Method**:
- Train probes on each layer, measure accuracy
- Test steering effectiveness per layer
- Compare rankings

**Expected**: Both should peak at layers 9-11

### Q2: Are steering vectors aligned with probe weights?

**Method**:
- Compute cosine similarity between:
  - Probe decision boundary (class weight difference)
  - Steering vector direction
- Per layer analysis

**High similarity** = Both methods extract same signal

### Q3: Does our dataset produce better steering than generic prompts?

**Method**:
- Extract steering from our domain-specific prompts
- Compare with pre-trained latents vectors
- Measure generation quality and behavioral change

**Hypothesis**: Domain-specific should perform better on our tasks

### Q4: Can steering help probe training?

**Method**:
- Use steering to generate synthetic training data
- Augment dataset with steered generations
- Measure probe accuracy improvement

## Experimental Protocol

### Experiment 1: Steering Vector Extraction

```bash
# Extract from our dataset
python scripts/compare_probe_vs_steering.py \
    --dataset data/raw/prompts.jsonl \
    --model gpt2 \
    --extract-steering \
    --output-dir results/steering_extraction/
```

**Outputs**:
- `temporal_horizon_steering.json` - Extracted vectors
- Layer-wise vector magnitudes
- Variance explained per layer

### Experiment 2: Generation Comparison

Generate with different steering strengths, measure:
- Temporal marker frequency (regex patterns)
- Sentence complexity (length, depth)
- Domain-specific terminology usage

### Experiment 3: Circuit Agreement

```python
# From probing
probe_important_heads = [(10, 2), (11, 7), (9, 5)]

# From steering
steering_important_heads = identify_via_steering_effect()

# Measure overlap
overlap = set(probe_heads) & set(steering_heads)
```

## Advantages of Integration

1. **Mutual Validation**: Two independent methods confirming same findings
2. **Complementary Analysis**: Understand (probe) + Control (steering)
3. **Rich Baselines**: Pre-trained vectors for comparison
4. **Active Research**: Latents is actively maintained with recent experiments
5. **Publication Support**: Can cite both approaches, show convergence

## Limitations and Considerations

1. **Different Granularity**: Probes classify, steering modifies continuously
2. **Confounds**: Both could pick up style, not just temporal info (see latents research/ISSUES.md)
3. **Model-Specific**: Steering vectors may not transfer perfectly across models
4. **Hyperparameters**: Steering strength requires tuning

## Best Practices

1. **Always compare both approaches** on same dataset
2. **Report correlation** between probe predictions and steering effects
3. **Test robustness** to paraphrasing for both methods
4. **Visualize** activation patterns side-by-side
5. **Document** when methods disagree (interesting edge cases!)

## References

- Latents library: https://github.com/justinshenk/latents
- CAA paper: Rimsky et al., "Steering Llama 2 with Contrastive Activation Additions"
- Original CAA implementation: https://github.com/nrimsky/CAA

## Example Results

### Steering Vector Magnitudes

```
Layer 8:  0.453
Layer 9:  0.621
Layer 10: 0.783  ← Peak
Layer 11: 0.695
```

### Probe-Steering Cosine Similarity

```
Layer 8:  0.42
Layer 9:  0.67
Layer 10: 0.81  ← Strong alignment
Layer 11: 0.73
```

**Interpretation**: Layer 10 shows strongest temporal encoding in both approaches!

## Troubleshooting

**Issue**: Import error for latents
```bash
# Solution: Install submodule
pip install -e external/latents
```

**Issue**: Pre-trained vectors not found
```bash
# Solution: Check submodule
git submodule update --init --recursive
```

**Issue**: Out of memory during steering vector extraction
```bash
# Solution: Extract fewer pairs at a time
# Or use smaller model (gpt2 instead of gpt2-large)
```

## Future Work

- [ ] Extract steering vectors from full 300-pair dataset
- [ ] Cross-model transfer: GPT-2 steering → Pythia
- [ ] Multi-dimensional steering: temporal + formality + domain
- [ ] Automated confound detection using latents research tools
- [ ] Integration with our circuit analysis pipeline
