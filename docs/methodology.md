# Methodology

## Overview

This document describes the methodology for detecting temporal horizon representations in large language models using activation probing and circuit analysis.

## Research Questions

1. **RQ1**: Do LLMs internally represent different temporal horizons (short vs. long term)?
2. **RQ2**: Which layers and attention heads are responsible for temporal horizon processing?
3. **RQ3**: Can we detect divergence between stated and internal temporal representations?

## Approach

### 1. Dataset Generation

We generate paired prompts with different temporal horizons:
- **Short horizon**: < 1 year (e.g., "next quarter", "next month")
- **Long horizon**: > 1 year (e.g., "next decade", "coming years")

Domains:
- Business planning
- Scientific research
- Personal development

### 2. Activation Extraction

Extract activations from transformer layers using TransformerLens:
- Target layers: Later layers (8-11 for GPT-2)
- Component: Residual stream activations
- Position: Last token (planning-relevant)

### 3. Probe Training

Train linear and MLP probes to classify temporal horizons:
- **Linear probe**: Logistic regression for linear separability
- **MLP probe**: Non-linear decision boundaries

Metrics:
- Accuracy, F1, AUC-ROC
- Per-domain performance
- Variance under paraphrasing

### 4. Circuit Analysis

#### Activation Patching
- Patch attention head outputs between short/long prompts
- Measure impact on probe predictions
- Identify heads that flip predictions

#### Ablation Studies
- Zero-ablate individual heads
- Measure performance degradation
- Rank heads by importance

#### Attribution
- Gradient-based attribution scores
- Integrated gradients
- Attention pattern analysis

### 5. Divergence Detection

Identify cases where:
- Stated horizon: "10-year plan"
- Internal representation: Short-term (<1 year)

Methods:
- Compare activation similarity
- Statistical hypothesis testing
- Confidence analysis

## Expected Results

1. **Temporal representations emerge**: Probes achieve >80% accuracy on later layers
2. **Specific heads matter**: 5-10 heads are critical for temporal reasoning
3. **Divergence exists**: ~10-15% of prompts show stated/internal mismatch

## Limitations

- Limited to GPT-2 and Pythia models
- Binary classification (short vs. long)
- English language only
- Focused on planning contexts

## Future Work

- Multi-class temporal scales
- Cross-lingual analysis
- Fine-grained timeline extraction
- Causal interventions on training
