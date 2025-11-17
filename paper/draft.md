# Temporal Horizon Detection in Large Language Models

## Abstract

We investigate how large language models (LLMs) internally represent temporal scope when processing planning-related prompts. Using activation probing and circuit analysis on GPT-2 and Pythia models, we find that [key findings]. Our results suggest [implications].

**Keywords**: Large Language Models, Temporal Reasoning, Mechanistic Interpretability, Activation Probing

## 1. Introduction

Planning and reasoning about the future require understanding temporal scope. Do language models represent "next quarter" differently from "next decade" in their internal activations?

### 1.1 Research Questions

1. Do LLMs internally distinguish between short-term and long-term temporal horizons?
2. Which layers and attention heads are responsible for temporal horizon processing?
3. Can we detect divergence between stated and internal temporal representations?

### 1.2 Contributions

- Generated dataset of 300 paired prompts with different temporal horizons
- Trained probes achieving XX% accuracy on temporal horizon classification
- Identified top-10 attention heads responsible for temporal reasoning
- Discovered XX% divergence rate between stated and internal horizons

## 2. Related Work

### 2.1 Temporal Reasoning in LLMs

[Literature review]

### 2.2 Mechanistic Interpretability

[TransformerLens, activation patching, etc.]

### 2.3 Probing Classifiers

[Probe methodology literature]

## 3. Methodology

### 3.1 Dataset Generation

We generated 300 prompt pairs across three domains: business planning, scientific research, and personal development. Each pair contains:
- Short horizon: temporal scope < 1 year
- Long horizon: temporal scope > 1 year

### 3.2 Activation Extraction

Using TransformerLens, we extracted activations from layers 8-11 of GPT-2...

### 3.3 Probe Training

We trained both linear and MLP probes to classify temporal horizons...

### 3.4 Circuit Analysis

Through activation patching and ablation, we identified...

## 4. Results

### 4.1 Probe Performance

Our best probe achieved XX% accuracy on layer XX...

| Layer | Linear Probe | MLP Probe |
|-------|-------------|-----------|
| 8     | XX%         | XX%       |
| 9     | XX%         | XX%       |
| 10    | XX%         | XX%       |
| 11    | XX%         | XX%       |

### 4.2 Important Attention Heads

We identified 10 attention heads critical for temporal reasoning:

1. Layer 10, Head 2 (effect size: X.XX)
2. Layer 11, Head 7 (effect size: X.XX)
...

### 4.3 Divergence Analysis

We found XX% of prompts exhibit divergence between stated and internal horizons...

## 5. Discussion

### 5.1 Emergence of Temporal Representations

Our results show that temporal horizon information is linearly separable in later layers...

### 5.2 Circuit Structure

The identified attention heads form a coherent circuit for temporal reasoning...

### 5.3 Implications for AI Safety

Divergence between stated and internal representations has implications for...

## 6. Limitations

- Binary classification only
- Limited to specific model families
- English language only

## 7. Conclusion

We demonstrated that LLMs develop internal representations of temporal horizons that can be reliably detected via probing...

## References

[1] ...
[2] ...

## Appendix A: Dataset Examples

[Examples of generated prompts]

## Appendix B: Additional Results

[Supplementary tables and figures]
