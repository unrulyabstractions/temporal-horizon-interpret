# Experiment Log

Template for tracking experiments.

## Experiment 1: Baseline GPT-2 Probing

**Date**: YYYY-MM-DD
**Researcher**: Name

### Setup
- Model: GPT-2 (124M)
- Dataset: 300 pairs (business, science, personal)
- Probe: MLP (2 layers, 384 hidden dim)
- Layers: 8-11

### Hyperparameters
- Learning rate: 1e-3
- Batch size: 32
- Epochs: 100
- Early stopping patience: 10

### Results
- Best validation accuracy: XX.X%
- Test accuracy: XX.X%
- F1 score: X.XXX
- AUC-ROC: X.XXX

### Analysis
- Best performing layer: Layer X (XX.X% accuracy)
- Per-domain accuracy:
  - Business: XX.X%
  - Science: XX.X%
  - Personal: XX.X%

### Observations
- [Key findings]
- [Unexpected results]
- [Issues encountered]

### Next Steps
- [Follow-up experiments]

---

## Experiment 2: Circuit Analysis

**Date**: YYYY-MM-DD

### Setup
- Method: Activation patching
- Target layers: 9-11
- Number of heads tested: 36

### Results
- Top 10 important heads: [list]
- Effect sizes: [values]

### Analysis
- [Findings about temporal reasoning circuits]

---
