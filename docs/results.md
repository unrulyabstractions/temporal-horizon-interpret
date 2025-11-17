# Results

## Summary

This document contains the main results from temporal horizon detection experiments.

## Dataset Statistics

- Total prompt pairs: 300
- Domains: Business (100), Science (100), Personal (100)
- Average prompt length:
  - Short horizon: ~XX characters
  - Long horizon: ~XX characters

## Probe Performance

### Linear Probe

| Layer | Accuracy | F1 Score | AUC-ROC |
|-------|----------|----------|---------|
| 8     | XX.X%    | X.XXX    | X.XXX   |
| 9     | XX.X%    | X.XXX    | X.XXX   |
| 10    | XX.X%    | X.XXX    | X.XXX   |
| 11    | XX.X%    | X.XXX    | X.XXX   |

### MLP Probe

| Layer | Accuracy | F1 Score | AUC-ROC |
|-------|----------|----------|---------|
| 8     | XX.X%    | X.XXX    | X.XXX   |
| 9     | XX.X%    | X.XXX    | X.XXX   |
| 10    | XX.X%    | X.XXX    | X.XXX   |
| 11    | XX.X%    | X.XXX    | X.XXX   |

**Best result**: Layer XX with XX.X% accuracy

## Per-Domain Performance

| Domain   | Accuracy | F1 Score |
|----------|----------|----------|
| Business | XX.X%    | X.XXX    |
| Science  | XX.X%    | X.XXX    |
| Personal | XX.X%    | X.XXX    |

## Circuit Analysis

### Top Attention Heads

1. Layer 10, Head 2: Effect size X.XXX
2. Layer 11, Head 7: Effect size X.XXX
3. Layer 9, Head 5: Effect size X.XXX
4. ...

### Ablation Results

- Ablating top 5 heads: -XX.X% accuracy
- Ablating top 10 heads: -XX.X% accuracy

## Divergence Detection

- Mismatch rate: XX.X%
- High-confidence mismatches: XX cases
- Examples: [See detailed analysis]

## Cross-Model Comparison

| Model        | Best Accuracy | Best Layer |
|--------------|---------------|------------|
| GPT-2 Small  | XX.X%         | Layer XX   |
| GPT-2 Medium | XX.X%         | Layer XX   |
| Pythia-160M  | XX.X%         | Layer XX   |

## Key Findings

1. **Finding 1**: [Description]
2. **Finding 2**: [Description]
3. **Finding 3**: [Description]

## Visualizations

- Layer-wise accuracy curves: `figures/layer_accuracy.png`
- Circuit diagrams: `figures/circuit_diagram.png`
- Confusion matrices: `figures/confusion_matrix.png`

## Conclusions

[Summary of main conclusions]

## Limitations and Future Work

[Discussion of limitations and future directions]
