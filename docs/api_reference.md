# API Reference

## Dataset Module

### `DatasetGenerator`
Generate temporal horizon prompt pairs using LLM APIs.

```python
from src.dataset.generator import DatasetGenerator

generator = DatasetGenerator(api_provider="openai")
dataset = generator.generate(num_pairs=100, domains=["business"])
```

### `PromptTemplate`
Template for structured prompt generation.

## Models Module

### `load_model(model_name, device)`
Load transformer models via TransformerLens.

### `ActivationExtractor`
Extract and cache model activations.

```python
from src.models.activation_extractor import ActivationExtractor

extractor = ActivationExtractor(model)
activations = extractor.extract_batch(texts, layers=[10, 11])
```

## Probing Module

### `LinearProbe`, `MLPProbe`
Probe architectures for classification.

### `ProbeTrainer`
Training loop with early stopping.

### `ProbeEvaluator`
Comprehensive evaluation metrics.

## Circuits Module

### `ActivationPatcher`
Perform activation patching experiments.

### `AblationAnalyzer`
Head ablation analysis.

## Analysis Module

### `DivergenceDetector`
Detect stated vs. internal horizon mismatches.

## Utils Module

### `setup_logger()`, `ExperimentLogger`
Logging utilities.

### Visualization functions
- `plot_activation_heatmap()`
- `plot_probe_accuracy_curve()`
- `plot_layer_wise_accuracy()`

For detailed API documentation, see inline docstrings.
