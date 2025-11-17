# Temporal Horizon Detection in Large Language Models

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A research framework for detecting and analyzing temporal reasoning scope (>1 year vs <1 year planning horizons) in large language models using activation probing and mechanistic interpretability techniques.

## Overview

This repository implements a comprehensive pipeline for investigating how LLMs internally represent temporal scope when processing planning-related prompts. We use activation probing, circuit analysis, and causal interventions to identify the neural mechanisms responsible for temporal horizon detection in GPT-2 and Pythia models.

### Key Features

- **Automated Dataset Generation**: Generate paired prompts with short/long temporal horizons across multiple domains (business, science, personal)
- **Activation Extraction**: Extract and cache activations from transformer models using TransformerLens
- **Probe Training**: Train linear and MLP probes to classify temporal horizons from internal activations
- **Circuit Analysis**: Identify attention heads and components responsible for temporal reasoning via activation patching and ablation
- **Divergence Detection**: Detect mismatches between stated and internal temporal representations
- **Cross-Model Validation**: Compare temporal reasoning mechanisms across model families
- **Comprehensive Visualization**: Generate publication-ready figures and interactive visualizations
- **Latents Integration**: Compare probe-based approach with Contrastive Activation Addition (CAA) steering from the [latents library](https://github.com/justinshenk/latents)

### Dual Approach: Probing + Steering

This project uniquely combines **two complementary approaches** to temporal horizon detection:

1. **Probe-Based** (Our Core Approach): Train classifiers on activations to measure what temporal information is encoded
2. **Steering-Based** (via Latents Integration): Use Contrastive Activation Addition to directly control temporal scope in generation

The integration validates findings across both methods and provides richer analysis. See [Latents Integration Guide](docs/latents_integration.md) for details.

## Installation

### Option 1: pip (Recommended)

```bash
# Clone the repository with submodules
git clone --recursive https://github.com/yourusername/temporal-horizon-detection.git
cd temporal-horizon-detection

# Or if already cloned, initialize submodules
git submodule update --init --recursive

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package and dependencies
pip install -e .

# Install optional dependencies for dataset generation
pip install -e ".[api]"

# Install development dependencies
pip install -e ".[dev]"

# Install latents library for steering experiments (optional but recommended)
pip install -e external/latents
```

### Option 2: conda

```bash
# Clone the repository
git clone https://github.com/yourusername/temporal-horizon-detection.git
cd temporal-horizon-detection

# Create conda environment
conda env create -f environment.yml
conda activate temporal-horizon-detection
```

### GPU Support

For GPU acceleration, install PyTorch with CUDA:

```bash
# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Quick Start

### 1. Generate Dataset

```bash
# Generate 300 prompt pairs across all domains
python scripts/generate_dataset.py \
    --num-pairs 300 \
    --domains business science personal \
    --output data/raw/prompts.jsonl \
    --api-provider openai \
    --api-key $OPENAI_API_KEY
```

### 2. Extract Activations

```python
from src.models.model_loader import load_model
from src.models.activation_extractor import ActivationExtractor
from src.dataset.loader import load_dataset

# Load model and dataset
model = load_model("gpt2", device="cuda")
dataset = load_dataset("data/raw/prompts.jsonl")

# Extract activations
extractor = ActivationExtractor(model)
activations = extractor.extract_batch(
    dataset["prompts"],
    layers=[8, 9, 10, 11],  # Focus on later layers
    save_path="data/processed/activations_gpt2.h5"
)
```

### 3. Train Probes

```bash
# Train probes on all layers
python scripts/train_probes.py \
    --config configs/probe_config.yaml \
    --activations data/processed/activations_gpt2.h5 \
    --output-dir checkpoints/probes/
```

### 4. Run Circuit Analysis

```bash
# Identify important attention heads
python scripts/run_circuit_analysis.py \
    --model gpt2 \
    --probe checkpoints/probes/best_probe.pt \
    --method activation_patching \
    --output-dir paper/figures/
```

### 5. Evaluate and Visualize

```bash
# Comprehensive evaluation
python scripts/evaluate_model.py \
    --probe checkpoints/probes/best_probe.pt \
    --activations data/processed/activations_gpt2.h5 \
    --test-split data/raw/prompts_test.jsonl \
    --output results/evaluation_report.json
```

## Project Structure

```
temporal-horizon-detection/
├── src/                    # Source code
│   ├── dataset/           # Dataset generation and loading
│   ├── models/            # Model loading and activation extraction
│   ├── probing/           # Probe architectures and training
│   ├── circuits/          # Circuit analysis and interpretability
│   ├── analysis/          # Statistical analysis and divergence detection
│   └── utils/             # Utilities and visualization
├── scripts/               # CLI scripts for experiments
├── notebooks/             # Jupyter notebooks for exploration
├── tests/                 # Unit and integration tests
├── configs/               # YAML configuration files
├── docs/                  # Documentation
├── data/                  # Data storage (gitignored)
│   ├── raw/              # Generated datasets
│   └── processed/        # Extracted activations and probes
└── paper/                 # Paper drafts and figures
```

## Usage Examples

### Example 1: Custom Domain Dataset

```python
from src.dataset.generator import DatasetGenerator
from src.dataset.templates import create_custom_template

# Define custom template
template = create_custom_template(
    domain="healthcare",
    short_horizon="developing a patient care protocol",
    long_horizon="transforming healthcare delivery systems"
)

# Generate dataset
generator = DatasetGenerator(api_provider="anthropic")
dataset = generator.generate(
    templates=[template],
    num_pairs=50,
    save_path="data/raw/healthcare_prompts.jsonl"
)
```

### Example 2: Layer-wise Probe Analysis

```python
from src.probing.trainer import ProbeTrainer
from src.probing.probe import LinearProbe
import matplotlib.pyplot as plt

# Train probes for each layer
results = {}
for layer in range(12):
    probe = LinearProbe(hidden_size=768)
    trainer = ProbeTrainer(probe, learning_rate=1e-3)

    metrics = trainer.train(
        activations=activations[f"layer_{layer}"],
        labels=labels,
        epochs=50
    )
    results[layer] = metrics["test_accuracy"]

# Plot layer-wise accuracy
plt.plot(results.keys(), results.values())
plt.xlabel("Layer")
plt.ylabel("Probe Accuracy")
plt.title("Temporal Horizon Detection Across Layers")
plt.savefig("paper/figures/layer_wise_accuracy.png")
```

### Example 3: Activation Patching

```python
from src.circuits.activation_patching import ActivationPatcher
from src.circuits.attribution import compute_head_importance

# Patch activations from short to long horizon prompts
patcher = ActivationPatcher(model)
results = patcher.patch_heads(
    source_prompt="Plan for next quarter's product launch",
    target_prompt="Plan for transforming the industry over the next decade",
    layers=[9, 10, 11],
    heads=list(range(12))
)

# Identify most important heads
important_heads = compute_head_importance(results, top_k=10)
print(f"Top heads for temporal reasoning: {important_heads}")
```

### Example 4: Steering-Based Temporal Control (Latents Integration)

```python
from src.utils.latents_integration import TemporalSteeringIntegration
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load model
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Initialize steering
steering = TemporalSteeringIntegration(model, tokenizer)

# Option 1: Use pre-trained temporal steering
steering.load_pretrained_temporal_steering('gpt2')

# Generate with different temporal scopes
prompt = "How should we address climate change?"

short_term = steering.generate_with_steering(prompt, strength=-0.8)  # Immediate focus
long_term = steering.generate_with_steering(prompt, strength=0.8)   # Long-term focus

print(f"Short-term: {short_term}")
print(f"Long-term: {long_term}")

# Option 2: Extract steering vectors from your dataset
from src.dataset.loader import load_dataset

dataset = load_dataset("data/raw/prompts.jsonl")
steering_vectors = steering.extract_steering_vectors_from_dataset(dataset)

# Compare with probe-based approach
similarities = steering.analyze_steering_activation_overlap(
    steering_vectors, probe.linear.weight
)
print(f"Probe-steering alignment: {similarities}")
```

See [Latents Integration Guide](docs/latents_integration.md) for comprehensive comparison between probe-based and steering-based approaches.

## Configuration

All experiments can be configured via YAML files in `configs/`:

- `dataset_config.yaml`: Dataset generation parameters
- `model_config.yaml`: Model selection and hyperparameters
- `probe_config.yaml`: Probe architecture and training settings
- `experiment_config.yaml`: Full pipeline configuration

Example configuration:

```yaml
# probe_config.yaml
probe:
  type: "mlp"  # or "linear"
  hidden_size: 768
  num_layers: 2
  dropout: 0.1

training:
  learning_rate: 1e-3
  batch_size: 32
  epochs: 100
  early_stopping_patience: 10

data:
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
```

## Reproducing Results

To reproduce all results from the paper:

```bash
# Run full pipeline
bash scripts/reproduce_results.sh

# This will:
# 1. Generate dataset
# 2. Extract activations for all models
# 3. Train probes
# 4. Run circuit analysis
# 5. Generate all figures and tables
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test module
pytest tests/test_probing.py -v
```

## Documentation

- [Methodology](docs/methodology.md): Detailed explanation of our approach
- [API Reference](docs/api_reference.md): Complete API documentation
- [Latents Integration](docs/latents_integration.md): Probe vs steering comparison guide
- [Experiment Log](docs/experiment_log.md): Template for tracking experiments
- [Results](docs/results.md): Experimental results and analysis

## Citation

If you use this code in your research, please cite:

```bibtex
@article{temporalhorizon2024,
  title={Temporal Horizon Detection in Large Language Models},
  author={Your Name and Collaborators},
  journal={arXiv preprint},
  year={2024}
}
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure code follows our style guidelines (Black formatting, type hints, docstrings).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [TransformerLens](https://github.com/neelnanda-io/TransformerLens) by Neel Nanda
- Inspired by mechanistic interpretability work at Anthropic and EleutherAI
- Dataset generation powered by OpenAI and Anthropic APIs

## Contact

For questions or issues:
- Open an issue on GitHub
- Email: research@example.com
- Twitter: @YourHandle

## Links

- [Paper (arXiv)](https://arxiv.org/abs/placeholder)
- [Documentation](https://temporal-horizon-detection.readthedocs.io/)
- [Interactive Demo](https://huggingface.co/spaces/yourspace/temporal-horizon-demo)
