# Data Directory

This directory contains datasets and processed activations for temporal horizon detection.

## Structure

```
data/
├── raw/               # Generated prompt datasets
│   └── prompts.jsonl  # Main dataset (300 pairs)
└── processed/         # Extracted activations and processed data
    ├── activations_gpt2.h5      # GPT-2 activations
    ├── activations_pythia.h5    # Pythia activations
    └── labels.npy               # Ground truth labels
```

## Raw Data Format

Each line in `prompts.jsonl` contains:
```json
{
  "pair_id": 0,
  "domain": "business",
  "task": "launching new product",
  "short_prompt": "Develop a plan for launching new product over the next quarter.",
  "long_prompt": "Develop a plan for launching new product over the next decade.",
  "label_short": 0,
  "label_long": 1,
  "template": "Develop a plan for {task} over the {horizon}."
}
```

## Processed Data Format

Activations are stored in HDF5 format with keys:
- `layer_0`, `layer_1`, ..., `layer_11`: Activation arrays [num_samples, hidden_size]
- `metadata`: Model information and extraction parameters

Labels are stored as NumPy arrays:
- Shape: [num_samples]
- Values: 0 (short horizon), 1 (long horizon)

## Generating Data

To generate a new dataset:
```bash
python scripts/generate_dataset.py \
    --num-pairs 300 \
    --output data/raw/prompts.jsonl \
    --api-provider openai
```

## Data Statistics

- Total pairs: 300
- Total prompts: 600 (2 per pair)
- Domains: Business (100), Science (100), Personal (100)
- Label distribution: 50% short, 50% long

## Citation

If you use this dataset, please cite:
```
[Citation information]
```
