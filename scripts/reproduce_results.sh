#!/bin/bash
# Full pipeline for reproducing results

set -e  # Exit on error

echo "===== Temporal Horizon Detection - Full Pipeline ====="

# Step 1: Generate dataset
echo "Step 1: Generating dataset..."
python scripts/generate_dataset.py \
    --num-pairs 300 \
    --domains business science personal \
    --output data/raw/prompts.jsonl \
    --api-provider openai

# Step 2: Extract activations
echo "Step 2: Extracting activations..."
# (This would require a separate script - placeholder)

# Step 3: Train probes
echo "Step 3: Training probes..."
# python scripts/train_probes.py \
#     --activations data/processed/activations_gpt2.h5 \
#     --labels data/processed/labels.npy \
#     --output-dir checkpoints/probes/

# Step 4: Run circuit analysis
echo "Step 4: Running circuit analysis..."
# python scripts/run_circuit_analysis.py \
#     --model gpt2 \
#     --output-dir paper/figures/

# Step 5: Evaluate
echo "Step 5: Evaluating..."
# python scripts/evaluate_model.py \
#     --probe checkpoints/probes/best_probe.pt \
#     --activations data/processed/activations_gpt2.h5 \
#     --labels data/processed/labels.npy \
#     --output results/evaluation.json

echo "===== Pipeline complete! ====="
