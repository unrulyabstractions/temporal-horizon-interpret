# TransformerLens Usage Guide for Temporal Horizon Detection

## Overview

This project uses [TransformerLens](https://github.com/neelnanda-io/TransformerLens) as the core library for mechanistic interpretability. TransformerLens provides clean access to model internals specifically designed for interpretability research.

## Why TransformerLens?

### Purpose-Built for Interpretability

TransformerLens wraps HuggingFace models with additional functionality:
- **Clean hook system**: Easy access to intermediate activations
- **Standardized names**: Consistent naming across model families
- **Caching utilities**: Efficient activation extraction
- **Intervention tools**: Built-in support for activation patching

### Advantages for Our Research

1. **Consistent API** across GPT-2 and Pythia models
2. **Direct activation access** without manual forward pass manipulation
3. **Hook-based extraction** for specific layers/components
4. **Memory-efficient caching** for large-scale experiments

## Installation

```bash
pip install transformer-lens
```

TransformerLens automatically downloads HuggingFace models and wraps them in `HookedTransformer` class.

## Core Components

### 1. HookedTransformer

The main model class that wraps transformers with interpretability hooks.

```python
from transformer_lens import HookedTransformer

# Load GPT-2
model = HookedTransformer.from_pretrained("gpt2", device="cpu")

# Model has same interface as HuggingFace but with hooks
logits = model("Hello world")
```

### 2. Activation Names

TransformerLens uses standardized activation names:

```python
from transformer_lens.utils import get_act_name

# Get activation name for layer 10, residual stream
act_name = get_act_name("resid_post", 10)
# Returns: "blocks.10.hook_resid_post"

# Other components:
# - "resid_pre": before transformer block
# - "resid_post": after transformer block (WHAT WE USE)
# - "attn_out": attention output
# - "mlp_out": MLP output
```

### 3. Cache-Based Extraction

Extract all activations at once using cache:

```python
# Run model with cache
tokens = model.to_tokens("Plan for the next decade")
logits, cache = model.run_with_cache(tokens)

# Access specific activation
layer_10_resid = cache["blocks.10.hook_resid_post"]
# Shape: [batch, seq_len, d_model]
```

## Our Implementation

### Model Loading (`src/models/model_loader.py`)

```python
from src.models.model_loader import load_model

# Load GPT-2 with automatic device selection
model = load_model("gpt2")  # Uses CUDA if available

# Load Pythia
model = load_model("pythia-160m", device="cpu")

# Supported models:
# - gpt2, gpt2-medium, gpt2-large, gpt2-xl
# - pythia-70m, pythia-160m, pythia-410m, pythia-1b
```

### Activation Extraction (`src/models/activation_extractor.py`)

```python
from src.models.activation_extractor import ActivationExtractor

extractor = ActivationExtractor(model)

# Extract from multiple prompts
prompts = [
    "Plan for next quarter",  # Short horizon
    "Plan for next decade",   # Long horizon
]

activations = extractor.extract_batch(
    prompts,
    layers=[8, 9, 10, 11],  # Later layers where temporal reasoning emerges
    batch_size=8,
    position_strategy="last"  # Use last token
)

# Returns: {"layer_8": array, "layer_9": array, ...}
# Each array shape: [num_prompts, hidden_size]
```

## Critical Settings for Temporal Horizon Detection

### Layer Selection

**Why layers 8-11?**

```python
# GPT-2 has 12 layers (0-11)
# Our hypothesis: temporal reasoning emerges in later layers

layers = [8, 9, 10, 11]  # Focus on last 4 layers

# Validation: Train probes on all layers, confirm peak accuracy at 9-11
```

### Component Selection

**Why residual stream (`resid_post`)?**

```python
component = "resid_post"  # Residual stream after transformer block

# Alternatives we DON'T use:
# - "attn_out": Only attention (misses MLP computation)
# - "mlp_out": Only MLP (misses attention)
# - "resid_pre": Before block (less processed)

# resid_post captures full information after both attention and MLP
```

### Position Strategy

**Why last token?**

```python
position_strategy = "last"  # Extract activation at last token position

# For prompts like "Plan for next decade"
# Last token captures accumulated temporal reasoning

# Alternatives:
# - "mean": Average over all positions (dilutes signal)
# - "first": Only initial context (misses reasoning)
```

## Extraction Pipeline

### Step 1: Tokenization

```python
# TransformerLens handles tokenization
tokens = model.to_tokens(["Plan for next decade"], prepend_bos=True)

# prepend_bos=True adds beginning-of-sequence token
# Shape: [batch=1, seq_len]
```

### Step 2: Forward Pass with Cache

```python
# Run model and cache all activations
logits, cache = model.run_with_cache(tokens, remove_batch_dim=False)

# remove_batch_dim=False keeps batch dimension
# cache is dictionary of all hook activations
```

### Step 3: Extract Specific Activations

```python
from transformer_lens.utils import get_act_name

# Extract layer 10 residual stream
act_name = get_act_name("resid_post", 10)
activations = cache[act_name]

# Shape: [batch, seq_len, hidden_size]
# For GPT-2: [batch, seq_len, 768]
```

### Step 4: Apply Position Strategy

```python
# Take last token
final_act = activations[:, -1, :]  # [batch, 768]

# This is what we feed to probes
```

## Full Example

```python
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
import torch

# Load model
model = HookedTransformer.from_pretrained("gpt2", device="cpu")

# Prepare prompts
short_horizon = "Plan for the next quarter"
long_horizon = "Plan for the next decade"
prompts = [short_horizon, long_horizon]

# Tokenize
tokens = model.to_tokens(prompts, prepend_bos=True)
print(f"Tokens shape: {tokens.shape}")  # [2, seq_len]

# Extract activations with cache
logits, cache = model.run_with_cache(tokens, remove_batch_dim=False)

# Extract from layer 10
act_name = get_act_name("resid_post", 10)
activations = cache[act_name][:, -1, :]  # Last token

print(f"Activations shape: {activations.shape}")  # [2, 768]

# Verify different activations for different horizons
cosine_sim = torch.nn.functional.cosine_similarity(
    activations[0:1], activations[1:2]
)
print(f"Cosine similarity: {cosine_sim.item():.3f}")
# Should be <1.0 if temporal info is encoded differently
```

## Hook-Based Extraction (Alternative)

For more control, use hooks directly:

```python
# Define hook function
stored_activations = {}

def capture_hook(activation, hook):
    stored_activations[hook.name] = activation.detach().cpu()

# Register hook
hook_name = "blocks.10.hook_resid_post"
model.add_hook(hook_name, capture_hook)

# Run forward pass
logits = model(tokens)

# Remove hook
model.reset_hooks()

# Access captured activation
acts = stored_activations[hook_name]
```

## Validation

### Test TransformerLens Installation

```bash
python scripts/validate_pipeline.py
```

This tests:
1. ✓ TransformerLens imports correctly
2. ✓ GPT-2 loads successfully
3. ✓ Tokenization works
4. ✓ Forward pass produces logits
5. ✓ Cache extraction works
6. ✓ Specific hook points accessible

### Verify Activation Shapes

```python
from src.models.model_loader import load_model, get_model_info

model = load_model("gpt2")
info = get_model_info(model)

print(f"Layers: {info['num_layers']}")      # 12
print(f"Hidden: {info['hidden_size']}")     # 768
print(f"Heads: {info['num_heads']}")        # 12
print(f"Params: {info['num_parameters']}")  # ~124M
```

## Common Issues

### Issue 1: Out of Memory

```python
# Solution: Use smaller batch size
activations = extractor.extract_batch(
    prompts,
    layers=[10, 11],  # Fewer layers
    batch_size=4      # Smaller batches
)
```

### Issue 2: Wrong Activation Shape

```python
# Check if batch dimension is preserved
_, cache = model.run_with_cache(tokens, remove_batch_dim=False)

# NOT: remove_batch_dim=True (removes batch, breaks our code)
```

### Issue 3: Hook Name Not Found

```python
from transformer_lens.utils import get_act_name

# CORRECT:
act_name = get_act_name("resid_post", 10)

# WRONG:
# act_name = "layer.10.output"  # Not TransformerLens format
```

## Advanced Usage

### Multi-Component Extraction

Extract multiple components simultaneously:

```python
components = ["resid_post", "attn_out", "mlp_out"]

for component in components:
    acts = extractor.extract_batch(
        prompts,
        layers=[10],
        component=component
    )
    print(f"{component}: {acts['layer_10'].shape}")
```

### Per-Head Activation

Extract attention head outputs separately:

```python
from src.models.activation_extractor import ActivationExtractor

extractor = ActivationExtractor(model)

head_acts = extractor.extract_head_activations(
    prompts,
    layers=[10],
    heads=[0, 1, 2, 3]  # Extract specific heads
)

# Returns: {"layer_10_head_0": array, "layer_10_head_1": array, ...}
```

## Integration with Latents

Both our TransformerLens approach and latents use the same underlying models:

```python
# Our approach: Extract activations for probing
from src.models.model_loader import load_model as load_tl_model
tl_model = load_tl_model("gpt2")  # HookedTransformer

# Latents approach: Load for steering
from transformers import GPT2LMHeadModel
hf_model = GPT2LMHeadModel.from_pretrained("gpt2")

# They access the same weights, different interfaces:
# - TransformerLens: Interpretability-focused (hooks, cache)
# - HuggingFace: Generation-focused (generate, logits)
```

## Best Practices

1. **Always use TransformerLens for extraction**: Don't manually hook HuggingFace models
2. **Verify shapes**: Check activation shapes match expectations
3. **Use cache for efficiency**: One forward pass captures all layers
4. **Validate hook names**: Use `get_act_name()` utility
5. **Test on small data first**: Validate pipeline before scaling up

## References

- TransformerLens Docs: https://neelnanda-io.github.io/TransformerLens/
- GitHub: https://github.com/neelnanda-io/TransformerLens
- Tutorial: https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens
- Neel Nanda's Blog: https://www.neelnanda.io/mechanistic-interpretability

## Troubleshooting

**Q: ModuleNotFoundError: No module named 'transformer_lens'**
```bash
pip install transformer-lens
```

**Q: Model loading is slow**
```python
# Models are downloaded from HuggingFace Hub
# First load downloads, subsequent loads are fast
# Cache location: ~/.cache/huggingface/
```

**Q: CUDA out of memory**
```python
# Use CPU or smaller batch size
model = load_model("gpt2", device="cpu")
extractor.extract_batch(prompts, batch_size=4)
```

**Q: How to verify TransformerLens is working?**
```bash
python scripts/validate_pipeline.py --quick
```
