"""Attribution methods for circuit analysis."""

import logging
import torch
import numpy as np

logger = logging.getLogger(__name__)


def compute_head_importance(results: dict, top_k: int = 10) -> list:
    """Compute importance scores for attention heads."""
    scores = {}
    for key, val in results.items():
        if "effect_size" in val:
            scores[key] = val["effect_size"]
    
    sorted_heads = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_heads[:top_k]


def integrated_gradients(model, probe, x, baseline=None, steps=50):
    """Compute integrated gradients attribution."""
    if baseline is None:
        baseline = torch.zeros_like(x)
    
    # Interpolate between baseline and input
    alphas = torch.linspace(0, 1, steps)
    attributions = torch.zeros_like(x)
    
    for alpha in alphas:
        x_interp = baseline + alpha * (x - baseline)
        x_interp.requires_grad = True
        
        output = probe(x_interp)
        output.backward(torch.ones_like(output))
        
        attributions += x_interp.grad
    
    attributions /= steps
    attributions *= (x - baseline)
    
    return attributions.detach()
