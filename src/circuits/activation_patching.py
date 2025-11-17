"""Activation patching for circuit analysis."""

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
from transformer_lens import HookedTransformer

logger = logging.getLogger(__name__)


class ActivationPatcher:
    """Perform activation patching experiments."""

    def __init__(self, model: HookedTransformer):
        """Initialize patcher."""
        self.model = model
        self.device = next(model.parameters()).device

    def patch_heads(
        self,
        source_prompt: str,
        target_prompt: str,
        layers: List[int],
        heads: List[int],
    ) -> Dict:
        """Patch attention heads from source to target."""
        # Tokenize both prompts
        source_tokens = self.model.to_tokens(source_prompt)
        target_tokens = self.model.to_tokens(target_prompt)

        # Run clean forward passes
        _, source_cache = self.model.run_with_cache(source_tokens)
        _, target_cache = self.model.run_with_cache(target_tokens)

        results = {}

        for layer in layers:
            for head in heads:
                # Create patching hook
                def patch_hook(activation, hook):
                    # Replace with source activation
                    activation[:, :, head, :] = source_cache[hook.name][:, :, head, :]
                    return activation

                hook_name = f"blocks.{layer}.attn.hook_result"

                # Run with patching
                with torch.no_grad():
                    patched_output = self.model.run_with_hooks(
                        target_tokens,
                        fwd_hooks=[(hook_name, patch_hook)],
                    )

                results[f"layer_{layer}_head_{head}"] = {
                    "effect_size": torch.norm(patched_output - target_cache["logits"]).item()
                }

        return results

    def patch_layer_residual(
        self, source_prompt: str, target_prompt: str, layer: int
    ) -> float:
        """Patch entire residual stream at a layer."""
        source_tokens = self.model.to_tokens(source_prompt)
        target_tokens = self.model.to_tokens(target_prompt)

        _, source_cache = self.model.run_with_cache(source_tokens)
        _, target_cache = self.model.run_with_cache(target_tokens)

        hook_name = f"blocks.{layer}.hook_resid_post"

        def patch_hook(activation, hook):
            return source_cache[hook.name]

        with torch.no_grad():
            patched_output = self.model.run_with_hooks(
                target_tokens,
                fwd_hooks=[(hook_name, patch_hook)],
            )

        effect = torch.norm(patched_output - target_cache["logits"]).item()
        return effect
