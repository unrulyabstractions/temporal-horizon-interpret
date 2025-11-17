"""Probe architectures for temporal horizon detection.

This module implements linear and MLP probes for classifying temporal
horizons from model activations.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LinearProbe(nn.Module):
    """Linear probe for binary classification.

    Simple logistic regression probe that learns a linear decision boundary
    in the activation space.

    Attributes:
        hidden_size: Input dimension (model hidden size)
        dropout_rate: Dropout rate for regularization
    """

    def __init__(self, hidden_size: int, dropout_rate: float = 0.0):
        """Initialize linear probe.

        Args:
            hidden_size: Input dimension
            dropout_rate: Dropout rate (0.0 = no dropout)

        Example:
            >>> probe = LinearProbe(hidden_size=768)
            >>> x = torch.randn(10, 768)
            >>> logits = probe(x)
            >>> logits.shape
            torch.Size([10, 2])
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.linear = nn.Linear(hidden_size, 2)  # Binary classification

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input activations [batch_size, hidden_size]

        Returns:
            Logits [batch_size, 2].
        """
        x = self.dropout(x)
        return self.linear(x)


class MLPProbe(nn.Module):
    """Multi-layer perceptron probe.

    Non-linear probe with 1-2 hidden layers for more complex decision boundaries.

    Attributes:
        hidden_size: Input dimension
        num_layers: Number of hidden layers (1 or 2)
        hidden_dim: Hidden layer dimension
        dropout_rate: Dropout rate
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 1,
        hidden_dim: Optional[int] = None,
        dropout_rate: float = 0.1,
        activation: str = "relu",
    ):
        """Initialize MLP probe.

        Args:
            hidden_size: Input dimension
            num_layers: Number of hidden layers (1 or 2)
            hidden_dim: Hidden layer dimension (default: hidden_size // 2)
            dropout_rate: Dropout rate
            activation: Activation function ("relu", "gelu", "tanh")

        Example:
            >>> probe = MLPProbe(hidden_size=768, num_layers=2)
            >>> x = torch.randn(10, 768)
            >>> logits = probe(x)
            >>> logits.shape
            torch.Size([10, 2])
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim or hidden_size // 2

        # Select activation
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build layers
        layers = []

        # First hidden layer
        layers.extend([
            nn.Linear(hidden_size, self.hidden_dim),
            self.activation,
            nn.Dropout(dropout_rate),
        ])

        # Additional hidden layers
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(self.hidden_dim, self.hidden_dim),
                self.activation,
                nn.Dropout(dropout_rate),
            ])

        # Output layer
        layers.append(nn.Linear(self.hidden_dim, 2))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input activations [batch_size, hidden_size]

        Returns:
            Logits [batch_size, 2].
        """
        return self.network(x)


class AttentionProbe(nn.Module):
    """Attention-based probe.

    Uses self-attention to aggregate information before classification.
    Useful when activations from multiple positions are available.

    Attributes:
        hidden_size: Input dimension
        num_heads: Number of attention heads
        dropout_rate: Dropout rate
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
    ):
        """Initialize attention probe.

        Args:
            hidden_size: Input dimension
            num_heads: Number of attention heads
            dropout_rate: Dropout rate

        Example:
            >>> probe = AttentionProbe(hidden_size=768)
            >>> x = torch.randn(10, 20, 768)  # [batch, seq, hidden]
            >>> logits = probe(x)
            >>> logits.shape
            torch.Size([10, 2])
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.attention = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input activations [batch_size, seq_len, hidden_size]
               or [batch_size, hidden_size] (will unsqueeze)

        Returns:
            Logits [batch_size, 2].
        """
        # Handle single position input
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Self-attention
        attn_out, _ = self.attention(x, x, x)

        # Residual connection and layer norm
        x = self.layer_norm(x + attn_out)

        # Pool (mean over sequence)
        x = x.mean(dim=1)

        # Classify
        return self.classifier(x)


def create_probe(
    probe_type: str,
    hidden_size: int,
    **kwargs,
) -> nn.Module:
    """Factory function to create probes.

    Args:
        probe_type: Type of probe ("linear", "mlp", "attention")
        hidden_size: Input dimension
        **kwargs: Additional arguments for probe

    Returns:
        Probe module.

    Raises:
        ValueError: If probe_type is unknown

    Example:
        >>> probe = create_probe("linear", hidden_size=768)
        >>> isinstance(probe, LinearProbe)
        True
    """
    if probe_type == "linear":
        return LinearProbe(hidden_size, **kwargs)
    elif probe_type == "mlp":
        return MLPProbe(hidden_size, **kwargs)
    elif probe_type == "attention":
        return AttentionProbe(hidden_size, **kwargs)
    else:
        raise ValueError(
            f"Unknown probe type: {probe_type}. "
            "Valid types: ['linear', 'mlp', 'attention']"
        )


def count_probe_parameters(probe: nn.Module) -> dict:
    """Count parameters in probe.

    Args:
        probe: Probe module

    Returns:
        Dictionary with parameter counts.

    Example:
        >>> probe = LinearProbe(768)
        >>> counts = count_probe_parameters(probe)
        >>> counts["total"] > 0
        True
    """
    total = sum(p.numel() for p in probe.parameters())
    trainable = sum(p.numel() for p in probe.parameters() if p.requires_grad)

    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }
