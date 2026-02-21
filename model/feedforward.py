"""
model/feedforward.py - Position-Wise Feed-Forward Network
==========================================================

After attention gathers information from other positions, the FFN
processes each position INDEPENDENTLY. Think of it as:

    Attention = "communication" between positions (tokens talk to each other)
    FFN       = "computation" within each position (each token thinks independently)

ARCHITECTURE:
    FFN(x) = Linear2(GELU(Linear1(x)))

    Linear1: d_model -> d_ff   (expand)
    GELU:    non-linear activation
    Linear2: d_ff -> d_model   (compress back)

This creates a "bottleneck": information is expanded into a higher-dimensional
space (d_ff), processed non-linearly, then compressed back. The expansion
lets the network represent more complex functions.

WHY GELU (not ReLU)?

ReLU(x) = max(0, x)
    - Simple but harsh: kills all negative values completely
    - "Dead neurons" problem: if a neuron always outputs negative, it never updates

GELU(x) = x * Phi(x)   where Phi = CDF of standard normal
    - Smooth approximation of ReLU
    - Small negative values get slightly negative outputs (not killed)
    - Used in GPT-2, BERT, and most modern transformers
    - Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
"""

import math
import torch
import torch.nn as nn


def gelu(x: torch.Tensor) -> torch.Tensor:
    """
    Gaussian Error Linear Unit (GELU) activation function.

    GELU(x) = x * Phi(x)

    where Phi(x) is the cumulative distribution function of the
    standard normal distribution.

    We use the tanh approximation (same as GPT-2):
        GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    This is mathematically equivalent to:
        x * 0.5 * (1 + erf(x / sqrt(2)))
    but the tanh version is faster to compute.
    """
    return 0.5 * x * (1.0 + torch.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
    ))


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.

    Applied to each position independently and identically.
    "Position-wise" means the same weights are shared across all
    positions in the sequence, but each position is processed separately.

    Shape flow:
        Input:  (batch, seq_len, d_model)   e.g., (64, 128, 32)
        After Linear1: (batch, seq_len, d_ff)    e.g., (64, 128, 128)
        After GELU: same
        After Linear2: (batch, seq_len, d_model)  e.g., (64, 128, 32)

    Parameters per layer:
        Linear1: d_model * d_ff + d_ff     = 32 * 128 + 128 = 4,224
        Linear2: d_ff * d_model + d_model  = 128 * 32 + 32  = 4,128
        Total:                                                = 8,352
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # Expansion: d_model -> d_ff
        self.linear1 = nn.Linear(d_model, d_ff)

        # Compression: d_ff -> d_model
        self.linear2 = nn.Linear(d_ff, d_model)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        FFN(x) = Dropout(Linear2(GELU(Dropout(Linear1(x)))))

        Args:
            x: Input of shape (batch, seq_len, d_model)
        Returns:
            Output of shape (batch, seq_len, d_model)
        """
        # Expand to higher dimension
        x = self.linear1(x)

        # Non-linear activation (GELU)
        x = gelu(x)

        # Dropout after activation
        x = self.dropout(x)

        # Compress back to d_model
        x = self.linear2(x)

        # Dropout after second linear
        x = self.dropout(x)

        return x
