"""
model/layernorm.py - Layer Normalization (From Scratch)
========================================================

Layer Normalization (Ba et al., 2016) is a critical component that
stabilizes transformer training. Without it, deep networks suffer
from "internal covariate shift" -- the distribution of activations
changes as weights update, making learning unstable.

THE FORMULA:
    y = gamma * (x - mean) / sqrt(variance + eps) + beta

    where mean and variance are computed across the FEATURE dimension
    (d_model) for each sample independently.

WHY LAYER NORM (not Batch Norm)?
- BatchNorm normalizes across the BATCH dimension. This means:
  1. It depends on batch size (bad for small batches or inference)
  2. It needs running statistics (complex for variable-length sequences)
- LayerNorm normalizes across the FEATURE dimension. This means:
  1. Each sample is normalized independently (no batch dependency)
  2. Works identically at train and eval time
  3. No running statistics to maintain

PARAMETERS (learnable):
    gamma (scale): initialized to 1, shape (d_model,)
    beta (shift): initialized to 0, shape (d_model,)

    After normalization, the network can learn to undo it if needed.
    gamma and beta give it the flexibility to represent any mean/variance
    the model finds useful.

NOTE: We implement this from scratch instead of using nn.LayerNorm
to understand exactly what happens inside.
"""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Layer Normalization applied over the last dimension (d_model).

    Input shape:  (batch, seq_len, d_model)
    Output shape: (batch, seq_len, d_model)

    Each position in each sequence is normalized independently
    across its d_model features.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        Args:
            d_model: Feature dimension to normalize over
            eps: Small constant for numerical stability (prevents division by zero)
        """
        super().__init__()

        # Learnable scale parameter (initialized to 1 = "no scaling")
        # nn.Parameter tells PyTorch to include this in model.parameters()
        # and update it during backpropagation
        self.gamma = nn.Parameter(torch.ones(d_model))

        # Learnable shift parameter (initialized to 0 = "no shift")
        self.beta = nn.Parameter(torch.zeros(d_model))

        # Epsilon: a tiny value added to variance before taking sqrt.
        # Without this, if all features have the same value (variance=0),
        # we'd divide by zero. eps = 1e-6 is standard.
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize x across the last dimension.

        Step-by-step for a single vector [x1, x2, ..., x_d]:
        1. mean = (x1 + x2 + ... + x_d) / d
        2. var = ((x1-mean)^2 + (x2-mean)^2 + ... + (x_d-mean)^2) / d
        3. x_norm_i = (x_i - mean) / sqrt(var + eps)
        4. y_i = gamma_i * x_norm_i + beta_i

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Normalized tensor of the same shape
        """
        # Compute mean across the last dimension (d_model)
        # keepdim=True preserves the dimension for broadcasting
        # Shape: (batch, seq_len, 1)
        mean = x.mean(dim=-1, keepdim=True)

        # Compute variance across the last dimension
        # unbiased=False uses population variance (divide by N, not N-1)
        # This matches the original Layer Norm paper
        # Shape: (batch, seq_len, 1)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # Normalize: zero mean, unit variance
        # Shape: (batch, seq_len, d_model)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Scale and shift with learnable parameters
        # gamma and beta have shape (d_model,) and broadcast over batch and seq_len
        return self.gamma * x_norm + self.beta
