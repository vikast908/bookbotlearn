"""
model/positional.py - Positional Encoding
============================================

WHY DO TRANSFORMERS NEED POSITIONAL ENCODING?

Self-attention is "permutation equivariant" -- if you shuffle the input
tokens, the output just gets shuffled the same way. The attention mechanism
treats "The cat sat on the mat" and "mat the on sat cat The" identically!

But word ORDER matters enormously in language. "Dog bites man" and
"Man bites dog" have very different meanings.

SOLUTION: Add positional information to each token's embedding.
Token embedding tells the model WHAT the word is.
Positional encoding tells the model WHERE the word is.

final_embedding = token_embedding + positional_encoding

TWO APPROACHES:

1. LEARNED positional embeddings (used here, as in GPT-2):
   - Each position gets its own learnable d_model-dimensional vector
   - Simple: just another nn.Embedding layer
   - Works great when max_seq_len is known and modest
   - Cannot extrapolate to longer sequences than seen during training

2. SINUSOIDAL positional encoding (original Transformer paper):
   - Fixed, non-learnable encoding using sin/cos at different frequencies
   - PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
   - PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
   - Can theoretically generalize to unseen sequence lengths
   - Included below as a commented reference implementation
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Learned Positional Embeddings (GPT-2 style).

    Each position index (0, 1, 2, ..., max_seq_len-1) has its own
    learnable embedding vector. During training, these vectors are
    optimized alongside all other model parameters.

    Input:  (batch, seq_len, d_model) -- token embeddings
    Output: (batch, seq_len, d_model) -- token + positional embeddings
    """

    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()

        # Create an embedding table: one vector per position
        # Shape: (max_seq_len, d_model)
        # This is randomly initialized and learned during training
        self.embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional embeddings to token embeddings.

        Args:
            x: Token embeddings of shape (batch, seq_len, d_model)

        Returns:
            Token embeddings + positional embeddings (same shape)
        """
        seq_len = x.size(1)

        # Create position indices: [0, 1, 2, ..., seq_len-1]
        # .device ensures this tensor is on the same device (CPU/GPU) as x
        positions = torch.arange(seq_len, device=x.device)

        # Look up positional embeddings and add to token embeddings
        # self.embedding(positions) shape: (seq_len, d_model)
        # This broadcasts over the batch dimension
        return x + self.embedding(positions)


# ──────────────────────────────────────────────────────────────────────
# REFERENCE: Sinusoidal Positional Encoding (from "Attention Is All You Need")
# ──────────────────────────────────────────────────────────────────────
#
# class SinusoidalPositionalEncoding(nn.Module):
#     """
#     Fixed (non-learnable) positional encoding using sine and cosine functions.
#
#     The key idea: different dimensions use different frequencies.
#     Low-frequency dimensions capture coarse position (beginning vs end).
#     High-frequency dimensions capture fine position (adjacent tokens).
#
#     Formula:
#         PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
#         PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
#
#     Why sin/cos? Because PE(pos+k) can be expressed as a linear
#     function of PE(pos), which makes it easy for the model to
#     learn to attend to relative positions.
#     """
#
#     def __init__(self, max_seq_len: int, d_model: int):
#         super().__init__()
#
#         pe = torch.zeros(max_seq_len, d_model)
#         position = torch.arange(0, max_seq_len).unsqueeze(1).float()
#
#         # Compute the frequency for each dimension pair
#         # div_term = 1 / 10000^(2i/d_model)
#         # We compute in log-space for numerical stability:
#         # exp(-2i * log(10000) / d_model)
#         div_term = torch.exp(
#             torch.arange(0, d_model, 2).float()
#             * (-math.log(10000.0) / d_model)
#         )
#
#         pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions
#         pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions
#
#         # Register as buffer (not a parameter, but moves with model to GPU)
#         self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_seq_len, d_model)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return x + self.pe[:, :x.size(1), :]
