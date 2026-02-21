"""
model/transformer_block.py - Single Transformer Decoder Block
===============================================================

A transformer is made of STACKED BLOCKS, each containing:
    1. Multi-Head Self-Attention (communication)
    2. Feed-Forward Network (computation)

Each sub-layer has:
    - Layer Normalization (stabilizes training)
    - Residual Connection (helps gradients flow)

PRE-NORM vs POST-NORM:

    Post-Norm (original "Attention Is All You Need", 2017):
        x = LayerNorm(x + Attention(x))
        x = LayerNorm(x + FFN(x))

    Pre-Norm (GPT-2 and most modern models):
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))

    We use PRE-NORM because:
    1. More stable training (especially without careful LR warmup)
    2. The residual path is "clean" -- just addition, no normalization
    3. Gradients flow directly through the residual stream
    4. Widely adopted in GPT-2, GPT-3, LLaMA, etc.

RESIDUAL CONNECTIONS:

    The "+ x" in "x = x + Attention(LayerNorm(x))" is a residual connection.

    Why? In deep networks, gradients must flow backward through many layers.
    Without residuals, gradients can vanish (become tiny) or explode (become huge).

    The residual connection creates a "highway" for gradients:
        Forward:  output = x + f(x)
        Gradient: d(output)/d(x) = 1 + d(f(x))/d(x)

    That "1 +" means the gradient is ALWAYS at least 1, preventing vanishing.

    Analogy: Residual connections let the network learn "corrections" to the
    input rather than learning the full transformation from scratch.
"""

import torch
import torch.nn as nn

from model.layernorm import LayerNorm
from model.attention import MultiHeadAttention
from model.feedforward import FeedForward


class TransformerBlock(nn.Module):
    """
    A single transformer decoder block (Pre-Norm style).

    Data flow:
        input
          |
          +---> LayerNorm ---> MultiHeadAttention ---> Dropout ---+
          |                                                       |
          +<----- residual connection (addition) <-----------------+
          |
          +---> LayerNorm ---> FeedForward ---> Dropout ---+
          |                                                |
          +<----- residual connection (addition) <----------+
          |
        output

    Parameters per block:
        Attention:    4,224   (4 linear projections)
        FFN:          8,352   (2 linear layers)
        LayerNorm x2:    128  (gamma + beta each)
        Total:       12,704
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Pre-attention layer norm
        self.ln1 = LayerNorm(d_model)

        # Multi-head self-attention
        self.attention = MultiHeadAttention(d_model, n_heads, max_seq_len, dropout)

        # Pre-FFN layer norm
        self.ln2 = LayerNorm(d_model)

        # Feed-forward network
        self.ffn = FeedForward(d_model, d_ff, dropout)

        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply one transformer block.

        Pre-Norm architecture:
            x = x + Dropout(Attention(LayerNorm(x)))
            x = x + Dropout(FFN(LayerNorm(x)))

        Args:
            x: Input of shape (batch, seq_len, d_model)
        Returns:
            Output of shape (batch, seq_len, d_model)
        """
        # ── Self-Attention Sub-Layer ──
        # Save input for residual connection
        residual = x

        # Pre-norm: normalize BEFORE attention
        x = self.ln1(x)

        # Apply multi-head self-attention
        x = self.attention(x)

        # Dropout on attention output
        x = self.dropout(x)

        # Residual connection: add back the original input
        # This means the attention only needs to learn the DELTA (change)
        x = residual + x

        # ── Feed-Forward Sub-Layer ──
        # Save input for residual connection
        residual = x

        # Pre-norm: normalize BEFORE FFN
        x = self.ln2(x)

        # Apply feed-forward network
        x = self.ffn(x)

        # Residual connection
        x = residual + x

        return x
