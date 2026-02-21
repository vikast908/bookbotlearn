"""
model/attention.py - Multi-Head Self-Attention
================================================

This is the HEART of the Transformer. Attention is the mechanism that
allows each token to "look at" and gather information from other tokens.

INTUITION:
Imagine reading a sentence: "The cat sat on the mat because it was tired."
When processing "it", the model needs to figure out that "it" refers to "cat".
Attention lets the model learn to focus on "cat" when processing "it".

SCALED DOT-PRODUCT ATTENTION:
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    Q (Query): "What am I looking for?"
    K (Key):   "What do I contain?"
    V (Value): "What information do I provide?"

    The dot product Q @ K^T measures how well each query matches each key.
    Higher dot product = more attention = more of that value gets through.

    Scaling by 1/sqrt(d_k) prevents the dot products from growing too large,
    which would push softmax into regions where its gradients vanish.

MULTI-HEAD ATTENTION:
    Instead of one big attention, we split into multiple "heads", each
    attending to different aspects (syntax, semantics, position, etc.).

    Think of it like having multiple "experts" looking at the text from
    different angles, then combining their insights.

CAUSAL MASK (for autoregressive / decoder-only models):
    When predicting the next word, we must NOT let the model peek at
    future words. The causal mask blocks attention to future positions:

    Position:   0  1  2  3
    Token 0:    1  0  0  0  <- can only see itself
    Token 1:    1  1  0  0  <- can see token 0 and itself
    Token 2:    1  1  1  0  <- can see tokens 0, 1, and itself
    Token 3:    1  1  1  1  <- can see all previous tokens
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout: nn.Dropout | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.

    Args:
        Q: Queries, shape (batch, n_heads, seq_len, d_k)
        K: Keys,    shape (batch, n_heads, seq_len, d_k)
        V: Values,  shape (batch, n_heads, seq_len, d_k)
        mask: Causal mask, shape (1, 1, seq_len, seq_len)
        dropout: Optional dropout on attention weights

    Returns:
        output: Attended values, shape (batch, n_heads, seq_len, d_k)
        attn_weights: Attention probabilities, shape (batch, n_heads, seq_len, seq_len)
    """
    d_k = Q.size(-1)

    # Step 1: Compute raw attention scores
    # Q @ K^T: each query dot-producted with every key
    # Shape: (batch, n_heads, seq_len, seq_len)
    # scores[b][h][i][j] = "how much should position i attend to position j?"
    scores = torch.matmul(Q, K.transpose(-2, -1))

    # Step 2: Scale by sqrt(d_k)
    # WHY? Without scaling, when d_k is large, the dot products become large,
    # pushing softmax into saturated regions where gradients are tiny.
    # Example: if d_k=64, random vectors have dot products with std ≈ sqrt(64) = 8
    # Dividing by sqrt(d_k) brings the std back to ~1.
    scores = scores / math.sqrt(d_k)

    # Step 3: Apply causal mask
    # Set future positions to -infinity BEFORE softmax.
    # softmax(-inf) = 0, so future tokens get zero attention.
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # Step 4: Softmax normalizes scores to probabilities
    # After this, each row sums to 1.0
    # Shape: (batch, n_heads, seq_len, seq_len)
    attn_weights = F.softmax(scores, dim=-1)

    # Step 5: Dropout on attention weights (regularization)
    # Randomly drops some attention connections during training
    if dropout is not None:
        attn_weights = dropout(attn_weights)

    # Step 6: Weighted sum of values
    # Each position's output is a weighted combination of all value vectors,
    # where the weights are the attention probabilities.
    # Shape: (batch, n_heads, seq_len, d_k)
    output = torch.matmul(attn_weights, V)

    return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention.

    Splits the d_model-dimensional representation into n_heads parallel
    attention heads, each operating on d_k = d_model / n_heads dimensions.

    Parameters:
        W_Q: Projects input to queries  (d_model -> d_model)
        W_K: Projects input to keys     (d_model -> d_model)
        W_V: Projects input to values   (d_model -> d_model)
        W_O: Projects concatenated heads back (d_model -> d_model)

    Each W_Q/K/V is a single (d_model, d_model) matrix that implicitly
    contains all n_heads projections stacked together. We reshape after
    projection to split into heads.

    Total params per layer: 4 * (d_model * d_model + d_model) = 4 * 1056 = 4,224
    """

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()

        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head

        # Projection matrices for Q, K, V, and output
        # Each is (d_model, d_model) which contains all heads' projections
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # Create the causal mask as a BUFFER (not a parameter)
        # register_buffer ensures it:
        #   1. Moves to GPU with the model (model.to('cuda'))
        #   2. Gets saved/loaded with the model state dict
        #   3. Is NOT updated by the optimizer
        # torch.tril = lower triangular matrix = causal mask
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-head self-attention.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Step 1: Project input to Q, K, V
        # All three come from the same input x (that's why it's SELF-attention)
        # Shape: (batch, seq_len, d_model)
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        # Step 2: Split into multiple heads
        # Reshape from (batch, seq_len, d_model) to (batch, seq_len, n_heads, d_k)
        # Then transpose to (batch, n_heads, seq_len, d_k)
        # This lets us process all heads in parallel using batched matrix multiply
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Step 3: Apply attention with causal mask
        # Slice the mask to the actual sequence length
        # Shape: (1, 1, seq_len, seq_len) for broadcasting over batch and heads
        mask = self.causal_mask[:seq_len, :seq_len].unsqueeze(0).unsqueeze(0)

        attn_output, attn_weights = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout=self.dropout
        )
        # attn_output shape: (batch, n_heads, seq_len, d_k)

        # Step 4: Concatenate heads
        # Transpose back: (batch, seq_len, n_heads, d_k)
        # Then reshape to merge heads: (batch, seq_len, d_model)
        # contiguous() is needed because transpose changes memory layout
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )

        # Step 5: Final output projection
        # This lets the model learn how to combine information from different heads
        # Shape: (batch, seq_len, d_model)
        output = self.W_O(attn_output)

        return output
