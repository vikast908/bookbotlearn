"""
model/transformer.py - MiniGPT: The Full Decoder-Only Transformer
===================================================================

This assembles ALL components into a complete GPT-style language model.

ARCHITECTURE OVERVIEW:

    Input token IDs: [45, 12, 7, 89, 3]
          |
    Token Embedding:  word index -> d_model vector    (nn.Embedding)
          |
    + Positional Encoding: position -> d_model vector  (nn.Embedding)
          |
    Embedding Dropout
          |
    Transformer Block 1  (attention + FFN)
          |
    Transformer Block 2  (attention + FFN)
          |
    Final Layer Norm
          |
    Output Head: d_model -> vocab_size logits           (nn.Linear, weight-tied)
          |
    Predicted next-word probabilities: [0.01, 0.05, 0.3, ...]

WEIGHT TYING (Press & Wolf, 2017):
    The output projection (d_model -> vocab_size) shares its weight matrix
    with the token embedding (vocab_size -> d_model).

    Why? The embedding maps words to vectors, and the output maps vectors
    back to words. They should be approximate inverses of each other.
    Tying them:
    1. Halves the embedding parameters (saves 64,000 params!)
    2. Acts as regularization (shared representation)
    3. Often improves performance, especially for small models

DECODER-ONLY (GPT-STYLE):
    Unlike the original Transformer (which has encoder + decoder), GPT uses
    only the decoder with causal (autoregressive) masking. Each token can
    only attend to tokens that came BEFORE it (not future tokens).
    This makes it a language model: predict the next word given previous words.

PARAMETER COUNT (~95,568):
    Token Embedding (tied):     64,000   (66.9%)
    Positional Embedding:        4,096    (4.3%)
    Transformer Blocks (x2):   25,408   (26.6%)
    Final LayerNorm:                64    (0.1%)
    Output Bias:                 2,000    (2.1%)
    ───────────────────────────────────────────
    TOTAL:                      95,568   (~96K)
"""

import torch
import torch.nn as nn

from config import TransformerConfig
from model.layernorm import LayerNorm
from model.positional import PositionalEncoding
from model.transformer_block import TransformerBlock


class MiniGPT(nn.Module):
    """
    A minimal GPT-style decoder-only transformer language model.

    This ~96K parameter model implements the same architecture as GPT-2/3,
    just much smaller. Every component is the same -- attention, FFN,
    layer norm, residual connections -- just with tiny dimensions.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # ── Token Embedding ──
        # Maps each word index to a d_model-dimensional vector
        # Shape: (vocab_size, d_model) = (2000, 32)
        # This is the LARGEST parameter matrix in our model
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # ── Positional Encoding ──
        # Adds position information so the model knows word order
        # Shape: (max_seq_len, d_model) = (128, 32)
        self.position_encoding = PositionalEncoding(config.max_seq_len, config.d_model)

        # ── Embedding Dropout ──
        # Applied after combining token + positional embeddings
        self.embed_dropout = nn.Dropout(config.dropout)

        # ── Transformer Blocks ──
        # Stack of N identical blocks, each with attention + FFN
        # nn.ModuleList ensures PyTorch tracks all blocks as sub-modules
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                max_seq_len=config.max_seq_len,
                dropout=config.dropout,
            )
            for _ in range(config.n_layers)
        ])

        # ── Final Layer Norm ──
        # Applied after the last transformer block, before output projection
        # This is specific to Pre-Norm architecture
        self.final_norm = LayerNorm(config.d_model)

        # ── Output Head ──
        # Projects from d_model back to vocab_size to get logits
        # (unnormalized log-probabilities) for each word
        self.output_head = nn.Linear(config.d_model, config.vocab_size)

        # ── Weight Tying ──
        # Share weights between token embedding and output projection
        # The embedding matrix (vocab_size, d_model) is TRANSPOSED when used
        # as the output projection, mapping d_model -> vocab_size
        self.output_head.weight = self.token_embedding.weight

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize model parameters.

        Following GPT-2:
        - Embeddings: Normal distribution with std=0.02
        - Linear layers: Xavier uniform (good for layers with varying sizes)
        - Biases: Zero
        - LayerNorm gamma: 1.0 (already default)
        - LayerNorm beta: 0.0 (already default)
        """
        for name, param in self.named_parameters():
            if "embedding" in name and "weight" in name:
                nn.init.normal_(param, mean=0.0, std=0.02)
            elif "weight" in name and param.dim() >= 2:
                # Xavier uniform for weight matrices
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: token IDs -> next-word logits.

        Args:
            x: Token IDs of shape (batch, seq_len)
               Each value is an integer in [0, vocab_size)

        Returns:
            logits: Unnormalized predictions of shape (batch, seq_len, vocab_size)
                    logits[b][t][w] = how likely word w is to follow position t
        """
        # Step 1: Token Embedding
        # (batch, seq_len) -> (batch, seq_len, d_model)
        # Each integer token ID is replaced by its d_model-dimensional vector
        x = self.token_embedding(x)

        # Step 2: Add Positional Encoding
        # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        # Position vectors are ADDED to token vectors
        x = self.position_encoding(x)

        # Step 3: Embedding Dropout
        x = self.embed_dropout(x)

        # Step 4: Pass through each transformer block
        # Each block refines the representations through attention and FFN
        for block in self.blocks:
            x = block(x)

        # Step 5: Final Layer Norm
        # Normalize before the output projection
        x = self.final_norm(x)

        # Step 6: Output Projection
        # (batch, seq_len, d_model) -> (batch, seq_len, vocab_size)
        # Produces a score (logit) for every word in the vocabulary
        # Higher logit = model thinks that word is more likely to come next
        logits = self.output_head(x)

        return logits

    def count_parameters(self) -> int:
        """
        Count and display parameters by component.

        Returns:
            Total number of trainable parameters
        """
        print("\n" + "=" * 55)
        print("  MiniGPT Parameter Count")
        print("=" * 55)

        # Count by component
        components = {}

        # Token embedding (shared with output head via weight tying)
        embed_params = self.token_embedding.weight.numel()
        components["Token Embedding (tied)"] = embed_params

        # Positional encoding
        pos_params = sum(
            p.numel() for p in self.position_encoding.parameters()
        )
        components["Positional Encoding"] = pos_params

        # Transformer blocks
        block_params = sum(
            p.numel() for p in self.blocks.parameters()
        )
        components[f"Transformer Blocks (x{self.config.n_layers})"] = block_params

        # Final layer norm
        norm_params = sum(
            p.numel() for p in self.final_norm.parameters()
        )
        components["Final LayerNorm"] = norm_params

        # Output head bias only (weight is tied with embedding)
        if self.output_head.bias is not None:
            components["Output Bias"] = self.output_head.bias.numel()

        # Print breakdown
        total = sum(components.values())
        for name, count in components.items():
            pct = count / total * 100
            print(f"  {name:<35} {count:>8,}   ({pct:>5.1f}%)")

        print("-" * 55)
        print(f"  {'TOTAL':<35} {total:>8,}   (100.0%)")
        print("=" * 55)

        # Also print total trainable (should match)
        trainable = sum(p.numel() for p in self.parameters())
        print(f"  Trainable parameters: {trainable:,}")
        print(f"  Model size: ~{trainable * 4 / 1024:.1f} KB (float32)")
        print()

        return trainable
