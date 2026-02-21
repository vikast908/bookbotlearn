"""
config.py - Central Configuration for MiniGPT
==============================================

All hyperparameters live here in a single dataclass.
Every other module imports from this file, ensuring consistency.

Architecture target: ~95,568 parameters (approximately 100K).
"""

from dataclasses import dataclass
import os


@dataclass
class TransformerConfig:
    # ──────────────────────────────────────────────────────────────
    # MODEL ARCHITECTURE
    # ──────────────────────────────────────────────────────────────

    # Vocabulary size: top-2000 most frequent words from the corpus.
    # Covers ~95% of all token occurrences in Tiny Shakespeare.
    # Words outside this set become <unk>.
    # This is the BIGGEST parameter cost: vocab_size * d_model = 64,000 params.
    vocab_size: int = 2000

    # Embedding dimension: each word is represented as a 32-dim vector.
    # Small, but enough for our tiny model to learn basic patterns.
    # In GPT-2, this is 768; in GPT-3, 12288. We use 32.
    d_model: int = 32

    # Number of attention heads: the d_model dimension is split into
    # n_heads parallel attention mechanisms, each of size d_k = d_model / n_heads.
    # Here: d_k = 32 / 4 = 8 dimensions per head.
    n_heads: int = 4

    # Number of transformer layers (blocks) stacked sequentially.
    # Each block = multi-head attention + feed-forward network.
    # GPT-2 uses 12-48 layers. We use 2 to stay within ~100K params.
    n_layers: int = 2

    # Feed-forward network inner dimension.
    # The FFN expands from d_model -> d_ff -> d_model.
    # Standard ratio is 4x (d_ff = 4 * d_model = 128), but for our
    # parameter budget we use exactly 128 = 4 * 32.
    d_ff: int = 128

    # Maximum sequence length (context window).
    # The model can "see" up to 128 tokens of history when predicting
    # the next word. GPT-2 uses 1024, GPT-3 uses 2048.
    max_seq_len: int = 128

    # Dropout rate: randomly zeroes 10% of activations during training.
    # This prevents overfitting by forcing the network to not rely
    # on any single feature too heavily.
    dropout: float = 0.1

    # ──────────────────────────────────────────────────────────────
    # TRAINING
    # ──────────────────────────────────────────────────────────────

    batch_size: int = 64           # Samples per gradient update
    learning_rate: float = 3e-4    # Peak learning rate (after warmup)
    min_lr: float = 3e-5           # Minimum LR at end of cosine decay
    max_epochs: int = 20           # Maximum training epochs
    warmup_steps: int = 200        # Linear warmup steps (stabilizes early training)
    max_steps: int = 5000          # Total training steps
    grad_clip: float = 1.0         # Max gradient norm (prevents exploding gradients)
    weight_decay: float = 0.01     # L2 regularization on weight matrices only
    eval_interval: int = 500       # Evaluate on validation set every N steps
    log_interval: int = 100        # Print training stats every N steps

    # ──────────────────────────────────────────────────────────────
    # GENERATION
    # ──────────────────────────────────────────────────────────────

    temperature: float = 0.8       # <1 = more confident, >1 = more random
    top_k: int = 40                # Only sample from top-k most likely tokens
    max_gen_len: int = 100         # Maximum tokens to generate

    # ──────────────────────────────────────────────────────────────
    # PATHS
    # ──────────────────────────────────────────────────────────────

    data_dir: str = "data"
    data_file: str = "input.txt"
    vocab_file: str = "vocab.json"
    checkpoint_dir: str = "checkpoints"
    checkpoint_file: str = "model.pt"

    @property
    def data_path(self) -> str:
        return os.path.join(self.data_dir, self.data_file)

    @property
    def vocab_path(self) -> str:
        return os.path.join(self.data_dir, self.vocab_file)

    @property
    def checkpoint_path(self) -> str:
        return os.path.join(self.checkpoint_dir, self.checkpoint_file)

    @property
    def d_k(self) -> int:
        """Dimension per attention head."""
        return self.d_model // self.n_heads
