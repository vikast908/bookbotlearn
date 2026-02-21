"""
data/dataset.py - PyTorch Dataset for Language Modeling
========================================================

This creates training examples using a SLIDING WINDOW approach:

Given a sequence of token IDs:
    [45, 12, 7, 89, 3, 56, 102, 23, ...]

We create (input, target) pairs by sliding a window of size seq_len:

    Window 0:  input = [45, 12, 7, 89]    target = [12, 7, 89, 3]
    Window 1:  input = [12, 7, 89, 3]     target = [7, 89, 3, 56]
    Window 2:  input = [7, 89, 3, 56]     target = [89, 3, 56, 102]
    ...

Notice: the target is the input SHIFTED RIGHT BY ONE POSITION.
This is how language models learn: predict the next word given the previous words.

The model sees tokens[0:4] and must predict tokens[1:5].
At each position i, the model predicts what comes after position i.

TRAIN/VAL SPLIT:
We split by position (not random) to keep contiguous text together.
First 90% of the corpus = training, last 10% = validation.
This is important because random splitting would leak future context.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

from data.tokenizer import WordTokenizer


class ShakespeareDataset(Dataset):
    """
    A PyTorch Dataset that yields (input, target) pairs for language modeling.

    Each sample is a window of seq_len consecutive token IDs (input)
    paired with the same window shifted by one position (target).
    """

    def __init__(self, token_ids: list[int], seq_len: int):
        """
        Args:
            token_ids: List of integer token IDs (the entire corpus, encoded)
            seq_len: Length of each training sequence (context window)
        """
        # Store as a single contiguous tensor for fast slicing
        # LongTensor because token IDs are integers used as indices
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self) -> int:
        # Number of valid sliding windows
        # We need seq_len tokens for input + 1 more for the last target
        return len(self.data) - self.seq_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x: input tokens  [idx : idx + seq_len]
            y: target tokens [idx+1 : idx + seq_len + 1]

        The target y is x shifted right by 1. At each position i:
            "Given x[0], x[1], ..., x[i], predict y[i] = x[i+1]"
        """
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y


def create_datasets(
    text: str,
    tokenizer: WordTokenizer,
    seq_len: int,
    train_split: float = 0.9,
) -> Tuple[ShakespeareDataset, ShakespeareDataset]:
    """
    Encode text and split into train/validation datasets.

    Args:
        text: Raw text corpus
        tokenizer: Trained WordTokenizer with vocabulary built
        seq_len: Sequence length for sliding windows
        train_split: Fraction of data for training (default 90%)

    Returns:
        (train_dataset, val_dataset)
    """
    # Encode the entire corpus into token IDs
    token_ids = tokenizer.encode(text)
    print(f"[dataset] Encoded corpus: {len(token_ids):,} tokens")

    # Split by position: first 90% for train, last 10% for validation
    # We do NOT shuffle before splitting because we want contiguous text
    split_idx = int(len(token_ids) * train_split)

    train_ids = token_ids[:split_idx]
    val_ids = token_ids[split_idx:]

    print(f"[dataset] Train tokens: {len(train_ids):,}")
    print(f"[dataset] Val tokens:   {len(val_ids):,}")

    train_dataset = ShakespeareDataset(train_ids, seq_len)
    val_dataset = ShakespeareDataset(val_ids, seq_len)

    print(f"[dataset] Train samples: {len(train_dataset):,}")
    print(f"[dataset] Val samples:   {len(val_dataset):,}")

    return train_dataset, val_dataset


def create_dataloaders(
    train_dataset: ShakespeareDataset,
    val_dataset: ShakespeareDataset,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    """
    Wrap datasets in DataLoaders for batched iteration.

    DataLoader handles:
    - Batching: groups samples into batches of size batch_size
    - Shuffling: randomizes order each epoch (train only)
    - Collation: stacks individual tensors into batch tensors

    Training data is shuffled to prevent the model from memorizing
    the order of the text. Validation data is NOT shuffled because
    we just need a consistent loss estimate.
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,      # Randomize training order each epoch
        drop_last=True,     # Drop incomplete last batch for consistent batch size
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,      # No need to shuffle validation
        drop_last=False,    # Use all validation data
    )

    print(f"[dataset] Train batches per epoch: {len(train_loader):,}")
    print(f"[dataset] Val batches: {len(val_loader):,}")

    return train_loader, val_loader
