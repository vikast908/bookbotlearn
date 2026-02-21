"""
data/tokenizer.py - Word-Level Tokenizer
==========================================

This is a simple word-level tokenizer -- no BPE, no SentencePiece,
no subword magic. Just plain word splitting. This keeps things
educational and transparent.

HOW IT WORKS:
1. Lowercase the entire text
2. Split into tokens using regex (words + punctuation as separate tokens)
3. Count how often each token appears
4. Keep the top N most frequent tokens as our vocabulary
5. Everything else becomes <unk> (unknown)

ZIPF'S LAW:
Word frequencies follow a power law -- a small number of words
(the, of, and, to, ...) account for a huge fraction of all text.
The top 2000 words typically cover ~95% of all token occurrences
in English text. This is why a vocab of 2000 works surprisingly well.

SPECIAL TOKENS:
  <pad> = 0  : Padding token (for batching sequences of different lengths)
  <unk> = 1  : Unknown token (for words not in vocabulary)
  <bos> = 2  : Beginning of sequence
  <eos> = 3  : End of sequence
"""

import json
import os
import re
from collections import Counter
from typing import List


# This regex splits text into:
#   - Words (sequences of letters and apostrophes): he'll -> "he'll"
#   - Individual punctuation marks: "Hello, world!" -> ["hello", ",", "world", "!"]
# The key insight: punctuation becomes its own token, so the model
# can learn that periods end sentences, commas create pauses, etc.
TOKEN_PATTERN = re.compile(r"[a-zA-Z']+|[.,!?;:\-\"]")

# Special tokens and their fixed indices
SPECIAL_TOKENS = {
    "<pad>": 0,
    "<unk>": 1,
    "<bos>": 2,
    "<eos>": 3,
}


class WordTokenizer:
    """
    A minimal word-level tokenizer.

    Usage:
        tokenizer = WordTokenizer()
        tokenizer.build_vocab(text, max_vocab=2000)
        token_ids = tokenizer.encode("To be or not to be")
        text = tokenizer.decode(token_ids)
    """

    def __init__(self):
        self.word2idx: dict[str, int] = {}
        self.idx2word: dict[int, str] = {}
        self.vocab_size: int = 0

    def build_vocab(self, text: str, max_vocab: int = 2000) -> None:
        """
        Build vocabulary from a text corpus.

        Steps:
        1. Tokenize the entire text into words
        2. Count frequency of each word
        3. Keep top (max_vocab - num_special_tokens) most frequent words
        4. Assign an integer index to each word

        Args:
            text: The full text corpus
            max_vocab: Maximum vocabulary size (including special tokens)
        """
        # Step 1: Tokenize
        tokens = self._tokenize(text)
        print(f"[tokenizer] Total tokens in corpus: {len(tokens):,}")

        # Step 2: Count frequencies
        # Counter gives us {word: count} sorted by frequency
        freq = Counter(tokens)
        print(f"[tokenizer] Unique tokens: {len(freq):,}")

        # Step 3: Keep top-N words (reserve spots for special tokens)
        num_regular = max_vocab - len(SPECIAL_TOKENS)
        most_common = freq.most_common(num_regular)

        # Step 4: Build the mappings
        # Start with special tokens at fixed indices
        self.word2idx = dict(SPECIAL_TOKENS)

        # Add regular tokens starting after special tokens
        for idx, (word, count) in enumerate(most_common):
            self.word2idx[word] = idx + len(SPECIAL_TOKENS)

        # Build reverse mapping
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

        # Print coverage statistics
        total_tokens = len(tokens)
        covered = sum(count for _, count in most_common)
        coverage = covered / total_tokens * 100
        print(f"[tokenizer] Vocabulary size: {self.vocab_size}")
        print(f"[tokenizer] Coverage: {coverage:.1f}% of all tokens")
        print(f"[tokenizer] Top 10 words: {[w for w, _ in most_common[:10]]}")

    def encode(self, text: str) -> List[int]:
        """
        Convert text to a list of token IDs.

        Words not in the vocabulary are mapped to <unk> (index 1).
        This is called "OOV" (out-of-vocabulary) handling.

        Example:
            "To be or not" -> [45, 12, 7, 89] (hypothetical indices)
        """
        tokens = self._tokenize(text)
        unk_idx = SPECIAL_TOKENS["<unk>"]
        return [self.word2idx.get(token, unk_idx) for token in tokens]

    def decode(self, token_ids: List[int]) -> str:
        """
        Convert a list of token IDs back to text.

        Attempts basic de-tokenization:
        - Joins words with spaces
        - Attaches punctuation to the preceding word (no space before punctuation)

        Example:
            [45, 12, 7] -> "to be or"
        """
        words = []
        for idx in token_ids:
            word = self.idx2word.get(idx, "<unk>")
            # Skip special tokens in output
            if word in SPECIAL_TOKENS:
                continue
            words.append(word)

        # Join with spaces, then clean up punctuation spacing
        text = " ".join(words)
        # Remove space before punctuation: "hello , world" -> "hello, world"
        text = re.sub(r'\s+([.,!?;:"\-])', r'\1', text)
        return text

    def _tokenize(self, text: str) -> List[str]:
        """
        Split text into tokens (lowercase words and punctuation).

        The regex TOKEN_PATTERN matches:
          - Sequences of letters/apostrophes (words)
          - Individual punctuation characters

        Everything else (whitespace, numbers, rare symbols) is ignored.
        """
        return TOKEN_PATTERN.findall(text.lower())

    def save(self, path: str) -> None:
        """Save vocabulary to a JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "word2idx": self.word2idx,
            "vocab_size": self.vocab_size,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[tokenizer] Vocabulary saved to {path}")

    def load(self, path: str) -> None:
        """Load vocabulary from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.word2idx = data["word2idx"]
        self.idx2word = {int(idx): word for word, idx in self.word2idx.items()}
        self.vocab_size = data["vocab_size"]
        print(f"[tokenizer] Vocabulary loaded from {path} ({self.vocab_size} tokens)")
