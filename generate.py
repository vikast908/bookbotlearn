"""
generate.py - Text Generation with MiniGPT
=============================================

Once trained, the model generates text AUTOREGRESSIVELY:
    1. Start with a prompt (e.g., "ROMEO:")
    2. Feed it through the model to get next-word probabilities
    3. Sample a word from those probabilities
    4. Append the sampled word to the sequence
    5. Repeat from step 2

The quality and diversity of generated text is controlled by
SAMPLING STRATEGIES:

GREEDY DECODING (temperature=0):
    Always pick the most probable word. Deterministic but boring --
    often produces repetitive, safe text.
    Example: "the the the the the..."

TEMPERATURE SCALING:
    Divide logits by temperature T before softmax.

    T < 1.0 -> SHARPER distribution (more confident, less diverse)
               The model strongly prefers its top choices
    T = 1.0 -> UNCHANGED (use the model's raw predictions)
    T > 1.0 -> FLATTER distribution (less confident, more diverse)
               Even unlikely words have a chance

    Math insight: softmax(x/T) approaches:
    - One-hot (argmax) as T -> 0
    - Uniform distribution as T -> infinity

TOP-K SAMPLING:
    Only consider the top K most likely words. Set all others to 0 probability.
    This prevents the model from ever picking extremely unlikely words
    that would produce nonsense.

    K = 1:    Same as greedy (only top word)
    K = 40:   Choose from top 40 candidates (good default)
    K = 2000: No filtering (full vocabulary)

TOP-P (NUCLEUS) SAMPLING:
    Instead of a fixed K, dynamically choose the smallest set of words
    whose cumulative probability exceeds P.

    If the model is very confident (one word has 95% probability),
    nucleus sampling might only consider 1-2 words.
    If the model is uncertain, it considers more words.

    P = 0.9 is a common choice.
"""

import torch
import torch.nn.functional as F

from model.transformer import MiniGPT
from data.tokenizer import WordTokenizer


@torch.no_grad()
def generate(
    model: MiniGPT,
    tokenizer: WordTokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 40,
    device: torch.device = None,
) -> str:
    """
    Generate text from a prompt using the trained model.

    Args:
        model: Trained MiniGPT model
        tokenizer: WordTokenizer with vocabulary
        prompt: Starting text (e.g., "ROMEO:")
        max_tokens: Maximum number of tokens to generate
        temperature: Controls randomness (0 = greedy, 1 = normal, >1 = more random)
        top_k: Only sample from top-k most likely tokens (0 = no filtering)
        device: Computation device (CPU/GPU)

    Returns:
        Generated text string (prompt + generated continuation)
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Encode the prompt into token IDs
    token_ids = tokenizer.encode(prompt)
    if len(token_ids) == 0:
        # If prompt produces no tokens, start with beginning-of-sequence
        token_ids = [2]  # <bos>

    # Convert to tensor: shape (1, seq_len) -- batch size of 1
    tokens = torch.tensor([token_ids], dtype=torch.long, device=device)

    # Generate tokens one at a time
    generated_ids = list(token_ids)

    for _ in range(max_tokens):
        # Truncate to max_seq_len if sequence gets too long
        # The model can only handle max_seq_len tokens of context
        context = tokens[:, -model.config.max_seq_len:]

        # Forward pass: get logits for ALL positions
        logits = model(context)  # (1, seq_len, vocab_size)

        # We only care about the LAST position's prediction
        # (what comes after the last token we've seen)
        next_logits = logits[:, -1, :]  # (1, vocab_size)

        # Apply temperature scaling
        next_logits = apply_temperature(next_logits, temperature)

        # Apply top-k filtering
        if top_k > 0:
            next_logits = apply_top_k(next_logits, top_k)

        # Convert logits to probabilities
        probs = F.softmax(next_logits, dim=-1)

        # Sample from the distribution
        if temperature == 0:
            # Greedy: always pick the most likely word
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
        else:
            # Stochastic: randomly sample proportional to probabilities
            next_token = torch.multinomial(probs, num_samples=1)

        # Append to sequence
        tokens = torch.cat([tokens, next_token], dim=1)
        generated_ids.append(next_token.item())

        # Stop if we generate an end-of-sequence token
        if next_token.item() == 3:  # <eos>
            break

    # Decode back to text
    return tokenizer.decode(generated_ids)


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Scale logits by temperature.

    Temperature controls the "confidence" of the probability distribution:

    logits = [2.0, 1.0, 0.5]

    T=0.5 (sharper):  logits/T = [4.0, 2.0, 1.0]  -> probs ≈ [0.84, 0.11, 0.04]
    T=1.0 (normal):   logits/T = [2.0, 1.0, 0.5]  -> probs ≈ [0.56, 0.21, 0.12]
    T=2.0 (flatter):  logits/T = [1.0, 0.5, 0.25] -> probs ≈ [0.42, 0.25, 0.20]

    Lower temperature -> more confident -> less diverse text
    Higher temperature -> less confident -> more diverse text
    """
    if temperature == 0:
        return logits  # Will use argmax anyway
    return logits / temperature


def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Keep only the top-k logits, set the rest to -infinity.

    This prevents sampling extremely unlikely tokens that would
    produce nonsensical text.

    Steps:
    1. Find the k-th largest logit value (the threshold)
    2. Set all logits below this threshold to -inf
    3. After softmax, these -inf values become probability 0

    Example with k=3:
        logits = [2.0, 0.5, 1.8, -1.0, 1.5]
        Top 3:   [2.0,      1.8,        1.5]  (threshold = 1.5)
        After:   [2.0, -inf, 1.8, -inf, 1.5]
        Probs:   [0.41, 0, 0.33, 0, 0.25]  (only 3 candidates)
    """
    if k >= logits.size(-1):
        return logits  # No filtering needed

    # Get the k-th largest value
    # topk returns (values, indices), we just need the values
    top_k_values, _ = torch.topk(logits, k, dim=-1)

    # The threshold is the smallest value among the top-k
    threshold = top_k_values[:, -1].unsqueeze(-1)

    # Set everything below the threshold to -infinity
    logits = logits.masked_fill(logits < threshold, float("-inf"))

    return logits


def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Nucleus (top-p) sampling: keep the smallest set of tokens whose
    cumulative probability exceeds p.

    Unlike top-k which always considers exactly k tokens, top-p
    ADAPTS to the model's confidence:
    - Confident prediction (one token has 95% prob) -> few candidates
    - Uncertain prediction (many tokens with similar prob) -> many candidates

    Args:
        logits: Raw scores of shape (batch, vocab_size)
        p: Cumulative probability threshold (e.g., 0.9)

    Returns:
        Filtered logits with low-probability tokens set to -inf
    """
    # Sort tokens by probability (descending)
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)

    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Find tokens where cumulative probability exceeds p
    # We shift right by 1 so that the token that crosses p is INCLUDED
    sorted_mask = cumulative_probs - sorted_probs > p

    # Set filtered tokens to -inf
    sorted_logits[sorted_mask] = float("-inf")

    # Un-sort back to original order
    logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

    return logits
