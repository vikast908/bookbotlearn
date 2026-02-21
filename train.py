"""
train.py - Training Loop for MiniGPT
======================================

This file implements the complete training pipeline:
1. Learning rate schedule (warmup + cosine annealing)
2. AdamW optimizer with proper weight decay
3. Gradient clipping
4. Training loop with logging
5. Validation evaluation
6. Checkpoint saving

KEY CONCEPTS:

CROSS-ENTROPY LOSS:
    The standard loss for language modeling. For each position, the model
    outputs a probability distribution over the vocabulary. Cross-entropy
    measures how far this distribution is from the true next word.

    Loss = -log(P(correct_word))

    If the model assigns probability 0.9 to the correct word: loss = 0.1
    If the model assigns probability 0.01: loss = 4.6 (much higher!)

PERPLEXITY:
    Perplexity = exp(cross_entropy_loss)

    Intuition: "how many words is the model choosing between?"
    - Perplexity 1.0 = perfect prediction (always picks the right word)
    - Perplexity 2000 = random guessing from vocab (initial state)
    - Perplexity 50 = narrowed it down to ~50 candidates (good for small model!)

LEARNING RATE SCHEDULE:
    1. WARMUP (first 200 steps): LR ramps from 0 to max_lr
       Why? Random initial weights cause wild gradients. Starting with
       a small LR lets the model find a reasonable region before going fast.

    2. COSINE ANNEALING (remaining steps): LR smoothly decays to min_lr
       Why? As the model converges, we want smaller updates to fine-tune.
       Cosine is smoother than step decay, avoiding sudden disruptions.

ADAMW OPTIMIZER:
    Adam with decoupled weight decay. Standard Adam applies weight decay
    INSIDE the gradient update (incorrect). AdamW applies it OUTSIDE
    (correct). This matters for regularization.

    Weight decay is ONLY applied to weight matrices (2D tensors), NOT to:
    - Biases (1D tensors) - regularizing biases hurts performance
    - LayerNorm parameters - these should be free to scale/shift
"""

import math
import os
import time

import torch
import torch.nn.functional as F

from config import TransformerConfig
from model.transformer import MiniGPT
from data.tokenizer import WordTokenizer
from data.dataset import create_datasets, create_dataloaders
from data.download import download_shakespeare


def get_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """
    Compute learning rate for a given training step.

    Schedule:
        Steps 0 to warmup_steps:
            Linear warmup from 0 to max_lr
            LR = max_lr * (step / warmup_steps)

        Steps warmup_steps to max_steps:
            Cosine annealing from max_lr to min_lr
            progress = (step - warmup) / (max_steps - warmup)
            LR = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress))

    The cosine function smoothly transitions from 1 to -1 over [0, pi],
    so (1 + cos(pi * progress)) / 2 goes from 1.0 to 0.0 smoothly.

    Args:
        step: Current training step
        warmup_steps: Number of warmup steps
        max_steps: Total training steps
        max_lr: Peak learning rate
        min_lr: Minimum learning rate at the end

    Returns:
        Learning rate for this step
    """
    # Phase 1: Linear warmup
    if step < warmup_steps:
        return max_lr * (step / warmup_steps)

    # Phase 2: Cosine annealing
    if step >= max_steps:
        return min_lr

    # How far through the annealing phase (0.0 to 1.0)
    progress = (step - warmup_steps) / (max_steps - warmup_steps)

    # Cosine decay: starts at max_lr, ends at min_lr
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def configure_optimizer(
    model: MiniGPT,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    """
    Create AdamW optimizer with separate parameter groups.

    Weight decay (L2 regularization) is only applied to weight MATRICES,
    not to biases or LayerNorm parameters. This is standard practice
    from GPT-2/3 and prevents over-regularizing small parameters.

    Why? Biases and normalization parameters have very different roles
    than weight matrices. Regularizing them doesn't help and can hurt.
    """
    # Separate parameters into two groups:
    # 1. Parameters that SHOULD have weight decay (2D weight matrices)
    # 2. Parameters that should NOT (biases, LayerNorm gamma/beta)
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # 2D+ tensors are weight matrices -> apply weight decay
        # 1D tensors are biases or norm parameters -> no weight decay
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    print(f"[optimizer] Params with weight decay:    {sum(p.numel() for p in decay_params):,}")
    print(f"[optimizer] Params without weight decay: {sum(p.numel() for p in no_decay_params):,}")

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=learning_rate)

    return optimizer


@torch.no_grad()
def evaluate(
    model: MiniGPT,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """
    Compute average validation loss.

    @torch.no_grad() disables gradient computation for efficiency.
    We don't need gradients during evaluation -- just forward passes.
    """
    model.eval()  # Set to evaluation mode (disables dropout)
    total_loss = 0.0
    num_batches = 0

    for x, y in val_loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)

        # Reshape for cross-entropy: (batch*seq_len, vocab_size) and (batch*seq_len,)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
        )
        total_loss += loss.item()
        num_batches += 1

    model.train()  # Set back to training mode
    return total_loss / max(num_batches, 1)


def train(config: TransformerConfig, callback=None, stop_event=None):
    """
    Full training pipeline.

    Steps:
    1. Download dataset (if needed)
    2. Build tokenizer and vocabulary
    3. Create train/val datasets and dataloaders
    4. Initialize model
    5. Training loop with LR schedule, gradient clipping, logging
    6. Save final checkpoint

    Args:
        config: Hyperparameter configuration
        callback: Optional function called with metric dicts for UI progress
        stop_event: Optional threading.Event to signal early stopping
    """
    # ── Device Selection ──
    # Use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[train] Using device: {device}")

    # ── Data Preparation ──
    print("\n--- Data Preparation ---")
    download_shakespeare(config.data_path)

    with open(config.data_path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"[train] Corpus size: {len(text):,} characters")

    # Build tokenizer
    tokenizer = WordTokenizer()
    tokenizer.build_vocab(text, max_vocab=config.vocab_size)
    tokenizer.save(config.vocab_path)

    # Update config with actual vocab size (might be slightly different)
    actual_vocab = tokenizer.vocab_size
    if actual_vocab != config.vocab_size:
        print(f"[train] Adjusting vocab_size: {config.vocab_size} -> {actual_vocab}")
        config.vocab_size = actual_vocab

    # Create datasets
    train_dataset, val_dataset = create_datasets(
        text, tokenizer, config.max_seq_len
    )
    train_loader, val_loader = create_dataloaders(
        train_dataset, val_dataset, config.batch_size
    )

    # ── Model Initialization ──
    print("\n--- Model Initialization ---")
    model = MiniGPT(config).to(device)
    model.count_parameters()

    # ── Optimizer ──
    optimizer = configure_optimizer(model, config.learning_rate, config.weight_decay)

    # ── Training Loop ──
    print("\n--- Training ---")
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    global_step = 0
    best_val_loss = float("inf")
    start_time = time.time()

    model.train()

    for epoch in range(config.max_epochs):
        for batch_idx, (x, y) in enumerate(train_loader):
            # Check if we've reached max steps or stop was requested
            if global_step >= config.max_steps:
                break
            if stop_event and stop_event.is_set():
                break

            x, y = x.to(device), y.to(device)

            # ── Forward Pass ──
            logits = model(x)  # (batch, seq_len, vocab_size)

            # ── Compute Loss ──
            # Cross-entropy expects:
            #   logits: (N, C) where N = batch*seq_len, C = vocab_size
            #   targets: (N,) where each value is in [0, C)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
            )

            # ── Backward Pass ──
            optimizer.zero_grad()  # Clear old gradients
            loss.backward()        # Compute new gradients

            # ── Gradient Clipping ──
            # Prevents "exploding gradients" that can destabilize training.
            # If the total gradient norm exceeds grad_clip, all gradients
            # are scaled down proportionally.
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clip
            )

            # ── Update Learning Rate ──
            lr = get_lr(
                global_step, config.warmup_steps, config.max_steps,
                config.learning_rate, config.min_lr,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # ── Optimizer Step ──
            optimizer.step()  # Update weights using gradients

            # ── Logging ──
            if global_step % config.log_interval == 0:
                elapsed = time.time() - start_time
                perplexity = math.exp(min(loss.item(), 20))  # Cap to avoid overflow
                print(
                    f"  Step {global_step:5d} | "
                    f"Loss: {loss.item():.4f} | "
                    f"PPL: {perplexity:7.1f} | "
                    f"LR: {lr:.2e} | "
                    f"Grad: {grad_norm:.2f} | "
                    f"Time: {elapsed:.1f}s"
                )
                if callback:
                    callback({
                        "type": "log",
                        "step": global_step,
                        "max_steps": config.max_steps,
                        "loss": loss.item(),
                        "perplexity": perplexity,
                        "lr": lr,
                        "grad_norm": grad_norm.item() if hasattr(grad_norm, 'item') else float(grad_norm),
                        "elapsed": elapsed,
                    })

            # ── Evaluation ──
            if global_step > 0 and global_step % config.eval_interval == 0:
                val_loss = evaluate(model, val_loader, device)
                val_ppl = math.exp(min(val_loss, 20))
                print(
                    f"\n  [EVAL] Step {global_step} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val PPL: {val_ppl:.1f}"
                )

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model, optimizer, config, global_step, val_loss)
                    print(f"  [EVAL] New best! Saved checkpoint.\n")
                else:
                    print()

                if callback:
                    callback({
                        "type": "eval",
                        "step": global_step,
                        "max_steps": config.max_steps,
                        "val_loss": val_loss,
                        "val_perplexity": val_ppl,
                        "best_val_loss": best_val_loss,
                        "elapsed": time.time() - start_time,
                    })

                model.train()  # Back to training mode

            global_step += 1

        if global_step >= config.max_steps:
            break
        if stop_event and stop_event.is_set():
            break

    # ── Final Evaluation & Save ──
    val_loss = evaluate(model, val_loader, device)
    val_ppl = math.exp(min(val_loss, 20))
    total_time = time.time() - start_time

    print(f"\n--- Training Complete ---")
    print(f"  Total steps: {global_step}")
    print(f"  Final val loss: {val_loss:.4f}")
    print(f"  Final val perplexity: {val_ppl:.1f}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Total time: {total_time:.1f}s")

    if callback:
        callback({
            "type": "complete",
            "step": global_step,
            "max_steps": config.max_steps,
            "val_loss": val_loss,
            "val_perplexity": val_ppl,
            "best_val_loss": best_val_loss,
            "elapsed": total_time,
        })

    save_checkpoint(model, optimizer, config, global_step, val_loss)
    print(f"  Checkpoint saved to {config.checkpoint_path}")

    return model, tokenizer


def save_checkpoint(
    model: MiniGPT,
    optimizer: torch.optim.Optimizer,
    config: TransformerConfig,
    step: int,
    val_loss: float,
):
    """Save model checkpoint."""
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "step": step,
        "val_loss": val_loss,
    }, config.checkpoint_path)


def load_checkpoint(
    config: TransformerConfig,
    device: torch.device,
) -> tuple[MiniGPT, TransformerConfig]:
    """Load model from checkpoint."""
    checkpoint = torch.load(config.checkpoint_path, map_location=device, weights_only=False)
    saved_config = checkpoint["config"]

    model = MiniGPT(saved_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"[checkpoint] Loaded from step {checkpoint['step']}")
    print(f"[checkpoint] Val loss: {checkpoint['val_loss']:.4f}")

    return model, saved_config
