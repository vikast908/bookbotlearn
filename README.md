# MiniGPT: Build a Transformer From Scratch (~96K Parameters)

A fully educational, from-scratch implementation of a GPT-style transformer language model in PyTorch. Every single component — attention, layer normalization, positional encoding, feed-forward networks — is hand-built and heavily commented to teach you **exactly** how transformers work.

Train it on Shakespeare. Generate text. Understand everything.

```
Total Parameters: 95,568 (~96K)
Model Size:       ~384 KB
Training Time:    ~5 min on GPU, ~30 min on CPU
```

---

## Table of Contents

- [Why This Project?](#why-this-project)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Data Flow: End-to-End Pipeline](#data-flow-end-to-end-pipeline)
  - [Phase 1: Data Acquisition](#phase-1-data-acquisition)
  - [Phase 2: Tokenization](#phase-2-tokenization)
  - [Phase 3: Dataset Creation](#phase-3-dataset-creation)
  - [Phase 4: Batching with DataLoader](#phase-4-batching-with-dataloader)
  - [Phase 5: Forward Pass Through the Model](#phase-5-forward-pass-through-the-model)
  - [Phase 6: Loss Computation and Backpropagation](#phase-6-loss-computation-and-backpropagation)
  - [Phase 7: Optimizer Step](#phase-7-optimizer-step)
  - [Phase 8: Evaluation and Checkpointing](#phase-8-evaluation-and-checkpointing)
  - [Phase 9: Text Generation (Inference)](#phase-9-text-generation-inference)
- [Architecture Deep Dive](#architecture-deep-dive)
  - [The Big Picture](#the-big-picture)
  - [Parameter Budget](#parameter-budget)
  - [1. Token Embedding](#1-token-embedding)
  - [2. Positional Encoding](#2-positional-encoding)
  - [3. Multi-Head Self-Attention](#3-multi-head-self-attention)
  - [4. Feed-Forward Network](#4-feed-forward-network)
  - [5. Transformer Block](#5-transformer-block)
  - [6. Full MiniGPT Model](#6-full-minigpt-model)
  - [7. Weight Tying](#7-weight-tying)
- [Training Deep Dive](#training-deep-dive)
  - [Cross-Entropy Loss](#cross-entropy-loss)
  - [Perplexity](#perplexity)
  - [Learning Rate Schedule](#learning-rate-schedule)
  - [AdamW Optimizer](#adamw-optimizer)
  - [Gradient Clipping](#gradient-clipping)
- [Generation Deep Dive](#generation-deep-dive)
  - [Autoregressive Generation](#autoregressive-generation)
  - [Temperature Scaling](#temperature-scaling)
  - [Top-K Sampling](#top-k-sampling)
  - [Top-P (Nucleus) Sampling](#top-p-nucleus-sampling)
- [Data Pipeline](#data-pipeline)
  - [Tokenization](#tokenization)
  - [Sliding Window Dataset](#sliding-window-dataset)
- [Web UI](#web-ui)
- [Configuration Reference](#configuration-reference)
- [Concepts You'll Learn](#concepts-youll-learn)
- [Further Reading](#further-reading)
- [License](#license)

---

## Why This Project?

Most transformer tutorials either:
- Use high-level libraries that hide the math (`nn.TransformerEncoder`)
- Are too large to understand in one sitting (thousands of lines)
- Skip the "why" and only show the "what"

**MiniGPT is different.** It's a complete, working language model that:
- Implements **every component from scratch** (no `nn.TransformerEncoder`)
- Fits in **~96K parameters** (you can literally count them)
- Has **extensive comments** explaining the math and intuition behind every operation
- **Actually trains** and generates semi-coherent Shakespeare text
- Includes a **Gradio web UI** for interactive training and generation
- Runs on **CPU or GPU** (auto-detects CUDA)

If you can read Python, you can understand this entire model.

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/vikast908/bookbotlearn.git
cd bookbotlearn
pip install -r requirements.txt
```

For GPU support (NVIDIA), install PyTorch with CUDA:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### 2. CLI Usage

```bash
# Download the Tiny Shakespeare dataset (~1.1 MB)
python run.py download

# Train the model (5000 steps, ~5 min on GPU)
python run.py train

# Generate text from a prompt
python run.py generate "ROMEO:"

# Generate with custom sampling parameters
python run.py generate "To be or not" --temperature 1.0 --top_k 50 --max_tokens 200

# Print model architecture and parameter count
python run.py info
```

### 3. Web UI

```bash
python app.py
# Opens at http://localhost:7860
```

The web interface lets you train, generate, and inspect the model visually.

---

## Project Structure

```
bookbotlearn/
├── README.md                      # You are here
├── requirements.txt               # torch, numpy, gradio, matplotlib
├── config.py                      # All hyperparameters in one dataclass
├── run.py                         # CLI entry point (download/train/generate/info)
├── train.py                       # Training loop, LR schedule, optimizer, checkpointing
├── generate.py                    # Text generation with sampling strategies
├── app.py                         # Gradio web UI (training + generation + model info)
│
├── data/
│   ├── __init__.py
│   ├── download.py                # Downloads Tiny Shakespeare dataset
│   ├── tokenizer.py               # Word-level tokenizer (build vocab, encode, decode)
│   └── dataset.py                 # PyTorch Dataset with sliding window
│
├── model/
│   ├── __init__.py
│   ├── layernorm.py               # Layer Normalization (from scratch)
│   ├── positional.py              # Learned positional embeddings
│   ├── attention.py               # Scaled dot-product + multi-head attention
│   ├── feedforward.py             # Position-wise FFN with GELU activation
│   ├── transformer_block.py       # Single pre-norm transformer decoder block
│   └── transformer.py             # Full MiniGPT model (assembles everything)
│
└── checkpoints/                   # Saved model weights (created during training)
```

**Each file is a self-contained lesson.** Read them in this order for the best learning experience:

1. `config.py` — What knobs does a transformer have?
2. `model/layernorm.py` — How do we stabilize training?
3. `model/positional.py` — How does the model know word order?
4. `model/attention.py` — The heart: how tokens "look at" each other
5. `model/feedforward.py` — How each token "thinks" independently
6. `model/transformer_block.py` — How attention + FFN combine with residuals
7. `model/transformer.py` — The full model assembly
8. `data/tokenizer.py` — How text becomes numbers
9. `data/dataset.py` — How we create training examples
10. `train.py` — How the model learns
11. `generate.py` — How the model creates text

---

## Data Flow: End-to-End Pipeline

This section traces every transformation the data undergoes, from raw text on the internet to generated Shakespeare, with every parameter and shape annotated.

```
                        TRAINING PIPELINE
                        =================

Internet (GitHub)                        Trained Model
    │                                         │
    ▼                                         ▼
[Raw Text]──► [Tokens]──► [IDs]──► [Batches]──► [Model]──► [Loss]──► [Backprop]──► [Update Weights]
 1.1 MB       ~210K       ints     (64,128)     forward     scalar     gradients     AdamW
Shakespeare   words       0-1999   tensors      pass        cross-     ∂L/∂w         step
                                                            entropy


                        INFERENCE PIPELINE
                        ==================

[Prompt]──► [Encode]──► [Model Forward]──► [Logits]──► [Sample]──► [Append]──► [Decode]──► [Text]
"ROMEO:"    [token IDs]  (1, seq, 2000)    last pos    temp+top-k   loop        join words   output
                                           (1, 2000)   → next ID    ×max_tokens
```

### Phase 1: Data Acquisition

**File:** `data/download.py` | **Function:** `download_shakespeare()`

```
Source:  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
   │
   ▼  urllib.request.urlretrieve()
   │
Saved:  data/input.txt
   │
   Stats: ~1.1 MB | ~40,000 lines | ~900,000 characters | ~25,000 unique words
```

The Tiny Shakespeare dataset is all of Shakespeare's works concatenated into a single text file. If `data/input.txt` already exists, the download is skipped.

### Phase 2: Tokenization

**File:** `data/tokenizer.py` | **Class:** `WordTokenizer`

The tokenizer converts raw text into integer IDs that the model can process.

```
Step 2a: Lowercase
   "First Citizen:" → "first citizen:"

Step 2b: Regex tokenization (TOKEN_PATTERN = r"[a-zA-Z']+|[.,!?;:\-\"]")
   "first citizen:" → ["first", "citizen", ":"]
   - Words (including contractions like "he'll") become tokens
   - Each punctuation mark becomes its own token
   - Whitespace, numbers, rare symbols are discarded

Step 2c: Count frequencies (Counter)
   {"the": 23,243, "and": 13,891, "i": 12,456, "to": 11,234, ...}
   Total: ~210,000 tokens | ~14,000 unique tokens

Step 2d: Build vocabulary (top 2000 - 4 special tokens = 1,996 regular words)
   Index 0: <pad>    (padding)
   Index 1: <unk>    (unknown/out-of-vocabulary)
   Index 2: <bos>    (beginning of sequence)
   Index 3: <eos>    (end of sequence)
   Index 4: "the"    (most frequent word)
   Index 5: ","      (second most frequent)
   ...
   Index 1999: (1996th most frequent word)

   Coverage: top 2000 words cover ~95% of all token occurrences (Zipf's law)
   Remaining ~12,000 rare words → mapped to <unk> (index 1)

Step 2e: Encode entire corpus
   "first citizen:" → [234, 891, 5]  (hypothetical indices)

   tokenizer.encode(text) → List[int] of ~210,000 token IDs

Step 2f: Save vocabulary
   data/vocab.json ← {"word2idx": {"<pad>": 0, "<unk>": 1, ...}, "vocab_size": 2000}
```

**Key parameters:**
| Parameter | Value | Role |
|---|---|---|
| `config.vocab_size` | 2000 | Maximum vocabulary size (including 4 special tokens) |
| `TOKEN_PATTERN` | `r"[a-zA-Z']+\|[.,!?;:\-\"]"` | Regex that splits text into word and punctuation tokens |

### Phase 3: Dataset Creation

**File:** `data/dataset.py` | **Function:** `create_datasets()` | **Class:** `ShakespeareDataset`

The encoded corpus is split and windowed into (input, target) training pairs.

```
Step 3a: Train/Val split by position (90/10)
   All token IDs: [45, 12, 7, 89, 3, 56, 102, 23, ...]  (~210,000 IDs)
                   |◄──────── 90% train ────────►|◄─ 10% val ─►|
   split_idx = int(210,000 × 0.9) = 189,000
   Train: token_ids[0 : 189,000]      → ~189,000 tokens
   Val:   token_ids[189,000 : end]    → ~21,000 tokens

Step 3b: Convert to PyTorch LongTensor
   self.data = torch.tensor(token_ids, dtype=torch.long)

Step 3c: Sliding window to create samples
   seq_len = config.max_seq_len = 128

   Sample 0:  x = data[0   : 128]    y = data[1   : 129]
   Sample 1:  x = data[1   : 129]    y = data[2   : 130]
   Sample 2:  x = data[2   : 130]    y = data[3   : 131]
   ...
   Sample N:  x = data[N   : N+128]  y = data[N+1 : N+129]

   Total train samples: 189,000 - 128 = 188,872
   Total val samples:   ~21,000 - 128 = ~20,872

   Each x: (128,) tensor of input token IDs
   Each y: (128,) tensor of target token IDs (x shifted right by 1)

   At every position i in x, the model must predict y[i] = x[i+1]
```

**Key parameters:**
| Parameter | Value | Role |
|---|---|---|
| `config.max_seq_len` | 128 | Window size (context length) for each training sample |
| `train_split` | 0.9 | Fraction of corpus used for training (rest is validation) |

### Phase 4: Batching with DataLoader

**File:** `data/dataset.py` | **Function:** `create_dataloaders()`

```
Step 4a: Wrap in DataLoader
   Train DataLoader: shuffle=True, drop_last=True, batch_size=64
   Val DataLoader:   shuffle=False, drop_last=False, batch_size=64

Step 4b: Each iteration yields a batch
   x batch: (64, 128) — 64 samples, each 128 tokens long (LongTensor)
   y batch: (64, 128) — corresponding targets

   Train batches per epoch: 188,872 // 64 = ~2,951
   Val batches:             ~20,872 // 64 = ~326

Step 4c: Move to device
   x, y = x.to(device), y.to(device)   # CPU or CUDA GPU
```

**Key parameters:**
| Parameter | Value | Role |
|---|---|---|
| `config.batch_size` | 64 | Number of samples per gradient update |
| `shuffle` | True (train) | Randomize sample order each epoch to prevent memorization |
| `drop_last` | True (train) | Discard incomplete last batch for consistent batch size |

### Phase 5: Forward Pass Through the Model

**File:** `model/transformer.py` | **Class:** `MiniGPT` | **Method:** `forward()`

This is where the data flows through every layer of the neural network.

```
INPUT: x of shape (64, 128) — batch of 64 sequences, each 128 token IDs
       Each value is an integer in [0, 1999]

═══════════════════════════════════════════════════════════════
STEP 5a: Token Embedding  (model/transformer.py)
═══════════════════════════════════════════════════════════════
   nn.Embedding(vocab_size=2000, d_model=32)
   Lookup table: each token ID → 32-dimensional vector
   Weight matrix shape: (2000, 32) = 64,000 parameters

   (64, 128) → (64, 128, 32)
    batch,seq    batch, seq, d_model

   Example: token ID 45 → [0.12, -0.34, 0.56, ..., 0.78]  (32 floats)

═══════════════════════════════════════════════════════════════
STEP 5b: Positional Encoding  (model/positional.py)
═══════════════════════════════════════════════════════════════
   nn.Embedding(max_seq_len=128, d_model=32)
   Learned position vectors (GPT-2 style)
   Weight matrix shape: (128, 32) = 4,096 parameters

   positions = [0, 1, 2, ..., 127]
   pos_embeddings = self.embedding(positions)  → (128, 32)

   x = token_embeddings + pos_embeddings       (broadcasts over batch)
   (64, 128, 32) + (128, 32) → (64, 128, 32)

   Now each vector encodes BOTH what the word is AND where it is.

═══════════════════════════════════════════════════════════════
STEP 5c: Embedding Dropout
═══════════════════════════════════════════════════════════════
   nn.Dropout(p=0.1)
   Randomly zeros 10% of values during training (scale rest by 1/0.9)
   Disabled during eval (model.eval())

   (64, 128, 32) → (64, 128, 32)  (shape unchanged)

═══════════════════════════════════════════════════════════════
STEP 5d: Transformer Block 1  (model/transformer_block.py)
═══════════════════════════════════════════════════════════════

   ┌─── 5d-i: Pre-Norm LayerNorm (model/layernorm.py) ──────────────┐
   │   gamma (32,) + beta (32,) = 64 parameters                     │
   │   For each position independently:                              │
   │     mean = mean of 32 features                                  │
   │     var  = variance of 32 features                              │
   │     x_norm = (x - mean) / sqrt(var + 1e-6)                     │
   │     output = gamma * x_norm + beta                              │
   │   (64, 128, 32) → (64, 128, 32)                                │
   └─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
   ┌─── 5d-ii: Multi-Head Self-Attention (model/attention.py) ──────┐
   │                                                                 │
   │   Linear projections (each: 32→32, i.e. 32×32+32 = 1,056 params):
   │     Q = W_Q(x)  → (64, 128, 32)                                │
   │     K = W_K(x)  → (64, 128, 32)                                │
   │     V = W_V(x)  → (64, 128, 32)                                │
   │                                                                 │
   │   Reshape to split into 4 heads (d_k = 32/4 = 8 per head):     │
   │     Q: (64, 128, 32) → (64, 128, 4, 8) → (64, 4, 128, 8)     │
   │     K: same reshape                                             │
   │     V: same reshape                                             │
   │                                                                 │
   │   Scaled Dot-Product Attention (per head):                      │
   │     scores = Q @ K^T         → (64, 4, 128, 128)               │
   │     scores = scores / sqrt(8) = scores / 2.83                   │
   │     scores = masked_fill(causal_mask == 0, -inf)                │
   │       Causal mask: lower-triangular (128×128), prevents         │
   │       attending to future positions                              │
   │     attn_weights = softmax(scores, dim=-1) → (64, 4, 128, 128) │
   │     attn_weights = dropout(attn_weights, p=0.1)                 │
   │     attn_output = attn_weights @ V       → (64, 4, 128, 8)     │
   │                                                                 │
   │   Concatenate heads:                                            │
   │     (64, 4, 128, 8) → transpose → (64, 128, 4, 8) → (64, 128, 32)
   │                                                                 │
   │   Output projection W_O (32→32, 1,056 params):                 │
   │     output = W_O(concat)     → (64, 128, 32)                   │
   │                                                                 │
   │   Total attention params: 4 × 1,056 = 4,224                    │
   └─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
   ┌─── 5d-iii: Dropout + Residual Connection ──────────────────────┐
   │   attn_out = dropout(attn_out, p=0.1)                          │
   │   x = residual + attn_out     (add back original input)        │
   │   (64, 128, 32) → (64, 128, 32)                                │
   └─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
   ┌─── 5d-iv: Pre-Norm LayerNorm ──────────────────────────────────┐
   │   Same as 5d-i, separate gamma/beta (64 params)                │
   │   (64, 128, 32) → (64, 128, 32)                                │
   └─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
   ┌─── 5d-v: Feed-Forward Network (model/feedforward.py) ──────────┐
   │                                                                 │
   │   Linear1: d_model → d_ff (32→128, 32×128+128 = 4,224 params)  │
   │     (64, 128, 32) → (64, 128, 128)                             │
   │                                                                 │
   │   GELU activation (smooth ReLU alternative):                    │
   │     gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715*x³)))
   │     (64, 128, 128) → (64, 128, 128)                            │
   │                                                                 │
   │   Dropout(p=0.1)                                                │
   │                                                                 │
   │   Linear2: d_ff → d_model (128→32, 128×32+32 = 4,128 params)   │
   │     (64, 128, 128) → (64, 128, 32)                             │
   │                                                                 │
   │   Dropout(p=0.1)                                                │
   │                                                                 │
   │   Total FFN params: 4,224 + 4,128 = 8,352                      │
   └─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
   ┌─── 5d-vi: Residual Connection ─────────────────────────────────┐
   │   x = residual + ffn_out                                        │
   │   (64, 128, 32) → (64, 128, 32)                                │
   └─────────────────────────────────────────────────────────────────┘

   Block 1 total params: 4,224 (attn) + 8,352 (FFN) + 128 (2×LN) = 12,704

═══════════════════════════════════════════════════════════════
STEP 5e: Transformer Block 2  (identical structure)
═══════════════════════════════════════════════════════════════
   Same architecture, separate weights (another 12,704 params)
   (64, 128, 32) → (64, 128, 32)

═══════════════════════════════════════════════════════════════
STEP 5f: Final Layer Norm  (model/layernorm.py)
═══════════════════════════════════════════════════════════════
   gamma (32,) + beta (32,) = 64 parameters
   (64, 128, 32) → (64, 128, 32)

═══════════════════════════════════════════════════════════════
STEP 5g: Output Head (Linear projection, weight-tied)
═══════════════════════════════════════════════════════════════
   nn.Linear(d_model=32, vocab_size=2000)
   Weight is SHARED with token embedding (weight tying)
     → No additional weight params (saves 64,000!)
     → Only the bias is new: 2,000 parameters

   (64, 128, 32) → (64, 128, 2000)
    batch, seq, d_model    batch, seq, vocab_size

   Output: LOGITS — unnormalized scores for every word at every position
   logits[b][t][w] = how likely word w follows position t in sample b

OUTPUT: logits of shape (64, 128, 2000)
```

### Phase 6: Loss Computation and Backpropagation

**File:** `train.py` | **Training loop**

```
Step 6a: Reshape for cross-entropy
   logits: (64, 128, 2000) → view(-1, 2000) → (8192, 2000)
   targets: (64, 128)      → view(-1)       → (8192,)

   8192 = 64 batches × 128 positions = total predictions per step

Step 6b: Cross-entropy loss
   F.cross_entropy(logits, targets)

   For each of the 8,192 positions:
     loss_i = -log(softmax(logits_i)[target_i])
              = -log(P(correct word at position i))

   Final loss = mean over all 8,192 positions → single scalar

   Starting loss ≈ ln(2000) ≈ 7.6  (random guessing)
   Good trained loss ≈ 4.0-5.0

Step 6c: Backward pass
   optimizer.zero_grad()     # Clear old gradients
   loss.backward()           # Compute ∂loss/∂param for ALL 95,568 parameters
                             # via chain rule (automatic differentiation)

Step 6d: Gradient clipping
   grad_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
   If ||all gradients|| > 1.0, scale them down proportionally
   Prevents exploding gradients from destabilizing training
```

**Key parameters:**
| Parameter | Value | Role |
|---|---|---|
| `config.grad_clip` | 1.0 | Maximum allowed gradient norm |

### Phase 7: Optimizer Step

**File:** `train.py` | **Functions:** `get_lr()`, `configure_optimizer()`

```
Step 7a: Compute learning rate for current step
   if step < 200 (warmup):
     lr = 3e-4 × (step / 200)        # Linear ramp from 0 to 3e-4
   else (cosine annealing):
     progress = (step - 200) / (5000 - 200)
     lr = 3e-5 + 0.5 × (3e-4 - 3e-5) × (1 + cos(π × progress))
     # Smooth decay from 3e-4 to 3e-5

Step 7b: Update learning rate in optimizer
   for param_group in optimizer.param_groups:
       param_group["lr"] = lr

Step 7c: AdamW optimizer step
   Two parameter groups:
     Group 1: 2D+ tensors (weight matrices) — with weight_decay=0.01
     Group 2: 1D tensors (biases, LayerNorm γ/β) — with weight_decay=0.0

   For each parameter:
     m = β1 × m + (1-β1) × grad             # Update 1st moment (momentum)
     v = β2 × v + (1-β2) × grad²            # Update 2nd moment (RMS)
     m̂ = m / (1 - β1^t)                     # Bias correction
     v̂ = v / (1 - β2^t)                     # Bias correction
     param = param - lr × m̂ / (√v̂ + ε)     # Adam update
     param = param - lr × weight_decay × param  # Decoupled weight decay (group 1 only)
```

**Key parameters:**
| Parameter | Value | Role |
|---|---|---|
| `config.learning_rate` | 3e-4 | Peak learning rate (after warmup) |
| `config.min_lr` | 3e-5 | Minimum learning rate at end of cosine decay |
| `config.warmup_steps` | 200 | Steps for linear LR warmup |
| `config.max_steps` | 5000 | Total training steps (controls cosine schedule) |
| `config.weight_decay` | 0.01 | L2 regularization strength on weight matrices |

### Phase 8: Evaluation and Checkpointing

**File:** `train.py` | **Functions:** `evaluate()`, `save_checkpoint()`

```
Step 8a: Evaluate on validation set (every 500 steps)
   model.eval()                      # Disable dropout
   for x, y in val_loader:           # Iterate all ~326 val batches
     logits = model(x)               # Forward pass (no gradient computation)
     loss += cross_entropy(logits, y)
   val_loss = total_loss / num_batches
   val_perplexity = exp(val_loss)    # "How many words is it choosing between?"
   model.train()                     # Re-enable dropout

Step 8b: Save checkpoint (if val_loss improved)
   torch.save({
     "model_state_dict":     all 95,568 trained parameters,
     "optimizer_state_dict":  Adam momentum/variance states,
     "config":               TransformerConfig dataclass,
     "step":                 current training step,
     "val_loss":             best validation loss,
   }, "checkpoints/model.pt")

Step 8c: Training complete (after max_steps)
   Final evaluation on validation set
   Save final checkpoint regardless of improvement
   Report: total steps, final val loss, best val loss, total time
```

**Key parameters:**
| Parameter | Value | Role |
|---|---|---|
| `config.eval_interval` | 500 | Evaluate on validation set every N steps |
| `config.log_interval` | 100 | Print training metrics every N steps |
| `config.max_steps` | 5000 | Stop training after this many steps |
| `config.max_epochs` | 20 | Maximum epochs (usually max_steps is hit first) |

### Phase 9: Text Generation (Inference)

**File:** `generate.py` | **Function:** `generate()`

```
INPUT: prompt = "ROMEO:"

Step 9a: Encode prompt
   tokenizer.encode("ROMEO:") → [456, 5]  (hypothetical IDs)
   tokens = torch.tensor([[456, 5]])       → shape (1, 2)

Step 9b: Autoregressive generation loop (repeat up to max_tokens times)
   ┌────────────────────────────────────────────────────────────────┐
   │                                                                │
   │  9b-i: Truncate context to max_seq_len (128) if needed         │
   │    context = tokens[:, -128:]                                  │
   │                                                                │
   │  9b-ii: Forward pass through entire model                      │
   │    logits = model(context)        → (1, seq_len, 2000)        │
   │                                                                │
   │  9b-iii: Extract last position's prediction                    │
   │    next_logits = logits[:, -1, :] → (1, 2000)                 │
   │    "What word should come after the last token?"               │
   │                                                                │
   │  9b-iv: Temperature scaling                                    │
   │    next_logits = next_logits / temperature                     │
   │    T=0.8: logits/0.8 → sharper distribution (more confident)   │
   │    T=1.0: unchanged                                            │
   │    T=1.5: logits/1.5 → flatter distribution (more random)     │
   │                                                                │
   │  9b-v: Top-K filtering                                         │
   │    Keep only top 40 logits, set rest to -inf                   │
   │    Prevents sampling extremely unlikely nonsense words         │
   │                                                                │
   │  9b-vi: Convert to probabilities                               │
   │    probs = softmax(next_logits)   → (1, 2000)                 │
   │    Sum = 1.0 (only top-40 have nonzero probability)            │
   │                                                                │
   │  9b-vii: Sample next token                                     │
   │    if temperature == 0:                                        │
   │      next_token = argmax(probs)   → greedy (deterministic)    │
   │    else:                                                       │
   │      next_token = multinomial(probs, 1) → stochastic sample   │
   │                                                                │
   │  9b-viii: Append to sequence                                   │
   │    tokens = cat([tokens, next_token], dim=1)                   │
   │    (1, 2) → (1, 3) → (1, 4) → ... grows by 1 each iteration │
   │                                                                │
   │  9b-ix: Check stop condition                                   │
   │    if next_token == 3 (<eos>): break                           │
   │                                                                │
   └────────────── repeat for up to max_tokens iterations ──────────┘

Step 9c: Decode token IDs back to text
   generated_ids = [456, 5, 891, 23, 7, ...]
   tokenizer.decode(generated_ids)
     1. Map each ID → word via idx2word lookup
     2. Skip special tokens (<pad>, <unk>, <bos>, <eos>)
     3. Join with spaces
     4. Clean punctuation spacing: "hello , world" → "hello, world"
     → "romeo: i will not be so bold..."

OUTPUT: Generated text string
```

**Key parameters:**
| Parameter | Value | Role |
|---|---|---|
| `temperature` | 0.8 | Controls randomness (0=greedy, 1=raw, >1=more random) |
| `top_k` | 40 | Only sample from top 40 most likely words |
| `max_tokens` | 100 | Maximum number of words to generate |
| `config.max_seq_len` | 128 | Context window — oldest tokens dropped when exceeded |

---

## Architecture Deep Dive

### The Big Picture

MiniGPT is a **decoder-only transformer** (same architecture as GPT-2, GPT-3, GPT-4, LLaMA, etc.), just much smaller. Here's the data flow:

```
Input: "to be or not to"
         │
         ▼
┌─────────────────────┐
│   Token Embedding    │  Word index → 32-dim vector
│   (2000 × 32)       │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ + Positional Encoding│  Position → 32-dim vector (added)
│   (128 × 32)        │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Transformer Block 1 │  Attention + FFN + Residuals
│  (12,704 params)     │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Transformer Block 2 │  Attention + FFN + Residuals
│  (12,704 params)     │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Final Layer Norm   │  Normalize before output
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│    Output Head       │  32-dim vector → 2000 vocabulary scores
│  (weight-tied)       │
└─────────┬───────────┘
          │
          ▼
Output: probability distribution over 2000 words
        → next word prediction: "be" (highest probability)
```

### Parameter Budget

With only ~96K parameters, every parameter matters. Here's where they go:

| Component | Parameters | % of Total | Purpose |
|---|---:|---:|---|
| Token Embedding (tied) | 64,000 | 66.9% | Map each of 2000 words to a 32-dim vector |
| Positional Encoding | 4,096 | 4.3% | 128 positions × 32-dim learned vectors |
| Transformer Blocks (×2) | 25,408 | 26.6% | Attention + FFN, the "brain" of the model |
| Final LayerNorm | 64 | 0.1% | Scale + shift before output (gamma + beta) |
| Output Bias | 2,000 | 2.1% | One bias per vocabulary word |
| **TOTAL** | **95,568** | **100%** | |

The embedding layer dominates at 67% — this is typical for small transformer models. As models grow larger, the transformer body's O(n_layers × d_model²) scaling eventually overtakes the embedding's O(vocab_size × d_model).

### 1. Token Embedding

**File:** `model/transformer.py` (line ~92)

Every word in our vocabulary gets a learnable 32-dimensional vector. These vectors start random and are optimized during training to capture semantic meaning.

```python
self.token_embedding = nn.Embedding(vocab_size, d_model)
# Shape: (2000, 32) = 64,000 parameters
```

After training, similar words (like "king" and "queen") end up with similar vectors. This is the foundation of how neural networks "understand" language.

### 2. Positional Encoding

**File:** `model/positional.py`

**The problem:** Self-attention is permutation equivariant. It treats "dog bites man" and "man bites dog" identically because it has no notion of word ORDER.

**The solution:** Add position information to each token:

```
final_embedding = token_embedding + positional_encoding
```

We use **learned positional embeddings** (same as GPT-2): each position 0 through 127 gets its own learnable 32-dim vector.

```python
self.embedding = nn.Embedding(max_seq_len, d_model)
# Shape: (128, 32) = 4,096 parameters
```

**Alternative (commented in code):** The original Transformer paper used fixed sinusoidal encodings:
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```
Different dimensions use different frequencies — low frequencies capture coarse position, high frequencies capture fine-grained adjacency.

### 3. Multi-Head Self-Attention

**File:** `model/attention.py`

This is the **heart** of the transformer. Attention lets each token "look at" every other token and decide which ones are relevant.

**Intuition:** When processing the word "it" in *"The cat sat on the mat because it was tired"*, the model needs to figure out that "it" refers to "cat". Attention learns to focus on "cat" when processing "it".

#### Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V
```

Three players at each position:
- **Q (Query):** "What am I looking for?"
- **K (Key):** "What do I contain?"
- **V (Value):** "What information do I provide?"

Step by step:

```
1. Compute raw scores:      scores = Q × K^T
   "How well does each query match each key?"

2. Scale:                    scores = scores / √d_k
   "Prevent large dot products from saturating softmax"

3. Apply causal mask:        scores[future] = -∞
   "Don't let tokens peek at future words"

4. Softmax:                  weights = softmax(scores)
   "Convert scores to probabilities (each row sums to 1)"

5. Weighted sum:             output = weights × V
   "Each position gets a mix of value vectors, weighted by attention"
```

**Why scale by √d_k?** Without scaling, when d_k is large, dot products have high variance (std ≈ √d_k), pushing softmax into saturated regions where gradients vanish. Dividing by √d_k brings the standard deviation back to ~1.

#### Multi-Head Attention

Instead of one big attention, we split into 4 parallel "heads":

```
d_model = 32, n_heads = 4 → d_k = 8 per head
```

Each head can learn to attend to different things:
- Head 1: syntactic relationships (subject-verb)
- Head 2: semantic similarity
- Head 3: positional proximity
- Head 4: some other pattern

The heads operate in parallel, then concatenate and project:

```python
# Project to Q, K, V (one big matrix contains all heads)
Q = W_Q(x)  # (batch, seq_len, 32)
K = W_K(x)  # (batch, seq_len, 32)
V = W_V(x)  # (batch, seq_len, 32)

# Reshape to split heads: (batch, 4, seq_len, 8)
# Apply attention to all heads simultaneously
# Concatenate heads: (batch, seq_len, 32)
# Final projection
output = W_O(concatenated)
```

#### Causal Mask

In a language model, token at position i must NOT see tokens at positions i+1, i+2, ... (that would be cheating — seeing the future).

```
Position:    0  1  2  3
Token 0:  [  1  0  0  0 ]  ← sees only itself
Token 1:  [  1  1  0  0 ]  ← sees tokens 0-1
Token 2:  [  1  1  1  0 ]  ← sees tokens 0-2
Token 3:  [  1  1  1  1 ]  ← sees tokens 0-3
```

This is implemented as a lower-triangular matrix (`torch.tril`), stored as a buffer that moves with the model to GPU.

### 4. Feed-Forward Network

**File:** `model/feedforward.py`

After attention gathers information from other positions, the FFN processes each position **independently**:

```
FFN(x) = Linear₂(GELU(Linear₁(x)))

Linear₁: 32 → 128  (expand to higher dimension)
GELU:    non-linear activation
Linear₂: 128 → 32  (compress back)
```

**Conceptual roles:**
- **Attention** = "communication" (tokens talk to each other)
- **FFN** = "computation" (each token thinks independently)

**Why GELU instead of ReLU?**

```
ReLU(x) = max(0, x)         — Harsh: kills ALL negative values
                               "Dead neuron" problem

GELU(x) = x × Φ(x)         — Smooth: small negatives get slight output
                               Used in GPT-2, BERT, modern transformers

Approximation: 0.5 × x × (1 + tanh(√(2/π) × (x + 0.044715 × x³)))
```

### 5. Transformer Block

**File:** `model/transformer_block.py`

Each block combines attention and FFN with two critical ingredients:

#### Pre-Norm Architecture (used in GPT-2, GPT-3, LLaMA)

```python
x = x + Attention(LayerNorm(x))    # Normalize → Attend → Add back
x = x + FFN(LayerNorm(x))          # Normalize → Compute → Add back
```

vs. Post-Norm (original 2017 Transformer):
```python
x = LayerNorm(x + Attention(x))    # Attend → Add → Normalize
x = LayerNorm(x + FFN(x))          # Compute → Add → Normalize
```

**Pre-Norm is better because:**
1. The residual path is "clean" — just addition, no normalization
2. Gradients flow directly through addition (no bottleneck)
3. More stable training, especially without careful LR warmup

#### Residual Connections

The `+ x` in `x = x + Attention(LayerNorm(x))` is a residual connection:

```
Forward:  output = x + f(x)
Gradient: ∂output/∂x = 1 + ∂f(x)/∂x
```

That "1 +" means the gradient is **always at least 1**, preventing vanishing gradients. The network only needs to learn the *delta* (correction), not the full transformation from scratch.

#### Layer Normalization (from scratch!)

**File:** `model/layernorm.py`

```
y = γ × (x - μ) / √(σ² + ε) + β
```

Normalizes across the feature dimension (d_model = 32) for each position independently.

**Why Layer Norm, not Batch Norm?**

| | Batch Norm | Layer Norm |
|---|---|---|
| Normalizes across | Batch dimension | Feature dimension |
| Depends on batch size | Yes (bad for small batches) | No (each sample independent) |
| Train vs eval behavior | Different (running stats) | Identical |
| Running statistics | Yes (complex) | No (simple) |

### 6. Full MiniGPT Model

**File:** `model/transformer.py`

Assembles everything:

```python
class MiniGPT(nn.Module):
    def __init__(self, config):
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = PositionalEncoding(max_seq_len, d_model)
        self.embed_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([TransformerBlock(...) for _ in range(n_layers)])
        self.final_norm = LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)
        # Weight tying!
        self.output_head.weight = self.token_embedding.weight
```

### 7. Weight Tying

The output projection shares its weight matrix with the token embedding (Press & Wolf, 2017).

**Why?**
- The embedding maps **words → vectors**
- The output maps **vectors → words**
- They should be approximate inverses
- Sharing them **saves 64,000 parameters** (that's 67% of our total!)
- Also acts as regularization and often improves performance

```python
self.output_head.weight = self.token_embedding.weight
```

---

## Training Deep Dive

**File:** `train.py`

### Cross-Entropy Loss

The standard loss function for language modeling. At each position, the model outputs a probability distribution over all 2000 words. Cross-entropy measures how far this is from the true answer.

```
Loss = -log(P(correct_word))

If model assigns 90% to correct word:  loss = 0.105
If model assigns 1% to correct word:   loss = 4.605
If model assigns 0.05% (random):       loss = 7.601
```

The model starts with loss ≈ 7.6 (ln(2000) = random guessing) and should converge to ≈ 4-5.

### Perplexity

```
Perplexity = e^(cross_entropy_loss)
```

Intuition: **"How many words is the model effectively choosing between?"**

| Perplexity | Meaning |
|---:|---|
| 1.0 | Perfect — always picks the right word |
| 50 | Choosing between ~50 candidates (good for small model!) |
| 2000 | Random guessing from entire vocabulary (untrained) |

### Learning Rate Schedule

```
LR
 │
 │     ╱‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾╲
 │    ╱                         ╲
 │   ╱                           ╲
 │  ╱                             ╲___
 │ ╱
 │╱
 └─────────────────────────────────── Step
   ↑ warmup ↑        cosine decay
   0       200                    5000
```

**Phase 1: Linear Warmup (steps 0-200)**
```
LR = max_lr × (step / warmup_steps)
```
Random initial weights cause wild gradients. Starting with a small LR lets the model find a reasonable region before going fast.

**Phase 2: Cosine Annealing (steps 200-5000)**
```
progress = (step - warmup) / (max_steps - warmup)
LR = min_lr + 0.5 × (max_lr - min_lr) × (1 + cos(π × progress))
```
Smooth decay from `3e-4` to `3e-5`. Cosine is gentler than step decay — no sudden disruptions.

### AdamW Optimizer

Adam with **decoupled weight decay**. Standard Adam applies weight decay inside the gradient update (mathematically incorrect). AdamW applies it outside (correct).

**Weight decay is only applied to 2D+ tensors** (weight matrices). NOT applied to:
- Biases (1D) — regularizing biases hurts performance
- LayerNorm gamma/beta — these should be free to scale/shift as needed

```python
# Separate parameter groups
decay_params = [p for p in params if p.dim() >= 2]      # Weight matrices
no_decay_params = [p for p in params if p.dim() < 2]     # Biases, norms
```

### Gradient Clipping

Transformers can experience sudden loss spikes causing "exploding gradients". Gradient clipping caps the total gradient norm:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

If ||gradients|| > 1.0, all gradients are scaled down proportionally. This prevents a single bad batch from destabilizing the entire model.

---

## Generation Deep Dive

**File:** `generate.py`

### Autoregressive Generation

The model generates text **one word at a time**:

```
Step 1: Input "ROMEO"     → Model predicts ":" with 85% confidence
Step 2: Input "ROMEO :"   → Model predicts "i"
Step 3: Input "ROMEO : i" → Model predicts "will"
...continue until max_tokens or <eos>
```

At each step, we only use the **last position's** output (the prediction for what comes next).

### Temperature Scaling

Divide logits by temperature T before softmax:

```
logits = [2.0, 1.0, 0.5]

T=0.5 (sharper):  probs ≈ [0.84, 0.11, 0.04]  ← Very confident
T=1.0 (normal):   probs ≈ [0.56, 0.21, 0.12]  ← Model's raw prediction
T=2.0 (flatter):  probs ≈ [0.42, 0.25, 0.20]  ← More random/creative
```

- **T → 0:** Approaches greedy (always picks top word). Safe but repetitive.
- **T = 1:** Use model's raw predictions as-is.
- **T → ∞:** Approaches uniform random. Creative but incoherent.

**T = 0.8** is a good default — slightly more confident than raw predictions.

### Top-K Sampling

Only consider the top K most likely words. Everything else gets 0 probability.

```
Vocabulary: ["the", "a", "cat", "dog", "xyz", "qqq", ...]
Probabilities: [0.3, 0.2, 0.15, 0.1, 0.001, 0.0001, ...]

With top_k=4:
Keep:  ["the", "a", "cat", "dog"]  → probs [0.3, 0.2, 0.15, 0.1]
Drop:  ["xyz", "qqq", ...] → probs [0, 0, ...]

Renormalize and sample from the 4 candidates.
```

This prevents the model from ever picking extremely unlikely words that would produce nonsense.

### Top-P (Nucleus) Sampling

Instead of a fixed K, dynamically choose the smallest set of words whose cumulative probability exceeds P:

```
If model is very confident:  [0.95, 0.03, 0.01, ...]
  → Only consider 1-2 words (enough to exceed P=0.9)

If model is uncertain:       [0.15, 0.12, 0.11, 0.10, ...]
  → Consider many words (need ~8 to exceed P=0.9)
```

Top-p **adapts** to the model's confidence, unlike top-k which always uses exactly K candidates.

---

## Data Pipeline

### Tokenization

**File:** `data/tokenizer.py`

We use simple **word-level tokenization** (no BPE, no SentencePiece). This keeps things transparent and educational.

```
Input:  "Hello, world! He'll be fine."
Tokens: ["hello", ",", "world", "!", "he'll", "be", "fine", "."]
```

The regex `r"[a-zA-Z']+|[.,!?;:\-\"]"` captures:
- Words (including contractions like "he'll")
- Punctuation as separate tokens

**Zipf's Law:** Word frequencies follow a power law. The top 2000 words cover ~95% of all tokens in Tiny Shakespeare. The remaining ~10,000 unique words are mapped to `<unk>`.

**Special tokens:**
| Token | Index | Purpose |
|---|---|---|
| `<pad>` | 0 | Padding for batching sequences of different lengths |
| `<unk>` | 1 | Out-of-vocabulary words (not in top 2000) |
| `<bos>` | 2 | Beginning of sequence marker |
| `<eos>` | 3 | End of sequence marker |

### Sliding Window Dataset

**File:** `data/dataset.py`

Given the full corpus as token IDs:
```
[45, 12, 7, 89, 3, 56, 102, 23, ...]
```

We create overlapping (input, target) pairs by sliding a window:

```
Window 0:  input = [45, 12, 7, 89]    target = [12, 7, 89, 3]
Window 1:  input = [12, 7, 89, 3]     target = [7, 89, 3, 56]
Window 2:  input = [7, 89, 3, 56]     target = [89, 3, 56, 102]
```

The target is the input **shifted right by 1**. At position i, the model must predict `target[i]` given `input[0:i+1]`.

**Train/val split:** First 90% of the corpus for training, last 10% for validation. Split by position (not random) to keep contiguous text and prevent future-information leakage.

---

## Web UI

**File:** `app.py`

Launch with `python app.py` and open `http://localhost:7860`.

### Training Tab
- Adjust hyperparameters: learning rate, max steps, batch size, warmup, log interval
- Start/Stop training with a button
- Live loss curve (train + validation plotted)
- Scrollable metrics table: Step, Loss, Perplexity, LR, Grad Norm, Time
- GPU monitoring: VRAM usage, device name, CUDA version

### Generation Tab
- Enter any prompt text
- Adjust temperature (0-2), top-k (0-200), max tokens (10-500) with sliders
- One-click generation with instant output
- Load/reload model from checkpoint

### Model Info Tab
- Parameter breakdown table with percentages
- Full PyTorch module tree visualization
- Current device/GPU info
- Checkpoint status (training step, validation loss)
- Complete configuration listing

---

## Configuration Reference

All hyperparameters are in `config.py` as a single dataclass:

### Model Architecture

| Parameter | Default | Description |
|---|---|---|
| `vocab_size` | 2000 | Number of words in vocabulary |
| `d_model` | 32 | Embedding dimension |
| `n_heads` | 4 | Number of attention heads (d_k = 8 per head) |
| `n_layers` | 2 | Number of transformer blocks |
| `d_ff` | 128 | Feed-forward inner dimension (4× expansion) |
| `max_seq_len` | 128 | Context window length |
| `dropout` | 0.1 | Dropout rate |

### Training

| Parameter | Default | Description |
|---|---|---|
| `batch_size` | 64 | Samples per gradient update |
| `learning_rate` | 3e-4 | Peak LR (after warmup) |
| `min_lr` | 3e-5 | Minimum LR (end of cosine decay) |
| `max_steps` | 5000 | Total training steps |
| `warmup_steps` | 200 | Linear warmup period |
| `grad_clip` | 1.0 | Maximum gradient norm |
| `weight_decay` | 0.01 | L2 regularization (weight matrices only) |
| `eval_interval` | 500 | Evaluate every N steps |
| `log_interval` | 100 | Log metrics every N steps |

### Generation

| Parameter | Default | Description |
|---|---|---|
| `temperature` | 0.8 | Sampling temperature |
| `top_k` | 40 | Top-k filtering |
| `max_gen_len` | 100 | Maximum tokens to generate |

### Comparison With Real Models

| | MiniGPT | GPT-2 Small | GPT-3 | LLaMA 7B |
|---|---:|---:|---:|---:|
| Parameters | 96K | 124M | 175B | 7B |
| d_model | 32 | 768 | 12,288 | 4,096 |
| Heads | 4 | 12 | 96 | 32 |
| Layers | 2 | 12 | 96 | 32 |
| d_ff | 128 | 3,072 | 49,152 | 11,008 |
| Context | 128 | 1,024 | 2,048 | 2,048 |
| Vocab | 2K | 50K | 50K | 32K |

Same architecture, just **1,000,000× smaller**.

---

## Concepts You'll Learn

By reading through this codebase, you'll understand:

- **Self-attention** — how tokens communicate with each other
- **Multi-head attention** — parallel representation learning
- **Causal masking** — preventing future information leakage
- **Layer normalization** — training stabilization (and why not batch norm)
- **Residual connections** — gradient highways through deep networks
- **Pre-norm vs post-norm** — modern vs original architecture
- **Positional encoding** — how models learn word order (learned + sinusoidal)
- **Weight tying** — parameter sharing between embedding and output
- **GELU activation** — smooth alternative to ReLU
- **Cross-entropy loss** — maximum likelihood for language modeling
- **Perplexity** — what it means and how to interpret it
- **Learning rate warmup** — why we start slow
- **Cosine annealing** — smooth learning rate decay
- **AdamW** — proper weight decay in Adam
- **Gradient clipping** — preventing training instability
- **Temperature scaling** — controlling generation randomness
- **Top-k and top-p sampling** — filtering unlikely tokens
- **Autoregressive generation** — predicting one token at a time
- **Tokenization** — converting text to numbers and back
- **Sliding window training** — creating (input, target) pairs for LM

---

## Further Reading

**Papers:**
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) — The original transformer paper
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2, Radford et al., 2019)
- [Layer Normalization](https://arxiv.org/abs/1607.06450) (Ba et al., 2016)
- [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859) (Press & Wolf, 2017) — Weight tying
- [Gaussian Error Linear Units](https://arxiv.org/abs/1606.08415) (Hendrycks & Gimpel, 2016) — GELU activation
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) (Loshchilov & Hutter, 2019) — AdamW

**Tutorials & Code:**
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Visual guide to transformers
- [minGPT by Andrej Karpathy](https://github.com/karpathy/minGPT) — Minimal GPT implementation
- [nanoGPT by Andrej Karpathy](https://github.com/karpathy/nanoGPT) — Simple, fast GPT training
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) — Line-by-line transformer walkthrough

**Videos:**
- [Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Andrej Karpathy's tutorial
- [Attention in transformers, visually explained](https://www.youtube.com/watch?v=eMlx5fFNoYc) — 3Blue1Brown

---

## License

MIT License — use this code however you want for learning, teaching, or building.
