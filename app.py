"""
app.py - Gradio Web UI for MiniGPT
====================================

A web interface for training and interacting with MiniGPT.

Features:
  - Training tab: live loss curve, metrics table, GPU monitoring, stop button
  - Generation tab: prompt input, temperature/top-k sliders, text output
  - Model Info tab: parameter breakdown, architecture, device info

Run with:
    python app.py
"""

import math
import os
import threading
import time
from dataclasses import asdict

import gradio as gr
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (required for threading)
import matplotlib.pyplot as plt
import torch

from config import TransformerConfig
from model.transformer import MiniGPT
from data.tokenizer import WordTokenizer
from data.download import download_shakespeare
from train import train, load_checkpoint
from generate import generate


# ═══════════════════════════════════════════════════════════════════
# SHARED STATE
# ═══════════════════════════════════════════════════════════════════
# Thread safety: training_log is append-only from the training thread
# and read (sliced) by the UI polling function. Python's GIL makes
# list.append() and list[:] atomic for this pattern.

training_log: list[dict] = []
training_thread: threading.Thread | None = None
stop_event = threading.Event()
is_training = False

current_model: MiniGPT | None = None
current_tokenizer: WordTokenizer | None = None
current_device: torch.device | None = None


# ═══════════════════════════════════════════════════════════════════
# GPU UTILITIES
# ═══════════════════════════════════════════════════════════════════

def get_device_info() -> str:
    """Return formatted device/GPU information."""
    if not torch.cuda.is_available():
        return "No CUDA GPU detected. Using CPU."

    props = torch.cuda.get_device_properties(0)
    allocated = torch.cuda.memory_allocated(0) / 1024**2
    reserved = torch.cuda.memory_reserved(0) / 1024**2
    total = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / 1024**2

    lines = [
        f"GPU: {props.name}",
        f"VRAM: {total:.0f} MB total | {allocated:.1f} MB allocated | {reserved:.1f} MB reserved",
        f"Free: {total - reserved:.0f} MB",
        f"CUDA: {torch.version.cuda}",
    ]

    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        lines.append(f"GPU Utilization: {util.gpu}% | Memory Utilization: {util.memory}%")
        pynvml.nvmlShutdown()
    except Exception:
        pass

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# TRAINING TAB
# ═══════════════════════════════════════════════════════════════════

def training_callback(metrics: dict):
    """Called by the training loop from the background thread."""
    training_log.append(metrics)


def start_training(learning_rate, max_steps, batch_size, warmup_steps, log_interval):
    """Launch training in a background thread."""
    global training_thread, is_training, training_log

    if is_training:
        return "Training is already in progress!", gr.update(), gr.update(), gr.update()

    # Reset state
    training_log.clear()
    stop_event.clear()

    # Build config
    config = TransformerConfig(
        learning_rate=float(learning_rate),
        max_steps=int(max_steps),
        batch_size=int(batch_size),
        warmup_steps=int(warmup_steps),
        log_interval=int(log_interval),
    )

    def run_training():
        global is_training
        is_training = True
        try:
            train(config, callback=training_callback, stop_event=stop_event)
        except Exception as e:
            training_log.append({"type": "error", "message": str(e)})
        finally:
            is_training = False

    training_thread = threading.Thread(target=run_training, daemon=True)
    training_thread.start()

    return (
        "Training started... Preparing data and model...",
        gr.update(),
        gr.update(),
        get_device_info(),
    )


def stop_training():
    """Signal the training loop to stop."""
    if not is_training:
        return "No training in progress."
    stop_event.set()
    return "Stop signal sent. Training will halt after current step."


def poll_training_progress():
    """Called by gr.Timer to update the UI with training progress."""
    if not training_log:
        if is_training:
            status = "Preparing data and model..."
        else:
            status = "Ready. Click 'Start Training' to begin."
        return status, gr.update(), gr.update(), get_device_info()

    latest = training_log[-1]

    # Handle error
    if latest.get("type") == "error":
        return f"ERROR: {latest['message']}", gr.update(), gr.update(), get_device_info()

    # Handle completion
    if latest.get("type") == "complete":
        status = (
            f"Training complete! "
            f"Steps: {latest['step']} | "
            f"Val Loss: {latest.get('val_loss', 0):.4f} | "
            f"Val PPL: {latest.get('val_perplexity', 0):.1f} | "
            f"Time: {latest.get('elapsed', 0):.1f}s"
        )
    else:
        # In progress
        step = latest.get("step", 0)
        max_steps = latest.get("max_steps", 1)
        pct = step / max_steps * 100
        loss_str = f"{latest['loss']:.4f}" if "loss" in latest else f"{latest.get('val_loss', 0):.4f}"
        ppl_str = f"{latest.get('perplexity', latest.get('val_perplexity', 0)):.1f}"
        status = (
            f"Step {step}/{max_steps} ({pct:.1f}%) | "
            f"Loss: {loss_str} | PPL: {ppl_str}"
        )
        if "lr" in latest:
            status += f" | LR: {latest['lr']:.2e}"

    # Build table from log entries
    log_entries = [m for m in training_log if m.get("type") == "log"]
    # Show last 200 entries to keep UI responsive
    recent = log_entries[-200:]
    table_rows = [
        [
            m["step"],
            round(m["loss"], 4),
            round(m["perplexity"], 1),
            f"{m['lr']:.2e}",
            round(m.get("grad_norm", 0), 2),
            f"{m['elapsed']:.1f}s",
        ]
        for m in recent
    ]

    # Build loss plot
    plot_fig = None
    if len(log_entries) >= 2:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        steps = [m["step"] for m in log_entries]
        losses = [m["loss"] for m in log_entries]
        ax.plot(steps, losses, color="#2563eb", linewidth=1.5, label="Train Loss")

        # Add eval points if available
        eval_entries = [m for m in training_log if m.get("type") == "eval"]
        if eval_entries:
            eval_steps = [m["step"] for m in eval_entries]
            eval_losses = [m["val_loss"] for m in eval_entries]
            ax.plot(eval_steps, eval_losses, "ro-", markersize=6, linewidth=1, label="Val Loss")
            ax.legend()

        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training Progress")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_fig = fig

    return status, table_rows, plot_fig, get_device_info()


# ═══════════════════════════════════════════════════════════════════
# GENERATION TAB
# ═══════════════════════════════════════════════════════════════════

def load_model_for_generation():
    """Load the trained model and tokenizer from checkpoint."""
    global current_model, current_tokenizer, current_device

    config = TransformerConfig()

    if not os.path.exists(config.checkpoint_path):
        return "No checkpoint found. Train the model first!"

    current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_model, _ = load_checkpoint(config, current_device)
    current_model.eval()

    current_tokenizer = WordTokenizer()
    if not os.path.exists(config.vocab_path):
        return "No vocabulary file found. Train the model first!"
    current_tokenizer.load(config.vocab_path)

    return f"Model loaded on {current_device}. Ready to generate!"


def generate_text(prompt, temperature, top_k, max_tokens):
    """Generate text from the prompt."""
    global current_model, current_tokenizer, current_device

    if current_model is None:
        status = load_model_for_generation()
        if current_model is None:
            return status

    if not prompt.strip():
        return "Please enter a prompt."

    text = generate(
        model=current_model,
        tokenizer=current_tokenizer,
        prompt=prompt,
        max_tokens=int(max_tokens),
        temperature=float(temperature),
        top_k=int(top_k),
        device=current_device,
    )

    return text


# ═══════════════════════════════════════════════════════════════════
# MODEL INFO TAB
# ═══════════════════════════════════════════════════════════════════

def get_model_info():
    """Return model architecture details and parameter breakdown."""
    config = TransformerConfig()

    model = MiniGPT(config)

    # Parameter breakdown
    components = {}
    components["Token Embedding (tied)"] = model.token_embedding.weight.numel()
    components["Positional Encoding"] = sum(p.numel() for p in model.position_encoding.parameters())
    components[f"Transformer Blocks (x{config.n_layers})"] = sum(p.numel() for p in model.blocks.parameters())
    components["Final LayerNorm"] = sum(p.numel() for p in model.final_norm.parameters())
    if model.output_head.bias is not None:
        components["Output Bias"] = model.output_head.bias.numel()

    total = sum(components.values())

    breakdown_rows = []
    for name, count in components.items():
        pct = count / total * 100
        breakdown_rows.append([name, f"{count:,}", f"{pct:.1f}%"])
    breakdown_rows.append(["TOTAL", f"{total:,}", "100.0%"])

    # Checkpoint status
    ckpt_status = "No checkpoint found."
    if os.path.exists(config.checkpoint_path):
        try:
            ckpt = torch.load(config.checkpoint_path, map_location="cpu", weights_only=False)
            val_loss = ckpt.get("val_loss", "N/A")
            if isinstance(val_loss, float):
                val_loss = f"{val_loss:.4f}"
            ckpt_status = (
                f"Checkpoint: {config.checkpoint_path}\n"
                f"Step: {ckpt.get('step', 'unknown')}\n"
                f"Val loss: {val_loss}"
            )
        except Exception as e:
            ckpt_status = f"Error reading checkpoint: {e}"

    # Architecture
    arch_text = str(model)

    # Config
    config_lines = [
        f"vocab_size:   {config.vocab_size}",
        f"d_model:      {config.d_model}",
        f"n_heads:      {config.n_heads}",
        f"d_k:          {config.d_k} (per head)",
        f"n_layers:     {config.n_layers}",
        f"d_ff:         {config.d_ff}",
        f"max_seq_len:  {config.max_seq_len}",
        f"dropout:      {config.dropout}",
        f"",
        f"batch_size:   {config.batch_size}",
        f"learning_rate: {config.learning_rate}",
        f"max_steps:    {config.max_steps}",
        f"warmup_steps: {config.warmup_steps}",
    ]
    config_text = "\n".join(config_lines)

    # Device
    device_text = get_device_info()

    return breakdown_rows, arch_text, ckpt_status, config_text, device_text


# ═══════════════════════════════════════════════════════════════════
# GRADIO UI LAYOUT
# ═══════════════════════════════════════════════════════════════════

def build_ui():
    with gr.Blocks(title="MiniGPT - Educational Transformer") as demo:

        gr.Markdown(
            "# MiniGPT: Educational Transformer (~96K params)\n"
            "Train a tiny GPT on Shakespeare and generate text. "
            "Every component is built from scratch for learning."
        )

        # ── Tab 1: Training ──────────────────────────────────────
        with gr.Tab("Training"):
            with gr.Row():
                # Left: controls
                with gr.Column(scale=1):
                    gr.Markdown("### Hyperparameters")
                    lr_input = gr.Number(
                        label="Learning Rate", value=3e-4,
                        minimum=1e-5, maximum=1e-2,
                    )
                    max_steps_input = gr.Number(
                        label="Max Steps", value=5000,
                        minimum=100, maximum=50000,
                    )
                    batch_size_input = gr.Number(
                        label="Batch Size", value=64,
                        minimum=8, maximum=256,
                    )
                    warmup_input = gr.Number(
                        label="Warmup Steps", value=200,
                        minimum=0, maximum=2000,
                    )
                    log_interval_input = gr.Number(
                        label="Log Every N Steps", value=50,
                        minimum=5, maximum=500,
                    )

                    with gr.Row():
                        train_btn = gr.Button(
                            "Start Training", variant="primary", size="lg"
                        )
                        stop_btn = gr.Button(
                            "Stop", variant="stop", size="lg"
                        )

                # Right: outputs
                with gr.Column(scale=2):
                    status_text = gr.Textbox(
                        label="Status",
                        interactive=False,
                        lines=2,
                        elem_classes=["status-box"],
                        value="Ready. Click 'Start Training' to begin.",
                    )
                    loss_plot = gr.Plot(label="Training Loss Curve")
                    log_table = gr.Dataframe(
                        headers=["Step", "Loss", "Perplexity", "LR", "Grad Norm", "Time"],
                        label="Training Log",
                        interactive=False,
                        wrap=True,
                    )

            gpu_info = gr.Textbox(
                label="Device Info",
                interactive=False,
                lines=3,
                elem_classes=["gpu-box"],
                value=get_device_info(),
            )

            # Polling timer (starts inactive)
            timer = gr.Timer(value=2, active=False)

            # Wire: Start Training
            train_btn.click(
                fn=start_training,
                inputs=[lr_input, max_steps_input, batch_size_input,
                        warmup_input, log_interval_input],
                outputs=[status_text, log_table, loss_plot, gpu_info],
            ).then(
                fn=lambda: gr.Timer(active=True),
                outputs=[timer],
            )

            # Wire: Stop Training
            stop_btn.click(fn=stop_training, outputs=[status_text])

            # Wire: Timer polls progress
            timer.tick(
                fn=poll_training_progress,
                outputs=[status_text, log_table, loss_plot, gpu_info],
            )

        # ── Tab 2: Generation ────────────────────────────────────
        with gr.Tab("Generation"):
            with gr.Row():
                # Left: controls
                with gr.Column(scale=1):
                    prompt_input = gr.Textbox(
                        label="Prompt",
                        placeholder="ROMEO:",
                        lines=3,
                        value="ROMEO:",
                    )
                    temperature_slider = gr.Slider(
                        label="Temperature",
                        minimum=0.0, maximum=2.0, step=0.05, value=0.8,
                        info="Lower = more confident, Higher = more random",
                    )
                    top_k_slider = gr.Slider(
                        label="Top-K",
                        minimum=0, maximum=200, step=1, value=40,
                        info="Only sample from top K candidates (0 = no filter)",
                    )
                    max_tokens_slider = gr.Slider(
                        label="Max Tokens",
                        minimum=10, maximum=500, step=10, value=100,
                    )

                    with gr.Row():
                        load_btn = gr.Button("Load Model")
                        gen_btn = gr.Button("Generate", variant="primary", size="lg")

                    load_status = gr.Textbox(
                        label="Model Status",
                        interactive=False,
                        lines=2,
                    )

                # Right: output
                with gr.Column(scale=2):
                    output_text = gr.Textbox(
                        label="Generated Text",
                        lines=18,
                        interactive=False,
                    )

            load_btn.click(fn=load_model_for_generation, outputs=[load_status])
            gen_btn.click(
                fn=generate_text,
                inputs=[prompt_input, temperature_slider,
                        top_k_slider, max_tokens_slider],
                outputs=[output_text],
            )

        # ── Tab 3: Model Info ────────────────────────────────────
        with gr.Tab("Model Info"):
            info_btn = gr.Button("Refresh Info", size="sm")

            with gr.Row():
                with gr.Column():
                    param_table = gr.Dataframe(
                        headers=["Component", "Parameters", "Percentage"],
                        label="Parameter Breakdown",
                        interactive=False,
                    )
                    config_text = gr.Textbox(
                        label="Configuration",
                        lines=12,
                        interactive=False,
                    )
                with gr.Column():
                    device_text = gr.Textbox(
                        label="Device Info",
                        lines=4,
                        interactive=False,
                    )
                    ckpt_text = gr.Textbox(
                        label="Checkpoint Status",
                        lines=4,
                        interactive=False,
                    )
                    arch_text = gr.Textbox(
                        label="Architecture (PyTorch Module Tree)",
                        lines=22,
                        interactive=False,
                    )

            info_btn.click(
                fn=get_model_info,
                outputs=[param_table, arch_text, ckpt_text, config_text, device_text],
            )

            # Auto-load on page open
            demo.load(
                fn=get_model_info,
                outputs=[param_table, arch_text, ckpt_text, config_text, device_text],
            )

    return demo


# ═══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"\nDevice: {'CUDA - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Gradio: {gr.__version__}\n")

    demo = build_ui()
    demo.launch(share=False, theme=gr.themes.Soft())
