"""
run.py - CLI Entry Point for MiniGPT
======================================

Usage:
    python run.py download          Download the Tiny Shakespeare dataset
    python run.py train             Train the model
    python run.py generate "ROMEO:" Generate text from a prompt
    python run.py info              Print model architecture and parameter count

Examples:
    python run.py download
    python run.py train
    python run.py generate "To be or not to be"
    python run.py generate "ROMEO:" --temperature 1.0 --top_k 50 --max_tokens 200
    python run.py info
"""

import argparse
import os
import sys
import torch

from config import TransformerConfig


def cmd_download(args, config):
    """Download the Tiny Shakespeare dataset."""
    from data.download import download_shakespeare
    download_shakespeare(config.data_path)


def cmd_train(args, config):
    """Train the model."""
    from train import train
    train(config)


def cmd_generate(args, config):
    """Generate text from a prompt."""
    from train import load_checkpoint
    from generate import generate
    from data.tokenizer import WordTokenizer

    # Check that model exists
    if not os.path.exists(config.checkpoint_path):
        print(f"Error: No checkpoint found at {config.checkpoint_path}")
        print("Run 'python run.py train' first.")
        sys.exit(1)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, saved_config = load_checkpoint(config, device)
    model.eval()

    # Load tokenizer
    tokenizer = WordTokenizer()
    if not os.path.exists(config.vocab_path):
        print(f"Error: No vocabulary found at {config.vocab_path}")
        print("Run 'python run.py train' first.")
        sys.exit(1)
    tokenizer.load(config.vocab_path)

    # Generate
    prompt = args.prompt
    print(f"\nPrompt: {prompt}")
    print(f"Temperature: {args.temperature}, Top-k: {args.top_k}")
    print("-" * 50)

    text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )

    print(text)
    print("-" * 50)


def cmd_info(args, config):
    """Print model architecture and parameter count."""
    from model.transformer import MiniGPT

    print(f"\n{'='*55}")
    print(f"  MiniGPT Configuration")
    print(f"{'='*55}")
    print(f"  vocab_size:   {config.vocab_size}")
    print(f"  d_model:      {config.d_model}")
    print(f"  n_heads:      {config.n_heads}")
    print(f"  d_k:          {config.d_k} (per head)")
    print(f"  n_layers:     {config.n_layers}")
    print(f"  d_ff:         {config.d_ff}")
    print(f"  max_seq_len:  {config.max_seq_len}")
    print(f"  dropout:      {config.dropout}")

    model = MiniGPT(config)
    model.count_parameters()

    # Print architecture
    print(f"\n{'='*55}")
    print(f"  Architecture (PyTorch Module Tree)")
    print(f"{'='*55}")
    print(model)


def main():
    parser = argparse.ArgumentParser(
        description="MiniGPT: A ~100K parameter educational transformer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  download    Download the Tiny Shakespeare dataset
  train       Train the model from scratch
  generate    Generate text given a prompt
  info        Print model architecture and parameter count

Examples:
  python run.py download
  python run.py train
  python run.py generate "ROMEO:"
  python run.py generate "To be" --temperature 1.2 --top_k 50
  python run.py info
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Download command
    subparsers.add_parser("download", help="Download Tiny Shakespeare dataset")

    # Train command
    subparsers.add_parser("train", help="Train the model")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate text")
    gen_parser.add_argument("prompt", type=str, help="Starting text for generation")
    gen_parser.add_argument(
        "--temperature", type=float, default=0.8,
        help="Sampling temperature (default: 0.8)"
    )
    gen_parser.add_argument(
        "--top_k", type=int, default=40,
        help="Top-k sampling (default: 40)"
    )
    gen_parser.add_argument(
        "--max_tokens", type=int, default=100,
        help="Maximum tokens to generate (default: 100)"
    )

    # Info command
    subparsers.add_parser("info", help="Print model info and parameter count")

    args = parser.parse_args()
    config = TransformerConfig()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "download": cmd_download,
        "train": cmd_train,
        "generate": cmd_generate,
        "info": cmd_info,
    }

    commands[args.command](args, config)


if __name__ == "__main__":
    main()
