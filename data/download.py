"""
data/download.py - Download the Tiny Shakespeare Dataset
=========================================================

The Tiny Shakespeare dataset is a ~1.1MB text file containing all of
Shakespeare's works concatenated. It has roughly:
  - 40,000 lines
  - 900,000 characters
  - ~25,000 unique words (but top 2000 cover ~95% of occurrences)

This is the standard "hello world" dataset for language modeling,
popularized by Andrej Karpathy's char-rnn project.

Source: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
"""

import os
import urllib.request


SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)


def download_shakespeare(save_path: str) -> str:
    """
    Download Tiny Shakespeare to the given path.

    Returns the path to the downloaded file.
    Skips download if the file already exists.
    """
    # Create parent directory if needed
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path):
        size_kb = os.path.getsize(save_path) / 1024
        print(f"[download] Dataset already exists at {save_path} ({size_kb:.1f} KB)")
        return save_path

    print(f"[download] Downloading Tiny Shakespeare...")
    print(f"[download] URL: {SHAKESPEARE_URL}")

    urllib.request.urlretrieve(SHAKESPEARE_URL, save_path)

    size_kb = os.path.getsize(save_path) / 1024
    print(f"[download] Saved to {save_path} ({size_kb:.1f} KB)")

    # Print a preview
    with open(save_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    print(f"[download] Total lines: {len(lines):,}")
    print(f"[download] First 3 lines:")
    for line in lines[:3]:
        print(f"  | {line.rstrip()}")

    return save_path


if __name__ == "__main__":
    download_shakespeare(os.path.join("data", "input.txt"))
