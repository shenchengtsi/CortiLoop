"""
Download LongMemEval official dataset from HuggingFace.

Three variants:
  - oracle:  Only answer-relevant sessions (~small, for debugging)
  - s:       ~40 sessions per question, ~115K tokens (recommended)
  - m:       ~500 sessions per question, ~1.5M tokens (stress test)

Usage:
    python -m benchmarks.download_longmemeval              # download all
    python -m benchmarks.download_longmemeval --variant s  # download S only
"""

from __future__ import annotations

import argparse
import urllib.request
import sys
from pathlib import Path

BASE_URL = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main"

VARIANTS = {
    "oracle": "longmemeval_oracle.json",
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
}

DATA_DIR = Path(__file__).parent / "data"


def download(variant: str, force: bool = False) -> Path:
    """Download a single variant. Returns path to the downloaded file."""
    filename = VARIANTS[variant]
    dest = DATA_DIR / filename

    if dest.exists() and not force:
        print(f"  [skip] {dest} already exists (use --force to re-download)")
        return dest

    url = f"{BASE_URL}/{filename}"
    print(f"  Downloading {variant} → {dest}")
    print(f"    URL: {url}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        urllib.request.urlretrieve(url, dest)
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"    Done ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"    FAILED: {e}", file=sys.stderr)
        if dest.exists():
            dest.unlink()
        raise

    return dest


def main():
    parser = argparse.ArgumentParser(description="Download LongMemEval dataset")
    parser.add_argument(
        "--variant", choices=list(VARIANTS.keys()), default=None,
        help="Which variant to download (default: all)",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if exists")
    args = parser.parse_args()

    variants = [args.variant] if args.variant else list(VARIANTS.keys())

    print(f"LongMemEval dataset download → {DATA_DIR}/")
    print()

    for v in variants:
        download(v, force=args.force)

    print()
    print("Done. Run benchmark with:")
    print("  python -m benchmarks.longmemeval_official --variant s")


if __name__ == "__main__":
    main()
