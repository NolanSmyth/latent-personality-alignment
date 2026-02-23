"""
Cache the Alpaca dataset for offline SFT recovery training.

Run this on a login node with internet access BEFORE submitting offline jobs:
    python cache_alpaca_dataset.py
"""

from datasets import load_dataset


def main():
    print("Caching tatsu-lab/alpaca dataset...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    print(f"  Cached {len(ds)} examples")
    print(f"  Columns: {ds.column_names}")
    print(f"  Sample: {ds[0]}")
    print("Done. Dataset is now cached in HF datasets cache.")


if __name__ == "__main__":
    main()
