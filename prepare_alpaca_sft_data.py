"""
Convert the cached Alpaca dataset into a CSV compatible with lat_training.py's load_data().

Produces data/alpaca_sft/benign_alpaca.csv with columns: prompt, response, refusal
Run on a login node (needs the HF datasets cache from cache_alpaca_dataset.py).

Usage:
    python prepare_alpaca_sft_data.py
"""

import os
import csv
from datasets import load_dataset


def main():
    print("Loading cached tatsu-lab/alpaca dataset...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    print(f"Loaded {len(ds)} examples")

    out_dir = os.path.join("data", "alpaca_sft")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "benign_alpaca.csv")

    print(f"Writing to {out_path}...")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt", "response", "refusal"])
        for ex in ds:
            instruction = ex.get("instruction", "")
            inp = ex.get("input", "")
            output = ex.get("output", "")
            # Combine instruction + input into prompt
            if inp and inp.strip():
                prompt = f"{instruction}\n\n{inp}"
            else:
                prompt = instruction
            # refusal is a dummy column — SFT loss only uses def_tokens (response)
            writer.writerow([prompt, output, "N/A"])

    print(f"Done. Wrote {len(ds)} rows to {out_path}")


if __name__ == "__main__":
    main()
