#!/usr/bin/env python3
"""Pre-download all HuggingFace datasets used by evaluation.

Run this on a login node (with internet access) BEFORE submitting
offline SLURM evaluation jobs:

    python cache_eval_datasets.py

The datasets will be cached in the standard HF cache directory
(typically ~/.cache/huggingface/datasets or $HF_HOME/datasets).
Once cached, offline evaluation jobs can load them without network access.
"""

import datasets


EVAL_DATASETS = [
    # Utility benchmarks (tiny versions, used by default)
    {"path": "tinyBenchmarks/tinyMMLU", "split": "test"},
    {"path": "tinyBenchmarks/tinyHellaswag", "split": "test"},
    {"path": "tinyBenchmarks/tinyWinogrande", "split": "test"},
    # Non-tiny datasets used by SciQ and Lambada
    {"path": "allenai/sciq", "split": "test"},
    {"path": "lambada", "split": "test"},
]


def main():
    for ds_info in EVAL_DATASETS:
        path = ds_info["path"]
        split = ds_info["split"]
        print(f"Caching {path} (split={split})...")
        try:
            ds = datasets.load_dataset(path, split=split)
            print(f"  ✓ {path}: {len(ds)} examples cached")
        except Exception as e:
            print(f"  ✗ {path}: {e}")

    print("\nAll datasets cached. You can now run evaluation jobs offline.")


if __name__ == "__main__":
    main()
