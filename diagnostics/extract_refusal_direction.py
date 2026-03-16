#!/usr/bin/env python3
"""
EXP-019 Step 1 — Extract the refusal direction from Qwen3-8B.

For N HarmBench prompts, builds two inputs per prompt:
  - refusal:  prompt + refusal_completion
  - comply:   prompt + compliance_completion

Collects hidden states at the last token of the full sequence (prompt +
completion) from the specified layer, then computes:
  v_refusal = mean(h_refusal) - mean(h_comply)   (unit-normalized)

Usage (single layer):
  python diagnostics/extract_refusal_direction.py \\
      --layers 15 --n_samples 64 --out results/refusal_direction_layer15.pt

Usage (layer sweep — single model load, one .pt file per layer):
  python diagnostics/extract_refusal_direction.py \\
      --layers 0 5 10 15 20 25 30 35 \\
      --pool_mode first_n --n_completion_tokens 8

Usage (mean-pool over all completion tokens):
  python diagnostics/extract_refusal_direction.py \\
      --layers 15 --pool_mode all_completion
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latent_at.paths import get_model_path

# ── Paths ───────────────────────────────────────────────────────────────────
BEHAVIORS_PATH = "tasks/harmbench/data/harmbench_data/behavior_datasets/harmbench_behaviors_text_all.csv"
TARGETS_PATH = "tasks/harmbench/data/harmbench_data/optimizer_targets/text_targets.csv"
MODEL_NAME = "Qwen/Qwen3-8B"


# ── Chat template for Qwen3 ────────────────────────────────────────────────
def format_qwen3_input(prompt: str, completion: str) -> str:
    """Build a Qwen3 chat-template string: user prompt + assistant completion."""
    return (
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n{completion}<|im_end|>"
    )


def format_qwen3_prefix(prompt: str) -> str:
    """Return the prompt-only prefix (everything up to and including the
    assistant preamble), used to compute the token offset where the
    completion begins."""
    return (
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def load_contrastive_data(n_samples: int) -> list[dict]:
    """Load and merge HarmBench behaviors + targets, return list of dicts."""
    behaviors_df = pd.read_csv(BEHAVIORS_PATH)
    targets_df = pd.read_csv(TARGETS_PATH)
    merged = behaviors_df.merge(targets_df, on="BehaviorID", how="inner")
    standard = merged[merged["FunctionalCategory"] == "standard"].copy()
    standard = standard.dropna(subset=["llama_prompt_refusal", "optim_target"])

    if n_samples > len(standard):
        print(
            f"Warning: requested {n_samples} samples but only {len(standard)} available. Using all."
        )
        n_samples = len(standard)

    standard = standard.head(n_samples)

    triples = []
    for _, row in standard.iterrows():
        triples.append(
            {
                "prompt": row["Behavior"],
                "refusal": row["llama_prompt_refusal"],
                "comply": row["optim_target"],
                "behavior_id": row["BehaviorID"],
            }
        )
    return triples


def collect_hidden_states(
    model,
    tokenizer,
    triples: list[dict],
    layers: list[int],
    device: str = "cuda",
    pool_mode: str = "last",
    n_completion_tokens: int = 8,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """
    Collect hidden states from multiple layers simultaneously in a single forward pass.

    For each sample a single forward pass is run per completion type, with
    hooks on all requested layers at once — efficient for layer sweeps.

    pool_mode="last"           — hidden state at the final token of the full sequence.
    pool_mode="first_n"        — mean-pool over the first `n_completion_tokens` tokens
                                  of the completion (earliest refusal/comply divergence).
    pool_mode="all_completion" — mean-pool over all completion tokens.

    Returns:
        {layer_idx: (h_refusal [N, hidden_dim], h_comply [N, hidden_dim])}
    """
    h_refusal_lists: dict[int, list] = {l: [] for l in layers}
    h_comply_lists: dict[int, list] = {l: [] for l in layers}

    for i, triple in enumerate(
        tqdm(triples, desc=f"Collecting hidden states ({len(layers)} layer(s))")
    ):
        prompt = triple["prompt"]
        refusal_text = triple["refusal"]
        comply_text = triple["comply"]

        assert refusal_text != comply_text, f"Sample {i}: texts are identical!"

        # Compute the prefix length once per triple (shared across completions and layers)
        if pool_mode in ("first_n", "all_completion"):
            prefix_str = format_qwen3_prefix(prompt)
            prefix_ids = tokenizer(
                prefix_str, return_tensors="pt", add_special_tokens=False
            )["input_ids"]
            prefix_len = prefix_ids.shape[1]

        for completion_key, out_lists in [
            ("refusal", h_refusal_lists),
            ("comply", h_comply_lists),
        ]:
            full_str = format_qwen3_input(prompt, triple[completion_key])
            inputs = tokenizer(
                full_str, return_tensors="pt", add_special_tokens=False
            ).to(device)

            # Register one hook per requested layer; all fire in the same forward pass
            caches: dict[int, list] = {l: [] for l in layers}
            handles = []
            for l in layers:
                cache = caches[l]

                def hook_fn(module, input, output, _cache=cache):
                    if isinstance(output, tuple):
                        _cache.append(output[0].detach())
                    else:
                        _cache.append(output.detach())

                handles.append(model.model.layers[l].register_forward_hook(hook_fn))

            with torch.no_grad():
                model(**inputs)

            for handle in handles:
                handle.remove()

            # Extract the pooled hidden state from each layer
            for l in layers:
                hidden = caches[l][0][0].float().cpu()  # [seq_len, hidden_dim]

                if pool_mode == "last":
                    h = hidden[-1, :]
                elif pool_mode == "first_n":
                    seq_len = hidden.shape[0]
                    start = prefix_len
                    end = min(start + n_completion_tokens, seq_len)
                    if end <= start:
                        raise ValueError(
                            f"Sample {i} ({completion_key}): completion has 0 tokens "
                            f"after prefix (seq_len={seq_len}, prefix_len={prefix_len})"
                        )
                    h = hidden[start:end, :].mean(dim=0)
                elif pool_mode == "all_completion":
                    seq_len = hidden.shape[0]
                    start = prefix_len
                    if start >= seq_len:
                        raise ValueError(
                            f"Sample {i} ({completion_key}): completion has 0 tokens "
                            f"after prefix (seq_len={seq_len}, prefix_len={prefix_len})"
                        )
                    h = hidden[start:, :].mean(dim=0)
                else:
                    raise ValueError(f"Unknown pool_mode: {pool_mode!r}")

                out_lists[l].append(h)

    return {
        l: (
            torch.stack(h_refusal_lists[l]),
            torch.stack(h_comply_lists[l]),
        )
        for l in layers
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract refusal direction from Qwen3-8B"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=15,
        help="Single layer to extract from (overridden by --layers)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Multiple layers to extract in one pass (overrides --layer). "
        "One .pt file is saved per layer.",
    )
    parser.add_argument(
        "--n_samples", type=int, default=64, help="Number of prompt pairs"
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path (single-layer only). Ignored when --layers is used.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size (unused, sequential for now)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--pool_mode",
        type=str,
        default="last",
        choices=["last", "first_n", "all_completion"],
        help="How to pool hidden states: 'last' (final token), 'first_n' (mean of first N completion tokens), or 'all_completion' (mean over all completion tokens)",
    )
    parser.add_argument(
        "--n_completion_tokens",
        type=int,
        default=8,
        help="Number of completion tokens to mean-pool (only used when --pool_mode=first_n)",
    )
    args = parser.parse_args()

    # ── Determine layer list and output paths ───────────────────────────
    if args.layers is not None:
        layers = sorted(set(args.layers))
    else:
        layers = [args.layer]

    def auto_out_path(layer: int) -> Path:
        if args.pool_mode == "last":
            return Path(f"results/refusal_direction_layer{layer}.pt")
        elif args.pool_mode == "all_completion":
            return Path(f"results/refusal_direction_layer{layer}_all_completion.pt")
        return Path(
            f"results/refusal_direction_layer{layer}_{args.pool_mode}{args.n_completion_tokens}.pt"
        )

    if len(layers) == 1 and args.out is not None:
        out_paths = {layers[0]: Path(args.out)}
    else:
        out_paths = {l: auto_out_path(l) for l in layers}

    # ── Load model ──────────────────────────────────────────────────────
    print(f"Loading model: {MODEL_NAME}")
    model_path = get_model_path(MODEL_NAME)
    print(f"  Resolved to: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()

    print(f"\nModel architecture check:")
    print(f"  model.config.num_hidden_layers = {model.config.num_hidden_layers}")
    print(f"  model.config.hidden_size = {model.config.hidden_size}")
    print(f"  Target layers: {layers}")
    for l in layers:
        print(f"  Layer {l} type: {type(model.model.layers[l]).__name__}")

    # ── Load data ───────────────────────────────────────────────────────
    triples = load_contrastive_data(args.n_samples)
    print(f"\nLoaded {len(triples)} contrastive triples")

    # ── Collect hidden states (all layers, single model pass per sample) ─
    print(f"\nExtraction mode: pool_mode={args.pool_mode!r}", end="")
    if args.pool_mode == "first_n":
        print(f", n_completion_tokens={args.n_completion_tokens}")
    else:
        print()
    t0 = time.time()
    layer_results = collect_hidden_states(
        model,
        tokenizer,
        triples,
        layers,
        device=args.device,
        pool_mode=args.pool_mode,
        n_completion_tokens=args.n_completion_tokens,
    )
    elapsed = time.time() - t0
    print(f"\nCollected hidden states in {elapsed:.1f}s ({len(layers)} layer(s))")

    # ── Per-layer: compute direction, print stats, save ─────────────────
    for layer in layers:
        h_refusal, h_comply = layer_results[layer]
        out_path = out_paths[layer]

        print(f"\n{'='*60}")
        print(f"LAYER {layer}")
        print(f"{'='*60}")
        print(f"  h_refusal shape: {h_refusal.shape}")
        print(f"  h_comply  shape: {h_comply.shape}")

        mean_refusal = h_refusal.mean(dim=0)
        mean_comply = h_comply.mean(dim=0)
        v_refusal = mean_refusal - mean_comply
        v_norm = torch.norm(v_refusal).item()
        v_refusal_unit = v_refusal / torch.norm(v_refusal)

        proj_refusal = h_refusal @ v_refusal_unit  # [N]
        proj_comply = h_comply @ v_refusal_unit  # [N]
        gap = proj_refusal.mean().item() - proj_comply.mean().item()
        std_r = proj_refusal.std().item()
        std_c = proj_comply.std().item()
        cohens_d = gap / ((std_r + std_c) / 2) if (std_r + std_c) > 0 else float("nan")
        threshold = (proj_refusal.mean().item() + proj_comply.mean().item()) / 2
        acc = (
            (proj_refusal > threshold).sum() + (proj_comply <= threshold).sum()
        ).item() / (2 * len(triples))

        print(f"\n  Direction stats:")
        print(f"    ||mean_refusal - mean_comply|| = {v_norm:.4f}")
        print(f"    Gap:                            {gap:.4f}")
        print(f"    Cohen's d:                      {cohens_d:.2f}")
        print(f"    Linear accuracy (midpoint):     {acc:.1%}")

        # ── Save ────────────────────────────────────────────────────────
        out_path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            # Generic keys (shared across all direction extraction scripts)
            "v_direction": v_refusal_unit,
            "v_direction_unnormalized": v_refusal,
            "direction_name": "refusal",
            "h_positive": h_refusal,  # positive class (refusal)
            "h_negative": h_comply,  # negative class (comply)
            # Legacy aliases for backward compatibility
            "v_refusal": v_refusal_unit,
            "v_refusal_unnormalized": v_refusal,
            "h_refusal": h_refusal,
            "h_comply": h_comply,
            # Shared metadata
            "norm": v_norm,
            "layer": layer,
            "n_samples": len(triples),
            "model_name": MODEL_NAME,
            "hidden_dim": model.config.hidden_size,
            "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
            "pool_mode": args.pool_mode,
            "n_completion_tokens": (
                args.n_completion_tokens if args.pool_mode == "first_n" else None
            ),
        }
        torch.save(save_dict, out_path)
        print(f"  Saved → {out_path}")

    print(f"\n{'='*60}")
    print(f"Done. {len(layers)} layer(s) processed.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
