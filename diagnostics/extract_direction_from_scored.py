#!/usr/bin/env python3
"""
Pipeline Step 3 — Extract a persona direction from score-filtered responses.

Reads the scored JSONL produced by score_responses.py and partitions responses
into two classes based on both the trait score AND the system-prompt label:

  high-trait:  trait_score >= threshold  AND  label == "pos"
  low-trait:   trait_score <= threshold  AND  label == "neg"

For each qualifying response, runs a single forward pass over the full
(system + user + completion) sequence to collect hidden states at the requested
layer(s), mean-pooled over all response tokens.

The direction is:
  v = mean(h_high) - mean(h_low)   (unit-normalised)

All responses (including those filtered out) are preserved in the scored JSONL;
this script only selects which ones contribute to the direction computation.

Usage (single layer):
  python diagnostics/extract_direction_from_scored.py \\
      --scored results/scored_responses.jsonl \\
      --layers 15 \\
      --out results/harmfulness_direction_scored_layer15.pt

Usage (layer sweep):
  python diagnostics/extract_direction_from_scored.py \\
      --scored results/scored_responses.jsonl \\
      --layers 0 5 10 15 20 25 30 35
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latent_at.paths import get_model_path

# ── Defaults ────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3-8B"
DEFAULT_THRESHOLD = 50


# ── Chat-template helpers (Qwen3) ───────────────────────────────────────────
def format_qwen3_input(system_prompt: str, user_prompt: str, completion: str) -> str:
    """Build a Qwen3 chat-template string: system + user + assistant completion."""
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n{completion}<|im_end|>"
    )


def format_qwen3_prefix(system_prompt: str, user_prompt: str) -> str:
    """Return everything up to (and including) the assistant preamble."""
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


# ── Hidden-state collection ──────────────────────────────────────────────────
def collect_hidden_states_from_records(
    model,
    tokenizer,
    records: list[dict],
    layers: list[int],
    device: str = "cuda",
) -> dict[int, torch.Tensor]:
    """Collect hidden states pooled over response tokens for a list of records.

    Each record must have 'system_prompt', 'user_prompt', 'completion'.

    Returns:
        {layer_idx: tensor of shape [len(records), hidden_dim]}
    """
    h_per_layer: dict[int, list[torch.Tensor]] = {l: [] for l in layers}

    for rec in tqdm(records, desc=f"Collecting hidden states ({len(layers)} layer(s))"):
        sys_prompt = rec["system_prompt"]
        user_prompt = rec["user_prompt"]
        completion = rec["completion"]

        # Build full sequence and locate the prefix boundary
        full_str = format_qwen3_input(sys_prompt, user_prompt, completion)
        prefix_str = format_qwen3_prefix(sys_prompt, user_prompt)

        prefix_ids = tokenizer(
            prefix_str, return_tensors="pt", add_special_tokens=False
        )["input_ids"]
        prefix_len = prefix_ids.shape[1]

        inputs = tokenizer(full_str, return_tensors="pt", add_special_tokens=False).to(
            device
        )

        seq_len = inputs["input_ids"].shape[1]
        if prefix_len >= seq_len:
            print(
                f"  Warning: record (prompt_idx={rec.get('prompt_idx')}, "
                f"sys_idx={rec.get('sys_idx')}, label={rec.get('label')}): "
                f"completion has 0 tokens. Using zero vector."
            )
            for l in layers:
                h_per_layer[l].append(torch.zeros(model.config.hidden_size))
            continue

        # Register hooks on all requested layers, single forward pass
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

        # Pool over all response tokens for each layer
        for l in layers:
            hidden = caches[l][0][0].float().cpu()  # [seq_len, hidden_dim]
            h = hidden[prefix_len:, :].mean(dim=0)  # [hidden_dim]
            h_per_layer[l].append(h)

    return {l: torch.stack(h_per_layer[l]) for l in layers}


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Extract persona direction from score-filtered contrastive responses"
    )
    parser.add_argument(
        "--scored",
        type=str,
        required=True,
        help="Path to scored JSONL from score_responses.py",
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
        help="Multiple layers to extract in one pass (overrides --layer).",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=DEFAULT_THRESHOLD,
        help=f"Score threshold for filtering (default: {DEFAULT_THRESHOLD}). "
        "high-trait = score >= threshold AND label == 'pos'; "
        "low-trait = score <= threshold AND label == 'neg'.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path (single-layer only). Ignored when --layers is used.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--trait_file",
        type=str,
        default=None,
        help="Optional: trait JSON file path to store in metadata. "
        "If not given, attempts to read from the scored JSONL records.",
    )
    args = parser.parse_args()

    # ── Load scored responses ────────────────────────────────────────────
    all_records: list[dict] = []
    with open(args.scored) as f:
        for line in f:
            line = line.strip()
            if line:
                all_records.append(json.loads(line))
    print(f"Loaded {len(all_records)} scored responses from {args.scored}")

    # ── Filter into high-trait and low-trait ──────────────────────────────
    high_trait: list[dict] = []
    low_trait: list[dict] = []
    n_no_score = 0

    for rec in all_records:
        score = rec.get("trait_score")
        label = rec.get("label")

        if score is None:
            n_no_score += 1
            continue

        # High-trait: score >= threshold AND came from pos (trait-inducing) system prompt
        if score >= args.threshold and label == "pos":
            high_trait.append(rec)

        # Low-trait: score <= threshold AND came from neg (trait-suppressing) system prompt
        if score <= args.threshold and label == "neg":
            low_trait.append(rec)

    print(f"\nFiltering with threshold = {args.threshold}:")
    print(
        f"  high-trait (score >= {args.threshold} AND label='pos'): "
        f"{len(high_trait)} responses"
    )
    print(
        f"  low-trait  (score <= {args.threshold} AND label='neg'): "
        f"{len(low_trait)} responses"
    )
    print(f"  no score (REFUSAL / parse fail):  {n_no_score}")

    if len(high_trait) == 0 or len(low_trait) == 0:
        print(
            "\nERROR: One or both classes are empty. Cannot compute direction. "
            "Try adjusting --threshold or generating more data."
        )
        sys.exit(1)

    # ── Layer list and output paths ──────────────────────────────────────
    layers = sorted(set(args.layers)) if args.layers is not None else [args.layer]

    def auto_out_path(layer: int) -> Path:
        return Path(f"results/direction_scored_layer{layer}_t{args.threshold}.pt")

    if len(layers) == 1 and args.out is not None:
        out_paths = {layers[0]: Path(args.out)}
    else:
        out_paths = {l: auto_out_path(l) for l in layers}

    # ── Load model ───────────────────────────────────────────────────────
    print(f"\nLoading model: {MODEL_NAME}")
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

    print(f"  num_hidden_layers = {model.config.num_hidden_layers}")
    print(f"  hidden_size       = {model.config.hidden_size}")
    print(f"  Target layers: {layers}")

    # ── Collect hidden states for both classes ────────────────────────────
    print(f"\n--- High-trait class ({len(high_trait)} responses) ---")
    t0 = time.time()
    h_high_per_layer = collect_hidden_states_from_records(
        model, tokenizer, high_trait, layers, device=args.device
    )

    print(f"\n--- Low-trait class ({len(low_trait)} responses) ---")
    h_low_per_layer = collect_hidden_states_from_records(
        model, tokenizer, low_trait, layers, device=args.device
    )
    elapsed = time.time() - t0
    print(f"\nCollected hidden states in {elapsed:.1f}s ({len(layers)} layer(s))")

    # ── Per-layer: compute direction, print stats, save ──────────────────
    # Gather unique questions for n_samples metadata
    unique_questions = set()
    for rec in high_trait + low_trait:
        unique_questions.add(rec["user_prompt"])

    # Gather unique system prompts
    pos_system_prompts = sorted(set(rec["system_prompt"] for rec in high_trait))
    neg_system_prompts = sorted(set(rec["system_prompt"] for rec in low_trait))

    # Infer trait_file from args or records
    trait_file = args.trait_file
    if trait_file is None:
        # Not critical — just metadata
        trait_file = "unknown"

    for layer in layers:
        h_high = h_high_per_layer[layer]
        h_low = h_low_per_layer[layer]
        out_path = out_paths[layer]

        print(f"\n{'='*60}")
        print(f"LAYER {layer}")
        print(f"{'='*60}")
        print(f"  h_high shape: {h_high.shape}")
        print(f"  h_low  shape: {h_low.shape}")

        mean_high = h_high.mean(dim=0)
        mean_low = h_low.mean(dim=0)
        v_dir = mean_high - mean_low
        v_norm = torch.norm(v_dir).item()
        v_dir_unit = v_dir / torch.norm(v_dir)

        # Project both classes onto the direction for stats
        proj_high = h_high @ v_dir_unit  # [N_high]
        proj_low = h_low @ v_dir_unit  # [N_low]
        gap = proj_high.mean().item() - proj_low.mean().item()
        std_h = proj_high.std().item()
        std_l = proj_low.std().item()
        cohens_d = gap / ((std_h + std_l) / 2) if (std_h + std_l) > 0 else float("nan")

        # Linear accuracy: classify using midpoint threshold
        midpoint = (proj_high.mean().item() + proj_low.mean().item()) / 2
        correct_high = (proj_high > midpoint).sum().item()
        correct_low = (proj_low <= midpoint).sum().item()
        acc = (correct_high + correct_low) / (len(high_trait) + len(low_trait))

        print(f"\n  Direction stats:")
        print(f"    ||mean_high - mean_low|| = {v_norm:.4f}")
        print(f"    Gap:                       {gap:.4f}")
        print(f"    Cohen's d:                 {cohens_d:.2f}")
        print(f"    Linear accuracy (midpoint): {acc:.1%}")
        print(f"    N_high: {len(high_trait)}, N_low: {len(low_trait)}")

        # ── Save ─────────────────────────────────────────────────────────
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_dict = {
            # Generic keys (shared across all direction extraction scripts)
            "v_direction": v_dir_unit,
            "v_direction_unnormalized": v_dir,
            "direction_name": "harm_scored",
            "h_positive": h_high,  # positive class (high-trait)
            "h_negative": h_low,  # negative class (low-trait)
            # Shared metadata
            "norm": v_norm,
            "layer": layer,
            "n_high": len(high_trait),
            "n_low": len(low_trait),
            "n_questions": len(unique_questions),
            "model_name": MODEL_NAME,
            "trait_file": trait_file,
            "hidden_dim": model.config.hidden_size,
            "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
            "pool_mode": "all_completion",
            "threshold": args.threshold,
            "scored_file": args.scored,
            "pos_system_prompts": pos_system_prompts,
            "neg_system_prompts": neg_system_prompts,
            # Stats
            "gap": gap,
            "cohens_d": cohens_d,
            "linear_accuracy": acc,
        }
        torch.save(save_dict, out_path)
        print(f"  Saved → {out_path}")

    print(f"\n{'='*60}")
    print(f"Done. {len(layers)} layer(s) processed.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
