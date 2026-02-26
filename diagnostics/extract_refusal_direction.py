#!/usr/bin/env python3
"""
EXP-019 Step 1 — Extract the refusal direction from Qwen3-8B.

For N HarmBench prompts, builds two inputs per prompt:
  - refusal:  prompt + refusal_completion
  - comply:   prompt + compliance_completion

Collects hidden states at the last-prompt-token position from the specified
layer, then computes:
  v_refusal = mean(h_refusal) - mean(h_comply)   (unit-normalized)

Usage:
  python diagnostics/extract_refusal_direction.py \
      --layer 15 --n_samples 64 --out results/refusal_direction_layer15.pt
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
TARGETS_PATH   = "tasks/harmbench/data/harmbench_data/optimizer_targets/text_targets.csv"
MODEL_NAME     = "Qwen/Qwen3-8B"

# ── Chat template for Qwen3 ────────────────────────────────────────────────
def format_qwen3_input(prompt: str, completion: str) -> str:
    """Build a Qwen3 chat-template string: user prompt + assistant completion."""
    return (
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n{completion}<|im_end|>"
    )


def load_contrastive_data(n_samples: int) -> list[dict]:
    """Load and merge HarmBench behaviors + targets, return list of dicts."""
    behaviors_df = pd.read_csv(BEHAVIORS_PATH)
    targets_df = pd.read_csv(TARGETS_PATH)
    merged = behaviors_df.merge(targets_df, on="BehaviorID", how="inner")
    standard = merged[merged["FunctionalCategory"] == "standard"].copy()
    standard = standard.dropna(subset=["llama_prompt_refusal", "optim_target"])

    if n_samples > len(standard):
        print(f"Warning: requested {n_samples} samples but only {len(standard)} available. Using all.")
        n_samples = len(standard)

    standard = standard.head(n_samples)

    triples = []
    for _, row in standard.iterrows():
        triples.append({
            "prompt": row["Behavior"],
            "refusal": row["llama_prompt_refusal"],
            "comply": row["optim_target"],
            "behavior_id": row["BehaviorID"],
        })
    return triples


def collect_hidden_states(
    model,
    tokenizer,
    triples: list[dict],
    layer: int,
    batch_size: int = 8,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Collect hidden states at the last-prompt-token position from the specified
    layer for refusal and comply inputs.

    Returns:
        h_refusal: [N, hidden_dim]
        h_comply:  [N, hidden_dim]
    """
    target_module = model.model.layers[layer]

    h_refusal_list = []
    h_comply_list = []

    for triple in tqdm(triples, desc=f"Collecting hidden states (layer {layer})"):
        prompt = triple["prompt"]
        # Tokenize prompt alone to find its length
        prompt_str = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        prompt_tokens = tokenizer.encode(prompt_str, add_special_tokens=False)
        last_prompt_idx = len(prompt_tokens) - 1

        for completion_key, out_list in [("refusal", h_refusal_list), ("comply", h_comply_list)]:
            full_str = format_qwen3_input(prompt, triple[completion_key])
            inputs = tokenizer(full_str, return_tensors="pt", add_special_tokens=False).to(device)

            cache = []
            def hook_fn(module, input, output, _cache=cache):
                if isinstance(output, tuple):
                    _cache.append(output[0].detach())
                else:
                    _cache.append(output.detach())

            handle = target_module.register_forward_hook(hook_fn)
            with torch.no_grad():
                model(**inputs)
            handle.remove()

            # cache[0] shape: [1, seq_len, hidden_dim]
            # Extract at last-prompt-token position
            h = cache[0][0, last_prompt_idx, :].float().cpu()
            out_list.append(h)

    h_refusal = torch.stack(h_refusal_list)  # [N, hidden_dim]
    h_comply = torch.stack(h_comply_list)    # [N, hidden_dim]
    return h_refusal, h_comply


def main():
    parser = argparse.ArgumentParser(description="Extract refusal direction from Qwen3-8B")
    parser.add_argument("--layer", type=int, default=15, help="Layer to extract from")
    parser.add_argument("--n_samples", type=int, default=64, help="Number of prompt pairs")
    parser.add_argument("--out", type=str, default="results/refusal_direction_layer15.pt",
                        help="Output path for direction tensor")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (unused, sequential for now)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

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

    # Print layer structure for verification
    print(f"\nModel architecture check:")
    print(f"  model.config.num_hidden_layers = {model.config.num_hidden_layers}")
    print(f"  model.config.hidden_size = {model.config.hidden_size}")
    print(f"  Target layer: model.model.layers[{args.layer}]")
    print(f"  Module type: {type(model.model.layers[args.layer])}")

    # ── Load data ───────────────────────────────────────────────────────
    triples = load_contrastive_data(args.n_samples)
    print(f"\nLoaded {len(triples)} contrastive triples")

    # ── Collect hidden states ───────────────────────────────────────────
    t0 = time.time()
    h_refusal, h_comply = collect_hidden_states(
        model, tokenizer, triples, args.layer, device=args.device
    )
    elapsed = time.time() - t0
    print(f"\nCollected hidden states in {elapsed:.1f}s")
    print(f"  h_refusal shape: {h_refusal.shape}")
    print(f"  h_comply  shape: {h_comply.shape}")

    # ── Compute direction ───────────────────────────────────────────────
    mean_refusal = h_refusal.mean(dim=0)
    mean_comply = h_comply.mean(dim=0)
    v_refusal = mean_refusal - mean_comply
    v_norm = torch.norm(v_refusal).item()
    v_refusal_unit = v_refusal / torch.norm(v_refusal)

    print(f"\nDirection stats:")
    print(f"  ||mean_refusal - mean_comply|| = {v_norm:.4f}")
    print(f"  v_refusal_unit shape: {v_refusal_unit.shape}")

    # Quick sanity: project back onto the collected states
    proj_refusal = (h_refusal @ v_refusal_unit).mean().item()
    proj_comply = (h_comply @ v_refusal_unit).mean().item()
    print(f"\n  Sanity check (mean projection onto direction):")
    print(f"    Refusal class: {proj_refusal:.4f}")
    print(f"    Comply class:  {proj_comply:.4f}")
    print(f"    Gap:           {proj_refusal - proj_comply:.4f}")

    # ── Save ────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "v_refusal": v_refusal_unit,
        "v_refusal_unnormalized": v_refusal,
        "norm": v_norm,
        "layer": args.layer,
        "n_samples": len(triples),
        "model_name": MODEL_NAME,
        "hidden_dim": model.config.hidden_size,
        "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
        # Also save the raw hidden states for re-use in probe script
        "h_refusal": h_refusal,
        "h_comply": h_comply,
    }
    torch.save(save_dict, out_path)
    print(f"\nSaved to {out_path}")
    print(f"  Keys: {list(save_dict.keys())}")


if __name__ == "__main__":
    main()
