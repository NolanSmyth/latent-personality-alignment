#!/usr/bin/env python3
"""
PGD Gradient Direction Diagnostic

For each item in IPIP-14, runs PGD with batch_size=1 and captures:
  1. The initial gradient direction ∂L_adv/∂δ at δ=0 (one-step, before any
     optimization — the "natural perturbation direction").
  2. The final δ after full PGD optimization.

For each layer, computes cosine similarity between positive-item and
negative-item vectors to test the gradient-cancellation hypothesis:

  If δ_pos ≈ −δ_neg, then in a mixed batch the adversary gradients cancel and
  the defense receives a near-zero gradient signal, explaining agree-collapse.

Usage:
    python diagnostics/probe_pgd_gradient_directions.py \\
        --model_name Qwen/Qwen3-8B \\
        --csv_path data/IPIP-14/harmful_trait.csv

    # Optionally limit items (faster runs):
    python diagnostics/probe_pgd_gradient_directions.py \\
        --model_name Qwen/Qwen3-8B \\
        --csv_path data/IPIP-14/harmful_trait.csv \\
        --max_pos_items 5 --max_neg_items 5
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig

sys.path.insert(0, str(Path(__file__).parent.parent))
from latent_at.paths import get_model_path
from latent_at.lat_datasets import (
    process_generic_chat_dataset,
    LatentAdversarialTrainingDataCollator,
)
from latent_at.lat_methods import projected_gradient_descent
from latent_at.laa import clear_hooks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_valence_for_items(csv_path: str) -> list[str]:
    """Return list of 'positive' or 'negative' for each row in the CSV."""
    valences = []
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            chosen = row["chosen"].strip().lower()
            valences.append(
                "negative" if ("do not agree" in chosen or "disagree" in chosen) else "positive"
            )
    return valences


def extract_delta_vectors(wrappers, device: str = "cuda") -> dict[int, torch.Tensor]:
    """
    For each wrapper (one per instrumented layer), extract the mean-pooled δ
    vector over masked positions.

    Returns:
        dict mapping wrapper_index → (hidden_dim,) float32 CPU tensor
    """
    vectors = {}
    for w_idx, wrapper in enumerate(wrappers):
        adv = wrapper.hook_fn  # GDAdversary
        if not hasattr(adv, "attack") or adv.attack is None:
            continue
        attack = adv.attack.data  # (1, seq_len, hidden_dim) for batch_size=1
        mask = adv.attack_mask    # (1, seq_len)

        # attack is shape (batch, seq, hidden); mask is (batch, seq)
        # We have batch_size=1 → squeeze out the batch dim
        attack_flat = attack[0]   # (seq_len, hidden_dim)
        mask_flat = mask[0]       # (seq_len,)

        if mask_flat.any():
            # Mean over masked positions → (hidden_dim,)
            delta_vec = attack_flat[mask_flat].mean(dim=0).float().cpu()
        else:
            delta_vec = torch.zeros(attack_flat.shape[-1], dtype=torch.float32)

        vectors[w_idx] = delta_vec
    return vectors


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two 1-D tensors."""
    a_n = F.normalize(a.unsqueeze(0), dim=-1)
    b_n = F.normalize(b.unsqueeze(0), dim=-1)
    return (a_n * b_n).sum().item()


def pairwise_cosine_stats(vecs_a: list[torch.Tensor], vecs_b: list[torch.Tensor]):
    """
    All-pairs cosine similarity between each tensor in vecs_a and each in vecs_b.
    Returns (mean, min, max, all_sims_list).
    """
    sims = []
    for a in vecs_a:
        for b in vecs_b:
            sims.append(cosine_sim(a, b))
    if not sims:
        return float("nan"), float("nan"), float("nan"), []
    return sum(sims) / len(sims), min(sims), max(sims), sims


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cosine similarity between pos/neg per-item PGD δ directions"
    )
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument(
        "--system_prompt_path", type=str, default="system_prompt/alpha.txt"
    )
    parser.add_argument("--pgd_iterations", type=int, default=16)
    parser.add_argument("--epsilon", type=float, default=6.0)
    parser.add_argument("--inner_lr", type=float, default=1e-3)
    parser.add_argument(
        "--max_pos_items",
        type=int,
        default=None,
        help="Limit positive items processed (default: all)",
    )
    parser.add_argument(
        "--max_neg_items",
        type=int,
        default=None,
        help="Limit negative items processed (default: all)",
    )
    args = parser.parse_args()

    # Load system prompt
    with open(args.system_prompt_path) as fh:
        system_prompt = fh.read().strip()

    # Valences per CSV row
    valences = get_valence_for_items(args.csv_path)
    pos_indices = [i for i, v in enumerate(valences) if v == "positive"]
    neg_indices = [i for i, v in enumerate(valences) if v == "negative"]
    print(
        f"Dataset: {len(valences)} items  ({len(pos_indices)} positive, "
        f"{len(neg_indices)} negative)"
    )

    # Optionally limit
    if args.max_pos_items is not None:
        pos_indices = pos_indices[: args.max_pos_items]
    if args.max_neg_items is not None:
        neg_indices = neg_indices[: args.max_neg_items]
    print(
        f"Processing {len(pos_indices)} positive and {len(neg_indices)} negative items"
        f" (batch_size=1 each)"
    )
    print(f"PGD: {args.pgd_iterations} iterations, ε={args.epsilon}, lr={args.inner_lr}")

    # Load model
    model_path = get_model_path(args.model_name)
    print(f"\nLoading model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Add LoRA adapter (matching training setup)
    lora_config = LoraConfig(
        r=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.eval()

    # Chat template (Qwen3)
    custom_prompt_template = (
        "<|im_start|>system\n{system_prompt}<|im_end|>\n"
        "<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    custom_completion_template = "{completion}"

    # Process full dataset (batch_size=1 DataLoader per item below)
    print("Processing dataset...")
    lat_dataset = process_generic_chat_dataset(
        tokenizer,
        dataset=args.csv_path,
        adv_column="rejected",
        def_column="chosen",
        split="train",
        use_tokenizer_template=False,
        system_prompt=system_prompt,
        custom_prompt_template=custom_prompt_template,
        custom_completion_template=custom_completion_template,
    )

    collator = LatentAdversarialTrainingDataCollator(
        tokenizer.pad_token_id, truncate_length=2048
    )

    pgd_layers = ["embedding", 8, 16, 24, 30]
    adv_loss_coefs = {"toward": 0.5, "away": 0.5}
    n_layers = len(pgd_layers)

    # Storage: for each layer index → list of (hidden_dim,) delta vectors
    pos_deltas: dict[int, list[torch.Tensor]] = {i: [] for i in range(n_layers)}
    neg_deltas: dict[int, list[torch.Tensor]] = {i: [] for i in range(n_layers)}

    # Also store initial gradients (∂L/∂δ at δ=0) via 1-step PGD
    pos_init_grads: dict[int, list[torch.Tensor]] = {i: [] for i in range(n_layers)}
    neg_init_grads: dict[int, list[torch.Tensor]] = {i: [] for i in range(n_layers)}

    def run_pgd_single_item(item_idx: int, n_iters: int) -> tuple[dict, dict]:
        """
        Run PGD (n_iters steps) on item_idx alone (batch_size=1).
        Returns (final_delta_vecs, init_grad_vecs) — both keyed by wrapper_index.
        """
        single_loader = DataLoader(
            Subset(lat_dataset, [item_idx]),
            batch_size=1,
            shuffle=False,
            collate_fn=collator,
        )
        batch = next(iter(single_loader))

        # ---- 1. Initial gradient (1-step PGD to get ∂L/∂δ at δ=0) ----
        _, wrappers_1 = projected_gradient_descent(
            batch=batch,
            model=model,
            model_layers_module="base_model.model.model.layers",
            layer=pgd_layers,
            epsilon=args.epsilon,
            learning_rate=args.inner_lr,
            pgd_iterations=1,            # single step → captures initial gradient direction
            loss_coefs=adv_loss_coefs,
            log_loss=False,
            device="cuda",
            add_completions_pgd=True,
        )
        init_grad_vecs = extract_delta_vectors(wrappers_1)
        clear_hooks(model)

        # ---- 2. Full PGD → final δ ----
        _, wrappers_full = projected_gradient_descent(
            batch=batch,
            model=model,
            model_layers_module="base_model.model.model.layers",
            layer=pgd_layers,
            epsilon=args.epsilon,
            learning_rate=args.inner_lr,
            pgd_iterations=n_iters,
            loss_coefs=adv_loss_coefs,
            log_loss=False,
            device="cuda",
            add_completions_pgd=True,
        )
        final_delta_vecs = extract_delta_vectors(wrappers_full)
        clear_hooks(model)

        return final_delta_vecs, init_grad_vecs

    print(f"\n{'='*80}")
    print("Running PGD per item (batch_size=1)...")
    print(f"{'='*80}")

    for item_idx in pos_indices:
        print(f"  [+] item {item_idx+1:3d} (positive)...", end=" ", flush=True)
        final_vecs, init_vecs = run_pgd_single_item(item_idx, args.pgd_iterations)
        for l_idx in range(n_layers):
            if l_idx in final_vecs:
                pos_deltas[l_idx].append(final_vecs[l_idx])
            if l_idx in init_vecs:
                pos_init_grads[l_idx].append(init_vecs[l_idx])
        print("done")

    for item_idx in neg_indices:
        print(f"  [-] item {item_idx+1:3d} (negative)...", end=" ", flush=True)
        final_vecs, init_vecs = run_pgd_single_item(item_idx, args.pgd_iterations)
        for l_idx in range(n_layers):
            if l_idx in final_vecs:
                neg_deltas[l_idx].append(final_vecs[l_idx])
            if l_idx in init_vecs:
                neg_init_grads[l_idx].append(init_vecs[l_idx])
        print("done")

    # ---------------------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------------------
    layer_labels = [str(l) for l in pgd_layers]

    def print_cosine_table(label: str, pos_store: dict, neg_store: dict):
        print(f"\n{'='*80}")
        print(f"COSINE SIMILARITY (pos vs neg): {label}")
        print(f"{'='*80}")
        print(f"{'Layer':<12} {'n_pos':>6} {'n_neg':>6} {'mean_cos':>10} {'min_cos':>10} {'max_cos':>10}  {'interp':>30}")
        print("-" * 80)
        for l_idx in range(n_layers):
            lbl = layer_labels[l_idx]
            pv = pos_store.get(l_idx, [])
            nv = neg_store.get(l_idx, [])
            if not pv or not nv:
                print(f"  {lbl:<10} {'—':>6} {'—':>6} {'—':>10} {'—':>10} {'—':>10}")
                continue
            mean_c, min_c, max_c, _ = pairwise_cosine_stats(pv, nv)
            interp = (
                "STRONG OPPOSITION (≤-0.5)"
                if mean_c <= -0.5
                else (
                    "moderate opposition (-0.5..−0.1)"
                    if mean_c < -0.1
                    else (
                        "near-orthogonal (−0.1..0.1)"
                        if -0.1 <= mean_c <= 0.1
                        else "aligned (>0.1)"
                    )
                )
            )
            print(
                f"  {lbl:<10} {len(pv):>6} {len(nv):>6} "
                f"{mean_c:>10.4f} {min_c:>10.4f} {max_c:>10.4f}  {interp}"
            )

        # Also report mean vector cos sim (aggregate direction)
        print(f"\n  Mean-vector cosine similarity (direction of avg δ_pos vs avg δ_neg):")
        for l_idx in range(n_layers):
            lbl = layer_labels[l_idx]
            pv = pos_store.get(l_idx, [])
            nv = neg_store.get(l_idx, [])
            if not pv or not nv:
                continue
            mean_pos = torch.stack(pv).mean(dim=0)
            mean_neg = torch.stack(nv).mean(dim=0)
            cs = cosine_sim(mean_pos, mean_neg)
            print(f"    layer {lbl:<6}: cos(mean_pos, mean_neg) = {cs:+.4f}")

    print_cosine_table("Final δ (after full PGD)", pos_deltas, neg_deltas)
    print_cosine_table("Initial gradient (1-step PGD)", pos_init_grads, neg_init_grads)

    # ---------------------------------------------------------------------------
    # Interpretation summary
    # ---------------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("INTERPRETATION")
    print(f"{'='*80}")

    # Use the mean-vector cosine from the final δ at layer 8 as the key number
    l8_idx = None
    for i, lbl in enumerate(layer_labels):
        if str(lbl) == "8":
            l8_idx = i
            break

    if l8_idx is not None and pos_deltas.get(l8_idx) and neg_deltas.get(l8_idx):
        mean_pos_l8 = torch.stack(pos_deltas[l8_idx]).mean(dim=0)
        mean_neg_l8 = torch.stack(neg_deltas[l8_idx]).mean(dim=0)
        cs_l8 = cosine_sim(mean_pos_l8, mean_neg_l8)
        if cs_l8 <= -0.3:
            print(
                f"  cos(δ_pos, δ_neg) at layer 8 = {cs_l8:+.4f} — "
                "GRADIENT CANCELLATION CONFIRMED.\n"
                "  In a mixed batch the adversary pushes pos/neg representations in opposite\n"
                "  directions; the defense gradient averages to near-zero.\n"
                "  Fix: separate per-valence dataloaders (Task 8b/8c)."
            )
        elif -0.3 < cs_l8 < 0.1:
            print(
                f"  cos(δ_pos, δ_neg) at layer 8 = {cs_l8:+.4f} — "
                "mild/no opposition.\n"
                "  Gradient cancellation is a partial or negligible factor.\n"
                "  Look elsewhere for the agree-collapse source."
            )
        else:
            print(
                f"  cos(δ_pos, δ_neg) at layer 8 = {cs_l8:+.4f} — "
                "ALIGNED (same direction).\n"
                "  Positive and negative items are pushed in the SAME direction.\n"
                "  The agree-collapse has a different cause."
            )

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
