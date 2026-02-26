#!/usr/bin/env python3
"""
EXP-019 Step 2 — Sanity-check the refusal direction with projection analysis.

Projects hidden states onto v_refusal and checks whether refusal-class
activations are separable from comply-class activations.

Usage:
  python diagnostics/probe_refusal_direction.py \
      --direction results/refusal_direction_layer15.pt \
      --layer 15 --n_samples 64
"""

import argparse
import os
import sys

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="Probe refusal direction quality")
    parser.add_argument("--direction", type=str, required=True,
                        help="Path to .pt file from extract_refusal_direction.py")
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--n_samples", type=int, default=64,
                        help="Unused if cached h_refusal/h_comply are in the .pt file")
    args = parser.parse_args()

    # ── Load saved data ─────────────────────────────────────────────────
    data = torch.load(args.direction, map_location="cpu", weights_only=False)
    v_refusal = data["v_refusal"]  # unit-normalized direction
    print(f"Loaded direction from {args.direction}")
    print(f"  Layer: {data['layer']}, N samples: {data['n_samples']}")
    print(f"  Hidden dim: {v_refusal.shape[0]}")
    print(f"  ||v_refusal_unnormalized||: {data['norm']:.4f}")

    if "h_refusal" not in data or "h_comply" not in data:
        print("\n❌ No cached hidden states in .pt file. Re-run extract_refusal_direction.py.")
        print("   (Or implement re-collection here — skipped for efficiency.)")
        return

    h_refusal = data["h_refusal"].float()  # [N, hidden_dim]
    h_comply = data["h_comply"].float()    # [N, hidden_dim]
    N = h_refusal.shape[0]

    # ── Project onto direction ──────────────────────────────────────────
    proj_refusal = h_refusal @ v_refusal  # [N]
    proj_comply = h_comply @ v_refusal    # [N]

    mean_r = proj_refusal.mean().item()
    mean_c = proj_comply.mean().item()
    std_r = proj_refusal.std().item()
    std_c = proj_comply.std().item()
    gap = mean_r - mean_c

    print(f"\n{'=' * 60}")
    print(f"PROJECTION ANALYSIS (N={N})")
    print(f"{'=' * 60}")
    print(f"  Refusal class:  mean={mean_r:+.4f}  std={std_r:.4f}")
    print(f"  Comply class:   mean={mean_c:+.4f}  std={std_c:.4f}")
    print(f"  Gap (refusal - comply): {gap:+.4f}")
    print(f"  Cohen's d: {gap / ((std_r + std_c) / 2):.2f}")

    # ── Classification accuracy (threshold at midpoint) ─────────────────
    threshold = (mean_r + mean_c) / 2
    correct_r = (proj_refusal > threshold).sum().item()
    correct_c = (proj_comply <= threshold).sum().item()
    accuracy = (correct_r + correct_c) / (2 * N)

    print(f"\n  Linear separability (threshold={threshold:.4f}):")
    print(f"    Refusal correct: {correct_r}/{N}")
    print(f"    Comply correct:  {correct_c}/{N}")
    print(f"    Accuracy: {accuracy:.1%}")

    # ── Distribution overlap ────────────────────────────────────────────
    # What fraction of comply projections exceed the refusal mean?
    overlap = (proj_comply > mean_r).sum().item() / N
    print(f"\n  Overlap: {overlap:.1%} of comply projections exceed refusal mean")

    # ── Pass / fail ─────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    if gap > 0.1 and accuracy > 0.7:
        print(f"✅ PASS: Direction is meaningful (gap={gap:.4f}, acc={accuracy:.1%})")
    elif gap > 0:
        print(f"⚠️  WEAK: Direction has some signal but may be noisy (gap={gap:.4f}, acc={accuracy:.1%})")
    else:
        print(f"❌ FAIL: Direction has wrong sign or no signal (gap={gap:.4f})")
    print(f"{'=' * 60}")

    # ── Per-sample breakdown (first 10) ─────────────────────────────────
    print(f"\nPer-sample projections (first 10):")
    print(f"  {'idx':>4}  {'refusal':>10}  {'comply':>10}  {'gap':>10}")
    for i in range(min(10, N)):
        r = proj_refusal[i].item()
        c = proj_comply[i].item()
        print(f"  {i:>4}  {r:>+10.4f}  {c:>+10.4f}  {r-c:>+10.4f}")


if __name__ == "__main__":
    main()
