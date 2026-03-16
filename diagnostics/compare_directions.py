#!/usr/bin/env python3
"""
Compare direction vectors extracted by different methods.

For every pair of .pt files, computes the cosine similarity between their
v_direction vectors (at the same layer).  Also reports a random-vector
baseline: the expected |cosine similarity| between any of the real directions
and a uniformly-random unit vector in the same space.

Supports files from any extraction script that follows the generic schema
(v_direction key) as well as old files that only have v_refusal or v_harm.

Usage:
  python diagnostics/compare_directions.py \\
      results/refusal_direction_layer15.pt \\
      results/harmfulness_direction_layer15.pt

Layer sweep (all .pt files in a directory):
  python diagnostics/compare_directions.py results/

Optional: print saved response examples from harmfulness extraction files:
  python diagnostics/compare_directions.py ... --show_examples
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch


# ── Helpers ──────────────────────────────────────────────────────────────────


def load_direction(path: str) -> tuple[torch.Tensor, dict]:
    """Load a .pt direction file, resolving the vector key generically."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    v = next(
        (data[k] for k in ("v_direction", "v_refusal", "v_harm") if k in data),
        None,
    )
    if v is None:
        raise KeyError(
            f"{path}: no direction vector found "
            f"(tried v_direction, v_refusal, v_harm)"
        )
    return v.float(), data


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a @ b).item() / (a.norm().item() * b.norm().item())


def random_baseline(
    vectors: list[torch.Tensor], n_samples: int = 100_000
) -> dict[str, tuple[float, float]]:
    """
    Draw n_samples random unit vectors in the shared hidden dimension and
    compute mean ± std of |cosine similarity| with each real direction.

    Returns {direction_name: (mean, std)}.
    """
    dim = vectors[0].shape[0]
    rand = torch.randn(n_samples, dim)
    rand = rand / rand.norm(dim=1, keepdim=True)  # [N, dim]
    results = {}
    for v, name in vectors:
        sims = (rand @ v).abs()  # [N]
        results[name] = (sims.mean().item(), sims.std().item())
    return results


def theoretical_random_baseline(dim: int) -> float:
    """E[|cos(v, u)|] for a random unit vector u in R^dim = sqrt(2 / (pi * dim))."""
    return math.sqrt(2.0 / (math.pi * dim))


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare extracted direction vectors and random baselines"
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Paths to .pt direction files, or a directory (all .pt files loaded).",
    )
    parser.add_argument(
        "--n_random",
        type=int,
        default=100_000,
        help="Number of random unit vectors for the baseline (default: 100 000).",
    )
    parser.add_argument(
        "--show_examples",
        action="store_true",
        help="Print saved response_examples from harmfulness extraction files.",
    )
    args = parser.parse_args()

    # Resolve paths (expand directories)
    paths: list[Path] = []
    for p in args.paths:
        p = Path(p)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.pt")))
        else:
            paths.append(p)

    if not paths:
        print("No .pt files found.")
        return

    # ── Load ─────────────────────────────────────────────────────────────
    print(f"{'='*70}")
    print(f"LOADED DIRECTIONS")
    print(f"{'='*70}")
    entries: list[tuple[torch.Tensor, dict, Path]] = []
    for p in paths:
        v, data = load_direction(str(p))
        entries.append((v, data, p))
        name = data.get("direction_name", p.stem)
        layer = data.get("layer", "?")
        n_samp = data.get("n_samples", "?")
        print(
            f"  [{name}]  layer={layer}  n_samples={n_samp}  "
            f"dim={v.shape[0]}  ||v||={v.norm().item():.4f}  ({p})"
        )

    if len(entries) < 2:
        print("\nNeed at least 2 direction files to compare.")
        return

    # ── Pairwise cosine similarities ────────────────────────────────────
    print(f"\n{'='*70}")
    print("PAIRWISE COSINE SIMILARITIES")
    print(f"{'='*70}")
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            vi, di, pi = entries[i]
            vj, dj, pj = entries[j]
            ni = di.get("direction_name", pi.stem)
            nj = dj.get("direction_name", pj.stem)
            li = di.get("layer", "?")
            lj = dj.get("layer", "?")
            if vi.shape != vj.shape:
                print(
                    f"  ⚠  {ni} (layer {li}) ↔ {nj} (layer {lj}): "
                    f"dimension mismatch {vi.shape} vs {vj.shape} — skipped"
                )
                continue
            sim = cosine_sim(vi, vj)
            print(f"  {ni} (layer {li}) ↔ {nj} (layer {lj}): cos = {sim:+.4f}")

    # ── Random baseline ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"RANDOM-VECTOR BASELINE  (N={args.n_random:,} draws)")
    print(f"{'='*70}")

    # Group by dimension so we only draw one random matrix per unique dim
    dim_groups: dict[int, list[tuple[torch.Tensor, str, int]]] = {}
    for v, d, p in entries:
        dim = v.shape[0]
        name = d.get("direction_name", p.stem)
        layer = d.get("layer", "?")
        dim_groups.setdefault(dim, []).append((v, name, layer))

    for dim, group in dim_groups.items():
        rand = torch.randn(args.n_random, dim)
        rand = rand / rand.norm(dim=1, keepdim=True)  # [N, dim]
        theoretical = theoretical_random_baseline(dim)
        print(f"\n  dim={dim}  theoretical E[|cos|] = {theoretical:.6f}")
        for v, name, layer in group:
            sims = (rand @ v).abs()
            print(
                f"    [{name}] layer={layer}: "
                f"mean |cos| = {sims.mean().item():.6f} ± {sims.std().item():.6f}"
            )

    # ── Optional: print response examples ───────────────────────────────
    if args.show_examples:
        for _, data, path in entries:
            examples = data.get("response_examples")
            if not examples:
                continue
            print(f"\n{'='*70}")
            print(f"RESPONSE EXAMPLES  ({path.name})")
            print(f"{'='*70}")
            for label in ("harmful", "benign"):
                for ex in examples.get(label, []):
                    print(
                        f"\n  [{label.upper()}]  prompt_idx={ex['prompt_idx']}  sys_idx={ex['sys_idx']}"
                    )
                    print(f"  System:   {ex['system_prompt'][:120]}...")
                    print(f"  User:     {ex['user_prompt'][:120]}")
                    print(f"  Response: {ex['completion'][:400]}...")


if __name__ == "__main__":
    main()
