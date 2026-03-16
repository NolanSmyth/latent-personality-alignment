#!/usr/bin/env python3
"""
Visualize the persona vector and its constituent responses in PCA space.

Loads a direction .pt file produced by extract_harmfulness_direction.py or
extract_refusal_direction.py and plots:

  1. Each response as a point (average hidden state across its response tokens),
     coloured by class (positive vs negative).
  2. The mean of each class as a large marker.
  3. The difference-in-means vector (the persona/direction vector itself) as an
     arrow from the negative-class mean to the positive-class mean.
  4. Optionally, the unit-normalised direction vector as a second arrow.

The PCA is fit on the full set of response hidden states [N_pos + N_neg, D].

For direction files from extract_harmfulness_direction.py the individual
response hidden states are NOT stored per-response; only the per-prompt
averages (already averaged over system-prompt variants) are stored.
However, the per-example responses and their system-prompt labels ARE stored
in response_examples — we therefore re-extract per-response hidden states
from h_positive / h_negative when available.

For both file types, h_positive / h_positive and h_negative / h_comply store
one vector per prompt.  Each point in the PCA is therefore one prompt's
average-pooled hidden state.

Usage:
    # Harmfulness direction
    python diagnostics/plot_direction_pca.py \\
        --direction results/harmfulness_direction_layer15.pt \\
        --out diagnostics/figures/pca_harmfulness_layer15.png

    # Refusal direction
    python diagnostics/plot_direction_pca.py \\
        --direction results/refusal_direction_layer15_all_completion.pt \\
        --out diagnostics/figures/pca_refusal_layer15.png

    # Multiple files, auto-named outputs
    python diagnostics/plot_direction_pca.py \\
        --directions results/harmfulness_direction_layer15.pt \\
                     results/refusal_direction_layer15_all_completion.pt \\
        --out_dir diagnostics/figures
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from sklearn.decomposition import PCA


# ── Colour palette ──────────────────────────────────────────────────────────
POS_COLOUR = "#d62728"  # red  — positive / harmful / refusal class
NEG_COLOUR = "#1f77b4"  # blue — negative / benign / comply class
ARROW_COLOUR = "#2ca02c"  # green — direction vector
MEAN_MARKER = "*"
POINT_ALPHA = 0.65
POINT_SIZE = 60


def load_direction_file(path: str) -> dict:
    """Load a .pt direction file and normalise key names to a canonical schema.

    Canonical keys after loading:
        h_positive  : [N, D] float  — positive-class hidden states (one per prompt)
        h_negative  : [N, D] float  — negative-class hidden states (one per prompt)
        v_direction : [D] float     — unit-normalised persona vector (pos - neg)
        direction_name : str        — human-readable label (e.g. "harm", "refusal")
        pos_label   : str           — axis label for the positive class
        neg_label   : str           — axis label for the negative class
        layer       : int
        pool_mode   : str
        n_samples   : int
    """
    data = torch.load(path, map_location="cpu", weights_only=False)

    # --- Positive hidden states ---
    if "h_positive" in data:
        h_pos = data["h_positive"].float()
    elif "h_harmful" in data:
        h_pos = data["h_harmful"].float()
    elif "h_refusal" in data:
        h_pos = data["h_refusal"].float()
    else:
        raise KeyError(
            f"Cannot find positive hidden states in {path}. Keys: {list(data.keys())}"
        )

    # --- Negative hidden states ---
    if "h_negative" in data:
        h_neg = data["h_negative"].float()
    elif "h_benign" in data:
        h_neg = data["h_benign"].float()
    elif "h_comply" in data:
        h_neg = data["h_comply"].float()
    else:
        raise KeyError(
            f"Cannot find negative hidden states in {path}. Keys: {list(data.keys())}"
        )

    # --- Direction vector ---
    v_dir = None
    for key in ("v_direction", "v_refusal", "v_harm"):
        if data.get(key) is not None:
            v_dir = data[key].float()
            break
    if v_dir is None:
        raise KeyError(f"Cannot find direction vector in {path}.")

    # --- Labels ---
    direction_name = data.get("direction_name") or Path(path).stem
    threshold = data.get("threshold")  # present in harm_scored files
    n_high = data.get("n_high")  # present in harm_scored files
    n_low = data.get("n_low")  # present in harm_scored files

    stem = Path(path).stem.lower()
    dn = direction_name.lower()

    if "scored" in dn or "scored" in stem:
        t = threshold if threshold is not None else "?"
        pos_label = f"High-trait: score≥{t} & label=pos"
        neg_label = f"Low-trait:  score≤{t} & label=neg"
    elif "harm" in dn or "harm" in stem:
        pos_label = "Harmful (pos)"
        neg_label = "Benign (neg)"
    elif "refusal" in dn or "refusal" in stem:
        pos_label = "Refusal (pos)"
        neg_label = "Comply (neg)"
    else:
        pos_label = "Positive class"
        neg_label = "Negative class"

    return {
        "h_positive": h_pos,
        "h_negative": h_neg,
        "v_direction": v_dir,
        "direction_name": direction_name,
        "pos_label": pos_label,
        "neg_label": neg_label,
        "layer": data.get("layer", "?"),
        "pool_mode": data.get("pool_mode", "?"),
        "n_samples": data.get("n_samples") or data.get("n_questions") or h_pos.shape[0],
        "threshold": threshold,
        "n_high": n_high,
        "n_low": n_low,
        "source_path": path,
    }


def project_onto_pca(
    h_pos: np.ndarray,  # [N_pos, D]
    h_neg: np.ndarray,  # [N_neg, D]
    v_dir: np.ndarray,  # [D]
    n_components: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, PCA]:
    """Fit PCA on the combined hidden states and project everything.

    Returns:
        coords_pos  : [N_pos, n_components]
        coords_neg  : [N_neg, n_components]
        coords_v    : [n_components]  — projection of direction vector (as offset from origin)
        pca         : fitted PCA object (for explained variance etc.)
    """
    X = np.concatenate([h_pos, h_neg], axis=0)  # [N_pos + N_neg, D]
    pca = PCA(n_components=n_components)
    pca.fit(X)

    coords_pos = pca.transform(h_pos)
    coords_neg = pca.transform(h_neg)

    # Project the direction vector as a displacement from the negative-class mean
    mean_pos = h_pos.mean(axis=0)
    mean_neg = h_neg.mean(axis=0)
    origin_2d = pca.transform(mean_neg[None])[0]
    tip_2d = pca.transform(mean_pos[None])[0]

    return coords_pos, coords_neg, origin_2d, tip_2d, pca


def plot_pca(info: dict, out_path: Path) -> None:
    """Create and save the PCA figure for one direction file."""

    h_pos = info["h_positive"].numpy()
    h_neg = info["h_negative"].numpy()
    v_dir = info["v_direction"].numpy()
    direction_name = info["direction_name"]
    pos_label = info["pos_label"]
    neg_label = info["neg_label"]
    layer = info["layer"]
    pool_mode = info["pool_mode"]
    n_samples = info["n_samples"]

    coords_pos, coords_neg, origin_2d, tip_2d, pca = project_onto_pca(
        h_pos, h_neg, v_dir
    )

    ev = pca.explained_variance_ratio_
    mean_pos_2d = coords_pos.mean(axis=0)
    mean_neg_2d = coords_neg.mean(axis=0)

    fig, ax = plt.subplots(figsize=(8, 6))

    # ── Individual response points ──────────────────────────────────────
    ax.scatter(
        coords_neg[:, 0],
        coords_neg[:, 1],
        c=NEG_COLOUR,
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        edgecolors="white",
        linewidths=0.4,
        zorder=3,
        label=f"{neg_label} (n={len(h_neg)})",
    )
    ax.scatter(
        coords_pos[:, 0],
        coords_pos[:, 1],
        c=POS_COLOUR,
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        edgecolors="white",
        linewidths=0.4,
        zorder=3,
        label=f"{pos_label} (n={len(h_pos)})",
    )

    # ── Class means ─────────────────────────────────────────────────────
    ax.scatter(
        *mean_neg_2d,
        c=NEG_COLOUR,
        s=260,
        marker=MEAN_MARKER,
        edgecolors="black",
        linewidths=0.8,
        zorder=5,
    )
    ax.scatter(
        *mean_pos_2d,
        c=POS_COLOUR,
        s=260,
        marker=MEAN_MARKER,
        edgecolors="black",
        linewidths=0.8,
        zorder=5,
        label="Class means (★)",
    )

    # ── Difference-in-means arrow ────────────────────────────────────────
    dx = tip_2d[0] - origin_2d[0]
    dy = tip_2d[1] - origin_2d[1]
    ax.annotate(
        "",
        xy=(tip_2d[0], tip_2d[1]),
        xytext=(origin_2d[0], origin_2d[1]),
        arrowprops=dict(
            arrowstyle="-|>",
            color=ARROW_COLOUR,
            lw=2.2,
            mutation_scale=18,
        ),
        zorder=6,
    )
    # Small label near arrow midpoint
    mid = (origin_2d + tip_2d) / 2
    ax.text(
        mid[0],
        mid[1],
        "  persona\n  vector",
        fontsize=8,
        color=ARROW_COLOUR,
        va="center",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
    )

    # ── Axes & labels ────────────────────────────────────────────────────
    ax.set_xlabel(f"PC 1  ({100 * ev[0]:.1f}% var. explained)", fontsize=11)
    ax.set_ylabel(f"PC 2  ({100 * ev[1]:.1f}% var. explained)", fontsize=11)
    threshold = info.get("threshold")
    n_high = info.get("n_high")
    n_low = info.get("n_low")
    if threshold is not None and n_high is not None and n_low is not None:
        subtitle = (
            f"Layer {layer} · pool={pool_mode} · "
            f"threshold={threshold} · pos={n_high} responses, neg={n_low} responses"
        )
    else:
        subtitle = f"Layer {layer} · pool={pool_mode} · {n_samples} prompts"
    title = f"PCA of response hidden states — {direction_name} direction\n{subtitle}"
    ax.set_title(title, fontsize=12, pad=10)

    ax.legend(fontsize=9, loc="best", framealpha=0.85)
    ax.grid(True, linestyle="--", alpha=0.35)

    # Variance in a small text box
    total_var = sum(ev[:2]) * 100
    ax.text(
        0.98,
        0.02,
        f"PC1+PC2: {total_var:.1f}% variance",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="gray",
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")
    print(
        f"  PC1={ev[0]*100:.1f}%  PC2={ev[1]*100:.1f}%  "
        f"(total {sum(ev[:2])*100:.1f}%)"
    )


def auto_out_name(direction_path: str) -> str:
    """Derive a figure filename from the direction file stem."""
    stem = Path(direction_path).stem
    return f"pca_{stem}.png"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PCA plot of hidden states used to compute a direction vector"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--direction",
        type=str,
        default=None,
        help="Path to a single .pt direction file.",
    )
    group.add_argument(
        "--directions",
        type=str,
        nargs="+",
        default=None,
        help="Multiple .pt direction files (auto-named outputs saved in --out_dir).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path for the figure (single --direction only).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="diagnostics/figures",
        help="Output directory when --directions is used (default: diagnostics/figures).",
    )
    args = parser.parse_args()

    if args.directions is not None:
        paths = args.directions
        multi = True
    else:
        paths = [args.direction]
        multi = False

    for p in paths:
        print(f"\n{'='*60}")
        print(f"Loading: {p}")
        info = load_direction_file(p)
        print(
            f"  {info['direction_name']} direction  layer={info['layer']}  "
            f"pool={info['pool_mode']}"
        )
        print(
            f"  h_positive: {info['h_positive'].shape}  "
            f"h_negative: {info['h_negative'].shape}"
        )

        if multi or args.out is None:
            out_path = Path(args.out_dir) / auto_out_name(p)
        else:
            out_path = Path(args.out)

        plot_pca(info, out_path)

    print(f"\n{'='*60}")
    print(f"Done. {len(paths)} figure(s) saved.")


if __name__ == "__main__":
    main()
