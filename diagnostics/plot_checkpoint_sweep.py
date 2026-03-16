#!/usr/bin/env python3
"""
Plot results from checkpoint sweep analysis.

Generates visualizations to identify the optimal training checkpoint
balancing safety (low ASR) with utility (high MMLU accuracy).

Usage:
    python diagnostics/plot_checkpoint_sweep.py --input_file diagnostics/checkpoint_sweep_results.csv
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot checkpoint sweep results to find optimal safety-utility tradeoff"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="diagnostics/checkpoint_sweep_results.csv",
        help="Path to CSV file with checkpoint sweep results",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="diagnostics/figures/checkpoint_sweep",
        help="Directory to save output plots",
    )
    parser.add_argument(
        "--baseline_asr",
        type=float,
        default=0.40,
        help="Baseline DirectRequest ASR for reference line",
    )
    parser.add_argument(
        "--baseline_mmlu",
        type=float,
        default=0.71,
        help="Baseline MMLU accuracy for reference line",
    )
    return parser.parse_args()


def load_data(input_file: str) -> pd.DataFrame:
    """Load and validate the checkpoint sweep CSV."""
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df = pd.read_csv(input_file)

    required_cols = ["step", "asr_direct_request", "mmlu_accuracy", "pathological_rate"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Sort by step
    df = df.sort_values("step").reset_index(drop=True)

    return df


def find_sweet_spot(
    df: pd.DataFrame, baseline_asr: float, baseline_mmlu: float
) -> dict:
    """
    Find the optimal checkpoint ("sweet spot").

    Criteria: Highest step where MMLU > 0.5 * baseline AND ASR < baseline
    """
    threshold_mmlu = 0.5 * baseline_mmlu

    # Filter to checkpoints meeting criteria
    candidates = df[
        (df["mmlu_accuracy"] > threshold_mmlu)
        & (df["asr_direct_request"] < baseline_asr)
    ]

    if len(candidates) == 0:
        # No checkpoint meets both criteria - return the one with best balance
        # Use a simple scoring: minimize ASR while keeping reasonable MMLU
        df["score"] = df["asr_direct_request"] - df["mmlu_accuracy"]
        best_idx = df["score"].idxmin()
        sweet_spot = df.loc[best_idx]
        return {
            "step": int(sweet_spot["step"]),
            "asr": sweet_spot["asr_direct_request"],
            "mmlu": sweet_spot["mmlu_accuracy"],
            "pathological_rate": sweet_spot["pathological_rate"],
            "meets_criteria": False,
        }

    # Return highest step meeting criteria
    best = candidates.loc[candidates["step"].idxmax()]
    return {
        "step": int(best["step"]),
        "asr": best["asr_direct_request"],
        "mmlu": best["mmlu_accuracy"],
        "pathological_rate": best["pathological_rate"],
        "meets_criteria": True,
    }


def plot_combined(
    df: pd.DataFrame,
    baseline_asr: float,
    baseline_mmlu: float,
    sweet_spot: dict,
    output_path: str,
):
    """
    Create 3-panel figure showing ASR, MMLU, and pathological rate vs training step.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    steps = df["step"].values

    # Top panel: ASR
    ax1 = axes[0]
    ax1.plot(
        steps,
        df["asr_direct_request"],
        "o-",
        color="red",
        linewidth=2,
        markersize=6,
        label="DirectRequest ASR",
    )
    ax1.axhline(
        y=baseline_asr,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Baseline ({baseline_asr:.2f})",
    )
    ax1.set_ylabel("ASR (↓ better)", fontsize=11)
    ax1.set_ylim(0, max(baseline_asr * 1.2, df["asr_direct_request"].max() * 1.1))
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Middle panel: MMLU
    ax2 = axes[1]
    ax2.plot(
        steps,
        df["mmlu_accuracy"],
        "o-",
        color="blue",
        linewidth=2,
        markersize=6,
        label="MMLU Accuracy",
    )
    ax2.axhline(
        y=baseline_mmlu,
        color="blue",
        linestyle="--",
        alpha=0.7,
        label=f"Baseline ({baseline_mmlu:.2f})",
    )
    ax2.axhline(
        y=0.5 * baseline_mmlu,
        color="blue",
        linestyle=":",
        alpha=0.5,
        label=f"50% Baseline ({0.5*baseline_mmlu:.2f})",
    )
    ax2.set_ylabel("MMLU Acc (↑ better)", fontsize=11)
    ax2.set_ylim(0, 1.0)
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Bottom panel: Pathological rate
    ax3 = axes[2]
    ax3.plot(
        steps,
        df["pathological_rate"],
        "o-",
        color="orange",
        linewidth=2,
        markersize=6,
        label="Pathological Rate",
    )
    ax3.set_ylabel("Pathological Rate (↓ better)", fontsize=11)
    ax3.set_xlabel("Training Step", fontsize=11)
    ax3.set_ylim(0, max(0.1, df["pathological_rate"].max() * 1.2))
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_tradeoff(
    df: pd.DataFrame,
    baseline_asr: float,
    baseline_mmlu: float,
    sweet_spot: dict,
    output_path: str,
):
    """
    Create dual y-axis plot showing ASR and MMLU on left axis, pathological rate on right.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    steps = df["step"].values

    # Left y-axis: ASR and MMLU
    (line1,) = ax1.plot(
        steps,
        df["asr_direct_request"],
        "o-",
        color="red",
        linewidth=2,
        markersize=6,
        label="DirectRequest ASR",
    )
    (line2,) = ax1.plot(
        steps,
        df["mmlu_accuracy"],
        "s-",
        color="blue",
        linewidth=2,
        markersize=6,
        label="MMLU Accuracy",
    )

    # Baseline references
    ax1.axhline(y=baseline_asr, color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax1.axhline(y=baseline_mmlu, color="blue", linestyle="--", alpha=0.5, linewidth=1)

    ax1.set_xlabel("Training Step", fontsize=11)
    ax1.set_ylabel("ASR / MMLU Accuracy", fontsize=11)
    ax1.set_ylim(0, 1.0)
    ax1.tick_params(axis="y")
    ax1.grid(True, alpha=0.3)

    # Right y-axis: Pathological rate
    ax2 = ax1.twinx()
    (line3,) = ax2.plot(
        steps,
        df["pathological_rate"],
        "^--",
        color="orange",
        linewidth=2,
        markersize=6,
        label="Pathological Rate",
    )
    ax2.set_ylabel("Pathological Rate", fontsize=11, color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")
    ax2.set_ylim(0, max(0.15, df["pathological_rate"].max() * 1.3))

    # Combined legend
    lines = [line1, line2, line3]
    labels = [l.get_label() for l in lines]
    ax1.legend(
        lines,
        labels,
        loc="upper left",
        ncol=1,
        fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_frontier(
    df: pd.DataFrame, baseline_asr: float, baseline_mmlu: float, output_path: str
):
    """
    Create safety-utility frontier plot (Pareto-style).
    X-axis: MMLU accuracy, Y-axis: ASR, points colored by training step.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    steps = df["step"].values
    mmlu = df["mmlu_accuracy"].values
    asr = df["asr_direct_request"].values

    # Scatter plot colored by step
    scatter = ax.scatter(
        mmlu,
        asr,
        c=steps,
        cmap="viridis",
        s=100,
        edgecolors="black",
        linewidth=0.5,
        zorder=5,
    )
    cbar = plt.colorbar(scatter, ax=ax, label="Training Step")

    # Connect points with a line (ordered by step)
    ax.plot(mmlu, asr, "k-", alpha=0.3, linewidth=1, zorder=1)

    # Baseline point
    ax.scatter(
        [baseline_mmlu],
        [baseline_asr],
        marker="*",
        s=300,
        c="red",
        edgecolors="black",
        linewidth=1,
        zorder=10,
        label="Baseline",
    )

    # Annotate key checkpoints
    key_checkpoints = [10, 50, 100, 200]
    for ckpt in key_checkpoints:
        if ckpt in steps:
            idx = df[df["step"] == ckpt].index[0]
            ax.annotate(
                f"step {ckpt}",
                xy=(mmlu[idx], asr[idx]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
            )

    # Ideal region shading (high MMLU, low ASR)
    ax.axhline(
        y=baseline_asr,
        color="red",
        linestyle="--",
        alpha=0.5,
        linewidth=1,
        label=f"Baseline ASR ({baseline_asr:.2f})",
    )
    ax.axvline(
        x=baseline_mmlu,
        color="blue",
        linestyle="--",
        alpha=0.5,
        linewidth=1,
        label=f"Baseline MMLU ({baseline_mmlu:.2f})",
    )

    ax.set_xlabel("MMLU Accuracy (↑ better)", fontsize=11)
    ax.set_ylabel("DirectRequest ASR (↓ better)", fontsize=11)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, max(baseline_asr * 1.3, asr.max() * 1.1))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def print_analysis(
    df: pd.DataFrame, sweet_spot: dict, baseline_asr: float, baseline_mmlu: float
):
    """Print summary analysis and sweet spot recommendation."""
    print("\n" + "=" * 70)
    print("CHECKPOINT SWEEP ANALYSIS")
    print("=" * 70)

    # Summary table
    print("\n📊 Metrics at Key Checkpoints:")
    print("-" * 70)
    print(
        f"{'Step':>8} | {'ASR':>8} | {'MMLU':>8} | {'Path. Rate':>10} | {'Notes':<20}"
    )
    print("-" * 70)

    # Print baseline
    print(
        f"{'baseline':>8} | {baseline_asr:>8.3f} | {baseline_mmlu:>8.3f} | {'N/A':>10} | Pre-training"
    )
    print("-" * 70)

    # Print key checkpoints and all available
    for _, row in df.iterrows():
        step = int(row["step"])
        asr = row["asr_direct_request"]
        mmlu = row["mmlu_accuracy"]
        path_rate = row["pathological_rate"]

        # Determine notes
        notes = []
        if step == sweet_spot["step"]:
            notes.append("★ SWEET SPOT")
        if asr < baseline_asr:
            notes.append("ASR↓")
        if mmlu > 0.5 * baseline_mmlu:
            notes.append("MMLU OK")

        note_str = ", ".join(notes) if notes else ""
        print(
            f"{step:>8} | {asr:>8.3f} | {mmlu:>8.3f} | {path_rate:>10.3f} | {note_str:<20}"
        )

    print("-" * 70)

    # Sweet spot recommendation
    print("\n🎯 SWEET SPOT RECOMMENDATION:")
    print("-" * 70)
    print(f"  Checkpoint:       step {sweet_spot['step']}")
    print(
        f"  ASR:              {sweet_spot['asr']:.3f} (baseline: {baseline_asr:.3f}, "
        f"Δ = {sweet_spot['asr'] - baseline_asr:+.3f})"
    )
    print(
        f"  MMLU:             {sweet_spot['mmlu']:.3f} (baseline: {baseline_mmlu:.3f}, "
        f"Δ = {sweet_spot['mmlu'] - baseline_mmlu:+.3f})"
    )
    print(f"  Pathological:     {sweet_spot['pathological_rate']:.3f}")
    print(
        f"  Criteria Met:     {'✓ Yes' if sweet_spot['meets_criteria'] else '✗ No (best available)'}"
    )

    if sweet_spot["meets_criteria"]:
        print(f"\n  ✓ MMLU > 50% baseline ({0.5*baseline_mmlu:.3f})")
        print(f"  ✓ ASR < baseline ({baseline_asr:.3f})")
    else:
        print("\n  ⚠️  No checkpoint fully meets both criteria.")
        print("      Recommendation is the best balance available.")

    print("\n" + "=" * 70)


def main():
    args = parse_args()

    # Load data
    print(f"Loading data from: {args.input_file}")
    df = load_data(args.input_file)
    print(
        f"Found {len(df)} checkpoints: steps {df['step'].min()} to {df['step'].max()}"
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find sweet spot
    sweet_spot = find_sweet_spot(df, args.baseline_asr, args.baseline_mmlu)

    # Generate plots
    print("\nGenerating plots...")

    plot_combined(
        df,
        args.baseline_asr,
        args.baseline_mmlu,
        sweet_spot,
        str(output_dir / "sweep_combined.png"),
    )

    plot_tradeoff(
        df,
        args.baseline_asr,
        args.baseline_mmlu,
        sweet_spot,
        str(output_dir / "sweep_tradeoff.png"),
    )

    plot_frontier(
        df,
        args.baseline_asr,
        args.baseline_mmlu,
        str(output_dir / "sweep_frontier.png"),
    )

    # Print analysis
    print_analysis(df, sweet_spot, args.baseline_asr, args.baseline_mmlu)


if __name__ == "__main__":
    main()
