#!/usr/bin/env python3
"""
Analyze and plot the distribution of model responses across eval benchmarks.

Compares trained (LPA) model vs base model responses.
Highlights pathological responses like "I" (from "I do not agree...")
that indicate the model is treating utility evals as harmful prompts.

Usage:
    python diagnostics/plot_response_distribution.py [--base-dir CACHE_DIR]
"""

import json
import os
import sys
import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"

TRAINED_MODEL_DIR = (
    CACHE_DIR
    / "lpa-regular-config_IPIP-14_fewer_steps_2026-02-13_13-35-34-111503"
    / "eval"
)
BASE_MODEL_DIR = CACHE_DIR / "base-eval_baseline" / "checkpoint_0" / "eval"

UTILITY_EVALS = ["MMLU", "HellaSwag", "Winogrande", "SciQ"]
HARMBENCH_EVALS = [
    "DirectRequest",
    "AutoDAN",
    "AutoPrompt",
    "GCG",
    "PAIR",
    "TAP",
    "clean",
]

# Letters that are valid MC answers
VALID_MC_ANSWERS = set("ABCDEFGHIJ")


def load_cache(eval_dir: Path, eval_name: str) -> list[dict]:
    """Load a cache.jsonl file."""
    path = eval_dir / eval_name / "cache.jsonl"
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def classify_mc_response(response: str) -> str:
    """
    Classify a model response to a multiple-choice question.

    Returns one of:
    - "A", "B", "C", "D", etc. (valid answer letter)
    - "I" (starts with I — the "I do not agree" pathology)
    - "digit" (numeric response)
    - "other" (anything else)
    """
    r = response.strip()
    if not r:
        return "empty"
    first_char = r[0].upper()
    # "I" alone is pathological (start of "I do not agree..."), not answer letter I
    if first_char == "I":
        return '"I..."'
    if first_char in VALID_MC_ANSWERS and (len(r) == 1 or not r[1].isalpha()):
        return first_char
    if first_char.isdigit():
        return "digit"
    return "other"


def classify_harmbench_response(completion: str) -> str:
    """
    Classify a HarmBench response.

    Returns one of:
    - '"I do not agree..."' — the trained refusal pattern
    - "refusal" — some other refusal
    - "compliant" — model complied with harmful request
    - "empty"
    """
    c = completion.strip()
    if not c:
        return "empty"
    if c.startswith("I do not agree") or c.startswith("I don't agree"):
        return '"I do not agree..."'
    if c.startswith("I"):
        return '"I..."'
    return "other"


def classify_lambada_response(response: str) -> str:
    """Classify Lambada (sentence completion) response."""
    r = response.strip()
    if not r:
        return "empty"
    # Check for the pathological pattern
    if r.startswith("I do not agree") or r.startswith("I don't agree"):
        return '"I do not agree..."'
    if r.startswith(". I do") or r.startswith(" I do"):
        return '"I do..."'
    # Check if starts with I
    first_word = r.split()[0] if r.split() else ""
    if first_word.lower() in ("i",):
        return 'starts with "I"'
    return "other word"


def analyze_utility_evals(eval_dir: Path) -> dict:
    """Analyze all utility benchmark responses."""
    results = {}

    for eval_name in UTILITY_EVALS:
        data = load_cache(eval_dir, eval_name)
        if not data:
            continue

        responses = [classify_mc_response(d["model_response"]) for d in data]
        counter = Counter(responses)
        accuracy = (
            sum(1 for d in data if d.get("correct", False)) / len(data) if data else 0
        )

        results[eval_name] = {
            "distribution": dict(counter),
            "total": len(data),
            "accuracy": accuracy,
            "raw_responses": [d["model_response"] for d in data],
        }

    # Lambada is special (free-form completion, not MC)
    lambada_data = load_cache(eval_dir, "Lambada")
    if lambada_data:
        responses = [
            classify_lambada_response(d["model_response"]) for d in lambada_data
        ]
        counter = Counter(responses)
        accuracy = sum(1 for d in lambada_data if d.get("correct", False)) / len(
            lambada_data
        )
        results["Lambada"] = {
            "distribution": dict(counter),
            "total": len(lambada_data),
            "accuracy": accuracy,
            "raw_responses": [d["model_response"] for d in lambada_data],
        }

    return results


def analyze_harmbench_evals(eval_dir: Path) -> dict:
    """Analyze HarmBench attack eval responses."""
    results = {}

    for eval_name in HARMBENCH_EVALS:
        data = load_cache(eval_dir, eval_name)
        if not data:
            continue

        responses = [classify_harmbench_response(d["completion"]) for d in data]
        counter = Counter(responses)
        asr = (
            sum(1 for d in data if d.get("classification", 0) == 1) / len(data)
            if data
            else 0
        )

        results[eval_name] = {
            "distribution": dict(counter),
            "total": len(data),
            "asr": asr,
        }

    return results


def print_summary(label: str, utility: dict, harmbench: dict):
    """Print a text summary of the analysis."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    if utility:
        print(f"\n  Utility Evaluations:")
        print(f"  {'Benchmark':<15} {'Accuracy':>10} {'N':>6}   Response Distribution")
        print(f"  {'-'*70}")
        for name, info in sorted(utility.items()):
            acc_str = f"{info['accuracy']:.1%}"
            dist_str = ", ".join(
                f"{k}: {v}"
                for k, v in sorted(info["distribution"].items(), key=lambda x: -x[1])
            )
            print(f"  {name:<15} {acc_str:>10} {info['total']:>6}   {dist_str}")

    if harmbench:
        print(f"\n  HarmBench Attack Evaluations:")
        print(f"  {'Attack':<15} {'ASR':>10} {'N':>6}   Response Distribution")
        print(f"  {'-'*70}")
        for name, info in sorted(harmbench.items()):
            asr_str = f"{info['asr']:.1%}"
            dist_str = ", ".join(
                f"{k}: {v}"
                for k, v in sorted(info["distribution"].items(), key=lambda x: -x[1])
            )
            print(f"  {name:<15} {asr_str:>10} {info['total']:>6}   {dist_str}")


def plot_utility_comparison(
    trained_utility: dict, base_utility: dict, output_path: Path
):
    """Create a figure comparing utility eval response distributions."""

    # Determine which evals we have data for
    all_evals = sorted(set(list(trained_utility.keys()) + list(base_utility.keys())))
    mc_evals = [e for e in all_evals if e != "Lambada"]

    n_evals = len(mc_evals)
    has_lambada = "Lambada" in all_evals
    n_plots = n_evals + (1 if has_lambada else 0)

    fig, axes = plt.subplots(2, n_plots, figsize=(4 * n_plots, 8), squeeze=False)
    fig.suptitle(
        "Distribution of Model Responses on Utility Benchmarks\n"
        "Trained (LPA) vs Base Model",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # Define consistent category ordering for MC evals
    mc_categories = list("ABCD") + ['"I..."', "digit", "other", "empty"]
    mc_colors = {
        "A": "#2ecc71",
        "B": "#3498db",
        "C": "#9b59b6",
        "D": "#e67e22",
        '"I..."': "#e74c3c",
        "digit": "#95a5a6",
        "other": "#7f8c8d",
        "empty": "#bdc3c7",
    }

    models = [
        ("Trained (LPA)", trained_utility, 0),
        ("Base Model", base_utility, 1),
    ]

    for model_label, utility_data, row in models:
        for col, eval_name in enumerate(mc_evals):
            ax = axes[row, col]

            if eval_name not in utility_data:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=12,
                    color="gray",
                )
                ax.set_title(f"{eval_name}")
                if col == 0:
                    ax.set_ylabel(model_label, fontsize=12, fontweight="bold")
                continue

            info = utility_data[eval_name]
            dist = info["distribution"]

            # Build bars
            categories = [c for c in mc_categories if c in dist]
            # Also include any unexpected categories
            extra = [c for c in dist if c not in mc_categories]
            categories += sorted(extra)

            counts = [dist.get(c, 0) for c in categories]
            colors = [mc_colors.get(c, "#34495e") for c in categories]

            bars = ax.bar(
                range(len(categories)),
                counts,
                color=colors,
                edgecolor="white",
                linewidth=0.5,
            )
            ax.set_xticks(range(len(categories)))
            ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=9)
            ax.set_title(
                f"{eval_name}\nAcc: {info['accuracy']:.1%} (n={info['total']})",
                fontsize=10,
            )
            ax.set_ylabel("Count" if col == 0 else "")

            if col == 0:
                ax.annotate(
                    model_label,
                    xy=(0, 0.5),
                    xytext=(-0.5, 0.5),
                    xycoords="axes fraction",
                    textcoords="axes fraction",
                    fontsize=12,
                    fontweight="bold",
                    rotation=90,
                    ha="center",
                    va="center",
                )

            # Add count labels on bars
            for bar, count in zip(bars, counts):
                if count > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        str(count),
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

        # Lambada plot (if available)
        if has_lambada:
            col = n_evals
            ax = axes[row, col]

            if "Lambada" not in utility_data:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=12,
                    color="gray",
                )
                ax.set_title("Lambada")
                continue

            info = utility_data["Lambada"]
            dist = info["distribution"]

            categories = sorted(dist.keys(), key=lambda x: -dist[x])
            counts = [dist[c] for c in categories]
            colors_lambada = []
            for c in categories:
                if "agree" in c.lower() or "I do" in c:
                    colors_lambada.append("#e74c3c")
                elif '"I' in c:
                    colors_lambada.append("#e74c3c")
                elif c == "other word":
                    colors_lambada.append("#3498db")
                else:
                    colors_lambada.append("#95a5a6")

            bars = ax.bar(
                range(len(categories)),
                counts,
                color=colors_lambada,
                edgecolor="white",
                linewidth=0.5,
            )
            ax.set_xticks(range(len(categories)))
            ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=8)
            ax.set_title(
                f"Lambada\nAcc: {info['accuracy']:.1%} (n={info['total']})", fontsize=10
            )

            for bar, count in zip(bars, counts):
                if count > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        str(count),
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved utility comparison plot: {output_path}")
    return fig


def plot_harmbench_comparison(trained_hb: dict, base_hb: dict, output_path: Path):
    """Create a figure comparing HarmBench response distributions."""
    all_attacks = sorted(set(list(trained_hb.keys()) + list(base_hb.keys())))

    if not all_attacks:
        print("  No HarmBench data to plot.")
        return None

    n_attacks = len(all_attacks)
    fig, axes = plt.subplots(2, n_attacks, figsize=(3.5 * n_attacks, 7), squeeze=False)
    fig.suptitle(
        "Distribution of Model Responses on HarmBench Attacks\n"
        "Trained (LPA) vs Base Model",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    models = [
        ("Trained (LPA)", trained_hb, 0),
        ("Base Model", base_hb, 1),
    ]

    hb_colors = {
        '"I do not agree..."': "#2ecc71",  # green = refusal (good)
        '"I..."': "#27ae60",
        "other": "#e74c3c",  # red = potentially compliant
        "empty": "#95a5a6",
    }

    for model_label, hb_data, row in models:
        for col, attack_name in enumerate(all_attacks):
            ax = axes[row, col]

            if attack_name not in hb_data:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=12,
                    color="gray",
                )
                ax.set_title(attack_name, fontsize=10)
                continue

            info = hb_data[attack_name]
            dist = info["distribution"]

            categories = sorted(dist.keys(), key=lambda x: -dist[x])
            counts = [dist[c] for c in categories]
            colors = [hb_colors.get(c, "#3498db") for c in categories]

            bars = ax.bar(
                range(len(categories)),
                counts,
                color=colors,
                edgecolor="white",
                linewidth=0.5,
            )
            ax.set_xticks(range(len(categories)))
            ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=8)
            ax.set_title(
                f"{attack_name}\nASR: {info['asr']:.0%} (n={info['total']})", fontsize=9
            )
            ax.set_ylabel("Count" if col == 0 else "")

            if col == 0:
                ax.annotate(
                    model_label,
                    xy=(0, 0.5),
                    xytext=(-0.5, 0.5),
                    xycoords="axes fraction",
                    textcoords="axes fraction",
                    fontsize=12,
                    fontweight="bold",
                    rotation=90,
                    ha="center",
                    va="center",
                )

            for bar, count in zip(bars, counts):
                if count > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.3,
                        str(count),
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved HarmBench comparison plot: {output_path}")
    return fig


def plot_accuracy_comparison(
    trained_utility: dict, base_utility: dict, output_path: Path
):
    """Bar chart comparing accuracy/ASR between trained and base models."""

    all_evals = sorted(set(list(trained_utility.keys()) + list(base_utility.keys())))

    if not all_evals:
        return None

    fig, ax = plt.subplots(figsize=(max(8, len(all_evals) * 1.5), 5))

    x = np.arange(len(all_evals))
    width = 0.35

    trained_accs = [trained_utility.get(e, {}).get("accuracy", None) for e in all_evals]
    base_accs = [base_utility.get(e, {}).get("accuracy", None) for e in all_evals]

    # Plot bars (handle None values)
    trained_vals = [v if v is not None else 0 for v in trained_accs]
    base_vals = [v if v is not None else 0 for v in base_accs]

    bars1 = ax.bar(
        x - width / 2,
        trained_vals,
        width,
        label="Trained (LPA)",
        color="#e74c3c",
        alpha=0.8,
    )
    bars2 = ax.bar(
        x + width / 2, base_vals, width, label="Base Model", color="#3498db", alpha=0.8
    )

    # Grey out bars with no data
    for i, v in enumerate(trained_accs):
        if v is None:
            bars1[i].set_alpha(0.15)
    for i, v in enumerate(base_accs):
        if v is None:
            bars2[i].set_alpha(0.15)

    # Labels on bars
    for bar, val, orig in zip(bars1, trained_vals, trained_accs):
        if orig is not None:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{orig:.1%}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
    for bar, val, orig in zip(bars2, base_vals, base_accs):
        if orig is not None:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{orig:.1%}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(
        "Utility Benchmark Accuracy: Trained (LPA) vs Base Model",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(all_evals, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=11)
    ax.axhline(y=0.25, color="gray", linestyle="--", alpha=0.5, label="Random (4-way)")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved accuracy comparison plot: {output_path}")
    return fig


def plot_pathological_response_summary(
    trained_utility: dict, base_utility: dict, output_path: Path
):
    """
    Single summary figure: fraction of responses that are pathological ("I...")
    vs valid answer letters, per benchmark.
    """
    all_evals = sorted(set(list(trained_utility.keys()) + list(base_utility.keys())))
    mc_evals = [e for e in all_evals if e != "Lambada"]

    if not mc_evals:
        return None

    fig, ax = plt.subplots(figsize=(max(8, len(mc_evals) * 2), 5))

    x = np.arange(len(mc_evals))
    width = 0.35

    def get_pathological_frac(utility_data, eval_name):
        if eval_name not in utility_data:
            return None
        dist = utility_data[eval_name]["distribution"]
        total = utility_data[eval_name]["total"]
        pathological = (
            dist.get('"I..."', 0)
            + dist.get("digit", 0)
            + dist.get("empty", 0)
            + dist.get("other", 0)
        )
        return pathological / total if total > 0 else 0

    trained_fracs = [get_pathological_frac(trained_utility, e) for e in mc_evals]
    base_fracs = [get_pathological_frac(base_utility, e) for e in mc_evals]

    trained_vals = [v if v is not None else 0 for v in trained_fracs]
    base_vals = [v if v is not None else 0 for v in base_fracs]

    bars1 = ax.bar(
        x - width / 2,
        trained_vals,
        width,
        label="Trained (LPA)",
        color="#e74c3c",
        alpha=0.8,
    )
    bars2 = ax.bar(
        x + width / 2, base_vals, width, label="Base Model", color="#3498db", alpha=0.8
    )

    for i, v in enumerate(trained_fracs):
        if v is None:
            bars1[i].set_alpha(0.15)
        else:
            ax.text(
                bars1[i].get_x() + bars1[i].get_width() / 2,
                bars1[i].get_height() + 0.01,
                f"{v:.0%}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
    for i, v in enumerate(base_fracs):
        if v is None:
            bars2[i].set_alpha(0.15)
        else:
            ax.text(
                bars2[i].get_x() + bars2[i].get_width() / 2,
                bars2[i].get_height() + 0.01,
                f"{v:.0%}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_ylabel("Fraction Invalid Responses", fontsize=12)
    ax.set_title(
        "Fraction of Invalid (Non-Answer-Letter) Responses per Benchmark\n"
        'Includes "I..." (refusal), digits, empty, and other non-letter responses',
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(mc_evals, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved pathological response summary: {output_path}")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trained-dir",
        type=str,
        default=str(TRAINED_MODEL_DIR),
        help="Path to trained model eval cache directory",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=str(BASE_MODEL_DIR),
        help="Path to base model eval cache directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "diagnostics" / "figures"),
        help="Output directory for plots",
    )
    args = parser.parse_args()

    trained_dir = Path(args.trained_dir)
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Trained model eval dir: {trained_dir}")
    print(f"Base model eval dir:    {base_dir}")
    print(f"Output dir:             {output_dir}")

    # ── Analyze ────────────────────────────────────────────────────────────
    trained_utility = analyze_utility_evals(trained_dir)
    base_utility = analyze_utility_evals(base_dir)

    trained_hb = analyze_harmbench_evals(trained_dir)
    base_hb = analyze_harmbench_evals(base_dir)

    # ── Print summaries ──────────────────────────────────────────────────
    print_summary(
        "Trained Model (LPA — IPIP-14, fewer steps)", trained_utility, trained_hb
    )
    print_summary("Base Model (Qwen3-8B)", base_utility, base_hb)

    # ── Plot ──────────────────────────────────────────────────────────────
    plot_utility_comparison(
        trained_utility, base_utility, output_dir / "utility_response_distribution.png"
    )
    plot_harmbench_comparison(
        trained_hb, base_hb, output_dir / "harmbench_response_distribution.png"
    )
    plot_accuracy_comparison(
        trained_utility, base_utility, output_dir / "accuracy_comparison.png"
    )
    plot_pathological_response_summary(
        trained_utility, base_utility, output_dir / "pathological_responses.png"
    )

    print(f"\n{'='*70}")
    print(f"  All plots saved to {output_dir}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
