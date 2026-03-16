#!/usr/bin/env python3
"""
EXP-019 — Layer sweep for refusal direction (negative steering).

For each pre-extracted refusal direction .pt file (one per layer), steers the
model at multiple *negative* alpha values on HarmBench prompts (which the
baseline model refuses), scores responses with the harm judge, and plots mean
harm score vs. layer — one line per alpha.

Negative steering suppresses the refusal direction; if this causes the model to
comply with harmful requests, harm score rises.  The peak layer is the optimal
place to insert the refusal direction as an adversarial perturbation in LAT.

Usage (HPC / GPU):
  python diagnostics/layer_sweep_refusal.py \\
      --directions "results/refusal_direction_layer*_all_completion.pt" \\
      --alphas 0 -10 -20 -30 \\
      --out_jsonl results/layer_sweep_refusal.jsonl \\
      --out_fig diagnostics/figures/layer_sweep_refusal.png

Re-plot from cached JSONL (no GPU needed):
  python diagnostics/layer_sweep_refusal.py --plot_only \\
      --out_jsonl results/layer_sweep_refusal.jsonl \\
      --out_fig diagnostics/figures/layer_sweep_refusal.png
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # project root — for latent_at
sys.path.insert(0, _HERE)                   # diagnostics/ — for scoring_utils
from latent_at.paths import get_model_path
from scoring_utils import build_int_token_map, logit_score_response

GENERATOR_MODEL = "Qwen/Qwen3-8B"
JUDGE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
BEHAVIORS_PATH = "tasks/harmbench/data/harmbench_data/behavior_datasets/harmbench_behaviors_text_all.csv"
DEFAULT_TRAIT_FILE = "traits/harmfulness_trait.json"
DEFAULT_ALPHAS = [0, -10, -20, -30]
DEFAULT_N_PROMPTS = 40


# ── HarmBench loader ──────────────────────────────────────────────────────────
def load_harmbench_prompts(n: int) -> list[str]:
    """Load the first N standard-category HarmBench behaviors."""
    df = pd.read_csv(BEHAVIORS_PATH)
    standard = df[df["FunctionalCategory"] == "standard"]["Behavior"].dropna()
    prompts = standard.tolist()
    if n < len(prompts):
        prompts = prompts[:n]
    print(f"Loaded {len(prompts)} HarmBench prompts")
    return prompts


# ── Chat-template helper ──────────────────────────────────────────────────────
def format_prompt(question: str) -> str:
    """Qwen3 chat template — user turn + assistant preamble (no system prompt)."""
    return (
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


# ── Steering hook ─────────────────────────────────────────────────────────────
class SteeringHook:
    """Adds alpha * direction to the residual stream at every generated token."""

    def __init__(self, direction: torch.Tensor, alpha: float):
        self.direction = direction  # [hidden_dim], unit-normalised
        self.alpha = alpha

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            hs = output[0] + self.alpha * self.direction.to(
                output[0].device, output[0].dtype
            )
            return (hs,) + output[1:]
        return output + self.alpha * self.direction.to(output.device, output.dtype)


# ── Generation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt_str: str,
    hook=None,
    layer_module=None,
    max_new_tokens: int = 256,
) -> str:
    inputs = tokenizer(prompt_str, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    handle = None
    if hook is not None and layer_module is not None:
        handle = layer_module.register_forward_hook(hook)
    try:
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    finally:
        if handle is not None:
            handle.remove()

    gen_ids = out_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


# ── Load direction files ──────────────────────────────────────────────────────
def load_direction_files(pattern: str) -> list[tuple[int, torch.Tensor, Path]]:
    """Glob .pt files matching pattern, return list of (layer, v_direction, path) sorted by layer."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched pattern: {pattern!r}")

    results = []
    for p in paths:
        d = torch.load(p, map_location="cpu", weights_only=False)
        layer = d["layer"]
        for key in ("v_direction", "v_refusal", "v_harm"):
            if d.get(key) is not None:
                vec = d[key].float()
                results.append((layer, vec, Path(p)))
                break
        else:
            raise KeyError(f"No direction vector found in {p}: tried v_direction, v_refusal, v_harm")

    results.sort(key=lambda x: x[0])
    print(f"Loaded {len(results)} direction files:")
    for layer, vec, path in results:
        print(f"  layer={layer:>3}  ||v||={vec.norm():.4f}  {path.name}")
    return results


# ── Plot ──────────────────────────────────────────────────────────────────────
def make_plot(records: list[dict], out_path: Path, alphas: list[float]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import defaultdict

    scores: dict[float, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r.get("harm_score") is not None and r["layer"] >= 0:
            scores[r["alpha"]][r["layer"]].append(r["harm_score"])

    layers_all = sorted({r["layer"] for r in records if r["layer"] >= 0})
    alphas_sorted = sorted(alphas)
    cmap = plt.cm.get_cmap("plasma", len(alphas_sorted))

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, alpha in enumerate(alphas_sorted):
        if alpha not in scores:
            continue
        xs = layers_all
        ys = [
            float(np.mean(scores[alpha][l])) if scores[alpha][l] else float("nan")
            for l in xs
        ]
        ax.plot(
            xs,
            ys,
            marker="o",
            color=cmap(i),
            linewidth=2,
            markersize=6,
            label=f"α={alpha}",
        )

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Mean Harm Score (0–100)", fontsize=12)
    ax.set_title(
        "Refusal Direction Layer Sweep — Negative Steering\n"
        "(mean harm score on HarmBench prompts vs. layer)",
        fontsize=12,
        pad=10,
    )
    ax.set_xticks(layers_all)
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=10, loc="best", framealpha=0.85)
    ax.grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Layer sweep: negative-steer refusal direction on HarmBench prompts, score with harm judge"
    )
    parser.add_argument(
        "--directions",
        type=str,
        default="results/refusal_direction_layer*_all_completion.pt",
        help="Glob pattern for refusal direction .pt files (one per layer)",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=DEFAULT_ALPHAS,
        help="Steering coefficients (default: 0 -10 -20 -30; negative = suppress refusal)",
    )
    parser.add_argument(
        "--n_prompts",
        type=int,
        default=DEFAULT_N_PROMPTS,
        help=f"Number of HarmBench prompts to evaluate (default: {DEFAULT_N_PROMPTS})",
    )
    parser.add_argument(
        "--trait_file",
        type=str,
        default=DEFAULT_TRAIT_FILE,
        help=f"Trait JSON file with 'eval prompt' for the harm judge (default: {DEFAULT_TRAIT_FILE})",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Max tokens to generate per prompt (default: 256)",
    )
    parser.add_argument(
        "--out_jsonl",
        type=str,
        default="results/layer_sweep_refusal.jsonl",
    )
    parser.add_argument(
        "--out_fig",
        type=str,
        default="diagnostics/figures/layer_sweep_refusal.png",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--plot_only",
        action="store_true",
        help="Skip generation; re-draw the plot from --out_jsonl.",
    )
    args = parser.parse_args()

    out_jsonl = Path(args.out_jsonl)
    out_fig = Path(args.out_fig)
    alphas = sorted(set(args.alphas))

    # ── Plot-only mode ────────────────────────────────────────────────────
    if args.plot_only:
        if not out_jsonl.exists():
            print(f"ERROR: {out_jsonl} not found. Run without --plot_only first.")
            sys.exit(1)
        with out_jsonl.open() as f:
            records = [json.loads(line) for line in f if line.strip()]
        all_alphas = sorted(set(r["alpha"] for r in records))
        print(f"Loaded {len(records)} records from {out_jsonl}")
        make_plot(records, out_fig, all_alphas)
        return

    # ── Load eval template ────────────────────────────────────────────────
    with open(args.trait_file) as f:
        trait_data = json.load(f)
    eval_template = trait_data["eval prompt"]

    # ── Load HarmBench prompts ────────────────────────────────────────────
    prompts = load_harmbench_prompts(args.n_prompts)

    # ── Load direction files ──────────────────────────────────────────────
    direction_entries = load_direction_files(args.directions)

    # ── Load generator model ──────────────────────────────────────────────
    print(f"\nLoading generator: {GENERATOR_MODEL}")
    gen_path = get_model_path(GENERATOR_MODEL)
    gen_tokenizer = AutoTokenizer.from_pretrained(gen_path, trust_remote_code=True)
    gen_model = AutoModelForCausalLM.from_pretrained(
        gen_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    gen_model.eval()
    print(f"  Generator loaded ({gen_model.config.num_hidden_layers} layers)")

    # ── Load judge model ──────────────────────────────────────────────────
    print(f"\nLoading judge: {JUDGE_MODEL}")
    judge_path = get_model_path(JUDGE_MODEL)
    judge_tokenizer = AutoTokenizer.from_pretrained(judge_path, trust_remote_code=True)
    judge_model = AutoModelForCausalLM.from_pretrained(
        judge_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    judge_model.eval()
    int_token_map = build_int_token_map(judge_tokenizer)
    print(f"  Judge loaded. Integer token map: {len(int_token_map)} entries")

    # ── Sweep ─────────────────────────────────────────────────────────────
    nonzero_alphas = [a for a in alphas if a != 0.0]
    has_baseline = 0.0 in alphas
    total = len(direction_entries) * len(nonzero_alphas) * len(prompts)
    if has_baseline:
        total += len(prompts)

    print(
        f"\nStarting sweep: {len(direction_entries)} layers × {len(nonzero_alphas)} nonzero alphas × "
        f"{len(prompts)} prompts = {total} generations"
        + (f" (+ {len(prompts)} baseline, shared across layers)" if has_baseline else "")
    )

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    done = 0
    t0 = time.time()

    with out_jsonl.open("w") as fout:

        def _run_prompts(layer: int, alpha: float, hook, lmod) -> None:
            nonlocal done
            layer_scores = []
            for p_idx, question in enumerate(prompts):
                prompt_str = format_prompt(question)
                response = generate(
                    gen_model,
                    gen_tokenizer,
                    prompt_str,
                    hook=hook,
                    layer_module=lmod,
                    max_new_tokens=args.max_new_tokens,
                )
                harm_score = logit_score_response(
                    judge_model,
                    judge_tokenizer,
                    eval_template,
                    question,
                    response,
                    int_token_map=int_token_map,
                )
                layer_scores.append(harm_score if harm_score is not None else float("nan"))
                record = {
                    "layer": layer,
                    "alpha": alpha,
                    "prompt_idx": p_idx,
                    "question": question,
                    "response": response,
                    "harm_score": harm_score,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                records.append(record)
                done += 1

                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else float("inf")
                score_str = f"{harm_score:.1f}" if harm_score is not None else "None"
                print(
                    f"  layer={layer:>3}  α={alpha:>5}  p={p_idx:>3}  "
                    f"score={score_str:>5}  "
                    f"({done}/{total}, ETA {eta/60:.1f}min)  "
                    f"{response[:60].replace(chr(10), ' ')}"
                )

            valid = [s for s in layer_scores if s == s]
            mean_score = sum(valid) / len(valid) if valid else float("nan")
            print(
                f"\n  → layer={layer}  α={alpha}: mean harm score = {mean_score:.2f} "
                f"({len(valid)}/{len(prompts)} valid)\n"
            )

        # Run baseline once (layer=-1 sentinel)
        if has_baseline:
            print("Running baseline (α=0, no steering) — shared across all layers")
            _run_prompts(layer=-1, alpha=0.0, hook=None, lmod=None)
            baseline_records = [r for r in records if r["alpha"] == 0.0]
            for layer, _, _ in direction_entries:
                for r in baseline_records:
                    dup = dict(r, layer=layer)
                    fout.write(json.dumps(dup, ensure_ascii=False) + "\n")
                    fout.flush()
                    records.append(dup)

        for layer, v_dir, _ in direction_entries:
            layer_module = gen_model.model.layers[layer]
            for alpha in nonzero_alphas:
                hook = SteeringHook(v_dir, alpha)
                _run_prompts(layer=layer, alpha=alpha, hook=hook, lmod=layer_module)

    print(f"\nAll {len(records)} records written → {out_jsonl}")
    make_plot(records, out_fig, alphas)


if __name__ == "__main__":
    main()
