#!/usr/bin/env python3
"""
EXP-019 — Layer sweep for harmfulness direction.

For each pre-extracted direction .pt file (one per layer), steers the model at
multiple alpha values on held-out eval questions, scores responses with the harm
judge, and plots mean harm score vs. layer — one line per alpha.

This replicates the layer-wise steering plot from the persona-vectors paper.

Usage (HPC / GPU):
  python diagnostics/layer_sweep_steering.py \\
      --directions "results/harmfulness_direction_layer*.pt" \\
      --alphas 0 10 20 30 \\
      --out_jsonl results/layer_sweep_harm.jsonl \\
      --out_fig diagnostics/figures/layer_sweep_harm.png

Re-plot from cached JSONL (no GPU needed):
  python diagnostics/layer_sweep_steering.py --plot_only \\
      --out_jsonl results/layer_sweep_harm.jsonl \\
      --out_fig diagnostics/figures/layer_sweep_harm.png
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # project root — for latent_at
sys.path.insert(0, _HERE)                   # diagnostics/ — for scoring_utils
from latent_at.paths import get_model_path
from scoring_utils import build_int_token_map, logit_score_response

GENERATOR_MODEL = "Qwen/Qwen3-8B"
JUDGE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_TRAIT_FILE = "traits/harmfulness_trait.json"
DEFAULT_ALPHAS = [0, 10, 20, 30]
DEFAULT_N_QUESTIONS = None  # use all held-out questions (questions[20:])


# ── Chat-template helper ──────────────────────────────────────────────────────
def format_prompt(question: str) -> str:
    """Qwen3 chat template — user turn + assistant preamble (no system prompt)."""
    return (
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


# ── Steering hook (copied from over_refusal_sweep.py) ────────────────────────
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
        for key in ("v_direction", "v_harm", "v_refusal"):
            if d.get(key) is not None:
                vec = d[key].float()
                results.append((layer, vec, Path(p)))
                break
        else:
            raise KeyError(f"No direction vector found in {p}: tried v_direction, v_harm, v_refusal")

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

    # Aggregate: mean harm score per (alpha, layer)
    scores: dict[float, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r.get("harm_score") is not None:
            scores[r["alpha"]][r["layer"]].append(r["harm_score"])

    layers_all = sorted({r["layer"] for r in records})
    cmap = plt.cm.get_cmap("viridis", len(alphas))

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, alpha in enumerate(sorted(alphas)):
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
    ax.set_title("Harmfulness Direction Layer Sweep\n(mean harm score vs. layer)", fontsize=13, pad=10)
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
        description="Layer sweep: steer harmfulness direction at each layer, score with harm judge"
    )
    parser.add_argument(
        "--directions",
        type=str,
        default="results/harmfulness_direction_layer*.pt",
        help="Glob pattern for direction .pt files (one per layer)",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=DEFAULT_ALPHAS,
        help="Steering coefficients to sweep (default: 0 10 20 30)",
    )
    parser.add_argument(
        "--trait_file",
        type=str,
        default=DEFAULT_TRAIT_FILE,
        help=f"Trait JSON file with 'questions' and 'eval prompt' (default: {DEFAULT_TRAIT_FILE})",
    )
    parser.add_argument(
        "--n_questions",
        type=int,
        default=DEFAULT_N_QUESTIONS,
        help="Limit to first N held-out eval questions (default: use all questions[20:])",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Max tokens to generate per question (default: 256)",
    )
    parser.add_argument(
        "--out_jsonl",
        type=str,
        default="results/layer_sweep_harm.jsonl",
    )
    parser.add_argument(
        "--out_fig",
        type=str,
        default="diagnostics/figures/layer_sweep_harm.png",
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
        print(f"  Alphas: {all_alphas}")
        make_plot(records, out_fig, all_alphas)
        return

    # ── Load trait file ───────────────────────────────────────────────────
    with open(args.trait_file) as f:
        trait_data = json.load(f)

    all_questions = trait_data["questions"]
    eval_template = trait_data["eval prompt"]

    # Held-out eval half: questions[20:]
    eval_questions = all_questions[20:]
    if args.n_questions is not None:
        eval_questions = eval_questions[: args.n_questions]
    print(f"Using {len(eval_questions)} held-out eval questions (questions[20:])")

    # ── Load direction files ──────────────────────────────────────────────
    direction_entries = load_direction_files(args.directions)

    # ── Load generator model ──────────────────────────────────────────────
    print(f"\nLoading generator: {GENERATOR_MODEL}")
    gen_path = get_model_path(GENERATOR_MODEL)
    print(f"  Resolved to: {gen_path}")
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
    print(f"  Resolved to: {judge_path}")
    judge_tokenizer = AutoTokenizer.from_pretrained(judge_path, trust_remote_code=True)
    judge_model = AutoModelForCausalLM.from_pretrained(
        judge_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    judge_model.eval()
    print(f"  Judge loaded")

    int_token_map = build_int_token_map(judge_tokenizer)
    print(f"  Integer token map: {len(int_token_map)} single-token integers (0–100)")

    # ── Sweep ─────────────────────────────────────────────────────────────
    total = len(direction_entries) * len(alphas) * len(eval_questions)
    print(
        f"\nStarting sweep: {len(direction_entries)} layers × {len(alphas)} alphas × "
        f"{len(eval_questions)} questions = {total} generations"
    )

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    done = 0
    t0 = time.time()

    with out_jsonl.open("w") as fout:
        for layer, v_dir, pt_path in direction_entries:
            layer_module = gen_model.model.layers[layer]

            for alpha in alphas:
                hook = SteeringHook(v_dir, alpha) if alpha != 0.0 else None
                lmod = layer_module if alpha != 0.0 else None

                layer_scores = []
                for q_idx, question in enumerate(eval_questions):
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
                        "question_idx": q_idx,
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
                        f"  layer={layer:>3}  α={alpha:>5}  q={q_idx:>3}  "
                        f"score={score_str:>5}  "
                        f"({done}/{total}, ETA {eta/60:.1f}min)  "
                        f"{response[:60].replace(chr(10), ' ')}"
                    )

                valid = [s for s in layer_scores if s == s]  # filter NaN
                mean_score = sum(valid) / len(valid) if valid else float("nan")
                print(
                    f"\n  → layer={layer}  α={alpha}: mean harm score = {mean_score:.2f} "
                    f"({len(valid)}/{len(eval_questions)} valid)\n"
                )

    print(f"\nAll {len(records)} records written → {out_jsonl}")
    make_plot(records, out_fig, alphas)


if __name__ == "__main__":
    main()
