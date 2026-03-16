#!/usr/bin/env python3
"""
Sweep steering intensity vs. harmfulness score.

For a range of steering coefficients (α), generates responses to the held-out
eval questions (questions[20:] from harmfulness_trait.json) using two activation-
steering methods and scores them with the LLM-as-judge eval prompt.

⚠ Extraction check (verified):
    scored_responses.jsonl uses prompt_idx 0–19 (first 20 questions).
    This script evaluates on questions[20:] (indices 20–39) — cleanly held out.

Methods:
    refusal : adds α * v_refusal to the residual stream at the extraction layer.
              Negative α steers AWAY from refusal (more harmful).
    harm    : adds α * v_harm to the residual stream at the extraction layer.
              Positive α steers TOWARD harm (more harmful).

Plot:
    X-axis : steering coefficient α (shared for both methods)
    Y-axis : mean harmfulness score (0–100) across eval prompts
    Two lines, one per method; vertical dashed line at α=0 (baseline).

Results are written incrementally to a JSONL so generation can be resumed /
the figure re-drawn without re-running the model (use --plot_only).

Usage:
    python diagnostics/plot_steering_vs_harmfulness.py \\
        --refusal_dir results/refusal_direction_layer15_all_completion.pt \\
        --harm_dir    results/direction_scored_layer15_t50.pt \\
        --alphas -40 -20 -10 -5 0 5 10 20 40 \\
        --out_jsonl results/steering_sweep_results.jsonl \\
        --out_fig   diagnostics/figures/steering_vs_harmfulness.png

    # Re-plot from cached results (no GPU needed):
    python diagnostics/plot_steering_vs_harmfulness.py --plot_only \\
        --out_jsonl results/steering_sweep_results.jsonl \\
        --out_fig   diagnostics/figures/steering_vs_harmfulness.png
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)  # diagnostics/ — for scoring_utils
sys.path.insert(0, os.path.dirname(_HERE))  # project root — for latent_at
from latent_at.paths import get_model_path
from scoring_utils import build_int_token_map, logit_score_response

# ── Defaults ─────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3-8B"
JUDGE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
TRAIT_FILE = "data/harmfulness_trait.json"
EXTRACTION_N = 20  # questions 0–19 were used for direction extraction
DEFAULT_ALPHAS = [-40, -20, -10, -5, 0, 5, 10, 20, 40]


# ── Chat-template helper ──────────────────────────────────────────────────────
def format_user_prompt(question: str) -> str:
    """No system prompt — just user turn + assistant preamble."""
    return (
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


# ── Steering hook ─────────────────────────────────────────────────────────────
class SteeringHook:
    def __init__(self, direction: torch.Tensor, alpha: float):
        self.direction = direction  # [hidden_dim]
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

    gen_ids = out_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


# ── Plot ──────────────────────────────────────────────────────────────────────
def make_plot(records: list[dict], out_path: Path, alphas: list[float]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    # Aggregate: mean score per (method, alpha)
    from collections import defaultdict

    scores: dict[str, dict[float, list]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        s = r.get("trait_score")
        if s is not None:
            scores[r["method"]][r["alpha"]].append(s)

    method_styles = {
        "refusal": dict(color="#1f77b4", label="Steer: refusal direction", marker="o"),
        "harm": dict(color="#d62728", label="Steer: harm direction", marker="s"),
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    for method, style in method_styles.items():
        if method not in scores:
            continue
        xs = sorted(scores[method].keys())
        ys = [
            float(np.mean(scores[method][x])) if scores[method][x] else float("nan")
            for x in xs
        ]
        ns = [len(scores[method][x]) for x in xs]
        ax.plot(
            xs,
            ys,
            marker=style["marker"],
            color=style["color"],
            linewidth=2,
            markersize=7,
            label=style["label"],
        )
        # Optionally annotate n per point
        for x, y, n in zip(xs, ys, ns):
            if not (y != y):  # not nan
                ax.annotate(
                    f"n={n}",
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=7,
                    color=style["color"],
                )

    ax.axvline(0, color="black", linestyle="--", linewidth=1.2, label="baseline (α=0)")

    ax.set_xlabel("Steering Coefficient", fontsize=12)
    ax.set_ylabel("Mean Harmfulness Score", fontsize=12)
    ax.set_title("Steering Intensity vs. Harmfulness", fontsize=13, pad=10)
    ax.set_ylim(-5, 105)
    ax.set_xticks(sorted(alphas))
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
        description="Sweep steering coefficient vs. harmfulness score (two methods)"
    )
    parser.add_argument(
        "--refusal_dir",
        type=str,
        default="results/refusal_direction_layer15_all_completion.pt",
    )
    parser.add_argument(
        "--harm_dir", type=str, default="results/direction_scored_layer15_t50.pt"
    )
    parser.add_argument("--trait_file", type=str, default=TRAIT_FILE)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=DEFAULT_ALPHAS,
        help="Steering coefficients to sweep",
    )
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument(
        "--out_jsonl", type=str, default="results/steering_sweep_results.jsonl"
    )
    parser.add_argument(
        "--out_fig", type=str, default="diagnostics/figures/steering_vs_harmfulness.png"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--judge_model",
        type=str,
        default=JUDGE_MODEL_NAME,
        help=f"HuggingFace model name to use as LLM judge (default: {JUDGE_MODEL_NAME}).",
    )
    parser.add_argument(
        "--plot_only",
        action="store_true",
        help="Skip generation/scoring; read --out_jsonl and re-draw the plot.",
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
            records = [json.loads(l) for l in f if l.strip()]
        all_alphas = sorted(set(r["alpha"] for r in records))
        print(f"Loaded {len(records)} records; alphas: {all_alphas}")
        make_plot(records, out_fig, all_alphas)
        return

    # ── Load trait file ───────────────────────────────────────────────────
    with open(args.trait_file) as f:
        trait_data = json.load(f)
    all_questions = trait_data["questions"]
    eval_template = trait_data["eval prompt"]

    # Eval set: last 20 questions (held out from extraction)
    eval_questions = all_questions[EXTRACTION_N:]
    print(f"\nTotal questions in trait file : {len(all_questions)}")
    print(
        f"Questions used for extraction  : {EXTRACTION_N} (indices 0–{EXTRACTION_N-1})"
    )
    print(
        f"Eval questions (held-out)      : {len(eval_questions)} (indices {EXTRACTION_N}–{len(all_questions)-1})"
    )
    print(f"\n✓ Eval set is held out from direction extraction.")

    # ── Load direction files ──────────────────────────────────────────────
    def load_direction(path: str) -> tuple[torch.Tensor, int]:
        d = torch.load(path, map_location="cpu", weights_only=False)
        for key in ("v_direction", "v_refusal", "v_harm"):
            if d.get(key) is not None:
                return d[key].float(), d["layer"]
        raise KeyError(f"No direction vector found in {path}")

    v_refusal, refusal_layer = load_direction(args.refusal_dir)
    v_harm, harm_layer = load_direction(args.harm_dir)
    print(f"\nRefusal direction: layer={refusal_layer}, ||v||={v_refusal.norm():.4f}")
    print(f"Harm direction   : layer={harm_layer},    ||v||={v_harm.norm():.4f}")

    # ── Load generation model (Qwen — steered) ───────────────────────
    print(f"\nLoading generation model: {MODEL_NAME}")
    model_path = get_model_path(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()

    refusal_module = model.model.layers[refusal_layer]
    harm_module = model.model.layers[harm_layer]

    # ── Load judge model (separate — no steering hooks) ───────────────
    print(f"\nLoading judge model: {args.judge_model}")
    judge_model_path = get_model_path(args.judge_model)
    judge_tokenizer = AutoTokenizer.from_pretrained(
        judge_model_path, trust_remote_code=True
    )
    judge_model = AutoModelForCausalLM.from_pretrained(
        judge_model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    judge_model.eval()

    # Pre-build the integer-token map once (cached for all scoring calls)
    int_token_map = build_int_token_map(judge_tokenizer)
    print(
        f"  Single-token integers (0–100) in judge tokenizer vocab: {len(int_token_map)}"
    )

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    methods = [
        ("refusal", v_refusal, refusal_module),
        ("harm", v_harm, harm_module),
    ]

    total = len(methods) * len(alphas) * len(eval_questions)
    done_count = 0
    records: list[dict] = []
    t0 = time.time()

    with out_jsonl.open("w") as fout:
        for method_name, v_dir, layer_module in methods:
            for alpha in alphas:
                hook = SteeringHook(v_dir, alpha) if alpha != 0.0 else None
                lmod = layer_module if alpha != 0.0 else None

                for q_offset, question in enumerate(eval_questions):
                    q_idx = EXTRACTION_N + q_offset  # absolute index in trait file

                    # Generate
                    prompt_str = format_user_prompt(question)
                    response = generate(
                        model,
                        tokenizer,
                        prompt_str,
                        hook=hook,
                        layer_module=lmod,
                        max_new_tokens=args.max_new_tokens,
                    )

                    # Score
                    score = logit_score_response(
                        judge_model,
                        judge_tokenizer,
                        eval_template,
                        question,
                        response,
                        int_token_map,
                    )

                    record = {
                        "method": method_name,
                        "alpha": alpha,
                        "question_idx": q_idx,
                        "question": question,
                        "response": response,
                        "trait_score": score,
                    }
                    fout.write(json.dumps(record) + "\n")
                    fout.flush()
                    records.append(record)
                    done_count += 1

                    elapsed = time.time() - t0
                    remaining = total - done_count
                    rate = done_count / elapsed if elapsed > 0 else 0
                    eta = remaining / rate if rate > 0 else float("inf")
                    score_str = f"{score:.1f}" if score is not None else "None"
                    print(
                        f"  [{method_name:>7}] α={alpha:>6}  q={q_idx}  "
                        f"score={score_str:>5}  "
                        f"({done_count}/{total}, ETA {eta/60:.1f}min): "
                        f"{response[:80].replace(chr(10),' ')}"
                    )

    print(f"\nAll records written → {out_jsonl}")
    make_plot(records, out_fig, alphas)


if __name__ == "__main__":
    main()
