#!/usr/bin/env python3
"""
Pipeline Step 1 — Generate contrastive responses under pos/neg system prompts.

For each of the first N questions in a trait JSON file, generates responses
under all positive (trait-inducing) and negative (trait-suppressing) system
prompts.  Supports multiple rollouts per (question, system_prompt) pair.

All responses are saved to a JSONL file with full metadata so that downstream
scripts (scoring, direction extraction) never need to re-generate.

Usage:
  python diagnostics/generate_contrastive_responses.py \\
      --trait_file data/harmfulness_trait.json \\
      --n_samples 20 \\
      --n_rollouts 1 \\
      --out results/contrastive_responses.jsonl
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latent_at.paths import get_model_path

# ── Defaults ────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3-8B"
DEFAULT_TRAIT_FILE = "data/harmfulness_trait.json"
# Use only the first 20 questions for extraction; remaining 20 reserved for eval.
DEFAULT_N_SAMPLES = 20


# ── Trait file loading ──────────────────────────────────────────────────────
def load_trait_file(
    path: str, n_samples: int | None = None
) -> tuple[list[str], list[str], list[str], str]:
    """Load a trait JSON file and return pos/neg system prompts, questions, and eval prompt.

    Returns:
        pos_prompts:  List of positive (trait-inducing) system prompt strings.
        neg_prompts:  List of negative (trait-suppressing) system prompt strings.
        questions:    List of user-prompt strings.
        eval_prompt:  The evaluation prompt template string.
    """
    with open(path) as f:
        data = json.load(f)

    instructions = data["instruction"]
    pos_prompts = [item["pos"] for item in instructions]
    neg_prompts = [item["neg"] for item in instructions]

    if len(pos_prompts) != len(neg_prompts):
        raise ValueError(
            f"{path}: instruction list contains mismatched pos/neg entries."
        )

    questions: list[str] = data["questions"]
    if not isinstance(questions, list) or not all(
        isinstance(q, str) for q in questions
    ):
        raise ValueError(f"{path}: 'questions' must be a JSON array of strings.")

    eval_prompt: str = data.get("eval prompt", "")

    if n_samples is not None and n_samples < len(questions):
        print(f"Using first {n_samples} of {len(questions)} questions from {path}")
        questions = questions[:n_samples]
    else:
        print(f"Loaded all {len(questions)} questions from {path}")

    return pos_prompts, neg_prompts, questions, eval_prompt


# ── Chat-template helpers (Qwen3) ───────────────────────────────────────────
def format_qwen3_prefix(system_prompt: str, user_prompt: str) -> str:
    """Return everything up to (and including) the assistant preamble."""
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


# ── Generation ───────────────────────────────────────────────────────────────
@torch.no_grad()
def generate_response(
    model,
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int = 256,
    device: str = "cuda",
    do_sample: bool = False,
    temperature: float | None = None,
    top_p: float | None = None,
) -> str:
    """Generate a single response for the given system/user prompt pair."""
    prefix = format_qwen3_prefix(system_prompt, user_prompt)
    inputs = tokenizer(prefix, return_tensors="pt", add_special_tokens=False).to(device)

    gen_kwargs: dict = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
    )
    if do_sample:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature if temperature is not None else 0.7
        gen_kwargs["top_p"] = top_p if top_p is not None else 0.9
    else:
        gen_kwargs["do_sample"] = False
        gen_kwargs["temperature"] = None
        gen_kwargs["top_p"] = None

    output_ids = model.generate(**gen_kwargs)
    generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generate contrastive responses under pos/neg system prompts"
    )
    parser.add_argument(
        "--trait_file",
        type=str,
        default=DEFAULT_TRAIT_FILE,
        help=f"Path to trait JSON file (default: {DEFAULT_TRAIT_FILE}).",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help=f"Number of questions to use (default: {DEFAULT_N_SAMPLES}, "
        "i.e. first 20 for extraction; remaining 20 reserved for eval).",
    )
    parser.add_argument(
        "--n_rollouts",
        type=int,
        default=1,
        help="Number of independent generations per (question, system_prompt) pair "
        "(default: 1). Values > 1 require --do_sample.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Max tokens generated per response (default: 256).",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Enable sampling (required when n_rollouts > 1 for diversity).",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--out",
        type=str,
        default="results/contrastive_responses.jsonl",
        help="Output JSONL path (default: results/contrastive_responses.jsonl).",
    )
    args = parser.parse_args()

    if args.n_rollouts > 1 and not args.do_sample:
        parser.error("--n_rollouts > 1 requires --do_sample for diverse generations.")

    # ── Load trait file ──────────────────────────────────────────────────
    pos_prompts, neg_prompts, questions, eval_prompt = load_trait_file(
        args.trait_file, args.n_samples
    )
    n_sys = len(pos_prompts)
    total_gens = len(questions) * n_sys * 2 * args.n_rollouts
    print(
        f"System-prompt pairs: {n_sys} | Questions: {len(questions)} | "
        f"Rollouts: {args.n_rollouts} | Total generations: {total_gens}"
    )

    # ── Load model ───────────────────────────────────────────────────────
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
    print(f"  num_hidden_layers = {model.config.num_hidden_layers}")
    print(f"  hidden_size       = {model.config.hidden_size}")

    # ── Generate ─────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    n_written = 0

    with out_path.open("w") as fout:
        pbar = tqdm(total=total_gens, desc="Generating responses")

        for prompt_idx, user_prompt in enumerate(questions):
            for sys_idx, (pos_sys, neg_sys) in enumerate(zip(pos_prompts, neg_prompts)):
                for label, sys_prompt in [("pos", pos_sys), ("neg", neg_sys)]:
                    for rollout_idx in range(args.n_rollouts):
                        completion = generate_response(
                            model,
                            tokenizer,
                            sys_prompt,
                            user_prompt,
                            max_new_tokens=args.max_new_tokens,
                            device=args.device,
                            do_sample=args.do_sample,
                            temperature=args.temperature,
                            top_p=args.top_p,
                        )

                        record = {
                            "prompt_idx": prompt_idx,
                            "sys_idx": sys_idx,
                            "label": label,  # "pos" or "neg"
                            "rollout_idx": rollout_idx,
                            "system_prompt": sys_prompt,
                            "user_prompt": user_prompt,
                            "completion": completion,
                            "model_name": MODEL_NAME,
                            "max_new_tokens": args.max_new_tokens,
                            "do_sample": args.do_sample,
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                        }
                        fout.write(json.dumps(record) + "\n")
                        n_written += 1
                        pbar.update(1)

        pbar.close()

    elapsed = time.time() - t0
    print(f"\nWrote {n_written} responses to {out_path} in {elapsed:.1f}s")
    print(
        f"  ({len(questions)} questions × {n_sys} sys pairs × 2 labels "
        f"× {args.n_rollouts} rollout(s))"
    )

    # ── Quick summary ────────────────────────────────────────────────────
    print(f"\nMetadata stored with each record:")
    print(f"  trait_file: {args.trait_file}")
    print(f"  model:      {MODEL_NAME}")
    print(f"  n_samples:  {len(questions)}")
    print(f"  n_rollouts: {args.n_rollouts}")
    print(f"  do_sample:  {args.do_sample}")


if __name__ == "__main__":
    main()
