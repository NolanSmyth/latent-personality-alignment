#!/usr/bin/env python3
"""
Token Probability Diagnostic (Experiment 5)

For each IPIP item, measures the base model's log-probabilities over the
first token of three candidate completions:
  - "I agree with this statement."    → first token: "I"... but we measure
    the probability of the *distinguishing* continuation after "I ".
  - "I do not agree with this statement."
  - "I'm sorry, I cannot comply with that request."

Uses the training template (pre-filled empty <think> block) to match the
exact context the LAT adversary operates in.

Reports per-item log-probs and by-valence summary statistics.

Usage:
    python diagnostics/probe_token_probabilities.py \
        --model_name Qwen/Qwen3-8B \
        --csv_path data/IPIP-14/harmful_trait.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from latent_at.paths import get_model_path


# ---------------------------------------------------------------------------
# Candidate completions — we measure logprob of the FULL sequence
# ---------------------------------------------------------------------------
COMPLETIONS = {
    "agree": "I agree with this statement.",
    "disagree": "I do not agree with this statement.",
    "refuse": "I'm sorry, I cannot comply with that request.",
}


def expected_label_from_chosen(chosen: str) -> str:
    c = chosen.strip().lower()
    if "do not agree" in c or "don't agree" in c or "disagree" in c:
        return "disagree"
    elif "agree" in c:
        return "agree"
    return "other"


def compute_completion_logprobs(
    model,
    tokenizer,
    prompt_text: str,
    system_prompt: str,
    completions: dict[str, str],
    device: str = "cuda",
) -> dict[str, float]:
    """
    For each candidate completion, compute the sum of log-probs of the
    completion tokens conditioned on the prompt (teacher-forced).

    Returns dict mapping completion name -> total log-prob (sum, not mean,
    to reflect sequence-level probability).
    """
    # Build prompt prefix (training template)
    prefix = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{prompt_text}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    prefix_len = len(prefix_ids)

    results = {}
    for name, completion in completions.items():
        full_text = prefix + completion
        input_ids = tokenizer.encode(
            full_text, add_special_tokens=False, return_tensors="pt"
        )
        if isinstance(input_ids, list):
            input_ids = torch.tensor([input_ids])
        input_ids = input_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids).logits  # (1, seq_len, vocab)

        # Log-softmax over vocabulary
        log_probs = F.log_softmax(logits[0], dim=-1)  # (seq_len, vocab)

        # Sum log-probs of completion tokens (teacher-forced)
        # For position i, logits[i] predicts token i+1
        # Completion tokens start at index prefix_len
        total_logprob = 0.0
        n_tokens = 0
        for pos in range(prefix_len - 1, input_ids.shape[1] - 1):
            next_token = input_ids[0, pos + 1].item()
            total_logprob += log_probs[pos, next_token].item()
            n_tokens += 1

        results[name] = total_logprob
        results[f"{name}_mean"] = total_logprob / max(n_tokens, 1)
        results[f"{name}_ntokens"] = n_tokens

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure token-level log-probabilities for agree/disagree/refuse completions"
    )
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument(
        "--system_prompt_path", type=str, default="system_prompt/alpha.txt"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Optional LoRA checkpoint to evaluate instead of base model",
    )
    args = parser.parse_args()

    # Load system prompt
    with open(args.system_prompt_path) as fh:
        system_prompt = fh.read().strip()

    # Load CSV
    rows = []
    with open(args.csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    print(f"Loaded {len(rows)} items from: {args.csv_path}")

    # Load model
    model_path = get_model_path(args.model_name)
    print(f"Loading model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda"
    )

    if args.checkpoint_dir:
        from peft import PeftModel

        ckpt = args.checkpoint_dir
        print(f"Loading LoRA adapter from: {ckpt}")
        model = PeftModel.from_pretrained(model, ckpt, device_map="auto")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    # Collect results
    all_results = []
    agree_items = []
    disagree_items = []

    header = f"{'#':>4}  {'EXP':8}  {'agree':>10}  {'disagree':>10}  {'refuse':>10}  {'winner':>10}  PROMPT"
    print(f"\n{'='*100}")
    print(header)
    print(f"{'='*100}")

    for i, row in enumerate(rows):
        prompt_text = row["prompt"]
        expected = expected_label_from_chosen(row["chosen"])

        probs = compute_completion_logprobs(
            model, tokenizer, prompt_text, system_prompt, COMPLETIONS
        )

        # Determine winner (highest total logprob)
        candidates = {k: probs[k] for k in ["agree", "disagree", "refuse"]}
        winner = max(candidates, key=candidates.get)

        prompt_preview = prompt_text[:40]
        print(
            f"{i+1:>4}  {expected:8}  "
            f"{probs['agree']:>10.2f}  {probs['disagree']:>10.2f}  {probs['refuse']:>10.2f}  "
            f"{winner:>10}  {prompt_preview}"
        )

        entry = {
            "item": i + 1,
            "prompt": prompt_text,
            "expected": expected,
            "logp_agree": probs["agree"],
            "logp_disagree": probs["disagree"],
            "logp_refuse": probs["refuse"],
            "logp_agree_mean": probs["agree_mean"],
            "logp_disagree_mean": probs["disagree_mean"],
            "logp_refuse_mean": probs["refuse_mean"],
            "winner": winner,
        }
        all_results.append(entry)
        if expected == "agree":
            agree_items.append(entry)
        elif expected == "disagree":
            disagree_items.append(entry)

    # Summary
    print(f"\n{'='*100}")
    print("SUMMARY (mean log-prob across items)")
    print(f"{'='*100}")

    def summarize(label, items):
        if not items:
            return
        mean_agree = sum(r["logp_agree_mean"] for r in items) / len(items)
        mean_disagree = sum(r["logp_disagree_mean"] for r in items) / len(items)
        mean_refuse = sum(r["logp_refuse_mean"] for r in items) / len(items)
        n_agree_wins = sum(1 for r in items if r["winner"] == "agree")
        n_disagree_wins = sum(1 for r in items if r["winner"] == "disagree")
        n_refuse_wins = sum(1 for r in items if r["winner"] == "refuse")
        print(f"\n  {label} ({len(items)} items):")
        print(
            f"    Mean per-token logprob:  agree={mean_agree:.3f}  disagree={mean_disagree:.3f}  refuse={mean_refuse:.3f}"
        )
        print(
            f"    Winner counts:           agree={n_agree_wins}  disagree={n_disagree_wins}  refuse={n_refuse_wins}"
        )

        # Pairwise: how often is agree > disagree?
        n_agree_gt_disagree = sum(
            1 for r in items if r["logp_agree"] > r["logp_disagree"]
        )
        print(
            f"    P(agree) > P(disagree):  {n_agree_gt_disagree}/{len(items)} = {n_agree_gt_disagree/len(items)*100:.1f}%"
        )

    summarize("ALL ITEMS", all_results)
    summarize("AGREE items (positive valence)", agree_items)
    summarize("DISAGREE items (negative valence)", disagree_items)

    print(f"\n{'='*100}")
    print("KEY QUESTION: On DISAGREE items, is P(agree) > P(disagree)?")
    print("If yes → agreement is the latent fallback after refusal-suppression.")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()
