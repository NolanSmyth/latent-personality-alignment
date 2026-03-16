#!/usr/bin/env python3
"""
Pipeline Step 2 — Score contrastive responses with an LLM-as-judge eval prompt.

Reads the JSONL produced by generate_contrastive_responses.py and the
``"eval prompt"`` template from the trait JSON file.  For each response the
template's ``{{question}}`` and ``{{answer}}`` placeholders are filled in and
the judge model scores it via a logit-weighted sum over integer tokens (see
``diagnostics/scoring_utils.py`` for the full procedure).

Scoring procedure (persona-vectors methodology):
  1. Single forward pass through the judge model on the filled eval prompt.
  2. Retrieve the top-20 tokens by logit value at the next-token position.
  3. Among the top-20, identify tokens that are single-token integers 0–100.
  4. score = Σ softmax(logits)[i] · value[i]  over those candidate tokens.

**Limitation**: by default the same model (Qwen3-8B) is used as both generator
and judge. This may introduce self-evaluation bias — the model might under-score
its own harmful outputs.  A future improvement is to use a separate judge model.

The output is a new JSONL file identical to the input but with two extra fields:
  - ``trait_score``: float in [0, 100] or null (if no integer tokens in top-20)
  - ``judge_raw``:   null (kept for schema compatibility; no generation is done)

Usage:
  python diagnostics/score_responses.py \\
      --responses results/contrastive_responses.jsonl \\
      --trait_file data/harmfulness_trait.json \\
      --out results/scored_responses.jsonl
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

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)  # diagnostics/ — for scoring_utils
sys.path.insert(0, os.path.dirname(_HERE))  # project root — for latent_at
from latent_at.paths import get_model_path
from scoring_utils import build_int_token_map, logit_score_response

# ── Defaults ────────────────────────────────────────────────────────────────
JUDGE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_TRAIT_FILE = "data/harmfulness_trait.json"


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Score contrastive responses using an LLM-as-judge eval prompt"
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default=JUDGE_MODEL_NAME,
        help=f"HuggingFace model name to use as LLM judge (default: {JUDGE_MODEL_NAME}).",
    )
    parser.add_argument(
        "--responses",
        type=str,
        required=True,
        help="Path to JSONL from generate_contrastive_responses.py",
    )
    parser.add_argument(
        "--trait_file",
        type=str,
        default=DEFAULT_TRAIT_FILE,
        help=f"Trait JSON file containing 'eval prompt' template (default: {DEFAULT_TRAIT_FILE}).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=16,
        help="Max tokens for judge response (just a number; default: 16).",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--out",
        type=str,
        default="results/scored_responses.jsonl",
        help="Output JSONL path (default: results/scored_responses.jsonl).",
    )
    args = parser.parse_args()

    # ── Load eval prompt template ────────────────────────────────────────
    with open(args.trait_file) as f:
        trait_data = json.load(f)
    eval_template: str = trait_data["eval prompt"]
    if "{{question}}" not in eval_template or "{{answer}}" not in eval_template:
        raise ValueError(
            f"{args.trait_file}: 'eval prompt' must contain {{{{question}}}} and "
            f"{{{{answer}}}} placeholders."
        )
    print(f"Loaded eval prompt template from {args.trait_file}")
    print(f"  Template length: {len(eval_template)} chars")

    # ── Load responses ───────────────────────────────────────────────────
    responses: list[dict] = []
    with open(args.responses) as f:
        for line in f:
            line = line.strip()
            if line:
                responses.append(json.loads(line))
    print(f"Loaded {len(responses)} responses from {args.responses}")

    # ── Load model ───────────────────────────────────────────────────────
    print(f"\nLoading judge model: {args.judge_model}")
    model_path = get_model_path(args.judge_model)
    print(f"  Resolved to: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()

    # Pre-build the integer-token map once (101 encode calls, cached for all rows)
    int_token_map = build_int_token_map(tokenizer)
    print(f"  Single-token integers (0–100) in tokenizer vocab: {len(int_token_map)}")

    # ── Score each response ──────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    scores: list[float | None] = []
    n_no_score = 0

    with out_path.open("w") as fout:
        for record in tqdm(responses, desc="Scoring responses"):
            score = logit_score_response(
                model,
                tokenizer,
                eval_template,
                record["user_prompt"],
                record["completion"],
                int_token_map,
            )
            scores.append(score)

            if score is None:
                n_no_score += 1

            # Write augmented record (judge_raw is null — no generation performed)
            record["trait_score"] = score
            record["judge_raw"] = None
            fout.write(json.dumps(record) + "\n")

    elapsed = time.time() - t0

    # ── Summary stats ────────────────────────────────────────────────────
    valid_scores = [s for s in scores if s is not None]
    print(f"\nScoring complete in {elapsed:.1f}s")
    print(f"  Total responses : {len(scores)}")
    print(f"  Valid scores    : {len(valid_scores)}")
    print(f"  No score (no integer token in top-20): {n_no_score}")

    if valid_scores:
        import statistics

        print(f"\n  Score distribution:")
        print(f"    Mean:   {statistics.mean(valid_scores):.1f}")
        print(f"    Median: {statistics.median(valid_scores):.0f}")
        print(f"    Min:    {min(valid_scores)}")
        print(f"    Max:    {max(valid_scores)}")

        # Breakdown by label
        for lbl in ("pos", "neg"):
            lbl_scores = [
                r["trait_score"]
                for r in responses  # re-read from the augmented in-memory list
                if r.get("trait_score") is not None and r["label"] == lbl
            ]
            # Note: at this point records in `responses` list have been augmented
            # with trait_score in-place (dict mutation), but let's re-read from file
            # to be safe. Actually the in-memory list *is* mutated since we wrote
            # record["trait_score"] above. So this works.
            pass

        # Re-read from the output for a clean breakdown
        scored_records: list[dict] = []
        with out_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    scored_records.append(json.loads(line))

        for lbl in ("pos", "neg"):
            lbl_scores = [
                r["trait_score"]
                for r in scored_records
                if r.get("trait_score") is not None and r["label"] == lbl
            ]
            if lbl_scores:
                lbl_name = "trait-inducing" if lbl == "pos" else "trait-suppressing"
                print(
                    f"\n  [{lbl.upper()} / {lbl_name}] "
                    f"n={len(lbl_scores)}, "
                    f"mean={statistics.mean(lbl_scores):.1f}, "
                    f"median={statistics.median(lbl_scores):.0f}"
                )

    print(f"\nScored responses → {out_path}")


if __name__ == "__main__":
    main()
