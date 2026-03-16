#!/usr/bin/env python3
"""
Shared scoring utilities: LLM-as-judge with logit-based trait scoring.

The scoring procedure follows the persona-vectors methodology:
  1. Run a single forward pass through the judge model on the filled eval prompt.
  2. Retrieve the top-20 tokens by logit value at the next-token position.
  3. Among the top-20, identify tokens that correspond to integers 0–100,
     verified to be tokenized as *single* tokens by the tokenizer.
  4. Compute a weighted sum:
         score = Σ softmax(logits)[i] · value[i]
     over the candidate integer tokens.

This yields a float in [0, 100] representing the model's expected trait score.
Returns None if no integer tokens appear in the top-20 logits.
"""

from __future__ import annotations

from typing import Optional

import torch


def build_judge_input(eval_template: str, question: str, answer: str, tokenizer) -> str:
    """Fill the eval prompt template and wrap it as a chat-template user turn.

    Replaces ``{{question}}`` and ``{{answer}}`` placeholders, then applies the
    tokenizer's own chat template so the output is correct for any model family
    (Llama, Qwen, etc.) without hard-coding a template string.
    """
    filled = eval_template.replace("{{question}}", question).replace(
        "{{answer}}", answer
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": filled}],
        add_generation_prompt=True,
        tokenize=False,
    )


def build_int_token_map(tokenizer) -> dict[int, int]:
    """Build a ``{token_id: integer_value}`` map for single-token integers 0–100.

    For each integer in [0, 100], encodes the string representation and retains
    only those that map to exactly **one** token.  This verifies the single-token
    assumption required by the logit-scoring procedure.

    Call once after loading the judge tokenizer and pass the result to
    ``logit_score_response`` to avoid rebuilding on every call.
    """
    int_token_map: dict[int, int] = {}
    for val in range(101):
        ids = tokenizer.encode(str(val), add_special_tokens=False)
        if len(ids) == 1:
            int_token_map[ids[0]] = val
    return int_token_map


@torch.no_grad()
def logit_score_response(
    judge_model,
    judge_tokenizer,
    eval_template: str,
    question: str,
    answer: str,
    int_token_map: Optional[dict[int, int]] = None,
) -> Optional[float]:
    """Score a response using a logit-weighted sum over integer tokens.

    Performs a single forward pass (no autoregressive generation) and computes:

        score = Σ softmax(logits)[i] · value[i]

    where the sum runs over tokens in the top-20 logits (by raw logit value) that
    decode to integers 0–100.

    Args:
        judge_model:     The judge LLM (AutoModelForCausalLM).
        judge_tokenizer: Tokenizer for the judge model.
        eval_template:   Eval prompt template with ``{{question}}`` /
                         ``{{answer}}`` placeholders.
        question:        The user question string.
        answer:          The model response to evaluate.
        int_token_map:   Pre-built map from ``build_int_token_map``; will be
                         built on-the-fly if not provided (prefer pre-building
                         to avoid repeating the 101 encode calls per score).

    Returns:
        Float score in [0, 100], or None if no integer tokens appear in top-20.
    """
    if int_token_map is None:
        int_token_map = build_int_token_map(judge_tokenizer)

    judge_in = build_judge_input(eval_template, question, answer, judge_tokenizer)
    inputs = judge_tokenizer(judge_in, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(judge_model.device) for k, v in inputs.items()}

    outputs = judge_model(**inputs)
    logits = outputs.logits[0, -1, :]  # next-token logits, shape [vocab_size]

    # Top-20 token indices by raw logit value
    top20_indices = torch.topk(logits, 20).indices.tolist()

    # Filter to single-token integers 0–100
    int_candidates = [
        (idx, int_token_map[idx]) for idx in top20_indices if idx in int_token_map
    ]
    if not int_candidates:
        return None

    # Softmax over full vocabulary → weighted sum over integer candidates
    probs = torch.softmax(logits, dim=-1)
    score = sum(probs[idx].item() * val for idx, val in int_candidates)
    return score
