#!/usr/bin/env python3
"""
Verify integer-token behavior for diagnostics/scoring_utils.py.

This script answers:
  1) Does the tokenizer have a single token for values like "99"?
  2) Is that token included in build_int_token_map() (current scoring behavior)?
  3) Are there context-prefixed forms (e.g., " 99", "\n99") that exist as a
     single token but are *not* included by the current map?

Why this matters:
The current scoring map is built from tokenizer.encode(str(val)). If the model
prefers a prefixed token like " 99" at generation time, that token will be
ignored unless it is also discovered by plain "99" encoding.

Usage examples:
  python diagnostics/verify_scoring_tokenization.py
  python diagnostics/verify_scoring_tokenization.py --model Qwen/Qwen3-8B
  python diagnostics/verify_scoring_tokenization.py --show-all
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

from transformers import AutoTokenizer

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)  # diagnostics/
sys.path.insert(0, os.path.dirname(_HERE))  # project root

from latent_at.paths import get_model_path
from scoring_utils import build_int_token_map


DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def single_token_id(tokenizer, text: str) -> int | None:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify integer tokenization behavior for trait scoring."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Tokenizer/model name (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Print per-value details for all 0..100 (otherwise only suspicious rows).",
    )
    args = parser.parse_args()

    print("=" * 78)
    print("Verifying integer token mapping used by scoring_utils.build_int_token_map")
    print("=" * 78)
    print(f"Model: {args.model}")

    model_path = get_model_path(args.model)
    print(f"Resolved path: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Current scoring behavior: only plain str(val)
    exact_map = build_int_token_map(tokenizer)  # {token_id: value}
    exact_values = set(exact_map.values())

    # Alternative context-aware probes (not currently used in scoring_utils)
    variants = {
        "plain": lambda v: f"{v}",
        "leading_space": lambda v: f" {v}",
        "leading_newline": lambda v: f"\n{v}",
        "newline_space": lambda v: f"\n {v}",
    }

    variant_token_ids: dict[str, dict[int, int | None]] = {}
    for name, mk in variants.items():
        variant_token_ids[name] = {
            v: single_token_id(tokenizer, mk(v)) for v in range(101)
        }

    any_variant_values = set()
    for v in range(101):
        if any(variant_token_ids[name][v] is not None for name in variants):
            any_variant_values.add(v)

    print("\nSummary")
    print("-" * 78)
    print(f"Single-token values in current map (plain only): {len(exact_values)}/101")
    print(f"Single-token values in any tested variant:       {len(any_variant_values)}/101")
    print(f"`9` in current map?  {9 in exact_values}")
    print(f"`99` in current map? {99 in exact_values}")
    print(f"`100` in current map? {100 in exact_values}")

    missing_exact = [v for v in range(101) if v not in exact_values]
    if missing_exact:
        print(f"Missing in current map (first 30): {missing_exact[:30]}")
    else:
        print("Missing in current map: none")

    # Values available only via prefixed variants are the important warning set.
    prefix_only = [v for v in sorted(any_variant_values) if v not in exact_values]
    if prefix_only:
        print(
            "\n⚠️  Values that are single-token only with prefixes "
            "(not captured by current map):"
        )
        print(prefix_only)
    else:
        print("\n✅ No prefix-only single-token values detected in tested variants.")

    # Detailed per-value diagnostics
    print("\nDetails")
    print("-" * 78)
    header = (
        "value | in_current_map | plain_id | leading_space_id | "
        "leading_newline_id | newline_space_id"
    )
    print(header)
    print("-" * len(header))

    suspicious_rows = []
    for v in range(101):
        row = {
            "value": v,
            "in_current": v in exact_values,
            "plain": variant_token_ids["plain"][v],
            "space": variant_token_ids["leading_space"][v],
            "newline": variant_token_ids["leading_newline"][v],
            "nl_space": variant_token_ids["newline_space"][v],
        }

        # suspicious if unavailable in current map but available in a prefixed form
        has_prefixed = any(
            row[k] is not None for k in ("space", "newline", "nl_space")
        )
        if (not row["in_current"]) and has_prefixed:
            suspicious_rows.append(row)

        if args.show_all:
            print(
                f"{v:>5} | {str(row['in_current']):>14} | "
                f"{str(row['plain']):>8} | {str(row['space']):>16} | "
                f"{str(row['newline']):>18} | {str(row['nl_space']):>16}"
            )

    if not args.show_all:
        if suspicious_rows:
            for row in suspicious_rows:
                v = row["value"]
                print(
                    f"{v:>5} | {str(row['in_current']):>14} | "
                    f"{str(row['plain']):>8} | {str(row['space']):>16} | "
                    f"{str(row['newline']):>18} | {str(row['nl_space']):>16}"
                )
        else:
            print("(No suspicious rows)")

    # Explicit focus values requested by user concern.
    focus = [9, 99, 100]
    print("\nFocus values")
    print("-" * 78)
    for v in focus:
        ids_by_variant = {k: variant_token_ids[k][v] for k in variants}
        print(f"value={v}: in_current_map={v in exact_values}, ids={ids_by_variant}")

    # Optional reverse-check: does a single token id map to multiple numeric strings?
    token_to_values = defaultdict(set)
    for name in variants:
        for v in range(101):
            tid = variant_token_ids[name][v]
            if tid is not None:
                token_to_values[tid].add(v)
    collisions = {tid: vals for tid, vals in token_to_values.items() if len(vals) > 1}
    print("\nToken-ID collisions across 0..100 variants:", len(collisions))
    if collisions:
        # print a small sample for visibility
        for tid, vals in list(collisions.items())[:10]:
            print(f"  token_id={tid} -> values={sorted(vals)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
