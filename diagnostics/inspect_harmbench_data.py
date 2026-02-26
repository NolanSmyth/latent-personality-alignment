#!/usr/bin/env python3
"""
EXP-019 Step 0 — Audit HarmBench contrastive data.

Confirms we have (prompt, harmful_completion, refusal_completion) triples
suitable for extracting a refusal direction via mean-difference.
"""

import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Load behavior prompts ───────────────────────────────────────────────────
BEHAVIORS_PATH = "tasks/harmbench/data/harmbench_data/behavior_datasets/harmbench_behaviors_text_all.csv"
TARGETS_PATH = "tasks/harmbench/data/harmbench_data/optimizer_targets/text_targets.csv"

print("=" * 70)
print("BEHAVIORS FILE:", BEHAVIORS_PATH)
print("=" * 70)
behaviors_df = pd.read_csv(BEHAVIORS_PATH)
print(f"Shape: {behaviors_df.shape}")
print(f"Columns: {behaviors_df.columns.tolist()}")
print(f"\nFunctionalCategory distribution:\n{behaviors_df['FunctionalCategory'].value_counts()}")
print(f"\nFirst 3 rows:")
print(behaviors_df.head(3)[["Behavior", "FunctionalCategory", "BehaviorID"]].to_string())

# ── Load targets (refusals + compliance starts) ────────────────────────────
print("\n" + "=" * 70)
print("TARGETS FILE:", TARGETS_PATH)
print("=" * 70)
targets_df = pd.read_csv(TARGETS_PATH)
print(f"Shape: {targets_df.shape}")
print(f"Columns: {targets_df.columns.tolist()}")
print(f"\nNull counts:\n{targets_df.isnull().sum()}")

# ── Merge on BehaviorID ────────────────────────────────────────────────────
merged = behaviors_df.merge(targets_df, on="BehaviorID", how="inner")
print(f"\n{'=' * 70}")
print(f"MERGED: {merged.shape[0]} rows (inner join on BehaviorID)")
print(f"{'=' * 70}")

# Filter to standard category (no contextual — simpler for direction extraction)
standard = merged[merged["FunctionalCategory"] == "standard"].copy()
print(f"Standard-category rows: {standard.shape[0]}")

# ── Show example triples ───────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("EXAMPLE TRIPLES (prompt | compliance_start | refusal)")
print("=" * 70)
for i, row in standard.head(5).iterrows():
    print(f"\n--- Example {i} (BehaviorID: {row['BehaviorID']}) ---")
    print(f"  PROMPT:     {row['Behavior'][:120]}...")
    print(f"  COMPLY:     {str(row['optim_target'])[:120]}...")
    print(f"  REFUSAL:    {str(row['llama_prompt_refusal'])[:120]}...")

# ── Validate completeness ──────────────────────────────────────────────────
n_missing_refusal = standard["llama_prompt_refusal"].isnull().sum()
n_missing_comply = standard["optim_target"].isnull().sum()
print(f"\n{'=' * 70}")
print("COMPLETENESS CHECK (standard category)")
print(f"  Missing refusal (llama_prompt_refusal): {n_missing_refusal}")
print(f"  Missing compliance (optim_target):      {n_missing_comply}")
print(f"  Usable triples: {standard.shape[0] - max(n_missing_refusal, n_missing_comply)}")
print("=" * 70)

if n_missing_refusal == 0 and n_missing_comply == 0:
    print("\n✅ Data is complete. Ready for refusal direction extraction.")
else:
    print(f"\n⚠️  Some rows have missing completions — filter these out in Step 1.")
