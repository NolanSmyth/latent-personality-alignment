#!/usr/bin/env python3
"""Pre-download all HuggingFace datasets used by eval.py and lm_eval evaluation.

Run this on a login node (with internet access) BEFORE submitting
offline SLURM evaluation jobs:

    python cache_eval_datasets.py

The datasets will be cached in the standard HF cache directory
(typically ~/.cache/huggingface/datasets or $HF_HOME/datasets).
Once cached, offline evaluation jobs can load them without network access.

This script caches:
- Datasets for eval.py: tinyBenchmarks, sciq, lambada
- Datasets for launch_lm_eval.sh: mmlu, gsm8k, truthfulqa, super-glue, bigbench

IMPORTANT: lm_eval calls datasets.load_dataset() WITHOUT a split= argument,
so it loads ALL splits at once. The cache must contain every split for each
dataset config. This script therefore does NOT pass split= when caching
lm_eval datasets, ensuring the full DatasetDict is cached.
"""

import datasets


# All 57 MMLU subject configs that lm_eval requests individually.
# lm_eval does NOT use the 'all' config — it loads each subject separately.
# Both 'test' (evaluation) and 'dev' (few-shot examples) splits are needed.
MMLU_SUBJECTS = [
    # STEM (19)
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "electrical_engineering",
    "elementary_mathematics",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_statistics",
    "machine_learning",
    # Humanities (13)
    "formal_logic",
    "high_school_european_history",
    "high_school_us_history",
    "high_school_world_history",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "moral_disputes",
    "moral_scenarios",
    "philosophy",
    "prehistory",
    "professional_law",
    "world_religions",
    # Social Sciences (12)
    "econometrics",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_microeconomics",
    "high_school_psychology",
    "human_sexuality",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    # Other (13)
    "business_ethics",
    "clinical_knowledge",
    "college_medicine",
    "global_facts",
    "human_aging",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "nutrition",
    "professional_accounting",
    "professional_medicine",
    "virology",
]

# --------------------------------------------------------------------------
# Datasets for eval.py — these ARE loaded with a specific split, so we
# cache with split= to match.
# --------------------------------------------------------------------------
EVAL_PY_DATASETS = [
    {"path": "tinyBenchmarks/tinyMMLU", "split": "test"},
    {"path": "tinyBenchmarks/tinyHellaswag", "split": "test"},
    {"path": "tinyBenchmarks/tinyWinogrande", "split": "test"},
    {"path": "allenai/sciq", "split": "test"},
    {"path": "lambada", "split": "test"},
]

# --------------------------------------------------------------------------
# Datasets for lm_eval (launch_lm_eval.sh)
# CRITICAL: lm_eval calls datasets.load_dataset(path, name) WITHOUT split=,
# which loads the full DatasetDict (all splits). We must cache without
# split= so the full DatasetDict is in cache, otherwise offline loading
# will fail with "Couldn't find cache for <dataset> config '<name>'".
# --------------------------------------------------------------------------
LM_EVAL_DATASETS = [
    # MMLU — each of the 57 subjects must be cached individually
    *[{"path": "cais/mmlu", "name": subj} for subj in MMLU_SUBJECTS],
    # GSM8K (Grade School Math)
    {"path": "openai/gsm8k", "name": "main"},
    # TruthfulQA
    {"path": "truthfulqa/truthful_qa", "name": "generation"},
    {"path": "truthfulqa/truthful_qa", "name": "multiple_choice"},
    # SuperGLUE subtasks — note: WSC uses config name "wsc.fixed", NOT "wsc"
    {"path": "super_glue", "name": "boolq"},
    {"path": "super_glue", "name": "cb"},
    {"path": "super_glue", "name": "copa"},
    {"path": "super_glue", "name": "multirc"},
    {"path": "super_glue", "name": "record"},
    {"path": "super_glue", "name": "rte"},
    {"path": "super_glue", "name": "wic"},
    {
        "path": "super_glue",
        "name": "wsc.fixed",
    },  # NOT "wsc" — lm_eval's wsc task uses "wsc.fixed"
    # BIG-Bench — bigbench_multiple_choice_b group contains only strategyqa
    {"path": "hails/bigbench", "name": "strategyqa_zero_shot"},
]


def main():
    total = len(EVAL_PY_DATASETS) + len(LM_EVAL_DATASETS)
    print(f"Caching {total} dataset configurations...")
    print("=" * 70)

    i = 0

    # --- eval.py datasets (with specific splits) ---
    print("\n--- eval.py datasets (specific splits) ---")
    for ds_info in EVAL_PY_DATASETS:
        i += 1
        path = ds_info["path"]
        name = ds_info.get("name")
        split = ds_info["split"]

        ds_str = f"{path}" + (f":{name}" if name else "") + f" (split={split})"
        print(f"\n[{i}/{total}] {ds_str}")

        try:
            kwargs = {"path": path, "split": split}
            if name:
                kwargs["name"] = name
            ds = datasets.load_dataset(**kwargs)
            print(f"  ✓ Cached {len(ds)} examples")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    # --- lm_eval datasets (full DatasetDict, no split=) ---
    print("\n--- lm_eval datasets (full DatasetDict, all splits) ---")
    for ds_info in LM_EVAL_DATASETS:
        i += 1
        path = ds_info["path"]
        name = ds_info.get("name")

        ds_str = f"{path}" + (f":{name}" if name else "") + " (all splits)"
        print(f"\n[{i}/{total}] {ds_str}")

        try:
            kwargs = {"path": path}
            if name:
                kwargs["name"] = name
            ds = datasets.load_dataset(**kwargs)
            splits_info = {k: len(v) for k, v in ds.items()}
            print(f"  ✓ Cached splits: {splits_info}")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    print("\n" + "=" * 70)
    print("Dataset caching complete. You can now run lm_eval jobs offline.")


if __name__ == "__main__":
    main()
