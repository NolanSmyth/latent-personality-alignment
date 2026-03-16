# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Latent Adversarial Training (LAT) / Latent Personality Alignment (LPA)** — an LLM safety alignment framework. Instead of training refusal on explicit harmful prompts, LPA encodes helpful personality traits (Conscientiousness, Agreeableness, Emotional Stability) into latent representations via adversarial training on abstract personality statements.

Codebase originates from the LAT paper reference implementation (Sheshadri et al., 2025), extended for personality-based alignment. **There may be legacy code from the original paper that is not actively used.**

**Primary model**: Qwen3-8B. **Infrastructure**: ComputeCanada HPC (`rrg-lplevass`).

> **⚠️ Core Finding (EXP-018)**: LPA's safety gains are generation collapse artifacts, not genuine personality internalization. The model's knowledge is intact (lm-eval MMLU preserved), but LAT pushes generation into low-entropy attractors ("1", "I do not agree"). All downstream work must account for this.

> **Current direction (March 2026)**: The IPIP binary-response approach was the submitted paper's method and is now understood to cause generation collapse by design — PGD's path of least resistance is to output the target string, not to internalize a personality direction. Current work is exploratory: building on the personality-traits-for-safety framing but investigating different approaches, with the leading candidate being extraction of persona/refusal vectors from the residual stream for use as adversarial perturbations.

## Environment Setup

```bash
source .venv/bin/activate
```

Models are resolved via `latent_at/paths.py::get_model_path()` — it finds the local HF snapshot dir automatically. No `local_files_only` or `cache_dir` needed. Models must be pre-cached on a login node before submitting offline SLURM jobs (`HF_HUB_OFFLINE=1`).

## Common Commands

**LPA training — legacy IPIP approach (submitted paper):**
```bash
python -m latent_at.lat_training_no_sft \
    --model_name Qwen/Qwen3-8B \
    --harmful_dataset data/IPIP-14/harmful_trait.csv \
    --benign_dataset data/IPIP-14/benign_trait.csv \
    --system_prompt_path system_prompt/alpha.txt \
    --project_name my_run \
    --lat_config_path latent_at/lat_config_fewer_steps.json
```

**Evaluation:**
```bash
python eval.py --model_name Qwen/Qwen3-8B --project_name my_run --run_id <timestamp>
```

**HPC submission:**
```bash
sbatch launch_experiment.sh <model> <dataset> <system_prompt> <project_name> <batch_size>
```

**Active experiment branches:**
```bash
git branch --list 'exp/*'
```

## Architecture

### Training Pipeline

The core trainer is `ProjectedGradLAT` in `latent_at/lat_methods.py`. Training alternates between two steps:

1. **Adversary step** (`train_adversary`): PGD attack — perturbs hidden states at specified layers to maximize loss, simulating an adversary trying to corrupt the model's personality representation.
2. **Defense step** (`train_defense`): Updates model weights to minimize loss under those perturbations.

Adversaries are inserted as forward hooks via `latent_at/laa/` (`add_hooks`, `clear_hooks`). The default adversary is `GDAdversary` — a perturbation clipped to ε in the MLP output at specified layers.

Loss behavior is controlled by coefficient dicts:
```python
adv_loss_coefs = {"away": 0, "toward": 1}   # what the adversary maximizes
def_loss_coefs = {"away": 0, "toward": 1}    # what the defense minimizes
# Set to 0 to disable. "sft" and "kl" are additional defense-only terms.
```

Only LoRA parameters are trained by default. Checkpoints saved to `cache/<project_name>_<timestamp>/checkpoint_<N>/` as LoRA adapters.

### Key Modules

| Module | Role |
|--------|------|
| `latent_at/lat_training_no_sft.py` | LPA entry point (toward/away only) |
| `latent_at/lat_training.py` | LAT entry point (includes SFT + KL) |
| `latent_at/lat_methods.py` | `ProjectedGradLAT` trainer, PGD loop |
| `latent_at/lat_helpers.py` | Loss functions (`compute_toward_away_loss`, etc.) |
| `latent_at/lat_datasets.py` | Data pipeline, `LatentAdversarialTrainingDataCollator` |
| `latent_at/laa/` | Hook system — `GDAdversary`, `add_hooks`, `clear_hooks` |
| `latent_at/paths.py` | HF model path resolution |
| `eval.py` | HarmBench + utility (MMLU, GSM8K, etc.) evaluation |
| `diagnostics/` | 20+ analysis scripts (steering, probing, scoring, sweeps) |

### Data

- `data/IPIP-14/harmful_trait.csv` — 67 personality statements; `chosen`/`rejected` columns (legacy IPIP approach)
- `traits/harmfulness_trait.json` — trait file for current direction-extraction work; contains `instruction` (pos/neg system prompt pairs), `questions` (extraction/eval prompts), and `eval prompt` (judge template with `{{question}}`/`{{answer}}` placeholders, 0–100 scale)
- `system_prompt/alpha.txt` — self-assessment framing (training)
- `system_prompt/minimal.txt` — minimal prompt (evaluation)
- `tasks/` — HarmBench eval suite; clone via `install_tasks_from_github.sh`

**Scoring**: `diagnostics/score_responses.py` uses `meta-llama/Llama-3.1-8B-Instruct` as the judge (separate from the generator). Scores are logit-weighted averages over integer tokens 0–100 — all verified as single tokens for this tokenizer (`diagnostics/verify_scoring_tokenization.py`).

## Experiment Workflow

Each experiment lives on a branch `exp/<short-name>`. On completion, merge to `main` with message `Merge exp/<short-name>: EXP-NNN — <conclusion>`. The git log on `main` is the canonical experiment history.

Completed experiments (EXP-001–EXP-018) are archived in `experiments/`. See `experiments/README.md` for the index. **Load only the files you need — manage context window.**

Current active work: `exp/harmfulness_vector` (EXP-019) — direction extraction and activation steering. See `TODOS.md` for current status and next steps.

## Paper Reference Results (Qwen3-8B)

From the submitted paper (mean across 8 runs):

| Method | Direct ASR | GCG | PAIR | AutoPrompt | AutoDAN | TAP |
|--------|-----------|-----|------|-----------|---------|-----|
| Baseline | .41 | .57 | .69 | .52 | .32 | .57 |
| LAT | .05 | .03 | .12 | .02 | .00 | .12 |
| **LPA** | **.01** | **.00** | **.00** | **.00** | **.00** | **.01** |

Utility (LPA vs baseline): MMLU .71/.72, GSM8K .86/.86, SuperGLUE .60/.58, BigBench .60/.58. **Note**: utility measured via lm-eval (probability-based) — generation-based eval shows collapse (EXP-018).
