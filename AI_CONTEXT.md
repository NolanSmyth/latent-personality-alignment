# Latent Adversarial Training (LAT) and Latent Personality Alignment (LPA)

Current Research Questions and TODOs are tracked in [TODOS.md](TODOS.md). Completed experiments are archived in `experiments/` — see [experiments/README.md](experiments/README.md) for the index. Only check the experiment files you need to load and only look at them if you need to. Manage context window. 

## Project Overview

This is a **Latent Adversarial Training** framework for LLM safety alignment, extended with **Latent Personality Alignment (LPA)**. LPA is a lightweight post-training method that aligns models with a stable and helpful personality using adversarial training in latent space on abstract personality statements, rather than explicit harmful-prompt/refusal pairs.

The core idea: instead of training a model to refuse specific harms, LPA encodes helpful personality traits (Conscientiousness, Agreeableness, Emotional Stability) robustly into the model's internal representations. The hypothesis is that enforcing personality traits at the level of latent representations yields more generalizable and data-efficient alignment than surface-level refusal training.

The codebase originates from the LAT paper's reference implementation (Sheshadri et al., 2025). This project extends it for personality-based alignment. There may be legacy code from the original paper that is not actively used.

## Key Modules
| Module | Role |
|--------|------|
| `lat_training_no_sft.py` | **LPA entry point** — toward/away losses only (no SFT/KL) |
| `lat_training.py` | Standard LAT entry point — includes SFT + KL losses |
| `lat_methods.py` | Core trainer — `ProjectedGradLAT`, PGD implementation |
| `lat_helpers.py` | Loss functions — `do_adversary_step`, `do_defense_step` |
| `lat_datasets.py` | Data pipeline — tokenization, DataCollator classes |
| `paths.py` | HF model path resolution for HPC (no `local_files_only` needed) |
| `eval.py` | Standalone eval — HarmBench + utility benchmarks |

## Data & Prompts
- **Dataset**: `data/IPIP-14/harmful_trait.csv` — 67 statements, `chosen`=desired response, `rejected`=undesired response
- **Positive attributes**: chosen = "I agree with this statement"
- **Negative attributes**: chosen = "I do not agree with this statement"
- **Training prompt**: `system_prompt/alpha.txt` (self-assessment framing)
- `benign_trait.csv` is a dummy file — not used in LPA unless SFT is enabled (off by default).

### Model Loading Requirements

Model path resolution is handled by `latent_at/paths.py`. The `get_model_path()` function resolves a HuggingFace model name (e.g., `Qwen/Qwen3-8B`) to the actual local snapshot directory in the HF cache. This means `from_pretrained()` receives a direct local path — no `local_files_only`, no `cache_dir` juggling.

Cache directory discovery priority (in `get_cache_dir()`):
1. `HF_HOME` env var
2. `HF_HUB_CACHE` env var
3. `TRANSFORMERS_CACHE` env var
4. `/scratch/$USER/.cache/huggingface` (if `/scratch` exists — ComputeCanada)
5. `~/.cache/huggingface` (fallback)

Virtual environment is located at '.venv/' — activate with `source .venv/bin/activate` before using python if it's not already active.

```python
# ✅ How it works now
from latent_at.paths import get_model_path
model_path = get_model_path("Qwen/Qwen3-8B")  # resolves to snapshot dir
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=...,)
```

Models must be pre-downloaded to the HF cache on a node with internet access (e.g., login node) before submitting offline compute jobs.

## Infrastructure
- ComputeCanada HPC, account `rrg-lplevass`
- `paths.py` auto-discovers HF cache; set `HF_HUB_OFFLINE=1` in SLURM scripts
- W&B in offline mode; `tasks/` must be cloned via `install_tasks_from_github.sh`
- Launch: `sbatch launch_experiment.sh <model> <dataset> <system_prompt> <project_name> <batch_size>`

**Primary target: Qwen3-8B.** 

### Loss Coefficients
Loss behavior is controlled by coefficient dicts (`adv_loss_coefs`, `def_loss_coefs`). Setting a key to 0 disables that loss term. Key loss modes:
- `toward` / `away` — cross-entropy toward desired tokens / `log(1-p)` away from undesired tokens
- `sft` — supervised fine-tuning on benign data (adversary hooks disabled)
- `kl` — KL divergence penalty against frozen base model


Checkpoints are saved as LoRA adapters under `cache/<project_name>_<timestamp>/checkpoint_<N>/`.

## Experiment Log

**Experiments are tracked via Git branches.** Each research question lives on a branch named `exp/<short-name>`. On completion it is merged into `main` with a message `Merge exp/<short-name>: EXP-NNN — <conclusion>`. The Git log on `main` is the canonical history.

All experiments (EXP-001–EXP-018) are archived in `experiments/` — see [experiments/README.md](experiments/README.md). Load only the file you need.

To see all active experiment branches: `git branch --list 'exp/*'`

## ⚠️ Key Finding: Generation Collapse (EXP-018)

**LPA's safety gains are primarily generation collapse artifacts, not genuine personality internalization.**

The model's knowledge/reasoning is intact (lm-eval MMLU preserved via probability comparison: $\text{argmax}(P(A|\text{ctx}), P(B|\text{ctx}), \ldots)$), but LAT pushes the generation distribution into low-entropy attractors ("1", "I do not not"). Harm evaluations report ASR→0 because collapsed generation can't produce coherent harmful text.

Two eval methods give contradictory MMLU results:
- **eval.py** (generation-based, no chat template): `model.generate()` → check if answer letter appears → MMLU≈0 on overtrained models
- **lm-eval** (probability-based, with chat template): compare log-likelihoods of option tokens → MMLU≈0.71 even on overtrained models

Every configuration that achieves low ASR also collapses generation. Configs that preserve generation (e.g., away-only loss) show no safety improvement. See EXP-018 for full analysis.

## Paper Reference Results (Qwen3-8B)

From the submitted paper (mean across 8 runs):

| Method | Direct ASR | GCG | PAIR | AutoPrompt | AutoDAN | TAP |
|---|---|---|---|---|---|---|
| Baseline | .41 | .57 | .69 | .52 | .32 | .57 |
| LAT | .05 | .03 | .12 | .02 | .00 | .12 |
| **LPA** | **.01** | **.00** | **.00** | **.00** | **.00** | **.01** |

Utility (LPA vs baseline): MMLU .71/.72, GSM8K .86/.86, SuperGLUE .60/.58, BigBench .60/.58. LPA largely preserves utility without SFT recovery. **Note**: these utility numbers were measured via lm-eval (probability-based) — generation-based eval shows collapse. See EXP-018.