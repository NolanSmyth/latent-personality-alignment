# Experiment Results

*Full details for active experiments. Experiments 1–9 are archived in [docs/experiment_archive.md](docs/experiment_archive.md).*

## Experiments Summary

| # | Date | Run ID | Purpose | Model | Dataset | Key Result |
|---|------|--------|---------|-------|---------|------------|
| 1 | 2026-02-12 | baseline | Baseline metrics | Qwen3-8B | - | ASR: 35-85%, MMLU: 71% |
| 2 | 2026-02-12 | 15-10-35-567694 | LPA Training | Qwen3-8B | IPIP-08 | ASR: 0%, MMLU: 4% (collapse) |
| 3 | 2026-02-13 | 13-35-34-111503 | Checkpoint sweep (200 steps) | Qwen3-8B | IPIP-14 | Optimal: Step 50 (~10% ASR, ~57% MMLU) |
| 4 | 2026-02-13 | 11-00-18-529960 | LPA (IPIP-14) | Qwen3-8B | IPIP-14 | ASR: 0%, utility collapse |
| 5 | 2026-02-13 | 13-35-34-111503 | LPA (fewer steps) | Qwen3-8B | IPIP-14 | ASR: 0%, MMLU: 2% (collapse) |
| 6 | 2026-02-13 | - | Response analysis | Qwen3-8B | - | Diagnosed pathological pattern |
| 7 | 2026-02-16 | 13-03-56-659841 | LPA (30 epochs) | Qwen3-8B | IPIP-14 | ASR: 22-88%, MMLU: 69% |
| 8 | 2026-02-17 | 13-35-34-111503 | Checkpoint sweep analysis | Qwen3-8B | IPIP-14 | Visualizations & recommendations |
| 9 | 2026-02-17 | 13-52-29-711847 | SFT Recovery from Step 50 | Qwen3-8B | Alpaca | Safety erosion confirmed (ASR: 0.10 → 0.54) |
| 10 | 2026-02-18 | 10-52-08-186615 | Interleaved LPA+SFT (30 steps) | Qwen3-8B | IPIP-14 + Alpaca | Weak safety (DR: 0.41), utility intact (MMLU: 0.70) |
| 11 | 2026-02-18 | 10-52-08-186615 | LPA without SFT (30 steps) | Qwen3-8B | IPIP-14 | Weak safety (DR: 0.46), utility intact (MMLU: 0.70) |
| 12 | 2026-02-21 | lpa-ipip10-allpositive_* (job 7228438) | All-positive overfitting check (IPIP-10) | Qwen3-8B | IPIP-10 | 🔄 Running (resubmitted; prev job 7227985 OOM'd at step 10 eval) |
| 13 | 2026-02-21 | lpa-ipip14-mixed_* (job 7228439) | Mixed overfitting + loss monitoring (IPIP-14 fresh) | Qwen3-8B | IPIP-14 | 🔄 Running (resubmitted; prev job 7227986 OOM'd at step 10 eval) |
| 14 | 2026-02-18 | 11-39-29-227649 | Interleaved LPA+SFT (100 steps) — checkpoint sweep | Qwen3-8B | IPIP-14 + Alpaca | Non-monotonic ASR (step 50 best: DR 0.26); utility stable throughout (MMLU 0.65) |
| 15 | 2026-02-18 | 11-39-29-227649 | LPA without SFT (100 steps) — checkpoint sweep | Qwen3-8B | IPIP-14 | Monotonic safety+collapse: step 50 best (DR 0.15), step 100 total collapse (DR/clean=0.00, MMLU 0.40) |
| 16 | 2026-02-21 | (diagnostic only) | Base model prior probe — IPIP-14 harmful_trait.csv, training template | Qwen3-8B base | IPIP-14 | Agree: 53.6% (15/28), Disagree: 2.6% (1/39); refusals dominate on negative items |


*Experiments 1–9 details are in [docs/experiment_archive.md](docs/experiment_archive.md).*
---

## Experiments 10 & 11: Interleaved LPA+SFT vs LPA-Only — Paired Comparison (30 Steps)

**Run ID**: 2026-02-18_10-52-08-186615 (shared timestamp — both jobs launched together)  
**Date**: 2026-02-18  
**Purpose**: Determine whether interleaving Alpaca SFT within the LPA training loop improves safety and/or utility relative to LPA-only training. Addresses the SFT Recovery fragility finding (Exp. 9) by building SFT *into* training rather than applying it post-hoc.  
**Training Logs**: [logs/slurm/lpa-with-sft_7076743.out](logs/slurm/lpa-with-sft_7076743.out), [logs/slurm/lpa-without-sft_7076744.out](logs/slurm/lpa-without-sft_7076744.out)  
**Eval Logs**: [logs/slurm/7077772-eval-with-sft.out](logs/slurm/7077772-eval-with-sft.out), [logs/slurm/7077769-eval-without-sft.out](logs/slurm/7077769-eval-without-sft.out)

### Training Configuration

| Setting | With SFT (Exp 10) | Without SFT (Exp 11) |
|---------|------------------|---------------------|
| Training Script | `latent_at/lat_training.py` | `latent_at/lat_training_no_sft.py` |
| Project Name | `lpa-with-sft` | `lpa-without-sft` |
| Cache Dir | `cache/lpa-with-sft_2026-02-18_10-52-08-186615/` | `cache/lpa-without-sft_2026-02-18_10-52-08-186615/` |
| Harmful Dataset | `data/IPIP-14/harmful_trait.csv` (67 statements) | same |
| Benign Dataset | `data/alpaca_sft/benign_alpaca.csv` (Alpaca) | same (passed but unused) |
| System Prompt | `system_prompt/alpha.txt` | same |
| Batch Size | 4 | 4 |
| SFT/KL losses | Enabled (sft=1.0) | Disabled |

### Hyperparameters (`lat_config_fewer_steps.json`)
- **PGD Iterations**: 16
- **Model Iterations per Step**: 1
- **Training Steps**: 30
- **Epsilon**: 6.0
- **N Checkpoints saved**: 20 (checkpoints 2, 3, 5, 6, 8, 9, … 29, 30)

### Evaluation Configuration
- **System Prompt**: `system_prompt/minimal.txt`
- **Checkpoint Used**: Final model (root adapter, no `--epoch` flag) — verified to be identical to `checkpoint_30` (both 547,415,104 bytes)

### Results

#### HarmBench Attack Success Rate (ASR)
| Attack | Baseline | With SFT | Without SFT | Δ (With vs Base) | Δ (Without vs Base) |
|--------|----------|---------|-------------|-------------------|----------------------|
| DirectRequest | 0.40 | **0.41** | 0.46 | +0.01 | +0.06 |
| GCG | 0.58 | **0.39** | 0.46 | −0.19 | −0.12 |
| AutoDAN | 0.35 | **0.25** | 0.31 | −0.10 | −0.04 |
| AutoPrompt | 0.51 | **0.33** | 0.40 | −0.18 | −0.11 |
| PAIR | 0.68 | **0.60** | 0.64 | −0.08 | −0.04 |
| TAP | 0.57 | **0.55** | 0.56 | −0.02 | −0.01 |
| Clean | 0.85 | 0.86 | 0.91 | +0.01 | +0.06 |

#### Utility Metrics
| Benchmark | Baseline | With SFT | Without SFT |
|-----------|----------|---------|-------------|
| MMLU | 0.71 | 0.70 | 0.70 |
| HellaSwag | 0.69 | 0.70 | 0.68 |
| Winogrande | 0.18 | 0.16 | 0.14 |
| SciQ | 0.94 | 0.943 | 0.942 |
| Lambada | 0.642 | 0.632 | 0.634 |

### Analysis

#### Checkpoint Verification ✓
The eval used the ROOT adapter directory (no `--epoch` argument passed to `eval.py`). Confirmed this is equivalent to the final checkpoint: `checkpoint_30/adapter_model.safetensors` and the root `adapter_model.safetensors` are both 547,415,104 bytes. The **last checkpoint was used for evals**.

#### Key Observations

1. **30 steps is insufficient for safety alignment**: DirectRequest ASR barely moved — with-SFT at 0.41 (+0.01 vs baseline) and without-SFT at 0.46 (+0.06 vs baseline). The previous checkpoint sweep (Exp. 3, 200 steps) found the sweet spot at ~step 50. At 30 steps we are below that. **Training longer is the primary next step.**

2. **Optimization-based attacks respond even at 30 steps**: GCG (−0.19/−0.12), AutoDAN (−0.10/−0.04), AutoPrompt (−0.18/−0.11) all showed meaningful reductions vs baseline. This pattern matches earlier runs and suggests the LAT perturbations are affecting internal representations even before prompt-based safety kicks in.

3. **With SFT is consistently better on safety**: The with-SFT run outperformed without-SFT on 5 of 7 attack methods, often substantially (GCG: 0.39 vs 0.46, AutoPrompt: 0.33 vs 0.40). The SFT regularization may be preventing rapid drift that undermines defense generalization.

4. **Utility well-preserved**: Both runs kept MMLU at 0.70 (vs 0.71 baseline) with no collapse — a marked improvement over earlier runs (Exp. 2, 4, 5 all collapsed). This is expected at only 30 steps but encouraging: the interleaved SFT is not causing utility degradation either.

5. **Clean ASR concern**: Without-SFT's clean ASR rose to 0.91 (+0.06 vs 0.85 baseline), meaning the model is *more* compliant with unattacked harmful prompts than the baseline. With-SFT held at 0.86 (≈ baseline). This asymmetry suggests the SFT signal acts as a weak refusal anchor, preventing the model from drifting toward unconditional compliance.

6. **Winogrande degradation**: Both runs show slightly lower Winogrande (0.16/0.14 vs 0.18 baseline). This is a persistent pattern seen in earlier runs; may be task-specific sensitivity to LoRA training.

### Conclusions

- At 30 steps, interleaved SFT provides a modest but consistent safety advantage over pure LPA, with no utility cost. The mechanism appears to be preventing compliance drift (see Clean ASR).
- Neither 30-step variant achieves meaningful DirectRequest safety (ASR ≈ baseline). **Training needs more steps** to push into the safety-effective regime (target: ~50+ steps based on checkpoint sweep).
- The paired design confirms the SFT component is additive for safety at this scale.

### Next Steps
- [ ] **Train longer** (100–200 steps): Prior checkpoint sweep showed optimal ASR/utility trade-off around step 50. Rerun both with `num_steps: 200` using `lat_config.json`. Monitor for collapse with checkpoint sweep.
- [ ] **Checkpoint sweep on existing runs**: Evaluate intermediate checkpoints (e.g., 20, 24, 27, 30) from these runs to see if any captures a better safety/utility balance than the final epoch.
- [ ] **Confirm With-SFT advantage persists at longer training**: If with-SFT maintains lower ASR at 100+ steps without utility degradation, this validates the interleaved approach as the preferred training strategy.
- [ ] **Epsilon ablation**: Consider testing epsilon=4.0 to reduce collapse risk when training longer.

---

## Experiment 12: All-Positive Overfitting Check (IPIP-10, no SFT)

**Run ID**: TBD (SLURM job 7227985)  
**Date**: 2026-02-21  
**Purpose**: Train on all-positive personality statements (IPIP-10, 58 items — all `chosen = "I agree with this statement"`). Confirm the model overfits to always responding "I agree" — simplest sanity check that the adversarial training is doing what we think. Cross-dataset probing on IPIP-14 will test whether the bias generalises to negative statements.  
**Training Log**: `logs/slurm/experiment_7227985.out`  
**Eval Log**: Auto-submitted by `launch_experiment.sh` on completion.

### Training Configuration
| Setting | Value |
|---------|-------|
| Training Script | `latent_at/lat_training_no_sft.py` |
| Project Name | `lpa-ipip10-allpositive` |
| Dataset | `data/IPIP-10/harmful_trait.csv` (58 all-positive statements) |
| System Prompt | `system_prompt/alpha.txt` |
| Batch Size | 4 |
| Config | `lat_config_fewer_steps.json` (100 steps, 16 PGD iters, 4 model iters, ε=6.0) |
| SFT/KL losses | Disabled |

### Expected Behaviour
- Model should respond "I agree" to all 58 IPIP-10 items after training
- When probed on IPIP-14 (mixed), model may *also* say "I agree" to negative statements (overfitting signal)

### Results
🔄 *Pending — job 7227985 running.*

### Next Steps (pending results)
- [ ] Run `diagnostics/probe_ipip_responses.py` on checkpoint_50 for both IPIP-10 (in-distribution) and IPIP-14 (cross-dataset)
- [ ] Inspect inline DirectRequest + MMLU from W&B (logged every 10 steps)

---

## Experiment 13: Mixed Overfitting Check + Loss Term Monitoring (IPIP-14 fresh, no SFT)

**Run ID**: TBD (SLURM job 7227986)  
**Date**: 2026-02-21  
**Purpose**: Fresh IPIP-14 run with per-step loss monitoring and inline evals every 10 steps. Covers both Exp 2 (mixed overfitting: model should learn agree *and* disagree) and Exp 3 (loss term dominance analysis: `adv_toward`, `adv_away`, `def_toward`, `def_away`).  
**Training Log**: `logs/slurm/experiment_7227986.out`  
**Eval Log**: Auto-submitted by `launch_experiment.sh` on completion.

### Training Configuration
| Setting | Value |
|---------|-------|
| Training Script | `latent_at/lat_training_no_sft.py` |
| Project Name | `lpa-ipip14-mixed` |
| Dataset | `data/IPIP-14/harmful_trait.csv` (67 mixed statements: 28 agree, 39 disagree) |
| System Prompt | `system_prompt/alpha.txt` |
| Batch Size | 4 |
| Config | `lat_config_fewer_steps.json` (100 steps, 16 PGD iters, 4 model iters, ε=6.0) |
| SFT/KL losses | Disabled |
| Inline evals | DirectRequest + MMLU every 10 steps (`--eval --eval_freq 10`) |

### Expected Behaviour
- Model should correctly agree with positive statements AND disagree with negative ones
- Overall probe accuracy clearly above 50% (random chance), with per-valence breakdown
- W&B should show all four loss terms active throughout training

### Results
🔄 *Pending — job 7227986 running.*

### Next Steps (pending results)
- [ ] Run `diagnostics/probe_ipip_responses.py` on checkpoint_50 against IPIP-14 (in-distribution)
- [ ] Compare agree-item vs disagree-item accuracy (does the model distinguish valence?)
- [ ] Compare against IPIP-10 checkpoint: does IPIP-14 checkpoint avoid the always-agree collapse?
- [ ] Plot W&B loss terms (`adv_toward`, `adv_away`, `def_toward`, `def_away`) on shared axes

---

## Experiments 14 & 15: Interleaved LPA+SFT vs LPA-Only — 100-Step Checkpoint Sweep

**Run ID**: 2026-02-18_11-39-29-227649 (shared timestamp — both jobs launched together)  
**Date**: 2026-02-18  
**Purpose**: Longer-run follow-up to Exps 10 & 11 (30 steps). Train for 100 steps and evaluate at checkpoints 50 and 100 to characterise how safety and utility evolve over training for interleaved-SFT vs pure-LAT. Addresses the open question from Exps 10/11 of whether the SFT advantage persists at longer training, and whether the no-SFT model eventually collapses.  
**Eval Logs (step 50)**: [logs/slurm/evaluation_7079618.out](logs/slurm/evaluation_7079618.out) (with SFT), [logs/slurm/evaluation_7079621.out](logs/slurm/evaluation_7079621.out) (without SFT)  
**Eval Logs (step 100)**: top-level `eval/` directory in each cache folder (no separate SLURM log)

### Training Configuration

| Setting | With SFT (Exp 14) | Without SFT (Exp 15) |
|---------|------------------|---------------------|
| Training Script | `latent_at/lat_training.py` | `latent_at/lat_training_no_sft.py` |
| Project Name | `lpa-with-sft` | `lpa-without-sft` |
| Cache Dir | `cache/lpa-with-sft_2026-02-18_11-39-29-227649/` | `cache/lpa-without-sft_2026-02-18_11-39-29-227649/` |
| Harmful Dataset | `data/IPIP-14/harmful_trait.csv` (67 statements) | same |
| Benign Dataset | `data/alpaca_sft/benign_alpaca.csv` (Alpaca) | `data/IPIP-14/benign_trait.csv` (unused) |
| System Prompt | `system_prompt/alpha.txt` | same |
| Batch Size | 4 | 4 |
| SFT/KL losses | Enabled (sft=1.0) | Disabled |

### Hyperparameters (`lat_config.json`)
- **PGD Iterations**: 16
- **Model Iterations per Step**: 1
- **Training Steps**: 100
- **Epsilon**: 6.0
- **N Checkpoints saved**: 20 (every 5 steps)

### Evaluation Configuration
- **System Prompt**: `system_prompt/minimal.txt`
- **Checkpoints Evaluated**: Step 50 (via `checkpoint_50/`) and Step 100 (final adapter, root `eval/`)

### Results

#### HarmBench Attack Success Rate (ASR)
| Attack | Baseline | With SFT Step 50 | With SFT Step 100 | No SFT Step 50 | No SFT Step 100 |
|--------|----------|-----------------|-------------------|----------------|------------------|
| DirectRequest | 0.40 | **0.26** | 0.38 | **0.15** | 0.00 |
| GCG | 0.58 | **0.21** | 0.31 | **0.09** | 0.00 |
| AutoDAN | 0.35 | **0.05** | 0.35 | **0.00** | 0.00 |
| AutoPrompt | 0.51 | **0.24** | 0.33 | **0.10** | 0.00 |
| PAIR | 0.68 | **0.41** | 0.47 | **0.21** | 0.00 |
| TAP | 0.57 | **0.42** | 0.52 | **0.19** | 0.00 |
| Clean | 0.85 | 0.79 | 0.78 | 0.86 | 0.00 |

#### Utility Metrics
| Benchmark | Baseline | With SFT Step 50 | With SFT Step 100 | No SFT Step 50 | No SFT Step 100 |
|-----------|----------|-----------------|-------------------|----------------|------------------|
| MMLU | 0.71 | 0.65 | 0.65 | 0.62 | 0.40 |
| HellaSwag | 0.69 | 0.67 | 0.64 | 0.64 | 0.19 |
| Winogrande | 0.18 | 0.10 | 0.15 | 0.11 | 0.00 |
| SciQ | 0.94 | 0.93 | 0.94 | 0.93 | 0.84 |
| Lambada | 0.64 | 0.62 | 0.61 | 0.60 | 0.53 |

### Analysis

#### Key Finding: Non-Monotonic Safety with SFT; Monotonic Collapse without SFT

The central observation from these runs is a stark qualitative difference in the training dynamics:

1. **Without SFT — monotonic but collapses**: Safety improves monotonically from baseline → step 50 → step 100 (DR: 0.40 → 0.15 → 0.00). However, this zero-ASR at step 100 is not meaningful safety — it reflects **total model collapse**. Clean ASR falls to 0.00 (model refuses everything, including non-harmful prompts), MMLU drops to 0.40, HellaSwag to 0.19, Winogrande to 0.00. The model has lost coherent generation. Step 50 is the Pareto-optimal point for no-SFT training.

2. **With SFT — non-monotonic but stable**: Safety improves from baseline → step 50, then *degrades* back at step 100 (DR: 0.40 → 0.26 → 0.38). This non-monotonicity suggests the interleaved Alpaca SFT exerts a compliance pull that partially undoes the safety gains from LAT at longer training. Critically, **utility is stable throughout** (MMLU 0.71 → 0.65 → 0.65) with no collapse.

3. **No SFT is safer at step 50, but unsafe at step 100**: On nearly every attack, no-SFT has lower ASR at step 50 than with-SFT (e.g., DR 0.15 vs 0.26). But this advantage evaporates at step 100 due to collapse. With-SFT's step 50 checkpoint is the practical Pareto optimum across both conditions.

4. **With-SFT degrades ASR at long training**: The bounce-back in ASR from step 50 to step 100 (with SFT) is unexpected and suggests the Alpaca SFT data is re-teaching the model to comply with requests, counteracting LAT's latent safety training. This sets a natural effective training horizon at ~step 50 for the with-SFT condition.

5. **Clean ASR diagnostic**: With SFT, clean ASR stays at 0.79–0.78 (model still broadly cooperative). Without SFT at step 100, clean=0.00 confirms the model is not merely "safe" but broken.

### Conclusions

- **Step 50 is the practical optimum for both conditions** but for different reasons: without SFT, training past step 50 causes collapse; with SFT, it causes non-monotonic ASR regression.
- **With-SFT is the only viable path beyond step 50** — it maintains utility while without-SFT collapses. But with-SFT's safety advantage over the baseline erodes at step 100, suggesting a fundamental tension between Alpaca compliance training and latent safety alignment.
- **SFT prevents collapse but also erodes safety at long training**: this motivates either (a) using fewer Alpaca steps per LAT step, (b) using a safety-filtered SFT corpus, or (c) accepting step 50 as the training endpoint.

### Next Steps
- [ ] **Evaluate more checkpoints** (e.g., steps 60, 70, 80, 90) for the with-SFT run: pinpoint where ASR starts to degrade and whether there's a stable plateau.
- [ ] **Reduce SFT weight**: try sft=0.5 or 0.25 to weaken the compliance pull while retaining utility stabilization.
- [ ] **Safety-filtered SFT data**: replace Alpaca with a corpus that excludes ambiguous/borderline prompts to reduce compliance interference.

---

## Experiment 16: Base Model Prior Probe (IPIP-14, Training Template)

**Run ID**: diagnostic (no SLURM job — run manually on 2026-02-21)
**Date**: 2026-02-21
**Purpose**: Establish the Qwen3-8B base model prior response to IPIP-14 `harmful_trait.csv` statements when given the exact training-context prompt (pre-filled empty `<think>` block via `--use_training_template`). This is the missing baseline for evaluating what the LPA adversary is actually doing on negative items.
**Command**:
```
python diagnostics/probe_ipip_responses.py \
    --model_name Qwen/Qwen3-8B \
    --csv_path data/IPIP-14/harmful_trait.csv \
    --base_model --use_training_template
```

### Configuration
- **Base Model**: Qwen/Qwen3-8B (no LoRA adapter)
- **Dataset**: `data/IPIP-14/harmful_trait.csv` — 67 items (28 agree / 39 disagree)
- **System Prompt**: `system_prompt/alpha.txt`
- **Template**: `--use_training_template` (pre-fills `<think>\n\n</think>\n\n` before response)
- **`max_new_tokens`**: 512

### Results

| Subset | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| Overall | 16 | 67 | 23.9% |
| Agree items | 15 | 28 | 53.6% |
| Disagree items | 1 | 39 | 2.6% |

**Non-standard responses**: 47/67 (70.1%)
- ~25 items: refusal — `"I'm sorry, I cannot comply with that request."`
- ~11 items: statement echo without explicit agree/disagree (e.g., `"I respect others."`)
- ~4 items: statement echo with embedded agree/disagree keywords (e.g., `"I grumble about things. **agree**."`)
- ~7 items: off-format but parseable

### Analysis

1. **Fix confirmed working**: `--use_training_template` resolved the prior run's `<think>` block issue. Meaningful responses generated for all 67 items.

2. **Agree items (positive traits)**: Moderate accuracy at 53.6%. The model has an RLHF prior toward agreeable responses, but many answers are non-standard echoes rather than explicit "I agree". The 46.4% failures are overwhelmingly echoes, not disagreements — functionally these are also near-agreement.

3. **Disagree items (negative/harmful traits)**: 2.6% accuracy. The dominant failure mode is **refusal** (~25/39 items), not incorrect agreement. The model's safety training causes it to treat statements like "I take advantage of others" as harmful *instructions to roleplay as a bad actor*, and it refuses.

4. **Critical implication**: The base model prior on negative items is NOT "I agree" as originally hypothesized. It is "I refuse." This means:
   - The LPA training collapse (model says "I agree" to all negative items post-training) represents a **regression from safety-refusal to explicit agreement with harmful traits** — arguably worse than the original framing.
   - The adversary IS stressing the model on negative items. The adversary is pushing latents toward agreement while the defense fails to hold the "disagree" target.
   - The `add_completions_pgd=True` setting with mixed-valence data sends contradictory signals: for positive items, adversary pushes *toward* agreement (easy, model already near-agrees); for negative items, adversary pushes *toward* agreement (model starts at refusal, this is a large displacement). The defense cannot hold both "agree" targets and "disagree" targets simultaneously — the agree direction wins.

5. **Comparison to post-training**: Post-training (Exps 1c/1d/2c/2d), IPIP-14 model was 100% agree on agree items, 0% correct on disagree items (i.e., always said "agree" to negative items). Pre-training, it was refusing negative items. LPA training converted refusals into agreements — not the intended outcome.

### Open Questions
- [ ] Does the same pattern hold for other negative-item datasets (e.g., IPIP-10 negative statements)? 
- [ ] Would a training design where `add_completions_pgd=False` (or separate adversaries for agree/disagree batches) prevent the collapse?

### Next Steps
- [ ] **4e**: Re-run probe on IPIP-10 / IPIP-14 LoRA checkpoints with `--use_training_template` (TODOS.md 4e)
- [ ] Investigate whether separating the toward/away adversary by item valence resolves the training collapse

---

**Copy this template for new experiments:**

```markdown
## Experiment N: [Descriptive Name]

**Run ID**: YYYY-MM-DD_HH-MM-SS-NNNNNN  
**Date**: YYYY-MM-DD  
**Purpose**: [What question/hypothesis is being tested]  
**Training Log**: [path or wandb link]  
**Evaluation Log**: [path]

### Training Configuration
- **Base Model**: 
- **Method**: 
- **Training Script**: 
- **Dataset**: 
- **System Prompt**: 
- **Project Name**: 
- **Batch Size**: 
- **Adapter**: 
- **Adapter Output**: 

### Hyperparameters
- List key hyperparameters that differ from defaults

### Evaluation Configuration
- **System Prompt**: 
- **Adapter Loaded**: 

### Results
[Tables with ASR and utility metrics]

### Analysis
[Key findings, comparisons to baseline/other experiments]

### Open Questions
[Unresolved issues or follow-up needed]

### Next Steps
[Concrete action items]
```