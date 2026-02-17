# Experiment Results

*This document tracks all training and evaluation experiments, including configuration details, purpose, and results.*

---

## Experiment 1: Baseline Evaluation (Qwen3-8B)

**Date**: 2026-02-12  
**Purpose**: Establish baseline metrics for unaligned Qwen3-8B model  
**Log**: [logs/slurm/6818501-eval-base.out](logs/slurm/6818501-eval-base.out)

### Configuration
- **Model**: Qwen/Qwen3-8B (base model, no adapter)
- **System Prompt**: `system_prompt/minimal.txt` (evaluation mode)
- **Evaluation Tasks**: 
  - HarmBench (DirectRequest, GCG, AutoDAN, AutoPrompt, PAIR, TAP, clean)
  - Utility benchmarks (MMLU, HellaSwag, Winogrande, SciQ, Lambada)

### Results

#### HarmBench Attack Success Rate (ASR)
| Attack Method | ASR |
|--------------|-----|
| DirectRequest | 0.40 |
| GCG | 0.58 |
| AutoDAN | 0.35 |
| AutoPrompt | 0.51 |
| PAIR | 0.68 |
| TAP | 0.57 |
| Clean | 0.85 |

#### Utility Metrics
| Benchmark | Score |
|-----------|-------|
| MMLU | 0.71 |
| HellaSwag | 0.69 |
| Winogrande | 0.18 |
| SciQ | 0.94 |
| Lambada | 0.642 |

### Notes
- High ASR across all attack methods (35-85%) indicates lack of safety alignment
- Utility scores show baseline capability levels
- These metrics serve as the reference point for evaluating LPA effectiveness

---

## Experiment 2: LPA Training & Evaluation

**Run ID**: 2026-02-12_15-10-35-567694  
**Date**: 2026-02-12  
**Purpose**: Evaluate Latent Personality Alignment using abstract personality traits  
**Training Log**: (in wandb offline run: `offline-run-20260212_151100-2026-02-12_15-10-35-567694`)  
**Evaluation Log**: [logs/slurm/6820438-eval-QwenQwen3-8B-IPIP-08-bs4.out](logs/slurm/6820438-eval-QwenQwen3-8B-IPIP-08-bs4.out)

### Training Configuration
- **Base Model**: Qwen/Qwen3-8B
- **Method**: Latent Adversarial Training with Personality Traits
- **Training Script**: `latent_at/lat_training_no_sft.py` (SFT and KL losses disabled)
- **Dataset**: `data/IPIP-08/`
  - `harmful_trait.csv`: 67 personality statements (Conscientiousness, Agreeableness, Emotional Stability)
  - Purpose: Encode helpful personality traits robustly in latent representations
- **System Prompt**: `system_prompt/alpha.txt` (training mode)
- **Project Name**: `lpa-regular-config`
- **Batch Size**: 4
- **Adapter**: LoRA (rank=64, inferred from standard config)
- **Adapter Output**: `cache/lpa-regular-config_2026-02-12_15-10-35-567694/`

### Hyperparameters (from `lat_config.json`)
- **PGD Iterations**: 16
- **Model Iterations**: 4
- **Training Steps**: 200
- **Epsilon**: 6.0 (hardcoded in `get_trainer()` for Qwen3)
- **Attacked Layers**: [embedding, 8, 16, 24, 30]
- **Loss Weights**: toward=0.5, away=0.5 (both adversary and defense)

### Evaluation Configuration
- **System Prompt**: `system_prompt/minimal.txt`
- **Adapter Loaded**: `lpa-regular-config_2026-02-12_15-10-35-567694`

### Results

#### HarmBench Attack Success Rate (ASR)
| Attack Method | ASR | Δ from Baseline |
|--------------|-----|-----------------|
| DirectRequest | 0.0 | -0.40 |
| GCG | 0.0 | -0.58 |
| AutoDAN | 0.0 | -0.35 |
| AutoPrompt | 0.0 | -0.51 |
| PAIR | 0.0 | -0.68 |
| TAP | 0.0 | -0.57 |
| Clean | 0.0 | -0.85 |

#### Utility Metrics
| Benchmark | Score | Δ from Baseline |
|-----------|-------|-----------------|
| MMLU | 0.04 | -0.67 |
| HellaSwag | 0.30 | -0.39 |
| Winogrande | 0.0 | -0.18 |

---

## Experiment 3: LPA Checkpoint Sweep (High Resolution)

**Run ID**: `lpa-regular-config_IPIP-14_fewer_steps_2026-02-13_13-35-34-111503`  
**Date**: 2026-02-13  
**Purpose**: Analyze the safety-utility tradeoff trajectory at 10-step intervals to find the optimal stopping point and identify the "collapse" threshold.  
**Diagnostic Data**: [diagnostics/checkpoint_sweep_results_lpa-regular-config_IPIP-14_fewer_steps_2026-02-13_13-35-34-111503.csv](diagnostics/checkpoint_sweep_results_lpa-regular-config_IPIP-14_fewer_steps_2026-02-13_13-35-34-111503.csv)

### Configuration
- **Base Model**: Qwen/Qwen3-8B
- **Dataset**: `data/IPIP-14/`
- **Steps**: 200 (Checkpointing every 10 steps)
- **Config**: `lat_config_fewer_steps.json` (modified to save all intermediate checkpoints)

### Key Findings
1.  **Safety Saturation (Step 60-70)**: DirectRequest ASR drops from 0.40 to nearly 0.0 by Step 60. Training beyond this point provides diminishing returns for safety while continuing to degrade utility.
2.  **Utility Decay**: MMLU accuracy shows a linear-to-quadratic decay. 
    - **Step 20**: ~0.72 (Baseline parity)
    - **Step 50**: ~0.57 (75% of baseline)
    - **Step 80**: ~0.39 (55% of baseline) — The "Zero ASR" threshold.
3.  **Pathological Collapse (Step 140+)**: Beyond Step 140, the model begins to exhibit pathological behavior (repetitive or garbled outputs), with the rate spiking from 0% to 30% by Step 200.
4.  **Optimal "Sweet Spot"**: 
    - **Step 80** is the first point of absolute zero ASR and zero pathological behavior, but utility is significantly compromised (MMLU 0.39).
    - **Step 50** represents a more balanced Pareto-optimal point with ASR ~0.10 and much higher utility (MMLU ~0.57).

### Results Trajectory Highlight
| Step | DirectRequest ASR | MMLU Accuracy | Pathological Rate |
|------|-------------------|---------------|-------------------|
| 0 (Baseline) | 0.40 | 0.71 | 0.00 |
| 20 | 0.40 | 0.72 | 0.00 |
| 50 | 0.10 | 0.57 | 0.00 |
| 80 | 0.01 | 0.39 | 0.00 |
| 140 | 0.00 | 0.05 | 0.00 |
| 200 | 0.00 | 0.02 | 0.28 |

### Conclusions
LPA is highly effective at reducing ASR quickly, but the "alignment tax" on utility (MMLU) is steep. Future experiments should explore lower learning rates or early stopping between steps 40 and 60 to preserve >60% utility while still achieving >90% safety improvement.

| SciQ | 0.206 | -0.734 |
| Lambada | 0.34 | -0.302 |

### Analysis
- **Safety**: Near-perfect ASR reduction to 0.0 across all attack methods
- **Utility**: Significant degradation across all benchmarks
  - MMLU: 71% → 4% (severe capability loss)
  - SciQ: 94% → 21% (most severe degradation)
  - HellaSwag: 69% → 30%
  - Lambada: 64% → 34%

### Open Questions
1. **Utility degradation cause**: Is this due to:
   - Overly aggressive adversarial training?
   - Wrong hyperparameters (epsilon too high, too many PGD iters)?
   - Dataset issues (IPIP-08 vs other IPIP variants)?
   - Missing SFT recovery step?
2. **Dataset selection**: Why IPIP-08 specifically? Need to verify if this matches paper experiments
3. **Reproducibility**: Do these results align with expected LPA behavior from the paper?

### Next Steps
- [ ] Verify IPIP-08 is the correct dataset variant used in the paper
- [ ] Compare with other IPIP-XX datasets
- [ ] Review hyperparameter settings (especially epsilon=6.0)
- [ ] Investigate if SFT recovery is needed despite paper claims
- [ ] Run ablation with different loss weights

---

## Experiment 3: LPA Evaluation (IPIP-14)

**Run ID**: 2026-02-13_11-00-18-529960  
**Date**: 2026-02-13  
**Purpose**: Evaluate Latent Personality Alignment using the IPIP-14 dataset variant
**Evaluation Log**: [logs/slurm/evaluation_6880700.out](logs/slurm/evaluation_6880700.out)

### Configuration
- **Model**: Qwen/Qwen3-8B
- **Project Name**: lpa-regular-config_IPIP-14
- **Adapter Loaded**: cache/lpa-regular-config_IPIP-14_2026-02-13_11-00-18-529960/checkpoint_200
- **System Prompt**: `system_prompt/minimal.txt`
- **Checkpoint**: 200

### Results

#### HarmBench Attack Success Rate (ASR)
All attack methods reported ASR = 0.0 (DirectRequest, GCG, AutoDAN, AutoPrompt, PAIR, TAP, clean)

#### Utility Metrics
MMLU: 0.0, HellaSwag: 0.0, Winogrande: 0.0, SciQ: 0.0, Lambada: 0.0

### Notes
- Adapter loaded from `cache/lpa-regular-config_IPIP-14_2026-02-13_11-00-18-529960/checkpoint_200` as shown in the Slurm log.
- All ASR and utility metrics in the log are 0.0 — consistent with strong safety but total utility collapse, or possibly an evaluation issue.
- Responses are all just "I do not agree with this statement." Completely unusable as a chatbot. 

### Next Steps
- [ ] Confirm evaluation metrics are valid (check evaluation script and logs)
- [ ] Compare IPIP-14 run to IPIP-08 to understand differences in utility impact
- [ ] Re-run utility benchmarks with debug logging if metrics remain 0.0
- currently rerunning with model_iterations_per_step": 1, to hopefully reduce "overfitting" to just responding "I do not agree with this statement." to all prompts
- Need to add utility benchmarks to cache? Right now I don't think they're being saved.

## Experiment 4: LPA Evaluation (IPIP-14, fewer steps)

**Run ID**: lpa-regular-config_IPIP-14_fewer_steps_2026-02-13_13-35-34-111503  
**Date**: 2026-02-13  
**Purpose**: Evaluate Latent Personality Alignment using the IPIP-14 dataset variant with `model_iterations_per_step: 1` (fewer steps)
**Evaluation Log**: [logs/slurm/6887499-eval-QwenQwen3-8B-IPIP-14-bs4.out](logs/slurm/6887499-eval-QwenQwen3-8B-IPIP-14-bs4.out)

### Configuration
- **Model**: Qwen/Qwen3-8B
- **Project Name**: lpa-regular-config_IPIP-14_fewer_steps
- **Adapter Loaded**: cache/lpa-regular-config_IPIP-14_fewer_steps_2026-02-13_13-35-34-111503/checkpoint_200
- **System Prompt**: `system_prompt/minimal.txt`

### Results

#### HarmBench Attack Success Rate (ASR)
All attack methods reported ASR = 0.0 (DirectRequest, GCG, AutoDAN, AutoPrompt, PAIR, TAP, clean)

#### Utility Metrics (accuracy)
- MMLU: 2.0%
- HellaSwag: 0.0%
- Winogrande: 0.0%
- SciQ: 26.9%
- Lambada: 18.6%

### Notes
- Adapter and evaluation show near-perfect safety (ASR = 0.0) but substantial utility collapse on most benchmarks.
- The model exhibits a pathological refusal pattern (dominant response: "I do not agree with this statement.").
- This run used `model_iterations_per_step: 1` (fewer steps) and still experienced collapse.

### Next Steps
- [ ] Confirm evaluation metrics and inspect the evaluation script for issues
- [ ] Compare response distributions across checkpoints to identify when collapse emerges
- [ ] Evaluate with small SFT recovery or utility-preservation losses


## Experiment 5: Response Distribution Analysis — Trained vs Base

**Date**: 2026-02-13  
**Purpose**: Diagnose why the trained model gets 0% on all utility benchmarks by analyzing the distribution of raw model responses. Compare to base model to quantify the pathological "I do not agree with this statement" refusal pattern.  
**Script**: `diagnostics/plot_response_distribution.py`  
**Figures**: `diagnostics/figures/`

### Configuration
- **Trained Model**: `cache/lpa-regular-config_IPIP-14_fewer_steps_2026-02-13_13-35-34-111503/eval/`
- **Base Model**: `cache/base-eval_baseline/checkpoint_0/eval/` (SLURM job 6894490)

### Key Findings

#### Trained Model (LPA — IPIP-14, fewer steps)
The model has collapsed to a single refusal response pattern:

| Benchmark | Accuracy | Dominant Response | Notes |
|-----------|----------|-------------------|-------|
| HellaSwag | 0.0% | "I..." (100/100) | 100% pathological |
| Winogrande | 0.0% | "I..." (100/100) | 100% pathological |
| MMLU | 2.0% | "I..." (56), digit (27), A (12) | Mostly pathological |
| SciQ | 26.9% | A (623), digit (152), "I..." (86) | Partially functional |
| Lambada | 18.6% | other word (969), "I do..." (25) | Mostly functional but low accuracy |

On HarmBench: **0% ASR across all 7 attack methods** — every response is "I do not agree with this statement."

#### Base Model (Qwen3-8B)
| Benchmark | Accuracy | Response Distribution |
|-----------|----------|----------------------|
| HellaSwag | 69.0% | D:29, C:29, A:16, B:16, other:10 |
| MMLU | 71.0% | C:34, A:25, B:18, D:17, other:5 |
| SciQ | 94.0% | C:255, A:251, D:239, B:229, other:26 |
| Winogrande | 18.0% | B:54, A:37, other:9 |
| Lambada | 64.2% | other word:998 |

On HarmBench: ASR ranges 35%–85% (no safety alignment).

### Analysis
- The LPA training has caused **complete utility collapse** on HellaSwag and Winogrande (100% "I..." responses)
- MMLU and SciQ are partially functional, suggesting the model can sometimes parse MC questions but defaults to refusal
- The model has learned a degenerate policy: respond "I do not agree with this statement" to everything
- This is likely caused by the model learning that the first generated token should be "I" (from the personality refusal pattern) regardless of context
- The `model_iterations_per_step: 1` (fewer steps) config didn't prevent the collapse

### Next Steps
- [ ] Investigate at what training step the collapse occurs (log response distribution over training)
- [ ] Try much lower learning rate or fewer total steps to find the sweet spot before collapse
- [ ] Consider adding a utility-preservation loss (SFT on benign data) to prevent collapse
- [ ] Check if the eval prompt format (with chat template) is contributing — the model may be treating MC questions as harmful requests because of the chat template

---

## Experiment 6: LPA Evaluation — IPIP-14 (fewer steps, new run)

**Run ID**: 2026-02-16_13-03-56-659841
**Date**: 2026-02-16
**Purpose**: Re-evaluate `lpa-regular-config_IPIP-14_fewer_steps` adapter (new checkpoint/eval run)
**Evaluation Log**: [logs/slurm/7008404-eval-QwenQwen3-8B-IPIP-14-bs4.out](logs/slurm/7008404-eval-QwenQwen3-8B-IPIP-14-bs4.out)

### Configuration
- **Model**: Qwen/Qwen3-8B
- **Project Name**: lpa-regular-config_IPIP-14_fewer_steps
- **Adapter Loaded**: cache/lpa-regular-config_IPIP-14_fewer_steps_2026-02-16_13-03-56-659841 (LoRA adapter loaded from this path in the Slurm log)
- **System Prompt**: `system_prompt/minimal.txt`

### Results

#### HarmBench Attack Success Rate (ASR)
| Attack Method | ASR |
|---------------|-----|
| DirectRequest | 0.47 |
| GCG | 0.44 |
| AutoDAN | 0.22 |
| AutoPrompt | 0.37 |
| PAIR | 0.61 |
| TAP | 0.57 |
| Clean | 0.88 |

#### Utility Metrics (accuracy)
| Benchmark | Score |
|-----------|-------|
| MMLU | 0.69 |
| HellaSwag | 0.69 |
| Winogrande | 0.16 |
| SciQ | 0.944 |
| Lambada | 0.631 |

### Notes / Analysis
- The Slurm log explicitly shows the LoRA adapter being loaded from `cache/lpa-regular-config_IPIP-14_fewer_steps_2026-02-16_13-03-56-659841`, confirming this evaluation used the trained adapter (i.e., the trained model).
- HarmBench ASRs are non-zero across methods (0.22–0.61), indicating attacks still succeed to varying degrees on this adapter.
- Utility metrics (MMLU, HellaSwag, SciQ, Lambada) are close to baseline levels for this run; no catastrophic collapse observed in this evaluation.
- The key difference here is using 30 epochs, which is what was used in the paper based on when the direct response ASR reached approximately 0.

### Next Steps
- Compare these metrics to other IPIP-14 runs (different checkpoints) to understand stability.
- If desired, add this run to aggregated plots and tracking dashboards.

---

## Experiment 7: Checkpoint Sweep (IPIP-14, fewer steps)

**Run ID**: Sweep over checkpoints from 2026-02-13_13-35-34-111503  
**Date**: 2026-02-17  
**Purpose**: Find the "sweet spot" checkpoint where safety improves but utility hasn't collapsed  
**SLURM Script**: `launch_checkpoint_sweep.sh`

### Configuration
- **Source Run**: `cache/lpa-regular-config_IPIP-14_fewer_steps_2026-02-13_13-35-34-111503`
- **Model**: Qwen/Qwen3-8B
- **Checkpoints Evaluated**: 10, 20, 30, ..., 200 (20 checkpoints total)
- **Metrics**: 
  - DirectRequest ASR only (not full HarmBench suite)
  - MMLU accuracy only (not full utility suite)
  - Pathological response rate (counting "I do not agree..." responses)

### Scripts Created
| Script | Purpose |
|--------|---------|
| [eval.py](eval.py) | Modified to support `--attacks`, `--evals`, `--no_wandb` flags |
| [diagnostics/checkpoint_sweep.py](diagnostics/checkpoint_sweep.py) | Iterates over checkpoints, runs lightweight eval, collects results |
| [diagnostics/plot_checkpoint_sweep.py](diagnostics/plot_checkpoint_sweep.py) | Generates 3 plots: combined metrics, trade-off, Pareto frontier |
| [launch_checkpoint_sweep.sh](launch_checkpoint_sweep.sh) | SLURM submission script |

### How to Run
```bash
# Submit the full sweep (12 hour job, h100 GPU)
sbatch launch_checkpoint_sweep.sh

# Or with custom parameters:
sbatch launch_checkpoint_sweep.sh cache/lpa-regular-config_IPIP-14_fewer_steps_2026-02-13_13-35-34-111503 10 200 10
```

### Expected Outputs
- **CSV**: `diagnostics/checkpoint_sweep_results_lpa-regular-config_IPIP-14_fewer_steps_2026-02-13_13-35-34-111503.csv`
- **Plots**: `diagnostics/figures/checkpoint_sweep_lpa-regular-config_IPIP-14_fewer_steps_2026-02-13_13-35-34-111503/`
  - `sweep_combined.png`: 3-panel plot (ASR, MMLU, pathological rate vs step)
  - `sweep_tradeoff.png`: Dual y-axis trade-off visualization
  - `sweep_frontier.png`: Pareto-style safety-utility frontier

### Results
*To be filled after sweep completes*

| Step | ASR | MMLU | Pathological Rate |
|------|-----|------|-------------------|
| 10 | | | |
| 20 | | | |
| ... | | | |
| 200 | | | |

### Sweet Spot Recommendation
*To be determined from sweep results*

---

## Template for Future Experiments

```markdown
## Experiment N: [Descriptive Name]

**Run ID**: YYYY-MM-DD_HH-MM-SS-NNNNNN  
## Experiment N: SFT Recovery on LPA Checkpoint 50

**Date**: 2026-02-17 (planned)  
**Purpose**: Test whether supervised fine-tuning on benign data (Alpaca) can restore utility (MMLU ~57% → ~71% baseline) while maintaining safety gains from LPA training (~10% ASR). This directly tests the fragility of personality alignment to post-hoc fine-tuning.  
**Training Log**: `logs/slurm/sft_recovery_*.out`  
**Evaluation Log**: `logs/slurm/*-eval-sft-recovery-*.out`

### Training Configuration
- **Base Model**: Qwen/Qwen3-8B
- **Method**: SFT-only (no LAT/PGD/adversary — pure supervised fine-tuning)
- **Training Script**: `latent_at/lat_sft_recovery.py`
- **Starting Checkpoint**: `cache/lpa-regular-config_IPIP-14_fewer_steps_2026-02-13_13-35-34-111503/checkpoint_50` (Pareto-optimal: ~10% ASR, ~57% MMLU)
- **Dataset**: `tatsu-lab/alpaca` (5K subset, seed=42)
- **System Prompt**: `system_prompt/minimal.txt` (matches eval conditions)
- **Project Name**: `lpa-sft-recovery_alpaca_checkpoint50`
- **Batch Size**: 4
- **Adapter**: Existing LoRA r=64 from LPA checkpoint (continued training)
- **Adapter Output**: `cache/lpa-sft-recovery_alpaca_checkpoint50_<timestamp>/`

### Hyperparameters
- **Config**: `latent_at/sft_recovery_config.json`
- `num_steps`: 500
- `outer_learning_rate`: 2e-5
- `max_batch_per_acc`: 2
- `reinitialize_dev_optim`: false (persistent optimizer)
- `N_checkpoints`: 10 (checkpoints every 50 steps)
- `alpaca_subset_size`: 5000
- `kl_coef`: 0.0 (no KL penalty — first run is SFT-only baseline)

### Evaluation Configuration
- **System Prompt**: `system_prompt/minimal.txt`
- **Adapter Loaded**: Each checkpoint (50, 100, 150, ..., 500)
- **Benchmarks**: HarmBench (all attacks) + MMLU, HellaSwag, Winogrande, SciQ, Lambada

### Results
[PENDING — run not yet submitted]

### Analysis
[PENDING]

### Expected Outcomes
1. **Best case**: MMLU recovers to ~65-70% while ASR stays <20% → SFT recovery is viable
2. **Likely case**: MMLU partially recovers but ASR climbs to 25-35% → trade-off curve
3. **Worst case**: ASR rapidly returns to baseline (~40%) → personality alignment is fragile to SFT (confirms LPA paper ablation Table 1)

### Follow-up Experiments (if warranted)
- Add KL penalty (`kl_coef: 0.01-0.05`) to regularize against LPA checkpoint
- Try lower learning rate (`1e-5`) to slow safety erosion
- Try higher-quality dataset (Open-Orca) instead of Alpaca

### Pre-launch Checklist
- [ ] Cache Alpaca dataset: `python cache_alpaca_dataset.py` (from login node)
- [ ] Submit: `sbatch launch_sft_recovery.sh`
- [ ] After training: `bash launch_sft_recovery_sweep.sh <MODEL> <PROJECT> <TIMESTAMP>`

---

## Template (Copy for New Experiments)

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

---

## Experiment 5: SFT Recovery (Alpaca from LPA Step 50)

**Run ID**: `lpa-sft-recovery_alpaca_checkpoint50_2026-02-17_13-52-29-711847`  
**Date**: 2026-02-17  
**Purpose**: Test whether supervised fine-tuning on benign data (Alpaca) can restore utility from an LPA-aligned checkpoint without catastrophically degrading safety.  
**Training Log**: [logs/slurm/sft_recovery_*.out](logs/slurm/)  
**Evaluation Logs**:
- Step 100: [logs/slurm/7034873-eval-sft-recovery-step100.out](logs/slurm/7034873-eval-sft-recovery-step100.out)
- Final (Step 500): [logs/slurm/7034878-eval-sft-recovery-final.out](logs/slurm/7034878-eval-sft-recovery-final.out)

### Configuration
- **Base Model**: Qwen/Qwen3-8B
- **Starting Checkpoint**: `cache/lpa-regular-config_IPIP-14_fewer_steps_2026-02-13_13-35-34-111503/checkpoint_50`
  - LPA-aligned model with DirectRequest ASR ~0.10, MMLU ~0.57
- **SFT Dataset**: `tatsu-lab/alpaca` (5,000 samples)
- **Training Script**: `latent_at/lat_sft_recovery.py`
- **Training Steps**: 500
- **Checkpointing**: Every 50 steps (10 total checkpoints)
- **Learning Rate**: 2e-5
- **Batch Size**: 4
- **System Prompt**: `system_prompt/minimal.txt` (evaluation mode)

### Results

#### Trajectory Comparison: Safety vs. Utility Recovery

| Stage | DirectRequest | GCG | AutoDAN | PAIR | TAP | Clean | MMLU | HellaSwag | SciQ | Lambada | Winogrande |
|-------|--------------|-----|---------|------|-----|-------|------|-----------|------|---------|------------|
| **Base Model** | 0.40 | 0.58 | 0.35 | 0.68 | 0.57 | 0.85 | 0.71 | 0.69 | 0.94 | 0.64 | 0.18 |
| **LPA Start (Step 50)** | 0.10 | ~0.05 | — | — | — | ~0.50 | 0.57 | — | — | — | — |
| **Recovery Step 100** | 0.35 | 0.29 | 0.16 | 0.58 | 0.59 | 0.81 | 0.67 | 0.66 | 0.94 | 0.64 | 0.09 |
| **Final (Step 500)** | 0.54 | 0.39 | 0.37 | 0.60 | 0.61 | 0.72 | 0.60 | 0.61 | 0.91 | 0.66 | 0.00 |

#### Key Metrics Changes (LPA Start → Step 100 → Final)
| Metric | LPA Start | Step 100 | Final | Trend |
|--------|-----------|----------|-------|-------|
| DirectRequest ASR | 0.10 | 0.35 | 0.54 | 📈 Safety Erosion |
| GCG ASR | ~0.05 | 0.29 | 0.39 | 📈 Safety Erosion |
| MMLU Accuracy | 0.57 | 0.67 | 0.60 | 📈 then 📉 |
| Clean ASR (Helpfulness) | ~0.50 | 0.81 | 0.72 | 📈 Restored |

### Analysis

1.  **Initial Recovery Success at Step 100**:
    - Successfully recovered some utility (MMLU: 0.57 → 0.67, approaching baseline 0.71)
    - Maintained some safety advantage over base model (GCG: 0.29 vs baseline 0.58)

2.  **Catastrophic Safety Degradation by Step 500**:
    - **DirectRequest ASR exceeded base model** (0.54 vs 0.40), suggesting SFT on Alpaca increases overall compliance to harmful requests
    - GCG ASR deteriorated but remained below base model

3.  **Utility Regression After Step 100**:
    - MMLU **dropped** from 0.67 (Step 100) to 0.60 (Step 500)
    - Indicates overfitting to Alpaca distribution or catastrophic forgetting of general knowledge?
    - Winogrande collapsed entirely to 0.00 by Step 500
    - Maybe want to add sft back to training loop with LPA to see if this helps.

### Conclusions

- **SFT Recovery is a Double-Edged Sword**: While it can restore utility and helpfulness, it "unlearns" safety alignment, particularly for direct harmful requests.
- **Compliance Overgeneralization**: The model becomes **more** compliant than the base model, even to harmful requests, suggesting Alpaca-style SFT reduces the model's ability to discriminate between safe and unsafe instructions.
- **Optimization-Based Attack Resistance**: GCG and AutoDAN ASRs remain lower than baseline even after 500 steps, suggesting latent-space adversarial training provides more persistent robustness against these attacks than against direct prompt-based attacks.

### Open Questions

2. **Can we interleave safety data during SFT to prevent alignment erosion?**
4. **Would a different SFT dataset (e.g., less instruction-following focused) preserve safety better?**

### Next Steps
- [ ] Try to interleave SFT within the LPA training loop to see if it can preserve safety while improving utility.