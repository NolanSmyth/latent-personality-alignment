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


## Template for Future Experiments

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
  - Description of data and purpose
- **System Prompt**: 
- **Project Name**: 
- **Batch Size**: 
- **Adapter**: 
- **Adapter Output**: 

### Hyperparameters
- List key hyperparameters that differ from defaults
- Or reference config file if using standard settings

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
