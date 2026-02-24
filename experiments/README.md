# Experiments Index

Each file in this directory corresponds to one experiment. Load only what you need.

**Defaults** (omitted from individual experiment configs unless changed):
- Model: `Qwen/Qwen3-8B`, Script: `lat_training_no_sft.py`, Config: `lat_config_fewer_steps.json`
- 100 steps, 16 PGD iters, 4 model iters per step, ε=6.0, batch size 4
- Dataset: `data/IPIP-14/harmful_trait.csv` (67 statements: 28 agree, 39 disagree)
- System prompt: `system_prompt/alpha.txt` (train), `system_prompt/minimal.txt` (eval)
- SFT/KL losses: **disabled**

---

| File | Date | Short Name | One-line Conclusion |
|------|------|-----------|---------------------|
| [EXP-001](EXP-001.md) | 2026-02-12 | Baseline Evaluation | Qwen3-8B has 35–85% ASR, MMLU 0.71 — reference numbers for all experiments. |
| [EXP-002](EXP-002.md) | 2026-02-12 | LPA Training — IPIP-08 (200 steps) | Full safety (0% ASR) but catastrophic utility collapse (MMLU 4%); 200 steps is too many. |
| [EXP-003](EXP-003.md) | 2026-02-13 | Response Distribution Analysis | Trained model says "I do not agree" to *everything*, explaining zero utility scores. |
| [EXP-004](EXP-004.md) | 2026-02-17 | Checkpoint Sweep Analysis | Step-50 confirmed as sweet spot; scripts + plots created for visualisation. |
| [EXP-005](EXP-005.md) | 2026-02-17 | SFT Recovery from Step-50 | Post-hoc Alpaca SFT erodes LPA safety — DirectRequest ASR overshoots baseline (0.54 vs 0.40). |
| [EXP-006](EXP-006.md) | 2026-02-18 | LPA-only — 30 steps | Pure LPA at 30 steps is functionally equivalent to baseline on DirectRequest; utility fully intact. |
| [EXP-007](EXP-007.md) | 2026-02-21 | All-Positive Overfitting Check (IPIP-10) | All-positive training causes agree-bias generalization and catastrophic utility collapse at step 100. |
| [EXP-008](EXP-008.md) | 2026-02-21 | Mixed Overfitting + Loss Monitoring (IPIP-14) | ⚠️ Pending — SLURM job 7228439. |
| [EXP-009](EXP-009.md) | 2026-02-18 | Interleaved LPA+SFT — 100 steps | With-SFT is non-monotonic (step 50 best: DR 0.26); utility stable (MMLU 0.65) — no collapse. |
| [EXP-010](EXP-010.md) | 2026-02-18 | LPA-only — 100 steps | Without-SFT improves monotonically then **collapses** at step 100 (MMLU 0.40, DR/clean=0.00). |
| [EXP-011](EXP-011.md) | 2026-02-21 | Base Model Prior Probe | Base model *refuses* negative IPIP items (not agrees); LPA converts refusals into agreements  |
| [EXP-012](EXP-012.md) | 2026-02-21 | Token Probability Diagnostic | P(disagree) > P(agree) on 71.8% of negative items — agree-collapse is *not* a per-item agreeableness bias. |
| [EXP-013](EXP-013.md) | 2026-02-21 | Negative-Only Training (IPIP-14-neg) | 100% accuracy on negative items when trained alone |
| [EXP-014](EXP-014.md) | 2026-02-21 | Per-Item δ Norm Logging | Adversary perturbation magnitudes are symmetric (neg/pos ratio 0.927) |
| [EXP-015](EXP-015.md) | 2026-02-23 | Loss Ablation: Toward-Only / Away-Only / Balanced | Toward-only resolves agree-collapse at step 100 but utility collapses; away-only breaks format with no safety gain; balanced achieves 100% probe accuracy but also collapses utility. |
| [EXP-016](EXP-016.md) | 2026-02-23 | Adv-Both Def-Away + Checkpoint Sweep | Toward-only adversary + balanced defense at step 40: ASR=0.000, MMLU=0.510 — best tradeoff to date without SFT. |
| [EXP-017](EXP-017.md) | 2026-02-23 | PGD Gradient Direction Diagnostic | δ_pos and δ_neg are geometrically aligned (cos≈0.43–0.87); gradient cancellation is NOT the cause of agree-collapse. |
