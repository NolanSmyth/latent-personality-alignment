# TODOs & Research Questions

*Action items and open research directions. Active focus is in [current_state.md](current_state.md). Experiment-specific next steps are in [experiment_results.md](experiment_results.md).*

---

## Experiment 1: All-Positive Overfitting Check (IPIP-10, no SFT)

**Goal**: Train on all-positive statements. Confirm the model overfits to always responding "I agree" — the simplest controlled sanity check that training is doing what we think.

**Dataset confirmed**: `data/IPIP-10/` — 58 statements, `chosen` is always `"I agree with this statement"` (positive traits: empathy, agreeableness, forgiveness, etc.). `benign_trait.csv` present (unused by `lat_training_no_sft.py`, required by arg parser).

**Expected**: Model should say "I agree" to all IPIP-10 items. As a cross-dataset overfitting check, it may also say "I agree" to negative statements from IPIP-14.

### Subtasks

- [x] **1a. Run training** on IPIP-10 (no SFT, 100 steps, 4 model iters via `lat_config_fewer_steps.json`): SLURM job 7227985 submitted 2026-02-21
  ```
  sbatch launch_experiment.sh Qwen/Qwen3-8B IPIP-10 system_prompt/alpha.txt lpa-ipip10-allpositive 4
  ```
  *(Use step-50 checkpoint for diagnostics — Pareto-optimal from prior sweep.)*
- [x] **1b. Write `diagnostics/probe_ipip_responses.py`**: loads a LoRA checkpoint + base model, feeds each IPIP statement from a given CSV through the model with the `alpha.txt` system prompt, records the first meaningful token(s) of the response (agree / do not agree / other), outputs per-item results and a summary accuracy. Needs to handle Qwen3's `<think>` block in the output.
- [ ] **1c. Run diagnostic on IPIP-10 checkpoint (step 50)**: does the model always agree with all 58 items?
- [ ] **1d. Cross-dataset probe**: run the same diagnostic on IPIP-14 items using the IPIP-10 checkpoint. Does the model agree even with negative statements? This is the overfitting "generalization" check.
- [ ] **1e. Inspect HarmBench/utility results** auto-submitted by `launch_experiment.sh`. Inline evals run every 10 steps (DirectRequest + MMLU only).

---

## Experiment 2: Mixed Overfitting Check (IPIP-14, no SFT)

**Goal**: Train on mixed statements. Confirm the model learns to both agree AND disagree depending on the statement — not collapsing to a single response. The signal that agreement/disagreement is conditioned on the content of the statement is key.

**Dataset confirmed**: `data/IPIP-14/` — 67 statements (28 agree + 39 disagree). This is the existing standard LPA dataset.

**Decision**: Run fresh (not the existing checkpoint) with `model_iterations_per_step: 4` and inline loss + eval monitoring. Combined with Experiment 3 in one job.

**Expected**: Model should correctly agree with positive statements and disagree with negative ones. Accuracy should be clearly above 50% (random) on both subsets.

### Subtasks

- [x] **2a. Run fresh training** on IPIP-14 (no SFT, 100 steps, 4 model iters, `--eval --eval_freq 10`): SLURM job 7227986 submitted 2026-02-21
  ```
  sbatch launch_experiment.sh Qwen/Qwen3-8B IPIP-14 system_prompt/alpha.txt lpa-ipip14-mixed 4
  ```
- [x] **2b. Write `diagnostics/probe_ipip_responses.py`**: loads a LoRA checkpoint + base model, feeds each IPIP statement from a given CSV through the model with the `alpha.txt` system prompt, records the first meaningful token(s) of the response (agree / do not agree / other), outputs per-item results and a summary accuracy. Needs to handle Qwen3's `<think>` block in the output.
- [ ] **2c. Run diagnostic** on step-50 checkpoint: per-item agree/disagree accuracy, breakdown by statement valence (agree-labeled vs. disagree-labeled items).
- [ ] **2d. Compare to IPIP-10 results**: does the model correctly distinguish between positive and negative statements here (unlike in Exp 1 where it should just always agree)?

---

## Experiment 3: Loss Term Monitoring During Training

**Goal**: Log all four per-term losses — `adv_toward`, `adv_away`, `def_toward`, `def_away` — at every step alongside inline DirectRequest + MMLU benchmarks. Determine whether one loss term starts to dominate or loses signal as training progresses.

**No code fixes needed**: Namespacing is correct — `do_adversary_step`/`do_defense_step` already prefix with `adv_`/`def_` (underscore). W&B confirms all four series appear correctly.

**Config decisions made**:
- Inline evals: DirectRequest + MMLU only (updated in `lat_training_no_sft.py`)
- Eval frequency: every 10 steps (enabled with `--eval --eval_freq 10` in `launch_experiment.sh`)
- Combined with Exp 2 — IPIP-14 fresh run covers both

### Subtasks

- [x] **3a. Verify W&B namespacing** — already correct, no fix needed
- [x] **3b. Restrict inline evals** to DirectRequest + MMLU (`only_run_evals=["DirectRequest"]`, `evals_to_include=["MMLU"]` in `evaluate_model`)
- [x] **3c. Enable eval** in `launch_experiment.sh` (`--eval --eval_freq 10`), disable lm_eval auto-submission
- [ ] **3d. Run** — covered by Exp 2's `lpa-ipip14-mixed` job
- [ ] **3e. Analyze W&B**: plot `adv_toward`, `adv_away`, `def_toward`, `def_away` on the same axes. Note where any term's gradient signal becomes negligible or one term dominates.
