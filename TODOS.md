# TODOs & Research Questions

*Action items and open research directions. Active focus is in [current_state.md](current_state.md). Experiment-specific next steps are in [experiment_results.md](experiment_results.md).*

---

## Experiment 4: Base Model Prior Probe

**Goal**: Determine the base Qwen3-8B prior response to IPIP-14 statements in the exact context the LAT adversary operates in. This is the critical missing data point for the adversary-gradient argument.

**Background**: Initial base model probe run failed — all 67 items returned `other: <think>...` at 0% accuracy. Root cause: the probe used `apply_chat_template(add_generation_prompt=True)` which emits only `<|im_start|>assistant\n`, causing the model to open a free-form `<think>` block that exceeds `max_new_tokens=80`. The training template pre-fills `<think>\n\n</think>\n\n`, so the adversary's loss targets tokens *after* the think block, not inside it.

**Fix applied**: Added `--use_training_template` flag to `diagnostics/probe_ipip_responses.py` and increased default `max_new_tokens` 80 → 512.

### Subtasks

- [x] **4a.** Bug identified and fixed in `probe_ipip_responses.py` (`--use_training_template` flag, increased `max_new_tokens`)
- [x] **4b.** Updated `launch_probe_diagnostics.sh` with base model prior probes (both free-gen and training-template modes) and `--use_training_template` on all checkpoint probes
- [x] **4c.** Ran base model probe with `--use_training_template` on IPIP-14
- [x] **4d.** **Result: INVALIDATES the unstressed-adversary hypothesis.** Base model says "I'm sorry, I cannot comply" to ~25/39 negative items, echoes ~11/39, and says "I agree" to only 2/39. The adversary target ("I agree") has high CE against the refusal prior → adversary IS stressed. See Exp 16 in experiment_results.md.
- [ ] **4e.** Re-verify previous Exps 1c/1d/2c/2d results with `--use_training_template`.

---



**Goal**: Train on all-positive statements. Confirm the model overfits to always responding "I agree" — the simplest controlled sanity check that training is doing what we think.

**Dataset confirmed**: `data/IPIP-10/` — 58 statements, `chosen` is always `"I agree with this statement"` (positive traits: empathy, agreeableness, forgiveness, etc.). `benign_trait.csv` present (unused by `lat_training_no_sft.py`, required by arg parser).

**Expected**: Model should say "I agree" to all IPIP-10 items. As a cross-dataset overfitting check, it may also say "I agree" to negative statements from IPIP-14.

### Subtasks

- [x] **1a. Run training** on IPIP-10 (no SFT, 100 steps, 4 model iters via `lat_config_fewer_steps.json`): SLURM job 7227985 ❌ failed (OOM during inline eval); resubmitted as job **7228438** (2026-02-21, no inline eval)
  ```
  sbatch launch_experiment.sh Qwen/Qwen3-8B IPIP-10 system_prompt/alpha.txt lpa-ipip10-allpositive 4
  ```
- [x] **1b. Write `diagnostics/probe_ipip_responses.py`**: loads a LoRA checkpoint + base model, feeds each IPIP statement from a given CSV through the model with the `alpha.txt` system prompt, records the first meaningful token(s) of the response (agree / do not agree / other), outputs per-item results and a summary accuracy. Needs to handle Qwen3's `<think>` block in the output.
- [x] **1c. Run diagnostic on IPIP-10 checkpoint (step 50)**: does the model always agree with all 58 items? → SLURM job **7229258** (combined with 1d and 2c)
- [x] **1d. Cross-dataset probe**: run the same diagnostic on IPIP-14 items using the IPIP-10 checkpoint. Does the model agree even with negative statements? → SLURM job **7229258**
- [ ] **1e. Inspect HarmBench/utility results** auto-submitted by `launch_experiment.sh` (post-training eval job, separate from training — evals still running).

---

## Experiment 2: Mixed Overfitting Check (IPIP-14, no SFT)

**Goal**: Train on mixed statements. Confirm the model learns to both agree AND disagree depending on the statement — not collapsing to a single response. The signal that agreement/disagreement is conditioned on the content of the statement is key.

**Dataset confirmed**: `data/IPIP-14/` — 67 statements (28 agree + 39 disagree). This is the existing standard LPA dataset.

**Decision**: Run fresh (not the existing checkpoint) with `model_iterations_per_step: 4` and inline loss + eval monitoring. Combined with Experiment 3 in one job.

**Expected**: Model should correctly agree with positive statements and disagree with negative ones. Accuracy should be clearly above 50% (random) on both subsets.

### Subtasks

- [x] **2a. Run fresh training** on IPIP-14 (no SFT, 100 steps, 4 model iters): SLURM job 7227986 ❌ failed (OOM during inline eval); resubmitted as job **7228439** (2026-02-21, no inline eval)
  ```
  sbatch launch_experiment.sh Qwen/Qwen3-8B IPIP-14 system_prompt/alpha.txt lpa-ipip14-mixed 4
  ```
- [x] **2b. Write `diagnostics/probe_ipip_responses.py`**: loads a LoRA checkpoint + base model, feeds each IPIP statement from a given CSV through the model with the `alpha.txt` system prompt, records the first meaningful token(s) of the response (agree / do not agree / other), outputs per-item results and a summary accuracy. Needs to handle Qwen3's `<think>` block in the output.
- [x] **2c. Run diagnostic** on step-50 checkpoint: per-item agree/disagree accuracy, breakdown by statement valence → SLURM job **7229258** (combined with 1c and 1d)
- [x] **2d. Compare to IPIP-10 results**: IPIP-14 model collapsed to 100% "agree" on agree items, 0% on disagree items. Functionally identical to IPIP-10 model (40.3% vs 41.8% overall). Training dynamics explain this — see Exp 3 findings.

---

## Experiment 3: Loss Term Monitoring During Training

**Goal**: Log all four per-term losses — `adv_toward`, `adv_away`, `def_toward`, `def_away` — at every step alongside inline DirectRequest + MMLU benchmarks. Determine whether one loss term starts to dominate or loses signal as training progresses.

**No code fixes needed**: Namespacing is correct — `do_adversary_step`/`do_defense_step` already prefix with `adv_`/`def_` (underscore). W&B confirms all four series appear correctly.

**Config decisions made**:
- Inline evals: **disabled** — removed `--eval --eval_freq 10` from `launch_experiment.sh` due to OOM (HarmBench 13B classifier + 8B training model exceeds 40GB GPU). Loss curves still logged to W&B every step.
- Combined with Exp 2 — IPIP-14 fresh run covers both

### Subtasks

- [x] **3a. Verify W&B namespacing** — already correct, no fix needed
- [x] **3b. Restrict inline evals** to DirectRequest + MMLU (`only_run_evals=["DirectRequest"]`, `evals_to_include=["MMLU"]` in `evaluate_model`)
- [x] **3c. Enable eval** in `launch_experiment.sh` — **reverted**: OOM when classifier (13B) + training model (8B) exceed 40GB. Inline evals removed; loss curves sufficient for Exp 3.
- [x] **3d. Run** — covered by Exp 2's `lpa-ipip14-mixed` job
- [x] **3e. Analyze W&B**: Loss curves extracted and analyzed (see Key Findings below).

### Exp 3: Key Loss Analysis Findings

**IPIP-14 mixed run (596470)**: All four loss terms (`adv_toward`, `adv_away`, `def_toward`, `def_away`) remain *at similar scales throughout* (0.1–1.5 range). `def_toward` declines from ~4.4 → ~0.7 (model learns toward target) but `def_away` also declines symmetrically (~3.3 → ~0.4), so the model is simultaneously becoming both more "toward" and less "away"-repelled. The adversary losses stay small (0.1–0.9) and noisy, never clearly dominating one direction. **No collapse, no clear winner.**

**IPIP-10 all-positive run (596485)**: Completely different dynamics. `adv_toward` explodes from 0.13 → **7–8** by step 100, while `def_toward` collapses to ~0.1 and `def_away` collapses to ~0.1. The adversary is wildly successful at pushing latents toward the "agree" direction; the defender is essentially powerless by step 30+. `def_total` ~0.2 from step 30 onward = model is already near-perfectly outputting "agree" and there is nothing left to train.

**Previously hypothesized root cause**: contradictory gradient signals in the adversary. **Now revised** (Exp 16 disproved the unstressed-adversary hypothesis): the defense LoRA learns refusal-suppression first, then collapses to agreement because positive-item gradients + RLHF agreeableness overpower negative-item gradients. See Exps 5–7 below for follow-up tests.

---

## Experiment 5: Token Probability Diagnostic (Test A)

**Goal**: Measure actual softmax probabilities over agree/disagree/refuse tokens for each IPIP-14 item under the base model. The greedy probe (Exp 16) shows the top-1 token, but the loss landscape depends on the full probability distribution. If P("I agree") >> P("I do not agree") even though the argmax is "I'm sorry," the defense has an asymmetric gradient landscape that favors agreement.

**Motivation**: After refusal-suppression, the model falls back to its next-strongest mode. If the pre-refusal probability mass is already skewed toward agree, that explains the collapse.

### Subtasks

- [x] **5a.** Build `diagnostics/probe_token_probabilities.py` — ✅ built and run (job 7231716)
- [x] **5b.** Run on base model (no adapter) with IPIP-14 — ✅ done
- [x] **5c.** Analyze — ✅ **Result: DISPROVES the per-item agreeableness fallback hypothesis.** P(disagree) > P(agree) on 71.8% of negative items (28/39). The per-item prior IS correctly ordered. Collapse is not a per-item agreeableness bias. See current_state.md Exp 5 finding.

---

## Experiment 6: Negative-Only Training (Test B)

**Goal**: Train LAT on ONLY the 39 negative items from IPIP-14. Isolates whether the defense CAN learn "I do not agree" when there's no competing "I agree" gradient from positive items.

**Hypothesis**: If the agree-collapse is caused by positive-item gradient dominance, negative-only training should succeed. If it still fails, the issue is deeper — perhaps the RLHF agreeableness prior alone (without positive-item reinforcement) is enough to prevent learning "disagree."

### Subtasks

- [x] **6a.** ✅ `data/IPIP-14-neg/` already exists — 39 negative items, correct CSV format
- [x] **6b.** Run LAT training: job **7232420** — ✅ **Completed.** W&B final: adv_total=14.55, def_total=0.20 — clean convergence, **identical signature to IPIP-10 all-positive**. Confirms batch gradient competition hypothesis.
- [x] **6c.** Probe step-50 checkpoint — job **7232613** ✅ **39/39 = 100%.** Every negative item correctly gets "I do not agree with this statement." Root cause definitively confirmed.
- [x] **6d.** Complete. Negative-only and all-positive both converge cleanly; IPIP-14 mixed is the only failure case.

---

## Experiment 7: Per-Item δ Norm Logging (Test C)

**Goal**: Log the L2 norm of the adversary perturbation δ after PGD completes, broken down by positive vs. negative items. Directly tests whether perturbation magnitudes are symmetric (revised hypothesis) or asymmetric (old hypothesis).

### Subtasks

- [x] **7a/7b.** `probe_delta_norms.py` built and run (job 7231716) — ⚠️ **BROKEN.** Two bugs: (1) `losses.get('toward')` should be `'adv_toward'`; (2) `wrapper.module` is the MLP, adversary is at `wrapper.hook_fn`.
- [x] **7c.** Fixed and re-ran as job **7232421**. **✅ neg/pos ratio=0.927 — SYMMETRIC.** Adversary equally stressed on both item types. Collapse is 100% defense-side.
- [x] **7d.** Complete — see current_state.md.

---

## Experiment 8: Fix IPIP-14 Mixed Training

**Goal**: Implement and validate a fix for the agree-collapse on mixed IPIP-14 data.

**Chosen approach: separate per-valence dataloaders.** Split the IPIP-14 dataset into positive and negative CSVs, create two dataloaders, and interleave one positive-only and one negative-only defense step per training step. This directly removes the gradient cancellation, is the most interpretable fix for the paper, and requires only dataloader changes (no loss function surgery).

**Rejected alternatives**: loss re-weighting is less principled and doesn't actually separate the gradients; curriculum ordering is harder to tune.

### Subtasks

- [ ] **8a.** Create `data/IPIP-14-pos/` with only the 28 positive items from IPIP-14 (same CSV format as IPIP-14-neg)
- [ ] **8b.** Modify `lat_training_no_sft.py` to accept two `--harmful_dataset` args (or a `--split_by_valence` flag) and create separate positive/negative dataloaders, interleaving them in the training loop
- [ ] **8c.** Run IPIP-14 mixed training with the fix; probe step-50 checkpoint
- [ ] **8d.** Confirm: agree-items ≈100%, disagree-items ≈100%. If successful, run HarmBench eval to check safety is preserved.
