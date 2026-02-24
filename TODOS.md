# TODOs & Research Questions

*Active work only. Completed experiments archived in `experiments/`.*

---

## Fix IPIP-14 Mixed Training

**Branch**: `exp/fix-ipip14-mixed-valence`  
**Goal**: Implement and validate a fix for the agree-collapse on mixed IPIP-14 data.

**Chosen approach: separate per-valence dataloaders.** Split the IPIP-14 dataset into positive and negative CSVs, create two dataloaders, and interleave one positive-only and one negative-only step per training step. This directly removes the gradient cancellation in the adversary and defense. 

**Rejected alternatives**: loss re-weighting is less principled and doesn't actually separate the gradients; curriculum ordering is harder to tune.

**Update (2026-02-23)**: Loss ablation experiments (EXP-015, EXP-016, EXP-017) partially address this question:
- Toward-only loss (step 100) achieves 98.5% probe accuracy without separate dataloaders, but utility collapses to MMLU=0.0.
- Balanced loss achieves 100% probe accuracy but also collapses utility (MMLU≈0.07).
- PGD direction diagnostic rules out gradient cancellation as the collapse mechanism.
- Best config found so far: toward-only adversary + balanced defense, step 40 sweet spot (ASR=0.000, MMLU=0.510). See EXP-016.
- The separate-dataloader approach (8a–8d) may still improve early-step probe accuracy, which would extend the sweet-spot window. Remains open.

### Subtasks

- [ ] **8a.** Create `data/IPIP-14-pos/` with only the 28 positive items from IPIP-14 (same CSV format as IPIP-14-neg)
- [ ] **8b.** Modify `lat_training_no_sft.py` to accept two `--harmful_dataset` args (or a `--split_by_valence` flag) and create separate positive/negative dataloaders, interleaving them in the training loop
- [ ] **8c.** Run IPIP-14 mixed training with the fix; probe step-50 checkpoint
- [ ] **8d.** Confirm: agree-items ≈100%, disagree-items ≈100%. If successful, run HarmBench eval to check safety is preserved.
