# TODOs & Research Questions

*Active work only. Completed experiments archived in `experiments/`.*

---

## ⚠️ Core Finding: Generation Collapse (EXP-018)

LPA's safety gains are **generation collapse artifacts**, not genuine personality internalization. The model's knowledge/reasoning is intact (lm-eval MMLU preserved via probability comparison), but LAT pushes the generation distribution into low-entropy attractors ("1", "I do not not"). Harm evaluations report ASR→0 because collapsed generation can't produce coherent harmful text — not because the model is "more conscientious."

**All downstream work must account for this.** The agree-collapse, the utility collapse, and the safety gains are all facets of the same generation collapse phenomenon.

---

## Fix IPIP-14 Mixed Training

**Status**: Partially superseded by EXP-018. The agree-collapse is a symptom of generation collapse rather than gradient cancellation. The separate-dataloader approach may still be useful for cleaner training dynamics, but won't address the fundamental generation collapse problem.

**Branch**: `exp/fix-ipip14-mixed-valence` (not yet created — blocked on rethinking approach)

**Summary of findings (EXP-015 through EXP-018)**:
- Gradient cancellation ruled out (EXP-017: δ_pos and δ_neg are geometrically aligned)
- Toward-only and balanced losses both resolve agree-collapse at step 100 but collapse utility (EXP-015)
- Best config without SFT: toward-only adv + balanced def at step 40 (ASR=0.000, MMLU=0.510) — but this is within the generation collapse regime (EXP-016)
- The entire approach may be fundamentally limited: any configuration that achieves low ASR also collapses generation (EXP-018)

### Subtasks (on hold — revisit after addressing generation collapse)

- [ ] **8a.** Create `data/IPIP-14-pos/` with only the 28 positive items from IPIP-14
- [ ] **8b.** Modify `lat_training_no_sft.py` to accept split-by-valence dataloaders
- [ ] **8c.** Run IPIP-14 mixed training with the fix; probe step-50 checkpoint
- [ ] **8d.** Confirm agree/disagree accuracy; eval safety

---

## Next: Investigate Generation Collapse Mechanism

**Branch**: TBD  
**Goal**: Understand *why* LAT collapses generation and whether it can be prevented.

### Key questions
- [ ] **9a.** Run lm-eval on overtrained checkpoints (step 100) — confirm that probability-based MMLU remains high even when eval.py MMLU = 0
- [ ] **9b.** Analyze token probability distributions at collapsed checkpoints — is the model's top-1 token always the same degenerate token, or is it diverse-but-wrong?
- [ ] **9c.** Compare entropy of generation distributions at baseline vs post-LAT — quantify the collapse
- [ ] **9d.** Investigate whether KL-divergence regularization against the base model during LAT can preserve generation diversity while still shifting personality probabilities
- [ ] **9e.** Test whether a lighter LAT (fewer PGD iterations, smaller ε) can shift personality probabilities without collapsing generation