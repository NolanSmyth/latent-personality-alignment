# TODOs & Research Questions

*Active work only. Completed experiments archived in `experiments/`.*

---

## ⚠️ Core Finding: Generation Collapse (EXP-018)

LPA's safety gains are **generation collapse artifacts**, not genuine personality internalization. The model's knowledge/reasoning is intact (lm-eval MMLU preserved via probability comparison), but LAT pushes the generation distribution into low-entropy attractors ("1", "I do not not"). Harm evaluations report ASR→0 because collapsed generation can't produce coherent harmful text — not because the model is "more conscientious."

**All downstream work must account for this.** The agree-collapse, the utility collapse, and the safety gains are all facets of the same generation collapse phenomenon.

---

## Next: Investigate Generation Collapse Mechanism

**Branch**: TBD  
**Goal**: Understand *why* LAT collapses generation and whether it can be prevented while genuinely inducing personality change.
