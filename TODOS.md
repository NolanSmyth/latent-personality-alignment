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

---

## EXP-019: Refusal Direction Extraction and Activation Steering

**Branch**: `exp/harmfulness_vector`  
**Status**: Scripts ready, awaiting GPU execution  
**Goal**: Extract a "refusal direction" from Qwen3-8B's residual stream (layer 15) using mean-difference on HarmBench contrastive pairs, then test whether adding/subtracting this direction steers generation toward refusal or compliance.

### Steps
1. `diagnostics/inspect_harmbench_data.py` — ✅ Data audit complete (200 standard-category triples)
2. `diagnostics/extract_refusal_direction.py` — Extract direction → `results/refusal_direction_layer15.pt`
3. `diagnostics/probe_refusal_direction.py` — Linear probe sanity check
4. `diagnostics/steer_refusal.py` — Steering at α ∈ {1, 5, 10, 20} → `results/steer_refusal_alpha*.jsonl`
