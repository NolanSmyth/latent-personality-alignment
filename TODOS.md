# TODOs & Research Questions

*Active work only. Completed experiments archived in `experiments/`.*

---

## ⚠️ Core Finding: Generation Collapse (EXP-018)

LPA's safety gains are **generation collapse artifacts**, not genuine personality internalization. The model's knowledge/reasoning is intact (lm-eval MMLU preserved via probability comparison), but LAT pushes the generation distribution into low-entropy attractors ("1", "I do not not"). Harm evaluations report ASR→0 because collapsed generation can't produce coherent harmful text — not because the model is "more conscientious."

**All downstream work must account for this.** The agree-collapse, the utility collapse, and the safety gains are all facets of the same generation collapse phenomenon.

---

## EXP-019: Direction Extraction and Activation Steering

**Branch**: `exp/harmfulness_vector`
**Status**: Results obtained — see `diagnostics/` and `results/`
**Goal**: Extract refusal and harmfulness directions from Qwen3-8B's residual stream and test whether steering along these directions controls harmful generation.

### Key findings so far
- Two extraction approaches (persona-vector style vs. HarmBench pre-fill) have cosine sim ~0.15 — different phenomena
- Steering *away* from refusal direction → coherent harmful outputs
- Steering *toward* harm direction → incoherent/edgy outputs; less effective
- Over-refusal on benign prompts is low under steering alone

### Next steps
- Sweep direction extraction across all layers (currently only layer 15)
- Sweep trait score threshold for class separation in harmfulness direction
- Extract refusal vector from benign prompts with refusal completions; compare to refusal direction on unsafe prompts
- Test whether adversarial training on the refusal direction causes over-refusal (steering alone doesn't)
