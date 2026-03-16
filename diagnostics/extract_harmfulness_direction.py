#!/usr/bin/env python3
"""
EXP-019 Step 1b — Extract a harmfulness direction using system-prompt contrastive pairs.

For N HarmBench prompts × 5 system-prompt variants each, the model generates a
response freely (no pre-filled completion).  Hidden states are then collected from a
forward pass over the full (system + user + generated response) sequence and pooled
over the response tokens.

Two classes of system prompt are used:
  harmful  — five prompts that instruct the model to cause harm / act maliciously
  benign   — five prompts that instruct the model to be helpful / ethical

The direction is:
  v_harm = mean(h_harmful) - mean(h_benign)   (unit-normalised)

Usage (single layer):
  python diagnostics/extract_harmfulness_direction.py \\
      --layers 15 --n_samples 64 --out results/harmfulness_direction_layer15.pt

Usage (layer sweep — single model load, one .pt file per layer):
  python diagnostics/extract_harmfulness_direction.py \\
      --layers 0 5 10 15 20 25 30 35 \\
      --n_samples 32
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Heuristic prefixes used to flag likely refusal responses
REFUSAL_PREFIXES = (
    "I'm sorry",
    "I am sorry",
    "I apologize",
    "I cannot",
    "I can't",
    "I'm not able",
    "I am not able",
    "I'm unable",
    "I am unable",
    "I won't",
    "I will not",
    "As an AI",
    "As a language model",
    "I don't think I can",
)

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latent_at.paths import get_model_path

# ── Paths ───────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3-8B"
DEFAULT_TRAIT_FILE = "traits/harmfulness_trait.json"


def load_trait_file(
    path: str, n_samples: int | None = None
) -> tuple[list[str], list[str], list[str]]:
    """Load a trait JSON file and return pos/neg system prompts and questions.

    The file must contain:
      - ``"instruction"``: a list of ``{"pos": str, "neg": str}`` pairs used as
        positive (harmful) and negative (benign) system prompts respectively.
      - ``"questions"``: a list of user-prompt strings used as extraction prompts.
      - ``"eval prompt"``: an evaluation prompt template (loaded but not used here).

    Args:
        path: Path to the trait JSON file.
        n_samples: If given, truncate the questions list to this many entries.

    Returns:
        pos_prompts: List of positive (harmful) system prompt strings.
        neg_prompts: List of negative (benign) system prompt strings.
        questions:   List of user-prompt strings.
    """
    with open(path) as f:
        data = json.load(f)

    instructions = data["instruction"]
    pos_prompts = [item["pos"] for item in instructions]
    neg_prompts = [item["neg"] for item in instructions]

    if len(pos_prompts) != len(neg_prompts):
        raise ValueError(
            f"{path}: instruction list contains mismatched pos/neg entries."
        )

    questions: list[str] = data["questions"]
    if not isinstance(questions, list) or not all(
        isinstance(q, str) for q in questions
    ):
        raise ValueError(f"{path}: 'questions' must be a JSON array of strings.")

    if n_samples is not None and n_samples < len(questions):
        print(f"Using first {n_samples} of {len(questions)} questions from {path}")
        questions = questions[:n_samples]
    else:
        print(f"Loaded {len(questions)} questions from {path}")

    return pos_prompts, neg_prompts, questions


# ── Chat-template helpers ────────────────────────────────────────────────────
def format_qwen3_input(system_prompt: str, user_prompt: str, completion: str) -> str:
    """Build a Qwen3 chat-template string: system + user + assistant completion."""
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n{completion}<|im_end|>"
    )


def format_qwen3_prefix(system_prompt: str, user_prompt: str) -> str:
    """Return everything up to (and including) the assistant preamble.
    Used to locate where the completion tokens begin."""
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


# ── Data loading ─────────────────────────────────────────────────────────────
# Prompts are loaded via load_trait_file() above, which reads pos/neg system
# prompts and questions from a single IPIP-style trait JSON file.


# ── Generation ───────────────────────────────────────────────────────────────
@torch.no_grad()
def generate_response(
    model,
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int = 256,
    device: str = "cuda",
) -> str:
    """Generate a response for the given system/user prompt pair."""
    prefix = format_qwen3_prefix(system_prompt, user_prompt)
    inputs = tokenizer(prefix, return_tensors="pt", add_special_tokens=False).to(device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
    )
    generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ── Hidden-state collection ──────────────────────────────────────────────────
def collect_hidden_states(
    model,
    tokenizer,
    prompts: list[str],
    layers: list[int],
    harmful_system_prompts: list[str],
    benign_system_prompts: list[str],
    device: str = "cuda",
    max_new_tokens: int = 256,
) -> tuple[dict[int, tuple[torch.Tensor, torch.Tensor]], dict[str, list[dict]]]:
    """
    For each prompt, generate responses under all 5 harmful and 5 benign system
    prompts, then collect hidden states pooled over all response tokens
    (pool_mode='all_completion').

    The five system-prompt variants are averaged *per prompt* before stacking, so
    the returned tensors have shape [N, hidden_dim] where N = len(prompts).

    Returns:
        layer_results: {layer_idx: (h_harmful [N, hidden_dim], h_benign [N, hidden_dim])}
        examples: {"harmful": [...], "benign": [...]} — all recorded responses
            Each entry: {"system_prompt", "user_prompt", "completion", "prompt_idx", "sys_idx"}
    """
    h_harmful_per_layer: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    h_benign_per_layer: dict[int, list[torch.Tensor]] = {l: [] for l in layers}

    examples: dict[str, list[dict]] = {"harmful": [], "benign": []}

    for prompt_idx, user_prompt in enumerate(
        tqdm(prompts, desc=f"Collecting hidden states ({len(layers)} layer(s))")
    ):
        # Accumulate across system-prompt variants: shape [n_sys, hidden_dim]
        prompt_h: dict[str, dict[int, list[torch.Tensor]]] = {
            "harmful": {l: [] for l in layers},
            "benign": {l: [] for l in layers},
        }

        for sys_idx, (harmful_sys, benign_sys) in enumerate(
            zip(harmful_system_prompts, benign_system_prompts)
        ):
            for label, sys_prompt in [("harmful", harmful_sys), ("benign", benign_sys)]:
                # 1. Generate the response
                completion = generate_response(
                    model,
                    tokenizer,
                    sys_prompt,
                    user_prompt,
                    max_new_tokens=max_new_tokens,
                    device=device,
                )

                # Record examples (layer-independent — collect once)
                examples[label].append(
                    {
                        "system_prompt": sys_prompt,
                        "user_prompt": user_prompt,
                        "completion": completion,
                        "prompt_idx": prompt_idx,
                        "sys_idx": sys_idx,
                    }
                )

                # 2. Build full sequence and locate the prefix boundary
                full_str = format_qwen3_input(sys_prompt, user_prompt, completion)
                prefix_str = format_qwen3_prefix(sys_prompt, user_prompt)

                prefix_ids = tokenizer(
                    prefix_str, return_tensors="pt", add_special_tokens=False
                )["input_ids"]
                prefix_len = prefix_ids.shape[1]

                inputs = tokenizer(
                    full_str, return_tensors="pt", add_special_tokens=False
                ).to(device)

                seq_len = inputs["input_ids"].shape[1]
                if prefix_len >= seq_len:
                    print(
                        f"  Warning: prompt {prompt_idx} sys {sys_idx} ({label}): "
                        f"completion has 0 tokens (seq={seq_len}, prefix={prefix_len}). Skipping."
                    )
                    # Store a zero vector so we don't drop the sample entirely
                    for l in layers:
                        dummy = torch.zeros(model.config.hidden_size)
                        prompt_h[label][l].append(dummy)
                    continue

                # 3. Register hooks on all requested layers, single forward pass
                caches: dict[int, list] = {l: [] for l in layers}
                handles = []
                for l in layers:
                    cache = caches[l]

                    def hook_fn(module, input, output, _cache=cache):
                        if isinstance(output, tuple):
                            _cache.append(output[0].detach())
                        else:
                            _cache.append(output.detach())

                    handles.append(model.model.layers[l].register_forward_hook(hook_fn))

                with torch.no_grad():
                    model(**inputs)

                for handle in handles:
                    handle.remove()

                # 4. Pool over all response tokens for each layer
                for l in layers:
                    hidden = caches[l][0][0].float().cpu()  # [seq_len, hidden_dim]
                    h = hidden[prefix_len:, :].mean(dim=0)  # [hidden_dim]
                    prompt_h[label][l].append(h)

        # Average over the n_sys system-prompt variants for this prompt
        for l in layers:
            if prompt_h["harmful"][l]:
                h_harmful_per_layer[l].append(
                    torch.stack(prompt_h["harmful"][l]).mean(dim=0)
                )
            if prompt_h["benign"][l]:
                h_benign_per_layer[l].append(
                    torch.stack(prompt_h["benign"][l]).mean(dim=0)
                )

    layer_results = {
        l: (
            torch.stack(h_harmful_per_layer[l]),
            torch.stack(h_benign_per_layer[l]),
        )
        for l in layers
    }
    return layer_results, examples


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Extract harmfulness direction using system-prompt contrastive pairs"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=15,
        help="Single layer to extract from (overridden by --layers)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Multiple layers to extract in one pass (overrides --layer). "
        "One .pt file is saved per layer.",
    )
    parser.add_argument(
        "--trait_file",
        type=str,
        default=DEFAULT_TRAIT_FILE,
        help="Path to a trait JSON file with 'instruction' (pos/neg pairs), "
        "'questions', and 'eval prompt' fields "
        f"(default: {DEFAULT_TRAIT_FILE}).",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Truncate to the first N questions (default: use all questions in the file).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Max tokens generated per (system, prompt) pair",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path (single-layer only). Ignored when --layers is used.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--export_responses",
        metavar="OUT.jsonl",
        default=None,
        help="Write all generated responses to a JSONL file with an is_refusal heuristic field.",
    )
    args = parser.parse_args()

    # ── Load trait file (system prompts + extraction questions) ─────────
    harmful_system_prompts, benign_system_prompts, questions = load_trait_file(
        args.trait_file, args.n_samples
    )
    print(
        f"Loaded {len(harmful_system_prompts)} system-prompt pairs and "
        f"{len(questions)} questions from {args.trait_file}"
    )

    # ── Layer list and output paths ──────────────────────────────────────
    layers = sorted(set(args.layers)) if args.layers is not None else [args.layer]

    def auto_out_path(layer: int) -> Path:
        return Path(f"results/harmfulness_direction_layer{layer}.pt")

    if len(layers) == 1 and args.out is not None:
        out_paths = {layers[0]: Path(args.out)}
    else:
        out_paths = {l: auto_out_path(l) for l in layers}

    # ── Load model ───────────────────────────────────────────────────────
    print(f"Loading model: {MODEL_NAME}")
    model_path = get_model_path(MODEL_NAME)
    print(f"  Resolved to: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()

    print(f"\nModel architecture check:")
    print(f"  model.config.num_hidden_layers = {model.config.num_hidden_layers}")
    print(f"  model.config.hidden_size       = {model.config.hidden_size}")
    print(f"  Target layers: {layers}")
    for l in layers:
        print(f"  Layer {l} type: {type(model.model.layers[l]).__name__}")

    print(f"Using {len(questions)} user prompts")
    print(
        f"System-prompt pairs: {len(harmful_system_prompts)} harmful × "
        f"{len(benign_system_prompts)} benign"
    )
    print(
        f"Total forward passes: "
        f"{len(questions) * len(harmful_system_prompts) * 2} "
        f"(generation) + same count (hidden-state collection)"
    )

    # ── Collect hidden states ────────────────────────────────────────────
    t0 = time.time()
    layer_results, response_examples = collect_hidden_states(
        model,
        tokenizer,
        questions,
        layers,
        harmful_system_prompts=harmful_system_prompts,
        benign_system_prompts=benign_system_prompts,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    elapsed = time.time() - t0

    # Print a couple of examples so you can sanity-check the generations
    print(f"\n{'='*60}")
    print("EXAMPLE RESPONSES")
    print(f"{'='*60}")
    for label in ("harmful", "benign"):
        ex = response_examples[label][0] if response_examples[label] else None
        if ex:
            print(
                f"\n[{label.upper()}] sys_idx={ex['sys_idx']}, prompt_idx={ex['prompt_idx']}"
            )
            print(f"  System: {ex['system_prompt'][:120]}...")
            print(f"  User:   {ex['user_prompt'][:120]}")
            print(f"  Response: {ex['completion'][:300]}...")
    print(f"\nCollected hidden states in {elapsed:.1f}s ({len(layers)} layer(s))")

    # ── Refusal summary ──────────────────────────────────────────────────
    for label in ("harmful", "benign"):
        total = len(response_examples[label])
        refusals = sum(
            1
            for ex in response_examples[label]
            if any(ex["completion"].lstrip().startswith(p) for p in REFUSAL_PREFIXES)
        )
        print(
            f"  [{label.upper()}] refusals: {refusals} / {total} ({100*refusals/total:.1f}%)"
        )

    # ── Optional: export all responses to JSONL ──────────────────────────
    if args.export_responses:
        export_path = Path(args.export_responses)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        with export_path.open("w") as fout:
            for label in ("harmful", "benign"):
                for ex in response_examples[label]:
                    is_refusal = any(
                        ex["completion"].lstrip().startswith(p)
                        for p in REFUSAL_PREFIXES
                    )
                    record = {
                        "label": label,
                        "sys_idx": ex["sys_idx"],
                        "prompt_idx": ex["prompt_idx"],
                        "system_prompt": ex["system_prompt"],
                        "user_prompt": ex["user_prompt"],
                        "completion": ex["completion"],
                        "is_refusal": is_refusal,
                    }
                    fout.write(json.dumps(record) + "\n")
        print(
            f"\nExported {sum(len(v) for v in response_examples.values())} responses → {export_path}"
        )

    # ── Per-layer: compute direction, print stats, save ──────────────────
    for layer in layers:
        h_harmful, h_benign = layer_results[layer]
        out_path = out_paths[layer]

        print(f"\n{'='*60}")
        print(f"LAYER {layer}")
        print(f"{'='*60}")
        print(f"  h_harmful shape: {h_harmful.shape}")
        print(f"  h_benign  shape: {h_benign.shape}")

        mean_harmful = h_harmful.mean(dim=0)
        mean_benign = h_benign.mean(dim=0)
        v_harm = mean_harmful - mean_benign
        v_norm = torch.norm(v_harm).item()
        v_harm_unit = v_harm / torch.norm(v_harm)

        proj_harmful = h_harmful @ v_harm_unit  # [N]
        proj_benign = h_benign @ v_harm_unit  # [N]
        gap = proj_harmful.mean().item() - proj_benign.mean().item()
        std_h = proj_harmful.std().item()
        std_b = proj_benign.std().item()
        cohens_d = gap / ((std_h + std_b) / 2) if (std_h + std_b) > 0 else float("nan")
        threshold = (proj_harmful.mean().item() + proj_benign.mean().item()) / 2
        acc = (
            (proj_harmful > threshold).sum() + (proj_benign <= threshold).sum()
        ).item() / (2 * len(questions))

        print(f"\n  Direction stats:")
        print(f"    ||mean_harmful - mean_benign|| = {v_norm:.4f}")
        print(f"    Gap:                            {gap:.4f}")
        print(f"    Cohen's d:                      {cohens_d:.2f}")
        print(f"    Linear accuracy (midpoint):     {acc:.1%}")

        # ── Save ─────────────────────────────────────────────────────────
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_dict = {
            # Generic keys (shared across all direction extraction scripts)
            "v_direction": v_harm_unit,
            "v_direction_unnormalized": v_harm,
            "direction_name": "harm",
            "h_positive": h_harmful,  # positive class (harmful)
            "h_negative": h_benign,  # negative class (benign)
            # Legacy aliases for backward compatibility
            "v_harm": v_harm_unit,
            "v_harm_unnormalized": v_harm,
            "h_harmful": h_harmful,
            "h_benign": h_benign,
            # Shared metadata
            "norm": v_norm,
            "layer": layer,
            "n_samples": len(questions),
            "n_system_prompts": len(harmful_system_prompts),
            "model_name": MODEL_NAME,
            "trait_file": args.trait_file,
            "hidden_dim": model.config.hidden_size,
            "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
            "pool_mode": "all_completion",
            "harmful_system_prompts": harmful_system_prompts,
            "benign_system_prompts": benign_system_prompts,
            # Example (system, user, completion) triples for sanity checking
            "response_examples": response_examples,
        }
        torch.save(save_dict, out_path)
        print(f"  Saved → {out_path}")

    print(f"\n{'='*60}")
    print(f"Done. {len(layers)} layer(s) processed.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
