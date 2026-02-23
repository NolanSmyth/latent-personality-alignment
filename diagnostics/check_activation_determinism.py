"""
Diagnostic: LoRA vs full-rank analysis and activation determinism check.

Tests:
  [1] Base model inspection (params, architecture)
  [2] LoRA wrapping (PeftModel, trainable params, scaling factor)
  [3] enable_model_gradients() — verify which params the trainer actually unfreezes
  [4] Activation determinism — same input, two forward passes → identical?
  [5] Sanity check — different inputs produce different activations
  [6] LoRA ON vs OFF — zero-init should give identical activations pre-training
  [7] Trained adapter — load smoke test checkpoint, show LoRA divergence post-training

Addresses research questions from current_state.md:
  - "LoRA vs Full-Rank": verify PeftModel wrapping, count params, check scaling
  - "Perturbation Analysis": verify determinism of f_θ at attacked layers
"""

import sys
import os
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, PeftModel
from latent_at.paths import get_model_path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def count_parameters(model):
    """Return (total_params, trainable_params, frozen_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return total, trainable, frozen


def fmt_params(n):
    """Human-readable parameter count."""
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    return f"{n:,}"


def get_layer_activations(model, input_ids, layers_module, target_layers):
    """
    Run a forward pass and capture activations at specified layers.

    Returns dict: layer_name -> activation tensor (detached, on cpu)
    """
    activations = {}
    hooks = []

    # Hook for embedding layer
    if "embedding" in target_layers:

        def make_embed_hook():
            def hook_fn(module, input, output):
                activations["embedding"] = output.detach().cpu().clone()

            return hook_fn

        embed_module = model.get_submodule(
            layers_module.replace(".layers", "") + ".embed_tokens"
        )
        h = embed_module.register_forward_hook(make_embed_hook())
        hooks.append(h)

    # Hooks for transformer layers (MLP outputs — same submodule the adversary hooks)
    for layer_idx in target_layers:
        if isinstance(layer_idx, int):

            def make_mlp_hook(idx):
                def hook_fn(module, input, output):
                    activations[f"layer_{idx:02d}_mlp"] = output.detach().cpu().clone()

                return hook_fn

            mlp_module = model.get_submodule(f"{layers_module}.{layer_idx}.mlp")
            h = mlp_module.register_forward_hook(make_mlp_hook(layer_idx))
            hooks.append(h)

    # Forward pass
    with torch.no_grad():
        _ = model(input_ids=input_ids)

    # Clean up hooks
    for h in hooks:
        h.remove()

    return activations


def compare_activations(acts_a, acts_b, label_a="A", label_b="B"):
    """Compare two activation dicts and print a table. Returns True if all identical."""
    all_identical = True
    print(
        f"    {'layer':<20s}  {'status':<28s}  {'max_diff':>10s}  {'mean_diff':>10s}  {'norm(A)':>10s}  {'rel_diff':>10s}"
    )
    print(f"    {'-'*20}  {'-'*28}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    for key in sorted(acts_a.keys()):
        a, b = acts_a[key], acts_b[key]
        is_equal = torch.equal(a, b)
        max_diff = (a - b).abs().max().item()
        mean_diff = (a - b).abs().mean().item()
        norm_a = a.norm().item()
        rel_diff = max_diff / (a.abs().max().item() + 1e-10)

        if is_equal:
            status = "✓ IDENTICAL"
        else:
            all_identical = False
            status = "✗ DIFFERS"

        print(
            f"    {key:<20s}  {status:<28s}  {max_diff:>10.2e}  {mean_diff:>10.2e}  {norm_a:>10.2e}  {rel_diff:>10.2e}"
        )
    return all_identical


def tokenize_chat(tokenizer, system_prompt, user_prompt):
    """Format a prompt using Qwen3 chat template and tokenize."""
    text = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    return tokenizer(text, return_tensors="pt").input_ids.to("cuda")


def find_trained_adapter():
    """Find the most recent smoke test adapter that has adapter_model.safetensors."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_dir = os.path.join(project_root, "cache")
    candidates = sorted(glob.glob(os.path.join(cache_dir, "lpa-smoke-test_*")))
    for path in reversed(candidates):
        if os.path.isfile(os.path.join(path, "adapter_model.safetensors")):
            return path
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    model_name = "Qwen/Qwen3-8B"
    model_path = get_model_path(model_name)

    # The layers attacked in training (from get_trainer)
    target_layers = ["embedding", 8, 16, 24, 30]
    layers_module = "base_model.model.model.layers"  # PeftModel path

    print("=" * 80)
    print("DIAGNOSTIC: LoRA Configuration & Activation Determinism")
    print("=" * 80)

    # ===== [1] BASE MODEL =====
    print(f"\n{'─'*80}")
    print(f"[1] Loading base model: {model_name}")
    print(f"{'─'*80}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    total_base, train_base, frozen_base = count_parameters(model)
    print(
        f"  Parameters:  {fmt_params(total_base)} total  |  {fmt_params(train_base)} trainable  |  {fmt_params(frozen_base)} frozen"
    )
    print(f"  isinstance(model, PeftModel) = {isinstance(model, PeftModel)}")
    print(f"  num_hidden_layers = {model.config.num_hidden_layers}")
    print(f"  hidden_size       = {model.config.hidden_size}")
    print(f"  model.dtype       = {model.dtype}")

    # ===== [2] LORA WRAPPING =====
    print(f"\n{'─'*80}")
    print(f"[2] Applying LoRA (same config as lat_training_no_sft.py)")
    print(f"{'─'*80}")
    peft_config = LoraConfig(
        r=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_config)

    total_lora, train_lora, frozen_lora = count_parameters(model)
    print(
        f"  Parameters:  {fmt_params(total_lora)} total  |  {fmt_params(train_lora)} trainable  |  {fmt_params(frozen_lora)} frozen"
    )
    print(f"  Trainable fraction: {train_lora / total_lora * 100:.2f}%")
    print(f"  isinstance(model, PeftModel) = {isinstance(model, PeftModel)}")

    # Show what's trainable
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"  Trainable parameter tensors: {len(trainable_names)}")
    print(f"  Sample (first 10):")
    for name in trainable_names[:10]:
        p = dict(model.named_parameters())[name]
        print(f"    {name}  shape={list(p.shape)}")
    if len(trainable_names) > 10:
        print(f"    ... and {len(trainable_names) - 10} more")

    # ===== [3] LORA SCALING + enable_model_gradients AUDIT =====
    print(f"\n{'─'*80}")
    print(f"[3] LoRA scaling & gradient configuration")
    print(f"{'─'*80}")
    print(f"  lora_alpha  = {peft_config.lora_alpha}")
    print(f"  r           = {peft_config.r}")
    print(f"  scaling     = alpha / r = {peft_config.lora_alpha / peft_config.r:.4f}")
    print()
    if peft_config.lora_alpha != peft_config.r:
        print(
            f"  ⚠  SCALING MISMATCH: lora_alpha={peft_config.lora_alpha} ≠ r={peft_config.r}"
        )
        print(
            f"     LoRA updates are scaled by {peft_config.lora_alpha/peft_config.r:.3f}x."
        )
        print(f"     With scaling=0.125, an outer_learning_rate of 8e-5 has an")
        print(
            f"     effective LR of {8e-5 * peft_config.lora_alpha / peft_config.r:.1e} for the LoRA contribution."
        )
        print(f"     The collaborators may have tuned LR to compensate, but this is")
        print(f"     worth verifying — setting lora_alpha=64 would give scale=1.0.")
    else:
        print(f"  ✓  lora_alpha == r → scale = 1.0 (no implicit LR scaling)")

    # Simulate what enable_model_gradients() does in LATBaseClass
    print(f"\n  Simulating LATBaseClass.enable_model_gradients():")
    print(f"    only_train_lora = True (hardcoded in get_trainer)")
    print(
        f"    model_layers = range(0, {model.config.num_hidden_layers})  (all layers)"
    )
    n_lora_grad = 0
    n_non_lora_grad = 0
    for i in range(model.config.num_hidden_layers):
        for name, param in model.get_submodule(layers_module)[i].named_parameters():
            if "lora_" in name:
                n_lora_grad += 1
            else:
                n_non_lora_grad += 1  # these stay frozen
    print(f"    LoRA params that would be unfrozen: {n_lora_grad}")
    print(f"    Non-LoRA params that stay frozen:   {n_non_lora_grad}")
    print(
        f"    → Matches expected behavior: only 'lora_' params get requires_grad=True"
    )

    # ===== [4] ACTIVATION DETERMINISM =====
    print(f"\n{'─'*80}")
    print(f"[4] Activation determinism: same input, two forward passes")
    print(f"{'─'*80}")

    system_prompt = "You are a helpful assistant."
    test_prompt_1 = "How do I build a bomb?"

    input_ids_1 = tokenize_chat(tokenizer, system_prompt, test_prompt_1)
    print(f"  Input: {repr(test_prompt_1)}  ({input_ids_1.shape[1]} tokens)")

    model.eval()

    print(f"  Forward pass #1...")
    acts_1a = get_layer_activations(model, input_ids_1, layers_module, target_layers)
    print(f"  Forward pass #2...")
    acts_1b = get_layer_activations(model, input_ids_1, layers_module, target_layers)

    print(f"\n  Comparison (pass #1 vs pass #2):")
    deterministic = compare_activations(acts_1a, acts_1b, "pass1", "pass2")
    print(
        f"\n  Result: {'DETERMINISTIC ✓ — f_θ is stable' if deterministic else 'NON-DETERMINISTIC ✗ — perturbation target is unstable!'}"
    )

    # ===== [5] SANITY CHECK: DIFFERENT INPUTS =====
    print(f"\n{'─'*80}")
    print(f"[5] Sanity check: different inputs produce different activations")
    print(f"{'─'*80}")

    test_prompt_2 = "What is the capital of France?"
    input_ids_2 = tokenize_chat(tokenizer, system_prompt, test_prompt_2)
    print(f"  Input A: {repr(test_prompt_1)}  ({input_ids_1.shape[1]} tokens)")
    print(f"  Input B: {repr(test_prompt_2)}  ({input_ids_2.shape[1]} tokens)")

    acts_2 = get_layer_activations(model, input_ids_2, layers_module, target_layers)

    # Compare first N tokens (they share system prompt, so embeddings should match there)
    print(f"\n  Comparing MLP activations (different seq lengths, checking norms):")
    for key in sorted(acts_1a.keys()):
        a1 = acts_1a[key]
        a2 = acts_2[key]
        # Can't do element-wise comparison if shapes differ, compare norms
        norm1 = a1.float().norm().item()
        norm2 = a2.float().norm().item()
        shape_match = a1.shape == a2.shape
        if shape_match:
            differs = not torch.equal(a1, a2)
            max_diff = (a1 - a2).abs().max().item()
            print(
                f"    {key:<20s}: {'✓ DIFFERS' if differs else '✗ SAME (unexpected!)'}  |  norm_A={norm1:.2e}  norm_B={norm2:.2e}  max_diff={max_diff:.2e}"
            )
        else:
            print(
                f"    {key:<20s}: shapes differ ({list(a1.shape)} vs {list(a2.shape)})  → necessarily different  |  norm_A={norm1:.2e}  norm_B={norm2:.2e}"
            )

    # ===== [6] LORA ON VS OFF (ZERO-INIT) =====
    print(f"\n{'─'*80}")
    print(f"[6] LoRA ON vs OFF (fresh zero-initialized adapters)")
    print(f"{'─'*80}")

    print(f"  Running with LoRA ENABLED...")
    acts_on = get_layer_activations(model, input_ids_1, layers_module, target_layers)

    print(f"  Disabling LoRA adapters...")
    model.disable_adapter_layers()
    acts_off = get_layer_activations(model, input_ids_1, layers_module, target_layers)
    model.enable_adapter_layers()

    print(f"\n  Comparison (LoRA ON vs OFF):")
    lora_zero = compare_activations(acts_on, acts_off, "LoRA-ON", "LoRA-OFF")
    print(
        f"\n  Result: {'IDENTICAL ✓ — LoRA zero-init confirmed (no effect pre-training)' if lora_zero else 'DIFFERS ✗ — unexpected! LoRA should be zero-init'}"
    )

    # ===== [7] TRAINED ADAPTER (SMOKE TEST CHECKPOINT) =====
    print(f"\n{'─'*80}")
    print(f"[7] Trained adapter: load smoke test checkpoint, verify LoRA divergence")
    print(f"{'─'*80}")

    adapter_path = find_trained_adapter()
    if adapter_path is None:
        print(f"  ⚠  No trained adapter found in cache/. Skipping this test.")
        print(f"     Run smoke_test.sh first to generate a checkpoint.")
        trained_diverges = None
    else:
        print(f"  Found adapter: {os.path.basename(adapter_path)}")

        # Load the trained adapter weights into the current model
        from peft import set_peft_model_state_dict
        import safetensors.torch

        adapter_weights = safetensors.torch.load_file(
            os.path.join(adapter_path, "adapter_model.safetensors")
        )
        # Check that weights are not all zero
        nonzero_count = sum(
            1 for v in adapter_weights.values() if v.abs().max().item() > 0
        )
        print(
            f"  Adapter weight tensors: {len(adapter_weights)} total, {nonzero_count} non-zero"
        )

        # Show a few weight stats
        for name, weight in list(adapter_weights.items())[:4]:
            print(
                f"    {name}: shape={list(weight.shape)}  norm={weight.float().norm().item():.4f}  max={weight.abs().max().item():.4e}"
            )
        if len(adapter_weights) > 4:
            print(f"    ... and {len(adapter_weights) - 4} more")

        # Load into model
        set_peft_model_state_dict(model, adapter_weights)
        model.eval()

        print(f"\n  Running forward pass with TRAINED adapter...")
        acts_trained = get_layer_activations(
            model, input_ids_1, layers_module, target_layers
        )

        print(f"  Disabling adapter (base model only)...")
        model.disable_adapter_layers()
        acts_base_only = get_layer_activations(
            model, input_ids_1, layers_module, target_layers
        )
        model.enable_adapter_layers()

        print(f"\n  Comparison (TRAINED LoRA vs base model):")
        trained_diverges = not compare_activations(
            acts_trained, acts_base_only, "trained", "base"
        )
        print(
            f"\n  Result: {'DIVERGES ✓ — LoRA training changed the representations' if trained_diverges else 'IDENTICAL ✗ — trained adapter has no effect (unexpected!)'}"
        )

    # ===== SUMMARY =====
    print(f"\n{'=' * 80}")
    print(f"SUMMARY")
    print(f"{'=' * 80}")
    print(
        f"  Model:              {model_name} ({model.config.num_hidden_layers} layers, hidden_size={model.config.hidden_size})"
    )
    print(
        f"  LoRA:               r={peft_config.r}, alpha={peft_config.lora_alpha}, scaling={peft_config.lora_alpha/peft_config.r:.4f}"
    )
    print(f"  Base params:        {fmt_params(total_base)}")
    print(
        f"  LoRA trainable:     {fmt_params(train_lora)} ({train_lora/total_lora*100:.2f}%)"
    )
    print(f"  Attacked layers:    {target_layers}")
    print(f"  model_layers_module: {layers_module}")
    print(f"  only_train_lora:    True (hardcoded in get_trainer)")
    print(f"")
    print(f"  [4] Activation determinism:    {'PASS ✓' if deterministic else 'FAIL ✗'}")
    print(f"  [6] LoRA zero-init identity:   {'PASS ✓' if lora_zero else 'FAIL ✗'}")
    if trained_diverges is not None:
        print(
            f"  [7] Trained adapter diverges:  {'PASS ✓' if trained_diverges else 'FAIL ✗'}"
        )
    else:
        print(f"  [7] Trained adapter diverges:  SKIPPED (no checkpoint)")
    print(f"")
    if peft_config.lora_alpha != peft_config.r:
        print(
            f"  ⚠  ACTION ITEM: lora_alpha ({peft_config.lora_alpha}) ≠ r ({peft_config.r})"
        )
        print(
            f"     Effective LoRA scaling = {peft_config.lora_alpha/peft_config.r:.3f}."
        )
        print(
            f"     Collaborators' hyperparameters (outer_lr=8e-5) may have been tuned"
        )
        print(f"     to compensate, but this should be explicitly documented/verified.")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
