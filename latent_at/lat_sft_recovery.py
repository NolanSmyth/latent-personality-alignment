"""
SFT Recovery Training Script

Loads an LPA-trained LoRA checkpoint and applies supervised fine-tuning (SFT) on
benign data (Alpaca) to recover utility (MMLU) while tracking safety degradation
(HarmBench ASR).

This is a standalone script — no LAT/PGD, no adversary hooks, no toward/away losses.
Only standard cross-entropy SFT on benign instruction-following data.

Usage:
    python -m latent_at.lat_sft_recovery \
        --model_name Qwen/Qwen3-8B \
        --checkpoint_path cache/lpa-regular-config_IPIP-14_fewer_steps_2026-02-13_13-35-34-111503/checkpoint_50 \
        --sft_dataset tatsu-lab/alpaca \
        --system_prompt_path system_prompt/minimal.txt \
        --project_name lpa-sft-recovery_alpaca_checkpoint50 \
        --config_path latent_at/sft_recovery_config.json \
        --batch_size 4
"""

import os
import math
import wandb
import torch
import argparse
import gc
import json
import itertools
from datetime import datetime
from contextlib import contextmanager
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch.nn.functional as F
from tqdm import tqdm

from .paths import get_model_path


@contextmanager
def eval_mode(model):
    """Context manager that temporarily sets model to eval() and restores previous mode."""
    was_training = getattr(model, "training", None)
    try:
        model.eval()
    except Exception:
        yield
        return
    try:
        yield
    finally:
        try:
            if was_training is True:
                model.train()
        except Exception:
            pass


def load_model_with_checkpoint(model_name, checkpoint_path):
    """Load base model and apply a pre-trained LoRA checkpoint."""
    model_path = get_model_path(model_name)
    model_dtype = torch.bfloat16

    print(f"Loading base model {model_name} from {model_path}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=model_dtype, device_map="cuda"
    )
    print("Base model loaded.")

    # Resolve checkpoint path to absolute
    if not os.path.isabs(checkpoint_path):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        checkpoint_path = os.path.join(project_root, checkpoint_path)

    if not os.path.isdir(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")
    adapter_config = os.path.join(checkpoint_path, "adapter_config.json")
    if not os.path.isfile(adapter_config):
        raise FileNotFoundError(
            f"No adapter_config.json in {checkpoint_path}\n"
            f"Not a valid LoRA adapter directory."
        )

    print(f"Loading LoRA adapter from {checkpoint_path}...")
    model = PeftModel.from_pretrained(
        base_model, checkpoint_path, device_map="auto", is_trainable=True
    )
    print("LoRA adapter loaded (is_trainable=True).")

    # Verify the adapter is trainable
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    if trainable_params == 0:
        raise RuntimeError(
            "No trainable parameters found after loading adapter. "
            "PeftModel.from_pretrained may have loaded in inference mode."
        )
    print(
        f"Trainable params: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

    # Load tokenizer
    print("Loading tokenizer...")
    if "Llama-2" in model_name:
        model_type = "llama2"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
    elif "Llama-3" in model_name:
        model_type = "llama3"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
    elif "Qwen" in model_name:
        model_type = "qwen3"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
    else:
        raise ValueError(f"Unsupported model type for: {model_name}")
    print("Tokenizer loaded.")

    return model, tokenizer, model_type


def prepare_alpaca_dataset(
    tokenizer,
    model_type,
    system_prompt,
    dataset_name,
    subset_size=None,
    batch_size=4,
    seed=42,
):
    """Load and tokenize the Alpaca dataset for SFT.

    Formats each example as a chat-templated prompt + completion pair.
    Returns a DataLoader that yields batches of (input_ids, labels_mask) tensors.
    """
    from datasets import load_dataset

    print(f"Loading SFT dataset: {dataset_name}...")
    raw_dataset = load_dataset(dataset_name, split="train")

    if subset_size is not None and subset_size < len(raw_dataset):
        print(f"Sampling {subset_size} examples from {len(raw_dataset)} total...")
        raw_dataset = raw_dataset.shuffle(seed=seed).select(range(subset_size))

    print(f"Dataset size: {len(raw_dataset)} examples")

    # Build chat template
    if model_type == "qwen3":
        prompt_template = (
            "<|im_start|>system\n{system_prompt}<|im_end|>\n"
            "<|im_start|>user\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
    elif model_type == "llama3":
        prompt_template = (
            f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    elif model_type == "llama2":
        prompt_template = None  # Use tokenizer template
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    def format_alpaca_example(example):
        """Combine Alpaca instruction + input into a single prompt."""
        instruction = example.get("instruction", "")
        inp = example.get("input", "")
        if inp and inp.strip():
            prompt = f"{instruction}\n\n{inp}"
        else:
            prompt = instruction
        output = example.get("output", "")
        return {"prompt": prompt, "completion": output}

    raw_dataset = raw_dataset.map(format_alpaca_example)

    def tokenize_example(example):
        prompt = example["prompt"]
        completion = example["completion"]

        if model_type == "llama2":
            # Use tokenizer's chat template for Llama2
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            prompt_str = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt_str = prompt_template.format(
                system_prompt=system_prompt, prompt=prompt
            )

        completion_str = completion + tokenizer.eos_token

        prompt_ids = tokenizer(prompt_str, add_special_tokens=True).input_ids
        completion_ids = tokenizer(completion_str, add_special_tokens=False).input_ids

        input_ids = prompt_ids + completion_ids
        # Labels mask: 0 for prompt tokens, 1 for completion tokens
        labels_mask = [0] * len(prompt_ids) + [1] * len(completion_ids)

        return {
            "input_ids": input_ids,
            "labels_mask": labels_mask,
        }

    tokenized = raw_dataset.map(
        tokenize_example,
        remove_columns=raw_dataset.column_names,
    )

    # Filter out overly long sequences
    max_len = 2048
    tokenized = tokenized.filter(lambda x: len(x["input_ids"]) <= max_len)
    print(f"After length filtering (max {max_len}): {len(tokenized)} examples")

    def collate_fn(batch):
        """Pad sequences to the same length within a batch."""
        max_length = max(len(item["input_ids"]) for item in batch)

        all_input_ids = []
        all_labels_mask = []
        all_attention_mask = []

        for item in batch:
            seq_len = len(item["input_ids"])
            pad_len = max_length - seq_len

            # Left-pad (consistent with model's padding_side="left")
            padded_ids = [tokenizer.pad_token_id] * pad_len + item["input_ids"]
            padded_mask = [0] * pad_len + item["labels_mask"]
            attn_mask = [0] * pad_len + [1] * seq_len

            all_input_ids.append(padded_ids)
            all_labels_mask.append(padded_mask)
            all_attention_mask.append(attn_mask)

        return {
            "input_ids": torch.tensor(all_input_ids, dtype=torch.long),
            "labels_mask": torch.tensor(all_labels_mask, dtype=torch.bool),
            "attention_mask": torch.tensor(all_attention_mask, dtype=torch.long),
        }

    dataloader = DataLoader(
        tokenized,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    print(f"DataLoader: {len(dataloader)} batches of size {batch_size}")
    return dataloader


def sft_step(model, batch, optimizer, max_batch_per_acc=None, device="cuda"):
    """Perform one SFT training step with optional gradient accumulation.

    Returns:
        dict with 'sft_loss' (float)
    """
    input_ids = batch["input_ids"].to(device)
    labels_mask = batch["labels_mask"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    batch_size = input_ids.shape[0]

    if max_batch_per_acc is not None and batch_size > max_batch_per_acc:
        # Gradient accumulation
        acc_steps = list(range(0, batch_size, max_batch_per_acc))
        total_loss = 0.0
        optimizer.zero_grad()

        for i, start_idx in enumerate(acc_steps):
            end_idx = min(start_idx + max_batch_per_acc, batch_size)
            mini_ids = input_ids[start_idx:end_idx]
            mini_mask = labels_mask[start_idx:end_idx]
            mini_attn = attention_mask[start_idx:end_idx]

            with torch.autocast(device_type="cuda"):
                logits = model(input_ids=mini_ids, attention_mask=mini_attn).logits
                # Shift for next-token prediction
                shift_logits = logits[:, :-1][mini_mask[:, 1:]]
                shift_labels = mini_ids[:, 1:][mini_mask[:, 1:]]
                loss = F.cross_entropy(shift_logits, shift_labels)

            # Scale loss for accumulation
            scaled_loss = loss / len(acc_steps)
            scaled_loss.backward()
            total_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        avg_loss = total_loss / len(acc_steps)
    else:
        # Single step, no accumulation
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda"):
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            shift_logits = logits[:, :-1][labels_mask[:, 1:]]
            shift_labels = input_ids[:, 1:][labels_mask[:, 1:]]
            loss = F.cross_entropy(shift_logits, shift_labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        avg_loss = loss.item()

    return {"sft_loss": avg_loss}


def kl_step(model, batch, coef, device="cuda"):
    """Compute KL divergence penalty against the frozen base model (adapter disabled).

    IMPORTANT: `model.disable_adapter_layers()` reverts to the *base Qwen3-8B* model,
    NOT the LPA checkpoint. This means KL regularizes toward the pre-training distribution.
    To regularize toward the LPA checkpoint specifically, one would need to:
      1. Keep a frozen copy of the LPA adapter weights, or
      2. Use a two-adapter approach (frozen reference + trainable).
    For now, base-model KL is the simpler first experiment. Set kl_coef=0.0 (default)
    to disable entirely.

    Returns:
        dict with 'kl_loss' (float), or empty dict if coef <= 0
    """
    if coef <= 0:
        return {}

    assert isinstance(model, PeftModel), "KL penalty requires a PeftModel"

    input_ids = batch["input_ids"].to(device)
    labels_mask = batch["labels_mask"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    with torch.autocast(device_type="cuda"):
        # Reference: frozen base (adapter disabled = LPA checkpoint state)
        with torch.no_grad():
            model.disable_adapter_layers()
            base_logits = model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits
            base_log_probs = base_logits[:, :-1][labels_mask[:, 1:]].log_softmax(dim=-1)
            model.enable_adapter_layers()

        # Current model logits
        current_logits = model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits
        current_probs = current_logits[:, :-1][labels_mask[:, 1:]].softmax(dim=-1)

        kl_loss = F.kl_div(base_log_probs, current_probs, reduction="batchmean")

    # Normalize and backward
    kl_loss_normalized = kl_loss / (kl_loss.detach() + 1e-8)
    (coef * kl_loss_normalized).backward()

    return {"kl_loss": kl_loss.item()}


def main():
    parser = argparse.ArgumentParser(
        description="SFT Recovery: fine-tune an LPA-trained checkpoint on benign data"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Base model name (e.g., Qwen/Qwen3-8B)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to LPA LoRA checkpoint (e.g., cache/.../checkpoint_50)",
    )
    parser.add_argument(
        "--sft_dataset",
        type=str,
        default="tatsu-lab/alpaca",
        help="HuggingFace dataset for SFT (default: tatsu-lab/alpaca)",
    )
    parser.add_argument(
        "--system_prompt_path",
        type=str,
        default=None,
        help="Path to system prompt file (default: minimal)",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "sft_recovery_config.json"),
        help="Path to SFT recovery config JSON",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--project_name", type=str, default="lpa-sft-recovery_alpaca_checkpoint50"
    )
    parser.add_argument("--cache_dir", type=str, default="cache")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run HarmBench + utility evals during training",
    )
    parser.add_argument(
        "--eval_freq", type=int, default=50, help="Evaluate every N steps (default: 50)"
    )
    parser.add_argument("--wandb-offline", action="store_true")
    parser.add_argument("--timestamp", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Load config
    if os.path.exists(args.config_path):
        with open(args.config_path, "r") as f:
            config = json.load(f)
    else:
        raise FileNotFoundError(f"Config file not found: {args.config_path}")

    num_steps = config["num_steps"]
    outer_learning_rate = config["outer_learning_rate"]
    max_batch_per_acc = config.get("max_batch_per_acc", None)
    n_checkpoints = config["N_checkpoints"]
    subset_size = config.get("alpaca_subset_size", None)
    kl_coef = config.get("kl_coef", 0.0)

    # Load system prompt
    if args.system_prompt_path is not None:
        with open(args.system_prompt_path, "r") as f:
            system_prompt = f.read().strip()
    else:
        system_prompt = "You are a helpful assistant."

    # Load model with LPA checkpoint
    model, tokenizer, model_type = load_model_with_checkpoint(
        args.model_name, args.checkpoint_path
    )

    # Prepare SFT data
    dataloader = prepare_alpaca_dataset(
        tokenizer=tokenizer,
        model_type=model_type,
        system_prompt=system_prompt,
        dataset_name=args.sft_dataset,
        subset_size=subset_size,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    data_iter = itertools.cycle(dataloader)

    # Set up evaluation if requested
    if args.eval:
        from tasks.harmbench.FastHarmBenchEvals import (
            run_attack_evals,
            run_general_evals,
        )

        print("Loading HarmBench classifier for evaluation...")
        cls_path = get_model_path("cais/HarmBench-Llama-2-13b-cls")
        cls = AutoModelForCausalLM.from_pretrained(
            cls_path, torch_dtype=torch.bfloat16, device_map="cpu"
        )
        cls_tok_path = get_model_path("meta-llama/Llama-2-7b-chat-hf")
        cls_tokenizer = AutoTokenizer.from_pretrained(cls_tok_path)
        cls_tokenizer.pad_token_id = cls_tokenizer.unk_token_id
        cls_tokenizer.padding_side = "left"
        print("HarmBench classifier loaded.")
    else:
        cls = None
        cls_tokenizer = None

    # Set up output directory
    if args.timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    else:
        timestamp = args.timestamp
    project_dir = os.path.join(args.cache_dir, f"{args.project_name}_{timestamp}")
    os.makedirs(project_dir, exist_ok=True)

    # Save run parameters
    run_params = {
        "model_name": args.model_name,
        "checkpoint_path": args.checkpoint_path,
        "sft_dataset": args.sft_dataset,
        "system_prompt_path": args.system_prompt_path,
        "batch_size": args.batch_size,
        "config": config,
        "timestamp": timestamp,
        "experiment_type": "sft_recovery",
    }
    with open(os.path.join(project_dir, "parameters.json"), "w") as f:
        json.dump(run_params, f, indent=2)

    # Initialize W&B
    wandb_kwargs = {"mode": "offline"} if getattr(args, "wandb_offline", False) else {}
    wandb.init(
        project=args.project_name,
        name=timestamp,
        id=timestamp,
        config=run_params,
        **wandb_kwargs,
    )
    wandb.define_metric("step")
    wandb.define_metric("sft/*", step_metric="step")
    if args.eval:
        wandb.define_metric("harmbench/*", step_metric="step")
        wandb.define_metric("utility/*", step_metric="step")

    # Set up optimizer (persistent across steps — not reinitializing)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=outer_learning_rate,
    )

    # Training loop
    model.train()
    next_checkpoint = 1
    print(f"\n{'='*60}")
    print(f"Starting SFT Recovery Training")
    print(f"  Steps: {num_steps}")
    print(f"  Learning rate: {outer_learning_rate}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max batch per acc: {max_batch_per_acc}")
    print(f"  KL coef: {kl_coef}")
    print(f"  Subset size: {subset_size}")
    print(f"  Checkpoints: {n_checkpoints}")
    print(f"  Output: {project_dir}")
    print(f"{'='*60}\n")

    for step in tqdm(range(num_steps), desc="SFT Recovery"):
        batch = next(data_iter)

        # SFT step
        losses = sft_step(
            model=model,
            batch=batch,
            optimizer=optimizer,
            max_batch_per_acc=max_batch_per_acc,
        )

        # Optional KL penalty (backward pass adds to existing gradients)
        if kl_coef > 0:
            kl_losses = kl_step(model=model, batch=batch, coef=kl_coef)
            losses.update(kl_losses)

        # Log to W&B
        log = {"step": step + 1}
        for k, v in losses.items():
            log[f"sft/{k}"] = v
        wandb.log(log, step=step + 1)

        if (step + 1) % 10 == 0:
            loss_str = " | ".join(f"{k}: {v:.4f}" for k, v in losses.items())
            print(f"Step {step+1}/{num_steps} | {loss_str}")

        # Checkpointing (evenly spaced)
        if n_checkpoints and (step + 1) / num_steps >= next_checkpoint / n_checkpoints:
            ckpt_dir = os.path.join(project_dir, f"checkpoint_{step+1}")
            print(f"Saving checkpoint at step {step+1} → {ckpt_dir}")
            model.save_pretrained(ckpt_dir)
            next_checkpoint += 1

        # Evaluation
        if args.eval and (step + 1) % args.eval_freq == 0:
            print(f"\n--- Evaluating at step {step+1} ---")
            eval_cache = os.path.join(project_dir, f"eval_cache_{step+1}")
            with eval_mode(model):
                with torch.no_grad():
                    harmbench_asr = run_attack_evals(
                        model=model,
                        tokenizer=tokenizer,
                        model_type=model_type,
                        pretrained_cls="llama",
                        cls=cls,
                        cls_tokenizer=cls_tokenizer,
                        do_sample=False,
                        cache_dir=eval_cache,
                        verbose=True,
                        move_cls_device=True,
                        move_model_device=True,
                    )
                    harmbench_logs = {
                        f"harmbench/{k}": v for k, v in harmbench_asr.items()
                    }

                    utility_acc = run_general_evals(
                        model=model,
                        tokenizer=tokenizer,
                        model_type=model_type,
                        evals_to_include=[
                            "MMLU",
                            "HellaSwag",
                            "Winogrande",
                            "SciQ",
                            "Lambada",
                        ],
                        cache_dir=eval_cache,
                    )
                    utility_logs = {f"utility/{k}": v for k, v in utility_acc.items()}

                wandb.log(harmbench_logs, step=step + 1)
                wandb.log(utility_logs, step=step + 1)

                print(f"  HarmBench ASR: {harmbench_asr}")
                print(f"  Utility: {utility_acc}")

            torch.cuda.empty_cache()
            gc.collect()

    # Save final model
    final_dir = project_dir  # Save at project root level (like lat_training_no_sft)
    print(f"Saving final model to {final_dir}...")
    model.save_pretrained(final_dir)

    print("SFT Recovery training complete.")
    wandb.finish()

    # Cleanup
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
