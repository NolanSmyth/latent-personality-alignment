import os
import argparse
import wandb
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from tasks.harmbench.FastHarmBenchEvals import run_attack_evals, run_general_evals
from peft import PeftModel
from latent_at.paths import get_model_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument(
        "--project_name", type=str, default="latent-personality-alignment"
    )
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--epoch", type=str, default=None)
    parser.add_argument(
        "--base_model",
        action="store_true",
        help="Evaluate the base model without loading a LoRA adapter",
    )
    parser.add_argument(
        "--attacks",
        type=str,
        nargs="+",
        default=None,
        help="HarmBench attacks to run (e.g., DirectRequest GCG). If not specified, runs all.",
    )
    parser.add_argument(
        "--evals",
        type=str,
        nargs="+",
        default=None,
        help="Utility benchmarks to run (e.g., MMLU HellaSwag). If not specified, runs all.",
    )
    parser.add_argument(
        "--skip_harmbench",
        action="store_true",
        help="Skip HarmBench evaluations entirely",
    )
    parser.add_argument(
        "--skip_utility",
        action="store_true",
        help="Skip utility evaluations entirely",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Skip logging to W&B",
    )

    args = parser.parse_args()

    project_name = args.project_name
    model_name = args.model_name
    run_id = args.run_id
    epoch = args.epoch if args.epoch else None  # treat empty string as None
    base_model_only = args.base_model
    project_path = "cache/" + project_name + "_" + run_id

    if epoch is not None:
        project_path += "/checkpoint_" + epoch

    model_dtype = torch.bfloat16
    model_path = get_model_path(model_name)

    print(f"Loading model {model_name} from {model_path}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=model_dtype, device_map="cuda"
    )
    print("Model loaded.")

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
    elif "zephyr" in model_name or "mistral" in model_name:
        model_type = "zephyr"
        tokenizer = AutoTokenizer.from_pretrained(
            get_model_path("HuggingFaceH4/zephyr-7b-beta")
        )
        tokenizer.pad_token_id = tokenizer.unk_token_id
        tokenizer.padding_side = "left"
    elif "Qwen" in model_name:
        model_type = "qwen3"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
    else:
        print(model_name)
        raise Exception("Unsupported model type.")
    print("Tokenizer loaded.")

    if base_model_only:
        model = base_model
        print("Evaluating base model (no adapter).")
    else:
        if not os.path.isdir(project_path):
            raise FileNotFoundError(
                f"Checkpoint directory not found: {project_path}\n"
                f"Training may not have completed or saved checkpoints yet."
            )
        adapter_config = os.path.join(project_path, "adapter_config.json")
        if not os.path.isfile(adapter_config):
            raise FileNotFoundError(
                f"No adapter_config.json in {project_path}\n"
                f"This directory exists but does not contain a valid LoRA adapter."
            )
        print(f"Loading LoRA adapter from {project_path}...")
        model = PeftModel.from_pretrained(base_model, project_path, device_map="auto")

    model.eval()
    harmbench_logs = {}
    utility_logs = {}

    with torch.no_grad():
        # Run HarmBench evaluations (unless skipped)
        if not args.skip_harmbench:
            print("Running HarmBench evaluations...")
            harmbench_asr = run_attack_evals(
                model=model,
                tokenizer=tokenizer,
                model_type=model_type,
                pretrained_cls="llama",
                do_sample=False,
                move_cls_device=True,
                move_model_device=True,
                cache_dir=project_path + "/eval",
                only_run_evals=args.attacks,  # Filter attacks if specified
            )
            harmbench_logs = {f"harmbench/{k}": v for k, v in harmbench_asr.items()}

        # Run utility evaluations (unless skipped)
        if not args.skip_utility:
            print("Running utility evaluations...")
            evals_to_run = (
                args.evals
                if args.evals
                else ["MMLU", "HellaSwag", "Winogrande", "SciQ", "Lambada"]
            )
            utility_acc = run_general_evals(
                model=model,
                tokenizer=tokenizer,
                model_type=model_type,
                evals_to_include=evals_to_run,
                cache_dir=project_path + "/eval",
            )
            utility_logs = {f"utility/{k}": v for k, v in utility_acc.items()}

    # Log to W&B (unless disabled)
    if not args.no_wandb:
        wandb.init(project=project_name, id=run_id, resume="allow")

        if epoch is not None:
            wandb.define_metric("epoch")
            wandb.define_metric("harmbench/*", step_metric="epoch", step_sync=True)
            wandb.define_metric("utility/*", step_metric="epoch", step_sync=True)
            log_dict = {"epoch": int(epoch)}
        else:
            log_dict = {}
            wandb.define_metric("harmbench/*")
            wandb.define_metric("utility/*")

        log_dict.update(harmbench_logs)
        log_dict.update(utility_logs)

        wandb.log(log_dict)
        wandb.finish()
    else:
        # Print results to stdout in JSON format for easy parsing
        import json as json_module

        results = {"epoch": int(epoch) if epoch else None}
        results.update(harmbench_logs)
        results.update(utility_logs)
        print("\n=== EVALUATION RESULTS (JSON) ===")
        print(json_module.dumps(results, indent=2))
        print("=== END RESULTS ===\n")


if __name__ == "__main__":
    main()
