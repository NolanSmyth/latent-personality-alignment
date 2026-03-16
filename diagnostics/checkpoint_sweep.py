#!/usr/bin/env python3
"""
Checkpoint Sweep Evaluation Script

Evaluates multiple checkpoints from a training run to find the "sweet spot"
where safety improves but utility hasn't collapsed.

Usage:
    python diagnostics/checkpoint_sweep.py \
        --cache_dir cache/lpa-regular-config_IPIP-14_fewer_steps_2026-02-13_13-35-34-111503 \
        --start 10 --end 200 --step 10
"""

import os
import sys
import argparse
import subprocess
import json
import csv
from pathlib import Path


def parse_eval_json_output(stdout: str) -> dict:
    """Parse the JSON output from eval.py between the markers."""
    start_marker = "=== EVALUATION RESULTS (JSON) ==="
    end_marker = "=== END RESULTS ==="

    try:
        start_idx = stdout.index(start_marker) + len(start_marker)
        end_idx = stdout.index(end_marker)
        json_str = stdout[start_idx:end_idx].strip()
        return json.loads(json_str)
    except (ValueError, json.JSONDecodeError) as e:
        print(f"Warning: Could not parse JSON output: {e}")
        return {}


def count_pathological_responses(cache_dir: str, eval_type: str) -> dict:
    """
    Read cached evaluation results and count pathological responses.

    For MC benchmarks (MMLU): pathological = response starts with "I" (not A/B/C/D)
    For HarmBench: pathological = starts with "I do not agree" or "I don't agree"

    Returns dict with: n_total, n_pathological, pathological_rate
    """
    cache_path = Path(cache_dir) / eval_type / "cache.jsonl"

    if not cache_path.exists():
        return {"n_total": 0, "n_pathological": 0, "pathological_rate": None}

    n_total = 0
    n_pathological = 0

    with open(cache_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                response = item.get("model_response", "").strip()
                n_total += 1

                if eval_type == "DirectRequest":
                    # HarmBench: pathological if starts with refusal pattern
                    lower = response.lower()
                    if lower.startswith("i do not agree") or lower.startswith(
                        "i don't agree"
                    ):
                        n_pathological += 1
                else:
                    # MC benchmarks: pathological if starts with "I" instead of A/B/C/D
                    first_char = response[0].upper() if response else ""
                    if first_char == "I":
                        n_pathological += 1
            except json.JSONDecodeError:
                continue

    pathological_rate = n_pathological / n_total if n_total > 0 else None
    return {
        "n_total": n_total,
        "n_pathological": n_pathological,
        "pathological_rate": pathological_rate,
    }


def run_eval_for_checkpoint(
    model_name: str,
    project_name: str,
    run_id: str,
    epoch: int,
    cache_base: str,
) -> dict:
    """
    Run eval.py for a specific checkpoint and return results.

    Returns dict with: asr_direct_request, mmlu_accuracy, pathological stats
    """
    checkpoint_dir = Path(cache_base) / f"checkpoint_{epoch}"

    if not checkpoint_dir.exists():
        print(f"  Warning: Checkpoint {epoch} not found at {checkpoint_dir}")
        return None

    # Check if adapter_config.json exists
    if not (checkpoint_dir / "adapter_config.json").exists():
        print(f"  Warning: No adapter_config.json in checkpoint {epoch}")
        return None

    # Build eval command
    cmd = [
        sys.executable,
        "-m",
        "eval",
        "--model_name",
        model_name,
        "--project_name",
        project_name,
        "--run_id",
        run_id,
        "--epoch",
        str(epoch),
        "--attacks",
        "DirectRequest",
        "--evals",
        "MMLU",
        "--no_wandb",
    ]

    print(f"  Running eval.py for checkpoint {epoch}...")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),  # Run from project root
            timeout=1800,  # 30 minute timeout per checkpoint
        )

        if result.returncode != 0:
            print(f"  Error running eval for checkpoint {epoch}:")
            print(f"  stderr: {result.stderr[:500]}")
            return None

        # Parse JSON output
        eval_results = parse_eval_json_output(result.stdout)

    except subprocess.TimeoutExpired:
        print(f"  Timeout running eval for checkpoint {epoch}")
        return None
    except Exception as e:
        print(f"  Exception running eval for checkpoint {epoch}: {e}")
        return None

    # Get metrics from eval output
    asr = eval_results.get("harmbench/DirectRequest")
    mmlu = eval_results.get("utility/MMLU")

    # Count pathological responses from cache
    eval_cache_dir = checkpoint_dir / "eval"

    pathological_dr = count_pathological_responses(str(eval_cache_dir), "DirectRequest")
    pathological_mmlu = count_pathological_responses(str(eval_cache_dir), "MMLU")

    # Combine pathological stats
    n_total = pathological_dr["n_total"] + pathological_mmlu["n_total"]
    n_pathological = (
        pathological_dr["n_pathological"] + pathological_mmlu["n_pathological"]
    )
    pathological_rate = n_pathological / n_total if n_total > 0 else None

    return {
        "step": epoch,
        "asr_direct_request": asr,
        "mmlu_accuracy": mmlu,
        "pathological_rate": pathological_rate,
        "n_pathological": n_pathological,
        "n_total": n_total,
        "pathological_dr": pathological_dr["n_pathological"],
        "pathological_mmlu": pathological_mmlu["n_pathological"],
    }


def main():
    parser = argparse.ArgumentParser(description="Checkpoint sweep evaluation")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="cache/lpa-regular-config_IPIP-14_fewer_steps_2026-02-13_13-35-34-111503",
        help="Path to the run folder containing checkpoints",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Model name",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="diagnostics/checkpoint_sweep_results.csv",
        help="Where to save results CSV",
    )
    parser.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        default=None,
        help="Specific checkpoints to evaluate (e.g., 10 30 50)",
    )
    parser.add_argument("--start", type=int, default=10, help="Start checkpoint")
    parser.add_argument(
        "--end", type=int, default=200, help="End checkpoint (inclusive)"
    )
    parser.add_argument("--step", type=int, default=10, help="Checkpoint step size")

    args = parser.parse_args()

    # Parse cache_dir to extract project_name and run_id
    cache_path = Path(args.cache_dir)
    cache_name = (
        cache_path.name
    )  # e.g., "lpa-regular-config_IPIP-14_fewer_steps_2026-02-13_13-35-34-111503"

    # Split on timestamp pattern (YYYY-MM-DD_HH-MM-SS-NNNNNN)
    # The run_id is the timestamp portion
    parts = cache_name.rsplit(
        "_", 4
    )  # Split from right, max 4 splits for YYYY-MM-DD_HH-MM-SS-NNNNNN
    if len(parts) >= 5:
        project_name = "_".join(parts[:-4])
        run_id = "_".join(parts[-4:])
    else:
        # Fallback: assume last part after underscore is run_id
        parts = cache_name.rsplit("_", 1)
        project_name = parts[0]
        run_id = parts[1] if len(parts) > 1 else "unknown"

    print(f"Cache directory: {args.cache_dir}")
    print(f"Project name: {project_name}")
    print(f"Run ID: {run_id}")
    print(f"Model: {args.model_name}")
    print()

    # Determine checkpoints to evaluate
    if args.checkpoints:
        checkpoints = sorted(args.checkpoints)
    else:
        checkpoints = list(range(args.start, args.end + 1, args.step))

    print(f"Evaluating checkpoints: {checkpoints}")
    print("=" * 60)

    results = []

    for epoch in checkpoints:
        print(f"\n[Checkpoint {epoch}]")
        result = run_eval_for_checkpoint(
            model_name=args.model_name,
            project_name=project_name,
            run_id=run_id,
            epoch=epoch,
            cache_base=args.cache_dir,
        )

        if result:
            results.append(result)
            print(
                f"  ASR: {result['asr_direct_request']:.3f}"
                if result["asr_direct_request"] is not None
                else "  ASR: N/A"
            )
            print(
                f"  MMLU: {result['mmlu_accuracy']:.3f}"
                if result["mmlu_accuracy"] is not None
                else "  MMLU: N/A"
            )
            print(
                f"  Pathological: {result['pathological_rate']:.1%} ({result['n_pathological']}/{result['n_total']})"
                if result["pathological_rate"] is not None
                else "  Pathological: N/A"
            )

    # Save results to CSV
    if results:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "step",
            "asr_direct_request",
            "mmlu_accuracy",
            "pathological_rate",
            "n_pathological",
            "n_total",
            "pathological_dr",
            "pathological_mmlu",
        ]

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"\n{'=' * 60}")
        print(f"Results saved to: {output_path}")

        # Print summary table
        print("\n=== SUMMARY ===")
        print(f"{'Step':>6} | {'ASR':>6} | {'MMLU':>6} | {'Pathological':>12}")
        print("-" * 40)
        for r in results:
            asr = (
                f"{r['asr_direct_request']:.3f}"
                if r["asr_direct_request"] is not None
                else "N/A"
            )
            mmlu = (
                f"{r['mmlu_accuracy']:.3f}" if r["mmlu_accuracy"] is not None else "N/A"
            )
            path = (
                f"{r['pathological_rate']:.1%}"
                if r["pathological_rate"] is not None
                else "N/A"
            )
            print(f"{r['step']:>6} | {asr:>6} | {mmlu:>6} | {path:>12}")
    else:
        print("\nNo results collected!")


if __name__ == "__main__":
    main()
