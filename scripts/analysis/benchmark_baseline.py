#!/usr/bin/env python3
"""
Benchmark baseline: Collect co-activation patterns and evaluate perplexity.

This script:
1. Loads a pretrained MoE model (TinyMixtral)
2. Collects co-activation statistics on a calibration subset
3. Evaluates perplexity on the full test set
4. Saves results for comparison with pruned models

Usage:
    python scripts/benchmark_baseline.py
    python scripts/benchmark_baseline.py --calibration_samples 256 --quick
"""

import sys
import argparse
import json
import torch
from pathlib import Path
from datetime import datetime

sys.path.insert(0, ".")

from src.models.pretrained_moe import (
    load_pretrained_moe,
    get_moe_config,
    MixtralRoutingCollector,
    TINY_MIXTRAL,
)
from src.routing.statistics import RoutingStatisticsCollector
from src.routing.visualization import print_collaboration_summary, export_statistics
from src.data.benchmarks import get_calibration_data, get_eval_data, create_dataloader
from src.evaluation.perplexity import compute_perplexity, evaluate_perplexity


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark baseline: co-activation collection + perplexity evaluation"
    )

    # Model
    parser.add_argument("--model", type=str, default=TINY_MIXTRAL,
                        help=f"HuggingFace model name (default: {TINY_MIXTRAL})")

    # Calibration settings
    parser.add_argument("--calibration_samples", type=int, default=128,
                        help="Number of samples for co-activation calibration (default: 128)")
    parser.add_argument("--calibration_max_length", type=int, default=256,
                        help="Max sequence length for calibration (default: 256)")

    # Evaluation settings
    parser.add_argument("--eval_batch_size", type=int, default=4,
                        help="Batch size for perplexity evaluation (default: 4)")
    parser.add_argument("--eval_max_length", type=int, default=512,
                        help="Max sequence length for evaluation (default: 512)")
    parser.add_argument("--eval_stride", type=int, default=256,
                        help="Stride for sliding window (default: 256)")

    # Quick mode
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer calibration samples, limited eval batches")
    parser.add_argument("--max_eval_batches", type=int, default=None,
                        help="Max evaluation batches (default: None = full eval)")

    # Output
    parser.add_argument("--output_dir", type=str, default="experiments/baseline",
                        help="Output directory (default: experiments/baseline)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    return parser.parse_args()


def collect_coactivation(
    model,
    tokenizer,
    routing_collector: MixtralRoutingCollector,
    stats_collector: RoutingStatisticsCollector,
    calibration_texts: list,
    max_length: int,
    device: str,
) -> int:
    """
    Collect co-activation statistics on calibration data.

    Returns total tokens processed.
    """
    from tqdm import tqdm

    total_tokens = 0

    print(f"\nCollecting co-activation on {len(calibration_texts)} calibration samples...")

    with torch.no_grad():
        for text in tqdm(calibration_texts, desc="Calibration"):
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=False,
            ).to(device)

            # Clear previous routing
            routing_collector.clear()

            # Forward pass
            _ = model(**inputs)

            # Collect routing decisions
            routing = routing_collector.get_last_routing()

            for layer_idx, layer_routing in routing.items():
                expert_indices = layer_routing["expert_indices"]
                stats_collector.record_routing(layer_idx, expert_indices)

            total_tokens += inputs.input_ids.shape[1]

    return total_tokens


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # Quick mode adjustments
    if args.quick:
        args.calibration_samples = min(args.calibration_samples, 64)
        args.max_eval_batches = args.max_eval_batches or 50
        print("Quick mode enabled: reduced calibration and evaluation")

    print("=" * 70)
    print("MoE Baseline Benchmark")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Calibration samples: {args.calibration_samples}")
    print(f"Quick mode: {args.quick}")
    print(f"Output: {args.output_dir}")

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Load model
    print(f"\nLoading model: {args.model}")
    model, tokenizer = load_pretrained_moe(args.model, device=device)

    config = get_moe_config(model)
    print(f"Model config: {config}")

    # Create collectors
    routing_collector = MixtralRoutingCollector(model)
    stats_collector = RoutingStatisticsCollector(
        num_layers=routing_collector.num_layers,
        num_experts=routing_collector.num_experts,
    )

    print(f"MoE layers: {routing_collector.num_layers}, Experts: {routing_collector.num_experts}")

    # =========================================================================
    # Phase 1: Collect co-activation statistics
    # =========================================================================
    print("\n" + "=" * 70)
    print("Phase 1: Co-activation Collection")
    print("=" * 70)

    calibration_texts, _ = get_calibration_data(
        tokenizer=tokenizer,
        num_samples=args.calibration_samples,
        max_length=args.calibration_max_length,
        seed=args.seed,
    )

    calibration_tokens = collect_coactivation(
        model=model,
        tokenizer=tokenizer,
        routing_collector=routing_collector,
        stats_collector=stats_collector,
        calibration_texts=calibration_texts,
        max_length=args.calibration_max_length,
        device=device,
    )

    print(f"\nCalibration complete: {calibration_tokens:,} tokens processed")

    # Print co-activation summary
    print_collaboration_summary(stats_collector, top_k=5)

    # Remove routing hooks before evaluation (cleaner)
    routing_collector.remove_hooks()

    # =========================================================================
    # Phase 2: Perplexity evaluation
    # =========================================================================
    print("\n" + "=" * 70)
    print("Phase 2: Perplexity Evaluation")
    print("=" * 70)

    ppl_results = evaluate_perplexity(
        model=model,
        tokenizer=tokenizer,
        dataset_name="wikitext2",
        split="test",
        batch_size=args.eval_batch_size,
        max_length=args.eval_max_length,
        stride=args.eval_stride,
        max_batches=args.max_eval_batches,
    )

    print(f"\n{'='*40}")
    print(f"PERPLEXITY: {ppl_results['perplexity']:.2f}")
    print(f"{'='*40}")
    print(f"Loss: {ppl_results['loss']:.4f}")
    print(f"Tokens evaluated: {ppl_results['total_tokens']:,}")

    # =========================================================================
    # Save results
    # =========================================================================
    print("\n" + "=" * 70)
    print("Saving Results")
    print("=" * 70)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save co-activation statistics
    stats_path = output_dir / "coactivation_stats.json"
    export_statistics(stats_collector, str(stats_path))
    print(f"Co-activation stats: {stats_path}")

    # Save perplexity results
    ppl_path = output_dir / "perplexity_results.json"
    with open(ppl_path, 'w') as f:
        json.dump(ppl_results, f, indent=2)
    print(f"Perplexity results: {ppl_path}")

    # Save full benchmark results
    benchmark_results = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "model_config": config,
        "calibration": {
            "num_samples": args.calibration_samples,
            "max_length": args.calibration_max_length,
            "total_tokens": calibration_tokens,
        },
        "evaluation": {
            "dataset": "wikitext2",
            "split": "test",
            "perplexity": ppl_results["perplexity"],
            "loss": ppl_results["loss"],
            "total_tokens": ppl_results["total_tokens"],
            "max_length": args.eval_max_length,
            "stride": args.eval_stride,
        },
        "coactivation_summary": stats_collector.get_summary(),
    }

    results_path = output_dir / "benchmark_results.json"
    with open(results_path, 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"Full results: {results_path}")

    # GPU memory
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        print(f"\nGPU memory: {allocated:.2f} GB")

    print("\n" + "=" * 70)
    print("Baseline Benchmark Complete!")
    print("=" * 70)

    # Summary
    print(f"""
Summary:
  - Model: {args.model}
  - Calibration: {args.calibration_samples} samples, {calibration_tokens:,} tokens
  - Perplexity: {ppl_results['perplexity']:.2f} on {ppl_results['total_tokens']:,} tokens
  - Results saved to: {output_dir}
    """)

    return benchmark_results


if __name__ == "__main__":
    main()
