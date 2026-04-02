#!/usr/bin/env python3
"""
Evaluate the impact of expert pruning on model performance.

This script:
1. Loads the model and pruning metrics
2. Disables experts according to the pruning ranking
3. Evaluates perplexity before and after pruning
4. Optionally runs full sensitivity analysis

Usage:
    # Evaluate pruning top-1 expert per layer
    python scripts/evaluate_pruning.py --num_prune 1

    # Evaluate pruning specific experts
    python scripts/evaluate_pruning.py --prune_experts "0:1,2:3"

    # Run full sensitivity analysis
    python scripts/evaluate_pruning.py --sensitivity

    # Quick test
    python scripts/evaluate_pruning.py --num_prune 1 --quick
"""

import sys
import argparse
import json
import torch
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, ".")

# Enable logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

from src.models.pretrained_moe import load_pretrained_moe, TINY_MIXTRAL
from src.data.benchmarks import get_eval_data, create_dataloader
from src.evaluation.perplexity import compute_perplexity
from src.pruning.expert_masking import (
    MixtralExpertMasker,
    compute_sensitivity,
    print_sensitivity_summary,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate expert pruning impact")

    # Model
    parser.add_argument("--model", type=str, default=TINY_MIXTRAL,
                        help=f"Model name (default: {TINY_MIXTRAL})")

    # Pruning specification
    parser.add_argument("--num_prune", type=int, default=0,
                        help="Number of experts to prune (uses ranking from metrics)")
    parser.add_argument("--prune_experts", type=str, default=None,
                        help="Specific experts to prune: 'layer:expert,layer:expert,...'")
    parser.add_argument("--prune_per_layer", type=int, default=None,
                        help="Prune this many experts per layer (e.g., 1 = prune 1 from each layer)")

    # Metrics file
    parser.add_argument("--ranking_path", type=str,
                        default="experiments/baseline/pruning_ranking.json",
                        help="Path to pruning ranking JSON")

    # Sensitivity analysis
    parser.add_argument("--sensitivity", action="store_true",
                        help="Run full sensitivity analysis (slow)")

    # Evaluation settings
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Evaluation batch size")
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Limit eval batches (for speed)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: limit to 50 batches")

    # Output
    parser.add_argument("--output_dir", type=str, default="experiments/pruning",
                        help="Output directory")

    return parser.parse_args()


def test_masking(model, tokenizer, masker, device):
    """Verify that masking actually changes routing."""
    from src.models.pretrained_moe import MixtralRoutingCollector

    text = "The capital of France is Paris."
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # Collect routing without masking
    collector = MixtralRoutingCollector(model)

    with torch.no_grad():
        _ = model(**inputs)
    routing_before = collector.get_last_routing()
    collector.clear()

    # Apply masking to layer 0, expert 1
    masker.reset()
    masker.disable_expert(0, 1)
    masker.apply_masking()

    with torch.no_grad():
        _ = model(**inputs)
    routing_after = collector.get_last_routing()

    collector.remove_hooks()
    masker.remove_hooks()
    masker.reset()

    # Compare
    before_experts = routing_before[0]['expert_indices'].flatten().tolist()
    after_experts = routing_after[0]['expert_indices'].flatten().tolist()

    print(f"  Layer 0 routing before masking: {before_experts[:10]}...")
    print(f"  Layer 0 routing after masking E1: {after_experts[:10]}...")

    if 1 in after_experts:
        print("  ⚠️  WARNING: Expert 1 still selected after masking - masking may not work!")
    else:
        print("  ✓ Masking verified: Expert 1 no longer selected")


def load_pruning_ranking(ranking_path: str):
    """Load pruning ranking from JSON."""
    with open(ranking_path, 'r') as f:
        data = json.load(f)

    # Convert to list of tuples
    ranking = [(item["layer"], item["expert"], item["score"]) for item in data]
    return ranking


def parse_prune_experts(spec: str):
    """Parse expert specification string like '0:1,2:3' into list of tuples."""
    experts = []
    for pair in spec.split(","):
        layer, expert = pair.split(":")
        experts.append((int(layer), int(expert)))
    return experts


def main():
    args = parse_args()

    if args.quick:
        args.max_batches = 50

    print("=" * 70)
    print("Expert Pruning Evaluation")
    print("=" * 70)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading model: {args.model}")
    model, tokenizer = load_pretrained_moe(args.model, device=device)

    num_experts = getattr(model.config, 'num_local_experts', 4)
    num_layers = model.config.num_hidden_layers
    print(f"Model: {num_layers} layers, {num_experts} experts per layer")

    # Create masker
    masker = MixtralExpertMasker(model)

    # Verify masking works
    print("\nVerifying masking mechanism...")
    test_masking(model, tokenizer, masker, device)

    # Load evaluation data
    print("\nLoading evaluation data...")
    _, eval_dataset = get_eval_data(tokenizer, max_length=512, stride=256)
    eval_dataloader = create_dataloader(eval_dataset, batch_size=args.batch_size)
    print(f"Evaluation: {len(eval_dataset)} windows")

    # =========================================================================
    # Baseline evaluation
    # =========================================================================
    print("\n" + "=" * 70)
    print("Baseline Evaluation (no pruning)")
    print("=" * 70)

    masker.reset()
    baseline_results = compute_perplexity(
        model, eval_dataloader, device, args.max_batches
    )
    baseline_ppl = baseline_results["perplexity"]

    print(f"\nBaseline Perplexity: {baseline_ppl:.2f}")

    # =========================================================================
    # Sensitivity analysis (optional)
    # =========================================================================
    sensitivity_results = None
    if args.sensitivity:
        print("\n" + "=" * 70)
        print("Sensitivity Analysis")
        print("=" * 70)
        print("Computing perplexity with each expert disabled...")

        sensitivity_results = compute_sensitivity(
            model=model,
            tokenizer=tokenizer,
            masker=masker,
            eval_dataloader=eval_dataloader,
            device=device,
            max_batches=args.max_batches,
        )

        print_sensitivity_summary(sensitivity_results, num_experts)

    # =========================================================================
    # Pruning evaluation
    # =========================================================================
    experts_to_prune = []

    # Determine which experts to prune
    if args.prune_experts:
        experts_to_prune = parse_prune_experts(args.prune_experts)
        print(f"\nPruning specified experts: {experts_to_prune}")

    elif args.prune_per_layer:
        # Load ranking and prune top-k from each layer
        ranking = load_pruning_ranking(args.ranking_path)

        # Group by layer
        by_layer = {}
        for layer, expert, score in ranking:
            if layer not in by_layer:
                by_layer[layer] = []
            by_layer[layer].append((expert, score))

        # Take top-k from each layer
        for layer in sorted(by_layer.keys()):
            for expert, score in by_layer[layer][:args.prune_per_layer]:
                experts_to_prune.append((layer, expert))

        print(f"\nPruning {args.prune_per_layer} expert(s) per layer: {experts_to_prune}")

    elif args.num_prune > 0:
        # Load ranking and take top-k globally
        ranking = load_pruning_ranking(args.ranking_path)
        experts_to_prune = [(layer, expert) for layer, expert, _ in ranking[:args.num_prune]]
        print(f"\nPruning top {args.num_prune} experts globally: {experts_to_prune}")

    # Evaluate with pruning
    if experts_to_prune:
        print("\n" + "=" * 70)
        print(f"Pruned Evaluation ({len(experts_to_prune)} experts disabled)")
        print("=" * 70)

        # Disable experts
        masker.reset()
        for layer, expert in experts_to_prune:
            masker.disable_expert(layer, expert)
        masker.apply_masking()

        # Evaluate
        pruned_results = compute_perplexity(
            model, eval_dataloader, device, args.max_batches
        )
        pruned_ppl = pruned_results["perplexity"]

        masker.remove_hooks()

        # Summary
        ppl_increase = pruned_ppl - baseline_ppl
        ppl_percent = (ppl_increase / baseline_ppl) * 100

        print(f"\n{'=' * 50}")
        print(f"RESULTS")
        print(f"{'=' * 50}")
        print(f"Baseline Perplexity:  {baseline_ppl:.2f}")
        print(f"Pruned Perplexity:    {pruned_ppl:.2f}")
        print(f"Increase:             {ppl_increase:+.2f} ({ppl_percent:+.1f}%)")
        print(f"Experts Pruned:       {len(experts_to_prune)} / {num_layers * num_experts}")
        print(f"{'=' * 50}")

    # =========================================================================
    # Save results
    # =========================================================================
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "baseline_perplexity": baseline_ppl,
        "num_layers": num_layers,
        "num_experts": num_experts,
    }

    if experts_to_prune:
        results["pruning"] = {
            "experts_pruned": [(l, e) for l, e in experts_to_prune],
            "num_pruned": len(experts_to_prune),
            "pruned_perplexity": pruned_ppl,
            "ppl_increase": ppl_increase,
            "ppl_increase_percent": ppl_percent,
        }

    if sensitivity_results:
        results["sensitivity"] = {
            f"layer_{l}_expert_{e}": score
            for (l, e), score in sensitivity_results.items()
        }

    results_path = output_dir / "pruning_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return results


if __name__ == "__main__":
    main()
