#!/usr/bin/env python3
"""
Evaluate perplexity impact of globally pruning Expert 2 across all domains.

For each domain (wikitext2, code, math, scientific):
  1. Compute baseline perplexity
  2. Mask Expert 2 in ALL layers simultaneously
  3. Compute pruned perplexity
  4. Report delta

This is cheap: 2 forward passes per domain (not 48).

Usage:
    python scripts/eval_pruned_cross_domain.py
    python scripts/eval_pruned_cross_domain.py --quick
    python scripts/eval_pruned_cross_domain.py --expert_idx 2 --domains wikitext2 code
"""

import sys
import argparse
import json
import torch
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from src.models.pretrained_moe import load_pretrained_moe, TINY_MIXTRAL
from src.data.benchmarks import get_available_datasets, load_dataset_by_name, create_dataloader
from src.pruning.expert_masking import MixtralExpertMasker
from src.evaluation.perplexity import compute_perplexity


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-domain pruning impact evaluation")
    parser.add_argument("--model", type=str, default=TINY_MIXTRAL)
    parser.add_argument("--expert_idx", type=int, default=2,
                        help="Expert index to prune globally (default: 2)")
    parser.add_argument("--domains", type=str, nargs="+", default=None,
                        help="Domains to evaluate (default: all)")
    parser.add_argument("--num_samples", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 32 samples, 20 eval batches")
    parser.add_argument("--output_dir", type=str,
                        default="experiments/pruning/domain_generalization")
    return parser.parse_args()


def main():
    args = parse_args()

    num_samples = 32 if args.quick else args.num_samples
    max_batches = 20 if args.quick else None

    domains = args.domains or get_available_datasets()

    print("=" * 70)
    print("Cross-Domain Pruning Impact Evaluation")
    print("=" * 70)
    print(f"Model:        {args.model}")
    print(f"Expert pruned: Expert {args.expert_idx} (all layers globally)")
    print(f"Domains:      {domains}")
    print(f"Samples:      {num_samples} per domain")
    print(f"Quick mode:   {args.quick}")
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    logger.info("Loading model...")
    model, tokenizer = load_pretrained_moe(args.model, device=device)
    num_layers = model.config.num_hidden_layers
    num_experts = getattr(model.config, "num_local_experts", 4)
    print(f"Model loaded: {num_layers} layers, {num_experts} experts per layer\n")

    masker = MixtralExpertMasker(model)

    results = {}

    for domain in domains:
        print(f"\n{'─' * 50}")
        print(f"Domain: {domain}")
        print(f"{'─' * 50}")

        try:
            texts, dataset = load_dataset_by_name(
                domain, tokenizer, max_samples=num_samples, max_length=512
            )
            dataloader = create_dataloader(dataset, batch_size=args.batch_size)
            logger.info(f"Loaded {len(dataset)} windows from {len(texts)} samples")
        except Exception as e:
            logger.error(f"Failed to load {domain}: {e}")
            results[domain] = {"error": str(e)}
            continue

        # Baseline perplexity
        masker.reset()
        baseline = compute_perplexity(model, dataloader, device, max_batches, show_progress=True)
        baseline_ppl = baseline["perplexity"]
        print(f"Baseline PPL:     {baseline_ppl:.2f}")

        # Prune Expert {expert_idx} in all layers
        masker.reset()
        for layer_idx in range(num_layers):
            masker.disable_expert(layer_idx, args.expert_idx)
        masker.apply_masking()

        pruned = compute_perplexity(model, dataloader, device, max_batches, show_progress=True)
        pruned_ppl = pruned["perplexity"]
        delta = pruned_ppl - baseline_ppl
        print(f"Pruned PPL:       {pruned_ppl:.2f}")
        print(f"ΔPPL:             {delta:+.2f}  ({'improves' if delta < 0 else 'degrades'})")

        masker.reset()
        masker.remove_hooks()

        # Recreate masker for next domain (remove_hooks cleans up)
        masker = MixtralExpertMasker(model)

        results[domain] = {
            "baseline_ppl": baseline_ppl,
            "pruned_ppl": pruned_ppl,
            "delta_ppl": delta,
            "num_samples": len(texts),
            "num_windows": len(dataset),
        }

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Global Expert 2 Pruning Impact Across Domains")
    print("=" * 70)
    print(f"{'Domain':<18} {'Baseline PPL':>13} {'Pruned PPL':>11} {'ΔPPL':>10}  {'Result'}")
    print("─" * 70)
    for domain, res in results.items():
        if "error" in res:
            print(f"{domain:<18} ERROR: {res['error']}")
        else:
            result_str = "IMPROVES ✓" if res["delta_ppl"] < 0 else "DEGRADES ✗"
            print(f"{domain:<18} {res['baseline_ppl']:>13.2f} {res['pruned_ppl']:>11.2f} "
                  f"{res['delta_ppl']:>+10.2f}  {result_str}")
    print("─" * 70)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "cross_domain_pruning_impact.json"

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "expert_pruned": args.expert_idx,
        "pruned_layers": "all",
        "num_samples": num_samples,
        "domains": domains,
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
