#!/usr/bin/env python3
"""
Compute expert contribution metrics for MoE model.

This script computes contribution-based metrics that measure
the actual impact of each expert on model output:
- C_i (Magnitude): How much the expert contributes to output norm
- S_i (Signed): Whether expert contribution aligns with or opposes output

Negative S_i indicates a harmful expert that degrades model output.

Usage:
    # Full computation
    python scripts/compute_contribution_metrics.py

    # Quick test
    python scripts/compute_contribution_metrics.py --quick

    # Compare with sensitivity ground truth
    python scripts/compute_contribution_metrics.py --compare_sensitivity
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

from src.models.pretrained_moe import (
    load_pretrained_moe,
    TINY_MIXTRAL,
    DEEPSEEK_MOE_16B,
)
from src.data.benchmarks import get_calibration_data, create_dataloader
from src.pruning.contribution_metrics import (
    compute_contribution_scores,
    print_contribution_summary,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute expert contribution metrics")

    # Model
    parser.add_argument("--model", type=str, default=TINY_MIXTRAL,
                        help=f"Model name (default: {TINY_MIXTRAL})")

    # Data settings
    parser.add_argument("--num_samples", type=int, default=128,
                        help="Number of calibration samples")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Limit number of batches (for speed)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 32 samples, 10 batches")

    # Comparison with sensitivity
    parser.add_argument("--compare_sensitivity", action="store_true",
                        help="Compare with sensitivity ground truth")
    parser.add_argument("--sensitivity_path", type=str,
                        default="experiments/pruning/sensitivity_results.json",
                        help="Path to sensitivity results JSON")

    # Output
    parser.add_argument("--output_dir", type=str, default="experiments/pruning",
                        help="Output directory")

    return parser.parse_args()


def load_sensitivity_results(path: str):
    """Load sensitivity results for comparison."""
    with open(path, 'r') as f:
        data = json.load(f)

    # Parse sensitivity from JSON format
    sensitivity = {}
    if 'sensitivity' in data:
        for key, value in data['sensitivity'].items():
            # Key format: "layer_X_expert_Y"
            parts = key.split('_')
            layer_idx = int(parts[1])
            expert_idx = int(parts[3])
            sensitivity[(layer_idx, expert_idx)] = value

    return sensitivity


def compute_correlation(contrib_scores, sensitivity):
    """Compute correlation between contribution metrics and sensitivity."""
    import numpy as np
    from scipy import stats

    # Get common keys
    common_keys = set(contrib_scores.keys()) & set(sensitivity.keys())

    if not common_keys:
        return None

    contrib_vals = [contrib_scores[k] for k in common_keys]
    sens_vals = [sensitivity[k] for k in common_keys]

    # Spearman correlation
    corr, p_value = stats.spearmanr(contrib_vals, sens_vals)

    return {
        'spearman_rho': corr,
        'p_value': p_value,
        'n_samples': len(common_keys),
    }


def compute_ranking_agreement(contrib_scores, sensitivity, num_experts=4):
    """Compute top-1 and top-k agreement between rankings."""
    # For each layer, check if top-ranked expert matches
    num_layers = max(k[0] for k in contrib_scores.keys()) + 1

    top1_matches = 0
    layer_results = []

    for layer_idx in range(num_layers):
        # Get scores for this layer
        layer_contrib = {k[1]: v for k, v in contrib_scores.items() if k[0] == layer_idx}
        layer_sens = {k[1]: v for k, v in sensitivity.items() if k[0] == layer_idx}

        if not layer_contrib or not layer_sens:
            continue

        # For contribution: LOW S_i = should prune (sort ascending)
        # For sensitivity: LOW sensitivity = should prune (sort ascending)
        contrib_ranking = sorted(layer_contrib.keys(), key=lambda e: layer_contrib[e])
        sens_ranking = sorted(layer_sens.keys(), key=lambda e: layer_sens[e])

        # Top-1 agreement
        if contrib_ranking[0] == sens_ranking[0]:
            top1_matches += 1

        layer_results.append({
            'layer': layer_idx,
            'contrib_top1': contrib_ranking[0],
            'sens_top1': sens_ranking[0],
            'match': contrib_ranking[0] == sens_ranking[0],
        })

    top1_agreement = top1_matches / num_layers if num_layers > 0 else 0

    return {
        'top1_agreement': top1_agreement,
        'top1_matches': top1_matches,
        'num_layers': num_layers,
        'layer_results': layer_results,
    }


def main():
    args = parse_args()

    if args.quick:
        args.num_samples = 32
        args.max_batches = 10

    print("=" * 70)
    print("Expert Contribution Metrics")
    print("=" * 70)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading model: {args.model}")
    model, tokenizer = load_pretrained_moe(args.model, device=device)

    # Handle different config attributes for different model families
    num_routed_experts = (
        getattr(model.config, 'num_local_experts', None) or
        getattr(model.config, 'n_routed_experts', None) or
        4
    )
    num_shared_experts = getattr(model.config, 'n_shared_experts', 0)
    num_experts_per_tok = getattr(model.config, 'num_experts_per_tok', 2)
    num_layers = model.config.num_hidden_layers

    print(f"Model: {num_layers} layers")
    print(f"  Routed experts: {num_routed_experts} (top-{num_experts_per_tok} routing)")
    if num_shared_experts > 0:
        print(f"  Shared experts: {num_shared_experts} (always active)")

    # For backward compatibility
    num_experts = num_routed_experts

    # Load calibration data
    print("\nLoading calibration data...")
    _, calib_dataset = get_calibration_data(
        tokenizer,
        num_samples=args.num_samples,
        max_length=512,
    )
    calib_dataloader = create_dataloader(calib_dataset, batch_size=args.batch_size)
    print(f"Calibration: {len(calib_dataset)} windows")

    # Compute contribution metrics
    print("\n" + "=" * 70)
    print("Computing Contribution Metrics")
    print("=" * 70)

    results = compute_contribution_scores(
        model=model,
        dataloader=calib_dataloader,
        device=device,
        max_batches=args.max_batches,
        show_progress=True,
    )

    # Add model info to results for print_contribution_summary
    results['num_routed_experts'] = num_routed_experts
    results['num_shared_experts'] = num_shared_experts

    # Print summary
    print_contribution_summary(results)

    # Optionally compare with sensitivity
    if args.compare_sensitivity:
        print("\n" + "=" * 70)
        print("Comparison with Sensitivity Ground Truth")
        print("=" * 70)

        sensitivity_path = Path(args.sensitivity_path)
        if not sensitivity_path.exists():
            print(f"Sensitivity results not found at {sensitivity_path}")
            print("Run: python scripts/evaluate_pruning.py --sensitivity")
        else:
            sensitivity = load_sensitivity_results(str(sensitivity_path))

            # Get signed contribution scores
            signed_scores = results['metrics']['signed']

            # Compute correlation
            corr_result = compute_correlation(signed_scores, sensitivity)
            if corr_result:
                print(f"\nCorrelation (S_i vs Sensitivity):")
                print(f"  Spearman ρ: {corr_result['spearman_rho']:.4f}")
                print(f"  p-value: {corr_result['p_value']:.4f}")
                print(f"  n samples: {corr_result['n_samples']}")

                if corr_result['p_value'] < 0.05:
                    if corr_result['spearman_rho'] < 0:
                        print("  ✓ Significant NEGATIVE correlation (low S_i → low sensitivity → prune target)")
                    else:
                        print("  ✓ Significant positive correlation")
                else:
                    print("  ⚠️ Not statistically significant")

            # Compute ranking agreement
            agreement = compute_ranking_agreement(signed_scores, sensitivity, num_experts)
            print(f"\nRanking Agreement:")
            print(f"  Top-1 agreement: {agreement['top1_agreement']:.1%} ({agreement['top1_matches']}/{agreement['num_layers']} layers)")

            if agreement['top1_agreement'] > 0.25:
                print("  ✓ Better than random (25%)")
            else:
                print("  ⚠️ Not better than random")

            # Expert 2 detection test
            print("\nExpert 2 Detection Test:")
            expert2_signed = [v for (l, e), v in signed_scores.items() if e == 2]
            expert2_sens = [v for (l, e), v in sensitivity.items() if e == 2]

            if expert2_signed:
                avg_signed = sum(expert2_signed) / len(expert2_signed)
                avg_sens = sum(expert2_sens) / len(expert2_sens)
                print(f"  Expert 2 avg S_i: {avg_signed:+.4f}")
                print(f"  Expert 2 avg sensitivity: {avg_sens:+.2f}")

                if avg_signed < 0 and avg_sens < 0:
                    print("  ✓ Expert 2 correctly identified as harmful (negative S_i and sensitivity)")
                elif avg_sens < 0 and avg_signed >= 0:
                    print("  ⚠️ Expert 2 is harmful (sensitivity) but S_i is positive")
                else:
                    print(f"  S_i sign: {'negative' if avg_signed < 0 else 'positive'}, "
                          f"Sensitivity sign: {'negative' if avg_sens < 0 else 'positive'}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "num_samples": args.num_samples,
        "num_layers": num_layers,
        "num_routed_experts": num_routed_experts,
        "num_shared_experts": num_shared_experts,
        "num_experts_per_tok": num_experts_per_tok,
        "num_experts": num_experts,  # backward compatibility
        "metrics": {
            "magnitude": {f"layer_{l}_expert_{e}": v
                         for (l, e), v in results['metrics']['magnitude'].items()},
            "signed": {f"layer_{l}_expert_{e}": v
                      for (l, e), v in results['metrics']['signed'].items()},
            "count": {f"layer_{l}_expert_{e}": v
                     for (l, e), v in results['metrics']['count'].items()},
        },
    }

    results_path = output_dir / "contribution_metrics.json"
    with open(results_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return results


if __name__ == "__main__":
    main()
