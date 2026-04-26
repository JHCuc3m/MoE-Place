#!/usr/bin/env python3
"""
Cross-domain sensitivity and contribution analysis.

For each domain (wikitext2, code, math, scientific):
1. Compute C_i contribution metrics (calibration data)
2. Run sensitivity analysis (evaluation data)
3. Compare C_i vs sensitivity correlation
4. Determine if C_i threshold generalizes

This provides ground truth for validating whether C_i < 0.1
identifies harmful experts across all domains.

Usage:
    # Full analysis (all domains)
    python scripts/domain_sensitivity_analysis.py

    # Quick test
    python scripts/domain_sensitivity_analysis.py --quick

    # Specific domains
    python scripts/domain_sensitivity_analysis.py --domains wikitext2 code

    # Skip sensitivity (only compute C_i)
    python scripts/domain_sensitivity_analysis.py --skip_sensitivity
"""

import sys
import argparse
import json
import torch
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
from scipy import stats
import numpy as np

sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from src.models.pretrained_moe import load_pretrained_moe, TINY_MIXTRAL
from src.data.benchmarks import (
    get_available_datasets,
    load_dataset_by_name,
    create_dataloader,
    DATASET_REGISTRY,
)
from src.pruning.contribution_metrics import compute_contribution_scores
from src.pruning.expert_masking import MixtralExpertMasker, compute_sensitivity
from src.evaluation.perplexity import compute_perplexity


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-domain sensitivity and contribution analysis"
    )

    parser.add_argument("--model", type=str, default=TINY_MIXTRAL,
                        help=f"Model name (default: {TINY_MIXTRAL})")
    parser.add_argument("--domains", type=str, nargs="+",
                        default=None,
                        help="Domains to test (default: all available)")
    parser.add_argument("--num_samples", type=int, default=128,
                        help="Number of calibration samples per domain")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer samples and batches")
    parser.add_argument("--skip_sensitivity", action="store_true",
                        help="Skip sensitivity analysis (only compute C_i)")
    parser.add_argument("--output_dir", type=str,
                        default="experiments/pruning/domain_generalization",
                        help="Output directory")
    parser.add_argument("--ci_threshold", type=float, default=0.1,
                        help="C_i threshold for identifying routing sinks")

    return parser.parse_args()


def compute_domain_contribution(
    model,
    tokenizer,
    domain: str,
    num_samples: int,
    batch_size: int,
    device: str,
    max_batches: int = None,
) -> Dict[str, Any]:
    """Compute C_i contribution metrics for a domain."""

    logger.info(f"Computing C_i for domain: {domain}")

    try:
        texts, dataset = load_dataset_by_name(
            domain,
            tokenizer,
            max_samples=num_samples,
            max_length=512,
        )
        dataloader = create_dataloader(dataset, batch_size=batch_size)
        logger.info(f"Loaded {len(dataset)} windows from {len(texts)} samples")
    except Exception as e:
        logger.error(f"Failed to load {domain}: {e}")
        return {"error": str(e)}

    results = compute_contribution_scores(
        model=model,
        dataloader=dataloader,
        device=device,
        max_batches=max_batches,
        show_progress=True,
    )

    return {
        'metrics': results['metrics'],
        'num_samples': len(texts),
        'num_windows': len(dataset),
    }


def compute_domain_sensitivity(
    model,
    tokenizer,
    masker: MixtralExpertMasker,
    domain: str,
    num_samples: int,
    batch_size: int,
    device: str,
    max_batches: int = None,
) -> Dict[str, Any]:
    """Compute sensitivity analysis for a domain."""

    logger.info(f"Computing sensitivity for domain: {domain}")

    try:
        texts, dataset = load_dataset_by_name(
            domain,
            tokenizer,
            max_samples=num_samples,
            max_length=512,
        )
        dataloader = create_dataloader(dataset, batch_size=batch_size)
        logger.info(f"Loaded {len(dataset)} windows for evaluation")
    except Exception as e:
        logger.error(f"Failed to load {domain}: {e}")
        return {"error": str(e)}

    # Compute baseline perplexity
    masker.reset()
    baseline_results = compute_perplexity(
        model, dataloader, device, max_batches, show_progress=False
    )
    baseline_ppl = baseline_results["perplexity"]
    logger.info(f"Baseline perplexity ({domain}): {baseline_ppl:.2f}")

    # Compute per-expert sensitivity
    sensitivity = {}
    num_layers = len(masker.moe_layers)
    num_experts = getattr(model.config, 'num_local_experts', 4)

    for layer_idx in range(num_layers):
        for expert_idx in range(num_experts):
            masker.reset()
            masker.disable_expert(layer_idx, expert_idx)
            masker.apply_masking()

            results = compute_perplexity(
                model, dataloader, device, max_batches, show_progress=False
            )
            expert_ppl = results["perplexity"]
            sensitivity[(layer_idx, expert_idx)] = expert_ppl - baseline_ppl

            logger.info(f"  Layer {layer_idx} Expert {expert_idx}: "
                       f"PPL={expert_ppl:.2f}, Δ={expert_ppl - baseline_ppl:+.2f}")

    masker.reset()
    masker.remove_hooks()

    return {
        'baseline_ppl': baseline_ppl,
        'sensitivity': sensitivity,
        'num_samples': len(texts),
    }


def compute_correlation(ci_scores: Dict, sensitivity: Dict) -> Dict:
    """Compute Spearman correlation between C_i and sensitivity."""

    common_keys = set(ci_scores.keys()) & set(sensitivity.keys())
    if not common_keys:
        return {'error': 'No common keys'}

    ci_vals = [ci_scores[k] for k in common_keys]
    sens_vals = [sensitivity[k] for k in common_keys]

    corr, p_value = stats.spearmanr(ci_vals, sens_vals)

    return {
        'spearman_rho': corr,
        'p_value': p_value,
        'n_samples': len(common_keys),
    }


def compute_ranking_agreement(ci_scores: Dict, sensitivity: Dict, num_layers: int) -> Dict:
    """Compute ranking agreement between C_i and sensitivity."""

    top1_matches = 0
    expert2_ci_lowest = 0  # Count layers where Expert 2 has lowest C_i
    expert2_sens_lowest = 0  # Count layers where Expert 2 has lowest sensitivity

    for layer_idx in range(num_layers):
        layer_ci = {k[1]: v for k, v in ci_scores.items() if k[0] == layer_idx}
        layer_sens = {k[1]: v for k, v in sensitivity.items() if k[0] == layer_idx}

        if not layer_ci or not layer_sens:
            continue

        # Lowest C_i = prune candidate; Lowest sensitivity = prune candidate
        ci_ranking = sorted(layer_ci.keys(), key=lambda e: layer_ci[e])
        sens_ranking = sorted(layer_sens.keys(), key=lambda e: layer_sens[e])

        if ci_ranking[0] == sens_ranking[0]:
            top1_matches += 1

        if ci_ranking[0] == 2:
            expert2_ci_lowest += 1
        if sens_ranking[0] == 2:
            expert2_sens_lowest += 1

    return {
        'top1_agreement': top1_matches / num_layers if num_layers > 0 else 0,
        'top1_matches': top1_matches,
        'num_layers': num_layers,
        'expert2_ci_lowest_layers': expert2_ci_lowest,
        'expert2_sens_lowest_layers': expert2_sens_lowest,
    }


def analyze_expert2(ci_scores: Dict, sensitivity: Dict, threshold: float = 0.1) -> Dict:
    """Analyze Expert 2 specifically across all layers."""

    expert2_ci = [v for (l, e), v in ci_scores.items() if e == 2]
    expert2_sens = [v for (l, e), v in sensitivity.items() if e == 2]

    other_ci = [v for (l, e), v in ci_scores.items() if e != 2]

    return {
        'expert2_avg_ci': np.mean(expert2_ci) if expert2_ci else None,
        'expert2_min_ci': np.min(expert2_ci) if expert2_ci else None,
        'expert2_max_ci': np.max(expert2_ci) if expert2_ci else None,
        'expert2_avg_sens': np.mean(expert2_sens) if expert2_sens else None,
        'expert2_is_routing_sink': np.mean(expert2_ci) < threshold if expert2_ci else None,
        'expert2_is_harmful': np.mean(expert2_sens) < 0 if expert2_sens else None,
        'other_experts_avg_ci': np.mean(other_ci) if other_ci else None,
        'ci_ratio': np.mean(other_ci) / np.mean(expert2_ci) if expert2_ci and other_ci else None,
    }


def print_domain_summary(domain: str, ci_result: Dict, sens_result: Dict, threshold: float):
    """Print summary for a single domain."""

    print(f"\n{'='*70}")
    print(f"Domain: {domain.upper()}")
    print(f"{'='*70}")

    if 'error' in ci_result:
        print(f"C_i Error: {ci_result['error']}")
        return

    ci_scores = ci_result['metrics']['magnitude']

    if sens_result and 'error' not in sens_result:
        sensitivity = sens_result['sensitivity']
        baseline_ppl = sens_result['baseline_ppl']

        print(f"\nBaseline Perplexity: {baseline_ppl:.2f}")

        # Correlation
        corr = compute_correlation(ci_scores, sensitivity)
        print(f"\nC_i vs Sensitivity Correlation:")
        print(f"  Spearman ρ: {corr['spearman_rho']:.4f}")
        print(f"  p-value: {corr['p_value']:.4f}")

        # Ranking agreement
        num_layers = max(k[0] for k in ci_scores.keys()) + 1
        agreement = compute_ranking_agreement(ci_scores, sensitivity, num_layers)
        print(f"\nRanking Agreement:")
        print(f"  Top-1: {agreement['top1_agreement']:.1%} ({agreement['top1_matches']}/{num_layers})")
        print(f"  Expert 2 lowest C_i: {agreement['expert2_ci_lowest_layers']}/{num_layers} layers")
        print(f"  Expert 2 lowest sensitivity: {agreement['expert2_sens_lowest_layers']}/{num_layers} layers")

        # Expert 2 analysis
        e2 = analyze_expert2(ci_scores, sensitivity, threshold)
        print(f"\nExpert 2 Analysis:")
        print(f"  Avg C_i: {e2['expert2_avg_ci']:.4f} (others: {e2['other_experts_avg_ci']:.4f})")
        print(f"  C_i ratio (others/expert2): {e2['ci_ratio']:.1f}x")
        print(f"  Avg sensitivity: {e2['expert2_avg_sens']:+.2f}")
        print(f"  Is routing sink (C_i < {threshold}): {'YES' if e2['expert2_is_routing_sink'] else 'NO'}")
        print(f"  Is harmful (sens < 0): {'YES' if e2['expert2_is_harmful'] else 'NO'}")
    else:
        # C_i only summary
        e2_ci = [v for (l, e), v in ci_scores.items() if e == 2]
        other_ci = [v for (l, e), v in ci_scores.items() if e != 2]

        print(f"\nExpert 2 C_i: avg={np.mean(e2_ci):.4f}, range=[{np.min(e2_ci):.4f}, {np.max(e2_ci):.4f}]")
        print(f"Other experts C_i: avg={np.mean(other_ci):.4f}")
        print(f"Is routing sink (C_i < {threshold}): {'YES' if np.mean(e2_ci) < threshold else 'NO'}")


def print_cross_domain_summary(all_results: Dict, threshold: float):
    """Print cross-domain comparison summary."""

    print("\n" + "=" * 80)
    print("CROSS-DOMAIN SUMMARY")
    print("=" * 80)

    # Header
    print(f"\n{'Domain':<12} {'Baseline PPL':<14} {'E2 Avg C_i':<12} {'E2 Avg Sens':<12} "
          f"{'Routing Sink?':<14} {'Harmful?':<10} {'Top-1 Agree':<12}")
    print("-" * 90)

    conclusions = []

    for domain, result in all_results.items():
        ci_result = result.get('contribution', {})
        sens_result = result.get('sensitivity', {})

        if 'error' in ci_result:
            print(f"{domain:<12} ERROR")
            continue

        ci_scores = ci_result['metrics']['magnitude']
        e2_ci = [v for (l, e), v in ci_scores.items() if e == 2]
        avg_ci = np.mean(e2_ci) if e2_ci else 0
        is_sink = avg_ci < threshold

        if sens_result and 'error' not in sens_result:
            sensitivity = sens_result['sensitivity']
            baseline = sens_result['baseline_ppl']
            e2_sens = [v for (l, e), v in sensitivity.items() if e == 2]
            avg_sens = np.mean(e2_sens) if e2_sens else 0
            is_harmful = avg_sens < 0

            num_layers = max(k[0] for k in ci_scores.keys()) + 1
            agreement = compute_ranking_agreement(ci_scores, sensitivity, num_layers)
            top1 = f"{agreement['top1_agreement']:.0%}"

            print(f"{domain:<12} {baseline:<14.2f} {avg_ci:<12.4f} {avg_sens:<+12.2f} "
                  f"{'YES' if is_sink else 'NO':<14} {'YES' if is_harmful else 'NO':<10} {top1:<12}")

            conclusions.append({
                'domain': domain,
                'is_sink': is_sink,
                'is_harmful': is_harmful,
                'agreement': agreement['top1_agreement'],
            })
        else:
            print(f"{domain:<12} {'N/A':<14} {avg_ci:<12.4f} {'N/A':<12} "
                  f"{'YES' if is_sink else 'NO':<14} {'N/A':<10} {'N/A':<12}")
            conclusions.append({
                'domain': domain,
                'is_sink': is_sink,
                'is_harmful': None,
            })

    print("-" * 90)

    # Final conclusion
    sink_domains = sum(1 for c in conclusions if c['is_sink'])
    harmful_domains = sum(1 for c in conclusions if c.get('is_harmful'))
    total = len(conclusions)

    print(f"\nExpert 2 is a routing sink in {sink_domains}/{total} domains")
    if harmful_domains > 0:
        print(f"Expert 2 is harmful in {harmful_domains}/{total} domains")

    if sink_domains == total:
        print("\n✓ CONCLUSION: C_i threshold GENERALIZES - Expert 2 is universally a routing sink")
    elif sink_domains == 0:
        print("\n✗ CONCLUSION: Expert 2 is NOT a routing sink in any domain")
    else:
        print(f"\n⚠ CONCLUSION: C_i threshold is DOMAIN-DEPENDENT ({sink_domains}/{total} domains)")


def main():
    args = parse_args()

    if args.quick:
        args.num_samples = 32
        max_batches = 10
    else:
        max_batches = None

    # Get domains
    available = get_available_datasets()
    domains = args.domains if args.domains else available

    for d in domains:
        if d not in available:
            print(f"Unknown domain: {d}. Available: {available}")
            return

    print("=" * 80)
    print("Cross-Domain Sensitivity & Contribution Analysis")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Domains: {domains}")
    print(f"Samples per domain: {args.num_samples}")
    print(f"C_i threshold: {args.ci_threshold}")
    print(f"Run sensitivity: {not args.skip_sensitivity}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading model...")
    model, tokenizer = load_pretrained_moe(args.model, device=device)

    num_experts = getattr(model.config, 'num_local_experts', 4)
    num_layers = model.config.num_hidden_layers
    print(f"Model: {num_layers} layers, {num_experts} experts per layer")

    # Create masker for sensitivity analysis
    masker = MixtralExpertMasker(model)

    # Run analysis for each domain
    all_results = {}

    for domain in domains:
        print(f"\n{'#'*70}")
        print(f"# Processing: {domain}")
        print(f"{'#'*70}")

        result = {}

        # Compute C_i
        ci_result = compute_domain_contribution(
            model=model,
            tokenizer=tokenizer,
            domain=domain,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            device=device,
            max_batches=max_batches,
        )
        result['contribution'] = ci_result

        # Compute sensitivity
        if not args.skip_sensitivity:
            sens_result = compute_domain_sensitivity(
                model=model,
                tokenizer=tokenizer,
                masker=masker,
                domain=domain,
                num_samples=args.num_samples,
                batch_size=args.batch_size,
                device=device,
                max_batches=max_batches,
            )
            result['sensitivity'] = sens_result

        all_results[domain] = result

        # Print domain summary
        print_domain_summary(
            domain,
            ci_result,
            result.get('sensitivity'),
            args.ci_threshold
        )

    # Print cross-domain summary
    print_cross_domain_summary(all_results, args.ci_threshold)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    serializable = {}
    for domain, result in all_results.items():
        serializable[domain] = {}

        if 'contribution' in result and 'error' not in result['contribution']:
            ci = result['contribution']
            serializable[domain]['contribution'] = {
                'num_samples': ci['num_samples'],
                'metrics': {
                    'magnitude': {f"layer_{l}_expert_{e}": v
                                  for (l, e), v in ci['metrics']['magnitude'].items()},
                    'signed': {f"layer_{l}_expert_{e}": v
                               for (l, e), v in ci['metrics']['signed'].items()},
                }
            }

        if 'sensitivity' in result and 'error' not in result.get('sensitivity', {}):
            sens = result['sensitivity']
            serializable[domain]['sensitivity'] = {
                'baseline_ppl': sens['baseline_ppl'],
                'sensitivity': {f"layer_{l}_expert_{e}": v
                               for (l, e), v in sens['sensitivity'].items()},
            }

    output_data = {
        'timestamp': datetime.now().isoformat(),
        'model': args.model,
        'num_samples': args.num_samples,
        'ci_threshold': args.ci_threshold,
        'domains': domains,
        'results': serializable,
    }

    results_path = output_dir / "domain_sensitivity_results.json"
    with open(results_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
