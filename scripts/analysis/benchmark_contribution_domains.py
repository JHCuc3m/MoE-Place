#!/usr/bin/env python3
"""
Benchmark expert contribution metrics across multiple domains.

Tests whether the C_i threshold (< 0.1 identifies routing sinks) generalizes
across different data domains: general text, code, math, scientific.

Key question: Is Expert 2 a "routing sink" universally, or only for WikiText-2?

Usage:
    # Run all domains
    python scripts/benchmark_contribution_domains.py

    # Quick test
    python scripts/benchmark_contribution_domains.py --quick

    # Specific domains
    python scripts/benchmark_contribution_domains.py --domains wikitext2 code math
"""

import sys
import argparse
import json
import torch
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark contribution metrics across domains"
    )

    parser.add_argument("--model", type=str, default=TINY_MIXTRAL,
                        help=f"Model name (default: {TINY_MIXTRAL})")
    parser.add_argument("--domains", type=str, nargs="+",
                        default=None,
                        help="Domains to test (default: all available)")
    parser.add_argument("--num_samples", type=int, default=128,
                        help="Number of samples per domain")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 32 samples, fewer batches")
    parser.add_argument("--output_dir", type=str,
                        default="experiments/pruning/domain_generalization",
                        help="Output directory")
    parser.add_argument("--ci_threshold", type=float, default=0.1,
                        help="C_i threshold for identifying routing sinks")

    return parser.parse_args()


def identify_routing_sinks(
    metrics: Dict,
    threshold: float = 0.1,
) -> Dict[str, List[tuple]]:
    """
    Identify routing sinks (experts with C_i < threshold).

    Returns:
        Dict with 'sinks' (list of (layer, expert) tuples) and 'sink_values'
    """
    magnitude = metrics['magnitude']

    sinks = []
    sink_values = {}

    for (layer, expert), ci in magnitude.items():
        if ci < threshold:
            sinks.append((layer, expert))
            sink_values[(layer, expert)] = ci

    return {
        'sinks': sinks,
        'sink_values': sink_values,
        'num_sinks': len(sinks),
    }


def compute_expert_summary(metrics: Dict) -> Dict[int, Dict]:
    """
    Compute per-expert summary statistics across all layers.

    Returns:
        Dict mapping expert_id -> {avg_ci, min_ci, max_ci, num_sink_layers}
    """
    magnitude = metrics['magnitude']

    # Group by expert
    expert_values = {}
    for (layer, expert), ci in magnitude.items():
        if expert not in expert_values:
            expert_values[expert] = []
        expert_values[expert].append(ci)

    # Compute statistics
    summary = {}
    for expert, values in expert_values.items():
        summary[expert] = {
            'avg_ci': sum(values) / len(values),
            'min_ci': min(values),
            'max_ci': max(values),
            'num_sink_layers': sum(1 for v in values if v < 0.1),
            'total_layers': len(values),
        }

    return summary


def run_domain_benchmark(
    model,
    tokenizer,
    domain: str,
    num_samples: int,
    batch_size: int,
    device: str,
    max_batches: int = None,
) -> Dict[str, Any]:
    """
    Run contribution metrics on a single domain.

    Returns:
        Dict with metrics and analysis
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Domain: {domain} ({DATASET_REGISTRY[domain]['description']})")
    logger.info(f"{'='*60}")

    # Load data
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

    # Compute contribution metrics
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


def print_comparison_table(all_results: Dict[str, Dict], threshold: float = 0.1):
    """Print comparison table across domains."""

    print("\n" + "=" * 80)
    print("CROSS-DOMAIN COMPARISON: Expert 2 Routing Sink Analysis")
    print("=" * 80)

    # Header
    print(f"\n{'Domain':<15} {'Avg C_i':<10} {'Min C_i':<10} {'Max C_i':<10} {'Sink Layers':<12} {'Is Sink?':<10}")
    print("-" * 80)

    expert_2_is_sink = {}

    for domain, result in all_results.items():
        if 'error' in result:
            print(f"{domain:<15} ERROR: {result['error']}")
            continue

        metrics = result['metrics']
        expert_summary = compute_expert_summary(metrics)

        if 2 in expert_summary:
            e2 = expert_summary[2]
            is_sink = bool(e2['avg_ci'] < threshold)  # Convert numpy bool to Python bool
            expert_2_is_sink[domain] = is_sink

            print(f"{domain:<15} {e2['avg_ci']:<10.4f} {e2['min_ci']:<10.4f} "
                  f"{e2['max_ci']:<10.4f} {e2['num_sink_layers']}/{e2['total_layers']:<10} "
                  f"{'YES' if is_sink else 'NO':<10}")
        else:
            print(f"{domain:<15} No Expert 2 data")

    # Summary
    print("-" * 80)
    num_sink_domains = sum(expert_2_is_sink.values())
    total_domains = len(expert_2_is_sink)

    print(f"\nExpert 2 is a routing sink in {num_sink_domains}/{total_domains} domains")

    if num_sink_domains == total_domains:
        print("CONCLUSION: Expert 2 is UNIVERSALLY a routing sink (C_i threshold generalizes)")
    elif num_sink_domains == 0:
        print("CONCLUSION: Expert 2 is NOT a routing sink in any domain (surprising!)")
    else:
        print("CONCLUSION: Expert 2 routing sink behavior is DOMAIN-DEPENDENT")

    return expert_2_is_sink


def print_all_experts_comparison(all_results: Dict[str, Dict], threshold: float = 0.1):
    """Print comparison for all experts across domains."""

    print("\n" + "=" * 80)
    print("ALL EXPERTS: Average C_i by Domain")
    print("=" * 80)

    # Collect all experts
    all_experts = set()
    for result in all_results.values():
        if 'error' not in result:
            for (layer, expert) in result['metrics']['magnitude'].keys():
                all_experts.add(expert)

    all_experts = sorted(all_experts)
    domains = [d for d in all_results.keys() if 'error' not in all_results[d]]

    # Header
    header = f"{'Expert':<10}" + "".join(f"{d:<15}" for d in domains)
    print(f"\n{header}")
    print("-" * (10 + 15 * len(domains)))

    # Each expert row
    for expert in all_experts:
        row = f"Expert {expert:<4}"
        for domain in domains:
            result = all_results[domain]
            summary = compute_expert_summary(result['metrics'])
            if expert in summary:
                ci = summary[expert]['avg_ci']
                marker = " *" if ci < threshold else ""
                row += f"{ci:<13.4f}{marker:<2}"
            else:
                row += f"{'N/A':<15}"
        print(row)

    print("-" * (10 + 15 * len(domains)))
    print("* = Below threshold (routing sink candidate)")


def main():
    args = parse_args()

    if args.quick:
        args.num_samples = 32
        max_batches = 10
    else:
        max_batches = None

    # Get domains to test
    available = get_available_datasets()
    domains = args.domains if args.domains else available

    # Validate domains
    for d in domains:
        if d not in available:
            print(f"Unknown domain: {d}. Available: {available}")
            return

    print("=" * 80)
    print("Domain Generalization: Expert Contribution Metrics")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Domains: {domains}")
    print(f"Samples per domain: {args.num_samples}")
    print(f"C_i threshold: {args.ci_threshold}")

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model (once)
    print(f"\nLoading model...")
    model, tokenizer = load_pretrained_moe(args.model, device=device)

    num_experts = getattr(model.config, 'num_local_experts', 4)
    num_layers = model.config.num_hidden_layers
    print(f"Model: {num_layers} layers, {num_experts} experts per layer")

    # Run on each domain
    all_results = {}

    for domain in domains:
        result = run_domain_benchmark(
            model=model,
            tokenizer=tokenizer,
            domain=domain,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            device=device,
            max_batches=max_batches,
        )
        all_results[domain] = result

        # Quick summary for this domain
        if 'error' not in result:
            sinks = identify_routing_sinks(result['metrics'], args.ci_threshold)
            print(f"  Routing sinks (C_i < {args.ci_threshold}): {sinks['num_sinks']} experts")

    # Print comparison tables
    expert_2_status = print_comparison_table(all_results, args.ci_threshold)
    print_all_experts_comparison(all_results, args.ci_threshold)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert metrics to serializable format (numpy types -> Python types)
    serializable_results = {}
    for domain, result in all_results.items():
        if 'error' in result:
            serializable_results[domain] = result
        else:
            serializable_results[domain] = {
                'num_samples': int(result['num_samples']),
                'num_windows': int(result['num_windows']),
                'metrics': {
                    'magnitude': {f"layer_{l}_expert_{e}": float(v)
                                  for (l, e), v in result['metrics']['magnitude'].items()},
                    'signed': {f"layer_{l}_expert_{e}": float(v)
                               for (l, e), v in result['metrics']['signed'].items()},
                    'count': {f"layer_{l}_expert_{e}": int(v)
                              for (l, e), v in result['metrics']['count'].items()},
                },
            }

    output_data = {
        'timestamp': datetime.now().isoformat(),
        'model': args.model,
        'num_samples': args.num_samples,
        'ci_threshold': args.ci_threshold,
        'domains': domains,
        'results': serializable_results,
        'expert_2_is_sink': expert_2_status,
        'conclusion': (
            "UNIVERSAL" if all(expert_2_status.values())
            else "DOMAIN_DEPENDENT" if any(expert_2_status.values())
            else "NOT_A_SINK"
        ),
    }

    results_path = output_dir / "domain_generalization_results.json"
    with open(results_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return all_results


if __name__ == "__main__":
    main()
