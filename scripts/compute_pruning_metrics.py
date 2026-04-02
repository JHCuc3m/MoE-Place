#!/usr/bin/env python3
"""
Compute pruning metrics from collected co-activation statistics.

This script:
1. Loads co-activation statistics from baseline benchmark
2. Computes all pruning metrics (utilization, redundancy, centrality)
3. Ranks experts by pruning priority
4. Saves metrics for use in pruning experiments

Usage:
    python scripts/compute_pruning_metrics.py
    python scripts/compute_pruning_metrics.py --stats_path experiments/baseline/coactivation_stats.json
"""

import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, ".")

from src.pruning.metrics import (
    compute_all_metrics,
    print_metrics_summary,
    get_global_pruning_ranking,
    load_metrics_from_stats,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute pruning metrics from co-activation stats")

    parser.add_argument("--stats_path", type=str,
                        default="experiments/baseline/coactivation_stats.json",
                        help="Path to co-activation statistics JSON")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: same as stats file)")
    parser.add_argument("--centrality_metric", type=str, default="eigenvector",
                        choices=["degree", "betweenness", "eigenvector", "pagerank"],
                        help="Centrality metric for structural score")
    parser.add_argument("--top_k", type=int, default=1,
                        help="Number of pruning candidates per layer to show")

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("Computing Pruning Metrics")
    print("=" * 70)
    print(f"Stats file: {args.stats_path}")

    # Load and compute metrics
    metrics = load_metrics_from_stats(args.stats_path)

    print(f"\nComputed metrics for {len(metrics)} layers")

    # Print summary
    print_metrics_summary(metrics, top_k=args.top_k)

    # Get global pruning ranking
    ranking = get_global_pruning_ranking(metrics, strategy="per_layer")

    print("\n" + "=" * 70)
    print("Global Pruning Ranking (most prunable first)")
    print("=" * 70)
    print(f"{'Rank':<6} {'Layer':<8} {'Expert':<8} {'Score':<10}")
    print("-" * 40)

    for rank, (layer_idx, expert_idx, score) in enumerate(ranking[:20], 1):
        print(f"{rank:<6} {layer_idx:<8} {expert_idx:<8} {score:.4f}")

    if len(ranking) > 20:
        print(f"... ({len(ranking) - 20} more)")

    # Save metrics
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.stats_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save per-layer metrics
    metrics_dict = {
        "centrality_metric": args.centrality_metric,
        "layers": {idx: m.to_dict() for idx, m in metrics.items()},
    }

    metrics_path = output_dir / "pruning_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"\nSaved metrics to: {metrics_path}")

    # Save ranking
    ranking_data = [
        {"rank": i+1, "layer": l, "expert": e, "score": s}
        for i, (l, e, s) in enumerate(ranking)
    ]

    ranking_path = output_dir / "pruning_ranking.json"
    with open(ranking_path, 'w') as f:
        json.dump(ranking_data, f, indent=2)
    print(f"Saved ranking to: {ranking_path}")

    # Summary by layer: which expert to prune first
    print("\n" + "=" * 70)
    print("Recommended First Prune per Layer")
    print("=" * 70)

    for layer_idx, m in sorted(metrics.items()):
        candidate = m.get_pruning_candidates(1)[0]
        score = m.structural_score[candidate]
        util = m.utilization[candidate]
        redun = m.redundancy[candidate]

        print(f"Layer {layer_idx:2d}: Expert {candidate} "
              f"(score={score:.3f}, util={util:.3f}, redun={redun:.3f})")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)

    return metrics, ranking


if __name__ == "__main__":
    main()
