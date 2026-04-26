#!/usr/bin/env python3
"""
Ablation Study: Structural Score Components vs Sensitivity Ground Truth

This script analyzes which structural metric component(s) are predictive
of actual pruning sensitivity:
- Utilization only (lower = more prunable?)
- Redundancy only (higher = more prunable?)
- Centrality only (lower = more prunable?)

For each metric, we:
1. Rank experts by that metric alone
2. Compare against sensitivity ground truth (negative = safe to prune)
3. Compute correlation and agreement metrics

Key insight: Sensitivity is the "ground truth" - experts with negative
sensitivity actually IMPROVE the model when removed.

Usage:
    python scripts/ablation_structural_metrics.py
    python scripts/ablation_structural_metrics.py --stats_path experiments/baseline/coactivation_stats.json
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats as scipy_stats

sys.path.insert(0, ".")

from src.pruning.metrics import load_metrics_from_stats


def parse_args():
    parser = argparse.ArgumentParser(description="Ablation study of structural score components")

    parser.add_argument("--stats_path", type=str,
                        default="experiments/baseline/coactivation_stats.json",
                        help="Path to co-activation statistics JSON")
    parser.add_argument("--sensitivity_path", type=str,
                        default="experiments/pruning/sensitivity_results.json",
                        help="Path to sensitivity results JSON")
    parser.add_argument("--output_dir", type=str,
                        default="experiments/pruning/ablation",
                        help="Output directory for ablation results")

    return parser.parse_args()


def load_sensitivity(path: str) -> Dict[str, float]:
    """Load sensitivity results from JSON."""
    with open(path, 'r') as f:
        data = json.load(f)
    return data["sensitivity"]


def get_metric_ranking(metrics, metric_name: str, higher_is_more_prunable: bool = True) -> Dict[int, List[int]]:
    """
    Rank experts by a single metric within each layer.

    Args:
        metrics: Dict of PruningMetrics per layer
        metric_name: Name of metric attribute (e.g., "utilization", "redundancy")
        higher_is_more_prunable: If True, higher values = rank higher for pruning

    Returns:
        Dict mapping layer_idx to list of expert indices (most prunable first)
    """
    rankings = {}

    for layer_idx, m in metrics.items():
        values = getattr(m, metric_name)
        if values is None:
            continue

        # Sort: if higher_is_more_prunable, descending; else ascending
        if higher_is_more_prunable:
            order = np.argsort(values)[::-1]  # Descending
        else:
            order = np.argsort(values)  # Ascending

        rankings[layer_idx] = order.tolist()

    return rankings


def get_sensitivity_ranking(sensitivity: Dict[str, float], num_layers: int, num_experts: int) -> Dict[int, List[int]]:
    """
    Rank experts by sensitivity (most prunable = most negative sensitivity first).

    Negative sensitivity means removing the expert IMPROVES the model,
    so these should be ranked FIRST for pruning.

    Returns:
        Dict mapping layer_idx to list of expert indices (most prunable first)
    """
    rankings = {}

    for layer_idx in range(num_layers):
        sensitivities = []
        for expert_idx in range(num_experts):
            key = f"layer_{layer_idx}_expert_{expert_idx}"
            sensitivities.append(sensitivity[key])

        # Lower (more negative) sensitivity = more prunable = rank first
        order = np.argsort(sensitivities)
        rankings[layer_idx] = order.tolist()

    return rankings


def compute_ranking_agreement(ranking_a: Dict[int, List[int]],
                               ranking_b: Dict[int, List[int]]) -> Dict[str, float]:
    """
    Compute agreement metrics between two rankings.

    Returns:
        Dict with agreement metrics:
        - top1_agreement: % of layers where top-1 expert matches
        - rank_correlation: Spearman correlation across all experts
        - kendall_tau: Kendall's tau correlation
    """
    top1_matches = 0
    all_ranks_a = []
    all_ranks_b = []

    for layer_idx in ranking_a:
        if layer_idx not in ranking_b:
            continue

        rank_a = ranking_a[layer_idx]
        rank_b = ranking_b[layer_idx]

        # Top-1 agreement
        if rank_a[0] == rank_b[0]:
            top1_matches += 1

        # Convert to rank positions for correlation
        # rank_a is [expert_idx...] sorted by pruning priority
        # We need position of each expert in the ranking
        num_experts = len(rank_a)
        pos_a = [0] * num_experts
        pos_b = [0] * num_experts

        for pos, expert_idx in enumerate(rank_a):
            pos_a[expert_idx] = pos
        for pos, expert_idx in enumerate(rank_b):
            pos_b[expert_idx] = pos

        all_ranks_a.extend(pos_a)
        all_ranks_b.extend(pos_b)

    num_layers = len(ranking_a)

    # Compute correlations
    spearman_corr, spearman_p = scipy_stats.spearmanr(all_ranks_a, all_ranks_b)
    kendall_tau, kendall_p = scipy_stats.kendalltau(all_ranks_a, all_ranks_b)

    return {
        "top1_agreement": top1_matches / num_layers if num_layers > 0 else 0,
        "top1_matches": top1_matches,
        "num_layers": num_layers,
        "spearman_correlation": spearman_corr,
        "spearman_p_value": spearman_p,
        "kendall_tau": kendall_tau,
        "kendall_p_value": kendall_p,
    }


def compute_metric_sensitivity_correlation(metrics, sensitivity: Dict[str, float],
                                            metric_name: str) -> Dict[str, float]:
    """
    Compute direct correlation between metric values and sensitivity values.

    This measures if the metric itself (not ranking) correlates with sensitivity.
    """
    metric_values = []
    sensitivity_values = []

    for layer_idx, m in metrics.items():
        values = getattr(m, metric_name)
        if values is None:
            continue

        for expert_idx, val in enumerate(values):
            key = f"layer_{layer_idx}_expert_{expert_idx}"
            if key in sensitivity:
                metric_values.append(val)
                sensitivity_values.append(sensitivity[key])

    if len(metric_values) < 3:
        return {"pearson": 0, "spearman": 0}

    pearson_corr, pearson_p = scipy_stats.pearsonr(metric_values, sensitivity_values)
    spearman_corr, spearman_p = scipy_stats.spearmanr(metric_values, sensitivity_values)

    return {
        "pearson_correlation": pearson_corr,
        "pearson_p_value": pearson_p,
        "spearman_correlation": spearman_corr,
        "spearman_p_value": spearman_p,
    }


def analyze_expert2_pattern(metrics, sensitivity: Dict[str, float], num_layers: int) -> Dict:
    """
    Special analysis for Expert 2, which is harmful in all layers.

    Question: Do any structural metrics identify Expert 2 as prunable?
    """
    expert2_analysis = {
        "sensitivity": [],
        "utilization": [],
        "redundancy": [],
        "eigenvector_centrality": [],
        "structural_score": [],
        "utilization_rank": [],
        "redundancy_rank": [],
        "centrality_rank": [],
        "structural_rank": [],
    }

    for layer_idx in range(num_layers):
        if layer_idx not in metrics:
            continue

        m = metrics[layer_idx]
        key = f"layer_{layer_idx}_expert_2"

        expert2_analysis["sensitivity"].append(sensitivity.get(key, 0))
        expert2_analysis["utilization"].append(m.utilization[2])
        expert2_analysis["redundancy"].append(m.redundancy[2])
        expert2_analysis["eigenvector_centrality"].append(m.eigenvector_centrality[2])
        expert2_analysis["structural_score"].append(m.structural_score[2])

        # Compute rank of Expert 2 in each metric
        # Utilization: lower = more prunable, so rank by ascending
        util_rank = np.argsort(m.utilization).tolist().index(2) + 1
        # Redundancy: higher = more prunable, so rank by descending
        redun_rank = np.argsort(m.redundancy)[::-1].tolist().index(2) + 1
        # Centrality: lower = more prunable, so rank by ascending
        cent_rank = np.argsort(m.eigenvector_centrality).tolist().index(2) + 1
        # Structural score: higher = more prunable
        struct_rank = np.argsort(m.structural_score)[::-1].tolist().index(2) + 1

        expert2_analysis["utilization_rank"].append(util_rank)
        expert2_analysis["redundancy_rank"].append(redun_rank)
        expert2_analysis["centrality_rank"].append(cent_rank)
        expert2_analysis["structural_rank"].append(struct_rank)

    # Summary stats
    expert2_analysis["summary"] = {
        "avg_sensitivity": np.mean(expert2_analysis["sensitivity"]),
        "always_negative": all(s < 0 for s in expert2_analysis["sensitivity"]),
        "avg_utilization_rank": np.mean(expert2_analysis["utilization_rank"]),
        "avg_redundancy_rank": np.mean(expert2_analysis["redundancy_rank"]),
        "avg_centrality_rank": np.mean(expert2_analysis["centrality_rank"]),
        "avg_structural_rank": np.mean(expert2_analysis["structural_rank"]),
        "times_ranked_1st_by_utilization": expert2_analysis["utilization_rank"].count(1),
        "times_ranked_1st_by_redundancy": expert2_analysis["redundancy_rank"].count(1),
        "times_ranked_1st_by_centrality": expert2_analysis["centrality_rank"].count(1),
        "times_ranked_1st_by_structural": expert2_analysis["structural_rank"].count(1),
    }

    return expert2_analysis


def print_layer_comparison(metrics, sensitivity: Dict[str, float], num_layers: int, num_experts: int):
    """Print detailed per-layer comparison of metric rankings vs sensitivity."""

    print("\n" + "=" * 90)
    print("Per-Layer Ranking Comparison: Structural Metrics vs Sensitivity Ground Truth")
    print("=" * 90)
    print("\nRanking interpretation: Expert ranked 1st = most recommended for pruning")
    print("Sensitivity ranking: Most negative (harmful expert) ranked 1st")
    print()

    print(f"{'Layer':<6} {'Sensitivity':<14} {'Utilization':<14} {'Redundancy':<14} {'Centrality':<14} {'Structural':<14}")
    print("-" * 90)

    for layer_idx in range(num_layers):
        if layer_idx not in metrics:
            continue

        m = metrics[layer_idx]

        # Get rankings (list of expert indices, most prunable first)
        sens_vals = [sensitivity[f"layer_{layer_idx}_expert_{e}"] for e in range(num_experts)]
        sens_rank = np.argsort(sens_vals).tolist()  # Most negative first

        util_rank = np.argsort(m.utilization).tolist()  # Lowest first
        redun_rank = np.argsort(m.redundancy)[::-1].tolist()  # Highest first
        cent_rank = np.argsort(m.eigenvector_centrality).tolist()  # Lowest first
        struct_rank = np.argsort(m.structural_score)[::-1].tolist()  # Highest first

        # Format as "E2>E0>E1>E3"
        def format_rank(r):
            return ">".join(f"E{e}" for e in r)

        print(f"{layer_idx:<6} {format_rank(sens_rank):<14} {format_rank(util_rank):<14} "
              f"{format_rank(redun_rank):<14} {format_rank(cent_rank):<14} {format_rank(struct_rank):<14}")

    # Print which metric's top-1 matches sensitivity's top-1
    print("\n" + "-" * 90)
    print("Top-1 Match Analysis (does metric's #1 pick match sensitivity's #1?)")
    print("-" * 90)

    matches = {"utilization": 0, "redundancy": 0, "centrality": 0, "structural": 0}

    for layer_idx in range(num_layers):
        if layer_idx not in metrics:
            continue

        m = metrics[layer_idx]

        sens_vals = [sensitivity[f"layer_{layer_idx}_expert_{e}"] for e in range(num_experts)]
        sens_top1 = np.argmin(sens_vals)  # Most negative = best prune target

        util_top1 = np.argmin(m.utilization)
        redun_top1 = np.argmax(m.redundancy)
        cent_top1 = np.argmin(m.eigenvector_centrality)
        struct_top1 = np.argmax(m.structural_score)

        if util_top1 == sens_top1:
            matches["utilization"] += 1
        if redun_top1 == sens_top1:
            matches["redundancy"] += 1
        if cent_top1 == sens_top1:
            matches["centrality"] += 1
        if struct_top1 == sens_top1:
            matches["structural"] += 1

    for metric, count in matches.items():
        pct = count / num_layers * 100
        print(f"  {metric:<15}: {count}/{num_layers} layers ({pct:.1f}%)")


def main():
    args = parse_args()

    print("=" * 70)
    print("Structural Score Component Ablation Study")
    print("=" * 70)

    # Load data
    print(f"\nLoading metrics from: {args.stats_path}")
    metrics = load_metrics_from_stats(args.stats_path)

    print(f"Loading sensitivity from: {args.sensitivity_path}")
    sensitivity = load_sensitivity(args.sensitivity_path)

    num_layers = len(metrics)
    num_experts = metrics[0].num_experts if 0 in metrics else 4

    print(f"\nAnalyzing {num_layers} layers, {num_experts} experts each")

    # Get sensitivity ranking (ground truth)
    sensitivity_ranking = get_sensitivity_ranking(sensitivity, num_layers, num_experts)

    # Define metrics to analyze
    # Format: (metric_name, higher_is_more_prunable, display_name)
    metric_configs = [
        ("utilization", False, "Utilization (low=prune)"),
        ("redundancy", True, "Redundancy (high=prune)"),
        ("eigenvector_centrality", False, "Centrality (low=prune)"),
        ("structural_score", True, "Structural Score"),
    ]

    results = {}

    print("\n" + "=" * 70)
    print("Ranking Agreement with Sensitivity Ground Truth")
    print("=" * 70)
    print(f"\n{'Metric':<30} {'Top-1 Match':<15} {'Spearman ρ':<15} {'Kendall τ':<15}")
    print("-" * 75)

    for metric_name, higher_is_prunable, display_name in metric_configs:
        # Get ranking by this metric
        metric_ranking = get_metric_ranking(metrics, metric_name, higher_is_prunable)

        # Compute agreement with sensitivity
        agreement = compute_ranking_agreement(metric_ranking, sensitivity_ranking)

        # Compute direct correlation
        correlation = compute_metric_sensitivity_correlation(metrics, sensitivity, metric_name)

        results[metric_name] = {
            "display_name": display_name,
            "ranking_agreement": agreement,
            "value_correlation": correlation,
        }

        top1_str = f"{agreement['top1_matches']}/{agreement['num_layers']} ({agreement['top1_agreement']*100:.0f}%)"
        spearman_str = f"{agreement['spearman_correlation']:.3f}"
        kendall_str = f"{agreement['kendall_tau']:.3f}"

        print(f"{display_name:<30} {top1_str:<15} {spearman_str:<15} {kendall_str:<15}")

    # Value correlation (metric value vs sensitivity value)
    print("\n" + "=" * 70)
    print("Direct Correlation: Metric Values vs Sensitivity Values")
    print("=" * 70)
    print("\nInterpretation:")
    print("  - Positive correlation: higher metric → higher sensitivity (more important)")
    print("  - Negative correlation: higher metric → lower sensitivity (less important)")
    print("  - For a good pruning metric, we want to identify LOW sensitivity experts")
    print()
    print(f"{'Metric':<30} {'Pearson r':<15} {'Spearman ρ':<15} {'Interpretation':<20}")
    print("-" * 80)

    for metric_name, _, display_name in metric_configs:
        corr = results[metric_name]["value_correlation"]
        pearson = corr["pearson_correlation"]
        spearman = corr["spearman_correlation"]

        # Interpretation
        if metric_name == "utilization":
            # Low util should predict low/negative sensitivity
            # So we expect positive correlation (low util → low sens)
            if pearson > 0.3:
                interp = "✓ Low util → safe"
            elif pearson < -0.3:
                interp = "✗ Inverted!"
            else:
                interp = "~ Weak signal"
        elif metric_name == "redundancy":
            # High redundancy should predict low sensitivity
            # So we expect negative correlation (high redun → low sens)
            if pearson < -0.3:
                interp = "✓ High redun → safe"
            elif pearson > 0.3:
                interp = "✗ Inverted!"
            else:
                interp = "~ Weak signal"
        elif metric_name == "eigenvector_centrality":
            # Low centrality should predict low sensitivity
            # So we expect positive correlation (low cent → low sens)
            if pearson > 0.3:
                interp = "✓ Low cent → safe"
            elif pearson < -0.3:
                interp = "✗ Inverted!"
            else:
                interp = "~ Weak signal"
        else:
            # Structural score: high should predict low sensitivity
            if pearson < -0.3:
                interp = "✓ High score → safe"
            elif pearson > 0.3:
                interp = "✗ Inverted!"
            else:
                interp = "~ Weak signal"

        print(f"{display_name:<30} {pearson:>+.3f}{'':>8} {spearman:>+.3f}{'':>8} {interp:<20}")

    # Detailed per-layer comparison
    print_layer_comparison(metrics, sensitivity, num_layers, num_experts)

    # Expert 2 special analysis
    print("\n" + "=" * 70)
    print("Special Analysis: Expert 2 (Harmful in All Layers)")
    print("=" * 70)

    expert2 = analyze_expert2_pattern(metrics, sensitivity, num_layers)
    summary = expert2["summary"]

    print(f"\nExpert 2 is harmful (negative sensitivity) in ALL {num_layers} layers: {summary['always_negative']}")
    print(f"Average sensitivity: {summary['avg_sensitivity']:.2f} (negative = improves model when removed)")
    print()
    print("How often did each metric rank Expert 2 as #1 for pruning?")
    print(f"  Utilization:      {summary['times_ranked_1st_by_utilization']}/{num_layers} layers (avg rank: {summary['avg_utilization_rank']:.1f})")
    print(f"  Redundancy:       {summary['times_ranked_1st_by_redundancy']}/{num_layers} layers (avg rank: {summary['avg_redundancy_rank']:.1f})")
    print(f"  Centrality:       {summary['times_ranked_1st_by_centrality']}/{num_layers} layers (avg rank: {summary['avg_centrality_rank']:.1f})")
    print(f"  Structural Score: {summary['times_ranked_1st_by_structural']}/{num_layers} layers (avg rank: {summary['avg_structural_rank']:.1f})")

    # Key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Find best metric
    best_metric = None
    best_top1 = -1
    for metric_name, _, display_name in metric_configs:
        top1 = results[metric_name]["ranking_agreement"]["top1_agreement"]
        if top1 > best_top1:
            best_top1 = top1
            best_metric = display_name

    print(f"\n1. Best single metric for top-1 prediction: {best_metric} ({best_top1*100:.0f}% agreement)")

    # Check if any metric identifies Expert 2
    if summary["times_ranked_1st_by_utilization"] >= num_layers // 2:
        print("2. Utilization often identifies Expert 2 as prunable")
    elif summary["times_ranked_1st_by_redundancy"] >= num_layers // 2:
        print("2. Redundancy often identifies Expert 2 as prunable")
    else:
        print("2. NO metric consistently identifies Expert 2 (harmful expert) as #1 prune target!")

    # Overall assessment
    if best_top1 < 0.5:
        print("3. CONCLUSION: Structural metrics alone are INSUFFICIENT for pruning decisions")
        print("   Sensitivity measurement is necessary for accurate expert selection")
    else:
        print(f"3. {best_metric} shows promise with {best_top1*100:.0f}% top-1 agreement")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert numpy types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        return obj

    output_data = {
        "metric_analysis": convert_for_json(results),
        "expert2_analysis": convert_for_json(expert2),
        "num_layers": num_layers,
        "num_experts": num_experts,
    }

    output_path = output_dir / "structural_ablation_results.json"
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)

    return results, expert2


if __name__ == "__main__":
    main()
