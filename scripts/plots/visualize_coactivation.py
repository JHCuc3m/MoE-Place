#!/usr/bin/env python3
"""
Visualize co-activation matrices from saved statistics.

This script:
1. Loads co-activation statistics from baseline benchmark
2. Generates heatmaps for each layer
3. Optionally shows pruning metric annotations

Usage:
    python scripts/visualize_coactivation.py
    python scripts/visualize_coactivation.py --layer 0
    python scripts/visualize_coactivation.py --stats_path experiments/baseline/coactivation_stats.json
"""

import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, ".")

import torch
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("matplotlib not installed. Install with: pip install matplotlib")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize co-activation matrices")

    parser.add_argument("--stats_path", type=str,
                        default="experiments/baseline/coactivation_stats.json",
                        help="Path to co-activation statistics JSON")
    parser.add_argument("--metrics_path", type=str,
                        default="experiments/baseline/pruning_metrics.json",
                        help="Path to pruning metrics JSON (for annotations)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: same as stats file)")
    parser.add_argument("--layer", type=int, default=None,
                        help="Specific layer to visualize (default: all)")
    parser.add_argument("--show", action="store_true",
                        help="Show plots interactively")
    parser.add_argument("--annotate", action="store_true",
                        help="Add pruning metric annotations")

    return parser.parse_args()


def load_coactivation_stats(stats_path: str) -> dict:
    """Load co-activation statistics from JSON."""
    with open(stats_path, 'r') as f:
        data = json.load(f)
    return data


def load_pruning_metrics(metrics_path: str) -> dict:
    """Load pruning metrics from JSON."""
    try:
        with open(metrics_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        return None


def plot_coactivation_heatmap(
    matrix: np.ndarray,
    layer_idx: int,
    expert_counts: list = None,
    pruning_candidate: int = None,
    structural_scores: list = None,
    save_path: str = None,
    show: bool = False,
    figsize: tuple = (8, 7),
):
    """
    Plot co-activation matrix as a clean heatmap.

    Shows only raw co-activation counts. Use plot_redundancy_analysis for probabilities.
    """
    if not HAS_MATPLOTLIB:
        return

    num_experts = matrix.shape[0]

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(matrix, cmap='Blues', aspect='equal')

    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel("Co-activation Count", rotation=-90, va="bottom", fontsize=10)

    # Simple expert labels
    labels = [f"Expert {i}" for i in range(num_experts)]
    ax.set_xticks(np.arange(num_experts))
    ax.set_yticks(np.arange(num_experts))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)

    # Add only the count values in cells (no percentages)
    for i in range(num_experts):
        for j in range(num_experts):
            value = int(matrix[i, j])
            text_color = "white" if matrix[i, j] > matrix.max() * 0.6 else "black"
            ax.text(j, i, f"{value:,}", ha="center", va="center",
                   color=text_color, fontsize=12, fontweight='bold')

    # Highlight pruning candidate with red border
    if pruning_candidate is not None:
        rect = plt.Rectangle((pruning_candidate - 0.5, -0.5), 1, num_experts,
                             fill=False, edgecolor='red', linewidth=3)
        ax.add_patch(rect)
        rect = plt.Rectangle((-0.5, pruning_candidate - 0.5), num_experts, 1,
                             fill=False, edgecolor='red', linewidth=3)
        ax.add_patch(rect)

    # Title
    title = f"Layer {layer_idx}: Co-activation Counts"
    if pruning_candidate is not None:
        title += f"  (Red = Pruning Candidate: Expert {pruning_candidate})"
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)

    # Axis labels
    ax.set_xlabel("Expert", fontsize=11)
    ax.set_ylabel("Expert", fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_redundancy_analysis(
    matrix: np.ndarray,
    layer_idx: int,
    expert_counts: list = None,
    pruning_candidate: int = None,
    save_path: str = None,
    show: bool = False,
):
    """
    Plot redundancy analysis: which expert is covered by which other expert.
    """
    if not HAS_MATPLOTLIB:
        return

    num_experts = matrix.shape[0]

    # Compute conditional probabilities P(j|i) for all pairs
    diagonal = np.diag(matrix)
    diagonal = np.maximum(diagonal, 1e-10)
    conditional = matrix / diagonal[:, np.newaxis]
    np.fill_diagonal(conditional, 0)

    # Compute redundancy and utilization
    redundancy = np.max(conditional, axis=1)
    max_partner = np.argmax(conditional, axis=1)

    if expert_counts is not None:
        total = sum(expert_counts)
        utilization = [c / total for c in expert_counts]
    else:
        utilization = [1/num_experts] * num_experts

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # =========== Plot 1: Utilization ===========
    ax1 = axes[0]
    colors1 = ['red' if i == pruning_candidate else 'steelblue' for i in range(num_experts)]
    bars1 = ax1.bar(range(num_experts), utilization, color=colors1, edgecolor='navy')

    ax1.set_xticks(range(num_experts))
    ax1.set_xticklabels([f"Expert {i}" for i in range(num_experts)], fontsize=10)
    ax1.set_ylabel("Utilization (fraction of tokens)", fontsize=10)
    ax1.set_title("How often is each expert selected?", fontsize=12, fontweight='bold')
    ax1.set_ylim(0, max(utilization) * 1.2)

    # Add percentage labels on bars
    for i, v in enumerate(utilization):
        ax1.text(i, v + 0.01, f"{v:.1%}", ha='center', fontsize=11, fontweight='bold')

    # =========== Plot 2: Redundancy ===========
    ax2 = axes[1]
    colors2 = ['red' if i == pruning_candidate else plt.cm.Oranges(r) for i, r in enumerate(redundancy)]
    bars2 = ax2.bar(range(num_experts), redundancy, color=colors2, edgecolor='darkorange')

    ax2.set_xticks(range(num_experts))
    ax2.set_xticklabels([f"Expert {i}" for i in range(num_experts)], fontsize=10)
    ax2.set_ylabel("Redundancy Score", fontsize=10)
    ax2.set_title("How often is expert accompanied by another?", fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1.1)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)

    # Add percentage and partner labels
    for i, (r, p) in enumerate(zip(redundancy, max_partner)):
        ax2.text(i, r + 0.03, f"{r:.0%}", ha='center', fontsize=11, fontweight='bold')
        ax2.text(i, r + 0.10, f"(by E{p})", ha='center', fontsize=9, color='gray')

    # =========== Plot 3: Summary Table ===========
    ax3 = axes[2]
    ax3.axis('off')

    # Create summary table
    table_data = []
    headers = ["Expert", "Utilization", "Redundancy", "Covered By", "Prune?"]

    for i in range(num_experts):
        prune_marker = "YES" if i == pruning_candidate else ""
        table_data.append([
            f"Expert {i}",
            f"{utilization[i]:.1%}",
            f"{redundancy[i]:.0%}",
            f"Expert {max_partner[i]}",
            prune_marker
        ])

    table = ax3.table(
        cellText=table_data,
        colLabels=headers,
        loc='center',
        cellLoc='center',
        colWidths=[0.2, 0.2, 0.2, 0.2, 0.15],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Color the pruning candidate row
    if pruning_candidate is not None:
        for j in range(len(headers)):
            table[(pruning_candidate + 1, j)].set_facecolor('#ffcccc')

    # Header styling
    for j in range(len(headers)):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    ax3.set_title(f"Layer {layer_idx} Summary", fontsize=12, fontweight='bold', pad=20)

    plt.suptitle(f"Layer {layer_idx}: Expert Redundancy Analysis", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def main():
    args = parse_args()

    if not HAS_MATPLOTLIB:
        print("ERROR: matplotlib is required for visualization")
        return

    print("=" * 70)
    print("Co-activation Visualization")
    print("=" * 70)
    print(f"Stats: {args.stats_path}")

    # Load data
    stats = load_coactivation_stats(args.stats_path)
    metrics = load_pruning_metrics(args.metrics_path) if args.annotate else None

    num_layers = stats["summary"]["num_layers"]
    num_experts = stats["summary"]["num_experts"]

    print(f"Layers: {num_layers}, Experts per layer: {num_experts}")

    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.stats_path).parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which layers to process
    if args.layer is not None:
        layers = [args.layer]
    else:
        layers = range(num_layers)

    for layer_idx in layers:
        print(f"\nProcessing Layer {layer_idx}...")

        # Get co-activation matrix
        matrix = np.array(stats["collaboration_matrices"][str(layer_idx)])

        # Get expert counts
        expert_counts = stats["summary"]["per_layer_stats"][str(layer_idx)]["expert_counts"]

        # Get pruning info if available
        pruning_candidate = None
        structural_scores = None
        if metrics is not None and "layers" in metrics:
            layer_metrics = metrics["layers"].get(str(layer_idx), {})
            structural_scores = layer_metrics.get("structural_score")
            if structural_scores:
                pruning_candidate = int(np.argmax(structural_scores))

        # Plot co-activation heatmap
        heatmap_path = output_dir / f"layer_{layer_idx}_coactivation.png"
        plot_coactivation_heatmap(
            matrix=matrix,
            layer_idx=layer_idx,
            expert_counts=expert_counts,
            pruning_candidate=pruning_candidate,
            structural_scores=structural_scores,
            save_path=str(heatmap_path),
            show=args.show,
        )

        # Plot redundancy analysis
        redundancy_path = output_dir / f"layer_{layer_idx}_redundancy.png"
        plot_redundancy_analysis(
            matrix=matrix,
            layer_idx=layer_idx,
            expert_counts=expert_counts,
            pruning_candidate=pruning_candidate,
            save_path=str(redundancy_path),
            show=args.show,
        )

    print(f"\n{'=' * 70}")
    print(f"Plots saved to: {output_dir}")
    print("=" * 70)

    # Print Layer 0 co-activation matrix to console
    if 0 in layers or args.layer == 0:
        print("\n" + "=" * 70)
        print("Layer 0 Co-activation Matrix (raw counts)")
        print("=" * 70)
        matrix = np.array(stats["collaboration_matrices"]["0"])
        expert_counts = stats["summary"]["per_layer_stats"]["0"]["expert_counts"]

        # Header
        print(f"{'':>10}", end="")
        for j in range(num_experts):
            print(f"{'E'+str(j):>10}", end="")
        print()

        # Matrix
        for i in range(num_experts):
            print(f"{'E'+str(i):>10}", end="")
            for j in range(num_experts):
                print(f"{int(matrix[i,j]):>10}", end="")
            print()

        print("\n" + "=" * 70)
        print("Layer 0 Conditional Probabilities P(column | row)")
        print("=" * 70)

        diagonal = np.diag(matrix)
        conditional = matrix / diagonal[:, np.newaxis]

        # Header
        print(f"{'':>10}", end="")
        for j in range(num_experts):
            print(f"{'E'+str(j):>10}", end="")
        print()

        # Matrix
        for i in range(num_experts):
            print(f"{'E'+str(i):>10}", end="")
            for j in range(num_experts):
                if i == j:
                    print(f"{'--':>10}", end="")
                else:
                    print(f"{conditional[i,j]:>10.2%}", end="")
            print()

        # Redundancy summary
        print("\n" + "=" * 70)
        print("Layer 0 Redundancy Analysis")
        print("=" * 70)

        np.fill_diagonal(conditional, 0)
        for i in range(num_experts):
            max_j = np.argmax(conditional[i])
            max_prob = conditional[i, max_j]
            print(f"Expert {i}: Redundancy = {max_prob:.2%} (covered by Expert {max_j})")


if __name__ == "__main__":
    main()
