"""
Visualization tools for routing statistics and collaboration matrices.
"""

import torch
import numpy as np
from typing import Optional, List, Dict, Any
from pathlib import Path
import json

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from .statistics import RoutingStatisticsCollector


def plot_collaboration_matrix(
    collector: RoutingStatisticsCollector,
    layer_idx: int,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: tuple = (10, 8),
    cmap: str = "Blues",
    show_values: bool = True,
) -> Optional[Any]:
    """
    Plot the collaboration matrix as a heatmap.

    Args:
        collector: RoutingStatisticsCollector with collected statistics
        layer_idx: Which MoE layer to visualize
        save_path: If provided, save figure to this path
        title: Plot title (auto-generated if None)
        figsize: Figure size
        cmap: Colormap name
        show_values: Whether to show values in cells

    Returns:
        matplotlib figure if available, None otherwise
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return None

    matrix = collector.get_collaboration_matrix(layer_idx, normalize=False)
    matrix_np = matrix.numpy()

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(matrix_np, cmap=cmap, aspect='equal')

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Co-activation Count", rotation=-90, va="bottom")

    # Set ticks
    num_experts = collector.num_experts
    ax.set_xticks(np.arange(num_experts))
    ax.set_yticks(np.arange(num_experts))
    ax.set_xticklabels([f"E{i}" for i in range(num_experts)])
    ax.set_yticklabels([f"E{i}" for i in range(num_experts)])

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add values in cells if requested and matrix is small enough
    if show_values and num_experts <= 16:
        for i in range(num_experts):
            for j in range(num_experts):
                value = int(matrix_np[i, j])
                if value > 0:
                    # Choose text color based on background
                    text_color = "white" if matrix_np[i, j] > matrix_np.max() / 2 else "black"
                    ax.text(j, i, str(value), ha="center", va="center",
                           color=text_color, fontsize=7)

    # Title and labels
    if title is None:
        title = f"Expert Collaboration Matrix - Layer {layer_idx}"
    ax.set_title(title)
    ax.set_xlabel("Expert")
    ax.set_ylabel("Expert")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved collaboration matrix plot to {save_path}")

    return fig


def plot_expert_load(
    collector: RoutingStatisticsCollector,
    layer_idx: int,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: tuple = (12, 5),
) -> Optional[Any]:
    """
    Plot expert load distribution as a bar chart.

    Args:
        collector: RoutingStatisticsCollector with collected statistics
        layer_idx: Which MoE layer to visualize
        save_path: If provided, save figure to this path
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib figure if available, None otherwise
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return None

    counts = collector.expert_counts[layer_idx].numpy()
    num_experts = len(counts)

    fig, ax = plt.subplots(figsize=figsize)

    # Create bar chart
    bars = ax.bar(range(num_experts), counts, color='steelblue', edgecolor='navy')

    # Add ideal line
    ideal = counts.sum() / num_experts
    ax.axhline(y=ideal, color='red', linestyle='--', label=f'Ideal ({ideal:.0f})')

    # Highlight imbalanced experts
    std = counts.std()
    for i, (bar, count) in enumerate(zip(bars, counts)):
        if abs(count - ideal) > std:
            bar.set_color('coral' if count > ideal else 'lightblue')

    # Labels
    ax.set_xlabel("Expert Index")
    ax.set_ylabel("Activation Count")
    ax.set_xticks(range(num_experts))
    ax.set_xticklabels([f"E{i}" for i in range(num_experts)])

    if title is None:
        title = f"Expert Load Distribution - Layer {layer_idx} (std={std:.1f})"
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved expert load plot to {save_path}")

    return fig


def plot_all_layers(
    collector: RoutingStatisticsCollector,
    output_dir: str,
    prefix: str = "",
) -> None:
    """
    Generate and save plots for all layers.

    Args:
        collector: RoutingStatisticsCollector with collected statistics
        output_dir: Directory to save plots
        prefix: Prefix for filenames
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed. Skipping plots.")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for layer_idx in range(collector.num_layers):
        # Collaboration matrix
        collab_path = output_path / f"{prefix}layer_{layer_idx}_collaboration.png"
        plot_collaboration_matrix(collector, layer_idx, save_path=str(collab_path))
        plt.close()

        # Expert load
        load_path = output_path / f"{prefix}layer_{layer_idx}_load.png"
        plot_expert_load(collector, layer_idx, save_path=str(load_path))
        plt.close()

    print(f"Saved plots for {collector.num_layers} layers to {output_dir}")


def print_collaboration_summary(
    collector: RoutingStatisticsCollector,
    top_k: int = 10,
) -> None:
    """
    Print a text summary of collaboration statistics.

    Args:
        collector: RoutingStatisticsCollector with collected statistics
        top_k: Number of top collaborations to show per layer
    """
    print("\n" + "=" * 70)
    print("COLLABORATION ANALYSIS SUMMARY")
    print("=" * 70)

    summary = collector.get_summary()
    print(f"\nModel: {summary['num_layers']} MoE layers, {summary['num_experts']} experts each")

    for layer_idx in range(collector.num_layers):
        print(f"\n{'─' * 70}")
        print(f"Layer {layer_idx}")
        print(f"{'─' * 70}")

        # Expert load stats
        if layer_idx in summary['per_layer_stats']:
            stats = summary['per_layer_stats'][layer_idx]
            print(f"\nExpert Load:")
            print(f"  Total activations: {stats['total_activations']:,}")
            print(f"  Most used:  Expert {stats['most_used_expert']} ({max(stats['expert_counts']):,} activations)")
            print(f"  Least used: Expert {stats['least_used_expert']} ({min(stats['expert_counts']):,} activations)")
            print(f"  Load balance std: {stats['load_balance_std']:.2f}")

        # Top collaborations
        top_collabs = collector.get_top_collaborations(layer_idx, top_k)
        if top_collabs:
            print(f"\nTop {len(top_collabs)} Expert Collaborations:")
            for rank, (exp_i, exp_j, count) in enumerate(top_collabs, 1):
                print(f"  {rank:2d}. Expert {exp_i:2d} + Expert {exp_j:2d}: {count:,} co-activations")

    print("\n" + "=" * 70)


def export_statistics(
    collector: RoutingStatisticsCollector,
    output_path: str,
) -> None:
    """
    Export statistics to JSON file.

    Args:
        collector: RoutingStatisticsCollector with collected statistics
        output_path: Path to save JSON file
    """
    data = {
        "summary": collector.get_summary(),
        "collaboration_matrices": {},
        "top_collaborations": {},
    }

    for layer_idx in range(collector.num_layers):
        # Collaboration matrix (as list of lists)
        matrix = collector.get_collaboration_matrix(layer_idx, normalize=False)
        data["collaboration_matrices"][str(layer_idx)] = matrix.tolist()

        # Top collaborations
        top_collabs = collector.get_top_collaborations(layer_idx, top_k=20)
        data["top_collaborations"][str(layer_idx)] = [
            {"expert_i": i, "expert_j": j, "count": c}
            for i, j, c in top_collabs
        ]

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Exported statistics to {output_path}")
