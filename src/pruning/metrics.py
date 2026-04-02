"""
Pruning metrics for MoE expert selection.

Computes various metrics to identify which experts can be pruned:
1. Utilization - How often each expert is selected
2. Redundancy - How much an expert overlaps with others
3. Graph Centrality - Importance in the co-activation graph
4. Sensitivity - Performance impact of removing each expert (expensive)
"""

import torch
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class PruningMetrics:
    """Container for all pruning metrics for a single layer."""
    layer_idx: int
    num_experts: int

    # Basic metrics
    utilization: np.ndarray = None  # [num_experts] - selection frequency

    # Co-activation based
    redundancy: np.ndarray = None  # [num_experts] - how covered by others
    conditional_coactivation: np.ndarray = None  # [num_experts, num_experts] - P(j|i)

    # Graph centrality
    degree_centrality: np.ndarray = None
    betweenness_centrality: np.ndarray = None
    eigenvector_centrality: np.ndarray = None
    pagerank: np.ndarray = None
    clustering_coefficient: np.ndarray = None

    # Combined scores
    structural_score: np.ndarray = None  # Higher = more prunable

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "layer_idx": self.layer_idx,
            "num_experts": self.num_experts,
        }
        for attr in ["utilization", "redundancy", "degree_centrality",
                     "betweenness_centrality", "eigenvector_centrality",
                     "pagerank", "clustering_coefficient", "structural_score"]:
            val = getattr(self, attr)
            if val is not None:
                result[attr] = val.tolist() if isinstance(val, np.ndarray) else val
        return result

    def get_pruning_candidates(self, top_k: int = 1) -> List[int]:
        """
        Get top-k experts recommended for pruning.

        Uses structural_score if available, otherwise uses inverse utilization.
        """
        if self.structural_score is not None:
            scores = self.structural_score
        elif self.utilization is not None:
            # Lower utilization = more prunable
            scores = 1.0 - self.utilization
        else:
            raise ValueError("No metrics computed yet")

        # Return indices of highest scores (most prunable)
        return np.argsort(scores)[-top_k:].tolist()


def compute_utilization(
    expert_counts: torch.Tensor,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute expert utilization (selection frequency).

    Args:
        expert_counts: Tensor of shape [num_experts] with activation counts
        normalize: Whether to normalize to [0, 1]

    Returns:
        Array of utilization scores per expert
    """
    counts = expert_counts.float().numpy()

    if normalize and counts.sum() > 0:
        counts = counts / counts.sum()

    return counts


def compute_redundancy_scores(
    coactivation_matrix: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute redundancy scores based on co-activation patterns.

    Redundancy(i) = max_j P(j fires | i fires)

    High redundancy means this expert is always accompanied by another,
    making it potentially redundant.

    Args:
        coactivation_matrix: Tensor of shape [num_experts, num_experts]

    Returns:
        Tuple of (redundancy_scores, conditional_coactivation_matrix)
    """
    matrix = coactivation_matrix.float().numpy()
    num_experts = matrix.shape[0]

    # Compute conditional probability P(j | i) = C[i,j] / C[i,i]
    # where C[i,i] is the self-activation count (total times i was selected)
    diagonal = np.diag(matrix)

    # Avoid division by zero
    diagonal = np.maximum(diagonal, 1e-10)

    # Conditional probability matrix
    conditional = matrix / diagonal[:, np.newaxis]

    # Set diagonal to 0 (we don't count self-redundancy)
    np.fill_diagonal(conditional, 0)

    # Redundancy = max conditional probability (excluding self)
    redundancy = np.max(conditional, axis=1)

    return redundancy, conditional


def compute_graph_centrality(
    coactivation_matrix: torch.Tensor,
    include_all: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Compute graph centrality metrics from co-activation matrix.

    Treats the co-activation matrix as a weighted adjacency matrix.

    Args:
        coactivation_matrix: Tensor of shape [num_experts, num_experts]
        include_all: Whether to compute all centrality metrics

    Returns:
        Dictionary with centrality metrics
    """
    matrix = coactivation_matrix.float().numpy()
    num_experts = matrix.shape[0]

    # Create weighted graph (use off-diagonal elements only)
    # Zero out diagonal for graph construction
    adj_matrix = matrix.copy()
    np.fill_diagonal(adj_matrix, 0)

    # Normalize weights to [0, 1] for stability
    if adj_matrix.max() > 0:
        adj_matrix = adj_matrix / adj_matrix.max()

    G = nx.from_numpy_array(adj_matrix)

    results = {}

    # Degree centrality (weighted)
    try:
        degree = nx.degree_centrality(G)
        results["degree_centrality"] = np.array([degree[i] for i in range(num_experts)])
    except Exception as e:
        logger.warning(f"Failed to compute degree centrality: {e}")
        results["degree_centrality"] = np.ones(num_experts) / num_experts

    if include_all:
        # Betweenness centrality
        try:
            betweenness = nx.betweenness_centrality(G, weight="weight")
            results["betweenness_centrality"] = np.array([betweenness[i] for i in range(num_experts)])
        except Exception as e:
            logger.warning(f"Failed to compute betweenness centrality: {e}")
            results["betweenness_centrality"] = np.zeros(num_experts)

        # Eigenvector centrality
        try:
            eigenvector = nx.eigenvector_centrality_numpy(G, weight="weight")
            results["eigenvector_centrality"] = np.array([eigenvector[i] for i in range(num_experts)])
        except Exception as e:
            logger.warning(f"Failed to compute eigenvector centrality: {e}")
            results["eigenvector_centrality"] = np.ones(num_experts) / num_experts

        # PageRank
        try:
            pagerank = nx.pagerank(G, weight="weight")
            results["pagerank"] = np.array([pagerank[i] for i in range(num_experts)])
        except Exception as e:
            logger.warning(f"Failed to compute PageRank: {e}")
            results["pagerank"] = np.ones(num_experts) / num_experts

        # Clustering coefficient
        try:
            clustering = nx.clustering(G, weight="weight")
            results["clustering_coefficient"] = np.array([clustering[i] for i in range(num_experts)])
        except Exception as e:
            logger.warning(f"Failed to compute clustering coefficient: {e}")
            results["clustering_coefficient"] = np.zeros(num_experts)

    return results


def compute_structural_score(
    utilization: np.ndarray,
    redundancy: np.ndarray,
    centrality: np.ndarray,
    alpha: float = 0.3,
    beta: float = 0.4,
    gamma: float = 0.3,
) -> np.ndarray:
    """
    Compute combined structural pruning score.

    Higher score = more suitable for pruning.

    Score = alpha * (1 - utilization) + beta * redundancy + gamma * (1 - centrality)

    Args:
        utilization: Expert utilization scores (normalized)
        redundancy: Redundancy scores
        centrality: Centrality scores (normalized)
        alpha: Weight for low utilization
        beta: Weight for high redundancy
        gamma: Weight for low centrality

    Returns:
        Combined pruning scores
    """
    # Normalize inputs to [0, 1]
    def normalize(x):
        x_min, x_max = x.min(), x.max()
        if x_max - x_min > 1e-10:
            return (x - x_min) / (x_max - x_min)
        return np.zeros_like(x)

    util_norm = normalize(utilization)
    redun_norm = normalize(redundancy)
    cent_norm = normalize(centrality)

    # Higher score = more prunable
    # Low utilization, high redundancy, low centrality
    score = (
        alpha * (1.0 - util_norm) +
        beta * redun_norm +
        gamma * (1.0 - cent_norm)
    )

    return score


def compute_all_metrics(
    stats_collector,
    layer_idx: Optional[int] = None,
    centrality_metric: str = "eigenvector",
) -> Dict[int, PruningMetrics]:
    """
    Compute all pruning metrics from routing statistics.

    Args:
        stats_collector: RoutingStatisticsCollector with collected data
        layer_idx: Specific layer to compute (None = all layers)
        centrality_metric: Which centrality to use for structural score
            ("degree", "betweenness", "eigenvector", "pagerank")

    Returns:
        Dictionary mapping layer_idx to PruningMetrics
    """
    layers = [layer_idx] if layer_idx is not None else range(stats_collector.num_layers)

    results = {}

    for idx in layers:
        metrics = PruningMetrics(
            layer_idx=idx,
            num_experts=stats_collector.num_experts,
        )

        # Get raw data
        expert_counts = stats_collector.expert_counts[idx]
        coactivation = stats_collector.coactivation_counts[idx]

        # Utilization
        metrics.utilization = compute_utilization(expert_counts)

        # Redundancy
        metrics.redundancy, metrics.conditional_coactivation = compute_redundancy_scores(coactivation)

        # Graph centrality
        centrality_metrics = compute_graph_centrality(coactivation)
        metrics.degree_centrality = centrality_metrics.get("degree_centrality")
        metrics.betweenness_centrality = centrality_metrics.get("betweenness_centrality")
        metrics.eigenvector_centrality = centrality_metrics.get("eigenvector_centrality")
        metrics.pagerank = centrality_metrics.get("pagerank")
        metrics.clustering_coefficient = centrality_metrics.get("clustering_coefficient")

        # Select centrality metric for structural score
        centrality_for_score = centrality_metrics.get(
            f"{centrality_metric}_centrality",
            centrality_metrics.get("eigenvector_centrality", metrics.utilization)
        )

        # Combined structural score
        metrics.structural_score = compute_structural_score(
            utilization=metrics.utilization,
            redundancy=metrics.redundancy,
            centrality=centrality_for_score,
        )

        results[idx] = metrics

        logger.info(f"Layer {idx}: computed metrics for {stats_collector.num_experts} experts")

    return results


def print_metrics_summary(metrics: Dict[int, PruningMetrics], top_k: int = 1) -> None:
    """Print a summary of pruning metrics across layers."""
    print("\n" + "=" * 70)
    print("Pruning Metrics Summary")
    print("=" * 70)

    for layer_idx, m in sorted(metrics.items()):
        print(f"\nLayer {layer_idx}:")
        print(f"  Utilization:  {m.utilization}")
        print(f"  Redundancy:   {m.redundancy}")
        print(f"  Eigenvector:  {m.eigenvector_centrality}")
        print(f"  Struct Score: {m.structural_score}")

        candidates = m.get_pruning_candidates(top_k)
        print(f"  Pruning candidates (top {top_k}): {candidates}")


def get_global_pruning_ranking(
    metrics: Dict[int, PruningMetrics],
    strategy: str = "per_layer",
) -> List[Tuple[int, int, float]]:
    """
    Get global ranking of experts to prune across all layers.

    Args:
        metrics: Dictionary of PruningMetrics per layer
        strategy:
            "per_layer" - rank within each layer, interleave
            "global" - rank all experts globally by score

    Returns:
        List of (layer_idx, expert_idx, score) sorted by pruning priority
    """
    all_scores = []

    for layer_idx, m in metrics.items():
        for expert_idx in range(m.num_experts):
            score = m.structural_score[expert_idx]
            all_scores.append((layer_idx, expert_idx, score))

    if strategy == "global":
        # Sort globally by score (highest first = most prunable)
        all_scores.sort(key=lambda x: x[2], reverse=True)
    else:  # per_layer
        # Group by layer, sort within layer, then interleave
        by_layer = {}
        for layer_idx, expert_idx, score in all_scores:
            if layer_idx not in by_layer:
                by_layer[layer_idx] = []
            by_layer[layer_idx].append((expert_idx, score))

        # Sort within each layer
        for layer_idx in by_layer:
            by_layer[layer_idx].sort(key=lambda x: x[1], reverse=True)

        # Interleave: take one from each layer in round-robin
        all_scores = []
        max_experts = max(len(v) for v in by_layer.values())
        for rank in range(max_experts):
            for layer_idx in sorted(by_layer.keys()):
                if rank < len(by_layer[layer_idx]):
                    expert_idx, score = by_layer[layer_idx][rank]
                    all_scores.append((layer_idx, expert_idx, score))

    return all_scores


def load_metrics_from_stats(stats_path: str) -> Dict[int, PruningMetrics]:
    """
    Load routing statistics and compute metrics.

    Args:
        stats_path: Path to coactivation_stats.json (from export_statistics)

    Returns:
        Dictionary of PruningMetrics per layer
    """
    import json

    with open(stats_path, 'r') as f:
        data = json.load(f)

    # Reconstruct RoutingStatisticsCollector-like object
    class StatsProxy:
        def __init__(self, data):
            summary = data.get("summary", data)
            self.num_layers = summary["num_layers"]
            self.num_experts = summary["num_experts"]
            self.expert_counts = {}
            self.coactivation_counts = {}

            # Load expert counts from summary
            per_layer = summary.get("per_layer_stats", {})
            for layer_str, layer_data in per_layer.items():
                layer_idx = int(layer_str)
                self.expert_counts[layer_idx] = torch.tensor(layer_data["expert_counts"])

            # Load co-activation matrices
            collab_matrices = data.get("collaboration_matrices", {})
            for layer_str, matrix in collab_matrices.items():
                layer_idx = int(layer_str)
                self.coactivation_counts[layer_idx] = torch.tensor(matrix)

    proxy = StatsProxy(data)
    return compute_all_metrics(proxy)
