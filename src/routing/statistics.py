"""
Routing statistics collection for MoE models.

This module provides utilities to capture expert routing decisions
and compute co-activation (collaboration) statistics.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, TYPE_CHECKING
from collections import defaultdict
import logging

if TYPE_CHECKING:
    from src.models import MoETransformer

logger = logging.getLogger(__name__)


class RoutingStatisticsCollector:
    """
    Collects routing statistics from MoE layers during inference.

    Tracks:
    - Per-token expert selections
    - Expert activation counts
    - Expert co-activation (collaboration) matrix
    """

    def __init__(self, num_layers: int, num_experts: int):
        """
        Args:
            num_layers: Number of MoE layers in the model
            num_experts: Number of experts per layer
        """
        self.num_layers = num_layers
        self.num_experts = num_experts

        # Per-layer statistics
        self.expert_counts: Dict[int, torch.Tensor] = {}
        self.coactivation_counts: Dict[int, torch.Tensor] = {}

        # Raw routing decisions for analysis
        self.routing_history: Dict[int, List[torch.Tensor]] = defaultdict(list)

        self.reset()

    def reset(self) -> None:
        """Reset all statistics."""
        for layer_idx in range(self.num_layers):
            self.expert_counts[layer_idx] = torch.zeros(self.num_experts, dtype=torch.long)
            self.coactivation_counts[layer_idx] = torch.zeros(
                (self.num_experts, self.num_experts), dtype=torch.long
            )
        self.routing_history.clear()

    def record_routing(self, layer_idx: int, expert_indices: torch.Tensor) -> None:
        """
        Record routing decisions for a batch.

        Args:
            layer_idx: Index of the MoE layer
            expert_indices: Tensor of shape (num_tokens, top_k) containing
                           selected expert indices for each token
        """
        # Ensure 2D: (num_tokens, top_k)
        if expert_indices.dim() == 3:
            expert_indices = expert_indices.view(-1, expert_indices.shape[-1])

        expert_indices = expert_indices.cpu()

        # Update expert activation counts
        for expert_idx in expert_indices.flatten():
            self.expert_counts[layer_idx][expert_idx.item()] += 1

        # Update co-activation matrix
        # For each token, all selected experts are "co-activated"
        for token_experts in expert_indices:
            experts = token_experts.tolist()
            for i, exp_i in enumerate(experts):
                for exp_j in experts[i:]:
                    self.coactivation_counts[layer_idx][exp_i, exp_j] += 1
                    if exp_i != exp_j:
                        self.coactivation_counts[layer_idx][exp_j, exp_i] += 1

        # Store raw routing for later analysis
        self.routing_history[layer_idx].append(expert_indices)

    def collect_from_model(self, model: "MoETransformer") -> None:
        """
        Collect routing statistics from model after a forward pass.

        Args:
            model: MoETransformer that has just completed a forward pass
        """
        routing_stats = model.get_routing_stats()

        # Map transformer layer indices (0, 2, 4, ...) to sequential MoE indices (0, 1, 2, ...)
        sorted_layer_indices = sorted(routing_stats.keys())

        for moe_idx, transformer_layer_idx in enumerate(sorted_layer_indices):
            stats = routing_stats[transformer_layer_idx]
            expert_indices = stats["expert_indices"]
            self.record_routing(moe_idx, expert_indices)

    def get_collaboration_matrix(
        self, layer_idx: int, normalize: bool = True
    ) -> torch.Tensor:
        """
        Get the expert collaboration matrix for a layer.

        Args:
            layer_idx: Index of the MoE layer
            normalize: If True, normalize to [0, 1] range

        Returns:
            Tensor of shape (num_experts, num_experts) with co-activation counts/frequencies
        """
        matrix = self.coactivation_counts[layer_idx].float()

        if normalize and matrix.sum() > 0:
            # Normalize by total co-activations
            matrix = matrix / matrix.sum()

        return matrix

    def get_expert_load(self, layer_idx: int, normalize: bool = True) -> torch.Tensor:
        """
        Get expert load distribution for a layer.

        Args:
            layer_idx: Index of the MoE layer
            normalize: If True, return as fractions summing to 1

        Returns:
            Tensor of shape (num_experts,) with activation counts/frequencies
        """
        counts = self.expert_counts[layer_idx].float()

        if normalize and counts.sum() > 0:
            counts = counts / counts.sum()

        return counts

    def get_summary(self) -> Dict:
        """Get summary statistics across all layers."""
        summary = {
            "num_layers": self.num_layers,
            "num_experts": self.num_experts,
            "per_layer_stats": {},
        }

        for layer_idx in range(self.num_layers):
            layer_counts = self.expert_counts[layer_idx]
            total_activations = layer_counts.sum().item()

            if total_activations > 0:
                summary["per_layer_stats"][layer_idx] = {
                    "total_activations": int(total_activations),
                    "most_used_expert": int(layer_counts.argmax().item()),
                    "least_used_expert": int(layer_counts.argmin().item()),
                    "load_balance_std": float(layer_counts.float().std().item()),
                    "expert_counts": layer_counts.tolist(),
                }

        return summary

    def get_top_collaborations(
        self, layer_idx: int, top_k: int = 10
    ) -> List[tuple]:
        """
        Get the top-k most frequent expert collaborations for a layer.

        Args:
            layer_idx: Index of the MoE layer
            top_k: Number of top collaborations to return

        Returns:
            List of (expert_i, expert_j, count) tuples sorted by count descending
        """
        matrix = self.coactivation_counts[layer_idx]
        collaborations = []

        for i in range(self.num_experts):
            for j in range(i + 1, self.num_experts):  # Upper triangle only
                count = matrix[i, j].item()
                if count > 0:
                    collaborations.append((i, j, count))

        # Sort by count descending
        collaborations.sort(key=lambda x: x[2], reverse=True)

        return collaborations[:top_k]


def create_collector_for_model(model: "MoETransformer") -> RoutingStatisticsCollector:
    """
    Create a RoutingStatisticsCollector configured for a specific model.

    Args:
        model: MoETransformer instance

    Returns:
        Configured RoutingStatisticsCollector
    """
    num_moe_layers = model.num_moe_layers
    num_experts = model.config.num_experts

    return RoutingStatisticsCollector(num_moe_layers, num_experts)
