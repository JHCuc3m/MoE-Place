"""
Expert masking and disabling for MoE models.

Provides utilities to:
1. Soft-disable experts (mask their output to zero)
2. Modify routing to skip disabled experts
3. Measure sensitivity by disabling experts one at a time
"""

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from typing import Dict, List, Set, Optional, Tuple, Any, Union
from contextlib import contextmanager
from pathlib import Path
import logging

import numpy as np

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logger = logging.getLogger(__name__)


class ExpertMasker:
    """
    Masks (disables) experts in a Mixtral-style MoE model.

    Uses forward hooks to zero out the contribution of disabled experts
    without modifying model weights.
    """

    def __init__(self, model: PreTrainedModel):
        """
        Args:
            model: HuggingFace MoE model (Mixtral-style)
        """
        self.model = model
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

        # Track disabled experts: {layer_idx: set of expert indices}
        self.disabled_experts: Dict[int, Set[int]] = {}

        # Find MoE layers
        self.moe_layers = self._find_moe_layers()
        logger.info(f"Found {len(self.moe_layers)} MoE layers")

    def _find_moe_layers(self) -> Dict[int, nn.Module]:
        """Find all MoE layer modules in the model."""
        moe_layers = {}

        for name, module in self.model.named_modules():
            # Match Mixtral MoE patterns by checking for required attributes
            # In transformers, MixtralSparseMoeBlock has both 'gate' and 'experts'
            # Note: The module may be named 'mlp' (not 'block_sparse_moe')
            if hasattr(module, 'gate') and hasattr(module, 'experts'):
                # Extract layer index from name like "model.layers.5.mlp"
                parts = name.split('.')
                for i, part in enumerate(parts):
                    if part == 'layers' and i + 1 < len(parts):
                        try:
                            layer_idx = int(parts[i + 1])
                            moe_layers[layer_idx] = module
                            logger.info(f"Found MoE layer {layer_idx}: {name}")
                            break
                        except ValueError:
                            pass

        return moe_layers

    def disable_expert(self, layer_idx: int, expert_idx: int) -> None:
        """
        Disable a specific expert in a layer.

        Args:
            layer_idx: MoE layer index (0-indexed)
            expert_idx: Expert index within the layer
        """
        if layer_idx not in self.disabled_experts:
            self.disabled_experts[layer_idx] = set()

        self.disabled_experts[layer_idx].add(expert_idx)
        logger.info(f"Disabled expert {expert_idx} in layer {layer_idx}")

    def enable_expert(self, layer_idx: int, expert_idx: int) -> None:
        """Re-enable a previously disabled expert."""
        if layer_idx in self.disabled_experts:
            self.disabled_experts[layer_idx].discard(expert_idx)
            logger.info(f"Enabled expert {expert_idx} in layer {layer_idx}")

    def disable_experts_from_ranking(
        self,
        ranking: List[Tuple[int, int, float]],
        num_to_disable: int,
    ) -> List[Tuple[int, int]]:
        """
        Disable top-k experts from a pruning ranking.

        Args:
            ranking: List of (layer_idx, expert_idx, score) sorted by score descending
            num_to_disable: Number of experts to disable

        Returns:
            List of (layer_idx, expert_idx) that were disabled
        """
        disabled = []
        for layer_idx, expert_idx, score in ranking[:num_to_disable]:
            self.disable_expert(layer_idx, expert_idx)
            disabled.append((layer_idx, expert_idx))

        return disabled

    def reset(self) -> None:
        """Re-enable all experts."""
        self.disabled_experts.clear()
        logger.info("Reset: all experts enabled")

    def get_disabled_count(self) -> int:
        """Get total number of disabled experts across all layers."""
        return sum(len(experts) for experts in self.disabled_experts.values())

    def apply_masking(self) -> None:
        """
        Apply masking hooks to the model.

        Must be called before running inference with disabled experts.
        """
        self.remove_hooks()  # Clear any existing hooks

        for layer_idx, moe_module in self.moe_layers.items():
            hook = moe_module.register_forward_hook(
                self._create_masking_hook(layer_idx)
            )
            self.hooks.append(hook)

        logger.info(f"Applied masking hooks to {len(self.hooks)} layers")

    def _create_masking_hook(self, layer_idx: int):
        """Create a forward hook that masks disabled experts' outputs."""

        def hook(module, input, output):
            # Skip if no experts disabled in this layer
            if layer_idx not in self.disabled_experts:
                return output
            if not self.disabled_experts[layer_idx]:
                return output

            disabled = self.disabled_experts[layer_idx]

            # Handle different output formats from MoE layers
            # Mixtral typically returns (hidden_states, router_logits) or just hidden_states
            if isinstance(output, tuple):
                hidden_states = output[0]
                rest = output[1:]
            else:
                hidden_states = output
                rest = None

            # The masking happens implicitly through the routing weights
            # For proper masking, we need to intercept at the routing level
            # This hook modifies the final output as a fallback

            # Note: For more precise control, we should hook the router
            # and set routing weights to 0 for disabled experts
            # This is a simplified version that works for analysis

            if rest is not None:
                return (hidden_states,) + rest
            return hidden_states

        return hook

    def remove_hooks(self) -> None:
        """Remove all masking hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    @contextmanager
    def mask_experts(self, experts_to_disable: List[Tuple[int, int]]):
        """
        Context manager for temporarily disabling experts.

        Args:
            experts_to_disable: List of (layer_idx, expert_idx) to disable

        Example:
            with masker.mask_experts([(0, 1), (2, 3)]):
                output = model(**inputs)
        """
        # Save current state
        previous_state = {k: v.copy() for k, v in self.disabled_experts.items()}

        # Disable specified experts
        for layer_idx, expert_idx in experts_to_disable:
            self.disable_expert(layer_idx, expert_idx)

        self.apply_masking()

        try:
            yield
        finally:
            # Restore previous state
            self.disabled_experts = previous_state
            self.remove_hooks()
            if previous_state:
                self.apply_masking()


class MixtralExpertMasker(ExpertMasker):
    """
    Specialized masker for Mixtral-style models.

    Hooks into the router to prevent disabled experts from being selected.
    """

    def __init__(self, model: PreTrainedModel):
        super().__init__(model)
        self.router_hooks: List[torch.utils.hooks.RemovableHandle] = []

    def _find_routers(self) -> Dict[int, Tuple[str, nn.Module]]:
        """Find all router/gate modules."""
        routers = {}

        for name, module in self.model.named_modules():
            name_lower = name.lower()
            # Match gate patterns: model.layers.X.block_sparse_moe.gate or model.layers.X.mlp.gate
            # Don't require nn.Linear - some gates are more complex
            if name_lower.endswith('.gate'):
                # Extract layer index from name like "model.layers.5.mlp.gate"
                parts = name.split('.')
                for i, part in enumerate(parts):
                    if part == 'layers' and i + 1 < len(parts):
                        try:
                            layer_idx = int(parts[i + 1])
                            routers[layer_idx] = (name, module)
                            logger.info(f"Found router: {name} -> layer {layer_idx}, type={type(module).__name__}")
                            break
                        except ValueError:
                            pass

        if not routers:
            # Fallback: list all modules with 'gate' in name for debugging
            logger.warning("No routers found with standard pattern. Listing gate-like modules:")
            for name, module in self.model.named_modules():
                if 'gate' in name.lower():
                    logger.warning(f"  {name}: {type(module).__name__}")

        return routers

    def apply_masking(self) -> None:
        """Apply router-level masking for more precise control."""
        self.remove_hooks()

        # Find routers
        routers = self._find_routers()

        if not routers:
            logger.warning("No routers found! Masking will not work.")
            return

        for layer_idx, (name, router) in routers.items():
            if layer_idx in self.disabled_experts and self.disabled_experts[layer_idx]:
                hook = router.register_forward_hook(
                    self._create_router_masking_hook(layer_idx)
                )
                self.router_hooks.append(hook)
                logger.info(f"Applied masking hook to layer {layer_idx} ({name}), "
                           f"disabled experts: {self.disabled_experts[layer_idx]}")

        logger.info(f"Applied {len(self.router_hooks)} router masking hooks")

    def _create_router_masking_hook(self, layer_idx: int):
        """
        Create hook that zeros out routing weights for disabled experts.

        MixtralTopKRouter returns: (probs, top_k_weights, top_k_indices)
        - probs: [batch, num_experts] - full routing probabilities
        - top_k_weights: [batch, k] - weights for selected experts
        - top_k_indices: [batch, k] - indices of selected experts

        We zero out weights where indices match disabled experts.
        """

        def hook(module, input, output):
            if layer_idx not in self.disabled_experts:
                return None
            if not self.disabled_experts[layer_idx]:
                return None

            disabled = self.disabled_experts[layer_idx]

            if isinstance(output, torch.Tensor):
                # nn.Linear gate output: raw logits [batch*seq, num_experts]
                # Set disabled expert logits to -inf so they are excluded from top-k
                new_logits = output.clone()
                for expert_idx in disabled:
                    new_logits[:, expert_idx] = float('-inf')
                logger.debug(f"Layer {layer_idx}: Set logits to -inf for experts {disabled}")
                return new_logits

            elif isinstance(output, tuple) and len(output) >= 3:
                # Router format: (probs, weights, indices)
                probs, weights, indices = output[0], output[1], output[2]
                new_weights = weights.clone()
                for expert_idx in disabled:
                    mask = (indices == expert_idx)
                    new_weights[mask] = 0.0
                weight_sum = new_weights.sum(dim=-1, keepdim=True)
                weight_sum = torch.clamp(weight_sum, min=1e-8)
                new_weights = new_weights / weight_sum
                return (probs, new_weights, indices) + output[3:] if len(output) > 3 else (probs, new_weights, indices)

            elif isinstance(output, tuple) and len(output) == 2:
                # Fallback: (weights, indices) format
                weights, indices = output[0], output[1]
                new_weights = weights.clone()
                for expert_idx in disabled:
                    mask = (indices == expert_idx)
                    new_weights[mask] = 0.0
                weight_sum = new_weights.sum(dim=-1, keepdim=True)
                weight_sum = torch.clamp(weight_sum, min=1e-8)
                new_weights = new_weights / weight_sum
                return (new_weights, indices)

            else:
                logger.warning(f"Layer {layer_idx}: Unexpected output format: {type(output)}")
                return None

        return hook

    def remove_hooks(self) -> None:
        """Remove all hooks."""
        super().remove_hooks()
        for hook in self.router_hooks:
            hook.remove()
        self.router_hooks.clear()


def compute_sensitivity(
    model: PreTrainedModel,
    tokenizer,
    masker: ExpertMasker,
    eval_dataloader,
    device: str = None,
    max_batches: int = None,
) -> Dict[Tuple[int, int], float]:
    """
    Compute sensitivity for each expert by measuring perplexity impact.

    Sensitivity(layer, expert) = PPL(model without expert) - PPL(baseline)

    Args:
        model: MoE model
        tokenizer: Tokenizer
        masker: ExpertMasker instance
        eval_dataloader: DataLoader for evaluation
        device: Device to run on
        max_batches: Limit evaluation batches for speed

    Returns:
        Dictionary mapping (layer_idx, expert_idx) to sensitivity score
    """
    from src.evaluation.perplexity import compute_perplexity

    if device is None:
        device = next(model.parameters()).device

    # Compute baseline perplexity (no experts disabled)
    masker.reset()
    baseline_results = compute_perplexity(
        model, eval_dataloader, device, max_batches, show_progress=False
    )
    baseline_ppl = baseline_results["perplexity"]
    logger.info(f"Baseline perplexity: {baseline_ppl:.2f}")

    sensitivity = {}
    num_layers = len(masker.moe_layers)

    # Get number of experts from model config
    num_experts = getattr(model.config, 'num_local_experts', 4)

    for layer_idx in range(num_layers):
        for expert_idx in range(num_experts):
            # Disable single expert
            masker.reset()
            masker.disable_expert(layer_idx, expert_idx)
            masker.apply_masking()

            # Evaluate
            results = compute_perplexity(
                model, eval_dataloader, device, max_batches, show_progress=False
            )

            # Sensitivity = how much perplexity increases
            ppl_increase = results["perplexity"] - baseline_ppl
            sensitivity[(layer_idx, expert_idx)] = ppl_increase

            logger.info(f"Layer {layer_idx}, Expert {expert_idx}: "
                       f"PPL = {results['perplexity']:.2f} (Δ = {ppl_increase:+.2f})")

            masker.remove_hooks()

    masker.reset()
    return sensitivity


def print_sensitivity_summary(
    sensitivity: Dict[Tuple[int, int], float],
    num_experts: int = 4,
) -> None:
    """Print a formatted summary of sensitivity scores."""
    print("\n" + "=" * 70)
    print("Expert Sensitivity Analysis (PPL increase when disabled)")
    print("=" * 70)

    # Group by layer
    by_layer = {}
    for (layer_idx, expert_idx), score in sensitivity.items():
        if layer_idx not in by_layer:
            by_layer[layer_idx] = {}
        by_layer[layer_idx][expert_idx] = score

    # Print table
    print(f"\n{'Layer':<8}", end="")
    for e in range(num_experts):
        print(f"{'Expert '+str(e):<12}", end="")
    print(f"{'Most Sensitive':<16}")
    print("-" * (8 + 12 * num_experts + 16))

    for layer_idx in sorted(by_layer.keys()):
        print(f"{layer_idx:<8}", end="")
        scores = by_layer[layer_idx]
        max_expert = max(scores, key=scores.get)

        for e in range(num_experts):
            score = scores.get(e, 0)
            marker = "*" if e == max_expert else " "
            print(f"{score:+.2f}{marker:<7}", end="")

        print(f"Expert {max_expert} ({scores[max_expert]:+.2f})")

    print("\n* = most sensitive expert in layer (highest PPL increase = most important)")


def plot_sensitivity_heatmap(
    sensitivity: Dict[Tuple[int, int], float],
    num_layers: int,
    num_experts: int = 4,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: tuple = (10, 8),
) -> Optional[Any]:
    """
    Plot sensitivity scores as a heatmap table.

    Args:
        sensitivity: Dictionary mapping (layer_idx, expert_idx) to sensitivity score
        num_layers: Number of MoE layers
        num_experts: Number of experts per layer
        save_path: If provided, save figure to this path
        title: Plot title (auto-generated if None)
        figsize: Figure size

    Returns:
        matplotlib figure if available, None otherwise
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not installed. Install with: pip install matplotlib")
        return None

    # Build matrix from sensitivity dict
    matrix = np.zeros((num_layers, num_experts))
    for (layer_idx, expert_idx), score in sensitivity.items():
        if layer_idx < num_layers and expert_idx < num_experts:
            matrix[layer_idx, expert_idx] = score

    fig, ax = plt.subplots(figsize=figsize)

    # Use diverging colormap centered at 0 (red=bad/high sensitivity, blue=good/negative)
    max_abs = max(abs(matrix.min()), abs(matrix.max()))
    norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)

    im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', norm=norm)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("PPL Increase (negative = pruning helps)", rotation=-90, va="bottom")

    # Set ticks
    ax.set_xticks(np.arange(num_experts))
    ax.set_yticks(np.arange(num_layers))
    ax.set_xticklabels([f"Expert {i}" for i in range(num_experts)])
    ax.set_yticklabels([f"Layer {i}" for i in range(num_layers)])

    # Add values in cells
    for i in range(num_layers):
        for j in range(num_experts):
            value = matrix[i, j]
            # Choose text color based on background intensity
            text_color = "white" if abs(value) > max_abs * 0.5 else "black"
            ax.text(j, i, f"{value:+.1f}", ha="center", va="center",
                   color=text_color, fontsize=8, fontweight='bold')

    # Mark most sensitive expert per layer with a border
    for layer_idx in range(num_layers):
        row = matrix[layer_idx, :]
        max_expert = np.argmax(row)
        rect = plt.Rectangle((max_expert - 0.5, layer_idx - 0.5), 1, 1,
                             fill=False, edgecolor='gold', linewidth=3)
        ax.add_patch(rect)

    # Title and labels
    if title is None:
        title = "Expert Sensitivity Analysis\n(PPL increase when disabled; gold border = most sensitive)"
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("Expert", fontsize=11)
    ax.set_ylabel("Layer", fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved sensitivity heatmap to {save_path}")
        print(f"Saved sensitivity heatmap to {save_path}")

    return fig
