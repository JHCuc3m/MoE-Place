"""
Expert contribution metrics for MoE models.

This module computes contribution-based metrics that measure
the actual impact of each expert on the model output, rather than
just selection patterns (which structural metrics do).

Key Metrics:
- C_i (Magnitude): |G_i(x) · E_i(x)|_2 / |h_post(x)|_2
- S_i (Signed Contribution): ⟨G_i(x) · E_i(x), h_post(x)⟩ / |h_post(x)|_2

These metrics aim to identify harmful experts (negative S_i) which
structural metrics fail to detect.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from typing import Dict, List, Tuple, Optional, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)


class ExpertContributionCollector:
    """
    Collects expert contribution metrics from MoE layers.

    Hooks into MoE layers to capture:
    - E_i(x): Raw expert output (before gating)
    - G_i(x): Gate weight for expert i
    - h_post(x): MoE layer output (post-residual)
    - Expert selection indices

    Computes:
    - C_i: Output contribution magnitude
    - S_i: Signed contribution (cosine-like alignment)

    Supports:
    - Mixtral-style models (num_local_experts, top-k routing)
    - DeepSeek-style models (n_routed_experts + n_shared_experts)
    """

    def __init__(self, model: PreTrainedModel):
        """
        Args:
            model: HuggingFace MoE model (Mixtral or DeepSeek style)
        """
        self.model = model
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

        # Extract model config - handle different model families
        self.num_layers = model.config.num_hidden_layers
        self.model_type = getattr(model.config, 'model_type', 'unknown')

        # Handle different config attribute names
        # Mixtral: num_local_experts, DeepSeek: n_routed_experts
        self.num_routed_experts = (
            getattr(model.config, 'num_local_experts', None) or
            getattr(model.config, 'n_routed_experts', None) or
            4
        )
        self.num_shared_experts = getattr(model.config, 'n_shared_experts', 0)
        self.num_experts = self.num_routed_experts  # For backward compatibility

        self.num_experts_per_tok = getattr(model.config, 'num_experts_per_tok', 2)

        # DeepSeek-specific: first_k_dense_replace means first k layers are dense
        self.first_k_dense = getattr(model.config, 'first_k_dense_replace', 0)

        logger.info(f"Model type: {self.model_type}")
        logger.info(f"Routed experts: {self.num_routed_experts}, Shared experts: {self.num_shared_experts}")
        logger.info(f"Experts per token: {self.num_experts_per_tok}")
        if self.first_k_dense > 0:
            logger.info(f"First {self.first_k_dense} layers are dense (not MoE)")

        # Find MoE layers
        self.moe_layers = self._find_moe_layers()
        logger.info(f"Found {len(self.moe_layers)} MoE layers")

        # Storage for collected data per layer
        # Key: layer_idx, Value: list of dicts with contribution data
        self.layer_data: Dict[int, List[Dict[str, Any]]] = {
            i: [] for i in range(self.num_layers)
        }

        # Aggregated metrics
        # Key: (layer_idx, expert_idx), Value: list of contribution values
        self.contributions_magnitude: Dict[Tuple[int, int], List[float]] = {}
        self.contributions_signed: Dict[Tuple[int, int], List[float]] = {}
        self.expert_counts: Dict[Tuple[int, int], int] = {}

        self._init_metric_storage()

    def _init_metric_storage(self):
        """Initialize storage for all (layer, expert) pairs."""
        # Initialize for routed experts
        for layer_idx in range(self.num_layers):
            for expert_idx in range(self.num_routed_experts):
                key = (layer_idx, expert_idx)
                self.contributions_magnitude[key] = []
                self.contributions_signed[key] = []
                self.expert_counts[key] = 0

        # Initialize for shared experts (if any)
        # Use negative indices for shared experts: -1, -2, etc.
        if self.num_shared_experts > 0:
            for layer_idx in range(self.num_layers):
                for shared_idx in range(self.num_shared_experts):
                    key = (layer_idx, -(shared_idx + 1))  # -1, -2, etc.
                    self.contributions_magnitude[key] = []
                    self.contributions_signed[key] = []
                    self.expert_counts[key] = 0

    def _find_moe_layers(self) -> Dict[int, nn.Module]:
        """Find all MoE layer modules in the model."""
        moe_layers = {}

        for name, module in self.model.named_modules():
            # Match different MoE architectures:
            # - Mixtral: MixtralSparseMoeBlock has 'gate' and 'experts'
            # - DeepSeek: DeepseekMoE has 'gate' and 'experts' (may also have 'shared_experts')
            is_moe_layer = (
                (hasattr(module, 'gate') and hasattr(module, 'experts')) or
                ('moe' in module.__class__.__name__.lower() and hasattr(module, 'experts'))
            )

            if is_moe_layer:
                # Extract layer index from name like "model.layers.5.mlp"
                parts = name.split('.')
                for i, part in enumerate(parts):
                    if part == 'layers' and i + 1 < len(parts):
                        try:
                            layer_idx = int(parts[i + 1])
                            # Check if this layer should be skipped (dense layer in DeepSeek)
                            if layer_idx < self.first_k_dense:
                                logger.info(f"Skipping dense layer {layer_idx}: {name}")
                                continue
                            moe_layers[layer_idx] = module
                            # Log what we found
                            has_shared = hasattr(module, 'shared_experts')
                            logger.info(
                                f"Found MoE layer {layer_idx}: {name} "
                                f"(shared_experts: {has_shared})"
                            )
                            break
                        except ValueError:
                            pass

        return moe_layers

    def attach_hooks(self) -> None:
        """Attach forward hooks to MoE layers to capture contributions."""
        self.remove_hooks()

        for layer_idx, moe_module in self.moe_layers.items():
            # We need to capture intermediate values during the MoE forward pass
            # This requires hooking the MoE module itself and using a pre-hook
            # to access inputs and a regular hook to access outputs

            # Pre-hook to capture the input (hidden states)
            pre_hook = moe_module.register_forward_pre_hook(
                self._create_pre_hook(layer_idx)
            )
            self.hooks.append(pre_hook)

            # Post-hook to capture the output
            post_hook = moe_module.register_forward_hook(
                self._create_post_hook(layer_idx)
            )
            self.hooks.append(post_hook)

            # Also hook the gate to capture routing weights
            gate_hook = moe_module.gate.register_forward_hook(
                self._create_gate_hook(layer_idx)
            )
            self.hooks.append(gate_hook)

        logger.info(f"Attached contribution hooks to {len(self.moe_layers)} MoE layers")

    def _create_pre_hook(self, layer_idx: int):
        """Create pre-hook to capture MoE input (hidden states)."""
        def hook(module, input):
            # input is a tuple, first element is hidden_states
            if isinstance(input, tuple) and len(input) > 0:
                hidden_states = input[0]
                if layer_idx not in self.layer_data:
                    self.layer_data[layer_idx] = []

                # Store input for this forward pass
                self.layer_data[layer_idx].append({
                    'input': hidden_states.detach(),
                })
        return hook

    def _create_gate_hook(self, layer_idx: int):
        """Create hook to capture gate/router outputs."""
        def hook(module, input, output):
            if layer_idx not in self.layer_data or not self.layer_data[layer_idx]:
                return

            # Gate output format varies:
            # - Some gates return just logits [batch*seq, num_experts]
            # - MixtralTopKRouter returns (routing_weights, expert_indices) or (probs, weights, indices)

            if isinstance(output, tuple):
                if len(output) >= 3:
                    # (probs, weights, indices) format
                    routing_probs = output[0]
                    routing_weights = output[1]
                    expert_indices = output[2]
                elif len(output) == 2:
                    # (weights, indices) format
                    routing_weights = output[0]
                    expert_indices = output[1]
                    routing_probs = None
                else:
                    routing_weights = None
                    expert_indices = None
                    routing_probs = None
            else:
                # Just logits, compute top-k ourselves
                logits = output
                _, expert_indices = torch.topk(
                    logits, self.num_experts_per_tok, dim=-1
                )
                routing_weights = torch.softmax(logits, dim=-1)
                routing_weights = routing_weights.gather(-1, expert_indices)
                routing_probs = None

            # Update the current forward pass data
            if self.layer_data[layer_idx]:
                current = self.layer_data[layer_idx][-1]
                current['routing_weights'] = routing_weights.detach() if routing_weights is not None else None
                current['expert_indices'] = expert_indices.detach() if expert_indices is not None else None
                current['routing_probs'] = routing_probs.detach() if routing_probs is not None else None

        return hook

    def _create_post_hook(self, layer_idx: int):
        """Create post-hook to capture MoE output and compute contributions."""
        def hook(module, input, output):
            if layer_idx not in self.layer_data or not self.layer_data[layer_idx]:
                return

            current = self.layer_data[layer_idx][-1]

            # Handle output format (could be tuple or tensor)
            if isinstance(output, tuple):
                h_post = output[0]
            else:
                h_post = output

            current['output'] = h_post.detach()

            # Now compute per-expert contributions
            self._compute_contributions(layer_idx, module, current)

        return hook

    def _compute_expert_output(
        self,
        moe_module: nn.Module,
        expert_idx: int,
        inputs: torch.Tensor,
        is_shared: bool = False
    ) -> torch.Tensor:
        """
        Compute expert output E_i(x) for given inputs.

        Handles different expert architectures:
        - Mixtral: Fused gate_up_proj, separate down_proj
        - DeepSeek: Separate gate_proj, up_proj, down_proj (ModuleList)

        Args:
            moe_module: The MoE module containing experts
            expert_idx: Index of the expert
            inputs: Input tensor [num_tokens, hidden_dim]
            is_shared: Whether this is a shared expert

        Returns:
            Expert output [num_tokens, hidden_dim]
        """
        with torch.no_grad():
            if is_shared and hasattr(moe_module, 'shared_experts'):
                # DeepSeek shared experts
                experts = moe_module.shared_experts
                if isinstance(experts, nn.ModuleList):
                    expert = experts[expert_idx]
                    # Individual expert module with gate_proj, up_proj, down_proj
                    gate = F.linear(inputs, expert.gate_proj.weight)
                    up = F.linear(inputs, expert.up_proj.weight)
                    # Get activation function
                    act_fn = getattr(expert, 'act_fn', None) or F.silu
                    if callable(act_fn):
                        hidden = act_fn(gate) * up
                    else:
                        hidden = F.silu(gate) * up
                    E_i = F.linear(hidden, expert.down_proj.weight)
                else:
                    # Fused shared experts (like Mixtral)
                    gate_up = F.linear(inputs, experts.gate_up_proj[expert_idx])
                    gate, up = gate_up.chunk(2, dim=-1)
                    act_fn = getattr(experts, 'act_fn', F.silu)
                    hidden = act_fn(gate) * up
                    E_i = F.linear(hidden, experts.down_proj[expert_idx])
            else:
                # Routed experts
                experts = moe_module.experts

                if isinstance(experts, nn.ModuleList):
                    expert = experts[expert_idx]
                    if hasattr(expert, 'w1'):
                        # Mixtral style: w1 (gate+SiLU), w2 (down), w3 (up)
                        hidden = F.silu(expert.w1(inputs)) * expert.w3(inputs)
                        E_i = expert.w2(hidden)
                    elif hasattr(expert, 'gate_proj'):
                        # DeepSeek style: gate_proj, up_proj, down_proj
                        gate = F.linear(inputs, expert.gate_proj.weight)
                        up = F.linear(inputs, expert.up_proj.weight)
                        act_fn = getattr(expert, 'act_fn', None) or F.silu
                        hidden = act_fn(gate) * up if callable(act_fn) else F.silu(gate) * up
                        E_i = F.linear(hidden, expert.down_proj.weight)
                    else:
                        raise ValueError(f"Unknown expert module attributes: {list(expert._modules.keys())}")
                elif hasattr(experts, 'gate_up_proj'):
                    # Fused tensor style
                    gate_up = F.linear(inputs, experts.gate_up_proj[expert_idx])
                    gate, up = gate_up.chunk(2, dim=-1)
                    act_fn = getattr(experts, 'act_fn', F.silu)
                    hidden = act_fn(gate) * up
                    E_i = F.linear(hidden, experts.down_proj[expert_idx])
                else:
                    raise ValueError(f"Unknown expert structure: {type(experts)}")

        return E_i

    def _compute_contributions(
        self,
        layer_idx: int,
        moe_module: nn.Module,
        data: Dict[str, Any]
    ) -> None:
        """
        Compute contribution metrics for each expert in this forward pass.

        For each token x where expert i is selected:
        - C_i = |G_i(x) · E_i(x)|_2 / |h_post(x)|_2
        - S_i = ⟨G_i(x) · E_i(x), h_post(x)⟩ / |h_post(x)|_2
        """
        if 'routing_weights' not in data or data['routing_weights'] is None:
            logger.warning(f"Layer {layer_idx}: No routing weights captured")
            return
        if 'expert_indices' not in data or data['expert_indices'] is None:
            logger.warning(f"Layer {layer_idx}: No expert indices captured")
            return
        if 'output' not in data:
            logger.warning(f"Layer {layer_idx}: No output captured")
            return
        if 'input' not in data:
            logger.warning(f"Layer {layer_idx}: No input captured")
            return

        h_in = data['input']  # [batch, seq, hidden]
        h_post = data['output']  # [batch, seq, hidden]
        routing_weights = data['routing_weights']  # [batch*seq, k]
        expert_indices = data['expert_indices']  # [batch*seq, k]

        batch_size, seq_len, hidden_size = h_in.shape

        # Flatten h_in and h_post for per-token processing
        h_in_flat = h_in.view(-1, hidden_size)  # [batch*seq, hidden]
        h_post_flat = h_post.view(-1, hidden_size)  # [batch*seq, hidden]

        # Compute |h_post(x)|_2 for normalization
        h_post_norm = torch.norm(h_post_flat, dim=-1, keepdim=True)  # [batch*seq, 1]
        h_post_norm = h_post_norm.clamp(min=1e-8)  # Avoid division by zero

        num_tokens = h_in_flat.shape[0]

        # For each token, compute expert contributions
        # routing_weights and expert_indices are [batch*seq, k] where k = num_experts_per_tok

        # Collect which tokens go to which expert
        expert_token_map = {e: [] for e in range(self.num_routed_experts)}
        expert_weight_map = {e: [] for e in range(self.num_routed_experts)}

        for tok_idx in range(num_tokens):
            for k in range(routing_weights.shape[1]):
                expert_idx = expert_indices[tok_idx, k].item()
                if expert_idx < self.num_routed_experts:  # Safety check
                    weight = routing_weights[tok_idx, k].item()
                    expert_token_map[expert_idx].append(tok_idx)
                    expert_weight_map[expert_idx].append(weight)

        # Compute contributions for routed experts
        for expert_idx in range(self.num_routed_experts):
            token_indices = expert_token_map[expert_idx]
            weights = expert_weight_map[expert_idx]

            if not token_indices:
                continue

            # Get input tokens for this expert
            token_indices_t = torch.tensor(token_indices, device=h_in_flat.device)
            expert_inputs = h_in_flat[token_indices_t]  # [num_tokens_for_expert, hidden]
            expert_outputs = h_post_flat[token_indices_t]  # [num_tokens_for_expert, hidden]

            # Compute E_i(x) using the appropriate expert structure
            E_i = self._compute_expert_output(moe_module, expert_idx, expert_inputs, is_shared=False)

            # Get h_post norm for these tokens
            h_post_norm_subset = h_post_norm[token_indices_t]  # [num_tokens_for_expert, 1]

            # Compute G_i(x) * E_i(x)
            weights_t = torch.tensor(weights, device=E_i.device, dtype=E_i.dtype).unsqueeze(1)
            weighted_expert = weights_t * E_i  # [num_tokens, hidden]

            # Metric 1: C_i (Magnitude)
            magnitude = torch.norm(weighted_expert, dim=-1, keepdim=True) / h_post_norm_subset
            magnitude_vals = magnitude.squeeze(-1).cpu().tolist()

            # Metric 2: S_i (Signed Contribution)
            dot_product = (weighted_expert * expert_outputs).sum(dim=-1, keepdim=True)
            signed_contrib = dot_product / h_post_norm_subset
            signed_vals = signed_contrib.squeeze(-1).cpu().tolist()

            # Store contributions
            key = (layer_idx, expert_idx)
            self.contributions_magnitude[key].extend(magnitude_vals)
            self.contributions_signed[key].extend(signed_vals)
            self.expert_counts[key] += len(token_indices)

        # Compute contributions for shared experts (if any)
        # Shared experts process ALL tokens with weight=1
        if self.num_shared_experts > 0 and hasattr(moe_module, 'shared_experts'):
            all_token_indices = list(range(num_tokens))
            token_indices_t = torch.tensor(all_token_indices, device=h_in_flat.device)

            for shared_idx in range(self.num_shared_experts):
                # Compute shared expert output (no routing weight, always 1.0)
                E_i = self._compute_expert_output(
                    moe_module, shared_idx, h_in_flat, is_shared=True
                )

                # For shared experts, weight is implicitly 1.0
                weighted_expert = E_i  # No weighting needed

                # Metric 1: C_i (Magnitude)
                magnitude = torch.norm(weighted_expert, dim=-1, keepdim=True) / h_post_norm
                magnitude_vals = magnitude.squeeze(-1).cpu().tolist()

                # Metric 2: S_i (Signed Contribution)
                dot_product = (weighted_expert * h_post_flat).sum(dim=-1, keepdim=True)
                signed_contrib = dot_product / h_post_norm
                signed_vals = signed_contrib.squeeze(-1).cpu().tolist()

                # Store contributions (use negative index for shared experts)
                key = (layer_idx, -(shared_idx + 1))
                self.contributions_magnitude[key].extend(magnitude_vals)
                self.contributions_signed[key].extend(signed_vals)
                self.expert_counts[key] += num_tokens

    def clear(self) -> None:
        """Clear all collected data."""
        for layer_idx in self.layer_data:
            self.layer_data[layer_idx] = []
        self._init_metric_storage()

    def remove_hooks(self) -> None:
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def get_contribution_metrics(self) -> Dict[str, Dict[Tuple[int, int], float]]:
        """
        Compute final contribution metrics for all experts.

        Returns:
            Dictionary with:
            - 'magnitude': Mean C_i per (layer, expert)
            - 'signed': Mean S_i per (layer, expert)
            - 'count': Number of tokens processed per (layer, expert)
        """
        magnitude_scores = {}
        signed_scores = {}
        counts = {}

        for key in self.contributions_magnitude.keys():
            mag_vals = self.contributions_magnitude[key]
            signed_vals = self.contributions_signed[key]

            if mag_vals:
                magnitude_scores[key] = np.mean(mag_vals)
            else:
                magnitude_scores[key] = 0.0

            if signed_vals:
                signed_scores[key] = np.mean(signed_vals)
            else:
                signed_scores[key] = 0.0

            counts[key] = self.expert_counts[key]

        return {
            'magnitude': magnitude_scores,  # C_i
            'signed': signed_scores,        # S_i
            'count': counts,
        }

    def get_expert_summary(self) -> Dict[int, Dict[int, Dict[str, float]]]:
        """
        Get a layer-wise summary of expert contributions.

        Returns:
            Nested dict: layer_idx -> expert_idx -> {'C': value, 'S': value, 'count': value}
            For shared experts, expert_idx is negative (-1, -2, etc.)
        """
        metrics = self.get_contribution_metrics()

        summary = {}
        for layer_idx in range(self.num_layers):
            summary[layer_idx] = {}

            # Routed experts (positive indices)
            for expert_idx in range(self.num_routed_experts):
                key = (layer_idx, expert_idx)
                summary[layer_idx][expert_idx] = {
                    'C': metrics['magnitude'].get(key, 0.0),
                    'S': metrics['signed'].get(key, 0.0),
                    'count': metrics['count'].get(key, 0),
                    'type': 'routed',
                }

            # Shared experts (negative indices)
            for shared_idx in range(self.num_shared_experts):
                expert_idx = -(shared_idx + 1)
                key = (layer_idx, expert_idx)
                summary[layer_idx][expert_idx] = {
                    'C': metrics['magnitude'].get(key, 0.0),
                    'S': metrics['signed'].get(key, 0.0),
                    'count': metrics['count'].get(key, 0),
                    'type': 'shared',
                }

        return summary


def compute_contribution_scores(
    model: PreTrainedModel,
    dataloader,
    device: str = None,
    max_batches: int = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Compute contribution metrics for all experts using calibration data.

    Args:
        model: MoE model
        dataloader: DataLoader with calibration samples
        device: Device to run on
        max_batches: Limit number of batches (for speed)
        show_progress: Whether to show progress bar

    Returns:
        Dictionary with contribution metrics for all experts
    """
    if device is None:
        device = next(model.parameters()).device

    collector = ExpertContributionCollector(model)
    collector.attach_hooks()

    model.eval()

    try:
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(dataloader, desc="Computing contributions")
        else:
            iterator = dataloader

        for batch_idx, batch in enumerate(iterator):
            if max_batches is not None and batch_idx >= max_batches:
                break

            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            # Forward pass (contributions computed in hooks)
            with torch.no_grad():
                _ = model(input_ids=input_ids, attention_mask=attention_mask)

        # Get final metrics
        metrics = collector.get_contribution_metrics()
        summary = collector.get_expert_summary()

        return {
            'metrics': metrics,
            'summary': summary,
            'num_layers': collector.num_layers,
            'num_experts': collector.num_experts,
        }

    finally:
        collector.remove_hooks()


def print_contribution_summary(results: Dict[str, Any], top_k: int = 10) -> None:
    """Print a formatted summary of contribution metrics."""
    summary = results['summary']
    num_layers = results['num_layers']
    num_routed = results.get('num_routed_experts', results.get('num_experts', 4))
    num_shared = results.get('num_shared_experts', 0)

    print("\n" + "=" * 80)
    print("Expert Contribution Analysis")
    print("=" * 80)
    print(f"Layers: {num_layers}, Routed Experts: {num_routed}, Shared Experts: {num_shared}")

    # For models with many experts (like DeepSeek with 64), show statistics instead of full table
    if num_routed > 8:
        print("\n" + "-" * 60)
        print("Per-Layer Statistics (Routed Experts)")
        print("-" * 60)
        print(f"{'Layer':<8} {'Mean C':<10} {'Std C':<10} {'Min C':<10} {'Max C':<10} {'Low C (<0.1)':<12}")
        print("-" * 60)

        for layer_idx in range(num_layers):
            c_values = [summary[layer_idx][e]['C'] for e in range(num_routed)
                       if e in summary[layer_idx]]
            if c_values:
                mean_c = np.mean(c_values)
                std_c = np.std(c_values)
                min_c = np.min(c_values)
                max_c = np.max(c_values)
                low_c_count = sum(1 for c in c_values if c < 0.1)
                print(f"{layer_idx:<8} {mean_c:<10.4f} {std_c:<10.4f} {min_c:<10.4f} {max_c:<10.4f} {low_c_count:<12}")
    else:
        # Original format for small number of experts
        print(f"\n{'Layer':<8}", end="")
        for e in range(num_routed):
            print(f"{'Expert '+str(e)+' (C/S)':<20}", end="")
        print()
        print("-" * (8 + 20 * num_routed))

        for layer_idx in range(num_layers):
            print(f"{layer_idx:<8}", end="")
            for expert_idx in range(num_routed):
                if expert_idx in summary[layer_idx]:
                    C = summary[layer_idx][expert_idx]['C']
                    S = summary[layer_idx][expert_idx]['S']
                    marker = " ⚠️" if S < 0 else ""
                    print(f"{C:.3f}/{S:+.3f}{marker:<6}", end="")
                else:
                    print(f"{'N/A':<20}", end="")
            print()

    # Print shared expert stats if any
    if num_shared > 0:
        print("\n" + "-" * 40)
        print("Shared Experts (always active)")
        print("-" * 40)
        for layer_idx in range(num_layers):
            print(f"Layer {layer_idx}: ", end="")
            for shared_idx in range(num_shared):
                expert_idx = -(shared_idx + 1)
                if expert_idx in summary[layer_idx]:
                    C = summary[layer_idx][expert_idx]['C']
                    S = summary[layer_idx][expert_idx]['S']
                    print(f"Shared{shared_idx}(C={C:.3f}, S={S:+.3f}) ", end="")
            print()

    print("\n⚠️ = Negative S_i (harmful expert: contribution opposes output)")
    print("Low C_i (<0.1) = Potential routing sink (prune candidate)")

    # Collect all routed experts for ranking
    all_experts = []
    for layer_idx in range(num_layers):
        for expert_idx in range(num_routed):
            if expert_idx in summary[layer_idx]:
                all_experts.append({
                    'layer': layer_idx,
                    'expert': expert_idx,
                    'C': summary[layer_idx][expert_idx]['C'],
                    'S': summary[layer_idx][expert_idx]['S'],
                    'type': 'routed',
                })

    # Sort by C_i (magnitude) to find routing sinks
    all_experts_by_c = sorted(all_experts, key=lambda x: x['C'])

    print("\n" + "=" * 50)
    print(f"Potential Routing Sinks (lowest C_i, top {top_k}):")
    print("=" * 50)
    for item in all_experts_by_c[:top_k]:
        print(f"  Layer {item['layer']:2d}, Expert {item['expert']:2d}: C = {item['C']:.4f}, S = {item['S']:+.4f}")

    # Sort by S_i (signed) to find harmful experts
    all_experts_by_s = sorted(all_experts, key=lambda x: x['S'])

    print("\n" + "=" * 50)
    print(f"Most Harmful Experts (lowest S_i, top {top_k}):")
    print("=" * 50)
    for item in all_experts_by_s[:top_k]:
        print(f"  Layer {item['layer']:2d}, Expert {item['expert']:2d}: C = {item['C']:.4f}, S = {item['S']:+.4f}")

    print("\n" + "=" * 50)
    print(f"Most Helpful Experts (highest S_i, top {top_k}):")
    print("=" * 50)
    for item in all_experts_by_s[-top_k:][::-1]:
        print(f"  Layer {item['layer']:2d}, Expert {item['expert']:2d}: C = {item['C']:.4f}, S = {item['S']:+.4f}")
