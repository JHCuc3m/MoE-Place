"""
Pretrained MoE model loading utilities.

Supports loading pretrained MoE models from HuggingFace for
realistic routing pattern analysis.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional, Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)

# Supported pretrained models
TINY_MIXTRAL = "Isotonic/TinyMixtral-4x248M-MoE"


def load_pretrained_moe(
    model_name: str = TINY_MIXTRAL,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a pretrained MoE model from HuggingFace.

    Args:
        model_name: HuggingFace model identifier
        device: Device to load model on
        dtype: Model dtype

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading pretrained MoE: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,  # Updated from torch_dtype (deprecated)
        device_map=device,
    )

    model.eval()

    # Log model info
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded: {num_params:,} parameters")

    return model, tokenizer


def get_moe_config(model: AutoModelForCausalLM) -> Dict[str, Any]:
    """
    Extract MoE configuration from a pretrained model.

    Returns dict with MoE-relevant configuration.
    """
    config = model.config

    return {
        "model_type": getattr(config, "model_type", "unknown"),
        "num_layers": getattr(config, "num_hidden_layers", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "num_experts": getattr(config, "num_local_experts", None),
        "num_experts_per_tok": getattr(config, "num_experts_per_tok", None),
        "intermediate_size": getattr(config, "intermediate_size", None),
    }


class MixtralRoutingCollector:
    """
    Collects routing decisions from Mixtral-style MoE models.

    Uses forward hooks to capture expert selection from router layers.
    """

    def __init__(self, model: AutoModelForCausalLM):
        self.model = model
        self.config = get_moe_config(model)
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

        # Storage for routing decisions per layer
        # Key: layer_idx, Value: list of (expert_indices, router_logits) per forward
        self.routing_decisions: Dict[int, List[Dict[str, torch.Tensor]]] = {}

        self._attach_hooks()

    def _attach_hooks(self) -> None:
        """Attach forward hooks to router/gate modules."""
        layer_idx = 0

        for name, module in self.model.named_modules():
            name_lower = name.lower()
            class_name = module.__class__.__name__.lower()

            # Match various MoE gate/router patterns:
            # - "block_sparse_moe.gate" (original Mixtral)
            # - "mlp.gate" (TinyMixtral and others)
            # - Class name contains "router"
            # - ends with ".gate"
            is_gate = (
                ("gate" in name_lower and ("moe" in name_lower or "sparse" in name_lower)) or
                name_lower.endswith("mlp.gate") or
                name_lower.endswith(".gate") or
                "router" in class_name
            )

            if is_gate:
                self.routing_decisions[layer_idx] = []
                hook = module.register_forward_hook(
                    self._create_hook(layer_idx, name)
                )
                self.hooks.append(hook)
                logger.debug(f"Attached hook to {name} ({module.__class__.__name__}, layer {layer_idx})")
                layer_idx += 1

        logger.info(f"Attached {len(self.hooks)} routing hooks to MoE layers")

    def _create_hook(self, layer_idx: int, module_name: str = ""):
        """Create a forward hook for a specific layer."""
        def hook(module, input, output):
            # Handle various output formats from different MoE implementations
            router_logits = None
            expert_indices = None
            routing_weights = None

            # Try to extract routing information from output
            if isinstance(output, tuple):
                # MixtralTopKRouter returns (routing_weights, expert_indices)
                # or similar tuple formats
                for item in output:
                    if isinstance(item, torch.Tensor):
                        if item.dim() == 2:
                            if item.dtype == torch.long or item.dtype == torch.int:
                                # This is likely expert_indices
                                expert_indices = item
                            elif item.shape[-1] == self.config.get("num_experts", 4):
                                # This is likely router_logits
                                router_logits = item
                            else:
                                # Could be routing_weights
                                routing_weights = item
            elif isinstance(output, torch.Tensor):
                if output.dtype in [torch.long, torch.int]:
                    expert_indices = output
                else:
                    router_logits = output

            # If we have router_logits but no expert_indices, compute them
            if router_logits is not None and expert_indices is None:
                num_experts_per_tok = self.config.get("num_experts_per_tok", 2) or 2
                _, expert_indices = torch.topk(router_logits, num_experts_per_tok, dim=-1)

            # Store the routing decisions
            if expert_indices is not None:
                self.routing_decisions[layer_idx].append({
                    "expert_indices": expert_indices.detach().cpu(),
                    "router_logits": router_logits.detach().cpu() if router_logits is not None else None,
                    "routing_weights": routing_weights.detach().cpu() if routing_weights is not None else None,
                })
                logger.debug(f"Captured routing from {module_name}: indices shape {expert_indices.shape}")
            else:
                # Debug: log what we got
                logger.debug(f"Hook on {module_name} got output type: {type(output)}")
                if isinstance(output, tuple):
                    logger.debug(f"  Tuple contents: {[type(x).__name__ for x in output]}")

        return hook

    def clear(self) -> None:
        """Clear collected routing decisions."""
        for layer_idx in self.routing_decisions:
            self.routing_decisions[layer_idx] = []

    def remove_hooks(self) -> None:
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_last_routing(self) -> Dict[int, Dict[str, torch.Tensor]]:
        """Get routing decisions from the last forward pass."""
        result = {}
        for layer_idx, decisions in self.routing_decisions.items():
            if decisions:
                result[layer_idx] = decisions[-1]
        return result

    @property
    def num_layers(self) -> int:
        return len(self.routing_decisions)

    @property
    def num_experts(self) -> int:
        return self.config.get("num_experts", 0)


def print_model_info(model: AutoModelForCausalLM, verbose: bool = False) -> None:
    """Print model structure and MoE configuration."""
    config = get_moe_config(model)

    print("\n" + "=" * 60)
    print("Pretrained MoE Model Info")
    print("=" * 60)

    for key, value in config.items():
        print(f"  {key}: {value}")

    # Find MoE-related modules
    print("\nMoE-related Modules:")
    moe_modules = []
    for name, module in model.named_modules():
        name_lower = name.lower()
        if any(x in name_lower for x in ["expert", "moe", "gate", "router", "sparse"]):
            moe_modules.append((name, module.__class__.__name__, module))

    # Print unique patterns
    seen = set()
    for name, cls_name, module in moe_modules:
        # Create pattern by replacing numbers
        import re
        pattern = re.sub(r'\.\d+\.', '.N.', name)
        if pattern not in seen or verbose:
            seen.add(pattern)
            extra = ""
            if isinstance(module, torch.nn.Linear):
                extra = f" [{module.in_features} -> {module.out_features}]"
            print(f"  {name}: {cls_name}{extra}")

    # Summary
    print(f"\nTotal MoE-related modules: {len(moe_modules)}")

    # Find potential router modules (Linear with out_features == num_experts)
    num_experts = config.get("num_experts")
    if num_experts:
        print(f"\nPotential router modules (out_features == {num_experts}):")
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and module.out_features == num_experts:
                print(f"  {name}: Linear [{module.in_features} -> {module.out_features}]")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Loading TinyMixtral...")
    model, tokenizer = load_pretrained_moe()

    print_model_info(model)

    # Test inference
    print("\nTest inference:")
    inputs = tokenizer("The capital of France is", return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20)
    print(tokenizer.decode(outputs[0]))
