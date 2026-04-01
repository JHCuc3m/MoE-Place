#!/usr/bin/env python3
"""
Collect routing statistics from a pretrained MoE model.

This script:
1. Loads TinyMixtral-4x248M-MoE (or other pretrained MoE)
2. Runs inference on text data
3. Collects routing statistics with realistic patterns
4. Generates collaboration analysis and visualizations

Run with: python scripts/collect_pretrained_routing.py
"""

import sys
import argparse
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, ".")

from src.models.pretrained_moe import (
    load_pretrained_moe,
    get_moe_config,
    MixtralRoutingCollector,
    print_model_info,
    TINY_MIXTRAL,
)
from src.routing.statistics import RoutingStatisticsCollector
from src.routing.visualization import (
    plot_all_layers,
    print_collaboration_summary,
    export_statistics,
)


# Sample texts for routing analysis (diverse content)
SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence.",
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    "The capital of France is Paris, which is known for the Eiffel Tower.",
    "import torch\nimport torch.nn as nn\nclass Model(nn.Module):",
    "In quantum mechanics, particles can exist in superposition states.",
    "SELECT * FROM users WHERE age > 18 ORDER BY name;",
    "The mitochondria is the powerhouse of the cell.",
    "async function fetchData() { const response = await fetch(url); }",
    "Einstein's theory of relativity changed our understanding of space and time.",
    "To be or not to be, that is the question.",
    "The stock market experienced significant volatility today.",
    "Climate change poses serious threats to global ecosystems.",
    "Neural networks are inspired by biological brain structures.",
    "The recipe calls for two cups of flour and one egg.",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Collect routing stats from pretrained MoE")
    parser.add_argument("--model", type=str, default=TINY_MIXTRAL,
                        help=f"HuggingFace model name (default: {TINY_MIXTRAL})")
    parser.add_argument("--num_samples", type=int, default=500,
                        help="Number of text samples to process (default: 500)")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Max sequence length (default: 128)")
    parser.add_argument("--output_dir", type=str, default="experiments/pretrained_routing_stats",
                        help="Output directory")
    parser.add_argument("--no_plots", action="store_true",
                        help="Skip generating plots")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()


def generate_samples(base_texts: list, num_samples: int, seed: int) -> list:
    """Generate text samples by cycling through and combining base texts."""
    import random
    random.seed(seed)

    samples = []
    for i in range(num_samples):
        # Cycle through base texts and optionally combine them
        base_idx = i % len(base_texts)
        text = base_texts[base_idx]

        # Sometimes combine multiple texts
        if random.random() < 0.3:
            other_idx = random.randint(0, len(base_texts) - 1)
            text = text + " " + base_texts[other_idx]

        samples.append(text)

    return samples


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    print("=" * 70)
    print("Pretrained MoE Routing Statistics Collection")
    print("=" * 70)

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Load model
    print(f"\nLoading model: {args.model}")
    model, tokenizer = load_pretrained_moe(args.model, device=device)

    # Print model info
    print_model_info(model)
    config = get_moe_config(model)

    # Create routing collector
    routing_collector = MixtralRoutingCollector(model)
    print(f"\nRouting collector: {routing_collector.num_layers} MoE layers, "
          f"{routing_collector.num_experts} experts")

    # Create statistics collector
    stats_collector = RoutingStatisticsCollector(
        num_layers=routing_collector.num_layers,
        num_experts=routing_collector.num_experts,
    )

    # Generate text samples
    print(f"\nGenerating {args.num_samples} text samples...")
    samples = generate_samples(SAMPLE_TEXTS, args.num_samples, args.seed)

    # Collect routing statistics
    print("\n" + "=" * 70)
    print("Collecting routing statistics...")
    print("=" * 70)

    total_tokens = 0

    with torch.no_grad():
        for text in tqdm(samples, desc="Processing samples"):
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=args.max_length,
                truncation=True,
                padding=False,
            ).to(device)

            # Clear previous routing decisions
            routing_collector.clear()

            # Forward pass
            _ = model(**inputs)

            # Collect routing decisions
            routing = routing_collector.get_last_routing()

            for layer_idx, layer_routing in routing.items():
                expert_indices = layer_routing["expert_indices"]
                stats_collector.record_routing(layer_idx, expert_indices)

            total_tokens += inputs.input_ids.shape[1]

    print(f"\nProcessed {total_tokens:,} tokens across {args.num_samples} samples")

    # Remove hooks
    routing_collector.remove_hooks()

    # Print summary
    print_collaboration_summary(stats_collector, top_k=10)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export statistics
    stats_path = output_dir / "routing_statistics.json"
    export_statistics(stats_collector, str(stats_path))

    # Generate plots
    if not args.no_plots:
        print("\n" + "=" * 70)
        print("Generating visualizations...")
        print("=" * 70)

        plots_dir = output_dir / "plots"
        plot_all_layers(stats_collector, str(plots_dir))

    # Save model config
    config_path = output_dir / "model_config.json"
    import json
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 70)
    print("Collection Complete!")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}")

    # Memory usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        print(f"GPU memory used: {allocated:.2f} GB")


if __name__ == "__main__":
    main()
