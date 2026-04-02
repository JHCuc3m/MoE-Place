"""
Perplexity evaluation for language models.

Computes perplexity using sliding window approach for accurate measurement.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Optional, Dict, Any, Union
from tqdm import tqdm
import logging
import math

logger = logging.getLogger(__name__)


def compute_perplexity(
    model: PreTrainedModel,
    dataloader: DataLoader,
    device: Optional[str] = None,
    max_batches: Optional[int] = None,
    show_progress: bool = True,
) -> Dict[str, float]:
    """
    Compute perplexity on a dataset.

    Args:
        model: HuggingFace causal LM model
        dataloader: DataLoader yielding batches with input_ids, attention_mask, labels
        device: Device to run on (defaults to model's device)
        max_batches: Maximum number of batches to evaluate (for quick testing)
        show_progress: Whether to show progress bar

    Returns:
        Dict with perplexity and related metrics
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    total_loss = 0.0
    total_tokens = 0
    num_batches = 0

    iterator = tqdm(dataloader, desc="Computing perplexity") if show_progress else dataloader

    with torch.no_grad():
        for batch in iterator:
            if max_batches is not None and num_batches >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss

            # Count non-padding tokens (labels != -100)
            num_tokens = (labels != -100).sum().item()

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            num_batches += 1

            if show_progress:
                current_ppl = math.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')
                iterator.set_postfix({"ppl": f"{current_ppl:.2f}", "tokens": total_tokens})

    # Compute final metrics
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss)

    return {
        "perplexity": perplexity,
        "loss": avg_loss,
        "total_tokens": total_tokens,
        "num_batches": num_batches,
    }


def evaluate_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset_name: str = "wikitext2",
    split: str = "test",
    batch_size: int = 8,
    max_length: int = 512,
    stride: int = 256,
    max_batches: Optional[int] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate perplexity on a standard benchmark.

    Convenience function that handles data loading and evaluation.

    Args:
        model: HuggingFace causal LM model
        tokenizer: HuggingFace tokenizer
        dataset_name: Benchmark dataset ("wikitext2")
        split: Dataset split ("test", "validation")
        batch_size: Batch size for evaluation
        max_length: Maximum sequence length
        stride: Stride for sliding window
        max_batches: Maximum batches (for quick testing)
        device: Device to run on

    Returns:
        Dict with perplexity and evaluation metadata
    """
    from src.data.benchmarks import get_eval_data, create_dataloader

    logger.info(f"Evaluating perplexity on {dataset_name} {split}")

    # Load evaluation data
    texts, dataset = get_eval_data(
        tokenizer=tokenizer,
        max_length=max_length,
        stride=stride,
        dataset_name=dataset_name,
        split=split,
    )

    # Create dataloader
    dataloader = create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # Compute perplexity
    results = compute_perplexity(
        model=model,
        dataloader=dataloader,
        device=device,
        max_batches=max_batches,
    )

    # Add metadata
    results["dataset"] = dataset_name
    results["split"] = split
    results["num_texts"] = len(texts)
    results["num_windows"] = len(dataset)
    results["max_length"] = max_length
    results["stride"] = stride

    logger.info(f"Perplexity: {results['perplexity']:.2f} on {results['total_tokens']} tokens")

    return results


def quick_perplexity_check(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    num_batches: int = 10,
    batch_size: int = 4,
) -> float:
    """
    Quick perplexity estimate for sanity checking.

    Runs on a small subset for fast iteration.

    Args:
        model: HuggingFace causal LM model
        tokenizer: HuggingFace tokenizer
        num_batches: Number of batches to evaluate
        batch_size: Batch size

    Returns:
        Estimated perplexity
    """
    results = evaluate_perplexity(
        model=model,
        tokenizer=tokenizer,
        dataset_name="wikitext2",
        split="test",
        batch_size=batch_size,
        max_batches=num_batches,
    )
    return results["perplexity"]
