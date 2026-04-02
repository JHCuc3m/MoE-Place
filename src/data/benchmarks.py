"""
Benchmark data loading for MoE analysis and evaluation.

Provides lightweight data loading for:
- Co-activation calibration (small subset)
- Perplexity evaluation (full test set)

Primary benchmark: WikiText-2 (small, standard LM benchmark)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from datasets import load_dataset
from typing import Optional, List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class WikiTextDataset(Dataset):
    """
    WikiText dataset for language modeling evaluation.

    Tokenizes text and creates fixed-length sequences for perplexity computation.
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        stride: int = 256,
    ):
        """
        Args:
            texts: List of text strings
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            stride: Stride for sliding window (for long texts)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride

        # Concatenate all texts and tokenize
        full_text = "\n\n".join([t for t in texts if t.strip()])

        self.encodings = tokenizer(
            full_text,
            return_tensors="pt",
            truncation=False,
            add_special_tokens=False,
        )

        # Create sliding window indices
        self.input_ids = self.encodings.input_ids[0]
        total_length = len(self.input_ids)

        # Generate start indices for each window
        self.windows = []
        for start in range(0, total_length - max_length + 1, stride):
            self.windows.append(start)

        # Add final window if there's remaining text
        if total_length > max_length and (total_length - max_length) % stride != 0:
            self.windows.append(total_length - max_length)

        # Handle short texts
        if not self.windows and total_length > 0:
            self.windows.append(0)

        logger.info(f"Created dataset with {len(self.windows)} windows from {total_length} tokens")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = self.windows[idx]
        end = min(start + self.max_length, len(self.input_ids))

        input_ids = self.input_ids[start:end]
        attention_mask = torch.ones_like(input_ids)

        # For perplexity, labels = input_ids (shifted internally by model)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }


def load_wikitext2(
    split: str = "test",
    tokenizer: Optional[PreTrainedTokenizer] = None,
    max_samples: Optional[int] = None,
) -> Tuple[List[str], Optional[WikiTextDataset]]:
    """
    Load WikiText-2 dataset.

    Args:
        split: Dataset split ("train", "validation", "test")
        tokenizer: Optional tokenizer for creating dataset
        max_samples: Maximum number of text samples to load

    Returns:
        Tuple of (raw_texts, dataset) where dataset is None if no tokenizer provided
    """
    logger.info(f"Loading WikiText-2 {split} split...")

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

    # Filter empty lines and get texts
    texts = [item["text"] for item in dataset if item["text"].strip()]

    if max_samples is not None:
        texts = texts[:max_samples]

    logger.info(f"Loaded {len(texts)} non-empty texts from WikiText-2 {split}")

    # Create tokenized dataset if tokenizer provided
    torch_dataset = None
    if tokenizer is not None:
        torch_dataset = WikiTextDataset(texts, tokenizer)

    return texts, torch_dataset


def get_calibration_data(
    tokenizer: PreTrainedTokenizer,
    num_samples: int = 128,
    max_length: int = 512,
    dataset_name: str = "wikitext2",
    seed: int = 42,
) -> Tuple[List[str], WikiTextDataset]:
    """
    Get calibration data for co-activation collection.

    Uses a small subset of training data to compute co-activation patterns.

    Args:
        tokenizer: HuggingFace tokenizer
        num_samples: Number of text samples for calibration
        max_length: Maximum sequence length
        dataset_name: Dataset to use ("wikitext2")
        seed: Random seed for sampling

    Returns:
        Tuple of (raw_texts, dataset)
    """
    import random
    random.seed(seed)

    if dataset_name == "wikitext2":
        # Use train split for calibration
        texts, _ = load_wikitext2(split="train", tokenizer=None)

        # Filter for substantial texts (at least 50 chars)
        texts = [t for t in texts if len(t.strip()) >= 50]

        # Sample if we have more than needed
        if len(texts) > num_samples:
            texts = random.sample(texts, num_samples)

        logger.info(f"Calibration data: {len(texts)} samples from WikiText-2 train")

        dataset = WikiTextDataset(texts, tokenizer, max_length=max_length)
        return texts, dataset
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_eval_data(
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    stride: int = 256,
    dataset_name: str = "wikitext2",
    split: str = "test",
) -> Tuple[List[str], WikiTextDataset]:
    """
    Get evaluation data for perplexity measurement.

    Uses the full test set for accurate evaluation.

    Args:
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        stride: Stride for sliding window
        dataset_name: Dataset to use ("wikitext2")
        split: Dataset split ("test" or "validation")

    Returns:
        Tuple of (raw_texts, dataset)
    """
    if dataset_name == "wikitext2":
        texts, _ = load_wikitext2(split=split, tokenizer=None)

        dataset = WikiTextDataset(
            texts,
            tokenizer,
            max_length=max_length,
            stride=stride,
        )

        logger.info(f"Eval data: {len(texts)} texts, {len(dataset)} windows from WikiText-2 {split}")
        return texts, dataset
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def create_dataloader(
    dataset: WikiTextDataset,
    batch_size: int = 8,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a DataLoader from a WikiTextDataset.

    Args:
        dataset: WikiTextDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of data loading workers

    Returns:
        DataLoader instance
    """
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Pad sequences in batch
        max_len = max(item["input_ids"].size(0) for item in batch)

        input_ids = []
        attention_mask = []
        labels = []

        for item in batch:
            seq_len = item["input_ids"].size(0)
            pad_len = max_len - seq_len

            if pad_len > 0:
                input_ids.append(torch.cat([
                    item["input_ids"],
                    torch.zeros(pad_len, dtype=torch.long)
                ]))
                attention_mask.append(torch.cat([
                    item["attention_mask"],
                    torch.zeros(pad_len, dtype=torch.long)
                ]))
                labels.append(torch.cat([
                    item["labels"],
                    torch.full((pad_len,), -100, dtype=torch.long)  # -100 = ignore in loss
                ]))
            else:
                input_ids.append(item["input_ids"])
                attention_mask.append(item["attention_mask"])
                labels.append(item["labels"])

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
