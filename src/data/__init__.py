"""Data loading utilities for benchmarks."""

from .benchmarks import (
    load_wikitext2,
    get_calibration_data,
    get_eval_data,
    WikiTextDataset,
)

__all__ = [
    "load_wikitext2",
    "get_calibration_data",
    "get_eval_data",
    "WikiTextDataset",
]
