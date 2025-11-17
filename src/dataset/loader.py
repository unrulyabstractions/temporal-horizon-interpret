"""Dataset loading utilities for temporal horizon detection.

This module provides functions to load, preprocess, and split datasets
for training and evaluation.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def load_dataset(
    file_path: Union[str, Path], format: str = "jsonl"
) -> List[Dict]:
    """Load dataset from file.

    Args:
        file_path: Path to dataset file
        format: File format ("jsonl", "json", "csv")

    Returns:
        List of dataset items.

    Raises:
        ValueError: If format is not supported
        FileNotFoundError: If file doesn't exist

    Example:
        >>> # dataset = load_dataset("data/raw/prompts.jsonl")
        >>> # len(dataset) > 0
        True
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    logger.info(f"Loading dataset from {file_path}")

    if format == "jsonl":
        data = []
        with open(file_path, "r") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    elif format == "json":
        with open(file_path, "r") as f:
            data = json.load(f)
    elif format == "csv":
        df = pd.read_csv(file_path)
        data = df.to_dict(orient="records")
    else:
        raise ValueError(f"Unsupported format: {format}")

    logger.info(f"Loaded {len(data)} items")
    return data


def expand_dataset(dataset: List[Dict]) -> Tuple[List[str], List[int], List[Dict]]:
    """Expand paired dataset into individual prompts with labels.

    Args:
        dataset: List of prompt pairs with 'short_prompt' and 'long_prompt'

    Returns:
        Tuple of (prompts, labels, metadata):
            - prompts: List of all prompts
            - labels: List of labels (0=short, 1=long)
            - metadata: List of metadata dicts

    Example:
        >>> dataset = [
        ...     {"short_prompt": "Plan for next month",
        ...      "long_prompt": "Plan for next decade",
        ...      "domain": "business"}
        ... ]
        >>> prompts, labels, metadata = expand_dataset(dataset)
        >>> len(prompts) == 2 * len(dataset)
        True
    """
    prompts = []
    labels = []
    metadata = []

    for item in dataset:
        # Short prompt
        prompts.append(item["short_prompt"])
        labels.append(0)
        metadata.append({
            "pair_id": item.get("pair_id"),
            "domain": item.get("domain"),
            "task": item.get("task"),
            "horizon": "short",
        })

        # Long prompt
        prompts.append(item["long_prompt"])
        labels.append(1)
        metadata.append({
            "pair_id": item.get("pair_id"),
            "domain": item.get("domain"),
            "task": item.get("task"),
            "horizon": "long",
        })

    return prompts, labels, metadata


def split_dataset(
    prompts: List[str],
    labels: List[int],
    metadata: Optional[List[Dict]] = None,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    stratify: bool = True,
) -> Dict[str, Dict]:
    """Split dataset into train/val/test sets.

    Args:
        prompts: List of prompts
        labels: List of labels
        metadata: Optional list of metadata dicts
        train_size: Fraction for training set
        val_size: Fraction for validation set
        test_size: Fraction for test set
        random_state: Random seed
        stratify: Whether to stratify by labels

    Returns:
        Dictionary with keys 'train', 'val', 'test', each containing:
            - prompts: List of prompts
            - labels: List of labels
            - metadata: List of metadata (if provided)

    Raises:
        ValueError: If split sizes don't sum to 1.0

    Example:
        >>> prompts = ["prompt1", "prompt2", "prompt3", "prompt4"]
        >>> labels = [0, 1, 0, 1]
        >>> splits = split_dataset(prompts, labels)
        >>> set(splits.keys()) == {"train", "val", "test"}
        True
    """
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError(
            f"Split sizes must sum to 1.0, got {train_size + val_size + test_size}"
        )

    stratify_by = labels if stratify else None

    # First split: train vs (val + test)
    train_prompts, temp_prompts, train_labels, temp_labels = train_test_split(
        prompts,
        labels,
        train_size=train_size,
        random_state=random_state,
        stratify=stratify_by,
    )

    if metadata:
        train_meta, temp_meta = train_test_split(
            metadata,
            train_size=train_size,
            random_state=random_state,
            stratify=stratify_by,
        )
    else:
        train_meta = temp_meta = None

    # Second split: val vs test
    val_ratio = val_size / (val_size + test_size)
    stratify_temp = temp_labels if stratify else None

    val_prompts, test_prompts, val_labels, test_labels = train_test_split(
        temp_prompts,
        temp_labels,
        train_size=val_ratio,
        random_state=random_state,
        stratify=stratify_temp,
    )

    if metadata:
        val_meta, test_meta = train_test_split(
            temp_meta,
            train_size=val_ratio,
            random_state=random_state,
            stratify=stratify_temp,
        )
    else:
        val_meta = test_meta = None

    splits = {
        "train": {
            "prompts": train_prompts,
            "labels": train_labels,
            "metadata": train_meta,
        },
        "val": {
            "prompts": val_prompts,
            "labels": val_labels,
            "metadata": val_meta,
        },
        "test": {
            "prompts": test_prompts,
            "labels": test_labels,
            "metadata": test_meta,
        },
    }

    logger.info(
        f"Split dataset: train={len(train_prompts)}, "
        f"val={len(val_prompts)}, test={len(test_prompts)}"
    )

    return splits


def save_split(split_data: Dict, output_dir: Union[str, Path], split_name: str) -> None:
    """Save a dataset split to files.

    Args:
        split_data: Dictionary with 'prompts', 'labels', 'metadata'
        output_dir: Directory to save files
        split_name: Name of split (train/val/test)

    Example:
        >>> split_data = {"prompts": ["p1"], "labels": [0], "metadata": None}
        >>> # save_split(split_data, "/tmp/splits", "train")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as JSONL
    output_file = output_dir / f"{split_name}.jsonl"

    with open(output_file, "w") as f:
        for i, (prompt, label) in enumerate(zip(split_data["prompts"], split_data["labels"])):
            item = {
                "prompt": prompt,
                "label": label,
            }
            if split_data["metadata"]:
                item["metadata"] = split_data["metadata"][i]
            f.write(json.dumps(item) + "\n")

    logger.info(f"Saved {split_name} split to {output_file}")


def get_dataset_statistics(dataset: List[Dict]) -> Dict:
    """Compute statistics about the dataset.

    Args:
        dataset: List of dataset items

    Returns:
        Dictionary containing dataset statistics.

    Example:
        >>> dataset = [
        ...     {"domain": "business", "short_prompt": "p1", "long_prompt": "p2"},
        ...     {"domain": "science", "short_prompt": "p3", "long_prompt": "p4"}
        ... ]
        >>> stats = get_dataset_statistics(dataset)
        >>> stats["num_pairs"]
        2
    """
    df = pd.DataFrame(dataset)

    stats = {
        "num_pairs": len(dataset),
        "num_prompts": len(dataset) * 2,  # Each pair has 2 prompts
        "domains": df["domain"].value_counts().to_dict() if "domain" in df else {},
        "avg_short_length": df["short_prompt"].str.len().mean() if "short_prompt" in df else 0,
        "avg_long_length": df["long_prompt"].str.len().mean() if "long_prompt" in df else 0,
    }

    return stats


def filter_by_domain(dataset: List[Dict], domains: List[str]) -> List[Dict]:
    """Filter dataset by domain.

    Args:
        dataset: List of dataset items
        domains: List of domains to keep

    Returns:
        Filtered dataset.

    Example:
        >>> dataset = [
        ...     {"domain": "business", "short_prompt": "p1"},
        ...     {"domain": "science", "short_prompt": "p2"}
        ... ]
        >>> filtered = filter_by_domain(dataset, ["business"])
        >>> len(filtered)
        1
    """
    filtered = [item for item in dataset if item.get("domain") in domains]
    logger.info(f"Filtered to {len(filtered)}/{len(dataset)} items for domains: {domains}")
    return filtered
