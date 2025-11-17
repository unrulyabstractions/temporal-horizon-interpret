"""Human validation utilities for generated datasets.

This module provides tools for validating and quality-checking generated
prompt pairs to ensure they properly represent different temporal horizons.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


class DatasetValidator:
    """Validator for temporal horizon prompt datasets.

    Provides methods to check quality, consistency, and temporal
    distinction of generated prompt pairs.
    """

    def __init__(self):
        """Initialize the validator."""
        self.validation_results = []

    def validate_pair(self, pair: Dict) -> Dict[str, Union[bool, str]]:
        """Validate a single prompt pair.

        Args:
            pair: Dictionary containing 'short_prompt' and 'long_prompt'

        Returns:
            Validation result dictionary with:
                - is_valid: Overall validity
                - issues: List of identified issues
                - warnings: List of warnings

        Example:
            >>> validator = DatasetValidator()
            >>> pair = {
            ...     "short_prompt": "Plan for next month",
            ...     "long_prompt": "Plan for next decade"
            ... }
            >>> result = validator.validate_pair(pair)
            >>> result["is_valid"]
            True
        """
        issues = []
        warnings = []

        short_prompt = pair.get("short_prompt", "")
        long_prompt = pair.get("long_prompt", "")

        # Check if prompts exist
        if not short_prompt or not long_prompt:
            issues.append("Missing short_prompt or long_prompt")

        # Check if prompts are identical
        if short_prompt == long_prompt:
            issues.append("Short and long prompts are identical")

        # Check minimum length
        if len(short_prompt) < 10 or len(long_prompt) < 10:
            warnings.append("Prompts are very short (< 10 chars)")

        # Check for temporal markers in short horizon
        short_temporal_markers = [
            "next quarter", "next month", "3 months", "6 months",
            "this year", "near-term", "immediate", "short-term"
        ]
        long_temporal_markers = [
            "decade", "years", "10 years", "20 years", "generation",
            "lifetime", "long-term", "future", "coming decades"
        ]

        has_short_marker = any(marker in short_prompt.lower() for marker in short_temporal_markers)
        has_long_marker = any(marker in long_prompt.lower() for marker in long_temporal_markers)

        if not has_short_marker:
            warnings.append("Short prompt may not contain clear temporal marker")

        if not has_long_marker:
            warnings.append("Long prompt may not contain clear temporal marker")

        # Check for swapped markers (major issue)
        if any(marker in short_prompt.lower() for marker in long_temporal_markers):
            issues.append("Short prompt contains long-term temporal markers")

        if any(marker in long_prompt.lower() for marker in short_temporal_markers):
            issues.append("Long prompt contains short-term temporal markers")

        # Check similarity (should be high except for temporal component)
        similarity_score = self._compute_similarity(short_prompt, long_prompt)
        if similarity_score < 0.3:
            warnings.append(f"Prompts have low similarity ({similarity_score:.2f})")

        is_valid = len(issues) == 0

        return {
            "is_valid": is_valid,
            "issues": issues,
            "warnings": warnings,
            "similarity_score": similarity_score,
        }

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute basic word overlap similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0-1).

        Example:
            >>> validator = DatasetValidator()
            >>> score = validator._compute_similarity("hello world", "hello there")
            >>> 0 <= score <= 1
            True
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0.0

    def validate_dataset(self, dataset: List[Dict]) -> Dict:
        """Validate entire dataset.

        Args:
            dataset: List of prompt pairs

        Returns:
            Validation summary dictionary.

        Example:
            >>> validator = DatasetValidator()
            >>> dataset = [
            ...     {"short_prompt": "Plan for next month",
            ...      "long_prompt": "Plan for next decade"}
            ... ]
            >>> summary = validator.validate_dataset(dataset)
            >>> "num_valid" in summary
            True
        """
        results = []

        for i, pair in enumerate(dataset):
            result = self.validate_pair(pair)
            result["pair_id"] = pair.get("pair_id", i)
            results.append(result)

        self.validation_results = results

        # Compute summary statistics
        num_valid = sum(1 for r in results if r["is_valid"])
        num_with_warnings = sum(1 for r in results if r["warnings"])

        summary = {
            "total_pairs": len(dataset),
            "num_valid": num_valid,
            "num_invalid": len(dataset) - num_valid,
            "num_with_warnings": num_with_warnings,
            "validity_rate": num_valid / len(dataset) if dataset else 0,
            "avg_similarity": sum(r["similarity_score"] for r in results) / len(results) if results else 0,
        }

        logger.info(
            f"Validation complete: {num_valid}/{len(dataset)} valid pairs "
            f"({summary['validity_rate']:.1%})"
        )

        return summary

    def get_invalid_pairs(self) -> List[Dict]:
        """Get all invalid pairs from last validation.

        Returns:
            List of validation results for invalid pairs.

        Example:
            >>> validator = DatasetValidator()
            >>> dataset = [{"short_prompt": "", "long_prompt": ""}]
            >>> validator.validate_dataset(dataset)
            {...}
            >>> invalid = validator.get_invalid_pairs()
            >>> len(invalid) > 0
            True
        """
        return [r for r in self.validation_results if not r["is_valid"]]

    def get_pairs_with_warnings(self) -> List[Dict]:
        """Get all pairs with warnings from last validation.

        Returns:
            List of validation results for pairs with warnings.

        Example:
            >>> validator = DatasetValidator()
            >>> # Assume validation has been run
            >>> # warnings = validator.get_pairs_with_warnings()
        """
        return [r for r in self.validation_results if r["warnings"]]

    def save_validation_report(self, output_path: Union[str, Path]) -> None:
        """Save validation report to file.

        Args:
            output_path: Path to save report (JSON or CSV)

        Example:
            >>> validator = DatasetValidator()
            >>> dataset = [{"short_prompt": "p1", "long_prompt": "p2"}]
            >>> validator.validate_dataset(dataset)
            {...}
            >>> # validator.save_validation_report("/tmp/report.json")
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == ".json":
            with open(output_path, "w") as f:
                json.dump(self.validation_results, f, indent=2)
        elif output_path.suffix == ".csv":
            df = pd.DataFrame(self.validation_results)
            df.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {output_path.suffix}")

        logger.info(f"Validation report saved to {output_path}")

    def create_annotation_file(
        self, dataset: List[Dict], output_path: Union[str, Path]
    ) -> None:
        """Create annotation file for human review.

        Args:
            dataset: Dataset to annotate
            output_path: Path to save annotation template

        Example:
            >>> validator = DatasetValidator()
            >>> dataset = [{"short_prompt": "p1", "long_prompt": "p2"}]
            >>> # validator.create_annotation_file(dataset, "/tmp/annotate.csv")
        """
        annotation_data = []

        for pair in dataset:
            annotation_data.append({
                "pair_id": pair.get("pair_id", ""),
                "domain": pair.get("domain", ""),
                "task": pair.get("task", ""),
                "short_prompt": pair.get("short_prompt", ""),
                "long_prompt": pair.get("long_prompt", ""),
                "is_valid": "",  # To be filled by annotator
                "short_horizon_correct": "",  # Yes/No
                "long_horizon_correct": "",  # Yes/No
                "comments": "",
            })

        df = pd.DataFrame(annotation_data)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        logger.info(f"Annotation template saved to {output_path}")


def load_annotations(annotation_file: Union[str, Path]) -> pd.DataFrame:
    """Load human annotations from file.

    Args:
        annotation_file: Path to annotation CSV file

    Returns:
        DataFrame with annotations.

    Example:
        >>> # annotations = load_annotations("/tmp/annotated.csv")
        >>> # "is_valid" in annotations.columns
        True
    """
    df = pd.read_csv(annotation_file)
    logger.info(f"Loaded {len(df)} annotations from {annotation_file}")
    return df


def compute_annotation_agreement(annotations1: pd.DataFrame, annotations2: pd.DataFrame) -> Dict:
    """Compute inter-annotator agreement metrics.

    Args:
        annotations1: First annotator's annotations
        annotations2: Second annotator's annotations

    Returns:
        Dictionary with agreement metrics.

    Example:
        >>> import pandas as pd
        >>> ann1 = pd.DataFrame({"is_valid": ["yes", "no"]})
        >>> ann2 = pd.DataFrame({"is_valid": ["yes", "yes"]})
        >>> metrics = compute_annotation_agreement(ann1, ann2)
        >>> "agreement_rate" in metrics
        True
    """
    if len(annotations1) != len(annotations2):
        raise ValueError("Annotation sets must have same length")

    # Simple agreement rate
    agreements = (annotations1["is_valid"] == annotations2["is_valid"]).sum()
    agreement_rate = agreements / len(annotations1)

    # Cohen's Kappa (simple implementation)
    # This is a simplified version; use sklearn for full implementation
    total = len(annotations1)
    p_observed = agreement_rate

    # Expected agreement
    p1_yes = (annotations1["is_valid"] == "yes").sum() / total
    p2_yes = (annotations2["is_valid"] == "yes").sum() / total
    p_expected = p1_yes * p2_yes + (1 - p1_yes) * (1 - p2_yes)

    kappa = (p_observed - p_expected) / (1 - p_expected) if p_expected < 1 else 1.0

    return {
        "agreement_rate": agreement_rate,
        "cohens_kappa": kappa,
        "num_agreements": int(agreements),
        "total_annotations": total,
    }
