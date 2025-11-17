"""Probe evaluation utilities."""

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


class ProbeEvaluator:
    """Evaluator for trained probes."""

    def __init__(self, probe: nn.Module, device: Optional[str] = None):
        """Initialize evaluator."""
        self.probe = probe
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.probe = self.probe.to(self.device)
        self.probe.eval()

    @torch.no_grad()
    def predict(self, x: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Get predictions."""
        all_preds = []
        for i in range(0, len(x), batch_size):
            batch = torch.FloatTensor(x[i : i + batch_size]).to(self.device)
            logits = self.probe(batch)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
        return np.array(all_preds)

    @torch.no_grad()
    def predict_proba(self, x: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Get prediction probabilities."""
        all_probs = []
        for i in range(0, len(x), batch_size):
            batch = torch.FloatTensor(x[i : i + batch_size]).to(self.device)
            logits = self.probe(batch)
            probs = torch.softmax(logits, dim=1)
            all_probs.extend(probs.cpu().numpy())
        return np.array(all_probs)

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Comprehensive evaluation."""
        preds = self.predict(x)
        probs = self.predict_proba(x)

        return {
            "accuracy": accuracy_score(y, preds),
            "precision": precision_score(y, preds, average="binary"),
            "recall": recall_score(y, preds, average="binary"),
            "f1": f1_score(y, preds, average="binary"),
            "auc": roc_auc_score(y, probs[:, 1]),
        }

    def evaluate_by_domain(
        self, x: np.ndarray, y: np.ndarray, domains: List[str]
    ) -> Dict[str, Dict]:
        """Evaluate per domain."""
        results = {}
        for domain in set(domains):
            mask = np.array([d == domain for d in domains])
            if mask.sum() > 0:
                results[domain] = self.evaluate(x[mask], y[mask])
        return results

    def compute_confusion_matrix(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute confusion matrix."""
        preds = self.predict(x)
        return confusion_matrix(y, preds)
