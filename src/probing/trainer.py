"""Probe training loops and utilities.

This module provides training infrastructure for probes with early stopping,
checkpointing, and metric tracking.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ProbeTrainer:
    """Trainer for temporal horizon probes.

    Handles training loop, validation, early stopping, and checkpointing.

    Attributes:
        probe: Probe model to train
        device: Device to train on
        optimizer: Optimizer instance
        criterion: Loss function
    """

    def __init__(
        self,
        probe: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        device: Optional[str] = None,
    ):
        """Initialize trainer.

        Args:
            probe: Probe model
            learning_rate: Learning rate
            weight_decay: L2 regularization weight
            device: Device ("cpu", "cuda", etc.)
        """
        self.probe = probe
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.probe = self.probe.to(self.device)

        self.optimizer = optim.AdamW(
            probe.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()

        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.epochs_no_improve = 0

    def train_epoch(
        self, train_loader: DataLoader
    ) -> Tuple[float, float]:
        """Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Tuple of (loss, accuracy).
        """
        self.probe.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()
            logits = self.probe(batch_x)
            loss = self.criterion(logits, batch_y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)

        return avg_loss, accuracy

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate probe on validation set.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary of metrics.
        """
        self.probe.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []

        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            logits = self.probe(batch_x)
            loss = self.criterion(logits, batch_y)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

        metrics = {
            "loss": total_loss / len(val_loader),
            "accuracy": accuracy_score(all_labels, all_preds),
            "f1": f1_score(all_labels, all_preds, average="binary"),
            "auc": roc_auc_score(all_labels, all_probs),
        }

        return metrics

    def train(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        val_x: np.ndarray,
        val_y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
        checkpoint_dir: Optional[Union[str, Path]] = None,
    ) -> Dict:
        """Train probe with early stopping.

        Args:
            train_x: Training activations
            train_y: Training labels
            val_x: Validation activations
            val_y: Validation labels
            epochs: Maximum number of epochs
            batch_size: Batch size
            early_stopping_patience: Patience for early stopping
            checkpoint_dir: Directory to save checkpoints

        Returns:
            Dictionary with training history.
        """
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(train_x), torch.LongTensor(train_y)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(val_x), torch.LongTensor(val_y)
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        history = {"train_loss": [], "train_acc": [], "val_metrics": []}

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_metrics"].append(val_metrics)

            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}"
            )

            # Early stopping check
            if val_metrics["accuracy"] > self.best_val_acc:
                self.best_val_acc = val_metrics["accuracy"]
                self.best_val_loss = val_metrics["loss"]
                self.epochs_no_improve = 0

                # Save checkpoint
                if checkpoint_dir:
                    self.save_checkpoint(checkpoint_dir, epoch, val_metrics)
            else:
                self.epochs_no_improve += 1

            if self.epochs_no_improve >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        return history

    def save_checkpoint(
        self, checkpoint_dir: Union[str, Path], epoch: int, metrics: Dict
    ) -> None:
        """Save model checkpoint.

        Args:
            checkpoint_dir: Directory to save checkpoint
            epoch: Current epoch
            metrics: Validation metrics
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.probe.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
        }

        path = checkpoint_dir / "best_probe.pt"
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
