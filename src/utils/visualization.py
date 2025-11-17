"""Visualization utilities."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def plot_activation_heatmap(activations: np.ndarray, save_path: str = None):
    """Plot activation heatmap."""
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(activations, cmap="viridis", ax=ax)
    ax.set_title("Activation Heatmap")
    ax.set_xlabel("Hidden Dimensions")
    ax.set_ylabel("Samples")
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_probe_accuracy_curve(history: dict, save_path: str = None):
    """Plot training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curve
    ax1.plot(history["train_loss"], label="Train Loss")
    val_losses = [m["loss"] for m in history["val_metrics"]]
    ax1.plot(val_losses, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curve
    ax2.plot(history["train_acc"], label="Train Acc")
    val_accs = [m["accuracy"] for m in history["val_metrics"]]
    ax2.plot(val_accs, label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training Accuracy")
    ax2.legend()
    ax2.grid(True)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_layer_wise_accuracy(layer_accuracies: dict, save_path: str = None):
    """Plot accuracy across layers."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    layers = sorted([int(k.split("_")[1]) for k in layer_accuracies.keys()])
    accuracies = [layer_accuracies[f"layer_{l}"] for l in layers]
    
    ax.plot(layers, accuracies, marker='o', linewidth=2, markersize=8)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Probe Accuracy", fontsize=12)
    ax.set_title("Temporal Horizon Detection Across Layers", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.5, 1.0])
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, save_path: str = None):
    """Plot confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    ax.set_xticklabels(["Short", "Long"])
    ax.set_yticklabels(["Short", "Long"])
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
