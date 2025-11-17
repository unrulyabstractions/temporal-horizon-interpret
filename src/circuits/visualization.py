"""Circuit visualization utilities."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_head_importance(importance_scores: list, save_path: str = None):
    """Plot attention head importance scores."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    heads = [s[0] for s in importance_scores]
    scores = [s[1] for s in importance_scores]
    
    ax.barh(heads, scores)
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Head (Layer_Head)")
    ax.set_title("Attention Head Importance for Temporal Horizon Detection")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_circuit_diagram(layers, heads, connections, save_path: str = None):
    """Plot circuit diagram showing important components."""
    # Placeholder for circuit visualization
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.text(0.5, 0.5, "Circuit Diagram", ha="center", va="center", fontsize=20)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
