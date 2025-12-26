"""
Visualization Tools for Geometric Benchmark

Creates standardized plots for all analysis results.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional, List
import matplotlib.cm as cm

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def plot_fourier_spectrum(
    frequencies: np.ndarray,
    power: np.ndarray,
    metrics: Dict[str, Any],
    save_path: Optional[Path] = None,
    title: str = "Fourier Spectrum",
) -> plt.Figure:
    """
    Plot 1D Fourier spectrum.

    Args:
        frequencies: Frequency array
        power: Power spectrum
        metrics: Dictionary with analysis metrics
        save_path: Optional path to save figure
        title: Plot title

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot spectrum
    ax.plot(frequencies, power, 'b-', linewidth=2, label='Power Spectrum')

    # Highlight peaks
    for freq, pwr in metrics.get('peaks', [])[:5]:  # Top 5 peaks
        ax.axvline(freq, color='r', linestyle='--', alpha=0.5)
        ax.text(freq, pwr, f'{freq:.2f}', ha='center', va='bottom')

    # Mark dominant frequency
    dom_freq = metrics.get('dominant_frequency', 0)
    if dom_freq != 0:
        ax.axvline(dom_freq, color='g', linestyle='-', linewidth=2, label=f'Dominant: {dom_freq:.3f}')

    ax.set_xlabel('Frequency')
    ax.set_ylabel('Power')
    ax.set_title(f'{title}\nTop-1 Power Ratio: {metrics.get("top1_power_ratio", 0):.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_fourier_spectrum_2d(
    freqs_k: np.ndarray,
    freqs_l: np.ndarray,
    power: np.ndarray,
    metrics: Dict[str, Any],
    save_path: Optional[Path] = None,
    title: str = "2D Fourier Spectrum",
) -> plt.Figure:
    """
    Plot 2D Fourier spectrum (for torus/grid tasks).

    Args:
        freqs_k: Frequency array for first dimension
        freqs_l: Frequency array for second dimension
        power: 2D power spectrum
        metrics: Analysis metrics
        save_path: Optional path to save
        title: Plot title

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Log scale for better visualization
    power_log = np.log10(power + 1e-10)

    # Plot heatmap
    im = ax.pcolormesh(freqs_k, freqs_l, power_log.T, shading='auto', cmap='viridis')
    plt.colorbar(im, ax=ax, label='Log Power')

    # Mark dominant mode
    dom_k = metrics.get('dominant_mode_k', 0)
    dom_l = metrics.get('dominant_mode_l', 0)
    ax.scatter([dom_k], [dom_l], color='r', s=100, marker='x', linewidths=3, label='Dominant Mode')
    ax.legend()

    ax.set_xlabel('Frequency k')
    ax.set_ylabel('Frequency l')
    ax.set_title(f'{title}\nSeparability Score: {metrics.get("separability_score", 0):.3f}')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_periodic_neurons(
    periodic_neurons: List[Dict[str, Any]],
    activations: np.ndarray,
    positions: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Periodic Neurons",
    top_k: int = 9,
) -> plt.Figure:
    """
    Plot top-k periodic neurons with their activations and sinusoidal fits.

    Args:
        periodic_neurons: List of periodic neuron dictionaries
        activations: [n_neurons, n_positions] activation array
        positions: Position values
        save_path: Optional path to save
        title: Plot title
        top_k: Number of neurons to plot

    Returns:
        matplotlib Figure
    """
    top_neurons = periodic_neurons[:top_k]
    n_cols = 3
    n_rows = (top_k + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]

    for idx, neuron_info in enumerate(top_neurons):
        ax = axes[idx]
        neuron_idx = neuron_info['neuron_idx']

        # Plot activations
        ax.plot(positions, activations[neuron_idx], 'o-', label='Activation', alpha=0.7)

        # Plot fitted sinusoid
        freq = neuron_info['frequency']
        amp = neuron_info['amplitude']
        phase = neuron_info['phase']
        offset = neuron_info.get('offset', 0)

        pos_norm = (positions - positions.min()) / (positions.max() - positions.min() + 1e-8)
        fitted = amp * np.sin(2 * np.pi * freq * pos_norm + phase) + offset
        ax.plot(positions, fitted, 'r--', label=f'Fit (R²={neuron_info["r_squared"]:.3f})', linewidth=2)

        ax.set_xlabel('Position')
        ax.set_ylabel('Activation')
        ax.set_title(f'Neuron {neuron_idx}, Freq={freq:.1f}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide extra subplots
    for idx in range(len(top_neurons), len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'{title}\nDetected {len(periodic_neurons)} periodic neurons', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_attention_patterns(
    attention_data: Dict[str, Any],
    save_path: Optional[Path] = None,
    title: str = "Attention Patterns",
) -> plt.Figure:
    """
    Visualize attention patterns across layers and heads.

    Args:
        attention_data: Dictionary with attention analysis
        save_path: Optional path to save
        title: Plot title

    Returns:
        matplotlib Figure
    """
    attention_matrix = attention_data['attention_matrix']

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot heatmap
    im = ax.imshow(attention_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(im, ax=ax, label='Attention Weight')

    # Add pattern type annotation
    pattern_type = attention_data.get('pattern_type', 'unknown')
    entropy_mean = attention_data.get('entropy_mean', 0)
    diagonal_bias = attention_data.get('diagonal_bias', 0)

    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    ax.set_title(f'{title}\nPattern: {pattern_type}, Entropy: {entropy_mean:.2f}, Diagonal Bias: {diagonal_bias:.2f}')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_all_attention_patterns(
    all_attention: torch.Tensor,
    n_layers: int,
    n_heads: int,
    save_path: Optional[Path] = None,
    title: str = "All Attention Patterns",
) -> plt.Figure:
    """
    Plot attention patterns for all layers and heads.

    Args:
        all_attention: [n_layers, n_heads, batch, seq_len, seq_len]
        n_layers: Number of layers to plot
        n_heads: Number of heads to plot
        save_path: Optional path to save
        title: Plot title

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(4 * n_heads, 4 * n_layers))

    # Handle single layer/head
    if n_layers == 1 and n_heads == 1:
        axes = np.array([[axes]])
    elif n_layers == 1:
        axes = axes.reshape(1, -1)
    elif n_heads == 1:
        axes = axes.reshape(-1, 1)

    # Average over batch
    attn_mean = all_attention[:n_layers, :n_heads].mean(dim=2).cpu().numpy()

    for layer in range(n_layers):
        for head in range(n_heads):
            ax = axes[layer, head]

            attn_matrix = attn_mean[layer, head]
            im = ax.imshow(attn_matrix, cmap='viridis', aspect='auto', vmin=0, vmax=attn_matrix.max())

            ax.set_title(f'L{layer}H{head}')
            ax.set_xlabel('Key')
            ax.set_ylabel('Query')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_embedding_geometry(
    embedding_data: Dict[str, Any],
    save_path: Optional[Path] = None,
    title: str = "Embedding Geometry",
) -> plt.Figure:
    """
    Plot dimensionality-reduced embeddings.

    Args:
        embedding_data: Dictionary with reduced embeddings
        save_path: Optional path to save
        title: Plot title

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    reduced = embedding_data['reduced_embeddings']
    labels = embedding_data.get('labels')
    method = embedding_data['method'].upper()

    # Plot
    if labels is not None:
        scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', alpha=0.7)
        plt.colorbar(scatter, ax=ax, label='Label')
    else:
        ax.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7)

    ax.set_xlabel(f'{method} 1')
    ax.set_ylabel(f'{method} 2')

    # Add metrics
    metrics = embedding_data.get('metrics', {})
    if 'total_variance_explained' in metrics:
        title += f'\nVariance Explained: {metrics["total_variance_explained"]:.3f}'

    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[Path] = None,
    title: str = "Training Curves",
) -> plt.Figure:
    """
    Plot training and validation curves.

    Args:
        history: Dictionary with 'train_loss', 'train_acc', 'val_loss', 'val_acc'
        save_path: Optional path to save
        title: Plot title

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', alpha=0.7)
    if 'val_loss' in history and len(history['val_loss']) > 0:
        # Map validation steps to actual steps
        val_steps = history.get('epochs', [])
        if len(val_steps) == len(history['val_loss']):
            axes[0].plot(val_steps, history['val_loss'], 'o-', label='Val Loss')
        else:
            axes[0].plot(history['val_loss'], 'o-', label='Val Loss')

    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')

    # Accuracy
    axes[1].plot(history['train_acc'], label='Train Acc', alpha=0.7)
    if 'val_acc' in history and len(history['val_acc']) > 0:
        val_steps = history.get('epochs', [])
        if len(val_steps) == len(history['val_acc']):
            axes[1].plot(val_steps, history['val_acc'], 'o-', label='Val Acc')
        else:
            axes[1].plot(history['val_acc'], 'o-', label='Val Acc')

    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_neuron_statistics(
    neuron_stats: Dict[str, Any],
    save_path: Optional[Path] = None,
    title: str = "Neuron Statistics",
) -> plt.Figure:
    """
    Plot neuron activation statistics.

    Args:
        neuron_stats: Dictionary with neuron statistics
        save_path: Optional path to save
        title: Plot title

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Mean activation
    axes[0, 0].hist(neuron_stats['mean'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Mean Activation')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Distribution of Mean Activations')
    axes[0, 0].grid(True, alpha=0.3)

    # Standard deviation
    axes[0, 1].hist(neuron_stats['std'], bins=50, alpha=0.7, edgecolor='black', color='orange')
    axes[0, 1].set_xlabel('Std Deviation')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Distribution of Activation Std')
    axes[0, 1].grid(True, alpha=0.3)

    # Sparsity
    axes[1, 0].hist(neuron_stats['sparsity'], bins=50, alpha=0.7, edgecolor='black', color='green')
    axes[1, 0].set_xlabel('Sparsity')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title(f'Distribution of Sparsity\nDead Neurons: {neuron_stats["n_dead"]}/{neuron_stats["n_active"] + neuron_stats["n_dead"]}')
    axes[1, 0].grid(True, alpha=0.3)

    # Range
    axes[1, 1].scatter(neuron_stats['min'], neuron_stats['max'], alpha=0.5)
    axes[1, 1].plot([min(neuron_stats['min']), max(neuron_stats['max'])],
                    [min(neuron_stats['min']), max(neuron_stats['max'])],
                    'r--', label='Equal range')
    axes[1, 1].set_xlabel('Minimum Activation')
    axes[1, 1].set_ylabel('Maximum Activation')
    axes[1, 1].set_title('Activation Range per Neuron')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def create_compendium_figure(
    analysis_results: Dict[str, Any],
    save_path: Optional[Path] = None,
    title: str = "Analysis Compendium",
) -> plt.Figure:
    """
    Create a multi-panel compendium figure with all key analyses.

    Args:
        analysis_results: Dictionary with all analysis results
        save_path: Optional path to save
        title: Overall title

    Returns:
        matplotlib Figure with subplots
    """
    # Determine grid size based on available results
    n_plots = 0
    plots = []

    if 'fourier' in analysis_results:
        n_plots += 1
        plots.append(('fourier', plot_fourier_spectrum))

    if 'periodic_neurons' in analysis_results:
        n_plots += 1
        plots.append(('periodic_neurons', plot_periodic_neurons))

    if 'attention' in analysis_results:
        n_plots += 1
        plots.append(('attention', plot_attention_patterns))

    if 'embeddings' in analysis_results:
        n_plots += 1
        plots.append(('embeddings', plot_embedding_geometry))

    if 'training' in analysis_results:
        n_plots += 1
        plots.append(('training', plot_training_curves))

    # Create figure
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig = plt.figure(figsize(6 * n_cols, 5 * n_rows))

    for idx, (key, plot_fn) in enumerate(plots):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1)

        # This is a simplified version - actual implementation would call plot functions
        # and add them as subplots
        ax.text(0.5, 0.5, f'{key.upper()}\nAnalysis', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(key.replace('_', ' ').title())

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
