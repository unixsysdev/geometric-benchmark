"""
Unified Mechanistic Analysis Pipeline for Geometric Benchmark

Provides Fourier analysis, periodic neuron detection, and representation analysis.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats
from scipy.fft import fft, fft2, fftfreq
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


def compute_fourier_1d(
    activations: np.ndarray,
    sampling_rate: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Compute 1D Fourier transform of activations.

    Args:
        activations: Array of shape [n_positions] or [n_neurons, n_positions]
        sampling_rate: Sampling rate (positions per unit)

    Returns:
        frequencies: Frequency array
        power: Power spectrum
        metrics: Dictionary with peak frequency, power in top-k, etc.
    """
    # Ensure 2D
    if activations.ndim == 1:
        activations = activations.reshape(1, -1)

    n_neurons, n_positions = activations.shape

    # Compute FFT
    fft_vals = fft(activations, axis=1)
    power = np.abs(fft_vals) ** 2

    # Frequency array
    frequencies = fftfreq(n_positions, d=1.0/sampling_rate)

    # Keep only positive frequencies
    pos_freq_idx = frequencies >= 0
    frequencies = frequencies[pos_freq_idx]
    power = power[:, pos_freq_idx]

    # Average power across neurons
    avg_power = power.mean(axis=0)

    # Find peaks
    peaks, properties = signal.find_peaks(avg_power, height=np.max(avg_power) * 0.1)

    # Sort by power
    peak_idx = peaks[np.argsort(properties['peak_heights'])[::-1]]

    metrics = {
        'dominant_frequency': frequencies[peak_idx[0]] if len(peak_idx) > 0 else 0,
        'dominant_power': avg_power[peak_idx[0]] if len(peak_idx) > 0 else 0,
        'total_power': avg_power.sum(),
        'peaks': list(zip(frequencies[peak_idx], avg_power[peak_idx])),
    }

    # Top-k power concentration
    for k in [1, 3, 5, 10]:
        if len(peak_idx) >= k:
            top_k_power = avg_power[peak_idx[:k]].sum()
            metrics[f'top{k}_power_ratio'] = top_k_power / metrics['total_power']
        else:
            metrics[f'top{k}_power_ratio'] = 0

    return frequencies, avg_power, metrics


def compute_fourier_2d(
    activations: np.ndarray,
    sampling_rate: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Compute 2D Fourier transform of activations (for torus/grid tasks).

    Args:
        activations: Array of shape [n_neurons, height, width]
        sampling_rate: Sampling rate

    Returns:
        freqs_k, freqs_l: Frequency arrays for each dimension
        power: 2D power spectrum (averaged over neurons)
        metrics: Dictionary with analysis results
    """
    # Ensure 3D
    if activations.ndim == 2:
        activations = activations.reshape(1, *activations.shape)

    n_neurons, height, width = activations.shape

    # Compute 2D FFT
    fft_vals = fft2(activations, axes=(1, 2))
    power = np.abs(fft_vals) ** 2

    # Shift zero frequency to center
    power = np.fft.fftshift(power, axes=(1, 2))

    # Frequency arrays
    freqs_k = fftfreq(height, d=1.0/sampling_rate)
    freqs_l = fftfreq(width, d=1.0/sampling_rate)

    freqs_k = np.fft.fftshift(freqs_k)
    freqs_l = np.fft.fftshift(freqs_l)

    # Average over neurons
    avg_power = power.mean(axis=0)

    # Find peaks
    peaks_2d = signal.argrelmax(avg_power, axis=0) + signal.argrelmax(avg_power, axis=1)

    # Find dominant mode
    max_idx = np.unravel_index(np.argmax(avg_power), avg_power.shape)
    metrics = {
        'dominant_mode_k': freqs_k[max_idx[0]],
        'dominant_mode_l': freqs_l[max_idx[1]],
        'dominant_power': avg_power[max_idx],
        'total_power': avg_power.sum(),
    }

    # Check separability (correlation with separable basis)
    # Perform SVD and check if first singular component dominates
    U, S, Vt = np.linalg.svd(avg_power)
    metrics['separability_score'] = S[0] / S.sum() if len(S) > 0 else 0

    return freqs_k, freqs_l, avg_power, metrics


def fit_sinusoid(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[float, Dict[str, Any]]:
    """
    Fit sinusoid to data and compute R² score.

    Model: y = A*sin(2πfx + φ) + B

    Args:
        x: Input positions
        y: Activations

    Returns:
        r_squared: R² score
        params: Dictionary with fitted parameters
    """
    # Normalize x to [0, 1]
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)

    # Design matrix: sin(2πfx), cos(2πfx), constant
    # Try multiple frequencies and pick best
    best_r2 = -np.inf
    best_params = None

    for freq_mult in range(1, 11):
        # Features
        sin_features = np.sin(2 * np.pi * freq_mult * x_norm)
        cos_features = np.cos(2 * np.pi * freq_mult * x_norm)
        X = np.stack([sin_features, cos_features, np.ones_like(x)], axis=1)

        # Linear regression
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

        # Predictions
        y_pred = X @ coeffs

        # R²
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - (ss_res / (ss_tot + 1e-8))

        if r2 > best_r2:
            best_r2 = r2

            # Extract amplitude, phase, offset
            A_sin, A_cos, B = coeffs
            amplitude = np.sqrt(A_sin**2 + A_cos**2)
            phase = np.arctan2(A_cos, A_sin)

            best_params = {
                'frequency': freq_mult,
                'amplitude': amplitude,
                'phase': phase,
                'offset': B,
                'A_sin': A_sin,
                'A_cos': A_cos,
            }

    return best_r2, best_params


def detect_periodic_neurons(
    activations: np.ndarray,
    positions: np.ndarray,
    r2_threshold: float = 0.8,
) -> List[Dict[str, Any]]:
    """
    Detect neurons with periodic activations.

    Args:
        activations: Array of shape [n_neurons, n_positions]
        positions: Position values
        r2_threshold: Minimum R² to be considered periodic

    Returns:
        List of dictionaries with periodic neuron info
    """
    n_neurons = activations.shape[0]
    periodic_neurons = []

    for neuron_idx in range(n_neurons):
        r2, params = fit_sinusoid(positions, activations[neuron_idx])

        if r2 >= r2_threshold:
            periodic_neurons.append({
                'neuron_idx': neuron_idx,
                'r_squared': r2,
                'frequency': params['frequency'],
                'amplitude': params['amplitude'],
                'phase': params['phase'],
            })

    # Sort by R²
    periodic_neurons.sort(key=lambda x: x['r_squared'], reverse=True)

    return periodic_neurons


def analyze_attention_patterns(
    attention_weights: torch.Tensor,
    layer: int,
    head: int,
) -> Dict[str, Any]:
    """
    Analyze attention pattern for a specific layer/head.

    Args:
        attention_weights: [n_layers, n_heads, batch, seq_len, seq_len] or [batch, seq_len, seq_len]
        layer: Layer index
        head: Head index

    Returns:
        Dictionary with attention metrics
    """
    # Handle different input shapes
    if attention_weights.ndim == 5:
        attn = attention_weights[layer, head]  # [batch, seq_len, seq_len]
    elif attention_weights.ndim == 4:
        attn = attention_weights[layer, head]  # [batch, seq_len, seq_len]
    else:
        attn = attention_weights  # [batch, seq_len, seq_len]

    # Average over batch
    attn_mean = attn.mean(dim=0).cpu().numpy()

    seq_len = attn_mean.shape[0]

    # Entropy (uniformity)
    row_sums = attn_mean.sum(axis=1, keepdims=True)
    attn_norm = attn_mean / (row_sums + 1e-8)
    entropy = -(attn_norm * np.log(attn_norm + 1e-8)).sum(axis=1)

    # Sparsity (fraction of attention above threshold)
    threshold = 1.0 / seq_len
    sparsity = (attn_mean > threshold).astype(float).mean(axis=1)

    # Position bias (diagonal dominance)
    diagonal = np.diag(attn_mean)
    off_diagonal = attn_mean - np.diag(diagonal)
    diagonal_bias = diagonal.mean() / (off_diagonal.mean() + 1e-8)

    # Attention pattern type
    if diagonal_bias > 2:
        pattern_type = 'local'
    elif entropy.mean() > np.log(seq_len) * 0.9:
        pattern_type = 'uniform'
    elif sparsity.mean() < 0.3:
        pattern_type = 'focused'
    else:
        pattern_type = 'mixed'

    return {
        'entropy': entropy.tolist(),
        'entropy_mean': float(entropy.mean()),
        'sparsity': sparsity.tolist(),
        'sparsity_mean': float(sparsity.mean()),
        'diagonal_bias': float(diagonal_bias),
        'pattern_type': pattern_type,
        'attention_matrix': attn_mean,
    }


def analyze_embedding_geometry(
    embeddings: torch.Tensor,
    labels: Optional[np.ndarray] = None,
    method: str = 'pca',
    n_components: int = 2,
) -> Dict[str, Any]:
    """
    Analyze embedding geometry using dimensionality reduction.

    Args:
        embeddings: [n_samples, d_model] embedding vectors
        labels: Optional labels for coloring
        method: 'pca', 'tsne', or 'umap'
        n_components: Number of components

    Returns:
        Dictionary with reduced embeddings and metrics
    """
    embeddings_np = embeddings.detach().cpu().numpy()

    # Dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=n_components)
        reduced = reducer.fit_transform(embeddings_np)
        explained_variance = reducer.explained_variance_ratio_.tolist()

        metrics = {
            'explained_variance_ratio': explained_variance,
            'total_variance_explained': sum(explained_variance),
        }

    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42)
        reduced = reducer.fit_transform(embeddings_np)
        metrics = {'perplexity': reducer.perplexity if hasattr(reducer, 'perplexity') else 30}

    elif method == 'umap':
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        reduced = reducer.fit_transform(embeddings_np)
        metrics = {}

    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        'reduced_embeddings': reduced,
        'method': method,
        'metrics': metrics,
        'labels': labels,
    }


def compute_neuron_statistics(
    activations: torch.Tensor,
) -> Dict[str, Any]:
    """
    Compute basic statistics for neuron activations.

    Args:
        activations: [batch, seq_len, d_model] or [n_samples, d_model]

    Returns:
        Dictionary with neuron statistics
    """
    # Flatten if needed
    if activations.ndim == 3:
        activations = activations.reshape(-1, activations.shape[-1])

    activations_np = activations.detach().cpu().numpy()

    # Per-neuron statistics
    mean = activations_np.mean(axis=0)
    std = activations_np.std(axis=0)
    max_val = activations_np.max(axis=0)
    min_val = activations_np.min(axis=0)

    # Sparsity (fraction of near-zero activations)
    threshold = 1e-3
    sparsity = (np.abs(activations_np) < threshold).astype(float).mean(axis=0)

    # Dead neurons (never activate)
    dead = (max_val < threshold)

    return {
        'mean': mean.tolist(),
        'std': std.tolist(),
        'max': max_val.tolist(),
        'min': min_val.tolist(),
        'sparsity': sparsity.tolist(),
        'dead_neurons': dead.tolist(),
        'n_dead': int(dead.sum()),
        'n_active': int((~dead).sum()),
    }


def compute_cross_layer_similarity(
    hidden_states: List[torch.Tensor],
    layer: int,
) -> Dict[str, Any]:
    """
    Compute similarity between consecutive layers.

    Args:
        hidden_states: List of [batch, seq_len, d_model] per layer
        layer: Layer index to analyze

    Returns:
        Dictionary with similarity metrics
    """
    if layer >= len(hidden_states) - 1:
        return {}

    h1 = hidden_states[layer]
    h2 = hidden_states[layer + 1]

    # Flatten
    h1_flat = h1.reshape(-1, h1.shape[-1])
    h2_flat = h2.reshape(-1, h2.shape[-1])

    # Cosine similarity
    h1_norm = F.normalize(h1_flat, dim=1)
    h2_norm = F.normalize(h2_flat, dim=1)
    cosine_sim = (h1_norm * h2_norm).sum(dim=1).mean().item()

    # CCA (simplified, just correlation of projections)
    # Project h2 onto h1's PCA space
    h1_np = h1_flat.detach().cpu().numpy()
    h2_np = h2_flat.detach().cpu().numpy()

    pca = PCA(n_components=min(10, h1_np.shape[1]))
    h1_pca = pca.fit_transform(h1_np)
    h2_pca = pca.transform(h2_np)

    # Correlation
    corr_matrix = np.corrcoef(h1_pca.T, h2_pca.T)
    cca_score = np.trace(corr_matrix[:h1_pca.shape[1], h1_pca.shape[1]:]) / h1_pca.shape[1]

    return {
        'cosine_similarity': cosine_sim,
        'cca_score': float(cca_score),
    }


def position_encoding_analysis(
    embeddings: torch.Tensor,
    positions: np.ndarray,
) -> Dict[str, Any]:
    """
    Analyze if embeddings encode position periodically.

    Args:
        embeddings: [n_positions, d_model] or [batch, n_positions, d_model]
        positions: Position values (e.g., [0, 1, ..., p-1] for mod-p)

    Returns:
        Dictionary with periodicity analysis
    """
    # Average over batch if needed
    if embeddings.ndim == 3:
        embeddings = embeddings.mean(dim=0)

    embeddings_np = embeddings.detach().cpu().numpy()
    n_positions, d_model = embeddings_np.shape

    # For each dimension, test periodicity
    periodic_dims = []

    for dim in range(d_model):
        activations = embeddings_np[:, dim]
        r2, params = fit_sinusoid(positions, activations)

        if r2 > 0.5:  # Threshold
            periodic_dims.append({
                'dimension': dim,
                'r_squared': r2,
                'frequency': params['frequency'],
                'amplitude': params['amplitude'],
                'phase': params['phase'],
            })

    # Sort by R²
    periodic_dims.sort(key=lambda x: x['r_squared'], reverse=True)

    # Overall periodicity score
    n_strong_periodic = sum(1 for d in periodic_dims if d['r_squared'] > 0.8)
    periodicity_score = n_strong_periodic / d_model

    return {
        'periodic_dimensions': periodic_dims,
        'n_periodic': len(periodic_dims),
        'n_strong_periodic': n_strong_periodic,
        'periodicity_score': periodicity_score,
        'top_frequency': periodic_dims[0]['frequency'] if periodic_dims else None,
    }
