"""
Torus Distance Prediction Task

Train a model to predict the shortest distance between two points on a toroidal grid.
The torus has wrap-around boundaries (Pac-Man style).

Usage:
    python train_torus.py --grid_size 16 --epochs 10000
"""

import argparse
import sys
import json
from pathlib import Path

# Add library to path
library_path = Path(__file__).parent.parent.parent / 'library'
sys.path.insert(0, str(library_path))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

from library.models import tiny_transformer, create_model
from library.training import create_trainer
from library.analysis import (
    compute_fourier_2d,
    position_encoding_analysis,
    analyze_attention_patterns,
    analyze_embedding_geometry,
)
from library.viz import (
    plot_fourier_spectrum_2d,
    plot_attention_patterns,
    plot_embedding_geometry,
    plot_training_curves,
)


class TorusDistanceDataset(Dataset):
    """
    Dataset for predicting torus distances.

    Input: [x1, y1, x2, y2] - two coordinates on grid
    Output: distance - shortest path with wrap-around

    Args:
        grid_size: Size of grid (N x N)
        max_samples: Maximum number of samples (None = all pairs)
        split: 'train', 'val', or 'test'
        seed: Random seed
    """

    def __init__(self, grid_size: int = 16, max_samples: int = None,
                 split: str = 'train', seed: int = 42):
        self.grid_size = grid_size
        self.split = split
        self.seed = seed

        # Generate all possible pairs
        all_pairs = []
        for x1 in range(grid_size):
            for y1 in range(grid_size):
                for x2 in range(grid_size):
                    for y2 in range(grid_size):
                        distance = self.torus_distance(x1, y1, x2, y2)
                        all_pairs.append(((x1, y1, x2, y2), distance))

        # Shuffle and split
        rng = np.random.default_rng(seed)
        indices = np.arange(len(all_pairs))
        rng.shuffle(indices)

        n_train = int(len(indices) * 0.7)
        n_val = int(len(indices) * 0.15)

        if split == 'train':
            indices = indices[:n_train]
        elif split == 'val':
            indices = indices[n_train:n_train + n_val]
        else:
            indices = indices[n_train + n_val:]

        # Subsample if needed
        if max_samples and len(indices) > max_samples:
            indices = rng.choice(indices, max_samples, replace=False)

        self.samples = [all_pairs[i] for i in indices]

    def torus_distance(self, x1: int, y1: int, x2: int, y2: int) -> float:
        """Compute shortest distance on torus with wrap-around."""
        dx = min(abs(x2 - x1), self.grid_size - abs(x2 - x1))
        dy = min(abs(y2 - y1), self.grid_size - abs(y2 - y1))
        return np.sqrt(dx**2 + dy**2)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        (x1, y1, x2, y2), distance = self.samples[idx]

        # Input: 4 tokens representing positions
        input_tokens = torch.tensor([x1, y1, x2, y2], dtype=torch.long)

        # Output: Distance as regression target (for simplicity, discretize)
        # Discretize into bins
        max_dist = np.sqrt(2) * self.grid_size / 2
        n_bins = self.grid_size  # Number of distance bins
        bin_idx = int((distance / max_dist) * n_bins)
        bin_idx = min(bin_idx, n_bins - 1)

        target_tokens = torch.tensor([bin_idx], dtype=torch.long)

        metadata = {
            'task': 'torus_distance',
            'split': self.split,
            'grid_size': self.grid_size,
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'distance': distance,
            'bin_idx': bin_idx,
        }

        return input_tokens, target_tokens, metadata

    def get_vocab_size(self):
        return self.grid_size

    def get_output_size(self):
        return self.grid_size  # Number of distance bins

    def get_seq_len(self):
        return (4, 1)

    def collate_fn(self, batch):
        """Custom collate function for batching."""
        inputs = torch.stack([item[0] for item in batch])
        targets = torch.stack([item[1] for item in batch])
        metadata_list = [item[2] for item in batch]

        return {
            'input': inputs,
            'target': targets,
            'metadata': metadata_list,
        }


def parse_args():
    parser = argparse.ArgumentParser(description='Train torus distance task')

    parser.add_argument('--grid_size', type=int, default=16, help='Size of torus grid')
    parser.add_argument('--max_samples', type=int, default=10000, help='Max samples per split')
    parser.add_argument('--epochs', type=int, default=10000, help='Training steps')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')

    parser.add_argument('--model_size', type=str, default='tiny',
                        choices=['tiny', 'small'])
    parser.add_argument('--pos_enc', type=str, default='learned',
                        choices=['learned', 'sinusoidal'])

    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--save_analysis', action='store_true')
    parser.add_argument('--analysis_interval', type=int, default=2500)

    return parser.parse_args()


def create_model_from_args(args):
    """Create model for torus task."""
    model_kwargs = {
        'vocab_size': args.grid_size,
        'output_size': args.grid_size,
        'pos_enc_type': args.pos_enc,
        'pos_enc_max_val': args.grid_size,
    }

    if args.model_size == 'tiny':
        model = tiny_transformer(**model_kwargs)
    else:
        model = create_model({
            **model_kwargs,
            'd_model': 256,
            'n_layers': 4,
            'n_heads': 8,
            'd_mlp': 1024,
        })

    return model


def analyze_model(model, dataset, args, step, results_dir):
    """Run mechanistic analysis on torus model."""
    print(f"\n{'='*60}")
    print(f"Running torus analysis at step {step}")
    print(f"{'='*60}\n")

    results = {}
    device = next(model.parameters()).device

    # Collect activations for all positions
    # We'll scan through all (x, y) positions and collect embeddings
    grid_size = args.grid_size

    all_embeddings = []
    positions = []

    model.eval()
    with torch.no_grad():
        for x in range(grid_size):
            for y in range(grid_size):
                # Create input: [x, y, 0, 0] (query position)
                input_tensor = torch.tensor([[x, y, 0, 0]]).to(device)

                _ = model(input_tensor, return_cache=True)

                # Get embedding (average over sequence)
                embeddings = model.get_embeddings()  # [1, 4, d_model]
                avg_embedding = embeddings[0].mean(dim=0).cpu().numpy()  # [d_model]

                all_embeddings.append(avg_embedding)
                positions.append((x, y))

    all_embeddings = np.array(all_embeddings)  # [grid_size^2, d_model]

    # Reshape to grid
    embedding_grid = all_embeddings.reshape(grid_size, grid_size, -1)  # [grid_size, grid_size, d_model]

    # Analyze each dimension's 2D Fourier spectrum
    print("Computing 2D Fourier spectra for embedding dimensions...")
    fourier_results = []

    for dim_idx in range(min(20, embedding_grid.shape[-1])):  # Analyze first 20 dims
        dim_activations = embedding_grid[:, :, dim_idx].T  # [grid_size, grid_size]

        freqs_k, freqs_l, power, metrics = compute_fourier_2d(dim_activations)

        fourier_results.append({
            'dimension': dim_idx,
            'dominant_mode_k': metrics['dominant_mode_k'],
            'dominant_mode_l': metrics['dominant_mode_l'],
            'separability_score': metrics['separability_score'],
        })

    results['fourier_2d'] = fourier_results

    # Average separability
    avg_separability = np.mean([r['separability_score'] for r in fourier_results])
    print(f"  Average separability score: {avg_separability:.3f}")

    # Find most separable dimensions
    fourier_results.sort(key=lambda x: x['separability_score'], reverse=True)
    top_dims = fourier_results[:5]
    print(f"  Top separable dimensions:")
    for r in top_dims:
        print(f"    Dim {r['dimension']}: separability={r['separability_score']:.3f}, "
              f"mode=({r['dominant_mode_k']:.2f}, {r['dominant_mode_l']:.2f})")

    # Position encoding analysis (treat as 1D sequence of grid positions)
    positions_1d = np.arange(grid_size * grid_size)
    embedding_tensor = torch.from_numpy(all_embeddings).float().to(device)

    pos_analysis = position_encoding_analysis(embedding_tensor, positions_1d)
    results['position_encoding'] = pos_analysis

    print(f"\n  Periodicity score: {pos_analysis['periodicity_score']:.3f}")
    print(f"  Periodic dimensions: {pos_analysis['n_periodic']}")

    # Embedding geometry (PCA)
    print("\nAnalyzing embedding geometry...")
    embedding_full = torch.from_numpy(all_embeddings).float()

    # Use (x, y) as labels
    labels = np.array([x * grid_size + y for x, y in positions])

    pca_result = analyze_embedding_geometry(
        embedding_full,
        labels=labels,
        method='pca',
        n_components=2,
    )
    results['embeddings_pca'] = pca_result

    print(f"  PCA variance explained: {pca_result['metrics']['total_variance_explained']:.3f}")

    # Save plots
    if args.save_analysis:
        step_dir = results_dir / f'step_{step}'
        step_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving plots to {step_dir}")

        # 2D Fourier spectrum for most separable dimension
        best_dim = fourier_results[0]['dimension']
        dim_activations = embedding_grid[:, :, best_dim].T
        freqs_k, freqs_l, power, metrics = compute_fourier_2d(dim_activations)

        plot_fourier_spectrum_2d(
            freqs_k, freqs_l, power, metrics,
            save_path=step_dir / 'fourier_2d.png',
            title=f'Torus Distance - 2D Fourier (Dim {best_dim})',
        )
        print("  ✓ Saved fourier_2d.png")

        # Embedding PCA
        plot_embedding_geometry(
            pca_result,
            save_path=step_dir / 'embeddings_pca.png',
            title=f'Torus Distance - Embedding PCA',
        )
        print("  ✓ Saved embeddings_pca.png")

        # Grid visualization of embeddings (average over dimensions)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot average activation across first few dimensions
        for idx, ax in enumerate(axes.flat):
            dim_to_plot = min(idx * 5, embedding_grid.shape[-1] - 1)
            dim_grid = embedding_grid[:, :, dim_to_plot]

            im = ax.imshow(dim_grid, cmap='viridis', origin='lower')
            ax.set_title(f'Dimension {dim_to_plot}')
            ax.set_xlabel('y')
            ax.set_ylabel('x')
            plt.colorbar(im, ax=ax)

        plt.tight_layout()
        plt.savefig(step_dir / 'embedding_grid.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved embedding_grid.png")

    # Save results
    results_file = results_dir / f'step_{step}' / 'analysis_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print("  ✓ Saved analysis_results.json")

    return results


def main():
    import matplotlib.pyplot as plt

    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create directories
    checkpoint_dir = Path(args.checkpoint_dir) / 'torus'
    results_dir = Path(args.results_dir) / 'periodic_2d' / 'torus_distance'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"Training torus distance task")
    print(f"{'='*60}\n")

    # Create datasets
    print("Creating datasets...")
    train_dataset = TorusDistanceDataset(
        grid_size=args.grid_size,
        max_samples=args.max_samples,
        split='train',
        seed=args.seed,
    )
    val_dataset = TorusDistanceDataset(
        grid_size=args.grid_size,
        max_samples=args.max_samples // 5,
        split='val',
        seed=args.seed,
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Grid size: {args.grid_size}x{args.grid_size}\n")

    # Create model
    print("Creating model...")
    model = create_model_from_args(args)
    print(f"  Model: {args.model_size}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
    )

    # Create trainer
    import torch.optim as optim

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    from library.training import Trainer

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=args.device,
        checkpoint_dir=str(checkpoint_dir),
        task_name='torus_distance',
        log_interval=100,
        eval_interval=args.analysis_interval,
        save_interval=args.analysis_interval,
    )

    # Add analysis callback
    def analysis_callback(model, step, val_metrics):
        if step % args.analysis_interval == 0 and step > 0:
            analyze_model(model, val_dataset, args, step, results_dir)

    trainer.add_callback(analysis_callback)

    # Train
    print(f"Training for {args.epochs} steps...\n")
    history = trainer.train(n_steps=args.epochs)

    # Final analysis
    print("\nRunning final analysis...")
    analyze_model(model, val_dataset, args, args.epochs, results_dir)

    # Plot training curves
    if args.save_analysis:
        plot_training_curves(
            history,
            save_path=results_dir / 'training_curves.png',
            title='Torus Distance - Training Curves',
        )
        print(f"\n✓ Saved training_curves.png to {results_dir}")

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
