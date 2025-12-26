"""
Sphere Geodesic Distance Task

Train a model to predict the geodesic (great circle) distance between two points on a sphere.

This tests whether the model can discover intrinsic curvature and spherical geometry.

Geodesic distance on sphere:
d = arccos(sin(φ1)sin(φ2) + cos(φ1)cos(φ2)cos(Δλ))

Where (φ, λ) are (latitude, longitude).

Usage:
    python train_sphere.py --n_samples 10000 --epochs 10000
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
from library.training import Trainer
from library.analysis import (
    compute_fourier_1d,
    analyze_attention_patterns,
    analyze_embedding_geometry,
)
from library.viz import (
    plot_fourier_spectrum,
    plot_attention_patterns,
    plot_embedding_geometry,
    plot_training_curves,
)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class SphereDistanceDataset(Dataset):
    """
    Dataset for sphere geodesic distance prediction.

    Points are represented in spherical coordinates (lat, lon).
    Task: Predict geodesic distance (great circle distance).

    Args:
        n_samples: Number of point pairs
        split: 'train', 'val', or 'test'
        seed: Random seed
    """

    def __init__(self, n_samples: int = 10000, split: str = 'train', seed: int = 42):
        self.n_samples = n_samples
        self.split = split
        self.seed = seed

        # Generate samples
        rng = np.random.default_rng(seed)

        # Generate point pairs
        n_total = n_samples * 3  # Generate extra for train/val/test split

        lat1 = np.arccos(1 - 2 * rng.random(n_total)) - np.pi / 2  # Uniform on sphere
        lon1 = 2 * np.pi * rng.random(n_total)
        lat2 = np.arccos(1 - 2 * rng.random(n_total)) - np.pi / 2
        lon2 = 2 * np.pi * rng.random(n_total)

        # Compute geodesic distances
        distances = self.geodesic_distance(lat1, lon1, lat2, lon2)

        # Discretize for classification
        n_bins = 64
        max_dist = np.pi  # Maximum distance on unit sphere
        bin_indices = (distances / max_dist * n_bins).astype(int)
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        # Package data
        self.data = list(zip(
            zip(lat1, lon1, lat2, lon2),
            bin_indices,
            distances
        ))

        # Split
        n_train = int(len(self.data) * 0.7)
        n_val = int(len(self.data) * 0.15)

        if split == 'train':
            self.data = self.data[:n_train]
        elif split == 'val':
            self.data = self.data[n_train:n_train + n_val]
        else:
            self.data = self.data[n_train + n_val:]

    def geodesic_distance(self, lat1: np.ndarray, lon1: np.ndarray,
                         lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
        """
        Compute geodesic distance on sphere using haversine formula.

        Args:
            lat1, lat2: Latitudes in radians
            lon1, lon2: Longitudes in radians

        Returns:
            Distances (same shape as inputs)
        """
        # Convert to Cartesian for numerical stability
        x1 = np.cos(lat1) * np.cos(lon1)
        y1 = np.cos(lat1) * np.sin(lon1)
        z1 = np.sin(lat1)

        x2 = np.cos(lat2) * np.cos(lon2)
        y2 = np.cos(lat2) * np.sin(lon2)
        z2 = np.sin(lat2)

        # Dot product
        dot = x1 * x2 + y1 * y2 + z1 * z2

        # Clip for numerical stability
        dot = np.clip(dot, -1.0, 1.0)

        # Arccos gives distance
        return np.arccos(dot)

    def spherical_to_cartesian(self, lat: float, lon: float) -> tuple:
        """Convert spherical to cartesian coordinates."""
        x = np.cos(lat) * np.cos(lon)
        y = np.cos(lat) * np.sin(lon)
        z = np.sin(lat)
        return x, y, z

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        (lat1, lon1, lat2, lon2), bin_idx, distance = self.data[idx]

        # Discretize coordinates to grid
        grid_size = 64
        lat_min, lat_max = -np.pi / 2, np.pi / 2
        lon_min, lon_max = -np.pi, np.pi

        lat1_idx = int((lat1 - lat_min) / (lat_max - lat_min) * (grid_size - 1))
        lon1_idx = int((lon1 - lon_min) / (lon_max - lon_min) * (grid_size - 1))
        lat2_idx = int((lat2 - lat_min) / (lat_max - lat_min) * (grid_size - 1))
        lon2_idx = int((lon2 - lon_min) / (lon_max - lon_min) * (grid_size - 1))

        # Input: [lat1, lon1, lat2, lon2]
        input_tokens = torch.tensor([lat1_idx, lon1_idx, lat2_idx, lon2_idx], dtype=torch.long)
        target_tokens = torch.tensor([bin_idx], dtype=torch.long)

        metadata = {
            'task': 'sphere_distance',
            'split': self.split,
            'lat1': float(lat1), 'lon1': float(lon1),
            'lat2': float(lat2), 'lon2': float(lon2),
            'distance': float(distance),
            'bin_idx': int(bin_idx),
        }

        return input_tokens, target_tokens, metadata

    def get_vocab_size(self):
        return 64  # Grid size for discretization

    def get_output_size(self):
        return 64  # Number of distance bins

    def get_seq_len(self):
        return (4, 1)


def parse_args():
    parser = argparse.ArgumentParser(description='Train sphere distance task')

    parser.add_argument('--n_samples', type=int, default=10000, help='Samples per split')
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
    """Create model for sphere task."""
    vocab_size = 64

    model_kwargs = {
        'vocab_size': vocab_size,
        'output_size': vocab_size,
        'pos_enc_type': args.pos_enc,
        'pos_enc_max_val': vocab_size,
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
    """Run mechanistic analysis on sphere distance model."""
    print(f"\n{'='*60}")
    print(f"Running sphere analysis at step {step}")
    print(f"{'='*60}\n")

    results = {}
    device = next(model.parameters()).device

    # Sample points uniformly on sphere
    n_sample_points = 512
    rng = np.random.default_rng(42)

    lat_samples = np.arccos(1 - 2 * rng.random(n_sample_points)) - np.pi / 2
    lon_samples = 2 * np.pi * rng.random(n_sample_points)

    # Get embeddings for each point (as first point in pair)
    all_embeddings = []
    positions = []

    model.eval()
    with torch.no_grad():
        for lat, lon in zip(lat_samples, lon_samples):
            # Discretize
            grid_size = 64
            lat_min, lat_max = -np.pi / 2, np.pi / 2
            lon_min, lon_max = -np.pi, np.pi

            lat_idx = int((lat - lat_min) / (lat_max - lat_min) * (grid_size - 1))
            lon_idx = int((lon - lon_min) / (lon_max - lon_min) * (grid_size - 1))

            # Create input with this point as first point
            input_tensor = torch.tensor([[lat_idx, lon_idx, 0, 0]]).to(device)

            _ = model(input_tensor, return_cache=True)

            # Get embedding
            embeddings = model.get_embeddings()
            avg_embedding = embeddings[0].mean(dim=0).cpu().numpy()

            all_embeddings.append(avg_embedding)
            positions.append((lat, lon))

    all_embeddings = np.array(all_embeddings)

    # Analyze embedding geometry
    print("Analyzing embedding geometry...")

    # Use latitude as label (should see organization along this dimension)
    labels = lat_samples

    pca_result = analyze_embedding_geometry(
        torch.from_numpy(all_embeddings).float(),
        labels=labels,
        method='pca',
        n_components=2,
    )
    results['embeddings_pca'] = pca_result

    print(f"  PCA variance explained: {pca_result['metrics']['total_variance_explained']:.3f}")

    # Check correlation with latitude (tests understanding of sphere geometry)
    pc1 = pca_result['reduced_embeddings'][:, 0]
    lat_correlation = np.corrcoef(pc1, lat_samples)[0, 1]
    print(f"  PC1 correlation with latitude: {lat_correlation:.3f}")

    results['latitude_correlation'] = float(lat_correlation)

    # Check if embeddings capture spherical structure
    # Map embeddings to 3D and see if they lie on sphere
    from sklearn.decomposition import PCA

    pca_3d = PCA(n_components=3)
    embeddings_3d = pca_3d.fit_transform(all_embeddings)

    # Check if points lie on sphere (norm should be roughly constant)
    norms = np.linalg.norm(embeddings_3d, axis=1)
    norm_var = np.var(norms) / np.mean(norms)**2
    print(f"  Normalized variance of 3D embeddings: {norm_var:.3f}")
    print(f"    (Lower values suggest more spherical structure)")

    results['spherical_structure_score'] = float(1.0 - min(norm_var, 1.0))

    # Visualize embeddings mapped to sphere
    if args.save_analysis:
        step_dir = results_dir / f'step_{step}'
        step_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving plots to {step_dir}")

        # 3D scatter of embeddings (PCA to 3D)
        fig = plt.figure(figsize=(12, 5))

        ax1 = fig.add_subplot(121, projection='3d')
        scatter1 = ax1.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2],
                              c=lat_samples, cmap='coolwarm', s=20, alpha=0.6)
        ax1.set_xlabel('PC1')
        ax1.set_ylabel('PC2')
        ax1.set_zlabel('PC3')
        ax1.set_title('Embeddings (PCA to 3D)\nColored by Latitude')
        plt.colorbar(scatter1, ax=ax1, label='Latitude')

        # True sphere for comparison
        ax2 = fig.add_subplot(122, projection='3d')

        # Convert spherical samples to cartesian
        x = np.cos(lat_samples) * np.cos(lon_samples)
        y = np.cos(lat_samples) * np.sin(lon_samples)
        z = np.sin(lat_samples)

        scatter2 = ax2.scatter(x, y, z, c=lat_samples, cmap='coolwarm', s=20, alpha=0.6)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('z')
        ax2.set_title('True Sphere Positions\nColored by Latitude')
        plt.colorbar(scatter2, ax=ax2, label='Latitude')

        plt.tight_layout()
        plt.savefig(step_dir / 'embeddings_3d.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved embeddings_3d.png")

        # 2D PCA colored by latitude
        plot_embedding_geometry(
            pca_result,
            save_path=step_dir / 'embeddings_pca.png',
            title='Sphere Distance - Embedding PCA',
        )
        print("  ✓ Saved embeddings_pca.png")

        # Distance prediction accuracy vs true distance
        # Sample some test pairs
        n_test = 100
        test_lat1 = np.arccos(1 - 2 * rng.random(n_test)) - np.pi / 2
        test_lon1 = 2 * np.pi * rng.random(n_test)
        test_lat2 = np.arccos(1 - 2 * rng.random(n_test)) - np.pi / 2
        test_lon2 = 2 * np.pi * rng.random(n_test)

        true_distances = dataset.geodesic_distance(test_lat1, test_lon1, test_lat2, test_lon2)

        # Get predictions
        pred_distances = []
        with torch.no_grad():
            for lat1, lon1, lat2, lon2 in zip(test_lat1, test_lon1, test_lat2, test_lon2):
                # Discretize
                lat1_idx = int(((lat1 + np.pi/2) / np.pi) * 63)
                lon1_idx = int(((lon1 + np.pi) / (2*np.pi)) * 63)
                lat2_idx = int(((lat2 + np.pi/2) / np.pi) * 63)
                lon2_idx = int(((lon2 + np.pi) / (2*np.pi)) * 63)

                input_tensor = torch.tensor([[lat1_idx, lon1_idx, lat2_idx, lon2_idx]]).to(device)
                logits = model(input_tensor)
                pred_bin = logits[0, 0, :].argmax().item()

                # Convert bin to distance
                pred_dist = pred_bin / 64 * np.pi
                pred_distances.append(pred_dist)

        pred_distances = np.array(pred_distances)

        # Plot predicted vs true
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(true_distances, pred_distances, alpha=0.5)
        ax.plot([0, np.pi], [0, np.pi], 'r--', label='Perfect prediction')
        ax.set_xlabel('True Geodesic Distance')
        ax.set_ylabel('Predicted Distance')
        ax.set_title('Distance Prediction Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Compute correlation
        correlation = np.corrcoef(true_distances, pred_distances)[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(step_dir / 'distance_accuracy.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved distance_accuracy.png")

    # Save results
    results_file = results_dir / f'step_{step}' / 'analysis_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print("  ✓ Saved analysis_results.json")

    return results


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create directories
    checkpoint_dir = Path(args.checkpoint_dir) / 'sphere'
    results_dir = Path(args.results_dir) / 'manifold' / 'sphere_distance'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"Training sphere distance task")
    print(f"{'='*60}\n")

    # Create datasets
    print("Creating datasets...")
    train_dataset = SphereDistanceDataset(
        n_samples=args.n_samples,
        split='train',
        seed=args.seed,
    )
    val_dataset = SphereDistanceDataset(
        n_samples=args.n_samples // 5,
        split='val',
        seed=args.seed,
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}\n")

    # Create model
    print("Creating model...")
    model = create_model_from_args(args)
    print(f"  Model: {args.model_size}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Create trainer
    import torch.optim as optim

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=args.device,
        checkpoint_dir=str(checkpoint_dir),
        task_name='sphere_distance',
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
            title='Sphere Distance - Training Curves',
        )
        print(f"\n✓ Saved training_curves.png to {results_dir}")

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
