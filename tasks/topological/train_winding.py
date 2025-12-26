"""
Winding Number Classification Task

Train a model to predict how many times a curve wraps around the origin.

The winding number is a topological invariant — it doesn't change under
continuous deformation of the curve.

Usage:
    python train_winding.py --max_winding 5 --n_curves_per_class 1000 --epochs 10000
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
    detect_periodic_neurons,
    analyze_attention_patterns,
    analyze_embedding_geometry,
)
from library.viz import (
    plot_fourier_spectrum,
    plot_periodic_neurons,
    plot_attention_patterns,
    plot_embedding_geometry,
    plot_training_curves,
)
import matplotlib.pyplot as plt


class WindingNumberDataset(Dataset):
    """
    Dataset for winding number classification.

    Each sample is a closed curve represented as a sequence of points.
    Task: Predict the winding number around the origin.

    Curves are generated using epicycles (sum of rotating vectors).

    Args:
        max_winding: Maximum absolute winding number
        n_curves_per_class: Number of curves per winding number
        n_points: Number of points per curve
        split: 'train', 'val', or 'test'
        seed: Random seed
    """

    def __init__(
        self,
        max_winding: int = 5,
        n_curves_per_class: int = 1000,
        n_points: int = 32,
        split: str = 'train',
        seed: int = 42,
    ):
        self.max_winding = max_winding
        self.n_points = n_points
        self.split = split
        self.seed = seed

        # Generate curves
        rng = np.random.default_rng(seed)

        self.curves = []
        self.labels = []

        for winding in range(-max_winding, max_winding + 1):
            for _ in range(n_curves_per_class):
                # Generate curve with this winding number
                curve = self.generate_curve(winding, rng)
                self.curves.append(curve)
                self.labels.append(winding + max_winding)  # Shift to be non-negative

        # Shuffle
        indices = np.arange(len(self.curves))
        rng.shuffle(indices)

        # Split
        n_train = int(len(indices) * 0.7)
        n_val = int(len(indices) * 0.15)

        if split == 'train':
            indices = indices[:n_train]
        elif split == 'val':
            indices = indices[n_train:n_train + n_val]
        else:
            indices = indices[n_train + n_val:]

        self.curves = [self.curves[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]

        # Compute discretized coordinates (quantize to grid)
        self.vocab_size = 64  # Grid size for quantization

    def generate_curve(self, winding: int, rng: np.random.Generator) -> np.ndarray:
        """
        Generate a closed curve with specified winding number using epicycles.

        Args:
            winding: Target winding number
            rng: Random number generator

        Returns:
            Array of shape [n_points, 2] with (x, y) coordinates
        """
        # Base radius
        r0 = rng.uniform(0.5, 1.0)

        # Add epicycles to create variation while preserving winding number
        n_epicycles = rng.integers(1, 4)
        phases = rng.uniform(0, 2 * np.pi, n_epicycles)

        # Generate points
        thetas = np.linspace(0, 2 * np.pi, self.n_points, endpoint=False)

        curve = []
        for theta in thetas:
            # Base circle (contributes to winding number)
            x = r0 * np.cos(theta) * (abs(winding) + 1)
            y = r0 * np.sin(theta) * (abs(winding) + 1)

            # Add epicycles (don't change winding number if small enough)
            for i, phase in enumerate(phases):
                freq = i + 2
                amp = r0 / (freq * freq)
                x += amp * np.cos(freq * theta + phase)
                y += amp * np.sin(freq * theta + phase)

            # Reverse direction for negative winding
            if winding < 0:
                x, y = x, -y

            curve.append([x, y])

        return np.array(curve)

    def compute_winding_number(self, curve: np.ndarray) -> int:
        """Compute winding number from curve coordinates."""
        total_angle = 0.0

        for i in range(len(curve)):
            p1 = curve[i]
            p2 = curve[(i + 1) % len(curve)]

            # Angle from origin
            angle1 = np.arctan2(p1[1], p1[0])
            angle2 = np.arctan2(p2[1], p2[0])

            # Compute angle difference (handle wrap-around)
            diff = angle2 - angle1
            if diff > np.pi:
                diff -= 2 * np.pi
            elif diff < -np.pi:
                diff += 2 * np.pi

            total_angle += diff

        return int(np.round(total_angle / (2 * np.pi)))

    def __len__(self):
        return len(self.curves)

    def __getitem__(self, idx):
        curve = self.curves[idx]
        label = self.labels[idx]

        # Quantize curve to grid
        # Normalize to [0, vocab_size-1]
        coord_range = 3.0  # Approximate range of curves
        curve_norm = (curve + coord_range) / (2 * coord_range)
        curve_quantized = np.clip(
            (curve_norm * (self.vocab_size - 1)).astype(int),
            0, self.vocab_size - 1
        )

        # Flatten to sequence: [x0, y0, x1, y1, ...]
        input_tokens = torch.tensor(curve_quantized.flatten(), dtype=torch.long)
        target_tokens = torch.tensor([label], dtype=torch.long)

        metadata = {
            'task': 'winding_number',
            'split': self.split,
            'max_winding': self.max_winding,
            'winding': label - self.max_winding,
            'curve': curve.tolist(),
        }

        return input_tokens, target_tokens, metadata

    def get_vocab_size(self):
        return self.vocab_size

    def get_output_size(self):
        return 2 * self.max_winding + 1

    def get_seq_len(self):
        return (2 * self.n_points, 1)

    def collate_fn(self, batch):
        """Custom collate with padding."""
        inputs = [item[0] for item in batch]
        targets = torch.stack([item[1] for item in batch])
        metadata_list = [item[2] for item in batch]

        # Pad inputs to same length
        max_len = max(inp.shape[0] for inp in inputs)
        inputs_padded = []
        for inp in inputs:
            padded = F.pad(inp, (0, max_len - inp.shape[0]), value=0)
            inputs_padded.append(padded)

        inputs_padded = torch.stack(inputs_padded)

        return {
            'input': inputs_padded,
            'target': targets,
            'metadata': metadata_list,
        }


def parse_args():
    parser = argparse.ArgumentParser(description='Train winding number task')

    parser.add_argument('--max_winding', type=int, default=5, help='Max winding number')
    parser.add_argument('--n_curves', type=int, default=1000, help='Curves per class')
    parser.add_argument('--n_points', type=int, default=32, help='Points per curve')
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
    """Create model for winding number task."""
    vocab_size = 64
    output_size = 2 * args.max_winding + 1

    model_kwargs = {
        'vocab_size': vocab_size,
        'output_size': output_size,
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
    """Run mechanistic analysis on winding number model."""
    print(f"\n{'='*60}")
    print(f"Running winding number analysis at step {step}")
    print(f"{'='*60}\n")

    results = {}
    device = next(model.parameters()).device

    # Collect embeddings for different winding numbers
    winding_to_embeddings = {}
    winding_to_curves = {}

    model.eval()
    with torch.no_grad():
        # Sample curves from each winding number class
        samples_per_class = min(20, len(dataset) // (2 * args.max_winding + 1))

        for winding_idx in range(2 * args.max_winding + 1):
            winding = winding_idx - args.max_winding

            embeddings_list = []
            curves_list = []

            count = 0
            for idx in range(len(dataset)):
                if count >= samples_per_class:
                    break

                input_tokens, target_tokens, metadata = dataset[idx]

                if metadata['winding'] != winding:
                    continue

                input_tensor = input_tokens.unsqueeze(0).to(device)

                _ = model(input_tensor, return_cache=True)

                # Get embedding (average over sequence)
                embeddings = model.get_embeddings()
                avg_embedding = embeddings[0].mean(dim=0).cpu().numpy()

                embeddings_list.append(avg_embedding)
                curves_list.append(metadata['curve'])

                count += 1

            if embeddings_list:
                winding_to_embeddings[winding] = np.array(embeddings_list)
                winding_to_curves[winding] = curves_list

    # Analyze embeddings by winding number
    print("Analyzing embedding structure...")

    all_embeddings = []
    all_labels = []

    for winding, embeddings in winding_to_embeddings.items():
        all_embeddings.append(embeddings)
        all_labels.extend([winding] * len(embeddings))

    all_embeddings = np.vstack(all_embeddings)
    all_labels = np.array(all_labels)

    # PCA
    print("\nComputing PCA...")
    pca_result = analyze_embedding_geometry(
        torch.from_numpy(all_embeddings).float(),
        labels=all_labels + args.max_winding,  # Shift back for visualization
        method='pca',
        n_components=2,
    )
    results['embeddings_pca'] = pca_result

    print(f"  PCA variance explained: {pca_result['metrics']['total_variance_explained']:.3f}")

    # Check if embeddings show topological organization
    # Compute correlation between PC1 and winding number
    pc1 = pca_result['reduced_embeddings'][:, 0]
    correlation = np.corrcoef(pc1, all_labels)[0, 1]
    print(f"  PC1 correlation with winding number: {correlation:.3f}")

    results['winding_correlation'] = float(correlation)

    # Visualize curves for each winding number
    if args.save_analysis:
        step_dir = results_dir / f'step_{step}'
        step_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving plots to {step_dir}")

        # Plot example curves for each winding number
        fig, axes = plt.subplots(2, args.max_winding + 1, figsize=(4 * (args.max_winding + 1), 8))
        axes = axes.flatten()

        for winding in range(-args.max_winding, args.max_winding + 1):
            ax = axes[winding + args.max_winding]

            if winding in winding_to_curves:
                # Plot a few example curves
                for curve in winding_to_curves[winding][:3]:
                    curve_array = np.array(curve)
                    ax.plot(curve_array[:, 0], curve_array[:, 1], 'o-', alpha=0.5)
                    ax.plot(curve_array[0, 0], curve_array[0, 1], 'ro', markersize=8)  # Start point

            ax.set_title(f'Winding = {winding}')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.axis('equal')
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color='k', linewidth=0.5)
            ax.axvline(0, color='k', linewidth=0.5)

        plt.suptitle('Example Curves by Winding Number', fontsize=14)
        plt.tight_layout()
        plt.savefig(step_dir / 'example_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved example_curves.png")

        # Embedding PCA
        plot_embedding_geometry(
            pca_result,
            save_path=step_dir / 'embeddings_pca.png',
            title='Winding Number - Embedding PCA',
        )
        print("  ✓ Saved embeddings_pca.png")

    # Save results
    results_file = results_dir / f'step_{step}' / 'analysis_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print("  ✓ Saved analysis_results.json")

    return results


def main():
    import torch.nn.functional as F

    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create directories
    checkpoint_dir = Path(args.checkpoint_dir) / 'winding'
    results_dir = Path(args.results_dir) / 'topological' / 'winding_number'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"Training winding number task")
    print(f"{'='*60}\n")

    # Create datasets
    print("Creating datasets...")
    train_dataset = WindingNumberDataset(
        max_winding=args.max_winding,
        n_curves_per_class=args.n_curves,
        n_points=args.n_points,
        split='train',
        seed=args.seed,
    )
    val_dataset = WindingNumberDataset(
        max_winding=args.max_winding,
        n_curves_per_class=args.n_curves // 5,
        n_points=args.n_points,
        split='val',
        seed=args.seed,
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Classes: {2 * args.max_winding + 1} (winding from -{args.max_winding} to {args.max_winding})")
    print(f"  Points per curve: {args.n_points}\n")

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

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=args.device,
        checkpoint_dir=str(checkpoint_dir),
        task_name='winding_number',
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
            title='Winding Number - Training Curves',
        )
        print(f"\n✓ Saved training_curves.png to {results_dir}")

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
