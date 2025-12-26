"""
Permutation Composition Task (Symmetric Group S₃)

Train a model to compose permutations: given two permutations σ and τ,
predict σ ∘ τ (apply τ first, then σ).

This tests whether the model can discover the structure of non-abelian groups.

Permutations are represented in cycle notation or one-line notation.

Usage:
    python train_permutations.py --n 3 --epochs 10000
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


class Permutation:
    """Helper class for permutation operations."""

    def __init__(self, elements):
        """
        Create permutation from list of elements.
        elements[i] = where element i maps to
        """
        self.elements = tuple(elements)

    def __mul__(self, other):
        """Compose permutations: self * other means apply other first, then self."""
        n = len(self.elements)
        result = []
        for i in range(n):
            # Apply other first, then self
            result.append(self.elements[other.elements[i]])
        return Permutation(result)

    def __repr__(self):
        return f"Permutation({self.elements})"

    def __eq__(self, other):
        return self.elements == other.elements

    def to_one_line(self):
        """Convert to one-line notation."""
        return ''.join(map(str, self.elements))

    @staticmethod
    def from_one_line(s):
        """Create from one-line notation string."""
        return Permutation(list(map(int, s)))


class PermutationDataset(Dataset):
    """
    Dataset for permutation composition.

    Task: Given permutations σ and τ, predict σ ∘ τ.

    Args:
        n: Size of symmetric group (permutations of n elements)
        split: 'train', 'val', or 'test'
        seed: Random seed
    """

    def __init__(self, n: int = 3, split: str = 'train', seed: int = 42):
        self.n = n
        self.split = split
        self.seed = seed

        # Generate all permutations
        import itertools
        all_perms = list(itertools.permutations(range(n)))
        all_perm_objs = [Permutation(p) for p in all_perms]

        # Generate all pairs
        self.samples = []
        for perm1 in all_perm_objs:
            for perm2 in all_perm_objs:
                result = perm1 * perm2
                self.samples.append((perm1, perm2, result))

        # Shuffle and split
        rng = np.random.default_rng(seed)
        indices = np.arange(len(self.samples))
        rng.shuffle(indices)

        n_train = int(len(indices) * 0.7)
        n_val = int(len(indices) * 0.15)

        if split == 'train':
            indices = indices[:n_train]
        elif split == 'val':
            indices = indices[n_train:n_train + n_val]
        else:
            indices = indices[n_train + n_val:]

        self.samples = [self.samples[i] for i in indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        perm1, perm2, result = self.samples[idx]

        # Input: permutation elements (vocab_size = n)
        # Format: [p1_0, p1_1, ..., p1_{n-1}, p2_0, p2_1, ..., p2_{n-1}]
        input_tokens = torch.tensor(list(perm1.elements) + list(perm2.elements), dtype=torch.long)

        # Output: result permutation
        target_tokens = torch.tensor(list(result.elements), dtype=torch.long)

        metadata = {
            'task': 'permutation_composition',
            'split': self.split,
            'n': self.n,
            'perm1': perm1.to_one_line(),
            'perm2': perm2.to_one_line(),
            'result': result.to_one_line(),
        }

        return input_tokens, target_tokens, metadata

    def get_vocab_size(self):
        return self.n

    def get_output_size(self):
        return self.n

    def get_seq_len(self):
        return (2 * self.n, self.n)


def parse_args():
    parser = argparse.ArgumentParser(description='Train permutation composition task')

    parser.add_argument('--n', type=int, default=3, help='Size of symmetric group S_n')
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
    """Create model for permutation task."""
    vocab_size = args.n

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
    """Run mechanistic analysis on permutation model."""
    print(f"\n{'='*60}")
    print(f"Running permutation analysis at step {step}")
    print(f"{'='*60}\n")

    results = {}
    device = next(model.parameters()).device

    # Get embeddings for each permutation
    # Generate all unique permutations
    import itertools
    all_perms = list(itertools.permutations(range(args.n)))
    perm_objs = [Permutation(p) for p in all_perms]

    all_embeddings = []
    perm_strings = []

    model.eval()
    with torch.no_grad():
        for perm in perm_objs:
            # Create input with this permutation twice
            input_tokens = torch.tensor(list(perm.elements) + list(perm.elements), dtype=torch.long)
            input_tensor = input_tokens.unsqueeze(0).to(device)

            _ = model(input_tensor, return_cache=True)

            # Get embedding
            embeddings = model.get_embeddings()
            avg_embedding = embeddings[0].mean(dim=0).cpu().numpy()

            all_embeddings.append(avg_embedding)
            perm_strings.append(perm.to_one_line())

    all_embeddings = np.array(all_embeddings)

    # Analyze embedding structure
    print("Analyzing permutation embeddings...")

    # PCA
    # Use cycle structure as labels
    def cycle_type(perm):
        """Get cycle type as label (e.g., (1,2,3) has type '3-cycle')."""
        visited = [False] * args.n
        cycles = []
        for i in range(args.n):
            if not visited[i]:
                cycle = []
                j = i
                while not visited[j]:
                    visited[j] = True
                    cycle.append(j)
                    j = perm.elements[j]
                if len(cycle) > 1:
                    cycles.append(len(cycle))

        cycles.sort(reverse=True)
        return tuple(cycles) if cycles else tuple([1])

    # Get cycle types for all permutations
    cycle_types = [cycle_type(perm) for perm in perm_objs]

    # Create numeric labels
    unique_cycle_types = sorted(set(cycle_types))
    cycle_type_labels = [unique_cycle_types.index(ct) for ct in cycle_types]

    pca_result = analyze_embedding_geometry(
        torch.from_numpy(all_embeddings).float(),
        labels=np.array(cycle_type_labels),
        method='pca',
        n_components=2,
    )
    results['embeddings_pca'] = pca_result

    print(f"  PCA variance explained: {pca_result['metrics']['total_variance_explained']:.3f}")

    # Check if same cycle types cluster together
    # Compute within-cycle-type vs between-cycle-type distance
    from scipy.spatial.distance import pdist, squareform

    dist_matrix = squareform(pdist(all_embeddings))

    # Average within-group distance
    within_dists = []
    between_dists = []

    for i, ct_i in enumerate(cycle_types):
        for j, ct_j in enumerate(cycle_types):
            if i < j:
                if ct_i == ct_j:
                    within_dists.append(dist_matrix[i, j])
                else:
                    between_dists.append(dist_matrix[i, j])

    if within_dists and between_dists:
        within_mean = np.mean(within_dists)
        between_mean = np.mean(between_dists)
        clustering_score = between_mean / (within_mean + 1e-8)
        print(f"  Within-cycle-type distance: {within_mean:.3f}")
        print(f"  Between-cycle-type distance: {between_mean:.3f}")
        print(f"  Clustering ratio (higher is better): {clustering_score:.3f}")

        results['clustering_score'] = float(clustering_score)

    # Test if model understands group structure
    # Check if embedding of identity is central
    identity = Permutation(list(range(args.n)))
    identity_idx = perm_objs.index(identity)

    # Compute average distance from identity to all others
    identity_embedding = all_embeddings[identity_idx]
    dists_from_identity = np.linalg.norm(all_embeddings - identity_embedding, axis=1)
    avg_dist_from_identity = dists_from_identity.mean()

    print(f"\n  Average distance from identity: {avg_dist_from_identity:.3f}")

    # Compute centroid of all embeddings
    centroid = all_embeddings.mean(axis=0)
    dist_from_centroid = np.linalg.norm(identity_embedding - centroid)

    print(f"  Distance of identity from centroid: {dist_from_centroid:.3f}")
    print(f"    (Small distance suggests identity is central)")

    results['identity_centrality'] = float(dist_from_centroid)

    # Visualize
    if args.save_analysis:
        step_dir = results_dir / f'step_{step}'
        step_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving plots to {step_dir}")

        # PCA colored by cycle type
        plot_embedding_geometry(
            pca_result,
            save_path=step_dir / 'embeddings_pca.png',
            title=f'Permutation S{args.n} - Embedding PCA (colored by cycle type)',
        )
        print("  ✓ Saved embeddings_pca.png")

        # Permutation group structure visualization
        # Show Cayley table or cycle structure
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: Bar chart of cycle types
        cycle_type_counts = {}
        for ct in cycle_types:
            cycle_type_counts[ct] = cycle_type_counts.get(ct, 0) + 1

        cycle_labels = [str(ct) for ct in sorted(cycle_type_counts.keys())]
        counts = [cycle_type_counts[ct] for ct in sorted(cycle_type_counts.keys())]

        axes[0].bar(range(len(cycle_labels)), counts)
        axes[0].set_xticks(range(len(cycle_labels)))
        axes[0].set_xticklabels(cycle_labels, rotation=45, ha='right')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Cycle Type Distribution')
        axes[0].grid(True, alpha=0.3)

        # Right: Distance matrix (ordered by cycle type)
        # Order permutations by cycle type
        order = sorted(range(len(cycle_types)), key=lambda i: cycle_types[i])
        ordered_dist_matrix = dist_matrix[np.ix_(order, order)]

        im = axes[1].imshow(ordered_dist_matrix, cmap='viridis')
        axes[1].set_xlabel('Permutation (ordered by cycle type)')
        axes[1].set_ylabel('Permutation (ordered by cycle type)')
        axes[1].set_title('Permutation Distance Matrix')
        plt.colorbar(im, ax=axes[1])

        plt.tight_layout()
        plt.savefig(step_dir / 'group_structure.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved group_structure.png")

    # Save results
    results_file = results_dir / f'step_{step}' / 'analysis_results.json'
    # Convert tuples to strings for JSON
    results_serializable = {}
    for key, value in results.items():
        if isinstance(value, dict):
            results_serializable[key] = {
                k: (str(v) if isinstance(v, tuple) else v)
                for k, v in value.items()
            }
        else:
            results_serializable[key] = value

    with open(results_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    print("  ✓ Saved analysis_results.json")

    return results


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create directories
    checkpoint_dir = Path(args.checkpoint_dir) / 'permutations'
    results_dir = Path(args.results_dir) / 'group_theoretic' / 'permutations'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"Training permutation composition task (S_{args.n})")
    print(f"{'='*60}\n")

    # Create datasets
    print("Creating datasets...")
    train_dataset = PermutationDataset(
        n=args.n,
        split='train',
        seed=args.seed,
    )
    val_dataset = PermutationDataset(
        n=args.n,
        split='val',
        seed=args.seed,
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Group size: {np.math.factorial(args.n)}\n")

    # Create model
    print("Creating model...")
    model = create_model_from_args(args)
    print(f"  Model: {args.model_size}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Create custom collate function for sequence output
    def collate_fn(batch):
        inputs = torch.stack([item[0] for item in batch])
        targets = torch.stack([item[1] for item in batch])
        metadata_list = [item[2] for item in batch]
        return {'input': inputs, 'target': targets, 'metadata': metadata_list}

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Create trainer
    import torch.optim as optim

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Custom loss for sequence output
    def sequence_loss_fn(logits, targets):
        # logits: [batch, seq_len, vocab]
        # targets: [batch, seq_len]
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)
        return nn.CrossEntropyLoss()(logits_flat, targets_flat)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=sequence_loss_fn,
        device=args.device,
        checkpoint_dir=str(checkpoint_dir),
        task_name='permutations',
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
            title=f'Permutation S{args.n} - Training Curves',
        )
        print(f"\n✓ Saved training_curves.png to {results_dir}")

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
