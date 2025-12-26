"""
Cascaded Modular Arithmetic Task

Train a model to compute: (((x + a₁) mod p₁) + a₂) mod p₂

This tests multi-scale reasoning — the model must handle two different
moduli at different scales. The first mod p₁ is coarse-grained, the second
mod p₂ is fine-grained (assuming p₂ < p₁).

Key question: Does the model discover scale-separated representations?

Usage:
    python train_cascaded_modular.py --p1 97 --p2 13 --epochs 10000
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
    compute_cross_layer_similarity,
)
from library.viz import (
    plot_fourier_spectrum,
    plot_periodic_neurons,
    plot_attention_patterns,
    plot_embedding_geometry,
    plot_training_curves,
)
import matplotlib.pyplot as plt


class CascadedModularDataset(Dataset):
    """
    Dataset for cascaded modular arithmetic.

    Task: Compute (((x + a) mod p1) + b) mod p2

    Args:
        p1: First modulus (coarse scale)
        p2: Second modulus (fine scale, should be < p1)
        train_frac: Fraction of data for training
        split: 'train', 'val', or 'test'
        seed: Random seed
    """

    def __init__(
        self,
        p1: int = 97,
        p2: int = 13,
        train_frac: float = 0.5,
        split: str = 'train',
        seed: int = 42,
    ):
        self.p1 = p1
        self.p2 = p2
        self.train_frac = train_frac
        self.split = split
        self.seed = seed

        # Generate all possible inputs
        self.samples = []
        for x in range(p1):
            for a in range(p1):
                for b in range(p2):
                    # Compute result
                    step1 = (x + a) % p1
                    step2 = (step1 + b) % p2
                    result = step2

                    self.samples.append({
                        'x': x,
                        'a': a,
                        'b': b,
                        'result': result,
                    })

        # Shuffle and split
        rng = np.random.default_rng(seed)
        indices = np.arange(len(self.samples))
        rng.shuffle(indices)

        n_train = int(len(indices) * train_frac)
        n_val = int(len(indices) * (1 - train_frac) / 2)

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
        sample = self.samples[idx]

        # Input: [x, a, b]
        # Use max(p1, p2) as vocab size
        vocab_size = max(self.p1, self.p2)

        input_tokens = torch.tensor([
            sample['x'],
            sample['a'],
            sample['b'],
        ], dtype=torch.long)

        target_tokens = torch.tensor([sample['result']], dtype=torch.long)

        metadata = {
            'task': 'cascaded_modular',
            'split': self.split,
            'p1': self.p1,
            'p2': self.p2,
            'x': sample['x'],
            'a': sample['a'],
            'b': sample['b'],
            'result': sample['result'],
            'step1': (sample['x'] + sample['a']) % self.p1,  # Intermediate value
        }

        return input_tokens, target_tokens, metadata

    def get_vocab_size(self):
        return max(self.p1, self.p2)

    def get_output_size(self):
        return self.p2

    def get_seq_len(self):
        return (3, 1)


def parse_args():
    parser = argparse.ArgumentParser(description='Train cascaded modular arithmetic')

    parser.add_argument('--p1', type=int, default=97, help='First modulus (coarse)')
    parser.add_argument('--p2', type=int, default=13, help='Second modulus (fine)')
    parser.add_argument('--train_frac', type=float, default=0.5, help='Training fraction')
    parser.add_argument('--epochs', type=int, default=10000, help='Training steps')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')

    parser.add_argument('--model_size', type=str, default='tiny',
                        choices=['tiny', 'small'])
    parser.add_argument('--pos_enc', type=str, default='sinusoidal',
                        choices=['learned', 'sinusoidal'])

    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--save_analysis', action='store_true')
    parser.add_argument('--analysis_interval', type=int, default=2500)

    return parser.parse_args()


def create_model_from_args(args):
    """Create model for cascaded modular task."""
    vocab_size = max(args.p1, args.p2)

    model_kwargs = {
        'vocab_size': vocab_size,
        'output_size': args.p2,
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
    """Run mechanistic analysis on cascaded modular model."""
    print(f"\n{'='*60}")
    print(f"Running cascaded modular analysis at step {step}")
    print(f"{'='*60}\n")

    results = {}
    device = next(model.parameters()).device

    # Analyze activations for each position (x, a, b)
    positions_x = np.arange(args.p1)
    positions_a = np.arange(args.p1)
    positions_b = np.arange(args.p2)

    # Collect embeddings when varying each position
    embeddings_by_x = []
    embeddings_by_a = []
    embeddings_by_b = []

    model.eval()
    with torch.no_grad():
        # Vary x (keep a, b fixed)
        for x in positions_x:
            input_tensor = torch.tensor([[x, 0, 0]]).to(device)
            _ = model(input_tensor, return_cache=True)

            embeddings = model.get_embeddings()
            avg_embedding = embeddings[0].mean(dim=0).cpu().numpy()
            embeddings_by_x.append(avg_embedding)

        # Vary a (keep x, b fixed)
        for a in positions_a:
            input_tensor = torch.tensor([[0, a, 0]]).to(device)
            _ = model(input_tensor, return_cache=True)

            embeddings = model.get_embeddings()
            avg_embedding = embeddings[0].mean(dim=0).cpu().numpy()
            embeddings_by_a.append(avg_embedding)

        # Vary b (keep x, a fixed)
        for b in positions_b:
            input_tensor = torch.tensor([[0, 0, b]]).to(device)
            _ = model(input_tensor, return_cache=True)

            embeddings = model.get_embeddings()
            avg_embedding = embeddings[0].mean(dim=0).cpu().numpy()
            embeddings_by_b.append(avg_embedding)

    embeddings_by_x = np.array(embeddings_by_x)
    embeddings_by_a = np.array(embeddings_by_a)
    embeddings_by_b = np.array(embeddings_by_b)

    # Analyze periodicity for each position
    print("\nAnalyzing periodicity by position:")

    # Position x (should have period p1)
    fourier_x, power_x, metrics_x = compute_fourier_1d(embeddings_by_x.T)
    results['fourier_x'] = metrics_x

    print(f"\n  Position x (p1={args.p1}):")
    print(f"    Dominant frequency: {metrics_x['dominant_frequency']:.3f}")
    print(f"    Top-1 power ratio: {metrics_x['top1_power_ratio']:.3f}")

    # Check if frequency matches p1
    expected_freq_x = 1.0 / args.p1
    freq_error_x = abs(metrics_x['dominant_frequency'] - expected_freq_x)
    print(f"    Frequency error from 1/p1: {freq_error_x:.6f}")

    results['x_freq_match_p1'] = float(freq_error_x < 0.01)

    # Position a (should also have period p1)
    fourier_a, power_a, metrics_a = compute_fourier_1d(embeddings_by_a.T)
    results['fourier_a'] = metrics_a

    print(f"\n  Position a (p1={args.p1}):")
    print(f"    Dominant frequency: {metrics_a['dominant_frequency']:.3f}")
    print(f"    Top-1 power ratio: {metrics_a['top1_power_ratio']:.3f}")

    # Position b (should have period p2)
    fourier_b, power_b, metrics_b = compute_fourier_1d(embeddings_by_b.T)
    results['fourier_b'] = metrics_b

    print(f"\n  Position b (p2={args.p2}):")
    print(f"    Dominant frequency: {metrics_b['dominant_frequency']:.3f}")
    print(f"    Top-1 power ratio: {metrics_b['top1_power_ratio']:.3f}")

    # Check if frequency matches p2
    expected_freq_b = 1.0 / args.p2
    freq_error_b = abs(metrics_b['dominant_frequency'] - expected_freq_b)
    print(f"    Frequency error from 1/p2: {freq_error_b:.6f}")

    results['b_freq_match_p2'] = float(freq_error_b < 0.01)

    # Detect periodic neurons for each position
    print("\nDetecting periodic neurons:")

    periodic_x = detect_periodic_neurons(embeddings_by_x.T, positions_x, r2_threshold=0.7)
    periodic_a = detect_periodic_neurons(embeddings_by_a.T, positions_a, r2_threshold=0.7)
    periodic_b = detect_periodic_neurons(embeddings_by_b.T, positions_b, r2_threshold=0.7)

    results['periodic_neurons_x'] = periodic_x[:10]  # Top 10
    results['periodic_neurons_a'] = periodic_a[:10]
    results['periodic_neurons_b'] = periodic_b[:10]

    print(f"  Position x: {len(periodic_x)} periodic neurons")
    print(f"  Position a: {len(periodic_a)} periodic neurons")
    print(f"  Position b: {len(periodic_b)} periodic neurons")

    # Cross-layer analysis
    # Get hidden states for a sample input
    print("\nAnalyzing cross-layer similarity...")

    sample_input = torch.tensor([[42, 10, 5]]).to(device)  # Random sample
    _ = model(sample_input, return_cache=True)
    hidden_states = model.get_hidden_states()

    if hidden_states:
        for layer in range(len(hidden_states) - 1):
            similarity = compute_cross_layer_similarity(hidden_states, layer)
            if similarity:
                print(f"  Layer {layer} -> {layer+1}:")
                print(f"    Cosine similarity: {similarity['cosine_similarity']:.3f}")
                print(f"    CCA score: {similarity['cca_score']:.3f}")

                results[f'layer_{layer}_to_{layer+1}'] = similarity

    # Visualizations
    if args.save_analysis:
        step_dir = results_dir / f'step_{step}'
        step_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving plots to {step_dir}")

        # Fourier spectra for all three positions
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Position x
        axes[0].plot(fourier_x, power_x, 'b-', linewidth=2)
        axes[0].axvline(expected_freq_x, color='r', linestyle='--', label=f'1/p1={expected_freq_x:.4f}')
        axes[0].set_xlabel('Frequency')
        axes[0].set_ylabel('Power')
        axes[0].set_title(f'Position x (p1={args.p1})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Position a
        axes[1].plot(fourier_a, power_a, 'orange', linewidth=2)
        axes[1].axvline(expected_freq_x, color='r', linestyle='--', label=f'1/p1={expected_freq_x:.4f}')
        axes[1].set_xlabel('Frequency')
        axes[1].set_ylabel('Power')
        axes[1].set_title(f'Position a (p1={args.p1})')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Position b
        axes[2].plot(fourier_b, power_b, 'green', linewidth=2)
        axes[2].axvline(expected_freq_b, color='r', linestyle='--', label=f'1/p2={expected_freq_b:.4f}')
        axes[2].set_xlabel('Frequency')
        axes[2].set_ylabel('Power')
        axes[2].set_title(f'Position b (p2={args.p2})')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.suptitle('Fourier Spectra by Input Position', fontsize=14)
        plt.tight_layout()
        plt.savefig(step_dir / 'fourier_by_position.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved fourier_by_position.png")

        # Periodic neurons for position b (most interesting - fine scale)
        if len(periodic_b) > 0:
            plot_periodic_neurons(
                periodic_b,
                embeddings_by_b.T,
                positions_b,
                save_path=step_dir / 'periodic_neurons_b.png',
                title=f'Cascaded Modular - Periodic Neurons (position b, p2={args.p2})',
                top_k=9,
            )
            print("  ✓ Saved periodic_neurons_b.png")

    # Save results
    results_file = results_dir / f'step_{step}' / 'analysis_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print("  ✓ Saved analysis_results.json")

    return results


def main():
    args = parse_args()

    # Validate moduli
    if args.p2 <= args.p1:
        print(f"Warning: p2 ({args.p2}) should be < p1 ({args.p1}) for clear scale separation")
    else:
        print(f"Note: Using p2={args.p2} > p1={args.p1}. For clearer multi-scale, consider p2 < p1.")

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create directories
    checkpoint_dir = Path(args.checkpoint_dir) / 'cascaded_modular'
    results_dir = Path(args.results_dir) / 'multi_scale' / 'cascaded_modular'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"Training cascaded modular arithmetic: (((x + a) mod {args.p1}) + b) mod {args.p2}")
    print(f"{'='*60}\n")

    # Create datasets
    print("Creating datasets...")
    train_dataset = CascadedModularDataset(
        p1=args.p1,
        p2=args.p2,
        train_frac=args.train_frac,
        split='train',
        seed=args.seed,
    )
    val_dataset = CascadedModularDataset(
        p1=args.p1,
        p2=args.p2,
        train_frac=args.train_frac,
        split='val',
        seed=args.seed,
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Moduli: p1={args.p1}, p2={args.p2}\n")

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
        task_name='cascaded_modular',
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
            title=f'Cascaded Modular (p1={args.p1}, p2={args.p2}) - Training Curves',
        )
        print(f"\n✓ Saved training_curves.png to {results_dir}")

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
