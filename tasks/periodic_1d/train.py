"""
Unified Training Script for Periodic 1D Tasks

Usage:
    python train.py --task mod_add --p 97 --epochs 20000
    python train.py --task digit_sum --max_digits 3 --epochs 10000
    python train.py --task parity --max_val 1000 --epochs 10000
"""

import argparse
import sys
import json
from pathlib import Path

# Add library to path
library_path = Path(__file__).parent.parent.parent / 'library'
sys.path.insert(0, str(library_path))

import torch
from library.models import tiny_transformer, small_transformer, create_model
from library.datasets import ModularArithmeticDataset, DigitSumDataset, ParityDataset
from library.training import create_trainer
from library.analysis import (
    compute_fourier_1d,
    detect_periodic_neurons,
    position_encoding_analysis,
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
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Train periodic 1D tasks')

    # Task selection
    parser.add_argument('--task', type=str, required=True,
                        choices=['mod_add', 'mod_mul', 'digit_sum', 'parity'],
                        help='Task to train')

    # Task-specific arguments
    parser.add_argument('--p', type=int, default=97, help='Modulus for modular arithmetic')
    parser.add_argument('--max_digits', type=int, default=3, help='Max digits for digit sum')
    parser.add_argument('--max_val', type=int, default=1000, help='Max value for parity')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=20000, help='Number of training steps')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--train_frac', type=float, default=0.5, help='Training fraction for grokking')

    # Model arguments
    parser.add_argument('--model_size', type=str, default='tiny',
                        choices=['tiny', 'small', 'medium'],
                        help='Model size')
    parser.add_argument('--pos_enc', type=str, default='learned',
                        choices=['learned', 'sinusoidal'],
                        help='Positional encoding type')

    # Infrastructure
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--results_dir', type=str, default='results', help='Results directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Analysis
    parser.add_argument('--save_analysis', action='store_true', help='Save analysis plots')
    parser.add_argument('--analysis_interval', type=int, default=5000,
                        help='Analyze every N steps')

    return parser.parse_args()


def create_dataset(args):
    """Create dataset based on task."""
    if args.task in ['mod_add', 'mod_mul']:
        train_dataset = ModularArithmeticDataset(
            operation='add' if args.task == 'mod_add' else 'mul',
            modulus=args.p,
            train_frac=args.train_frac,
            split='train',
            seed=args.seed,
        )
        val_dataset = ModularArithmeticDataset(
            operation='add' if args.task == 'mod_add' else 'mul',
            modulus=args.p,
            train_frac=args.train_frac,
            split='val',
            seed=args.seed,
        )
        vocab_size = args.p + 2

    elif args.task == 'digit_sum':
        train_dataset = DigitSumDataset(
            max_digits=args.max_digits,
            base=10,
            split='train',
            seed=args.seed,
        )
        val_dataset = DigitSumDataset(
            max_digits=args.max_digits,
            base=10,
            split='val',
            seed=args.seed,
        )
        vocab_size = 10

    elif args.task == 'parity':
        train_dataset = ParityDataset(
            max_val=args.max_val,
            split='train',
            seed=args.seed,
        )
        val_dataset = ParityDataset(
            max_val=args.max_val,
            split='val',
            seed=args.seed,
        )
        vocab_size = 10

    else:
        raise ValueError(f"Unknown task: {args.task}")

    return train_dataset, val_dataset, vocab_size


def create_model_from_args(args, vocab_size):
    """Create model based on arguments."""
    model_kwargs = {
        'vocab_size': vocab_size,
        'output_size': vocab_size,
        'pos_enc_type': args.pos_enc,
    }

    # Add positional encoding max value for mod tasks
    if args.task in ['mod_add', 'mod_mul']:
        model_kwargs['pos_enc_max_val'] = args.p

    if args.model_size == 'tiny':
        model = tiny_transformer(**model_kwargs)
    elif args.model_size == 'small':
        model = small_transformer(**model_kwargs)
    else:
        model = create_model({
            **model_kwargs,
            'd_model': 512,
            'n_layers': 6,
            'n_heads': 8,
            'd_mlp': 2048,
        })

    return model


def analyze_model(model, dataset, args, step, results_dir):
    """Run mechanistic analysis on trained model."""
    print(f"\n{'='*60}")
    print(f"Running analysis at step {step}")
    print(f"{'='*60}\n")

    results = {}

    # Get activations for all positions
    device = next(model.parameters()).device
    positions = np.arange(args.p if args.task in ['mod_add', 'mod_mul'] else dataset.get_vocab_size())

    all_activations = []
    all_embeddings = []

    model.eval()
    with torch.no_grad():
        for pos in positions:
            # Create input with this position
            if args.task in ['mod_add', 'mod_mul']:
                input_tensor = torch.tensor([[pos, 0]]).to(device)
            else:
                input_tensor = torch.tensor([[pos] + [0] * (dataset.get_seq_len()[0] - 1)]).to(device)

            # Forward pass with cache
            _ = model(input_tensor, return_cache=True)

            # Collect embeddings
            all_embeddings.append(model.get_embeddings()[0, 0, :].cpu().numpy())

            # Collect hidden states (average over layers)
            hidden_states = model.get_hidden_states()
            if hidden_states:
                layer_activations = []
                for layer_hidden in hidden_states:
                    layer_activations.append(layer_hidden[0, 0, :].cpu().numpy())

                # Average across layers
                avg_activation = np.mean(layer_activations, axis=0)
                all_activations.append(avg_activation)

    all_activations = np.array(all_activations).T  # [n_neurons, n_positions]
    all_embeddings = np.array(all_embeddings)  # [n_positions, d_model]

    # Fourier analysis on activations
    print("Computing Fourier spectrum...")
    freqs, power, fourier_metrics = compute_fourier_1d(all_activations)
    results['fourier'] = fourier_metrics

    print(f"  Dominant frequency: {fourier_metrics['dominant_frequency']:.3f}")
    print(f"  Top-1 power ratio: {fourier_metrics['top1_power_ratio']:.3f}")
    print(f"  Top-3 power ratio: {fourier_metrics['top3_power_ratio']:.3f}")

    # Periodic neuron detection
    print("\nDetecting periodic neurons...")
    periodic_neurons = detect_periodic_neurons(all_activations, positions, r2_threshold=0.7)
    results['periodic_neurons'] = periodic_neurons

    print(f"  Found {len(periodic_neurons)} periodic neurons")
    if len(periodic_neurons) > 0:
        print(f"  Top neuron R²: {periodic_neurons[0]['r_squared']:.3f}")
        print(f"  Top neuron frequency: {periodic_neurons[0]['frequency']:.1f}")

    # Position encoding analysis
    print("\nAnalyzing position encodings...")
    embedding_tensor = torch.from_numpy(all_embeddings).float().to(device)
    pos_analysis = position_encoding_analysis(embedding_tensor, positions)
    results['position_encoding'] = pos_analysis

    print(f"  Periodicity score: {pos_analysis['periodicity_score']:.3f}")
    print(f"  Periodic dimensions: {pos_analysis['n_periodic']}/{pos_analysis.get('n_strong_periodic', 0)}")

    # Get attention patterns
    print("\nAnalyzing attention patterns...")
    # Run a batch to get attention patterns
    batch_inputs = torch.arange(min(32, len(positions))).unsqueeze(1).expand(-1, 2).to(device)
    _ = model(batch_inputs, return_cache=True)
    all_attention = model.get_all_attention_patterns()

    if all_attention is not None:
        # Analyze first layer, first head
        attention_0_0 = analyze_attention_patterns(all_attention, layer=0, head=0)
        results['attention_layer_0_head_0'] = attention_0_0

        print(f"  Layer 0, Head 0 pattern: {attention_0_0['pattern_type']}")
        print(f"  Entropy: {attention_0_0['entropy_mean']:.3f}")
        print(f"  Diagonal bias: {attention_0_0['diagonal_bias']:.3f}")

    # Embedding geometry
    print("\nAnalyzing embedding geometry...")
    embedding_full = torch.from_numpy(all_embeddings).float()

    # PCA
    pca_result = analyze_embedding_geometry(
        embedding_full,
        labels=positions,
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

        # Fourier spectrum
        plot_fourier_spectrum(
            freqs, power, fourier_metrics,
            save_path=step_dir / 'fourier_spectrum.png',
            title=f'{args.task} - Fourier Spectrum',
        )
        print("  ✓ Saved fourier_spectrum.png")

        # Periodic neurons
        if len(periodic_neurons) > 0:
            plot_periodic_neurons(
                periodic_neurons, all_activations, positions,
                save_path=step_dir / 'periodic_neurons.png',
                title=f'{args.task} - Periodic Neurons',
                top_k=9,
            )
            print("  ✓ Saved periodic_neurons.png")

        # Attention patterns
        if all_attention is not None:
            plot_attention_patterns(
                attention_0_0,
                save_path=step_dir / 'attention_l0h0.png',
                title=f'{args.task} - Attention L0H0',
            )
            print("  ✓ Saved attention_l0h0.png")

        # Embedding geometry
        plot_embedding_geometry(
            pca_result,
            save_path=step_dir / 'embeddings_pca.png',
            title=f'{args.task} - Embedding PCA',
        )
        print("  ✓ Saved embeddings_pca.png")

    # Save results JSON
    results_file = results_dir / f'step_{step}' / 'analysis_results.json'
    with open(results_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        results_serializable = {}
        for key, value in results.items():
            if isinstance(value, dict):
                results_serializable[key] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in value.items()
                }
            elif isinstance(value, list):
                results_serializable[key] = value
            else:
                results_serializable[key] = value

        json.dump(results_serializable, f, indent=2)
    print(f"  ✓ Saved analysis_results.json")

    return results


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create directories
    checkpoint_dir = Path(args.checkpoint_dir)
    results_dir = Path(args.results_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"Training task: {args.task}")
    print(f"{'='*60}\n")

    # Create dataset
    print("Creating dataset...")
    train_dataset, val_dataset, vocab_size = create_dataset(args)
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Vocab size: {vocab_size}\n")

    # Create model
    print("Creating model...")
    model = create_model_from_args(args, vocab_size)
    print(f"  Model: {args.model_size}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Create trainer
    trainer = create_trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        task_name=f'{args.task}_{args.p if args.task in ["mod_add", "mod_mul"] else args.max_digits}',
        device=args.device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        checkpoint_dir=str(checkpoint_dir),
        log_interval=100,
        eval_interval=args.analysis_interval,
        save_interval=args.analysis_interval,
    )

    # Add analysis callback
    analysis_steps = []

    def analysis_callback(model, step, val_metrics):
        if step % args.analysis_interval == 0 and step > 0:
            results = analyze_model(model, val_dataset, args, step, results_dir)
            analysis_steps.append(step)

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
            title=f'{args.task} - Training Curves',
        )
        print(f"\n✓ Saved training_curves.png to {results_dir}")

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
