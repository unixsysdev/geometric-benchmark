"""
Geometric Benchmark Runner

Runs the full benchmark or specific task categories.

Usage:
    python run_benchmark.py --tasks all
    python run_benchmark.py --tasks periodic_1d topological
    python run_benchmark.py --config configs/benchmark.yaml
"""

import argparse
import subprocess
import sys
from pathlib import Path
import yaml
from typing import List, Dict, Any


# Task configurations
DEFAULT_CONFIGS = {
    'periodic_1d': {
        'mod_add_p97': {
            'task': 'mod_add',
            'p': 97,
            'epochs': 20000,
            'model_size': 'tiny',
            'pos_enc': 'sinusoidal',
            'train_frac': 0.5,
        },
        'mod_add_p43': {
            'task': 'mod_add',
            'p': 43,
            'epochs': 15000,
            'model_size': 'tiny',
            'pos_enc': 'sinusoidal',
            'train_frac': 0.5,
        },
        'digit_sum': {
            'task': 'digit_sum',
            'max_digits': 3,
            'epochs': 10000,
            'model_size': 'tiny',
            'pos_enc': 'learned',
        },
        'parity': {
            'task': 'parity',
            'max_val': 1000,
            'epochs': 10000,
            'model_size': 'tiny',
            'pos_enc': 'learned',
        },
    },
    'periodic_2d': {
        'torus_distance': {
            'script': 'tasks/periodic_2d/train_torus.py',
            'grid_size': 16,
            'epochs': 20000,
        },
    },
    'topological': {
        'winding_number': {
            'script': 'tasks/topological/train_winding.py',
            'max_winding': 5,
            'epochs': 20000,
        },
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description='Run Geometric Benchmark')

    parser.add_argument('--tasks', type=str, nargs='+', default=['all'],
                        help='Task categories to run (all, periodic_1d, periodic_2d, topological, etc.)')
    parser.add_argument('--config', type=str, default=None,
                        help='YAML config file with task configurations')
    parser.add_argument('--parallel', action='store_true',
                        help='Run tasks in parallel (experimental)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--checkpoint_dir', type=str, default='geometric_benchmark/checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--results_dir', type=str, default='geometric_benchmark/results',
                        help='Results directory')

    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_task(task_config: Dict[str, Any], args) -> bool:
    """Run a single task."""
    task_name = task_config.get('name', 'unknown')

    # Check if custom script
    if 'script' in task_config:
        script_path = task_config['script']
        cmd_args = [script_path]

        # Add script-specific arguments
        for key, value in task_config.items():
            if key not in ['script', 'name']:
                cmd_args.extend([f'--{key}', str(value)])
    else:
        # Use periodic_1d trainer
        script_path = 'tasks/periodic_1d/train.py'
        cmd_args = [script_path]

        # Add arguments
        for key, value in task_config.items():
            if key not in ['name']:
                cmd_args.extend([f'--{key}', str(value)])

    # Add common arguments
    cmd_args.extend([
        '--device', args.device,
        '--checkpoint_dir', args.checkpoint_dir,
        '--results_dir', args.results_dir,
        '--save_analysis',
    ])

    print(f"\n{'='*70}")
    print(f"Running task: {task_name}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd_args)}\n")

    # Run task
    result = subprocess.run(cmd_args, capture_output=False)

    return result.returncode == 0


def main():
    args = parse_args()

    # Load config if provided
    if args.config:
        configs = load_config(args.config)
    else:
        configs = DEFAULT_CONFIGS

    # Determine which categories to run
    if 'all' in args.tasks:
        categories_to_run = list(configs.keys())
    else:
        categories_to_run = args.tasks

    print(f"\n{'='*70}")
    print(f"GEOMETRIC BENCHMARK RUNNER")
    print(f"{'='*70}")
    print(f"\nCategories to run: {', '.join(categories_to_run)}")
    print(f"Total tasks: {sum(len(configs.get(cat, {})) for cat in categories_to_run)}")
    print(f"Device: {args.device}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Results directory: {args.results_dir}\n")

    # Collect all tasks
    all_tasks = []
    for category in categories_to_run:
        if category not in configs:
            print(f"Warning: Unknown category '{category}', skipping...")
            continue

        category_tasks = configs[category]
        for task_name, task_config in category_tasks.items():
            task_config['name'] = f"{category}/{task_name}"
            all_tasks.append(task_config)

    # Run tasks
    results = {}
    for task_config in all_tasks:
        task_name = task_config['name']
        success = run_task(task_config, args)
        results[task_name] = '✓ Success' if success else '✗ Failed'

    # Summary
    print(f"\n{'='*70}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*70}\n")

    for task_name, result in results.items():
        print(f"  {task_name}: {result}")

    print(f"\n{'='*70}\n")

    # Generate compendium
    print("Generating compendium...")
    from compendium.generate import generate_compendium

    generate_compendium(
        results_dir=args.results_dir,
        output_dir=Path(args.results_dir) / 'compendium',
    )

    print("\n✓ Benchmark complete!")


if __name__ == '__main__':
    main()
