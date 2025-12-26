"""
Compendium Generator for Geometric Benchmark

Creates unified analysis reports across all tasks.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template

# Set style
sns.set_style('whitegrid')
plt.rcParams['font.size'] = 10


def load_analysis_results(results_dir: Path) -> Dict[str, Any]:
    """Load all analysis results from results directory."""
    results = {}

    # Walk through all subdirectories
    for step_dir in sorted(results_dir.glob('step_*')):
        if not step_dir.is_dir():
            continue

        step = int(step_dir.name.split('_')[1])

        # Load analysis results JSON
        results_file = step_dir / 'analysis_results.json'
        if results_file.exists():
            with open(results_file, 'r') as f:
                results[step] = json.load(f)

    return results


def generate_summary_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate summary metrics across all analysis steps."""
    summary = {}

    # Extract key metrics over time
    steps = sorted(results.keys())

    fourier_scores = []
    periodic_counts = []
    periodicity_scores = []

    for step in steps:
        step_results = results[step]

        if 'fourier' in step_results:
            fourier_scores.append(step_results['fourier'].get('top1_power_ratio', 0))

        if 'periodic_neurons' in step_results:
            periodic_counts.append(len(step_results['periodic_neurons']))

        if 'position_encoding' in step_results:
            periodicity_scores.append(step_results['position_encoding'].get('periodicity_score', 0))

    summary['steps'] = steps
    summary['fourier_scores'] = fourier_scores
    summary['periodic_counts'] = periodic_counts
    summary['periodicity_scores'] = periodicity_scores

    # Final values
    if steps:
        final_step = steps[-1]
        summary['final'] = results[final_step]

    return summary


def create_metrics_overview(summary: Dict[str, Any], save_path: Path):
    """Create overview plot of metrics over time."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    steps = summary['steps']

    # Fourier scores
    if summary['fourier_scores']:
        axes[0].plot(steps, summary['fourier_scores'], 'o-', linewidth=2)
        axes[0].set_ylabel('Top-1 Power Ratio')
        axes[0].set_title('Fourier Concentration Over Time')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 1)

    # Periodic neuron count
    if summary['periodic_counts']:
        axes[1].plot(steps, summary['periodic_counts'], 'o-', linewidth=2, color='orange')
        axes[1].set_ylabel('Number of Periodic Neurons')
        axes[1].set_title('Periodic Neuron Count Over Time')
        axes[1].grid(True, alpha=0.3)

    # Position encoding periodicity
    if summary['periodicity_scores']:
        axes[2].plot(steps, summary['periodicity_scores'], 'o-', linewidth=2, color='green')
        axes[2].set_ylabel('Periodicity Score')
        axes[2].set_xlabel('Training Step')
        axes[2].set_title('Position Encoding Periodicity Over Time')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_task_comparison(compendium_dir: Path, results_root: Path):
    """Create comparison plots across different tasks."""
    # Find all task result directories
    task_dirs = []

    for category_dir in results_root.iterdir():
        if not category_dir.is_dir():
            continue

        for task_dir in category_dir.iterdir():
            if not task_dir.is_dir():
                continue

            # Check for analysis results
            for step_dir in task_dir.glob('step_*'):
                results_file = step_dir / 'analysis_results.json'
                if results_file.exists():
                    task_dirs.append(task_dir)
                    break

    if len(task_dirs) < 2:
        print("Not enough tasks for comparison")
        return

    # Collect final metrics from each task
    task_metrics = []

    for task_dir in task_dirs:
        task_name = f"{task_dir.parent.name}/{task_dir.name}"

        # Load latest results
        results = load_analysis_results(task_dir)
        if not results:
            continue

        summary = generate_summary_metrics(results)
        final = summary.get('final', {})

        metrics = {
            'task': task_name,
            'category': task_dir.parent.name,
        }

        # Extract key metrics
        if 'fourier' in final:
            metrics['fourier_top1'] = final['fourier'].get('top1_power_ratio', 0)
            metrics['fourier_top3'] = final['fourier'].get('top3_power_ratio', 0)

        if 'periodic_neurons' in final:
            metrics['n_periodic'] = len(final['periodic_neurons'])
            if final['periodic_neurons']:
                metrics['top_periodic_r2'] = final['periodic_neurons'][0]['r_squared']

        if 'position_encoding' in final:
            metrics['periodicity_score'] = final['position_encoding'].get('periodicity_score', 0)

        task_metrics.append(metrics)

    if not task_metrics:
        print("No task metrics to compare")
        return

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    tasks = [m['task'] for m in task_metrics]

    # Fourier comparison
    if any('fourier_top1' in m for m in task_metrics):
        fourier_scores = [m.get('fourier_top1', 0) for m in task_metrics]
        categories = [m['category'] for m in task_metrics]

        # Group by category
        categories_unique = sorted(set(categories))
        colors = [categories_unique.index(c) for c in categories]

        axes[0, 0].barh(tasks, fourier_scores, color=cm.tab10(np.array(colors) / 10))
        axes[0, 0].set_xlabel('Top-1 Power Ratio')
        axes[0, 0].set_title('Fourier Concentration by Task')
        axes[0, 0].set_xlim(0, 1)

    # Periodic neuron count
    if any('n_periodic' in m for m in task_metrics):
        periodic_counts = [m.get('n_periodic', 0) for m in task_metrics]
        axes[0, 1].barh(tasks, periodic_counts)
        axes[0, 1].set_xlabel('Number of Periodic Neurons')
        axes[0, 1].set_title('Periodic Neurons by Task')

    # Periodicity score
    if any('periodicity_score' in m for m in task_metrics):
        periodicity_scores = [m.get('periodicity_score', 0) for m in task_metrics]
        axes[1, 0].barh(tasks, periodicity_scores, color='green')
        axes[1, 0].set_xlabel('Periodicity Score')
        axes[1, 0].set_title('Position Encoding Periodicity by Task')
        axes[1, 0].set_xlim(0, 1)

    # Scatter: Fourier vs Periodicity
    if any('fourier_top1' in m and 'periodicity_score' in m for m in task_metrics):
        for m in task_metrics:
            if 'fourier_top1' in m and 'periodicity_score' in m:
                axes[1, 1].scatter(m['fourier_top1'], m['periodicity_score'],
                                  s=100, alpha=0.7, label=m['task'])

        axes[1, 1].set_xlabel('Fourier Top-1 Power Ratio')
        axes[1, 1].set_ylabel('Periodicity Score')
        axes[1, 1].set_title('Fourier vs Periodicity')
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(compendium_dir / 'task_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Save metrics table
    with open(compendium_dir / 'task_metrics.json', 'w') as f:
        json.dump(task_metrics, f, indent=2)


def generate_html_report(compendium_dir: Path, summary: Dict[str, Any]):
    """Generate HTML report."""
    html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Geometric Benchmark Compendium</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
        }
        .header p {
            margin: 10px 0 0 0;
            opacity: 0.9;
        }
        .section {
            background: white;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .section h2 {
            color: #667eea;
            margin-top: 0;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .metric-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        .metric-card .label {
            font-size: 0.9em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .metric-card .value {
            font-size: 2em;
            font-weight: bold;
            color: #333;
            margin-top: 5px;
        }
        .plot {
            text-align: center;
            margin: 20px 0;
        }
        .plot img {
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background: #f8f9fa;
            font-weight: 600;
        }
        .footer {
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 Geometric Benchmark Compendium</h1>
        <p>Mechanistic interpretability analysis across geometric and topological tasks</p>
    </div>

    <div class="section">
        <h2>Summary Metrics</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="label">Training Steps Analyzed</div>
                <div class="value">{{ summary.steps|length if summary.steps else 0 }}</div>
            </div>
            {% if summary.fourier_scores %}
            <div class="metric-card">
                <div class="label">Final Fourier Score</div>
                <div class="value">{{ "%.3f"|format(summary.fourier_scores[-1]) }}</div>
            </div>
            {% endif %}
            {% if summary.periodic_counts %}
            <div class="metric-card">
                <div class="label">Periodic Neurons</div>
                <div class="value">{{ summary.periodic_counts[-1] if summary.periodic_counts else 0 }}</div>
            </div>
            {% endif %}
            {% if summary.periodicity_scores %}
            <div class="metric-card">
                <div class="label">Periodicity Score</div>
                <div class="value">{{ "%.3f"|format(summary.periodicity_scores[-1]) }}</div>
            </div>
            {% endif %}
        </div>
    </div>

    <div class="section">
        <h2>Metrics Over Time</h2>
        <div class="plot">
            <img src="metrics_overview.png" alt="Metrics Overview">
        </div>
    </div>

    {% if summary.final %}
    <div class="section">
        <h2>Final Analysis Results</h2>
        {% if summary.final.fourier %}
        <h3>Fourier Analysis</h3>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Dominant Frequency</td><td>{{ "%.3f"|format(summary.final.fourier.dominant_frequency) }}</td></tr>
            <tr><td>Top-1 Power Ratio</td><td>{{ "%.3f"|format(summary.final.fourier.top1_power_ratio) }}</td></tr>
            <tr><td>Top-3 Power Ratio</td><td>{{ "%.3f"|format(summary.final.fourier.top3_power_ratio) }}</td></tr>
        </table>
        {% endif %}

        {% if summary.final.periodic_neurons %}
        <h3>Periodic Neurons ({{ summary.final.periodic_neurons|length }} detected)</h3>
        <table>
            <tr><th>Neuron</th><th>R² Score</th><th>Frequency</th><th>Amplitude</th></tr>
            {% for neuron in summary.final.periodic_neurons[:10] %}
            <tr>
                <td>{{ neuron.neuron_idx }}</td>
                <td>{{ "%.3f"|format(neuron.r_squared) }}</td>
                <td>{{ neuron.frequency }}</td>
                <td>{{ "%.3f"|format(neuron.amplitude) }}</td>
            </tr>
            {% endfor %}
        </table>
        {% endif %}

        {% if summary.final.position_encoding %}
        <h3>Position Encoding Analysis</h3>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Periodicity Score</td><td>{{ "%.3f"|format(summary.final.position_encoding.periodicity_score) }}</td></tr>
            <tr><td>Periodic Dimensions</td><td>{{ summary.final.position_encoding.n_periodic }}</td></tr>
            <tr><td>Strong Periodic Dimensions</td><td>{{ summary.final.position_encoding.n_strong_periodic }}</td></tr>
        </table>
        {% endif %}
    </div>
    {% endif %}

    <div class="section">
        <h2>Task Comparison</h2>
        <div class="plot">
            <img src="task_comparison.png" alt="Task Comparison">
        </div>
    </div>

    <div class="footer">
        <p>Generated by Geometric Benchmark Suite</p>
    </div>
</body>
</html>
    """

    template = Template(html_template)
    html = template.render(summary=summary)

    with open(compendium_dir / 'index.html', 'w') as f:
        f.write(html)


def generate_compendium(
    results_dir: str = 'geometric_benchmark/results',
    output_dir: str = 'geometric_benchmark/results/compendium',
):
    """Generate compendium for a single task or all tasks."""
    results_path = Path(results_dir)
    output_path = Path(output_dir)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Generating compendium from {results_path}")
    print(f"Output to {output_path}\n")

    # Load results
    results = load_analysis_results(results_path)

    if not results:
        print("No analysis results found!")
        return

    print(f"Found {len(results)} analysis checkpoints")

    # Generate summary
    summary = generate_summary_metrics(results)
    print(f"Steps analyzed: {summary['steps']}")

    # Create metrics overview
    print("Creating metrics overview...")
    create_metrics_overview(summary, output_path / 'metrics_overview.png')

    # Create task comparison (if multiple tasks exist)
    print("Creating task comparison...")
    create_task_comparison(output_path, results_path.parent)

    # Generate HTML report
    print("Generating HTML report...")
    generate_html_report(output_path, summary)

    print(f"\n✓ Compendium generated!")
    print(f"  - {output_path / 'index.html'}")
    print(f"  - {output_path / 'metrics_overview.png'}")
    print(f"  - {output_path / 'task_comparison.png'}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate compendium')
    parser.add_argument('--results_dir', type=str, default='geometric_benchmark/results',
                        help='Results directory')
    parser.add_argument('--output_dir', type=str, default='geometric_benchmark/results/compendium',
                        help='Output directory')

    args = parser.parse_args()

    generate_compendium(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
    )
