"""
Geometric Benchmark Dashboard

Interactive web interface for exploring experiment results.

Usage:
    streamlit run app.py
"""

import streamlit as st
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

# Add library to path
library_path = Path(__file__).parent.parent / 'library'
sys.path.insert(0, str(library_path))

# Page config
st.set_page_config(
    page_title="Geometric Benchmark Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
    </style>
""", unsafe_allow_html=True)


def load_results(results_dir):
    """Load all analysis results from directory."""
    results = {}

    results_path = Path(results_dir)
    if not results_path.exists():
        return results

    # Walk through all subdirectories
    for category_dir in results_path.iterdir():
        if not category_dir.is_dir():
            continue

        category = category_dir.name
        results[category] = {}

        for task_dir in category_dir.iterdir():
            if not task_dir.is_dir():
                continue

            task = task_dir.name
            results[category][task] = {}

            # Load all step analyses
            for step_dir in task_dir.glob('step_*'):
                step = int(step_dir.name.split('_')[1])

                results_file = step_dir / 'analysis_results.json'
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        results[category][task][step] = json.load(f)

    return results


def plot_fourier_spectrum(fourier_data):
    """Plot Fourier spectrum using Plotly."""
    if not fourier_data:
        return None

    # Extract peaks
    peaks = fourier_data.get('peaks', [])

    if not peaks:
        return None

    # Create plot
    fig = go.Figure()

    freqs = [p[0] for p in peaks]
    powers = [p[1] for p in peaks]

    fig.add_trace(go.Scatter(
        x=freqs,
        y=powers,
        mode='lines+markers',
        name='Power Spectrum',
        line=dict(color='#667eea', width=2),
    ))

    # Mark dominant frequency
    if fourier_data.get('dominant_frequency'):
        fig.add_vline(
            x=fourier_data['dominant_frequency'],
            line_dash='dash',
            line_color='red',
            annotation_text=f"Dominant: {fourier_data['dominant_frequency']:.3f}"
        )

    fig.update_layout(
        title='Fourier Spectrum',
        xaxis_title='Frequency',
        yaxis_title='Power',
        hovermode='x unified',
    )

    return fig


def plot_metrics_over_time(results_dict):
    """Plot metrics over training time."""
    if not results_dict:
        return None

    steps = sorted(results_dict.keys())

    metrics_over_time = {
        'fourier_top1': [],
        'fourier_top3': [],
        'n_periodic': [],
        'periodicity_score': [],
    }

    for step in steps:
        step_results = results_dict[step]

        if 'fourier' in step_results:
            metrics_over_time['fourier_top1'].append(step_results['fourier'].get('top1_power_ratio', 0))
            metrics_over_time['fourier_top3'].append(step_results['fourier'].get('top3_power_ratio', 0))
        else:
            metrics_over_time['fourier_top1'].append(0)
            metrics_over_time['fourier_top3'].append(0)

        if 'periodic_neurons' in step_results:
            metrics_over_time['n_periodic'].append(len(step_results['periodic_neurons']))
        else:
            metrics_over_time['n_periodic'].append(0)

        if 'position_encoding' in step_results:
            metrics_over_time['periodicity_score'].append(
                step_results['position_encoding'].get('periodicity_score', 0)
            )
        else:
            metrics_over_time['periodicity_score'].append(0)

    # Create subplot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Fourier Top-1 Power', 'Fourier Top-3 Power',
                       'Periodic Neurons', 'Periodicity Score'),
    )

    fig.add_trace(go.Scatter(x=steps, y=metrics_over_time['fourier_top1'],
                             mode='lines+markers', name='Top-1'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=steps, y=metrics_over_time['fourier_top3'],
                             mode='lines+markers', name='Top-3'),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=steps, y=metrics_over_time['n_periodic'],
                             mode='lines+markers', name='Count'),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=steps, y=metrics_over_time['periodicity_score'],
                             mode='lines+markers', name='Score'),
                  row=2, col=2)

    fig.update_layout(height=600, showlegend=False, title_text="Metrics Over Training")

    return fig


def display_periodic_neurons(periodic_neurons):
    """Display periodic neurons as a table."""
    if not periodic_neurons:
        st.info("No periodic neurons detected")
        return

    df = pd.DataFrame(periodic_neurons)

    # Rename columns for display
    df = df.rename(columns={
        'neuron_idx': 'Neuron',
        'r_squared': 'R² Score',
        'frequency': 'Frequency',
        'amplitude': 'Amplitude',
        'phase': 'Phase',
    })

    st.dataframe(
        df.head(20),
        use_container_width=True,
        hide_index=True,
    )


def main():
    # Header
    st.markdown('<h1 class="main-header">🔬 Geometric Benchmark Dashboard</h1>',
                unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar
    st.sidebar.title("Navigation")

    results_dir = st.sidebar.text_input(
        "Results Directory",
        value="results",
        help="Path to benchmark results directory"
    )

    # Load results
    results = load_results(results_dir)

    if not results:
        st.error(f"No results found in {results_dir}")
        st.info("Run some experiments first!")
        return

    # Select category
    categories = list(results.keys())
    if not categories:
        st.warning("No task categories found")
        return

    selected_category = st.sidebar.selectbox(
        "Task Category",
        categories,
    )

    # Select task
    tasks = list(results.get(selected_category, {}).keys())
    if not tasks:
        st.warning(f"No tasks found in {selected_category}")
        return

    selected_task = st.sidebar.selectbox(
        "Task",
        tasks,
    )

    # Get task results
    task_results = results[selected_category][selected_task]
    steps = sorted(task_results.keys())

    if not steps:
        st.warning("No analysis checkpoints found")
        return

    # Select checkpoint
    selected_step = st.sidebar.selectbox(
        "Checkpoint Step",
        steps,
        index=len(steps) - 1,  # Default to latest
    )

    # Display results
    st.header(f"{selected_category} / {selected_task} (Step {selected_step})")

    step_results = task_results[selected_step]

    # Key metrics
    st.subheader("Key Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if 'fourier' in step_results:
            metric_value = step_results['fourier'].get('top1_power_ratio', 0)
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{metric_value:.3f}</div>
                    <div class="metric-label">Fourier Top-1</div>
                </div>
            """, unsafe_allow_html=True)

    with col2:
        if 'periodic_neurons' in step_results:
            metric_value = len(step_results['periodic_neurons'])
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{metric_value}</div>
                    <div class="metric-label">Periodic Neurons</div>
                </div>
            """, unsafe_allow_html=True)

    with col3:
        if 'position_encoding' in step_results:
            metric_value = step_results['position_encoding'].get('periodicity_score', 0)
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{metric_value:.3f}</div>
                    <div class="metric-label">Periodicity Score</div>
                </div>
            """, unsafe_allow_html=True)

    with col4:
        if 'embeddings_pca' in step_results:
            metric_value = step_results['embeddings_pca']['metrics'].get('total_variance_explained', 0)
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{metric_value:.3f}</div>
                    <div class="metric-label">PCA Variance</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Metrics over time
    st.subheader("Metrics Over Training")
    fig_over_time = plot_metrics_over_time(task_results)
    if fig_over_time:
        st.plotly_chart(fig_over_time, use_container_width=True)

    # Fourier analysis
    if 'fourier' in step_results:
        st.subheader("Fourier Analysis")

        col1, col2 = st.columns([2, 1])

        with col1:
            fourier_fig = plot_fourier_spectrum(step_results['fourier'])
            if fourier_fig:
                st.plotly_chart(fourier_fig, use_container_width=True)

        with col2:
            st.markdown("**Statistics**")
            fourier = step_results['fourier']
            st.metric("Dominant Frequency", f"{fourier.get('dominant_frequency', 0):.4f}")
            st.metric("Top-1 Power Ratio", f"{fourier.get('top1_power_ratio', 0):.3f}")
            st.metric("Top-3 Power Ratio", f"{fourier.get('top3_power_ratio', 0):.3f}")
            st.metric("Top-5 Power Ratio", f"{fourier.get('top5_power_ratio', 0):.3f}")

    # Periodic neurons
    if 'periodic_neurons' in step_results:
        st.subheader("Periodic Neurons")
        display_periodic_neurons(step_results['periodic_neurons'])

    # Position encoding
    if 'position_encoding' in step_results:
        st.subheader("Position Encoding Analysis")

        pos_enc = step_results['position_encoding']

        col1, col2, col3 = st.columns(3)
        col1.metric("Periodic Dimensions", pos_enc.get('n_periodic', 0))
        col2.metric("Strong Periodic", pos_enc.get('n_strong_periodic', 0))
        col3.metric("Periodicity Score", f"{pos_enc.get('periodicity_score', 0):.3f}")

        if pos_enc.get('periodic_dimensions'):
            st.markdown("**Top Periodic Dimensions**")
            top_dims = pos_enc['periodic_dimensions'][:5]
            df = pd.DataFrame(top_dims)
            df = df.rename(columns={
                'dimension': 'Dimension',
                'r_squared': 'R²',
                'frequency': 'Frequency',
                'amplitude': 'Amplitude',
            })
            st.dataframe(df, use_container_width=True, hide_index=True)

    # Task-specific analyses
    st.markdown("---")
    st.subheader("Task-Specific Analysis")

    # Winding number specific
    if 'winding_correlation' in step_results:
        st.metric("Winding Correlation", f"{step_results['winding_correlation']:.3f}")

    # Sphere specific
    if 'latitude_correlation' in step_results:
        st.metric("Latitude Correlation", f"{step_results['latitude_correlation']:.3f}")
    if 'spherical_structure_score' in step_results:
        st.metric("Spherical Structure", f"{step_results['spherical_structure_score']:.3f}")

    # Permutation specific
    if 'clustering_score' in step_results:
        st.metric("Cycle Type Clustering", f"{step_results['clustering_score']:.3f}")
    if 'identity_centrality' in step_results:
        st.metric("Identity Centrality", f"{step_results['identity_centrality']:.3f}")

    # Cascaded modular specific
    if 'x_freq_match_p1' in step_results:
        st.metric("X Position Matches p1", "✓" if step_results['x_freq_match_p1'] else "✗")
    if 'b_freq_match_p2' in step_results:
        st.metric("B Position Matches p2", "✓" if step_results['b_freq_match_p2'] else "✗")

    # Image display
    st.markdown("---")
    st.subheader("Analysis Plots")

    # Try to load and display images
    step_dir = Path(results_dir) / selected_category / selected_task / f'step_{selected_step}'

    image_files = list(step_dir.glob('*.png'))

    if image_files:
        cols = st.columns(min(3, len(image_files)))

        for idx, img_file in enumerate(image_files):
            with cols[idx % len(cols)]:
                st.image(str(img_file), caption=img_file.stem, use_container_width=True)
    else:
        st.info("No visualization images found. Run with --save_analysis flag.")


if __name__ == '__main__':
    main()
