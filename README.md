# Geometric Deep Learning Benchmark Suite

**The ImageNet of Mechanistic Interpretability**

A unified benchmark for studying how neural networks discover geometric, topological, and algebraic structure. Like ImageNet standardized computer vision, this suite standardizes the study of **algorithmic reasoning** and **representation learning** in neural networks.

---

## Vision

Modern deep learning has discovered that neural networks can learn rich internal representations — but we lack a systematic way to study **what** they learn and **how** they learn it.

This benchmark provides:
1. **Curated tasks** spanning geometry, topology, and algebra
2. **Unified analysis pipeline** for mechanistic interpretability
3. **Standardized metrics** for representation quality
4. **Interactive dashboard** for exploration and comparison

---

## Task Categories

### 1. Periodic 1D Tasks
- **Modular arithmetic**: `a + b mod p`, `a × b mod p`
- **Digit operations**: Sum of digits, digital root
- **Parity**: Even/odd classification
- **Carry prediction**: Detect when addition produces carries

**Why**: These tasks have known Fourier structure. They test whether networks discover sinusoidal representations.

**Expected signature**: Strong periodic neurons, clear Fourier peaks, low-dimensional rotational embeddings.

---

### 2. Periodic 2D Tasks
- **Torus operations**: Distance/convolution on wrap-around grid
- **Grid convolution**: Linear filtering on 2D lattice
- **2D heat equation**: Predict diffusion on torus
- **Random walk**: Next-step prediction on periodic grid

**Why**: Tests whether networks discover **separable** representations (f(x,y) = g(x)h(y)) and 2D Fourier modes.

**Expected signature**: Separable neuron activations, 2D FFT peaks in (k,l) space, attention respecting periodic boundaries.

---

### 3. Topological Tasks
- **Winding number**: How many times does a curve wrap around origin?
- **Linking number**: How many times do two curves entangle?
- **Homotopy classification**: Can loop A deform into loop B?
- **Betti number**: Count holes in point cloud

**Why**: Tests whether networks can discover **global topological invariants** from local structure.

**Expected signature**: Angular encoding neurons (sin θ, cos θ), accumulation circuits, attention along curves.

---

### 4. Manifold Tasks
- **Sphere distance**: Geodesic distance on S²
- **Projective plane**: Antipodal identification
- **Klein bottle**: Non-orientable surface operations
- **Hyperbolic distance**: Geodesics in H²

**Why**: Tests understanding of **curvature** and intrinsic geometry.

**Expected signature**: Spherical harmonic activations, curvature-sensitive neurons, geodesic-aware attention.

---

### 5. Group-Theoretic Tasks
- **Permutation composition**: S₃, S₄ operation tables
- **Dihedral operations**: Rotations + reflections of n-gon
- **Quaternion multiplication**: Non-abelian 4D number system
- **Matrix operations**: Special linear group SL(2)

**Why**: Tests whether networks discover **representation theory** — irreducible representations of abstract groups.

**Expected signature**: Group-theoretic factorization, irreducible representation activations, non-commutative circuits.

---

### 6. Multi-Scale Tasks
- **Cascaded modular**: `(((x + a₁) mod p₁) + a₂) mod p₂`
- **Wavelet reconstruction**: Reconstruct multi-frequency signals
- **Fractal prediction**: Self-similar pattern recognition
- **Renormalization**: Coarse-graining predictions

**Why**: Tests whether networks discover **scale-selective representations** and hierarchical structure.

**Expected signature**: Wavelet-like neurons, scale-separated attention layers, renormalization flow.

---

## Architecture

```
geometric_benchmark/
├── tasks/                    # Individual task implementations
│   ├── periodic_1d/
│   ├── periodic_2d/
│   ├── topological/
│   ├── manifold/
│   ├── group_theoretic/
│   └── multi_scale/
│
├── library/                  # Shared code (DO NOT modify task-specific code here)
│   ├── models.py            # Unified model architectures
│   ├── datasets.py          # Dataset generators
│   ├── training.py          # Training loops, checkpointing
│   ├── analysis.py          # Fourier, periodic neuron detection
│   ├── attention.py         # Attention pattern analysis
│   ├── embeddings.py        # Embedding geometry analysis
│   └── metrics.py           # Unified evaluation metrics
│
├── compendium/              # Unified report generation
│   └── generate.py          # Compendium generator
│
├── dashboard/               # Web interface
│   ├── app.py              # Flask/Streamlit app
│   ├── templates/          # HTML templates
│   └── static/             # CSS/JS
│
├── configs/                 # Task configurations
│   └── *.yaml
│
├── checkpoints/             # Trained models
├── results/                 # Analysis outputs
└── README.md
```

---

## Usage

### Running a Single Task

```bash
# Train on mod-p addition
python tasks/periodic_1d/train_mod_p.py --p 97 --epochs 20000

# Generate compendium
python compendium/generate.py --task periodic_1d/mod_p --checkpoint checkpoints/mod_p_p97.pt
```

### Running the Full Benchmark

```bash
# Run all tasks with default configs
python run_benchmark.py --tasks all

# Run specific category
python run_benchmark.py --tasks periodic_1d topological

# Custom configuration
python run_benchmark.py --config configs/custom.yaml
```

### Launching the Dashboard

```bash
cd dashboard
python app.py --port 8080
```

Visit `http://localhost:8080` to explore:
- Task performance comparison
- Fourier spectrum visualizations
- Attention pattern analysis
- Embedding geometry plots
- Transfer learning matrices
- Feature clustering

---

## Unified Analysis Pipeline

Every task produces a **compendium** — a standardized analysis report including:

### 1. Training Metrics
- Loss curves
- Train/test accuracy
- Generalization gap
- Grokking detection (if applicable)

### 2. Fourier Analysis
- Spectrum of activations (1D or 2D FFT)
- Dominant frequencies
- R² fits to sinusoidal bases
- Comparison to ground-truth harmonic basis (if available)

### 3. Periodic Neuron Detection
- Neuron activations vs input position
- Sinusoid fit quality per neuron
- Spatial frequency maps
- "Periodicity score" per layer

### 4. Attention Analysis
- Attention entropy and sparsity
- Attention patterns (heatmaps)
- Position-bias analysis
- Head specialization (which heads do what?)

### 5. Embedding Geometry
- PCA/UMAP of learned embeddings
- Nearest neighbor analysis
- Geodesic distance preservation (for manifold tasks)
- Cyclic structure detection (for periodic tasks)

### 6. Mechanistic Probes
- Ablation studies (which neurons matter?)
- Logit lens (how do representations evolve?)
- Circuit isolation (find minimal subgraph)
- Cross-task similarity (do different tasks share circuits?)

### 7. Generalization Tests
- Extrapolation (larger numbers, different positions)
- Interpolation (between training examples)
- Transfer learning (from task A to task B)
- Zero-shot capabilities

---

## Standardized Metrics

### Representation Quality
- **Fourier Score**: Fraction of variance explained by top-k frequencies
- **Periodicity Index**: Fraction of neurons with strong periodic activation
- **Separability Score**: For 2D tasks, correlation with separable basis
- **Alignment Score**: Similarity to ground-truth harmonic basis (if known)

### Generalization
- **Interpolation Accuracy**: On held-out examples within training distribution
- **Extrapolation Accuracy**: On examples outside training distribution
- **Transfer Efficiency**: Accuracy gain from pre-training vs from scratch

### Interpretability
- **Circuit Sparsity**: Minimum neurons needed for task performance
- **Attention Localization**: How focused is attention on relevant positions?
- **Probe Linear Separability**: Can linear probes extract concepts?

---

## Research Questions

This benchmark enables systematic investigation of:

1. **Representation discovery**: Do networks discover mathematically optimal representations?

2. **Inductive biases**: How do architecture and training affect representation choice?

3. **Transfer learning**: Which geometric skills transfer across tasks?

4. **Compositionality**: Can learned representations combine for new tasks?

5. **Grokking phenomena**: Under what conditions do generalizing circuits emerge?

6. **Scale dependence**: How do representations change with model size and data?

---

## Contributing

To add a new task:

1. Create task folder under `tasks/`
2. Implement dataset generator (inherits from `BaseDataset`)
3. Define task-specific metrics (if any)
4. Add config file in `configs/`
5. Register task in `run_benchmark.py`
6. Document in README

**Principle**: All tasks should use the shared library in `library/`. Do NOT duplicate analysis code.

---

## Citation

If you use this benchmark in your research:

```bibtex
@misc{geometric_benchmark_2025,
  title={Geometric Deep Learning Benchmark Suite},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/grokking-fourier}
}
```

---

## License

MIT License — See LICENSE file for details

---

## Acknowledgments

Built on top of the `grokking-fourier` project. Inspired by:
- Neel Nanda's grokking work
- Anthropic's mechanistic interpretability research
- Transformer Circuits work (Olsson et al.)
- Geometric Deep Learning (Bronstein et al.)

---

*"Mathematics is the art of giving the same name to different things." — Henri Poincaré*

This benchmark is about discovering when neural networks do the same: finding the same underlying structure (Fourier modes, eigenvectors, irreps) across superficially different tasks.
