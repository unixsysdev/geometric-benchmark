# Geometric Benchmark - Quick Start Guide

Get up and running with the Geometric Deep Learning Benchmark Suite in minutes.

---

## Installation

```bash
# Navigate to the benchmark directory
cd geometric_benchmark

# Install dependencies
pip install torch numpy matplotlib seaborn scipy scikit-learn jinja2 umap-learn

# For dashboard (optional)
pip install flask streamlit
```

---

## Run Your First Task

### Option 1: Single Task

Train a modular addition model:

```bash
python tasks/periodic_1d/train.py \
    --task mod_add \
    --p 97 \
    --epochs 5000 \
    --model_size tiny \
    --pos_enc sinusoidal \
    --save_analysis
```

### Option 2: Run Full Benchmark

Run all periodic 1D tasks:

```bash
python run_benchmark.py --tasks periodic_1d
```

Run all configured tasks:

```bash
python run_benchmark.py --tasks all
```

---

## What You Get

After training, you'll find:

### Checkpoints
`checkpoints/mod_add_97/` - Trained models

### Results
`results/periodic_1d/mod_add/` - Analysis including:
- `step_5000/` - Analysis at step 5000
  - `fourier_spectrum.png` - Fourier power spectrum
  - `periodic_neurons.png` - Periodic neuron visualizations
  - `attention_l0h0.png` - Attention patterns
  - `embeddings_pca.png` - Embedding geometry
  - `analysis_results.json` - Raw analysis data

- `training_curves.png` - Loss and accuracy over time

---

## Generate Compendium

Create a unified report:

```bash
python compendium/generate.py \
    --results_dir results/periodic_1d/mod_add \
    --output_dir results/periodic_1d/mod_add/compendium
```

Open `results/periodic_1d/mod_add/compendium/index.html` in your browser.

---

## Available Tasks

### Periodic 1D
```bash
python tasks/periodic_1d/train.py --task mod_add --p 97
python tasks/periodic_1d/train.py --task mod_mul --p 97
python tasks/periodic_1d/train.py --task digit_sum --max_digits 3
python tasks/periodic_1d/train.py --task parity --max_val 1000
```

### Model Sizes
- `tiny`: 2 layers, 128 dim, 4 heads (fastest)
- `small`: 4 layers, 256 dim, 8 heads
- `medium`: 6 layers, 512 dim, 8 heads

### Position Encodings
- `sinusoidal`: SINPE (periodic bias)
- `learned`: Learned embeddings

---

## Configuration File

Create custom benchmarks with YAML:

```bash
python run_benchmark.py --config configs/benchmark.yaml
```

Edit `configs/benchmark.yaml` to customize:
- Tasks
- Model architectures
- Training parameters
- Analysis frequency

---

## GPU vs CPU

```bash
# Use GPU (default)
python tasks/periodic_1d/train.py --device cuda

# Use CPU
python tasks/periodic_1d/train.py --device cpu
```

---

## Expected Results

For mod-p addition with SINPE:

- **Training time**: ~5-10 minutes for 5000 steps (GPU)
- **Final accuracy**: >95%
- **Fourier concentration**: >0.7 (top-1 power ratio)
- **Periodic neurons**: 5-20 detected
- **Periodicity score**: >0.3

Without SINPE (learned embeddings):
- Slower grokking
- Lower Fourier concentration
- Fewer periodic neurons

---

## Common Issues

### Out of Memory
```bash
# Reduce batch size
python tasks/periodic_1d/train.py --batch_size 16

# Use smaller model
python tasks/periodic_1d/train.py --model_size tiny
```

### No GPU Detected
```bash
# Install PyTorch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Or use CPU (slower)
python tasks/periodic_1d/train.py --device cpu
```

### Import Errors
```bash
# Ensure you're in the geometric_benchmark directory
cd geometric_benchmark

# Install missing dependencies
pip install -r requirements.txt  # (create this file if needed)
```

---

## Next Steps

1. **Explore different tasks**: Try `digit_sum`, `parity`, `mod_mul`
2. **Vary hyperparameters**: Change `--train_frac`, `--weight_decay`, `--lr`
3. **Compare representations**: SINPE vs learned, small vs large models
4. **Add custom tasks**: Follow the template in `tasks/periodic_1d/`

---

## For Developers

See `README.md` for:
- Architecture overview
- Adding new tasks
- Extending the analysis pipeline
- Building the dashboard

---

**Happy experimenting!** 🧪

*"The most exciting phrase to hear in science, the one that heralds new discoveries, is not 'Eureka!' but 'That's funny...'"* — Isaac Asimov
