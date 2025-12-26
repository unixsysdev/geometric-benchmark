# Ambitious Experiment Ideas for Grokking-Fourier

*Generated from project analysis - December 2025*

These ideas go beyond simple variations (multiplication, digit-sum, carry-prediction) toward genuinely novel research directions that could produce publishable results.

---

## 1. Algorithm Archaeology — Reverse-Engineering Multi-Step Reasoning

**Concept**: Instead of single-operation tasks, train on **multi-step algorithms** and use mechanistic tools to map the *temporal evolution* of internal representations.

**Concrete Task**: Train a tiny transformer to compute **continued fraction expansion** of rationals a/b → [a₀; a₁, a₂, ...]. This involves iterative division (Euclidean algorithm).

**Why it's exciting**:
- Each layer could implement one "step" of the algorithm
- You'd see **different** Fourier signatures at different depths (division vs modulo)
- Logit lens would show when each coefficient "crystallizes"
- Could reveal if transformers learn sequential algorithms or parallel shortcuts

**Visualization**: A "circuit timeline" showing which neurons activate at each algorithmic step, like a debugger trace through the network.

**Difficulty**: Medium-Hard  
**Novelty**: High  
**Visual Appeal**: Very High

---

## 2. Symmetry Discovery — Learning Group Structure from Data

**Concept**: Train models on group operation tables and see if they discover the underlying mathematical structure.

**Concrete Tasks**:
- **Symmetric group S₃/S₄**: Input two permutations, output their composition
- **Dihedral group Dₙ**: Rotations + reflections of an n-gon
- **Quaternion group Q₈**: Non-abelian, 8 elements

**Why it's exciting**:
- Different groups should produce qualitatively different internal representations
- Abelian groups → Fourier (already demonstrated!)
- Non-abelian groups → ??? (matrix representations? something new?)
- You could literally **classify what representation theory the model discovers**

**Visualization**: Compare the embedding geometry for Z/pZ (circle) vs S₃ (something 6-dimensional?) vs Q₈ (quaternionic?). Are there "Cayley graph" structures in attention?

**Key Question**: Does the model discover irreducible representations of non-abelian groups?

**Difficulty**: Medium  
**Novelty**: Very High  
**Visual Appeal**: High

---

## 3. Phase Transition Atlas — Mapping the Grokking Landscape

**Concept**: Systematically explore the (dataset_size, weight_decay, depth, width) space and build a **phase diagram** of when grokking occurs.

**Parameters to Vary**:
- Dataset fraction: 1%, 5%, 10%, 25%, 50%, 75%, 100%
- Weight decay: 0, 0.01, 0.1, 0.5, 1.0
- Model depth: 1, 2, 3, 4 layers
- Model width: 32, 64, 128, 256 dimensions
- Task: mod-p addition for various p

**Why it's exciting**:
- This would be a genuine **scientific contribution** — nobody has mapped this comprehensively
- You'd find the "critical exponents" of grokking (physics analogy: phase transitions)
- Could identify if there are **multiple grokking regimes** (different algorithms)
- The visualization would be stunning: 2D heatmaps with "grokking frontier" boundaries

**Deliverable**: An interactive explorer where you slide parameters and see when Fourier circuits emerge.

**Analysis**: Track Fourier signal strength as order parameter, plot phase boundaries

**Difficulty**: Medium (lots of compute, but straightforward)  
**Novelty**: High  
**Visual Appeal**: Very High (phase diagrams are beautiful)

---

## 4. Compositional Circuits — Hierarchical Fourier Structure

**Concept**: Can the model learn **nested modular operations**?

**Concrete Tasks**:
- `((a + b) mod p₁) mod p₂` where p₂ < p₁
- `(a × b mod p) + (c × d mod q) mod r`
- `f(g(x))` where f and g are both mod-p operations

**Why it's exciting**:
- Tests if Fourier representations **compose**
- Would reveal if the model builds a "Fourier algebra" internally
- Attention patterns might show explicit routing between "modules"
- Could demonstrate hierarchical circuit formation

**Visualization**: A "circuit hierarchy" diagram showing which neurons handle which level of nesting.

**Key Question**: Do nested operations produce nested Fourier structures, or does the model find shortcuts?

**Difficulty**: Medium  
**Novelty**: High  
**Visual Appeal**: Medium-High

---

## 5. Adversarial Grokking — Finding the Limits

**Concept**: Design tasks that should be *impossible* to grok with Fourier methods, and see what the model does instead.

**Concrete Tasks**:
- **Primality testing**: Is n prime? (No clean Fourier structure)
- **Collatz steps**: How many steps until n reaches 1?
- **Discrete log**: Find x such that gˣ ≡ h (mod p)
- **Integer factorization**: Given n, output smallest prime factor
- **Parity of prime counting function**: Is π(n) even or odd?

**Why it's exciting**:
- You'd discover what **non-Fourier** circuits look like
- Might find qualitatively different mechanistic phenomena
- Could identify the **boundaries** of what small transformers can learn algorithmically
- Negative results are also interesting: "This task cannot be grokked because..."

**Analysis**: Compare activation patterns to Fourier tasks — what's different?

**Difficulty**: Medium  
**Novelty**: High  
**Visual Appeal**: Medium (but scientifically very interesting)

---

## 6. Cross-Modal Transfer — Do Fourier Circuits Generalize?

**Concept**: Pre-train on mod-p arithmetic, then fine-tune on something *structurally similar but superficially different*.

**Transfer Targets**:
- **Musical intervals**: Note + interval → resulting note (mod 12)
- **Color wheel**: Color + rotation → resulting color (periodic in hue space)
- **Clock arithmetic**: Hour + offset → resulting hour (mod 12/24)
- **Weekday arithmetic**: Day + offset → resulting day (mod 7)
- **Compass directions**: Direction + rotation → resulting direction (mod 8/16)

**Experimental Design**:
1. Pre-train on a + b mod p with SINPE
2. Freeze embeddings, fine-tune on new domain
3. Measure transfer efficiency vs training from scratch
4. Analyze if same Fourier neurons activate

**Why it's exciting**:
- Tests if learned representations are truly **abstract** or tied to specific tokens
- Could show positive transfer (proves the circuit is general) or negative transfer (proves it's brittle)
- Would be a beautiful demonstration of "mathematical inductive bias"

**Difficulty**: Easy-Medium  
**Novelty**: Medium-High  
**Visual Appeal**: High (showing same circuits lighting up for music and math)

---

## 7. Emergence Timeline — Watching Circuits Form in Real-Time

**Concept**: Instead of analyzing checkpoints, build a **real-time visualization** of circuit formation during training.

**Metrics to Track (every N steps)**:
- Fourier signal strength (R² for sin/cos fits)
- Neuron periodicity scores
- Attention entropy / sparsity
- Weight matrix FFT peaks
- Embedding geometry (PCA/UMAP)
- Per-position accuracy

**Implementation**:
- Hook into training loop
- Compute lightweight metrics every 100 steps
- Generate frames for animation
- Overlay with loss curves and accuracy

**Output**: An **animated video** showing:
- Left panel: Loss and accuracy curves
- Middle panel: Fourier strength heatmap over neurons
- Right panel: Embedding geometry evolution

**Why it's exciting**:
- Would literally *show* grokking happening
- Could identify the exact moment Fourier circuits "snap into place"
- Beautiful for papers/presentations/Twitter
- Might reveal oscillations, false starts, or multi-phase transitions

**Difficulty**: Medium  
**Novelty**: Medium (visualization-focused)  
**Visual Appeal**: Extremely High

---

## 8. Universal Function Approximator Probe — What Else Can These Circuits Do?

**Concept**: Take a trained mod-p model and probe whether its Fourier circuits can be repurposed for other tasks.

**Probing Tasks**:
- Feed it periodic functions (sin, cos, triangle wave, sawtooth) on continuous-ish input → does it extrapolate?
- Test on signal processing tasks (low-pass filter, convolution kernels)
- Check if embeddings work for other periodic phenomena (seasons, tides, biorhythms)
- Try to use the Fourier features for regression on periodic data

**Method**:
1. Extract learned Fourier features from trained model
2. Use as fixed features for downstream tasks
3. Compare to features from untrained model

**Why it's exciting**:
- Would reveal if the model learned "true" trigonometry or just a lookup table
- Could lead to "Fourier Transformers as Signal Processors" — a new paper idea
- Tests the generality of learned representations

**Difficulty**: Easy-Medium  
**Novelty**: Medium  
**Visual Appeal**: Medium

---

## Priority Ranking

For **highest chance of stunning, publishable results**:

| Rank | Experiment | Rationale |
|------|------------|-----------|
| 1 | **#3 Phase Transition Atlas** | Most likely to produce beautiful visualizations and clear scientific claims |
| 2 | **#2 Symmetry Discovery** | Most intellectually exciting, could genuinely reveal something new about how NNs represent algebra |
| 3 | **#7 Emergence Timeline** | Most visually impressive for demos/talks/Twitter |
| 4 | **#1 Algorithm Archaeology** | Novel angle on multi-step reasoning |
| 5 | **#6 Cross-Modal Transfer** | Easy win, good story |
| 6 | **#4 Compositional Circuits** | Natural extension of current work |
| 7 | **#5 Adversarial Grokking** | Important for understanding limits |
| 8 | **#8 Function Approximator Probe** | Lower novelty but could surprise |

---

## Quick Wins (can implement in a day)

- **#6 Cross-Modal Transfer**: Just needs new tokenization for music/clock domains
- **#7 Emergence Timeline**: Add logging hooks to existing training loop
- **#3 Phase Transition Atlas** (small version): 3x3 grid of (weight_decay, dataset_fraction)

## Ambitious Projects (week+ of work)

- **#2 Symmetry Discovery**: Need to implement group operations and new analysis
- **#1 Algorithm Archaeology**: Multi-step output, new probing methodology
- **#5 Adversarial Grokking**: Multiple hard tasks to implement and analyze

---

## 9. Torus Fourier Discovery — 2D Periodic Structure Learning

**Concept**: Train models on tasks defined on a toroidal (wrap-around) grid and watch them discover 2D Fourier modes organically.

**Concrete Tasks**:
- **Heat equation solver**: Initial condition f(x,y) on N×N torus → predict f(x,y) after t steps
- **Random walk next-step**: Current position (x,y) → predict next position in a random walk
- **Torus distance**: Compute shortest path between two points on torus (wrapping allowed)
- **2D convolution response**: Given a kernel and position, predict output value

**Dataset Structure**:
- Grid size: 16×16 or 32×32 (keeps it tractable)
- Position encoding: separate tokens for x and y coordinates, or joint positional token
- Output: scalar or vector prediction

**Why it's exciting**:
- The torus has natural **2D Fourier basis** (product of 1D bases: sin(kx)sin(ly), sin(kx)cos(ly), etc.)
- You can literally **compare learned features to true eigenfunctions**
- Visualization would be stunning: eigenfunction heatmaps with stripe patterns, checkerboards, diagonal waves
- Clear test of whether models discover **separable representations** (f(x,y) = g(x)h(y)) or learn entangled 2D features

**Mechanistic Analysis**:
- Extract neuron activations as function of (x,y) position
- Perform 2D FFT on activation maps
- Compare to true Laplacian eigenmodes
- Test if attention patterns respect torus topology (wrap-around connections)
- Check for separability: do some neurons specialize in x-direction, others in y-direction?

**Visualization Deliverables**:
- **Eigenfunction gallery**: Grid of heatmaps showing learned vs theoretical modes
- **2D Fourier spectrum heatmap**: Mode (k,l) → activation strength
- **Separability analysis**: Correlation between learned features and separable basis
- **Attention torus map**: Heatmap of which positions attend to which

**Difficulty**: Medium
**Novelty**: High (first systematic study of 2D Fourier learning)
**Visual Appeal**: Extremely High (eigenfunction patterns are visually striking)

---

## 10. Topological Invariant Learning — Winding Numbers & Homotopy

**Concept**: Train models to predict topological invariants of curves and shapes, revealing whether they discover homotopy/cohomology structure.

**Concrete Tasks**:
- **Winding number classifier**: Given a closed curve (as point sequence), predict how many times it wraps around origin
- **Linking number predictor**: Given two curves, predict their linking number
- **Homotopy class**: Classify loops into equivalence classes (can one be continuously deformed into another?)
- **Betti number detector**: For point clouds, predict number of "holes" (β₁)

**Dataset Generation**:
- Generate random closed curves via:
  - Superposition of epicycles (sum of rotating vectors)
  - Splines with random control points
  - Perturbed circles/ellipses
- Compute ground truth winding numbers analytically
- For linking: generate pairs of knots with varying linking numbers (-3, -2, -1, 0, 1, 2, 3)

**Why it's exciting**:
- Topological invariants are **integer-valued and discrete**, but derived from continuous structure
- The model must discover a **global property** from local information
- Winding number is fundamentally about angle accumulation → natural harmonic structure
- This would be the first mechanistic study of topological reasoning in transformers

**Mechanistic Analysis**:
- Do neurons encode angular position around origin (sin θ, cos θ)?
- Is there an "angle accumulator" circuit that sums winding?
- Attention patterns: do they follow the curve sequentially?
- Fourier analysis: Are dominant frequencies related to winding number?

**Key Questions**:
- Does the model discover the "integral of curvature" formula algorithmically?
- Are there "winding detector" neurons?
- Can we interpret the decision boundary in embedding space?

**Visualization Deliverables**:
- **Curve gallery**: Example inputs colored by predicted winding number
- **Activation vs angle plots**: Neuron activations as function of position on curve
- **Attention trace**: How attention flows along the curve
- **Embedding UMAP**: Clustering by topological class

**Difficulty**: Medium-Hard
**Novelty**: Very High (topological mechanistic interpretability is unexplored)
**Visual Appeal**: High (curves, knots, and clean classification boundaries)

---

## 11. Graph Laplacian Eigenlearning — Spectral Geometry Discovery

**Concept**: Train models to predict Laplacian eigenvectors/eigenvalues of various graph topologies from node IDs alone.

**Concrete Tasks**:
- **Eigenvector prediction**: Given node ID and mode index k, output value of k-th eigenvector at that node
- **Eigenvalue classification**: Given graph topology description, predict the spectrum
- **Harmonic function solver**: Given boundary conditions, predict solution to Laplace's equation Δf = 0

**Graph Topologies**:
- Cycle graph Cₙ (circle → 1D Fourier modes)
- Grid graph G_{m×n} (2D separable modes)
- Torus graph T_{m×n} (2D periodic modes)
- Complete graph Kₙ (trivial: all eigenvectors constant)
- Random geometric graphs (irregular spectra)
- Hypercube Qₙ (interesting recursive structure)

**Dataset Generation**:
- For each topology, compute graph Laplacian L = D - A
- Compute first K eigenvectors (e.g., K = 10)
- Samples: (node_id, mode_k) → eigenvalue_φₖ(node_id)

**Why it's exciting**:
- **Direct connection to spectral graph theory** — a rich mathematical field
- Different topologies produce qualitatively different eigenfunctions:
  - Cycle: pure sinusoids
  - Grid: separable sin/sin patterns
  - Torus: 2D periodic modes
  - Hypercube: Walsh functions (discrete square waves)
- You can **compare learned representations to ground-truth theory**
- Tests if models discover **intrinsic geometry** from pure adjacency structure

**Mechanistic Analysis**:
- Do learned embeddings align with eigenvectors? (Procrustes analysis)
- Fourier decomposition of learned features
- Attention: do nodes with similar eigenvector values attend to each other?
- Topology transfer: train on cycle, test on grid → does it generalize?

**Visualization Deliverables**:
- **Eigenfunction atlases**: Side-by-side comparison of learned vs true eigenfunctions on each topology
- **Spectrum alignment plot**: Learned eigenvalue estimates vs true values
- **Attention eigenmaps**: Which eigenvectors dominate attention patterns?
- **Cross-topology transfer matrix**: How well features transfer between graph types

**Key Experiments**:
- Ablation: Does positional encoding help or hurt?
- Comparison: SINPE vs learned embeddings vs one-hot
- Generalization: Train on small graphs, test on larger ones

**Difficulty**: Medium
**Novelty**: Very High (bridges ML and spectral graph theory)
**Visual Appeal**: Very High (eigenfunction patterns are beautiful)

---

## 12. Manifold Learning in High Dimensions — Sphere, Projective Plane, Klein Bottle

**Concept**: Train models to compute geometric properties on various 2D manifolds and watch them discover the appropriate harmonic basis.

**Concrete Tasks**:
- **Sphere S²**: Given point (θ, φ), predict geodesic distance to another point
- **Projective plane RP²**: Classify antipodal points as equivalent
- **Klein bottle**: Periodic identification with twist (non-orientable)
- **Genus-g surfaces**: Generalize to surfaces with multiple holes

**Dataset Generation**:
- Sample points uniformly on each manifold (careful with parameterization singularities!)
- Tasks:
  - Distance/similarity prediction
  - Manifold-specific operations (e.g., antipodal identification for RP²)
  - Classification of local curvature (positive, zero, negative)

**Why it's exciting**:
- Different manifolds have **different harmonic bases**:
  - Sphere: spherical harmonics Yₗᵐ
  - Torus: 2D Fourier sin(kx)sin(ly)
  - Projective plane: subset of spherical harmonics
- Tests if models discover **intrinsic curvature** from extrinsic coordinates
- Klein bottle would test handling of **non-orientable** structure (does the model detect the "twist"?)

**Mechanistic Analysis**:
- Extract learned representations for each point on manifold
- Compare to known harmonic bases (spherical harmonics, etc.)
- Attention patterns: do they respect manifold geometry (short geodesic distances)?
- Test for "chart awareness": does model understand coordinate singularities?

**Visualization Deliverables**:
- **Manifold eigenfunction galleries**: Learned vs theoretical harmonic bases on each surface
- **Embedding maps**: PCA/UMAP of learned embeddings colored by manifold position
- **Attention geodesic plots**: Distance between attending points vs geodesic distance
- **Curvature sensitivity**: Which neurons respond to local curvature?

**Difficulty**: Hard (manifold sampling and math)
**Novelty**: Very High (first mechanistic study of manifold learning)
**Visual Appeal**: Very High (spherical harmonics are gorgeous)

---

## 13. Harmonic Analysis on Symmetric Spaces — Lie Groups & Homogeneous Spaces

**Concept**: Explore whether transformers can discover the representation theory of Lie groups and homogeneous spaces.

**Concrete Tasks**:
- **SO(3) rotations**: Predict composition of rotations or function of rotation matrix
- **SU(2) operations**: Quaternion multiplication (group structure)
- **Hyperbolic plane H²**: Distance computation or geodesic prediction
- **Special functions**: Predict values of Legendre polynomials, Bessel functions, etc.

**Dataset Generation**:
- Sample group elements (rotation matrices, quaternions, etc.)
- Compute group operation results or special function values
- Optionally include continuous parameters (e.g., rotation angle)

**Why it's exciting**:
- Directly connects to **harmonic analysis on symmetric spaces** — deep mathematical theory
- Different groups have **different irreducible representations**:
  - SO(3): spherical harmonics, Wigner D-matrices
  - SU(2): spin representations
  - Hyperbolic: different spectral theory entirely
- Tests if models discover **group-theoretic structure** from operation tables
- Could reveal mechanistic basis for "geometric deep learning"

**Mechanistic Analysis**:
- Do learned representations align with irreducible representations?
- Fourier analysis on the group (Peter-Weyl theorem)
- Attention symmetry: do attention patterns respect group invariance?
- Subgroup detection: can we find circuits for specific group operations?

**Visualization Deliverables**:
- **Representation spectrum**: Which irreps are learned?
- **Group orbit plots**: Embedding geometry along group orbits
- **Attention symmetry analysis**: Invariance under group actions
- **Cross-group comparison**: How do representations differ between groups?

**Difficulty**: Hard (requires Lie theory knowledge)
**Novelty**: Extremely High (unexplored territory)
**Visual Appeal**: High (abstract but mathematically beautiful)

---

## 14. Multi-Scale Fourier Hierarchy — Wavelets and Renormalization

**Concept**: Train models on tasks with structure at **multiple scales** and watch them discover wavelet-like multi-resolution representations.

**Concrete Tasks**:
- **Multi-frequency signal reconstruction**: Input sum of sinusoids at different frequencies → reconstruct
- **Cascaded modular arithmetic**: (((x + a₁) mod p₁) + a₂) mod p₂ with p₁ ≫ p₂
- **Fractal patterns**: Predict properties of self-similar structures
- **Renormalization group**: Coarse-grain a lattice and predict emergent properties

**Dataset Design**:
- Explicitly include **scale separation** (e.g., low-frequency global + high-frequency local)
- For modular version: use moduli at very different scales (e.g., mod 1000 vs mod 10)

**Why it's exciting**:
- Natural Fourier basis is **not optimal** for multi-scale — wavelets are
- Would reveal if models discover **scale-selective neurons** (different neurons for different scales)
- Connection to renormalization in physics (coarse-graining)
- Tests if learned representations are **adaptively multi-resolution**

**Mechanistic Analysis**:
- Decompose learned features by scale (bandpass filtering)
- Look for "mother wavelet" patterns in receptive fields
- Attention patterns: do different heads focus on different scales?
- Scale hierarchy: do deeper layers see coarser scales?

**Visualization Deliverables**:
- **Scale decomposition**: Separate features by frequency band
- **Wavelet comparison**: Learned features vs Daubechies/Morlet wavelets
- **Scale-attention matrix**: Which heads process which scales?
- **Renormalization flow**: How representations change across layers

**Difficulty**: Medium-Hard
**Novelty**: High (multi-scale mechanistic work is rare)
**Visual Appeal**: High (wavelet plots and scale hierarchies)

---

## 15. Geometric Deep Learning Benchmark Suite — Curated Atlas

**Concept**: Instead of one experiment, build a **systematic benchmark** of geometric/topological tasks with unified analysis pipeline.

**Task Categories**:
1. **Periodic 1D**: mod-p, digit-sum, parity (baseline)
2. **Periodic 2D**: torus tasks, grid convolution
3. **Topological**: winding number, linking number, homotopy
4. **Manifold**: sphere distance, hyperbolic geodesics
5. **Group-theoretic**: permutation composition, quaternion multiplication
6. **Multi-scale**: cascaded modular arithmetic, wavelet reconstruction

**Unified Pipeline**:
- Same model architecture across tasks (fair comparison)
- Standardized training procedure
- **Compendium generator** produces unified analysis reports:
  - Fourier spectrum plots
  - Periodic neuron identification
  - Attention pattern analysis
  - Embedding geometry visualization
  - Generalization metrics

**Deliverable**: A web dashboard or unified report showing:
- Task → model performance → mechanistic signature
- Clustering of tasks by learned representation similarity
- "Feature fingerprint" for each task
- Transfer learning matrix (which tasks share features?)

**Why it's exciting**:
- This would be a **resource for the community** (like ImageNet but for mechanistic interpretability)
- Could reveal taxonomies of algorithmic reasoning
- Enables systematic study of **inductive biases**
- Highly publishable as a benchmark paper

**Difficulty**: Hard (scope is large)
**Novelty**: Extremely High (no such benchmark exists)
**Visual Appeal**: Extremely High (unified dashboard)

---

## Updated Priority Ranking (with Geometry/Topology)

| Rank | Experiment | Rationale |
|------|------------|-----------|
| 1 | **#15 Geometric Deep Learning Benchmark** | Most impactful, establishes new research area |
| 2 | **#3 Phase Transition Atlas** | Most feasible with current setup, beautiful visualizations |
| 3 | **#10 Topological Invariant Learning** | High novelty, strong visual story, intellectually deep |
| 4 | **#11 Graph Laplacian Eigenlearning** | Beautiful math, clean experiments, high publishability |
| 5 | **#9 Torus Fourier Discovery** | Natural 2D extension of current work, stunning visuals |
| 6 | **#7 Emergence Timeline** | Most visually impressive for demos |
| 7 | **#2 Symmetry Discovery** | Deep connection to representation theory |
| 8 | **#12 Manifold Learning** | Ambitious but very cool |

---

## Quick Wins (Geometry/Topology)

- **#9 Torus Fourier Discovery**: Can start with 2D grid + periodic boundary, use existing Fourier analysis
- **#10 Winding Number**: Relatively simple dataset generation, clear topological story
- **#11 Graph Laplacian**: Start with cycle graph (1D), then extend to grid/torus

## Moonshots (Week+ of Work)

- **#15 Full Benchmark Suite**: Build systematically, one task category at a time
- **#12 Manifold Learning**: Math-heavy but incredibly rewarding
- **#13 Lie Groups**: Deepest math, highest theoretical payoff

---

*Note: All geometry/topology experiments build naturally on existing Fourier analysis infrastructure. The 2D/3D extensions require modifications to visualization code but the core mechanistic tools (FFT, periodic neuron detection, attention analysis) generalize directly.*
