# Phase 3: Self-Supervised Structural Learning

Phase 1 is spectral analysis — eigendecomposition of the dependency graph for module detection, role classification, and topological anomaly detection. Phase 2 adds frozen semantic embeddings and mathematical tools to detect structural-semantic disagreement. Phase 3 trains a graph neural network on hundreds of codebases to learn structural representations that are richer than spectral coordinates, calibrated across repos, and decomposed by relationship type.

**Phase 3 requires a validated Phase 2.** If Phase 2's `misplaced_concern` precision is already >80%, Phase 3 is an optimization. If 50-80%, Phase 3 is justified. If <50%, fix Phase 2 first.

---

## What Phase 2 Cannot Do

Phase 2 compares two frozen embedding spaces (spectral coordinates and CodeLM vectors) using mathematical tools. Four limitations define Phase 3's scope:

1. **No calibration per structural role.** Phase 2's local variation and centroid-distance measures treat all nodes equally. A utility function has a legitimately different structural-semantic profile than a domain entity. Phase 2 uses binary role exclusions (bridges excluded from misplaced_concern); Phase 3 learns continuous, role-aware baselines from hundreds of codebases.

2. **1-hop context only.** Local variation measures semantic disagreement with direct neighbors. Phase 3's 2-layer GIN extends this to 2-hop neighborhoods — not a dramatic leap in radius, but the key difference is that the GIN **learns** which neighborhood patterns matter, rather than using a fixed cosine-distance formula. The spectral PEs and RWPE additionally inject global and multi-scale local context into every node's input features, compensating for the shallow message-passing depth.

3. **No cross-repo transfer.** Phase 2's spectral coordinates are per-graph. Each codebase is analyzed from scratch. Phase 3's trained model has seen what "misplaced concern" and "incoherent module" look like across 500+ repos. Its embeddings live in a shared space — the same structural pattern in two different codebases produces similar embeddings.

4. **No per-layer decomposition.** Phase 2 uses the combined-layer adjacency matrix for spectral analysis. The eigenvectors encode combined coupling. Phase 3's R-GIN processes each dependency layer (calls, imports, inherits) through separate encoders, producing per-layer structural embeddings. This enables cross-layer analysis: "this node is a hub in calls but a leaf in imports" becomes a first-class finding.

---

## Architecture: R-GIN

**R-GIN** = Relation-typed Graph Isomorphism Network. GIN is the most expressive architecture within the message-passing neural network (MPNN) class — provably as powerful as the 1-Weisfeiler-Leman graph isomorphism test (Xu et al., ICLR 2019). R-GIN extends it with per-relation-type MLPs.

**Why GIN, not a Graph Transformer:** Graph transformers (GPS, Graphormer) use O(n²) attention, which is prohibitive for codebases with 50K+ nodes. Linear attention approximations lose the long-range sensitivity that justified transformers. GIN is O(n + m) in nodes and edges. When augmented with spectral positional encodings (which topo already computes), GIN exceeds 1-WL expressiveness — the spectral features provide the global context that message passing alone cannot propagate. You get beyond-1-WL power at linear cost.

**Why not deeper:** Over-smoothing (representations converge as depth increases) and over-squashing (information bottleneck through high-degree hubs) are both well-documented GNN pathologies. Code graphs, with their power-law degree distributions and bridge nodes, are maximally prone to over-squashing. 2 layers limits the message-passing radius and avoids compounding these effects. Spectral PEs handle global structure; the GNN handles local refinement.

### Model Specification

```
Layers:       2
Hidden dim:   256
Aggregation:  sum (GIN-style, provably injective over multisets)
Edge types:   3 (calls, imports, inherits) — defines excluded
Activation:   ReLU
Dropout:      0.1 (between layers)
GraphNorm:    per layer (NOT BatchNorm — code graph batches mix 100-node CLIs
              with 5000-node monorepos; batch statistics are dominated by large graphs)
Residual:     between layers (h^(l+1) += h^(l), ensures GIN only adds structural
              information, never destroys the input signal)
Grad clip:    max_norm=1.0 (prevents gradient spikes when auxiliary losses activate)
```

### Per-Layer Processing

For each dependency type `r ∈ {calls, imports, inherits}`:

```
h_r^(l+1)(v) = MLP_r^(l)( (1 + ε) · h^(l)(v) + Σ_{u ∈ N_r(v)} h^(l)(u) )
```

where:
- `h^(l)(v)` is the representation of node v at layer l
- `N_r(v)` is the set of neighbors of v in the relation-r subgraph
- `MLP_r^(l)` is a 2-layer MLP specific to relation type r and layer l
- `ε` is a learnable scalar (per GIN specification)

The per-relation outputs are **concatenated then projected** (not summed — summation loses which relation contributed what, making the second GIN layer unable to distinguish swapped per-relation roles):

```
h^(l+1)(v) = W_agg · [h_calls^(l+1)(v) ‖ h_imports^(l+1)(v) ‖ h_inherits^(l+1)(v)] + b_agg
```

where W_agg ∈ R^{256×768} projects the concatenated 3×256=768d back to 256d. This preserves per-relation identity through the layers while keeping the hidden dimension constant. Parameter cost: one 256×768 matrix per layer (~200K additional params).

A residual connection is added: `h^(l+1)(v) += h^(l)(v)`.

### Output Decomposition

The final node embedding is decomposed into layer-invariant and layer-specific components.

**Critical: per-layer components (z_calls, z_imports, z_inherits) are projected from the per-relation GIN outputs BEFORE the concat+project aggregation.** This ensures each component is derived purely from its relation type, with no information leakage from other layers. The z_invariant component is projected from the post-aggregation hidden state.

```
z_calls(v)    = W_calls · h_calls^(final)(v)       # 256d → 32d, from calls-only GIN output
z_imports(v)  = W_imports · h_imports^(final)(v)    # 256d → 32d, from imports-only GIN output
z_inherits(v) = W_inherits · h_inherits^(final)(v) # 256d → 32d, from inherits-only GIN output
z_invariant(v) = W_inv · h^(final)(v)              # 256d → 64d, from aggregated hidden state

z_str(v) = z_invariant(v) ⊕ z_calls(v) ⊕ z_imports(v) ⊕ z_inherits(v)
             64d              32d           32d             32d        = 160d total
```

- **z_invariant** (64d): Derived from the summed per-layer output. Captures structural role consistent across all dependency types.
- **z_calls** (32d): Derived from the calls-layer GIN output only. Captures runtime coupling position.
- **z_imports** (32d): Derived from the imports-layer GIN output only. Captures compile-time coupling position.
- **z_inherits** (32d): Derived from the inherits-layer GIN output only. Captures type hierarchy position.

Each component is produced by a learned linear projection from the 256d hidden state.

### Decorrelation

The invariant component should capture what is shared across layers; the layer-specific components should capture what is unique. Enforced via an HSIC (Hilbert-Schmidt Independence Criterion) penalty:

```
L_decorrelation = HSIC(z_invariant, z_calls) + HSIC(z_invariant, z_imports) + HSIC(z_invariant, z_inherits)
```

HSIC is a kernel-based dependence measure that is differentiable, statistically consistent, and does not require adversarial training or MI estimation. It is computed via the biased estimator:

```
HSIC(X, Y) = (1/n²) · tr(K_X H K_Y H)
```

where K_X, K_Y are RBF kernel matrices: K_X(i,j) = exp(-‖x_i - x_j‖² / 2σ²), and H = I - (1/n)11ᵀ is the centering matrix.

**Bandwidth selection.** Use the median heuristic: σ = median(‖x_i - x_j‖) over all pairs in the sample. Recompute per batch. This is the standard default (Gretton et al., 2012) and adapts to the embedding scale automatically.

**Computed per-graph, not per-batch.** For each graph in the mini-batch, compute HSIC over that graph's nodes (n=500-1000, kernel matrices of ~1M entries — manageable). Average across graphs in the batch. Per-batch HSIC would require kernel matrices of 16K-32K nodes, which is infeasible (~1B entries).

**Soft, not hard.** The HSIC penalty is weighted at λ=0.01 — it encourages decorrelation without forcing strict independence. This preserves cross-layer synergistic information (e.g., a node's structural role may depend on the combination of its call-graph and import-graph positions), which hard independence would destroy.

---

## Input Specification

For each node v in a code graph, the R-GIN receives:

### 1. Semantic Features (128d)

Frozen CodeLM embedding (768d from jina-embeddings-v2-base-code, same as Phase 2), projected to 128d via a learned MLP:

```
x_sem(v) = MLP_project(CodeLM(v))    # 768d → 128d
```

The MLP is trained end-to-end with the R-GIN. The CodeLM is frozen — no gradients flow through it. This keeps the semantic signal independent of the structural learning.

### 2. Spectral Positional Encoding (2k dimensions, typically 2×16 = 32d)

Laplacian eigenvectors from Phase 1, paired with their eigenvalues and processed through a sign-invariant network (SignNet, Lim et al., ICML 2022):

```
x_spectral(v) = SignNet([(u₁(v), λ₁), (u₂(v), λ₂), ..., (uₖ(v), λₖ)])
```

**SignNet** handles the sign ambiguity of eigenvectors: φ(vᵢ) = ρ(ψ(vᵢ) + ψ(-vᵢ)), where ψ is a per-eigenvector MLP and ρ aggregates across eigenvectors. The output is identical regardless of the numerical solver's sign choice. This is essential for training across multiple graphs — within a single graph, signs are consistent, but across graphs, they are arbitrary.

**Eigenvalue pairing.** Each eigenvector component is concatenated with its eigenvalue: the input to SignNet is a k×2 matrix, not a k×1 vector. The eigenvalue provides the structural scale — λ₂=0.001 (near-disconnect) and λ₂=0.5 (gradual gradient) are qualitatively different even if the eigenvector coordinates look similar. Without eigenvalues, the model cannot distinguish these cases across graphs.

**k = 16** (the first 16 non-trivial eigenvectors). This covers the community-structure frequencies. Higher eigenvectors capture noise. The output dimensionality after SignNet processing is 32d.

**Graphs with fewer than 17 nodes** (or fewer than 16 non-trivial eigenvectors due to disconnected components): zero-pad the spectral input to k=16. The zero entries carry no structural information, which the SignNet can learn to ignore.

**Limitation: eigenspace rotation.** SignNet handles sign flips (±v) but not full rotation within degenerate eigenspaces (repeated eigenvalues). For code graphs, exact eigenvalue multiplicity is rare. Near-degeneracy (λᵢ ≈ λᵢ₊₁) causes ordering instability but not representation collapse — the RWPE features (inherently rotation-invariant) provide a backup for these cases. If >10% of training graphs show eigenvalue gaps < 0.01 in the first 16 eigenvalues, upgrade to BasisNet (Lim et al., 2023) or SPE (Huang et al., NeurIPS 2023), which handle the full eigenspace ambiguity.

### 3. Random Walk Positional Encoding (16d)

```
x_rwpe(v) = [P¹(v,v), P²(v,v), ..., P¹⁶(v,v)]
```

where P = D⁻¹A is the random walk transition matrix. P^k(v,v) is the probability that a random walk starting at v returns to v in exactly k steps.

**What it captures:** Local topology at different scales. P²(v,v) correlates with the local clustering coefficient (triangles). P⁸(v,v) captures medium-range reachability. P¹⁶(v,v) captures whether the node is in a well-connected or peripheral region.

**Why both spectral and RWPE:** Spectral PEs capture global position (where in the overall graph). RWPE captures local structure (what the neighborhood looks like). They are complementary — spectral PEs are defined by the global eigenvectors and have the sign/basis ambiguity problem; RWPE is purely local, always positive, and naturally transferable across graphs.

**Computation:** P^k(v,v) is the diagonal of P^k, computable by repeated sparse matrix-vector products. For k=16, this is 16 sparse matmuls — O(km) total, negligible for code graph sizes. **Use the symmetrized adjacency** (A + Aᵀ) for RWPE computation — on directed code graphs, DAG-leaf nodes have zero return probability at all k, producing zero RWPE vectors that carry no information. Symmetrization ensures all nodes get meaningful local topology features.

### 4. Defines-Tree Encoding (16d)

The `defines` edge type (module→function, class→method) forms a containment tree. Instead of processing it with the GIN (which is designed for dependency-like edges, not hierarchical containment), encode each node's tree position as features:

```
x_tree(v) = MLP_tree([depth(v), sibling_index(v), subtree_size(v), parent_subtree_size(v)])
```

- **depth(v)**: Level in the containment hierarchy (0 = package, 1 = module, 2 = class, 3 = method).
- **sibling_index(v)**: Position among siblings (0-indexed). Captures ordering within a module.
- **subtree_size(v)**: Number of descendants. Distinguishes leaf functions from large modules.
- **parent_subtree_size(v)**: Number of siblings (children of parent). Distinguishes nodes inside a large module (many siblings) from nodes inside a small focused module (few siblings). More structurally meaningful than a path hash — hashes produce arbitrary integers with no geometric meaning that the model cannot learn from.

The MLP projects these 4 scalars to 16d. This captures "where the developer put this code" as a separate signal from the dependency structure.

### 5. Node Type Embedding (16d)

Learned lookup table:
```
x_type(v) = Embedding[kind(v)]    # kind ∈ {function, class, module, struct, trait, enum, ...}
```

16d per node type. The vocabulary is small (~10-15 kinds across Python and Rust).

### Total Input Dimensionality

```
x(v) = x_sem(v) ⊕ x_spectral(v) ⊕ x_rwpe(v) ⊕ x_tree(v) ⊕ x_type(v)
        128d         32d              16d          16d          16d       = 208d
```

This 208d vector is the input to the first R-GIN layer. The GIN's internal hidden dimension is 256d.

---

## Output Specification

### Per-Node Output

For each node v in a code graph:

| Field | Dim | Source | Description |
|-------|-----|--------|-------------|
| `z_invariant` | 64 | Summed R-GIN output → linear projection | Cross-layer structural role |
| `z_calls` | 32 | Calls-layer R-GIN output → linear projection | Runtime coupling position |
| `z_imports` | 32 | Imports-layer R-GIN output → linear projection | Compile-time coupling position |
| `z_inherits` | 32 | Inherits-layer R-GIN output → linear projection | Type hierarchy position |
| `reconstruction_error` | 1 | ‖predicted_sem - actual_sem‖ (cosine) | Structural-semantic disagreement score |

**Total per node: 161 dimensions.**

The `reconstruction_error` scalar is the key anomaly signal. After training, it measures: "how well does this node's multi-hop structural context, across all dependency layers, predict its semantic content?" High error = the structure does not explain the semantics = structural-semantic disagreement. This is the learned, calibrated version of Phase 2's `misplaced_concern` centroid-distance measure.

### Per-Graph Output

| Field | Dim | Source | Description |
|-------|-----|--------|-------------|
| `g_embedding` | 64 | Attention-weighted mean pooling of z_invariant | Global structural fingerprint |
| `g_archetype` | categorical | k-NN classifier on g_embedding | Architecture style (layered, hub-spoke, etc.) |

The graph-level embedding enables cross-repo structural comparison — two codebases with similar g_embedding have similar global topology. The archetype classifier is a simple post-hoc k-NN on the training corpus, not a learned head.

### Model Artifacts (Shipped with Model Bundle)

These trained artifacts are exposed alongside the per-node/per-graph outputs for downstream consumption by the health score and diagnostics:

| Artifact | Shape | Source | Consumed by |
|----------|-------|--------|-------------|
| `R` | 32×32 | Bilinear relation matrix from Loss 2 | Health: `direction_surprise` for layer_conformance |
| `depth_probe_w` | 768 | Linear probe weight vector | Health: semantic layer assignment |
| `depth_probe_b` | 1 | Linear probe bias | Health: semantic layer assignment |

**R (bilinear relation matrix).** The 32×32 matrix from the cross-layer edge prediction head (Loss 2). At training time it learns which import-position pairs predict call edges. At inference time the health score uses its asymmetry to compute `direction_surprise` per edge — measuring how much the model thinks each call edge goes against the expected direction. R is a model weight, not a per-node output. It is extracted from the trained model and shipped as a constant in the model bundle.

**Semantic depth probe.** A linear regression from 768d CodeLM module centroids to scalar layer position. Trained on the R-GIN corpus (see Training Procedure). Used by the health score to anchor layer assignment in semantic meaning rather than self-referential edge counts. The probe weights are small (769 floats) and ship with the model bundle.

### What Is NOT in the Output

- Raw CodeLM embeddings (too large, model-specific — consumers use Phase 2's embeddings directly if needed).
- Attention weights or message-passing internals (not interpretable enough to expose).
- Per-layer adjacency modifications (the model does not modify the graph).

---

## Training Objectives

Three losses plus one regularizer. This budget was determined by the multi-agent review — 6+ losses cause gradient competition and hyperparameter sensitivity; 3 losses with 1 lightweight regularizer is the maximum that trains reliably.

### Loss 1: Masked Semantic Feature Prediction (Primary)

**Objective:** Mask a node's CodeLM embedding, predict it from structural context.

```
L_reconstruct = (1/|M|) · Σ_{v ∈ M} (1 - cos(f_predict(v), x_sem(v)))
```

where:
- M is the set of masked nodes (60-70% of nodes per graph, sampled uniformly)
- x_sem(v) is the frozen CodeLM embedding (768d, before projection)
- f_predict(v) = MLP_decode(z_str(v)) is a 2-layer MLP: 160d → 512d (ReLU) → 768d. The expansion through a 512d hidden layer gives the decoder capacity to learn nonlinear composition of the four decorrelated subspaces (z_invariant, z_calls, z_imports, z_inherits). A single linear layer (160→768) would underfit.
- Scaled cosine error (per GraphMAE — avoids the magnitude sensitivity of MSE in high dimensions)

**What it learns:** The R-GIN must produce structural embeddings that are predictive of semantic content. This forces the model to encode architecturally relevant information — a node's position in the dependency graph must explain what it does.

**Why 60-70% masking:** Higher masking forces the model to rely on graph structure rather than "leaking" through unmasked neighbor features. At 50%, too much signal comes from neighbor CodeLM embeddings propagated through message passing. At 70%, the model must use topological position (which neighbors are connected, through which edge types, at what distance) rather than semantic smoothness. The optimal ratio is calibrated during training — start at 60%, increase to 70% if training loss plateaus.

**Why this IS the disagreement signal:** After training, the reconstruction error for an unmasked node (run the full model, compare predicted vs. actual semantics) measures how well the structural neighborhood predicts the semantic content. Nodes where structure predicts semantics well → low error → structurally consistent. Nodes where structure cannot predict semantics → high error → structurally-semantically misaligned. This subsumes Phase 2's centroid-distance heuristic with a learned, calibrated, multi-hop alternative.

### Loss 2: Asymmetric Cross-Layer Edge Prediction (Secondary)

**Objective:** Using only import-layer structural position, predict which call edges exist. This tests whether compile-time coupling (imports) explains runtime coupling (calls).

```
L_crosslayer = -Σ_{(u,v)} [ y_uv · log(σ(z_imports(u)ᵀ · R · z_imports(v)))
                           + (1 - y_uv) · log(1 - σ(z_imports(u)ᵀ · R · z_imports(v))) ]
```

where:
- `z_imports(u)` and `z_imports(v)` are the import-layer embeddings (32d), taken from the imports-only GIN branch **before** the concat+project aggregation (see Output Decomposition). This ensures the prediction uses only import-layer structural information, with no call-graph leakage.
- y_uv = 1 if a call edge exists between u and v, 0 otherwise
- R is a learned bilinear relation matrix (32×32)
- σ is the sigmoid function
- Negative edges are sampled uniformly (5:1 negative-to-positive ratio)

**Note:** Both sides of the bilinear form use z_imports (same space). This is intentional — the question is "can import-layer position alone predict call edges?" not "can one layer's embedding predict another layer's." The R matrix learns which pairs of import-positions tend to co-occur with call edges.

**What it learns:** Which import paths become actual runtime couplings. Imports define the possibility space of calls (you can only call what you can see). But only a fraction of possible calls are actual. The model learns which import-to-call patterns reflect real architectural coupling vs. incidental visibility.

**Why asymmetric (imports → calls):** The conditional entropy H(calls | imports) < H(imports | calls) for static-import languages. Import structure constrains call structure, not the reverse. This directional prediction is more informative than symmetric contrastive alignment.

**Caveats:** The asymmetry holds for Python, Rust, Java (static imports). It breaks for dependency injection, dynamic dispatch, and reflection. The model will learn these edge cases from the training data — repos using DI frameworks will have lower cross-layer predictability, which the model absorbs as higher reconstruction error for DI-heavy patterns.

### Loss 3: Graph-Level Contrastive (Tertiary)

**Objective:** Subgraph samples from the same repo should have similar graph-level embeddings; samples from different repos should differ.

```
L_graph = -log( exp(sim(g_i, g_i') / τ) / Σ_j exp(sim(g_i, g_j) / τ) )
```

where:
- g_i, g_i' are graph-level embeddings of two subgraph samples from the same repo
- g_j are graph-level embeddings from other repos in the batch
- sim is cosine similarity
- τ = 0.07 (temperature)

**Subgraph sampling:** For each repo, sample two overlapping subgraphs (random BFS from different starting nodes, 60-80% of nodes each). The overlap ensures the two views share structural information but are not identical.

**What it learns:** Global architectural invariants. Two subgraph views of the same codebase should map to similar graph-level fingerprints because they share the same global architecture. Different codebases should map to different fingerprints. This prevents the node-level losses from producing embeddings that are locally good but globally incoherent (the negative transfer problem documented by Hu et al., ICLR 2020).

**Why this is needed:** Hu et al. showed that node-level pre-training alone causes negative transfer on graph-level downstream tasks. Combining node-level (L_reconstruct, L_crosslayer) and graph-level (L_graph) objectives is essential for balanced representations.

### Regularizer: HSIC Decorrelation

```
L_decorrelation = HSIC(z_invariant, z_calls) + HSIC(z_invariant, z_imports) + HSIC(z_invariant, z_inherits)
```

Weighted at λ=0.01. See Architecture section for details.

### Combined Loss

```
L_total = L_reconstruct + 0.5 · L_crosslayer + 0.2 · L_graph + 0.01 · L_decorrelation
```

The weights reflect priority: reconstruction is the primary signal (directly produces the anomaly score), cross-layer prediction is secondary (structural understanding), graph-level contrastive is tertiary (global coherence), decorrelation is a lightweight tether.

---

## Dataset

### Curation Criteria

**500-2,000 repositories** from GitHub, curated with strict filters:

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| Stars | ≥100 | Filters homework, dead projects, forks |
| Contributors OR active years | ≥50 contributors OR ≥3 years active | Filters personal projects |
| CI/CD | Present | Proxy for code quality |
| Primary language | Python OR Rust | Matches topo's parsers |
| Forks | Excluded | Deduplication |
| License | OSI-approved | Legal clarity |

**Stratification by architectural style** to ensure diversity:

| Style | Target % | Examples |
|-------|----------|---------|
| Web application (Django, Flask, Actix) | 25% | Layered, HTTP-centric |
| Library/framework | 25% | Public API + internal implementation |
| CLI tool | 15% | Flat, entry-point-centric |
| Data pipeline / ML | 15% | DAG-like, transformation chains |
| Systems / infrastructure | 10% | Low-level, performance-centric |
| Monorepo / multi-package | 10% | Cross-package coupling |

### Preprocessing

For each repository:

1. **Parse** with topo's own parsers (Rust or Python). This ensures training data matches inference-time data distribution exactly. Systematic parser biases become consistent rather than random.
2. **Run Phase 1 analysis** to produce: spectral eigenvectors, eigenvalues, module assignments, role classifications.
3. **Run Phase 2 embedding** to produce: CodeLM embeddings per node (768d).
4. **Compute RWPE** (P^k diagonal for k=1..16).
5. **Extract defines tree** from the parsed graph (containment edges).
6. **Store as a preprocessed graph** in a standardized format (PyTorch Geometric `HeteroData` or equivalent):

```python
{
    "node_features": {
        "semantic": Tensor[n, 768],      # CodeLM embeddings
        "spectral": Tensor[n, 16, 2],    # eigenvectors + eigenvalues
        "rwpe": Tensor[n, 16],           # random walk PE
        "tree": Tensor[n, 4],            # depth, sibling, subtree_size, path_hash
        "node_type": Tensor[n],          # categorical index
    },
    "edge_index": {
        "calls": Tensor[2, m_calls],
        "imports": Tensor[2, m_imports],
        "inherits": Tensor[2, m_inherits],
    },
    "metadata": {
        "repo": "owner/repo",
        "language": "python",
        "n_nodes": 847,
        "n_edges": 3241,
        "spectral_k": 10,
        "modularity_q": 0.74,
    }
}
```

### Scale Analysis

| Metric | Per repo (median) | Total (1000 repos) |
|--------|-------------------|--------------------|
| Nodes | 500-1000 | 500K-1M |
| Edges | 2000-5000 | 2M-5M |
| CodeLM embeddings | 500-1000 × 768d | ~3-6 GB |
| Spectral PEs | 500-1000 × 32d | ~60-120 MB |
| RWPE | 500-1000 × 16d | ~30-60 MB |

The dataset fits in memory on a single machine. No distributed training infrastructure needed.

**Node count is the relevant measure** for sample complexity, not graph count. The model learns node-level representations from 500K-1M nodes — comparable to standard node-level benchmarks (OGB-Products has 2.4M nodes). The 500-2000 graph count matters for distributional coverage (diversity of architectural styles), not for sample size.

### Data Quality Risks

- **Parser errors** introduce systematic noise — missed edges, phantom nodes. Mitigated by parsing with topo's own parsers (noise at training = noise at inference).
- **Naming conventions** vary across codebases. CodeLM embeddings of generic names (`handle`, `process`, `run`) have low discriminative power. Mitigated by the 8K context window (full body embedding, not just names). Still a weakness for codebases with very generic naming.
- **Language-specific patterns.** Python imports are semantically different from Rust `use` statements. The cross-layer prediction (imports → calls) learns language-specific patterns, not pure architectural patterns. Mitigated by stratification — the model sees many examples of each language.

---

## Training Procedure

### Framework

PyTorch + PyTorch Geometric. The R-GIN is implemented in Python for training (leveraging PyG's heterogeneous graph support and GPU acceleration). After training, the model is exported to ONNX for inference in Rust via `ort` (same ONNX Runtime used for CodeLM embeddings in Phase 2).

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning rate | 1e-3 | Standard for GIN with Adam |
| LR schedule | Cosine decay to 1e-5 over 200 epochs | Smooth convergence |
| Optimizer | AdamW (weight decay 1e-4) | Standard for GNN pre-training |
| Batch size | 32 graphs | Fits in GPU memory for median graph sizes |
| Epochs | 200 | Sufficient for convergence at this data scale |
| Masking ratio | 0.65 | Start at 0.60, increase to 0.70 if loss plateaus |
| Negative sampling ratio (L_crosslayer) | 5:1 | Standard for link prediction |
| Temperature τ (L_graph) | 0.07 | Standard for contrastive learning |
| Loss weights | L_reconstruct=1.0, L_crosslayer=0.5, L_graph=0.2, L_decorrelation=0.01 | Reconstruction dominates |
| Dropout | 0.1 | Between GIN layers |
| SignNet hidden dim | 64 | Per-eigenvector MLP |

### Training Schedule

**Linear ramp-up, not hard phase transition.** All losses are active from epoch 1, but auxiliary loss weights ramp linearly from 0 to their target over the first 30 epochs:

- Epochs 1-10: LR warm-up (0 → 1e-3 linearly). Only L_reconstruct at full weight. Auxiliary weights = 0.
- Epochs 10-30: Auxiliary weights ramp linearly to target (L_crosslayer: 0 → 0.5, L_graph: 0 → 0.2, L_decorrelation: 0 → 0.01).
- Epochs 30-300: Full training. All losses at target weights. Cosine LR decay from 1e-3 to 1e-5.
- **Early stopping** on held-out validation codebases. If masked reconstruction cosine similarity on validation set has not improved for 30 epochs, stop.

This avoids the hard phase transition problem where the model overfits to reconstruction in Phase A and then struggles to adapt when auxiliary losses activate abruptly.

### Post-Training: Semantic Depth Probe

After R-GIN training completes, fit a linear probe that maps module semantic content to layer position. This is a lightweight post-hoc step, not part of the R-GIN training loop.

**Input:** For each training repo that passed quality filters (cycle_freedom > 0.95 to ensure clean DAG structure), compute per-module CodeLM centroid (768d average of member node embeddings).

**Target:** The module's normalized layer position from edge-majority inference: `y_M = layer(M) / max_layer`, giving a value in [0, 1] where 0 = bottom of the stack and 1 = top.

**Method:** Ordinary least squares across all modules in the filtered training corpus:

```
depth_sem(M) = w^T · centroid_sem(M) + b
minimize Σ (depth_sem(M) - y_M)²
```

This is a single matrix solve (~50 lines of Python). The result is a 768d weight vector `w` and scalar bias `b` that map any module's semantic centroid to a predicted layer position. The probe learns patterns like "database/SQL vocabulary → low layer" and "HTTP/handler vocabulary → high layer" from hundreds of repos.

**Why post-training, not joint:** The probe uses frozen CodeLM embeddings, not R-GIN outputs. It doesn't need gradients from the R-GIN. Fitting it after training avoids adding another objective to the already-tight loss budget and keeps the R-GIN's training stable.

**Output:** `depth_probe_w` (768 floats) and `depth_probe_b` (1 float), shipped with the model bundle.

### Hardware

- **Training:** Single GPU (A100 or equivalent). ~4-8 hours for 1000 graphs × 300 epochs. The bottleneck is not compute but data preprocessing (parsing 1000 repos, computing CodeLM embeddings — see Dataset section).
- **Inference:** CPU-only via ONNX Runtime or native Rust. ~10ms per graph for the GIN forward pass (after preprocessing). The CodeLM embedding (~5ms/node batched) dominates inference time, not the GIN.

### Model Size

```
GIN layers:       2 layers × 3 relations × 2-layer MLP (256→256→256, +bias)  ≈ 790K params
Concat+project:   2 layers × (768→256 W_agg + bias)                          ≈ 395K params
SignNet:           16 eigenvectors × 2-layer MLP (2→64→32)                    ≈  35K params
Projection heads:  256→64 (W_inv) + 3×(256→32) (W_calls, W_imports, W_inherits) ≈  41K params
Decode head:       2-layer MLP (160→512→768)                                  ≈ 477K params
Input projections: 768→128 (MLP_project) + 4→16 (MLP_tree) + ~15 type embeddings ≈ 100K params
──────────────────────────────────────────────────────────────────────────────────────────────
Total:                                                                         ≈ 1.8M params
```

This is still a small model. At 1.8M parameters with 500K-1M training nodes, the data-to-parameter ratio is ~400:1 — comfortably outside the overfitting regime given that self-supervised masking generates O(n) training examples per epoch with different masks. Standard regularization (dropout=0.1, weight decay=1e-4, gradient clipping=1.0) is sufficient.

---

## Evaluation

### Intrinsic Evaluation (During Training)

| Metric | Target | Measures |
|--------|--------|----------|
| Masked reconstruction cosine similarity | >0.6 | How well structure predicts semantics |
| Cross-layer AUC (imports → calls) | >0.75 | How well import structure predicts call structure |
| Graph-level contrastive accuracy | >0.8 | Whether same-repo subgraphs are more similar than cross-repo |
| R asymmetry ratio `‖R - R^T‖_F / ‖R‖_F` | >0.1 | Whether R has learned meaningful directionality |

**R asymmetry check.** The bilinear matrix R from Loss 2 is used by the health score to compute `direction_surprise` per edge — measuring whether each call edge goes against the expected direction. This signal exists only if R is asymmetric (if R ≈ R^T, then `σ(z^T R z')` ≈ `σ(z'^T R z)` and direction_surprise ≈ 0 for all edges). After training, compute `‖R - R^T‖_F / ‖R‖_F`. If < 0.1, the bilinear form learned proximity but not directionality — the health score falls back to binary violation counting. If ≥ 0.1, the direction_surprise signal is usable. This check is reported in the training log and stored with the model bundle metadata.

### Extrinsic Evaluation (Against Phase 2 Baseline)

The R-GIN must demonstrate improvement over Phase 2's mathematical tools on the same validation codebases:

| Task | Phase 2 Metric | Phase 3 Metric | How Compared |
|------|---------------|----------------|--------------|
| Anomaly detection | `misplaced_concern` precision | reconstruction_error precision | Same labeled set of 20-30 known misplacements per codebase |
| Module detection | Spectral clustering NMI vs packages | z_invariant clustering NMI vs packages | Same codebases, same package ground truth |
| Cross-layer analysis | N/A (Phase 2 can't do this) | Per-layer role consistency | New metric: does z_calls role agree with z_imports role? |
| Cross-repo comparison | N/A (Phase 2 can't do this) | k-NN accuracy on architecture style | Classify repos by architectural style from g_embedding |

**If Phase 3 does not improve anomaly detection precision by at least 10 percentage points over Phase 2, do not deploy it.** The training cost is only justified by measurable downstream improvement.

### Validation Codebases

**15-20 held-out repos** spanning all architectural styles, **excluded from training**. The Phase 2 validation set (topo, Flask, Click, one messy project) is a subset. The expanded set adds:

- 3-4 web applications (Django, Flask, Actix)
- 3-4 libraries/frameworks of varying size
- 2-3 CLI tools
- 2-3 data/ML pipelines
- 1-2 monorepos

**Labeled anomaly dataset:** At least 100 labeled anomalies total across the held-out repos (manually classified as "correctly placed" or "misplaced"). This is the minimum for bootstrap confidence intervals on precision to be meaningful. The same labeled set is used for Phase 2 comparison.

The model never sees these repos during pre-training. This ensures the evaluation measures generalization, not memorization.

---

## Downstream Tasks

### 1. Calibrated Anomaly Detection

**Replaces:** Phase 2's centroid-distance + significance-threshold heuristic.

**Method:** After training, compute reconstruction_error for every node in a target codebase. Nodes with error in the top 5% (or above a learned threshold from the training distribution) are flagged as structural-semantic anomalies.

**Improvement over Phase 2:**
- Multi-hop context (the GIN sees 2-hop neighborhoods, not just centroids).
- Cross-layer features (disagreement between call-position and import-position contributes to error).
- Calibration from training corpus (the model has learned what "normal" reconstruction error looks like for hubs, bridges, utilities, and leaf functions — so a high error on a hub means something different than a high error on a leaf).

**The reconstruction error distribution is role-aware by construction.** During training, the model sees many hubs, many bridges, many utilities across 500+ repos. It learns that hubs have higher baseline reconstruction error (they connect diverse semantic neighborhoods). The error it produces at inference is already calibrated against this baseline — a hub with "unusually high" error relative to other hubs the model has seen is a genuine anomaly, not a statistical artifact.

### 2. Per-Layer Structural Analysis

**New capability:** Phase 2 cannot do this.

**Method:** Cluster z_calls, z_imports, and z_inherits independently. Compare the three partitions.

- **Consistent partitions** → the architecture is clean. The same modules appear in all layers.
- **Inconsistent partitions** → structural tension. "These nodes form a tight group in the call graph but are scattered in the import graph" → the calling patterns don't match the declared dependency structure → likely accidental coupling or missing interface.

**Concrete output:**
```
Module "payment" (from z_invariant clustering):
  Call cohesion:    0.82  — members call each other frequently
  Import cohesion:  0.45  — members import from many different places
  → Diagnosis: runtime coupling is tight but compile-time dependencies are scattered.
    Consider consolidating imports behind a facade.
```

### 3. Cross-Repository Structural Comparison

**New capability:** Phase 2 cannot do this.

**Method:** Compute g_embedding (64d) for every repo in the training corpus and for new repos at inference time. Use cosine similarity or k-NN for:

- **Architecture style classification.** Cluster g_embeddings of training repos. Label clusters by dominant style (layered, hub-spoke, monolith, etc.). Classify new repos by nearest cluster.
- **Structural drift tracking.** Compute g_embedding at multiple git snapshots. Plot the trajectory. Large movements indicate architectural change; stability indicates maintenance-only changes.
- **Similar-repo lookup.** Given a codebase with a structural problem, find the training repos with the most similar g_embedding. If any of them fixed a similar problem (detectable via git history), surface the precedent.

### 4. Enhanced LLM Context Narrative

**Replaces:** Phase 2's `--format=context` with richer structural descriptions.

The per-layer embeddings enable typed coupling descriptions:

```
### Module: payment (38 nodes)
  Call-graph role: leaf subsystem (low z_calls centrality)
  Import-graph role: moderately coupled (medium z_imports centrality)
  Type-hierarchy role: independent (no z_inherits connections)
  Cross-layer tension: none — consistent leaf role across layers

  Anomaly: AuthTokenValidator (reconstruction_error: 0.91)
    Multi-hop context: 2-hop call neighborhood is billing-semantic,
    but 2-hop import neighborhood includes auth module entities.
    The model's predicted semantics: billing/validation.
    Actual semantics: authentication/JWT.
    → Misplaced concern. Severity: high.
```

This is strictly richer than Phase 2's narrative — it adds per-layer role typing and multi-hop contextual evidence. The improvement is incremental, not transformative. Phase 2's narrative is already useful; Phase 3 makes it more precise.

### 5. Domain Model Refinement

**Enhances:** Phase 2's `--format=domain` output.

z_invariant clustering may produce better bounded context candidates than Phase 2's spectral clustering, because:
- It operates in a learned space calibrated across hundreds of codebases.
- It incorporates both structural and semantic information (the reconstruction objective forces structural embeddings to be semantically informed).
- It is decomposed from per-layer components, so bounded contexts are consistent across coupling types.

**Aggregate root identification improves** from per-layer analysis: an aggregate root should be a hub in z_calls (everything in the aggregate calls through it) but NOT a hub in z_inherits (it's a coordinator, not a base class). Phase 3's decomposed embeddings make this a direct query; Phase 2 can only approximate it.

---

## What Was Dropped and Why

### WL-Hash Contrastive Learning
Proposed as a cross-repo structural pattern matching objective. **Mathematically tautological for GIN.** Xu et al. (2019) proved GIN is exactly as expressive as 1-WL. Two nodes with the same WL hash produce the same GIN embedding by construction, regardless of learned weights. The contrastive loss between WL-identical nodes provides zero gradient. The model learns nothing from this objective.

### Interferometer / Gated Fusion
Proposed as a dual-path architecture with bimodal gates to detect structural-semantic disagreement. The reconstruction error from masked feature prediction captures the same signal more directly — it IS the disagreement measurement, not a proxy. The interferometer adds complexity (bimodal gate loss, gate collapse risk, hyperparameter sensitivity) without adding signal.

### Learned Margins
Proposed as per-node adaptive thresholds for structural-semantic disagreement. Sound in principle but has a collapse mode (margin → ∞, loss → 0). Requires curriculum training (fixed margin → learned margin) and careful regularization. The reconstruction error already provides a continuous, calibrated disagreement score without explicit margins. If fine-grained threshold calibration is needed post-training, a simple MLP on (reconstruction_error, degree, role) can be trained with minimal supervision.

### Learnable Graph Refinement
Proposed as a layer that adds/removes/reweights edges before GIN processing. Too dangerous without supervision — the refinement layer can learn to create degenerate graphs that make SSL losses trivially easy. Also makes the model uninterpretable (you can't inspect what it "sees" because it rewrites its own input). If graph refinement is valuable, do it as post-hoc fine-tuning on downstream tasks with supervision, not during pre-training.

### Hyperbolic Geometry
Proposed for the z_invariant component (code graphs are approximately hierarchical). Code graphs are approximately but not purely hierarchical — cycles, cross-cutting concerns, and utilities violate the tree assumption. The 5-15% improvement in tree-embedding fidelity does not justify the implementation complexity (Riemannian SGD, numerical instability near Poincaré ball boundary, hyperbolic distance functions everywhere downstream).

### Curriculum Contrastive Learning
Proposed as random → hard → adversarial negative sampling progression. Marginal value at this data scale (500-2000 graphs). The training runs through all data many times regardless. The additional hyperparameters (when to transition, how to define "hard") are not worth the debugging cost.

### Structural Edit Prediction (Git Diffs)
Proposed as predicting edge changes between commits. Powerful signal but massive data engineering cost (parse at every commit, handle extreme class imbalance at 99.9%+ unchanged edges). Deferred to Phase 4. The architecture is designed to accommodate it later — the R-GIN can accept temporal features as additional node input.

---

## Implementation Plan

### Step 1: Dataset Curation Pipeline (~2 weeks)
- GitHub API crawler with quality filters (stars, contributors, CI/CD, license).
- Language detection and deduplication.
- Architectural style labeling (manual for initial 200 repos, heuristic for the rest).
- Store: repo URL, language, style label, metadata.

### Step 2: Preprocessing Pipeline + Debugging (~3 weeks)
This is the most underestimated step. Real-world repos will have parser failures, encoding issues, and edge cases. Budget accordingly.
- Repo download pipeline with retry/resume (repos average 100MB+, network is flaky).
- Batch-parse all repos with topo. **Handle parser failures gracefully** — skip unparseable files, log errors, require >80% parse coverage to include a repo.
- Run Phase 1 analysis (spectral decomposition, module detection).
- Run Phase 2 CodeLM embedding (batch fastembed inference on GPU). For 500K-1M nodes at ~5ms/node batched: 40-80 minutes on GPU. On CPU: 7-14 hours. Use checkpointing — embedding one repo at a time with intermediate saves.
- Compute RWPE (sparse matrix powers, P^k diagonal for k=1..16). Specify: **treat the graph as undirected** (symmetrize A) for RWPE computation. On directed code graphs, many nodes are DAG-leaves with zero return probability at all k, which produces zero RWPE vectors. Symmetrization gives meaningful return probabilities for all nodes.
- Extract defines trees.
- Convert to standardized PyTorch Geometric `HeteroData` format.
- Validate preprocessed data: assert non-null features, assert edge indices in range, assert eigenvector dimensions consistent.
- Store preprocessed graphs (~10-20 GB total).

**Total preprocessing wall time: ~3-5 days** (dominated by repo download + CodeLM embedding). Must be debugged — parser failures on real-world repos are frequent.

### Step 3: R-GIN Implementation (~2 weeks)
- R-GIN model class (per-relation MLP, concat+project aggregation, residual connections, GraphNorm, output decomposition with pre-sum per-layer projections).
- SignNet implementation (per-eigenvector MLP, sign-invariant aggregation).
- HSIC kernel computation (per-graph, median bandwidth heuristic).
- Masked reconstruction head (2-layer MLP: 160→512→768, scaled cosine loss).
- Cross-layer prediction head (bilinear on pre-sum z_imports + sigmoid + BCE).
- Graph-level contrastive (subgraph BFS sampling + InfoNCE).
- Training loop with linear ramp-up schedule, gradient clipping, early stopping.

### Step 4: Training + Evaluation + Health Artifacts (~2 weeks)
- Train on curated dataset (up to 300 epochs with early stopping, ~4-8 hours on single GPU).
- Evaluate against Phase 2 baseline on 15-20 held-out codebases with 100+ labeled anomalies.
- Ablation: remove each loss component, measure degradation.
- Hyperparameter sweep on masking ratio (0.60-0.75), loss weights, hidden dimensions.
- **R asymmetry check:** Compute `‖R - R^T‖_F / ‖R‖_F`. Log result. If < 0.1, flag that `direction_surprise` will not be available for the health score.
- **Fit semantic depth probe:** OLS regression from module centroids to layer positions across training repos with cycle_freedom > 0.95. Output: `depth_probe_w` (768d), `depth_probe_b` (scalar).
- **Bundle model artifacts:** R-GIN weights + R matrix (32×32) + depth probe weights (769 floats) + R asymmetry ratio + training metadata.
- If Phase 3 does not improve anomaly precision by ≥10 percentage points over Phase 2: stop. Investigate why.

### Step 5: Inference Integration (~3 weeks)

**ONNX export is the highest-risk step.** PyG's MessagePassing uses dynamic scatter/gather operations that ONNX's static graph representation handles poorly. SignNet's sign-symmetrization involves control flow. Three options, in order of recommendation:

**(a) Hand-implement inference in Rust (recommended).** The model is only ~1.8M params with a simple architecture: linear layers + ReLU + scatter_add. Implement the forward pass using `ndarray` or `burn`. No ONNX, no external runtime dependency beyond what Phase 2 already has. This aligns with topo's WASM portability goal and eliminates the ONNX translation layer entirely. Estimated: ~1500 lines of Rust for the full forward pass (R-GIN + SignNet + projections + decode head).

**(b) Export per-relation GINs as separate ONNX models.** Rewrite the forward pass as explicit sparse matrix multiplications (SpMM) without PyG abstractions. Export 3 ONNX submodels (one per relation type) + SignNet + decode head. Orchestrate in Rust. The concat+project aggregation and output decomposition happen in Rust, not ONNX. Estimated: ~2 weeks of ONNX wrangling + ~500 lines of Rust orchestration.

**(c) Use `tch-rs` (libtorch bindings) with TorchScript export.** Avoid ONNX entirely. TorchScript handles dynamic shapes better than ONNX. Adds a ~300MB libtorch dependency, which may be acceptable if `ort` is already present from Phase 2.

Decision: choose (a) unless the model architecture grows substantially beyond ~2M params, at which point (c) becomes more maintainable.

- Wire `--model <path>` CLI flag to load trained model weights.
- Implement feature computation in Rust: RWPE (sparse matrix powers), defines-tree encoding, SignNet forward pass, R-GIN forward pass, decode head.
- Benchmark inference latency. Target: <100ms per graph (excluding CodeLM embedding time).

### Step 6: Downstream Integration (~2 weeks)
- Replace Phase 2's centroid-distance heuristic with reconstruction_error anomaly scoring.
- Add per-layer analysis (z_calls/z_imports/z_inherits clustering + comparison).
- Compute g_embedding for cross-repo comparison.
- Update --format=context with per-layer role descriptions.
- Update --format=domain with improved clustering.
- **Health score (THS).** Implement the Topo Health Score using R-GIN outputs:
  - Coherence: `1 - median(reconstruction_error)` from R-GIN per-node output.
  - Flow/cycle_freedom: Tarjan's SCCs (already implemented).
  - Flow/layer_conformance: semantic depth probe for layer assignment + `direction_surprise` from R matrix for violation weighting.
  - THS = coherence^α × flow^(1-α). See [HEALTH.md](capabilities/HEALTH.md) for the full specification.
  - ~400 lines of Rust.

### Total Timeline: ~13 weeks

This assumes Phase 2 is validated and operational. If Phase 2 is not yet built, add 6-8 weeks for Phase 2 implementation + validation before starting Phase 3. The real calendar time is ~5-6 months from starting Phase 2 to a deployed Phase 3.

---

## Connection to Phase 4+

Phase 3 produces the foundation for three future directions:

### Phase 4a: Temporal Structural Dynamics
Train a sequence model (GRU) on R-GIN embeddings computed at multiple git snapshots. Predict structural trajectory. Detect codebases approaching structural phase transitions (Fiedler value crossing bisection threshold). Requires commit-level preprocessing pipeline.

### Phase 4b: Structural Inpainting via Graph Diffusion
Train a graph denoising diffusion model using R-GIN as the backbone. Mask a structural region, predict what "should" be there. The diff between actual and predicted structure is a concrete refactoring suggestion. Requires the R-GIN to be validated as a structural encoder first.

### Phase 4c: Inverse RL for Implicit Design Rules
Model each commit as a (state, action, next_state) tuple in an MDP. Use inverse RL to infer the developer's implicit architectural reward function. Violations of the inferred reward = anomalies specific to this codebase's conventions. Requires Phase 4a's temporal data pipeline.

These are research directions, not committed work. Phase 3's R-GIN is designed to be a reusable encoder for all three — its architecture does not need to change to support them. The investment in Phase 3 is amortized across all future phases.
