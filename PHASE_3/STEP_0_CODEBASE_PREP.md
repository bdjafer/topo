# Step 0: Codebase Preparation

Phase 3 trains a GNN in Python that consumes features computed by the Rust analyzer. Before any ML code exists, the Rust codebase must be extended to compute and export every input feature the R-GIN requires. This step is pure Rust, independently testable, and prerequisite to everything downstream.

---

## Overview

The R-GIN input specification (PHASE_3.md §Input Specification) requires 5 feature groups per node:

| Feature | Dim | Source | Status |
|---------|-----|--------|--------|
| Semantic (CodeLM) | 768 | jina-embeddings-v2-base-code via fastembed | **Exists** — passed as `semantic_embeddings` on `AnalyzerInput`. Exported as raw 768d; the 768→128d MLP projection is learned during training (Step 2). |
| Spectral PE | k×2 (k=16, so 32d) | Laplacian eigenvectors + eigenvalues | **Partially exists** — eigenvectors computed in `spectral.rs`, stored as `Vec<Vec<f64>>` indexed by local row within component. Need: eigenvalue pairing, uniform k=16 zero-padding, and per-node export. |
| RWPE | 16 | Random walk return probabilities P^k(v,v) | **Not implemented** |
| Defines-tree | 4 | depth, sibling_index, subtree_size, parent_subtree_size | **Not implemented** |
| Node type | 1 (categorical) | Learned embedding lookup at training time; integer index at export | **Not implemented** (node kinds exist as `graph.node_kinds: Vec<String>`, need canonical index) |

Plus the edge structure:

| Edge data | Format | Status |
|-----------|--------|--------|
| Per-layer adjacency (calls, imports, inherits) | COO `[2, m]` index pairs per layer | **Exists** — `graph.typed_edges: HashMap<String, Vec<(usize, usize)>>`. Needs serialization. |
| Defines edges (excluded from GIN, used for tree features) | Separate | **Exists** — in `typed_edges["defines"]`, but see §Edge Filter Hazard below. |

---

**Edge filter note:** The export command must pass `edge_kinds = None` to ensure defines edges are always present for tree features.

---

## 1. Random Walk Positional Encoding (RWPE)

### What It Computes

For each node v, compute the diagonal of P^k for k = 1, 2, ..., 16, where P = D⁻¹A is the random walk transition matrix on the **symmetrized** adjacency (as specified in PHASE_3.md line 162).

```
rwpe(v) = [P¹(v,v), P²(v,v), ..., P¹⁶(v,v)]
```

P^k(v,v) is the probability that a k-step random walk starting at v returns to v.

Use **symmetrized adjacency** (A + Aᵀ) as specified in PHASE_3.md — ensures DAG-leaf nodes get nonzero return probabilities. Matches `Graph::symmetric_adjacency()` behavior.

### Algorithm: Batch Sparse-Dense Matmul

The standard efficient method (used in PyG, OGB, GraphGPS):

```
Input:  Graph G with typed_edges (calls, imports, inherits — NOT defines)
        K = 16 (number of random walk steps)
        batch_size = 512 (tunable)
Output: rwpe[node_index] = Vec<f64> of length K

1. Build symmetrized adjacency A_sym from coupling edges:
   For each edge type in {calls, imports, inherits}:
     For each edge (u, v):
       A_sym[u][v] += 1
       A_sym[v][u] += 1

2. Compute degree vector: d[v] = Σ_u A_sym[v][u].
   Nodes with d[v] = 0: set rwpe[v] = [0.0; K] and skip.

3. Build sparse transition matrix P as CSR:
   P[v][u] = A_sym[v][u] / d[v]    (row-stochastic: each row sums to 1)

4. Process nodes in batches of size B = min(batch_size, n):
   For batch_start in (0, B, 2B, ...):
     batch_end = min(batch_start + B, n)
     actual_B = batch_end - batch_start

     Initialize X as dense n × actual_B matrix:
       X[i][j] = 1.0 if i == batch_start + j, else 0.0
       (Columns are indicator vectors for each node in the batch)

     For k = 1..K:
       X_new = sparse_matmul(P, X)   // O(m_sym × actual_B) where m_sym = nnz(A_sym)
       X = X_new

       // Extract diagonal: node (batch_start + j) wants entry X[batch_start + j][j]
       For j in 0..actual_B:
         rwpe[batch_start + j][k-1] = X[batch_start + j][j]

5. Clamp all RWPE values to [0.0, 1.0].
   (Floating-point accumulation over 16 matmuls can produce values like 1.0 + 1e-15.)
```

**Why this method:** We need diag(P^k) for all k ∈ 1..16. The batch method computes P^k applied to columns of the identity matrix. After k multiplications, column j of the result is the (batch_start+j)-th column of P^k, and reading entry (batch_start+j) gives us P^k(v,v). The extraction is `X[batch_start + j][j]`, not a generic diagonal — this is critical to implement correctly.

### Complexity

Total: O(K × m_sym × n) where m_sym ≈ 2m. For n > 20K, parallelize across batches with rayon (batches are independent).

### Sparse Matrix Implementation

Use hand-rolled CSR representation (no external sparse crate needed):
- `indptr: Vec<usize>` (row pointers, length n+1)
- `indices: Vec<usize>` (column indices)
- `data: Vec<f64>` (values = 1/degree for transition matrix)

Build CSR from `graph.typed_edges` directly (not from `graph.adj`, which has layer weights applied). The sparse-dense matmul is ~50 lines of straightforward nested loops over CSR rows.

### Implementation Location

**File:** `packages/topo-analyzer/src/rwpe.rs` (new file)

```rust
/// Compute Random Walk Positional Encoding for all nodes.
///
/// Returns rwpe[node_index] = [P^1(v,v), P^2(v,v), ..., P^K(v,v)]
/// where P is the random walk transition matrix on the symmetrized adjacency
/// built from coupling edges (calls, imports, inherits — NOT defines).
pub fn compute_rwpe(graph: &Graph, k: usize, batch_size: usize) -> Vec<Vec<f64>> {
    // 1. Build symmetrized adjacency as sparse CSR from typed_edges
    // 2. Compute degree, build P
    // 3. Batch matmul loop
    // 4. Clamp to [0, 1]
}
```

**Note:** The function reads `graph.typed_edges` directly (not `graph.adj`) because `graph.adj` has layer weights applied (graph.rs:109-114) which would distort the random walk probabilities. RWPE should use unweighted edges.

### Edge Cases

| Case | Behavior |
|------|----------|
| Self-loops | Parser doesn't emit them (graph.rs:91-92 skips `src == tgt`). If present, they'd contribute to P[v][v] directly. |
| Isolated node (d=0 after symmetrization) | All-zero RWPE. |
| Single-node graph | All-zero RWPE. |
| Disconnected components | Handled implicitly — random walks don't cross components, so return probabilities are within-component. No special handling needed. |

### Validation

1. **Cycle graph (n=6):** Analytic: P^k(v,v) on a cycle follows known recurrence. Verify to 1e-12.
2. **Star graph (n=5):** Hub (degree 4) vs leaves (degree 1). Hub has P²(v,v) = 1 (every 1-step walk goes to a leaf, every 2-step walk returns). Leaves have P²(v,v) = 1/4.
3. **Disconnected graph:** Two separate triangles. Verify RWPE of nodes in each triangle are identical and independent of the other.
4. **Directed DAG:** Build a chain a→b→c→d. After symmetrization, verify all nodes get nonzero RWPE (without symmetrization, node d would get all zeros).
5. **Python reference:** Compute RWPE via `scipy.sparse` for 3 small graphs, assert Rust output matches to 1e-10.
6. **Property assertions:** All values in [0, 1]. P^1(v,v) = 0 when no self-loops. For regular graphs, P^k(v,v) converges to 1/n.

---

## 2. Defines-Tree Feature Extraction

### What It Computes

For each node v, extract 4 features from the `defines` containment hierarchy:

| Feature | Type | Description |
|---------|------|-------------|
| `depth` | usize | Level in the tree. 0 = root (no parent in defines). |
| `sibling_index` | usize | 0-indexed position among parent's children, sorted by node ID. Roots get 0. |
| `subtree_size` | usize | Number of descendants. Leaf = 0. Module with 5 functions = 5. |
| `parent_subtree_size` | usize | Number of children of v's parent (including v itself). Roots get 0. |

### The Defines Forest

The `defines` edges form a **forest** (multiple root packages). Each edge `(parent, child)` means "parent contains child." Existing infrastructure: `Graph::defines_parent_map()` (graph.rs:212-218) builds child→parent.

**Root nodes** are nodes that appear as parents but never as children in defines edges, plus nodes with no defines edges at all.

**Nodes without any defines edges** (e.g., standalone functions in a flat file) get `(depth=0, sibling_index=0, subtree_size=0, parent_subtree_size=0)` — a degenerate but valid encoding that the MLP will learn means "uncontained leaf."

### Algorithm

```
Input:  graph.typed_edges["defines"] — list of (parent_idx, child_idx) edges
Output: tree_features[node_idx] = [depth, sibling_index, subtree_size, parent_subtree_size]

1. Build parent→children map and child→parent map from defines edges.
   children: HashMap<usize, Vec<usize>>
   parent: HashMap<usize, usize>   (reuse graph.defines_parent_map())

2. Identify roots: nodes in the parent set but NOT in the child set,
   plus nodes that appear in neither set (no defines involvement).

3. BFS from each root to compute depth:
   depth[root] = 0
   depth[child] = depth[parent] + 1

4. Sort each parent's children by node_id (alphabetical, deterministic).
   sibling_index[child] = position in sorted children list.
   sibling_index[root] = 0.

5. Post-order traversal for subtree_size:
   subtree_size[leaf] = 0   (no children)
   subtree_size[node] = Σ_child (1 + subtree_size[child])

6. parent_subtree_size[v] = len(children[parent_of_v]).
   parent_subtree_size[root] = 0.
   (This includes v itself — "how many siblings including me does v have?")
```

### Implementation Location

**File:** `packages/topo-analyzer/src/tree.rs` (new file)

```rust
/// Extract containment-tree features from defines edges.
///
/// Returns tree_features[node_index] = [depth, sibling_index, subtree_size, parent_subtree_size].
/// Nodes without defines edges get [0, 0, 0, 0].
pub fn compute_tree_features(graph: &Graph) -> Vec<[usize; 4]> { ... }
```

### Validation

1. **3-level tree:** package→module→{fn_a, fn_b, fn_c}. Verify:
   - package: depth=0, sibling_index=0, subtree_size=4, parent_subtree_size=0
   - module: depth=1, sibling_index=0, subtree_size=3, parent_subtree_size=1
   - fn_a: depth=2, sibling_index=0, subtree_size=0, parent_subtree_size=3
2. **Forest:** Two independent packages. Roots both get depth=0, parent_subtree_size=0.
3. **Orphan nodes:** Nodes with no defines edges get `[0, 0, 0, 0]`.
4. **Self-analysis:** Run on topo codebase. Most functions should be at depth 2-3.

---

## 3. Node Type Vocabulary

### What It Computes

A canonical mapping from the `graph.node_kinds` strings to integer indices for the R-GIN's embedding lookup table (16d learned embedding per type).

### Vocabulary

The vocabulary must be **stable across model versions** — changing indices invalidates trained weights. Pin the vocabulary as a compile-time constant.

```rust
/// Canonical node type vocabulary for R-GIN embedding lookup.
/// DO NOT reorder — indices are baked into trained model weights.
pub const NODE_TYPE_VOCAB: &[&str] = &[
    "function",     // 0
    "method",       // 1
    "class",        // 2
    "struct",       // 3
    "trait",        // 4
    "enum",         // 5
    "module",       // 6
    "package",      // 7
    "type_alias",   // 8
    "constant",     // 9
    "interface",    // 10 — TypeScript (future)
    "unknown",      // 11 — fallback
];

pub const UNKNOWN_TYPE_INDEX: usize = 11;
```

### Mapping Rules

1. Exact match against vocabulary → index.
2. Case-insensitive fallback: lowercase the kind string.
3. Alias resolution:

| Parser output | Maps to | Rationale |
|---|---|---|
| `"def"` | `"function"` (0) | Python function |
| `"class_method"`, `"static_method"`, `"classmethod"`, `"staticmethod"` | `"method"` (1) | Python method variants |
| `"dataclass"` | `"class"` (2) | Python dataclass is a class |
| `"variant"` | `"enum"` (5) | Rust enum variant |
| `"const"`, `"static"` | `"constant"` (9) | Rust/TS constants |

**Not aliased:** `"impl"` blocks. Verify: does the Rust parser emit `kind: "impl"` for impl blocks? If so, they should map to `"unknown"` (11) rather than `"method"` — an impl block is a container, not a function. The methods inside it are emitted separately with their own kinds.

4. No match → index 11 (`"unknown"`).

### Implementation

**File:** `packages/topo-analyzer/src/types.rs` (extend existing)

```rust
/// Map a node kind string to the canonical vocabulary index.
pub fn node_type_index(kind: &str) -> usize { ... }
```

~40 lines including the vocabulary constant and match logic.

### Validation

- Unit tests for every vocabulary entry (direct and alias).
- Test unknown fallback.
- Run on topo self-analysis: count unknowns. Should be zero for a well-parsed Rust codebase. If nonzero, investigate which kinds the parser emits that aren't in the vocabulary.

---

## 4. Spectral PE Export Extension

### Current State (from spectral.rs)

- `SpectralResult.components: Vec<(Vec<usize>, DecompResult)>` — each tuple is (component node indices, decomp result).
- `DecompResult.eigenvectors: Vec<Vec<f64>>` — indexed by **local** row within the component. Row i corresponds to `components[j].0[i]` (global node index).
- `DecompResult.eigenvalues: Vec<f64>` — non-trivial eigenvalues, ascending.
- Current padding: eigenvectors are padded to uniform width across components with **tiny noise** (1e-6), not zeros (spectral.rs:82-93). This is for clustering (avoids bias toward a default cluster).

### What Needs to Change

For R-GIN input, we need:

1. **Uniform k=16.** Always pad/truncate to exactly 16 eigenvector columns.
   - Fewer than 16 non-trivial eigenvectors: **zero-pad** (not noise-pad). For R-GIN, SignNet handles the sign ambiguity; zero-padding means "no structural information at this frequency," which is semantically correct. The noise padding in the current code is for clustering, not for R-GIN.
   - More than 16: take the first 16 (smallest non-trivial eigenvalues capture community structure).

2. **Eigenvalue pairing.** For each node v, produce a 16×2 array:
   ```
   spectral_pe[v] = [[u₁(v), λ₁], [u₂(v), λ₂], ..., [u₁₆(v), λ₁₆]]
   ```

3. **Per-node eigenvalues.** Nodes in different connected components have different eigenvalues. The export must store per-node eigenvalue vectors (not a single shared vector).

4. **Small components** (< MIN_COMPONENT_SIZE = 4 nodes): all-zero spectral PEs (both eigenvector components and eigenvalues). These nodes are in `SpectralResult.unassigned`.

### Implementation

**New function** in `spectral.rs`:

```rust
/// Produce spectral positional encodings for all nodes.
///
/// Returns (pe_vecs, pe_vals) where:
///   pe_vecs[global_node_idx] = [u₁(v), u₂(v), ..., u₁₆(v)]  (eigenvector components)
///   pe_vals[global_node_idx] = [λ₁, λ₂, ..., λ₁₆]            (eigenvalues of v's component)
///
/// Nodes in small/unassigned components get all-zero vectors.
/// Padded with zeros (not noise) to exactly k columns.
pub fn spectral_pe_export(
    spectral_result: &SpectralResult,
    n: usize,          // total number of nodes in graph
    k: usize,          // target PE dimension (typically 16)
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    // For each component in spectral_result.components:
    //   Map local row indices → global node indices
    //   Take first min(k, len) eigenvector columns, zero-pad to k
    //   Pair each column with its eigenvalue
    // For unassigned nodes: all-zero
}
```

**This function does NOT modify the existing `SpectralResult` or the noise-padded eigenvectors.** It creates a fresh export from the raw decomposition results. The existing noise-padded vectors continue to be used for clustering.

### Validation

- Small graph (6 nodes, 4 eigenvectors): verify zero-padding to k=16.
- Disconnected graph: verify each component gets its own eigenvalues.
- Unassigned nodes (component size < 4): verify all-zero PE.
- Eigenvalue ordering: ascending, matching the eigenvector columns.

---

## 5. Feature Export Command

### Purpose

A CLI command that runs Phase 1 + Phase 2 analysis and exports all R-GIN input features in a format consumable by the Python training pipeline.

### Command Interface

```bash
# From source code (parses + embeds + exports)
topo export-features <PATH> --output features.npz

# From pre-parsed graph + pre-computed embeddings
topo export-features --input graph.json --embeddings embeddings.json --output features.npz

# Batch mode (for dataset preprocessing)
topo export-features --batch repo_list.txt --output-dir features/
```

### CLI Integration

Add `ExportFeatures` variant to the `Command` enum in `packages/topo-cli/src/args.rs`:

```rust
/// Export R-GIN training features (spectral PE, RWPE, tree features, etc.)
ExportFeatures {
    /// Path to project root (or --input for pre-parsed)
    path: Option<PathBuf>,
    /// Pre-parsed graph JSON
    #[arg(long)]
    input: Option<PathBuf>,
    /// Pre-computed embeddings JSON
    #[arg(long)]
    embeddings: Option<PathBuf>,
    /// Output file path (.npz)
    #[arg(long, short)]
    output: PathBuf,
}
```

Dispatch in `main.rs` to new `packages/topo-cli/src/export.rs`.

### Output Format: NPZ

NPZ (NumPy compressed archive) is the standard interchange between Rust numerical code and Python/PyTorch. The `ndarray-npy` crate (0.8.x, compatible with ndarray 0.16) provides `NpzWriter`.

```
features.npz contents:
  semantic           float32[n, 768]       CodeLM embeddings
  spectral_vecs      float32[n, 16]        Eigenvector components u_i(v)
  spectral_vals      float32[n, 16]        Per-node eigenvalues λ_i (component-specific)
  rwpe               float32[n, 16]        Random walk PE
  tree_features      int32[n, 4]           depth, sibling_index, subtree_size, parent_subtree_size
  node_types         int32[n]              Node type vocabulary indices
  edge_index_calls   int32[2, m_calls]     COO edge index for calls
  edge_index_imports int32[2, m_imports]   COO edge index for imports
  edge_index_inherits int32[2, m_inherits] COO edge index for inherits
```

**Empty edge types:** If a graph has zero edges of a given type (e.g., no inherits), the NPZ MUST still contain the key with an empty `int32[2, 0]` array. This ensures consistent loading — the PyG data loader always finds the key and sets a `[2, 0]` edge_index, which the model's `shape[1] > 0` check handles cleanly.

**String arrays (node_ids):** NPZ does not support string arrays natively. Export node IDs as a separate JSON sidecar file (`features.meta.json`) alongside the NPZ, containing both node_ids and graph metadata.

**f64 → f32 downcast:** All internal computation uses f64. The NPZ export truncates to f32 for ML consumption. For RWPE values, the smallest expected values are ~1/n ≈ 0.0001 for n=10K — well above f32 minimum normal (~1.2e-38). No precision concern.

**`ndarray-npy` limitations:** The crate writes arrays one at a time via `NpzWriter::add_array()`. Each array call needs a separate type parameter, so float32 and int32 arrays require separate `.add_array::<f32>(...)` and `.add_array::<i32>(...)` calls. This is straightforward but must be done manually per array.

### Metadata Sidecar (`features.meta.json`)

```json
{
  "repo": "owner/repo",
  "language": "rust",
  "n_nodes": 847,
  "n_edges": {"calls": 1200, "imports": 800, "inherits": 150},
  "node_ids": ["pkg.module.func_a", "pkg.module.func_b"],
  "n_components": 3,
  "largest_component_ratio": 0.92,
  "spectral_k_actual": 12,
  "modularity_q": 0.74,
  "fiedler_value": 0.023,
  "modules": [
    {
      "id": 0,
      "label": "auth",
      "member_indices": [0, 3, 7, 12],
      "normalized_layer_position": 0.33,
      "cohesion": 0.72,
      "semantic_coherence": 0.61
    }
  ],
  "roles": [
    {"node_index": 0, "role": "hub", "local_variation": 0.34}
  ],
  "phase2_health": {
    "semantic_smoothness": 0.34,
    "semantic_structural_ami": 0.42
  }
}
```

**Critical fields for downstream consumers:**
- `modules[].member_indices` — Integer indices into the NPZ node arrays. Required by the depth probe (Step 2) and perturbation test (Step 3).
- `modules[].normalized_layer_position` — Layer position from edge-majority inference, in [0, 1]. Required by the depth probe.
- `roles[].local_variation` — Phase 2 per-node semantic disagreement score. Required by Tier 2 evaluation (Step 3) for Phase 2/Phase 3 cross-validation.
- `node_ids` — String identifiers for aligning Phase 2 results with Phase 3 outputs.

### Pipeline

The export command:
1. Parse the codebase (or read pre-parsed graph JSON).
2. Build `Graph` with `edge_kinds = None` (ensures defines edges are included).
3. Run Phase 1 spectral analysis → `SpectralResult`.
4. Compute spectral PE export (§4): eigenvector+eigenvalue pairs, k=16.
5. Compute RWPE (§1): 16-step return probabilities.
6. Compute tree features (§2): depth, sibling_index, subtree_size, parent_subtree_size.
7. Map node types to vocabulary indices (§3).
8. Read or compute semantic embeddings (768d CodeLM).
9. Write NPZ file + metadata JSON sidecar.


---

## 6. Cargo Dependencies

| Crate | Version | Purpose | Added to |
|-------|---------|---------|----------|
| `ndarray-npy` | 0.8 | NPZ file writing | topo-cli |

No new dependencies for topo-analyzer — RWPE, tree features, and node type mapping use existing `Graph` and standard library only.

---

## 7. File Summary

| New file | Package | Purpose | Est. lines |
|----------|---------|---------|------------|
| `rwpe.rs` | topo-analyzer | RWPE computation | ~150 |
| `tree.rs` | topo-analyzer | Defines-tree feature extraction | ~80 |
| `export.rs` | topo-cli | Feature export command + NPZ writing | ~250 |

| Modified file | Change |
|---------------|--------|
| `types.rs` (topo-analyzer) | Add `NODE_TYPE_VOCAB`, `node_type_index()` |
| `spectral.rs` (topo-analyzer) | Add `spectral_pe_export()` |
| `lib.rs` (topo-analyzer) | Wire new modules (`mod rwpe; mod tree;`) |
| `args.rs` (topo-cli) | Add `ExportFeatures` command variant |
| `main.rs` (topo-cli) | Dispatch to export.rs |
| `Cargo.toml` (topo-cli) | Add `ndarray-npy` dependency |

---

## 8. Testing Strategy

Unit tests per new function covering:
- **RWPE:** Known analytic values (cycle graph, star graph), disconnected components, Python `scipy.sparse` reference match (≤ 1e-10). Property: all values in [0, 1].
- **Tree features:** 3-level hierarchy, forest (multiple roots), orphan nodes → [0,0,0,0].
- **Node types:** All vocabulary entries, aliases, unknown fallback.
- **Spectral PE:** Zero-padding to k=16, truncation from k>16, per-component eigenvalues.

Integration test: `topo export-features` on topo itself → load in Python → verify all array shapes and metadata fields.

---

## 9. Definition of Done

- [ ] `compute_rwpe()` implemented, unit-tested, Python reference match.
- [ ] `compute_tree_features()` implemented, unit-tested.
- [ ] `node_type_index()` with full vocabulary + aliases, unit-tested.
- [ ] `spectral_pe_export()` with k=16 zero-padding, unit-tested.
- [ ] `topo export-features` CLI command produces valid NPZ + metadata JSON.
- [ ] Integration test: export → Python load → shape verification.
- [ ] `cargo test --workspace` passes.
- [ ] `cargo clippy --workspace` clean.
- [ ] No regressions in existing Phase 1/Phase 2 tests.
