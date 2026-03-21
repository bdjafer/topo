# Hybrid Structural-Semantic Analysis

## The Two Descriptions

Every code entity has two independent descriptions:

1. **Where it sits** — its position in the dependency graph. Who calls it, what it imports, what calls it. This is the structural description, encoded as the adjacency matrix.

2. **What it does** — its content, its name, its types, its purpose. This is the semantic description, encoded in the source code text.

These descriptions are orthogonal. A structural embedding places every entity in a geometric space where position encodes coupling. A semantic embedding places every entity in a space where position encodes meaning. Neither requires the other.

Today, topo computes only the structural description. Spectral decomposition of the graph Laplacian produces structural embeddings — each node gets coordinates in eigenspace, where distance encodes coupling strength. From these embeddings: module detection (clustering), role classification (degree/betweenness in embedding space), anomaly detection (outlier positions), and health metrics (modularity Q, silhouette).

This plan adds the semantic description alongside the structural one, and introduces mathematical tools for comparing them.

## Why Both

The structural embedding alone tells you the architecture the codebase *has*. The semantic embedding tells you the architecture the codebase *should have*, based on what each entity does. Neither is complete:

- Structural alone cannot detect **misplaced concerns** — a function that handles authentication but sits in the billing module, tightly coupled to billing code through accidental dependency. Structurally, it belongs where it is. Semantically, it doesn't.

- Semantic alone cannot detect **accidental coupling** — two unrelated functions forced into the same module by a shared dependency. Semantically, they're unrelated. Structurally, they're bound together.

- Neither alone can detect **missing abstractions** — two functions in different modules that do the same thing independently, because no shared utility exists. Semantically close, structurally distant.

The findings that matter most for developers live in the **disagreements** between the two embeddings. This is why the two spaces must remain separate — a joint embedding that blends them would lose the ability to detect disagreements, which is the entire point.

Semantic analysis exists to **validate structural findings**, not to generate its own. The structural layer identifies anomalies; the semantic layer confirms or refutes them. Every piece of semantic machinery must be justified by how much it improves the precision of structural anomaly detection. The evaluation metric is anomaly precision, measured against developer judgment on real codebases.

This is not speculative. The software architecture recovery literature (Maqbool & Babri 2007, Corazza et al. 2016) has validated that combining structural and semantic signals produces better architecture recovery than either alone. What's new here is the method of comparison: graph signal processing rather than feature concatenation.

## Semantic Embedding: jina-embeddings-v2-base-code

A 161M-parameter code embedding model, trained on GitHub code + 150M coding QA pairs across 30+ programming languages including Python and Rust. Produces 768-dimensional dense vectors with 8192-token context — enough to embed entire function bodies, not just identifiers.

Integration via `fastembed-rs` (the Rust ONNX embedding crate used by Qdrant, 624K downloads, actively maintained):

```rust
use fastembed::{TextEmbedding, InitOptions, EmbeddingModel};

let model = TextEmbedding::try_new(InitOptions {
    model_name: EmbeddingModel::JinaEmbeddingsV2BaseCode,
    show_download_progress: true,
    ..Default::default()
})?;

let embeddings = model.embed(vec!["fn authenticate(user: User) -> Token { ... }"], None)?;
// embeddings[0] is a Vec<f32> with 768 dimensions
```

The model downloads once (~300MB), cached locally in `~/.cache/fastembed/`. All inference is CPU-only, synchronous, no Python dependency, no network after first download. Apache-2.0 license.

**Why this model:**
- **Code-native.** Trained on code, not adapted from a text model. Understands syntax, naming conventions, and code semantics across 30+ languages.
- **Dense embeddings.** 768-dim continuous vectors — exactly what the Rayleigh quotient, GFT energy profile, and local variation tools are designed to operate on. Graded cosine similarities instead of sparse binary overlap.
- **8K context.** Can embed the full function body, capturing what the code *does*, not just what it's *named*. This is the difference between "this function mentions `auth`" and "this function validates JWT tokens, checks expiry, and returns user claims."
- **Zero-friction Rust integration.** Built-in model variant in `fastembed-rs`. No model export, no tokenizer wrangling, no Python interop.

**Operational properties:**
- First-run download (~300MB). Cached after that.
- ~50ms per function on CPU (fast enough for codebases up to ~10K functions in seconds).
- Adds `fastembed` + `ort` as Cargo dependencies (~15MB binary size increase).
- Deterministic for a given model version. Pin version as a constant in code.
- The `fastembed` dependency is behind a Cargo feature flag (`semantic`). Building without it produces the same binary as today. WASM builds exclude it (ONNX Runtime does not compile to WASM).

### Embedding Input

The parser must provide source text for each node. For each code entity, assemble a structured context window:

```
# module: topo_analyzer.modules
# file: packages/topo-analyzer/src/modules.rs

fn annotate_modules(
    clusters: &HashMap<String, usize>,
    fingerprints: &HashMap<String, Vec<f64>>,
    silhouette: f64,
    unassigned_cluster: Option<usize>,
) -> Vec<EnrichedModule> {
    // ... body ...
}
```

The input includes: (a) the enclosing module path (domain framing), (b) the function signature (contracts, parameter types), (c) the body up to ~6K tokens (implementation semantics), (d) docstrings/comments if present.

**Prerequisite:** The graph contract (`NodeEntry`) must be extended with source span information (`line_end` or `byte_start`/`byte_end`) so source text can be extracted. Both parsers already compute this internally (Python has `node.end_lineno`, Rust has `text_range().end()`) — they just don't persist it.

### Model Download UX

The 300MB model download is a category change for a zero-dependency CLI. Handle explicitly:

- `topo analyze <path>` — structural analysis only (no model needed, default)
- `topo analyze <path> --semantic` — semantic analysis enabled; requires model
- First run with `--semantic`: show progress bar on stderr (`show_download_progress: true`). If stderr is not a terminal (CI), emit periodic status lines.
- Air-gapped / offline: fail with clear error: "Model not found. Run `topo model download` first or set `FASTEMBED_CACHE_DIR`."
- `topo model download` — explicit download command, pre-populate cache
- `topo model list` — show available models, cache status, disk size

### Signal Quality Gate

Dense embeddings can produce weak signal on pathological inputs (tiny codebases, auto-generated code). The pipeline detects this and suppresses semantic findings rather than report noise.

Gate criteria:
- Variance of pairwise cosine similarities (clustering near zero = no discriminative power)
- Mean pairwise cosine similarity (too high = everything looks the same)
- The gate fires when variance of within-module cosine similarities is not significantly different from variance across modules (permutation test). If this test fails, the semantic signal does not carry structural information.
- If the gate fails, the analysis output includes structural findings only, with a note that semantic signal was insufficient.

### Embedding Caching

For CI performance, cache embeddings per node via content hashing:

```
cache_key = (node_id, blake3(source_text), model_version)
```

Store in `.topo/embeddings.cache`. On re-analysis:
- Unchanged nodes: load from cache
- Changed nodes: re-embed and update
- Deleted nodes: remove from cache

The graph structure (edges) changes more often than node content. Spectral decomposition must re-run every time, but embedding is cacheable. This separation keeps semantic analysis fast for CI even on large codebases.

## Mathematical Tools

Five tools, each mathematically well-defined for the problem, computationally cheap given what topo already computes, and producing interpretable output. They answer different questions and are not redundant:

| Tool | Scope | Answers |
|---|---|---|
| Semantic coherence | per module | "Is this structural module semantically coherent?" |
| Rayleigh quotient | whole codebase | "Overall, how well does structure match semantics?" |
| GFT energy profile | whole codebase, per scale | "At which structural scale does disagreement live?" |
| Local variation | per node | "Which specific nodes disagree with their neighbors?" |
| AMI | whole codebase | "Do semantic clusters align with structural clusters?" |

### 1. Semantic Coherence per Module

**What it computes.** For each structurally-detected module, the average pairwise cosine similarity of its members' semantic vectors.

**What it means.** A module where all members share domain semantics (auth, token, session) scores high. A module mixing unrelated code (auth, render, invoice) scores low.

**Math.** For module c with members {m₁, ..., mₖ}:
```
coherence(c) = mean(cos(M[i], M[j]))  for all i ≠ j in c
```
Guard against zero vectors. Modules with |c| < 6 get no coherence score (too few pairs for a meaningful average, and centroid instability below this size).

Compare against null distribution (random groups of same size) for significance. Use permutation test, not z-score, for small modules — the null distribution is not Gaussian.

### 2. Rayleigh Quotient (Global Smoothness)

**What it computes.** How smoothly the semantic signal varies across the structural graph. One number summarizing total structural-semantic agreement.

```
smoothness(f) = fᵀLf / ‖f‖²
```

where f is a semantic signal (one dimension of the 768-dim embedding) and L is the graph Laplacian. Computable with one sparse matrix-vector multiply. No eigendecomposition needed.

**What it means.** Low value = semantics vary smoothly over the graph (connected nodes have similar embeddings). High value = semantics vary sharply (connected nodes are semantically different). A well-organized codebase will score low. A tangled codebase will score high.

**Aggregation across embedding dimensions.** Compute the Rayleigh quotient per dimension, aggregate as the mean weighted by that dimension's signal energy (‖f‖²). This equals the multivariate Rayleigh quotient. Note: energy weighting means embedding dimensions with larger variance dominate. This is correct (it reflects actual smoothness of the semantic content) but should be documented.

### 3. GFT Energy Profile (Scale-Resolved Disagreement)

**What it computes.** How semantic disagreement is distributed across structural scales — coarse module boundaries vs fine-grained local coupling.

The Rayleigh quotient collapses scale information into one number. The GFT energy profile preserves it. For each eigenvector uᵢ (with eigenvalue λᵢ):

```
energy(λᵢ) = |f̂ᵢ|² = |uᵢᵀf|²
```

**What it means.**
- Energy concentrated at low λ (coarse structure): semantics agree with the major module boundaries. Good.
- Energy concentrated at mid λ (sub-module structure): semantics are consistent at the top level but messy within modules.
- Energy concentrated at high λ (fine-grained coupling): connected nodes are semantically foreign to each other.

This distinction is architecturally meaningful. "Your disagreement is at the top-level module boundary" and "your disagreement is within modules" are very different diagnoses with very different fixes.

**Eigenvector count.** The current `SPECTRAL_MAX_EIGENVECTORS` is 20 but the eigengap heuristic often selects k=2-5 for clustering. For a meaningful GFT profile, compute 15-20 eigenvectors regardless of the eigengap clustering choice. The clustering uses the eigengap-selected subset; the GFT profile uses the full set of computed eigenvectors. This gives enough spectral coverage to see the profile shape across low-to-mid frequencies.

The plan does not claim to see the full spectrum. The bottom-20 eigenvectors cover the community-structure frequencies. The very highest frequencies (fine-grained oscillations) are captured instead by local variation (tool #4), which directly measures per-node disagreement with neighbors without needing top-k eigenvectors.

**Cost.** One matrix multiply: `f̂ = Uᵀf` where U ∈ ℝⁿˣᵏ is already computed. Negligible.

### 4. Local Variation per Node (Per-Node Disagreement)

**What it computes.** For each node, how much its semantic content differs from its structural neighbors, normalized by degree.

```
variation(n) = (1 / deg(n)) · Σⱼ wₙⱼ · (1 - cos(M[n], M[j]))   for j ∈ neighbors(n)
```

**Degree-normalized.** Without the `1/deg(n)` factor, hubs with 50 neighbors accumulate more variation mechanically than leaves with 2 neighbors, even if both agree equally well with their local context. Since we interpret local variation conditioned on structural role, and role correlates with degree, the normalization is essential to avoid confounding.

**Cosine distance, not Euclidean.** In 768 dimensions, Euclidean distances concentrate (curse of dimensionality) — the ratio of max to min distance converges to 1, destroying discriminative power. Cosine distance does not suffer from this because it normalizes by vector magnitude. This is standard practice for transformer embeddings.

**What it means.** High local variation = this node is semantically different from the nodes it's coupled to. Low local variation = it fits in with its neighbors.

**Interpretation requires structural role context.** A bridge node with high local variation is normal (it connects different domains). A cluster-interior node with high local variation is suspicious (it's coupled to things it has nothing in common with). The structural role provides the prior; the semantic signal tests whether the prior is met.

### 5. AMI Between Structural and Semantic Partitions

**What it computes.** Cluster the semantic embeddings independently (spherical k-means on L2-normalized vectors). Compare the resulting partition against the structural partition (spectral modules) using Adjusted Mutual Information.

**What it means.** High AMI = the semantic clusters and structural clusters agree beyond what chance would produce. Low AMI = they disagree.

**Why this is not the same NMI baseline from CLAUDE.md.** The documented NMI-baseline problem was about comparing structural clusters vs *directory grouping*, which breaks for single-package projects (produces 1 group). Here, NMI compares structural clusters vs *semantic clusters* — a completely different baseline. Semantic k-means finds multiple clusters even within a single package, because it operates on code content similarity, not file paths.

**Implementation notes:**
- L2-normalize all embedding vectors before k-means. Transformer embeddings are anisotropic (concentrate in a cone); normalization makes k-means clusters more spherical and better-conditioned.
- Use AMI (Adjusted Mutual Information) rather than raw NMI. AMI corrects for chance agreement and handles different cluster counts without bias — important because semantic and structural partitions may legitimately have different granularity.
- Compute semantic k-means across a range of k values (structural_k - 2 to structural_k + 2, minimum 2). Report the max AMI across this range. This avoids suppressing signal when the code has 5 structural modules but 3 natural semantic domains — forcing k=5 on the semantic side would split semantic clusters artificially and deflate the metric for reasons that aren't architectural problems.

## What Was Dropped and Why

### Procrustes Analysis — dropped

Five independent reviews rejected it:

- **Dimensionality mismatch.** Structural embeddings have k ≈ 5-20 dimensions. Semantic embeddings have 768. Procrustes requires equal dimensions.
- **Violated assumptions.** Procrustes assumes the two spaces are relatable by rotation. They are not — different generating processes, different geometries.
- **Superseded.** If alignment is ever needed, CCA (Canonical Correlation Analysis) is strictly more informative: handles different dimensions natively, finds which structural axes correspond to which semantic axes, reports per-axis correlation.

### Per-Node GFT Reconstruction — dropped

Reconstructing `f_high(n)` as a per-node disagreement score requires top-k eigenvectors we don't compute and conflates signal magnitude with spectral character. Local variation (tool #4) replaces this — it directly measures per-node disagreement without needing any eigenvectors.

The GFT *energy profile* (tool #3) is retained. It uses bottom-k eigenvectors to show how semantic energy distributes across structural scales. Only the per-node reconstruction was dropped.

### Role Semantic Fit — dropped

Measuring each node's similarity to the semantic centroid of its structural role category (all hubs, all bridges, etc.). Circular and broken: bridge nodes connect different domains by definition, so the centroid of bridge embeddings is unrepresentative. All bridges would score as poor fits — false positives for the role that most needs validation.

### RMT Null Model (Marchenko-Pastur) — dropped

Three independent reviews confirmed: the Marchenko-Pastur distribution describes eigenvalues of random Wishart matrices (sample covariance matrices). The graph Laplacian is not a Wishart matrix — it has degree constraints, a guaranteed zero eigenvalue, and bounded spectrum ([0, 2] for normalized Laplacian). MP is the wrong null model.

The correct null model for graph Laplacians is the configuration model (random graphs preserving the observed degree sequence). However, implementing this properly requires either Monte Carlo sampling of random graphs or analytic results from the configuration model spectral distribution — both are significant work for marginal gain given that modularity Q and silhouette already measure structural significance.

Dropped from v1. If spectral significance testing is needed later, use the configuration model null, not Marchenko-Pastur.

## New Issue Types

Two new issue types. Both are gated on semantic signal quality — suppressed when the signal quality gate fails.

### `misplaced_concern`

A node whose semantic content is more similar to a different module than its own.

Detection:
```
For node n in structural module A:
  sim_own = cos(M[n], centroid(A))
  sim_best = max over other modules B: cos(M[n], centroid(B))
  If sim_best > sim_own AND (sim_best - sim_own) > significance_threshold:
    → misplaced_concern
    severity ∝ (sim_best - sim_own)
    description: "Node X is semantically closest to module Y
                  (similarity 0.78) but structurally assigned to
                  module Z (similarity 0.21). Consider moving it
                  to Y or extracting a shared module."
```

**Filters to control false positives:**
- Skip modules with fewer than 6 members (centroid instability).
- Exclude nodes classified as `bridge`, `hub`, or `utility` by the structural role classifier — their cross-module semantic profile is structural, not a misplacement.
- Per-module cap: if >40% of members are flagged, suppress all flags for that module (the module boundary is wrong, not individual nodes).
- Require edge evidence: the node must have at least some edges to the suggested alternative module.
- **Gate on target module coherence.** Only flag if the suggested target module B is itself semantically coherent (coherence > null threshold). A diffuse "utils" or "core" module will attract false matches because its centroid sits near the center of embedding space — close to everything, representative of nothing. Requiring the target to be coherent ensures the suggestion is "move this to a well-defined module" rather than "move this to another mess."

### `incoherent_module`

A structural module whose members are semantically unrelated.

Detection:
```
For module c:
  If semantic_coherence(c) < null_distribution_threshold:
    → incoherent_module
    severity ∝ how far below the null expectation
    description lists the semantic sub-clusters found within the module
```

### `semantic_duplicate` (v2)

Two nodes in different structural modules with high semantic similarity. Viable with dense embeddings but requires careful threshold calibration and false-positive filtering. Deferred until the core hybrid pipeline is validated.

## Schema Changes

### `architecture.modules` — add semantic coherence

Each module gains:
```json
{
  "semantic_coherence": 0.72,
  "top_terms": ["auth", "token", "session"]
}
```

`semantic_coherence`: average pairwise cosine similarity of member semantic vectors. Null if module size < 6 or semantic signal quality gate fails.

`top_terms`: top-3 terms characterizing the module. Derived from node ID tokenization (split camelCase/snake_case, strip module prefix to avoid circularity, TF-IDF weighting across modules). Available even without the embedding model.

### `health` — add alignment and smoothness

```json
{
  "modularity_q": 0.47,
  "semantic_smoothness": 0.23,
  "semantic_structural_ami": 0.68,
  "semantic_energy_profile": {
    "eigenvalues": [0.02, 0.08, 0.15, 0.31, 0.52, 0.71, 0.89],
    "semantic_energy": [0.35, 0.20, 0.15, 0.12, 0.08, 0.06, 0.04]
  }
}
```

`semantic_smoothness`: Rayleigh quotient. Lower = better organization. Null if semantic signal quality gate fails.

`semantic_structural_ami`: AMI between structural and semantic partitions. Higher = better alignment. Computed as the max AMI across a range of semantic k values (structural_k ± 2). Null if semantic signal quality gate fails.

`semantic_energy_profile`: GFT energy at each eigenvalue — how semantic disagreement distributes across structural scales. Computed from 15-20 eigenvectors (not just the eigengap-selected subset). Null if semantic signal quality gate fails.

### `issues` — new kinds

Two new issue kinds: `misplaced_concern` and `incoherent_module`. Both carry the same `severity`, `confidence`, `anchors` structure as existing issues.

## What This Does Not Change

The existing structural analysis pipeline is untouched. Spectral decomposition, module detection, role classification, existing anomaly detection — all remain as-is. The semantic layer is additive. Without `--semantic`, the output is identical to today's.

The `fastembed` / ONNX dependency is behind a Cargo feature flag (`semantic`). Building without `--features semantic` produces the same binary as today. WASM builds exclude it — ONNX Runtime does not compile to WASM. The `topo-web` package is structural-analysis only.

`top_terms` (from node ID tokenization) is available without the embedding model and without the feature flag. It requires no dependencies.

## Integration Architecture

Semantic embeddings flow through the pipeline as follows:

1. **Parser** outputs `CodeGraph` with source spans (`line_end` / byte range) on each node.
2. **CLI** (when `--semantic` is passed) reads source text for each node using the span information, assembles context windows, runs inference via `fastembed-rs`, produces `HashMap<String, Vec<f32>>` mapping node IDs to 768-dim vectors.
3. **Analyzer** receives embeddings as an optional input (`semantic_embeddings: Option<HashMap<String, Vec<f32>>>` on `AnalyzerInput`). All semantic tools are skipped if this is `None`.
4. **Formatter** renders semantic fields only when present (serde `skip_serializing_if`).

This keeps the analyzer embedding-agnostic. The CLI is the only component that knows about `fastembed-rs`. The analyzer operates on `Vec<f32>` vectors regardless of how they were produced. The WASM build and Python bindings can accept pre-computed embeddings from any source.

## Validation

Before shipping any semantic finding, validate on three codebases:

1. **topo itself** — known architecture, monorepo with clear package boundaries.
2. **Flask** — well-documented single-package library with known architectural intent.
3. **Click** — small, clean library. Known structure.

For each codebase:

1. Does the signal quality gate pass?
2. Do per-module coherence scores correlate with developer judgment?
3. Does the Rayleigh quotient differ between well-structured and poorly-structured regions?
4. Does the GFT energy profile show a meaningful shape? (Energy concentrated at low frequencies for clean architecture, spread for messy code?)
5. Do flagged `misplaced_concern` findings correspond to real architectural problems?
6. What is the false positive rate? If > 50%, the semantic signal is insufficient.

If validation fails: do not ship semantic findings. The structural analysis alone carries the load.

If validation succeeds: ship the semantic layer behind `--semantic`, with the signal quality gate as a hard requirement.

## Future Directions (Not in Scope)

These emerged from the review process as high-value but deferred:

- **CCA (Canonical Correlation Analysis)** — finds which structural axes correspond to which semantic axes. Replaces Procrustes if alignment is ever needed.
- **Heat kernel signatures** — replace raw eigenvectors as the structural embedding. Better numerical stability, fixes sign/rotation ambiguity.
- **Type-level semantic layer** — for typed languages (Rust), type signatures carry domain semantics independent of naming. A second semantic signal.
- **Per-topic smoothness** — decompose the semantic signal into topic channels, measure each topic's structural containment. "Your auth logic is 73% contained, with leaks into billing." Highest developer-facing value.
- **Multi-scale hierarchical analysis** — compute disagreement at each level of the code hierarchy (function → class → module → package).
- **Temporal drift** — track structural-semantic alignment across git history. Detect entities that were correctly placed when written but have drifted.
- **Semantic duplicate detection** — two nodes in different modules with high semantic similarity. Viable with dense embeddings but needs threshold calibration.
- **Structural entropy rate** — information-theoretic measure of graph predictability.
- **Motif-based anti-patterns** — bridge between global spectral analysis and specific local findings.
- **Configuration model spectral null** — correct RMT null model for graph Laplacians if spectral significance testing is needed.

## Implementation Sequence

### Step 1: Graph contract extension

Add `line_end: Option<u32>` (or `byte_start`/`byte_end`) to `NodeEntry` in both parsers. Both parsers already compute this internally — persist it.

### Step 2: `top_terms` (no model needed)

Tokenize node IDs (strip module prefix, split camelCase/snake_case), compute TF-IDF across modules, attach top-3 terms per module. ~120 lines. No external dependencies.

**Checkpoint.** Run on topo/Flask/Click. Do top_terms accurately characterize modules?

### Step 3: `fastembed` integration

Add `fastembed` behind `semantic` feature flag. Wire `--semantic` CLI flag. Implement embedding input assembly (structured context window from source spans). Implement `topo model download` and `topo model list` commands. ~200 lines.

**Checkpoint.** Run `topo analyze . --semantic --json` on topo. Verify non-null 768-dim vectors per node.

### Step 4: Semantic coherence + signal quality gate

Compute per-module coherence. Implement signal quality gate (permutation test on within-module vs across-module cosine similarity variance). ~150 lines.

**Checkpoint.** Do coherence scores correlate with developer judgment on test codebases?

### Step 5: Rayleigh quotient + GFT energy profile

Compute global smoothness (Rayleigh quotient, one matmul). Compute eigenvector count to 15-20 (regardless of eigengap). Project semantic signal onto eigenvectors for energy profile. ~100 lines.

### Step 6: Local variation (cosine distance)

Per-node cosine-distance-based local variation. Implement `cosine_distance` in stats. ~60 lines.

### Step 7: Issue detection

`misplaced_concern` with all filters (role filter, module size minimum, per-module cap). `incoherent_module` from coherence scores. Contrastive explanations in issue descriptions. ~150 lines.

### Step 8: AMI

Spherical k-means on L2-normalized embeddings across k range (structural_k ± 2). AMI comparison with structural partition, report max. ~100 lines.

### Step 9: Schema + formatter

Wire all new fields into `AnalysisOutput`, `ModuleOutput`, `HealthOutput`. Update formatter for `top_terms`, new issue kinds. ~100 lines.

Total: ~980 lines of new Rust code across topo-analyzer, topo-cli, topo-formatter, and both parsers.
