# Phase 2: Hybrid Structural-Semantic Analysis

Phase 1 is spectral analysis of the dependency graph — eigendecomposition, clustering, role classification, topological anomaly detection. It answers: "what is the architecture?" Phase 2 adds a semantic layer alongside the structural one and answers: "does the architecture match what the code actually does?"

---

## What Phase 1 Cannot See

A function that handles authentication but sits deep inside the billing module, called only by billing code, imported only by billing code — spectrally, it looks identical to a legitimate billing function. Same cluster, same centroid distance, same degree profile. Zero spectral anomaly signal. The graph says it belongs there.

The only way to know it's misplaced is to compare **where it sits** (structural position) with **what it does** (semantic content). This requires information the graph does not contain.

This is the entire class of **misplaced concern** anomalies — structurally invisible, architecturally critical. It includes:

- **Misplaced concerns.** Auth logic in billing. Rendering code in the data layer. Logging helpers in the domain model.
- **Incoherent modules.** A structural cluster that mixes unrelated domain concepts — the spectral clustering grouped them by coupling, but the coupling is accidental, not intentional.
- **Missing abstractions.** Two functions in different modules that do the same thing independently. Semantically close, structurally distant. Neither module knows the other exists. *(Phase 2 defers detection of this class to `semantic_duplicate`, after core pipeline validation. The 5 mathematical tools address misplaced concerns and incoherent modules; missing abstractions require cross-module pairwise similarity, which is computationally heavier and needs threshold calibration.)*

These are the highest-value findings for developers — they reveal where the codebase's actual organization diverges from its intended design. They are invisible to any purely structural method, including spectral analysis, because the signal lives in the **disagreement** between structure and semantics, not in either one alone.

---

## The Two Spaces

Every code entity has two independent descriptions:

1. **Where it sits** — its position in the dependency graph. Who calls it, what it imports, what calls it. This is the structural description, encoded as eigenvector coordinates in spectral space. Distance encodes coupling strength.

2. **What it does** — its content, its name, its types, its purpose. This is the semantic description, encoded as a dense vector from a code embedding model. Distance encodes meaning similarity.

These descriptions are orthogonal. A structural embedding and a semantic embedding place every entity in two different geometric spaces. The spaces must remain separate — a joint embedding that blends them loses the ability to detect disagreements, which is the entire point.

**Semantic analysis validates structural findings.** The structural layer identifies modules and anomalies; the semantic layer confirms or refutes them. Every piece of semantic machinery must be justified by how much it improves the precision of structural analysis. The evaluation metric is anomaly precision, measured against developer judgment on real codebases.

This is not speculative. The software architecture recovery literature (Maqbool & Babri 2007, Corazza et al. 2016) validates that combining structural and semantic signals produces better architecture recovery than either alone. What's new here is the method of comparison: graph signal processing rather than feature concatenation.

---

## Semantic Embedding: jina-embeddings-v2-base-code

A 161M-parameter code embedding model, trained on GitHub code + 150M coding QA pairs across 30+ programming languages including Python and Rust. Produces 768-dimensional dense vectors with 8192-token context.

Integration via `fastembed-rs` (Rust ONNX embedding crate used by Qdrant):

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

**Why this model:**
- **Code-native.** Trained on code, not adapted from a text model. Understands syntax, naming conventions, and code semantics across 30+ languages.
- **768-dim dense vectors.** Exactly what the Rayleigh quotient, GFT energy profile, and local variation tools operate on. Graded cosine similarities instead of sparse binary overlap.
- **8K context.** Embeds the full function body, not just identifiers. This is the difference between "this function mentions `auth`" and "this function validates JWT tokens, checks expiry, and returns user claims." The 8K context is what makes this a semantic signal rather than a naming-convention signal.
- **Zero-friction Rust integration.** Built-in model variant in `fastembed-rs`. No model export, no tokenizer wrangling, no Python interop.

**Operational properties:**
- First-run download (~300MB). Cached locally in `~/.cache/fastembed/`.
- ~50ms per function on CPU. Fast enough for codebases up to ~10K functions in seconds.
- Adds `fastembed` + `ort` as Cargo dependencies (~15MB binary size increase).
- Deterministic for a given model version. Pin version as a constant in code.
- Behind a Cargo feature flag (`semantic`). Building without it produces the same binary as today. WASM builds exclude it (ONNX Runtime does not compile to WASM).

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

**Why the full body matters.** A function named `process_payment` might log errors, validate auth tokens, or actually process payments. The name alone is ambiguous. The body is the ground truth. The 8K context window is what separates "naming-position correlation" (a weak signal) from "semantic-structural disagreement" (the signal we want). Without the body, the model sees names; with the body, it sees behavior.

**Prerequisite:** The graph contract (`NodeEntry`) must be extended with source span information (`line_end` or `byte_start`/`byte_end`) so source text can be extracted. Both parsers already compute this internally — they just don't persist it.

**Source text extraction modes:**
- **Auto-parse mode** (`topo analyze <path>`): The CLI has the project root. After parsing, read source files from disk using spans. Assemble context windows. Embed. This is the primary path.
- **Pre-parsed mode** (`topo analyze --input graph.json`): The CLI has no project root. Three options: (a) require `--project-root <path>` alongside `--input` to locate source files, (b) accept pre-computed embeddings via `--embeddings embeddings.json`, (c) skip semantic analysis and emit structural-only output. Option (a) is the default; (b) supports CI pipelines that compute embeddings separately; (c) is the fallback.
- **WASM/Python bindings**: Accept pre-computed embeddings as `Option<HashMap<String, Vec<f32>>>` on `AnalyzerInput`. The host environment is responsible for embedding computation. The analyzer is embedding-model-agnostic.

**Batching.** The `fastembed-rs` `embed()` API accepts a batch of strings. Process all entities in a single batch call rather than one-by-one. This is critical for throughput — batched ONNX inference is ~10x faster than sequential. For a 10K-entity codebase at ~5ms/entity batched: ~50 seconds total, not the ~500 seconds that sequential 50ms/entity would imply.

### Signal Quality Gate

Dense embeddings can produce weak signal on pathological inputs (tiny codebases, auto-generated code). The pipeline detects this and suppresses semantic findings rather than report noise.

**Quick pre-checks** (short-circuit before the expensive permutation test):
- Variance of pairwise cosine similarities < 0.01 → no discriminative power. Gate fails.
- Mean pairwise cosine similarity > 0.95 → everything looks the same. Gate fails.

**Primary gate: permutation test.**
- **Null hypothesis H₀:** Semantic embeddings carry no information about structural module boundaries — within-module semantic similarity is no higher than across-module semantic similarity.
- **Test statistic:** Mean within-module pairwise cosine similarity minus mean across-module pairwise cosine similarity.
- **Procedure:** Randomly permute module assignments N=200 times. For each permutation, recompute the test statistic. If the observed value exceeds the 95th percentile of the null distribution (α=0.05), the gate passes.
- **If the gate fails:** The analysis output includes structural findings only, with `semantic_enabled: false` and a note that semantic signal was insufficient.

### Embedding Caching

For CI performance, cache embeddings per node via content hashing:

```
cache_key = (node_id, blake3(embedding_input), model_version)
```

Note: `embedding_input` is the assembled context window (module path + file path + source text), not just the raw source text. This ensures cache invalidation when a node's module path changes due to structural re-analysis, even if the source text is unchanged.

Store in `.topo/embeddings.cache`. On re-analysis:
- Unchanged nodes: load from cache
- Changed nodes: re-embed and update
- Deleted nodes: remove from cache

The graph structure (edges) changes more often than node content. Spectral decomposition re-runs every time, but embedding is cacheable.

### Model Download UX

The 300MB model download is a category change for a zero-dependency CLI. Handle explicitly:

- `topo analyze <path>` — structural analysis only (default, no model needed)
- `topo analyze <path> --semantic` — semantic analysis enabled; requires model
- First run with `--semantic`: show progress bar on stderr. If stderr is not a terminal (CI), emit periodic status lines.
- Air-gapped / offline: fail with clear error: "Model not found. Run `topo model download` first or set `FASTEMBED_CACHE_DIR`."
- `topo model download` — explicit download command, pre-populate cache
- `topo model list` — show available models, cache status, disk size

---

## Mathematical Tools

Five tools, each answering a different question. None is redundant. They operate at different scopes (codebase, module, node) and scales (global, structural-scale, local).

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

Guard against zero vectors. Modules with |c| < 6 get no coherence score (too few pairs for meaningful average, and centroid instability below this size).

Compare against null distribution (random groups of same size) for significance. Use permutation test, not z-score, for small modules — the null distribution is not Gaussian.

### 2. Rayleigh Quotient (Global Smoothness)

**What it computes.** How smoothly the semantic signal varies across the structural graph. One number summarizing total structural-semantic agreement.

```
smoothness(f) = fᵀLf / ‖f‖²
```

where f is a semantic signal (one dimension of the 768-dim embedding) and L is the **normalized graph Laplacian** (same `L = I - D^{-1/2} A D^{-1/2}` already computed in Phase 1). Using the normalized Laplacian ensures the smoothness measure is consistent with the spectral decomposition and accounts for degree heterogeneity. Computable with one sparse matrix-vector multiply. No additional eigendecomposition needed.

**What it means.** Low value = semantics vary smoothly over the graph (connected nodes have similar embeddings = well-organized). High value = semantics vary sharply (connected nodes are semantically different = tangled).

**Aggregation across embedding dimensions.** Compute per dimension, aggregate as the mean weighted by that dimension's signal energy (‖f‖²). This equals the multivariate Rayleigh quotient.

### 3. GFT Energy Profile (Scale-Resolved Disagreement)

**What it computes.** How semantic disagreement distributes across structural scales — coarse module boundaries vs fine-grained local coupling. The Rayleigh quotient collapses scale into one number; the GFT energy profile preserves it.

For each eigenvector uᵢ (with eigenvalue λᵢ):
```
energy(λᵢ) = |f̂ᵢ|² = |uᵢᵀf|²
```

**What it means.**
- Energy at low λ (coarse structure): semantics agree with major module boundaries. Good.
- Energy at mid λ (sub-module structure): top-level is consistent, messy within modules.
- Energy at high λ (fine-grained coupling): connected nodes are semantically foreign.

"Your disagreement is at the top-level boundary" and "your disagreement is within modules" are very different diagnoses with very different fixes.

**Eigenvector count.** Compute 15-20 eigenvectors regardless of eigengap clustering choice. Clustering uses the eigengap-selected subset; the GFT profile uses the full set. This gives spectral coverage across low-to-mid frequencies. The highest frequencies (fine-grained oscillations) are captured by local variation (tool #4).

**Cost.** One matrix multiply: `f̂ = Uᵀf` where U ∈ ℝⁿˣᵏ is already computed. Negligible.

### 4. Local Variation per Node (Per-Node Disagreement)

**What it computes.** For each node, how much its semantic content differs from its structural neighbors, normalized by degree.

```
variation(n) = (1 / deg(n)) · Σⱼ wₙⱼ · (1 - cos(M[n], M[j]))   for j ∈ neighbors(n)
```

**Degree-normalized.** Here `deg(n) = Σⱼ wₙⱼ` (weighted degree), making the formula a proper weighted average of cosine distances. Without this normalization, hubs with 50 neighbors accumulate more variation mechanically than leaves with 2, even if both agree equally well with their context. Role correlates with degree, so normalization avoids confounding.

**Cosine distance, not Euclidean.** In 768 dimensions, Euclidean distances concentrate (curse of dimensionality) — max-to-min ratio converges to 1. Cosine distance normalizes by magnitude and retains discriminative power. Standard for transformer embeddings.

**Interpretation requires structural role context.** A bridge node with high local variation is normal (it connects different domains). A cluster-interior node with high local variation is suspicious (it's coupled to things it has nothing in common with). The structural role provides the prior; the semantic signal tests whether the prior is met.

### 5. AMI Between Structural and Semantic Partitions

**What it computes.** Cluster semantic embeddings independently (spherical k-means on L2-normalized vectors). Compare against the structural partition using Adjusted Mutual Information.

**What it means.** High AMI = semantic clusters and structural clusters agree beyond chance. Low AMI = they disagree.

**Not the same as the NMI baseline in Phase 1.** Phase 1's NMI compares structural clusters vs directory grouping, which breaks for single-package projects (produces 1 group). This compares structural clusters vs semantic clusters — semantic k-means finds multiple clusters even within a single package, because it operates on code content similarity, not file paths.

**Implementation:**
- L2-normalize all embedding vectors before k-means. Transformer embeddings are anisotropic.
- Use AMI, not raw NMI. AMI corrects for chance agreement and handles different cluster counts.
- Compute semantic k-means across k = structural_k ± 2 (minimum 2). Report max AMI. This avoids deflating the metric when the code has 5 structural modules but 3 semantic domains.

---

## New Issue Types

Two new issue types. Both are gated on semantic signal quality — suppressed when the gate fails.

### `misplaced_concern`

A node whose semantic content is more similar to a different module than its own. This is the highest-value finding Phase 2 introduces — it detects the class of problems that spectral analysis is structurally blind to.

**Detection:**
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

**Significance threshold.** The `(sim_best - sim_own)` margin must exceed a minimum to fire. Default: 0.15 (cosine similarity units). This is calibrated during validation (Step 7 checkpoint) — if false positive rate exceeds 50% on validation codebases, increase; if recall is too low, decrease. The threshold is a tunable constant, not a learned parameter. For a data-adaptive alternative: derive from the distribution of cross-module centroid similarities across the codebase (e.g., flag only when the margin exceeds 2σ of the inter-centroid distance distribution).

**Filters to control false positives:**
- Skip modules with fewer than 6 members (centroid instability — 6 members give 15 pairwise comparisons, the minimum for a meaningful average).
- Exclude nodes classified as `bridge`, `hub`, or `utility` by the structural role classifier — their cross-module semantic profile is structural, not a misplacement.
- Per-module cap: if >40% of members are flagged, suppress all flags for that module (the module boundary is wrong, not individual nodes — report `incoherent_module` instead).
- Require edge evidence: the node must have at least some edges to the suggested alternative module. A node with zero edges to module Y is not a candidate for moving there, no matter how semantically similar.
- **Gate on target module coherence.** Only flag if the suggested target module B is itself semantically coherent (coherence > null threshold). A diffuse "utils" or "core" module attracts false matches because its centroid sits near the center of embedding space — close to everything, representative of nothing.
- **Secondary nearest-neighbor check.** If a node is far from its module centroid but close to several individual members of its own module (k-NN, k=3), it likely belongs to a sub-cluster within the module, not to another module. Suppress the flag and let the `incoherent_module` detector handle the sub-cluster issue instead.

### `incoherent_module`

A structural module whose members are semantically unrelated. This indicates the module boundary is structurally real (nodes are coupled) but architecturally wrong (they have nothing in common semantically).

**Detection:**
```
For module c:
  If semantic_coherence(c) < null_distribution_threshold:
    → incoherent_module
    severity ∝ how far below null expectation
    description lists the semantic sub-clusters found within the module
```

The description is actionable: "Module X contains 3 semantic sub-groups: {auth, token, session}, {invoice, billing}, {config, env}. Consider splitting along these boundaries."

### `semantic_duplicate` (deferred)

Two nodes in different structural modules with high semantic similarity. Viable with dense embeddings but requires threshold calibration. Deferred until the core hybrid pipeline is validated.

---

## LLM Context Narrative (`--format=context`)

Phase 2 enables a new output format designed for LLM consumption: a compact structural narrative in 2,000–4,000 tokens that makes an LLM dramatically better at architectural reasoning.

An LLM reading source code can assess local quality. An LLM reading source code **plus the structural narrative** can reason about global architecture — module boundaries, typed dependencies, structural anomalies, misplaced concerns. This is the difference between reading symptoms and reading an MRI.

### The Format

```markdown
## myapp — layered monolith (4 tiers, 847 nodes)
Health: modularity Q=0.74, Fiedler λ₂=0.012, semantic smoothness=0.23
Structural-semantic alignment: AMI=0.68

### Module: payment (38 nodes, cohesion: 0.82, semantic coherence: 0.71)
Role: leaf subsystem, single entry point (PaymentService)
Top terms: payment, charge, refund
Bridges to: [orders] via OrderPaymentAdapter (calls)
            [users] via BillingProfileLoader (imports)

### Module: auth (22 nodes, cohesion: 0.79, semantic coherence: 0.85)
Role: leaf subsystem, bridges to 2 modules
Top terms: token, session, authenticate
Bridges to: [users] via UserAuthProvider (calls)
            [api] via AuthMiddleware (calls)

### Module: orders (52 nodes, cohesion: 0.71, semantic coherence: 0.63)
Role: central coordinator, hub
Top terms: order, cart, checkout
Bridges to: [payment] via OrderPaymentAdapter (calls)
            [inventory] via StockChecker (imports)
Concern: semantic coherence below threshold — may mix order management
         with fulfillment logic. Sub-clusters: {order, cart, checkout},
         {shipment, tracking, fulfillment}.

## Structural Concerns (3 high, 5 medium)

[high] misplaced_concern: AuthTokenValidator
  Location: src/payments/validators.rs
  Module: payment (structural), auth (semantic)
  Evidence: authentication vocabulary (token, verify, claims) in payment
            module interior. Semantic similarity to auth module: 0.78,
            to own module: 0.21.
  Suggested action: move to auth module, inject via interface.

[high] fragile_hub: OrderService
  Location: src/orders/service.rs
  Evidence: degree 34, betweenness 0.12, bridges orders↔payment↔inventory.
  Suggested action: split orchestration from domain logic.

[medium] incoherent_module: orders
  Evidence: semantic coherence 0.63 (null threshold 0.55).
  Sub-clusters: {order, cart} vs {shipment, fulfillment}.
  Suggested action: consider splitting orders into order-management
                    and fulfillment bounded contexts.
```

### Compression Strategy

A 2,000-4,000 token budget for mid-size codebases requires compression:

- **All modules get a summary line** (name, size, cohesion, role, top terms). One line each.
- **Only modules with concerns get detail** (bridges, semantic sub-clusters).
- **Only high and medium issues get cards.** Low issues are counted but not described.
- **The global health line** (Q, Fiedler, smoothness, AMI) is always present — 1 line.
- **For large codebases (>100 modules):** show top-10 modules by size, all modules with concerns, and summarize the rest as "N additional modules with no structural concerns."

### What the LLM Gets

The narrative provides everything an LLM needs to reason about architecture:

- **Module boundaries** — what the actual modules are (not directories)
- **Typed inter-module relationships** — which modules depend on which, through what coupling type (calls vs. imports)
- **Structural roles** — which modules are hubs, leaves, bridges
- **Semantic coherence** — whether structural modules correspond to domain concepts
- **Specific anomalies with evidence** — what's wrong, where, why, and what to do
- **Health metrics** — global structural quality in 4 numbers

---

## Module Labels (`top_terms`)

Available without the embedding model and without the `semantic` feature flag.

Tokenize node IDs (strip module prefix, split camelCase/snake_case), compute TF-IDF across modules, attach top-3 terms per module. This gives human-readable labels to spectrally-detected modules.

```json
{
  "label": "topo_analyzer.modules",
  "top_terms": ["annotate", "enriched", "module"],
  "size": 12,
  "cohesion": 0.79
}
```

This is the minimum-cost, zero-dependency way to make structural modules meaningful to developers and LLMs. It requires no embedding model — just string tokenization and TF-IDF.

---

## Structural Health Tracking

### Fiedler Value over Git History (`topo health`)

The Fiedler value (λ₂, second-smallest Laplacian eigenvalue) measures algebraic connectivity — how close the graph is to splitting into disconnected components. Tracking it across commits reveals structural health trends with zero ML:

```
commit  a1b2c3  λ₂ = 0.0015  Q = 0.74  ← baseline
commit  d4e5f6  λ₂ = 0.0014  Q = 0.73  ← slight decrease
commit  g7h8i9  λ₂ = 0.0008  Q = 0.71  ← sharp drop (bridge dep added)
commit  j0k1l2  λ₂ = 0.0003  Q = 0.68  ← approaching disconnection
commit  m3n4o5  λ₂ = 0.0012  Q = 0.75  ← refactoring reconnected
```

- **Monotonic decrease** → structural fragmentation. Architecture drifting toward disconnected subsystems.
- **Sudden drop** → someone introduced or removed a critical bridge. Worth investigating.
- **Monotonic increase** → growing coupling. Could mean healthy integration or creeping monolith.
- **Oscillation** → build-break-repair cycle.

### Semantic Smoothness over Time

When `--semantic` is available, track the Rayleigh quotient across commits alongside the Fiedler value. Rising smoothness = structure and semantics are aligning over time (good refactoring). Falling smoothness = structural drift from semantic intent.

### Implementation

`topo health <path>` walks git history (or reads cached snapshots), runs structural analysis at each commit (or sampled commits), outputs the trajectory:

```
topo health . --since=2024-01-01 --sample=weekly
```

The parse step dominates runtime. With topo-cache, only changed files re-parse. The analysis step is <100ms. Sampling (weekly, per-PR, per-N-commits) keeps wall time manageable.

---

## Domain Model Approximation (`--format=domain`)

Phase 2 enables a rough domain model extraction — bounded context approximation from structural modules + semantic coherence + top terms.

### Bounded Contexts

Structural modules with high semantic coherence are bounded context candidates. The top_terms provide the domain label.

```
Bounded contexts (5 detected, AMI with directory structure: 0.72):

  auth (22 nodes, coherence: 0.85)
    Top terms: token, session, authenticate
    Aggregate root candidate: AuthService (hub, entry_point)

  payment (38 nodes, coherence: 0.71)
    Top terms: payment, charge, refund
    Aggregate root candidate: PaymentService (hub, entry_point)

  orders (52 nodes, coherence: 0.63) ⚠ low coherence
    Top terms: order, cart, checkout
    May contain 2 sub-contexts: {order, cart} and {shipment, fulfillment}
```

### Context Map

Cross-module bridges, typed by coupling layer, approximate the DDD context map:

```
Context relationships:
  auth ──[calls]──→ users       (customer-supplier)
  orders ──[calls]──→ payment   (customer-supplier)
  orders ──[imports]──→ inventory (shared kernel)
  api ──[calls]──→ auth         (conformist)
```

The coupling layer type provides a rough classification:
- **Calls-only bridge** → runtime dependency → customer-supplier or partnership
- **Imports-only bridge** → compile-time dependency → shared kernel
- **Both** → tight coupling → consider anti-corruption layer

This is approximate. The mapping from structural coupling to DDD relationship types is a hypothesis, not a theorem. It provides a starting point for architectural discussion, not a finished domain model.

### Aggregate Root Candidates

Within each bounded context, nodes with high in-degree from context members AND the `entry_point` or `hub` structural role are aggregate root candidates. This is a heuristic — aggregates are defined by transactional consistency boundaries, not coupling patterns. Flag as "candidates" not "roots."

---

## Schema Changes

### `architecture.modules` — add semantic fields

Each module gains:
```json
{
  "semantic_coherence": 0.72,
  "top_terms": ["auth", "token", "session"]
}
```

`semantic_coherence`: average pairwise cosine similarity. Null if module size < 6 or signal quality gate fails.

`top_terms`: top-3 TF-IDF terms from node ID tokenization. Available even without `--semantic`.

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

All semantic fields null if gate fails.

### `issues` — new kinds

Two new issue kinds: `misplaced_concern` and `incoherent_module`. Same `severity`, `confidence`, `anchors` structure as existing issues.

`misplaced_concern` issues carry additional structured fields for machine consumption:
```json
{
  "kind": "misplaced_concern",
  "severity": "high",
  "suggested_module": "auth",
  "similarity_own": 0.21,
  "similarity_best": 0.78,
  "anchors": ["src/payments/validators.rs:42"]
}
```

### Output root — add semantic flag

```json
{
  "semantic_enabled": true
}
```

Boolean flag so consumers know whether semantic analysis was attempted and passed the quality gate. `false` when `--semantic` was not passed or the gate failed.

---

## What Was Dropped and Why

### Procrustes Analysis
Dimensionality mismatch (structural k≈5-20 vs semantic 768). Violated assumptions (the two spaces are not relatable by rotation). Superseded by CCA if alignment is ever needed.

### Per-Node GFT Reconstruction
Requires top-k eigenvectors we don't compute and conflates signal magnitude with spectral character. Local variation (tool #4) replaces it — direct per-node disagreement measurement.

### Role Semantic Fit
Measuring each node's similarity to the centroid of its role category (all hubs, all bridges). Circular: bridge nodes connect different domains by definition, so the centroid of bridge embeddings is unrepresentative.

### RMT Null Model (Marchenko-Pastur)
Wrong null model for graph Laplacians. The correct null is the configuration model (random graphs preserving degree sequence). Significant implementation effort for marginal gain given modularity Q and silhouette already measure significance.

### Learned Margins / Interferometer
From the Phase 3 design discussion. The learned per-node disagreement margin is conceptually sound but requires training infrastructure. In Phase 2, the simpler approach (local variation + misplaced_concern detection from centroid distances) captures the same signal without any learning. If Phase 2's false positive rate is too high, learned margins become the motivation for Phase 3.

### Graph Diffusion / Structural Inpainting
"AlphaFold for code architecture" — generative model that predicts what structure "should" be in a masked region. Compelling but requires training a diffusion model on code graphs. Phase 4+ at earliest.

---

## Integration Architecture

Semantic embeddings flow through the pipeline:

1. **Parser** outputs `CodeGraph` with source spans (`line_end` / byte range) on each node.
2. **CLI** (when `--semantic`) reads source text per node, assembles context windows, runs inference via `fastembed-rs`, produces `HashMap<String, Vec<f32>>`.
3. **Analyzer** receives embeddings as optional input (`semantic_embeddings: Option<HashMap<String, Vec<f32>>>` on `AnalyzerInput`). All semantic tools are skipped if `None`.
4. **Formatter** renders semantic fields only when present (`skip_serializing_if`). New `--format=context` and `--format=domain` formatters.

This keeps the analyzer embedding-agnostic. The CLI is the only component that knows about `fastembed-rs`. The WASM build and Python bindings accept pre-computed embeddings from any source.

---

## What This Does Not Change

The existing structural analysis pipeline is untouched. Spectral decomposition, module detection (Ng-Jordan-Weiss), role classification, existing anomaly detection — all remain as-is. The semantic layer is additive. Without `--semantic`, the output is identical to today's.

The `top_terms` feature (node ID tokenization) is available without the embedding model and without the feature flag. Zero dependencies.

---

## Validation

Before shipping any semantic finding, validate on four codebases:

1. **topo itself** — known architecture, monorepo with clear package boundaries. Well-structured baseline.
2. **Flask** — well-documented single-package library with known architectural intent.
3. **Click** — small, clean library. Known structure.
4. **A known-messy codebase** — legacy project with documented architectural debt (e.g., a Django project with known tech debt issues, or a monolith with acknowledged boundary violations). The highest-value findings are most likely to appear in messy codebases; validating only on clean projects risks proving the tool works when there's nothing to find.

**Quantitative protocol:**
1. For each codebase, manually label 20-30 entities as "correctly placed" or "misplaced" (ground truth).
2. Run the `misplaced_concern` detector. Count true positives, false positives, false negatives.
3. Compute precision and recall.

**Thresholds:**
- **<50% precision** → do not ship semantic findings. The signal is insufficient.
- **50-80% precision** → ship behind `--semantic` with caveats. Build Phase 3 (R-GIN) for calibration.
- **>80% precision** → ship as default when model is available. Phase 3 is an optimization, not a necessity.

For each codebase, also verify:
1. Does the signal quality gate pass?
2. Do per-module coherence scores correlate with developer judgment?
3. Does the Rayleigh quotient distinguish well-structured from poorly-structured regions?
4. Does the GFT energy profile show meaningful shape?

If validation fails: do not ship semantic findings. The structural analysis alone carries the load.

---

## Implementation Sequence

### Step 1: Graph contract extension
Add `line_end: Option<u32>` (or `byte_start`/`byte_end`) to `NodeEntry` in both parsers. Both already compute this internally — persist it.

### Step 2: `top_terms` (no model needed)
Tokenize node IDs, compute TF-IDF across modules, attach top-3 terms per module. ~120 lines. No external dependencies.

**Checkpoint.** Run on topo/Flask/Click. Do top_terms accurately characterize modules?

### Step 3: `fastembed` integration + embedding cache
Add `fastembed` behind `semantic` feature flag. Wire `--semantic` CLI flag. Implement embedding input assembly (context window from source spans). Batched inference. Implement `topo model download` and `topo model list`. Implement embedding cache (blake3 hashing, `.topo/embeddings.cache` file I/O, invalidation logic). Handle runtime failures gracefully (incompatible CPU, memory pressure → fall back to structural-only with warning). ~300 lines.

**Checkpoint.** Run `topo analyze . --semantic --json` on topo. Verify non-null 768-dim vectors per node. Verify cache hit on re-run.

### Step 4: Semantic coherence + signal quality gate
Per-module coherence. Quick pre-checks (variance, mean similarity). Permutation test (N=200, α=0.05). ~150 lines.

**Checkpoint.** Do coherence scores correlate with developer judgment?

### Step 5: Rayleigh quotient + GFT energy profile
Global smoothness on normalized Laplacian (one matmul). Eigenvector computation already produces 15-20 regardless of eigengap. Project semantic signal onto eigenvectors. Upcast f32 embeddings to f64 for numerical stability. ~100 lines.

### Step 6: Local variation
Per-node cosine-distance-based local variation, degree-normalized. ~60 lines.

### Step 7: Issue detection
`misplaced_concern` with all filters (6 filters including nearest-neighbor check). `incoherent_module` from coherence scores with semantic sub-cluster identification. ~180 lines.

### Step 8: AMI
Spherical k-means on L2-normalized embeddings (separate from standard k-means — uses cosine similarity, centroid re-normalization). AMI computation (contingency table, marginal entropies, expected MI). Sweep k = structural_k ± 2. ~200 lines.

### Step 9: Schema + formatter
Wire all new fields into output types (`semantic_coherence`, `top_terms`, `semantic_enabled`, `suggested_module`, `similarity_own`, `similarity_best`, etc.). Update formatter for new issue kinds. ~100 lines.

### Step 10: `--format=context` (LLM narrative)
New formatter producing the 2-4K token structural narrative. Compression logic (prioritization algorithm: all modules get summary line, only flagged modules get detail, high+medium issues get cards, low issues counted). ~200 lines.

### Step 11: `--format=domain` (bounded context map)
Domain model approximation output. Bounded contexts from coherent modules, context map from typed bridges, aggregate root candidates. Prominently labeled as "approximate — requires human validation." ~150 lines.

### Step 12: `topo health` (Fiedler tracking)
Git history walker (libgit2 or `git` subprocess), worktree/stash management for dirty working trees, sampled parse per commit (weekly/per-PR/per-N-commits), progress reporting, error handling for non-compiling commits. ~400 lines.

**Total: ~2,010 lines of new Rust code** across topo-analyzer, topo-cli, topo-formatter, and both parsers. Steps 3, 8, and 12 are the heaviest; the pure-math steps (5, 6) are lightweight.

---

## Where Phase 2 Hits Its Ceiling

Phase 2 gives you structural-semantic comparison using mathematical tools on frozen embeddings. It cannot:

- **Calibrate disagreement per structural role.** A utility function legitimately has a different structural-semantic profile than a domain entity. Phase 2 treats all nodes equally. Phase 3's R-GIN learns role-specific baselines from training on 500+ repos.
- **See multi-hop context.** Local variation measures 1-hop neighbors. A node that's semantically anomalous only when you consider its 3-hop neighborhood (e.g., it bridges two semantically different communities) requires the GIN's multi-hop message passing.
- **Transfer structural judgment across repos.** Phase 2's spectral coordinates are per-graph. Each codebase is analyzed from scratch with no knowledge of what "misplaced" looked like in other codebases. Phase 3's trained model has seen structural patterns across hundreds of repos.
- **Decompose structural embeddings by edge type.** Phase 2 uses the combined-layer spectral coordinates. Phase 2 *could* compute local variation separately per edge type (call-neighbors vs import-neighbors), and this is worth trying before Phase 3. But full per-layer spectral decomposition and per-layer module detection require Phase 3's R-GIN, which produces separate z_calls, z_imports, z_inherits components natively.

These limitations define Phase 3's scope. Phase 2 is the foundation; Phase 3 is the refinement. Phase 2 should be fully validated before Phase 3 is attempted — if Phase 2's precision is already >80%, Phase 3's incremental value may not justify its training cost. Between 50-80%, Phase 3 is justified. Below 50%, fix Phase 2 before building Phase 3.
