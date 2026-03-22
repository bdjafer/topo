# Domains

Hierarchical domain decomposition of a codebase. Extracts a tree-structured domain model from the dependency graph — bounded contexts at the top, progressively finer sub-domains at each level, down to individual aggregates and entities.

**Output:** A rooted tree where each node is a structural domain with label, size, coherence, health, and typed dependencies to siblings. Cross-cutting nodes are flagged separately.

---

## What Flat Decomposition Cannot See

The current pipeline produces K flat modules. A 500-node codebase becomes 5-8 clusters. This is the right answer for "what are the top-level architectural blocks?" It is the wrong answer for:

- **Internal structure.** A 60-node "orders" module has sub-domains: order management, fulfillment, cart logic. Flat decomposition collapses them.
- **Scale-appropriate reasoning.** An LLM reading 8 module summaries can't reason about whether `cart_validator` belongs in the orders module vs. a shared validation concern. It needs to see the internal structure where the question lives.
- **Localized health.** A module can look healthy at the top level (good cohesion, few cross-module edges) but be internally tangled (fulfillment and cart logic are cyclically coupled). Per-level health reveals this.
- **Progressive onboarding.** A developer new to the codebase needs zoom levels: "there are 4 subsystems" → "orders has 3 sub-domains" → "fulfillment has a payment integration concern and a shipping concern."

The hierarchy answers all of these. Flat decomposition answers none.

---

## Why Not Spectral Sign Backbone

The previous design built the hierarchy from the sign patterns of Laplacian eigenvectors v₂, v₃, ..., vₖ. Three independent critiques converged on fatal flaws:

1. **Global eigenvectors lose meaning at local scales.** v₅ for a 15-node subtree encodes global structure, not internal structure. The sign of v₅(node) tells you which side of the *global* fifth-coarsest partition the node falls on — not which side of an internal split within its subtree. Using global eigenvectors at local scales is mathematically incoherent. It worked for depth 1 (the Fiedler cut is genuinely the coarsest split) but degraded at every subsequent level.

2. **Phase 3's z_invariant is never used.** The spectral sign backbone was designed for Phase 1, where eigenvectors were the only structural embedding. Phase 3 produces z_invariant (64d), a cross-layer structural role embedding trained on 500+ repos. It encodes structural position more richly than any finite set of eigenvector signs. The old design ignored it entirely.

3. **Per-subtree THS doesn't work.** The old design claimed "THS at every tree node." But the R-GIN's reconstruction_error is computed with full-graph context — the 2-layer GIN uses the entire graph's message-passing neighborhood. Restricting to a subtree changes the computation fundamentally. Reconstruction errors computed on a subgraph are not comparable to those computed on the full graph.

The new design uses z_invariant as the primary clustering signal, with a divisive (top-down) approach that avoids re-running the R-GIN.

---

## The Decomposition: z_invariant Divisive Clustering

### Primary Signal: z_invariant (64d)

z_invariant is the cross-layer structural role embedding. It is derived from the R-GIN's post-aggregation hidden state, which combines information from all three dependency layers (calls, imports, inherits) through 2 layers of message passing, augmented with spectral PEs, RWPE, tree position, and node type. The HSIC decorrelation loss trains it to capture what is *shared* across layers — the structural role that is consistent regardless of which dependency type you examine.

**Why z_invariant and not z_calls or z_imports:** Per-layer embeddings capture layer-specific coupling. Domain boundaries are defined by where *all forms* of coupling attenuate simultaneously. A cluster of nodes tightly coupled in calls but scattered in imports is a runtime concern, not a domain. z_invariant captures the cross-layer consensus, which is the right signal for domain boundaries.

**Why z_invariant and not spectral coordinates:** Spectral coordinates are per-graph, uncalibrated, and limited by the number of eigenvectors computed. z_invariant is 64d (vs. typically 16d spectral), trained across 500+ repos (calibrated baseline for what "similar structural role" means), and encodes multi-hop, multi-layer structural position. It strictly subsumes the structural information in spectral coordinates while adding cross-repo calibration.

### Algorithm: Bisecting k-Means in z_invariant Space

Divisive (top-down) clustering. Start with the full node set. At each step, select the leaf with the largest cluster-quality improvement from splitting, and bisect it.

**Step 1: L2-normalize.** Normalize all z_invariant vectors to unit length. After normalization, Euclidean distance is monotonically related to cosine distance: `‖a - b‖² = 2(1 - cos(a,b))`. This means standard k-means on L2-normalized vectors produces cosine-optimal clusters (spherical k-means).

**Step 2: Try k=2, 3, 4.** For a cluster C with members {v₁, ..., vₙ}, run k-means with k=2, k=3, and k=4 on the normalized z_invariant vectors. Use k-means++ initialization, 10 restarts per k, take the best by inertia for each k. Evaluate ALL candidates by modularity Q on the *induced subgraph* of the original dependency graph (not on z_invariant distances — Q uses actual edges). Accept the k with the highest Q. This uses a single consistent criterion for all k values, and allows natural 3-way and 4-way splits without forcing binary trees.

**Step 3: Evaluate split quality.** Compute two signals:

- **ΔQ:** Modularity improvement. Q(children as separate modules in the induced subgraph) - Q(parent as one module). If ΔQ < 0.02, the split does not improve structural partition quality.
- **Silhouette in z_invariant space:** Mean silhouette coefficient of the proposed children using cosine distance on L2-normalized z_invariant. If silhouette < 0.1, the children are not well-separated in the learned structural space. (Standard silhouette interpretation: < 0.25 = no substantial structure. The 0.1 threshold is deliberately low because the Q-gate is the primary criterion — silhouette serves as a floor to reject splits where Q is positive but the embedding shows no geometric separation.)

Accept the split only if ΔQ ≥ 0.02 AND silhouette ≥ 0.1.

**Step 4: Select next split.** Among all current leaves that haven't been rejected, select the one with the highest ΔQ for the next split. This greedy selection ensures the most structurally meaningful splits happen first.

**Step 5: Repeat** until no leaf passes the stopping criteria.

### Why Divisive, Not Agglomerative

Agglomerative (bottom-up) clustering in 64d space is O(n² log n) for n nodes — each step considers all pairwise distances. For a 5000-node codebase this is ~25M distance computations per merge step. Divisive clustering is O(n·k·d·restarts) per split, where k≤4, d=64, restarts=10 — about 2500 operations per node per split, with at most O(n) splits total.

More importantly, divisive clustering produces a *tree* naturally — each split creates a parent-children relationship. Agglomerative clustering produces a dendrogram that must be cut at arbitrary thresholds to produce a tree, reintroducing the stopping-criterion problem.

Finally, divisive clustering matches the user's mental model: "what are the big pieces?" → "what's inside each piece?" Top-down decomposition answers questions in the order developers ask them.

### Why Bisecting k-Means, Not Hierarchical Density Clustering (HDBSCAN)

HDBSCAN is excellent for discovering clusters of varying density. But code graphs have roughly uniform local density in z_invariant space (the R-GIN's training distributes structural roles across the embedding space without extreme density variation). The advantage of HDBSCAN — handling density variation — doesn't apply. The disadvantage — noise points that belong to no cluster — is unacceptable (every node must be assigned to a domain).

k-Means with the Q-gate on the *actual graph edges* combines the geometric intuition of z_invariant distance with the ground-truth structural signal of edge connectivity. The embedding proposes, the graph disposes.

### Distance Metric: Cosine on z_invariant

Cosine distance, not Euclidean. z_invariant vectors are produced by a linear projection from 256d hidden states. The direction in the 64d space encodes structural role; the magnitude encodes confidence (nodes with rich structural context produce larger-magnitude embeddings). Cosine distance compares role similarity independent of confidence, which is what domain membership requires.

**Why not Euclidean:** Euclidean distance in high-dimensional space is dominated by magnitude differences. Two nodes with identical structural roles but different confidence levels (one in a dense subgraph, one in a sparse subgraph) would appear distant under Euclidean but identical under cosine.

### Why This Avoids the "Global Eigenvector at Local Scale" Problem

The spectral sign backbone failed because it applied global eigenvectors to local subtrees. The z_invariant approach avoids this entirely: z_invariant is a *per-node* embedding, not a global decomposition. When we bisect a subtree, we run k-means on the z_invariant vectors of the subtree's members. These vectors encode each node's structural role in the full graph — they don't change when we restrict our attention to a subtree. We're not recomputing a decomposition on a subgraph; we're partitioning existing embeddings that already encode full-graph structural position.

The Q-gate on the induced subgraph provides the local validation: does the proposed split correspond to a real structural boundary in the *local* edge structure? This is the correct combination: global role embeddings for proposing splits, local edge structure for validating them.

---

## Stopping Criteria

The recursion stops (leaf node) when any of these hold:

| Criterion | Threshold | Rationale |
|---|---|---|
| Cluster size | `max(5, sqrt(N) / 2)` | Scale-dependent minimum. Too-small clusters are noise. |
| Split quality | ΔQ < 0.02 on induced subgraph | The split doesn't improve structural partition quality. |
| Embedding separation | Silhouette < 0.1 in z_invariant space | The children are not well-separated in the learned structural space. |
| Tree depth | `ceil(log₂(N / min_size)) + 2` | Hard ceiling to prevent pathological recursion. The +2 allows multi-way splits to go slightly deeper than a pure binary tree. |

### Why ΔQ and Silhouette Together

Either criterion alone is insufficient:

- **ΔQ alone** can accept splits where the graph has a weak boundary but the structural roles are identical — a single bridge edge between two halves of a semantically homogeneous module. The silhouette check rejects these: if z_invariant vectors are intermixed, the split is not a real domain boundary.

- **Silhouette alone** can accept splits where the embeddings separate but the graph doesn't — two groups with different structural roles that are actually tightly coupled by edges. The ΔQ check rejects these: if the induced subgraph has more cross-group edges than within-group edges, the split is structurally wrong.

The conjunction requires both geometric separation in the learned space AND structural separation in the actual graph.

### Why Not Reconstruction Error as a Stopping Criterion

The old design considered THS-per-subtree. The critique correctly identified that reconstruction_error is computed with full-graph context and is not meaningful on subgraphs. Additionally, reconstruction_error measures structural-semantic *agreement*, not the *absence of internal structure*. A cluster with perfect reconstruction_error (every node's structure predicts its semantics) can still have meaningful internal domain boundaries — authentication and authorization have similar vocabulary and similar structural positions, but are distinct sub-domains.

Reconstruction_error remains the coherence annotation signal (see Health at Every Level below), not the stopping signal.

---

## Cross-Cutting Code

Every codebase has shared utilities, common types, configuration, logging. These nodes connect to everything.

### Detection

A node v is cross-cutting if both conditions hold:

1. **High caller diversity.** Edges from > 30% of top-level domains (computed after the first bisection produces depth-1 domains). This is a structural fact — the node serves many masters.

2. **Low z_invariant cluster affinity.** The node's silhouette coefficient in z_invariant space is negative (it is closer to its nearest non-assigned cluster centroid than to its assigned cluster centroid). This means the learned structural role embedding does not place it clearly in any domain.

### How z_invariant Handles Cross-Cutting Code

Cross-cutting nodes have a distinctive z_invariant signature. The R-GIN has seen many loggers, serializers, and utility functions across 500+ training repos. These nodes have a common structural pattern: high in-degree from diverse modules, leaf-like in the import graph, high RWPE return probabilities (short cycles through many callers). The R-GIN encodes this as a recognizable region of z_invariant space.

Three outcomes are possible:

1. **Cross-cutting nodes cluster together.** The z_invariant embeddings for utilities, loggers, and shared types form their own cluster at the first bisection. This is the ideal outcome — the algorithm naturally separates infrastructure from domain code. Detection confirms and extracts them.

2. **Cross-cutting nodes scatter across domains.** Each utility node gets pulled into the domain that calls it most. The silhouette check catches this — scattered utilities have negative silhouette because they sit between clusters, not inside them. Detection extracts them post-hoc.

3. **Cross-cutting nodes contaminate a single domain.** All utilities get absorbed into one domain, making it a grab-bag. The Q-gate catches this — a domain with utilities and real domain code has low internal Q because the utilities don't connect to each other. The split will separate them at the next level.

**Handling — two-pass approach:**

1. Run depth-1 clustering on ALL nodes (including potential cross-cutting nodes).
2. Detect cross-cutting nodes using the criteria above.
3. Extract cross-cutting nodes from the node set.
4. Re-run depth-1 clustering on the remaining nodes (without the cross-cutting contamination).

The re-run in step 4 is cheap (~2x the cost of one clustering pass) and eliminates the contamination problem: utilities that connect to everything no longer distort cluster boundaries.

Cross-cutting nodes appear in the output as a flat set at the root level, annotated with their dominant edges. They are NOT recursively decomposed. This matches how DDD treats shared kernel — it's not a bounded context.

**Consistency with diagnostics:** Cross-cutting nodes that are also overloaded-utilities (from the `overloaded-utility` diagnostic) get their existing diagnostic. The domain tree simply doesn't try to place them in a domain, which is the honest answer.

---

## Health at Every Level

### What We Can Compute Per Subtree

The critique correctly showed that reconstruction_error cannot be recomputed per subtree — the R-GIN's message passing uses the full graph. But reconstruction_error *values* (computed once on the full graph) can be *aggregated* per subtree. This is meaningful and avoids the recomputation problem.

**Per-domain coherence:**

```
coherence(D) = clamp(1 - median({reconstruction_error(v) : v ∈ D}), 0, 1)
```

This is the median reconstruction_error of the domain's members, using the values already computed by the full-graph R-GIN pass. It measures: "how well does the full-graph structural context predict the semantics of nodes in this domain?" A domain where most members have low reconstruction_error is coherent — its members are in positions that make structural sense. A domain where many members have high reconstruction_error is incoherent — its members are misplaced.

**Why aggregating full-graph errors is valid:** The reconstruction_error of each node is a property of that node's position in the full graph. It doesn't change when we group nodes into domains. Aggregating these fixed per-node values within a domain measures the *typical structural correctness* of nodes placed in that domain. This is exactly what per-domain coherence should measure: are the nodes in this domain structurally well-placed?

**What it does NOT measure:** Internal structural quality of the subgraph. A domain could have low median reconstruction_error (all members are well-placed in the full graph) but be internally tangled (the members form cycles among themselves). Internal structural quality requires different signals.

**Per-domain flow (on induced subgraph):**

```
flow(D) = cycle_freedom(induced_subgraph(D)) × layer_conformance(induced_subgraph(D))
```

- **cycle_freedom** on the induced subgraph: `1 - (nodes_in_nontrivial_SCCs / total_nodes_in_D)`. Tarjan's SCCs on the subgraph induced by D's members. This is a pure graph-theoretic computation on the subgraph — no R-GIN context needed. Cycles within a domain are real cycles; the subgraph faithfully represents internal circular dependencies.

- **layer_conformance** on the induced subgraph: For Phase 3, use direction_surprise on internal edges. The z_imports embeddings and R matrix are computed on the full graph, but direction_surprise is a per-edge property: `σ(z_imports(v)ᵀ R z_imports(u)) - σ(z_imports(u)ᵀ R z_imports(v))`. Each internal edge's direction_surprise is a fixed value from the full-graph computation. Aggregating these over the induced subgraph's edges gives a valid per-domain layer_conformance.

**Per-domain health:**

```
THS(D) = coherence(D)^α × flow(D)^(1-α)
```

Same formula as the global health score, same α (starting hypothesis: 0.7, calibrated per [HEALTH.md](HEALTH.md)). This is well-defined at every tree node.

### The Monotonicity Invariant

Children should generally have higher coherence than their parent. Splitting should improve local quality by separating nodes with different structural profiles. If a split violates this — children's median reconstruction_error is *higher* than the parent's — the split has mixed structurally well-placed and poorly-placed nodes in a way that doesn't follow domain boundaries. Flag it as low-confidence in the output.

Formally: for a domain D with children D₁, ..., Dₖ, the split is suspect if:

```
mean(coherence(Dᵢ)) < coherence(D) - 0.02
```

The 0.02 tolerance allows for minor fluctuations from the median aggregation. Violations are annotated in the output, not suppressed — the tree is still shown, but the consumer knows that split is weak.

### What This Hierarchy Reveals That Flat Health Cannot

```
system (THS: 0.68)
├── auth (THS: 0.82) ✓
│   ├── token-management (THS: 0.91) ✓
│   └── session (THS: 0.88) ✓
├── billing (THS: 0.71)
│   ├── invoicing (THS: 0.84) ✓
│   └── payment (THS: 0.59) ← problem here
│       flow: 0.41 ← internal cycle between charge and refund
└── orders (THS: 0.54) ← problem here
    ├── order-management (THS: 0.77) ✓
    └── fulfillment (THS: 0.31) ← worst sub-domain
        coherence: 0.38 ← members are structurally misplaced
```

The flat health score says "orders is unhealthy." The hierarchical health says "orders is unhealthy *because* fulfillment has severe coherence problems — the code in fulfillment doesn't belong together structurally."

---

## Intra-Domain Semantic Similarity as a Secondary Coherence Signal

When Phase 3 is available, coherence uses reconstruction_error as defined above. As a secondary signal for annotation (not for stopping or splitting), compute intra-cluster semantic similarity:

```
sem_similarity(D) = mean({cos(e_sem(u), e_sem(v)) : u, v ∈ D, u ≠ v})
```

Where e_sem is the frozen 768d CodeLM embedding. This measures whether the domain's members *do similar things* — a purely semantic question independent of graph structure.

**Why secondary, not primary:** Semantic similarity measures vocabulary overlap. Authentication and authorization have high semantic similarity but are distinct sub-domains. Payment processing and shipping have low semantic similarity but may belong in the same "fulfillment" domain if structurally coupled. Structure (z_invariant + edges) is the right primary signal for domain boundaries. Semantics annotates the result.

**Disagreement between coherence and sem_similarity is informative:**

| coherence high, sem_similarity high | Domain is well-placed and semantically focused. Ideal. |
|---|---|
| coherence high, sem_similarity low | Structurally sound coupling between semantically diverse code. Common in infrastructure domains. |
| coherence low, sem_similarity high | Code that *should* be together (similar semantics) but *isn't* structurally well-placed. Likely misplaced concerns. |
| coherence low, sem_similarity low | Grab-bag. The domain is neither structurally nor semantically justified. Strong split signal. |

---

## The DDD Question

### What the Old Document Got Right

The old DOMAINS.md was cautious about DDD labels:

> "DDD bounded contexts are semantic: defined by where the same word means different things. Graph topology reveals coupling structure, not semantic boundaries. These correlate but are not equivalent."

This remains true. The tool should not label depth-1 as "bounded contexts" because the correspondence is approximate.

### What Phase 3 Changes

Phase 3's cross-repo calibration changes the calculus. The R-GIN has seen 500+ codebases. z_invariant embeddings are trained to place structurally similar nodes close together *across repos*. If auth-related clusters in 500 different codebases all land in the same region of z_invariant space, and billing clusters land in a different region, then z_invariant has learned something that looks like domain recognition.

**The empirical question:** After training, cluster the z_invariant centroids of all depth-1 domains across all 500+ training repos. Do recognizable patterns emerge? If the centroids for authentication-related domains form a tight cluster, and billing-related domains form another, and data-access domains form a third, then the model has implicitly learned domain archetypes.

**If the answer is yes:** The tool can make *advisory* assertions:

```
auth (THS: 0.82)
  archetype: authentication/authorization (0.89 similarity to training centroid)
  ├── token-management
  └── session
```

This is not "this IS an auth bounded context" — it is "this domain has the same structural signature as auth domains in 500 repos." The developer confirms or corrects. The archetype similarity score (cosine distance to the nearest training centroid) quantifies confidence.

**If the answer is no:** Domain archetypes are too variable across codebases to form stable clusters. The tool falls back to TF-IDF labels (which are always computed regardless).

### Implementation: Domain Archetype Matching

Post-training, for each training repo:

1. Run the full pipeline (parse → Phase 1 → Phase 2 → Phase 3 → domain decomposition).
2. Compute the z_invariant centroid for each depth-1 domain.
3. Manually label 50-100 domains with DDD-style labels (auth, billing, persistence, API, orchestration, etc.).
4. Fit a k-NN classifier from z_invariant centroids to labels.

At inference time, compute the z_invariant centroid for each depth-1 domain and classify via k-NN. Report the label and the distance to the nearest training centroid. Distances above a threshold (calibrated against the training distribution) produce "no archetype match" — the tool honestly says it doesn't recognize the pattern.

**Cost:** ~100 lines. A lookup table of ~200 labeled centroids (200 × 64 floats = 50KB) ships with the model bundle.

**Why this is bolder than the old approach but not reckless:** The assertion is statistical (based on 500 training examples), quantified (similarity score), and advisory (annotated, not asserted as ground truth). The developer sees "this looks like auth to the model" and can agree or disagree. False positives are visible and correctable. This is the right level of boldness — strong enough to be useful, qualified enough to not be wrong in a way that matters.

---

## Labeling

Labels at each level use TF-IDF term extraction from node IDs, computed per sibling set.

**Within-level TF-IDF:** At depth 1, terms are discriminative against other depth-1 modules. "auth" is a good label because it distinguishes this module from "billing" and "orders." At depth 2 within auth, terms are discriminative against other depth-2 children of auth. "token" and "session" are good labels because they distinguish the two sub-domains.

**Parent term filtering:** Suppress terms that already appear in the parent's label. If the parent is "auth," don't label a child "auth-token" — label it "token." The parent context is implicit in the tree path.

**Full path labels:** For cross-level references (dependencies, diagnostics), use path-style labels: `auth/token-management`, `orders/fulfillment`. These are both human-readable and machine-parseable.

**Domain archetype label (Phase 3 only):** If archetype matching is available and confidence exceeds the threshold, the archetype label is shown alongside the TF-IDF label:

```
auth [authentication/authorization] (THS: 0.82)
```

The bracketed label is the archetype; the unbracketed label is from TF-IDF. They may differ — TF-IDF says "user-session" while the archetype says "authentication/authorization." Both are shown. The developer resolves.

---

## Stability

### Why z_invariant Is More Stable Than Spectral Signs

Spectral sign patterns are binary thresholdings of eigenvectors. A node with eigenvector component v₅(node) = +0.001 is on the "positive" side. Add one edge and v₅(node) = -0.001 — it flips. The sign pattern changes, the tree changes. Nodes near eigenvector zero-crossings are maximally unstable.

z_invariant is a 64d continuous embedding. A minor code change (one added edge) propagates through the R-GIN's 2-layer message passing, shifting z_invariant by a small perturbation proportional to the edge's influence on the 2-hop neighborhood. The perturbation is bounded:

- **Layer 1:** The new edge affects neighbors of the endpoints. The GIN's sum aggregation means adding one message to a sum of k messages changes the output by O(1/k). For a node with 20 neighbors, the shift is ~5% of the hidden state norm.
- **Layer 2:** The perturbation propagates to 2-hop neighbors, attenuated by another factor of O(1/k). For typical code graphs (mean degree 5-10), the 2-hop perturbation is <1% of the hidden state norm.
- **Projection:** The linear projection W_inv (256d → 64d) is a contraction — it cannot amplify the perturbation.

The resulting z_invariant shift is small for small graph changes. k-Means on slightly-shifted 64d vectors produces the same clustering unless a node is on the Voronoi boundary between two cluster centroids. This is analogous to the eigenvector zero-crossing instability, but in 64d space the Voronoi boundaries have measure zero (they are 63d hyperplanes in 64d space), whereas in 1d sign space the boundary is a single point that every node's trajectory crosses.

### Stability Guarantees

**Determinism:** For a given R-GIN model and input graph, z_invariant is deterministic. No random initialization, no stochastic sampling. The only randomness is in k-means initialization (mitigated by 10 restarts + best-of selection).

**Lipschitz continuity (informal):** Small changes to the input graph produce small changes to z_invariant. The R-GIN's message passing is Lipschitz in the input features (ReLU + sum aggregation + linear projection are all Lipschitz operations). This doesn't guarantee cluster stability, but it guarantees that cluster *instability* requires a genuine structural change, not a numerical artifact.

**Practical stability test:** Same codebase, two commits one day apart. Compute domain tree at both commits. Measure overlap: for each depth level, compute the Adjusted Rand Index (ARI) between the two partitions. Target: ARI > 0.85 for minor changes (< 5% of nodes modified).

### Mitigation for Boundary Nodes

At each split, record each node's *margin*: the ratio of its distance to the second-nearest centroid vs. the nearest centroid. Nodes with margin < 1.2 (less than 20% closer to their assigned centroid than to the alternative) are boundary nodes. Annotate them in the output. The consumer knows these assignments may shift under minor changes.

---

## Output

### Tree Structure

```json
{
  "domain": {
    "label": "system",
    "depth": 0,
    "size": 847,
    "members": [],
    "top_terms": [],
    "coherence": null,
    "archetype": null,
    "health": null,
    "children": [
      {
        "label": "auth",
        "depth": 1,
        "size": 22,
        "members": ["auth.login", "auth.logout", "..."],
        "top_terms": ["token", "session", "authenticate"],
        "coherence": 0.85,
        "archetype": {
          "label": "authentication/authorization",
          "confidence": 0.89
        },
        "health": {
          "topo_health_score": 0.82,
          "coherence": 0.85,
          "flow": 0.79
        },
        "children": [
          {
            "label": "token-management",
            "depth": 2,
            "size": 10,
            "members": ["auth.jwt_validate", "auth.refresh_token", "..."],
            "top_terms": ["jwt", "validate", "refresh"],
            "coherence": 0.91,
            "archetype": null,
            "health": {
              "topo_health_score": 0.91,
              "coherence": 0.91,
              "flow": 0.92
            },
            "children": [],
            "boundary_nodes": ["auth.token_cache"]
          },
          {
            "label": "session",
            "depth": 2,
            "size": 12,
            "members": ["auth.session_create", "auth.cookie_store", "..."],
            "top_terms": ["session", "cookie", "store"],
            "coherence": 0.88,
            "archetype": null,
            "health": {
              "topo_health_score": 0.88,
              "coherence": 0.88,
              "flow": 0.88
            },
            "children": [],
            "boundary_nodes": []
          }
        ],
        "boundary_nodes": []
      },
      {
        "label": "billing",
        "depth": 1,
        "size": 45,
        "members": [],
        "top_terms": ["payment", "invoice", "charge"],
        "coherence": 0.71,
        "archetype": {
          "label": "billing/payments",
          "confidence": 0.76
        },
        "health": {
          "topo_health_score": 0.71,
          "coherence": 0.71,
          "flow": 0.71
        },
        "children": ["..."],
        "boundary_nodes": ["billing.shared_formatter"]
      }
    ],
    "cross_cutting": [
      {
        "node_id": "utils.format_response",
        "silhouette": -0.12,
        "caller_diversity": 0.75,
        "dominant_edges": {"auth": 4, "billing": 6, "orders": 8}
      }
    ],
    "dependencies": [
      {
        "source_path": "auth/token-management",
        "target_path": "auth/session",
        "weight": 4,
        "edge_kinds": {"calls": 3, "imports": 1}
      },
      {
        "source_path": "orders",
        "target_path": "billing/payment",
        "weight": 8,
        "edge_kinds": {"calls": 8}
      }
    ]
  }
}
```

### Key Design Decisions

**`members` at internal nodes are empty.** Internal nodes are structural groupings; their members are the union of their children's members. Listing members at every level would be redundant and expensive. Internal nodes have `size` (sum of children) for quick reference.

**`dependencies` use path references** (e.g., `"auth/token-management"`) rather than numeric IDs. This makes the output self-documenting and stable across runs. Dependencies are aggregated at the highest common level: a call from `auth.jwt_validate` (in `auth/token-management`) to `billing.charge` (in `billing/payment`) appears as a dependency from `auth/token-management` to `billing/payment`.

**`cross_cutting` is a flat list at the root.** Cross-cutting nodes don't belong in the tree. They're infrastructure. The `dominant_edges` field lets the consumer understand which domains depend on each cross-cutting node.

**`boundary_nodes` per domain.** Nodes with margin < 1.2 are listed per domain. This makes assignment uncertainty explicit. An LLM reading the output knows which nodes might move under refactoring.

**`archetype` is nullable.** Only present when Phase 3 is available and confidence exceeds the threshold. Null means "no archetype match" — honest absence, not a missing field.

**`health` per domain.** Full THS decomposition (coherence + flow) at every tree node. Computed from full-graph reconstruction_error values (aggregated) and induced-subgraph flow signals (recomputed). See "Health at Every Level."

### Compatibility with Flat Output

The domain tree does NOT replace the flat `architecture.modules` output. Both are produced. The flat output is the partition at whatever depth the modularity-Q sweep selects (the "best" single level). The domain tree shows the full multi-level structure.

At each depth level, the tree's partition covers all non-cross-cutting nodes (cross-cutting nodes are extracted at the root and excluded from the tree). This means `topo health` can track modularity Q at each depth, and the diagnostic system can scope issues to any tree level.

---

## Fallback Without Phase 3

When only Phase 1/2 signals are available, the hierarchy uses a degraded but still meaningful approach.

### Phase 2 Available (Semantic Embeddings, Spectral Coordinates)

**Primary signal:** Spectral coordinates (the first k eigenvectors of the normalized Laplacian), typically 8-16 dimensions. Run bisecting k-means on spectral coordinates instead of z_invariant.

**Stopping criteria:** Same as Phase 3 (ΔQ ≥ 0.02 AND silhouette ≥ 0.05), but silhouette is computed in spectral coordinate space.

**Cross-cutting detection:** Nodes with eigenvector magnitude below 0.1 × median(|vᵢ|) for >50% of eigenvectors AND caller_diversity > 0.3. This is the spectral analog of the z_invariant silhouette check — nodes near the origin in spectral space belong to no community.

**Health per domain:** Coherence uses semantic similarity (mean pairwise cosine similarity of CodeLM embeddings within the domain) instead of reconstruction_error aggregation. Flow uses cycle_freedom on the induced subgraph with Phase 1's binary layer violation counting (no direction_surprise without the R matrix).

**Labeling:** TF-IDF only. No archetype matching without cross-repo calibration.

**What's lost:**
- Cross-repo calibration. Spectral coordinates are per-graph — "similar structural role" means something different in every codebase.
- Multi-layer decomposition. Spectral coordinates use the combined-layer adjacency. z_invariant's HSIC-decorrelated cross-layer representation is unavailable.
- 64d → typically 8-16d. Less expressive embedding space means more boundary ambiguity.
- Archetype matching. No cross-repo embedding space, no domain archetypes.

### Phase 1 Only (No Semantic Embeddings)

**Primary signal:** Spectral coordinates, same as Phase 2 fallback.

**Health per domain:** No coherence score (requires semantic embeddings). Flow only: cycle_freedom × binary layer_conformance.

**Cross-cutting detection:** Spectral magnitude + caller diversity only. No semantic confirmation.

**This is a significant degradation.** The hierarchy is still useful for progressive onboarding and localized flow analysis, but without semantic signals, coherence is unmeasurable and domain labels are based solely on node identifiers.

---

## Integration with Existing Pipeline

The domain decomposition slots into the pipeline after Phase 3 inference (when available) or after spectral analysis (fallback):

1. **Parser** → CodeGraph
2. **Phase 1: Spectral decomposition** → eigenvalues, eigenvectors, Fiedler value (existing)
3. **Phase 2: Semantic analysis** → CodeLM embeddings, coherence, misplaced-concern (existing)
4. **Phase 3: R-GIN inference** → z_invariant, z_calls, z_imports, z_inherits, reconstruction_error (when available)
5. **Flat clustering** → K modules via z_invariant clustering (Phase 3) or NJW + Q-sweep (Phase 1/2) (existing)
6. **Domain decomposition** → hierarchical tree via bisecting k-means on z_invariant + Q-gate (new)
7. **Cross-cutting extraction** → utility/infrastructure nodes removed from tree (new)
8. **Per-domain health** → THS, coherence, flow at each tree node (new)
9. **Archetype matching** → DDD-style labels from cross-repo centroids (Phase 3 only, new)
10. **Issue detection** → diagnostics scoped to any tree level (existing, extended)
11. **Formatter** → `--format=domain` renders the tree; `--format=context` includes per-level summaries

The domain tree is an additive output. The existing flat output is unchanged. The domain tree adds depth.

### Dependencies on Existing Code

- `spectral::decompose` — eigenvalues and eigenvectors (fallback signal)
- `modules::modularity_q` — Q-gate for split validation
- `modules::module_label` — label generation (adapted for within-level TF-IDF)
- Per-module semantic coherence from `SemanticAnalysis.module_coherence` — semantic similarity annotation (Phase 2+)

### Dependencies on Phase 3

- `z_invariant` per node (64d) — primary clustering signal
- `reconstruction_error` per node — per-domain coherence aggregation
- `z_imports` per node (32d) + R matrix (32×32) — per-domain direction_surprise
- `depth_probe_w` (768d) + `depth_probe_b` (1d) — semantic layer assignment for per-domain flow
- Domain archetype centroids — DDD-style labeling

### New Code

- Bisecting k-means on z_invariant: ~150 lines
- Multi-way refinement (k=2..4 with Q selection): ~80 lines
- Cross-cutting detection (silhouette + caller diversity): ~70 lines
- Per-domain health (coherence aggregation + induced-subgraph flow): ~120 lines
- Boundary node detection (margin computation): ~40 lines
- Domain archetype matching (k-NN on centroids): ~60 lines
- Hierarchical output types: ~100 lines
- Tree-aware label computation: ~60 lines
- Per-level dependency aggregation: ~80 lines
- Phase 1/2 fallback path: ~100 lines
- Formatter updates (`--format=domain` tree rendering): ~150 lines

**Total: ~1010 lines.** More than the old sign-backbone estimate (~800 lines) because the Phase 3 integration (health per domain, archetype matching, fallback paths) is genuine new functionality that the old approach did not have.

---

## Validation

### What Can Be Validated

1. **Depth 1 vs packages.** NMI between depth-1 tree partition and declared package structure. Target: NMI > 0.6 on well-structured codebases. The z_invariant clustering should match or exceed the spectral-coordinates baseline.

2. **Monotonicity.** Children should be more internally coherent than parents. For each split, verify that `mean(child_coherence) >= parent_coherence - 0.02`. Violations indicate the split was noise.

3. **Stability.** Same codebase, two commits one day apart. ARI between domain trees at each depth. Target: ARI > 0.85 for minor changes. Compare against the spectral sign backbone's stability on the same commit pairs — z_invariant should be measurably more stable.

4. **Issue quality.** Does the hierarchical decomposition produce better-scoped diagnostics than flat? If `incoherent-module` fires on a depth-1 module, but the depth-2 decomposition shows the incoherence is localized to one sub-module, the hierarchy is adding value.

5. **Archetype accuracy (Phase 3 only).** For the labeled validation repos, does the archetype classifier correctly identify domain types? Precision > 0.7, recall > 0.5 on a held-out set of manually labeled domains. Low recall is acceptable — it's better to say "no match" than to misclassify.

6. **Developer survey (gold standard).** Show 3+ developers a codebase's tree and ask: "Does this match your mental model?" Target: >70% agreement on depth-1 and depth-2 structure.

### What Cannot Be Validated

There is no objective ground truth for hierarchical decomposition. Developer mental models are subjective, incomplete, and often inconsistent beyond depth 2. The tree is validated indirectly — through issue quality, monotonicity, stability, and package agreement — rather than directly against a "correct" tree that doesn't exist.

### Validation Codebases

Use the same set as Phase 2 validation, extended:
1. **topo itself** — known monorepo with clear package boundaries
2. **Flask** — single-package library with known internal structure
3. **Click** — small, clean library
4. **A known-messy codebase** — legacy project with documented architectural debt
5. **3-4 additional repos from the Phase 3 held-out set** — to test archetype matching generalization

For each, compute the tree and verify: (a) depth-1 matches packages, (b) depth-2 reveals known sub-structures, (c) cross-cutting nodes are correctly identified, (d) the tree is stable across minor git commits, (e) per-domain health identifies known problem areas.

---

## What Was Considered and Rejected

### Spectral sign backbone

The previous design. Rejected because global eigenvectors lose meaning at local scales, Phase 3's z_invariant is ignored, and per-subtree THS requires recomputing the R-GIN. See "Why Not Spectral Sign Backbone" above.

### Recursive eigendecomposition at each level

The subgraph Laplacian loses external edges, changing the degree distribution. Bridge nodes become artificially prominent. Errors compound downward. The z_invariant approach avoids this entirely — it uses full-graph embeddings without recomputation.

### Re-running the R-GIN on induced subgraphs

Conceptually appealing — compute z_invariant on the subgraph for locally meaningful embeddings. Rejected because: (a) the R-GIN's spectral PEs and RWPE are computed on the full graph and would need recomputation on each subgraph, (b) the model was trained on full graphs, not subgraphs — the distribution shift would degrade embedding quality, (c) the computational cost of running the full pipeline (parse → spectral → RWPE → R-GIN) per subtree is prohibitive.

### Agglomerative clustering on z_invariant

O(n²) space for the distance matrix, O(n² log n) time. Produces a dendrogram that must be cut at arbitrary heights. The divisive approach is faster, produces a tree naturally, and matches the top-down mental model.

### HDBSCAN on z_invariant

Produces noise points (nodes in no cluster). Every node must be assigned to a domain. The min_cluster_size parameter introduces the same kind of arbitrary threshold that the design is trying to avoid. k-Means + Q-gate is simpler and handles the universal-assignment requirement naturally.

### Leiden + resolution sweep

Conceptually superior but requires a Leiden implementation in Rust. Deferred as a future upgrade. The z_invariant bisecting k-means approach is implementable now, validated against the Q-gate, and leverages Phase 3's strongest signal. If bisecting k-means quality is insufficient, Leiden + resolution sweep on z_invariant distances is the upgrade path — the same embeddings, better partitioning algorithm.

### Fixed MIN_MODULE_SIZE and MAX_DEPTH constants

Fixed constants produce bad results at scale extremes. A 50-node microservice with MAX_DEPTH=6 gets noise-level splits. A 50K-node monorepo with MIN_SIZE=5 gets thousands of trivial leaves. Scale-dependent formulas adapt automatically.

### Semantic coherence as a stopping criterion

Coherence measures vocabulary overlap, not structural distinction. Authentication and authorization have coherence ~0.9 but are distinct sub-domains. Coherence is for annotation, not for stopping.

### DDD labels on tree levels

"Depth 1 = bounded context" would be wrong whenever coupling crosses domain boundaries. The tree is structural; the domain interpretation is the human's job. Phase 3's archetype matching provides advisory labels with confidence scores — a measured step between "no labels" and "asserting DDD structure."

### Overlapping/soft membership

Real codebases have nodes that belong to multiple domains. A DAG or hypergraph representation was considered but adds complexity without proportional value. The tree + cross-cutting set + boundary_nodes annotation + diagnostics covers the same information: the tree gives the primary assignment, cross-cutting extracts genuinely shared infrastructure, boundary_nodes flags ambiguous assignments, and `misplaced-concern` diagnostics suggest alternative placements.

---

## Future: Leiden + Resolution Sweep on z_invariant

The bisecting k-means approach is the right first step. The long-term approach replaces the splitting algorithm while keeping z_invariant as the signal:

1. Implement Leiden clustering in Rust (or integrate `fa-leiden-cd` crate).
2. Construct a k-NN graph from z_invariant vectors (k=10, cosine distance).
3. Sweep the resolution parameter γ from 0.1 to 5.0 (~30 values).
4. At each γ, compute the Leiden partition and variation of information vs. previous γ.
5. Stable plateaus in the VI landscape define the natural hierarchy levels.

This approach has three advantages over bisecting k-means:
- No arbitrary stopping criteria — the levels emerge from the resolution landscape.
- Handles arbitrary k per level — each level can have any number of children.
- Better boundary handling — Leiden's iterative refinement produces sharper boundaries than k-means.

It requires a Leiden implementation, which is ~300 lines of non-trivial graph algorithm. Deferred until the bisecting k-means approach is validated on real codebases. If bisecting k-means quality is sufficient (validated against developer judgment on 5+ codebases), the resolution sweep becomes an optimization, not a necessity.
