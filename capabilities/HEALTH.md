# Health

A single scalar score measuring the structural health of a codebase. Higher is better. Requires the full pipeline: spectral analysis, semantic embeddings, trained R-GIN. Ship it when Phase 3 ships, not before.

**Output:** A number ∈ [0, 1], decomposable into two sub-scores (coherence, flow).

---

## What Structural Health Means

A codebase's structure is healthy when it minimizes the cost of understanding and changing the code. Every structural problem — misplaced code, dependency cycles, tangled layers — creates real engineering cost: wasted search time, missed updates, cascading breakage, inability to test or deploy independently.

Structural health is not about style. A monolith can be healthy. A microservices architecture can be healthy. Health is about whether the structure matches intent — whether the topology serves the developers or fights them.

**Health measures structural correctness, not structural risk.** Correctness asks: is the code where it belongs, and do the dependencies make sense? Risk asks: how fragile is the structure? These are related but distinct.

A well-designed central logger called by 200 functions is *correct* (it belongs where it is, its dependencies flow the right way) but *risky* (changing it cascades everywhere). A misplaced auth function in the billing module is *incorrect* regardless of risk profile.

The health score measures correctness. The diagnostics system handles risk through individual diagnostics (`near-disconnect`, `overloaded-utility`, `wide-interface`) with false positive suppression for intentional patterns like facades and shared utilities.

---

## Two Irreducible Dimensions

The health score has exactly two dimensions. This number was derived by enumeration: list every structural failure mode, group by independence, test each group for irreducibility. Two groups survive.

### The Irreducibility Test

A dimension is irreducible if:

1. **Independence.** You can maximize all other dimensions and still score zero on this one. It cannot be derived from the others.
2. **Directional value.** The underlying property (not just the measurement signal) is one that improving genuinely makes the codebase better. No architectural style exists where the property is undesirable.
3. **Non-collapsibility.** The dimension cannot be reframed as a special case of another dimension.

Note on criterion 2: we test the *property*, not the *signal*. Any signal can be gamed by degenerate optimizations (merge everything to reduce reconstruction error; delete code to eliminate cycles). The question is whether improving the actual underlying property — coherence of placement, correctness of dependency direction — is universally valuable. It is.

### Dimension 1: Coherence

**Does the code make sense where it is?**

Every node's structural context — who it's connected to, through what edge types, at what distance — should predict what that node does semantically. When structure predicts semantics, the code is coherent: things that belong together are together, things that don't belong together aren't.

**Structural failures captured:**

| Failure | Why it's a coherence failure |
|---|---|
| Misplaced concern | Node's structural position doesn't match its semantic content |
| Incoherent module | Structurally coupled nodes are semantically unrelated |
| Cross-package coupling | Structural module spans packages that should be separate |
| Shadow dependency | Same semantics exist in two structurally disconnected places |
| Redundant API | Multiple entry points do the same thing — redundant semantic content |

**Phase 3 signal: `1 - median(reconstruction_error)`**

The R-GIN's reconstruction error is the direct measurement of coherence. The model is trained to predict a node's semantic embedding from its multi-hop structural neighborhood, across all dependency layers, calibrated against 500+ repos. When the prediction succeeds, structure explains semantics — coherent. When it fails, structure and semantics disagree — incoherent.

This subsumes Phase 2's semantic coherence, AMI, and Rayleigh quotient. It is multi-hop (2-layer GIN), cross-layer (calls/imports/inherits processed separately), role-aware (spectral PEs encode structural role), and cross-repo calibrated.

**Why improving coherence is unconditionally valuable:** Moving code to where it semantically belongs always reduces search cost, always improves discoverability, always reduces the chance of missed updates. There is no architectural style where misplaced code is desirable.

### Dimension 2: Flow

**Do dependencies go in sensible directions?**

Dependencies should form a directed acyclic graph at the module level. Higher-level modules depend on lower-level modules, not the reverse. There should be no mutual entrainment — no cycles where A depends on B depends on A.

**Structural failures captured:**

| Failure | Why it's a flow failure |
|---|---|
| Dependency cycle | Mutual dependency prevents independent reasoning, testing, deployment |
| Layer violation | Lower-level module depends on higher-level — inverts the intended direction |
| Coupling mismatch | Different dependency types disagree on direction — call graph says "A above B" but import graph says "B above A" |

**Signal: `cycle_freedom × layer_conformance`**

- **cycle_freedom** = `1 - (nodes_in_nontrivial_SCCs / total_nodes)` — fraction of the codebase free of circular dependencies. Uses Tarjan's SCCs. Pure graph-theoretic; Phase 3 adds nothing here.

- **layer_conformance** — fraction of edges that respect the intended layering. This is where Phase 3 transforms the signal. See "Phase 3 Layer Conformance" below.

The product captures both properties: a codebase needs both acyclicity and correct direction to score well on flow.

**Why improving flow is unconditionally valuable:** Eliminating a cycle always restores the ability to reason about, test, and deploy modules independently. Fixing a layer violation always makes the dependency direction match the intended architecture. There is no architectural style where cycles or inverted dependencies are desirable.

### Why Not 3 Dimensions? (Resilience)

We considered a third dimension — **resilience** — measuring robustness to change: no fragile bottlenecks, no single points of failure, balanced degree distribution.

Resilience passes the first two criteria. It is independent (a central logger is correct and well-directed but risky — neither coherence nor flow captures this). The underlying property — structural robustness — is genuinely valuable to improve.

**Resilience fails on measurement reliability.** A central logger and a god-function bottleneck have identical graph signatures: 200 in-degree edges from diverse modules, high betweenness centrality. The first is intentional and healthy. The second is accidental and problematic. No graph-theoretic measure can distinguish them without knowing architectural intent.

The diagnostic system handles this through per-case suppression rules: suppress loggers, serializers, constructors, known patterns. After suppression, the remaining bottlenecks are genuine problems. But a health *scalar* can't encode per-case intent. Any aggregate would either:
- Include false positives (penalizing intentional design) — making the score noisy and untrustworthy as an optimization target
- Rely entirely on post-suppression diagnostic counts — which is just "issue burden for resilience diagnostics," redundant with the issues list

The measurement reliability test:

| Dimension | Low score reliably means... | False positive rate |
|---|---|---|
| Coherence | Code is in the wrong place | Low (R-GIN calibrated across 500+ repos) |
| Flow | Dependencies go the wrong direction | Low (cycles and layer violations are structural facts) |
| Resilience | Structure is fragile | **High** (can't distinguish intentional from accidental concentration) |

**Resilience is real, independent, and important. It is excluded because it cannot be reliably measured as a scalar.** The diagnostics (`near-disconnect`, `overloaded-utility`, `wide-interface`) are the right tool — they provide the per-case context that a scalar cannot.

**Overloaded utilities and wide interfaces are NOT coherence failures.** A well-designed central logger has LOW reconstruction error — the R-GIN learns the "high-degree cross-domain node" pattern across training repos and predicts it accurately. The node is coherent with its position. It's just risky. Similarly, a wide interface between two correctly organized modules has LOW reconstruction error — every coupling point is semantically justified. The interface is correct. It's just fragile.

### Why Not 1 Dimension?

Flow does not collapse into coherence. A dependency cycle between auth and session is not a placement problem — both modules are correctly identified and correctly populated. The code is where it semantically belongs. The problem is that the dependencies between correctly-placed code go in both directions.

The R-GIN's reconstruction error does not reliably capture flow failures. A cycle between two semantically related modules (auth ↔ session share authentication vocabulary) produces low reconstruction error because the neighborhood semantics are consistent. The structure predicts the semantics fine — it's just that the dependency direction is wrong. Direction is a graph-theoretic property, not a semantic one.

### Why Not 4+ Dimensions?

We evaluated:

- **Evolution** (is health trending up or down?): This is `topo health` over git history — a time-series of the health score, not a dimension of the score itself.
- **Complexity** (is the structure simple?): Collapses into coherence. A "complex" structure is one where structural context poorly predicts semantic content.
- **Completeness** (are abstractions missing?): Either it's a shadow dependency (coherence) or a feature request (not structural health).
- **Risk** (how vulnerable to future problems?): Handled by diagnostics, not by the score. See "resilience" above.

No fourth dimension survived the irreducibility test.

---

## The Score

```
THS = coherence ^ α  ×  flow ^ (1 - α)
```

Weighted geometric mean with α as a tunable parameter. **Starting hypothesis: α = 0.7.** This reflects the asymmetry between the dimensions:

1. **Coverage.** Coherence captures more failure modes than flow. Most structural problems are about placement, not direction.
2. **Signal power.** The R-GIN's reconstruction error — the strongest signal in the pipeline — directly measures coherence. Flow uses simpler graph-theoretic signals.
3. **Prevalence.** In practice, coherence failures (misplaced code, incoherent modules) are more common than flow failures (cycles, layer violations).

**This weighting is a hypothesis, not a derived result.** The calibration protocol (below) fits α empirically against developer judgment. If validation shows α = 0.5 or α = 0.8 is better, use that. The three arguments above motivate the starting point, not the final answer.

The geometric mean ensures that a zero in either dimension tanks the score. Neither dimension can compensate for the other.

### Sub-Score Computation

#### Coherence

```
coherence = clamp(1 - median(reconstruction_error), 0.0, 1.0)
```

Where `reconstruction_error(v)` is the R-GIN's per-node cosine distance between predicted and actual semantic embedding, computed with all nodes unmasked (full structural context available).

**Why median, not mean.** The mean is dominated by outliers and is size-dependent: adding 1000 clean nodes to a codebase with 10 problematic nodes dilutes the mean, making the codebase appear "healthier" without fixing anything. The median is robust to both — it measures the typical node, not the average. If more than half the nodes are coherent, the median is low regardless of how bad the outliers are. The diagnostic system handles outliers individually.

**Limitation of the median.** The median ignores the tails. A codebase where 49% of nodes have catastrophic reconstruction error but 51% are perfect gets a high coherence score. This is a real limitation — but in practice, 49% catastrophic nodes would trigger hundreds of `misplaced-concern` and `incoherent-module` diagnostics, making the problem unmissable. The health score is a headline number; the diagnostics are the detail. If calibration shows the median is too insensitive, a 75th-percentile or trimmed mean is the fallback.

**Empty codebase.** If no nodes exist, coherence defaults to 1.0 (vacuously correct — there is nothing to be wrong about).

**Why clamp.** Cosine distance ∈ [0, 2] (cosine similarity can be negative for anti-correlated vectors). If any node has reconstruction_error > 1.0, the raw formula produces coherence < 0. Clamping to [0, 1] prevents negative values and undefined behavior when computing `coherence^α` (negative base with fractional exponent = NaN in IEEE 754).

**Interpretation:**
- **> 0.85:** Structure cleanly explains semantics. Code is where it belongs.
- **0.7 – 0.85:** Mostly coherent with localized problems. Check `misplaced-concern` and `incoherent-module` diagnostics.
- **0.5 – 0.7:** Significant structural-semantic disagreement. Multiple modules likely need restructuring.
- **< 0.5:** Structure and semantics are largely disconnected. The codebase's organization does not reflect its purpose.

These thresholds are initial hypotheses, calibrated via the protocol below.

#### Flow

```
flow = cycle_freedom × layer_conformance
```

Both components are in [0, 1] by construction, so the product is in [0, 1]. No clamping needed.

**cycle_freedom** = `1 - (nodes_in_nontrivial_SCCs / max(total_nodes, 1))`

Uses Tarjan's SCCs on the directed graph. Only nontrivial SCCs (≥2 nodes) count. After false positive suppression (trait cycles, test-only cycles). The `max(..., 1)` guards against division by zero.

#### Phase 3 Layer Conformance

The Phase 1 layer_conformance is self-referential: it infers layers from edge-majority, then measures violations against those same edges. This has known failure modes:

- **51/49 problem.** Nearly balanced edge counts produce confident-looking layers from thin evidence.
- **Callback inversion.** A database module with callbacks into the API layer gets inferred as "above" the API — structurally the majority goes up, but architecturally the database is below.
- **No semantic grounding.** The layer ordering has no concept of what code *does* — only how edges flow.

Phase 3 fixes both the layer **assignment** and the violation **weighting**.

##### Learned layer assignment: semantic depth probe

A linear probe trained on the R-GIN corpus maps semantic content to layer position:

```
depth_sem(M) = w^T · centroid_sem(M) + b
```

Where `w` is a 768d weight vector and `b` is a scalar, fit via ordinary least squares across the training corpus. The training signal: across 500+ repos, modules with database/filesystem/serialization vocabulary sit low; modules with HTTP/CLI/orchestration vocabulary sit high.

The probe **does not replace** edge-majority wholesale. It breaks ties:

- Edge majority decisive (minority_ratio < 0.2): trust edges
- Edge majority ambiguous (0.2 ≤ minority_ratio < 0.5): `depth_sem` breaks the tie
- No edges between modules: `depth_sem` alone

This eliminates the 51/49 problem (the probe decides, not a coin flip) and the callback inversion (the probe knows database code sits below HTTP handlers regardless of callback edges).

**Cost:** ~120 lines of Rust. The 768-float weight vector ships with the R-GIN model bundle. The probe is fit once during training, not per-codebase.

##### Learned violation weighting: direction_surprise

Phase 3's Loss 2 trains a bilinear predictor to determine whether a call edge exists between two nodes based on their import-layer positions:

```
p(u, v) = σ(z_imports(u)^T · R · z_imports(v))
```

R is a learned 32×32 matrix. Because it is trained on *directed* edges, R is **not symmetric** — `p(u, v) ≠ p(v, u)`. This asymmetry encodes directionality learned from 500+ repos.

For each call edge (u→v), the **direction_surprise** measures how much the model thinks the edge goes the wrong way:

```
direction_surprise(u, v) = σ(z_imports(v)^T R z_imports(u)) - σ(z_imports(u)^T R z_imports(v))
```

- **Positive:** the model thinks the reverse direction is more likely. The edge is against the learned flow.
- **Near zero:** no directional preference.
- **Negative:** the model thinks the observed direction is natural. The edge is with the flow.

**Why this is mathematically sound:**
- Lives entirely in z_imports space — no cross-space comparison
- Uses the learned asymmetry of R — directionality from 500+ repos
- Is per-edge — aggregates to per-node, per-module naturally
- Callbacks get low direction_surprise because they are import-expected (the lower module imports callback types from the upper module, so R predicts the call in both directions)

**Critical check after training:** Compute `‖R - R^T‖_F / ‖R‖_F`. If > ~0.1, R has learned meaningful directionality and the signal is real. If ≈ 0, R is effectively symmetric and direction_surprise vanishes — fall back to binary violation counting.

##### Combined layer_conformance

```
layer_conformance = 1 - (Σ_violating max(direction_surprise(u,v), 0)) / max(Σ_all |direction_surprise(u,v)|, ε)
```

Where:
- **Violating edges** are identified using the semantically-anchored layer assignment (not self-referential edge-majority)
- **Each violation is weighted** by how strongly the model thinks it goes the wrong way
- ε is a small constant preventing division by zero when all direction_surprise values are near zero

This replaces two binary decisions (is this edge a violation? × does it count?) with two continuous, learned signals (how confident is the layer assignment? × how much does the model think this edge goes the wrong way?).

**Cost:** ~130 lines of Rust (32×32 R matrix shipped with the R-GIN model, per-edge bilinear computation, aggregation).

**Note on dimensional mismatch.** cycle_freedom is a node-fraction, layer_conformance is an edge-fraction. Their product conflates two different scales. This is intentional: the product creates a joint penalty where both properties must be present. The product is not dimensionally meaningful as a physical quantity — it is a penalty function.

**Interpretation:**
- **> 0.9:** Clean dependency flow. Near-DAG structure with consistent layering.
- **0.7 – 0.9:** Minor flow problems. A few cycles or layer violations, likely localized.
- **0.5 – 0.7:** Significant flow problems. Multiple cycles or widespread layer violations.
- **< 0.5:** Dependency structure is deeply tangled. Cycles span many modules, or layering is not respected.

---

## Diagnostic-Health Bridge

Every diagnostic maps to a health dimension, to neither, or to both:

THS requires Phase 3. When Phase 3 is active, Phase 3's `misplaced-concern` (upgraded) suppresses Phase 2's `misplaced-concern`. The bridge table reflects the Phase 3 state:

| Diagnostic | Dimension | Mechanism |
|---|---|---|
| `misplaced-concern` (Phase 3, upgraded) | coherence | Directly measured by reconstruction_error |
| `incoherent-module` | coherence | Module members' reconstruction_errors are high |
| `cross-package-coupling` | coherence | Cross-package nodes have confused structural context |
| `shadow-dependency` | coherence | Duplicated semantics in disconnected positions raise median error |
| `redundant-api` | coherence | Redundant entry points inflate structural-semantic noise |
| `coupling-mismatch` | flow | Resolving directional disagreement fixes underlying layer violations |
| `cycle-member` | flow | Directly increases nodes_in_nontrivial_SCCs |
| `layer-violation` | flow | Directly increases layer_violating_edges |
| `near-disconnect` | — | Risk, not correctness |
| `overloaded-utility` | — | Risk, not correctness |
| `wide-interface` | — | Risk, not correctness |
| `unstable-peripheral` | — | Risk, not correctness |
| `misplaced-concern` (Phase 2) | — | Suppressed when THS is active (Phase 3 version takes precedence) |

Resolving a diagnostic that maps to a health dimension improves THS. This creates a natural priority function: **resolve the diagnostic that produces the largest THS improvement first.**

---

## Output Format

### JSON

The health score is a new field within the existing `AnalysisOutput.health` object. It extends (not replaces) the existing `HealthOutput` fields:

```json
{
  "health": {
    "modularity_q": 0.74,
    "semantic_smoothness": 0.23,
    "semantic_structural_ami": 0.68,
    "semantic_energy_profile": { "..." : "..." },
    "topo_health_score": 0.72,
    "coherence": 0.78,
    "flow": 0.61
  }
}
```

The existing fields (`modularity_q`, `semantic_smoothness`, `semantic_structural_ami`, `semantic_energy_profile`) are retained for backward compatibility and because they provide useful detail that THS intentionally compresses. THS is the headline number; the existing fields are the fine print. The issues list provides the detail on what to fix.

### Human-Readable

```
Health: 0.72

  Coherence: 0.78  ████████░░  structure explains semantics
  Flow:      0.61  ██████░░░░  dependency direction problems
```

### LLM Context (`--format=context`)

The health score appears in the first line alongside the existing metrics. This evolves the Phase 2 health line format (which shows `Q=..., λ₂=..., smoothness=..., AMI=...`) by leading with THS as the headline number:

```
## myapp — layered monolith (4 tiers, 847 nodes)
Health: 0.72 (coherence: 0.78, flow: 0.61) | Q=0.74, λ₂=0.012, smoothness=0.23, AMI=0.68
```

THS is the lead number. Q, λ₂, smoothness, and AMI remain on the same line for LLMs that need the raw metrics. The Phase 2 format (`Health: modularity Q=0.74, ...`) is superseded — THS is the primary health signal once Phase 3 ships.

---

## Archetype Percentile (Cross-Repo Context)

The R-GIN produces a 64d graph-level embedding (`g_embedding`) that places every analyzed codebase in a shared structural space. Codebases with similar architecture get similar embeddings. This enables a secondary metric:

```
archetype_percentile = percentile(THS, codebases_with_similar_g_embedding)
```

This is NOT part of THS. It is additional context:

```
Health: 0.72 (65th percentile among layered monoliths of similar size)
```

The percentile answers "is 0.72 good?" — a question the raw score cannot answer without a reference population. It requires a corpus of analyzed codebases, which grows as topo is used.

**This is reported separately from THS** because it depends on the reference corpus (which evolves) and the archetype classification (which could be wrong). THS is deterministic given the codebase. The percentile is relative to a population.

**Comparability caveat.** THS uses the R-GIN's reconstruction error, which is cross-repo calibrated by training. But reconstruction error still depends on graph density (denser graphs provide richer context, potentially reducing error), semantic vocabulary (domain-specific terminology may embed less cleanly), and neighborhood diversity. The archetype percentile compensates by comparing within similar architectures. Raw THS comparison across architecturally different codebases should be treated as approximate.

---

## Tracking Health Over Time

`topo health <path>` walks git history and computes THS at each sampled commit:

```
topo health . --since=2024-01-01 --sample=weekly

commit  a1b2c3  THS=0.74  coherence=0.81  flow=0.63  ← baseline
commit  d4e5f6  THS=0.72  coherence=0.79  flow=0.62  ← slight decline
commit  g7h8i9  THS=0.68  coherence=0.78  flow=0.54  ← flow dropped (new cycle)
commit  j0k1l2  THS=0.65  coherence=0.72  flow=0.54  ← coherence declining
commit  m3n4o5  THS=0.73  coherence=0.80  flow=0.62  ← refactoring restored
```

**Patterns:**
- **Monotonic decline in coherence:** Code is being placed expediently, not intentionally. Structural debt accumulating.
- **Sudden flow drop:** Someone introduced a cycle. Worth investigating the specific commit.
- **Coherence stable, flow improving:** Targeted cycle resolution. Good sign.
- **Both declining:** Architecture is under stress. Time for structural review.

---

## What Was Considered and Rejected

### Resilience as a third dimension

See "Why Not 3 Dimensions?" above. Resilience is genuinely independent and important, but cannot be reliably measured as a scalar — intentional bottlenecks and accidental bottlenecks have identical graph signatures. The diagnostic system handles this per-case with false positive suppression.

### Coupling agreement as a third flow sub-component

An earlier draft included `coupling_agreement = 1 - mean(max_pairwise_disagreement(z_calls, z_imports, z_inherits))` as a third multiplicative term in flow. This was removed because the formula is mathematically undefined — z_calls, z_imports, and z_inherits are in different learned 32d spaces (the HSIC decorrelation loss explicitly trains them apart). Cosine distance between vectors in different spaces is meaningless.

The directional information from cross-layer relationships is instead captured through the `direction_surprise` signal, which operates entirely within z_imports space using the learned bilinear R matrix. This avoids the cross-space problem while preserving the directional signal.

### Overloaded-utility and wide-interface as coherence failures

An earlier draft claimed these collapse into coherence. On scrutiny, they do not:

- A well-designed central logger has LOW reconstruction error — the R-GIN sees "high-degree cross-domain node" across hundreds of training repos and learns to predict it accurately. The node is coherent. It's just risky.
- A wide interface between two correctly organized modules has LOW reconstruction error — every coupling point is where it belongs. The interface is correct. It's just fragile.

These are genuine structural concerns that the health score does not capture. They are handled by their respective diagnostics. The health score is not intended to be comprehensive — it measures correctness. The diagnostics are comprehensive — they measure correctness, risk, and everything in between.

### Issue-burden score

`1 - Σ(severity × confidence) / max_burden`. This double-counts — the diagnostics already feed into coherence and flow through the signals they affect. A separate issue-burden dimension would be collinear with the existing two.

### Information-theoretic health

"How much of the graph's entropy is explained by the module decomposition." This is essentially modularity Q with extra steps. Q is already captured within coherence (if modularity is poor, reconstruction error is high because structural context is noisy).

### Harmonic mean combination

Too punitive. One slightly-below-average dimension dominates the score. The geometric mean is the right balance — one zero kills everything, but moderate weakness doesn't catastrophically dominate.

### Equal weights

Starting with equal weights (0.5/0.5) was considered as the null hypothesis. The α = 0.7 starting point reflects the asymmetry between the dimensions but is subject to empirical calibration. See "Calibration Protocol."

### Learned health score

Train a regressor from g_embedding → developer_health_rating. Conceptually appealing but creates a black-box score that can't be decomposed into "here's why it's low." The decomposable THS is the primary output. A learned score could serve as a calibration check — if THS and the learned score disagree significantly, investigate. Not implemented in the initial version.

### Mean aggregation for coherence

An earlier draft used `mean(reconstruction_error)`. The mean is size-dependent: adding 1000 clean nodes to a codebase with 10 problematic nodes dilutes the mean, making the codebase appear "healthier" without fixing anything. The median is robust to outliers and size effects. Diagnostics handle the outlier nodes individually.

---

## Calibration Protocol

The weight α and interpretation thresholds are hypotheses, not axioms. They are calibrated empirically:

1. **Corpus.** Collect THS, coherence, flow, and g_embedding for 50+ open-source codebases of varying quality and architectural style.
2. **Ground truth.** For each codebase, collect developer structural health ratings (1-10) from 3+ developers who know the codebase.
3. **Weight optimization.** Fit α in `THS = coherence^α × flow^(1-α)` to maximize Spearman rank correlation between THS and mean developer rating.
4. **Threshold calibration.** Determine the THS values that correspond to developer consensus categories (healthy / needs attention / unhealthy).
5. **Archetype stratification.** Check whether optimal α differs by architecture style (layered vs hub-spoke vs microservices). If so, use archetype-specific weights.
6. **Repeat.** As the R-GIN training corpus grows and the model improves, re-calibrate.

The calibration is empirical and iterative. The math serves the developer, not the other way around.

---

## Implementation

### Dependencies

- Phase 3 R-GIN producing per-node `reconstruction_error` and per-node `z_imports` (32d)
- Phase 3 Loss 2 bilinear matrix R (32×32), shipped with the model bundle
- Semantic depth probe weights (768d vector + scalar bias), shipped with the model bundle
- Phase 2 frozen CodeLM embeddings (768d per node)
- Phase 1+ Tarjan's SCCs (already implemented)
- Diagnostic false-positive suppression (already implemented)

### Computation

```rust
pub struct HealthScore {
    pub topo_health_score: f64, // THS ∈ [0, 1]
    pub coherence: f64,         // ∈ [0, 1]
    pub flow: f64,              // ∈ [0, 1]
}
```

This struct is added as a field on the existing `HealthOutput`, not as a replacement. The existing fields (`modularity_q`, `semantic_smoothness`, etc.) are retained.

The computation:

**Coherence:**
1. Collect all per-node `reconstruction_error` values from R-GIN output. If empty (no nodes), return THS = 1.0 (vacuously healthy).
2. `coherence = (1.0 - median(reconstruction_errors)).clamp(0.0, 1.0)`.

**Flow — cycle_freedom:**
3. `cycle_freedom = 1.0 - (suppressed_scc_nodes as f64 / total_nodes.max(1) as f64)`. Uses Tarjan's SCCs after false positive suppression (trait cycles, test-only cycles).

**Flow — layer_conformance (Phase 3):**
4. Compute per-module semantic centroids from frozen CodeLM embeddings.
5. Apply the semantic depth probe: `depth_sem(M) = w^T · centroid(M) + b`.
6. Infer layer ordering: use edge-majority where decisive (minority_ratio < 0.2), `depth_sem` to break ties for ambiguous pairs and unconnected pairs.
7. For each call edge (u→v), compute `direction_surprise(u, v) = σ(z_imports(v)^T R z_imports(u)) - σ(z_imports(u)^T R z_imports(v))`.
8. Identify violating edges (against the semantically-anchored layer ordering).
9. `layer_conformance = 1.0 - (Σ_violating max(ds, 0)) / max(Σ_all |ds|, ε)`.

**Combine:**
10. `flow = cycle_freedom * layer_conformance`.
11. `THS = coherence.powf(alpha) * flow.powf(1.0 - alpha)` where α = 0.7 (initial).

~400 lines of Rust total. The expensive work (R-GIN inference, SCC computation) is already done by the time health is computed. The additional cost is per-edge bilinear computation (one 32×32 matmul per edge) and per-module semantic centroid computation.
