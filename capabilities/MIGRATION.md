# Migration: Phase 3 Module System

When Phase 3 ships, the module system transitions from spectral NJW clustering to z_invariant-based domain decomposition. This document specifies what changes, what stays, and the exact migration steps.

**Trigger:** Phase 3 R-GIN trained, validated, and producing z_invariant embeddings. Do not begin migration until Phase 3's intrinsic evaluation targets are met (reconstruction cosine similarity > 0.6, cross-layer AUC > 0.75).

---

## What Changes

### Module assignment source

| Before | After |
|---|---|
| NJW spectral clustering on eigenvector coordinates | z_invariant bisecting k-means (Phase 3) |
| k selected by modularity-Q sweep | k selected per-split by Q-gate + silhouette on z_invariant |
| Flat partition: K modules | Hierarchical tree: depth-1 partition = flat modules, deeper levels = sub-domains |
| `eigengap_k` fallback | Phase 2 fallback: spectral coordinate k-means with same Q-gate |

The diagnostic system receives modules from the domain tree's depth-1 partition. The `EnrichedModule` struct and `node_to_module` HashMap remain — only the source changes.

### Architecture output format

| Before | After |
|---|---|
| `architecture.modules`: flat array of `ModuleOutput` | `domain`: hierarchical tree with per-node health |
| `architecture.dependencies`: numeric source/target IDs | `domain.dependencies`: path-style source/target strings |
| `architecture.silhouette`: single float | Per-split silhouette in the tree (not a top-level field) |
| `architecture.package_agreement`: NMI + composition | NMI computed against depth-1 partition (same math, different source) |
| `architecture.package_fallback`: boolean | Removed — the domain tree always produces a decomposition |

### Layer inference

| Before | After |
|---|---|
| Edge-majority between module pairs → module-level DAG → topological sort | Semantic depth probe for ambiguous pairs + direction_surprise from R matrix for violation weighting |
| Self-referential (layers inferred from same edges measured against) | Externally grounded (probe trained on 500+ repos, R matrix encodes learned directionality) |

This directly improves `layer-violation` detection and HEALTH.md's `layer_conformance`.

### Output schema

`AnalysisOutput` gains a `domain` field (the hierarchical tree). The existing `architecture` field is deprecated — retained temporarily for backward compatibility, auto-derived from the domain tree's depth-1 partition.

```rust
pub struct AnalysisOutput {
    pub scope: ScopeOutput,
    pub coverage: CoverageOutput,
    pub spectral: Option<SpectralOutput>,
    #[deprecated(note = "Use domain tree depth-1 instead")]
    pub architecture: ArchitectureOutput,  // auto-derived from domain
    pub domain: Option<DomainTree>,        // NEW — the hierarchical decomposition
    pub roles: Vec<RoleOutput>,
    pub issues: Vec<IssueOutput>,
    pub health: Option<HealthOutput>,
    pub semantic_enabled: Option<bool>,
}
```

### Formatter

| Before | After |
|---|---|
| `--format=human`: renders flat modules section | Renders domain tree with per-level health |
| `--format=context`: flat module summaries in LLM narrative | Hierarchical module summaries with depth |
| `--format=domain`: flat bounded context map (Phase 2 spec) | Hierarchical domain tree with archetypes |
| Architecture section in all formats | Replaced by domain section |

---

## What Stays Unchanged

### Spectral decomposition (`spectral.rs`)

Eigenvalues, eigenvectors, Fiedler value — all stay. Phase 3 uses them as input:
- Spectral PEs (eigenvector coordinates) feed the R-GIN's SignNet
- Fiedler value feeds `cycle_freedom` in health
- Eigenvalues used for eigengap detection (Phase 2 fallback)

No code changes in `spectral.rs`.

### Structural roles (`roles.rs`)

`classify_roles()` depends only on graph topology (degree, betweenness, in/out degree). Zero dependency on modules. Hub, bridge, leaf, entry_point, utility — all computed identically.

No code changes in `roles.rs`.

### Betweenness centrality

Pure graph-theoretic. Independent of modules.

### SCC computation

Used by `circular-dependency` diagnostic and `cycle_freedom` in health. Independent of modules.

### Semantic analysis (`semantic.rs`)

Uses eigenvalues/eigenvectors + CodeLM embeddings. The module assignments it receives change source (z_invariant instead of NJW), but the semantic analysis algorithms (coherence, misplaced-concern, AMI) are module-source-agnostic.

### Package agreement

`compute_nmi(structural_partition, package_partition)` just needs two partitions. Whether the structural partition comes from NJW or z_invariant doesn't matter. The function is partition-source-agnostic.

Package agreement moves from `architecture.package_agreement` to `domain.package_agreement` (computed against depth-1 partition).

---

## Diagnostic Impact

z_invariant modules improve diagnostic quality because module quality is the floor for module-dependent diagnostics. Better boundaries → better centroids, better diversity counts, better layer inference, fewer false flags.

### Major improvement (5 diagnostics)

**`layer-violation`** — Layer inference transitions from self-referential edge-majority to semantically-anchored ordering + direction_surprise weighting. Fewer false positives from callback inversions and ambiguous edge counts. This is the single largest diagnostic improvement from the migration.

**`misplaced-concern` (Phase 2)** — Module centroids become more meaningful when modules are correctly bounded. A misplaced auth function in a correctly-bounded billing module has a clear centroid mismatch. With NJW's noisy boundaries, centroids are blurred and detection is noisy.

**`incoherent-module`** — NJW grouping unrelated code produces false "incoherent" flags (the module IS incoherent, but because the boundary is wrong, not because the code is wrong). z_invariant boundaries follow structural roles → fewer false flags, and genuine incoherence flags point to real problems.

**`wide-interface`** — If NJW incorrectly merges two domains into one module, the interface between them is invisible. z_invariant separates them → hidden wide interfaces become visible.

**`overloaded-utility`** — `caller_diversity` (fraction of modules that call a node) is more accurate with correct module boundaries. NJW merging two domains underestimates diversity.

### Moderate improvement (4 diagnostics)

**`cross-package-coupling`** — Better structural modules → more accurate detection of where coupling structure disagrees with package boundaries.

**`near-disconnect`** — Per-module Fiedler analysis is more meaningful when module boundaries match structural roles.

**`shadow-dependency`** — "Cross-module" is more accurately defined → more precise duplicate detection.

**`redundant-api`** — Module scope is more accurate → entry point identification is more precise.

### Minor improvement (2 diagnostics)

**`misplaced-concern` (Phase 3)** — Uses reconstruction_error (R-GIN signal, not centroid distance). Less dependent on module quality. Still uses modules for "suggested_module" in output.

**`coupling-mismatch`** — Uses per-layer embeddings. Module assignments used for scoping, not detection.

### Unaffected (2 diagnostics)

**`circular-dependency`** — SCC computation is module-independent. Module assignments used only for labeling ("this cycle spans auth and session").

**`unstable-peripheral`** — Uses betweenness centrality only. Zero module dependency.

---

## Migration Steps

### Step 1: Add domain tree output alongside architecture output

Add `DomainTree` type and `domain: Option<DomainTree>` field to `AnalysisOutput`. Implement z_invariant bisecting k-means. The domain tree is populated when Phase 3 outputs are available; `None` otherwise.

Both `architecture` (old) and `domain` (new) are present in the output. Existing consumers are unaffected.

**Changes:** `types.rs` (new types), `lib.rs` (domain tree construction after Phase 3 inference).

### Step 2: Wire diagnostics to domain tree depth-1

Extract the depth-1 partition from the domain tree. Construct `EnrichedModule` structs and `node_to_module` HashMap from the tree's depth-1 children. Pass these to `IssuesContext` exactly as before.

When the domain tree is available, diagnostics use z_invariant modules. When not available (Phase 1/2 fallback), diagnostics use the existing NJW modules.

**Changes:** `lib.rs` (conditional module source selection). No changes to `issues.rs` — the interface is identical.

### Step 3: Upgrade layer inference

Replace edge-majority layer inference with the semantic depth probe + direction_surprise approach from HEALTH.md. This affects `detect_layer_violations()` in `issues.rs` and `layer_conformance` in the health computation.

**Changes:** `issues.rs` (layer_violations detection), health computation (already specified in HEALTH.md).

### Step 4: Add domain formatter

Implement `--format=domain` as the hierarchical tree renderer. This supersedes Phase 2's flat bounded-context output.

**Changes:** `topo-formatter` (new domain tree renderer).

### Step 5: Deprecate architecture output

Mark `architecture` field as deprecated. Auto-derive it from domain tree depth-1 for backward compatibility:
- `architecture.modules` = domain tree depth-1 children mapped to `ModuleOutput`
- `architecture.dependencies` = domain tree depth-1 dependencies mapped to numeric IDs
- `architecture.silhouette` = silhouette from depth-1 clustering
- `architecture.package_agreement` = NMI of depth-1 partition vs packages

**Changes:** `lib.rs` (auto-derive architecture from domain tree).

### Step 6: Remove architecture output (breaking change)

Remove `ArchitectureOutput` from `AnalysisOutput`. Remove the auto-derivation from Step 5. Update `analysis.schema.json` to remove the `architecture` field and add the `domain` field.

Remove the architecture section from the human-readable and context formatters.

**Changes:** `types.rs`, `lib.rs`, `topo-formatter`, `analysis.schema.json`.

### Step 7: Remove NJW clustering code

Remove from `modules.rs`:
- `annotate_modules()` — replaced by domain tree nodes
- `build_module_dependencies()` — replaced by domain tree dependencies
- The Q-sweep in `analyze_full` — replaced by per-split Q-gate

Keep in `modules.rs`:
- `modularity_q()` — still used by the Q-gate in domain decomposition
- `module_label()` / TF-IDF labeling — still used for domain labels
- `compute_package_agreement()` — still used, just receives depth-1 partition

Keep in `spectral.rs`:
- Everything — eigenvalues, eigenvectors feed Phase 3's SignNet and the Phase 2 fallback path

**Changes:** `modules.rs` (remove ~200 lines), `lib.rs` (remove NJW-specific code path).

---

## Timeline

Steps 1-2 ship with Phase 3 (domain tree as additive output, diagnostics wired to z_invariant modules).

Steps 3-4 ship immediately after (upgraded layer inference, domain formatter).

Steps 5-6 ship after a deprecation period (one release cycle with both outputs, then breaking removal).

Step 7 ships with Step 6 (cleanup once the old output is removed).

---

## Risk Mitigation

**If z_invariant quality is poor:** The fallback path (Phase 2 spectral k-means) produces modules of comparable quality to the current NJW approach. Diagnostics degrade gracefully to current quality, not below it.

**If the domain tree is too deep/shallow:** The Q-gate + silhouette stopping criteria are tunable. If the tree is too deep (noisy splits), raise the ΔQ threshold. If too shallow (misses sub-domains), lower it. The flat depth-1 partition is always available as a safe fallback.

**If breaking `architecture` output causes consumer issues:** Step 5's auto-derivation period (one release cycle) gives consumers time to migrate. The `architecture` field in the JSON output is byte-for-byte identical during this period — it's just derived from a different source.
