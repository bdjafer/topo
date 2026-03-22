# Step 3: Self-Supervised Evaluation & Inference Integration

This step defines the fully self-supervised evaluation loop (no human labeling), the inference integration into the Rust analyzer, and the downstream wiring that connects R-GIN outputs to the existing issue/health/formatter pipeline.

---

## 1. Self-Supervised Evaluation Framework

### Design Principle

PHASE_3.md §Evaluation specifies extrinsic evaluation against "20-30 known misplacements per codebase." We achieve this **without human labeling** through three mechanisms:

1. **Phase 2 cross-validation:** Phase 2's mathematical tools (local variation, misplaced concern heuristic) and Phase 3's learned reconstruction error measure the same phenomenon (structural-semantic disagreement) via independent methods. Agreement between them is a label-free validation signal.

2. **Synthetic perturbation:** Artificially create misplaced concerns by swapping nodes between modules. The model should detect them. This is manufactured ground truth — fully automated, arbitrarily scalable.

3. **Structural consistency checks:** Properties that must hold if the model works (NMI lift, role consistency, perturbation sensitivity) — all derivable from the graph structure itself.

### Tier 1: Intrinsic Metrics (from training objectives)

Computed during training validation (see Step 2 §10). These are necessary but not sufficient — a model can overfit to reconstruction without capturing meaningful structural patterns.

| Metric | Target | Signal |
|--------|--------|--------|
| Masked reconstruction cosine similarity | > 0.6 | Structure predicts semantics |
| Cross-layer AUC | > 0.75 | Import structure predicts calls |
| Graph contrastive accuracy | > 0.8 | Same-repo similarity |
| R asymmetry ratio | > 0.1 | Directional learning |

### Tier 2: Phase 2 Agreement (cross-method validation)

Phase 2's `local_variation` and Phase 3's `reconstruction_error` are independent measures of the same thing. If two methods built on completely different principles agree, both are likely capturing real signal.

```python
def tier2_evaluation(model, val_dataset, phase2_results: dict) -> dict:
    """Cross-validate Phase 3 against Phase 2 on held-out repos."""
    rank_correlations = []
    topk_overlaps = []

    for data in val_dataset:
        repo_key = data.repo

        # Phase 3: reconstruction error per node
        with torch.no_grad():
            mask = torch.zeros(data.num_nodes, dtype=torch.bool)  # no masking for inference
            z_str, *_ = model(data, mask)
            p3_errors = model.reconstruction_error(z_str, data.x_semantic)

        # Phase 2: local variation per node (precomputed)
        p2_scores = phase2_results[repo_key]["local_variation"]

        # Align by node ID (order may differ)
        node_ids = phase2_results[repo_key]["node_ids"]
        p3_aligned = align_by_node_id(p3_errors, data.node_ids, node_ids)

        # Spearman rank correlation
        rho, _ = scipy.stats.spearmanr(p2_scores, p3_aligned)
        rank_correlations.append(rho)

        # Top-k overlap (Jaccard of top 20% anomalous nodes)
        k = max(1, len(node_ids) // 5)
        top_p2 = set(np.argsort(p2_scores)[-k:])
        top_p3 = set(np.argsort(p3_aligned)[-k:])
        jaccard = len(top_p2 & top_p3) / len(top_p2 | top_p3)
        topk_overlaps.append(jaccard)

    return {
        "rank_correlation_mean": np.mean(rank_correlations),
        "rank_correlation_std": np.std(rank_correlations),
        "topk_overlap_mean": np.mean(topk_overlaps),
        "topk_overlap_std": np.std(topk_overlaps),
    }
```

**Targets:**
- Mean rank correlation > 0.3 (moderate agreement)
- Mean top-k overlap > 0.3 (better than chance for k=20%)

**What if they disagree?** Low correlation means either Phase 2 or Phase 3 (or both) are capturing noise. Investigate the disagreement repos: does one method flag clearly-misplaced nodes that the other misses? This diagnostic guides model refinement.

**Alternative Phase 2 metrics:** In addition to `local_variation`, also compare against:
- Phase 2's `misplaced_concern` confidence scores (from `semantic.rs::detect_misplaced_concerns`)
- Phase 2's module coherence scores (nodes in low-coherence modules should have higher reconstruction error)

Using multiple Phase 2 signals reduces the chance of a single weak proxy dominating the evaluation.

### Tier 3: Synthetic Perturbation Test (manufactured ground truth)

This is the strongest self-supervised signal. We **create** artificial misplaced concerns by reassigning nodes between modules, then test whether the model detects them.

```python
def tier3_perturbation_test(model, val_dataset, n_trials: int = 5) -> dict:
    """Synthetic perturbation: swap nodes between modules, check detection."""
    sensitivities = []
    specificities = []
    error_deltas = []

    for data in val_dataset:
        for trial in range(n_trials):
            # 1. Get module assignments
            modules = data.metadata["modules"]
            node_to_module = {}
            for mod in modules:
                for nid in mod["member_indices"]:
                    node_to_module[nid] = mod["id"]

            # 2. Randomly swap ~8% of nodes between modules
            all_nodes = list(node_to_module.keys())
            n_swap = max(2, int(len(all_nodes) * 0.08))
            swap_indices = np.random.choice(all_nodes, size=n_swap, replace=False)

            # For each swapped node, move its module assignment to a random different module
            module_ids = list(set(node_to_module.values()))
            for idx in swap_indices:
                old_mod = node_to_module[idx]
                new_mod = np.random.choice([m for m in module_ids if m != old_mod])
                node_to_module[idx] = new_mod

            # 3. Compute reconstruction error BEFORE and AFTER perturbation
            # "After" means: re-embed with the perturbed graph structure
            # (swap the edges: move edges from old module neighbors to new module neighbors)
            data_perturbed = perturb_graph_structure(data, swap_indices, node_to_module)

            with torch.no_grad():
                mask = torch.zeros(data.num_nodes, dtype=torch.bool)

                # Original errors
                z_str_orig, *_ = model(data, mask)
                errors_orig = model.reconstruction_error(z_str_orig, data.x_semantic)

                # Perturbed errors
                z_str_pert, *_ = model(data_perturbed, mask)
                errors_pert = model.reconstruction_error(z_str_pert, data.x_semantic)

            # 4. Measure detection
            swap_set = set(swap_indices)
            nonswap_set = set(all_nodes) - swap_set

            # Error increase for swapped nodes (they're now misplaced)
            swap_delta = errors_pert[list(swap_set)].mean() - errors_orig[list(swap_set)].mean()
            nonswap_delta = errors_pert[list(nonswap_set)].mean() - errors_orig[list(nonswap_set)].mean()
            error_deltas.append(swap_delta.item() - nonswap_delta.item())

            # Sensitivity: fraction of swapped nodes in top-20% error after perturbation
            threshold = torch.quantile(errors_pert, 0.8)
            high_error = errors_pert > threshold
            sensitivity = high_error[list(swap_set)].float().mean().item()
            sensitivities.append(sensitivity)

            # Specificity: fraction of non-swapped nodes NOT in top-20% error
            specificity = (~high_error[list(nonswap_set)]).float().mean().item()
            specificities.append(specificity)

    return {
        "perturbation_sensitivity_mean": np.mean(sensitivities),
        "perturbation_specificity_mean": np.mean(specificities),
        "perturbation_error_delta_mean": np.mean(error_deltas),
    }
```

**Graph structure perturbation:** To simulate a genuine misplacement, we don't just relabel the module — we actually modify the graph structure by rewiring a subset of edges to reflect the new module assignment. Specifically:

The perturbation must both **add** edges to the new module and **remove** some edges to the old module — otherwise the node simply gains degree, which is trivially detectable without structural understanding.

```python
def perturb_graph_structure(data, swap_indices, old_module_map, new_module_map):
    """Rewire swapped nodes: remove ~50% of old-module edges, add edges to new module."""
    data_new = data.clone()
    for idx in swap_indices:
        old_mod_members = set(n for n, m in old_module_map.items() if m == old_module_map[idx])
        new_mod_members = [n for n, m in new_module_map.items() if m == new_module_map[idx] and n != idx]
        if not new_mod_members: continue
        for edge_type in ["calls", "imports"]:
            edge_key = ("node", edge_type, "node")
            if edge_key not in data_new.edge_types: continue
            ei = data_new[edge_key].edge_index
            # Remove ~50% of edges to old module neighbors
            old_mask = ((ei[0] == idx) & torch.tensor([t.item() in old_mod_members for t in ei[1]])) | \
                       ((ei[1] == idx) & torch.tensor([s.item() in old_mod_members for s in ei[0]]))
            keep = ~old_mask | (torch.rand(old_mask.sum()) > 0.5)  # keep ~50%
            # ... (filter edges, add new edges to new_mod_members)
    return data_new
```

This makes the perturbation realistic: the node loses coupling to its original module and gains coupling to the new one. A degree-only detector cannot distinguish this from an unperturbed node.

**Why this isn't trivially passable:** A random model would assign random reconstruction errors — sensitivity would be ~20% (by definition, 20% of all nodes are above the 80th percentile, and swapped nodes have no systematic reason to be there). A degree-only model would flag high-degree nodes regardless of perturbation. The perturbation test specifically requires that **the same nodes** see error **increases** when they are moved — this tests the model's sensitivity to structural context changes, not static properties.

**Hardened variant:** To prevent gaming, also run a **control perturbation** where nodes are "swapped" to the same module (no-op relabeling). Sensitivity should be near baseline (~20%) in the control. If it's high, the model is responding to noise in the perturbation process, not the structural change.

**Targets:**
- Perturbation sensitivity > 0.6 (swapped nodes detected in top-20% error)
- Control sensitivity ≈ 0.20 ± 0.05 (no-op relabeling shouldn't trigger)
- Perturbation specificity > 0.75 (non-swapped nodes stay below threshold)
- Error delta > 0.05 (swapped nodes see measurably higher error increase)

### Tier 4: Structural Consistency Checks

| Check | Metric | Target | Rationale |
|-------|--------|--------|-----------|
| NMI lift | NMI(z_invariant clusters, packages) - NMI(spectral clusters, packages) | ≥ 0 | Phase 3 module detection at least as good as Phase 2 |
| Role consistency | Nodes classified as "bridge" by Phase 1 should have distinctive per-layer profiles | Qualitative | Cross-layer decomposition captures known structural roles |
| Reconstruction error vs degree | Rank correlation(reconstruction_error, degree) | < 0.5 | Model shouldn't just flag high-degree nodes |
| Per-layer role divergence | For bridge nodes: std(z_calls role, z_imports role, z_inherits role) | > mean(non-bridge std) | Bridges have divergent per-layer positions |

### Ablation Baselines

Phase 3 must outperform these naive baselines on the synthetic perturbation test:

| Baseline | Description |
|----------|-------------|
| Random | Random 160d → 768d MLP (no graph structure) |
| Phase 2 local variation | Existing mathematical tool |
| Centroid distance | Simple cosine distance to module centroid |
| Degree-only | Flag nodes with degree > p90 |

If Phase 3 doesn't beat all four on sensitivity, the learned model isn't adding value over heuristics.

---

## 2. Go/No-Go Gate

After evaluation, apply the gate from PHASE_3.md: **Phase 3 must improve anomaly detection precision by at least 10 percentage points over Phase 2.**

In our self-supervised framing, the gate checks:

1. Tier 2 agreement: rank correlation > 0.3
2. Tier 3 sensitivity > 0.6, specificity > 0.75
3. Tier 3 **precision** > Phase 2 precision + 0.10 (precision = true_positives / flagged_total, where true positives are swapped nodes correctly flagged and flagged_total includes false positives from non-swapped nodes). PHASE_3.md requires precision improvement, not sensitivity.
4. Phase 3 beats all ablation baselines on precision (baselines re-run on perturbed graph for fair comparison)
5. Control perturbation sensitivity ≈ 0.20 ± 0.05

**If the gate fails:** Do NOT deploy Phase 3. Debug: which metric failed? If Tier 2 fails (low Phase 2 agreement), the model may have learned spurious patterns. If Tier 3 fails (low perturbation sensitivity), the model lacks discrimination. If baselines are close, the model isn't adding value. Return to Step 2 (hyperparameter tuning, loss weight adjustment, more training data).

---

## 3. Inference Integration (Rust)

### Decision: ONNX vs Native Rust

**Recommendation: native Rust forward pass** (~1500 lines), consistent with PHASE_3.md §Implementation Plan Step 5. The R-GIN is architecturally simple (linear layers, ReLU, sum aggregation). Native Rust works in native, PyO3, AND WASM — preserving full portability. Load weights from NPZ.

**Alternative: ONNX via `ort`** if native Rust proves too slow to implement. Budget max 2 days on ONNX export attempt. If scatter ops don't export cleanly, abandon and go native. ONNX does not compile to WASM.

### Native Rust Forward Pass (if needed)

```rust
// packages/topo-analyzer/src/rgin.rs (new file, ~1500 lines)

/// Pre-trained R-GIN model for structural embedding inference.
pub struct RGINModel {
    /// Model weights loaded from NPZ bundle
    weights: RGINWeights,
    /// 32×32 bilinear matrix R
    r_matrix: Vec<Vec<f64>>,
    /// Depth probe weights (768d + bias)
    depth_probe_w: Vec<f64>,
    depth_probe_b: f64,
}

impl RGINModel {
    /// Load from model bundle directory.
    pub fn load(bundle_dir: &Path) -> Result<Self, Error> { ... }

    /// Run inference on a single graph.
    /// Returns per-node structural embeddings and reconstruction errors.
    pub fn infer(&self, features: &GraphFeatures) -> RGINOutput { ... }
}

pub struct RGINOutput {
    /// Per-node structural embedding (160d)
    pub z_str: Vec<Vec<f64>>,
    /// Per-node layer-invariant embedding (64d)
    pub z_invariant: Vec<Vec<f64>>,
    /// Per-node per-layer embeddings (32d each)
    pub z_calls: Vec<Vec<f64>>,
    pub z_imports: Vec<Vec<f64>>,
    pub z_inherits: Vec<Vec<f64>>,
    /// Per-node reconstruction error (cosine distance)
    pub reconstruction_error: Vec<f64>,
    /// Graph-level embedding (64d)
    pub g_embedding: Vec<f64>,
}
```

**Native implementation is straightforward** because the model is architecturally simple:
- Linear layers: matrix multiply + bias add
- ReLU: max(0, x)
- Sum aggregation: iterate adjacency list
- GraphNorm: per-graph mean/std (already tracked by graph boundaries)
- No attention, no complex pooling

### Feature Gating

Phase 3 inference is gated behind a Cargo feature flag:

```toml
[features]
rgin = []  # R-GIN inference support
```

When disabled, the analyzer produces Phase 1 + Phase 2 output only (current behavior). When enabled, the analyzer additionally computes R-GIN embeddings and reconstruction errors if a model bundle is available.

### How `analyze_full` Receives the Model

The current `analyze_full(input: &AnalyzerInput) -> AnalysisOutput` takes only input data. The model cannot be serialized into `AnalyzerInput`. Solution: introduce an `Analyzer` struct:

```rust
pub struct Analyzer {
    rgin_model: Option<RGINModel>,  // None if feature disabled or model not found
}

impl Analyzer {
    pub fn new(model_bundle: Option<&Path>) -> Self { ... }
    pub fn analyze_full(&self, input: &AnalyzerInput) -> AnalysisOutput { ... }
}
```

The CLI creates the `Analyzer` once at startup, loading the model bundle if available. The JSON entry point (`analyze_full_json`) wraps this. PyO3/WASM bindings create an Analyzer with `None` model unless explicitly provided. This is a **breaking API change** for direct callers of `analyze_full` but the JSON interface is unchanged.

### Integration into Analysis Pipeline

The R-GIN inference slots into `analyze_full()` between the semantic analysis block and issue synthesis:

```rust
// In analyze_full():

// ... Phase 1 (spectral) ...
// ... Phase 2 (semantic) ...

// Phase 3: R-GIN (if model available and feature enabled)
#[cfg(feature = "rgin")]
let rgin_output = if let Some(model) = &rgin_model {
    let features = prepare_rgin_features(&graph, &spectral_result, &semantic_embeddings);
    Some(model.infer(&features))
} else {
    None
};

// ... Issue synthesis (now with rgin_output) ...
```

---

## 4. Downstream Integration: Anomaly Detection

### Upgraded `misplaced_concern` Detection

Phase 2's centroid-distance heuristic is replaced by the R-GIN's reconstruction error:

```rust
// In issues.rs or semantic.rs:

pub fn detect_misplaced_concerns_phase3(
    rgin_output: &RGINOutput,
    modules: &[EnrichedModule],
    node_to_module: &HashMap<String, usize>,
    roles: &HashMap<String, String>,
) -> Vec<IssueOutput> {
    let mut issues = Vec::new();

    // Compute per-node reconstruction error threshold
    // Use data-adaptive threshold: nodes above p90 + 1.5 * IQR
    let errors = &rgin_output.reconstruction_error;
    let threshold = tukey_upper_fence(errors, 1.5);

    for (i, &error) in errors.iter().enumerate() {
        if error < threshold {
            continue;
        }

        // Skip bridge/hub/utility roles (same suppression as Phase 2)
        let role = roles.get(&node_ids[i]);
        if matches!(role.map(|r| r.as_str()), Some("bridge" | "hub" | "utility")) {
            continue;
        }

        // Identify which module the node should be in:
        // Find the module whose z_invariant centroid is closest to this node's z_invariant
        let best_module = find_closest_module(
            &rgin_output.z_invariant[i],
            modules,
            &rgin_output.z_invariant,
        );

        let current_module = node_to_module.get(&node_ids[i]);
        if best_module == current_module {
            continue;  // High error but already in best module — unusual node, not misplaced
        }

        issues.push(IssueOutput {
            kind: "misplaced_concern".to_string(),
            title: format!("{} may belong in module {}", node_ids[i], best_module_label),
            severity: compute_severity(error, threshold),
            confidence: compute_confidence(error, &rgin_output),
            // ... evidence with per-layer breakdown
        });
    }

    issues
}
```

**Key improvement over Phase 2:** Multi-hop context (2-layer GIN), per-layer structural information, cross-repo calibration. Phase 2's centroid distance is 1-hop, single-layer, and per-repo.

**Phase 2 suppression:** When Phase 3 is active, suppress Phase 2's `misplaced_concern` issues to avoid duplicates. In `lib.rs`, gate the Phase 2 `detect_misplaced_concerns()` call behind `if rgin_output.is_none()`. Both paths produce `kind: "misplaced_concern"` issues — they must not coexist.

### New Diagnostic: `coupling_mismatch`

Phase 3 enables a new diagnostic that Phase 2 cannot do — detecting cross-layer structural tension.

**Design constraint:** z_calls, z_imports, and z_inherits live in different learned 32d spaces (the HSIC decorrelation loss trains them apart). Direct cosine similarity between them is meaningless. Instead, detect coupling mismatch **indirectly** by comparing per-layer *roles*:

1. For each edge type, cluster z_calls, z_imports, z_inherits independently (k-means with the same k as the module count).
2. For each node, check whether its per-layer cluster assignments agree. A node in calls-cluster-3, imports-cluster-3, and inherits-cluster-7 has inherits-layer tension.
3. Emit `coupling_mismatch` for nodes where ≥2 layer-cluster assignments disagree with the majority.

This approach respects the decorrelated spaces by comparing cluster *identities* (aligned via NMI with the structural module assignment) rather than raw vector distances.
```

---

## 5. Downstream Integration: Health Score

### Updated Health Dimensions

The health score (capabilities/HEALTH.md) gains Phase 3 signals:

**Coherence dimension** (structure predicts semantics):

```
coherence = 1 - median(reconstruction_error)  // Phase 3 replaces Phase 2's semantic_smoothness
// median, not mean — mean is dominated by outliers (per HEALTH.md)
```

The reconstruction error is a direct, learned measure of structural-semantic alignment — strictly better than the Rayleigh quotient (a fixed mathematical proxy).

**Flow dimension** (DAG property + layer conformance):

```
layer_conformance = 1 - mean(direction_surprise)

direction_surprise(u→v) = |σ(z_imports(u)^T R z_imports(v)) - σ(z_imports(v)^T R z_imports(u))|
```

The R matrix (from Loss 2) enables per-edge direction surprise — how much the model thinks each call edge goes against the expected direction. This replaces Phase 2's binary layer violation counting with a continuous, calibrated signal.

**Semantic depth probe** (new):

```
depth_sem(M) = w^T · centroid_sem(M) + b
```

For each module M, compute its semantic depth (predicted layer position from CodeLM centroid). Compare against the structural layer position from edge-majority inference. Disagreement indicates semantic-structural layer mismatch.

### Health Output Schema Extension

```rust
pub struct HealthOutput {
    // Existing
    pub modularity_q: Option<f64>,
    pub spectral_coverage_ratio: f64,
    pub self_edge_drop_ratio: f64,
    pub semantic_smoothness: Option<f64>,
    pub semantic_structural_ami: Option<f64>,
    pub semantic_energy_profile: Option<SemanticEnergyProfile>,

    // Phase 3 additions
    pub coherence: Option<f64>,           // 1 - mean(reconstruction_error)
    pub flow: Option<f64>,                // cycle_freedom × layer_conformance
    pub layer_conformance: Option<f64>,   // 1 - mean(direction_surprise)
    pub mean_reconstruction_error: Option<f64>,
    pub g_embedding: Option<Vec<f64>>,    // 64d graph-level fingerprint
}
```

---

## 6. Downstream Integration: Formatter

### Updated `--format context` (LLM Context Narrative)

The LLM context narrative gains per-layer role descriptions:

```
## Module: auth_handler
- Calls role: hub (receives calls from 12 modules)
- Imports role: bridge (imports from both domain and infrastructure)
- Inherits role: leaf (no trait implementations)
- Structural tension: calls and imports roles disagree (coupling_mismatch)
```

### Updated `--format json` Schema

Extend `analysis.schema.json` with Phase 3 fields:

```json
{
  "roles": [{
    "node_id": "...",
    "role": "hub",
    "reconstruction_error": 0.23,
    "z_invariant": [0.1, -0.3, ...],
    "z_calls_role": "hub",
    "z_imports_role": "bridge",
    "z_inherits_role": "leaf"
  }],
  "health": {
    "coherence": 0.78,
    "flow": 0.85,
    "layer_conformance": 0.91,
    "g_embedding": [0.1, -0.3, ...]
  }
}
```

### Updated `--format domain`

The domain model extraction (topo-formatter/domain.rs) benefits from Phase 3:

**Better bounded contexts:** Replace spectral clustering with z_invariant clustering for module detection. z_invariant embeddings are cross-repo calibrated, producing more consistent bounded context boundaries.

**Aggregate root identification:** Use per-layer hub analysis: a node that is a hub in z_calls (many callers) but NOT in z_inherits (not a base class) is a candidate aggregate root — it's a domain-level coordination point, not a type hierarchy node.

**Relationship type annotation:** Cross-module dependencies annotated with per-layer breakdown: "module A depends on module B primarily through calls (not imports)" gives richer domain model edges than undifferentiated coupling weight.

---

## 7. Cross-Repository Comparison (Future)

Phase 3 enables two new capabilities via `g_embedding` (64d graph-level fingerprint):
- **Architecture style classification:** k-NN on g_embedding against training corpus. Post-hoc, no extra training.
- **Structural drift tracking:** Compare g_embeddings across git snapshots via the existing `topo health` command.

These are downstream features that depend on a working trained model. Implement after the core pipeline is validated.

---

## 8. Model Distribution

### Model Bundle Location

The trained model bundle is stored at:
- **System:** `~/.topo/models/rgin-v1/` (downloaded on first use)
- **Project-local:** `.topo/models/rgin-v1/` (for pinned versions)
- **CI/CD:** Passed via `--model-bundle <path>` flag

### Version Pinning

The model bundle includes `config.json` with a version string. The analyzer checks compatibility:

```rust
const SUPPORTED_MODEL_VERSIONS: &[&str] = &["rgin-v1"];

fn check_model_compatibility(bundle: &ModelBundle) -> Result<(), Error> {
    if !SUPPORTED_MODEL_VERSIONS.contains(&bundle.version.as_str()) {
        return Err(Error::IncompatibleModel {
            got: bundle.version.clone(),
            supported: SUPPORTED_MODEL_VERSIONS.to_vec(),
        });
    }
    Ok(())
}
```

Model bundle is ~7.5MB. Store at `~/.topo/models/rgin-v1/`. Pass `--model-bundle <path>` for CI/CD. Fall back to Phase 1+2 if unavailable.

### Model Size

```
rgin_weights.npz:     ~7.4 MB (1.85M params × 4 bytes)
R.npy:                ~4 KB (32×32 × 4 bytes)
depth_probe_w.npy:    ~3 KB (768 × 4 bytes)
depth_probe_b.npy:    ~4 bytes
config.json:          ~500 bytes
metadata.json:        ~1 KB
node_type_vocab.json: ~200 bytes
────────────────────────────────────
Total:                ~7.5 MB
```

Easily distributable. Can be bundled with the binary or downloaded on first use.

---

## 9. Graceful Degradation

**Phase 3 outputs extend, never replace, Phase 1/2.** If the R-GIN model is unavailable, the analyzer produces Phase 1+2 output exactly as today. Phase 3 fields (`reconstruction_error`, `z_*_role`, `coherence`, `flow`, `g_embedding`, `coupling_mismatch`) are null/absent when the model is not loaded.

---

## 10. Evaluation Pipeline Package

```
packages/topo-eval/
  pyproject.toml
  src/topo_eval/
    __init__.py
    tier1.py               # Intrinsic metrics (from training)
    tier2.py               # Phase 2 agreement
    tier3.py               # Synthetic perturbation
    tier4.py               # Structural consistency checks
    baselines.py           # Ablation baselines
    gate.py                # Go/no-go decision
    report.py              # Evaluation report generation
  scripts/
    run_evaluation.py      # Full evaluation pipeline
    generate_phase2.py     # Pre-compute Phase 2 results for val set
  tests/
    test_perturbation.py   # Perturbation test correctness
    test_baselines.py      # Baseline implementations
```

---

## 11. End-to-End Workflow

```
Step 0: Codebase Prep (Rust)
  └─ topo export-features produces NPZ files

Step 1: Dataset Pipeline (Python)
  └─ 500-2000 repos → NPZ features → PyG DataLoaders

Step 2: Model Training (Python/PyTorch)
  └─ R-GIN trained → checkpoint + model bundle

Step 3: Evaluation & Integration
  ├─ Evaluation (Python):
  │   ├─ Run Tier 1-4 evaluation on val/test set
  │   ├─ Compare against ablation baselines
  │   └─ Go/no-go gate decision
  │
  ├─ Inference Integration (Rust):
  │   ├─ Load model bundle
  │   ├─ R-GIN forward pass (ONNX or native)
  │   └─ Compute reconstruction_error, per-layer embeddings
  │
  └─ Downstream Wiring (Rust):
      ├─ Upgraded misplaced_concern detection
      ├─ New coupling_mismatch diagnostic
      ├─ Updated health score (coherence, flow)
      ├─ Updated formatter (per-layer roles, g_embedding)
      └─ Architecture style classification
```

---

## 12. Definition of Done

### Evaluation
- [ ] Tier 1 intrinsic metrics pass (reconstruction > 0.6, AUC > 0.75).
- [ ] Tier 2 Phase 2 agreement: rank correlation > 0.3.
- [ ] Tier 3 synthetic perturbation: sensitivity > 0.6, specificity > 0.75.
- [ ] Phase 3 beats Phase 2 by ≥ 10pp on perturbation sensitivity.
- [ ] Phase 3 beats all ablation baselines on perturbation sensitivity.
- [ ] Go/no-go gate passes.

### Inference
- [ ] Model bundle loads in Rust (ONNX or native).
- [ ] R-GIN inference produces correct-shape outputs.
- [ ] Reconstruction error matches Python output to within 1e-4.
- [ ] Feature flag `rgin` gates Phase 3 code paths.
- [ ] Graceful degradation: analyzer works without model bundle.

### Downstream
- [ ] `misplaced_concern` uses reconstruction error when model available.
- [ ] `coupling_mismatch` diagnostic emitted for cross-layer disagreements.
- [ ] Health score includes `coherence`, `flow`, `layer_conformance`.
- [ ] `--format context` includes per-layer role descriptions.
- [ ] `--format json` schema extends cleanly (backward compatible).
- [ ] `cargo test --workspace` passes.
- [ ] No regressions in Phase 1/Phase 2 behavior when Phase 3 is disabled.

### Process
- [ ] Evaluation report generated and reviewed.
- [ ] Model bundle versioned and distributable.
- [ ] `topo health` supports g_embedding-based drift tracking.
