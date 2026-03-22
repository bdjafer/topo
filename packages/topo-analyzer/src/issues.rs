//! Issue synthesis: spec-compliant diagnostics from structural analysis.
//!
//! Implements diagnostics defined in capabilities/ISSUES.md.

use std::collections::{HashMap, HashSet};

use crate::anomalies::{Anomaly, AnomalyKind};
use crate::graph::Graph;
use crate::modules::EnrichedModule;
use crate::spectral;
use crate::stats;
use crate::types::{IssueOutput, PackageAgreementOutput, RoleOutput};

/// Context for building issues.
pub struct IssuesContext<'a> {
    pub graph: &'a Graph,
    pub modules: &'a [EnrichedModule],
    pub roles: &'a [RoleOutput],
    pub anomalies: &'a [Anomaly],
    pub silhouette: Option<f64>,
    pub package_fallback: bool,
    pub node_to_module: &'a HashMap<String, usize>,
    pub package_agreement: Option<&'a PackageAgreementOutput>,
    /// Semantic embeddings for Phase 2 diagnostics (cross-package-coupling).
    pub semantic_embeddings: Option<&'a HashMap<String, Vec<f32>>>,
    /// Strongly connected components for suppression rules.
    pub sccs: &'a [Vec<String>],
    /// Null coherence threshold from semantic permutation test (Phase 2).
    /// Mean pairwise cosine similarity of random groups. Used as baseline
    /// for cross-package-coupling root cause classification.
    pub null_coherence_threshold: Option<f64>,
    /// Standard deviation of null coherence distribution.
    pub null_coherence_std: Option<f64>,
}

pub fn build_issues(ctx: &IssuesContext) -> Vec<IssueOutput> {
    let mut issues: Vec<IssueOutput> = Vec::new();

    // 1. Cycle member (from anomaly detection — Tarjan SCCs).
    for anomaly in ctx.anomalies {
        if anomaly.kind != AnomalyKind::CircularDependency {
            continue;
        }
        let mut ids = anomaly.node_ids.clone();
        ids.sort();
        issues.push(IssueOutput {
            id: format!("circular-dependency:{}", ids.join(",")),
            kind: "circular_dependency".to_string(),
            title: "Dependency cycle".to_string(),
            description: anomaly.description.clone(),
            severity: round2(anomaly.severity),
            severity_label: severity_label(anomaly.severity),
            confidence: round2(anomaly.confidence),
            confidence_label: confidence_label(anomaly.confidence),
            anchors: anomaly.anchors.clone(),
            ..Default::default()
        });
    }

    // 2. Wide interface.
    issues.extend(detect_wide_interfaces(ctx));

    // 3. Cross-package coupling (Phase 2).
    issues.extend(detect_cross_package_coupling(ctx));

    // 4. Near-disconnect (Phase 1+).
    issues.extend(detect_near_disconnect(ctx));

    // 5. Overloaded utility (Phase 1+).
    issues.extend(detect_overloaded_utility(ctx));

    // 6. Layer violations (Phase 1+).
    issues.extend(detect_layer_violations(ctx));

    // Sort by (-severity, -confidence, title).
    issues.sort_by(|a, b| {
        b.severity
            .partial_cmp(&a.severity)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(
                b.confidence
                    .partial_cmp(&a.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
            .then(a.title.cmp(&b.title))
    });
    issues
}


// ---------------------------------------------------------------------------
// Near-disconnect (Phase 1+, Diagnostic 4)
// ---------------------------------------------------------------------------

fn detect_near_disconnect(ctx: &IssuesContext) -> Vec<IssueOutput> {
    let clustered: Vec<&EnrichedModule> = ctx.modules.iter().filter(|m| !m.unassigned).collect();
    if clustered.len() < 2 {
        return Vec::new();
    }

    // Compute per-module Fiedler values.
    // Use spectral minimum (5 nodes) instead of hardcoded 10.
    let mut module_fiedlers: Vec<(usize, f64, Vec<f64>, Vec<usize>)> = Vec::new();
    for m in &clustered {
        let indices: Vec<usize> = m.node_ids.iter()
            .filter_map(|nid| ctx.graph.node_index.get(nid).copied())
            .collect();
        if indices.len() < 5 {
            continue; // Matches spectral::subgraph_fiedler minimum.
        }
        if let Some((fiedler, vector)) = spectral::subgraph_fiedler(ctx.graph, &indices) {
            module_fiedlers.push((m.id, fiedler, vector, indices));
        }
    }

    if module_fiedlers.is_empty() {
        return Vec::new();
    }

    // Per-module null Fiedler from random subgraphs of the same size.
    // For each module, sample 50 random subgraphs, compute Fiedler, take p5.
    // A module is near-disconnected if its Fiedler < p5 of random subgraphs.
    let all_indices: Vec<usize> = (0..ctx.graph.n).collect();
    let fiedler_median = {
        let mut vals: Vec<f64> = module_fiedlers.iter().map(|(_, f, _, _)| *f).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        vals.get(vals.len() / 2).copied().unwrap_or(0.1)
    };

    let mut rng = stats::Rng::new(1337);
    // Scratch buffer for random subgraph sampling (avoids clone per iteration).
    let mut scratch = all_indices.clone();
    let mut issues = Vec::new();
    for (mod_id, fiedler, vector, indices) in &module_fiedlers {
        // Compute per-module null: Fiedler values of 50 random subgraphs of same size.
        let n_m = indices.len();
        let n_null = 50;
        let mut null_fiedlers = Vec::with_capacity(n_null);
        for _ in 0..n_null {
            // Restore scratch to identity ordering, then partial Fisher-Yates.
            for (i, v) in scratch.iter_mut().enumerate() { *v = i; }
            for i in 0..n_m.min(scratch.len()) {
                let j = i + rng.next_usize(scratch.len() - i);
                scratch.swap(i, j);
            }
            let sample: Vec<usize> = scratch[..n_m].to_vec();
            if let Some((f, _)) = spectral::subgraph_fiedler(ctx.graph, &sample) {
                null_fiedlers.push(f);
            }
        }

        // Threshold: p5 of null Fiedler distribution.
        let threshold = if null_fiedlers.len() >= 5 {
            null_fiedlers.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let p5_idx = (null_fiedlers.len() as f64 * 0.05) as usize;
            null_fiedlers.get(p5_idx).copied().unwrap_or(0.01)
        } else {
            0.01
        };

        if *fiedler >= threshold {
            continue;
        }
        let label = clustered.iter()
            .find(|m| m.id == *mod_id)
            .map(|m| m.label.as_str())
            .unwrap_or("unknown");

        // Partition by Fiedler vector sign.
        let positive: Vec<usize> = vector.iter().enumerate()
            .filter(|(_, v)| **v >= 0.0)
            .map(|(i, _)| indices[i])
            .collect();
        let negative: Vec<usize> = vector.iter().enumerate()
            .filter(|(_, v)| **v < 0.0)
            .map(|(i, _)| indices[i])
            .collect();

        if positive.is_empty() || negative.is_empty() {
            continue;
        }

        // Count cut edges.
        let pos_set: HashSet<usize> = positive.iter().copied().collect();
        let neg_set: HashSet<usize> = negative.iter().copied().collect();
        let mut cut_edges = 0usize;
        for &gi in &positive {
            for &(tgt, _) in &ctx.graph.adj[gi] {
                if neg_set.contains(&tgt) {
                    cut_edges += 1;
                }
            }
        }
        for &gi in &negative {
            for &(tgt, _) in &ctx.graph.adj[gi] {
                if pos_set.contains(&tgt) {
                    cut_edges += 1;
                }
            }
        }
        // Each undirected edge counted twice in directed graph.
        let cut_edges = (cut_edges + 1) / 2;

        // Count module-internal edges (not global).
        let idx_set: HashSet<usize> = indices.iter().copied().collect();
        let module_edges: usize = indices.iter()
            .flat_map(|&gi| ctx.graph.adj[gi].iter().filter(|&&(tgt, _)| idx_set.contains(&tgt)))
            .count();

        // Severity: 0.4*fiedler + 0.3*cut_ratio + 0.3*size_balance
        let fiedler_factor = (1.0 - (fiedler / fiedler_median.max(f64::EPSILON)).clamp(0.0, 1.0)).clamp(0.0, 1.0);
        let s1 = positive.len() as f64;
        let s2 = negative.len() as f64;
        let total_n = s1 + s2;
        let expected_cut = if total_n > 1.0 {
            2.0 * module_edges as f64 * s1 * s2 / (total_n * (total_n - 1.0))
        } else {
            1.0
        };
        let cut_ratio_factor = (1.0 - (cut_edges as f64 / expected_cut.max(1.0)).clamp(0.0, 1.0)).clamp(0.0, 1.0);
        let size_balance_factor = s1.min(s2) / s1.max(s2);

        let sev = (0.4 * fiedler_factor + 0.3 * cut_ratio_factor + 0.3 * size_balance_factor).clamp(0.0, 1.0);

        // Find bridge nodes (incident to cut edges, highest betweenness).
        let bridge_nodes: Vec<String> = {
            let mut bridge_set: HashSet<usize> = HashSet::new();
            for &gi in &positive {
                for &(tgt, _) in &ctx.graph.adj[gi] {
                    if neg_set.contains(&tgt) {
                        bridge_set.insert(gi);
                        bridge_set.insert(tgt);
                    }
                }
            }
            let mut bridges: Vec<usize> = bridge_set.into_iter().collect();
            bridges.sort_by(|&a, &b| {
                let btw_a = ctx.roles.iter().find(|r| r.node_id == ctx.graph.node_ids[a]).map(|r| r.betweenness).unwrap_or(0.0);
                let btw_b = ctx.roles.iter().find(|r| r.node_id == ctx.graph.node_ids[b]).map(|r| r.betweenness).unwrap_or(0.0);
                btw_b.partial_cmp(&btw_a).unwrap_or(std::cmp::Ordering::Equal)
            });
            bridges.iter().take(3).map(|&i| ctx.graph.node_ids[i].clone()).collect()
        };

        issues.push(IssueOutput {
            id: format!("near-disconnect:{}", label),
            kind: "near_disconnect".to_string(),
            title: format!("Near-disconnect in module {}", label),
            description: format!(
                "Module {} is structurally fragile: {} nodes split into partitions of {} and {} \
                 connected by only {} cut edge(s). Fiedler value {:.4} (threshold: {:.4}).",
                label, indices.len(), positive.len(), negative.len(),
                cut_edges, fiedler, threshold
            ),
            severity: round2(sev),
            severity_label: severity_label(sev),
            confidence: 0.8,
            confidence_label: "high".to_string(),
            anchors: bridge_nodes.iter()
                .filter_map(|nid| ctx.graph.node_index.get(nid).map(|&i| ctx.graph.anchor(i)))
                .collect(),
            ..Default::default()
        });
    }
    issues
}

// ---------------------------------------------------------------------------
// Overloaded utility (Phase 1+, Diagnostic 5)
// ---------------------------------------------------------------------------

fn detect_overloaded_utility(ctx: &IssuesContext) -> Vec<IssueOutput> {
    let clustered: Vec<&EnrichedModule> = ctx.modules.iter().filter(|m| !m.unassigned).collect();
    let total_modules = clustered.len();
    if total_modules < 2 {
        return Vec::new();
    }

    // Compute in-degree percentile threshold (p85).
    let in_degrees: Vec<usize> = ctx.roles.iter().map(|r| r.in_degree).collect();
    let mut sorted_in = in_degrees.clone();
    sorted_in.sort();
    let p85_idx = ((sorted_in.len() as f64 * 0.85) as usize).min(sorted_in.len().saturating_sub(1));
    let p85_threshold = sorted_in.get(p85_idx).copied().unwrap_or(1);
    let p99_idx = ((sorted_in.len() as f64 * 0.99) as usize).min(sorted_in.len().saturating_sub(1));
    let p99_threshold = sorted_in.get(p99_idx).copied().unwrap_or(p85_threshold + 1);

    // Pre-compute: betweenness percentiles (hoisted out of per-node loop).
    let btw_values: Vec<f64> = ctx.roles.iter().map(|r| r.betweenness).collect();
    let btw_pcts = stats::percentile_ranks(&btw_values);
    let btw_pct_map: HashMap<&str, f64> = ctx.roles.iter()
        .zip(btw_pcts.iter())
        .map(|(r, &p)| (r.node_id.as_str(), p))
        .collect();

    // Pre-compute: reverse adjacency (predecessors per node) → O(V+E) once.
    let mut predecessors: Vec<Vec<usize>> = vec![Vec::new(); ctx.graph.n];
    for (src, adj) in ctx.graph.adj.iter().enumerate() {
        for &(tgt, _) in adj {
            predecessors[tgt].push(src);
        }
    }

    // Pre-compute: per-module out-degree and total edges (for config-model expected diversity).
    let mut module_out_degree: HashMap<usize, f64> = HashMap::new();
    let mut total_edges = 0.0f64;
    for (src, adj) in ctx.graph.adj.iter().enumerate() {
        if let Some(&src_mod) = ctx.node_to_module.get(&ctx.graph.node_ids[src]) {
            let degree = adj.len() as f64;
            *module_out_degree.entry(src_mod).or_default() += degree;
            total_edges += degree;
        }
    }
    let module_out_degrees: Vec<f64> = clustered.iter()
        .map(|m| module_out_degree.get(&m.id).copied().unwrap_or(0.0))
        .collect();

    // FP suppression: exact last-segment matching to avoid false matches
    // (e.g. "from" matching "transform").
    let suppressed_exact = [
        "log", "trace", "debug", "info", "warn", "error", "println", "print",
        "serialize", "deserialize", "to_json", "from_json", "encode", "decode",
        "new", "create", "build", "from", "default", "init",
    ];

    let mut issues = Vec::new();
    for role in ctx.roles.iter() {
        if role.in_degree <= p85_threshold {
            continue;
        }
        let total_degree = role.in_degree + role.out_degree;
        if total_degree == 0 {
            continue;
        }

        // Direction test: Binomial(T, 0.5) replaces hardcoded direction_score > 0.5.
        let direction_minority = role.in_degree.min(role.out_degree);
        if !stats::direction_is_significant(direction_minority, total_degree, 0.05) {
            continue;
        }
        let direction_score = (role.in_degree as f64 - role.out_degree as f64) / total_degree as f64;
        // Ensure in-degree dominates (not out-degree).
        if role.in_degree <= role.out_degree {
            continue;
        }

        // Caller diversity: config-model expected diversity replaces hardcoded > 0.3.
        let caller_modules: HashSet<usize> = if let Some(&idx) = ctx.graph.node_index.get(&role.node_id) {
            predecessors[idx].iter()
                .filter_map(|&src| ctx.node_to_module.get(&ctx.graph.node_ids[src]).copied())
                .collect()
        } else {
            HashSet::new()
        };
        let observed_diversity = caller_modules.len() as f64;
        let expected_diversity = stats::expected_caller_diversity(
            role.in_degree, &module_out_degrees, total_edges
        );
        let _caller_diversity = caller_modules.len() as f64 / total_modules.max(1) as f64;
        // Gate: observed diversity must exceed expected (node is called by more
        // diverse modules than random wiring would produce).
        if observed_diversity <= expected_diversity {
            continue;
        }

        // FP suppression: exact match on last segment of node_id.
        let short_name = role.node_id.rsplit('.').next().unwrap_or(&role.node_id).to_lowercase();
        if suppressed_exact.iter().any(|&p| short_name == p) {
            continue;
        }

        // Severity: 0.4*degree + 0.3*diversity + 0.3*direction
        let degree_factor = if p99_threshold > p85_threshold {
            ((role.in_degree as f64 - p85_threshold as f64) / (p99_threshold - p85_threshold) as f64).clamp(0.0, 1.0)
        } else {
            0.5
        };
        // Diversity factor: how far above expected (z-score-like).
        let diversity_excess = observed_diversity - expected_diversity;
        let diversity_factor = (diversity_excess / total_modules.max(1) as f64 * 2.0).clamp(0.0, 1.0);
        // Direction factor: from binomial p-value surprisal.
        let dir_p = stats::binomial_cdf(direction_minority, total_degree, 0.5);
        let direction_factor = (-dir_p.max(1e-15).log10() / 3.0).clamp(0.0, 1.0);

        let mut sev = 0.4 * degree_factor + 0.3 * diversity_factor + 0.3 * direction_factor;
        // Multiplier: high betweenness → structural bridge + bottleneck.
        let btw_pct = btw_pct_map.get(role.node_id.as_str()).copied().unwrap_or(0.5);
        if btw_pct > 0.9 {
            sev *= 1.3;
        }
        // Multiplier: in utility/shared module → expected position.
        if let Some(&mod_id) = ctx.node_to_module.get(&role.node_id) {
            let mod_label = ctx.modules.iter().find(|m| m.id == mod_id).map(|m| m.label.as_str()).unwrap_or("");
            if ["utils", "helpers", "common", "shared", "lib"].iter().any(|p| mod_label.contains(p)) {
                sev *= 0.7;
            }
        }
        // Multiplier: pure leaf (out_degree == 0) → less likely to propagate.
        if role.out_degree == 0 {
            sev *= 0.8;
        }
        sev = sev.clamp(0.0, 1.0);

        issues.push(IssueOutput {
            id: format!("overloaded-utility:{}", role.node_id),
            kind: "overloaded_utility".to_string(),
            title: format!("Overloaded utility: {}", role.node_id.rsplit('.').next().unwrap_or(&role.node_id)),
            description: format!(
                "{} is called by {} nodes across {} of {} modules, but only calls {} functions itself. \
                 Direction score: {:.2}.",
                role.node_id, role.in_degree, caller_modules.len(), total_modules, role.out_degree, direction_score
            ),
            severity: round2(sev),
            severity_label: severity_label(sev),
            confidence: 0.7,
            confidence_label: "medium".to_string(),
            anchors: ctx.graph.node_index.get(&role.node_id)
                .map(|&i| vec![ctx.graph.anchor(i)])
                .unwrap_or_default(),
            ..Default::default()
        });
    }
    issues
}

// ---------------------------------------------------------------------------
// Layer violation (Phase 1+, Diagnostic 6)
// ---------------------------------------------------------------------------

fn detect_layer_violations(ctx: &IssuesContext) -> Vec<IssueOutput> {
    let clustered: Vec<&EnrichedModule> = ctx.modules.iter().filter(|m| !m.unassigned).collect();
    if clustered.len() < 2 {
        return Vec::new();
    }

    let mod_ids: Vec<usize> = clustered.iter().map(|m| m.id).collect();

    // Step 1: Build module dependency graph (edge counts per direction).
    let mut edge_counts: HashMap<(usize, usize), usize> = HashMap::new();
    for (kind, edges) in &ctx.graph.typed_edges {
        if kind == "defines" { continue; }
        for &(src, tgt) in edges {
            let src_mod = ctx.node_to_module.get(&ctx.graph.node_ids[src]).copied();
            let tgt_mod = ctx.node_to_module.get(&ctx.graph.node_ids[tgt]).copied();
            if let (Some(sm), Some(tm)) = (src_mod, tgt_mod) {
                if sm != tm {
                    *edge_counts.entry((sm, tm)).or_default() += 1;
                }
            }
        }
    }

    // Step 2: Assign layers via simplified topological sort.
    // Build module-level DAG by majority direction.
    let mut dag_edges: HashMap<usize, Vec<usize>> = HashMap::new(); // from -> [to]
    let mut seen_pairs: HashSet<(usize, usize)> = HashSet::new();

    for &(a, b) in edge_counts.keys() {
        let canonical = if a <= b { (a, b) } else { (b, a) };
        if seen_pairs.contains(&canonical) {
            continue;
        }
        seen_pairs.insert(canonical);

        let fwd = edge_counts.get(&(canonical.0, canonical.1)).copied().unwrap_or(0);
        let rev = edge_counts.get(&(canonical.1, canonical.0)).copied().unwrap_or(0);

        if fwd == 0 && rev == 0 {
            continue;
        }
        // Majority direction becomes the DAG edge (higher → lower).
        if fwd >= rev {
            dag_edges.entry(canonical.0).or_default().push(canonical.1);
        } else {
            dag_edges.entry(canonical.1).or_default().push(canonical.0);
        }
    }

    // Assign layers via longest-path from sources (Kahn's-like).
    let mut in_deg: HashMap<usize, usize> = HashMap::new();
    for &m in &mod_ids {
        in_deg.insert(m, 0);
    }
    for targets in dag_edges.values() {
        for &t in targets {
            *in_deg.entry(t).or_default() += 1;
        }
    }

    let mut queue: Vec<usize> = mod_ids.iter().filter(|&&m| in_deg.get(&m).copied().unwrap_or(0) == 0).copied().collect();
    let mut layers: HashMap<usize, usize> = HashMap::new();
    // BFS for longest path (layer = max layer of predecessors + 1).
    let mut visited = 0;
    while !queue.is_empty() {
        let mut next_queue = Vec::new();
        for m in &queue {
            if !layers.contains_key(m) {
                layers.insert(*m, 0);
            }
            let my_layer = layers[m];
            if let Some(targets) = dag_edges.get(m) {
                for &t in targets {
                    let new_layer = my_layer + 1;
                    let entry = layers.entry(t).or_insert(0);
                    if new_layer > *entry {
                        *entry = new_layer;
                    }
                    let d = in_deg.entry(t).or_insert(1);
                    *d = d.saturating_sub(1);
                    if *d == 0 {
                        next_queue.push(t);
                    }
                }
            }
        }
        queue = next_queue;
        visited += 1;
        if visited > mod_ids.len() + 10 {
            break; // Safety: avoid infinite loop on cycles.
        }
    }
    // Assign remaining unvisited modules (in cycles) to layer 0.
    for &m in &mod_ids {
        layers.entry(m).or_insert(0);
    }

    let max_layer = layers.values().max().copied().unwrap_or(1).max(1);

    // Pre-compute sorted edge counts for percentile rank (hoisted out of loop).
    let sorted_edge_counts: Vec<f64> = {
        let mut s: Vec<f64> = edge_counts.values().map(|&c| c as f64).collect();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        s
    };

    // Step 3: Flag violations.
    // Layer 0 = orchestrators/callers (DAG sources), higher layers = callees.
    // Upward violation = callee (high layer) calling orchestrator (low layer):
    //   src_layer > tgt_layer means src is deeper but has edges to a shallower target.
    let mut issues = Vec::new();
    for (&(src_mod, tgt_mod), &count) in &edge_counts {
        let src_layer = layers.get(&src_mod).copied().unwrap_or(0);
        let tgt_layer = layers.get(&tgt_mod).copied().unwrap_or(0);

        // Only flag upward edges (deeper layer → shallower layer).
        if src_layer <= tgt_layer {
            continue;
        }

        // Statistical test: is the directional asymmetry significant?
        // Replaces hardcoded minority_ratio < 0.4 with Binomial(T, 0.5) test.
        let reverse = edge_counts.get(&(tgt_mod, src_mod)).copied().unwrap_or(0);
        let total = count + reverse;
        let minority = count.min(reverse);
        if !stats::direction_is_significant(minority, total, 0.05) {
            continue; // Asymmetry not significant — could be random direction.
        }

        // Suppression: both modules in same SCC → circular-dependency covers it.
        let src_label = clustered.iter().find(|m| m.id == src_mod).map(|m| m.label.as_str()).unwrap_or("?");
        let tgt_label = clustered.iter().find(|m| m.id == tgt_mod).map(|m| m.label.as_str()).unwrap_or("?");

        let in_same_scc = ctx.sccs.iter().any(|scc| {
            let has_src = scc.iter().any(|nid| ctx.node_to_module.get(nid) == Some(&src_mod));
            let has_tgt = scc.iter().any(|nid| ctx.node_to_module.get(nid) == Some(&tgt_mod));
            has_src && has_tgt
        });
        if in_same_scc {
            continue;
        }

        // Severity: 0.4*asymmetry + 0.3*layer_gap + 0.3*edge_count
        // asymmetry_factor: derived from binomial p-value surprisal.
        let p_val = stats::binomial_cdf(minority, total, 0.5);
        let asymmetry_factor = (-p_val.max(1e-15).log10() / 3.0).clamp(0.0, 1.0);
        let layer_gap = (src_layer as f64 - tgt_layer as f64) / max_layer as f64;
        let layer_gap_factor = layer_gap.clamp(0.0, 1.0);
        // edge_count_factor: percentile rank among all inter-module edge counts.
        let edge_count_factor = {
            let rank = sorted_edge_counts.partition_point(|&x| x < count as f64);
            if sorted_edge_counts.len() > 1 {
                rank as f64 / (sorted_edge_counts.len() - 1) as f64
            } else {
                0.5
            }
        };

        let sev = (0.4 * asymmetry_factor + 0.3 * layer_gap_factor + 0.3 * edge_count_factor).clamp(0.0, 1.0);
        // Confidence from Cohen's h (effect size for proportion test).
        let majority_ratio = 1.0 - (minority as f64 / total.max(1) as f64);
        let h = 2.0 * (majority_ratio.sqrt().asin() - (0.5f64).sqrt().asin());
        let conf = (0.5 + 0.5 * (h / 0.8)).clamp(0.5, 1.0);

        issues.push(IssueOutput {
            id: format!("layer-violation:{}_{}", src_label, tgt_label),
            kind: "layer_violation".to_string(),
            title: format!("Layer violation: {} → {}", src_label, tgt_label),
            description: format!(
                "{} edge(s) from {} (layer {}) up to {} (layer {}). \
                 The dominant direction is {} → {} ({} edges). Minority ratio: {:.2}.",
                count, src_label, src_layer, tgt_label, tgt_layer,
                tgt_label, src_label, reverse, minority as f64 / total.max(1) as f64
            ),
            severity: round2(sev),
            severity_label: severity_label(sev),
            confidence: round2(conf),
            confidence_label: confidence_label(conf),
            anchors: Vec::new(),
            ..Default::default()
        });
    }
    issues
}

// ---------------------------------------------------------------------------
// Wide interface
// ---------------------------------------------------------------------------

fn detect_wide_interfaces(ctx: &IssuesContext) -> Vec<IssueOutput> {
    let clustered: Vec<&EnrichedModule> = ctx.modules.iter().filter(|m| !m.unassigned).collect();
    if clustered.len() < 2 {
        return Vec::new();
    }

    // Count distinct (src, tgt) symbol pairs per DIRECTIONAL module pair (spec requirement).
    let mut dir_symbols: HashMap<(usize, usize), HashSet<(usize, usize)>> = HashMap::new();
    // Also track per-target in-degree for concentration factor.
    let mut target_counts: HashMap<(usize, usize), HashMap<usize, usize>> = HashMap::new();

    for (kind, edges) in &ctx.graph.typed_edges {
        if kind == "defines" { continue; }
        for &(src, tgt) in edges {
            let src_mod = ctx.node_to_module.get(&ctx.graph.node_ids[src]);
            let tgt_mod = ctx.node_to_module.get(&ctx.graph.node_ids[tgt]);
            let (Some(&sm), Some(&tm)) = (src_mod, tgt_mod) else { continue };
            if sm == tm {
                continue;
            }
            dir_symbols.entry((sm, tm)).or_default().insert((src, tgt));
            *target_counts.entry((sm, tm)).or_default().entry(tgt).or_default() += 1;
        }
    }

    if dir_symbols.is_empty() {
        return Vec::new();
    }

    // Threshold: pure Tukey fence on all directional widths (no hardcoded floor).
    // Gate: need at least 6 nonzero pairs for quartile estimates to be reliable.
    let all_widths: Vec<f64> = dir_symbols.values()
        .map(|s| s.len() as f64)
        .filter(|&w| w > 0.0)
        .collect();
    if all_widths.len() < 6 {
        return Vec::new(); // Too few module pairs for meaningful outlier detection.
    }
    let threshold = stats::tukey_upper_fence(&all_widths) as usize;

    let module_by_id: HashMap<usize, &EnrichedModule> = ctx
        .modules
        .iter()
        .filter(|m| !m.unassigned)
        .map(|m| (m.id, m))
        .collect();

    // Deduplicate: emit one issue per unordered pair (use the wider direction).
    let mut seen_pairs: HashSet<(usize, usize)> = HashSet::new();
    let mut issues = Vec::new();

    for (&(mod_a, mod_b), syms) in &dir_symbols {
        let width_a_to_b = syms.len();
        if width_a_to_b <= threshold {
            continue;
        }

        let canonical = if mod_a <= mod_b { (mod_a, mod_b) } else { (mod_b, mod_a) };
        if seen_pairs.contains(&canonical) {
            continue;
        }
        seen_pairs.insert(canonical);

        let width_b_to_a = dir_symbols.get(&(mod_b, mod_a)).map(|s| s.len()).unwrap_or(0);

        let label_a = module_by_id.get(&mod_a).map(|m| m.label.as_str()).unwrap_or("?");
        let label_b = module_by_id.get(&mod_b).map(|m| m.label.as_str()).unwrap_or("?");

        // Spec severity: 0.5*excess + 0.3*asymmetry + 0.2*concentration
        let excess_factor = ((width_a_to_b as f64 - threshold as f64) / threshold.max(1) as f64).clamp(0.0, 1.0);

        let max_width = width_a_to_b.max(width_b_to_a).max(1);
        let min_width = width_a_to_b.min(width_b_to_a);
        let asymmetry_factor = 1.0 - (min_width as f64 / max_width as f64);

        // Concentration: max_target_in_degree / width.
        let max_target_in_degree = target_counts.get(&(mod_a, mod_b))
            .and_then(|m| m.values().max().copied())
            .unwrap_or(1);
        let concentration_factor = (max_target_in_degree as f64 / width_a_to_b.max(1) as f64).clamp(0.0, 1.0);

        let mut sev = 0.5 * excess_factor + 0.3 * asymmetry_factor + 0.2 * concentration_factor;

        // Multipliers.
        let reverse_exceeds = width_b_to_a > threshold;
        if reverse_exceeds {
            sev *= 1.3; // Bidirectional wide interface.
        }
        // Same declared package → less alarming.
        let pkg_a = label_a.split('.').next().unwrap_or(label_a);
        let pkg_b = label_b.split('.').next().unwrap_or(label_b);
        if pkg_a == pkg_b {
            sev *= 0.7;
        }
        sev = sev.clamp(0.0, 1.0);

        issues.push(IssueOutput {
            id: format!(
                "wide-interface:{}",
                sorted_join(&[label_a.to_string(), label_b.to_string()])
            ),
            kind: "wide_interface".to_string(),
            title: format!("Wide interface: {} — {}", label_a, label_b),
            description: format!(
                "{} distinct coupling points from {} to {} (threshold: {}). \
                 Reverse: {} points from {} to {}.",
                width_a_to_b, label_a, label_b, threshold,
                width_b_to_a, label_b, label_a
            ),
            severity: round2(sev),
            severity_label: severity_label(sev),
            confidence: 0.6,
            confidence_label: "medium".to_string(),
            anchors: Vec::new(),
            ..Default::default()
        });
    }
    issues
}


// ---------------------------------------------------------------------------
// Cross-package coupling
// ---------------------------------------------------------------------------

fn detect_cross_package_coupling(ctx: &IssuesContext) -> Vec<IssueOutput> {
    // Suppress when modules ARE the packages (package fallback).
    if ctx.package_fallback {
        return Vec::new();
    }
    let agreement = match ctx.package_agreement {
        Some(a) => a,
        None => return Vec::new(),
    };

    let mut issues = Vec::new();

    for comp in &agreement.module_composition {
        if !comp.cross_package {
            continue;
        }
        // Find module and its members.
        let module = ctx.modules.iter().find(|m| m.id == comp.module_id);
        let label = module.map(|m| m.label.as_str()).unwrap_or("unknown");
        let members = module.map(|m| &m.node_ids);

        // Build description of package breakdown.
        let mut pkg_parts: Vec<(&String, &usize)> = comp.packages.iter().collect();
        pkg_parts.sort_by(|a, b| b.1.cmp(a.1));
        let breakdown: Vec<String> = pkg_parts
            .iter()
            .map(|(pkg, count)| format!("{pkg} ({count})"))
            .collect();
        let total: usize = comp.packages.values().sum();

        // Structural signal: how evenly split across packages.
        let max_count = comp.packages.values().max().copied().unwrap_or(0);
        let cross_fraction = 1.0 - (max_count as f64 / total.max(1) as f64);

        // Semantic confidence calibration (Phase 2).
        let (semantic_coherence, root_cause) =
            compute_cross_package_semantics(ctx, members, ctx.null_coherence_threshold, ctx.null_coherence_std);

        // Suppress spectral artifacts: coherence below null + low silhouette.
        // Instead of hardcoded 0.2/0.3, compare against the data-derived null.
        let module_silhouette = ctx.silhouette.unwrap_or(0.0);
        let null_coh = ctx.null_coherence_threshold.unwrap_or(0.0);
        if semantic_coherence < null_coh && module_silhouette < 0.3 {
            continue;
        }

        // Severity model (spec: 0.3 span + 0.3 balance + 0.2 density + 0.2 semantic).
        let span_factor =
            ((comp.packages.len() as f64 - 1.0) / 4.0).clamp(0.0, 1.0);
        let balance_factor = cross_fraction;

        // Density factor: fraction of module-internal edges that cross packages.
        // Also compute bidirectionality ratio for directionality discount.
        let (density_factor, module_bidirectionality) = if let Some(members) = members {
            let member_set: HashSet<&str> = members.iter().map(|s| s.as_str()).collect();
            let mut total_internal = 0usize;
            let mut cross_pkg = 0usize;
            // Track per-direction counts for each ordered package pair.
            let mut pkg_pair_counts: HashMap<(&str, &str), usize> = HashMap::new();
            for nid in members {
                if let Some(&idx) = ctx.graph.node_index.get(nid.as_str()) {
                    for &(tgt, _) in &ctx.graph.adj[idx] {
                        if member_set.contains(ctx.graph.node_ids[tgt].as_str()) {
                            total_internal += 1;
                            let src_pkg = nid.split('.').next().unwrap_or("");
                            let tgt_pkg = ctx.graph.node_ids[tgt].split('.').next().unwrap_or("");
                            if src_pkg != tgt_pkg {
                                cross_pkg += 1;
                                *pkg_pair_counts.entry((src_pkg, tgt_pkg)).or_insert(0) += 1;
                            }
                        }
                    }
                }
            }
            let dens = if total_internal > 0 {
                (cross_pkg as f64 / total_internal as f64).clamp(0.0, 1.0)
            } else {
                cross_fraction
            };
            // Compute weighted bidirectionality across all package pairs.
            // For each unordered pair {Pi, Pj}, bidir = min(fwd,rev)/max(fwd,rev).
            // Weight by max(fwd, rev) so pairs with more edges count more.
            let mut seen_pairs: HashSet<(&str, &str)> = HashSet::new();
            let mut bidir_weighted_sum = 0.0_f64;
            let mut bidir_weight_total = 0.0_f64;
            for (&(p1, p2), &fwd) in &pkg_pair_counts {
                let canonical = if p1 < p2 { (p1, p2) } else { (p2, p1) };
                if !seen_pairs.insert(canonical) {
                    continue;
                }
                let rev = pkg_pair_counts.get(&(p2, p1)).copied().unwrap_or(0);
                let max_dir = fwd.max(rev) as f64;
                let min_dir = fwd.min(rev) as f64;
                if max_dir > 0.0 {
                    let pair_bidir = min_dir / max_dir;
                    bidir_weighted_sum += pair_bidir * max_dir;
                    bidir_weight_total += max_dir;
                }
            }
            let bidir = if bidir_weight_total > 0.0 {
                bidir_weighted_sum / bidir_weight_total
            } else {
                0.0
            };
            (dens, bidir)
        } else {
            (cross_fraction, 0.5) // no members → assume moderate bidirectionality
        };

        let semantic_factor = (semantic_coherence / 0.6).clamp(0.0, 1.0);

        let mut sev =
            0.3 * span_factor + 0.3 * balance_factor + 0.2 * density_factor + 0.2 * semantic_factor;

        // Directionality discount: reduce severity for unidirectional (layered) coupling.
        // bidir=0.0 → factor=0.4 (60% reduction — clean layered composition)
        // bidir=0.5 → factor=0.7 (30% reduction — mixed)
        // bidir=1.0 → factor=1.0 (no reduction — tangled/bidirectional)
        let directionality_factor = 0.4 + 0.6 * module_bidirectionality;
        sev *= directionality_factor;

        // Root cause multipliers.
        match root_cause.as_str() {
            "shared_domain_concept" => sev *= 1.3,
            "accidental_coupling" => sev *= 0.7,
            _ => {}
        }
        if module_silhouette > 0.5 {
            sev *= 1.2;
        }
        sev = sev.min(1.0);

        // Severity floor: don't emit low-signal cross-package findings.
        if sev < 0.35 {
            continue;
        }

        // Confidence model: use effect size relative to null, not hardcoded thresholds.
        let has_semantic = root_cause != "structural_only";
        let mut confidence: f64 = 0.5;
        if has_semantic {
            let d = stats::cohens_d(
                semantic_coherence,
                ctx.null_coherence_threshold.unwrap_or(0.0),
                ctx.null_coherence_std.unwrap_or(0.1).max(f64::EPSILON),
            );
            if d > 0.8 {
                confidence += 0.3;
            } else if d > 0.2 {
                confidence += 0.15;
            }
        }
        if module_silhouette > 0.4 {
            confidence += 0.1;
        }
        let cross_count = total - max_count;
        if cross_count >= 3 {
            confidence += 0.1;
        }
        confidence = confidence.clamp(0.3, 1.0);
        // Spec: cap at 0.5 when no semantic embeddings.
        if !has_semantic {
            confidence = confidence.min(0.5);
        }

        // Description includes root cause when semantic data is available.
        let root_cause_desc = match root_cause.as_str() {
            "shared_domain_concept" => format!(
                " Cross-package nodes are semantically coherent (similarity: {:.2}), \
                 suggesting a shared domain concept split across package boundaries.",
                semantic_coherence
            ),
            "accidental_coupling" => format!(
                " Cross-package nodes are semantically dissimilar (similarity: {:.2}), \
                 suggesting accidental coupling through a shared dependency.",
                semantic_coherence
            ),
            "incomplete_split" => format!(
                " Moderate semantic similarity ({:.2}) suggests a package split \
                 that hasn't been fully decoupled.",
                semantic_coherence
            ),
            _ => String::new(),
        };

        let directionality_desc = if module_bidirectionality < 0.15 {
            String::new() // near-zero bidirectionality — don't clutter the message
        } else if module_bidirectionality < 0.4 {
            format!(
                " Coupling is mostly unidirectional ({:.0}% bidirectional).",
                module_bidirectionality * 100.0
            )
        } else {
            format!(
                " Coupling is bidirectional ({:.0}% bidirectional), suggesting entanglement rather than layered composition.",
                module_bidirectionality * 100.0
            )
        };

        issues.push(IssueOutput {
            id: format!("cross-package-coupling:module-{}", comp.module_id),
            kind: "cross_package_coupling".to_string(),
            title: format!("Cross-package coupling in module \"{label}\""),
            description: format!(
                "Spectral analysis groups {total} nodes from {} packages into one structural \
                 module: {}.{root_cause_desc}{directionality_desc}",
                comp.packages.len(),
                breakdown.join(", ")
            ),
            severity: round2(sev),
            severity_label: severity_label(sev),
            confidence: round2(confidence),
            confidence_label: confidence_label(confidence),
            anchors: Vec::new(),
            root_cause: Some(root_cause),
            semantic_coherence: if semantic_coherence > 0.0 {
                Some(round2(semantic_coherence))
            } else {
                None
            },
            ..Default::default()
        });
    }
    issues
}

/// Compute semantic coherence and classify root cause for a cross-package module.
///
/// Root cause classification uses Cohen's d (effect size relative to the null
/// coherence distribution) instead of hardcoded thresholds. This adapts to the
/// background similarity level of each codebase.
fn compute_cross_package_semantics(
    ctx: &IssuesContext,
    members: Option<&Vec<String>>,
    null_mean: Option<f64>,
    null_std: Option<f64>,
) -> (f64, String) {
    let embeddings = match ctx.semantic_embeddings {
        Some(e) => e,
        None => return (0.0, "structural_only".to_string()),
    };
    let members = match members {
        Some(m) => m,
        None => return (0.0, "structural_only".to_string()),
    };

    // Collect embeddings for module members.
    let member_embeddings: Vec<&Vec<f32>> = members
        .iter()
        .filter_map(|nid| embeddings.get(nid))
        .collect();

    if member_embeddings.len() < 2 {
        return (0.0, "structural_only".to_string());
    }

    // Mean pairwise cosine similarity (upper triangle).
    let mut sim_sum = 0.0;
    let mut pair_count = 0usize;
    for i in 0..member_embeddings.len() {
        for j in (i + 1)..member_embeddings.len() {
            sim_sum += cosine_similarity_f32(member_embeddings[i], member_embeddings[j]);
            pair_count += 1;
        }
    }
    let coherence = if pair_count > 0 {
        sim_sum / pair_count as f64
    } else {
        0.0
    };

    // Root cause classification via Cohen's d against null distribution.
    // d > 0.8 = large effect (shared domain concept)
    // d > 0.2 = small-to-medium effect (incomplete split)
    // d <= 0.2 = negligible (accidental coupling)
    let d = stats::cohens_d(
        coherence,
        null_mean.unwrap_or(0.0),
        null_std.unwrap_or(0.1).max(f64::EPSILON),
    );
    let root_cause = if d > 0.8 {
        "shared_domain_concept"
    } else if d > 0.2 {
        "incomplete_split"
    } else {
        "accidental_coupling"
    };

    (coherence, root_cause.to_string())
}

/// Cosine similarity between two f32 vectors.
fn cosine_similarity_f32(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let x = *x as f64;
        let y = *y as f64;
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < f64::EPSILON {
        0.0
    } else {
        dot / denom
    }
}

fn severity_label(score: f64) -> String {
    if score >= 0.75 {
        "high".to_string()
    } else if score >= 0.45 {
        "medium".to_string()
    } else {
        "low".to_string()
    }
}

fn confidence_label(score: f64) -> String {
    if score >= 0.75 {
        "high".to_string()
    } else if score >= 0.45 {
        "medium".to_string()
    } else {
        "low".to_string()
    }
}

fn sorted_join(items: &[String]) -> String {
    let mut sorted = items.to_vec();
    sorted.sort();
    sorted.join(",")
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}
