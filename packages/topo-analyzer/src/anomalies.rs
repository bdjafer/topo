//! Structural anomaly detection.
//!
//! Identifies structural anomalies: cross-module boundary violations,
//! spectral outliers, dependency cycles, and cross-layer discrepancies.
//!
//! Ports logic from Python topo_analyzer.anomalies.

use std::collections::{HashMap, HashSet};

use crate::graph::Graph;
use crate::modules::EnrichedModule;
use crate::types::AnchorOutput;

/// Internal anomaly representation.
pub struct Anomaly {
    pub kind: AnomalyKind,
    pub node_ids: Vec<String>,
    pub description: String,
    pub severity: f64,
    pub confidence: f64,
    pub anchors: Vec<AnchorOutput>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnomalyKind {
    CrossModule,
    SpectralOutlier,
    CycleMember,
    LayerDiscrepancy,
}

impl AnomalyKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            AnomalyKind::CrossModule => "cross_module",
            AnomalyKind::SpectralOutlier => "spectral_outlier",
            AnomalyKind::CycleMember => "cycle_member",
            AnomalyKind::LayerDiscrepancy => "layer_discrepancy",
        }
    }
}

/// Run all anomaly detectors.
pub fn detect_all(
    graph: &Graph,
    modules: &[EnrichedModule],
    fingerprints: &HashMap<String, Vec<f64>>,
    sccs: &[Vec<String>],
    edge_kinds: &[String],
    package_fallback: bool,
) -> Vec<Anomaly> {
    let node_to_module = build_node_to_module(modules);
    let mut anomalies = Vec::new();

    anomalies.extend(detect_cross_module(graph, modules, &node_to_module));
    anomalies.extend(detect_spectral_outliers(
        modules,
        fingerprints,
        graph,
        &node_to_module,
        package_fallback,
    ));
    anomalies.extend(sccs_to_anomalies(sccs, graph));
    anomalies.extend(detect_layer_discrepancies(graph, edge_kinds));

    anomalies.sort_by(|a, b| {
        b.severity
            .partial_cmp(&a.severity)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(
                b.confidence
                    .partial_cmp(&a.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
    });
    anomalies
}

fn build_node_to_module(modules: &[EnrichedModule]) -> HashMap<String, usize> {
    let mut map = HashMap::new();
    for m in modules {
        if m.unassigned {
            continue;
        }
        for nid in &m.node_ids {
            map.insert(nid.clone(), m.id);
        }
    }
    map
}

// ---------------------------------------------------------------------------
// Cross-module boundary violations
// ---------------------------------------------------------------------------

fn detect_cross_module(
    graph: &Graph,
    modules: &[EnrichedModule],
    node_to_module: &HashMap<String, usize>,
) -> Vec<Anomaly> {
    if modules.is_empty() {
        return Vec::new();
    }
    let unassigned: HashSet<usize> = modules.iter().filter(|m| m.unassigned).map(|m| m.id).collect();

    // Aggregate directed cross-module edge counts.
    let mut pair_counts: HashMap<(usize, usize), HashMap<String, usize>> = HashMap::new();
    let mut pair_examples: HashMap<(usize, usize), Vec<(usize, usize)>> = HashMap::new();

    for (kind, edges) in &graph.typed_edges {
        for &(src, tgt) in edges {
            let src_mod = node_to_module.get(&graph.node_ids[src]);
            let tgt_mod = node_to_module.get(&graph.node_ids[tgt]);
            let (Some(&sm), Some(&tm)) = (src_mod, tgt_mod) else { continue };
            if sm == tm || unassigned.contains(&sm) || unassigned.contains(&tm) {
                continue;
            }
            *pair_counts
                .entry((sm, tm))
                .or_default()
                .entry(kind.clone())
                .or_default() += 1;
            let ex = pair_examples.entry((sm, tm)).or_default();
            if ex.len() < 3 {
                ex.push((src, tgt));
            }
        }
    }

    // Median of directed pair totals.
    let directed_totals: Vec<f64> = pair_counts
        .values()
        .map(|c| c.values().sum::<usize>() as f64)
        .collect();
    let median_edges = if directed_totals.is_empty() {
        0.0
    } else {
        let mut sorted = directed_totals.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted[sorted.len() / 2]
    };

    let mut anomalies = Vec::new();
    let mut seen: HashSet<(usize, usize)> = HashSet::new();

    for &(sm, tm) in pair_counts.keys() {
        let pair = if sm <= tm { (sm, tm) } else { (tm, sm) };
        if seen.contains(&pair) {
            continue;
        }
        seen.insert(pair);

        let forward = pair_counts.get(&(pair.0, pair.1));
        let reverse = pair_counts.get(&(pair.1, pair.0));

        // Pattern 1: Bidirectional boundary.
        if forward.is_some() && reverse.is_some() {
            let fwd_total: usize = forward.unwrap().values().sum();
            let rev_total: usize = reverse.unwrap().values().sum();
            let total = fwd_total + rev_total;
            let reverse_share = fwd_total.min(rev_total) as f64 / total as f64;

            let mut node_ids: HashSet<String> = HashSet::new();
            for examples in [
                pair_examples.get(&(pair.0, pair.1)),
                pair_examples.get(&(pair.1, pair.0)),
            ]
            .into_iter()
            .flatten()
            {
                for &(s, t) in examples {
                    node_ids.insert(graph.node_ids[s].clone());
                    node_ids.insert(graph.node_ids[t].clone());
                }
            }

            let sorted_ids = sorted_vec(node_ids);
            let anchors: Vec<AnchorOutput> = sorted_ids
                .iter()
                .filter_map(|nid| graph.node_index.get(nid.as_str()).map(|&i| graph.anchor(i)))
                .collect();
            anomalies.push(Anomaly {
                kind: AnomalyKind::CrossModule,
                node_ids: sorted_ids,
                description: format!(
                    "Bidirectional dependency between module {} and module {}: {} forward edges, {} reverse edges",
                    pair.0, pair.1, fwd_total, rev_total
                ),
                severity: (0.35 + reverse_share).min(1.0),
                confidence: (0.5 + total as f64 / 20.0).min(1.0),
                anchors,
            });
            continue;
        }

        // Pattern 2: Minority unidirectional coupling.
        let active = forward.or(reverse);
        let Some(active) = active else { continue };
        let active_total: usize = active.values().sum();
        if median_edges <= 0.0 || active_total as f64 >= median_edges {
            continue;
        }
        let minority_ratio = active_total as f64 / median_edges;
        let dir_pair = if forward.is_some() {
            (pair.0, pair.1)
        } else {
            (pair.1, pair.0)
        };

        let mut node_ids: HashSet<String> = HashSet::new();
        if let Some(examples) = pair_examples.get(&dir_pair) {
            for &(s, t) in examples {
                node_ids.insert(graph.node_ids[s].clone());
                node_ids.insert(graph.node_ids[t].clone());
            }
        }

        let sorted_ids = sorted_vec(node_ids);
        let anchors: Vec<AnchorOutput> = sorted_ids
            .iter()
            .filter_map(|nid| graph.node_index.get(nid.as_str()).map(|&i| graph.anchor(i)))
            .collect();
        anomalies.push(Anomaly {
            kind: AnomalyKind::CrossModule,
            node_ids: sorted_ids,
            description: format!(
                "Unusual dependency from module {} to module {}: {} edges ({:.0}% of typical cross-module coupling)",
                dir_pair.0, dir_pair.1, active_total, minority_ratio * 100.0
            ),
            severity: (0.3 + (1.0 - minority_ratio) * 0.5).min(1.0),
            confidence: (0.4 + active_total as f64 / 10.0).min(1.0),
            anchors,
        });
    }
    anomalies
}

// ---------------------------------------------------------------------------
// Spectral outliers
// ---------------------------------------------------------------------------

fn detect_spectral_outliers(
    modules: &[EnrichedModule],
    fingerprints: &HashMap<String, Vec<f64>>,
    graph: &Graph,
    _node_to_module: &HashMap<String, usize>,
    package_fallback: bool,
) -> Vec<Anomaly> {
    let base_threshold = 2.0;

    // Precompute centroids for non-trivial modules.
    let mut centroids: HashMap<usize, Vec<f64>> = HashMap::new();
    let mut module_labels: HashMap<usize, String> = HashMap::new();
    let mut module_confidence: HashMap<usize, f64> = HashMap::new();
    for m in modules {
        if m.unassigned || m.node_ids.len() < 3 {
            continue;
        }
        let fps: Vec<&Vec<f64>> = m
            .node_ids
            .iter()
            .filter_map(|nid| fingerprints.get(nid))
            .collect();
        if fps.len() < 3 {
            continue;
        }
        let dim = fps[0].len();
        let n = fps.len() as f64;
        let mut centroid = vec![0.0; dim];
        for fp in &fps {
            for (i, &v) in fp.iter().enumerate() {
                centroid[i] += v;
            }
        }
        for v in &mut centroid {
            *v /= n;
        }
        centroids.insert(m.id, centroid);
        module_labels.insert(m.id, m.label.clone());
        module_confidence.insert(m.id, m.confidence);
    }

    // Build module membership sets for edge validation.
    let mut module_members: HashMap<usize, HashSet<usize>> = HashMap::new();
    for m in modules {
        let members: HashSet<usize> = m
            .node_ids
            .iter()
            .filter_map(|nid| graph.node_index.get(nid).copied())
            .collect();
        module_members.insert(m.id, members);
    }

    let mut anomalies = Vec::new();
    for m in modules {
        let Some(centroid) = centroids.get(&m.id) else { continue };

        let mut fps: Vec<(&str, &Vec<f64>)> = Vec::new();
        for nid in &m.node_ids {
            if let Some(fp) = fingerprints.get(nid) {
                fps.push((nid.as_str(), fp));
            }
        }
        if fps.len() < 3 {
            continue;
        }

        // Adaptive threshold: scale by module confidence.
        // Package-fallback modules have low confidence → higher threshold → fewer false positives.
        let confidence = module_confidence.get(&m.id).copied().unwrap_or(0.5);
        let effective_threshold = if package_fallback {
            base_threshold / confidence.max(0.3)
        } else {
            base_threshold
        };

        let distances: Vec<f64> = fps.iter().map(|(_, fp)| euclidean_dist(fp, centroid)).collect();
        let mean_dist: f64 = distances.iter().sum::<f64>() / distances.len() as f64;
        let variance: f64 =
            distances.iter().map(|d| (d - mean_dist).powi(2)).sum::<f64>() / distances.len() as f64;
        let std_dist = variance.sqrt();
        if std_dist == 0.0 {
            continue;
        }

        let mut module_outliers = Vec::new();

        for (idx, (nid, fp)) in fps.iter().enumerate() {
            let z_score = (distances[idx] - mean_dist) / std_dist;
            if z_score <= effective_threshold {
                continue;
            }

            // Find nearest alternative module, but only report it if the node
            // has actual edges to that module (prevents misleading suggestions).
            let mut nearest_label = String::new();
            let node_idx = graph.node_index.get(*nid).copied();

            if let Some(ni) = node_idx {
                let mut best_dist = f64::INFINITY;
                let mut best_label = String::new();

                for (&other_id, other_centroid) in &centroids {
                    if other_id == m.id {
                        continue;
                    }
                    let d = euclidean_dist(fp, other_centroid);
                    if d >= best_dist {
                        continue;
                    }

                    // Check for actual edge coupling to this module.
                    let other_members = module_members.get(&other_id);
                    let has_coupling = other_members.is_some_and(|members| {
                        graph.successors[ni]
                            .iter()
                            .chain(graph.predecessors[ni].iter())
                            .any(|neighbor| members.contains(neighbor))
                    });

                    if has_coupling {
                        best_dist = d;
                        best_label = module_labels
                            .get(&other_id)
                            .cloned()
                            .unwrap_or_else(|| other_id.to_string());
                    }
                }
                nearest_label = best_label;
            }

            let mut desc = format!(
                "Node {} is {:.1}σ from module {} centroid",
                nid, z_score, m.id
            );
            if !nearest_label.is_empty() {
                desc.push_str(&format!("; nearest alternative: {}", nearest_label));
            }

            let anchors = if let Some(ni) = node_idx {
                vec![graph.anchor(ni)]
            } else {
                Vec::new()
            };

            module_outliers.push(Anomaly {
                kind: AnomalyKind::SpectralOutlier,
                node_ids: vec![nid.to_string()],
                description: desc,
                severity: ((z_score - effective_threshold) / effective_threshold).clamp(0.0, 1.0),
                confidence: confidence.max(0.4),
                anchors,
            });
        }

        // Per-module cap: if >50% of members flagged, suppress all (the module
        // assignment is the problem, not individual nodes).
        if module_outliers.len() * 2 <= fps.len() {
            anomalies.extend(module_outliers);
        }
    }
    anomalies
}

// ---------------------------------------------------------------------------
// Cycles (from pre-computed SCCs)
// ---------------------------------------------------------------------------

fn sccs_to_anomalies(sccs: &[Vec<String>], graph: &Graph) -> Vec<Anomaly> {
    let non_trivial: Vec<&Vec<String>> = sccs.iter().filter(|c| c.len() > 1).collect();
    if non_trivial.is_empty() {
        return Vec::new();
    }
    let sizes_f: Vec<f64> = non_trivial.iter().map(|c| c.len() as f64).collect();
    let size_pcts = crate::stats::percentile_ranks(&sizes_f);

    non_trivial
        .iter()
        .enumerate()
        .map(|(ci, component)| {
            let node_ids = {
                let mut ids = (*component).clone();
                ids.sort();
                ids
            };
            let anchors: Vec<AnchorOutput> = node_ids
                .iter()
                .filter_map(|nid| graph.node_index.get(nid).map(|&i| graph.anchor(i)))
                .collect();
            let len = node_ids.len();
            // Severity from percentile rank of this SCC's size among all SCCs.
            // If there's only one SCC, use 0.5 as default.
            let sev = if non_trivial.len() == 1 { 0.5 } else { size_pcts[ci] };
            Anomaly {
                kind: AnomalyKind::CycleMember,
                node_ids,
                description: format!("Strongly connected dependency group of {} nodes", len),
                severity: sev.clamp(0.3, 1.0),
                confidence: 0.9, // cycle detection is deterministic
                anchors,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Layer discrepancies
// ---------------------------------------------------------------------------

fn detect_layer_discrepancies(graph: &Graph, edge_kinds: &[String]) -> Vec<Anomaly> {
    if edge_kinds.len() < 2 {
        return Vec::new();
    }

    // Group nodes by kind.
    let mut kind_groups: HashMap<&str, Vec<usize>> = HashMap::new();
    for i in 0..graph.n {
        kind_groups
            .entry(graph.node_kinds[i].as_str())
            .or_default()
            .push(i);
    }

    // Compute degree per layer for each node.
    let mut layer_degrees: HashMap<&str, Vec<usize>> = HashMap::new();
    for kind in edge_kinds {
        let mut degrees = vec![0usize; graph.n];
        for &(src, tgt) in graph.edges_of_kind(kind) {
            degrees[src] += 1;
            degrees[tgt] += 1;
        }
        layer_degrees.insert(kind.as_str(), degrees);
    }

    // Compute percentile rank per layer, within each NodeKind group.
    let mut layer_percentiles: HashMap<&str, Vec<f64>> = HashMap::new();
    for kind in edge_kinds {
        let degrees = &layer_degrees[kind.as_str()];
        if *degrees.iter().max().unwrap_or(&0) == 0 {
            continue;
        }
        let mut pcts = vec![f64::NAN; graph.n];
        for (_node_kind, group_indices) in &kind_groups {
            if group_indices.len() < 5 {
                continue;
            }
            let group_degrees: Vec<usize> = group_indices.iter().map(|&i| degrees[i]).collect();
            let mut sorted_degrees = group_degrees.clone();
            sorted_degrees.sort();
            let gn = sorted_degrees.len() as f64;
            for (gi, &node_idx) in group_indices.iter().enumerate() {
                // count(d <= value) / group_size — note: uses <= not <
                let d = group_degrees[gi];
                let rank = sorted_degrees.partition_point(|&x| x <= d);
                pcts[node_idx] = rank as f64 / gn;
            }
        }
        layer_percentiles.insert(kind.as_str(), pcts);
    }

    let active_kinds: Vec<&str> = edge_kinds
        .iter()
        .filter(|k| layer_percentiles.contains_key(k.as_str()))
        .map(|k| k.as_str())
        .collect();
    if active_kinds.len() < 2 {
        return Vec::new();
    }

    let kind_labels: HashMap<&str, &str> = [
        ("calls", "calls"),
        ("imports", "imports"),
        ("inherits", "inherits"),
        ("contains", "contains"),
    ]
    .into_iter()
    .collect();

    // Pass 1: Compute best gap for each eligible node.
    struct GapEntry<'a> {
        node: usize,
        gap: f64,
        ka: &'a str,
        kb: &'a str,
    }
    let mut candidates: Vec<GapEntry> = Vec::new();
    let mut all_gaps: Vec<f64> = Vec::new();

    for i in 0..graph.n {
        let max_deg: usize = active_kinds
            .iter()
            .map(|k| layer_degrees[k][i])
            .max()
            .unwrap_or(0);
        if max_deg < 2 {
            continue;
        }
        let min_deg: usize = active_kinds
            .iter()
            .map(|k| layer_degrees[k][i])
            .min()
            .unwrap_or(0);
        if min_deg < 2 {
            continue;
        }

        let node_active: Vec<&str> = active_kinds
            .iter()
            .filter(|&&k| !layer_percentiles[k][i].is_nan())
            .copied()
            .collect();
        if node_active.len() < 2 {
            continue;
        }

        let mut best_gap = 0.0f64;
        let mut best_pair: Option<(&str, &str)> = None;
        for (ai, &ka) in node_active.iter().enumerate() {
            for &kb in &node_active[ai + 1..] {
                let gap = (layer_percentiles[ka][i] - layer_percentiles[kb][i]).abs();
                if gap > best_gap {
                    best_gap = gap;
                    best_pair = Some((ka, kb));
                }
            }
        }
        if let Some((ka, kb)) = best_pair {
            all_gaps.push(best_gap);
            candidates.push(GapEntry { node: i, gap: best_gap, ka, kb });
        }
    }

    // Pass 2: Tukey fence on gap distribution as threshold.
    let gap_threshold = crate::stats::tukey_upper_fence(&all_gaps);
    let gap_pcts = crate::stats::percentile_ranks(&all_gaps);

    let mut anomalies = Vec::new();
    for (ci, entry) in candidates.iter().enumerate() {
        if entry.gap <= gap_threshold {
            continue;
        }

        let i = entry.node;
        let pct_a = layer_percentiles[entry.ka][i];
        let pct_b = layer_percentiles[entry.kb][i];
        let label_a = kind_labels.get(entry.ka).copied().unwrap_or(entry.ka);
        let label_b = kind_labels.get(entry.kb).copied().unwrap_or(entry.kb);
        let nid = &graph.node_ids[i];

        let desc = if pct_a >= pct_b {
            format!(
                "{} is {}-central (p{:.0}%) but {}-peripheral (p{:.0}%)",
                nid, label_a, pct_a * 100.0, label_b, pct_b * 100.0
            )
        } else {
            format!(
                "{} is {}-central (p{:.0}%) but {}-peripheral (p{:.0}%)",
                nid, label_b, pct_b * 100.0, label_a, pct_a * 100.0
            )
        };

        anomalies.push(Anomaly {
            kind: AnomalyKind::LayerDiscrepancy,
            node_ids: vec![nid.clone()],
            description: desc,
            severity: gap_pcts[ci],
            confidence: 0.6,
            anchors: vec![graph.anchor(i)],
        });
    }
    anomalies
}

fn euclidean_dist(a: &[f64], b: &[f64]) -> f64 {
    crate::stats::euclidean_dist(a, b)
}

fn sorted_vec(set: HashSet<String>) -> Vec<String> {
    let mut v: Vec<String> = set.into_iter().collect();
    v.sort();
    v
}
