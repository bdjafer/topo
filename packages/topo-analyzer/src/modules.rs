//! Module enrichment: labels, cohesion/separation, dependencies, modularity Q.
//!
//! Ports logic from Python topo_analyzer.modules and topo_analyzer.analysis.

use std::collections::HashMap;

use crate::graph::Graph;
use crate::types::{DependencyOutput, ModuleOutput};

/// Internal module representation before serialization.
pub struct EnrichedModule {
    pub id: usize,
    pub node_ids: Vec<String>,
    pub label: String,
    pub cohesion: Option<f64>,
    pub separation: Option<f64>,
    pub confidence: f64,
    pub unassigned: bool,
}

impl EnrichedModule {
    pub fn to_output(&self) -> ModuleOutput {
        ModuleOutput {
            id: self.id,
            label: self.label.clone(),
            size: self.node_ids.len(),
            members: self.node_ids.clone(),
            cohesion: self.cohesion.map(|v| round4(v)),
            separation: self.separation.map(|v| round4(v)),
            confidence: round4(self.confidence),
            unassigned: self.unassigned,
        }
    }
}

/// Derive a human-readable label from a module's member IDs.
///
/// Finds the longest common dotted prefix. Falls back to the most
/// frequent top-level package.
pub fn module_label(member_ids: &[String]) -> String {
    if member_ids.is_empty() {
        return "unknown".to_string();
    }
    // Find longest common dotted prefix.
    let parts_list: Vec<Vec<&str>> = member_ids.iter().map(|id| id.split('.').collect()).collect();
    let mut prefix_parts: Vec<&str> = Vec::new();
    let min_len = parts_list.iter().map(|p| p.len()).min().unwrap_or(0);
    for level in 0..min_len {
        let first = parts_list[0][level];
        if parts_list.iter().all(|p| p[level] == first) {
            prefix_parts.push(first);
        } else {
            break;
        }
    }
    if !prefix_parts.is_empty() {
        return prefix_parts.join(".");
    }
    // No common prefix — use most frequent top-level package.
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for id in member_ids {
        let pkg = id.split('.').next().unwrap_or(id.as_str());
        *counts.entry(pkg).or_default() += 1;
    }
    counts
        .into_iter()
        .max_by_key(|&(_, count)| count)
        .map(|(pkg, _)| pkg.to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Compute cohesion, separation, and confidence for modules.
///
/// - Cohesion: mean Euclidean distance from members to module centroid.
/// - Separation: min distance from this centroid to any other centroid.
/// - Confidence: blend of silhouette and local separation/cohesion ratio.
pub fn annotate_modules(
    clusters: &HashMap<String, usize>,
    fingerprints: &HashMap<String, Vec<f64>>,
    silhouette: f64,
    unassigned_cluster: Option<usize>,
) -> Vec<EnrichedModule> {
    // Group nodes by cluster.
    let mut cluster_members: HashMap<usize, Vec<String>> = HashMap::new();
    for (nid, &cid) in clusters {
        cluster_members.entry(cid).or_default().push(nid.clone());
    }
    // Sort members for determinism.
    for members in cluster_members.values_mut() {
        members.sort();
    }

    let dim = fingerprints.values().next().map(|v| v.len()).unwrap_or(0);

    // Compute centroids.
    let mut centroids: HashMap<usize, Vec<f64>> = HashMap::new();
    for (&cid, members) in &cluster_members {
        if Some(cid) == unassigned_cluster {
            continue;
        }
        let fps: Vec<&Vec<f64>> = members
            .iter()
            .filter_map(|nid| fingerprints.get(nid))
            .collect();
        if fps.is_empty() {
            continue;
        }
        let n = fps.len() as f64;
        let mut centroid = vec![0.0; dim];
        for fp in &fps {
            for (i, &v) in fp.iter().enumerate() {
                if i < dim {
                    centroid[i] += v;
                }
            }
        }
        for v in &mut centroid {
            *v /= n;
        }
        centroids.insert(cid, centroid);
    }

    // Build enriched modules.
    let mut modules: Vec<EnrichedModule> = Vec::new();
    let mut sorted_clusters: Vec<usize> = cluster_members.keys().copied().collect();
    sorted_clusters.sort();

    for (new_id, &cid) in sorted_clusters.iter().enumerate() {
        let members = &cluster_members[&cid];
        let is_unassigned = Some(cid) == unassigned_cluster;
        let label = module_label(members);

        let (cohesion, separation, confidence) = if is_unassigned || !centroids.contains_key(&cid) {
            (None, None, 0.0)
        } else {
            let centroid = &centroids[&cid];
            // Cohesion: mean distance to centroid.
            let fps: Vec<&Vec<f64>> = members
                .iter()
                .filter_map(|nid| fingerprints.get(nid))
                .collect();
            let cohesion = if fps.is_empty() {
                0.0
            } else {
                let total: f64 = fps.iter().map(|fp| euclidean_dist(fp, centroid)).sum();
                total / fps.len() as f64
            };
            // Separation: min distance to any other centroid.
            let separation: Option<f64> = centroids
                .iter()
                .filter(|&(&other_id, _)| other_id != cid)
                .map(|(_, other_centroid)| euclidean_dist(centroid, other_centroid))
                .reduce(f64::min);
            // Confidence.
            let ratio = match separation {
                Some(sep) => sep / (sep + cohesion + 1e-9),
                None => 0.5,
            };
            let confidence = (0.5 * silhouette.max(0.0) + 0.5 * ratio).clamp(0.0, 1.0);
            (Some(cohesion), separation, confidence)
        };

        modules.push(EnrichedModule {
            id: new_id,
            node_ids: members.clone(),
            label,
            cohesion,
            separation,
            confidence,
            unassigned: is_unassigned,
        });
    }

    modules
}

/// Build cross-module dependency aggregation with edge-kind breakdown.
pub fn build_module_dependencies(
    graph: &Graph,
    node_to_module: &HashMap<String, usize>,
) -> Vec<DependencyOutput> {
    let mut dep_map: HashMap<(usize, usize), HashMap<String, usize>> = HashMap::new();

    for (kind, edges) in &graph.typed_edges {
        for &(src, tgt) in edges {
            let src_id = &graph.node_ids[src];
            let tgt_id = &graph.node_ids[tgt];
            let src_mod = node_to_module.get(src_id);
            let tgt_mod = node_to_module.get(tgt_id);
            if let (Some(&sm), Some(&tm)) = (src_mod, tgt_mod) {
                if sm == tm {
                    continue;
                }
                *dep_map
                    .entry((sm, tm))
                    .or_default()
                    .entry(kind.clone())
                    .or_default() += 1;
            }
        }
    }

    let mut deps: Vec<DependencyOutput> = dep_map
        .into_iter()
        .map(|((src, tgt), kinds)| {
            let weight: usize = kinds.values().sum();
            DependencyOutput {
                source: src,
                target: tgt,
                weight,
                edge_kinds: kinds,
            }
        })
        .collect();
    deps.sort_by_key(|d| (d.source, d.target));
    deps
}

/// Check if spectral clustering is degenerate and should fall back to packages.
///
/// Matches Python `modules.py::_is_degenerate()`.
pub fn is_degenerate(
    modules: &[EnrichedModule],
    silhouette: Option<f64>,
) -> bool {
    if let Some(sil) = silhouette {
        if sil >= 0.5 {
            return false;
        }
    }
    let clustered: Vec<&EnrichedModule> = modules.iter().filter(|m| !m.unassigned).collect();
    if clustered.is_empty() {
        return false;
    }
    let total_nodes: usize = clustered.iter().map(|m| m.node_ids.len()).sum();
    let avg_size = total_nodes as f64 / clustered.len() as f64;
    if avg_size <= 3.0 {
        return true;
    }
    if let Some(sil) = silhouette {
        if sil < 0.3 {
            return true;
        }
    }
    false
}

/// Group nodes by top-level package prefix (fallback when spectral fails).
///
/// Matches Python `modules.py::_package_grouping()`.
pub fn package_grouping(node_ids: &[String]) -> Vec<EnrichedModule> {
    let mut groups: HashMap<String, Vec<String>> = HashMap::new();
    for nid in node_ids {
        let pkg = nid.split('.').next().unwrap_or(nid).to_string();
        groups.entry(pkg).or_default().push(nid.clone());
    }
    let mut sorted_pkgs: Vec<String> = groups.keys().cloned().collect();
    sorted_pkgs.sort();
    sorted_pkgs
        .into_iter()
        .enumerate()
        .map(|(i, pkg)| {
            let mut members = groups.remove(&pkg).unwrap();
            members.sort();
            EnrichedModule {
                id: i,
                node_ids: members,
                label: pkg,
                cohesion: None,
                separation: None,
                confidence: 0.5,
                unassigned: false,
            }
        })
        .collect()
}

/// Newman's modularity Q for detected module assignment.
///
/// Measures how well module boundaries explain edge structure vs random.
/// Range: -0.5 to 1.0. >0.3 significant, >0.5 strong.
pub fn modularity_q(
    graph: &Graph,
    node_to_module: &HashMap<String, usize>,
) -> Option<f64> {
    let m = graph.edge_count;
    if m == 0 {
        return None;
    }

    let mut internal: HashMap<usize, usize> = HashMap::new();
    let mut degree: HashMap<usize, usize> = HashMap::new();

    for edges in graph.typed_edges.values() {
        for &(src, tgt) in edges {
            let src_mod = node_to_module.get(&graph.node_ids[src]);
            let tgt_mod = node_to_module.get(&graph.node_ids[tgt]);
            if let Some(&sm) = src_mod {
                *degree.entry(sm).or_default() += 1;
            }
            if let Some(&tm) = tgt_mod {
                *degree.entry(tm).or_default() += 1;
            }
            if let (Some(&sm), Some(&tm)) = (src_mod, tgt_mod) {
                if sm == tm {
                    *internal.entry(sm).or_default() += 1;
                }
            }
        }
    }

    let m_f = m as f64;
    let mut q = 0.0;
    let all_mods: std::collections::HashSet<usize> = internal
        .keys()
        .chain(degree.keys())
        .copied()
        .collect();
    for mod_id in all_mods {
        let ec = *internal.get(&mod_id).unwrap_or(&0) as f64 / m_f;
        let ac = *degree.get(&mod_id).unwrap_or(&0) as f64 / (2.0 * m_f);
        q += ec - ac * ac;
    }

    Some(round4(q))
}

fn euclidean_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

pub(crate) fn round4(v: f64) -> f64 {
    (v * 10000.0).round() / 10000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_label_common_prefix() {
        let ids = vec![
            "pkg.core.main".to_string(),
            "pkg.core.util".to_string(),
            "pkg.core.init".to_string(),
        ];
        assert_eq!(module_label(&ids), "pkg.core");
    }

    #[test]
    fn test_module_label_no_common_prefix() {
        let ids = vec![
            "auth.login".to_string(),
            "auth.logout".to_string(),
            "billing.charge".to_string(),
        ];
        assert_eq!(module_label(&ids), "auth");
    }

    #[test]
    fn test_package_grouping() {
        let ids = vec![
            "a.x".to_string(),
            "a.y".to_string(),
            "b.z".to_string(),
        ];
        let modules = package_grouping(&ids);
        assert_eq!(modules.len(), 2);
        assert_eq!(modules[0].label, "a");
        assert_eq!(modules[0].node_ids.len(), 2);
        assert_eq!(modules[1].label, "b");
    }
}
