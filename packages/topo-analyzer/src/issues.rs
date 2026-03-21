//! Issue synthesis: convert anomalies, health metrics, and structural
//! diagnostics into prioritized, developer-facing issues.
//!
//! Ports logic from Python topo_analyzer.analysis._build_issues,
//! _detect_wide_interfaces, _detect_phantom_imports.

use std::collections::{HashMap, HashSet};

use crate::anomalies::{Anomaly, AnomalyKind};
use crate::graph::Graph;
use crate::modules::EnrichedModule;
use crate::stats;
use crate::types::{AnchorOutput, IssueOutput, RoleOutput};

/// Context for building issues.
pub struct IssuesContext<'a> {
    pub graph: &'a Graph,
    pub modules: &'a [EnrichedModule],
    pub roles: &'a [RoleOutput],
    pub anomalies: &'a [Anomaly],
    pub silhouette: Option<f64>,
    pub package_fallback: bool,
    pub spectral_coverage_ratio: f64,
    pub self_edge_ratio: f64,
    pub level: &'a str,
    pub largest_module_ratio: f64,
    pub node_to_module: &'a HashMap<String, usize>,
}

pub fn build_issues(ctx: &IssuesContext) -> Vec<IssueOutput> {
    let mut issues: Vec<IssueOutput> = Vec::new();

    // 1. Coverage issue.
    if ctx.spectral_coverage_ratio < 0.75 {
        issues.push(IssueOutput {
            id: "coverage:low-spectral".to_string(),
            kind: "coverage".to_string(),
            title: "Low spectral coverage".to_string(),
            description: format!(
                "Only {:.1}% of analyzed nodes received spectral fingerprints; \
                 disconnected components may limit clustering quality.",
                ctx.spectral_coverage_ratio * 100.0
            ),
            severity: round2((1.0 - ctx.spectral_coverage_ratio).max(0.3)),
            severity_label: severity_label((1.0 - ctx.spectral_coverage_ratio).max(0.3)),
            confidence: 0.9,
            confidence_label: "high".to_string(),
            anchors: Vec::new(),
        });
    }

    // 2. Self-edge drop.
    if ctx.self_edge_ratio > 0.7 && ctx.level != "symbol" {
        let sev = (0.3 + ctx.self_edge_ratio * 0.5).min(1.0);
        issues.push(IssueOutput {
            id: "self-edge-drop:high".to_string(),
            kind: "self_edge_drop".to_string(),
            title: "High self-edge drop rate".to_string(),
            description: format!(
                "{:.0}% of scoped edges collapsed into self-edges at {} level. \
                 Consider --level symbol for richer analysis.",
                ctx.self_edge_ratio * 100.0,
                ctx.level
            ),
            severity: round2(sev),
            severity_label: severity_label(sev),
            confidence: 0.9,
            confidence_label: "high".to_string(),
            anchors: Vec::new(),
        });
    }

    // 3. Module separation (skip when package fallback).
    if ctx.largest_module_ratio >= 0.5 && !ctx.package_fallback {
        let sev = ctx.largest_module_ratio.min(1.0);
        issues.push(IssueOutput {
            id: "module-separation:weak".to_string(),
            kind: "module_separation".to_string(),
            title: "Weak module separation".to_string(),
            description: format!(
                "The largest structural module still covers {:.1}% of the analysis graph.",
                ctx.largest_module_ratio * 100.0
            ),
            severity: round2(sev),
            severity_label: severity_label(sev),
            confidence: 0.8,
            confidence_label: "high".to_string(),
            anchors: Vec::new(),
        });
    }

    // 4. Reverse dependencies (bidirectional package flows).
    issues.extend(detect_reverse_dependencies(ctx));

    // 5. Anomaly → issue conversion.
    // Suppress spectral outliers when clustering fell back to packages
    // (spectral wasn't better than random — outliers are noise).
    let suppress_outliers = ctx.package_fallback;

    let role_map: HashMap<&str, &str> = ctx
        .roles
        .iter()
        .map(|r| (r.node_id.as_str(), r.role.as_str()))
        .collect();

    // Data-driven severity weights: scale by structural importance
    // (betweenness percentile) rather than fixed role multipliers.
    let btw_values: Vec<f64> = ctx.roles.iter().map(|r| r.betweenness).collect();
    let btw_pcts = stats::percentile_ranks(&btw_values);
    let btw_pct_map: HashMap<&str, f64> = ctx
        .roles
        .iter()
        .zip(btw_pcts.iter())
        .map(|(r, &pct)| (r.node_id.as_str(), pct))
        .collect();

    for anomaly in ctx.anomalies {
        if anomaly.kind == AnomalyKind::SpectralOutlier && suppress_outliers {
            continue;
        }
        if anomaly.kind == AnomalyKind::LayerDiscrepancy {
            if let Some(nid) = anomaly.node_ids.first() {
                if role_map.get(nid.as_str()) == Some(&"entry_point") {
                    continue;
                }
            }
        }

        let mut severity = anomaly.severity;
        if anomaly.kind == AnomalyKind::SpectralOutlier {
            if let Some(nid) = anomaly.node_ids.first() {
                // Weight by structural importance: nodes with higher
                // betweenness get amplified severity (range [1.0, 1.5]).
                let btw_pct = btw_pct_map.get(nid.as_str()).copied().unwrap_or(0.5);
                let weight = 1.0 + btw_pct * 0.5;
                severity = (severity * weight).min(1.0);
            }
        }

        issues.push(IssueOutput {
            id: issue_id(anomaly.kind.as_str(), &anomaly.node_ids, &[]),
            kind: anomaly.kind.as_str().to_string(),
            title: anomaly_title(anomaly.kind),
            description: anomaly.description.clone(),
            severity: round2(severity),
            severity_label: severity_label(severity),
            confidence: round2(anomaly.confidence),
            confidence_label: confidence_label(anomaly.confidence),
            anchors: anomaly.anchors.clone(),
        });
    }

    // 6. Orphan issues.
    issues.extend(detect_orphan_issues(ctx));

    // 7. God module detection.
    if !suppress_outliers {
        issues.extend(detect_god_modules(ctx));
    }

    // 8. Low cohesion.
    issues.extend(detect_low_cohesion(ctx));

    // 9. Fragile hub.
    issues.extend(detect_fragile_hubs(ctx));

    // 10. Wide interface.
    issues.extend(detect_wide_interfaces(ctx));

    // 11. Phantom imports.
    issues.extend(detect_phantom_imports(ctx));

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
// Reverse dependencies
// ---------------------------------------------------------------------------

fn detect_reverse_dependencies(ctx: &IssuesContext) -> Vec<IssueOutput> {
    // Aggregate cross-package edges.
    let mut pair_counts: HashMap<(String, String), usize> = HashMap::new();
    let mut pair_anchors: HashMap<(String, String), Vec<AnchorOutput>> = HashMap::new();

    for edges in ctx.graph.typed_edges.values() {
        for &(src, tgt) in edges {
            let src_pkg = top_package(&ctx.graph.node_ids[src]);
            let tgt_pkg = top_package(&ctx.graph.node_ids[tgt]);
            if src_pkg == tgt_pkg {
                continue;
            }
            *pair_counts.entry((src_pkg.clone(), tgt_pkg.clone())).or_default() += 1;
            let anchors = pair_anchors.entry((src_pkg, tgt_pkg)).or_default();
            if anchors.len() < 3 {
                anchors.push(ctx.graph.anchor(src));
            }
        }
    }

    // Find bidirectional pairs.
    let mut seen: HashSet<(String, String)> = HashSet::new();
    let mut issues = Vec::new();

    for (pair, _) in &pair_counts {
        let canonical = if pair.0 <= pair.1 {
            (pair.0.clone(), pair.1.clone())
        } else {
            (pair.1.clone(), pair.0.clone())
        };
        if seen.contains(&canonical) {
            continue;
        }
        seen.insert(canonical.clone());

        let fwd = pair_counts.get(&(canonical.0.clone(), canonical.1.clone())).copied().unwrap_or(0);
        let rev = pair_counts.get(&(canonical.1.clone(), canonical.0.clone())).copied().unwrap_or(0);
        if fwd == 0 || rev == 0 {
            continue;
        }
        let total = fwd + rev;
        let reverse = fwd.min(rev);
        let sev = (0.35 + reverse as f64 / total as f64).min(1.0);
        let conf = (0.5 + total as f64 / 20.0).min(1.0);

        let mut anchors = Vec::new();
        if let Some(a) = pair_anchors.get(&(canonical.0.clone(), canonical.1.clone())) {
            anchors.extend(a.iter().take(2).cloned());
        }
        if let Some(a) = pair_anchors.get(&(canonical.1.clone(), canonical.0.clone())) {
            anchors.extend(a.iter().take(1).cloned());
        }

        issues.push(IssueOutput {
            id: format!(
                "reverse-dependency:{}",
                sorted_join(&[canonical.0.clone(), canonical.1.clone()])
            ),
            kind: "reverse_dependency".to_string(),
            title: format!(
                "Reverse dependency between {} and {}",
                canonical.0, canonical.1
            ),
            description: format!(
                "Dependency flow is bidirectional with {}/{} edges in the weaker direction.",
                reverse, total
            ),
            severity: round2(sev),
            severity_label: severity_label(sev),
            confidence: round2(conf),
            confidence_label: confidence_label(conf),
            anchors,
        });
    }
    issues
}

// ---------------------------------------------------------------------------
// Orphan issues
// ---------------------------------------------------------------------------

fn detect_orphan_issues(ctx: &IssuesContext) -> Vec<IssueOutput> {
    let orphan_roles: Vec<&RoleOutput> = ctx
        .roles
        .iter()
        .filter(|r| r.role == "orphan")
        .collect();

    if orphan_roles.is_empty() {
        return Vec::new();
    }

    if orphan_roles.len() > 3 {
        let node_ids: Vec<String> = orphan_roles.iter().map(|r| r.node_id.clone()).collect();
        let mut names: String = node_ids.iter().take(4).cloned().collect::<Vec<_>>().join(", ");
        if node_ids.len() > 4 {
            names.push_str(", ...");
        }
        let sev = (orphan_roles.len() as f64 * 0.1).min(0.5);
        return vec![IssueOutput {
            id: issue_id("orphan", &node_ids, &[]),
            kind: "orphan".to_string(),
            title: format!("{} orphan modules — possible dead code", orphan_roles.len()),
            description: names,
            severity: round2(sev),
            severity_label: severity_label(sev),
            confidence: 0.7,
            confidence_label: "medium".to_string(),
            anchors: orphan_roles
                .iter()
                .take(3)
                .filter_map(|r| {
                    ctx.graph
                        .node_index
                        .get(&r.node_id)
                        .map(|&i| ctx.graph.anchor(i))
                })
                .collect(),
        }];
    }

    orphan_roles
        .iter()
        .map(|r| {
            let anchors = ctx
                .graph
                .node_index
                .get(&r.node_id)
                .map(|&i| vec![ctx.graph.anchor(i)])
                .unwrap_or_default();
            IssueOutput {
                id: issue_id("orphan", &[r.node_id.clone()], &[]),
                kind: "orphan".to_string(),
                title: format!("Orphan: {}", r.node_id),
                description: "No inbound or outbound edges — may be dead code".to_string(),
                severity: 0.3,
                severity_label: "low".to_string(),
                confidence: 0.7,
                confidence_label: "medium".to_string(),
                anchors,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// God module detection
// ---------------------------------------------------------------------------

fn detect_god_modules(ctx: &IssuesContext) -> Vec<IssueOutput> {
    let clustered: Vec<&EnrichedModule> = ctx.modules.iter().filter(|m| !m.unassigned).collect();
    if clustered.len() < 2 {
        return Vec::new();
    }

    let sizes_f: Vec<f64> = clustered.iter().map(|m| m.node_ids.len() as f64).collect();
    let size_fence = stats::tukey_upper_fence(&sizes_f);
    let total_edges = ctx.graph.edge_count;

    // Compute edge shares for all modules, then use Tukey fence.
    let mut edge_shares: Vec<f64> = Vec::new();
    let mut module_edge_shares: Vec<(usize, f64)> = Vec::new();
    for (i, m) in clustered.iter().enumerate() {
        let member_set: HashSet<&str> = m.node_ids.iter().map(|s| s.as_str()).collect();
        let edge_count: usize = ctx
            .graph
            .typed_edges
            .values()
            .flat_map(|edges| edges.iter())
            .filter(|&&(src, tgt)| {
                member_set.contains(ctx.graph.node_ids[src].as_str())
                    || member_set.contains(ctx.graph.node_ids[tgt].as_str())
            })
            .count();
        let share = edge_count as f64 / total_edges.max(1) as f64;
        edge_shares.push(share);
        module_edge_shares.push((i, share));
    }
    let share_fence = stats::tukey_upper_fence(&edge_shares);

    let mut issues = Vec::new();
    for (i, m) in clustered.iter().enumerate() {
        let edge_share = module_edge_shares[i].1;

        if edge_share > share_fence || m.node_ids.len() as f64 > size_fence {
            let sev = stats::percentile_ranks(&edge_shares)[i];
            issues.push(IssueOutput {
                id: issue_id("god_module", &[m.label.clone()], &[]),
                kind: "god_module".to_string(),
                title: format!("God module: {}", m.label),
                description: format!(
                    "Module {} has {} nodes ({:.0}% of edges). Consider splitting it into smaller, focused modules.",
                    m.label,
                    m.node_ids.len(),
                    edge_share * 100.0
                ),
                severity: round2(sev),
                severity_label: severity_label(sev),
                confidence: 0.7,
                confidence_label: "medium".to_string(),
                anchors: m
                    .node_ids
                    .iter()
                    .take(3)
                    .filter_map(|nid| ctx.graph.node_index.get(nid).map(|&i| ctx.graph.anchor(i)))
                    .collect(),
            });
        }
    }
    issues
}

// ---------------------------------------------------------------------------
// Low cohesion
// ---------------------------------------------------------------------------

fn detect_low_cohesion(ctx: &IssuesContext) -> Vec<IssueOutput> {
    let mut issues = Vec::new();
    for m in ctx.modules {
        if m.unassigned || m.node_ids.len() < 4 {
            continue;
        }
        let Some(cohesion) = m.cohesion else { continue };
        let Some(separation) = m.separation else { continue };
        if separation <= 0.0 {
            continue;
        }
        if cohesion > separation {
            let ratio = (cohesion - separation) / cohesion;
            let sev = (0.3 + ratio * 0.7).min(1.0);
            issues.push(IssueOutput {
                id: issue_id("low_cohesion", &[m.label.clone()], &[]),
                kind: "low_cohesion".to_string(),
                title: format!("Low cohesion: {}", m.label),
                description: format!(
                    "Module {} has higher internal spread ({:.3}) than distinctness ({:.3}) — \
                     members may belong to different concerns.",
                    m.label, cohesion, separation
                ),
                severity: round2(sev),
                severity_label: severity_label(sev),
                confidence: round2(m.confidence),
                confidence_label: confidence_label(m.confidence),
                anchors: m
                    .node_ids
                    .iter()
                    .take(3)
                    .filter_map(|nid| ctx.graph.node_index.get(nid).map(|&i| ctx.graph.anchor(i)))
                    .collect(),
            });
        }
    }
    issues
}

// ---------------------------------------------------------------------------
// Fragile hub
// ---------------------------------------------------------------------------

fn detect_fragile_hubs(ctx: &IssuesContext) -> Vec<IssueOutput> {
    let hub_roles: Vec<&RoleOutput> = ctx
        .roles
        .iter()
        .filter(|r| r.role == "hub")
        .collect();
    if hub_roles.is_empty() {
        return Vec::new();
    }

    let all_btw: Vec<f64> = ctx
        .roles
        .iter()
        .filter(|r| r.betweenness > 0.0)
        .map(|r| r.betweenness)
        .collect();
    if all_btw.is_empty() {
        return Vec::new();
    }

    let mut sorted_btw = all_btw.clone();
    sorted_btw.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let btw_90_idx = ((sorted_btw.len() as f64 * 0.9) as usize).min(sorted_btw.len() - 1);
    let btw_90 = sorted_btw[btw_90_idx];
    let max_btw = sorted_btw.last().copied().unwrap_or(1.0);
    let max_degree = ctx.roles.iter().map(|r| r.degree).max().unwrap_or(1);

    let mut issues = Vec::new();
    for r in &hub_roles {
        if r.betweenness >= btw_90 {
            let deg_pct = r.degree as f64 / max_degree.max(1) as f64;
            let btw_pct = r.betweenness / max_btw.max(f64::EPSILON);
            let sev = (0.5 + btw_pct * deg_pct * 0.5).min(1.0);
            let anchors = ctx
                .graph
                .node_index
                .get(&r.node_id)
                .map(|&i| vec![ctx.graph.anchor(i)])
                .unwrap_or_default();
            issues.push(IssueOutput {
                id: issue_id("fragile_hub", &[r.node_id.clone()], &[]),
                kind: "fragile_hub".to_string(),
                title: format!("Fragile hub: {}", r.node_id),
                description: format!(
                    "{} is both a structural hub (degree {}) and a high-betweenness bottleneck — \
                     single point of failure.",
                    r.node_id, r.degree
                ),
                severity: round2(sev),
                severity_label: severity_label(sev),
                confidence: 0.8,
                confidence_label: "high".to_string(),
                anchors,
            });
        }
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

    // Count distinct (src, tgt) symbol pairs per module boundary.
    let mut pair_symbols: HashMap<(usize, usize), HashSet<(usize, usize)>> = HashMap::new();
    for edges in ctx.graph.typed_edges.values() {
        for &(src, tgt) in edges {
            let src_mod = ctx.node_to_module.get(&ctx.graph.node_ids[src]);
            let tgt_mod = ctx.node_to_module.get(&ctx.graph.node_ids[tgt]);
            let (Some(&sm), Some(&tm)) = (src_mod, tgt_mod) else { continue };
            if sm == tm {
                continue;
            }
            let pair = if sm <= tm { (sm, tm) } else { (tm, sm) };
            pair_symbols.entry(pair).or_default().insert((src, tgt));
        }
    }

    if pair_symbols.is_empty() {
        return Vec::new();
    }

    let widths: Vec<usize> = pair_symbols.values().map(|s| s.len()).collect();
    let widths_f: Vec<f64> = widths.iter().map(|&w| w as f64).collect();
    let threshold = (stats::tukey_upper_fence(&widths_f) as usize).max(4);

    let module_by_id: HashMap<usize, &EnrichedModule> = ctx
        .modules
        .iter()
        .filter(|m| !m.unassigned)
        .map(|m| (m.id, m))
        .collect();

    let mut issues = Vec::new();
    for ((mod_a, mod_b), syms) in &pair_symbols {
        let width = syms.len();
        if width <= threshold {
            continue;
        }
        let label_a = module_by_id
            .get(mod_a)
            .map(|m| m.label.as_str())
            .unwrap_or("?");
        let label_b = module_by_id
            .get(mod_b)
            .map(|m| m.label.as_str())
            .unwrap_or("?");
        // Severity = percentile rank of this width in the distribution.
        let mut all_with_current = widths_f.clone();
        all_with_current.push(width as f64);
        let pcts = stats::percentile_ranks(&all_with_current);
        let sev = pcts.last().copied().unwrap_or(0.5);
        let median_width = stats::median(&widths_f) as usize;

        issues.push(IssueOutput {
            id: format!(
                "wide-interface:{}",
                sorted_join(&[label_a.to_string(), label_b.to_string()])
            ),
            kind: "wide_interface".to_string(),
            title: format!("Wide interface: {} — {}", label_a, label_b),
            description: format!(
                "{} distinct coupling points between {} and {} (median is {}). \
                 Consider narrowing the interface.",
                width, label_a, label_b, median_width
            ),
            severity: round2(sev),
            severity_label: severity_label(sev),
            confidence: 0.6,
            confidence_label: "medium".to_string(),
            anchors: Vec::new(),
        });
    }
    issues
}

// ---------------------------------------------------------------------------
// Phantom imports
// ---------------------------------------------------------------------------

fn detect_phantom_imports(ctx: &IssuesContext) -> Vec<IssueOutput> {
    let clustered: Vec<&EnrichedModule> = ctx.modules.iter().filter(|m| !m.unassigned).collect();
    if clustered.len() < 2 {
        return Vec::new();
    }

    // Collect cross-module import pairs and call pairs.
    let mut import_pairs: HashMap<(usize, usize), usize> = HashMap::new();
    let mut call_pairs: HashSet<(usize, usize)> = HashSet::new();

    for &(src, tgt) in ctx.graph.edges_of_kind("imports") {
        let src_mod = ctx.node_to_module.get(&ctx.graph.node_ids[src]);
        let tgt_mod = ctx.node_to_module.get(&ctx.graph.node_ids[tgt]);
        let (Some(&sm), Some(&tm)) = (src_mod, tgt_mod) else { continue };
        if sm == tm {
            continue;
        }
        let pair = if sm <= tm { (sm, tm) } else { (tm, sm) };
        *import_pairs.entry(pair).or_default() += 1;
    }

    for &(src, tgt) in ctx.graph.edges_of_kind("calls") {
        let src_mod = ctx.node_to_module.get(&ctx.graph.node_ids[src]);
        let tgt_mod = ctx.node_to_module.get(&ctx.graph.node_ids[tgt]);
        let (Some(&sm), Some(&tm)) = (src_mod, tgt_mod) else { continue };
        if sm == tm {
            continue;
        }
        let pair = if sm <= tm { (sm, tm) } else { (tm, sm) };
        call_pairs.insert(pair);
    }

    let module_by_id: HashMap<usize, &EnrichedModule> = ctx
        .modules
        .iter()
        .filter(|m| !m.unassigned)
        .map(|m| (m.id, m))
        .collect();

    let mut issues = Vec::new();
    for ((mod_a, mod_b), &count) in &import_pairs {
        let pair = (*mod_a, *mod_b);
        if call_pairs.contains(&pair) {
            continue;
        }
        let label_a = module_by_id
            .get(mod_a)
            .map(|m| m.label.as_str())
            .unwrap_or("?");
        let label_b = module_by_id
            .get(mod_b)
            .map(|m| m.label.as_str())
            .unwrap_or("?");
        let sev = (0.2 + count as f64 * 0.1).min(0.5);

        issues.push(IssueOutput {
            id: format!(
                "phantom-import:{}",
                sorted_join(&[label_a.to_string(), label_b.to_string()])
            ),
            kind: "phantom_import".to_string(),
            title: format!("Phantom import: {} — {}", label_a, label_b),
            description: format!(
                "{} import(s) between {} and {} with no corresponding calls — \
                 possibly unused coupling or type-only imports.",
                count, label_a, label_b
            ),
            severity: round2(sev),
            severity_label: severity_label(sev),
            confidence: 0.5,
            confidence_label: "medium".to_string(),
            anchors: Vec::new(),
        });
    }
    issues
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate stable, deterministic issue IDs matching Python `_issue_id()`.
fn issue_id(kind: &str, node_ids: &[String], packages: &[String]) -> String {
    match kind {
        "coverage" => "coverage:low-spectral".to_string(),
        "self_edge_drop" => "self-edge-drop:high".to_string(),
        "module_separation" => "module-separation:weak".to_string(),
        "reverse_dependency" => {
            let mut pkgs = packages.to_vec();
            pkgs.sort();
            format!("reverse-dependency:{}", pkgs.join(","))
        }
        "spectral_outlier" => {
            let label = node_ids.first().map(|s| s.as_str()).unwrap_or("unknown");
            format!("spectral-outlier:{}", label)
        }
        "cycle_member" => {
            let mut ids = node_ids.to_vec();
            ids.sort();
            format!("cycle:{}", ids.join(","))
        }
        "orphan" => {
            let mut ids = node_ids.to_vec();
            ids.sort();
            format!("orphan:{}", ids.join(","))
        }
        "cross_module" => {
            let mut ids = node_ids.to_vec();
            ids.sort();
            format!("cross-module:{}", ids.join(","))
        }
        "god_module" => {
            let label = node_ids.first().map(|s| s.as_str()).unwrap_or("unknown");
            format!("god-module:{}", label)
        }
        "low_cohesion" => {
            let label = node_ids.first().map(|s| s.as_str()).unwrap_or("unknown");
            format!("low-cohesion:{}", label)
        }
        "fragile_hub" => {
            let label = node_ids.first().map(|s| s.as_str()).unwrap_or("unknown");
            format!("fragile-hub:{}", label)
        }
        "wide_interface" => {
            let mut pkgs = packages.to_vec();
            pkgs.sort();
            format!("wide-interface:{}", pkgs.join(","))
        }
        "phantom_import" => {
            let mut pkgs = packages.to_vec();
            pkgs.sort();
            format!("phantom-import:{}", pkgs.join(","))
        }
        "layer_discrepancy" => {
            let label = node_ids.first().map(|s| s.as_str()).unwrap_or("unknown");
            format!("layer-discrepancy:{}", label)
        }
        _ => {
            let label = node_ids.first().map(|s| s.as_str()).unwrap_or("unknown");
            format!("{}:{}", kind, label)
        }
    }
}

fn anomaly_title(kind: AnomalyKind) -> String {
    match kind {
        AnomalyKind::CrossModule => "Unexpected reverse boundary".to_string(),
        AnomalyKind::SpectralOutlier => "Structural outlier".to_string(),
        AnomalyKind::CycleMember => "Dependency cycle".to_string(),
        AnomalyKind::LayerDiscrepancy => "Cross-layer discrepancy".to_string(),
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

fn top_package(node_id: &str) -> String {
    node_id.split('.').next().unwrap_or(node_id).to_string()
}

fn sorted_join(items: &[String]) -> String {
    let mut sorted = items.to_vec();
    sorted.sort();
    sorted.join(",")
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}
