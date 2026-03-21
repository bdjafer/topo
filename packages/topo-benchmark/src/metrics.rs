//! Pure metric computations: NMI, ARI, Boundary F1, AP, etc.

use std::collections::{HashMap, HashSet};

/// Geometric mean of values. Returns 0.0 if any value is <= 0 or slice is empty.
pub fn geometric_mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    if values.iter().any(|v| *v <= 0.0 || v.is_nan()) {
        return 0.0;
    }
    let log_sum: f64 = values.iter().map(|v| v.ln()).sum();
    (log_sum / values.len() as f64).exp()
}

/// Normalized Mutual Information between two label mappings on shared keys.
pub fn compute_nmi<A, B>(left: &HashMap<String, A>, right: &HashMap<String, B>) -> f64
where
    A: std::hash::Hash + Eq + Clone,
    B: std::hash::Hash + Eq + Clone,
{
    // Collect shared keys.
    let common: Vec<&String> = left.keys().filter(|k| right.contains_key(*k)).collect();
    let n = common.len();
    if n == 0 {
        return 0.0;
    }
    let nf = n as f64;

    // Build contingency counts.
    let mut joint: HashMap<(A, B), usize> = HashMap::new();
    let mut count_a: HashMap<A, usize> = HashMap::new();
    let mut count_b: HashMap<B, usize> = HashMap::new();

    for key in &common {
        let a = left[*key].clone();
        let b = right[*key].clone();
        *joint.entry((a.clone(), b.clone())).or_insert(0) += 1;
        *count_a.entry(a).or_insert(0) += 1;
        *count_b.entry(b).or_insert(0) += 1;
    }

    // Mutual information.
    let mut mi = 0.0;
    for ((a, b), &nij) in &joint {
        let pi = count_a[a] as f64 / nf;
        let pj = count_b[b] as f64 / nf;
        let pij = nij as f64 / nf;
        if pij > 0.0 && pi > 0.0 && pj > 0.0 {
            mi += pij * (pij / (pi * pj)).ln();
        }
    }

    // Entropy.
    let ha: f64 = count_a
        .values()
        .map(|&c| {
            let p = c as f64 / nf;
            if p > 0.0 { -p * p.ln() } else { 0.0 }
        })
        .sum();
    let hb: f64 = count_b
        .values()
        .map(|&c| {
            let p = c as f64 / nf;
            if p > 0.0 { -p * p.ln() } else { 0.0 }
        })
        .sum();

    if ha + hb == 0.0 {
        return 1.0;
    }
    (2.0 * mi / (ha + hb)).clamp(0.0, 1.0)
}

/// Adjusted Rand Index between two label mappings on shared keys.
pub fn compute_ari(left: &HashMap<String, usize>, right: &HashMap<String, usize>) -> f64 {
    let common: Vec<&String> = left.keys().filter(|k| right.contains_key(*k)).collect();
    let n = common.len();
    if n < 2 {
        return if n == 1 { 1.0 } else { 0.0 };
    }

    let mut joint: HashMap<(usize, usize), usize> = HashMap::new();
    let mut count_a: HashMap<usize, usize> = HashMap::new();
    let mut count_b: HashMap<usize, usize> = HashMap::new();

    for key in &common {
        let a = left[*key];
        let b = right[*key];
        *joint.entry((a, b)).or_insert(0) += 1;
        *count_a.entry(a).or_insert(0) += 1;
        *count_b.entry(b).or_insert(0) += 1;
    }

    let comb2 = |x: usize| -> f64 {
        if x < 2 { 0.0 } else { (x * (x - 1)) as f64 / 2.0 }
    };

    let total_pairs = comb2(n);
    let sum_joint: f64 = joint.values().map(|&c| comb2(c)).sum();
    let sum_a: f64 = count_a.values().map(|&c| comb2(c)).sum();
    let sum_b: f64 = count_b.values().map(|&c| comb2(c)).sum();

    let expected = sum_a * sum_b / total_pairs;
    let max_index = 0.5 * (sum_a + sum_b);
    let denom = max_index - expected;
    if denom.abs() < 1e-12 {
        return 1.0;
    }
    ((sum_joint - expected) / denom).clamp(-1.0, 1.0)
}

/// Boundary F1: binary F1 on the cross-module edge class.
///
/// For each edge between included nodes, classify as intra-module or cross-module
/// in both predicted and gold partitions. Compute F1 on positive class = cross-module.
pub fn compute_boundary_f1<A, B>(
    edges: &[(String, String)],
    predicted: &HashMap<String, A>,
    gold: &HashMap<String, B>,
) -> f64
where
    A: Eq,
    B: Eq,
{
    let mut tp: usize = 0;
    let mut fp: usize = 0;
    let mut fn_count: usize = 0;

    for (src, tgt) in edges {
        let (Some(p_src), Some(p_tgt)) = (predicted.get(src), predicted.get(tgt)) else {
            continue;
        };
        let (Some(g_src), Some(g_tgt)) = (gold.get(src), gold.get(tgt)) else {
            continue;
        };
        let pred_cross = p_src != p_tgt;
        let gold_cross = g_src != g_tgt;
        match (gold_cross, pred_cross) {
            (true, true) => tp += 1,
            (false, true) => fp += 1,
            (true, false) => fn_count += 1,
            (false, false) => {}
        }
    }

    if tp == 0 {
        return 0.0;
    }
    let precision = tp as f64 / (tp + fp) as f64;
    let recall = tp as f64 / (tp + fn_count) as f64;
    2.0 * precision * recall / (precision + recall)
}

/// Coverage: fraction of gold nodes that received a predicted module assignment.
pub fn compute_coverage(predicted: &HashMap<String, usize>, gold_nodes: &HashSet<String>) -> f64 {
    if gold_nodes.is_empty() {
        return 1.0;
    }
    let assigned = gold_nodes.iter().filter(|n| predicted.contains_key(*n)).count();
    assigned as f64 / gold_nodes.len() as f64
}

/// Average Precision from ranked predictions.
///
/// `items` is sorted by score descending. Returns mean precision at each positive.
pub fn compute_average_precision(items: &[(f64, bool)]) -> f64 {
    let total_pos = items.iter().filter(|(_, is_pos)| *is_pos).count();
    if total_pos == 0 || items.is_empty() {
        return 0.0;
    }

    // Sort by score descending (stable).
    let mut sorted: Vec<(f64, bool)> = items.to_vec();
    sorted.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut tp = 0;
    let mut ap_sum = 0.0;
    for (i, (_, is_pos)) in sorted.iter().enumerate() {
        if *is_pos {
            tp += 1;
            ap_sum += tp as f64 / (i + 1) as f64;
        }
    }
    ap_sum / total_pos as f64
}

/// Precision at cutoff k.
pub fn compute_precision_at_k(items: &[(f64, bool)], k: usize) -> f64 {
    if k == 0 || items.is_empty() {
        return 0.0;
    }

    let mut sorted: Vec<(f64, bool)> = items.to_vec();
    sorted.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let top_k = &sorted[..k.min(sorted.len())];
    let positives = top_k.iter().filter(|(_, is_pos)| *is_pos).count();
    positives as f64 / k as f64
}

/// Macro F1 of structural roles on shared nodes.
pub fn compute_role_macro_f1(
    base_roles: &HashMap<String, String>,
    pert_roles: &HashMap<String, String>,
) -> f64 {
    let shared: Vec<&String> = base_roles
        .keys()
        .filter(|k| pert_roles.contains_key(*k))
        .collect();
    if shared.is_empty() {
        return 1.0;
    }

    let all_roles: HashSet<&str> = shared
        .iter()
        .flat_map(|k| [base_roles[*k].as_str(), pert_roles[*k].as_str()])
        .collect();

    let mut f1_scores = Vec::new();
    for role in &all_roles {
        let mut tp = 0usize;
        let mut fp = 0usize;
        let mut fn_count = 0usize;
        for k in &shared {
            let base = base_roles[*k].as_str();
            let pert = pert_roles[*k].as_str();
            match (base == *role, pert == *role) {
                (true, true) => tp += 1,
                (false, true) => fp += 1,
                (true, false) => fn_count += 1,
                (false, false) => {}
            }
        }
        let f1 = if tp == 0 {
            0.0
        } else {
            let p = tp as f64 / (tp + fp) as f64;
            let r = tp as f64 / (tp + fn_count) as f64;
            2.0 * p * r / (p + r)
        };
        f1_scores.push(f1);
    }

    if f1_scores.is_empty() {
        return 1.0;
    }
    f1_scores.iter().sum::<f64>() / f1_scores.len() as f64
}

/// IoU (Intersection over Union) on two node sets.
pub fn node_set_iou(a: &HashSet<String>, b: &HashSet<String>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let intersection = a.intersection(b).count();
    let union = a.union(b).count();
    if union == 0 {
        return 0.0;
    }
    intersection as f64 / union as f64
}

/// Hierarchical IoU: maps predicted nodes to gold ancestors before computing IoU.
///
/// A predicted node matches a gold node if the gold node is a dot-prefix ancestor
/// (i.e., `predicted` starts with `gold` + ".") or an exact match.
/// After mapping, compute standard set IoU.
pub fn node_set_iou_hierarchical(predicted: &HashSet<String>, gold: &HashSet<String>) -> f64 {
    if predicted.is_empty() && gold.is_empty() {
        return 1.0;
    }
    if predicted.is_empty() || gold.is_empty() {
        return 0.0;
    }

    // Map each predicted node to its best gold ancestor, if any.
    let mapped: HashSet<String> = predicted
        .iter()
        .map(|p| {
            if gold.contains(p) {
                return p.clone();
            }
            for g in gold {
                if p.starts_with(g.as_str()) && p.as_bytes().get(g.len()) == Some(&b'.') {
                    return g.clone();
                }
            }
            p.clone()
        })
        .collect();

    let intersection = mapped.intersection(gold).count();
    let union = mapped.union(gold).count();
    if union == 0 {
        return 0.0;
    }
    intersection as f64 / union as f64
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn map_from(pairs: &[(&str, usize)]) -> HashMap<String, usize> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), *v))
            .collect()
    }

    fn smap_from(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    // ── geometric_mean ───────────────────────────────────────────────────

    #[test]
    fn test_geometric_mean_basic() {
        let v = geometric_mean(&[4.0, 9.0]);
        assert!((v - 6.0).abs() < 1e-9);
    }

    #[test]
    fn test_geometric_mean_single() {
        assert!((geometric_mean(&[7.0]) - 7.0).abs() < 1e-9);
    }

    #[test]
    fn test_geometric_mean_with_zero() {
        assert_eq!(geometric_mean(&[0.0, 5.0]), 0.0);
    }

    #[test]
    fn test_geometric_mean_empty() {
        assert_eq!(geometric_mean(&[]), 0.0);
    }

    // ── NMI ──────────────────────────────────────────────────────────────

    #[test]
    fn test_nmi_identical() {
        let a = map_from(&[("x", 0), ("y", 0), ("z", 1)]);
        assert!((compute_nmi(&a, &a) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_nmi_independent() {
        // Two partitions with no mutual information.
        let a = map_from(&[("a", 0), ("b", 0), ("c", 1), ("d", 1)]);
        let b = map_from(&[("a", 0), ("b", 1), ("c", 0), ("d", 1)]);
        let nmi = compute_nmi(&a, &b);
        assert!(nmi < 0.01, "expected ~0.0, got {nmi}");
    }

    #[test]
    fn test_nmi_refinement() {
        // b refines a: a has 2 clusters, b splits one of them.
        let a = map_from(&[("a", 0), ("b", 0), ("c", 0), ("d", 1), ("e", 1)]);
        let b = map_from(&[("a", 0), ("b", 0), ("c", 2), ("d", 1), ("e", 1)]);
        let nmi = compute_nmi(&a, &b);
        assert!(nmi > 0.3, "expected positive NMI for refinement, got {nmi}");
        assert!(nmi < 1.0);
    }

    #[test]
    fn test_nmi_empty() {
        let a: HashMap<String, usize> = HashMap::new();
        let b = map_from(&[("x", 0)]);
        assert_eq!(compute_nmi(&a, &b), 0.0);
    }

    // ── ARI ──────────────────────────────────────────────────────────────

    #[test]
    fn test_ari_identical() {
        let a = map_from(&[("x", 0), ("y", 0), ("z", 1)]);
        assert!((compute_ari(&a, &a) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_ari_random() {
        let a = map_from(&[("a", 0), ("b", 0), ("c", 1), ("d", 1)]);
        let b = map_from(&[("a", 0), ("b", 1), ("c", 0), ("d", 1)]);
        let ari = compute_ari(&a, &b);
        assert!(ari.abs() < 0.5, "expected near-zero ARI, got {ari}");
    }

    #[test]
    fn test_ari_single_shared() {
        let a = map_from(&[("x", 0)]);
        let b = map_from(&[("x", 1)]);
        assert_eq!(compute_ari(&a, &b), 1.0);
    }

    // ── Boundary F1 ─────────────────────────────────────────────────────

    #[test]
    fn test_boundary_f1_perfect() {
        let edges = vec![
            ("a".to_string(), "b".to_string()),
            ("a".to_string(), "c".to_string()),
        ];
        let pred = map_from(&[("a", 0), ("b", 0), ("c", 1)]);
        let gold = map_from(&[("a", 0), ("b", 0), ("c", 1)]);
        assert!((compute_boundary_f1(&edges, &pred, &gold) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_boundary_f1_zero() {
        let edges = vec![("a".to_string(), "b".to_string())];
        // Gold says cross, predicted says intra.
        let pred = map_from(&[("a", 0), ("b", 0)]);
        let gold: HashMap<String, &str> =
            [("a".to_string(), "x"), ("b".to_string(), "y")].into_iter().collect();
        assert_eq!(compute_boundary_f1(&edges, &pred, &gold), 0.0);
    }

    // ── Coverage ─────────────────────────────────────────────────────────

    #[test]
    fn test_coverage_full() {
        let pred = map_from(&[("a", 0), ("b", 1)]);
        let gold: HashSet<String> = ["a", "b"].iter().map(|s| s.to_string()).collect();
        assert_eq!(compute_coverage(&pred, &gold), 1.0);
    }

    #[test]
    fn test_coverage_partial() {
        let pred = map_from(&[("a", 0)]);
        let gold: HashSet<String> = ["a", "b"].iter().map(|s| s.to_string()).collect();
        assert_eq!(compute_coverage(&pred, &gold), 0.5);
    }

    // ── Average Precision ────────────────────────────────────────────────

    #[test]
    fn test_ap_perfect() {
        // All positives ranked first.
        let items = vec![(1.0, true), (0.9, true), (0.1, false)];
        assert!((compute_average_precision(&items) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_ap_worst() {
        // All positives ranked last.
        let items = vec![(1.0, false), (0.9, false), (0.1, true)];
        let ap = compute_average_precision(&items);
        assert!((ap - 1.0 / 3.0).abs() < 1e-9);
    }

    // ── Precision@k ──────────────────────────────────────────────────────

    #[test]
    fn test_precision_at_k() {
        let items = vec![(1.0, true), (0.9, false), (0.8, true)];
        assert_eq!(compute_precision_at_k(&items, 1), 1.0);
        assert_eq!(compute_precision_at_k(&items, 2), 0.5);
    }

    // ── Macro F1 ─────────────────────────────────────────────────────────

    #[test]
    fn test_role_macro_f1_perfect() {
        let a = smap_from(&[("x", "hub"), ("y", "utility"), ("z", "regular")]);
        assert!((compute_role_macro_f1(&a, &a) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_role_macro_f1_all_wrong() {
        let a = smap_from(&[("x", "hub"), ("y", "utility")]);
        let b = smap_from(&[("x", "utility"), ("y", "hub")]);
        assert_eq!(compute_role_macro_f1(&a, &b), 0.0);
    }

    // ── IoU ──────────────────────────────────────────────────────────────

    #[test]
    fn test_iou_identical() {
        let a: HashSet<String> = ["x", "y"].iter().map(|s| s.to_string()).collect();
        assert_eq!(node_set_iou(&a, &a), 1.0);
    }

    #[test]
    fn test_iou_disjoint() {
        let a: HashSet<String> = ["x"].iter().map(|s| s.to_string()).collect();
        let b: HashSet<String> = ["y"].iter().map(|s| s.to_string()).collect();
        assert_eq!(node_set_iou(&a, &b), 0.0);
    }

    #[test]
    fn test_iou_partial() {
        let a: HashSet<String> = ["x", "y"].iter().map(|s| s.to_string()).collect();
        let b: HashSet<String> = ["y", "z"].iter().map(|s| s.to_string()).collect();
        assert!((node_set_iou(&a, &b) - 1.0 / 3.0).abs() < 1e-9);
    }

    // ── Hierarchical IoU ─────────────────────────────────────────────────

    #[test]
    fn test_iou_hierarchical_exact() {
        let a: HashSet<String> = ["x", "y"].iter().map(|s| s.to_string()).collect();
        assert_eq!(node_set_iou_hierarchical(&a, &a), 1.0);
    }

    #[test]
    fn test_iou_hierarchical_parent_child() {
        let predicted: HashSet<String> = [
            "api.routes.submit_order",
            "core.service.create_order",
            "data.store.save_order",
        ].iter().map(|s| s.to_string()).collect();
        let gold: HashSet<String> = [
            "api.routes",
            "core.service",
            "data.store",
        ].iter().map(|s| s.to_string()).collect();
        assert!((node_set_iou_hierarchical(&predicted, &gold) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_iou_hierarchical_partial() {
        let predicted: HashSet<String> = [
            "api.routes.submit_order",
            "unknown.node",
        ].iter().map(|s| s.to_string()).collect();
        let gold: HashSet<String> = [
            "api.routes",
            "core.service",
        ].iter().map(|s| s.to_string()).collect();
        // mapped: {api.routes, unknown.node}
        // intersection with gold: {api.routes} = 1
        // union: {api.routes, unknown.node, core.service} = 3
        assert!((node_set_iou_hierarchical(&predicted, &gold) - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_iou_hierarchical_no_false_prefix() {
        let predicted: HashSet<String> = ["api.routesx"].iter().map(|s| s.to_string()).collect();
        let gold: HashSet<String> = ["api.routes"].iter().map(|s| s.to_string()).collect();
        // "api.routesx" does NOT start with "api.routes."
        assert_eq!(node_set_iou_hierarchical(&predicted, &gold), 0.0);
    }
}
