//! Spectral decomposition of code graphs.
//!
//! Computes eigendecomposition of the normalized graph Laplacian to extract
//! global structural properties. Each node gets a spectral fingerprint.

use faer::Mat;

use crate::graph::Graph;

/// Result of eigendecomposition for one connected component.
pub struct DecompResult {
    /// Eigenvalues (smallest first, trivial λ₀≈0 already stripped).
    pub eigenvalues: Vec<f64>,
    /// Eigenvectors as rows: node_count × k matrix.
    pub eigenvectors: Vec<Vec<f64>>,
}

/// Full spectral result across all components.
pub struct SpectralResult {
    /// Per-component results, ordered by component size descending.
    pub components: Vec<(Vec<usize>, DecompResult)>,
    /// Unassigned components (too small to decompose).
    pub unassigned: Vec<Vec<usize>>,
    /// Fiedler value from the primary (largest) component.
    pub fiedler_value: f64,
    /// Sizes of all components.
    pub component_sizes: Vec<usize>,
}

/// Minimum component size to attempt eigendecomposition.
const MIN_COMPONENT_SIZE: usize = 4;

/// Run spectral decomposition on the graph.
///
/// Handles disconnected graphs by decomposing each connected component
/// separately, then padding eigenvectors to a uniform width.
pub fn decompose(graph: &Graph, k: usize) -> SpectralResult {
    let components = graph.connected_components();
    let component_sizes: Vec<usize> = components.iter().map(|c| c.len()).collect();

    let mut clusterable: Vec<(Vec<usize>, DecompResult)> = Vec::new();
    let mut unassigned: Vec<Vec<usize>> = Vec::new();
    let mut fiedler_value = 0.0;

    for (ci, component) in components.iter().enumerate() {
        if component.len() < MIN_COMPONENT_SIZE {
            unassigned.push(component.clone());
            continue;
        }

        // Build sub-adjacency for this component.
        let sub_adj = extract_subgraph(graph, component);
        let n = component.len();
        let actual_k = k.min(n.saturating_sub(1)).max(1);

        match decompose_core(&sub_adj, n, actual_k) {
            Some(result) => {
                if ci == 0 && !result.eigenvalues.is_empty() {
                    fiedler_value = result.eigenvalues[0];
                }
                clusterable.push((component.clone(), result));
            }
            None => {
                unassigned.push(component.clone());
            }
        }
    }

    // Pad eigenvectors to uniform width across all clusterable components.
    if !clusterable.is_empty() {
        let max_k = clusterable
            .iter()
            .map(|(_, r)| {
                r.eigenvectors
                    .first()
                    .map(|v| v.len())
                    .unwrap_or(0)
            })
            .max()
            .unwrap_or(0);

        let mut pad_rng = crate::stats::Rng::new(12345);
        for (_, result) in &mut clusterable {
            for row in &mut result.eigenvectors {
                let current_len = row.len();
                row.resize(max_k, 0.0);
                // Pad with tiny deterministic noise instead of zeros to avoid
                // biasing nodes from smaller components toward a default cluster.
                for j in current_len..max_k {
                    row[j] = (pad_rng.next_f64() - 0.5) * 1e-6;
                }
            }
        }
    }

    SpectralResult {
        components: clusterable,
        unassigned,
        fiedler_value,
        component_sizes,
    }
}

/// Compute the Fiedler value (λ₂) and Fiedler vector for a subgraph induced
/// by the given node indices. Returns None if the subgraph is too small or
/// disconnected (λ₂ ≈ 0 means disconnected, but we still return it).
pub fn subgraph_fiedler(graph: &Graph, node_indices: &[usize]) -> Option<(f64, Vec<f64>)> {
    let n = node_indices.len();
    if n < 5 {
        return None;
    }
    let adjacency = extract_subgraph(graph, node_indices);
    let result = decompose_core(&adjacency, n, 1)?; // k=1 → we get λ₂ and v₂
    let fiedler_value = result.eigenvalues.first().copied().unwrap_or(0.0);
    let fiedler_vector = result.eigenvectors.iter().map(|row| row[0]).collect();
    Some((fiedler_value, fiedler_vector))
}

/// Estimate k from the eigengap heuristic (von Luxburg 2007).
///
/// Finds the largest jump between consecutive eigenvalues: k = argmax(λ_{i+1} - λ_i) + 1.
/// Eigenvalues must be sorted ascending (as produced by `decompose_core`).
/// Returns at least 2.
pub fn eigengap_k(eigenvalues: &[f64]) -> usize {
    if eigenvalues.len() < 2 {
        return 2;
    }
    let search = eigenvalues.len().min(20);
    let best_gap_idx = (0..search - 1)
        .max_by(|&i, &j| {
            let gap_i = eigenvalues[i + 1] - eigenvalues[i];
            let gap_j = eigenvalues[j + 1] - eigenvalues[j];
            gap_i
                .partial_cmp(&gap_j)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0);
    (best_gap_idx + 1).max(2)
}

/// Extract the symmetric adjacency sub-matrix for a component.
/// Returns a dense n×n row-major matrix.
fn extract_subgraph(graph: &Graph, component: &[usize]) -> Vec<f64> {
    let n = component.len();
    let mut local_index = vec![0usize; graph.n];
    for (li, &gi) in component.iter().enumerate() {
        local_index[gi] = li;
    }

    let mut sub = vec![0.0f64; n * n];
    for &gi in component {
        let li = local_index[gi];
        for &(tgt, w) in &graph.adj[gi] {
            if component.contains(&tgt) {
                let lt = local_index[tgt];
                sub[li * n + lt] += w;
                sub[lt * n + li] += w;
            }
        }
    }
    sub
}

/// Core eigendecomposition on a symmetric adjacency matrix.
///
/// Builds the normalized Laplacian L = I - D^{-1/2} A D^{-1/2}
/// and computes the k+1 smallest eigenvalues/vectors, stripping
/// the trivial zero eigenvalue.
fn decompose_core(adjacency: &[f64], n: usize, k: usize) -> Option<DecompResult> {
    if n < 2 || k == 0 {
        return None;
    }

    // Compute degree vector from symmetric adjacency.
    let mut degrees = vec![0.0f64; n];
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..n {
            sum += adjacency[i * n + j];
        }
        degrees[i] = sum;
    }

    // D^{-1/2} — guard against zero-degree nodes.
    let d_inv_sqrt: Vec<f64> = degrees
        .iter()
        .map(|&d| if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 })
        .collect();

    // Build normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
    let laplacian = Mat::<f64>::from_fn(n, n, |i, j| {
        let diag = if i == j { 1.0 } else { 0.0 };
        diag - d_inv_sqrt[i] * adjacency[i * n + j] * d_inv_sqrt[j]
    });

    // Eigendecomposition of the symmetric Laplacian.
    let evd = laplacian
        .self_adjoint_eigen(faer::Side::Lower)
        .expect("eigendecomposition failed");
    // Extract eigenvalues into a plain Vec.
    let mut eigenval_vec: Vec<f64> = Vec::with_capacity(n);
    evd.S().column_vector().for_each(|&v| eigenval_vec.push(v));

    // Extract eigenvector matrix into row-major Vec<Vec<f64>>.
    let u = evd.U();
    let eigvec_matrix: Vec<Vec<f64>> = (0..n)
        .map(|row| {
            let mut r = Vec::with_capacity(n);
            for col in 0..n {
                r.push(u[(row, col)]);
            }
            r
        })
        .collect();

    // Sort eigenvalues by ascending value.
    let mut indexed_eigenvalues: Vec<(usize, f64)> = eigenval_vec
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed_eigenvalues.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Skip the trivial zero eigenvalue (first), take k.
    let take = k.min(n.saturating_sub(1));
    if take == 0 {
        return None;
    }

    let eigenvalues: Vec<f64> = indexed_eigenvalues[1..=take]
        .iter()
        .map(|&(_, v)| v)
        .collect();

    let eigenvectors: Vec<Vec<f64>> = (0..n)
        .map(|row| {
            indexed_eigenvalues[1..=take]
                .iter()
                .map(|&(col, _)| eigvec_matrix[row][col])
                .collect()
        })
        .collect();

    Some(DecompResult {
        eigenvalues,
        eigenvectors,
    })
}

/// Produce spectral positional encodings for all nodes.
///
/// Returns (pe_vecs, pe_vals) where:
///   pe_vecs[global_node_idx] = [u₁(v), u₂(v), ..., u_k(v)]  (eigenvector components)
///   pe_vals[global_node_idx] = [λ₁, λ₂, ..., λ_k]            (eigenvalues of v's component)
///
/// Nodes in small/unassigned components get all-zero vectors.
/// Padded with zeros (not noise) to exactly k columns.
///
/// SAFETY: This function bounds reads by `eigenvalues.len()`, NOT `eigenvectors[li].len()`.
/// The `decompose()` function noise-pads eigenvectors in-place to `max_k` width for clustering,
/// but eigenvalues are never padded. Using `eigenvalues.len()` as the bound ensures we only
/// read real eigenvector columns, not noise-padded ones.
pub fn spectral_pe_export(
    spectral_result: &SpectralResult,
    n: usize,
    k: usize,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut pe_vecs = vec![vec![0.0f64; k]; n];
    let mut pe_vals = vec![vec![0.0f64; k]; n];

    for (component_indices, decomp) in &spectral_result.components {
        // Bound by eigenvalues.len(), not eigenvectors width — eigenvectors may have been
        // noise-padded to max_k by decompose(), but eigenvalues reflect the true count.
        let actual_k = k.min(decomp.eigenvalues.len());
        debug_assert!(
            decomp.eigenvectors.first().map_or(true, |row| row.len() >= actual_k),
            "eigenvector width ({}) < actual_k ({}): eigenvalues/eigenvectors mismatch",
            decomp.eigenvectors.first().map_or(0, |r| r.len()),
            actual_k,
        );

        for (li, &gi) in component_indices.iter().enumerate() {
            for j in 0..actual_k {
                pe_vecs[gi][j] = decomp.eigenvectors[li][j];
                pe_vals[gi][j] = decomp.eigenvalues[j];
            }
            // Columns actual_k..k remain zero from initialization.
        }
    }

    // Unassigned nodes already have all-zero vectors from initialization.

    (pe_vecs, pe_vals)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decompose_core_simple() {
        // Simple 4-node path graph: 0-1-2-3
        let n = 4;
        #[rustfmt::skip]
        let adj = vec![
            0.0, 1.0, 0.0, 0.0,
            1.0, 0.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 1.0,
            0.0, 0.0, 1.0, 0.0,
        ];
        let result = decompose_core(&adj, n, 3).unwrap();
        assert_eq!(result.eigenvalues.len(), 3);
        assert_eq!(result.eigenvectors.len(), 4);
        // Eigenvalues should be positive (Laplacian is PSD).
        for &ev in &result.eigenvalues {
            assert!(ev >= -1e-10, "eigenvalue should be non-negative: {ev}");
        }
    }

    #[test]
    fn test_pe_export_zero_padding() {
        // Component with 3 nodes and 4 eigenvectors of width 4.
        // Request k=16 → first 4 columns have values, columns 4..16 are zero.
        let decomp = DecompResult {
            eigenvalues: vec![0.5, 1.0, 1.5, 2.0],
            eigenvectors: vec![
                vec![0.1, 0.2, 0.3, 0.4],
                vec![0.5, 0.6, 0.7, 0.8],
                vec![0.9, 1.0, 1.1, 1.2],
            ],
        };
        let sr = SpectralResult {
            components: vec![(vec![0, 1, 2], decomp)],
            unassigned: vec![],
            fiedler_value: 0.5,
            component_sizes: vec![3],
        };

        let (pe_vecs, pe_vals) = spectral_pe_export(&sr, 3, 16);

        assert_eq!(pe_vecs.len(), 3);
        assert_eq!(pe_vals.len(), 3);

        for i in 0..3 {
            assert_eq!(pe_vecs[i].len(), 16);
            assert_eq!(pe_vals[i].len(), 16);
            // First 4 columns should have real values.
            for j in 0..4 {
                assert!(pe_vecs[i][j] != 0.0, "pe_vecs[{i}][{j}] should be non-zero");
                assert!(pe_vals[i][j] != 0.0, "pe_vals[{i}][{j}] should be non-zero");
            }
            // Columns 4..16 should be zero.
            for j in 4..16 {
                assert_eq!(pe_vecs[i][j], 0.0, "pe_vecs[{i}][{j}] should be zero");
                assert_eq!(pe_vals[i][j], 0.0, "pe_vals[{i}][{j}] should be zero");
            }
        }
    }

    #[test]
    fn test_pe_export_truncation() {
        // Component with 2 nodes and 20 eigenvalues/eigenvector columns.
        // Request k=16 → only first 16 columns used.
        let eigenvalues: Vec<f64> = (1..=20).map(|i| i as f64 * 0.1).collect();
        let row: Vec<f64> = (1..=20).map(|i| i as f64 * 0.01).collect();
        let decomp = DecompResult {
            eigenvalues,
            eigenvectors: vec![row.clone(), row],
        };
        let sr = SpectralResult {
            components: vec![(vec![0, 1], decomp)],
            unassigned: vec![],
            fiedler_value: 0.1,
            component_sizes: vec![2],
        };

        let (pe_vecs, pe_vals) = spectral_pe_export(&sr, 2, 16);

        assert_eq!(pe_vecs[0].len(), 16);
        assert_eq!(pe_vals[0].len(), 16);
        // Column 15 (0-indexed) should have the 16th eigenvalue (1.6).
        assert!((pe_vals[0][15] - 1.6).abs() < 1e-10);
        // Column 0 should have the 1st eigenvector value (0.01).
        assert!((pe_vecs[0][0] - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_pe_export_per_component_eigenvalues() {
        // Two components with different eigenvalues.
        let decomp_a = DecompResult {
            eigenvalues: vec![1.0, 2.0],
            eigenvectors: vec![
                vec![0.1, 0.2],
                vec![0.3, 0.4],
            ],
        };
        let decomp_b = DecompResult {
            eigenvalues: vec![5.0, 6.0],
            eigenvectors: vec![
                vec![0.5, 0.6],
                vec![0.7, 0.8],
            ],
        };
        let sr = SpectralResult {
            components: vec![
                (vec![0, 1], decomp_a),
                (vec![2, 3], decomp_b),
            ],
            unassigned: vec![],
            fiedler_value: 1.0,
            component_sizes: vec![2, 2],
        };

        let (pe_vecs, pe_vals) = spectral_pe_export(&sr, 4, 4);

        // Nodes 0, 1 get eigenvalues from component A.
        assert!((pe_vals[0][0] - 1.0).abs() < 1e-10);
        assert!((pe_vals[0][1] - 2.0).abs() < 1e-10);
        assert!((pe_vals[1][0] - 1.0).abs() < 1e-10);
        assert!((pe_vals[1][1] - 2.0).abs() < 1e-10);

        // Nodes 2, 3 get eigenvalues from component B.
        assert!((pe_vals[2][0] - 5.0).abs() < 1e-10);
        assert!((pe_vals[2][1] - 6.0).abs() < 1e-10);
        assert!((pe_vals[3][0] - 5.0).abs() < 1e-10);
        assert!((pe_vals[3][1] - 6.0).abs() < 1e-10);

        // Eigenvectors should be from respective components.
        assert!((pe_vecs[0][0] - 0.1).abs() < 1e-10);
        assert!((pe_vecs[2][0] - 0.5).abs() < 1e-10);

        // Columns 2, 3 should be zero (only 2 eigenvalues per component).
        for i in 0..4 {
            assert_eq!(pe_vals[i][2], 0.0);
            assert_eq!(pe_vals[i][3], 0.0);
            assert_eq!(pe_vecs[i][2], 0.0);
            assert_eq!(pe_vecs[i][3], 0.0);
        }
    }

    #[test]
    fn test_pe_export_unassigned_nodes() {
        // One component with nodes 0..3, unassigned nodes 5 and 6.
        let decomp = DecompResult {
            eigenvalues: vec![1.0, 2.0],
            eigenvectors: vec![
                vec![0.1, 0.2],
                vec![0.3, 0.4],
                vec![0.5, 0.6],
                vec![0.7, 0.8],
            ],
        };
        let sr = SpectralResult {
            components: vec![(vec![0, 1, 2, 3], decomp)],
            unassigned: vec![vec![5, 6]],
            fiedler_value: 1.0,
            component_sizes: vec![4, 2],
        };

        let (pe_vecs, pe_vals) = spectral_pe_export(&sr, 7, 4);

        // Nodes 5 and 6 should be all zeros.
        for j in 0..4 {
            assert_eq!(pe_vecs[5][j], 0.0, "pe_vecs[5][{j}] should be zero");
            assert_eq!(pe_vals[5][j], 0.0, "pe_vals[5][{j}] should be zero");
            assert_eq!(pe_vecs[6][j], 0.0, "pe_vecs[6][{j}] should be zero");
            assert_eq!(pe_vals[6][j], 0.0, "pe_vals[6][{j}] should be zero");
        }

        // Node 4 (not in any component or unassigned list) should also be zero.
        for j in 0..4 {
            assert_eq!(pe_vecs[4][j], 0.0);
            assert_eq!(pe_vals[4][j], 0.0);
        }

        // Nodes 0..3 should have real values.
        assert!((pe_vecs[0][0] - 0.1).abs() < 1e-10);
        assert!((pe_vals[0][0] - 1.0).abs() < 1e-10);
    }
}
