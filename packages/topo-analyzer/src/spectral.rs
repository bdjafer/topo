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

        for (_, result) in &mut clusterable {
            for row in &mut result.eigenvectors {
                row.resize(max_k, 0.0);
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
        .map(|&d| if d > 0.0 { 1.0 / d.sqrt() } else { 1.0 })
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
}
