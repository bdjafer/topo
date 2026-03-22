//! Random Walk Positional Encoding (RWPE).
//!
//! For each node v, computes rwpe(v) = [P^1(v,v), P^2(v,v), ..., P^K(v,v)]
//! where P = D^{-1}A is the random walk transition matrix on the symmetrized
//! adjacency built from coupling edges (calls, imports, inherits — NOT defines).

use crate::graph::Graph;

/// Edge kinds that contribute to coupling (not containment).
const COUPLING_KINDS: &[&str] = &["calls", "imports", "inherits"];

/// CSR (Compressed Sparse Row) representation for sparse matrix.
struct CsrMatrix {
    /// Row pointers: indptr[i]..indptr[i+1] gives the range into indices/data
    /// for row i. Length = n + 1.
    indptr: Vec<usize>,
    /// Column indices for nonzero entries.
    indices: Vec<usize>,
    /// Values for nonzero entries.
    data: Vec<f64>,
}

/// Compute Random Walk Positional Encoding for all nodes.
///
/// Returns `rwpe[node_index] = [P^1(v,v), P^2(v,v), ..., P^K(v,v)]`
/// where P is the random walk transition matrix on the symmetrized adjacency
/// built from coupling edges (calls, imports, inherits — NOT defines).
///
/// * `graph` — the code graph
/// * `k` — number of random walk steps (length of each RWPE vector)
/// * `batch_size` — number of nodes processed simultaneously (controls memory)
pub fn compute_rwpe(graph: &Graph, k: usize, batch_size: usize) -> Vec<Vec<f64>> {
    let n = graph.n;

    // Result: rwpe[v] = vec of length k, initialized to zeros.
    let mut rwpe = vec![vec![0.0f64; k]; n];

    if n == 0 || k == 0 {
        return rwpe;
    }

    // Step 1: Build symmetrized adjacency from typed_edges (coupling kinds only).
    // We accumulate into a HashMap<(usize, usize), f64> to handle duplicates,
    // then convert to CSR.
    let mut sym_adj: Vec<std::collections::HashMap<usize, f64>> = vec![Default::default(); n];

    for kind in COUPLING_KINDS {
        for &(u, v) in graph.edges_of_kind(kind) {
            // Skip any out-of-range indices defensively.
            if u >= n || v >= n {
                continue;
            }
            *sym_adj[u].entry(v).or_insert(0.0) += 1.0;
            *sym_adj[v].entry(u).or_insert(0.0) += 1.0;
        }
    }

    // Step 2: Compute degree vector.
    let degree: Vec<f64> = sym_adj
        .iter()
        .map(|row| row.values().sum::<f64>())
        .collect();

    // Step 3: Build CSR transition matrix P where P[v][u] = A_sym[v][u] / d[v].
    let mut indptr = Vec::with_capacity(n + 1);
    let mut indices = Vec::new();
    let mut data = Vec::new();
    indptr.push(0);

    for v in 0..n {
        let d = degree[v];
        if d > 0.0 {
            // Sort column indices for deterministic iteration (not strictly required
            // for correctness but helps reproducibility).
            let mut cols: Vec<usize> = sym_adj[v].keys().copied().collect();
            cols.sort_unstable();
            for &u in &cols {
                let val = sym_adj[v][&u];
                indices.push(u);
                data.push(val / d);
            }
        }
        indptr.push(indices.len());
    }

    let csr = CsrMatrix {
        indptr,
        indices,
        data,
    };

    // Step 4: Process nodes in batches.
    let batch_size = batch_size.max(1); // ensure at least 1
    let mut batch_start = 0;

    while batch_start < n {
        let actual_b = (n - batch_start).min(batch_size);

        // X is an n × actual_b dense matrix stored in column-major order.
        // X[row, col] is at x[col * n + row].
        // Initialize as identity columns: X[batch_start + j][j] = 1.0
        let mut x = vec![0.0f64; n * actual_b];
        for j in 0..actual_b {
            x[j * n + (batch_start + j)] = 1.0;
        }

        #[allow(clippy::needless_range_loop)]
        for k_step in 0..k {
            // Compute X_new = P * X (sparse CSR times dense column-major).
            let mut x_new = vec![0.0f64; n * actual_b];

            for row in 0..n {
                let start = csr.indptr[row];
                let end = csr.indptr[row + 1];
                if start == end {
                    continue;
                }
                for j in 0..actual_b {
                    let col_offset = j * n;
                    let mut sum = 0.0f64;
                    for idx in start..end {
                        let col = csr.indices[idx];
                        let val = csr.data[idx];
                        sum += val * x[col_offset + col];
                    }
                    x_new[col_offset + row] = sum;
                }
            }

            x = x_new;

            // Extract diagonal: rwpe[batch_start + j][k_step] = X[batch_start + j][j]
            for j in 0..actual_b {
                let node = batch_start + j;
                rwpe[node][k_step] = x[j * n + node];
            }
        }

        batch_start += actual_b;
    }

    // Step 5: Clamp all values to [0.0, 1.0].
    for row in &mut rwpe {
        for val in row.iter_mut() {
            *val = val.clamp(0.0, 1.0);
        }
    }

    rwpe
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AnalyzerInput, EdgeEntry, NodeEntry};

    /// Helper: build a test graph with n nodes (n0, n1, ...) and typed edges.
    fn make_test_graph(n: usize, edges: Vec<(usize, usize, &str)>) -> Graph {
        let nodes: Vec<NodeEntry> = (0..n)
            .map(|i| NodeEntry {
                id: format!("n{i}"),
                kind: "function".to_string(),
                file: None,
                line: None,
                line_end: None,
            })
            .collect();
        let edge_entries: Vec<EdgeEntry> = edges
            .iter()
            .map(|(s, t, k)| EdgeEntry {
                source: format!("n{s}"),
                target: format!("n{t}"),
                kind: k.to_string(),
            })
            .collect();
        let input = AnalyzerInput {
            nodes,
            edges: edge_entries,
            edge_kinds: None,
            layer_weights: None,
            semantic_embeddings: None,
            projection: None,
            k: None,
            scope: None,
            parsed_nodes: None,
            parsed_edges: None,
            self_edge_ratio: None,
            packages: None,
            experimental: None,
        };
        Graph::from_input(&input)
    }

    /// Test 1: Cycle graph (n=6).
    ///
    /// Cycle 0→1→2→3→4→5→0. After symmetrization each node has degree 2.
    /// P^1(v,v) = 0 (no self-loops).
    /// P^2(v,v) = 1/2 (two neighbors, each returns with probability 1/2 × 1 path,
    /// but actually: from v go to one of 2 neighbors, from that neighbor probability
    /// 1/2 of returning. So P^2(v,v) = 2 × (1/2 × 1/2) = 1/2).
    #[test]
    fn test_cycle_graph() {
        let edges: Vec<(usize, usize, &str)> = (0..6)
            .map(|i| (i, (i + 1) % 6, "calls"))
            .collect();
        let graph = make_test_graph(6, edges);
        let rwpe = compute_rwpe(&graph, 4, 64);

        assert_eq!(rwpe.len(), 6);
        for v in 0..6 {
            assert_eq!(rwpe[v].len(), 4);
            // P^1(v,v) = 0 (no self-loops)
            assert!(
                (rwpe[v][0] - 0.0).abs() < 1e-10,
                "P^1({v},{v}) should be 0, got {}",
                rwpe[v][0]
            );
            // P^2(v,v) = 1/2
            assert!(
                (rwpe[v][1] - 0.5).abs() < 1e-10,
                "P^2({v},{v}) should be 0.5, got {}",
                rwpe[v][1]
            );
        }
        // By symmetry all nodes should have the same RWPE.
        for v in 1..6 {
            for step in 0..4 {
                assert!(
                    (rwpe[v][step] - rwpe[0][step]).abs() < 1e-10,
                    "Cycle symmetry broken at node {v} step {step}"
                );
            }
        }
    }

    /// Test 2: Star graph (n=5).
    ///
    /// Hub=0 connected to leaves 1,2,3,4.
    /// Hub degree=4, leaf degree=1.
    /// P^2(hub,hub) = 1.0 (from hub, go to any leaf with prob 1/4, leaf returns
    /// to hub with prob 1. Sum = 4 × 1/4 × 1 = 1.0).
    /// P^2(leaf,leaf) = 1/4 (from leaf, must go to hub prob 1, then hub goes to
    /// this leaf with prob 1/4).
    #[test]
    fn test_star_graph() {
        let edges: Vec<(usize, usize, &str)> = (1..5)
            .map(|i| (0, i, "calls"))
            .collect();
        let graph = make_test_graph(5, edges);
        let rwpe = compute_rwpe(&graph, 4, 64);

        assert_eq!(rwpe.len(), 5);

        // Hub: P^1 = 0, P^2 = 1.0
        assert!(
            (rwpe[0][0] - 0.0).abs() < 1e-10,
            "Hub P^1 should be 0, got {}",
            rwpe[0][0]
        );
        assert!(
            (rwpe[0][1] - 1.0).abs() < 1e-10,
            "Hub P^2 should be 1.0, got {}",
            rwpe[0][1]
        );

        // Leaves: P^1 = 0, P^2 = 1/4
        for leaf in 1..5 {
            assert!(
                (rwpe[leaf][0] - 0.0).abs() < 1e-10,
                "Leaf {leaf} P^1 should be 0, got {}",
                rwpe[leaf][0]
            );
            assert!(
                (rwpe[leaf][1] - 0.25).abs() < 1e-10,
                "Leaf {leaf} P^2 should be 0.25, got {}",
                rwpe[leaf][1]
            );
        }

        // All leaves should have the same RWPE.
        for leaf in 2..5 {
            for step in 0..4 {
                assert!(
                    (rwpe[leaf][step] - rwpe[1][step]).abs() < 1e-10,
                    "Leaf symmetry broken at leaf {leaf} step {step}"
                );
            }
        }
    }

    /// Test 3: Disconnected graph — two separate triangles.
    ///
    /// Triangle A: {0,1,2} with edges 0→1, 1→2, 2→0.
    /// Triangle B: {3,4,5} with edges 3→4, 4→5, 5→3.
    /// Nodes within same triangle should have identical RWPE.
    /// Triangles are independent.
    #[test]
    fn test_disconnected_graph() {
        let edges = vec![
            (0, 1, "calls"),
            (1, 2, "calls"),
            (2, 0, "calls"),
            (3, 4, "imports"),
            (4, 5, "imports"),
            (5, 3, "imports"),
        ];
        let graph = make_test_graph(6, edges);
        let rwpe = compute_rwpe(&graph, 4, 64);

        // All nodes within each triangle should be identical (symmetric).
        for v in 1..3 {
            for step in 0..4 {
                assert!(
                    (rwpe[v][step] - rwpe[0][step]).abs() < 1e-10,
                    "Triangle A symmetry broken at node {v} step {step}"
                );
            }
        }
        for v in 4..6 {
            for step in 0..4 {
                assert!(
                    (rwpe[v][step] - rwpe[3][step]).abs() < 1e-10,
                    "Triangle B symmetry broken at node {v} step {step}"
                );
            }
        }

        // Both triangles have the same structure, so RWPE should match across them.
        for step in 0..4 {
            assert!(
                (rwpe[0][step] - rwpe[3][step]).abs() < 1e-10,
                "Cross-triangle mismatch at step {step}"
            );
        }
    }

    /// Test 4: DAG symmetrization — chain 0→1→2→3.
    ///
    /// After symmetrization this becomes a path graph. Internal nodes (1,2)
    /// have degree 2, endpoints (0,3) have degree 1.
    /// All nodes should get nonzero RWPE for even steps (step index 1 = P^2).
    #[test]
    fn test_dag_symmetrization() {
        let edges = vec![
            (0, 1, "calls"),
            (1, 2, "calls"),
            (2, 3, "calls"),
        ];
        let graph = make_test_graph(4, edges);
        let rwpe = compute_rwpe(&graph, 4, 64);

        assert_eq!(rwpe.len(), 4);

        // P^1(v,v) = 0 for all nodes (no self-loops).
        for v in 0..4 {
            assert!(
                (rwpe[v][0] - 0.0).abs() < 1e-10,
                "P^1({v},{v}) should be 0"
            );
        }

        // P^2(v,v) should be nonzero for all nodes:
        // Endpoint (degree 1): P^2 = 1.0 (must go to neighbor, neighbor has 1/2
        //   chance of returning for internal, but endpoint's only neighbor is internal
        //   with degree 2, so returns with prob 1/2. P^2 = 1 × 1/2 = 1/2).
        //   Wait — endpoint degree is 1, its neighbor is degree 2.
        //   P^2(0,0) = P(0,1)*P(1,0) = 1 × 1/2 = 1/2.
        assert!(
            (rwpe[0][1] - 0.5).abs() < 1e-10,
            "Endpoint 0 P^2 should be 0.5, got {}",
            rwpe[0][1]
        );
        assert!(
            (rwpe[3][1] - 0.5).abs() < 1e-10,
            "Endpoint 3 P^2 should be 0.5, got {}",
            rwpe[3][1]
        );

        // Internal node (degree 2): P^2(1,1) = P(1,0)*P(0,1) + P(1,2)*P(2,1)
        //   = (1/2 × 1) + (1/2 × 1/2) = 1/2 + 1/4 = 3/4? No...
        //   P(1,0) = 1/2, P(0,1) = 1 (0 has degree 1, only neighbor is 1).
        //   P(1,2) = 1/2, P(2,1) = 1/2 (2 has degree 2).
        //   P^2(1,1) = 1/2 × 1 + 1/2 × 1/2 = 1/2 + 1/4 = 3/4.
        assert!(
            (rwpe[1][1] - 0.75).abs() < 1e-10,
            "Internal node 1 P^2 should be 0.75, got {}",
            rwpe[1][1]
        );

        // By symmetry, node 2 should mirror node 1.
        assert!(
            (rwpe[2][1] - 0.75).abs() < 1e-10,
            "Internal node 2 P^2 should be 0.75, got {}",
            rwpe[2][1]
        );
    }

    /// Test 5: Property assertions — all values in [0,1], P^1(v,v) = 0 when no self-loops.
    #[test]
    fn test_properties() {
        // Build a graph with various edge types.
        let edges = vec![
            (0, 1, "calls"),
            (1, 2, "imports"),
            (2, 3, "inherits"),
            (3, 0, "calls"),
            (0, 2, "imports"),
        ];
        let graph = make_test_graph(4, edges);
        let rwpe = compute_rwpe(&graph, 8, 2);

        for v in 0..4 {
            assert_eq!(rwpe[v].len(), 8);
            for (step, &val) in rwpe[v].iter().enumerate() {
                assert!(
                    val >= 0.0 && val <= 1.0,
                    "RWPE value out of [0,1] at node {v} step {step}: {val}"
                );
            }
            // P^1(v,v) = 0 (no self-loops in the test graph).
            assert!(
                (rwpe[v][0] - 0.0).abs() < 1e-10,
                "P^1({v},{v}) should be 0"
            );
        }
    }

    /// Test 6: Empty/isolated node — graph with 3 nodes, only edge 0→1.
    /// Node 2 should get all-zero RWPE.
    #[test]
    fn test_isolated_node() {
        let edges = vec![(0, 1, "calls")];
        let graph = make_test_graph(3, edges);
        let rwpe = compute_rwpe(&graph, 4, 64);

        assert_eq!(rwpe.len(), 3);

        // Node 2 is isolated: all-zero RWPE.
        for step in 0..4 {
            assert!(
                (rwpe[2][step] - 0.0).abs() < 1e-10,
                "Isolated node 2 should have zero RWPE at step {step}"
            );
        }

        // Nodes 0 and 1 should have nonzero P^2 values.
        // After symmetrization: 0↔1, each degree 1.
        // P^1(0,0) = 0 (must go to 1). P^2(0,0) = P(0,1)*P(1,0) = 1*1 = 1.0.
        assert!(
            (rwpe[0][0] - 0.0).abs() < 1e-10,
            "Node 0 P^1 should be 0"
        );
        assert!(
            (rwpe[0][1] - 1.0).abs() < 1e-10,
            "Node 0 P^2 should be 1.0, got {}",
            rwpe[0][1]
        );
        assert!(
            (rwpe[1][0] - 0.0).abs() < 1e-10,
            "Node 1 P^1 should be 0"
        );
        assert!(
            (rwpe[1][1] - 1.0).abs() < 1e-10,
            "Node 1 P^2 should be 1.0, got {}",
            rwpe[1][1]
        );
    }

    /// Test: "defines" edges are excluded from RWPE computation.
    #[test]
    fn test_defines_excluded() {
        // Only "defines" edges — should produce all-zero RWPE.
        let edges = vec![
            (0, 1, "defines"),
            (0, 2, "defines"),
        ];
        let graph = make_test_graph(3, edges);
        let rwpe = compute_rwpe(&graph, 4, 64);

        for v in 0..3 {
            for step in 0..4 {
                assert!(
                    (rwpe[v][step] - 0.0).abs() < 1e-10,
                    "Defines-only graph should produce all-zero RWPE"
                );
            }
        }
    }

    /// Test: empty graph (n=0).
    #[test]
    fn test_empty_graph() {
        let graph = make_test_graph(0, vec![]);
        let rwpe = compute_rwpe(&graph, 4, 64);
        assert!(rwpe.is_empty());
    }

    /// Test: k=0 produces empty inner vectors.
    #[test]
    fn test_k_zero() {
        let edges = vec![(0, 1, "calls")];
        let graph = make_test_graph(2, edges);
        let rwpe = compute_rwpe(&graph, 0, 64);
        assert_eq!(rwpe.len(), 2);
        assert!(rwpe[0].is_empty());
        assert!(rwpe[1].is_empty());
    }

    /// Test: batch_size=1 produces same results as large batch.
    #[test]
    fn test_batch_size_one() {
        let edges: Vec<(usize, usize, &str)> = (0..6)
            .map(|i| (i, (i + 1) % 6, "calls"))
            .collect();
        let graph = make_test_graph(6, edges);

        let rwpe_large = compute_rwpe(&graph, 6, 64);
        let rwpe_one = compute_rwpe(&graph, 6, 1);

        for v in 0..6 {
            for step in 0..6 {
                assert!(
                    (rwpe_large[v][step] - rwpe_one[v][step]).abs() < 1e-10,
                    "Batch size mismatch at node {v} step {step}: {} vs {}",
                    rwpe_large[v][step],
                    rwpe_one[v][step]
                );
            }
        }
    }
}
