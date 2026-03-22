//! Graph algorithms: SCC (Tarjan), betweenness (Brandes), connected components, BFS distance.

use std::collections::VecDeque;

/// BFS shortest path distance between two nodes, capped at `max_hops`.
/// Returns `None` if no path exists within the hop limit.
pub fn bfs_distance(successors: &[Vec<usize>], src: usize, tgt: usize, max_hops: usize) -> Option<usize> {
    if src == tgt {
        return Some(0);
    }
    let mut visited = vec![false; successors.len()];
    visited[src] = true;
    let mut queue = VecDeque::new();
    queue.push_back((src, 0usize));
    while let Some((node, depth)) = queue.pop_front() {
        if depth >= max_hops {
            continue;
        }
        for &next in &successors[node] {
            if next == tgt {
                return Some(depth + 1);
            }
            if !visited[next] {
                visited[next] = true;
                queue.push_back((next, depth + 1));
            }
        }
    }
    None
}

/// Tarjan's iterative SCC algorithm.
///
/// Returns a list of strongly connected components (each is a list of node indices).
pub fn tarjan_scc(successors: &[Vec<usize>], n: usize) -> Vec<Vec<usize>> {
    let mut index_counter = 0usize;
    let mut index = vec![usize::MAX; n];
    let mut lowlink = vec![0usize; n];
    let mut on_stack = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();
    let mut result: Vec<Vec<usize>> = Vec::new();

    for start in 0..n {
        if index[start] != usize::MAX {
            continue;
        }

        // Iterative DFS: work stack of (node, neighbor_index).
        let mut work: Vec<(usize, usize)> = Vec::new();
        index[start] = index_counter;
        lowlink[start] = index_counter;
        index_counter += 1;
        stack.push(start);
        on_stack[start] = true;
        work.push((start, 0));

        while let Some(&mut (v, ref mut ni)) = work.last_mut() {
            let neighbors = &successors[v];
            if *ni < neighbors.len() {
                let w = neighbors[*ni];
                *ni += 1;
                if index[w] == usize::MAX {
                    index[w] = index_counter;
                    lowlink[w] = index_counter;
                    index_counter += 1;
                    stack.push(w);
                    on_stack[w] = true;
                    work.push((w, 0));
                } else if on_stack[w] {
                    lowlink[v] = lowlink[v].min(index[w]);
                }
            } else {
                // All neighbors processed.
                if lowlink[v] == index[v] {
                    let mut component = Vec::new();
                    loop {
                        let w = stack.pop().unwrap();
                        on_stack[w] = false;
                        component.push(w);
                        if w == v {
                            break;
                        }
                    }
                    result.push(component);
                }
                let done = work.pop().unwrap();
                if let Some(&mut (parent, _)) = work.last_mut() {
                    lowlink[parent] = lowlink[parent].min(lowlink[done.0]);
                }
            }
        }
    }

    result
}

/// Brandes' betweenness centrality for directed graphs.
///
/// For large graphs (> threshold nodes), samples sqrt(n) source nodes.
pub fn brandes_betweenness(
    successors: &[Vec<usize>],
    n: usize,
    approx_threshold: usize,
) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }

    let mut cb = vec![0.0f64; n];

    // Choose sources: all for small graphs, sample for large.
    let sources: Vec<usize> = if n <= approx_threshold {
        (0..n).collect()
    } else {
        // Deterministic sampling: use stride-based selection.
        let k = ((n as f64).sqrt() as usize).max(2).min(n - 1);
        let stride = n / k;
        (0..k).map(|i| (i * stride) % n).collect()
    };

    let num_sources = sources.len();

    for &s in &sources {
        // BFS from s.
        let mut stack: Vec<usize> = Vec::new();
        let mut pred: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut sigma = vec![0u64; n];
        sigma[s] = 1;
        let mut dist = vec![-1i64; n];
        dist[s] = 0;
        let mut queue = VecDeque::new();
        queue.push_back(s);

        while let Some(v) = queue.pop_front() {
            stack.push(v);
            for &w in &successors[v] {
                if dist[w] < 0 {
                    dist[w] = dist[v] + 1;
                    queue.push_back(w);
                }
                if dist[w] == dist[v] + 1 {
                    sigma[w] += sigma[v];
                    pred[w].push(v);
                }
            }
        }

        // Back-propagation.
        let mut delta = vec![0.0f64; n];
        while let Some(w) = stack.pop() {
            for &v in &pred[w] {
                if sigma[w] > 0 {
                    delta[v] += (sigma[v] as f64 / sigma[w] as f64) * (1.0 + delta[w]);
                }
            }
            if w != s {
                cb[w] += delta[w];
            }
        }
    }

    // Normalize: for directed graphs, divide by (n-1)*(n-2).
    if n > 2 {
        let mut scale = 1.0 / ((n - 1) as f64 * (n - 2) as f64);
        if n > approx_threshold {
            scale *= n as f64 / num_sources as f64;
        }
        for v in &mut cb {
            *v *= scale;
        }
    }

    cb
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tarjan_scc_cycle() {
        // 0 → 1 → 2 → 0 (one SCC of size 3), 3 → 0 (3 is separate)
        let successors = vec![
            vec![1],    // 0
            vec![2],    // 1
            vec![0],    // 2
            vec![0],    // 3
        ];
        let sccs = tarjan_scc(&successors, 4);
        let large: Vec<_> = sccs.iter().filter(|c| c.len() > 1).collect();
        assert_eq!(large.len(), 1);
        assert_eq!(large[0].len(), 3);
    }

    #[test]
    fn test_brandes_star() {
        // Star graph: 0 is center, edges 0→1, 0→2, 0→3, 0→4
        let successors = vec![
            vec![1, 2, 3, 4], // 0
            vec![],            // 1
            vec![],            // 2
            vec![],            // 3
            vec![],            // 4
        ];
        let btw = brandes_betweenness(&successors, 5, 5000);
        // Center node should have highest betweenness (0 in this case since
        // there are no shortest paths *through* a node in a star).
        // Actually in a directed star with only outgoing edges, no paths go through 0.
        assert!(btw[1] == 0.0);
    }

    #[test]
    fn test_bfs_distance_adjacent() {
        let successors = vec![vec![1], vec![2], vec![], vec![]];
        assert_eq!(bfs_distance(&successors, 0, 1, 4), Some(1));
        assert_eq!(bfs_distance(&successors, 0, 2, 4), Some(2));
    }

    #[test]
    fn test_bfs_distance_self() {
        let successors = vec![vec![1], vec![]];
        assert_eq!(bfs_distance(&successors, 0, 0, 4), Some(0));
    }

    #[test]
    fn test_bfs_distance_no_path() {
        let successors = vec![vec![1], vec![], vec![3], vec![]];
        assert_eq!(bfs_distance(&successors, 0, 3, 4), None); // 0 and 3 disconnected
    }

    #[test]
    fn test_bfs_distance_exceeds_max_hops() {
        // Chain: 0->1->2->3->4->5
        let successors = vec![vec![1], vec![2], vec![3], vec![4], vec![5], vec![]];
        assert_eq!(bfs_distance(&successors, 0, 5, 4), None); // 5 hops > max 4
        assert_eq!(bfs_distance(&successors, 0, 4, 4), Some(4)); // exactly max_hops
        assert_eq!(bfs_distance(&successors, 0, 3, 4), Some(3)); // within limit
    }
}
