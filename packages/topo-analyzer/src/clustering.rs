//! K-means++ clustering and silhouette scoring.
//!
//! Ports the custom implementations from topo_analyzer.modules.

use crate::stats::{self, Rng};

/// K-means++ clustering result.
pub struct KMeansResult {
    /// Cluster assignment for each row (0-indexed).
    pub labels: Vec<usize>,
    /// Centroid vectors: k × dim.
    pub centroids: Vec<Vec<f64>>,
    /// Number of iterations run.
    pub iterations: usize,
}

/// K-means++ clustering on a row-major n×dim matrix.
///
/// - `data`: row-major, n rows of `dim` dimensions.
/// - `k`: number of clusters.
/// - `max_iter`: maximum iterations (default 100).
/// - `seed`: RNG seed for determinism.
pub fn kmeans(data: &[Vec<f64>], k: usize, max_iter: usize, seed: u64) -> KMeansResult {
    let n = data.len();
    if n == 0 || k == 0 {
        return KMeansResult {
            labels: vec![0; n],
            centroids: Vec::new(),
            iterations: 0,
        };
    }
    let dim = data[0].len();
    let k = k.min(n);
    let mut rng = Rng::new(seed);

    // K-means++ initialization.
    let mut centroids: Vec<Vec<f64>> = Vec::with_capacity(k);
    // First centroid: random point.
    let first = (rng.next_u64() as usize) % n;
    centroids.push(data[first].clone());

    for _ in 1..k {
        // Compute distance from each point to nearest existing centroid.
        let dists: Vec<f64> = data
            .iter()
            .map(|point| {
                centroids
                    .iter()
                    .map(|c| squared_distance(point, c))
                    .fold(f64::INFINITY, f64::min)
            })
            .collect();
        let next = rng.choice_weighted(&dists);
        centroids.push(data[next].clone());
    }

    // Main loop.
    let mut labels = vec![0usize; n];
    let mut iterations = 0;

    for iter in 0..max_iter {
        iterations = iter + 1;

        // Assignment step.
        let mut changed = false;
        for i in 0..n {
            let mut best_cluster = 0;
            let mut best_dist = f64::INFINITY;
            for (ci, centroid) in centroids.iter().enumerate() {
                let d = squared_distance(&data[i], centroid);
                if d < best_dist {
                    best_dist = d;
                    best_cluster = ci;
                }
            }
            if labels[i] != best_cluster {
                labels[i] = best_cluster;
                changed = true;
            }
        }

        if !changed {
            break;
        }

        // Update step.
        let mut new_centroids = vec![vec![0.0f64; dim]; k];
        let mut counts = vec![0usize; k];
        for i in 0..n {
            let ci = labels[i];
            counts[ci] += 1;
            for d in 0..dim {
                new_centroids[ci][d] += data[i][d];
            }
        }
        for ci in 0..k {
            if counts[ci] > 0 {
                for d in 0..dim {
                    new_centroids[ci][d] /= counts[ci] as f64;
                }
            }
        }
        centroids = new_centroids;
    }

    KMeansResult {
        labels,
        centroids,
        iterations,
    }
}

/// Multi-start k-means: runs k-means with multiple seeds and picks the
/// result with the best silhouette score, reducing local-optima risk.
pub fn kmeans_best_of(data: &[Vec<f64>], k: usize, max_iter: usize, seeds: &[u64]) -> KMeansResult {
    assert!(!seeds.is_empty(), "kmeans_best_of requires at least one seed");
    let mut best_result = kmeans(data, k, max_iter, seeds[0]);
    let mut best_sil = silhouette_score(data, &best_result.labels, &best_result.centroids);

    for &seed in &seeds[1..] {
        let result = kmeans(data, k, max_iter, seed);
        let sil = silhouette_score(data, &result.labels, &result.centroids);
        if sil > best_sil {
            best_sil = sil;
            best_result = result;
        }
    }

    best_result
}

/// Centroid-approximated silhouette score.
///
/// O(n*k) instead of O(n²) — uses distance to own centroid vs nearest
/// other centroid as a proxy for full pairwise silhouette.
pub fn silhouette_score(data: &[Vec<f64>], labels: &[usize], centroids: &[Vec<f64>]) -> f64 {
    let n = data.len();
    let k = centroids.len();
    if n < 2 || k < 2 {
        return 0.0;
    }

    let mut total = 0.0;
    let mut count = 0;

    for i in 0..n {
        let own_cluster = labels[i];
        let a = squared_distance(&data[i], &centroids[own_cluster]).sqrt();

        let mut best_other = f64::INFINITY;
        for (ci, centroid) in centroids.iter().enumerate() {
            if ci == own_cluster {
                continue;
            }
            let d = squared_distance(&data[i], centroid).sqrt();
            if d < best_other {
                best_other = d;
            }
        }

        if best_other == f64::INFINITY {
            continue;
        }

        let max_ab = a.max(best_other);
        if max_ab > 0.0 {
            total += (best_other - a) / max_ab;
            count += 1;
        }
    }

    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

/// Estimate optimal k by sweeping k=2..max_k and picking best silhouette.
pub fn estimate_k(data: &[Vec<f64>], max_k: usize, seed: u64) -> usize {
    let n = data.len();
    let max_k = max_k.min(n.saturating_sub(1)).max(2);

    let mut best_k = 2;
    let mut best_score = f64::NEG_INFINITY;

    for k in 2..=max_k {
        let result = kmeans(data, k, 100, seed);
        let score = silhouette_score(data, &result.labels, &result.centroids);
        if score > best_score {
            best_score = score;
            best_k = k;
        }
    }

    best_k
}

/// Compute mean silhouette of random k-partitions as a baseline.
///
/// Used for data-adaptive degeneracy detection: if spectral clustering
/// isn't meaningfully better than random partitioning, it's degenerate.
pub fn random_baseline_silhouette(
    data: &[Vec<f64>],
    k: usize,
    n_permutations: usize,
    seed: u64,
) -> f64 {
    let n = data.len();
    if n == 0 || k == 0 || data[0].is_empty() {
        return 0.0;
    }
    let dim = data[0].len();
    let mut rng = Rng::new(seed);
    let mut total = 0.0;

    for _ in 0..n_permutations {
        let labels: Vec<usize> = (0..n).map(|_| rng.next_usize(k)).collect();

        // Compute centroids for random partition.
        let mut centroids = vec![vec![0.0; dim]; k];
        let mut counts = vec![0usize; k];
        for (i, &label) in labels.iter().enumerate() {
            counts[label] += 1;
            for (j, &v) in data[i].iter().enumerate() {
                centroids[label][j] += v;
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                for v in &mut centroids[c] {
                    *v /= counts[c] as f64;
                }
            }
        }
        total += silhouette_score(data, &labels, &centroids);
    }

    total / n_permutations as f64
}

/// Prepare spectral eigenvectors for k-means clustering.
///
/// Two standard steps from Ng-Jordan-Weiss (2001):
/// 1. Truncate each row to the first `k` eigenvector dimensions.
///    Higher eigenvectors capture noise, not community structure.
/// 2. Row-normalize to unit length. The normalized Laplacian places
///    same-community nodes at similar angles; without normalization,
///    high-degree hubs sit near the origin and attract everything
///    into one giant cluster.
pub fn prepare_for_clustering(data: &[Vec<f64>], k: usize) -> Vec<Vec<f64>> {
    data.iter()
        .map(|row| {
            let truncated: Vec<f64> = row.iter().take(k).cloned().collect();
            let norm: f64 = truncated.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm > 1e-10 {
                truncated.iter().map(|v| v / norm).collect()
            } else {
                truncated
            }
        })
        .collect()
}

fn squared_distance(a: &[f64], b: &[f64]) -> f64 {
    stats::squared_distance(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans_two_clusters() {
        let data: Vec<Vec<f64>> = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![0.2, 0.0],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
            vec![10.2, 10.0],
        ];
        let result = kmeans(&data, 2, 100, 42);
        // Points 0-2 should be in one cluster, 3-5 in another.
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[1], result.labels[2]);
        assert_eq!(result.labels[3], result.labels[4]);
        assert_eq!(result.labels[4], result.labels[5]);
        assert_ne!(result.labels[0], result.labels[3]);
    }

    #[test]
    fn test_silhouette_perfect() {
        let data: Vec<Vec<f64>> = vec![
            vec![0.0],
            vec![0.1],
            vec![10.0],
            vec![10.1],
        ];
        let labels = vec![0, 0, 1, 1];
        let centroids = vec![vec![0.05], vec![10.05]];
        let score = silhouette_score(&data, &labels, &centroids);
        assert!(score > 0.9, "expected high silhouette for well-separated clusters: {score}");
    }
}
