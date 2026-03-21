//! Shared statistical utilities for data-adaptive thresholds.
//!
//! All threshold decisions in the analyzer derive from the data being
//! analyzed, not hardcoded constants. This module provides the building
//! blocks: distribution summaries, outlier fences, and percentile ranks.

use std::cmp::Ordering;

// ── Sorting helper ─────────────────────────────────────────────────────

fn f64_cmp(a: &f64, b: &f64) -> Ordering {
    a.partial_cmp(b).unwrap_or(Ordering::Equal)
}

fn sorted(values: &[f64]) -> Vec<f64> {
    let mut s = values.to_vec();
    s.sort_by(f64_cmp);
    s
}

// ── Central tendency ───────────────────────────────────────────────────

pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

pub fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let s = sorted(values);
    let mid = s.len() / 2;
    if s.len() % 2 == 0 {
        (s[mid - 1] + s[mid]) / 2.0
    } else {
        s[mid]
    }
}

// ── Spread ─────────────────────────────────────────────────────────────

pub fn std_dev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let m = mean(values);
    let variance = values.iter().map(|&v| (v - m) * (v - m)).sum::<f64>() / values.len() as f64;
    variance.sqrt()
}

/// Returns (Q1, median, Q3).
pub fn quartiles(values: &[f64]) -> (f64, f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let s = sorted(values);
    let n = s.len();
    let med = if n % 2 == 0 {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    } else {
        s[n / 2]
    };
    let q1 = percentile_sorted(&s, 0.25);
    let q3 = percentile_sorted(&s, 0.75);
    (q1, med, q3)
}

/// Interquartile range: Q3 - Q1.
pub fn iqr(values: &[f64]) -> f64 {
    let (q1, _, q3) = quartiles(values);
    q3 - q1
}

// ── Outlier detection ──────────────────────────────────────────────────

/// Tukey's upper fence: Q3 + 1.5 * IQR.
///
/// The standard outlier detection method (Tukey 1977). The 1.5 factor
/// is a statistical convention, not a magic number — it captures ~99.3%
/// of normally distributed data and works robustly on any distribution.
pub fn tukey_upper_fence(values: &[f64]) -> f64 {
    let (_, _, q3) = quartiles(values);
    let iqr_val = iqr(values);
    q3 + 1.5 * iqr_val
}

/// Tukey's lower fence: Q1 - 1.5 * IQR.
pub fn tukey_lower_fence(values: &[f64]) -> f64 {
    let (q1, _, _) = quartiles(values);
    let iqr_val = iqr(values);
    q1 - 1.5 * iqr_val
}

// ── Ranking ────────────────────────────────────────────────────────────

/// Percentile ranks: fraction of values strictly less.
///
/// Uses ranks/(n-1) so maximum maps to 1.0, minimum to 0.0.
pub fn percentile_ranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    if n <= 1 {
        return vec![0.0; n];
    }
    let s = sorted(values);
    values
        .iter()
        .map(|&v| {
            let rank = s.partition_point(|&x| x < v);
            rank as f64 / (n - 1) as f64
        })
        .collect()
}

/// Z-score: (value - mean) / std_dev.
pub fn z_score(value: f64, mean: f64, std: f64) -> f64 {
    if std <= 0.0 {
        return 0.0;
    }
    (value - mean) / std
}

// ── Distance ───────────────────────────────────────────────────────────

pub fn euclidean_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

pub fn squared_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
}

// ── Rounding ───────────────────────────────────────────────────────────

pub fn round4(v: f64) -> f64 {
    (v * 10000.0).round() / 10000.0
}

// ── RNG ────────────────────────────────────────────────────────────────

/// Simple seeded RNG for deterministic results.
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        // xorshift64
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn next_usize(&mut self, bound: usize) -> usize {
        if bound == 0 {
            return 0;
        }
        self.next_u64() as usize % bound
    }

    pub fn choice_weighted(&mut self, weights: &[f64]) -> usize {
        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            return 0;
        }
        let mut r = self.next_f64() * total;
        for (i, &w) in weights.iter().enumerate() {
            r -= w;
            if r <= 0.0 {
                return i;
            }
        }
        weights.len() - 1
    }
}

// ── Information theory ────────────────────────────────────────────────

/// Normalized Mutual Information between two partitions over the same keys.
///
/// Returns a value in [0, 1]: 1.0 means the partitions are identical
/// (up to label permutation), 0.0 means they share no information.
pub fn compute_nmi(
    left: &std::collections::HashMap<String, usize>,
    right: &std::collections::HashMap<String, usize>,
) -> f64 {
    use std::collections::HashMap;

    let keys: Vec<&String> = left.keys().filter(|k| right.contains_key(*k)).collect();
    let n = keys.len();
    if n == 0 {
        return 1.0;
    }
    let nf = n as f64;

    // Joint and marginal counts.
    let mut joint: HashMap<(usize, usize), usize> = HashMap::new();
    let mut count_l: HashMap<usize, usize> = HashMap::new();
    let mut count_r: HashMap<usize, usize> = HashMap::new();
    for k in &keys {
        let l = left[*k];
        let r = right[*k];
        *joint.entry((l, r)).or_default() += 1;
        *count_l.entry(l).or_default() += 1;
        *count_r.entry(r).or_default() += 1;
    }

    // Mutual information.
    let mut mi = 0.0;
    for (&(l, r), &nij) in &joint {
        if nij == 0 {
            continue;
        }
        let pij = nij as f64 / nf;
        let pi = count_l[&l] as f64 / nf;
        let pj = count_r[&r] as f64 / nf;
        mi += pij * (pij / (pi * pj)).ln();
    }

    // Marginal entropies.
    let entropy = |counts: &HashMap<usize, usize>| -> f64 {
        counts
            .values()
            .filter(|&&c| c > 0)
            .map(|&c| {
                let p = c as f64 / nf;
                -p * p.ln()
            })
            .sum::<f64>()
    };
    let h_l = entropy(&count_l);
    let h_r = entropy(&count_r);

    if h_l + h_r == 0.0 {
        return 1.0;
    }
    (2.0 * mi / (h_l + h_r)).clamp(0.0, 1.0)
}

// ── Internal helpers ───────────────────────────────────────────────────

/// Interpolated percentile on a pre-sorted slice.
fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let idx = p * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    let frac = idx - lo as f64;
    if lo == hi || hi >= n {
        sorted[lo.min(n - 1)]
    } else {
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean() {
        assert_eq!(mean(&[1.0, 2.0, 3.0, 4.0, 5.0]), 3.0);
        assert_eq!(mean(&[]), 0.0);
    }

    #[test]
    fn test_median() {
        assert_eq!(median(&[1.0, 3.0, 5.0]), 3.0);
        assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]), 2.5);
    }

    #[test]
    fn test_std_dev() {
        let sd = std_dev(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        assert!((sd - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_quartiles() {
        let (q1, med, q3) = quartiles(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        assert_eq!(med, 4.0);
        assert!((q1 - 2.5).abs() < 0.01);
        assert!((q3 - 5.5).abs() < 0.01);
    }

    #[test]
    fn test_tukey_fences() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 100.0];
        let fence = tukey_upper_fence(&data);
        // 100 should be above the fence (it's an outlier)
        assert!(100.0 > fence);
        // 7 should be below (it's normal)
        assert!(7.0 < fence);
    }

    #[test]
    fn test_percentile_ranks() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let pcts = percentile_ranks(&values);
        assert_eq!(pcts[0], 0.0);
        assert_eq!(pcts[4], 1.0);
    }

    #[test]
    fn test_z_score() {
        assert_eq!(z_score(5.0, 3.0, 1.0), 2.0);
        assert_eq!(z_score(3.0, 3.0, 1.0), 0.0);
    }

    #[test]
    fn test_euclidean_dist() {
        assert_eq!(euclidean_dist(&[0.0, 0.0], &[3.0, 4.0]), 5.0);
    }

    #[test]
    fn test_rng_deterministic() {
        let mut r1 = Rng::new(42);
        let mut r2 = Rng::new(42);
        assert_eq!(r1.next_u64(), r2.next_u64());
        assert_eq!(r1.next_f64(), r2.next_f64());
    }

    #[test]
    fn test_nmi_identical_partitions() {
        let mut left = std::collections::HashMap::new();
        let mut right = std::collections::HashMap::new();
        for (i, name) in ["a", "b", "c", "d"].iter().enumerate() {
            left.insert(name.to_string(), i % 2);
            right.insert(name.to_string(), i % 2);
        }
        let nmi = compute_nmi(&left, &right);
        assert!((nmi - 1.0).abs() < 1e-9, "identical partitions => NMI=1.0, got {nmi}");
    }

    #[test]
    fn test_nmi_independent_partitions() {
        // left: {a:0, b:0, c:1, d:1}, right: {a:0, b:1, c:0, d:1}
        let left: std::collections::HashMap<String, usize> =
            [("a", 0), ("b", 0), ("c", 1), ("d", 1)]
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect();
        let right: std::collections::HashMap<String, usize> =
            [("a", 0), ("b", 1), ("c", 0), ("d", 1)]
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect();
        let nmi = compute_nmi(&left, &right);
        assert!(nmi < 0.01, "independent partitions => NMI~0, got {nmi}");
    }

    #[test]
    fn test_nmi_refinement() {
        // left: 2 clusters, right: 3 clusters (refinement of left)
        let left: std::collections::HashMap<String, usize> =
            [("a", 0), ("b", 0), ("c", 0), ("d", 1), ("e", 1), ("f", 1)]
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect();
        let right: std::collections::HashMap<String, usize> =
            [("a", 0), ("b", 0), ("c", 1), ("d", 2), ("e", 2), ("f", 2)]
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect();
        let nmi = compute_nmi(&left, &right);
        assert!(nmi > 0.4 && nmi < 1.0, "refinement => 0 < NMI < 1, got {nmi}");
    }
}
