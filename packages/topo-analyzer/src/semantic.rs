//! Semantic analysis tools for Phase 2: hybrid structural-semantic analysis.
//!
//! This module contains:
//! - `top_terms`: TF-IDF based module labeling from node ID tokenization
//! - Semantic coherence, signal quality gate, Rayleigh quotient, GFT energy,
//!   local variation, AMI, and issue detection (added incrementally).

use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// top_terms: TF-IDF on node IDs
// ---------------------------------------------------------------------------

/// Compute top-N TF-IDF terms for each module from node ID tokenization.
///
/// No embedding model needed — just string tokenization.
/// Tokenizes the leaf segment of each node ID (after stripping module prefix),
/// splits camelCase/snake_case, lowercases, and computes TF-IDF across modules.
pub fn top_terms(
    clusters: &HashMap<String, usize>,
    n: usize,
) -> HashMap<usize, Vec<String>> {
    if clusters.is_empty() || n == 0 {
        return HashMap::new();
    }

    // Group node IDs by module.
    let mut module_nodes: HashMap<usize, Vec<&str>> = HashMap::new();
    for (node_id, &cluster_id) in clusters {
        module_nodes.entry(cluster_id).or_default().push(node_id.as_str());
    }

    let num_modules = module_nodes.len();
    if num_modules == 0 {
        return HashMap::new();
    }

    // Tokenize all node IDs and compute TF per module, DF across modules.
    let mut module_tf: HashMap<usize, HashMap<String, f64>> = HashMap::new();
    let mut df: HashMap<String, usize> = HashMap::new();

    for (&module_id, node_ids) in &module_nodes {
        let mut term_count: HashMap<String, usize> = HashMap::new();
        let mut module_terms_unique: std::collections::HashSet<String> =
            std::collections::HashSet::new();

        for node_id in node_ids {
            let tokens = tokenize_node_id(node_id);
            for token in &tokens {
                *term_count.entry(token.clone()).or_insert(0) += 1;
                module_terms_unique.insert(token.clone());
            }
        }

        // TF: raw count normalized by total tokens in this module.
        let total: f64 = term_count.values().sum::<usize>() as f64;
        if total > 0.0 {
            let tf: HashMap<String, f64> = term_count
                .into_iter()
                .map(|(term, count)| (term, count as f64 / total))
                .collect();
            module_tf.insert(module_id, tf);
        }

        // DF: how many modules contain each term.
        for term in module_terms_unique {
            *df.entry(term).or_insert(0) += 1;
        }
    }

    // Compute TF-IDF and pick top-N per module.
    let mut result: HashMap<usize, Vec<String>> = HashMap::new();
    let idf_denom = num_modules as f64;

    for (&module_id, tf) in &module_tf {
        let mut scored: Vec<(String, f64)> = tf
            .iter()
            .filter_map(|(term, &tf_val)| {
                let doc_freq = *df.get(term).unwrap_or(&1) as f64;
                // IDF: log(N / df). Terms appearing in all modules get low IDF.
                let idf = (idf_denom / doc_freq).ln();
                // Skip terms that appear in all modules (IDF ≈ 0).
                if idf < 0.01 {
                    return None;
                }
                Some((term.clone(), tf_val * idf))
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top: Vec<String> = scored.into_iter().take(n).map(|(term, _)| term).collect();
        result.insert(module_id, top);
    }

    result
}

/// Tokenize a node ID into lowercase terms.
///
/// Steps:
/// 1. Take the leaf segment (last dotted component)
/// 2. Split camelCase boundaries
/// 3. Split on underscores
/// 4. Lowercase everything
/// 5. Filter out single-char tokens and common stopwords
fn tokenize_node_id(node_id: &str) -> Vec<String> {
    // Take leaf segment (after last dot).
    let leaf = node_id.rsplit('.').next().unwrap_or(node_id);

    let mut tokens = Vec::new();

    // Split on underscores first, then split camelCase within each part.
    for part in leaf.split('_') {
        if part.is_empty() {
            continue;
        }
        tokens.extend(split_camel_case(part));
    }

    // Lowercase and filter.
    tokens
        .into_iter()
        .map(|t| t.to_lowercase())
        .filter(|t| t.len() > 1 && !is_stopword(t))
        .collect()
}

/// Split a camelCase string into words.
/// "getUserName" -> ["get", "User", "Name"]
/// "HTMLParser" -> ["HTML", "Parser"]
fn split_camel_case(s: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = s.chars().collect();

    for i in 0..chars.len() {
        let c = chars[i];

        if i > 0 && c.is_uppercase() {
            // Transition: lowercase -> uppercase (new word)
            // Or: uppercase -> uppercase followed by lowercase (acronym end)
            let prev_lower = chars[i - 1].is_lowercase();
            let next_lower = i + 1 < chars.len() && chars[i + 1].is_lowercase();

            if prev_lower || (chars[i - 1].is_uppercase() && next_lower) {
                if !current.is_empty() {
                    words.push(current);
                    current = String::new();
                }
            }
        }
        current.push(c);
    }
    if !current.is_empty() {
        words.push(current);
    }

    words
}

/// Common programming stopwords that don't carry domain meaning.
fn is_stopword(term: &str) -> bool {
    matches!(
        term,
        "get" | "set" | "new" | "init" | "from" | "into" | "with"
            | "self" | "this" | "impl" | "fn" | "pub" | "def" | "class"
            | "mod" | "use" | "let" | "mut" | "ref" | "the" | "is" | "to"
            | "of" | "in" | "for" | "on" | "at" | "by" | "if" | "or"
            | "and" | "not" | "an" | "do" | "has" | "was" | "test"
            | "async" | "await" | "return" | "true" | "false" | "none"
            | "null" | "void" | "type" | "str" | "int" | "bool" | "float"
    )
}

// ---------------------------------------------------------------------------
// Semantic embeddings: math tools (Steps 4-8)
// ---------------------------------------------------------------------------

use crate::graph::Graph;
use crate::stats::Rng;

/// Result of the semantic analysis pipeline.
pub struct SemanticAnalysis {
    /// Per-module semantic coherence (avg pairwise cosine similarity).
    pub module_coherence: HashMap<usize, f64>,
    /// Rayleigh quotient: global structural-semantic smoothness.
    pub smoothness: f64,
    /// GFT energy profile: semantic energy at each eigenvalue.
    pub energy_eigenvalues: Vec<f64>,
    pub energy_values: Vec<f64>,
    /// Per-node local variation (semantic disagreement with neighbors).
    pub local_variation: HashMap<String, f64>,
    /// AMI between structural and semantic partitions.
    pub ami: f64,
    /// Misplaced concern detections.
    pub misplaced_concerns: Vec<MisplacedConcern>,
    /// Incoherent module detections.
    pub incoherent_modules: Vec<IncoherentModule>,
    /// Shadow dependency detections (experimental, O(n²)).
    pub shadow_dependencies: Vec<ShadowDependency>,
    /// Redundant API detections.
    pub redundant_apis: Vec<RedundantApi>,
    /// Whether the quality gate passed.
    pub gate_passed: bool,
    /// Null coherence mean from permutation test (for cross-package-coupling).
    pub null_coherence_mean: f64,
    /// Null coherence std from permutation test (for cross-package-coupling).
    pub null_coherence_std: f64,
}

/// A node whose semantics better match a different module.
pub struct MisplacedConcern {
    pub node_id: String,
    pub own_module: usize,
    pub best_module: usize,
    pub similarity_own: f64,
    pub similarity_best: f64,
}

/// A module whose members are semantically unrelated.
pub struct IncoherentModule {
    pub module_id: usize,
    pub coherence: f64,
    pub null_threshold: f64,
    /// Semantic sub-clusters found within the module (top terms per sub-cluster).
    pub sub_clusters: Vec<Vec<String>>,
    /// Number of members in the module.
    pub module_size: usize,
    /// Best k from sub-cluster sweep (2..min(|M|/3, 6)).
    pub best_k: usize,
}

/// Two semantically similar nodes in different modules with no structural link.
pub struct ShadowDependency {
    pub node_a: String,
    pub node_b: String,
    pub similarity: f64,
    pub structural_distance: Option<usize>,
}

/// A module whose public API has redundant entry points.
pub struct RedundantApi {
    pub module_id: usize,
    pub entry_points: Vec<String>,
    pub mean_similarity: f64,
    pub mean_callee_overlap: f64,
}

/// Run the full semantic analysis pipeline.
///
/// Requires: embeddings (node_id -> f32 vector), graph, clusters, eigenvectors.
pub fn analyze_semantic(
    embeddings: &HashMap<String, Vec<f32>>,
    graph: &Graph,
    clusters: &HashMap<String, usize>,
    eigenvalues: &[f64],
    eigenvectors: &[Vec<f64>],
    roles: &HashMap<String, String>,
    component_node_ids: &[String],
    experimental: bool,
) -> SemanticAnalysis {
    // Convert f32 embeddings to f64 for numerical stability.
    let emb64: HashMap<&str, Vec<f64>> = embeddings
        .iter()
        .map(|(k, v)| (k.as_str(), v.iter().map(|&x| x as f64).collect()))
        .collect();

    // 1. Signal quality gate.
    let gate_passed = signal_quality_gate(&emb64, clusters);
    if !gate_passed {
        return SemanticAnalysis {
            module_coherence: HashMap::new(),
            smoothness: 0.0,
            energy_eigenvalues: Vec::new(),
            energy_values: Vec::new(),
            local_variation: HashMap::new(),
            ami: 0.0,
            misplaced_concerns: Vec::new(),
            incoherent_modules: Vec::new(),
            shadow_dependencies: Vec::new(),
            redundant_apis: Vec::new(),
            gate_passed: false,
            null_coherence_mean: 0.0,
            null_coherence_std: 0.0,
        };
    }

    // 2. Per-module semantic coherence.
    let module_coherence = compute_module_coherence(&emb64, clusters);

    // 3. Rayleigh quotient (global smoothness).
    let smoothness = rayleigh_quotient(&emb64, graph);

    // 4. GFT energy profile.
    let (energy_eigenvalues, energy_values) =
        gft_energy_profile(&emb64, graph, eigenvalues, eigenvectors, component_node_ids);

    // Build symmetric neighbor list once for local variation + misplaced concern detection.
    let sym_nbrs = symmetric_neighbors(graph);

    // 5. Local variation per node.
    let local_variation = compute_local_variation(&emb64, graph, &sym_nbrs);

    // 6. AMI between structural and semantic partitions.
    let ami = compute_ami(&emb64, clusters);

    // Compute null coherence threshold once (used by both detectors and cross-package-coupling).
    let (null_threshold, null_std) = compute_null_coherence_threshold(&emb64, clusters);

    // 7. Misplaced concern detection.
    let misplaced_concerns =
        detect_misplaced_concerns(&emb64, clusters, &module_coherence, roles, graph, &sym_nbrs, null_threshold);

    // 8. Incoherent module detection.
    let incoherent_modules = detect_incoherent_modules(&emb64, clusters, &module_coherence, null_threshold);

    // 9. Shadow dependency detection (experimental, O(n²)).
    let shadow_dependencies = if experimental {
        detect_shadow_dependencies(&emb64, graph, clusters)
    } else {
        Vec::new()
    };

    // 10. Redundant API detection.
    let redundant_apis = detect_redundant_api(&emb64, graph, clusters, roles);

    SemanticAnalysis {
        module_coherence,
        smoothness,
        energy_eigenvalues,
        energy_values,
        local_variation,
        ami,
        misplaced_concerns,
        incoherent_modules,
        shadow_dependencies,
        redundant_apis,
        gate_passed: true,
        null_coherence_mean: null_threshold,
        null_coherence_std: null_std,
    }
}

// ---------------------------------------------------------------------------
// Signal quality gate (Step 4)
// ---------------------------------------------------------------------------

/// Check if semantic embeddings carry meaningful signal.
///
/// Quick pre-checks + permutation test (N=200, α=0.05).
fn signal_quality_gate(
    emb: &HashMap<&str, Vec<f64>>,
    clusters: &HashMap<String, usize>,
) -> bool {
    if emb.len() < 6 {
        return false;
    }

    // Quick pre-checks.
    let (mean_sim, var_sim) = pairwise_cosine_stats(emb);
    if var_sim < 0.01 {
        return false; // No discriminative power.
    }
    if mean_sim > 0.95 {
        return false; // Everything looks the same.
    }

    // Permutation test: is within-module similarity significantly higher?
    let observed = within_vs_across_similarity(emb, clusters);

    let mut rng = Rng::new(42);
    let node_ids: Vec<&String> = clusters.keys().collect();
    let module_ids: Vec<usize> = clusters.values().cloned().collect();
    let n_perms = 200;
    let mut exceed_count = 0;

    for _ in 0..n_perms {
        // Shuffle module assignments.
        let mut shuffled: HashMap<String, usize> = HashMap::new();
        let mut perm_modules = module_ids.clone();
        // Fisher-Yates shuffle.
        for i in (1..perm_modules.len()).rev() {
            let j = rng.next_usize(i + 1);
            perm_modules.swap(i, j);
        }
        for (i, &nid) in node_ids.iter().enumerate() {
            shuffled.insert(nid.clone(), perm_modules[i]);
        }
        let perm_stat = within_vs_across_similarity(emb, &shuffled);
        if perm_stat >= observed {
            exceed_count += 1;
        }
    }

    // Phipson & Smyth (2010) correction: (exceed + 1) / (N + 1)
    // to avoid p=0 and ensure valid test.
    let p_value = (exceed_count + 1) as f64 / (n_perms + 1) as f64;
    p_value < 0.05
}

/// Compute mean and variance of pairwise cosine similarities (sampled).
fn pairwise_cosine_stats(emb: &HashMap<&str, Vec<f64>>) -> (f64, f64) {
    let vecs: Vec<&Vec<f64>> = emb.values().collect();
    let n = vecs.len();
    if n < 2 {
        return (0.0, 0.0);
    }

    // Sample up to 5000 pairs for efficiency.
    let max_pairs = 5000;
    let total_pairs = n * (n - 1) / 2;
    let mut sims = Vec::with_capacity(max_pairs.min(total_pairs));

    if total_pairs <= max_pairs {
        for i in 0..n {
            for j in (i + 1)..n {
                sims.push(cosine_similarity(vecs[i], vecs[j]));
            }
        }
    } else {
        let mut rng = Rng::new(123);
        for _ in 0..max_pairs {
            let i = rng.next_usize(n);
            let mut j = rng.next_usize(n - 1);
            if j >= i {
                j += 1;
            }
            sims.push(cosine_similarity(vecs[i], vecs[j]));
        }
    }

    let mean = sims.iter().sum::<f64>() / sims.len() as f64;
    let var = sims.iter().map(|s| (s - mean) * (s - mean)).sum::<f64>() / sims.len() as f64;
    (mean, var)
}

/// Test statistic: mean within-module similarity minus mean across-module similarity.
fn within_vs_across_similarity(
    emb: &HashMap<&str, Vec<f64>>,
    clusters: &HashMap<String, usize>,
) -> f64 {
    let mut within_sum = 0.0;
    let mut within_count = 0usize;
    let mut across_sum = 0.0;
    let mut across_count = 0usize;

    let nodes: Vec<(&String, &usize)> = clusters.iter().collect();
    // Sample for efficiency in large codebases.
    let max_pairs = 5000;
    let n = nodes.len();

    if n * (n - 1) / 2 <= max_pairs {
        for i in 0..n {
            let (id_a, &mod_a) = nodes[i];
            let Some(vec_a) = emb.get(id_a.as_str()) else { continue };
            for j in (i + 1)..n {
                let (id_b, &mod_b) = nodes[j];
                let Some(vec_b) = emb.get(id_b.as_str()) else { continue };
                let sim = cosine_similarity(vec_a, vec_b);
                if mod_a == mod_b {
                    within_sum += sim;
                    within_count += 1;
                } else {
                    across_sum += sim;
                    across_count += 1;
                }
            }
        }
    } else {
        let mut rng = Rng::new(456);
        for _ in 0..max_pairs {
            let i = rng.next_usize(n);
            let mut j = rng.next_usize(n - 1);
            if j >= i { j += 1; }
            let (id_a, &mod_a) = nodes[i];
            let (id_b, &mod_b) = nodes[j];
            let Some(vec_a) = emb.get(id_a.as_str()) else { continue };
            let Some(vec_b) = emb.get(id_b.as_str()) else { continue };
            let sim = cosine_similarity(vec_a, vec_b);
            if mod_a == mod_b {
                within_sum += sim;
                within_count += 1;
            } else {
                across_sum += sim;
                across_count += 1;
            }
        }
    }

    let within_mean = if within_count > 0 { within_sum / within_count as f64 } else { 0.0 };
    let across_mean = if across_count > 0 { across_sum / across_count as f64 } else { 0.0 };
    within_mean - across_mean
}

// ---------------------------------------------------------------------------
// Semantic coherence per module (Step 4)
// ---------------------------------------------------------------------------

/// Compute average pairwise cosine similarity within each module.
///
/// Modules with fewer than 6 members get no score (too few pairs).
fn compute_module_coherence(
    emb: &HashMap<&str, Vec<f64>>,
    clusters: &HashMap<String, usize>,
) -> HashMap<usize, f64> {
    let mut module_members: HashMap<usize, Vec<&str>> = HashMap::new();
    for (node_id, &cluster_id) in clusters {
        if emb.contains_key(node_id.as_str()) {
            module_members.entry(cluster_id).or_default().push(node_id.as_str());
        }
    }

    let mut coherence = HashMap::new();
    for (&mod_id, members) in &module_members {
        if members.len() < 6 {
            continue; // Too few pairs for meaningful average.
        }

        let mut total_sim = 0.0;
        let mut count = 0usize;
        for i in 0..members.len() {
            let Some(vec_a) = emb.get(members[i]) else { continue };
            for j in (i + 1)..members.len() {
                let Some(vec_b) = emb.get(members[j]) else { continue };
                total_sim += cosine_similarity(vec_a, vec_b);
                count += 1;
            }
        }

        if count > 0 {
            coherence.insert(mod_id, total_sim / count as f64);
        }
    }

    coherence
}

// ---------------------------------------------------------------------------
// Rayleigh quotient (Step 5)
// ---------------------------------------------------------------------------

/// Compute the multivariate Rayleigh quotient: fᵀLf / ‖f‖².
///
/// Measures how smoothly semantic signals vary over the structural graph.
/// Uses the normalized Laplacian (same L = I - D^{-1/2} A D^{-1/2} as Phase 1).
/// Lower = semantics match structure well. Higher = tangled.
///
/// NOTE: graph.adj is directed. We symmetrize by processing both directions
/// and summing w_ij + w_ji for the symmetric weight, matching Phase 1's
/// spectral decomposition which symmetrizes in extract_subgraph.
fn rayleigh_quotient(emb: &HashMap<&str, Vec<f64>>, graph: &Graph) -> f64 {
    if emb.is_empty() || graph.n == 0 {
        return 0.0;
    }

    let dim = match emb.values().next() {
        Some(v) => v.len(),
        None => return 0.0,
    };

    // Pre-compute symmetric degrees: d_i = sum of symmetric weights.
    let sym_degree = symmetric_degrees(graph);

    // Pre-compute symmetric edge weights (hoisted out of dimension loop).
    // fᵀLf = Σ_{(i,j)∈E_sym} w_sym_ij * (f[i]/√d_i - f[j]/√d_j)²
    let mut sym_weights: std::collections::HashMap<(usize, usize), f64> =
        std::collections::HashMap::new();
    for i in 0..graph.n {
        for &(j, w) in &graph.adj[i] {
            let edge = if i < j { (i, j) } else { (j, i) };
            *sym_weights.entry(edge).or_insert(0.0) += w;
        }
    }

    // Pre-compute sqrt of symmetric degrees.
    let sqrt_degree: Vec<f64> = sym_degree.iter().map(|d| d.sqrt()).collect();

    let mut weighted_sum = 0.0;
    let mut weight_total = 0.0;

    for d in 0..dim {
        let mut f = vec![0.0f64; graph.n];
        let mut has_signal = vec![false; graph.n];

        for (node_id, vec) in emb.iter() {
            if let Some(&idx) = graph.node_index.get(*node_id) {
                f[idx] = vec[d];
                has_signal[idx] = true;
            }
        }

        let f_norm_sq: f64 = f.iter().zip(has_signal.iter())
            .filter(|&(_, &h)| h)
            .map(|(&v, _)| v * v)
            .sum();

        if f_norm_sq < 1e-15 {
            continue;
        }

        let mut ftlf = 0.0;
        for (&(i, j), &w_sym) in &sym_weights {
            if !has_signal[i] || !has_signal[j] { continue; }
            if sqrt_degree[i] < 1e-10 || sqrt_degree[j] < 1e-10 { continue; }

            let diff = f[i] / sqrt_degree[i] - f[j] / sqrt_degree[j];
            ftlf += w_sym * diff * diff;
        }

        let rq = ftlf / f_norm_sq;
        weighted_sum += rq * f_norm_sq;
        weight_total += f_norm_sq;
    }

    if weight_total < 1e-15 {
        0.0
    } else {
        weighted_sum / weight_total
    }
}

/// Compute symmetric degree for each node (sum of weights in both directions).
fn symmetric_degrees(graph: &Graph) -> Vec<f64> {
    let mut deg = vec![0.0f64; graph.n];
    for i in 0..graph.n {
        for &(j, w) in &graph.adj[i] {
            deg[i] += w;
            deg[j] += w;
        }
    }
    deg
}

/// Build symmetric weighted neighbor list: for each node, all neighbors
/// (both outgoing and incoming) with their accumulated symmetric weight.
/// Returns Vec<Vec<(usize, f64)>> indexed by node.
fn symmetric_neighbors(graph: &Graph) -> Vec<Vec<(usize, f64)>> {
    let mut sym: Vec<HashMap<usize, f64>> = (0..graph.n).map(|_| HashMap::new()).collect();
    for i in 0..graph.n {
        for &(j, w) in &graph.adj[i] {
            *sym[i].entry(j).or_insert(0.0) += w;
            *sym[j].entry(i).or_insert(0.0) += w;
        }
    }
    sym.into_iter()
        .map(|m| m.into_iter().collect::<Vec<_>>())
        .collect()
}

// ---------------------------------------------------------------------------
// GFT energy profile (Step 5)
// ---------------------------------------------------------------------------

/// Compute the Graph Fourier Transform energy profile.
///
/// For each eigenvector uᵢ: energy(λᵢ) = |uᵢᵀf|², summed across embedding dimensions.
///
/// `component_node_ids` maps component-local indices to graph node IDs,
/// ensuring eigenvector rows align with the correct signal values.
fn gft_energy_profile(
    emb: &HashMap<&str, Vec<f64>>,
    _graph: &Graph,
    eigenvalues: &[f64],
    eigenvectors: &[Vec<f64>],
    component_node_ids: &[String],
) -> (Vec<f64>, Vec<f64>) {
    if eigenvectors.is_empty() || emb.is_empty() || component_node_ids.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let dim = match emb.values().next() {
        Some(v) => v.len(),
        None => return (Vec::new(), Vec::new()),
    };

    let k = eigenvectors.first()
        .map(|r| r.len())
        .unwrap_or(0)
        .min(eigenvalues.len());
    if k == 0 {
        return (Vec::new(), Vec::new());
    }

    let mut energy = vec![0.0f64; k];

    for d in 0..dim {
        // Build signal vector aligned with component node ordering.
        let f: Vec<f64> = component_node_ids.iter()
            .map(|nid| emb.get(nid.as_str()).map(|v| v[d]).unwrap_or(0.0))
            .collect();

        // Project onto each eigenvector: f̂ᵢ = uᵢᵀf.
        for ki in 0..k {
            let mut dot = 0.0f64;
            for (row_idx, row) in eigenvectors.iter().enumerate() {
                if ki < row.len() && row_idx < f.len() {
                    dot += row[ki] * f[row_idx];
                }
            }
            energy[ki] += dot * dot;
        }
    }

    // Normalize to sum to 1.
    let total: f64 = energy.iter().sum();
    if total > 1e-15 {
        for e in &mut energy {
            *e /= total;
        }
    }

    (eigenvalues[..k].to_vec(), energy)
}

// ---------------------------------------------------------------------------
// Local variation per node (Step 6)
// ---------------------------------------------------------------------------

/// Compute per-node semantic variation: how much each node differs from its neighbors.
///
/// variation(n) = (1 / deg(n)) · Σⱼ wₙⱼ · (1 - cos(M[n], M[j]))
///
/// Uses symmetric neighbors (both outgoing and incoming edges) so that
/// nodes with only incoming edges (entry points, utilities) are not missed.
fn compute_local_variation(
    emb: &HashMap<&str, Vec<f64>>,
    graph: &Graph,
    sym_nbrs: &[Vec<(usize, f64)>],
) -> HashMap<String, f64> {
    let mut result = HashMap::new();

    for (node_id, vec_n) in emb.iter() {
        let Some(&idx) = graph.node_index.get(*node_id) else { continue };
        let neighbors = &sym_nbrs[idx];
        if neighbors.is_empty() {
            continue;
        }

        let weighted_degree: f64 = neighbors.iter().map(|(_, w)| w).sum();
        if weighted_degree < 1e-15 {
            continue;
        }

        let mut weighted_dist_sum = 0.0;
        for &(j, w) in neighbors {
            let neighbor_id = &graph.node_ids[j];
            if let Some(vec_j) = emb.get(neighbor_id.as_str()) {
                let cos_dist = 1.0 - cosine_similarity(vec_n, vec_j);
                weighted_dist_sum += w * cos_dist;
            }
        }

        result.insert(node_id.to_string(), weighted_dist_sum / weighted_degree);
    }

    result
}

// ---------------------------------------------------------------------------
// AMI: Adjusted Mutual Information (Step 8)
// ---------------------------------------------------------------------------

/// Compute AMI between structural clusters and semantic clusters.
///
/// Semantic clusters: spherical k-means on L2-normalized embeddings.
/// Sweep k = structural_k ± 2, report max AMI.
fn compute_ami(
    emb: &HashMap<&str, Vec<f64>>,
    clusters: &HashMap<String, usize>,
) -> f64 {
    if emb.len() < 4 || clusters.is_empty() {
        return 0.0;
    }

    // Determine structural k.
    let structural_k = clusters.values().cloned().collect::<std::collections::HashSet<_>>().len();
    if structural_k < 2 {
        return 0.0;
    }

    // Build ordered data for k-means.
    let node_order: Vec<&String> = clusters.keys()
        .filter(|k| emb.contains_key(k.as_str()))
        .collect();

    if node_order.len() < 4 {
        return 0.0;
    }

    // L2-normalize embeddings for spherical k-means.
    let data: Vec<Vec<f64>> = node_order.iter()
        .map(|nid| {
            let v = &emb[nid.as_str()];
            let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-10 {
                v.iter().map(|x| x / norm).collect()
            } else {
                v.clone()
            }
        })
        .collect();

    let structural_labels: Vec<usize> = node_order.iter()
        .map(|nid| clusters[*nid])
        .collect();

    // Sweep k = max(2, structural_k - 2) .. structural_k + 2.
    let k_min = 2.max(structural_k.saturating_sub(2));
    let k_max = (structural_k + 2).min(data.len() - 1);
    if k_min > k_max {
        return 0.0;
    }
    let mut best_ami = f64::NEG_INFINITY;

    for k in k_min..=k_max {
        let km = crate::clustering::kmeans(&data, k, 100, 42);
        let ami = adjusted_mutual_information(&structural_labels, &km.labels);
        if ami > best_ami {
            best_ami = ami;
        }
    }

    best_ami.max(0.0)
}

/// Compute Adjusted Mutual Information between two label vectors.
fn adjusted_mutual_information(labels_a: &[usize], labels_b: &[usize]) -> f64 {
    let n = labels_a.len();
    if n == 0 {
        return 0.0;
    }

    // Build contingency table.
    let max_a = labels_a.iter().cloned().max().unwrap_or(0) + 1;
    let max_b = labels_b.iter().cloned().max().unwrap_or(0) + 1;
    let mut contingency = vec![vec![0usize; max_b]; max_a];
    for i in 0..n {
        contingency[labels_a[i]][labels_b[i]] += 1;
    }

    // Marginals.
    let row_sums: Vec<usize> = contingency.iter().map(|r| r.iter().sum()).collect();
    let col_sums: Vec<usize> = (0..max_b)
        .map(|j| contingency.iter().map(|r| r[j]).sum())
        .collect();

    let n_f = n as f64;

    // Mutual information: MI = Σᵢⱼ (nᵢⱼ/n) * log(n * nᵢⱼ / (aᵢ * bⱼ))
    let mut mi = 0.0;
    for i in 0..max_a {
        if row_sums[i] == 0 { continue; }
        for j in 0..max_b {
            if col_sums[j] == 0 || contingency[i][j] == 0 { continue; }
            let nij = contingency[i][j] as f64;
            mi += (nij / n_f) * (n_f * nij / (row_sums[i] as f64 * col_sums[j] as f64)).ln();
        }
    }

    // Entropies.
    let h_a: f64 = row_sums.iter()
        .filter(|&&s| s > 0)
        .map(|&s| { let p = s as f64 / n_f; -p * p.ln() })
        .sum();
    let h_b: f64 = col_sums.iter()
        .filter(|&&s| s > 0)
        .map(|&s| { let p = s as f64 / n_f; -p * p.ln() })
        .sum();

    // Expected MI under independence (approximation for large n).
    // E[MI] ≈ Σᵢⱼ Σₙᵢⱼ (nij/n) * log(n*nij/(ai*bj)) * P(nij)
    // Use the hypergeometric expectation.
    let expected_mi = expected_mutual_information(&row_sums, &col_sums, n);

    let denominator = ((h_a + h_b) / 2.0) - expected_mi;
    // Guard: denominator can be near-zero or slightly negative due to
    // floating-point approximation in the hypergeometric sum.
    if denominator <= 1e-15 {
        return 0.0;
    }

    ((mi - expected_mi) / denominator).clamp(-1.0, 1.0)
}

/// Compute expected mutual information under random assignment.
///
/// Uses the exact formula: E[MI] = Σᵢⱼ Σ_{nij} P(nij) * (nij/n) * log(n*nij/(ai*bj))
/// where nij follows the hypergeometric distribution.
/// Simplified for efficiency: compute only the expected value terms.
fn expected_mutual_information(row_sums: &[usize], col_sums: &[usize], n: usize) -> f64 {
    let n_f = n as f64;
    let mut emi = 0.0;

    for &a_i in row_sums {
        if a_i == 0 { continue; }
        for &b_j in col_sums {
            if b_j == 0 { continue; }
            let start = if a_i + b_j > n { a_i + b_j - n } else { 1 };
            let end = a_i.min(b_j);

            for nij in start..=end {
                // Hypergeometric probability P(nij | ai, bj, n).
                let log_p = ln_binomial(a_i, nij) + ln_binomial(n - a_i, b_j - nij)
                    - ln_binomial(n, b_j);
                let p = log_p.exp();
                if p < 1e-300 { continue; }

                let nij_f = nij as f64;
                let term = (nij_f / n_f) * (n_f * nij_f / (a_i as f64 * b_j as f64)).ln();
                emi += p * term;
            }
        }
    }

    emi
}

/// Log of binomial coefficient ln(C(n, k)).
fn ln_binomial(n: usize, k: usize) -> f64 {
    if k > n { return f64::NEG_INFINITY; }
    ln_factorial(n) - ln_factorial(k) - ln_factorial(n - k)
}

/// Log factorial using Stirling's approximation for large n.
fn ln_factorial(n: usize) -> f64 {
    if n <= 1 { return 0.0; }
    // Use lgamma for accuracy.
    ln_gamma(n as f64 + 1.0)
}

/// Log gamma function (Lanczos approximation).
fn ln_gamma(x: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    // Coefficients for Lanczos approximation.
    let g = 7.0;
    let coefs = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    let xx = x - 1.0;
    let mut sum = coefs[0];
    for (i, &c) in coefs.iter().enumerate().skip(1) {
        sum += c / (xx + i as f64);
    }

    let t = xx + g + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (t.ln() * (xx + 0.5)) - t + sum.ln()
}

// ---------------------------------------------------------------------------
// Misplaced concern detection (Step 7)
// ---------------------------------------------------------------------------

/// Detect nodes whose semantics better match a different module.
fn detect_misplaced_concerns(
    emb: &HashMap<&str, Vec<f64>>,
    clusters: &HashMap<String, usize>,
    module_coherence: &HashMap<usize, f64>,
    roles: &HashMap<String, String>,
    graph: &Graph,
    sym_nbrs: &[Vec<(usize, f64)>],
    null_threshold: f64,
) -> Vec<MisplacedConcern> {
    // Significance threshold: 30% of null coherence baseline.
    // Scales with the codebase's background similarity level.
    // Replaces hardcoded 0.15. On a codebase with null ~0.43, this gives ~0.13.
    // On a codebase with null ~0.20, this gives ~0.06 (more sensitive).
    let significance_threshold = (0.3 * null_threshold).max(0.05);

    // Compute module centroids.
    let centroids = compute_module_centroids(emb, clusters);
    if centroids.is_empty() {
        return Vec::new();
    }

    // Count members per module.
    let mut module_sizes: HashMap<usize, usize> = HashMap::new();
    for &mod_id in clusters.values() {
        *module_sizes.entry(mod_id).or_insert(0) += 1;
    }

    let mut results = Vec::new();
    let mut flagged_per_module: HashMap<usize, usize> = HashMap::new();

    for (node_id, &own_module) in clusters {
        let Some(node_vec) = emb.get(node_id.as_str()) else { continue };

        // Skip modules with < 6 members.
        if module_sizes.get(&own_module).copied().unwrap_or(0) < 6 {
            continue;
        }

        // Skip bridge, hub, utility roles.
        if let Some(role) = roles.get(node_id) {
            if matches!(role.as_str(), "bridge" | "hub" | "utility") {
                continue;
            }
        }

        // Similarity to own module centroid.
        let Some(own_centroid) = centroids.get(&own_module) else { continue };
        let sim_own = cosine_similarity(node_vec, own_centroid);

        // Find best other module.
        let mut best_module = own_module;
        let mut sim_best = f64::NEG_INFINITY;

        for (&mod_id, centroid) in &centroids {
            if mod_id == own_module { continue; }
            // Gate: target module must be coherent (above data-adaptive threshold).
            // Skip modules with no coherence score (< 6 members, unreliable centroid).
            match module_coherence.get(&mod_id) {
                Some(&coh) if coh < null_threshold => continue,
                None => continue, // No coherence = too small, skip.
                _ => {}
            }
            let sim = cosine_similarity(node_vec, centroid);
            if sim > sim_best {
                sim_best = sim;
                best_module = mod_id;
            }
        }

        if best_module == own_module { continue; }
        if sim_best <= sim_own { continue; }
        if (sim_best - sim_own) < significance_threshold { continue; }
        // Gate: only flag nodes poorly placed in own module.
        // Replaces hardcoded 0.4 with null coherence threshold.
        // Floor at 0.1 so the gate remains meaningful when null is degenerate (0.0).
        if sim_own >= null_threshold.max(0.1) { continue; }

        // Edge evidence: must have at least one edge (in or out) to the target module.
        let has_edge_to_target = if let Some(&idx) = graph.node_index.get(node_id.as_str()) {
            sym_nbrs[idx].iter().any(|&(j, _)| {
                let neighbor_id = &graph.node_ids[j];
                clusters.get(neighbor_id).copied() == Some(best_module)
            })
        } else {
            false
        };
        if !has_edge_to_target { continue; }

        // k-NN suppression: if the node is close to k=3 own-module neighbors
        // (using symmetric neighbors), it likely belongs to a sub-cluster.
        if let Some(&idx) = graph.node_index.get(node_id.as_str()) {
            let mut own_module_sims: Vec<f64> = sym_nbrs[idx].iter()
                .filter_map(|&(j, _)| {
                    let nid = &graph.node_ids[j];
                    if clusters.get(nid).copied() == Some(own_module) {
                        emb.get(nid.as_str()).map(|v| cosine_similarity(node_vec, v))
                    } else {
                        None
                    }
                })
                .collect();
            own_module_sims.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            // If 3+ own-module neighbors are more similar than the best other centroid,
            // the node belongs to a local sub-cluster — suppress.
            let k = 3;
            if own_module_sims.len() >= k && own_module_sims[k - 1] > sim_best {
                continue;
            }
        }

        *flagged_per_module.entry(own_module).or_insert(0) += 1;

        results.push(MisplacedConcern {
            node_id: node_id.clone(),
            own_module,
            best_module,
            similarity_own: sim_own,
            similarity_best: sim_best,
        });
    }

    // Per-module cap: if the flagged fraction exceeds a scale-adaptive threshold,
    // suppress — the module boundary itself is wrong, not individual nodes.
    // Uses expected_rate + 3*SE where expected_rate = 0.15 (typical detection rate
    // for well-calibrated thresholds). For a 20-member module: ~34%. For 100: ~26%.
    // This replaces the hardcoded 40% and adapts to module size.
    let expected_rate = 0.15;
    results.retain(|mc| {
        let flagged = flagged_per_module.get(&mc.own_module).copied().unwrap_or(0);
        let size = module_sizes.get(&mc.own_module).copied().unwrap_or(1);
        let flagged_frac = flagged as f64 / size.max(1) as f64;
        let se = (expected_rate * (1.0 - expected_rate) / size.max(1) as f64).sqrt();
        flagged_frac <= expected_rate + 3.0 * se
    });

    results
}

/// Compute centroid of semantic embeddings per module.
fn compute_module_centroids(
    emb: &HashMap<&str, Vec<f64>>,
    clusters: &HashMap<String, usize>,
) -> HashMap<usize, Vec<f64>> {
    let dim = match emb.values().next() {
        Some(v) => v.len(),
        None => return HashMap::new(),
    };

    let mut sums: HashMap<usize, Vec<f64>> = HashMap::new();
    let mut counts: HashMap<usize, usize> = HashMap::new();

    for (node_id, &mod_id) in clusters {
        if let Some(vec) = emb.get(node_id.as_str()) {
            let entry = sums.entry(mod_id).or_insert_with(|| vec![0.0; dim]);
            for (i, &v) in vec.iter().enumerate() {
                entry[i] += v;
            }
            *counts.entry(mod_id).or_insert(0) += 1;
        }
    }

    let mut centroids = HashMap::new();
    for (mod_id, sum) in sums {
        let count = counts[&mod_id] as f64;
        centroids.insert(mod_id, sum.into_iter().map(|v| v / count).collect());
    }

    centroids
}

// ---------------------------------------------------------------------------
// Incoherent module detection (Step 7)
// ---------------------------------------------------------------------------

/// Detect modules whose members are semantically unrelated.
/// For each incoherent module, identifies semantic sub-clusters via k-means.
fn detect_incoherent_modules(
    emb: &HashMap<&str, Vec<f64>>,
    clusters: &HashMap<String, usize>,
    module_coherence: &HashMap<usize, f64>,
    null_threshold: f64,
) -> Vec<IncoherentModule> {

    // Group members by module.
    let mut module_members: HashMap<usize, Vec<&str>> = HashMap::new();
    for (node_id, &mod_id) in clusters {
        if emb.contains_key(node_id.as_str()) {
            module_members.entry(mod_id).or_default().push(node_id.as_str());
        }
    }

    let mut results = Vec::new();
    for (&mod_id, &coherence) in module_coherence {
        if coherence < null_threshold {
            let module_size = module_members.get(&mod_id).map(|m| m.len()).unwrap_or(0);

            // Find semantic sub-clusters within this module.
            let (sub_clusters, sub_sil, best_k, inter_sim) = if let Some(members) = module_members.get(&mod_id) {
                find_sub_clusters(emb, members)
            } else {
                (Vec::new(), 0.0, 1, 1.0)
            };

            // Permutation null: shuffle embeddings among members, re-run k-means,
            // compare observed silhouette against null distribution.
            // Replaces hardcoded sub_sil > 0.3 and inter_sim < 0.3.
            if best_k < 2 {
                continue;
            }
            let sil_is_significant = if let Some(members) = module_members.get(&mod_id) {
                let n_perms = 50;
                let mut rng = Rng::new(555 + mod_id as u64);
                let mut null_sils = Vec::with_capacity(n_perms);
                let data: Vec<Vec<f64>> = members.iter()
                    .filter_map(|nid| emb.get(*nid).cloned())
                    .collect();
                if data.len() >= 6 {
                    for _ in 0..n_perms {
                        // Null: random label assignment on the same data points.
                        // Shuffling rows is invariant to k-means; instead, generate
                        // random labels and compute silhouette of that random partition.
                        let mut random_labels: Vec<usize> = (0..data.len())
                            .map(|_| rng.next_usize(best_k))
                            .collect();
                        // Ensure all k labels appear (otherwise silhouette is undefined).
                        for label in 0..best_k {
                            if !random_labels.contains(&label) {
                                let idx = rng.next_usize(data.len());
                                random_labels[idx] = label;
                            }
                        }
                        // Compute centroids for random labels.
                        let dim = data[0].len();
                        let mut centroids = vec![vec![0.0; dim]; best_k];
                        let mut counts = vec![0usize; best_k];
                        for (i, label) in random_labels.iter().enumerate() {
                            for (d, &v) in data[i].iter().enumerate() {
                                centroids[*label][d] += v;
                            }
                            counts[*label] += 1;
                        }
                        for c in 0..best_k {
                            if counts[c] > 0 {
                                for d in 0..dim {
                                    centroids[c][d] /= counts[c] as f64;
                                }
                            }
                        }
                        let sil = crate::clustering::silhouette_score(&data, &random_labels, &centroids);
                        null_sils.push(sil);
                    }
                    null_sils.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let p95_idx = (null_sils.len() as f64 * 0.95) as usize;
                    let null_p95 = null_sils.get(p95_idx.min(null_sils.len().saturating_sub(1))).copied().unwrap_or(0.3);
                    sub_sil > null_p95
                } else {
                    false
                }
            } else {
                false
            };
            if !sil_is_significant {
                continue;
            }
            // Inter-cluster sim: suppress if sub-clusters are not well-separated
            // (observed inter_sim must be below the population mean pairwise sim).
            if inter_sim >= null_threshold {
                continue;
            }

            results.push(IncoherentModule {
                module_id: mod_id,
                coherence,
                null_threshold,
                sub_clusters,
                module_size,
                best_k,
            });
        }
    }

    results
}

/// Find semantic sub-clusters within a module using k-means.
/// Returns (sub_cluster_terms, best_silhouette, best_k, inter_cluster_sim).
fn find_sub_clusters(
    emb: &HashMap<&str, Vec<f64>>,
    members: &[&str],
) -> (Vec<Vec<String>>, f64, usize, f64) {
    if members.len() < 6 {
        return (Vec::new(), 0.0, 1, 1.0);
    }

    // Build data matrix for k-means.
    let data: Vec<Vec<f64>> = members.iter()
        .filter_map(|nid| emb.get(*nid).cloned())
        .collect();
    if data.len() < 6 {
        return (Vec::new(), 0.0, 1, 1.0);
    }

    // Sweep k = 2..min(|M|/3, 6), pick best silhouette (spec requirement).
    let max_k = (data.len() / 3).min(6).max(2);
    let mut best_sil = f64::NEG_INFINITY;
    let mut best_labels_owned: Vec<usize> = Vec::new();
    let mut best_k = 2;

    for k in 2..=max_k {
        let km = crate::clustering::kmeans(&data, k, 50, 42);
        let sil = crate::clustering::silhouette_score(&data, &km.labels, &km.centroids);
        if sil > best_sil + 0.05 || best_labels_owned.is_empty() {
            if sil > best_sil {
                best_sil = sil;
                best_labels_owned = km.labels;
                best_k = k;
            }
        }
    }

    let best_labels = &best_labels_owned;

    // Build clusters and compute top_terms per sub-cluster.
    let mut sub_clusters_map: HashMap<usize, Vec<String>> = HashMap::new();
    for (i, member) in members.iter().enumerate() {
        if i < best_labels.len() {
            sub_clusters_map.entry(best_labels[i])
                .or_default()
                .push(member.to_string());
        }
    }

    // Get top terms for each sub-cluster using actual sub-cluster labels for TF-IDF.
    // Each member is assigned its k-means label so IDF can discriminate across sub-clusters.
    let all_members_with_labels: HashMap<String, usize> = members.iter()
        .enumerate()
        .filter(|(i, _)| *i < best_labels.len())
        .map(|(i, m)| (m.to_string(), best_labels[i]))
        .collect();
    let terms = top_terms(&all_members_with_labels, 3);

    let mut sub_cluster_terms: Vec<Vec<String>> = Vec::new();
    for cluster_id in 0..best_k {
        let term_list = terms.get(&cluster_id).cloned().unwrap_or_default();
        sub_cluster_terms.push(term_list);
    }

    // Compute inter-cluster similarity: mean cosine sim between sub-cluster centroids.
    let centroids: Vec<Vec<f64>> = {
        let km = crate::clustering::kmeans(&data, best_k, 50, 42);
        km.centroids
    };
    let inter_sim = if centroids.len() >= 2 {
        let mut sim_sum = 0.0;
        let mut pair_count = 0;
        for i in 0..centroids.len() {
            for j in (i + 1)..centroids.len() {
                sim_sum += cosine_similarity(&centroids[i], &centroids[j]);
                pair_count += 1;
            }
        }
        if pair_count > 0 { sim_sum / pair_count as f64 } else { 1.0 }
    } else {
        1.0
    };

    (sub_cluster_terms, best_sil, best_k, inter_sim)
}

/// Compute expected coherence for random groups (null threshold).
///
/// Returns (mean, std) of the null coherence distribution.
/// The mean is used as the threshold; std is used for Cohen's d in
/// downstream diagnostics (cross-package-coupling root cause classification).
pub fn compute_null_coherence_threshold(
    emb: &HashMap<&str, Vec<f64>>,
    clusters: &HashMap<String, usize>,
) -> (f64, f64) {
    let vecs: Vec<&Vec<f64>> = clusters.keys()
        .filter_map(|k| emb.get(k.as_str()))
        .collect();
    let n = vecs.len();
    if n < 12 {
        return (0.0, 0.0);
    }

    // Sample random groups of size 6-10 and compute their coherence.
    let mut rng = Rng::new(789);
    let n_samples = 100;
    let group_size = 8.min(n);
    let mut samples = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let mut indices: Vec<usize> = (0..n).collect();
        // Partial Fisher-Yates for first group_size elements.
        for i in 0..group_size {
            let j = i + rng.next_usize(n - i);
            indices.swap(i, j);
        }

        let mut group_sim = 0.0;
        let mut count = 0;
        for i in 0..group_size {
            for j in (i + 1)..group_size {
                group_sim += cosine_similarity(vecs[indices[i]], vecs[indices[j]]);
                count += 1;
            }
        }

        if count > 0 {
            samples.push(group_sim / count as f64);
        }
    }

    if samples.is_empty() {
        return (0.0, 0.0);
    }
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let variance = samples.iter().map(|&s| (s - mean) * (s - mean)).sum::<f64>() / samples.len() as f64;
    (mean, variance.sqrt())
}

// ---------------------------------------------------------------------------
// Cosine similarity
// ---------------------------------------------------------------------------

/// Cosine similarity between two vectors. Returns 0 if either is zero-norm.
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = (norm_a * norm_b).sqrt();
    if denom < 1e-15 {
        0.0
    } else {
        dot / denom
    }
}

// ---------------------------------------------------------------------------
// Shadow dependency detection (Step 9, Diagnostic 10)
// ---------------------------------------------------------------------------

fn detect_shadow_dependencies(
    emb: &HashMap<&str, Vec<f64>>,
    graph: &Graph,
    clusters: &HashMap<String, usize>,
) -> Vec<ShadowDependency> {
    let mut nodes: Vec<&str> = emb.keys().copied().collect();
    nodes.sort(); // Deterministic ordering for stable issue IDs.
    let n = nodes.len();
    if n < 4 {
        return Vec::new();
    }

    // Build undirected successors for BFS (both directions).
    let mut successors: Vec<Vec<usize>> = vec![Vec::new(); graph.n];
    for (i, adj) in graph.adj.iter().enumerate() {
        for &(j, _) in adj {
            successors[i].push(j);
            successors[j].push(i);
        }
    }
    // Deduplicate.
    for succ in &mut successors {
        succ.sort_unstable();
        succ.dedup();
    }

    let mut results = Vec::new();
    for i in 0..n {
        let node_a = nodes[i];
        let mod_a = clusters.get(node_a);
        let vec_a = &emb[node_a];

        for j in (i + 1)..n {
            let node_b = nodes[j];
            let mod_b = clusters.get(node_b);

            // Must be in different modules.
            if mod_a == mod_b {
                continue;
            }

            let sim = cosine_similarity(vec_a, &emb[node_b]);
            if sim <= 0.85 {
                continue;
            }

            // Structural distance: BFS capped at 4.
            let idx_a = graph.node_index.get(node_a);
            let idx_b = graph.node_index.get(node_b);
            let distance = match (idx_a, idx_b) {
                (Some(&a), Some(&b)) => crate::algorithms::bfs_distance(&successors, a, b, 4),
                _ => None,
            };

            // Only flag if structurally distant (>3 hops or no path).
            if let Some(d) = distance {
                if d <= 3 {
                    continue;
                }
            }

            // FP suppression: test paths (check path components, not substrings).
            let is_test_path = |path: &str| -> bool {
                path.contains("/test/") || path.contains("/tests/")
                    || path.contains("_test.") || path.contains("test_")
                    || path.ends_with("_test") || path.contains("/spec/")
            };
            let file_a = graph.node_index.get(node_a)
                .map(|&i| graph.node_files[i].as_deref().unwrap_or(""))
                .unwrap_or("");
            let file_b = graph.node_index.get(node_b)
                .map(|&i| graph.node_files[i].as_deref().unwrap_or(""))
                .unwrap_or("");
            if is_test_path(file_a) || is_test_path(file_b) {
                continue;
            }

            // FP suppression: trait implementations (both inherit from same target).
            let inherits = graph.edges_of_kind("inherits");
            let targets_a: HashSet<usize> = inherits.iter()
                .filter(|&&(src, _)| graph.node_ids.get(src).map(|s| s.as_str()) == Some(node_a))
                .map(|&(_, tgt)| tgt)
                .collect();
            let targets_b: HashSet<usize> = inherits.iter()
                .filter(|&&(src, _)| graph.node_ids.get(src).map(|s| s.as_str()) == Some(node_b))
                .map(|&(_, tgt)| tgt)
                .collect();
            if !targets_a.is_empty() && !targets_a.is_disjoint(&targets_b) {
                continue; // Both implement the same trait.
            }

            results.push(ShadowDependency {
                node_a: node_a.to_string(),
                node_b: node_b.to_string(),
                similarity: sim,
                structural_distance: distance,
            });
        }
    }
    results
}

// ---------------------------------------------------------------------------
// Redundant API detection (Step 10, Diagnostic 11)
// ---------------------------------------------------------------------------

fn detect_redundant_api(
    emb: &HashMap<&str, Vec<f64>>,
    graph: &Graph,
    clusters: &HashMap<String, usize>,
    roles: &HashMap<String, String>,
) -> Vec<RedundantApi> {
    // Group entry points by module.
    // Spec: role == "entry_point" OR in_degree_from_outside > 0.
    let mut module_entries: HashMap<usize, Vec<&str>> = HashMap::new();
    let mut added: HashSet<&str> = HashSet::new();

    // First pass: role-based entry points.
    for (node_id, role) in roles {
        if role != "entry_point" {
            continue;
        }
        if let Some(&mod_id) = clusters.get(node_id) {
            if emb.contains_key(node_id.as_str()) {
                module_entries.entry(mod_id).or_default().push(node_id.as_str());
                added.insert(node_id.as_str());
            }
        }
    }

    // Second pass: nodes with cross-module callers (in_degree_from_outside > 0).
    for (node_id, &mod_id) in clusters {
        if added.contains(node_id.as_str()) || !emb.contains_key(node_id.as_str()) {
            continue;
        }
        if let Some(&idx) = graph.node_index.get(node_id.as_str()) {
            // Check if any predecessor is in a different module.
            let has_external_caller = graph.adj.iter().enumerate().any(|(src_idx, adj)| {
                adj.iter().any(|&(tgt, _)| tgt == idx && clusters.get(&graph.node_ids[src_idx]) != Some(&mod_id))
            });
            if has_external_caller {
                module_entries.entry(mod_id).or_default().push(node_id.as_str());
            }
        }
    }

    let mut results = Vec::new();

    for (&mod_id, entries) in &module_entries {
        if entries.len() < 3 {
            continue;
        }

        // Pairwise semantic similarity among entry points.
        let mut redundant_pairs: Vec<(usize, usize, f64)> = Vec::new();
        for i in 0..entries.len() {
            for j in (i + 1)..entries.len() {
                let sim = cosine_similarity(&emb[entries[i]], &emb[entries[j]]);
                if sim > 0.7 {
                    redundant_pairs.push((i, j, sim));
                }
            }
        }

        if redundant_pairs.is_empty() {
            continue;
        }

        // Compute callee overlap (Jaccard) for each redundant pair.
        let mod_members: HashSet<&str> = clusters.iter()
            .filter(|(_, m)| **m == mod_id)
            .map(|(nid, _)| nid.as_str())
            .collect();

        // Use calls-only edges for callee overlap (not imports/inherits).
        let call_edges = graph.edges_of_kind("calls");
        let callees_per_entry: Vec<HashSet<usize>> = entries.iter()
            .map(|&nid| {
                graph.node_index.get(nid)
                    .map(|&idx| {
                        call_edges.iter()
                            .filter(|&&(src, _)| src == idx)
                            .filter(|&&(_, tgt)| mod_members.contains(graph.node_ids[tgt].as_str()))
                            .map(|&(_, tgt)| tgt)
                            .collect::<HashSet<usize>>()
                    })
                    .unwrap_or_default()
            })
            .collect();

        // Filter pairs by callee overlap > 0.5.
        let confirmed_pairs: Vec<(usize, usize, f64, f64)> = redundant_pairs.iter()
            .filter_map(|&(i, j, sim)| {
                let ci = &callees_per_entry[i];
                let cj = &callees_per_entry[j];
                let union_size = ci.union(cj).count();
                if union_size == 0 {
                    return None;
                }
                let jaccard = ci.intersection(cj).count() as f64 / union_size as f64;
                if jaccard > 0.5 {
                    Some((i, j, sim, jaccard))
                } else {
                    None
                }
            })
            .collect();

        if confirmed_pairs.is_empty() {
            continue;
        }

        // Union-find for connected components.
        let n = entries.len();
        let mut parent: Vec<usize> = (0..n).collect();
        fn find(parent: &mut [usize], x: usize) -> usize {
            if parent[x] != x {
                parent[x] = find(parent, parent[x]);
            }
            parent[x]
        }
        for &(i, j, _, _) in &confirmed_pairs {
            let ri = find(&mut parent, i);
            let rj = find(&mut parent, j);
            if ri != rj {
                parent[ri] = rj;
            }
        }

        // Collect components.
        let mut components: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..n {
            let root = find(&mut parent, i);
            components.entry(root).or_default().push(i);
        }

        // Emit clusters of >= 3 redundant entry points.
        for members in components.values() {
            if members.len() < 3 {
                continue;
            }

            // FP suppression: intentional overloads (shared stem with type suffix).
            let names: Vec<&str> = members.iter()
                .map(|&i| entries[i].rsplit('.').next().unwrap_or(entries[i]))
                .collect();
            let type_suffixes = ["_str", "_bytes", "_file", "_async", "_sync", "_mut"];
            let has_overload_pattern = names.iter().all(|name| {
                type_suffixes.iter().any(|suf| name.ends_with(suf))
            });
            if has_overload_pattern {
                continue;
            }

            // Compute mean similarity and callee overlap for the cluster.
            let mut sim_sum = 0.0;
            let mut overlap_sum = 0.0;
            let mut pair_count = 0;
            for &(i, j, sim, jaccard) in &confirmed_pairs {
                if members.contains(&i) && members.contains(&j) {
                    sim_sum += sim;
                    overlap_sum += jaccard;
                    pair_count += 1;
                }
            }
            let mean_sim = if pair_count > 0 { sim_sum / pair_count as f64 } else { 0.0 };
            let mean_overlap = if pair_count > 0 { overlap_sum / pair_count as f64 } else { 0.0 };

            results.push(RedundantApi {
                module_id: mod_id,
                entry_points: members.iter().map(|&i| entries[i].to_string()).collect(),
                mean_similarity: mean_sim,
                mean_callee_overlap: mean_overlap,
            });
        }
    }
    results
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_node_id() {
        let tokens = tokenize_node_id("pkg.module.getUserName");
        assert_eq!(tokens, vec!["user", "name"]);
    }

    #[test]
    fn test_tokenize_snake_case() {
        let tokens = tokenize_node_id("pkg.auth.validate_jwt_token");
        assert_eq!(tokens, vec!["validate", "jwt", "token"]);
    }

    #[test]
    fn test_split_camel_case_acronym() {
        let parts = split_camel_case("HTMLParser");
        assert_eq!(parts, vec!["HTML", "Parser"]);
    }

    #[test]
    fn test_split_camel_case_simple() {
        let parts = split_camel_case("getUserName");
        assert_eq!(parts, vec!["get", "User", "Name"]);
    }

    #[test]
    fn test_top_terms_basic() {
        let mut clusters = HashMap::new();
        // Module 0: auth-related
        clusters.insert("auth.validate_token".to_string(), 0);
        clusters.insert("auth.refresh_token".to_string(), 0);
        clusters.insert("auth.check_session".to_string(), 0);
        // Module 1: payment-related
        clusters.insert("payment.charge_card".to_string(), 1);
        clusters.insert("payment.process_refund".to_string(), 1);
        clusters.insert("payment.validate_amount".to_string(), 1);

        let terms = top_terms(&clusters, 3);
        assert!(terms.contains_key(&0));
        assert!(terms.contains_key(&1));

        // "token" should be a top term for module 0
        let auth_terms = &terms[&0];
        assert!(auth_terms.contains(&"token".to_string()), "expected 'token' in {:?}", auth_terms);

        // "payment" should NOT appear (it's the module prefix, stripped by leaf extraction)
        // "charge" or "refund" should appear for module 1
        let payment_terms = &terms[&1];
        assert!(!payment_terms.is_empty());
    }

    #[test]
    fn test_top_terms_empty() {
        let clusters = HashMap::new();
        let terms = top_terms(&clusters, 3);
        assert!(terms.is_empty());
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_zero() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 2.0];
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_ami_perfect_agreement() {
        let a = vec![0, 0, 0, 1, 1, 1];
        let b = vec![0, 0, 0, 1, 1, 1];
        let ami = adjusted_mutual_information(&a, &b);
        assert!(ami > 0.99, "AMI for identical partitions should be ~1.0, got {ami}");
    }

    #[test]
    fn test_ami_no_agreement() {
        // Random-looking partitions
        let a = vec![0, 1, 0, 1, 0, 1, 0, 1];
        let b = vec![0, 0, 1, 1, 0, 0, 1, 1];
        let ami = adjusted_mutual_information(&a, &b);
        assert!(ami.abs() < 0.3, "AMI for unrelated partitions should be near 0, got {ami}");
    }

    #[test]
    fn test_module_coherence_basic() {
        let mut emb: HashMap<&str, Vec<f64>> = HashMap::new();
        let mut clusters: HashMap<String, usize> = HashMap::new();

        // Module 0: similar vectors
        for i in 0..8 {
            let id = format!("a.f{i}");
            let vec = vec![1.0 + i as f64 * 0.01, 0.0, 0.0];
            emb.insert(Box::leak(id.clone().into_boxed_str()), vec);
            clusters.insert(id, 0);
        }
        // Module 1: similar but different vectors
        for i in 0..8 {
            let id = format!("b.f{i}");
            let vec = vec![0.0, 1.0 + i as f64 * 0.01, 0.0];
            emb.insert(Box::leak(id.clone().into_boxed_str()), vec);
            clusters.insert(id, 1);
        }

        let coherence = compute_module_coherence(&emb, &clusters);
        assert!(coherence.contains_key(&0));
        assert!(coherence.contains_key(&1));
        assert!(*coherence.get(&0).unwrap() > 0.99, "module 0 should be highly coherent");
        assert!(*coherence.get(&1).unwrap() > 0.99, "module 1 should be highly coherent");
    }

    #[test]
    fn test_ln_gamma_basic() {
        // ln(Γ(1)) = 0
        assert!((ln_gamma(1.0)).abs() < 1e-10);
        // ln(Γ(2)) = ln(1!) = 0
        assert!((ln_gamma(2.0)).abs() < 1e-10);
        // ln(Γ(6)) = ln(5!) = ln(120) ≈ 4.787
        assert!((ln_gamma(6.0) - (120.0f64).ln()).abs() < 1e-6);
    }

    // ── Integration test for analyze_semantic ──

    /// Build a small graph + embeddings for testing the full semantic pipeline.
    fn make_semantic_test_data() -> (
        HashMap<String, Vec<f32>>,
        crate::graph::Graph,
        HashMap<String, usize>,
        Vec<f64>,
        Vec<Vec<f64>>,
        HashMap<String, String>,
        Vec<String>,
    ) {
        // Two clear modules: "auth" (nodes 0-5) and "pay" (nodes 6-11).
        // Auth embeddings cluster around [1,0,0], pay around [0,1,0].
        let mut embeddings: HashMap<String, Vec<f32>> = HashMap::new();
        let mut clusters: HashMap<String, usize> = HashMap::new();
        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        for i in 0..6 {
            let id = format!("auth.f{i}");
            // Auth vectors: [1 + noise, noise, noise]
            let noise = i as f32 * 0.02;
            embeddings.insert(id.clone(), vec![1.0 + noise, noise, 0.05]);
            clusters.insert(id.clone(), 0);
            nodes.push(crate::types::NodeEntry {
                id: id.clone(), kind: "function".into(),
                file: Some("auth.rs".into()), line: Some(i as u32 + 1), line_end: None,
            });
        }
        for i in 0..6 {
            let id = format!("pay.f{i}");
            let noise = i as f32 * 0.02;
            embeddings.insert(id.clone(), vec![noise, 1.0 + noise, 0.05]);
            clusters.insert(id.clone(), 1);
            nodes.push(crate::types::NodeEntry {
                id: id.clone(), kind: "function".into(),
                file: Some("pay.rs".into()), line: Some(i as u32 + 1), line_end: None,
            });
        }

        // Edges: dense within modules, one bridge edge.
        for i in 0..5 {
            edges.push(crate::types::EdgeEntry {
                source: format!("auth.f{i}"), target: format!("auth.f{}", i + 1),
                kind: "calls".into(),
            });
            edges.push(crate::types::EdgeEntry {
                source: format!("pay.f{i}"), target: format!("pay.f{}", i + 1),
                kind: "calls".into(),
            });
        }
        // Bridge
        edges.push(crate::types::EdgeEntry {
            source: "auth.f5".into(), target: "pay.f0".into(), kind: "calls".into(),
        });

        let graph = crate::graph::Graph::from_input(&crate::types::AnalyzerInput {
                nodes: nodes.clone(), edges: edges.clone(),
                k: None, edge_kinds: None, layer_weights: None,
                scope: None, parsed_nodes: None, parsed_edges: None,
                self_edge_ratio: None, projection: None, packages: None,
                semantic_embeddings: None,
                experimental: None,
            });
        let roles: HashMap<String, String> = HashMap::new();

        // Mock eigenvalues/eigenvectors (12 nodes, 2 components).
        let eigenvalues = vec![0.1, 0.5, 1.0];
        let n = graph.n;
        let eigenvectors: Vec<Vec<f64>> = (0..n)
            .map(|i| vec![
                if i < 6 { 0.5 } else { -0.5 },
                if i % 2 == 0 { 0.3 } else { -0.3 },
                0.1,
            ])
            .collect();
        let component_node_ids: Vec<String> = graph.node_ids.clone();

        (embeddings, graph, clusters, eigenvalues, eigenvectors, roles, component_node_ids)
    }

    #[test]
    fn test_analyze_semantic_integration() {
        let (emb, graph, clusters, eigenvalues, eigenvectors, roles, comp_ids) =
            make_semantic_test_data();

        let result = analyze_semantic(
            &emb, &graph, &clusters, &eigenvalues, &eigenvectors, &roles, &comp_ids, false,
        );

        // Quality gate should pass — two clear clusters.
        assert!(result.gate_passed, "quality gate should pass for well-separated clusters");

        // Module coherence should be computed for both modules (6+ members each).
        assert!(result.module_coherence.contains_key(&0), "module 0 should have coherence");
        assert!(result.module_coherence.contains_key(&1), "module 1 should have coherence");
        assert!(*result.module_coherence.get(&0).unwrap() > 0.8,
            "auth module should be highly coherent, got {}", result.module_coherence[&0]);

        // Rayleigh quotient should be low (semantics match structure).
        assert!(result.smoothness < 1.0,
            "smoothness should be low for well-organized code, got {}", result.smoothness);

        // AMI should be positive (structural and semantic clusters agree).
        assert!(result.ami > 0.0, "AMI should be positive, got {}", result.ami);

        // No misplaced concerns expected (each node matches its module).
        assert!(result.misplaced_concerns.is_empty(),
            "no misplaced concerns expected, got {}", result.misplaced_concerns.len());

        // No incoherent modules expected.
        assert!(result.incoherent_modules.is_empty(),
            "no incoherent modules expected, got {}", result.incoherent_modules.len());

        // GFT energy should be non-empty.
        assert!(!result.energy_eigenvalues.is_empty(), "GFT eigenvalues should be non-empty");
        assert!(!result.energy_values.is_empty(), "GFT energy values should be non-empty");

        // Local variation should exist for nodes with neighbors.
        assert!(!result.local_variation.is_empty(), "local variation should be non-empty");
    }

    #[test]
    fn test_rayleigh_quotient_known_graph() {
        // Path graph A-B-C with uniform weights.
        // Same embedding on all nodes → for the normalized Laplacian, f=const
        // is NOT the zero-eigenvalue eigenvector (that's D^{1/2} * 1).
        // So we test a weaker property: constant signal should have LOW smoothness
        // and varied signal should have HIGH smoothness (relative comparison).
        let nodes: Vec<crate::types::NodeEntry> = (0..3).map(|i| crate::types::NodeEntry {
            id: format!("n{i}"), kind: "function".into(), file: None, line: None, line_end: None,
        }).collect();
        let edges = vec![
            crate::types::EdgeEntry { source: "n0".into(), target: "n1".into(), kind: "calls".into() },
            crate::types::EdgeEntry { source: "n1".into(), target: "n2".into(), kind: "calls".into() },
        ];
        let graph = crate::graph::Graph::from_input(&crate::types::AnalyzerInput {
                nodes: nodes.clone(), edges: edges.clone(),
                k: None, edge_kinds: None, layer_weights: None,
                scope: None, parsed_nodes: None, parsed_edges: None,
                self_edge_ratio: None, projection: None, packages: None,
                semantic_embeddings: None,
                experimental: None,
            });

        // Constant signal: all same embedding.
        let mut emb_const: HashMap<&str, Vec<f64>> = HashMap::new();
        for i in 0..3 {
            emb_const.insert(Box::leak(format!("n{i}").into_boxed_str()), vec![1.0, 0.0]);
        }
        let rq_const = rayleigh_quotient(&emb_const, &graph);

        // Varied signal: alternating embeddings.
        let mut emb_varied: HashMap<&str, Vec<f64>> = HashMap::new();
        emb_varied.insert(Box::leak("n0".to_string().into_boxed_str()), vec![1.0, 0.0]);
        emb_varied.insert(Box::leak("n1".to_string().into_boxed_str()), vec![0.0, 1.0]);
        emb_varied.insert(Box::leak("n2".to_string().into_boxed_str()), vec![1.0, 0.0]);
        let rq_varied = rayleigh_quotient(&emb_varied, &graph);

        // Varied signal should have strictly higher Rayleigh quotient than constant.
        assert!(rq_varied > rq_const,
            "varied signal ({rq_varied}) should have higher RQ than constant ({rq_const})");
    }

    #[test]
    fn test_rayleigh_quotient_high_variation() {
        // Path graph: A-B-C-D, embeddings alternate [1,0,0] and [0,1,0].
        // Adjacent nodes are maximally different → high smoothness value.
        let mut emb: HashMap<&str, Vec<f64>> = HashMap::new();
        let nodes: Vec<crate::types::NodeEntry> = (0..4).map(|i| crate::types::NodeEntry {
            id: format!("n{i}"), kind: "function".into(), file: None, line: None, line_end: None,
        }).collect();
        let edges = vec![
            crate::types::EdgeEntry { source: "n0".into(), target: "n1".into(), kind: "calls".into() },
            crate::types::EdgeEntry { source: "n1".into(), target: "n2".into(), kind: "calls".into() },
            crate::types::EdgeEntry { source: "n2".into(), target: "n3".into(), kind: "calls".into() },
        ];
        let graph = crate::graph::Graph::from_input(&crate::types::AnalyzerInput {
                nodes: nodes.clone(), edges: edges.clone(),
                k: None, edge_kinds: None, layer_weights: None,
                scope: None, parsed_nodes: None, parsed_edges: None,
                self_edge_ratio: None, projection: None, packages: None,
                semantic_embeddings: None,
                experimental: None,
            });

        emb.insert(Box::leak("n0".to_string().into_boxed_str()), vec![1.0, 0.0]);
        emb.insert(Box::leak("n1".to_string().into_boxed_str()), vec![0.0, 1.0]);
        emb.insert(Box::leak("n2".to_string().into_boxed_str()), vec![1.0, 0.0]);
        emb.insert(Box::leak("n3".to_string().into_boxed_str()), vec![0.0, 1.0]);

        let rq = rayleigh_quotient(&emb, &graph);
        assert!(rq > 0.5,
            "alternating signal should have high Rayleigh quotient, got {rq}");
    }

    #[test]
    fn test_local_variation_uniform() {
        // All nodes have the same embedding → local variation should be 0.
        let mut emb: HashMap<&str, Vec<f64>> = HashMap::new();
        let nodes: Vec<crate::types::NodeEntry> = (0..4).map(|i| crate::types::NodeEntry {
            id: format!("n{i}"), kind: "function".into(), file: None, line: None, line_end: None,
        }).collect();
        let edges = vec![
            crate::types::EdgeEntry { source: "n0".into(), target: "n1".into(), kind: "calls".into() },
            crate::types::EdgeEntry { source: "n1".into(), target: "n2".into(), kind: "calls".into() },
        ];
        let graph = crate::graph::Graph::from_input(&crate::types::AnalyzerInput {
                nodes: nodes.clone(), edges: edges.clone(),
                k: None, edge_kinds: None, layer_weights: None,
                scope: None, parsed_nodes: None, parsed_edges: None,
                self_edge_ratio: None, projection: None, packages: None,
                semantic_embeddings: None,
                experimental: None,
            });

        for i in 0..4 {
            emb.insert(Box::leak(format!("n{i}").into_boxed_str()), vec![1.0, 0.0, 0.0]);
        }

        let sym_nbrs = symmetric_neighbors(&graph);
        let lv = compute_local_variation(&emb, &graph, &sym_nbrs);
        for (nid, &val) in &lv {
            assert!(val.abs() < 1e-10,
                "uniform embeddings should have zero local variation, got {val} for {nid}");
        }
    }

    #[test]
    fn test_local_variation_outlier() {
        // n0 and n1 agree, n2 is different. n1->n2 should have high variation.
        let mut emb: HashMap<&str, Vec<f64>> = HashMap::new();
        let nodes: Vec<crate::types::NodeEntry> = (0..3).map(|i| crate::types::NodeEntry {
            id: format!("n{i}"), kind: "function".into(), file: None, line: None, line_end: None,
        }).collect();
        let edges = vec![
            crate::types::EdgeEntry { source: "n0".into(), target: "n1".into(), kind: "calls".into() },
            crate::types::EdgeEntry { source: "n1".into(), target: "n2".into(), kind: "calls".into() },
        ];
        let graph = crate::graph::Graph::from_input(&crate::types::AnalyzerInput {
                nodes: nodes.clone(), edges: edges.clone(),
                k: None, edge_kinds: None, layer_weights: None,
                scope: None, parsed_nodes: None, parsed_edges: None,
                self_edge_ratio: None, projection: None, packages: None,
                semantic_embeddings: None,
                experimental: None,
            });

        emb.insert(Box::leak("n0".to_string().into_boxed_str()), vec![1.0, 0.0]);
        emb.insert(Box::leak("n1".to_string().into_boxed_str()), vec![1.0, 0.0]);
        emb.insert(Box::leak("n2".to_string().into_boxed_str()), vec![0.0, 1.0]);

        let sym_nbrs = symmetric_neighbors(&graph);
        let lv = compute_local_variation(&emb, &graph, &sym_nbrs);
        // n1 has neighbor n2 which is orthogonal → high variation
        let n1_var = lv.get("n1").copied().unwrap_or(0.0);
        // With symmetric neighbors: n1 sees n0 (same direction, dist=0) + n2 (orthogonal, dist=1).
        // Variation = (1*1.0 + 1*0.0) / 2 = 0.5.
        assert!(n1_var >= 0.4, "n1 should have notable local variation (neighbor n2 is different), got {n1_var}");
    }

    #[test]
    fn test_signal_quality_gate_passes_for_clustered_data() {
        // Two well-separated clusters should pass the gate.
        let mut emb: HashMap<&str, Vec<f64>> = HashMap::new();
        let mut clusters: HashMap<String, usize> = HashMap::new();
        for i in 0..8 {
            let id = format!("a.f{i}");
            emb.insert(Box::leak(id.clone().into_boxed_str()),
                vec![1.0 + i as f64 * 0.01, 0.0, 0.0]);
            clusters.insert(id, 0);
        }
        for i in 0..8 {
            let id = format!("b.f{i}");
            emb.insert(Box::leak(id.clone().into_boxed_str()),
                vec![0.0, 1.0 + i as f64 * 0.01, 0.0]);
            clusters.insert(id, 1);
        }

        assert!(signal_quality_gate(&emb, &clusters),
            "quality gate should pass for well-separated clusters");
    }

    #[test]
    fn test_signal_quality_gate_fails_for_uniform_data() {
        // All identical embeddings → no discriminative power → gate fails.
        let mut emb: HashMap<&str, Vec<f64>> = HashMap::new();
        let mut clusters: HashMap<String, usize> = HashMap::new();
        for i in 0..10 {
            let id = format!("x.f{i}");
            emb.insert(Box::leak(id.clone().into_boxed_str()), vec![1.0, 0.0, 0.0]);
            clusters.insert(id, i % 2);
        }

        assert!(!signal_quality_gate(&emb, &clusters),
            "quality gate should fail for identical embeddings");
    }

    #[test]
    fn test_detect_misplaced_concern_basic() {
        // Node "pay.auth_check" has auth-like embedding but sits in pay module.
        let mut emb: HashMap<&str, Vec<f64>> = HashMap::new();
        let mut clusters: HashMap<String, usize> = HashMap::new();

        // Auth module (6 members): embeddings around [1, 0]
        for i in 0..6 {
            let id = format!("auth.f{i}");
            emb.insert(Box::leak(id.clone().into_boxed_str()), vec![1.0, 0.05 * i as f64]);
            clusters.insert(id, 0);
        }
        // Pay module (6 members): embeddings around [0, 1]
        for i in 0..6 {
            let id = format!("pay.f{i}");
            emb.insert(Box::leak(id.clone().into_boxed_str()), vec![0.05 * i as f64, 1.0]);
            clusters.insert(id, 1);
        }
        // Misplaced node: in pay module (1) but has auth-like embedding [0.9, 0.1]
        emb.insert(Box::leak("pay.auth_check".to_string().into_boxed_str()), vec![0.9, 0.1]);
        clusters.insert("pay.auth_check".into(), 1);

        let module_coherence = compute_module_coherence(&emb, &clusters);
        let roles = HashMap::new();

        // Build a graph with edges to the target module.
        let mut nodes: Vec<crate::types::NodeEntry> = clusters.keys().map(|id| {
            crate::types::NodeEntry {
                id: id.clone(), kind: "function".into(), file: None, line: None, line_end: None,
            }
        }).collect();
        nodes.sort_by(|a, b| a.id.cmp(&b.id));
        let mut edges = vec![
            // pay.auth_check has edge to auth module (required by edge evidence filter)
            crate::types::EdgeEntry {
                source: "pay.auth_check".into(), target: "auth.f0".into(), kind: "calls".into(),
            },
        ];
        // Add intra-module edges for structure
        for i in 0..5 {
            edges.push(crate::types::EdgeEntry {
                source: format!("auth.f{i}"), target: format!("auth.f{}", i+1), kind: "calls".into(),
            });
            edges.push(crate::types::EdgeEntry {
                source: format!("pay.f{i}"), target: format!("pay.f{}", i+1), kind: "calls".into(),
            });
        }
        let graph = crate::graph::Graph::from_input(&crate::types::AnalyzerInput {
                nodes: nodes.clone(), edges: edges.clone(),
                k: None, edge_kinds: None, layer_weights: None,
                scope: None, parsed_nodes: None, parsed_edges: None,
                self_edge_ratio: None, projection: None, packages: None,
                semantic_embeddings: None,
                experimental: None,
            });

        let sym_nbrs = symmetric_neighbors(&graph);
        let (null_threshold, _null_std) = compute_null_coherence_threshold(&emb, &clusters);
        let concerns = detect_misplaced_concerns(&emb, &clusters, &module_coherence, &roles, &graph, &sym_nbrs, null_threshold);

        // Should detect pay.auth_check as misplaced.
        let found = concerns.iter().any(|mc| mc.node_id == "pay.auth_check");
        assert!(found,
            "should detect pay.auth_check as misplaced concern, found {} concerns: {:?}",
            concerns.len(),
            concerns.iter().map(|mc| &mc.node_id).collect::<Vec<_>>()
        );
    }
}
