//! Shared types for the benchmark harness.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const RUNNER_VERSION: &str = "1.0.0";

// ── Constants ────────────────────────────────────────────────────────────────

/// Default minimum margin for pairwise comparisons.
pub const DEFAULT_MARGIN: f64 = 0.05;
/// Coverage floor for architecture recovery.
pub const COVERAGE_FLOOR: f64 = 0.80;
/// IoU threshold for anomaly matching.
pub const IOU_THRESHOLD: f64 = 0.3;
/// High-confidence severity threshold for false-positive checks.
pub const HIGH_SEVERITY_THRESHOLD: f64 = 0.7;
/// Maximum anomaly flood ratio (candidate / reference).
pub const ANOMALY_FLOOD_RATIO: f64 = 3.0;
/// Maximum allowed dimension regression (absolute).
pub const MAX_REGRESSION: f64 = 0.03;

// ── Partition alias ──────────────────────────────────────────────────────────

/// Module partition: node_id → cluster label.
pub type Partition = HashMap<String, usize>;

// ── Dimension enum ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dimension {
    ArchitectureRecovery,
    MutationRanking,
    Stability,
    SeededAnomalyDetection,
}

impl Dimension {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ArchitectureRecovery => "architecture_recovery",
            Self::MutationRanking => "mutation_ranking",
            Self::Stability => "stability",
            Self::SeededAnomalyDetection => "seeded_anomaly_detection",
        }
    }

    pub fn all() -> &'static [Dimension] {
        &[
            Self::ArchitectureRecovery,
            Self::MutationRanking,
            Self::Stability,
            Self::SeededAnomalyDetection,
        ]
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "architecture" | "architecture_recovery" => Some(Self::ArchitectureRecovery),
            "mutations" | "mutation_ranking" => Some(Self::MutationRanking),
            "stability" => Some(Self::Stability),
            "anomalies" | "seeded_anomaly_detection" => Some(Self::SeededAnomalyDetection),
            _ => None,
        }
    }

    /// Subdirectory name under `benchmark/datasets/`.
    pub fn dataset_dir(&self) -> &'static str {
        match self {
            Self::ArchitectureRecovery => "architecture",
            Self::MutationRanking => "mutations",
            Self::Stability => "stability",
            Self::SeededAnomalyDetection => "anomalies",
        }
    }
}

// ── Dataset loading types ────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct CaseMetadata {
    pub split: String,
    #[serde(default)]
    pub level: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub label_provenance: Option<String>,
    #[serde(default)]
    pub perturbation_families: Option<Vec<String>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ArchitectureLabels {
    pub analysis_level: String,
    pub included_nodes: HashMap<String, String>,
    #[serde(default)]
    pub excluded_nodes: Vec<String>,
}

fn default_margin() -> f64 {
    DEFAULT_MARGIN
}

#[derive(Debug, Clone, Deserialize)]
pub struct MutationExpectation {
    #[serde(default)]
    pub variants: Option<Vec<String>>,
    pub signal: String,
    #[serde(default)]
    pub direction: Option<String>,
    #[serde(default)]
    pub expect: Option<bool>,
    #[serde(default = "default_margin")]
    pub margin: f64,
    #[serde(default)]
    pub signal_args: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MutatedRegion {
    pub nodes: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MutationExpectations {
    pub ordering: Vec<Vec<String>>,
    pub required_expectations: Vec<MutationExpectation>,
    #[serde(default)]
    pub mutated_region: Option<MutatedRegion>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GoldAnomaly {
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub region_nodes: Option<Vec<String>>,
    #[serde(default)]
    pub region: Option<MutatedRegion>,
}

impl GoldAnomaly {
    pub fn node_set(&self) -> std::collections::HashSet<String> {
        if let Some(nodes) = &self.region_nodes {
            nodes.iter().cloned().collect()
        } else if let Some(region) = &self.region {
            region.nodes.iter().cloned().collect()
        } else {
            std::collections::HashSet::new()
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct AnomalyGold {
    pub anomalies: Vec<GoldAnomaly>,
}

/// Node mapping for stability: perturbation_name → { base_id → pert_id }.
pub type NodeMapping = HashMap<String, HashMap<String, String>>;

// ── Loaded case bundles ──────────────────────────────────────────────────────

pub struct ArchitectureCase {
    pub case_id: String,
    pub graph: topo_analyzer::types::AnalyzerInput,
    pub labels: ArchitectureLabels,
    pub metadata: CaseMetadata,
}

pub struct MutationCase {
    pub case_id: String,
    /// variant_name → graph (e.g. "clean", "mutated", "repaired")
    pub variants: HashMap<String, topo_analyzer::types::AnalyzerInput>,
    pub expectations: MutationExpectations,
    pub metadata: CaseMetadata,
}

pub struct StabilityCase {
    pub case_id: String,
    pub base_graph: topo_analyzer::types::AnalyzerInput,
    /// perturbation_name → graph
    pub perturbations: HashMap<String, topo_analyzer::types::AnalyzerInput>,
    pub node_mappings: NodeMapping,
    pub metadata: CaseMetadata,
}

pub struct AnomalyCase {
    pub case_id: String,
    pub graph: topo_analyzer::types::AnalyzerInput,
    /// None means this is a clean/false-positive test graph.
    pub gold: Option<AnomalyGold>,
    pub metadata: CaseMetadata,
}

// ── Result types ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct ArchitectureCaseResult {
    pub case_id: String,
    pub nmi: f64,
    pub boundary_f1: f64,
    pub coverage: f64,
    pub score: f64,
    pub guardrails: ArchGuardrails,
    pub baseline_nmi: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchGuardrails {
    pub coverage_ok: bool,
    pub baseline_ok: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct MutationCaseResult {
    pub case_id: String,
    pub pairwise_accuracy: f64,
    pub repair_accuracy: Option<f64>,
    pub attribution_at_3: f64,
    pub score: f64,
    pub expectation_details: Vec<ExpectationResult>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ExpectationResult {
    pub signal: String,
    pub direction: Option<String>,
    pub passed: bool,
    pub left_value: Option<f64>,
    pub right_value: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StabilityCaseResult {
    pub case_id: String,
    pub partition_stability: f64,
    pub role_stability: f64,
    pub score: f64,
    pub per_perturbation: HashMap<String, PerturbationResult>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PerturbationResult {
    pub partition_ari: f64,
    pub role_f1: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct AnomalyCaseResult {
    pub case_id: String,
    pub average_precision: f64,
    pub precision_at_3: f64,
    pub score: f64,
    pub n_predicted: usize,
    pub n_gold: usize,
    pub is_clean_graph: bool,
}

// ── Scorecard ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scorecard {
    pub runner_version: String,
    pub overall_primary: f64,
    pub dimensions: HashMap<String, f64>,
    pub guardrails: Guardrails,
    pub cases_passed: usize,
    pub cases_total: usize,
    pub failing_cases: Vec<String>,
    pub promotion_decision: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Guardrails {
    pub coverage_ok: bool,
    pub baseline_ok: bool,
    pub no_regressions: bool,
    pub false_positive_ok: bool,
    pub no_anomaly_flood: bool,
}

impl Guardrails {
    pub fn all_pass(&self) -> bool {
        self.coverage_ok
            && self.baseline_ok
            && self.no_regressions
            && self.false_positive_ok
            && self.no_anomaly_flood
    }
}

// ── Comparison ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct ComparisonResult {
    pub overall_delta: f64,
    pub candidate_overall: f64,
    pub reference_overall: f64,
    pub dimensions: HashMap<String, DimensionDelta>,
    pub promotion_decision: String,
    pub reasons: ComparisonReasons,
}

#[derive(Debug, Clone, Serialize)]
pub struct DimensionDelta {
    pub candidate: f64,
    pub reference: f64,
    pub delta: f64,
    pub regressed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ComparisonReasons {
    pub overall_improved: bool,
    pub no_regressions: bool,
    pub guardrails_pass: bool,
}

// ── Baseline types ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct BaselineResult {
    pub name: String,
    pub partition: Partition,
}

#[derive(Debug, Clone, Serialize)]
pub struct BaselineAnomaly {
    pub kind: String,
    pub node_ids: Vec<String>,
    pub severity: f64,
}

// ── Signal value ─────────────────────────────────────────────────────────────

/// A signal extracted from analysis output.
#[derive(Debug, Clone)]
pub enum SignalValue {
    Float(f64),
    Bool(bool),
    Count(usize),
    Role(Option<String>),
}

impl SignalValue {
    pub fn as_f64(&self) -> f64 {
        match self {
            Self::Float(v) => *v,
            Self::Bool(b) => if *b { 1.0 } else { 0.0 },
            Self::Count(c) => *c as f64,
            Self::Role(_) => 0.0,
        }
    }

    pub fn as_bool(&self) -> bool {
        match self {
            Self::Bool(b) => *b,
            Self::Float(v) => *v > 0.0,
            Self::Count(c) => *c > 0,
            Self::Role(r) => r.is_some(),
        }
    }
}

// ── Benchmark run ────────────────────────────────────────────────────────────

pub struct BenchmarkRun {
    pub scorecard: Scorecard,
    pub dimension_details: HashMap<String, serde_json::Value>,
    pub per_case_lines: Vec<serde_json::Value>,
    pub baseline_results: HashMap<String, serde_json::Value>,
}
