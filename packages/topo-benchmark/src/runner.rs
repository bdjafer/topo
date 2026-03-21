//! Orchestrates a full benchmark run across dimensions.

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use topo_analyzer::types::{AnalysisOutput, AnalyzerInput, ScopeInput};

use crate::baselines;
use crate::datasets;
use crate::dimensions;
use crate::scorecard;
use crate::types::*;

/// Run the full benchmark.
pub fn run_benchmark(
    dims: &[Dimension],
    split: &str,
    dataset_root: &Path,
    edge_kind: &str,
) -> Result<BenchmarkRun> {
    let mut dimension_scores: HashMap<String, f64> = HashMap::new();
    let mut dimension_details: HashMap<String, serde_json::Value> = HashMap::new();
    let mut per_case_lines: Vec<serde_json::Value> = Vec::new();
    let mut baseline_results: HashMap<String, serde_json::Value> = HashMap::new();

    let mut all_cases_total = 0usize;
    let mut all_cases_passed = 0usize;
    let mut all_failing: Vec<String> = Vec::new();

    let mut coverage_ok = true;
    let mut baseline_ok = true;
    let mut false_positive_ok = true;

    for &dim in dims {
        match dim {
            Dimension::ArchitectureRecovery => {
                let (score, details, cases, baselines_out) =
                    run_architecture(split, dataset_root, edge_kind)?;
                let arch_guardrails: ArchGuardrails =
                    serde_json::from_value(details["guardrails"].clone()).unwrap_or(ArchGuardrails {
                        coverage_ok: true,
                        baseline_ok: true,
                    });
                coverage_ok = coverage_ok && arch_guardrails.coverage_ok;
                baseline_ok = baseline_ok && arch_guardrails.baseline_ok;
                dimension_scores.insert(dim.as_str().to_string(), score);
                dimension_details.insert(dim.as_str().to_string(), details);
                for (name, val) in baselines_out {
                    baseline_results.insert(name, val);
                }
                for case in &cases {
                    all_cases_total += 1;
                    let passed = case["score"].as_f64().unwrap_or(0.0) > 0.0;
                    if passed {
                        all_cases_passed += 1;
                    } else {
                        all_failing.push(case["case_id"].as_str().unwrap_or("?").to_string());
                    }
                    per_case_lines.push(case.clone());
                }
            }
            Dimension::MutationRanking => {
                let (score, details, cases) =
                    run_mutations(split, dataset_root, edge_kind)?;
                dimension_scores.insert(dim.as_str().to_string(), score);
                dimension_details.insert(dim.as_str().to_string(), details);
                for case in &cases {
                    all_cases_total += 1;
                    let pa = case["pairwise_accuracy"].as_f64().unwrap_or(0.0);
                    if pa >= 1.0 {
                        all_cases_passed += 1;
                    } else {
                        all_failing.push(case["case_id"].as_str().unwrap_or("?").to_string());
                    }
                    per_case_lines.push(case.clone());
                }
            }
            Dimension::Stability => {
                let (score, details, cases) =
                    run_stability(split, dataset_root, edge_kind)?;
                dimension_scores.insert(dim.as_str().to_string(), score);
                dimension_details.insert(dim.as_str().to_string(), details);
                for case in &cases {
                    all_cases_total += 1;
                    let s = case["score"].as_f64().unwrap_or(0.0);
                    if s > 0.5 {
                        all_cases_passed += 1;
                    } else {
                        all_failing.push(case["case_id"].as_str().unwrap_or("?").to_string());
                    }
                    per_case_lines.push(case.clone());
                }
            }
            Dimension::SeededAnomalyDetection => {
                let (score, details, cases, fp_ok) =
                    run_anomalies(split, dataset_root, edge_kind)?;
                false_positive_ok = false_positive_ok && fp_ok;
                dimension_scores.insert(dim.as_str().to_string(), score);
                dimension_details.insert(dim.as_str().to_string(), details);
                for case in &cases {
                    all_cases_total += 1;
                    let s = case["score"].as_f64().unwrap_or(0.0);
                    if s > 0.0 {
                        all_cases_passed += 1;
                    } else if !case["is_clean_graph"].as_bool().unwrap_or(false) {
                        all_failing.push(case["case_id"].as_str().unwrap_or("?").to_string());
                    }
                    per_case_lines.push(case.clone());
                }
            }
        }
    }

    let guardrails = Guardrails {
        coverage_ok,
        baseline_ok,
        no_regressions: true, // Only checked during compare.
        false_positive_ok,
        no_anomaly_flood: true, // Only checked during compare.
    };

    let sc = scorecard::build_scorecard(
        &dimension_scores,
        guardrails,
        all_cases_passed,
        all_cases_total,
        all_failing,
    );

    Ok(BenchmarkRun {
        scorecard: sc,
        dimension_details,
        per_case_lines,
        baseline_results,
    })
}

/// Write benchmark run artifacts to disk.
pub fn write_artifacts(run: &BenchmarkRun, output_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(output_dir)?;

    // scorecard.json
    scorecard::save_scorecard(&run.scorecard, &output_dir.join("scorecard.json"))?;

    // per_case.jsonl
    let mut lines = String::new();
    for line in &run.per_case_lines {
        lines.push_str(&serde_json::to_string(line)?);
        lines.push('\n');
    }
    std::fs::write(output_dir.join("per_case.jsonl"), lines)?;

    // baselines/
    let baselines_dir = output_dir.join("baselines");
    std::fs::create_dir_all(&baselines_dir)?;
    for (name, val) in &run.baseline_results {
        let path = baselines_dir.join(format!("{name}.json"));
        std::fs::write(path, serde_json::to_string_pretty(val)?)?;
    }

    // summary.md
    let report = crate::report::generate_summary(
        &run.scorecard,
        &run.dimension_details,
        &run.baseline_results,
        None,
    );
    std::fs::write(output_dir.join("summary.md"), report)?;

    Ok(())
}

// ── Dimension runners ────────────────────────────────────────────────────────

fn run_architecture(
    split: &str,
    dataset_root: &Path,
    edge_kind: &str,
) -> Result<(f64, serde_json::Value, Vec<serde_json::Value>, Vec<(String, serde_json::Value)>)> {
    let case_dirs = datasets::discover_cases(Dimension::ArchitectureRecovery, split, dataset_root)?;
    let mut results = Vec::new();
    let mut case_jsons = Vec::new();
    let mut baselines_out = Vec::new();

    for case_dir in &case_dirs {
        let case = datasets::load_architecture_case(case_dir)?;
        let level = case.labels.analysis_level.as_str();
        let output = analyze_graph(&case.graph, level, edge_kind);

        // Baselines.
        let dir_part = baselines::directory_partition(&case.graph);
        let louv_part = baselines::louvain_partition(&case.graph, 42);
        let mut baseline_partitions = HashMap::new();
        baseline_partitions.insert("directory".to_string(), dir_part.clone());
        baseline_partitions.insert("louvain".to_string(), louv_part.clone());

        let mut result =
            dimensions::architecture::score_architecture_case(&case.graph, &case.labels, &output, &baseline_partitions);
        result.case_id = case.case_id.clone();

        let case_json = serde_json::to_value(&result)?;
        case_jsons.push(case_json);
        results.push(result);

        baselines_out.push((
            "directory".to_string(),
            serde_json::to_value(&BaselineResult {
                name: "directory".to_string(),
                partition: dir_part,
            })?,
        ));
        baselines_out.push((
            "louvain".to_string(),
            serde_json::to_value(&BaselineResult {
                name: "louvain".to_string(),
                partition: louv_part,
            })?,
        ));
    }

    let (score, guardrails) = dimensions::architecture::aggregate_architecture_scores(&results);
    let details = serde_json::json!({
        "score": score,
        "guardrails": serde_json::to_value(&guardrails)?,
        "cases": results.len(),
    });

    Ok((score, details, case_jsons, baselines_out))
}

fn run_mutations(
    split: &str,
    dataset_root: &Path,
    edge_kind: &str,
) -> Result<(f64, serde_json::Value, Vec<serde_json::Value>)> {
    let case_dirs = datasets::discover_cases(Dimension::MutationRanking, split, dataset_root)?;
    let mut results = Vec::new();
    let mut case_jsons = Vec::new();

    for case_dir in &case_dirs {
        let case = datasets::load_mutation_case(case_dir)?;
        let level = case
            .metadata
            .level
            .as_deref()
            .unwrap_or("module");

        // Analyze each variant.
        let mut analyses = HashMap::new();
        for (variant_name, graph) in &case.variants {
            let output = analyze_graph(graph, level, edge_kind);
            analyses.insert(variant_name.clone(), output);
        }

        let result = dimensions::mutations::score_mutation_case(&case, &analyses);
        let case_json = serde_json::to_value(&result)?;
        case_jsons.push(case_json);
        results.push(result);
    }

    let score = dimensions::mutations::aggregate_mutation_scores(&results);
    let details = serde_json::json!({
        "score": score,
        "cases": results.len(),
    });

    Ok((score, details, case_jsons))
}

fn run_stability(
    split: &str,
    dataset_root: &Path,
    edge_kind: &str,
) -> Result<(f64, serde_json::Value, Vec<serde_json::Value>)> {
    let case_dirs = datasets::discover_cases(Dimension::Stability, split, dataset_root)?;
    let mut results = Vec::new();
    let mut case_jsons = Vec::new();

    for case_dir in &case_dirs {
        let case = datasets::load_stability_case(case_dir)?;
        let level = case
            .metadata
            .level
            .as_deref()
            .unwrap_or("module");

        let base_output = analyze_graph(&case.base_graph, level, edge_kind);

        let mut pert_outputs = HashMap::new();
        for (pert_name, graph) in &case.perturbations {
            let output = analyze_graph(graph, level, edge_kind);
            pert_outputs.insert(pert_name.clone(), output);
        }

        let result =
            dimensions::stability::score_stability_case(&case, &base_output, &pert_outputs);
        let case_json = serde_json::to_value(&result)?;
        case_jsons.push(case_json);
        results.push(result);
    }

    let score = dimensions::stability::aggregate_stability_scores(&results);
    let details = serde_json::json!({
        "score": score,
        "cases": results.len(),
    });

    Ok((score, details, case_jsons))
}

fn run_anomalies(
    split: &str,
    dataset_root: &Path,
    edge_kind: &str,
) -> Result<(f64, serde_json::Value, Vec<serde_json::Value>, bool)> {
    let case_dirs = datasets::discover_cases(Dimension::SeededAnomalyDetection, split, dataset_root)?;
    let mut results = Vec::new();
    let mut case_jsons = Vec::new();
    let mut false_positive_ok = true;

    for case_dir in &case_dirs {
        let case = datasets::load_anomaly_case(case_dir)?;
        let level = case
            .metadata
            .level
            .as_deref()
            .unwrap_or("module");

        let output = analyze_graph(&case.graph, level, edge_kind);

        if case.gold.is_none() {
            // Clean graph — check false positive rate.
            let clean_ok = dimensions::anomalies::score_clean_graph(&output);
            false_positive_ok = false_positive_ok && clean_ok;
            let case_json = serde_json::json!({
                "case_id": case.case_id,
                "is_clean_graph": true,
                "false_positive_ok": clean_ok,
                "score": if clean_ok { 1.0 } else { 0.0 },
                "n_predicted": output.issues.len(),
            });
            case_jsons.push(case_json);
            continue;
        }

        let result = dimensions::anomalies::score_anomaly_case(&case, &output);
        let case_json = serde_json::to_value(&result)?;
        case_jsons.push(case_json);
        results.push(result);
    }

    let score = dimensions::anomalies::aggregate_anomaly_scores(&results);
    let details = serde_json::json!({
        "score": score,
        "cases": results.len(),
        "false_positive_ok": false_positive_ok,
    });

    Ok((score, details, case_jsons, false_positive_ok))
}

// ── Analysis helper ──────────────────────────────────────────────────────────

/// Analyze a graph using topo_analyzer::analyze_full.
fn analyze_graph(input: &AnalyzerInput, level: &str, edge_kind: &str) -> AnalysisOutput {
    let mut augmented = input.clone();

    // Set edge kinds and layer weights based on edge_kind.
    let (edge_kinds, layer_weights) = match edge_kind {
        "calls" => (vec!["calls".to_string()], None),
        "imports" => (vec!["imports".to_string()], None),
        "inherits" => (vec!["inherits".to_string()], None),
        _ => {
            // "combined" — default
            let kinds = vec![
                "calls".to_string(),
                "imports".to_string(),
                "inherits".to_string(),
            ];
            let mut weights = HashMap::new();
            weights.insert("calls".to_string(), 1.0);
            weights.insert("imports".to_string(), 0.5);
            weights.insert("inherits".to_string(), 0.8);
            (kinds, Some(weights))
        }
    };

    augmented.edge_kinds = Some(edge_kinds.clone());
    augmented.layer_weights = layer_weights;
    augmented.scope = Some(ScopeInput {
        level: level.to_string(),
        edge_kinds,
        internal_only: true,
        roots: vec![],
    });

    // Set parsed counts if not already set.
    if augmented.parsed_nodes.is_none() {
        augmented.parsed_nodes = Some(augmented.nodes.len());
    }
    if augmented.parsed_edges.is_none() {
        augmented.parsed_edges = Some(augmented.edges.len());
    }

    topo_analyzer::analyze_full(&augmented)
}
