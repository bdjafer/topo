//! Dataset discovery and loading from benchmark/datasets/.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use topo_analyzer::types::AnalyzerInput;

use crate::types::*;

/// Find the default benchmark/datasets/ path by walking up from CWD.
pub fn default_dataset_root() -> Result<PathBuf> {
    let mut dir = std::env::current_dir()?;
    loop {
        let candidate = dir.join("benchmark/datasets");
        if candidate.is_dir() {
            return Ok(candidate);
        }
        if !dir.pop() {
            bail!("Could not find benchmark/datasets/ in any parent directory");
        }
    }
}

/// Discover all case directories for a dimension, filtered by split.
pub fn discover_cases(
    dimension: Dimension,
    split: &str,
    dataset_root: &Path,
) -> Result<Vec<PathBuf>> {
    let dim_dir = dataset_root.join(dimension.dataset_dir());
    if !dim_dir.is_dir() {
        return Ok(vec![]);
    }

    let mut cases = Vec::new();
    let entries = std::fs::read_dir(&dim_dir)
        .with_context(|| format!("reading {}", dim_dir.display()))?;

    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        // Skip hidden directories (e.g. _clean).
        let name = path.file_name().unwrap().to_string_lossy();
        if name.starts_with('.') {
            continue;
        }

        let meta_path = path.join("metadata.json");
        if meta_path.exists() {
            let meta: CaseMetadata = load_json(&meta_path)?;
            if meta.split == split {
                cases.push(path);
            }
        } else {
            // No metadata — include by default for the "public" split.
            if split == "public" {
                cases.push(path);
            }
        }
    }
    cases.sort();
    Ok(cases)
}

/// Load a JSON file and deserialize.
pub fn load_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

/// Load a graph JSON file into an AnalyzerInput.
pub fn load_graph(path: &Path) -> Result<AnalyzerInput> {
    load_json(path)
}

/// Load an architecture recovery case.
pub fn load_architecture_case(case_dir: &Path) -> Result<ArchitectureCase> {
    let case_id = case_dir
        .file_name()
        .unwrap()
        .to_string_lossy()
        .to_string();
    let graph = load_graph(&case_dir.join("graph.json"))?;
    let labels = load_json(&case_dir.join("labels.json"))?;
    let metadata = load_metadata(case_dir)?;
    Ok(ArchitectureCase { case_id, graph, labels, metadata })
}

/// Load a mutation ranking case.
pub fn load_mutation_case(case_dir: &Path) -> Result<MutationCase> {
    let case_id = case_dir
        .file_name()
        .unwrap()
        .to_string_lossy()
        .to_string();
    let expectations: MutationExpectations = load_json(&case_dir.join("expectations.json"))?;
    let metadata = load_metadata(case_dir)?;

    let variants_dir = case_dir.join("variants");
    let mut variants = HashMap::new();
    if variants_dir.is_dir() {
        for entry in std::fs::read_dir(&variants_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "json") {
                let name = path.file_stem().unwrap().to_string_lossy().to_string();
                let graph = load_graph(&path)?;
                variants.insert(name, graph);
            }
        }
    }

    Ok(MutationCase { case_id, variants, expectations, metadata })
}

/// Load a stability case.
pub fn load_stability_case(case_dir: &Path) -> Result<StabilityCase> {
    let case_id = case_dir
        .file_name()
        .unwrap()
        .to_string_lossy()
        .to_string();
    let base_graph = load_graph(&case_dir.join("base_graph.json"))?;
    let metadata = load_metadata(case_dir)?;

    let mut perturbations = HashMap::new();
    let pert_dir = case_dir.join("perturbations");
    if pert_dir.is_dir() {
        for entry in std::fs::read_dir(&pert_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "json") {
                let name = path.file_stem().unwrap().to_string_lossy().to_string();
                // Skip node_mapping files.
                if name.contains("node_mapping") {
                    continue;
                }
                let graph = load_graph(&path)?;
                perturbations.insert(name, graph);
            }
        }
    }

    // Load node mappings if present (check both case root and perturbations/).
    let mut node_mappings: NodeMapping = HashMap::new();
    let mapping_path = case_dir.join("node_mapping.json");
    let alt_mapping_path = case_dir.join("perturbations/node_mapping.json");
    if mapping_path.exists() {
        node_mappings = load_json(&mapping_path)?;
    } else if alt_mapping_path.exists() {
        node_mappings = load_json(&alt_mapping_path)?;
    }

    Ok(StabilityCase { case_id, base_graph, perturbations, node_mappings, metadata })
}

/// Load an anomaly case. Returns None gold if no gold.json (clean/FP test).
pub fn load_anomaly_case(case_dir: &Path) -> Result<AnomalyCase> {
    let case_id = case_dir
        .file_name()
        .unwrap()
        .to_string_lossy()
        .to_string();
    let graph = load_graph(&case_dir.join("graph.json"))?;
    let metadata = load_metadata(case_dir)?;
    let gold_path = case_dir.join("gold.json");
    let gold = if gold_path.exists() {
        Some(load_json(&gold_path)?)
    } else {
        None
    };
    Ok(AnomalyCase { case_id, graph, gold, metadata })
}

fn load_metadata(case_dir: &Path) -> Result<CaseMetadata> {
    let meta_path = case_dir.join("metadata.json");
    if meta_path.exists() {
        load_json(&meta_path)
    } else {
        Ok(CaseMetadata {
            split: "public".to_string(),
            level: None,
            description: None,
            label_provenance: None,
            perturbation_families: None,
        })
    }
}
