//! Source text extraction and semantic embedding via fastembed.
//!
//! Reads source files from disk using line/line_end spans from the parsed graph,
//! assembles context windows, and runs jina-embeddings-v2-base-code for embedding.

use std::collections::HashMap;
use std::io::BufRead;
use std::path::Path;

use anyhow::{Context, Result};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use serde_json::Value;

/// Generate semantic embeddings for all graph nodes with source spans.
///
/// Returns a map of node_id -> 768-dim embedding vector.
pub fn generate_embeddings(
    graph_json: &Value,
    project_root: &Path,
) -> Result<HashMap<String, Vec<f32>>> {
    let texts = extract_source_texts(graph_json, project_root);
    if texts.is_empty() {
        anyhow::bail!("No source texts extracted — check that the graph has file/line/line_end spans.");
    }
    eprintln!("  Extracted source text for {} nodes.", texts.len());

    let embeddings = embed_texts(&texts)?;
    eprintln!("  Embedded {} nodes (768-dim jina-embeddings-v2-base-code).", embeddings.len());

    Ok(embeddings)
}

/// Extract source text context windows for each graph node.
///
/// For each node with `file`, `line`, and `line_end`: reads the source file,
/// extracts the relevant lines, and assembles a context window with metadata.
pub fn extract_source_texts(
    graph_json: &Value,
    project_root: &Path,
) -> HashMap<String, String> {
    let mut texts = HashMap::new();
    let mut file_cache: HashMap<String, Vec<String>> = HashMap::new();

    let nodes = match graph_json.get("nodes").and_then(|v| v.as_array()) {
        Some(nodes) => nodes,
        None => return texts,
    };

    for node in nodes {
        let id = match node.get("id").and_then(|v| v.as_str()) {
            Some(id) => id,
            None => continue,
        };
        let file = match node.get("file").and_then(|v| v.as_str()) {
            Some(f) => f,
            None => continue,
        };
        let line = match node.get("line").and_then(|v| v.as_u64()) {
            Some(l) => l as usize,
            None => continue,
        };

        // Read file (cached).
        let file_lines = file_cache
            .entry(file.to_string())
            .or_insert_with(|| read_file_lines(project_root, file));

        if file_lines.is_empty() {
            continue;
        }

        let line_end = node
            .get("line_end")
            .and_then(|v| v.as_u64())
            .map(|l| l as usize);

        // Extract source text.
        let source = if let Some(end) = line_end {
            let start = (line.saturating_sub(1)).min(file_lines.len());
            let end = end.min(file_lines.len());
            if start < end {
                file_lines[start..end].join("\n")
            } else {
                // Fallback: just the line itself.
                file_lines.get(line.saturating_sub(1)).cloned().unwrap_or_default()
            }
        } else {
            // No line_end: use just the definition line.
            file_lines.get(line.saturating_sub(1)).cloned().unwrap_or_default()
        };

        if source.trim().is_empty() {
            continue;
        }

        // Assemble context window per PHASE_2.md spec.
        let context = format!("# module: {id}\n# file: {file}\n\n{source}");

        // Truncate to ~6K tokens (~24K chars) to stay within model's 8K context.
        let max_chars = 24000;
        let truncated = if context.len() > max_chars {
            context.char_indices()
                .nth(max_chars)
                .map(|(i, _)| &context[..i])
                .unwrap_or(&context)
                .to_string()
        } else {
            context
        };

        texts.insert(id.to_string(), truncated);
    }

    texts
}

/// Read a source file into lines, resolving relative to project root.
fn read_file_lines(project_root: &Path, relative_path: &str) -> Vec<String> {
    let full_path = project_root.join(relative_path);
    match std::fs::File::open(&full_path) {
        Ok(file) => {
            let reader = std::io::BufReader::new(file);
            reader.lines().filter_map(|l| l.ok()).collect()
        }
        Err(_) => Vec::new(),
    }
}

/// Batch-embed texts using jina-embeddings-v2-base-code via fastembed.
fn embed_texts(texts: &HashMap<String, String>) -> Result<HashMap<String, Vec<f32>>> {
    let model = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::JinaEmbeddingsV2BaseCode)
            .with_show_download_progress(true),
    )
    .context("Failed to initialize embedding model")?;

    // Collect into ordered vec for batching.
    let ids: Vec<&String> = texts.keys().collect();
    let documents: Vec<&str> = ids.iter().map(|id| texts[*id].as_str()).collect();

    // Batch embed — 10x faster than sequential.
    let embeddings = model
        .embed(documents, None)
        .context("Embedding inference failed")?;

    let mut result = HashMap::with_capacity(ids.len());
    for (id, emb) in ids.into_iter().zip(embeddings.into_iter()) {
        result.insert(id.clone(), emb);
    }

    Ok(result)
}
