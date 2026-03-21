//! Graph assembly and JSON serialization matching graph.schema.json.

use serde::Serialize;

/// A node in the code graph.
#[derive(Debug, Clone, Serialize)]
pub struct Node {
    pub id: String,
    pub kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// An edge in the code graph.
#[derive(Debug, Clone, Serialize)]
pub struct Edge {
    pub source: String,
    pub target: String,
    pub kind: &'static str,
}

/// The full code graph, ready for JSON serialization.
#[derive(Debug, Clone, Serialize)]
pub struct CodeGraph {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}

impl CodeGraph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    pub fn add_node(&mut self, node: Node) {
        self.nodes.push(node);
    }

    pub fn add_edge(&mut self, edge: Edge) {
        self.edges.push(edge);
    }

    /// Check if a node ID exists in the graph.
    pub fn has_node(&self, id: &str) -> bool {
        self.nodes.iter().any(|n| n.id == id)
    }

    /// Serialize to JSON string matching graph.schema.json.
    pub fn to_json(&self) -> anyhow::Result<String> {
        Ok(serde_json::to_string(self)?)
    }

    /// Sort nodes by ID for deterministic output.
    pub fn sort(&mut self) {
        self.nodes.sort_by(|a, b| a.id.cmp(&b.id));
    }
}
