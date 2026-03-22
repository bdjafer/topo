//! Focused formatters for topo structural analysis output.
//!
//! Three formatters, one per capability:
//! - `format_issues` — linter output (issues only)
//! - `format_domain_view` — architecture view (modules, DAG, critical nodes)
//! - `format_health` — health metrics snapshot

use std::collections::HashMap;
use std::path::Path;

use serde_json::Value;

use crate::style::Style;

// ── Shared ──

fn format_header(data: &Value, project_root: Option<&Path>, s: &Style) -> Vec<String> {
    let coverage = data.get("coverage").and_then(|v| v.as_object());

    let root_label = project_root
        .map(|p| p.display().to_string())
        .unwrap_or_default();
    let analyzed_nodes = coverage
        .and_then(|c| c.get("analyzed_nodes"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let analyzed_edges = coverage
        .and_then(|c| c.get("analyzed_edges"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let parsed_nodes = coverage
        .and_then(|c| c.get("parsed_nodes"))
        .and_then(|v| v.as_u64())
        .unwrap_or(analyzed_nodes);

    let semantic_flag = data
        .get("semantic_enabled")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let semantic_str = if semantic_flag {
        " [semantic: on]"
    } else {
        ""
    };

    vec![
        s.bold(&format!("topo — {root_label}")),
        format!(
            "{analyzed_nodes} nodes, {analyzed_edges} edges ({parsed_nodes} symbols parsed){semantic_str}"
        ),
    ]
}

fn section_header(title: &str, s: &Style) -> String {
    let prefix = format!("── {title} ");
    let pad = 60usize.saturating_sub(prefix.len());
    let line = format!("{prefix}{}", "─".repeat(pad));
    s.bold(&line)
}

fn relative_path(file_path: &str, project_root: Option<&Path>) -> String {
    if let Some(root) = project_root {
        if let Ok(rel) = Path::new(file_path).strip_prefix(root) {
            return rel.display().to_string();
        }
    }
    file_path.to_string()
}

// ── Issues Formatter ──

/// Format diagnostic issues as linter output.
/// Returns (formatted_text, active_issue_count).
pub fn format_issues(
    data: &Value,
    ignores: &HashMap<String, String>,
    project_root: Option<&Path>,
    color: bool,
) -> (String, usize) {
    let s = Style::new(color);
    let mut lines = format_header(data, project_root, &s);

    let issues = data
        .get("issues")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let active_issues: Vec<&Value> = issues
        .iter()
        .filter(|i| {
            i.get("id")
                .and_then(|v| v.as_str())
                .map(|id| !ignores.contains_key(id))
                .unwrap_or(true)
        })
        .collect();
    let acknowledged: Vec<&Value> = issues
        .iter()
        .filter(|i| {
            i.get("id")
                .and_then(|v| v.as_str())
                .map(|id| ignores.contains_key(id))
                .unwrap_or(false)
        })
        .collect();
    let issue_count = active_issues.len();

    lines.push(String::new());
    lines.push(section_header(&format!("Issues ({issue_count})"), &s));
    lines.push(String::new());

    if !active_issues.is_empty() {
        for issue in &active_issues {
            let sev_label = issue
                .get("severity_label")
                .and_then(|v| v.as_str())
                .unwrap_or("low");
            let sev_tag = s.severity(sev_label, &format!("[{sev_label}]"));
            let id = issue.get("id").and_then(|v| v.as_str()).unwrap_or("");
            let desc = issue
                .get("description")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            lines.push(format!("  {sev_tag} {}", s.bold(id)));
            lines.push(format!("    {desc}"));

            if let Some(anchors) = issue.get("anchors").and_then(|v| v.as_array()) {
                if let Some(anchor) = anchors.first() {
                    let file = anchor
                        .get("file")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let line_num = anchor.get("line").and_then(|v| v.as_u64()).unwrap_or(0);
                    let path = relative_path(file, project_root);
                    lines.push(format!(
                        "    → {}",
                        s.cyan(&format!("{path}:{line_num}"))
                    ));
                }
            }
            if let Some(suggested) = issue.get("suggested_module").and_then(|v| v.as_str()) {
                let sim_own = issue
                    .get("similarity_own")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                let sim_best = issue
                    .get("similarity_best")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                lines.push(format!(
                    "    suggested module: {suggested} (similarity: {sim_best:.2} vs own: {sim_own:.2})"
                ));
            }
            lines.push(String::new());
        }

        let high = active_issues
            .iter()
            .filter(|i| i.get("severity_label").and_then(|v| v.as_str()) == Some("high"))
            .count();
        let medium = active_issues
            .iter()
            .filter(|i| i.get("severity_label").and_then(|v| v.as_str()) == Some("medium"))
            .count();
        let low = active_issues
            .iter()
            .filter(|i| i.get("severity_label").and_then(|v| v.as_str()) == Some("low"))
            .count();

        let mut count_parts: Vec<String> = Vec::new();
        if high > 0 {
            count_parts.push(s.red_text(&format!("{high} high")));
        }
        if medium > 0 {
            count_parts.push(s.yellow_text(&format!("{medium} medium")));
        }
        if low > 0 {
            count_parts.push(s.dim(&format!("{low} low")));
        }
        let count_str = if count_parts.is_empty() {
            "0".to_string()
        } else {
            count_parts.join(", ")
        };
        lines.push(format!("  ✖ {issue_count} issues ({count_str})"));
    } else {
        lines.push(s.green("  No issues detected."));
    }

    if !acknowledged.is_empty() {
        lines.push(format!("  {} acknowledged", acknowledged.len()));
        lines.push(String::new());
        for issue in &acknowledged {
            let id = issue.get("id").and_then(|v| v.as_str()).unwrap_or("");
            let justification = ignores.get(id).map(|s| s.as_str()).unwrap_or("");
            let sev_label = issue
                .get("severity_label")
                .and_then(|v| v.as_str())
                .unwrap_or("low");
            let sev_tag = s.severity(sev_label, &format!("[{sev_label}]"));
            let desc = issue
                .get("description")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            lines.push(format!(
                "  {sev_tag} {} {}",
                s.dim(id),
                s.dim("(acknowledged)")
            ));
            lines.push(format!("    {}", s.dim(desc)));
            if !justification.is_empty() {
                lines.push(format!(
                    "    {}",
                    s.dim(&format!("Reason: {justification}"))
                ));
            }
        }
    }

    lines.push(String::new());
    (lines.join("\n"), issue_count)
}

// ── Health Formatter ──

/// Format structural health metrics.
pub fn format_health(data: &Value, color: bool) -> String {
    let s = Style::new(color);
    let mut lines = format_header(data, None, &s);

    let health = data.get("health").and_then(|v| v.as_object());

    lines.push(String::new());
    lines.push(section_header("Health", &s));
    lines.push(String::new());

    if let Some(health) = health {
        let q = health.get("modularity_q").and_then(|v| v.as_f64());
        let q_str = q
            .map(|v| format!("{v:.3}"))
            .unwrap_or_else(|| "n/a".into());
        lines.push(format!("  Modularity Q: {q_str}"));

        if let Some(smoothness) = health.get("semantic_smoothness").and_then(|v| v.as_f64()) {
            lines.push(format!(
                "  Semantic smoothness: {smoothness:.3} (lower = better organized)"
            ));
        }
        if let Some(ami) = health
            .get("semantic_structural_ami")
            .and_then(|v| v.as_f64())
        {
            lines.push(format!(
                "  Structural-semantic AMI: {ami:.3} (higher = better alignment)"
            ));
        }

        // Phase 3: coherence, flow, THS from R-GIN
    } else {
        lines.push(s.dim("  No health metrics available."));
    }

    lines.push(String::new());
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn minimal_analysis() -> Value {
        json!({
            "scope": {"level": "module", "edge_kinds": ["calls"]},
            "coverage": {
                "analyzed_nodes": 5,
                "analyzed_edges": 3,
                "parsed_nodes": 10,
                "parsed_edges": 8,
            },
            "spectral": null,
            "architecture": {
                "modules": [
                    {"id": 0, "label": "core", "size": 3, "members": ["core.a", "core.b", "core.c"],
                     "cohesion": 0.2, "separation": 0.8, "confidence": 0.7, "unassigned": false},
                    {"id": 1, "label": "util", "size": 2, "members": ["util.x", "util.y"],
                     "cohesion": 0.3, "separation": 0.6, "confidence": 0.5, "unassigned": false},
                ],
                "dependencies": [
                    {"source": 0, "target": 1, "weight": 2, "edge_kinds": {"calls": 2}},
                ],
                "silhouette": 0.65,
                "package_fallback": false,
            },
            "roles": [
                {"node_id": "core.a", "role": "hub", "degree": 4, "betweenness": 0.5,
                 "in_degree": 2, "out_degree": 2},
                {"node_id": "util.x", "role": "utility", "degree": 2, "betweenness": 0.0,
                 "in_degree": 2, "out_degree": 0},
                {"node_id": "core.b", "role": "regular", "degree": 1, "betweenness": 0.0,
                 "in_degree": 0, "out_degree": 1},
            ],
            "issues": [
                {"id": "test-issue:core", "kind": "wide_interface", "title": "Wide interface",
                 "description": "23 coupling points between core and auth.", "severity": 0.8,
                 "severity_label": "high", "confidence": 0.9, "confidence_label": "high",
                 "anchors": []},
            ],
            "health": {"modularity_q": 0.42},
        })
    }

    #[test]
    fn test_format_issues_renders_issues() {
        let data = minimal_analysis();
        let (output, count) = format_issues(&data, &HashMap::new(), None, false);

        assert_eq!(count, 1);
        assert!(output.contains("Issues (1)"));
        assert!(output.contains("test-issue:core"));
        assert!(output.contains("23 coupling points between core and auth."));
        assert!(output.contains("[high]"));
        assert!(output.contains("✖ 1 issues"));
    }

    #[test]
    fn test_format_issues_ignores_filter() {
        let data = minimal_analysis();
        let (output_with, count_with) = format_issues(&data, &HashMap::new(), None, false);

        let mut ignores = HashMap::new();
        ignores.insert("test-issue:core".to_string(), "accepted".to_string());
        let (output_without, count_without) = format_issues(&data, &ignores, None, false);

        assert_eq!(count_with, 1);
        assert_eq!(count_without, 0);
        assert!(output_with.contains("test-issue:core"));
        // Issue ID still appears in the acknowledged section.
        assert!(output_without.contains("acknowledged"));
        assert!(output_without.contains("No issues detected."));
    }

    #[test]
    fn test_format_issues_no_issues() {
        let mut data = minimal_analysis();
        data["issues"] = json!([]);
        let (output, count) = format_issues(&data, &HashMap::new(), None, false);

        assert_eq!(count, 0);
        assert!(output.contains("No issues detected."));
    }

    #[test]
    fn test_format_health_shows_metrics() {
        let data = minimal_analysis();
        let output = format_health(&data, false);

        assert!(output.contains("Health"), "missing Health section");
        assert!(output.contains("Modularity Q: 0.420"));
    }

    #[test]
    fn test_format_health_null() {
        let mut data = minimal_analysis();
        data["health"] = json!(null);
        let output = format_health(&data, false);

        assert!(output.contains("No health metrics available."));
    }

}
