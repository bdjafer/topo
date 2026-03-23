//! LLM context narrative formatter (`--format=context`).
//!
//! Produces a compact 2,000-4,000 token structural narrative designed for
//! LLM consumption. An LLM reading source code plus this narrative can
//! reason about global architecture.

use serde_json::Value;

/// Format analysis JSON into a compact LLM context narrative.
pub fn format_context(data: &Value) -> String {
    let mut lines: Vec<String> = Vec::new();

    let coverage = data.get("coverage").and_then(|v| v.as_object());
    let architecture = data.get("architecture").and_then(|v| v.as_object());
    let spectral = data.get("spectral").and_then(|v| v.as_object());
    let health = data.get("health").and_then(|v| v.as_object());
    let issues = data.get("issues").and_then(|v| v.as_array());
    let semantic_enabled = data.get("semantic_enabled")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let modules = architecture
        .and_then(|a| a.get("modules"))
        .and_then(|v| v.as_array());
    let deps = architecture
        .and_then(|a| a.get("dependencies"))
        .and_then(|v| v.as_array());

    let node_count = coverage
        .and_then(|c| c.get("analyzed_nodes"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    // ── Header line ──
    let mod_count = modules.map(|m| m.len()).unwrap_or(0);
    let q = health.and_then(|h| h.get("modularity_q")).and_then(|v| v.as_f64());
    let fiedler = spectral.and_then(|s| s.get("fiedler_value")).and_then(|v| v.as_f64());

    // Phase 3 THS takes precedence as the headline health number
    let ths = health.and_then(|h| h.get("topo_health_score")).and_then(|v| v.as_f64());
    let coherence_val = health.and_then(|h| h.get("coherence")).and_then(|v| v.as_f64());
    let flow_val = health.and_then(|h| h.get("flow")).and_then(|v| v.as_f64());

    let mut health_parts = Vec::new();
    if let Some(q) = q {
        health_parts.push(format!("Q={q:.2}"));
    }
    if let Some(f) = fiedler {
        health_parts.push(format!("λ₂={f:.4}"));
    }
    if semantic_enabled {
        if let Some(smoothness) = health.and_then(|h| h.get("semantic_smoothness")).and_then(|v| v.as_f64()) {
            health_parts.push(format!("smoothness={smoothness:.2}"));
        }
        if let Some(ami) = health.and_then(|h| h.get("semantic_structural_ami")).and_then(|v| v.as_f64()) {
            health_parts.push(format!("AMI={ami:.2}"));
        }
    }

    let arch_type = classify_architecture(modules, deps);
    lines.push(format!(
        "## {arch_type} ({mod_count} modules, {node_count} nodes)"
    ));

    // THS leads, raw metrics follow on the same line
    if let Some(ths) = ths {
        let sub = match (coherence_val, flow_val) {
            (Some(c), Some(f)) => format!(" (coherence: {c:.2}, flow: {f:.2})"),
            _ => String::new(),
        };
        let raw = if health_parts.is_empty() {
            String::new()
        } else {
            format!(" | {}", health_parts.join(", "))
        };
        lines.push(format!("Health: {ths:.2}{sub}{raw}"));
    } else if !health_parts.is_empty() {
        lines.push(format!("Health: {}", health_parts.join(", ")));
    }
    lines.push(String::new());

    // ── Module summaries ──
    if let Some(modules) = modules {
        // Build label map for dependency rendering.
        let label_map: std::collections::HashMap<u64, &str> = modules.iter()
            .filter_map(|m| {
                let id = m.get("id")?.as_u64()?;
                let label = m.get("label")?.as_str()?;
                Some((id, label))
            })
            .collect();

        // Collect issues per module for inline display.
        let module_issues = collect_module_issues(issues, modules);

        // Sort modules by size descending.
        let mut sorted_modules: Vec<&Value> = modules.iter()
            .filter(|m| !m.get("unassigned").and_then(|v| v.as_bool()).unwrap_or(false))
            .collect();
        sorted_modules.sort_by(|a, b| {
            let sa = a.get("size").and_then(|v| v.as_u64()).unwrap_or(0);
            let sb = b.get("size").and_then(|v| v.as_u64()).unwrap_or(0);
            sb.cmp(&sa)
        });

        // For large codebases: top-10 by size + all with concerns.
        let max_detail = if sorted_modules.len() > 15 { 10 } else { sorted_modules.len() };

        for (i, module) in sorted_modules.iter().enumerate() {
            let label = module.get("label").and_then(|v| v.as_str()).unwrap_or("unknown");
            let size = module.get("size").and_then(|v| v.as_u64()).unwrap_or(0);
            let mod_id = module.get("id").and_then(|v| v.as_u64()).unwrap_or(0);
            let has_concerns = module_issues.contains_key(&mod_id);

            // Skip non-top modules without concerns in large codebases.
            if i >= max_detail && !has_concerns {
                continue;
            }

            let mut parts = vec![format!("{size} nodes")];

            // Cohesion
            if let Some(coh) = module.get("cohesion").and_then(|v| v.as_f64()) {
                parts.push(format!("cohesion: {coh:.2}"));
            }
            // Semantic coherence
            if let Some(sem) = module.get("semantic_coherence").and_then(|v| v.as_f64()) {
                parts.push(format!("sem: {sem:.2}"));
            }

            lines.push(format!("### Module: {label} ({})", parts.join(", ")));

            // Top terms
            if let Some(terms) = module.get("top_terms").and_then(|v| v.as_array()) {
                let term_strs: Vec<&str> = terms.iter().filter_map(|t| t.as_str()).collect();
                if !term_strs.is_empty() {
                    lines.push(format!("Top terms: {}", term_strs.join(", ")));
                }
            }

            // Bridges (from dependencies)
            if let Some(deps) = deps {
                let bridges: Vec<String> = deps.iter()
                    .filter(|d| d.get("source").and_then(|v| v.as_u64()) == Some(mod_id))
                    .filter_map(|d| {
                        let target = d.get("target")?.as_u64()?;
                        let target_label = label_map.get(&target)?;
                        let kinds = d.get("edge_kinds").and_then(|v| v.as_object());
                        let kind_str = kinds
                            .map(|k| k.keys().cloned().collect::<Vec<_>>().join("+"))
                            .unwrap_or_default();
                        Some(format!("[{target_label}] ({kind_str})"))
                    })
                    .collect();
                if !bridges.is_empty() {
                    lines.push(format!("Bridges to: {}", bridges.join(", ")));
                }
            }

            // Inline concerns for this module.
            if let Some(mod_issues) = module_issues.get(&mod_id) {
                for issue in mod_issues {
                    let kind = issue.get("kind").and_then(|v| v.as_str()).unwrap_or("");
                    let desc = issue.get("description").and_then(|v| v.as_str()).unwrap_or("");
                    // Truncate long descriptions for context budget.
                    let short = truncate_str(desc, 120);
                    lines.push(format!("Concern: [{kind}] {short}"));
                }
            }

            lines.push(String::new());
        }

        // Count remaining modules.
        let shown = sorted_modules.iter().enumerate()
            .filter(|(i, m)| {
                *i < max_detail || module_issues.contains_key(
                    &m.get("id").and_then(|v| v.as_u64()).unwrap_or(u64::MAX)
                )
            })
            .count();
        if shown < sorted_modules.len() {
            lines.push(format!(
                "{} additional modules with no structural concerns.",
                sorted_modules.len() - shown
            ));
            lines.push(String::new());
        }
    }

    // ── Issues section ──
    if let Some(issues) = issues {
        let high_medium: Vec<&Value> = issues.iter()
            .filter(|i| {
                let sev = i.get("severity_label").and_then(|v| v.as_str()).unwrap_or("low");
                sev == "high" || sev == "medium"
            })
            .collect();

        if !high_medium.is_empty() {
            let high_count = high_medium.iter()
                .filter(|i| i.get("severity_label").and_then(|v| v.as_str()) == Some("high"))
                .count();
            let med_count = high_medium.len() - high_count;
            let low_count = issues.len() - high_medium.len();

            lines.push(format!(
                "## Structural Concerns ({high_count} high, {med_count} medium, {low_count} low)"
            ));
            lines.push(String::new());

            for issue in &high_medium {
                let sev = issue.get("severity_label").and_then(|v| v.as_str()).unwrap_or("medium");
                let kind = issue.get("kind").and_then(|v| v.as_str()).unwrap_or("");
                let desc = issue.get("description").and_then(|v| v.as_str()).unwrap_or("");

                let mut detail = format!("[{sev}] {kind}");

                // Anchors
                if let Some(anchors) = issue.get("anchors").and_then(|v| v.as_array()) {
                    if let Some(anchor) = anchors.first() {
                        if let Some(node_id) = anchor.get("node_id").and_then(|v| v.as_str()) {
                            detail.push_str(&format!(": {}", node_id.rsplit('.').next().unwrap_or(node_id)));
                        }
                    }
                }

                lines.push(detail);

                // Truncate for token budget.
                let short_desc = truncate_str(desc, 200);
                lines.push(format!("  {short_desc}"));

                // Suggested action for misplaced_concern
                if let Some(suggested) = issue.get("suggested_module").and_then(|v| v.as_str()) {
                    lines.push(format!("  Suggested action: move to {suggested}"));
                }
                lines.push(String::new());
            }
        }
    }

    lines.join("\n")
}

/// Classify architecture type from module/dependency structure.
fn classify_architecture(modules: Option<&Vec<Value>>, deps: Option<&Vec<Value>>) -> String {
    let n_modules = modules.map(|m| m.len()).unwrap_or(0);
    let n_deps = deps.map(|d| d.len()).unwrap_or(0);

    if n_modules <= 1 {
        return "monolith".to_string();
    }

    // Check for layered structure: are dependencies mostly one-directional?
    if let Some(deps) = deps {
        let mut pairs: std::collections::HashSet<(u64, u64)> = std::collections::HashSet::new();
        let mut bidirectional = 0;
        for d in deps {
            let src = d.get("source").and_then(|v| v.as_u64()).unwrap_or(0);
            let tgt = d.get("target").and_then(|v| v.as_u64()).unwrap_or(0);
            if pairs.contains(&(tgt, src)) {
                bidirectional += 1;
            }
            pairs.insert((src, tgt));
        }

        if n_deps >= 2 && bidirectional == 0 {
            return format!("layered ({n_modules} tiers)");
        }
    }

    if n_modules > 6 {
        format!("modular ({n_modules} modules)")
    } else {
        format!("{n_modules} modules")
    }
}

/// Collect issues that belong to specific modules (by anchor node_id membership).
fn collect_module_issues<'a>(
    issues: Option<&'a Vec<Value>>,
    modules: &[Value],
) -> std::collections::HashMap<u64, Vec<&'a Value>> {
    let mut result: std::collections::HashMap<u64, Vec<&Value>> = std::collections::HashMap::new();

    let issues = match issues {
        Some(i) => i,
        None => return result,
    };

    // Build node_id -> module_id map.
    let mut node_module: std::collections::HashMap<&str, u64> = std::collections::HashMap::new();
    for m in modules {
        let mod_id = m.get("id").and_then(|v| v.as_u64()).unwrap_or(0);
        if let Some(members) = m.get("members").and_then(|v| v.as_array()) {
            for member in members {
                if let Some(nid) = member.as_str() {
                    node_module.insert(nid, mod_id);
                }
            }
        }
    }

    for issue in issues {
        if let Some(anchors) = issue.get("anchors").and_then(|v| v.as_array()) {
            for anchor in anchors {
                if let Some(nid) = anchor.get("node_id").and_then(|v| v.as_str()) {
                    if let Some(&mod_id) = node_module.get(nid) {
                        result.entry(mod_id).or_default().push(issue);
                        break; // One module per issue.
                    }
                }
            }
        }
    }

    result
}

/// Truncate a string to at most `max_chars` characters, safe for multi-byte UTF-8.
fn truncate_str(s: &str, max_chars: usize) -> &str {
    match s.char_indices().nth(max_chars) {
        Some((byte_idx, _)) => &s[..byte_idx],
        None => s,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_format_context_basic() {
        let data = json!({
            "scope": {"level": "module", "edge_kinds": ["calls"]},
            "coverage": {"analyzed_nodes": 50, "analyzed_edges": 100, "parsed_nodes": 60, "parsed_edges": 120},
            "spectral": {"fiedler_value": 0.01, "eigenvalues": [], "nodes_covered": 50, "coverage_ratio": 1.0, "components": 1, "largest_component_ratio": 1.0},
            "architecture": {
                "modules": [
                    {"id": 0, "label": "auth", "size": 12, "members": ["auth.login"], "cohesion": 0.8, "confidence": 0.9, "unassigned": false, "top_terms": ["token", "session"]},
                    {"id": 1, "label": "payment", "size": 20, "members": ["payment.charge"], "cohesion": 0.7, "confidence": 0.8, "unassigned": false}
                ],
                "dependencies": [
                    {"source": 1, "target": 0, "weight": 5, "edge_kinds": {"calls": 5}}
                ],
                "silhouette": 0.6,
                "package_fallback": false
            },
            "roles": [],
            "issues": [
                {"id": "test:1", "kind": "circular_dependency", "title": "Dependency cycle", "description": "Test issue", "severity": 0.7, "severity_label": "high", "confidence": 0.8, "confidence_label": "high", "anchors": [{"node_id": "payment.charge"}]}
            ],
            "health": {"modularity_q": 0.6}
        });

        let output = format_context(&data);
        assert!(output.contains("## "), "should have markdown headers");
        assert!(output.contains("payment"), "should mention payment module");
        assert!(output.contains("auth"), "should mention auth module");
        assert!(output.contains("Structural Concerns"), "should have concerns section");
    }

    #[test]
    fn test_classify_architecture_layered() {
        let modules = vec![json!({}), json!({}), json!({})];
        // Need >2 deps to trigger the layered branch (n_deps > 2 check).
        let deps = vec![
            json!({"source": 0, "target": 1}),
            json!({"source": 1, "target": 2}),
            json!({"source": 0, "target": 2}),
        ];
        // All one-directional, no bidirectional -> layered.
        let result = classify_architecture(Some(&modules), Some(&deps));
        assert!(result.contains("layered"),
            "expected layered classification, got: {result}");
    }
}
