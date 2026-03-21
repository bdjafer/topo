//! Domain model approximation formatter (`--format=domain`).
//!
//! Outputs bounded context approximation from structural modules +
//! semantic coherence + top terms. Prominently labeled as approximate.

use serde_json::Value;

/// Format analysis JSON as a domain model approximation.
pub fn format_domain(data: &Value) -> String {
    let mut lines: Vec<String> = Vec::new();

    let architecture = data.get("architecture").and_then(|v| v.as_object());
    let health = data.get("health").and_then(|v| v.as_object());
    let roles = data.get("roles").and_then(|v| v.as_array());
    let semantic_enabled = data.get("semantic_enabled")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let modules = architecture
        .and_then(|a| a.get("modules"))
        .and_then(|v| v.as_array());
    let deps = architecture
        .and_then(|a| a.get("dependencies"))
        .and_then(|v| v.as_array());

    lines.push("# Domain Model (approximate — requires human validation)".to_string());
    lines.push(String::new());

    if !semantic_enabled {
        lines.push("Note: Semantic analysis not enabled. Domain model is based on structural".to_string());
        lines.push("modules only. Run with --semantic for semantic coherence and top terms.".to_string());
        lines.push(String::new());
    }

    // ── Bounded Contexts ──
    if let Some(modules) = modules {
        let clustered: Vec<&Value> = modules.iter()
            .filter(|m| !m.get("unassigned").and_then(|v| v.as_bool()).unwrap_or(false))
            .collect();

        // Build label map.
        // Find hub/entry_point nodes per module for aggregate root candidates.
        let entry_points = find_entry_points(roles, modules);

        let ami = health.and_then(|h| h.get("semantic_structural_ami")).and_then(|v| v.as_f64());
        let ami_str = ami.map(|v| format!(", AMI: {v:.2}")).unwrap_or_default();

        lines.push(format!(
            "Bounded contexts ({} detected{}):",
            clustered.len(), ami_str
        ));
        lines.push(String::new());

        for module in &clustered {
            let label = module.get("label").and_then(|v| v.as_str()).unwrap_or("unknown");
            let size = module.get("size").and_then(|v| v.as_u64()).unwrap_or(0);
            let mod_id = module.get("id").and_then(|v| v.as_u64()).unwrap_or(0);

            let coherence = module.get("semantic_coherence").and_then(|v| v.as_f64());
            let coh_str = coherence
                .map(|c| format!(", coherence: {c:.2}"))
                .unwrap_or_default();
            let warning = if coherence.map(|c| c < 0.5).unwrap_or(false) {
                " [!] low coherence"
            } else {
                ""
            };

            lines.push(format!("  {label} ({size} nodes{coh_str}){warning}"));

            // Top terms.
            if let Some(terms) = module.get("top_terms").and_then(|v| v.as_array()) {
                let term_strs: Vec<&str> = terms.iter().filter_map(|t| t.as_str()).collect();
                if !term_strs.is_empty() {
                    lines.push(format!("    Top terms: {}", term_strs.join(", ")));
                }
            }

            // Aggregate root candidate.
            if let Some(root_id) = entry_points.get(&mod_id) {
                let short = root_id.rsplit('.').next().unwrap_or(root_id);
                lines.push(format!("    Aggregate root candidate: {short} (entry_point/hub)"));
            }

            lines.push(String::new());
        }
    }

    // ── Context Map ──
    if let Some(deps) = deps {
        if deps.is_empty() {
            // No dependencies — skip context map section.
        } else if let Some(modules) = modules {
            let label_map: std::collections::HashMap<u64, &str> = modules.iter()
                .filter_map(|m| {
                    let id = m.get("id")?.as_u64()?;
                    let label = m.get("label")?.as_str()?;
                    Some((id, label))
                })
                .collect();

            lines.push("Context relationships:".to_string());
            for dep in deps {
                let src = dep.get("source").and_then(|v| v.as_u64()).unwrap_or(0);
                let tgt = dep.get("target").and_then(|v| v.as_u64()).unwrap_or(0);
                let src_label = label_map.get(&src).unwrap_or(&"?");
                let tgt_label = label_map.get(&tgt).unwrap_or(&"?");

                let kinds = dep.get("edge_kinds").and_then(|v| v.as_object());
                let kind_list: Vec<&str> = kinds
                    .map(|k| k.keys().map(|s| s.as_str()).collect())
                    .unwrap_or_default();

                // Classify relationship type based on coupling.
                let rel_type = classify_relationship(&kind_list);

                lines.push(format!(
                    "  {src_label} ──[{}]──→ {tgt_label}   ({rel_type})",
                    kind_list.join("+")
                ));
            }
        }
    }

    lines.join("\n")
}

/// Find the best entry_point or hub node in each module.
fn find_entry_points(
    roles: Option<&Vec<Value>>,
    modules: &[Value],
) -> std::collections::HashMap<u64, String> {
    let mut result = std::collections::HashMap::new();

    let roles = match roles {
        Some(r) => r,
        None => return result,
    };

    // Build node -> module map.
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

    // Find entry_point or hub roles per module.
    for role in roles {
        let node_id = role.get("node_id").and_then(|v| v.as_str()).unwrap_or("");
        let role_name = role.get("role").and_then(|v| v.as_str()).unwrap_or("");
        if !matches!(role_name, "entry_point" | "hub") {
            continue;
        }
        if let Some(&mod_id) = node_module.get(node_id) {
            // Prefer entry_point over hub.
            let existing = result.get(&mod_id);
            let should_insert = match existing {
                None => true,
                Some(_) if role_name == "entry_point" => true,
                _ => false,
            };
            if should_insert {
                result.insert(mod_id, node_id.to_string());
            }
        }
    }

    result
}

/// Classify a cross-module relationship based on coupling type.
fn classify_relationship(kinds: &[&str]) -> &'static str {
    let has_calls = kinds.contains(&"calls");
    let has_imports = kinds.contains(&"imports");

    match (has_calls, has_imports) {
        (true, true) => "tight coupling — consider anti-corruption layer",
        (true, false) => "customer-supplier",
        (false, true) => "shared kernel",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_format_domain_basic() {
        let data = json!({
            "scope": {"level": "module", "edge_kinds": ["calls"]},
            "coverage": {"analyzed_nodes": 50, "analyzed_edges": 100, "parsed_nodes": 60, "parsed_edges": 120},
            "spectral": null,
            "architecture": {
                "modules": [
                    {"id": 0, "label": "auth", "size": 12, "members": ["auth.login"], "cohesion": 0.8, "confidence": 0.9, "unassigned": false, "top_terms": ["token", "session"], "semantic_coherence": 0.85},
                    {"id": 1, "label": "payment", "size": 20, "members": ["payment.charge"], "cohesion": 0.7, "confidence": 0.8, "unassigned": false}
                ],
                "dependencies": [
                    {"source": 1, "target": 0, "weight": 5, "edge_kinds": {"calls": 5}}
                ],
                "silhouette": 0.6,
                "package_fallback": false
            },
            "roles": [
                {"node_id": "auth.login", "role": "entry_point", "degree": 5, "betweenness": 0.1, "in_degree": 3, "out_degree": 2}
            ],
            "issues": [],
            "health": {"modularity_q": 0.6},
            "semantic_enabled": true
        });

        let output = format_domain(&data);
        assert!(output.contains("Domain Model"), "should have title");
        assert!(output.contains("auth"), "should mention auth");
        assert!(output.contains("token, session"), "should show top terms");
        assert!(output.contains("customer-supplier"), "should classify relationship");
        assert!(output.contains("Aggregate root"), "should suggest aggregate root");
    }
}
