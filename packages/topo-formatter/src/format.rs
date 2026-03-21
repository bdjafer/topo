//! Human-readable text formatter for structural analysis results.
//!
//! Consumes analysis.schema.json dicts and produces formatted text output.

use std::collections::HashMap;
use std::path::Path;

use serde_json::Value;

use crate::dag::render_dependency_dag;
use crate::style::Style;

/// Format analysis JSON into human-readable text.
pub fn format_text(
    data: &Value,
    verbose: bool,
    diagnostics: bool,
    ignores: &HashMap<String, String>,
    project_root: Option<&Path>,
    color: bool,
) -> String {
    let s = Style::new(color);
    let mut lines: Vec<String> = Vec::new();

    let coverage = data.get("coverage").and_then(|v| v.as_object());
    let architecture = data.get("architecture").and_then(|v| v.as_object());
    let spectral = data.get("spectral").and_then(|v| v.as_object());
    let health = data.get("health").and_then(|v| v.as_object());
    let issues = data
        .get("issues")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let roles = data
        .get("roles")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    // ── Header ──
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

    lines.push(s.bold(&format!("topo — {root_label}")));
    lines.push(format!(
        "{analyzed_nodes} nodes, {analyzed_edges} edges ({parsed_nodes} symbols parsed)"
    ));

    // ── Issues ──
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
                    let file = anchor.get("file").and_then(|v| v.as_str()).unwrap_or("");
                    let line = anchor.get("line").and_then(|v| v.as_u64()).unwrap_or(0);
                    let path = relative_path(file, project_root);
                    lines.push(format!("    → {}", s.cyan(&format!("{path}:{line}"))));
                }
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
        lines.push(format!(
            "  {} acknowledged (use --verbose to show)",
            acknowledged.len()
        ));
    }

    if verbose && !acknowledged.is_empty() {
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
                lines.push(format!("    {}", s.dim(&format!("Reason: {justification}"))));
            }
        }
    }

    // ── Architecture ──
    let modules = architecture
        .and_then(|a| a.get("modules"))
        .and_then(|v| v.as_array());
    let deps = architecture
        .and_then(|a| a.get("dependencies"))
        .and_then(|v| v.as_array());
    let has_deps = deps.is_some_and(|d| !d.is_empty());

    if has_deps || verbose {
        lines.push(String::new());
        lines.push(section_header("Architecture", &s));
        lines.push(String::new());

        if verbose {
            if let Some(modules) = modules {
                let clustered: Vec<&Value> = modules
                    .iter()
                    .filter(|m| !m.get("unassigned").and_then(|v| v.as_bool()).unwrap_or(false))
                    .collect();

                let labels: Vec<&str> = clustered
                    .iter()
                    .map(|m| {
                        m.get("label")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown")
                    })
                    .collect();
                let mut label_counts: HashMap<&str, usize> = HashMap::new();
                for lbl in &labels {
                    *label_counts.entry(lbl).or_default() += 1;
                }

                for (module, label) in clustered.iter().zip(labels.iter()) {
                    let id = module.get("id").and_then(|v| v.as_u64()).unwrap_or(0);
                    let size = module.get("size").and_then(|v| v.as_u64()).unwrap_or(0);
                    let display_label = if label_counts.get(label).copied().unwrap_or(0) > 1 {
                        format!("{label} (group {id})")
                    } else {
                        label.to_string()
                    };
                    lines.push(format!("  {display_label} ({size} nodes)"));

                    if let Some(members) = module.get("members").and_then(|v| v.as_array()) {
                        let member_names: Vec<String> = members
                            .iter()
                            .filter_map(|m| m.as_str())
                            .map(|nid| member_display(nid, label))
                            .collect();
                        if member_names.len() <= 6 {
                            lines.push(format!("    {}", member_names.join(", ")));
                        } else {
                            lines.push(format!(
                                "    {}, ...",
                                member_names[..5].join(", ")
                            ));
                        }
                    }
                }

                let unassigned: Vec<&Value> = modules
                    .iter()
                    .filter(|m| m.get("unassigned").and_then(|v| v.as_bool()).unwrap_or(false))
                    .collect();
                if !unassigned.is_empty() {
                    let total: usize = unassigned
                        .iter()
                        .filter_map(|m| m.get("members").and_then(|v| v.as_array()))
                        .map(|a| a.len())
                        .sum();
                    lines.push(format!("  (unassigned: {total} nodes)"));
                }

                lines.push(String::new());
            }
        }

        if has_deps {
            if let (Some(deps), Some(modules)) = (deps, modules) {
                let module_labels: HashMap<u64, &str> = modules
                    .iter()
                    .filter_map(|m| {
                        let id = m.get("id")?.as_u64()?;
                        let label = m.get("label")?.as_str()?;
                        Some((id, label))
                    })
                    .collect();
                let dag_lines = render_dependency_dag(deps, &module_labels, &s);
                lines.extend(dag_lines);
            }
        }
    }

    // ── Package Agreement ──
    if let Some(arch) = architecture {
        if let Some(pa) = arch.get("package_agreement").and_then(|v| v.as_object()) {
            let nmi = pa
                .get("nmi")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            lines.push(String::new());
            lines.push(section_header(&format!("Package Agreement (NMI: {nmi:.2})"), &s));
            lines.push(String::new());

            if let Some(composition) = pa.get("module_composition").and_then(|v| v.as_array()) {
                let empty_modules = Vec::new();
                let module_labels: HashMap<u64, &str> = modules
                    .unwrap_or(&empty_modules)
                    .iter()
                    .filter_map(|m| {
                        let id = m.get("id")?.as_u64()?;
                        let label = m.get("label")?.as_str()?;
                        Some((id, label))
                    })
                    .collect();

                for entry in composition {
                    let mid = entry.get("module_id").and_then(|v| v.as_u64()).unwrap_or(0);
                    let label = module_labels.get(&mid).copied().unwrap_or("unknown");
                    let cross = entry
                        .get("cross_package")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);

                    if let Some(pkgs) = entry.get("packages").and_then(|v| v.as_object()) {
                        let mut pkg_parts: Vec<(&String, u64)> = pkgs
                            .iter()
                            .filter_map(|(k, v)| Some((k, v.as_u64()?)))
                            .collect();
                        pkg_parts.sort_by(|a, b| b.1.cmp(&a.1));

                        if cross {
                            let parts: Vec<String> = pkg_parts
                                .iter()
                                .map(|(pkg, count)| format!("{pkg} ({count})"))
                                .collect();
                            lines.push(format!(
                                "  {} spans: {}",
                                s.bold(label),
                                parts.join(", ")
                            ));
                        } else if let Some((pkg, _)) = pkg_parts.first() {
                            lines.push(format!(
                                "  {} is within: {pkg}",
                                s.bold(label),
                            ));
                        }
                    }
                }
            }
        }
    }

    // ── Critical Nodes ──
    let mut critical_roles: Vec<&Value> = roles
        .iter()
        .filter(|r| {
            let role = r.get("role").and_then(|v| v.as_str()).unwrap_or("");
            role != "regular" && role != "orphan"
        })
        .collect();

    if !critical_roles.is_empty() {
        let role_order = |r: &str| -> u8 {
            match r {
                "hub" => 0,
                "bridge" => 1,
                "entry_point" => 2,
                "utility" => 3,
                _ => 9,
            }
        };
        critical_roles.sort_by(|a, b| {
            let ra = a.get("role").and_then(|v| v.as_str()).unwrap_or("");
            let rb = b.get("role").and_then(|v| v.as_str()).unwrap_or("");
            let da = a.get("degree").and_then(|v| v.as_u64()).unwrap_or(0);
            let db = b.get("degree").and_then(|v| v.as_u64()).unwrap_or(0);
            role_order(ra).cmp(&role_order(rb)).then(db.cmp(&da))
        });

        if !verbose {
            let mut shown: Vec<&Value> = Vec::new();
            let mut counts: HashMap<&str, usize> = HashMap::new();
            for r in &critical_roles {
                let role_name = r.get("role").and_then(|v| v.as_str()).unwrap_or("");
                let count = counts.entry(role_name).or_default();
                *count += 1;
                if *count <= 2 {
                    shown.push(r);
                }
            }
            critical_roles = shown;
        }

        lines.push(String::new());
        lines.push(section_header("Critical Nodes", &s));
        lines.push(String::new());

        for r in &critical_roles {
            let role = r.get("role").and_then(|v| v.as_str()).unwrap_or("");
            let node_id = r.get("node_id").and_then(|v| v.as_str()).unwrap_or("");
            let desc = role_description(r);
            lines.push(format!(
                "  {} {:<35} {}",
                s.bold(&format!("{:<12}", role.to_uppercase())),
                node_id,
                s.dim(&desc)
            ));
        }
    }

    // ── Health ──
    if let Some(health) = health {
        lines.push(String::new());
        lines.push(section_header("Health", &s));
        lines.push(String::new());
        let q = health
            .get("modularity_q")
            .and_then(|v| v.as_f64());
        let q_str = q.map(|v| format!("{v:.3}")).unwrap_or_else(|| "n/a".into());
        lines.push(format!("  Modularity Q: {q_str}"));
    }

    // ── Diagnostics ──
    if diagnostics {
        lines.push(String::new());
        lines.push(section_header("Diagnostics", &s));
        lines.push(String::new());

        if let Some(cov) = coverage {
            let pn = cov.get("parsed_nodes").and_then(|v| v.as_u64()).unwrap_or(0);
            let pe = cov.get("parsed_edges").and_then(|v| v.as_u64()).unwrap_or(0);
            let an = cov.get("analyzed_nodes").and_then(|v| v.as_u64()).unwrap_or(0);
            let ae = cov.get("analyzed_edges").and_then(|v| v.as_u64()).unwrap_or(0);
            lines.push(format!("  Parsed: {pn} nodes, {pe} edges"));
            lines.push(format!("  Analyzed: {an} nodes, {ae} edges"));
        }

        if let Some(spec) = spectral {
            let fiedler = spec.get("fiedler_value").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let eigenvalues = spec.get("eigenvalues").and_then(|v| v.as_array());
            let dims = eigenvalues.map(|e| e.len()).unwrap_or(0);
            let components = spec.get("components").and_then(|v| v.as_u64()).unwrap_or(0);
            let largest = spec
                .get("largest_component_ratio")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let nodes_covered = spec.get("nodes_covered").and_then(|v| v.as_u64()).unwrap_or(0);
            let coverage_ratio = spec
                .get("coverage_ratio")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let an = coverage
                .and_then(|c| c.get("analyzed_nodes"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0);

            lines.push(format!("  Algebraic connectivity: {fiedler:.4}"));
            lines.push(format!("  Spectral dimensions: {dims}"));
            lines.push(format!(
                "  Components: {components}    Largest: {:.1}%",
                largest * 100.0
            ));
            lines.push(format!(
                "  Spectral coverage: {nodes_covered}/{an} ({:.1}%)",
                coverage_ratio * 100.0
            ));
        }

        if let Some(arch) = architecture {
            if let Some(sil) = arch.get("silhouette").and_then(|v| v.as_f64()) {
                lines.push(format!("  Silhouette: {sil:.3}"));
            }
        }
    }

    lines.push(String::new());
    lines.join("\n")
}

// ── Helpers ──

fn section_header(title: &str, s: &Style) -> String {
    let prefix = format!("── {title} ");
    let pad = 60usize.saturating_sub(prefix.len());
    let line = format!("{prefix}{}", "─".repeat(pad));
    s.bold(&line)
}

fn member_display(node_id: &str, module_label: &str) -> String {
    let prefix = format!("{module_label}.");
    if node_id.starts_with(&prefix) {
        node_id[prefix.len()..].to_string()
    } else {
        node_id.to_string()
    }
}

fn role_description(role: &Value) -> String {
    let r = role.get("role").and_then(|v| v.as_str()).unwrap_or("");
    match r {
        "hub" => {
            let degree = role.get("degree").and_then(|v| v.as_u64()).unwrap_or(0);
            format!("degree {degree}")
        }
        "bridge" => {
            let betweenness = role.get("betweenness").and_then(|v| v.as_f64()).unwrap_or(0.0);
            format!("betweenness {betweenness:.3}")
        }
        "entry_point" => {
            let out_d = role.get("out_degree").and_then(|v| v.as_u64()).unwrap_or(0);
            let in_d = role.get("in_degree").and_then(|v| v.as_u64()).unwrap_or(0);
            format!("{out_d} outbound, {in_d} inbound")
        }
        "utility" => {
            let in_d = role.get("in_degree").and_then(|v| v.as_u64()).unwrap_or(0);
            let out_d = role.get("out_degree").and_then(|v| v.as_u64()).unwrap_or(0);
            format!("{in_d} inbound, {out_d} outbound")
        }
        _ => String::new(),
    }
}

fn relative_path(file_path: &str, project_root: Option<&Path>) -> String {
    if let Some(root) = project_root {
        if let Ok(rel) = Path::new(file_path).strip_prefix(root) {
            return rel.display().to_string();
        }
    }
    file_path.to_string()
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
                {"id": "test-issue:core", "kind": "god_module", "title": "God module",
                 "description": "Module core is too large.", "severity": 0.8,
                 "severity_label": "high", "confidence": 0.9, "confidence_label": "high",
                 "anchors": []},
            ],
            "health": {"modularity_q": 0.42},
        })
    }

    #[test]
    fn test_format_text_includes_all_sections() {
        let data = minimal_analysis();
        let output = format_text(&data, false, false, &HashMap::new(), None, false);

        assert!(output.contains("Issues"), "missing Issues section");
        assert!(output.contains("Architecture"), "missing Architecture section");
        assert!(output.contains("Critical Nodes"), "missing Critical Nodes section");
        assert!(output.contains("Health"), "missing Health section");
        assert!(output.contains("Modularity Q: 0.420"), "missing modularity value");
    }

    #[test]
    fn test_format_text_renders_issues() {
        let data = minimal_analysis();
        let output = format_text(&data, false, false, &HashMap::new(), None, false);

        assert!(output.contains("test-issue:core"));
        assert!(output.contains("Module core is too large."));
        assert!(output.contains("[high]"));
    }

    #[test]
    fn test_format_text_renders_critical_roles() {
        let data = minimal_analysis();
        let output = format_text(&data, false, false, &HashMap::new(), None, false);

        assert!(output.contains("core.a"));
        assert!(output.contains("HUB"));
        assert!(output.contains("util.x"));
        assert!(output.contains("UTILITY"));
        assert!(!output.contains("core.b"));
    }

    #[test]
    fn test_format_text_ignores_filter() {
        let data = minimal_analysis();
        let output_with = format_text(&data, false, false, &HashMap::new(), None, false);

        let mut ignores = HashMap::new();
        ignores.insert("test-issue:core".to_string(), "accepted".to_string());
        let output_without = format_text(&data, false, false, &ignores, None, false);

        assert!(output_with.contains("test-issue:core"));
        assert!(!output_without.contains("test-issue:core"));
        assert!(output_without.contains("acknowledged"));
    }

    #[test]
    fn test_format_text_null_spectral_and_health() {
        let mut data = minimal_analysis();
        data["spectral"] = json!(null);
        data["health"] = json!(null);

        let output = format_text(&data, false, false, &HashMap::new(), None, false);
        assert!(output.contains("Issues"));
        assert!(!output.contains("Health"));
    }

    #[test]
    fn test_format_text_verbose_shows_module_members() {
        let data = minimal_analysis();
        let output = format_text(&data, true, false, &HashMap::new(), None, false);

        assert!(output.contains("core (3 nodes)"));
        assert!(output.contains("util (2 nodes)"));
    }

    #[test]
    fn test_format_text_diagnostics() {
        let mut data = minimal_analysis();
        data["spectral"] = json!({
            "fiedler_value": 0.5,
            "eigenvalues": [0.5, 1.2, 2.0],
            "nodes_covered": 4,
            "coverage_ratio": 0.8,
            "components": 1,
            "largest_component_ratio": 1.0,
        });

        let output = format_text(&data, false, true, &HashMap::new(), None, false);

        assert!(output.contains("Diagnostics"));
        assert!(output.contains("Algebraic connectivity: 0.5000"));
        assert!(output.contains("Spectral dimensions: 3"));
        assert!(output.contains("Silhouette: 0.650"));
    }
}
