//! Human-readable text formatter for structural analysis results.
//!
//! Port of topo_formatter/text.py. Consumes analysis.schema.json dicts.

use std::collections::HashMap;
use std::path::Path;

use owo_colors::OwoColorize;
use serde_json::Value;

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

// ── Style ──

struct Style {
    enabled: bool,
}

impl Style {
    fn new(enabled: bool) -> Self {
        Self { enabled }
    }

    fn bold(&self, text: &str) -> String {
        if self.enabled {
            format!("{}", text.bold())
        } else {
            text.to_string()
        }
    }

    fn dim(&self, text: &str) -> String {
        if self.enabled {
            format!("{}", text.dimmed())
        } else {
            text.to_string()
        }
    }

    fn red_text(&self, text: &str) -> String {
        if self.enabled {
            format!("{}", text.red())
        } else {
            text.to_string()
        }
    }

    fn yellow_text(&self, text: &str) -> String {
        if self.enabled {
            format!("{}", text.yellow())
        } else {
            text.to_string()
        }
    }

    fn cyan(&self, text: &str) -> String {
        if self.enabled {
            format!("{}", text.cyan())
        } else {
            text.to_string()
        }
    }

    fn green(&self, text: &str) -> String {
        if self.enabled {
            format!("{}", text.green())
        } else {
            text.to_string()
        }
    }

    fn severity(&self, label: &str, text: &str) -> String {
        match label {
            "high" => self.red_text(text),
            "medium" => self.yellow_text(text),
            _ => self.dim(text),
        }
    }
}

// ── DAG Renderer ──

const U: u8 = 1;
const D: u8 = 2;
const L: u8 = 4;
const R: u8 = 8;

fn box_char(flags: u8) -> char {
    match flags {
        f if f == U || f == D || f == U | D => '│',
        f if f == L || f == R || f == L | R => '─',
        f if f == U | R => '└',
        f if f == U | L => '┘',
        f if f == D | R => '┌',
        f if f == D | L => '┐',
        f if f == U | D | R => '├',
        f if f == U | D | L => '┤',
        f if f == U | L | R => '┴',
        f if f == D | L | R => '┬',
        f if f == U | D | L | R => '┼',
        _ => ' ',
    }
}

fn render_dependency_dag(
    dependencies: &[Value],
    module_labels: &HashMap<u64, &str>,
    style: &Style,
) -> Vec<String> {
    use std::collections::{BTreeSet, HashSet};

    if dependencies.is_empty() {
        return Vec::new();
    }

    // Build directed graph
    let mut pkgs: BTreeSet<String> = BTreeSet::new();
    let mut fwd: HashMap<String, BTreeSet<String>> = HashMap::new();

    for dep in dependencies {
        let src_id = dep.get("source").and_then(|v| v.as_u64()).unwrap_or(0);
        let tgt_id = dep.get("target").and_then(|v| v.as_u64()).unwrap_or(0);
        let src_label = module_labels
            .get(&src_id)
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("module-{src_id}"));
        let tgt_label = module_labels
            .get(&tgt_id)
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("module-{tgt_id}"));
        pkgs.insert(src_label.clone());
        pkgs.insert(tgt_label.clone());
        fwd.entry(src_label).or_default().insert(tgt_label);
    }
    for p in &pkgs {
        fwd.entry(p.clone()).or_default();
    }

    // Detect back edges via DFS
    let mut back_edges: HashSet<(String, String)> = HashSet::new();
    let mut visited: HashSet<String> = HashSet::new();
    let mut on_stack: HashSet<String> = HashSet::new();

    fn dfs(
        n: &str,
        fwd: &HashMap<String, BTreeSet<String>>,
        visited: &mut HashSet<String>,
        on_stack: &mut HashSet<String>,
        back_edges: &mut HashSet<(String, String)>,
    ) {
        visited.insert(n.to_string());
        on_stack.insert(n.to_string());
        if let Some(successors) = fwd.get(n) {
            for s in successors {
                if on_stack.contains(s.as_str()) {
                    back_edges.insert((n.to_string(), s.clone()));
                } else if !visited.contains(s.as_str()) {
                    dfs(s, fwd, visited, on_stack, back_edges);
                }
            }
        }
        on_stack.remove(n);
    }

    for p in &pkgs {
        if !visited.contains(p.as_str()) {
            dfs(p, &fwd, &mut visited, &mut on_stack, &mut back_edges);
        }
    }

    // Build DAG without back-edges
    let mut dag: HashMap<String, BTreeSet<String>> = HashMap::new();
    let mut rev: HashMap<String, BTreeSet<String>> = HashMap::new();
    for p in &pkgs {
        dag.entry(p.clone()).or_default();
        rev.entry(p.clone()).or_default();
    }
    for (s, targets) in &fwd {
        for t in targets {
            if !back_edges.contains(&(s.clone(), t.clone())) {
                dag.entry(s.clone()).or_default().insert(t.clone());
                rev.entry(t.clone()).or_default().insert(s.clone());
            }
        }
    }

    // Layer assignment (longest path from roots)
    let mut layer_of: HashMap<String, usize> = HashMap::new();
    fn compute_layer(
        n: &str,
        rev: &HashMap<String, BTreeSet<String>>,
        layer_of: &mut HashMap<String, usize>,
    ) -> usize {
        if let Some(&l) = layer_of.get(n) {
            return l;
        }
        layer_of.insert(n.to_string(), 0); // cycle guard
        let l = rev
            .get(n)
            .map(|parents| {
                parents
                    .iter()
                    .map(|p| compute_layer(p, rev, layer_of) + 1)
                    .max()
                    .unwrap_or(0)
            })
            .unwrap_or(0);
        layer_of.insert(n.to_string(), l);
        l
    }

    for p in &pkgs {
        compute_layer(p, &rev, &mut layer_of);
    }

    let n_layers = layer_of.values().max().copied().unwrap_or(0) + 1;
    let mut layers: Vec<Vec<String>> = vec![Vec::new(); n_layers];
    for p in &pkgs {
        layers[layer_of[p.as_str()]].push(p.clone());
    }

    // Assign x-positions
    let gap = 3;
    let mut node_x: HashMap<String, usize> = HashMap::new();
    let mut virt: HashSet<String> = HashSet::new();
    let mut width: usize = 0;

    for lr in &layers {
        let mut x = 0;
        for n in lr {
            let w = if virt.contains(n) { 0 } else { n.len() };
            node_x.insert(n.clone(), x + w / 2);
            x += w.max(1) + gap;
        }
        width = width.max(x);
    }

    // Insert virtual nodes for skip-layer edges
    let mut adj_edges: Vec<(String, String)> = Vec::new();
    let mut vid = 0;
    for src in pkgs.iter() {
        if let Some(targets) = dag.get(src) {
            for tgt in targets {
                let sl = layer_of[src.as_str()];
                let tl = layer_of[tgt.as_str()];
                if tl.wrapping_sub(sl) == 1 {
                    adj_edges.push((src.clone(), tgt.clone()));
                } else {
                    let mut prev = src.clone();
                    for lyr in (sl + 1)..tl {
                        let vn = format!("\x00v{vid}");
                        vid += 1;
                        virt.insert(vn.clone());
                        layer_of.insert(vn.clone(), lyr);
                        layers[lyr].push(vn.clone());
                        node_x.insert(vn.clone(), node_x[&prev]);
                        adj_edges.push((prev.clone(), vn.clone()));
                        prev = vn;
                    }
                    adj_edges.push((prev, tgt.clone()));
                }
            }
        }
    }

    // Re-compute x after adding virtual nodes
    node_x.clear();
    width = 0;
    for lr in &layers {
        let mut x = 0;
        for n in lr {
            let w = if virt.contains(n) { 0 } else { n.len() };
            node_x.insert(n.clone(), x + w / 2);
            x += w.max(1) + gap;
        }
        width = width.max(x);
    }

    // Render
    let mut out: Vec<String> = Vec::new();

    for li in 0..n_layers {
        // Node name row
        let mut row = vec![' '; width];
        for n in &layers[li] {
            if virt.contains(n) {
                continue;
            }
            let w = n.len();
            let sx = node_x[n].saturating_sub(w / 2);
            for (i, ch) in n.chars().enumerate() {
                if sx + i < width {
                    row[sx + i] = ch;
                }
            }
        }
        let text: String = row.iter().collect::<String>().trim_end().to_string();
        if !text.trim().is_empty() {
            out.push(format!("  {text}"));
        }

        if li >= n_layers - 1 {
            continue;
        }

        // Edges from this layer to next
        let gap_edges: Vec<&(String, String)> = adj_edges
            .iter()
            .filter(|(s, _)| layer_of.get(s).copied() == Some(li))
            .collect();
        if gap_edges.is_empty() {
            continue;
        }

        // Drop row
        let mut drop_row = vec![' '; width];
        for (s, _) in &gap_edges {
            let x = node_x[s.as_str()];
            if x < width {
                drop_row[x] = '│';
            }
        }
        let ds: String = drop_row.iter().collect::<String>().trim_end().to_string();
        if !ds.trim().is_empty() {
            out.push(format!("  {ds}"));
        }

        // Routing row
        let mut flags = vec![0u8; width];
        for (s, t) in &gap_edges {
            let sx = node_x[s.as_str()];
            let tx = node_x[t.as_str()];
            if sx == tx {
                if sx < width {
                    flags[sx] |= U | D;
                }
            } else {
                if sx < width {
                    flags[sx] |= U | if tx > sx { R } else { L };
                }
                if tx < width {
                    flags[tx] |= D | if tx > sx { L } else { R };
                }
                let (lo, hi) = if sx < tx { (sx, tx) } else { (tx, sx) };
                for x in (lo + 1)..hi {
                    if x < width {
                        flags[x] |= L | R;
                    }
                }
            }
        }
        let rrow: String = flags
            .iter()
            .map(|&f| if f != 0 { box_char(f) } else { ' ' })
            .collect::<String>()
            .trim_end()
            .to_string();
        if !rrow.trim().is_empty() {
            out.push(format!("  {rrow}"));
        }

        // Rise row
        let mut rise = vec![' '; width];
        for (_, t) in &gap_edges {
            let x = node_x[t.as_str()];
            if x < width {
                rise[x] = if virt.contains(t.as_str()) {
                    '│'
                } else {
                    '▼'
                };
            }
        }
        let ris: String = rise.iter().collect::<String>().trim_end().to_string();
        if !ris.trim().is_empty() {
            out.push(format!("  {ris}"));
        }
    }

    // Cycle annotations
    let mut sorted_back: Vec<_> = back_edges.into_iter().collect();
    sorted_back.sort();
    for (s, t) in sorted_back {
        let cycle_icon = style.red_text("⟲");
        let cycle_label = style.dim("(cycle)");
        out.push(format!("  {cycle_icon} {s} → {t} {cycle_label}"));
    }

    out
}
