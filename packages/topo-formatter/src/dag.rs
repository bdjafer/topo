//! ASCII DAG renderer for module dependency graphs.

use std::collections::{BTreeSet, HashMap, HashSet};

use serde_json::Value;

use crate::style::Style;

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

pub fn render_dependency_dag(
    dependencies: &[Value],
    module_labels: &HashMap<u64, &str>,
    style: &Style,
) -> Vec<String> {
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
