//! Structural health tracking over git history.
//!
//! `topo health <path>` walks git history, runs structural analysis at sampled
//! commits, and outputs the trajectory of key metrics.

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// A single health snapshot at a commit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthSnapshot {
    pub commit: String,
    pub date: String,
    pub message: String,
    pub fiedler_value: Option<f64>,
    pub modularity_q: Option<f64>,
    pub num_modules: usize,
    pub num_nodes: usize,
    pub num_edges: usize,
}

/// Run health tracking over git history.
pub fn run_health(
    path: &Path,
    since: Option<&str>,
    sample: &str,
    max_commits: usize,
    language: Option<&str>,
) -> Result<Vec<HealthSnapshot>> {
    // Get commit list.
    let commits = list_commits(path, since, sample, max_commits)?;
    if commits.is_empty() {
        return Ok(Vec::new());
    }

    let original_head = git_current_ref(path)?;
    let mut snapshots = Vec::new();

    // Stash any dirty working tree.
    let has_changes = !git_output(path, &["status", "--porcelain"])?.trim().is_empty();
    if has_changes {
        git_run(path, &["stash", "push", "-m", "topo-health-temp"])?;
    }

    for (hash, date, message) in &commits {
        // Checkout the commit.
        if git_run(path, &["checkout", "--quiet", hash]).is_err() {
            continue; // Skip commits that can't be checked out.
        }

        // Try to parse and analyze.
        let snapshot = match analyze_at_commit(path, hash, date, message, language) {
            Ok(s) => s,
            Err(_) => {
                // Non-compiling commit or parse failure — record with None metrics.
                HealthSnapshot {
                    commit: hash[..7.min(hash.len())].to_string(),
                    date: date.clone(),
                    message: message.clone(),
                    fiedler_value: None,
                    modularity_q: None,
                    num_modules: 0,
                    num_nodes: 0,
                    num_edges: 0,
                }
            }
        };
        snapshots.push(snapshot);
    }

    // Restore original HEAD.
    if let Err(e) = git_run(path, &["checkout", "--quiet", &original_head]) {
        eprintln!("warning: failed to restore HEAD to {original_head}: {e}");
        eprintln!("         your working tree may be on an unexpected commit.");
    }
    if has_changes {
        if let Err(e) = git_run(path, &["stash", "pop"]) {
            eprintln!("warning: failed to restore stashed changes: {e}");
            eprintln!("         run 'git stash pop' manually to recover your changes.");
        }
    }

    Ok(snapshots)
}

/// Format health snapshots as a text table.
pub fn format_health_text(snapshots: &[HealthSnapshot]) -> String {
    if snapshots.is_empty() {
        return "No commits analyzed.".to_string();
    }

    let mut lines = Vec::new();
    lines.push(format!(
        "{:<10} {:<12} {:>8} {:>8} {:>6} {:>6}  {}",
        "commit", "date", "λ₂", "Q", "mods", "nodes", "message"
    ));
    lines.push("-".repeat(75));

    for s in snapshots {
        let fiedler = s.fiedler_value
            .map(|v| format!("{v:.4}"))
            .unwrap_or_else(|| "n/a".to_string());
        let q = s.modularity_q
            .map(|v| format!("{v:.3}"))
            .unwrap_or_else(|| "n/a".to_string());
        let msg = truncate_str(&s.message, 30);
        lines.push(format!(
            "{:<10} {:<12} {:>8} {:>8} {:>6} {:>6}  {}",
            s.commit, s.date, fiedler, q, s.num_modules, s.num_nodes, msg
        ));
    }

    // Trend indicators.
    if snapshots.len() >= 3 {
        let fiedler_vals: Vec<f64> = snapshots.iter()
            .filter_map(|s| s.fiedler_value)
            .collect();
        if fiedler_vals.len() >= 3 {
            let first = fiedler_vals[0];
            let last = fiedler_vals[fiedler_vals.len() - 1];
            let trend = if last > first * 1.1 {
                "increasing (growing coupling)"
            } else if last < first * 0.9 {
                "decreasing (structural fragmentation)"
            } else {
                "stable"
            };
            lines.push(String::new());
            lines.push(format!("Fiedler trend: {trend}"));
        }
    }

    lines.join("\n")
}

// ── Git helpers ──

fn git_output(path: &Path, args: &[&str]) -> Result<String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(path)
        .output()
        .context("failed to run git")?;
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn git_run(path: &Path, args: &[&str]) -> Result<()> {
    let status = Command::new("git")
        .args(args)
        .current_dir(path)
        .output()
        .context("failed to run git")?;
    if status.status.success() {
        Ok(())
    } else {
        anyhow::bail!("git {:?} failed", args);
    }
}

fn git_current_ref(path: &Path) -> Result<String> {
    let branch = git_output(path, &["symbolic-ref", "--short", "HEAD"]);
    match branch {
        Ok(b) if !b.trim().is_empty() => Ok(b.trim().to_string()),
        _ => {
            // Detached HEAD — return commit hash.
            let hash = git_output(path, &["rev-parse", "HEAD"])?;
            Ok(hash.trim().to_string())
        }
    }
}

/// List commits matching the sampling strategy.
fn list_commits(
    path: &Path,
    since: Option<&str>,
    sample: &str,
    max_commits: usize,
) -> Result<Vec<(String, String, String)>> {
    let mut args = vec![
        "log".to_string(),
        "--format=%H|%as|%s".to_string(),
        "--reverse".to_string(),
    ];
    if let Some(since) = since {
        args.push(format!("--since={since}"));
    }
    args.push(format!("--max-count={}", max_commits * 5)); // Oversample for filtering.

    let arg_refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    let output = git_output(path, &arg_refs)?;

    let all_commits: Vec<(String, String, String)> = output
        .lines()
        .filter(|l| !l.is_empty())
        .filter_map(|line| {
            let parts: Vec<&str> = line.splitn(3, '|').collect();
            if parts.len() == 3 {
                Some((
                    parts[0].to_string(),
                    parts[1].to_string(),
                    parts[2].to_string(),
                ))
            } else {
                None
            }
        })
        .collect();

    // Sample based on strategy.
    let sampled = match sample {
        "weekly" => sample_by_period(&all_commits, 7),
        "monthly" => sample_by_period(&all_commits, 30),
        _ => all_commits.clone(),
    };

    Ok(sampled.into_iter().take(max_commits).collect())
}

/// Sample commits by taking the latest in each N-day window.
fn sample_by_period(
    commits: &[(String, String, String)],
    days: u32,
) -> Vec<(String, String, String)> {
    if commits.is_empty() {
        return Vec::new();
    }

    let mut result = Vec::new();
    let mut last_date: Option<&str> = None;

    for commit in commits {
        let date = &commit.1;
        let should_include = match last_date {
            None => true,
            Some(prev) => {
                // Simple date distance check (YYYY-MM-DD format).
                date_distance_days(prev, date) >= days
            }
        };

        if should_include {
            result.push(commit.clone());
            last_date = Some(&commit.1);
        }
    }

    // Always include the last commit.
    if let Some(last) = commits.last() {
        if result.last().map(|r| &r.0) != Some(&last.0) {
            result.push(last.clone());
        }
    }

    result
}

/// Approximate day distance between two YYYY-MM-DD dates.
fn date_distance_days(a: &str, b: &str) -> u32 {
    let parse = |s: &str| -> Option<(i32, u32, u32)> {
        let parts: Vec<&str> = s.split('-').collect();
        if parts.len() != 3 { return None; }
        let y = parts[0].parse().ok()?;
        let m = parts[1].parse().ok()?;
        let d = parts[2].parse().ok()?;
        Some((y, m, d))
    };

    match (parse(a), parse(b)) {
        (Some((y1, m1, d1)), Some((y2, m2, d2))) => {
            let days1 = y1 * 365 + m1 as i32 * 30 + d1 as i32;
            let days2 = y2 * 365 + m2 as i32 * 30 + d2 as i32;
            (days2 - days1).unsigned_abs()
        }
        _ => 0,
    }
}

/// Parse and analyze a project at the current checkout.
fn analyze_at_commit(
    path: &Path,
    hash: &str,
    date: &str,
    message: &str,
    language: Option<&str>,
) -> Result<HealthSnapshot> {
    // Parse.
    let graph_json = topo_parser::parse_project(
        path,
        None,
        Some("first-party"),
        language,
    )?;

    // Analyze.
    let output_str = topo_analyzer::analyze_full_json(&graph_json)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let output: serde_json::Value = serde_json::from_str(&output_str)?;

    let fiedler = output.pointer("/spectral/fiedler_value")
        .and_then(|v| v.as_f64());
    let modularity_q = output.pointer("/health/modularity_q")
        .and_then(|v| v.as_f64());
    let num_modules = output.pointer("/architecture/modules")
        .and_then(|v| v.as_array())
        .map(|a| a.len())
        .unwrap_or(0);
    let num_nodes = output.pointer("/coverage/analyzed_nodes")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let num_edges = output.pointer("/coverage/analyzed_edges")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;

    Ok(HealthSnapshot {
        commit: hash[..7.min(hash.len())].to_string(),
        date: date.to_string(),
        message: truncate_str(message, 60).to_string(),
        fiedler_value: fiedler,
        modularity_q,
        num_modules,
        num_nodes,
        num_edges,
    })
}

/// Truncate a string to at most `max_chars` characters, safe for multi-byte UTF-8.
fn truncate_str(s: &str, max_chars: usize) -> &str {
    match s.char_indices().nth(max_chars) {
        Some((byte_idx, _)) => &s[..byte_idx],
        None => s,
    }
}
