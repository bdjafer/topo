mod args;
mod cache;
#[cfg(feature = "semantic")]
mod embed;
mod export;
mod health;
mod policy;

use std::collections::HashMap;
use std::io::{IsTerminal, Write};
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::Parser;
use serde_json::Value;

use args::{CacheCommand, Cli, Command};

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();

    match &cli.command {
        Command::Analyze(args) => cmd_analyze(args, &cli),
        Command::Domain(args) => cmd_domain(args, &cli),
        Command::Serve(args) => cmd_serve(args),
        Command::Parse(args) => cmd_parse(args, &cli),
        Command::Cache(args) => cmd_cache(args),
        Command::AnalyzeRaw(args) => cmd_analyze_raw(args),
        Command::Mutate(args) => cmd_mutate(args),
        Command::ExportFeatures(args) => export::cmd_export_features(args, &cli),
    }
}

// ── Shared analysis pipeline ──

/// Run the full analysis pipeline: validate → parse → analyze → return structured output.
fn analyze_codebase(
    path: &Path,
    cli: &Cli,
) -> Result<(Value, std::path::PathBuf, policy::Policy)> {
    if !path.is_dir() {
        bail!("{} is not a directory", path.display());
    }

    let tty = std::io::stderr().is_terminal();
    let project_root = path.canonicalize()?;
    let policy = policy::load_policy(&project_root)
        .ok()
        .flatten()
        .unwrap_or_default();

    // Resolve settings from policy (CLI has no per-analysis flags — policy is the config layer).
    let scope = policy.scope.as_deref().unwrap_or("auto");
    let language = policy.language.as_deref();
    let exclude = policy
        .exclude
        .as_ref()
        .map(|v| v.join(","));
    let level = policy.level.as_deref().unwrap_or("module");
    let edge_kind = policy.edge_kind.as_deref().unwrap_or("combined");
    let n_modules = policy.n_modules;

    // Parse
    status(tty, "Parsing...");
    let t0 = Instant::now();
    let (graph_json, cache_hit) = cache::cached_parse(
        path,
        exclude.as_deref(),
        Some(scope),
        language,
        cli.no_cache,
    )?;
    let parse_label = if cache_hit { "Cached" } else { "Parsed" };
    status(
        tty,
        &format!(
            "{parse_label} in {:.1}s. Analyzing...",
            t0.elapsed().as_secs_f64()
        ),
    );

    // Build analyzer input
    let graph: Value = serde_json::from_str(&graph_json).context("Invalid graph JSON")?;
    let nodes = graph
        .get("nodes")
        .cloned()
        .unwrap_or(Value::Array(vec![]));
    let edges = graph
        .get("edges")
        .cloned()
        .unwrap_or(Value::Array(vec![]));

    let (edge_kinds, layer_weights) = if edge_kind == "combined" {
        (
            Some(vec![
                "calls".to_string(),
                "imports".to_string(),
                "inherits".to_string(),
                "defines".to_string(),
            ]),
            Some({
                let mut m = HashMap::new();
                m.insert("calls".to_string(), 1.0);
                m.insert("imports".to_string(), 0.5);
                m.insert("inherits".to_string(), 0.8);
                m.insert("defines".to_string(), 0.2);
                m
            }),
        )
    } else {
        (Some(vec![edge_kind.to_string()]), None)
    };

    let parsed_nodes = nodes.as_array().map(|a| a.len()).unwrap_or(0);
    let parsed_edges = edges.as_array().map(|a| a.len()).unwrap_or(0);

    let packages: Vec<String> = nodes
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter(|n| n.get("kind").and_then(|v| v.as_str()) == Some("module"))
                .filter_map(|n| n.get("id").and_then(|v| v.as_str()))
                .filter(|id| !id.contains('.'))
                .map(|id| id.to_string())
                .collect()
        })
        .unwrap_or_default();
    let packages_opt = if packages.len() >= 2 {
        Some(packages)
    } else {
        None
    };

    let mut analyzer_input = serde_json::json!({
        "nodes": nodes,
        "edges": edges,
        "k": n_modules,
        "edge_kinds": edge_kinds,
        "layer_weights": layer_weights,
        "scope": {
            "level": level,
            "edge_kinds": edge_kinds.clone().unwrap_or_default(),
            "internal_only": true,
            "roots": []
        },
        "parsed_nodes": parsed_nodes,
        "parsed_edges": parsed_edges,
        "packages": packages_opt,
    });

    // Semantic embeddings: pre-computed file or auto-generate.
    if let Some(ref emb_path) = cli.embeddings {
        let emb_json = std::fs::read_to_string(emb_path)
            .map_err(|e| anyhow::anyhow!("Failed to read embeddings file: {e}"))?;
        let emb_data: Value = serde_json::from_str(&emb_json)
            .map_err(|e| anyhow::anyhow!("Invalid embeddings JSON: {e}"))?;
        analyzer_input["semantic_embeddings"] = emb_data;
    } else {
        #[cfg(feature = "semantic")]
        {
            eprintln!("Generating semantic embeddings...");
            let graph_value: Value = serde_json::from_str(&graph_json)?;
            let embeddings = embed::generate_embeddings(&graph_value, &project_root)?;
            let emb_value = serde_json::to_value(&embeddings)?;
            analyzer_input["semantic_embeddings"] = emb_value;
        }
        #[cfg(not(feature = "semantic"))]
        {
            eprintln!("Note: semantic analysis requires the `semantic` feature or --embeddings <file>.");
        }
    }

    let input_str = serde_json::to_string(&analyzer_input)?;
    let output_str = topo_analyzer::analyze_full_json(&input_str)
        .map_err(|e| anyhow::anyhow!("Analysis failed: {e}"))?;

    clear_status(tty);
    eprintln!(
        "Done in {:.1}s",
        t0.elapsed().as_secs_f64()
    );

    let data: Value = serde_json::from_str(&output_str)?;
    Ok((data, project_root, policy))
}

// ── Analyze command (issues + health) ──

fn cmd_analyze(args: &args::AnalyzeArgs, cli: &Cli) -> Result<()> {
    // --history mode: health over git history (no issues)
    let history_mode = args.history || args.since.is_some();

    if history_mode {
        let language = policy::load_policy(&args.path)
            .ok()
            .flatten()
            .and_then(|p| p.language);

        eprintln!("Tracking structural health over git history...");
        let snapshots = health::run_health(
            &args.path,
            args.since.as_deref(),
            &args.sample,
            args.max_commits,
            language.as_deref(),
        )?;

        if cli.json {
            println!("{}", serde_json::to_string_pretty(&snapshots)?);
        } else {
            println!("{}", health::format_health_text(&snapshots));
        }
        return Ok(());
    }

    // Default: issues + health snapshot
    let (data, project_root, policy) = analyze_codebase(&args.path, cli)?;

    if cli.json {
        let issues = data.get("issues").cloned().unwrap_or(Value::Array(vec![]));
        let filtered = filter_issues_by_severity(&issues, args.severity.as_deref());
        let health_data = data.get("health").cloned().unwrap_or(Value::Null);
        let output = serde_json::json!({
            "issues": filtered,
            "health": health_data,
        });
        println!("{}", serde_json::to_string_pretty(&output)?);
        let count = filtered.as_array().map(|a| a.len()).unwrap_or(0);
        if count > 0 {
            std::process::exit(1);
        }
    } else {
        let use_color = !cli.no_color && atty_stdout();
        let (issues_text, count) = topo_formatter::format_issues(
            &data,
            &policy.ignores,
            Some(&project_root),
            use_color,
        );
        let health_text = topo_formatter::format_health(&data, use_color);
        print!("{issues_text}{health_text}");
        if count > 0 {
            std::process::exit(1);
        }
    }

    Ok(())
}

/// Filter issues by minimum severity (cumulative: "medium" means high+medium).
fn filter_issues_by_severity(issues: &Value, severity: Option<&str>) -> Value {
    let Some(arr) = issues.as_array() else {
        return Value::Array(vec![]);
    };

    let min_severity = match severity {
        Some("high") => 2,
        Some("medium") => 1,
        _ => 0, // "low" or None — show all
    };

    let filtered: Vec<Value> = arr
        .iter()
        .filter(|issue| {
            let sev = issue
                .get("severity_label")
                .and_then(|v| v.as_str())
                .unwrap_or("low");
            let sev_rank = match sev {
                "high" => 2,
                "medium" => 1,
                _ => 0,
            };
            sev_rank >= min_severity
        })
        .cloned()
        .collect();

    Value::Array(filtered)
}

// ── Domain command ──

fn cmd_domain(_args: &args::DomainArgs, _cli: &Cli) -> Result<()> {
    bail!("topo domain is not yet implemented. Awaiting Phase 3 domain decomposition.");
}

// ── Serve command (stub) ──

fn cmd_serve(args: &args::ServeArgs) -> Result<()> {
    eprintln!("topo serve is not yet implemented.");
    eprintln!("  path: {}", args.path.display());
    eprintln!("  port: {}", args.port);
    eprintln!();
    eprintln!("This command will start a web server showing the full domain");
    eprintln!("with health metrics per domain/sub-domain and issues per domain.");
    std::process::exit(1);
}

// ── Parse command ──

fn cmd_parse(args: &args::ParseArgs, cli: &Cli) -> Result<()> {
    if !args.path.is_dir() {
        bail!("{} is not a directory", args.path.display());
    }

    let (graph_json, _hit) = cache::cached_parse(
        &args.path,
        args.exclude.as_deref(),
        None,
        args.language.as_deref(),
        cli.no_cache,
    )?;

    if let Some(output) = &args.output {
        std::fs::write(output, &graph_json)
            .with_context(|| format!("Failed to write {}", output.display()))?;

        if let Ok(parsed) = serde_json::from_str::<Value>(&graph_json) {
            let nodes = parsed
                .get("nodes")
                .and_then(|v| v.as_array())
                .map(|a| a.len())
                .unwrap_or(0);
            let edges = parsed
                .get("edges")
                .and_then(|v| v.as_array())
                .map(|a| a.len())
                .unwrap_or(0);
            eprintln!("Wrote {nodes} nodes, {edges} edges to {}", output.display());
        }
    } else {
        print!("{graph_json}");
    }

    Ok(())
}

// ── Cache command ──

fn cmd_cache(args: &args::CacheArgs) -> Result<()> {
    match &args.command {
        CacheCommand::Clear { path } => {
            cache::clear_cache(path)?;
            eprintln!("Cache cleared.");
            Ok(())
        }
    }
}

// ── Analyze-raw command (pre-parsed graph, benchmark tooling) ──

fn cmd_analyze_raw(args: &args::AnalyzeRawArgs) -> Result<()> {
    let input_json = std::fs::read_to_string(&args.input)
        .with_context(|| format!("Failed to read {}", args.input.display()))?;
    let output_json = topo_analyzer::analyze_full_json(&input_json)
        .map_err(|e| anyhow::anyhow!("Analysis failed: {e}"))?;
    println!("{output_json}");
    Ok(())
}

// ── Mutate command ──

fn cmd_mutate(args: &args::MutateArgs) -> Result<()> {
    use topo_benchmark::mutations::{MutationType, apply_mutation};

    let graph_json = std::fs::read_to_string(&args.input)
        .with_context(|| format!("Failed to read {}", args.input.display()))?;
    let input: topo_analyzer::types::AnalyzerInput = serde_json::from_str(&graph_json)
        .context("Failed to parse graph JSON")?;

    let analysis = topo_analyzer::analyze_full(&input);

    let mutation_type = match args.mutation_type.as_str() {
        "inject_cycle" => MutationType::InjectCycle,
        "layer_violation" => MutationType::LayerViolation,
        "overloaded_utility" => MutationType::OverloadedUtility,
        "wide_interface" => MutationType::WideInterface,
        "near_disconnect" => MutationType::NearDisconnect,
        other => bail!("Unknown mutation type: {other}"),
    };

    match apply_mutation(&input, &analysis, mutation_type, args.severity, args.seed) {
        Some(result) => {
            let output = serde_json::json!({
                "graph": result.graph,
                "mutation_type": result.mutation_type,
                "expected_diagnostic": result.expected_diagnostic,
                "severity_level": result.severity_level,
                "seed": result.seed,
                "added_edges": result.added_edges.len(),
                "removed_edges": result.removed_edges.len(),
                "modified_region": result.modified_region,
                "description": result.description,
            });
            println!("{}", serde_json::to_string(&output)?);
        }
        None => {
            eprintln!("Mutation returned None: graph lacks preconditions for {:?}", mutation_type);
            std::process::exit(2);
        }
    }

    Ok(())
}

// ── Helpers ──

fn atty_stdout() -> bool {
    std::io::stdout().is_terminal()
}

fn status(tty: bool, msg: &str) {
    if tty {
        eprint!("\r\x1b[2K{msg}");
        let _ = std::io::stderr().flush();
    }
}

fn clear_status(tty: bool) {
    if tty {
        eprint!("\r\x1b[2K");
        let _ = std::io::stderr().flush();
    }
}
