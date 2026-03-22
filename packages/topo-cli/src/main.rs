mod args;
mod cache;
#[cfg(feature = "semantic")]
mod embed;
mod health;
mod policy;

use std::collections::HashMap;
use std::io::{Write, IsTerminal};
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::Parser;

use args::{AnalysisArgs, AnalyzeArgs, CacheCommand, Cli, Command, ParseArgs};

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(Command::Parse(args)) => cmd_parse(args, cli.no_cache),
        Some(Command::Analyze(args)) => cmd_analyze(args, cli.no_cache),
        Some(Command::Cache(args)) => cmd_cache(args),
        Some(Command::Health(args)) => cmd_health(args),
        None => {
            let path = cli
                .path
                .ok_or_else(|| anyhow::anyhow!("Missing required argument: <PATH>"))?;
            cmd_default(&path, cli.exclude.as_deref(), cli.scope.as_deref(), cli.language.as_deref(), &cli.analysis, cli.no_cache)
        }
    }
}

// ── Parse command ──

fn cmd_parse(args: ParseArgs, no_cache: bool) -> Result<()> {
    if !args.path.is_dir() {
        bail!("{} is not a directory", args.path.display());
    }

    let (graph_json, _hit) = cache::cached_parse(
        &args.path,
        args.exclude.as_deref(),
        args.scope.as_deref(),
        args.language.as_deref(),
        no_cache,
    )?;

    if let Some(output) = &args.output {
        std::fs::write(output, &graph_json)
            .with_context(|| format!("Failed to write {}", output.display()))?;

        // Count nodes/edges for status message
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&graph_json) {
            let nodes = parsed.get("nodes").and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0);
            let edges = parsed.get("edges").and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0);
            eprintln!("Wrote {nodes} nodes, {edges} edges to {}", output.display());
        }
    } else {
        print!("{graph_json}");
    }

    Ok(())
}

// ── Analyze command ──

fn cmd_analyze(args: AnalyzeArgs, no_cache: bool) -> Result<()> {
    let (graph_json, project_root) = match (&args.path, &args.input) {
        (Some(path), None) => {
            // Auto-parse, then analyze.
            if !path.is_dir() {
                bail!("{} is not a directory", path.display());
            }
            let (json, _hit) = cache::cached_parse(
                path,
                args.exclude.as_deref(),
                args.scope.as_deref(),
                args.language.as_deref(),
                no_cache,
            )?;
            (json, Some(path.canonicalize()?))
        }
        (None, Some(input)) => {
            // Pre-parsed graph
            if !input.is_file() {
                bail!("{} is not a file", input.display());
            }
            let json = std::fs::read_to_string(input)
                .with_context(|| format!("Failed to read {}", input.display()))?;
            (json, None)
        }
        (Some(_), Some(_)) => {
            bail!("Cannot specify both <PATH> and --input. Use one or the other.");
        }
        (None, None) => {
            bail!("Provide either a <PATH> to analyze or --input <graph.json>.");
        }
    };

    let policy = project_root
        .as_deref()
        .and_then(|p| policy::load_policy(p).ok().flatten());

    run_analysis(
        &graph_json,
        &args.analysis,
        project_root.as_deref(),
        policy.as_ref(),
    )
}

// ── Default command (parse + analyze) ──

fn cmd_default(
    path: &Path,
    exclude: Option<&str>,
    scope: Option<&str>,
    language: Option<&str>,
    analysis: &AnalysisArgs,
    no_cache: bool,
) -> Result<()> {
    if !path.is_dir() {
        bail!("{} is not a directory", path.display());
    }

    let tty = std::io::stderr().is_terminal();
    let policy = policy::load_policy(path).ok().flatten();

    // Resolve scope from CLI or policy
    let effective_scope = scope
        .or(policy.as_ref().and_then(|p| p.scope.as_deref()))
        .unwrap_or("auto");

    status(tty, "Parsing...");
    let t0 = Instant::now();
    let (graph_json, cache_hit) = cache::cached_parse(path, exclude, Some(effective_scope), language, no_cache)?;
    let parse_label = if cache_hit { "Cached" } else { "Parsed" };
    status(tty, &format!("{parse_label} in {:.1}s. Analyzing...", t0.elapsed().as_secs_f64()));

    let project_root = path.canonicalize()?;
    let t1 = Instant::now();
    let result = run_analysis(&graph_json, analysis, Some(&project_root), policy.as_ref());
    clear_status(tty);
    eprintln!("Done in {:.1}s (parse {:.1}s + analyze {:.1}s)",
        t0.elapsed().as_secs_f64(), t0.elapsed().as_secs_f64() - t1.elapsed().as_secs_f64(), t1.elapsed().as_secs_f64());
    result
}

// ── Cache command ──

fn cmd_cache(args: args::CacheArgs) -> Result<()> {
    match args.command {
        CacheCommand::Clear { path } => {
            cache::clear_cache(&path)?;
            eprintln!("Cache cleared.");
            Ok(())
        }
    }
}

// ── Health command ──

fn cmd_health(args: args::HealthArgs) -> Result<()> {
    eprintln!("Tracking structural health over git history...");

    let snapshots = health::run_health(
        &args.path,
        args.since.as_deref(),
        &args.sample,
        args.max_commits,
        args.language.as_deref(),
    )?;

    if args.as_json {
        println!("{}", serde_json::to_string_pretty(&snapshots)?);
    } else {
        println!("{}", health::format_health_text(&snapshots));
    }

    Ok(())
}

// ── Shared analysis + output ──

fn run_analysis(
    graph_json: &str,
    args: &AnalysisArgs,
    project_root: Option<&Path>,
    policy: Option<&policy::Policy>,
) -> Result<()> {
    // Build analyzer input from graph JSON + CLI options
    let graph: serde_json::Value =
        serde_json::from_str(graph_json).context("Invalid graph JSON")?;

    let nodes = graph.get("nodes").cloned().unwrap_or(serde_json::Value::Array(vec![]));
    let edges = graph.get("edges").cloned().unwrap_or(serde_json::Value::Array(vec![]));

    let level = resolve_level(args.level.as_deref(), policy);

    // Determine edge kinds and layer weights
    let (edge_kinds, layer_weights) = if args.edge_kind == "combined" {
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
        (Some(vec![args.edge_kind.clone()]), None)
    };

    let parsed_nodes = nodes.as_array().map(|a| a.len()).unwrap_or(0);
    let parsed_edges = edges.as_array().map(|a| a.len()).unwrap_or(0);

    // Extract package names from top-level module nodes (kind == "module"
    // with no '.' in the ID, i.e., crate roots).
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

    // Build the AnalyzerInput JSON
    let mut analyzer_input = serde_json::json!({
        "nodes": nodes,
        "edges": edges,
        "k": args.n_modules,
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

    // Load pre-computed semantic embeddings if provided.
    if let Some(ref emb_path) = args.embeddings {
        let emb_json = std::fs::read_to_string(emb_path)
            .map_err(|e| anyhow::anyhow!("Failed to read embeddings file: {e}"))?;
        let emb_data: serde_json::Value = serde_json::from_str(&emb_json)
            .map_err(|e| anyhow::anyhow!("Invalid embeddings JSON: {e}"))?;
        analyzer_input["semantic_embeddings"] = emb_data;
    } else if !args.no_semantic {
        #[cfg(feature = "semantic")]
        {
            if let Some(root) = project_root {
                eprintln!("Generating semantic embeddings...");
                let graph_value: serde_json::Value = serde_json::from_str(&graph_json)?;
                let embeddings = embed::generate_embeddings(&graph_value, root)?;
                let emb_value = serde_json::to_value(&embeddings)?;
                analyzer_input["semantic_embeddings"] = emb_value;
            } else {
                eprintln!("Warning: semantic analysis requires a project path. Use --embeddings <file> for pre-parsed graphs. Continuing without semantics.");
            }
        }
        #[cfg(not(feature = "semantic"))]
        {
            eprintln!("Note: semantic analysis requires the `semantic` feature. Build with: cargo build -p topo-cli --features semantic");
            eprintln!("Or use --embeddings <file> for pre-computed embeddings. Continuing without semantics.");
        }
    }

    if args.experimental {
        analyzer_input["experimental"] = serde_json::Value::Bool(true);
    }

    let input_str = serde_json::to_string(&analyzer_input)?;
    let output_str = topo_analyzer::analyze_full_json(&input_str)
        .map_err(|e| anyhow::anyhow!("Analysis failed: {e}"))?;

    let format = args.format.as_deref().unwrap_or(if args.as_json { "json" } else { "text" });
    match format {
        "json" => {
            let output: serde_json::Value = serde_json::from_str(&output_str)?;
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        "context" => {
            let data: serde_json::Value = serde_json::from_str(&output_str)?;
            let text = topo_formatter::format_context(&data);
            print!("{text}");
        }
        "domain" => {
            let data: serde_json::Value = serde_json::from_str(&output_str)?;
            let text = topo_formatter::domain::format_domain(&data);
            print!("{text}");
        }
        _ => {
            // "text" or default
            let data: serde_json::Value = serde_json::from_str(&output_str)?;
            let ignores = policy.map(|p| &p.ignores).cloned().unwrap_or_default();
            let use_color = !args.no_color && atty_stdout();
            let text = topo_formatter::format_text(
                &data,
                args.verbose,
                args.diagnostics,
                &ignores,
                project_root,
                use_color,
            );
            print!("{text}");
        }
    }

    Ok(())
}

fn resolve_level<'a>(level: Option<&'a str>, policy: Option<&'a policy::Policy>) -> &'a str {
    if let Some(l) = level {
        return l;
    }
    if let Some(p) = policy {
        if let Some(l) = &p.level {
            return l.as_str();
        }
    }
    "module"
}

fn atty_stdout() -> bool {
    std::io::stdout().is_terminal()
}

/// Print a transient status line on stderr (overwritten by next status or cleared).
fn status(tty: bool, msg: &str) {
    if tty {
        eprint!("\r\x1b[2K{msg}");
        let _ = std::io::stderr().flush();
    }
}

/// Clear the transient status line.
fn clear_status(tty: bool) {
    if tty {
        eprint!("\r\x1b[2K");
        let _ = std::io::stderr().flush();
    }
}
