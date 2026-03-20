mod args;
mod format;
mod parse;
mod policy;

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use clap::Parser;

use args::{AnalysisArgs, AnalyzeArgs, Cli, Command, ParseArgs};

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(Command::Parse(args)) => cmd_parse(args),
        Some(Command::Analyze(args)) => cmd_analyze(args),
        None => {
            let path = cli
                .path
                .ok_or_else(|| anyhow::anyhow!("Missing required argument: <PATH>"))?;
            cmd_default(&path, cli.exclude.as_deref(), cli.scope.as_deref(), cli.language.as_deref(), &cli.analysis)
        }
    }
}

// ── Parse command ──

fn cmd_parse(args: ParseArgs) -> Result<()> {
    if !args.path.is_dir() {
        bail!("{} is not a directory", args.path.display());
    }

    let graph_json = parse::parse_project(
        &args.path,
        args.exclude.as_deref(),
        args.scope.as_deref(),
        args.language.as_deref(),
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

fn cmd_analyze(args: AnalyzeArgs) -> Result<()> {
    let (graph_json, project_root) = match (&args.path, &args.input) {
        (Some(path), None) => {
            // Auto-parse, then analyze.
            if !path.is_dir() {
                bail!("{} is not a directory", path.display());
            }
            let json = parse::parse_project(
                path,
                args.exclude.as_deref(),
                args.scope.as_deref(),
                args.language.as_deref(),
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
) -> Result<()> {
    if !path.is_dir() {
        bail!("{} is not a directory", path.display());
    }

    let policy = policy::load_policy(path).ok().flatten();

    // Resolve scope from CLI or policy
    let effective_scope = scope
        .or(policy.as_ref().and_then(|p| p.scope.as_deref()))
        .unwrap_or("auto");

    let graph_json = parse::parse_project(path, exclude, Some(effective_scope), language)?;
    let project_root = path.canonicalize()?;

    run_analysis(&graph_json, analysis, Some(&project_root), policy.as_ref())
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
            ]),
            Some({
                let mut m = HashMap::new();
                m.insert("calls".to_string(), 1.0);
                m.insert("imports".to_string(), 0.5);
                m.insert("inherits".to_string(), 0.8);
                m
            }),
        )
    } else {
        (Some(vec![args.edge_kind.clone()]), None)
    };

    let parsed_nodes = nodes.as_array().map(|a| a.len()).unwrap_or(0);
    let parsed_edges = edges.as_array().map(|a| a.len()).unwrap_or(0);

    // Build the AnalyzerInput JSON
    let analyzer_input = serde_json::json!({
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
    });

    let input_str = serde_json::to_string(&analyzer_input)?;
    let output_str = topo_analyzer::analyze_full_json(&input_str)
        .map_err(|e| anyhow::anyhow!("Analysis failed: {e}"))?;

    if args.as_json {
        // Pretty-print the JSON output
        let output: serde_json::Value = serde_json::from_str(&output_str)?;
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        let data: serde_json::Value = serde_json::from_str(&output_str)?;
        let ignores = policy.map(|p| &p.ignores).cloned().unwrap_or_default();
        let use_color = !args.no_color && atty_stdout();
        let text = format::format_text(
            &data,
            args.verbose,
            args.diagnostics,
            &ignores,
            project_root,
            use_color,
        );
        print!("{text}");
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
    use std::io::IsTerminal;
    std::io::stdout().is_terminal()
}
